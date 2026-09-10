"""Authenticated loopback bridge to Arena in an existing browser profile.

The companion extension executes only the packaged Arena page controller.
Neither website cookies nor login tokens cross this bridge.
"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
import atexit
import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import secrets
import threading
import time
from urllib.parse import urlsplit

import requests

PORT = 18874
VERSION = 1
_start_lock = threading.RLock()
_running = None


def _root():
    return Path.home() / '.glossarion'


@contextmanager
def _state_lock():
    root = _root()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (root / '.autharena-bridge.lock').open('a+b') as handle:
        handle.seek(0, 2)
        if not handle.tell():
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def _save_settings(settings):
    from token_encryption import save_encrypted_tokens
    root = _root()
    temporary = root / ('.arena-' + secrets.token_hex(8) + '.tmp')
    try:
        save_encrypted_tokens(settings, str(temporary))
        os.chmod(temporary, 0o600)
        os.replace(temporary, root / 'autharena-bridge.enc')
    finally:
        temporary.unlink(missing_ok=True)


def _settings():
    from token_encryption import load_encrypted_tokens
    path = _root() / 'autharena-bridge.enc'
    if path.exists():
        data = load_encrypted_tokens(str(path))
        if not isinstance(data, dict) or not isinstance(data.get('control_token'), str):
            raise RuntimeError('Arena browser pairing data could not be read. Restore or remove autharena-bridge.enc, then pair again.')
        return data
    data = {'version': VERSION, 'control_token': secrets.token_urlsafe(32), 'devices': {}, 'accounts': {}}
    _save_settings(data)
    return data


def _extension_source():
    candidates = [Path(__file__).resolve().parent / 'autharena_extension',
                  Path(__file__).resolve().parent.parent / 'assets' / 'autharena_extension']
    return next((path for path in candidates if (path / 'manifest.json').is_file()), None)


def prepare_extension():
    """Materialize an unpacked extension, including the adapter's shared engine."""
    from autharena import _PREPARE_JS
    source = _extension_source()
    if source is None:
        raise ImportError('The Arena browser helper extension is missing from this installation.')
    target = _root() / 'autharena_extension'
    target.mkdir(parents=True, exist_ok=True)
    for name in ('manifest.json', 'background.js', 'connect.js', 'README.md'):
        path = source / name
        if path.is_file():
            contents = path.read_bytes()
            destination = target / name
            if not destination.is_file() or destination.read_bytes() != contents:
                destination.write_bytes(contents)
    replacements = {'__PAYLOAD__': 'config.payload', '__MODEL__': 'config.model',
                    '__TIMEOUT_MS__': 'config.timeout_ms', '__V2_SITEKEY__': 'config.recaptcha_v2_sitekey',
                    '__REJECTIONS__': 'config.rejections', '__LOGIN_ONLY__': 'config.login_only',
                    '__ALLOW_INTERACTIVE__': 'config.allow_interactive'}
    body = re.sub('|'.join(map(re.escape, replacements)), lambda m: replacements[m.group()], _PREPARE_JS)
    script = 'function prepareArena(config) {\n' + body + '\n}\n'
    page = target / 'arena_page.js'
    if not page.is_file() or page.read_text(encoding='utf-8') != script:
        page.write_text(script, encoding='utf-8')
    return target


def request(bridge, method, path, body=None, *, timeout=5):
    """Control-plane requests never use environment proxies or browser CORS."""
    with requests.Session() as session:
        session.trust_env = False
        response = session.request(method, bridge['url'] + path, json=body,
                                   headers={'Authorization': 'Bearer ' + bridge['control_token']}, timeout=timeout)
    if not response.ok:
        try:
            message = response.json().get('error', 'Arena browser bridge request failed')
        except ValueError:
            message = 'Arena browser bridge returned an unexpected response'
        raise RuntimeError(message)
    return response.json()


class _Problem(Exception):
    def __init__(self, status, message):
        self.status, self.message = status, message


class _State:
    def __init__(self, settings, extension_path):
        self.settings = settings
        self.extension_path = extension_path
        self.cv = threading.Condition(threading.RLock())
        self.jobs = {}
        self.pairings = {}
        self.device_commands = {}
        self.last_seen = {}

    def _save(self):
        with _state_lock():
            _save_settings(self.settings)

    def role(self, token):
        if secrets.compare_digest(token, self.settings['control_token']):
            return 'control', None
        for device, entry in self.settings['devices'].items():
            if secrets.compare_digest(token, entry['token']):
                return 'extension', device
        return None, None

    def _queue(self, device, command):
        self.device_commands.setdefault(device, deque()).append(command)
        self.cv.notify_all()

    def _cancel(self, job, message='Arena request cancelled'):
        if job['terminal']:
            return
        job['terminal'] = True
        job['events'].append({'event': 'error', 'message': message, 'error_type': 'cancelled'})
        if job['device']:
            self._queue(job['device'], {'type': 'cancel', 'job_id': job['id']})

    def _expire(self):
        now = time.monotonic()
        for job in list(self.jobs.values()):
            if not job['terminal'] and (now > job['deadline'] or now - job['last_control'] > 15):
                self._cancel(job, 'Arena request cancelled because the desktop request ended or timed out')
            if (not job['terminal'] and job['device'] and not job['delivered']
                    and now - job['queued_at'] >= 35):
                job['terminal'] = True
                job['events'].append({'event': 'error', 'status_code': 503, 'safe_to_rotate': True,
                                      'request_dispatched': False,
                                      'message': f'Arena account {job["account"]} browser helper is offline. Open its paired browser profile and enable the Arena helper.'})
            if job['terminal'] and now - job['last_control'] > 60:
                self.jobs.pop(job['id'], None)
        for nonce, (job_id, expires) in list(self.pairings.items()):
            if now > expires or job_id not in self.jobs:
                self.pairings.pop(nonce, None)

    def create(self, config, base_url):
        account = config.get('account_id', 0)
        if type(account) is not int or not 0 <= account <= 9999:
            raise _Problem(400, 'Invalid Arena account number')
        duration = float(config.get('timeout', 180))
        if not 0 < duration <= 86400:
            raise _Problem(400, 'Invalid Arena request timeout')
        self._expire()
        if any(not j['terminal'] and j['account'] == account for j in self.jobs.values()):
            raise _Problem(409, f'Arena account {account} already has an active request')
        login = bool(config.get('login'))
        interactive = login or bool(config.get('allow_interactive', True))
        device = None if login else self.settings['accounts'].get(str(account))
        job_id = secrets.token_urlsafe(24)
        page_config = {key: config.get(key) for key in ('model', 'payload', 'timeout', 'login', 'allow_interactive', 'recaptcha_v2_sitekey')}
        page_config.update(model='' if login else str(config.get('model') or ''), login=login,
                           allow_interactive=interactive, timeout=duration, account_id=account)
        created_at = time.monotonic()
        job = {'id': job_id, 'account': account, 'device': device, 'config': page_config,
               'created_at': created_at, 'queued_at': created_at if device else None, 'delivered': False,
               'deadline': created_at + duration, 'last_control': created_at,
               'owner': config.get('owner_pid'), 'events': deque(), 'terminal': False,
               'ready': False, 'dispatched': False, 'verified': False}
        self.jobs[job_id] = job
        connect_url = None
        if device:
            self._queue(device, {'type': 'run', 'job_id': job_id, 'config': page_config})
        elif interactive:
            nonce = secrets.token_urlsafe(32)
            self.pairings[nonce] = (job_id, job['deadline'])
            connect_url = base_url + '/connect#' + nonce
        else:
            job['terminal'] = True
            job['events'].append({'event': 'error', 'status_code': 401, 'safe_to_rotate': True,
                                  'message': f'Arena account {account} is not paired. Use Arena Login first.'})
        return {'job_id': job_id, 'connect_url': connect_url}

    def pair(self, body):
        self._expire()
        nonce, device = body.get('nonce'), body.get('device_id')
        if not isinstance(device, str) or not re.fullmatch(r'[a-zA-Z0-9_-]{16,128}', device):
            raise _Problem(400, 'Invalid browser profile identifier')
        pairing = self.pairings.get(nonce)
        if not pairing or pairing[1] <= time.monotonic():
            raise _Problem(403, 'Arena login link expired. Click Arena Login again.')
        job = self.jobs.get(pairing[0])
        if not job or job['terminal']:
            raise _Problem(410, 'This Arena login is no longer active')
        for account, paired_device in self.settings['accounts'].items():
            if paired_device == device and account != str(job['account']):
                raise _Problem(409, f'This browser profile is already account {account}. Open this login link in another browser profile for account {job["account"]}.')
        for pending in self.jobs.values():
            if (not pending['terminal'] and pending['id'] != job['id']
                    and pending['device'] == device and pending['account'] != job['account']):
                raise _Problem(409, 'This browser profile is already completing another account login.')
        # A nonce is single-use. Persistent device secrets never appear in URLs.
        self.pairings.pop(nonce, None)
        if device not in self.settings['devices']:
            self.settings['devices'][device] = {'token': secrets.token_urlsafe(32)}
            self._save()
        job['device'] = device
        job['queued_at'] = self.last_seen[device] = time.monotonic()
        self._queue(device, {'type': 'run', 'job_id': job['id'], 'config': job['config']})
        return {'device_id': device, 'device_token': self.settings['devices'][device]['token'], 'account_id': job['account']}

    def poll(self, device):
        self.last_seen[device] = time.monotonic()
        end = time.monotonic() + 20
        while True:
            self._expire()
            pending = self.device_commands.setdefault(device, deque())
            while pending:
                command = pending.popleft()
                job = self.jobs.get(command['job_id'])
                if command['type'] == 'cancel' or (job and not job['terminal']):
                    if command['type'] == 'run':
                        job['delivered'] = True
                    return {'command': command}
            remaining = end - time.monotonic()
            if remaining <= 0:
                return {'command': None}
            self.cv.wait(min(remaining, 1))

    def job(self, job_id):
        job = self.jobs.get(job_id)
        if job is None:
            raise _Problem(404, 'Arena request no longer exists; it will not be replayed')
        return job

    def events(self, device, body):
        self._expire()
        job = self.job(body.get('job_id'))
        if job['device'] != device:
            raise _Problem(403, 'This request belongs to another browser profile')
        if job['terminal']:
            return {'ok': True}
        events = body.get('events')
        if not isinstance(events, list) or len(events) > 256:
            raise _Problem(400, 'Invalid Arena event batch')
        for event in events:
            if not isinstance(event, dict):
                raise _Problem(400, 'Invalid Arena event')
            kind = event.get('event')
            if kind == 'ready':
                if job['dispatched'] or job['ready']:
                    raise _Problem(409, 'Duplicate Arena dispatch readiness')
                job['ready'] = True
            elif kind == 'rejected':
                job['dispatched'] = False
                job['ready'] = False
            elif kind == 'verified':
                if event.get('logged_in') is not True or event.get('tou_accepted') is not True:
                    raise _Problem(400, 'Arena login was not verified')
                if any(paired_device == device and account != str(job['account'])
                       for account, paired_device in self.settings['accounts'].items()):
                    raise _Problem(409, 'This browser profile is already verified for another account.')
                job['verified'] = True
                self.settings['accounts'][str(job['account'])] = device
                self._save()
            elif kind == 'logged_out':
                job['verified'] = False
            elif kind == 'chunk':
                if not job['dispatched']:
                    raise _Problem(409, 'Arena output arrived before dispatch')
            elif kind == 'done':
                if job['config']['login'] and not job['verified']:
                    raise _Problem(409, 'Arena login ended without verification')
                if not job['config']['login'] and not job['dispatched']:
                    raise _Problem(409, 'Arena completion arrived before desktop dispatch')
                job['terminal'] = True
            elif kind == 'error':
                job['terminal'] = True
            elif kind not in {'status', 'action'}:
                raise _Problem(400, 'Unknown Arena event')
            job['events'].append(event)
            if job['terminal']:
                break
        if len(job['events']) > 2048:
            self._cancel(job, 'Arena request cancelled because the desktop stopped reading the stream')
        self.cv.notify_all()
        return {'ok': True}


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False


class _Handler(BaseHTTPRequestHandler):
    server_version = 'ArenaBridge/1'

    def log_message(self, *args):
        pass

    def _origin(self):
        origin = self.headers.get('Origin')
        if origin and not re.fullmatch(r'(?:chrome|moz)-extension://[a-zA-Z0-9_-]+', origin):
            raise _Problem(403, 'Only the paired Arena browser extension can access this service')
        return origin

    def _send(self, status, value, *, html_page=False):
        raw = value.encode('utf-8') if html_page else json.dumps(value, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'text/html; charset=utf-8' if html_page else 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(raw)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        origin = self.headers.get('Origin', '')
        if re.fullmatch(r'(?:chrome|moz)-extension://[a-zA-Z0-9_-]+', origin):
            self.send_header('Access-Control-Allow-Origin', origin)
            self.send_header('Access-Control-Allow-Headers', 'Authorization, Content-Type')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Private-Network', 'true')
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self):
        self._handle(options=True)

    def do_GET(self):
        self._handle()

    def do_POST(self):
        self._handle()

    def _handle(self, options=False):
        try:
            host = self.headers.get('Host', '')
            if host != f'127.0.0.1:{self.server.server_port}':
                raise _Problem(403, 'Invalid Arena bridge host')
            path = urlsplit(self.path).path
            if self.command == 'GET' and path == '/connect':
                return self._send(200, _connect_page(self.server.state.extension_path), html_page=True)
            self._origin()
            if options:
                return self._send(200, {})
            length = int(self.headers.get('Content-Length', '0'))
            if length < 0 or length > 8 * 1024 * 1024:
                raise _Problem(413, 'Arena bridge request is too large')
            body = json.loads(self.rfile.read(length)) if length else {}
            if not isinstance(body, dict):
                raise _Problem(400, 'Invalid Arena bridge request')
            state = self.server.state
            with state.cv:
                if self.command == 'POST' and path == '/pair':
                    return self._send(200, state.pair(body))
                header = self.headers.get('Authorization', '')
                token = header[7:] if header.startswith('Bearer ') else ''
                role, device = state.role(token)
                if role is None:
                    raise _Problem(401, 'Arena browser bridge authentication failed')
                if role == 'extension':
                    if path == '/extension/poll' and self.command == 'GET':
                        return self._send(200, state.poll(device))
                    if path == '/extension/events' and self.command == 'POST':
                        return self._send(200, state.events(device, body))
                    raise _Problem(403, 'Browser extensions cannot use desktop control endpoints')
                if self.headers.get('Origin'):
                    raise _Problem(403, 'Desktop control is unavailable to browser origins')
                if path == '/control/health':
                    return self._send(200, {'version': VERSION})
                if path == '/control/jobs' and self.command == 'POST':
                    return self._send(200, state.create(body, f'http://127.0.0.1:{self.server.server_port}'))
                if path == '/control/cancel-owner' and self.command == 'POST':
                    for job in state.jobs.values():
                        if job['owner'] == body.get('owner_pid'):
                            state._cancel(job)
                    return self._send(200, {'ok': True})
                match = re.fullmatch(r'/control/jobs/([a-zA-Z0-9_-]+)/(events|command)', path)
                if match:
                    state._expire()
                    job = state.job(match.group(1))
                    job['last_control'] = time.monotonic()
                    if match.group(2) == 'events' and self.command == 'GET':
                        if not job['terminal'] and not job['events']:
                            state.cv.wait(.2)
                        events = list(job['events'])
                        job['events'].clear()
                        return self._send(200, {'events': events, 'terminal': job['terminal']})
                    if match.group(2) == 'command' and self.command == 'POST':
                        command = body.get('command')
                        if command == 'cancel':
                            state._cancel(job)
                        elif command == 'dispatch' and not job['terminal'] and job['ready'] and not job['dispatched']:
                            job['dispatched'], job['ready'] = True, False
                            state._queue(job['device'], {'type': 'dispatch', 'job_id': job['id']})
                        else:
                            raise _Problem(409, 'Arena request cannot be dispatched in its current state')
                        return self._send(200, {'ok': True})
                raise _Problem(404, 'Unknown Arena bridge endpoint')
        except _Problem as error:
            self._send(error.status, {'error': error.message})
        except (ValueError, TypeError):
            self._send(400, {'error': 'Invalid Arena bridge request'})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            self._send(500, {'error': 'Arena browser bridge failed to handle the request'})


def _connect_page(extension_path):
    folder = html.escape(str(extension_path))
    return f'''<!doctype html><meta charset="utf-8"><title>Arena Login</title>
<style>body{{font:17px system-ui;max-width:760px;margin:70px auto;padding:24px;background:#202124;color:#eee}}h1{{font-size:28px}}code{{display:block;padding:16px;background:#303134;overflow-wrap:anywhere}}li{{margin:16px 0}}#status{{color:#9ad}}button{{padding:8px}}</style>
<h1>Arena Login</h1><p id="status">Connecting to the Arena helper in this browser…</p>
<p>If this is your first login, enable the helper extension once. Arena sign-in will then open automatically in this browser.</p>
<ol><li>Open your browser’s Extensions page and turn on Developer mode.</li>
<li>Choose <b>Load unpacked</b> and select this folder:<code>{folder}</code></li>
<li>Refresh this page. The helper will open Arena’s sign-in form.</li></ol>
<p>The helper only works with Arena and this local app. Your Arena session stays in this browser.</p>
<script>addEventListener('message',e=>{{if(e.source===window&&e.data?.type==='arena-pair-result'&&e.data.error)document.getElementById('status').textContent=e.data.error;}});</script>'''


def ensure_broker():
    global _running
    with _start_lock:
        extension_path = prepare_extension()
        with _state_lock():
            settings = _settings()
        bridge = {'url': f'http://127.0.0.1:{PORT}', 'control_token': settings['control_token']}
        try:
            if request(bridge, 'GET', '/control/health', timeout=.5).get('version') == VERSION:
                return bridge
        except (requests.RequestException, RuntimeError):
            pass
        try:
            server = _Server(('127.0.0.1', PORT), _Handler)
        except OSError as error:
            # Another application instance may have won the same startup race.
            try:
                if request(bridge, 'GET', '/control/health', timeout=2).get('version') == VERSION:
                    return bridge
            except (requests.RequestException, RuntimeError):
                pass
            raise RuntimeError(f'Arena browser bridge could not use local port {PORT}. Close the other app using it and retry.') from error
        server.state = _State(settings, extension_path)
        threading.Thread(target=server.serve_forever, name='autharena-browser-bridge', daemon=True).start()
        _running = server
        return bridge


def close_broker():
    global _running
    server, _running = _running, None
    if server:
        server.shutdown()
        server.server_close()


atexit.register(close_broker)
