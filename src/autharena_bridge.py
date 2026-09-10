"""Authenticated loopback bridge to Arena in an existing browser profile.

The companion extension executes only the packaged Arena page controller.
Neither website cookies nor login tokens cross this bridge.
"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
import atexit
import ast
import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import secrets
import sys
import threading
import time
from urllib.parse import urlsplit

import requests

PORT = 18874
VERSION = 1
# Bump when a running server must not be reused after installer/bridge updates.
# This is separate from the persistent pairing data and extension protocol.
SERVER_REVISION = 5
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
    bundle = getattr(sys, '_MEIPASS', None)
    if bundle is not None:
        # A one-file extraction is temporary. It is a source only, never the
        # path Chromium should remember for the unpacked extension.
        source = Path(bundle) / 'autharena_extension'
        return source if source.is_dir() else None
    candidates = [Path(__file__).resolve().parent / 'autharena_extension',
                  Path(__file__).resolve().parent.parent / 'assets' / 'autharena_extension']
    return next((path for path in candidates if (path / 'manifest.json').is_file()), None)


def update_extension_from_github(*, cancel_check, progress):
    """Download one coherent repository revision; never execute remote Python."""
    def fetch(url, limit):
        if cancel_check():
            raise RuntimeError('Arena extension download was cancelled.')
        with requests.get(url, timeout=(5, 15), stream=True, allow_redirects=False,
                          headers={'Accept': 'application/vnd.github+json',
                                   'User-Agent': 'Glossarion-Arena-Setup'}) as response:
            response.raise_for_status()
            if response.status_code != 200:
                raise RuntimeError('GitHub did not return the requested extension file.')
            data = bytearray()
            for chunk in response.iter_content(65536):
                if cancel_check():
                    raise RuntimeError('Arena extension download was cancelled.')
                data.extend(chunk)
                if len(data) > limit:
                    raise RuntimeError('The GitHub extension download exceeded its size limit.')
            return bytes(data)

    progress({'status': 'running', 'message': 'Downloading the Arena extension from Shirochi-stack/Glossarion on GitHub.'})
    try:
        commit = json.loads(fetch('https://api.github.com/repos/Shirochi-stack/Glossarion/commits/main', 1000000))['sha']
        if not isinstance(commit, str) or not re.fullmatch(r'[0-9a-f]{40}', commit):
            raise ValueError('Invalid repository revision')
        base = f'https://raw.githubusercontent.com/Shirochi-stack/Glossarion/{commit}/'
        assets = {}
        for name in ('manifest.json', 'background.js', 'connect.js', 'README.md'):
            progress({'status': 'running', 'message': f'Downloading Arena extension: {name}.'})
            assets[name] = fetch(base + 'assets/autharena_extension/' + name, 2000000)
        # The repository stores the page controller as a Python string. Extract
        # that literal from the SAME commit without importing or executing it.
        tree = ast.parse(fetch(base + 'src/autharena.py', 2000000).decode('utf-8-sig'))
        values = [node.value for node in tree.body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == '_PREPARE_JS' for target in node.targets)]
        if len(values) != 1 or not isinstance(values[0], ast.Constant) or not isinstance(values[0].value, str):
            raise ValueError('The repository page controller is not a literal string')
        if cancel_check():
            raise RuntimeError('Arena extension download was cancelled.')
        return prepare_extension(assets=assets, prepare_js=values[0].value, source_revision=commit)
    except (requests.RequestException, ValueError, KeyError, TypeError, SyntaxError, ImportError) as error:
        raise RuntimeError('Could not download a complete Arena extension from GitHub. '
                           'The installed files were kept. Retry installation.') from error


def prepare_extension(*, assets=None, prepare_js=None, source_revision=None):
    """Validate and atomically update files at the extension's persistent path."""
    target = _root() / 'autharena_extension'
    if assets is None:
        if (target / '.github-revision').is_file():
            from autharena_setup import _validated_folder
            try:
                return _validated_folder(target)
            except (OSError, ValueError):
                pass
        from autharena import _PREPARE_JS
        prepare_js = _PREPARE_JS
        source = _extension_source()
        if source is None:
            raise ImportError('The Arena browser helper extension is missing from this installation.')
        assets = {}
        for name in ('manifest.json', 'background.js', 'connect.js', 'README.md'):
            path = source / name
            if not path.is_file():
                raise ImportError(f'The Arena browser helper installation is missing {name}.')
            assets[name] = path.read_bytes()
    else:
        assets = dict(assets)
    for name in ('manifest.json', 'background.js', 'connect.js', 'README.md'):
        if not assets[name].strip():
            raise ImportError(f'The Arena browser helper installation contains an empty {name}.')
    try:
        manifest = json.loads(assets['manifest.json'].decode('utf-8'))
        if (not isinstance(manifest, dict) or manifest.get('manifest_version') != 3
                or manifest.get('name') != 'Glossarion Arena Browser Companion'):
            raise ValueError('expected a Manifest V3 object')
        referenced = [manifest['background']['service_worker']]
        for content_script in manifest.get('content_scripts', []):
            referenced.extend(content_script.get('js', []))
        if any(name not in assets for name in referenced):
            raise ValueError('manifest references an unpackaged script')
    except (ValueError, KeyError, TypeError, AttributeError, UnicodeError) as exc:
        raise ImportError(f'The Arena browser helper manifest is invalid: {exc}') from exc
    replacements = {'__PAYLOAD__': 'config.payload', '__MODEL__': 'config.model',
                    '__TIMEOUT_MS__': 'config.timeout_ms', '__V2_SITEKEY__': 'config.recaptcha_v2_sitekey',
                    '__REJECTIONS__': 'config.rejections', '__LOGIN_ONLY__': 'config.login_only',
                    '__ALLOW_INTERACTIVE__': 'config.allow_interactive'}
    body = re.sub('|'.join(map(re.escape, replacements)), lambda m: replacements[m.group()], prepare_js)
    script = 'function prepareArena(config) {\n' + body + '\n}\n'
    assets['arena_page.js'] = script.encode('utf-8')
    if source_revision:
        assets['.github-revision'] = source_revision.encode('ascii')
    # Serialize concurrent desktop/helper processes. The target path and
    # manifest identity stay stable across one-file extraction directories.
    with _state_lock():
        target.mkdir(parents=True, exist_ok=True)
        staged = []
        committed = []
        temporary_paths = []

        def stage(contents):
            temporary = target.parent / ('.arena-extension-' + secrets.token_hex(12) + '.tmp')
            temporary_paths.append(temporary)
            with temporary.open('xb') as handle:
                handle.write(contents)
                handle.flush()
                os.fsync(handle.fileno())
            return temporary

        try:
            # Stage the complete update before touching installed files; put
            # the manifest last so a browser reload sees complete script files.
            for name in sorted(assets, key=lambda value: value == 'manifest.json'):
                destination = target / name
                previous = destination.read_bytes() if destination.is_file() else None
                if previous == assets[name]:
                    continue
                replacement = stage(assets[name])
                backup = stage(previous) if previous is not None else None
                staged.append((destination, replacement, backup))
            for destination, replacement, backup in staged:
                os.replace(replacement, destination)
                committed.append((destination, backup))
        except OSError:
            # Restore already replaced files if a later replace fails (for
            # example because a browser temporarily locks a file on Windows).
            for destination, backup in reversed(committed):
                if backup is None:
                    destination.unlink(missing_ok=True)
                else:
                    os.replace(backup, destination)
            raise
        finally:
            for temporary in temporary_paths:
                temporary.unlink(missing_ok=True)
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
        self.setup_results = {}
        self.active_setup = None

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
        for job_id in list(self.setup_results):
            if job_id not in self.jobs and job_id != self.active_setup:
                self.setup_results.pop(job_id, None)

    def setup_job(self, nonce):
        self._expire()
        if not isinstance(nonce, str) or not re.fullmatch(r'[a-zA-Z0-9_-]{16,256}', nonce):
            raise _Problem(403, 'Invalid Arena login link. Click Arena Login in Glossarion again.')
        pairing = self.pairings.get(nonce)
        if not pairing or pairing[1] <= time.monotonic():
            raise _Problem(403, 'Arena login link expired or already connected. Click Arena Login again if needed.')
        job = self.job(pairing[0])
        if job['terminal']:
            raise _Problem(410, 'This Arena login is no longer active')
        return job

    def setup_status(self, body):
        if set(body) != {'nonce'}:
            raise _Problem(400, 'Invalid Arena setup request')
        job = self.setup_job(body.get('nonce'))
        return dict(self.setup_results.get(job['id'], {
            'status': 'ready', 'message': 'Click Install in browser to set up the Arena helper.',
        }))

    def start_setup(self, body):
        if set(body) - {'nonce', 'browser_hint'} or 'nonce' not in body:
            raise _Problem(400, 'Invalid Arena setup request')
        hint = body.get('browser_hint', '')
        if hint not in ('', 'chrome', 'edge'):
            raise _Problem(400, 'Unsupported browser selection')
        job = self.setup_job(body.get('nonce'))
        job_id = job['id']
        if self.active_setup == job_id:
            return dict(self.setup_results[job_id])
        if self.active_setup is not None:
            raise _Problem(409, 'Another Arena browser installation is in progress. Finish it first.')
        attempt_id = secrets.token_urlsafe(12)
        self.active_setup = job_id
        self.setup_results[job_id] = {'status': 'running', 'attempt_id': attempt_id,
                                     'message': 'Keep this Arena Login page in front while setup finds its browser window.'}

        def cancelled():
            with self.cv:
                self._expire()
                current = self.jobs.get(job_id)
                return not current or current['terminal'] or body['nonce'] not in self.pairings

        def public_message(message):
            return message.replace(job['connect_url'], 'Arena Login').replace(body['nonce'], '[login link]')[:2000]

        def progress(update):
            if (not isinstance(update, dict) or update.get('status') != 'running'
                    or not isinstance(update.get('message'), str)):
                return
            with self.cv:
                if self.active_setup != job_id or cancelled():
                    return
                self.setup_results[job_id] = {'status': 'running', 'attempt_id': attempt_id,
                                             'message': public_message(update['message'])}
                self.cv.notify_all()

        def install():
            result = {'status': 'error', 'message': 'Automatic setup could not finish. Use the manual steps below.'}
            try:
                from autharena_setup import install_extension
                with self.cv:
                    self.setup_job(body['nonce'])
                extension_path = update_extension_from_github(cancel_check=cancelled, progress=progress)
                returned = install_extension(extension_path, browser_hint=hint, cancel_check=cancelled,
                                             connect_url=job['connect_url'], progress=progress)
                if (isinstance(returned, dict)
                        and returned.get('status') in {'awaiting_connection', 'manual_required', 'error', 'cancelled'}
                        and isinstance(returned.get('message'), str)):
                    result = {'status': returned['status'], 'message': public_message(returned['message'])}
            except RuntimeError as error:
                result = {'status': 'error', 'message': public_message(str(error))}
            except Exception:
                pass
            finally:
                with self.cv:
                    self.setup_results[job_id] = dict(result, attempt_id=attempt_id)
                    if self.active_setup == job_id:
                        self.active_setup = None
                    self.cv.notify_all()

        threading.Thread(target=install, name='autharena-browser-setup', daemon=True).start()
        return dict(self.setup_results[job_id])

    def open_setup_folder(self, body):
        if set(body) != {'nonce'}:
            raise _Problem(400, 'Invalid Arena setup request')
        self.setup_job(body.get('nonce'))
        from autharena_setup import open_extension_folder
        result = open_extension_folder(self.extension_path)
        if not isinstance(result, dict) or not isinstance(result.get('message'), str):
            raise _Problem(500, 'Could not open the Arena helper folder')
        return {'status': str(result.get('status', 'manual_required')), 'message': result['message'][:2000]}

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
        # This URL comes only from the server's own loopback address and nonce.
        # It binds installation to the browser window displaying this login.
        job['connect_url'] = connect_url
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
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header('Content-Security-Policy', "frame-ancestors 'none'")
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
            setup_route = path in {'/setup/install', '/setup/status', '/setup/open-folder'}
            if setup_route:
                if (self.headers.get('Origin') != f'http://127.0.0.1:{self.server.server_port}'
                        or self.headers.get('Sec-Fetch-Site', 'same-origin') != 'same-origin'):
                    raise _Problem(403, 'Arena setup is only available from its local login page')
                if self.command != 'POST' or self.headers.get('Content-Type', '').split(';', 1)[0].strip() != 'application/json':
                    raise _Problem(400, 'Invalid Arena setup request')
            else:
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
                if setup_route:
                    action = {'/setup/install': state.start_setup, '/setup/status': state.setup_status,
                              '/setup/open-folder': state.open_setup_folder}[path]
                    return self._send(200, action(body))
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
                    return self._send(200, {'version': VERSION, 'revision': SERVER_REVISION, 'pid': os.getpid()})
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
    return '''<!doctype html><meta charset="utf-8"><title>Arena Login</title>
<style>body{font:17px system-ui;max-width:760px;margin:56px auto;padding:24px;background:#202124;color:#eee}h1{font-size:28px}code{display:block;padding:16px;background:#303134;overflow-wrap:anywhere}li{margin:16px 0}#status{color:#9ad;min-height:48px}button{padding:12px 18px;border-radius:8px;border:0;cursor:pointer}#install{background:#a8c7fa;color:#172238;font-size:17px}button:disabled{opacity:.6;cursor:default}details{margin-top:28px}summary{cursor:pointer}</style>
<h1>Arena Login</h1>
<p id="status" role="status">Connecting to the Arena helper in this browser…</p>
<p>Set up the helper once, then Arena sign-in opens in this browser. Your browser may ask you to confirm installation.</p>
<p>Keep this Arena Login page in front until setup opens Extensions in the same browser window. Then leave Extensions in front until setup finishes.</p>
<button id="install" type="button">Install in browser</button>
<details id="manual"><summary>Manual installation</summary>
<ol><li>Open your browser’s Extensions page and turn on Developer mode.</li>
<li>Choose <b>Load unpacked</b> and select this folder:<code id="folder">__EXTENSION_FOLDER__</code>
<button id="copy" type="button">Copy folder path</button> <button id="open-folder" type="button">Open folder</button></li>
<li>Return to this page and choose <button id="reconnect" type="button">Connect helper</button>.</li></ol></details>
<p>The helper only works with Arena and this local app. Your Arena session stays in this browser.</p>
<script>
(() => {
  const nonce = location.hash.slice(1);
  const status = document.getElementById('status');
  const install = document.getElementById('install');
  const manual = document.getElementById('manual');
  const storageKey = 'autharena-setup:' + nonce;
  let stopped = false, polling = false;
  const valid = /^[A-Za-z0-9_-]{16,256}$/.test(nonce);
  const browserHint = /Edg\\//.test(navigator.userAgent) ? 'edge' :
    /Chrome\\//.test(navigator.userAgent) ? 'chrome' : '';
  async function request(path, details = {}) {
    const response = await fetch(path, {method:'POST', credentials:'omit', cache:'no-store',
      headers:{'Content-Type':'application/json'}, body:JSON.stringify({nonce, ...details})});
    const result = await response.json();
    if (!response.ok) throw Error(result.error || 'Arena setup is unavailable.');
    return result;
  }
  function reconnect(attempt) {
    if (stopped) return;
    let count = 0;
    try {
      const saved = JSON.parse(sessionStorage.getItem(storageKey) || '{}');
      count = saved.attempt === attempt ? Number(saved.count) || 0 : 0;
      if (count >= 15) {
        status.textContent = 'The helper has not connected yet. Complete browser installation, then choose Connect helper below.';
        manual.open = true; return;
      }
      sessionStorage.setItem(storageKey, JSON.stringify({attempt, count:count + 1}));
    } catch (_) {
      status.textContent = 'Complete browser installation, then choose Connect helper below.';
      manual.open = true; return;
    }
    setTimeout(() => { if (!stopped) location.reload(); }, 2000);
  }
  async function poll() {
    if (stopped) return;
    if (polling) { setTimeout(poll, 500); return; }
    polling = true;
    try {
      const result = await request('/setup/status');
      if (stopped) return;
      if (result.status !== 'ready') status.textContent = result.message;
      install.disabled = result.status === 'running';
      install.textContent = result.status === 'error' ? 'Retry installation' : 'Install in browser';
      if (result.status === 'running') setTimeout(poll, 1500);
      else if (result.status === 'awaiting_connection') reconnect(result.attempt_id);
      else if (result.status === 'manual_required') manual.open = true;
    } catch (error) {
      if (!stopped) { status.textContent = error.message; install.disabled = true; stopped = true; }
    } finally { polling = false; }
  }
  install.addEventListener('click', async () => {
    if (stopped || install.disabled || !valid) return;
    install.disabled = true;
    manual.open = false;
    try {
      const result = await request('/setup/install', {browser_hint:browserHint});
      if (stopped) return;
      status.textContent = result.message;
      setTimeout(poll, 500);
    } catch (error) {
      status.textContent = error.message; install.disabled = false; install.textContent = 'Retry installation';
    }
  });
  document.getElementById('copy').addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(document.getElementById('folder').textContent);
      status.textContent = 'Folder path copied.';
    } catch (_) { status.textContent = 'Select the folder path above and copy it.'; }
  });
  document.getElementById('reconnect').addEventListener('click', async () => {
    if (stopped || !valid) return;
    try { await request('/setup/status'); location.reload(); }
    catch (error) { status.textContent = error.message; }
  });
  document.getElementById('open-folder').addEventListener('click', async () => {
    if (stopped || !valid) return;
    try { const result = await request('/setup/open-folder'); status.textContent = result.message; }
    catch (error) { status.textContent = error.message; }
  });
  addEventListener('message', event => {
    if (event.source !== window || event.origin !== location.origin || event.data?.type !== 'arena-pair-result') return;
    stopped = true; install.disabled = true;
    if (event.data.ok) {
      status.textContent = 'Arena helper connected. Opening Arena sign-in…';
      install.textContent = 'Helper connected';
      try { sessionStorage.removeItem(storageKey); } catch (_) {}
    } else { status.textContent = event.data.error || 'Arena pairing failed. Open a new Arena Login link.'; manual.open = true; }
  });
  if (!valid) {
    stopped = true; install.disabled = true;
    status.textContent = 'Open Arena Login in Glossarion to start a new connection.';
  } else setTimeout(poll, 1200);
})();
</script>'''.replace('__EXTENSION_FOLDER__', html.escape(str(extension_path)))


def _check_server_revision(health):
    if health.get('version') != VERSION or health.get('revision') != SERVER_REVISION:
        raise RuntimeError(
            'Another Glossarion instance is running a different version of the Arena login helper. '
            'Save your work and close all Glossarion windows, then reopen the updated app and click Arena Login. '
            'Refreshing the browser page cannot update the running helper.'
        )


def ensure_broker():
    global _running
    with _start_lock:
        extension_path = prepare_extension()
        with _state_lock():
            settings = _settings()
        bridge = {'url': f'http://127.0.0.1:{PORT}', 'control_token': settings['control_token']}
        health = None
        try:
            health = request(bridge, 'GET', '/control/health', timeout=.5)
        except (requests.RequestException, RuntimeError):
            pass
        if health is not None:
            _check_server_revision(health)
            return bridge
        try:
            server = _Server(('127.0.0.1', PORT), _Handler)
        except OSError as error:
            # Another application instance may have won the same startup race.
            try:
                health = request(bridge, 'GET', '/control/health', timeout=2)
            except (requests.RequestException, RuntimeError):
                pass
            if health is not None:
                _check_server_revision(health)
                return bridge
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
