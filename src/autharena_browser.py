"""JSON-line adapter between AuthArena and its current-browser companion."""
from __future__ import annotations

import json
import queue
import sys
import threading
import time
import webbrowser

from autharena_bridge import ensure_broker, request


def _control_request(config, commands, emit, *, cancelled=lambda: False):
    initial = []
    while not commands.empty():
        initial.append(commands.get_nowait())
    if cancelled() or 'cancel' in initial:
        raise RuntimeError('Arena request cancelled')
    if 'dispatch' in initial:
        raise RuntimeError('Arena dispatch arrived before browser readiness')
    bridge = config.get('bridge') or ensure_broker()
    deadline = time.monotonic() + float(config['timeout'])
    job_id = None
    finished = False
    try:
        job = request(bridge, 'POST', '/control/jobs', {
            key: value for key, value in config.items() if key not in {'bridge', 'profile'}
        })
        job_id = job['job_id']
        connect_url = job.get('connect_url')
        if connect_url:
            emit('status', message='Opening Arena Login in your current browser. Enable the Arena helper once if prompted.')
            if webbrowser.open(connect_url) is False:
                raise RuntimeError('Could not open your current browser for Arena Login')
        else:
            emit('status', message=f'Connecting to Arena account {config.get("account_id", 0)} in its browser profile.')
        base = '/control/jobs/' + job_id
        while time.monotonic() < deadline:
            pending = []
            while True:
                try:
                    pending.append(commands.get_nowait())
                except queue.Empty:
                    break
            if cancelled() or 'cancel' in pending:
                raise RuntimeError('Arena request cancelled')
            if 'dispatch' in pending:
                request(bridge, 'POST', base + '/command', {'command': 'dispatch'})
            response = request(bridge, 'GET', base + '/events', timeout=min(5, max(.05, deadline - time.monotonic())))
            for event in response.get('events', []):
                kind = event.get('event')
                details = {key: value for key, value in event.items() if key != 'event'}
                if kind == 'action':
                    emit('status', message=event.get('message', 'Complete Arena sign-in in your browser.'))
                elif kind in {'verified', 'logged_out', 'ready', 'rejected', 'chunk', 'done', 'error', 'status'}:
                    emit(kind, **details)
                if kind in {'done', 'error'}:
                    finished = True
                    return 0 if kind == 'done' else 1
            if response.get('terminal'):
                raise RuntimeError('Arena browser request ended without a result; it was not replayed')
        raise TimeoutError('Arena timed out waiting for its browser profile. Open that profile with the Arena helper enabled, or allow more time to sign in.')
    finally:
        if job_id and not finished:
            try:
                request(bridge, 'POST', '/control/jobs/' + job_id + '/command', {'command': 'cancel'}, timeout=2)
            except Exception:
                pass


def run_browser_helper(config, *, prepare_script=None):
    """No browser process/profile is launched, copied, inspected, or closed here."""
    def emit(event, **details):
        print(json.dumps({'autharena': 1, 'event': event, **details}, ensure_ascii=False), flush=True)

    commands = queue.Queue()
    stopped = threading.Event()

    def read_commands():
        try:
            for line in sys.stdin:
                try:
                    command = json.loads(line).get('command')
                except (ValueError, AttributeError):
                    continue
                if command in {'dispatch', 'cancel'}:
                    commands.put(command)
                if command == 'cancel':
                    stopped.set()
        finally:
            stopped.set()
            commands.put('cancel')

    threading.Thread(target=read_commands, daemon=True).start()
    try:
        return _control_request(config, commands, emit, cancelled=stopped.is_set)
    except Exception as exc:
        emit('error', message=str(exc), error_type=('timeout' if isinstance(exc, TimeoutError)
                                                  else 'configuration' if isinstance(exc, ImportError) else 'transport'))
        return 1
