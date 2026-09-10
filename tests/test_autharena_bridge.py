"""Exercise the broker on an ephemeral loopback port, without personal state."""

import copy
from contextlib import nullcontext
from http.client import HTTPConnection
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest
import requests

import autharena_bridge as bridge_module


CONTROL = "test-desktop-control-secret"


@pytest.mark.parametrize('race', [False, True])
@pytest.mark.parametrize('revision', [None, 1, bridge_module.SERVER_REVISION])
def test_broker_reuses_only_current_server_revision(monkeypatch, tmp_path, race, revision):
    monkeypatch.setattr(bridge_module, 'prepare_extension', lambda: tmp_path)
    monkeypatch.setattr(bridge_module, '_state_lock', nullcontext)
    monkeypatch.setattr(bridge_module, '_settings', lambda: {'control_token': CONTROL})
    calls = []

    def health(*args, **kwargs):
        calls.append(kwargs['timeout'])
        if race and len(calls) == 1:
            raise requests.ConnectionError('Not listening yet')
        result = {'version': bridge_module.VERSION}
        if revision is not None:
            result['revision'] = revision
        return result

    def server(*args, **kwargs):
        if not race:
            pytest.fail('Do not try binding over a responding server')
        raise OSError('Another instance won the startup race')

    monkeypatch.setattr(bridge_module, 'request', health)
    monkeypatch.setattr(bridge_module, '_Server', server)
    if revision == bridge_module.SERVER_REVISION:
        assert bridge_module.ensure_broker()['control_token'] == CONTROL
    else:
        with pytest.raises(RuntimeError, match='close all Glossarion windows'):
            bridge_module.ensure_broker()
    assert len(calls) == (2 if race else 1)


DEVICE_A = "test_browser_profile_a"
DEVICE_B = "test_browser_profile_b"
DEVICE_TOKEN_A = "test-browser-a-token"
DEVICE_TOKEN_B = "test-browser-b-token"
EXTENSION_ORIGIN = "chrome-extension://aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"


@pytest.fixture
def broker(monkeypatch, tmp_path):
    settings = {
        "version": bridge_module.VERSION, "control_token": CONTROL,
        "devices": {DEVICE_A: {"token": DEVICE_TOKEN_A}, DEVICE_B: {"token": DEVICE_TOKEN_B}},
        "accounts": {},
    }
    saved = []
    monkeypatch.setattr(bridge_module._State, "_save", lambda self: saved.append(copy.deepcopy(self.settings)))
    state = bridge_module._State(settings, tmp_path / "extension")
    server = bridge_module._Server(("127.0.0.1", 0), bridge_module._Handler)
    server.state = state
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=.01), daemon=True)
    thread.start()
    session = requests.Session()
    session.trust_env = False
    base = f"http://127.0.0.1:{server.server_port}"

    def call(method, path, body=None, *, token=CONTROL, origin=None, headers=None):
        actual_headers = dict(headers or {})
        if token is not None:
            actual_headers["Authorization"] = "Bearer " + token
        if origin is not None:
            actual_headers["Origin"] = origin
        actual_headers.setdefault("Content-Type", "application/json")
        return session.request(method, base + path, data=json.dumps(body).encode("utf-8") if body is not None else None,
                               headers=actual_headers, timeout=3)

    def create(account=0, login=False, interactive=True, **options):
        response = call("POST", "/control/jobs", {
            "account_id": account, "login": login, "allow_interactive": interactive,
            "timeout": 30, "owner_pid": 1001, "model": "test-model",
            "payload": {"userMessage": {"content": "PRIVATE-PROMPT-TEST"}}, **options,
        })
        assert response.status_code == 200, response.text
        return response.json()

    fixture = SimpleNamespace(state=state, settings=settings, saved=saved, call=call, create=create,
                              server=server, base=base, session=session)
    try:
        yield fixture
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        session.close()


def extension_events(broker, job_id, events, device_token=DEVICE_TOKEN_A):
    return broker.call("POST", "/extension/events", {"job_id": job_id, "events": events},
                       token=device_token, origin=EXTENSION_ORIGIN)


def control_events(broker, job_id):
    response = broker.call("GET", f"/control/jobs/{job_id}/events")
    assert response.status_code == 200, response.text
    return response.json()


def command(broker, job_id, name):
    return broker.call("POST", f"/control/jobs/{job_id}/command", {"command": name})


def poll_command(broker, device_token=DEVICE_TOKEN_A):
    response = broker.call("GET", "/extension/poll", token=device_token, origin=EXTENSION_ORIGIN)
    assert response.status_code == 200, response.text
    return response.json()["command"]


def test_real_helper_process_streams_first_fragment_before_completion(broker, monkeypatch, tmp_path):
    import autharena as arena

    monkeypatch.setenv('TRANSLATION_CANCELLED', '0')
    arena._cancel_event.clear()
    monkeypatch.setattr(arena, '_profiles_root', lambda: tmp_path / 'profiles')
    monkeypatch.setattr(arena, '_ensure_browser_bridge', lambda: {
        'url': broker.base, 'control_token': CONTROL,
    })
    broker.settings['accounts']['0'] = DEVICE_A
    first_fragment = threading.Event()
    failures, progress, fragments = [], [], []

    def browser_peer():
        try:
            run = poll_command(broker)
            assert run['type'] == 'run'
            job_id = run['job_id']
            assert run['config']['model'] == 'selected-model'
            assert extension_events(broker, job_id, [
                {'event': 'verified', 'logged_in': True, 'tou_accepted': True},
                {'event': 'ready'},
            ]).status_code == 200
            assert poll_command(broker) == {'type': 'dispatch', 'job_id': job_id}
            assert extension_events(broker, job_id, [
                {'event': 'chunk', 'data': 'a0:"Hello"\n'},
            ]).status_code == 200
            # Completion is deliberately withheld until Python receives the
            # first fragment, proving the real IPC/HTTP path is incremental.
            assert first_fragment.wait(3), 'The stream was buffered until completion'
            assert extension_events(broker, job_id, [
                {'event': 'chunk', 'data': 'a0:" world"\nad:{"finishReason":"stop"}\n'},
                {'event': 'done'},
            ]).status_code == 200
        except Exception as error:
            failures.append(error)

    peer = threading.Thread(target=browser_peer, daemon=True)
    peer.start()

    def chunk(kind, text):
        fragments.append((kind, text))
        first_fragment.set()

    result = arena._run_browser_helper({
        'account_id': 0, 'model': 'selected-model', 'login': False,
        'allow_interactive': False, 'timeout': 10,
        'payload': arena._build_payload(None, 'Test prompt'),
    }, time.monotonic() + 10, before_send_callback=lambda: progress.append('dispatch'), on_chunk=chunk)
    peer.join(3)
    assert failures == []
    assert not peer.is_alive()
    assert progress == ['dispatch']
    assert fragments == [('content', 'Hello'), ('content', ' world')]
    assert result['content'] == 'Hello world'
    assert result['finish_reason_explicit'] is True
    assert arena.get_account_status(0)['logged_in'] is True


def paired_job(broker, account=0):
    broker.settings["accounts"][str(account)] = DEVICE_A
    job = broker.create(account=account)
    run = poll_command(broker)
    assert run["type"] == "run"
    assert run["job_id"] == job["job_id"]
    return job["job_id"]


@pytest.fixture
def setup_installer(monkeypatch, broker):
    entered, release = threading.Event(), threading.Event()
    calls, folders, callbacks = [], [], []
    outcome = {"status": "awaiting_connection", "message": "Browser setup submitted; waiting for the helper."}

    def install(path, browser_hint="", cancel_check=None, *, connect_url, progress):
        calls.append((path, browser_hint, connect_url))
        callbacks.append(progress)
        entered.set()
        for _ in range(100):
            if cancel_check():
                return {"status": "cancelled", "message": "Login cancelled."}
            if release.wait(.02):
                return dict(outcome)
        return {"status": "error", "message": "Test installer timed out"}

    def open_folder(path):
        folders.append(path)
        return {"status": "opened", "message": "Folder opened."}

    monkeypatch.setitem(sys.modules, "autharena_setup", SimpleNamespace(
        install_extension=install, open_extension_folder=open_folder,
    ))
    monkeypatch.setattr(bridge_module, 'update_extension_from_github',
                        lambda **kwargs: broker.state.extension_path)
    yield SimpleNamespace(entered=entered, release=release, calls=calls, outcome=outcome, folders=folders,
                          progress=callbacks)
    release.set()


def setup_nonce(broker, account=0):
    job = broker.create(account=account, login=True, timeout=600)
    return job, urlsplit(job["connect_url"]).fragment


def setup_request(broker, action, nonce, **extra):
    return broker.call("POST", "/setup/" + action, {"nonce": nonce, **extra}, token=None, origin=broker.base)


def wait_for_setup(broker, job_id):
    end = time.monotonic() + 3
    while broker.state.active_setup == job_id and time.monotonic() < end:
        time.sleep(.01)
    assert broker.state.active_setup is None
    return broker.state.setup_results[job_id]


def test_connect_page_and_status_never_start_install_without_a_click(broker, setup_installer):
    _, nonce = setup_nonce(broker)
    page = broker.call("GET", "/connect", token=None)
    assert page.status_code == 200
    assert "Install in browser" in page.text and "Copy folder path" in page.text and "Open folder" in page.text
    assert "Keep this Arena Login page in front" in page.text
    assert "in the same browser window" in page.text
    assert page.headers["X-Frame-Options"] == "DENY"
    assert page.headers["Content-Security-Policy"] == "frame-ancestors 'none'"
    assert setup_request(broker, "status", nonce).json()["status"] == "ready"
    assert setup_installer.calls == []


@pytest.mark.parametrize("origin", [None, "null", "https://arena.ai", "https://evil.example", EXTENSION_ORIGIN])
@pytest.mark.parametrize("action", ["install", "status", "open-folder"])
def test_setup_requires_local_page_origin_even_with_valid_nonce(broker, setup_installer, origin, action):
    _, nonce = setup_nonce(broker)
    response = broker.call("POST", "/setup/" + action, {"nonce": nonce}, token=CONTROL, origin=origin)
    assert response.status_code == 403
    assert setup_installer.calls == setup_installer.folders == []


@pytest.mark.parametrize("headers", [
    {"Host": "evil.example"}, {"Sec-Fetch-Site": "cross-site"},
    {"Content-Type": "text/plain"}, {"Content-Type": "application/x-www-form-urlencoded"},
])
def test_setup_rejects_host_csrf_and_non_json_requests(broker, setup_installer, headers):
    _, nonce = setup_nonce(broker)
    response = broker.call("POST", "/setup/install", {"nonce": nonce},
                           token=None, origin=broker.base, headers=headers)
    assert response.status_code in (400, 403)
    assert setup_installer.calls == []


@pytest.mark.parametrize("invalid", ["unknown_nonce_with_valid_length", [], None, "short"])
def test_setup_requires_a_real_active_pairing_nonce(broker, setup_installer, invalid):
    assert setup_request(broker, "install", invalid).status_code == 403
    assert setup_installer.calls == []


@pytest.mark.parametrize("extra", [
    {"path": "C:/arbitrary"}, {"url": "https://evil.example"}, {"browser_hint": "--load-extension=x"},
    {"browser_hint": "firefox"}, {"account_id": 9},
    {"connect_url": "http://127.0.0.1:18874/connect#attacker_nonce"}, {"window_handle": 123},
])
def test_setup_cannot_select_arbitrary_paths_urls_or_commands(broker, setup_installer, extra):
    _, nonce = setup_nonce(broker)
    assert setup_request(broker, "install", nonce, **extra).status_code == 400
    assert setup_installer.calls == []


def test_setup_is_async_single_active_and_does_not_claim_installation_or_login(broker, setup_installer):
    first, nonce = setup_nonce(broker)
    _, other_nonce = setup_nonce(broker, account=1)
    response = setup_request(broker, "install", nonce, browser_hint="edge")
    assert response.status_code == 200 and response.json()["status"] == "running"
    assert setup_installer.entered.wait(1)
    assert setup_request(broker, "status", nonce).json()["status"] == "running"
    assert setup_request(broker, "install", nonce, browser_hint="edge").json()["status"] == "running"
    assert setup_request(broker, "install", other_nonce).status_code == 409
    assert setup_installer.calls == [(broker.state.extension_path, "edge", first["connect_url"])]
    setup_installer.release.set()
    result = wait_for_setup(broker, first["job_id"])
    assert result["status"] == "awaiting_connection"
    assert setup_request(broker, "status", nonce).json() == result
    assert broker.settings["accounts"] == {}
    assert broker.state.jobs[first["job_id"]]["verified"] is False
    assert "PRIVATE-PROMPT-TEST" not in str(result) and CONTROL not in str(result)


def test_setup_target_is_derived_from_server_not_desktop_or_page_arguments(broker, setup_installer):
    job = broker.create(login=True, timeout=600, connect_url="https://evil.example/connect#fake",
                        base_url="https://evil.example", window_handle=123)
    nonce = urlsplit(job["connect_url"]).fragment
    assert setup_request(broker, "install", nonce, browser_hint="chrome").status_code == 200
    assert setup_installer.entered.wait(1)
    assert setup_installer.calls == [(broker.state.extension_path, "chrome", broker.base + "/connect#" + nonce)]


def test_installer_progress_is_visible_before_completion_and_cannot_claim_success(broker, setup_installer):
    job, nonce = setup_nonce(broker)
    started = setup_request(broker, "install", nonce).json()
    assert setup_installer.entered.wait(1)
    progress = setup_installer.progress[0]
    progress({"status": "running", "message": "Opening Extensions in the current window."})
    visible = setup_request(broker, "status", nonce).json()
    assert visible == {"status": "running", "attempt_id": started["attempt_id"],
                       "message": "Opening Extensions in the current window."}
    assert broker.state.active_setup == job["job_id"]
    for invalid in ({"status": "installed", "message": "Success"}, {"status": "running", "message": []}, "text"):
        progress(invalid)
    assert setup_request(broker, "status", nonce).json() == visible
    progress({"status": "running", "message": "Locating " + job["connect_url"] + " " + nonce})
    assert nonce not in setup_request(broker, "status", nonce).text
    setup_installer.release.set()
    final = wait_for_setup(broker, job["job_id"])
    progress({"status": "running", "message": "Late progress"})
    assert setup_request(broker, "status", nonce).json() == final


def test_cancelled_login_stops_the_installer_and_invalidates_setup_requests(broker, setup_installer):
    job, nonce = setup_nonce(broker)
    assert setup_request(broker, "install", nonce).status_code == 200
    assert setup_installer.entered.wait(1)
    assert command(broker, job["job_id"], "cancel").status_code == 200
    assert wait_for_setup(broker, job["job_id"])["status"] == "cancelled"
    assert setup_request(broker, "status", nonce).status_code == 410
    assert setup_request(broker, "install", nonce).status_code == 410
    assert len(setup_installer.calls) == 1
    setup_installer.progress[0]({"status": "running", "message": "Stale progress"})
    assert broker.state.setup_results[job["job_id"]]["status"] == "cancelled"


def test_consumed_and_expired_nonce_cannot_start_or_inspect_setup(broker, setup_installer):
    job, nonce = setup_nonce(broker)
    assert broker.call("POST", "/pair", {"nonce": nonce, "device_id": DEVICE_A},
                       token=None, origin=EXTENSION_ORIGIN).status_code == 200
    assert setup_request(broker, "install", nonce).status_code == 403
    assert setup_request(broker, "status", nonce).status_code == 403
    _, expired = setup_nonce(broker, account=2)
    broker.state.pairings[expired] = (job["job_id"], time.monotonic() - 1)
    assert setup_request(broker, "install", expired).status_code == 403
    assert setup_installer.calls == []


def test_manual_folder_action_opens_only_prepared_folder(broker, setup_installer):
    _, nonce = setup_nonce(broker)
    assert setup_request(broker, "open-folder", nonce, path="C:/arbitrary").status_code == 400
    response = setup_request(broker, "open-folder", nonce)
    assert response.status_code == 200 and response.json()["status"] == "opened"
    assert setup_installer.folders == [broker.state.extension_path]
    assert setup_installer.calls == []


def test_unexpected_installer_success_is_not_reported_as_installed(broker, setup_installer):
    job, nonce = setup_nonce(broker)
    setup_installer.outcome.update(status="installed", message="unverified success")
    setup_installer.release.set()
    setup_request(broker, "install", nonce)
    assert wait_for_setup(broker, job["job_id"])["status"] == "error"
    assert broker.settings["accounts"] == {}


@pytest.mark.parametrize("headers,origin", [
    ({"Host": "evil.example"}, None),
    ({"Host": "localhost:18874"}, None),
    ({}, "https://arena.ai"),
    ({}, "https://evil.example"),
    ({}, "null"),
])
def test_host_and_origin_protection_rejects_webpage_requests(broker, headers, origin):
    response = broker.call("GET", "/control/health", headers=headers, origin=origin)
    assert response.status_code == 403
    assert "Access-Control-Allow-Origin" not in response.headers


def test_desktop_and_extension_credentials_have_separate_roles(broker):
    assert broker.call("GET", "/control/health", token=None).status_code == 401
    assert broker.call("GET", "/control/health", token="incorrect").status_code == 401
    assert broker.call("GET", "/control/health").json() == {
        "version": bridge_module.VERSION, "revision": bridge_module.SERVER_REVISION,
        "pid": bridge_module.os.getpid(),
    }
    assert broker.call("GET", "/control/health", token=DEVICE_TOKEN_A,
                       origin=EXTENSION_ORIGIN).status_code == 403
    assert broker.call("GET", "/control/health", origin=EXTENSION_ORIGIN).status_code == 403
    assert broker.call("GET", "/extension/poll").status_code == 404
    preflight = broker.call("OPTIONS", "/extension/events", token=None, origin=EXTENSION_ORIGIN)
    assert preflight.status_code == 200
    assert preflight.headers["Access-Control-Allow-Origin"] == EXTENSION_ORIGIN


def test_connect_page_exposes_neither_control_secret_nor_prompt(broker, capsys):
    job = broker.create(login=True)
    assert job["connect_url"].startswith(broker.base + "/connect#")
    nonce = urlsplit(job["connect_url"]).fragment
    response = broker.call("GET", "/connect", token=None)
    assert response.status_code == 200
    assert response.headers["Referrer-Policy"] == "no-referrer"
    assert response.headers["Cache-Control"] == "no-store"
    for private in (CONTROL, DEVICE_TOKEN_A, "PRIVATE-PROMPT-TEST", nonce):
        assert private not in response.text
    captured = capsys.readouterr()
    assert "PRIVATE-PROMPT-TEST" not in captured.out + captured.err
    assert CONTROL not in captured.out + captured.err


def test_pair_nonce_is_single_use_and_account_binds_only_after_verification(broker):
    job = broker.create(account=3, login=True)
    nonce = urlsplit(job["connect_url"]).fragment
    response = broker.call("POST", "/pair", {"nonce": nonce, "device_id": DEVICE_A},
                           token=None, origin=EXTENSION_ORIGIN)
    assert response.status_code == 200
    assert response.json() == {"device_id": DEVICE_A, "device_token": DEVICE_TOKEN_A, "account_id": 3}
    assert "3" not in broker.settings["accounts"]
    assert poll_command(broker)["job_id"] == job["job_id"]
    assert broker.call("POST", "/pair", {"nonce": nonce, "device_id": DEVICE_A},
                       token=None, origin=EXTENSION_ORIGIN).status_code == 403
    invalid = extension_events(broker, job["job_id"], [{"event": "verified", "logged_in": True}])
    assert invalid.status_code == 400
    assert "3" not in broker.settings["accounts"]
    valid = extension_events(broker, job["job_id"], [{"event": "verified", "logged_in": True, "tou_accepted": True}])
    assert valid.status_code == 200
    assert broker.settings["accounts"]["3"] == DEVICE_A
    assert broker.saved[-1]["accounts"]["3"] == DEVICE_A
    assert extension_events(broker, job["job_id"], [{"event": "done"}]).status_code == 200
    assert control_events(broker, job["job_id"])["terminal"] is True


def test_expired_pair_link_cannot_register_or_queue_a_device(broker):
    job = broker.create(login=True)
    nonce = urlsplit(job["connect_url"]).fragment
    with broker.state.cv:
        broker.state.pairings[nonce] = (job["job_id"], time.monotonic() - 1)
    response = broker.call("POST", "/pair", {"nonce": nonce, "device_id": "new_test_browser_profile"},
                           token=None, origin=EXTENSION_ORIGIN)
    assert response.status_code == 403
    assert "new_test_browser_profile" not in broker.settings["devices"]
    assert not broker.state.device_commands


def test_already_bound_browser_profile_cannot_pair_to_another_account(broker):
    broker.settings["accounts"]["1"] = DEVICE_A
    job = broker.create(account=2, login=True)
    response = broker.call("POST", "/pair", {"nonce": urlsplit(job["connect_url"]).fragment, "device_id": DEVICE_A},
                           token=None, origin=EXTENSION_ORIGIN)
    assert response.status_code == 409
    assert broker.settings["accounts"] == {"1": DEVICE_A}


def test_parallel_login_verification_cannot_bind_same_browser_twice(broker):
    first = broker.create(account=1, login=True)
    second = broker.create(account=2, login=True)
    paired = []
    for job in (first, second):
        response = broker.call("POST", "/pair", {
            "nonce": urlsplit(job["connect_url"]).fragment, "device_id": DEVICE_A,
        }, token=None, origin=EXTENSION_ORIGIN)
        if response.status_code == 200:
            paired.append(job["job_id"])
        else:
            assert response.status_code == 409
    assert len(paired) == 1
    assert extension_events(broker, paired[0], [{"event": "verified", "logged_in": True, "tou_accepted": True}]).status_code == 200
    assert sum(device == DEVICE_A for device in broker.settings["accounts"].values()) == 1


def test_wrong_browser_cannot_send_events_for_another_profiles_request(broker):
    job_id = paired_job(broker)
    response = extension_events(broker, job_id, [{"event": "ready"}], device_token=DEVICE_TOKEN_B)
    assert response.status_code == 403
    assert broker.state.jobs[job_id]["ready"] is False
    assert not broker.state.jobs[job_id]["events"]


def test_ready_does_not_dispatch_until_desktop_approves_and_duplicates_are_rejected(broker):
    job_id = paired_job(broker)
    assert command(broker, job_id, "dispatch").status_code == 409
    assert extension_events(broker, job_id, [{"event": "ready"}]).status_code == 200
    assert broker.state.jobs[job_id]["dispatched"] is False
    assert not broker.state.device_commands[DEVICE_A]
    assert command(broker, job_id, "dispatch").status_code == 200
    assert command(broker, job_id, "dispatch").status_code == 409
    assert extension_events(broker, job_id, [{"event": "ready"}]).status_code == 409
    assert poll_command(broker) == {"type": "dispatch", "job_id": job_id}
    assert not broker.state.device_commands[DEVICE_A]


def test_normal_done_and_stream_chunks_require_desktop_dispatch(broker):
    job_id = paired_job(broker)
    assert extension_events(broker, job_id, [{"event": "chunk", "data": 'a0:"not yet"\n'}]).status_code == 409
    assert extension_events(broker, job_id, [{"event": "done"}]).status_code == 409
    assert broker.state.jobs[job_id]["terminal"] is False


def test_terminal_event_stops_later_account_mutations_in_same_batch(broker):
    job_id = paired_job(broker, account=4)
    broker.settings["accounts"].clear()
    response = extension_events(broker, job_id, [
        {"event": "error", "message": "ended"},
        {"event": "verified", "logged_in": True, "tou_accepted": True},
        {"event": "ready"},
    ])
    assert response.status_code == 200
    assert broker.settings["accounts"] == {}
    assert broker.state.jobs[job_id]["ready"] is False
    assert control_events(broker, job_id) == {
        "events": [{"event": "error", "message": "ended"}], "terminal": True,
    }


def test_verified_binding_rechecks_existing_account_ownership(broker):
    job = broker.create(account=2, login=True)
    nonce = urlsplit(job["connect_url"]).fragment
    assert broker.call("POST", "/pair", {"nonce": nonce, "device_id": DEVICE_A},
                       token=None, origin=EXTENSION_ORIGIN).status_code == 200
    # A competing/stale pairing cannot overwrite another confirmed account.
    broker.settings["accounts"]["1"] = DEVICE_A
    response = extension_events(broker, job["job_id"], [{"event": "verified", "logged_in": True, "tou_accepted": True}])
    assert response.status_code == 409
    assert broker.settings["accounts"] == {"1": DEVICE_A}
    assert broker.state.jobs[job["job_id"]]["verified"] is False


def test_stream_events_arrive_before_finish_and_are_drained_once(broker):
    job_id = paired_job(broker)
    extension_events(broker, job_id, [{"event": "ready"}])
    command(broker, job_id, "dispatch")
    poll_command(broker)
    first = {"event": "chunk", "data": 'ag:"think"\na0:"你"\n'}
    assert extension_events(broker, job_id, [first]).status_code == 200
    partial = control_events(broker, job_id)
    assert partial["terminal"] is False
    assert first in partial["events"]
    assert control_events(broker, job_id)["events"] == []
    terminal = [{"event": "chunk", "data": 'ad:{"finishReason":"stop"}\n'}, {"event": "done"}]
    assert extension_events(broker, job_id, terminal).status_code == 200
    final = control_events(broker, job_id)
    assert final == {"events": terminal, "terminal": True}
    assert command(broker, job_id, "dispatch").status_code == 409


def test_explicit_rejection_allows_one_new_readiness_but_no_automatic_dispatch(broker):
    job_id = paired_job(broker)
    extension_events(broker, job_id, [{"event": "ready"}])
    command(broker, job_id, "dispatch")
    poll_command(broker)
    assert extension_events(broker, job_id, [{"event": "rejected"}, {"event": "ready"}]).status_code == 200
    assert broker.state.jobs[job_id]["dispatched"] is False
    assert not broker.state.device_commands[DEVICE_A]
    assert command(broker, job_id, "dispatch").status_code == 200
    assert poll_command(broker)["type"] == "dispatch"


def test_cancel_owner_queues_cancel_only_for_its_jobs_and_cannot_replay(broker):
    broker.settings["accounts"].update({"0": DEVICE_A, "1": DEVICE_B})
    first = broker.create(account=0, owner_pid=101)
    other = broker.create(account=1, owner_pid=202)
    poll_command(broker)
    poll_command(broker, DEVICE_TOKEN_B)
    assert broker.call("POST", "/control/cancel-owner", {"owner_pid": 101}).status_code == 200
    assert poll_command(broker) == {"type": "cancel", "job_id": first["job_id"]}
    assert broker.state.jobs[first["job_id"]]["terminal"] is True
    assert broker.state.jobs[other["job_id"]]["terminal"] is False
    events = control_events(broker, first["job_id"])
    assert events["events"][0]["error_type"] == "cancelled"
    assert command(broker, first["job_id"], "dispatch").status_code == 409
    assert extension_events(broker, first["job_id"], [{"event": "ready"}]).status_code == 200
    assert not broker.state.device_commands[DEVICE_A]


@pytest.mark.parametrize("expiration", ["deadline", "disconnected"])
def test_expired_or_abandoned_jobs_queue_cancel_and_do_not_restart(broker, expiration):
    job_id = paired_job(broker)
    with broker.state.cv:
        job = broker.state.jobs[job_id]
        if expiration == "deadline":
            job["deadline"] = time.monotonic() - 1
        else:
            job["last_control"] = time.monotonic() - 16
        broker.state._expire()
    assert poll_command(broker) == {"type": "cancel", "job_id": job_id}
    result = control_events(broker, job_id)
    assert result["terminal"] is True
    assert result["events"][0]["event"] == "error"
    assert command(broker, job_id, "dispatch").status_code == 409


@pytest.mark.parametrize("operation", ["events", "dispatch"])
def test_resumed_desktop_cannot_revive_an_abandoned_request(broker, operation):
    job_id = paired_job(broker)
    extension_events(broker, job_id, [{"event": "ready"}])
    broker.state.jobs[job_id]["last_control"] = time.monotonic() - 16
    if operation == "dispatch":
        assert command(broker, job_id, "dispatch").status_code == 409
    else:
        assert control_events(broker, job_id)["terminal"] is True
    assert broker.state.jobs[job_id]["dispatched"] is False
    assert poll_command(broker) == {"type": "cancel", "job_id": job_id}


def test_pairing_cannot_revive_an_abandoned_login(broker):
    job = broker.create(login=True)
    broker.state.jobs[job["job_id"]]["last_control"] = time.monotonic() - 16
    response = broker.call("POST", "/pair", {
        "nonce": urlsplit(job["connect_url"]).fragment, "device_id": DEVICE_A,
    }, token=None, origin=EXTENSION_ORIGIN)
    assert response.status_code == 410
    assert not broker.state.device_commands
    assert not broker.settings["accounts"]


def test_browser_events_cannot_revive_or_verify_an_abandoned_request(broker):
    job_id = paired_job(broker)
    broker.settings["accounts"].clear()
    broker.state.jobs[job_id]["last_control"] = time.monotonic() - 16
    response = extension_events(broker, job_id, [
        {"event": "verified", "logged_in": True, "tou_accepted": True}, {"event": "ready"},
    ])
    assert response.status_code == 200
    assert broker.state.jobs[job_id]["terminal"] is True
    assert broker.state.jobs[job_id]["ready"] is False
    assert not broker.settings["accounts"]
    assert poll_command(broker) == {"type": "cancel", "job_id": job_id}


def test_offline_browser_fails_before_dispatch_and_queued_run_is_not_replayed(broker):
    broker.settings["accounts"]["2"] = DEVICE_A
    first = broker.create(account=2, interactive=False, timeout=180)
    broker.state.jobs[first["job_id"]]["queued_at"] = time.monotonic() - 35.1
    expired = control_events(broker, first["job_id"])
    assert expired["terminal"] is True
    error, = expired["events"]
    assert error["status_code"] == 503
    assert error["safe_to_rotate"] is True
    assert error["request_dispatched"] is False
    assert not error.get("auth_invalid")
    assert broker.settings["accounts"]["2"] == DEVICE_A
    second = broker.create(account=2, interactive=False)
    assert poll_command(broker)["job_id"] == second["job_id"]
    assert broker.state.jobs[second["job_id"]]["delivered"] is True
    assert not broker.state.device_commands[DEVICE_A]


def test_delivered_job_is_not_subject_to_offline_preflight_deadline(broker):
    job_id = paired_job(broker)
    broker.state.jobs[job_id]["queued_at"] = time.monotonic() - 36
    assert broker.state.jobs[job_id]["delivered"] is True
    assert control_events(broker, job_id) == {"events": [], "terminal": False}


def test_pairing_starts_a_fresh_device_delivery_window(broker):
    job = broker.create(account=3, login=True, timeout=180)
    record = broker.state.jobs[job["job_id"]]
    record["created_at"] = time.monotonic() - 40
    response = broker.call("POST", "/pair", {
        "nonce": urlsplit(job["connect_url"]).fragment, "device_id": DEVICE_A,
    }, token=None, origin=EXTENSION_ORIGIN)
    assert response.status_code == 200
    assert record["queued_at"] > record["created_at"] + 39
    assert poll_command(broker)["job_id"] == job["job_id"]
    assert record["terminal"] is False


def test_cancel_before_browser_poll_discards_the_queued_run(broker):
    broker.settings["accounts"]["0"] = DEVICE_A
    job = broker.create()
    assert command(broker, job["job_id"], "cancel").status_code == 200
    assert poll_command(broker) == {"type": "cancel", "job_id": job["job_id"]}
    assert not broker.state.device_commands[DEVICE_A]
    assert broker.state.jobs[job["job_id"]]["delivered"] is False


def test_single_account_cannot_run_two_concurrent_jobs(broker):
    first = broker.create(account=7)
    response = broker.call("POST", "/control/jobs", {"account_id": 7, "timeout": 30})
    assert response.status_code == 409
    command(broker, first["job_id"], "cancel")
    assert broker.create(account=7)["job_id"] != first["job_id"]


def test_noninteractive_unpaired_account_fails_without_opening_pair_link(broker):
    job = broker.create(account=8, interactive=False)
    assert job["connect_url"] is None
    assert not broker.state.pairings
    result = control_events(broker, job["job_id"])
    assert result["terminal"]
    assert result["events"][0]["status_code"] == 401
    assert result["events"][0]["safe_to_rotate"] is True


@pytest.mark.parametrize("body", [
    {"account_id": -1}, {"account_id": 10000}, {"account_id": True},
    {"timeout": 0}, {"timeout": float("nan")}, {"timeout": 999999}, [],
])
def test_invalid_job_requests_fail_as_json_without_leaking_inputs(broker, body):
    response = broker.call("POST", "/control/jobs", body)
    assert response.status_code == 400
    assert set(response.json()) == {"error"}
    assert CONTROL not in response.text


def test_oversized_body_and_unauthorized_or_unknown_profile_events_are_rejected(broker):
    connection = HTTPConnection("127.0.0.1", broker.server.server_port, timeout=3)
    try:
        connection.putrequest("POST", "/control/jobs")
        connection.putheader("Authorization", "Bearer " + CONTROL)
        connection.putheader("Content-Length", str(8 * 1024 * 1024 + 1))
        connection.endheaders()
        oversized = connection.getresponse()
        assert oversized.status == 413
        assert set(json.loads(oversized.read())) == {"error"}
    finally:
        connection.close()
    unauthorized = broker.call("POST", "/extension/events", {"job_id": "unknown", "events": []},
                               token="unknown-device-token", origin=EXTENSION_ORIGIN)
    assert unauthorized.status_code == 401
    unknown = extension_events(broker, "unknown-job-id", [])
    assert unknown.status_code == 404
    assert "unknown-device-token" not in unauthorized.text


def test_event_batch_limit_is_enforced_before_readiness_changes(broker):
    job_id = paired_job(broker)
    response = extension_events(broker, job_id, [{"event": "ready"}] * 257)
    assert response.status_code == 400
    assert broker.state.jobs[job_id]["ready"] is False
    assert not broker.state.jobs[job_id]["events"]


def test_materialized_extension_uses_static_packaged_engine_and_temporary_root(monkeypatch, tmp_path):
    source = Path(__file__).resolve().parents[1] / "assets" / "autharena_extension"
    assert (source / "manifest.json").is_file()
    monkeypatch.setattr(bridge_module, "_root", lambda: tmp_path)
    monkeypatch.setattr(bridge_module, "_extension_source", lambda: source)
    target = bridge_module.prepare_extension()
    assert target == tmp_path / "autharena_extension"
    assert (target / "manifest.json").read_bytes() == (source / "manifest.json").read_bytes()
    script = (target / "arena_page.js").read_text(encoding="utf-8")
    assert script.startswith("function prepareArena(config)")
    assert "config.payload" in script
    assert "config.login_only" in script
    assert "__PAYLOAD__" not in script and "__LOGIN_ONLY__" not in script
    assert not re.search(r"\beval\s*\(|new\s+Function\s*\(", script)
    original_mtime = (target / "arena_page.js").stat().st_mtime_ns
    bridge_module.prepare_extension()
    assert (target / "arena_page.js").stat().st_mtime_ns == original_mtime
    if shutil.which("node"):
        result = subprocess.run([shutil.which("node"), "--check", str(target / "arena_page.js")],
                                capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stderr
