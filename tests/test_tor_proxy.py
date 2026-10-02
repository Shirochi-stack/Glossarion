import io
import concurrent.futures
import threading
import socket
import socketserver
from contextlib import ExitStack
from urllib.parse import urlsplit
import tarfile
from pathlib import Path
from unittest.mock import Mock

import pytest

import ocagy_cli
import opera_aria
import tor_proxy


@pytest.fixture(autouse=True)
def enable_tor_for_existing_route_tests(monkeypatch):
    monkeypatch.setenv("GLOSSARION_TOR_ENABLED", "1")
    monkeypatch.setenv("GLOSSARION_TOR_INSTANCES", "1")


def test_parallel_pool_rotates_between_independent_instances(monkeypatch):
    monkeypatch.setenv('GLOSSARION_TOR_INSTANCES', '4')
    instances = [tor_proxy._TorInstance(index) for index in range(4)]
    for index, instance in enumerate(instances):
        instance.process = Mock(poll=lambda: None)
        instance.port = 19050 + index
    monkeypatch.setattr(tor_proxy, '_INSTANCES', instances)
    monkeypatch.setattr(tor_proxy, '_POOL_NEXT', 0)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        urls = list(pool.map(lambda _: tor_proxy.new_proxy_url(), range(16)))
    ports = [urlsplit(url).port for url in urls]
    assert set(ports) == {19050, 19051, 19052, 19053}
    assert all(ports.count(port) == 4 for port in set(ports))
    assert len({urlsplit(url).username for url in urls}) == 16


def test_newnym_authenticates_with_cookie_and_observes_cooldown(monkeypatch, tmp_path):
    commands = []
    class Control(socketserver.StreamRequestHandler):
        def handle(self):
            for _ in range(2):
                commands.append(self.rfile.readline().decode().strip())
                self.wfile.write(b'250 OK\r\n')
                self.wfile.flush()
    server = socketserver.ThreadingTCPServer(('127.0.0.1', 0), Control)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    instance = tor_proxy._TorInstance(2)
    instance.runtime = Mock(name=str(tmp_path))
    instance.runtime.name = str(tmp_path)
    instance.control_port = server.server_address[1]
    (tmp_path / 'data').mkdir()
    (tmp_path / 'data' / 'control_auth_cookie').write_bytes(bytes(range(32)))
    monkeypatch.setattr(tor_proxy, '_REQUEST_INSTANCES', {12345: instance})
    logs = []
    try:
        tor_proxy.notify_block('http://user:tor@127.0.0.1:12345', logs.append)
        tor_proxy.notify_block('http://user:tor@127.0.0.1:12345', logs.append)
        assert commands == ['AUTHENTICATE ' + bytes(range(32)).hex(), 'SIGNAL NEWNYM']
        assert any('accepted' in log for log in logs)
        assert any('cooldown' in log for log in logs)
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


@pytest.mark.parametrize('status', [403, 429])
def test_opera_block_requests_newnym(monkeypatch, status):
    monkeypatch.setattr(tor_proxy, 'new_proxy_url', lambda **kw: 'http://user:tor@127.0.0.1:19050')
    notify = Mock()
    monkeypatch.setattr(tor_proxy, 'notify_block', notify)
    monkeypatch.setattr(opera_aria.requests, 'post', Mock(return_value=Mock(status_code=status)))
    response = opera_aria._post_chat('token', 'Hello', 30)
    try:
        notify.assert_called_once()
        assert urlsplit(notify.call_args.args[0]).port != 19050
    finally:
        response.close()


@pytest.mark.parametrize('message', ['OpenCode Zen HTTP error: 403', 'OpenCode Zen quota/rate limit: 429'])
def test_ocz_block_requests_newnym_before_relay_closes(monkeypatch, message):
    monkeypatch.setattr(ocagy_cli, 'is_cancelled', lambda: False)
    monkeypatch.setattr(tor_proxy, 'new_proxy_url', lambda **kw: 'http://user:tor@127.0.0.1:19050')
    monkeypatch.setattr(ocagy_cli, '_send_opencode_zen_completion_impl', Mock(side_effect=ocagy_cli.OcAgyError(message)))
    notify = Mock()
    monkeypatch.setattr(tor_proxy, 'notify_block', notify)
    with pytest.raises(ocagy_cli.OcAgyError, match=message):
        ocagy_cli.send_opencode_zen_completion(messages=[], model='ocz/example-free')
    notify.assert_called_once()


def test_unique_identities_and_no_environment_mutation(monkeypatch):
    monkeypatch.setattr(tor_proxy._INSTANCES[0], "process", Mock(poll=lambda: None))
    monkeypatch.setattr(tor_proxy._INSTANCES[0], "port", 19050)
    first = tor_proxy.new_proxy_url()
    second = tor_proxy.new_proxy_url()
    assert first != second
    assert first.endswith("@127.0.0.1:19050")
    original = {"HTTPS_PROXY": "old", "no_proxy": "*", "KEEP": "value"}
    env = tor_proxy.proxy_environment(original, first)
    assert original["HTTPS_PROXY"] == "old"
    assert env["http_proxy"] == env["HTTPS_PROXY"] == first
    assert env["NO_PROXY"] == env["no_proxy"] == "localhost,127.0.0.1,::1"
    assert env["KEEP"] == "value"


def test_bundle_selection_uses_official_stable_architecture(monkeypatch):
    monkeypatch.setattr(tor_proxy.sys, "platform", "win32")
    monkeypatch.setattr(tor_proxy.platform, "machine", lambda: "AMD64")
    stable = "https://dist.torproject.org/torbrowser/15.0.24/tor-expert-bundle-windows-x86_64-15.0.24.tar.gz"
    html = (f'<a href="https://evil.example/{stable.rsplit("/", 1)[-1]}">bad</a>'
            '<a href="https://dist.torproject.org/torbrowser/16.0a13/tor-expert-bundle-windows-x86_64-16.0a13.tar.gz">alpha</a>'
            f'<a href="{stable}">stable</a>')
    assert tor_proxy._bundle_url(html) == stable


def test_archive_cannot_escape_destination(tmp_path):
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        member = tarfile.TarInfo("../escaped")
        member.size = 1
        tar.addfile(member, io.BytesIO(b"x"))
    destination = tmp_path / "bundle"
    destination.mkdir()
    with pytest.raises((tarfile.TarError, tor_proxy.TorProxyError)):
        tor_proxy._extract_bundle(archive, destination)
    assert not (tmp_path / "escaped").exists()


@pytest.mark.parametrize("streaming", [True, False])
def test_ocz_both_modes_get_tor_but_paid_alias_does_not(monkeypatch, streaming):
    monkeypatch.setattr(ocagy_cli, "is_cancelled", lambda: False)
    monkeypatch.setattr(ocagy_cli, "ensure_opencode_installed", lambda **kw: "fake")
    monkeypatch.setattr(ocagy_cli, "_subprocess_env", lambda: {"NO_PROXY": "*"})
    proxy = Mock(return_value="http://fresh:tor@127.0.0.1:19050")
    monkeypatch.setattr(tor_proxy, "new_proxy_url", proxy)
    seen = []
    logs = []

    def run(**kwargs):
        seen.append(kwargs["subprocess_env"])
        return {"content": "OK", "finish_reason": "stop"}

    monkeypatch.setattr(ocagy_cli, "_send_via_server_zen", run)
    monkeypatch.setattr(ocagy_cli, "_run_opencode_zen_buffered", run)
    for model in ("ocz/example-free", "oc/example-paid"):
        result = ocagy_cli.send_opencode_zen_completion(
            messages=[{"role": "user", "content": "Hello"}], model=model,
            log_stream=streaming, log_fn=logs.append)
        assert result["content"] == "OK"
    assert urlsplit(seen[0]["HTTPS_PROXY"]).username == "fresh"
    assert urlsplit(seen[0]["HTTPS_PROXY"]).hostname == "127.0.0.1"
    assert seen[0]["NO_PROXY"] != "*"
    assert "HTTPS_PROXY" not in seen[1]
    assert proxy.call_count == 1
    routing_logs = [line for line in logs if "routing request through Tor" in line]
    assert len(routing_logs) == 1
    assert routing_logs[0].startswith("🧄")
    assert "127.0.0.1:" in routing_logs[0]
    assert "fresh circuit identity" in routing_logs[0]
    assert ":tor@" not in routing_logs[0]


@pytest.mark.parametrize('streaming', [True, False])
def test_ocz_retry_after_failure_gets_new_port_and_identity(monkeypatch, streaming):
    monkeypatch.setattr(ocagy_cli, 'is_cancelled', lambda: False)
    monkeypatch.setattr(ocagy_cli, 'ensure_opencode_installed', lambda **kw: 'fake')
    monkeypatch.setattr(ocagy_cli, '_subprocess_env', lambda: {})
    monkeypatch.setattr(tor_proxy, 'new_proxy_url', lambda **kw:
                        f'http://{tor_proxy.uuid.uuid4().hex}:tor@127.0.0.1:19050')
    attempts = []

    def run(**kwargs):
        attempts.append(urlsplit(kwargs['subprocess_env']['HTTPS_PROXY']))
        if len(attempts) == 1:
            raise ocagy_cli.OcAgyError('Temporary upstream failure')
        return {'content': 'OK', 'finish_reason': 'stop'}

    monkeypatch.setattr(ocagy_cli, '_send_via_server_zen', run)
    monkeypatch.setattr(ocagy_cli, '_run_opencode_zen_buffered', run)
    params = dict(messages=[{'role': 'user', 'content': 'Hello'}],
                  model='ocz/example-free', log_stream=streaming)
    with pytest.raises(ocagy_cli.OcAgyError, match='Temporary upstream failure'):
        ocagy_cli.send_opencode_zen_completion(**params)
    assert ocagy_cli.send_opencode_zen_completion(**params)['content'] == 'OK'
    assert attempts[0].port != attempts[1].port
    assert attempts[0].username != attempts[1].username


def test_opera_retry_rotates_and_closes_responses(monkeypatch):
    opera_aria.reset_cancel()
    monkeypatch.setattr(opera_aria, "get_token", lambda **kw: "token")
    proxies = iter(["http://one:tor@127.0.0.1:1", "http://two:tor@127.0.0.1:1"])
    monkeypatch.setattr(tor_proxy, "new_proxy_url", lambda **kw: next(proxies))
    rejected = Mock(status_code=401)
    success = Mock(status_code=200)
    rejected_close, success_close = rejected.close, success.close
    success.iter_lines.return_value = iter(['data: {"text":"OK"}'])
    post = Mock(side_effect=[rejected, success])
    monkeypatch.setattr(opera_aria.requests, "post", post)
    monkeypatch.setattr(opera_aria, "_classify", lambda *args: ("text", "OK"))
    result = opera_aria.send_chat_completion(
        messages=[{"role": "user", "content": "Hello"}], log_stream=False)
    assert result["content"] == "OK"
    assert post.call_args_list[0].kwargs["proxies"] != post.call_args_list[1].kwargs["proxies"]
    attempt_urls = [urlsplit(call.kwargs['proxies']['https']) for call in post.call_args_list]
    assert attempt_urls[0].port != attempt_urls[1].port
    assert attempt_urls[0].username != attempt_urls[1].username
    assert post.call_args.kwargs["allow_redirects"] is False
    assert rejected_close.called and success_close.called


def test_tor_failure_does_not_send_direct_opera_request(monkeypatch):
    def fail(**kw):
        raise tor_proxy.TorProxyError("bootstrap failed")

    monkeypatch.setattr(tor_proxy, "new_proxy_url", fail)
    post = Mock()
    monkeypatch.setattr(opera_aria.requests, "post", post)
    with pytest.raises(opera_aria.OperaAriaError, match="bootstrap failed"):
        opera_aria._post_chat("token", "Hello", 30)
    post.assert_not_called()


def test_missing_tor_downloads_and_reuses_bundle(monkeypatch, tmp_path):
    monkeypatch.setenv("GLOSSARION_TOR_DIR", str(tmp_path))
    expected_binary = tmp_path / "bundle" / "tor" / "tor.exe"
    monkeypatch.setattr(tor_proxy, "_find_binary", lambda:
                        str(expected_binary) if expected_binary.is_file() else None)
    monkeypatch.delenv("GLOSSARION_TOR_BINARY", raising=False)
    monkeypatch.setattr(tor_proxy.shutil, "which", lambda _: None)
    monkeypatch.setattr(tor_proxy.sys, "platform", "win32")
    monkeypatch.setattr(tor_proxy.platform, "machine", lambda: "AMD64")
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:gz") as tar:
        member = tarfile.TarInfo("tor/tor.exe")
        member.size = 4
        tar.addfile(member, io.BytesIO(b"fake"))
    url = "https://dist.torproject.org/torbrowser/15.0.24/tor-expert-bundle-windows-x86_64-15.0.24.tar.gz"
    page = Mock(text=f'<a href="{url}">Download</a>')
    archive = Mock()
    archive.iter_content.return_value = [data.getvalue()]
    for response in (page, archive):
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
    session = Mock()
    session.__enter__ = Mock(return_value=session)
    session.__exit__ = Mock(return_value=False)
    session.get.side_effect = [page, archive]
    monkeypatch.setattr(tor_proxy.requests, "Session", lambda: session)
    installed = tor_proxy.ensure_tor_installed()
    assert Path(installed).read_bytes() == b"fake"
    assert tor_proxy.ensure_tor_installed() == installed
    assert session.get.call_count == 2
    assert session.trust_env is False


@pytest.mark.parametrize("failure", [False, True])
def test_bootstrap_waits_for_ready_and_cleans_failed_process(monkeypatch, tmp_path, failure):
    monkeypatch.setattr(tor_proxy._INSTANCES[0], "process", None)
    monkeypatch.setattr(tor_proxy._INSTANCES[0], "port", None)
    monkeypatch.setattr(tor_proxy._INSTANCES[0], "runtime", None)
    monkeypatch.setattr(tor_proxy, "ensure_tor_installed", lambda *args: str(tmp_path / "tor"))
    process = Mock()
    process.poll.return_value = 1 if failure else None

    def launch(command, **kwargs):
        config = Path(command[-1])
        text = config.read_text()
        assert "IsolateSOCKSAuth" in text
        assert "SocksPort 0" in text
        assert "ClientOnly 1" in text
        assert "Log notice stdout" in text
        assert "Log notice file" not in text
        assert kwargs["stdout"].name == str(config.parent / "tor.log")
        (config.parent / "tor.log").write_text("failed" if failure else "Bootstrapped 100% (done)")
        return process

    monkeypatch.setattr(tor_proxy, "popen_no_window", launch)
    try:
        if failure:
            with pytest.raises(tor_proxy.TorProxyError, match="exited during startup"):
                tor_proxy.new_proxy_url()
            assert process.terminate.called
            assert tor_proxy._INSTANCES[0].process is None
        else:
            assert "@127.0.0.1:" in tor_proxy.new_proxy_url()
            assert tor_proxy._INSTANCES[0].process is process
            assert tor_proxy.new_proxy_url()
    finally:
        tor_proxy._stop()


def test_parallel_first_requests_start_one_tor_and_use_unique_identities(monkeypatch, tmp_path):
    monkeypatch.setattr(tor_proxy._INSTANCES[0], "process", None)
    monkeypatch.setattr(tor_proxy._INSTANCES[0], "port", None)
    monkeypatch.setattr(tor_proxy._INSTANCES[0], "runtime", None)
    install = Mock(return_value=str(tmp_path / "tor"))
    monkeypatch.setattr(tor_proxy, "ensure_tor_installed", install)
    process = Mock(poll=lambda: None)
    started = threading.Event()
    release_bootstrap = threading.Event()

    def launch(command, **kwargs):
        config = Path(command[-1])
        started.set()

        def ready():
            assert release_bootstrap.wait(5)
            (config.parent / "tor.log").write_text("Bootstrapped 100% (done)")

        threading.Thread(target=ready).start()
        return process

    launcher = Mock(side_effect=launch)
    monkeypatch.setattr(tor_proxy, "popen_no_window", launcher)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            futures = [pool.submit(tor_proxy.new_proxy_url) for _ in range(16)]
            assert started.wait(5)
            assert not any(f.done() for f in futures)
            release_bootstrap.set()
            urls = [f.result(timeout=5) for f in futures]
        assert len(set(urls)) == 16
        assert len({url.split("@", 1)[1] for url in urls}) == 1
        install.assert_called_once()
        launcher.assert_called_once()
    finally:
        release_bootstrap.set()
        tor_proxy._stop()


def test_waiting_parallel_request_can_cancel(monkeypatch):
    with tor_proxy._INSTANCES[0].lock:
        with pytest.raises(tor_proxy.TorProxyError, match="waiting for startup"):
            tor_proxy.new_proxy_url(cancelled=lambda: True)


@pytest.mark.parametrize("streaming", [True, False])
def test_parallel_ocz_requests_keep_independent_proxy_environments(monkeypatch, streaming):
    monkeypatch.setattr(tor_proxy._INSTANCES[0], "process", Mock(poll=lambda: None))
    monkeypatch.setattr(tor_proxy._INSTANCES[0], "port", 19050)
    monkeypatch.setattr(ocagy_cli, "is_cancelled", lambda: False)
    monkeypatch.setattr(ocagy_cli, "ensure_opencode_installed", lambda **kw: "fake")
    original = {"NO_PROXY": "*", "HTTPS_PROXY": "ambient"}
    monkeypatch.setattr(ocagy_cli, "_subprocess_env", lambda: dict(original))
    rendezvous = threading.Barrier(8)

    def run(**kwargs):
        env = kwargs["subprocess_env"]
        before = dict(env)
        rendezvous.wait(timeout=5)
        assert env == before
        return {"content": env["HTTPS_PROXY"], "finish_reason": "stop"}

    monkeypatch.setattr(ocagy_cli, "_send_via_server_zen", run)
    monkeypatch.setattr(ocagy_cli, "_run_opencode_zen_buffered", run)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(ocagy_cli.send_opencode_zen_completion,
                               messages=[{"role": "user", "content": "Hello"}],
                               model="ocz/example-free", log_stream=streaming) for _ in range(8)]
        results = [f.result(timeout=10) for f in futures]
    assert len({result["content"] for result in results}) == 8
    assert original == {"NO_PROXY": "*", "HTTPS_PROXY": "ambient"}


def test_parallel_opera_posts_have_independent_proxies(monkeypatch):
    monkeypatch.setattr(tor_proxy._INSTANCES[0], "process", Mock(poll=lambda: None))
    monkeypatch.setattr(tor_proxy._INSTANCES[0], "port", 19050)
    monkeypatch.setattr(opera_aria, "_is_cancelled", lambda: False)
    rendezvous = threading.Barrier(8)

    def post(url, **kwargs):
        proxies = kwargs["proxies"]
        before = dict(proxies)
        rendezvous.wait(timeout=5)
        assert proxies == before
        assert proxies["https"] == proxies["http"]
        return Mock(proxy_url=proxies["https"])

    monkeypatch.setattr(opera_aria.requests, "post", post)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(opera_aria._post_chat, "token", "Hello", 30) for _ in range(8)]
        responses = [f.result(timeout=10) for f in futures]
        urls = [response.proxy_url for response in responses]
        for response in responses:
            response.close()
    assert len(set(urls)) == 8


def test_startup_emits_only_one_condensed_status(monkeypatch, tmp_path):
    monkeypatch.setattr(tor_proxy._INSTANCES[0], "process", None)
    monkeypatch.setattr(tor_proxy._INSTANCES[0], "port", None)
    monkeypatch.setattr(tor_proxy._INSTANCES[0], "runtime", None)
    monkeypatch.setattr(tor_proxy, "ensure_tor_installed", lambda *args: str(tmp_path / "tor"))
    process = Mock(poll=lambda: None)
    logs = []
    runtime_log = []

    def launch(command, **kwargs):
        log = Path(command[-1]).parent / "tor.log"
        runtime_log.append(log)
        log.write_text("[notice] Bootstrapped 55% (loading_descriptors): Loading relay descriptors\n")
        return process

    def next_stage(_):
        with runtime_log[0].open("a") as log:
            log.write("[notice] Bootstrapped 100% (done): Done\n")

    monkeypatch.setattr(tor_proxy, "popen_no_window", launch)
    monkeypatch.setattr(tor_proxy.time, "sleep", next_stage)
    try:
        tor_proxy.new_proxy_url(log_fn=logs.append)
        assert len(logs) == 1
        assert logs[0].startswith('🚀 Starting Tor instance 1')
        assert not any('Bootstrapped' in line or 'selected' in line for line in logs)
    finally:
        tor_proxy._stop()


def test_timeout_preserves_process_and_next_request_resumes(monkeypatch, tmp_path):
    monkeypatch.setattr(tor_proxy._INSTANCES[0], "process", None)
    monkeypatch.setattr(tor_proxy._INSTANCES[0], "port", None)
    monkeypatch.setattr(tor_proxy._INSTANCES[0], "runtime", None)
    monkeypatch.setattr(tor_proxy, "_BOOTSTRAP_WAIT_SECONDS", 0.02)
    monkeypatch.setattr(tor_proxy, "ensure_tor_installed", lambda *args: str(tmp_path / "tor"))
    process = Mock(poll=lambda: None)
    logs = []

    def launch(command, **kwargs):
        (Path(command[-1]).parent / "tor.log").write_text(
            "Bootstrapped 55% (loading_descriptors): Loading relay descriptors\n")
        return process

    launcher = Mock(side_effect=launch)
    monkeypatch.setattr(tor_proxy, "popen_no_window", launcher)
    try:
        with pytest.raises(tor_proxy.TorBootstrapTimeout, match="55%"):
            tor_proxy.new_proxy_url(log_fn=logs.append)
        assert tor_proxy._INSTANCES[0].process is process
        assert tor_proxy._INSTANCES[0].port is None  # Never return a not-yet-ready proxy.
        process.terminate.assert_not_called()
        runtime = Path(tor_proxy._INSTANCES[0].runtime.name)
        assert runtime.exists()
        with (runtime / "tor.log").open("a") as log:
            log.write("Bootstrapped 100% (done): Done\n")
        assert "@127.0.0.1:" in tor_proxy.new_proxy_url(log_fn=logs.append)
        launcher.assert_called_once()
    finally:
        tor_proxy._stop()


def test_existing_installation_is_reported_without_download(monkeypatch):
    monkeypatch.setattr(tor_proxy, "_find_binary", lambda: "existing/tor.exe")
    session = Mock()
    monkeypatch.setattr(tor_proxy.requests, "Session", session)
    logs = []
    assert tor_proxy.ensure_tor_installed(log_fn=logs.append) == "existing/tor.exe"
    assert logs == ["📍 Found existing Tor installation: existing/tor.exe"]
    session.assert_not_called()


def test_opera_logs_tor_routing_for_each_post_without_credentials(monkeypatch):
    monkeypatch.setattr(opera_aria, "_is_cancelled", lambda: False)
    proxies = iter(["http://first-identity:tor@127.0.0.1:19050",
                    "http://second-identity:tor@127.0.0.1:19050"])
    monkeypatch.setattr(tor_proxy, "new_proxy_url", lambda **kw: next(proxies))
    post = Mock()
    monkeypatch.setattr(opera_aria.requests, "post", post)
    logs = []
    for _ in range(2):
        response = opera_aria._post_chat("private-bearer", "Hello", 30, log_fn=logs.append)
        response.close()
    assert len(logs) == 2
    assert all(line.startswith("🧄") for line in logs)
    assert all("routing chat POST through Tor" in line for line in logs)
    assert all("127.0.0.1:" in line for line in logs)
    assert logs[0] != logs[1]
    assert all("private-bearer" not in line and ":tor@" not in line for line in logs)
    assert urlsplit(post.call_args_list[0].kwargs["proxies"]["https"]).username == "first-identity"
    assert urlsplit(post.call_args_list[1].kwargs["proxies"]["https"]).username == "second-identity"


def test_ocz_error_logging_surfaces_hidden_model_error():
    log = ('timestamp=now level=ERROR message=failed error="ProviderModelNotFoundError: '
           'Model not found: opencode/example-free" cause="stack trace omitted"')
    detail = ocagy_cli._zen_error_detail(log)
    assert detail == "ProviderModelNotFoundError: Model not found: opencode/example-free"
    assert "stack trace" not in detail


@pytest.mark.parametrize("detail", [
    "ProviderModelNotFoundError: Model not found: opencode/deepseek-v4-flash-free",
    "Error from provider (Console): Upstream request failed: Model is unavailable.",
])
def test_buffered_ocz_classifies_real_model_failures(monkeypatch, detail):
    process = Mock(returncode=1)
    process.communicate.return_value = (
        ocagy_cli.json.dumps({"type": "error", "error": {"data": {
            "message": "Unexpected server error. Check server logs for details."}}}),
        'timestamp=now level=ERROR error=' + ocagy_cli.json.dumps(detail),
    )
    popen = Mock(return_value=process)
    monkeypatch.setattr(ocagy_cli.subprocess, "Popen", popen)
    with pytest.raises(ocagy_cli.OcAgyError, match="OpenCode Zen model error") as error:
        ocagy_cli._run_opencode_zen_buffered(
            exe="fake", prompt="Hello", effective_model="opencode/deepseek-v4-flash-free",
            timeout_seconds=30, logger=lambda _: None, subprocess_env={})
    assert detail in str(error.value)
    assert "--print-logs" in popen.call_args.args[0]


def test_missing_ocz_model_stops_without_global_api_retries(monkeypatch):
    import unified_api_client as unified

    send = Mock(side_effect=ocagy_cli.OcAgyError(
        "OpenCode Zen model error: ProviderModelNotFoundError: "
        "Model not found: opencode/deepseek-v4-flash-free"))
    monkeypatch.setattr(unified, "_opencode_zen_send", send)
    monkeypatch.setattr(unified, "OPENCODE_ZEN_AVAILABLE", True)
    monkeypatch.setenv("MAX_RETRIES", "3")
    monkeypatch.setenv("USE_FALLBACK_KEYS", "0")
    client = unified.UnifiedClient("", "ocz/deepseek-v4-flash-free", _skip_cancel_reset=True)
    monkeypatch.setattr(client, "_save_payload", lambda *args, **kwargs: None)
    monkeypatch.setattr(client, "_save_failed_request", lambda *args, **kwargs: None)
    monkeypatch.setattr(client, "_track_stats", lambda *args, **kwargs: None)
    with pytest.raises(unified.UnifiedClientError) as error:
        client._send_internal(
            [{"role": "user", "content": "Hello"}], temperature=0.2,
            max_tokens=1024, context="translation", request_id="ocz-model-error")
    assert error.value.error_type == "config_error"
    send.assert_called_once()


def test_geoip_paths_resolve_tor_browser_data_in_paths_with_spaces(tmp_path):
    root = tmp_path / "Tor Browser" / "Browser" / "TorBrowser"
    binary = root / "Tor" / "tor.exe"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"fake")
    data = root / "Data" / "Tor"
    data.mkdir(parents=True)
    for name in ("geoip", "geoip6"):
        (data / name).write_text("data")
    config = tor_proxy._geoip_config(str(binary))
    options = dict(line.split(" ", 1) for line in config.splitlines())
    assert ocagy_cli.json.loads(options["GeoIPFile"]) == str((data / "geoip").resolve())
    assert ocagy_cli.json.loads(options["GeoIPv6File"]) == str((data / "geoip6").resolve())


def test_tor_explicitly_disabled_skips_setup_for_both_routes(monkeypatch):
    monkeypatch.setenv("GLOSSARION_TOR_ENABLED", "0")
    assert not tor_proxy.enabled()
    setup = Mock(side_effect=AssertionError("Tor must not start while disabled"))
    monkeypatch.setattr(tor_proxy, "new_proxy_url", setup)
    with tor_proxy.request_proxy() as proxy:
        assert proxy is None
    monkeypatch.setattr(ocagy_cli, "is_cancelled", lambda: False)
    monkeypatch.setattr(ocagy_cli, "ensure_opencode_installed", lambda **kw: "fake")
    monkeypatch.setattr(ocagy_cli, "_subprocess_env", lambda: {})
    run = Mock(return_value={"content": "OK", "finish_reason": "stop"})
    monkeypatch.setattr(ocagy_cli, "_run_opencode_zen_buffered", run)
    ocagy_cli.send_opencode_zen_completion(
        messages=[{"role": "user", "content": "Hello"}], model="ocz/example-free", log_stream=False)
    assert "HTTPS_PROXY" not in run.call_args.kwargs["subprocess_env"]
    post = Mock()
    monkeypatch.setattr(opera_aria.requests, "post", post)
    response = opera_aria._post_chat("token", "Hello", 30)
    response.close()
    assert "proxies" not in post.call_args.kwargs
    setup.assert_not_called()


def test_request_ports_forward_parallel_traffic_and_close_after_use(monkeypatch):
    class Echo(socketserver.BaseRequestHandler):
        def handle(self):
            while data := self.request.recv(65536):
                self.request.sendall(data)

    upstream = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Echo)
    upstream.daemon_threads = True
    worker = threading.Thread(target=upstream.serve_forever, daemon=True)
    worker.start()
    monkeypatch.setattr(tor_proxy, "new_proxy_url", lambda **kw:
                        f"http://{tor_proxy.uuid.uuid4().hex}:tor@127.0.0.1:{upstream.server_address[1]}")
    try:
        with ExitStack() as contexts:
            urls = [contexts.enter_context(tor_proxy.request_proxy()) for _ in range(8)]
            ports = [urlsplit(url).port for url in urls]
            assert len(set(ports)) == 8
            assert len({urlsplit(url).username for url in urls}) == 8

            def transfer(port):
                with socket.create_connection(("127.0.0.1", port), timeout=3) as connection:
                    payload = f"CONNECT sample-{port}:443 HTTP/1.1\r\n\r\n".encode()
                    connection.sendall(payload)
                    assert connection.recv(65536) == payload

            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(transfer, ports))
        for port in ports:
            with pytest.raises(OSError):
                socket.create_connection(("127.0.0.1", port), timeout=0.1)
    finally:
        upstream.shutdown()
        upstream.server_close()
        worker.join(timeout=2)


def test_tor_disabled_by_default(monkeypatch):
    monkeypatch.delenv("GLOSSARION_TOR_ENABLED", raising=False)
    assert not tor_proxy.enabled()
