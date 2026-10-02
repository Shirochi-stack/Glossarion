import io
import tarfile
from pathlib import Path
from unittest.mock import Mock

import pytest

import ocagy_cli
import opera_aria
import tor_proxy


def test_unique_identities_and_no_environment_mutation(monkeypatch):
    monkeypatch.setattr(tor_proxy, "_PROCESS", Mock(poll=lambda: None))
    monkeypatch.setattr(tor_proxy, "_PORT", 19050)
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

    def run(**kwargs):
        seen.append(kwargs["subprocess_env"])
        return {"content": "OK", "finish_reason": "stop"}

    monkeypatch.setattr(ocagy_cli, "_send_via_server_zen", run)
    monkeypatch.setattr(ocagy_cli, "_run_opencode_zen_buffered", run)
    for model in ("ocz/example-free", "oc/example-paid"):
        result = ocagy_cli.send_opencode_zen_completion(
            messages=[{"role": "user", "content": "Hello"}], model=model,
            log_stream=streaming)
        assert result["content"] == "OK"
    assert seen[0]["HTTPS_PROXY"] == proxy.return_value
    assert seen[0]["NO_PROXY"] != "*"
    assert "HTTPS_PROXY" not in seen[1]
    assert proxy.call_count == 1


def test_opera_retry_rotates_and_closes_responses(monkeypatch):
    opera_aria.reset_cancel()
    monkeypatch.setattr(opera_aria, "get_token", lambda **kw: "token")
    proxies = iter(["http://one:tor@127.0.0.1:1", "http://two:tor@127.0.0.1:1"])
    monkeypatch.setattr(tor_proxy, "new_proxy_url", lambda **kw: next(proxies))
    rejected = Mock(status_code=401)
    success = Mock(status_code=200)
    success.iter_lines.return_value = iter(['data: {"text":"OK"}'])
    post = Mock(side_effect=[rejected, success])
    monkeypatch.setattr(opera_aria.requests, "post", post)
    monkeypatch.setattr(opera_aria, "_classify", lambda *args: ("text", "OK"))
    result = opera_aria.send_chat_completion(
        messages=[{"role": "user", "content": "Hello"}], log_stream=False)
    assert result["content"] == "OK"
    assert post.call_args_list[0].kwargs["proxies"] != post.call_args_list[1].kwargs["proxies"]
    assert post.call_args.kwargs["allow_redirects"] is False
    assert rejected.close.called and success.close.called


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
    monkeypatch.setattr(tor_proxy, "_PROCESS", None)
    monkeypatch.setattr(tor_proxy, "_PORT", None)
    monkeypatch.setattr(tor_proxy, "_RUNTIME", None)
    monkeypatch.setattr(tor_proxy, "ensure_tor_installed", lambda *args: str(tmp_path / "tor"))
    process = Mock()
    process.poll.return_value = 1 if failure else None

    def launch(command, **kwargs):
        config = Path(command[-1])
        text = config.read_text()
        assert "IsolateSOCKSAuth" in text
        assert "SocksPort 0" in text
        assert "ClientOnly 1" in text
        (config.parent / "tor.log").write_text("failed" if failure else "Bootstrapped 100% (done)")
        return process

    monkeypatch.setattr(tor_proxy, "popen_no_window", launch)
    try:
        if failure:
            with pytest.raises(tor_proxy.TorProxyError, match="exited during startup"):
                tor_proxy.new_proxy_url()
            assert process.terminate.called
            assert tor_proxy._PROCESS is None
        else:
            assert "@127.0.0.1:" in tor_proxy.new_proxy_url()
            assert tor_proxy._PROCESS is process
            assert tor_proxy.new_proxy_url()
    finally:
        tor_proxy._stop()
