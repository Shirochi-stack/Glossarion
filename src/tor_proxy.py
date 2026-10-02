"""Managed Tor HTTP CONNECT proxy with a fresh circuit identity per request.

Only provider HTTPS traffic uses this proxy. Installers and loopback control
connections stay direct. Circuit isolation does not guarantee a unique exit IP.
"""
from __future__ import annotations

import atexit
import os
import platform
import re
import shutil
import socket
import sys
import tarfile
import tempfile
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests

from shutdown_utils import popen_no_window


class TorProxyError(RuntimeError):
    pass


_LOCK = threading.Lock()
_PROCESS = None
_PORT = None
_RUNTIME = None
DOWNLOAD_PAGE = "https://download.torproject.org/tor/"


def _root():
    return Path(os.getenv("GLOSSARION_TOR_DIR") or
                str(Path.home() / ".glossarion" / "tor"))


def _find_binary():
    override = os.getenv("GLOSSARION_TOR_BINARY", "").strip()
    if override:
        if not Path(override).is_file():
            raise TorProxyError(f"Configured Tor binary was not found: {override}")
        return str(Path(override).resolve())
    found = shutil.which("tor")
    if found:
        return found
    name = "tor.exe" if sys.platform.startswith("win") else "tor"
    roots = [_root() / "bundle"]
    if sys.platform.startswith("win"):
        roots += [Path.home() / "Desktop" / "Tor Browser" / "Browser" / "TorBrowser",
                  Path(os.getenv("LOCALAPPDATA", str(Path.home()))) / "Tor Browser" / "Browser" / "TorBrowser"]
    elif sys.platform == "darwin":
        roots += [Path("/Applications/Tor Browser.app/Contents"),
                  Path.home() / "Applications/Tor Browser.app/Contents"]
    for root in roots:
        if root.is_dir():
            for candidate in root.rglob(name):
                if candidate.is_file():
                    return str(candidate)
    return None


def _bundle_url(html):
    system = "windows" if sys.platform.startswith("win") else (
        "macos" if sys.platform == "darwin" else "linux")
    machine = platform.machine().lower()
    arch = {"amd64": "x86_64", "x86_64": "x86_64", "arm64": "aarch64",
            "aarch64": "aarch64", "i386": "i686", "i686": "i686", "x86": "i686"}.get(machine)
    if not arch:
        raise TorProxyError(f"Tor automatic installation does not support {system}/{machine}.")
    pattern = rf"tor-expert-bundle-{system}-{arch}-[0-9.]+\.tar\.gz"
    for href in re.findall(r'href=[\"\']([^\"\']+)[\"\']', html):
        url = urljoin(DOWNLOAD_PAGE, href)
        if (urlparse(url).scheme == "https" and
                urlparse(url).hostname == "dist.torproject.org" and
                re.fullmatch(pattern, url.rsplit("/", 1)[-1])):
            return url  # Stable column appears before alpha on the official page.
    raise TorProxyError(f"No stable Tor expert bundle found for {system}/{arch}.")


def _extract_bundle(archive, destination):
    with tarfile.open(archive, "r:gz") as bundle:
        if hasattr(tarfile, "data_filter"):
            bundle.extractall(destination, filter="data")
        else:
            # Older Python: never extract links or paths outside the destination.
            target = destination.resolve()
            for member in bundle.getmembers():
                resolved = (target / member.name).resolve()
                if not resolved.is_relative_to(target) or not (member.isfile() or member.isdir()):
                    raise TorProxyError("Unsafe entry in Tor archive.")
            bundle.extractall(destination)


def ensure_tor_installed(log_fn=None, cancelled=None):
    """Called under the manager lock, installing a per-user expert bundle if absent."""
    binary = _find_binary()
    if binary:
        return binary
    if log_fn:
        log_fn("Tor was not found; downloading the official expert bundle...")
    root = _root()
    root.mkdir(parents=True, exist_ok=True)
    # Explicitly avoid ambient proxies while bootstrapping the proxy itself.
    with requests.Session() as session, tempfile.TemporaryDirectory(dir=root) as stage:
        session.trust_env = False
        with session.get(DOWNLOAD_PAGE, timeout=30) as response:
            response.raise_for_status()
            url = _bundle_url(response.text)
        archive = Path(stage) / "tor.tar.gz"
        with session.get(url, stream=True, timeout=(30, 60)) as response:
            response.raise_for_status()
            with archive.open("wb") as out:
                for chunk in response.iter_content(1024 * 1024):
                    if cancelled and cancelled():
                        raise TorProxyError("Tor installation cancelled.")
                    out.write(chunk)
        extracted = Path(stage) / "bundle"
        extracted.mkdir()
        _extract_bundle(archive, extracted)
        name = "tor.exe" if sys.platform.startswith("win") else "tor"
        if not any(p.is_file() for p in extracted.rglob(name)):
            raise TorProxyError("The Tor expert bundle does not contain a Tor executable.")
        destination = root / "bundle"
        if destination.exists():
            # Preserve an incomplete previous install for inspection.
            destination.rename(root / ("bundle-incomplete-" + uuid.uuid4().hex))
        extracted.rename(destination)
    return _find_binary()


def _stop():
    global _PROCESS, _PORT, _RUNTIME
    if _PROCESS is not None:
        try:
            _PROCESS.terminate()
            _PROCESS.wait(timeout=5)
        except Exception:
            try:
                _PROCESS.kill()
                _PROCESS.wait(timeout=5)
            except Exception:
                pass
    _PROCESS = _PORT = None
    if _RUNTIME is not None:
        _RUNTIME.cleanup()
        _RUNTIME = None


atexit.register(_stop)


def new_proxy_url(log_fn=None, cancelled=None):
    """Return a unique authenticated HTTP proxy URL; never fall back to direct."""
    global _PROCESS, _PORT, _RUNTIME
    try:
        with _LOCK:
            if cancelled and cancelled():
                raise TorProxyError("Tor request cancelled.")
            if _PROCESS is None or _PROCESS.poll() is not None:
                _stop()
                binary = ensure_tor_installed(log_fn, cancelled)
                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", 0))
                    port = sock.getsockname()[1]
                _RUNTIME = tempfile.TemporaryDirectory(prefix="glossarion-tor-")
                runtime = Path(_RUNTIME.name)
                log = runtime / "tor.log"
                config = runtime / "torrc"
                config.write_text(
                    'ClientOnly 1\nSocksPort 0\n'
                    f'HTTPTunnelPort 127.0.0.1:{port} IsolateSOCKSAuth\n'
                    f'DataDirectory "{runtime.as_posix()}/data"\n'
                    f'Log notice file "{log.as_posix()}"\n', encoding="utf-8")
                if log_fn:
                    log_fn("Starting Tor; waiting for network bootstrap...")
                try:
                    _PROCESS = popen_no_window([binary, "-f", str(config)],
                                               cwd=str(Path(binary).parent),
                                               stdout=-3, stderr=-3)
                    deadline = time.monotonic() + 120
                    while time.monotonic() < deadline:
                        if cancelled and cancelled():
                            raise TorProxyError("Tor startup cancelled.")
                        status = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
                        if _PROCESS.poll() is not None:
                            raise TorProxyError("Tor exited during startup: " + status[-1500:])
                        if "Bootstrapped 100%" in status:
                            _PORT = port
                            break
                        time.sleep(0.1)
                    else:
                        raise TorProxyError("Tor did not bootstrap within 120 seconds.")
                except BaseException:
                    _stop()
                    raise
            identity = uuid.uuid4().hex
            return f"http://{identity}:tor@127.0.0.1:{_PORT}"
    except TorProxyError:
        raise
    except Exception as exc:
        raise TorProxyError(f"Unable to set up Tor: {exc}") from exc


def proxy_environment(env, proxy_url):
    """Copy CLI environment, overriding ambient proxy and bypass settings."""
    env = dict(env)
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        env[name] = env[name.lower()] = proxy_url
    env["NO_PROXY"] = env["no_proxy"] = "localhost,127.0.0.1,::1"
    return env
