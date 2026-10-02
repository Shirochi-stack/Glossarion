"""Managed Tor HTTP CONNECT proxy with a fresh circuit identity per request.

Only provider HTTPS traffic uses this proxy. Installers and loopback control
connections stay direct. Circuit isolation does not guarantee a unique exit IP.
"""
from __future__ import annotations

import atexit
from contextlib import contextmanager
import json
import os
import platform
import re
import select
import shutil
import socket
import socketserver
import subprocess
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


class TorBootstrapTimeout(TorProxyError):
    """The running Tor process is retained so retries can resume bootstrap."""


class _TorInstance:
    def __init__(self, index):
        self.index = index
        self.lock = threading.Lock()
        self.control_lock = threading.Lock()
        self.process = self.port = self.runtime = self.starting_port = self.binary = None
        self.control_port = None
        self.last_newnym = float('-inf')


_INSTANCES = [_TorInstance(index) for index in range(8)]
_POOL_LOCK = threading.Lock()
_POOL_NEXT = 0
_INSTALL_LOCK = threading.Lock()
_REQUEST_INSTANCES = {}
_BOOTSTRAP_WAIT_SECONDS = 120
DOWNLOAD_PAGE = "https://download.torproject.org/tor/"
_LAST_REQUEST_PORT = None
_RELAY_LOCK = threading.Lock()


def enabled():
    return os.getenv("GLOSSARION_TOR_ENABLED", "0").strip().lower() in ("1", "true", "yes", "on")


class _RelayHandler(socketserver.BaseRequestHandler):
    def handle(self):
        upstream = None
        try:
            upstream = socket.create_connection(self.server.upstream, timeout=30)
            pair = (self.request, upstream)
            with self.server.active_lock:
                if self.server.closing:
                    return
                self.server.active.update(pair)
            while True:
                readable, _, _ = select.select(pair, [], [], 1)
                for source in readable:
                    data = source.recv(65536)
                    if not data:
                        return
                    destination = upstream if source is self.request else self.request
                    destination.sendall(data)
        except (OSError, ValueError):
            pass  # Closing a request also closes its relay connections.
        finally:
            with self.server.active_lock:
                self.server.active.discard(self.request)
                self.server.active.discard(upstream)
            if upstream is not None:
                upstream.close()


class _RequestRelay(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = False


@contextmanager
def request_proxy(log_fn=None, cancelled=None):
    """Keep a distinct loopback port alive for the full request/stream lifetime."""
    global _LAST_REQUEST_PORT
    if not enabled():
        yield None
        return
    proxy = new_proxy_url(log_fn=log_fn, cancelled=cancelled)
    address = urlparse(proxy)
    with _RELAY_LOCK:
        for _ in range(32):
            relay = _RequestRelay(("127.0.0.1", 0), _RelayHandler)
            port = relay.server_address[1]
            if port != _LAST_REQUEST_PORT:
                break
            relay.server_close()
        else:
            raise TorProxyError("Unable to allocate a new request proxy port.")
        _LAST_REQUEST_PORT = port
    with _RELAY_LOCK:
        _REQUEST_INSTANCES[port] = next((item for item in _INSTANCES if item.port == address.port), None)
    relay.upstream = (address.hostname, address.port)
    relay.active = set()
    relay.closing = False
    relay.active_lock = threading.Lock()
    worker = threading.Thread(target=relay.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    worker.start()
    try:
        yield address._replace(netloc=f"{address.username}:{address.password}@127.0.0.1:{port}").geturl()
    finally:
        with _RELAY_LOCK:
            _REQUEST_INSTANCES.pop(port, None)
        relay.shutdown()
        relay.server_close()
        with relay.active_lock:
            relay.closing = True
            for connection in relay.active:
                try:
                    connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                connection.close()
        worker.join(timeout=2)


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
        if log_fn:
            log_fn(f"Found existing Tor installation: {binary}")
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


def _stop(instance=None):
    if instance is None:
        for item in _INSTANCES:
            with item.lock:
                _stop(item)
        return
    if instance.process is not None:
        try:
            instance.process.terminate()
            instance.process.wait(timeout=5)
        except Exception:
            try:
                instance.process.kill()
                instance.process.wait(timeout=5)
            except Exception:
                pass
    instance.process = instance.port = None
    instance.starting_port = instance.binary = None
    instance.control_port = None
    instance.last_newnym = float('-inf')
    if instance.runtime is not None:
        instance.runtime.cleanup()
        instance.runtime = None


atexit.register(_stop)


def _geoip_config(binary):
    """Locate GeoIP data beside Tor Browser, expert bundles, or system Tor."""
    base = Path(binary).resolve().parent
    directories = [base, base / "data", base.parent / "data",
                   base.parent / "Data" / "Tor", base.parent / "share" / "tor",
                   Path("/usr/share/tor"), Path("/usr/local/share/tor"),
                   Path("/opt/homebrew/share/tor")]
    lines = []
    for option, filename in (("GeoIPFile", "geoip"), ("GeoIPv6File", "geoip6")):
        for directory in directories:
            candidate = directory / filename
            if candidate.is_file():
                lines.append(f"{option} {json.dumps(str(candidate.resolve()), ensure_ascii=False)}\n")
                break
    return "".join(lines)


def _wait_for_bootstrap(log_fn, cancelled, instance=None):
    instance = instance or _INSTANCES[0]
    log = Path(instance.runtime.name) / "tor.log"
    started = time.monotonic()
    deadline = started + _BOOTSTRAP_WAIT_SECONDS
    last_progress = "no bootstrap progress reported"
    last_notice = started
    seen_warnings = set()
    status = ""
    while time.monotonic() < deadline:
        if cancelled and cancelled():
            raise TorProxyError("Tor startup cancelled.")
        status = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        progress = re.findall(r"Bootstrapped \d+%[^\r\n]*", status)
        now = time.monotonic()
        if progress and progress[-1] != last_progress:
            last_progress = progress[-1]
            last_notice = now
            if log_fn:
                log_fn(f"Tor: {last_progress}")
        elif log_fn and now - last_notice >= 10:
            log_fn(f"Tor is still starting: {last_progress} ({int(now - started)}s elapsed)")
            last_notice = now
        for warning in re.findall(r"\[(?:warn|err)\] ([^\r\n]*)", status):
            if warning not in seen_warnings:
                seen_warnings.add(warning)
                if log_fn:
                    log_fn(f"Tor warning: {warning}")
        if instance.process.poll() is not None:
            raise TorProxyError(
                f"Tor exited during startup (code {instance.process.returncode}, binary {instance.binary}): "
                + (status[-2500:] or "No console output was produced."))
        if "Bootstrapped 100%" in status:
            instance.port = instance.starting_port
            if log_fn:
                log_fn("Tor proxy is ready; sending the API request.")
            return
        time.sleep(0.1)
    raise TorBootstrapTimeout(
        f"Tor did not bootstrap within {_BOOTSTRAP_WAIT_SECONDS} seconds. "
        f"Last stage: {last_progress}. Tor is still running; retries will continue "
        f"waiting without restarting it. Recent Tor output:\n"
        f"{status[-2500:] or 'No console output was produced.'}")


def _instance_proxy_url(instance, log_fn=None, cancelled=None):
    """Return a unique authenticated HTTP proxy URL; never fall back to direct."""
    wait_started = time.monotonic()
    wait_reported = False
    try:
        # Waiting callers can still cancel while another thread bootstraps Tor.
        while not instance.lock.acquire(timeout=0.1):
            if cancelled and cancelled():
                raise TorProxyError("Tor request cancelled while waiting for startup.")
            if log_fn and not wait_reported and time.monotonic() - wait_started >= 2:
                log_fn("Tor is starting for another request; waiting for the shared proxy...")
                wait_reported = True
        try:
            if cancelled and cancelled():
                raise TorProxyError("Tor request cancelled.")
            if instance.process is None or instance.process.poll() is not None:
                _stop(instance)
                with _INSTALL_LOCK:
                    binary = ensure_tor_installed(log_fn, cancelled)
                instance.binary = binary
                with socket.socket() as sock, socket.socket() as control_sock:
                    sock.bind(("127.0.0.1", 0))
                    port = sock.getsockname()[1]
                    control_sock.bind(("127.0.0.1", 0))
                    instance.control_port = control_sock.getsockname()[1]
                instance.starting_port = port
                instance.runtime = tempfile.TemporaryDirectory(prefix="glossarion-tor-")
                runtime = Path(instance.runtime.name)
                log = runtime / "tor.log"
                config = runtime / "torrc"
                config.write_text(
                    'ClientOnly 1\nSocksPort 0\n'
                    f'ControlPort 127.0.0.1:{instance.control_port}\nCookieAuthentication 1\n'
                    f'HTTPTunnelPort 127.0.0.1:{port} IsolateSOCKSAuth\n'
                    f'DataDirectory {json.dumps(str(runtime / "data"), ensure_ascii=False)}\n'
                    'Log notice stdout\n' + _geoip_config(binary), encoding="utf-8")
                if log_fn:
                    log_fn(f"Starting Tor instance {instance.index + 1} ({binary}); waiting for network bootstrap (up to {_BOOTSTRAP_WAIT_SECONDS}s)...")
                try:
                    # Capture early configuration errors too. Tor's Log option
                    # does not parse quoted filenames like ordinary path options.
                    with log.open("wb") as console:
                        instance.process = popen_no_window([binary, "-f", str(config)],
                                                   cwd=str(Path(binary).parent),
                                                   stdout=console, stderr=subprocess.STDOUT)
                except BaseException:
                    _stop(instance)
                    raise
            if instance.port is None:
                try:
                    _wait_for_bootstrap(log_fn, cancelled, instance)
                except TorBootstrapTimeout:
                    # A timeout is a caller's wait limit, not a dead Tor process.
                    raise
                except BaseException:
                    _stop(instance)
                    raise
            identity = uuid.uuid4().hex
            return f"http://{identity}:tor@127.0.0.1:{instance.port}"
        finally:
            instance.lock.release()
    except TorProxyError:
        raise
    except Exception as exc:
        raise TorProxyError(f"Unable to set up Tor: {exc}") from exc


def new_proxy_url(log_fn=None, cancelled=None):
    """Rotate through independent Tor instances, starting each lazily."""
    global _POOL_NEXT
    try:
        size = max(1, min(8, int(os.getenv('GLOSSARION_TOR_INSTANCES', '4'))))
    except ValueError:
        size = 4
    with _POOL_LOCK:
        instance = _INSTANCES[_POOL_NEXT % size]
        _POOL_NEXT += 1
    proxy = _instance_proxy_url(instance, log_fn, cancelled)
    if log_fn:
        log_fn(f'Tor instance {instance.index + 1}/{size} selected for this request.')
    return proxy


def notify_block(proxy_url, log_fn=None):
    """Ask the affected instance for new circuits without disturbing active streams."""
    if not proxy_url:
        return
    port = urlparse(proxy_url).port
    with _RELAY_LOCK:
        instance = _REQUEST_INSTANCES.get(port)
    if instance is None:
        return
    with instance.control_lock:
        if time.monotonic() - instance.last_newnym < 10:
            if log_fn:
                log_fn('Tor NEWNYM cooldown active; the next attempt rotates to another instance.')
            return
        try:
            cookie = (Path(instance.runtime.name) / 'data' / 'control_auth_cookie').read_bytes()
            with socket.create_connection(('127.0.0.1', instance.control_port), timeout=5) as connection:
                with connection.makefile('rwb') as control:
                    for command in (f'AUTHENTICATE {cookie.hex()}', 'SIGNAL NEWNYM'):
                        control.write((command + '\r\n').encode('ascii'))
                        control.flush()
                        while True:
                            reply = control.readline(4096)
                            if not reply.startswith(b'250'):
                                raise TorProxyError('Tor control command was rejected.')
                            if reply[3:4] == b' ':
                                break
            instance.last_newnym = time.monotonic()
            if log_fn:
                log_fn(f'Tor instance {instance.index + 1}: SIGNAL NEWNYM accepted after blocked request.')
        except Exception as exc:
            # Retain the original provider error and allow its normal retry/backoff.
            if log_fn:
                log_fn(f'Tor circuit renewal failed: {exc}')


def proxy_environment(env, proxy_url):
    """Copy CLI environment, overriding ambient proxy and bypass settings."""
    env = dict(env)
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        env[name] = env[name.lower()] = proxy_url
    env["NO_PROXY"] = env["no_proxy"] = "localhost,127.0.0.1,::1"
    return env
