"""Logistics.json: a compact record of every API request, for tracking API key usage (desktop only).

One line per request in the logs folder (``GLOSSARION_LOG_DIR``, else ``<app dir>/logs``): when it
was sent, which API key (pool label, index, masked key), provider / model / endpoint, the request
parameters, the anti-duplicate settings of the section the request used (translation or glossary)
and what of them was sent, the book's output folder, and the full path of the saved payload
(Payloads/...). ``unified_api_client.UnifiedClient._save_payload`` records each request.

The file stays a valid JSON array: each record is appended before the closing ``]`` under a
lock file, so several translation / glossary processes can write to it at once. Over
``MAX_BYTES`` it is renamed ``Logistics_<date>_<time>.json`` and a new one starts.
``GLOSSARION_LOGISTICS=0`` switches it off; Glossarion Mobile never writes it.
"""

import json
import os
import threading
import time
from datetime import datetime

FILE_NAME = "Logistics.json"
MAX_BYTES = 25 * 1024 * 1024
_thread_lock = threading.Lock()


def enabled():
    if str(os.environ.get("GLOSSARION_LOGISTICS", "1")).strip().lower() in ("0", "false", "no", "off"):
        return False
    try:
        import mobile_runtime
        return not mobile_runtime.is_mobile()
    except Exception:
        return True


def logs_dir():
    folder = os.environ.get("GLOSSARION_LOG_DIR")
    if not folder:
        try:
            from app_paths import _get_app_dir
            folder = os.path.join(_get_app_dir(), "logs")
        except Exception:
            folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    return folder


def logistics_path():
    return os.path.join(logs_dir(), FILE_NAME)


def mask_key(key):
    """``AIzaSyAB...x9Qk``: enough to tell keys apart, never the key itself."""
    key = str(key or "")
    if not key:
        return None
    if len(key) <= 12:
        return key[:2] + "..." + key[-2:] if len(key) > 4 else "***"
    return f"{key[:8]}...{key[-4:]}"


class _FileLock:
    """An exclusive lock on ``<path>.lock`` (other Glossarion processes append too)."""

    def __init__(self, path, timeout=10.0):
        self.path = path + ".lock"
        self.timeout = timeout
        self.handle = None

    def __enter__(self):
        self.handle = open(self.path, "a+b")
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                if os.name == "nt":
                    import msvcrt
                    self.handle.seek(0)
                    msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except OSError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.02)

    def __exit__(self, *exc):
        try:
            if os.name == "nt":
                import msvcrt
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()


def _append(path, line):
    """Append one JSON value to the array in ``path`` (created as ``[\\n...\\n]``)."""
    data = line.encode("utf-8")
    if os.path.isfile(path) and os.path.getsize(path) > MAX_BYTES:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.replace(path, os.path.join(os.path.dirname(path), f"Logistics_{stamp}.json"))
    if not os.path.isfile(path) or os.path.getsize(path) < 2:
        with open(path, "wb") as f:
            f.write(b"[\n" + data + b"\n]\n")
        return
    with open(path, "r+b") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - 64))
        tail = f.read()
        close = tail.rfind(b"]")
        if close < 0:  # not an array we wrote: start a fresh one beside it
            f.close()
            os.replace(path, path + ".broken")
            with open(path, "wb") as fresh:
                fresh.write(b"[\n" + data + b"\n]\n")
            return
        while close > 0 and tail[close - 1:close] in (b" ", b"\t", b"\r", b"\n"):
            close -= 1  # the line break before "]" too: one record per line
        end = size - len(tail) + close
        f.seek(0)
        empty = f.read(end).strip() == b"["
        f.seek(end)
        f.truncate()
        f.write((b"" if empty else b",\n") + data + b"\n]\n")


def record(entry):
    """Append ``entry`` (a dict) to Logistics.json; never raises."""
    if not enabled():
        return None
    try:
        path = logistics_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        line = json.dumps({k: v for k, v in entry.items() if v not in (None, "", {}, [])},
                          ensure_ascii=False, separators=(",", ":"), default=str)
        with _thread_lock, _FileLock(path):
            _append(path, line)
        return path
    except Exception:
        return None
