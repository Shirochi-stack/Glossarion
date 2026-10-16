"""GGUF downloads (resumable, cancellable, verified) and the registry of downloaded models.

Storage (``models_dir()`` = ``GLOSSARION_MODELS_DIR``, ``<data>/models`` on the device):

* ``<stem>.gguf``: a verified model; ``<stem>`` is the file name without ``.gguf`` (``-<owner>`` is
  appended when another repository already brought that name). The model id is ``local/<stem>``.
* ``models.json``: the registry ``{stem: {repo, file, revision, sha256, size, downloaded_at}}``,
  written atomically (temp file, fsync, ``os.replace``).
* ``<stem>.gguf.part`` + ``<stem>.gguf.part.json``: an unfinished download and what it is
  (``{url, repo, file, revision, size, sha256, stem}``). Pause keeps both, Cancel deletes both; a part
  found at start is listed as paused and never resumes on its own.

A transfer resumes with ``Range: bytes=<have>-`` (206 appends; 200 means the server ignored the
range: start over; 416 with every byte present goes straight to the check), retries transient errors
(connection, timeout, 429, 5xx) five times with 2/4/8/16/30 s back-off, re-hashes an existing part
first so the final sha256 covers every byte, and checks the size and Hugging Face's sha256 before
the file is renamed into place. Downloads run outside JobService: a 5 GB model must not queue behind
a translation, and a killed transfer is cheap to resume.

``DownloadManager`` keeps the plumbing of the Ollama ``PullManager`` it replaces: one worker thread per
model (``run_io``), progress marshalled to the event loop with ``call_soon_threadsafe`` and throttled,
``subscribe`` / unsubscribe.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import logging
import os
import shutil
import threading
import time
from typing import Any, Callable, Optional

from glossarion_mobile.services.local_llm import hf

log = logging.getLogger("glossarion.local_llm")

__all__ = [
    "DownloadCancelled", "DownloadError", "DownloadManager", "DownloadSpec", "DownloadState", "Registry",
    "format_bytes", "models_dir",
]

MODELS_ENV = "GLOSSARION_MODELS_DIR"
REGISTRY_NAME = "models.json"
PART_SUFFIX = ".part"
CHUNK = 1024 * 1024
CONNECT_TIMEOUT = 30.0
READ_TIMEOUT = 60.0
BACKOFF = (2, 4, 8, 16, 30)
SPARE_BYTES = 256 * 1024 * 1024
CONFIRM_BYTES = 1_000_000_000  # "Download 5.7 GB from Hugging Face?"
#: ``shutil.disk_usage`` (tests replace it)
disk_usage = shutil.disk_usage


def models_dir() -> str:
    path = os.environ.get(MODELS_ENV, "").strip()
    if not path:
        base = os.environ.get("GLOSSARION_DATA_DIR", "").strip() or os.path.join(os.path.expanduser("~"), ".glossarion")
        path = os.path.join(base, "models")
    return os.path.abspath(path)


def format_bytes(count: Any) -> str:
    try:
        value = float(count or 0)
    except (TypeError, ValueError):
        value = 0.0
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1000 or unit == "TB":
            if unit in ("B", "KB"):
                return f"{value:.0f} {unit}"
            return f"{value:.1f} {unit}" if value < 100 else f"{value:.0f} {unit}"
        value /= 1000.0
    return f"{value:.1f} TB"


def _atomic_write_json(path: str, data: Any) -> None:
    from glossarion_mobile.state.prefs import atomic_write_json

    atomic_write_json(path, data)


# ---- registry ---------------------------------------------------------------------------------------


class Registry:
    """``models.json`` of one models folder (thread-safe; re-read on every call: cheap and always fresh)."""

    def __init__(self, root: Optional[str] = None) -> None:
        self.root = os.path.abspath(root) if root else models_dir()
        self._lock = threading.RLock()

    @property
    def path(self) -> str:
        return os.path.join(self.root, REGISTRY_NAME)

    def file_path(self, stem: str) -> str:
        return os.path.join(self.root, f"{stem}.gguf")

    def load(self) -> dict:
        with self._lock:
            try:
                with open(self.path, encoding="utf-8") as handle:
                    data = json.load(handle)
            except (OSError, ValueError):
                return {}
            if not isinstance(data, dict):
                return {}
            out: dict = {}
            for stem, entry in data.items():
                if isinstance(entry, dict) and os.path.isfile(self.file_path(str(stem))):
                    out[str(stem)] = dict(entry)
            return out

    def save(self, entries: dict) -> None:
        with self._lock:
            os.makedirs(self.root, exist_ok=True)
            _atomic_write_json(self.path, entries)

    def add(self, stem: str, entry: dict) -> None:
        with self._lock:
            entries = self.load()
            entries[stem] = dict(entry)
            self.save(entries)

    def remove(self, stem: str) -> bool:
        with self._lock:
            entries = self.load()
            found = entries.pop(stem, None) is not None
            try:
                os.remove(self.file_path(stem))
                found = True
            except FileNotFoundError:
                pass
            self.save(entries)
            return found

    def resolve(self, name: Any) -> Optional[str]:
        """The registered stem for ``name`` (``local/`` optional; case-insensitive), else None."""
        from glossarion_mobile.services.local_llm import stem_of

        wanted = stem_of(name).casefold()
        if not wanted:
            return None
        for stem in self.load():
            if stem.casefold() == wanted:
                return stem
        return None

    def find(self, repo: str, file: str) -> Optional[str]:
        """The stem of ``repo``'s ``file`` when it is downloaded."""
        for stem, entry in self.load().items():
            if str(entry.get("repo", "")).casefold() == str(repo).casefold() and entry.get("file") == file:
                return stem
        return None

    def stem_for(self, repo: str, file: str) -> str:
        """The stem a download of ``repo``/``file`` gets (existing entry, else a free name)."""
        existing = self.find(repo, file)
        if existing:
            return existing
        base = file.rsplit("/", 1)[-1]
        stem = base[:-5] if base.lower().endswith(".gguf") else base
        entries = self.load()
        taken = {s.casefold() for s in entries}
        part = self.pending_parts()
        taken |= {s.casefold() for s, meta in part.items()
                  if not (meta.get("repo") == repo and meta.get("file") == file)}
        if stem.casefold() not in taken:
            return stem
        owner = repo.split("/", 1)[0]
        candidate = f"{stem}-{owner}"
        number = 2
        while candidate.casefold() in taken:
            candidate = f"{stem}-{owner}{number}"
            number += 1
        return candidate

    def pending_parts(self) -> dict:
        """``{stem: sidecar}`` of unfinished downloads in the folder."""
        out: dict = {}
        try:
            names = os.listdir(self.root)
        except OSError:
            return out
        for name in names:
            if not name.endswith(".gguf.part.json"):
                continue
            stem = name[: -len(".gguf.part.json")]
            try:
                with open(os.path.join(self.root, name), encoding="utf-8") as handle:
                    meta = json.load(handle)
            except (OSError, ValueError):
                continue
            if isinstance(meta, dict):
                meta = dict(meta)
                try:
                    meta["have"] = os.path.getsize(os.path.join(self.root, f"{stem}.gguf{PART_SUFFIX}"))
                except OSError:
                    meta["have"] = 0
                out[stem] = meta
        return out


# ---- downloads --------------------------------------------------------------------------------------


class DownloadError(Exception):
    """A download that cannot go on (the message is what the row shows)."""


class DownloadCancelled(Exception):
    def __init__(self, mode: str) -> None:
        super().__init__(mode)
        self.mode = mode  # "pause" | "cancel"


@dataclasses.dataclass
class DownloadSpec:
    repo: str
    file: str
    revision: str
    size: int
    sha256: Optional[str]
    stem: str

    @property
    def url(self) -> str:
        return hf.resolve_url(self.repo, self.revision, self.file)


@dataclasses.dataclass
class DownloadState:
    stem: str
    repo: str = ""
    file: str = ""
    status: str = "Starting…"
    phase: str = "checking"  # checking | downloading | verifying | done
    bytes_done: int = 0
    bytes_total: int = 0
    speed: float = 0.0  # bytes per second
    eta: Optional[float] = None  # seconds
    running: bool = True
    paused: bool = False
    cancelled: bool = False
    done: bool = False
    error: Optional[str] = None

    @property
    def fraction(self) -> Optional[float]:
        if self.done:
            return 1.0
        if self.bytes_total <= 0:
            return None
        return max(0.0, min(1.0, self.bytes_done / self.bytes_total))

    @property
    def percent(self) -> int:
        fraction = self.fraction
        return int(fraction * 100) if fraction is not None else 0


def _transient(exc: BaseException) -> bool:
    try:
        import requests
    except ImportError:  # pragma: no cover
        return isinstance(exc, (ConnectionError, TimeoutError))
    return isinstance(exc, (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError,
                            ConnectionError, TimeoutError))


class _HttpStatus(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(status)
        self.status = status


def download_blocking(spec: DownloadSpec, registry: Registry, *, progress: Callable[[dict], None],
                      should_stop: Callable[[], Optional[str]], session: Any = None,
                      sleep: Optional[Callable[[float], bool]] = None) -> dict:
    """Download ``spec`` into ``registry``'s folder (blocking; worker thread). ``should_stop()`` returns
    "pause" / "cancel" / None; ``sleep(seconds)`` returns True when interrupted. Returns the registry
    entry; raises ``DownloadCancelled`` / ``DownloadError``."""
    import requests

    root = registry.root
    os.makedirs(root, exist_ok=True)
    final = registry.file_path(spec.stem)
    part = final + PART_SUFFIX
    sidecar = part + ".json"
    meta = {"url": spec.url, "repo": spec.repo, "file": spec.file, "revision": spec.revision,
            "size": spec.size, "sha256": spec.sha256, "stem": spec.stem}
    try:
        with open(sidecar, encoding="utf-8") as handle:
            old = json.load(handle)
        if not isinstance(old, dict) or old.get("sha256") != spec.sha256 or old.get("size") != spec.size \
                or old.get("file") != spec.file or old.get("repo") != spec.repo:
            _remove(part)
    except (OSError, ValueError):
        _remove(part)  # a part without its sidecar is of unknown origin
    _atomic_write_json(sidecar, meta)

    def stop_check() -> None:
        mode = should_stop()
        if mode:
            raise DownloadCancelled(mode)

    have = _size(part)
    if spec.size and have > spec.size:
        _remove(part)
        have = 0
    free = int(disk_usage(root).free)
    needed = max(0, int(spec.size) - have) + SPARE_BYTES
    if spec.size and free < needed:
        raise DownloadError(f"Not enough storage: {format_bytes(needed)} needed, {format_bytes(free)} free")

    hasher = hashlib.sha256()
    if have:
        progress({"phase": "checking", "status": "Checking the downloaded part…", "bytes_done": 0,
                  "bytes_total": have})
        done = 0
        with open(part, "rb") as handle:
            while True:
                stop_check()
                block = handle.read(CHUNK * 4)
                if not block:
                    break
                hasher.update(block)
                done += len(block)
                progress({"phase": "checking", "bytes_done": done, "bytes_total": have})
    total = int(spec.size or 0)
    attempt = 0
    wait = sleep or (lambda seconds: (time.sleep(seconds), False)[1])
    client = session or requests
    while True:
        stop_check()
        if total and have >= total:
            break
        headers = {"Accept-Encoding": "identity", "User-Agent": "Glossarion-Mobile"}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            response = client.get(spec.url, headers=headers, stream=True, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
                                  allow_redirects=True)
            try:
                status = response.status_code
                if status == 416 and total and have >= total:
                    break
                if status == 200 and have:
                    # the server ignored the range: start over
                    have = 0
                    hasher = hashlib.sha256()
                    _truncate(part)
                elif status == 206:
                    start = _content_range_start(response.headers.get("Content-Range"))
                    if start is not None and start != have:
                        raise DownloadError(f"Hugging Face resumed at byte {start}, not {have}")
                elif status != 200:
                    raise _HttpStatus(status)
                if not total:
                    total = _total_from(response, have)
                last = 0.0
                with open(part, "ab") as out:
                    for block in response.iter_content(chunk_size=CHUNK):
                        if not block:
                            continue
                        out.write(block)
                        hasher.update(block)
                        have += len(block)
                        now = time.monotonic()
                        if now - last >= 0.1:
                            last = now
                            progress({"phase": "downloading", "bytes_done": have, "bytes_total": total})
                            stop_check()
                    out.flush()
                    os.fsync(out.fileno())
                attempt = 0
                if total and have < total:
                    raise ConnectionError("the connection ended early")
                if not total:
                    total = have
                break
            finally:
                try:
                    response.close()
                except Exception:
                    pass
        except DownloadCancelled:
            raise
        except DownloadError:
            raise
        except _HttpStatus as exc:
            if exc.status in (401, 403, 404):
                raise DownloadError(str(hf._http_error(exc.status, spec.repo))) from None
            if exc.status != 429 and exc.status < 500:
                raise DownloadError(f"Hugging Face answered HTTP {exc.status}") from None
            reason = f"HTTP {exc.status}"
        except Exception as exc:
            if not _transient(exc):
                raise DownloadError(f"Download failed: {exc}") from exc
            reason = type(exc).__name__
        if attempt >= len(BACKOFF):
            raise DownloadError(f"Download failed after {attempt + 1} tries ({reason})")
        delay = BACKOFF[attempt]
        attempt += 1
        have = _size(part)
        progress({"phase": "downloading", "status": f"Connection problem ({reason}); retrying in {delay} s…",
                  "bytes_done": have, "bytes_total": total})
        if wait(delay):
            stop_check()
        # re-hash what is on disk (the failed write may have left a partial block)
        hasher = hashlib.sha256()
        with open(part, "rb") as handle:
            for block in iter(lambda: handle.read(CHUNK * 4), b""):
                hasher.update(block)
        have = _size(part)

    progress({"phase": "verifying", "status": "Verifying…", "bytes_done": have, "bytes_total": total or have})
    digest = hasher.hexdigest()
    if (spec.size and have != spec.size) or (spec.sha256 and digest != spec.sha256.lower()):
        _remove(part)
        _remove(sidecar)
        raise DownloadError("checksum mismatch")
    os.replace(part, final)
    _remove(sidecar)
    entry = {"repo": spec.repo, "file": spec.file, "revision": spec.revision, "sha256": digest, "size": have,
             "downloaded_at": int(time.time())}
    registry.add(spec.stem, entry)
    return entry


def _content_range_start(value: Any) -> Optional[int]:
    text = str(value or "")
    if not text.startswith("bytes "):
        return None
    try:
        return int(text[6:].split("-", 1)[0])
    except ValueError:
        return None


def _total_from(response: Any, have: int) -> int:
    for header in ("X-Linked-Size",):
        try:
            value = int(response.headers.get(header) or 0)
            if value:
                return value
        except ValueError:
            pass
    text = str(response.headers.get("Content-Range") or "")
    if "/" in text:
        try:
            return int(text.rsplit("/", 1)[1])
        except ValueError:
            pass
    try:
        length = int(response.headers.get("Content-Length") or 0)
    except ValueError:
        length = 0
    return have + length if response.status_code == 206 else length


def _size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError:
        log.debug("could not remove %s", path, exc_info=True)


def _truncate(path: str) -> None:
    with open(path, "wb"):
        pass


class DownloadManager:
    """Downloads on worker threads, one per model at a time.

    ``listener(state)`` runs on the event loop: progress at most every ``throttle`` seconds, phase
    changes and the end always. ``download_fn`` (tests) replaces :func:`download_blocking`."""

    def __init__(self, *, registry: Optional[Registry] = None, run_io: Optional[Callable[..., Any]] = None,
                 throttle: float = 0.25, session: Any = None,
                 download_fn: Optional[Callable[..., dict]] = None) -> None:
        self.registry = registry or Registry()
        self._run_io = run_io
        self.throttle = throttle
        self.session = session
        self._download = download_fn or download_blocking
        self.states: dict = {}
        self._stops: dict = {}
        self._wakes: dict = {}
        self.listeners: list = []
        self._scanned = False

    def subscribe(self, listener: Callable[[DownloadState], Any]) -> Callable[[], None]:
        self.listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self.listeners:
                self.listeners.remove(listener)

        return unsubscribe

    def _emit(self, state: DownloadState) -> None:
        for listener in list(self.listeners):
            try:
                listener(state)
            except Exception:
                log.exception("download listener failed")

    def scan(self) -> list:
        """Unfinished downloads in the folder as paused states (once; later calls return the list)."""
        if not self._scanned:
            self._scanned = True
            for stem, meta in self.registry.pending_parts().items():
                if stem in self.states:
                    continue
                total = int(meta.get("size") or 0)
                have = int(meta.get("have") or 0)
                self.states[stem] = DownloadState(
                    stem, str(meta.get("repo") or ""), str(meta.get("file") or ""),
                    status=f"Paused · {format_bytes(have)} of {format_bytes(total)}", phase="downloading",
                    bytes_done=have, bytes_total=total, running=False, paused=True)
        return [s for s in self.states.values() if s.paused]

    def pending_spec(self, stem: str) -> Optional[DownloadSpec]:
        """The spec of a paused part (from its sidecar) for Resume."""
        meta = self.registry.pending_parts().get(stem)
        if not meta:
            return None
        try:
            return DownloadSpec(str(meta["repo"]), str(meta["file"]), str(meta["revision"]), int(meta.get("size") or 0),
                                meta.get("sha256") or None, stem)
        except (KeyError, TypeError, ValueError):
            return None

    def running(self, stem: str) -> bool:
        state = self.states.get(stem)
        return bool(state and state.running)

    def any_running(self) -> bool:
        return any(s.running for s in self.states.values())

    def _stop(self, stem: str, mode: str) -> bool:
        holder = self._stops.get(stem)
        if holder is None or not self.running(stem):
            return False
        if holder[0] == mode or holder[0] == "cancel":  # already asked (a listener reacting to its own emit)
            return True
        holder[0] = mode
        wake = self._wakes.get(stem)
        if wake is not None:
            wake.set()
        state = self.states[stem]
        state.status = "Pausing…" if mode == "pause" else "Cancelling…"
        self._emit(state)
        return True

    def pause(self, stem: str) -> bool:
        return self._stop(stem, "pause")

    def cancel(self, stem: str) -> bool:
        """Stop a running download and delete its part; a paused part is deleted at once."""
        if self._stop(stem, "cancel"):
            return True
        state = self.states.get(stem)
        part = self.registry.file_path(stem) + PART_SUFFIX
        existed = os.path.exists(part) or os.path.exists(part + ".json")
        _remove(part)
        _remove(part + ".json")
        if state is not None and not state.running:
            state.paused = False
            state.cancelled = True
            state.status = "Cancelled"
            self.states.pop(stem, None)
            self._emit(state)
        return existed

    async def download(self, spec: DownloadSpec) -> DownloadState:
        """Download ``spec``; returns the final state. A second call for a running model returns its
        running state at once."""
        if self.running(spec.stem):
            return self.states[spec.stem]
        state = DownloadState(spec.stem, spec.repo, spec.file, bytes_total=int(spec.size or 0))
        self.states[spec.stem] = state
        holder: list = [None]
        wake = threading.Event()
        self._stops[spec.stem] = holder
        self._wakes[spec.stem] = wake
        loop = asyncio.get_running_loop()
        last = [0.0, 0, 0.0]  # emit time, bytes at the last speed sample, its time

        def apply(update: dict) -> None:
            if not state.running:
                return
            phase = update.get("phase", state.phase)
            changed = phase != state.phase or "status" in update
            state.phase = phase
            if "bytes_done" in update:
                state.bytes_done = int(update["bytes_done"])
            if "bytes_total" in update and update["bytes_total"]:
                state.bytes_total = int(update["bytes_total"])
            now = time.monotonic()
            if phase == "downloading":
                if last[2] and now - last[2] >= 0.5:
                    sample = (state.bytes_done - last[1]) / (now - last[2])
                    state.speed = sample if not state.speed else 0.7 * state.speed + 0.3 * sample
                    last[1], last[2] = state.bytes_done, now
                elif not last[2]:
                    last[1], last[2] = state.bytes_done, now
                remaining = max(0, state.bytes_total - state.bytes_done)
                state.eta = remaining / state.speed if state.speed > 1 else None
            if "status" in update:
                state.status = str(update["status"])
            elif phase == "downloading":
                state.status = _progress_text(state)
            elif phase == "checking":
                state.status = f"Checking the downloaded part… {state.percent}%"
            if changed or now - last[0] >= self.throttle:
                last[0] = now
                self._emit(state)

        def progress(update: dict) -> None:  # worker thread
            try:
                loop.call_soon_threadsafe(apply, dict(update))
            except RuntimeError:  # loop closed
                pass

        def sleep(seconds: float) -> bool:
            return wake.wait(seconds)

        def work() -> dict:
            return self._download(spec, self.registry, progress=progress, should_stop=lambda: holder[0],
                                  session=self.session, sleep=sleep)

        self._emit(state)
        try:
            if self._run_io is not None:
                await self._run_io(work)
            else:
                await asyncio.to_thread(work)
            await asyncio.sleep(0)  # let queued progress land before the final state
            state.done = True
            state.phase = "done"
            state.bytes_done = state.bytes_total = max(state.bytes_total, state.bytes_done)
            state.status = "Downloaded ✓"
        except DownloadCancelled as exc:
            await asyncio.sleep(0)
            if exc.mode == "pause":
                state.paused = True
                state.bytes_done = _size(self.registry.file_path(spec.stem) + PART_SUFFIX)
                state.status = f"Paused · {format_bytes(state.bytes_done)} of {format_bytes(state.bytes_total)}"
            else:
                state.cancelled = True
                part = self.registry.file_path(spec.stem) + PART_SUFFIX
                _remove(part)
                _remove(part + ".json")
                state.status = "Cancelled"
        except Exception as exc:
            await asyncio.sleep(0)
            state.error = str(exc) or type(exc).__name__
            state.status = f"Failed: {state.error}"
            log.info("download of %s failed: %s", spec.stem, state.error)
        finally:
            state.running = False
            self._stops.pop(spec.stem, None)
            self._wakes.pop(spec.stem, None)
        if state.cancelled:
            self.states.pop(spec.stem, None)
        self._emit(state)
        return state


def _progress_text(state: DownloadState) -> str:
    text = f"{state.percent}% · {format_bytes(state.bytes_done)} of {format_bytes(state.bytes_total)}"
    if state.speed > 1:
        text += f" · {format_bytes(state.speed)}/s"
    if state.eta is not None and state.eta < 360000:
        minutes, seconds = divmod(int(state.eta), 60)
        hours, minutes = divmod(minutes, 60)
        text += f" · {hours}:{minutes:02d}:{seconds:02d} left" if hours else f" · {minutes}:{seconds:02d} left"
    return text
