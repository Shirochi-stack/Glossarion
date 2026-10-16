"""Ollama on the phone through Termux, model presets and pulls (Model manager › Ollama).

A phone cannot run the desktop's managed Ollama (``ollamapull`` installs and starts a binary). Termux
can: ``pkg install ollama`` then ``ollama serve`` gives the usual Ollama API on
``http://127.0.0.1:11434``, which the shared ``ollamapull/`` route already talks to (its address is
``OLLAMAPULL_BASE_URL``, ``state.remote_routes``; empty means that loopback address). This module is
the Flet-free part:

* **Termux commands**: the one-tap setup command run through Termux's ``com.termux.RUN_COMMAND``
  intent (``flet_glossarion_native`` ``termux_run``; Termux must allow it with
  ``allow-external-apps=true``), with the copy-and-open-Termux fallback and the manual steps.
* **Presets** (``PREF_KEY`` in Prefs, ``mobile_state.json``): named model entries the Model manager
  lists, edits, renames and removes. ``DEFAULT_PRESETS`` are only the first-run data (the Qwen3.5
  small models of ``model_options.LOCAL_CHAT_MODEL_TAGS``); a saved list (even an empty one) wins.
* **Pulls**: the shared ``ollamapull.pull_model`` (``/api/pull`` with its own progress, speed and ETA
  text) run off the UI loop, its progress marshalled back to the loop and throttled.

Model names go through the shared ``ollamapull.normalize_pull_name``: an Ollama library tag or a
Hugging Face GGUF repo (``hf.co/<user>/<repo>[:<quant>]``).
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import logging
import re
import threading
import time
from typing import Any, Callable, Optional, Sequence

log = logging.getLogger("glossarion.ollama_termux")

__all__ = [
    "DEFAULT_PRESETS", "PREF_KEY", "PullManager", "PullState", "SETUP_COMMAND", "TERMUX_ALLOW_COMMAND",
    "TERMUX_URL", "TermuxOutcome", "installed", "is_loopback", "load_presets", "manual_steps", "normalize_model",
    "pull_command", "restore_defaults", "route_model", "run_in_termux", "save_presets", "server_status",
    "validate_preset",
]

#: The Ollama server ``ollama serve`` starts inside Termux (the ``ollamapull`` default address).
TERMUX_URL = "http://127.0.0.1:11434"
PREF_KEY = "ollama_presets"
TERMUX_PACKAGE = "com.termux"
TERMUX_FDROID_URL = "https://f-droid.org/packages/com.termux/"
#: Run once in Termux so other apps (Glossarion) may send it commands (RUN_COMMAND).
TERMUX_ALLOW_COMMAND = ("mkdir -p ~/.termux && (grep -qs '^allow-external-apps *= *true' ~/.termux/termux.properties "
                        "|| echo 'allow-external-apps=true' >> ~/.termux/termux.properties) && termux-reload-settings")
INSTALL_COMMAND = "pkg install -y ollama"
SERVE_COMMAND = "ollama serve"
#: The one-tap setup: keep the CPU awake, install Ollama when missing, start its server (it keeps running in
#: that Termux session; the app pulls and chats through http://127.0.0.1:11434).
SETUP_COMMAND = ("termux-wake-lock; command -v ollama >/dev/null 2>&1 || pkg install -y ollama; "
                 "ollama serve")
#: What the copy-and-paste fallback pastes for the setup: the setup itself plus allow-external-apps, so the
#: first manual paste also turns on one-tap setup for every later run.
FIRST_RUN_SETUP_COMMAND = TERMUX_ALLOW_COMMAND + "; " + SETUP_COMMAND
MAX_PRESETS = 50

#: First-run presets (data, not UI): the Qwen3.5 small models of the Ollama library
#: (``model_options.LOCAL_CHAT_MODEL_TAGS``). Users rename, edit, remove and add their own.
DEFAULT_PRESETS: tuple = (
    {"name": "Garbage", "model": "qwen3.5:0.8b", "note": "Qwen3.5 0.8B · fastest, rough translations"},
    {"name": "Light", "model": "qwen3.5:2b", "note": "Qwen3.5 2B · quick drafts on most phones"},
    {"name": "Medium", "model": "qwen3.5:4b", "note": "Qwen3.5 4B · better quality, needs about 4 GB free RAM"},
    {"name": "Heavy", "model": "qwen3.5:9b", "note": "Qwen3.5 9B · best of the four, needs about 8 GB free RAM"},
)


def _pull_module() -> Any:
    import ollamapull  # shared (src/ollamapull.py)

    return ollamapull


def normalize_model(text: str) -> tuple:
    """(name for ``ollama pull``, None) or (None, reason) — the shared ``ollamapull.normalize_pull_name``."""
    module = _pull_module()
    try:
        return module.normalize_pull_name(text), None
    except module.OllamaPullError as exc:
        return None, str(exc)


def route_model(model: str) -> str:
    """The model id the app uses for a pulled model: ``ollamapull/<name>``."""
    name = str(model or "").strip()
    return name if name.lower().startswith("ollamapull/") else f"ollamapull/{name}"


def pull_command(model: str) -> str:
    return f"ollama pull {model}"


def is_loopback(url: str) -> bool:
    text = str(url or "").strip().lower()
    return not text or bool(re.match(r"^(https?://)?(127\.0\.0\.1|localhost|\[::1\])(:\d+)?/?$", text))


def manual_steps(model: Optional[str] = None) -> list:
    """(title, command or None) rows of the manual Termux setup, in order."""
    rows: list = [
        ("Install Termux from F-Droid (the Play Store build is outdated) and open it once.", None),
        ("Install Ollama:", INSTALL_COMMAND),
        ("Start the server and leave Termux running:", SERVE_COMMAND),
    ]
    if model:
        rows.append(("Optional, download in Termux instead of here:", pull_command(model)))
    rows.append(("Optional, let Glossarion start Termux commands with one tap "
                 "(then allow “Run commands in Termux” for Glossarion):", TERMUX_ALLOW_COMMAND))
    return rows


# ---- presets ----------------------------------------------------------------------------------------


def _clean(row: Any) -> Optional[dict]:
    if not isinstance(row, dict):
        return None
    name = str(row.get("name") or "").strip()
    model = str(row.get("model") or "").strip()
    if not name or not model:
        return None
    return {"name": name[:60], "model": model, "note": str(row.get("note") or "").strip()[:200]}


def load_presets(prefs: Any) -> list:
    """The saved presets (Prefs ``PREF_KEY``); the defaults until the user saved a list."""
    saved = None
    try:
        saved = prefs.get(PREF_KEY, None) if prefs is not None else None
    except Exception:
        saved = None
    if not isinstance(saved, list):
        return [dict(p) for p in DEFAULT_PRESETS]
    return [row for row in (_clean(r) for r in saved) if row is not None]


def save_presets(prefs: Any, presets: Sequence[dict]) -> list:
    rows = [row for row in (_clean(r) for r in presets) if row is not None][:MAX_PRESETS]
    if prefs is not None:
        prefs.set(PREF_KEY, copy.deepcopy(rows))
    return rows


def restore_defaults(prefs: Any) -> list:
    return save_presets(prefs, [dict(p) for p in DEFAULT_PRESETS])


def validate_preset(row: dict, presets: Sequence[dict], index: Optional[int] = None) -> tuple:
    """(cleaned row, None) or (None, message). ``index``: the row being edited (its own name may stay)."""
    name = str(row.get("name") or "").strip()
    if not name:
        return None, "Give the preset a name"
    if len(name) > 60:
        return None, "Keep the name under 60 characters"
    for i, other in enumerate(presets):
        if i != index and str(other.get("name") or "").strip().casefold() == name.casefold():
            return None, f"A preset is already called {name}"
    model, error = normalize_model(str(row.get("model") or ""))
    if error:
        return None, error
    return {"name": name, "model": model, "note": str(row.get("note") or "").strip()[:200]}, None


# ---- Termux -----------------------------------------------------------------------------------------


@dataclasses.dataclass
class TermuxOutcome:
    """How a Termux command went: ``mode`` "intent" (sent with RUN_COMMAND), "clipboard" (copied for
    pasting), or "missing" (Termux is not installed); ``message`` is what the screen says."""

    mode: str
    ok: bool
    message: str
    error: Optional[str] = None


async def _maybe_await(value: Any) -> Any:
    return await value if asyncio.iscoroutine(value) else value


async def run_in_termux(native: Any, command: str, *, label: str = "Glossarion", background: bool = False,
                        copy_text: Optional[Callable[[str], Any]] = None,
                        fallback_command: Optional[str] = None) -> TermuxOutcome:
    """Send ``command`` to Termux (RUN_COMMAND); otherwise copy it (or ``fallback_command``) and open
    Termux for pasting.

    ``native``: the app's ``NativeBridge`` (None or a stub off-device)."""
    result: Any = None
    if native is not None and not getattr(native, "is_stub", False):
        try:
            result = await native.termux_run(command, label=label, background=background)
        except Exception as exc:  # never let the bridge break the screen
            log.info("termux_run failed: %s", exc)
            result = {"ok": False, "error": "failed", "message": str(exc)}
    error = str(result.get("error") or "") if isinstance(result, dict) else "unavailable"
    if isinstance(result, dict) and result.get("ok"):
        if not background:
            await _open_termux(native)
        return TermuxOutcome("intent", True, "Sent to Termux. Watch it there; come back when it says "
                                             "it is listening, then tap Check. If nothing runs, turn on "
                                             "allow-external-apps (Termux steps).")
    if error == "not_installed":
        return TermuxOutcome("missing", False, "Termux is not installed. Get it from F-Droid, open it once, "
                                               "then try again.", error)
    command = fallback_command or command
    copied = False
    if copy_text is not None:
        try:
            await _maybe_await(copy_text(command))
            copied = True
        except Exception as exc:
            log.info("copy failed: %s", exc)
    if error == "permission" and isinstance(result, dict) and result.get("requested"):
        # Android is showing its permission prompt over Glossarion: opening Termux now would hide it.
        return TermuxOutcome("clipboard", copied, "Allow Glossarion to “Run commands in Termux” in the prompt, then "
                                                  "tap “Set up in Termux” again. "
                                                  + ("The command is also copied for pasting in Termux." if copied
                                                     else f"Or type in Termux: {command}"), error)
    opened = await _open_termux(native)
    if error == "permission":
        lead = ("Allow Glossarion to “Run commands in Termux” (Android asked you, or App info › Permissions), "
                "and turn on allow-external-apps in Termux for one-tap setup. ")
    else:
        lead = ""
    where = "Termux is open: " if opened else "Open Termux, "
    action = "long-press › Paste, then press Enter." if copied else f"type: {command}"
    return TermuxOutcome("clipboard", copied, lead + ("The command is copied. " if copied else "") + where + action,
                         error or None)


async def _open_termux(native: Any) -> bool:
    if native is None or getattr(native, "is_stub", False):
        return False
    try:
        return bool(await native.open_termux())
    except Exception:
        return False


# ---- pulls ------------------------------------------------------------------------------------------

_PERCENT_RE = re.compile(r"(\d{1,3})%")


@dataclasses.dataclass
class PullState:
    model: str
    status: str = "Starting…"
    fraction: Optional[float] = None
    running: bool = True
    error: Optional[str] = None
    cancelled: bool = False
    done: bool = False


class PullManager:
    """Pulls through the shared ``ollamapull.pull_model`` on a worker thread; one per model at a time.

    ``listener(state)`` runs on the event loop: progress at most every ``throttle`` seconds, the end
    always. ``pull`` (tests) replaces ``ollamapull.pull_model``; ``run_io(fn)`` runs it off the loop."""

    def __init__(self, *, pull: Optional[Callable[..., Any]] = None,
                 run_io: Optional[Callable[..., Any]] = None, throttle: float = 0.25) -> None:
        self._pull = pull
        self._run_io = run_io
        self.throttle = throttle
        self.states: dict = {}
        self._stops: dict = {}
        self.listeners: list = []

    def subscribe(self, listener: Callable[[PullState], Any]) -> Callable[[], None]:
        self.listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self.listeners:
                self.listeners.remove(listener)

        return unsubscribe

    def _emit(self, state: PullState) -> None:
        for listener in list(self.listeners):
            try:
                listener(state)
            except Exception:
                log.exception("pull listener failed")

    def running(self, model: str) -> bool:
        state = self.states.get(model)
        return bool(state and state.running)

    def cancel(self, model: str) -> bool:
        stop = self._stops.get(model)
        if stop is None or not self.running(model):
            return False
        stop.set()
        state = self.states[model]
        state.status = "Stopping…"
        self._emit(state)
        return True

    async def pull(self, model: str) -> PullState:
        """Pull ``model`` (already normalised); returns its final state. A second call for a model
        that is still pulling returns the running state at once."""
        if self.running(model):
            return self.states[model]
        module = None
        pull_fn = self._pull
        if pull_fn is None:
            module = _pull_module()
            pull_fn = module.pull_model
        state = PullState(model)
        self.states[model] = state
        stop = threading.Event()
        self._stops[model] = stop
        loop = asyncio.get_running_loop()
        last = [0.0]

        def apply(message: str) -> None:
            if not state.running:
                return
            text = str(message or "")
            if text.startswith(f"{model}: "):
                text = text[len(model) + 2:]
            state.status = text or state.status
            match = None
            for match in _PERCENT_RE.finditer(text):
                pass
            if match is not None:
                state.fraction = max(0.0, min(1.0, int(match.group(1)) / 100))
            now = time.monotonic()
            if now - last[0] >= self.throttle:
                last[0] = now
                self._emit(state)

        def progress(message: str) -> None:  # worker thread
            try:
                loop.call_soon_threadsafe(apply, message)
            except RuntimeError:  # loop closed
                pass

        def work() -> None:
            pull_fn(model, progress=progress, should_stop=stop.is_set)

        self._emit(state)
        try:
            if self._run_io is not None:
                await self._run_io(work)
            else:
                await asyncio.to_thread(work)
            state.done = True
            state.fraction = 1.0
            state.status = "Downloaded ✓"
        except Exception as exc:  # OllamaPullError / OllamaPullCancelled / transport
            cancelled_type = getattr(module or _safe_pull_module(), "OllamaPullCancelled", None)
            if stop.is_set() or (cancelled_type is not None and isinstance(exc, cancelled_type)):
                state.cancelled = True
                state.status = "Stopped"
            else:
                state.error = str(exc) or type(exc).__name__
                state.status = f"Failed: {state.error}"
        finally:
            state.running = False
            self._stops.pop(model, None)
        self._emit(state)
        return state


def _safe_pull_module() -> Any:
    try:
        return _pull_module()
    except Exception:
        return None


async def server_status(*, run_io: Optional[Callable[..., Any]] = None, probe: Optional[Callable[[], Any]] = None) -> dict:
    """``{"running": bool, "version": str|None, "models": [names], "error": str|None}`` of the Ollama
    server ``ollamapull/`` uses (``ollamapull.base_url()``), read off the loop."""

    def read() -> dict:
        if probe is not None:
            return probe()
        module = _pull_module()
        out: dict = {"running": False, "version": None, "models": [], "error": None, "url": module.base_url()}
        version = module._server_version()
        if not version:
            out["error"] = f"No Ollama answered at {module.base_url()}"
            return out
        out["running"], out["version"] = True, version
        try:
            out["models"] = [str(m.get("name") or m.get("model") or "") for m in module._models() if isinstance(m, dict)]
        except Exception as exc:
            out["error"] = str(exc)
        return out

    try:
        if run_io is not None:
            return await run_io(read)
        return await asyncio.to_thread(read)
    except Exception as exc:
        return {"running": False, "version": None, "models": [], "error": str(exc)}


def installed(model: str, names: Sequence[str]) -> bool:
    """Is ``model`` among the server's ``/api/tags`` names (``name`` alone means ``name:latest``)."""
    wanted = str(model or "").strip().lower()
    aliases = {wanted} if ":" in wanted.rsplit("/", 1)[-1] else {wanted, f"{wanted}:latest"}
    return any(str(n).strip().lower() in aliases for n in names or ())
