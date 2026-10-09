"""OAuthBridge: account sign-in on the phone (plan §4, UI_SPEC §4.13, §4.17).

Nothing here re-implements OAuth. The bridge drives the shared ``begin`` /
``complete`` split of each backend auth module (U1 for ChatGPT, U4 for the rest):

* **Loopback providers** (ChatGPT ``authgpt_auth``, Claude ``authcd_auth``, Gemini
  ``authgem_auth``): ``begin_oauth(store=..., account_id=..., timeout=...)`` starts
  PKCE + state and the loopback callback server and persists the pending state
  encrypted next to the token file (paste fallback); ``complete_from_redirect(
  session_or_store, url_or_code)`` checks the state, exchanges the code and saves
  the tokens to the slot's token store (``HOME/.glossarion``, encrypted with the
  SecureStorage key installed at boot).
* **Device code** (Grok ``authgrok_auth``, RFC 8628): ``begin_device_login(store,
  account_id)`` returns the verification URI + user code; ``poll_device_login(
  session, should_stop)`` polls until the browser approves (it checks the other slots
  and saves the tokens). The app shows the code (copy) and opens the verification page
  itself. A device login the app was killed in the middle of is picked up again with
  ``resume_device_login`` (same code, while it is valid).
* Claude's session also carries ``manual_auth_url``: the same sign-in with Anthropic's
  code page as redirect, which shows ``code#state`` to paste (``open_manual_page``).

Flow (``LoginSheet`` steps Opening browser -> Waiting for sign-in -> Exchanging token
-> Done):

1. ``begin_*`` on a worker thread; ``opener(url)`` = ``webbrowser.open``, which the
   bootstrap routes to ``UrlLauncher.launch_url(mode=IN_APP_BROWSER_VIEW)`` (Custom
   Tabs / SFSafariViewController);
2. on Android a short "Signing in…" foreground service keeps the process (and the
   loopback server) alive while the browser is in front. Without it Android caches
   and then freezes the app behind the Custom Tab: the kernel still accepts the
   browser's connection to the loopback, nobody answers, and the page spins after
   the account / organization is chosen. A running job service is shared
   (``ServiceHolds``: whoever finishes last stops it). When the service is gone
   once the browser is in front (checked after ``fgs_check_delay`` seconds and on
   its ``destroyed`` event) the sign-in gets a ``notice`` and the LoginPanel shows
   the paste field at once; **Reopen browser** then starts the service again first;
3. the loopback answers the callback with a page that returns to the app
   (``glossarion://app/oauth/return?p=<provider>``, ``GLOSSARION_OAUTH_RETURN_URL``)
   and says the tab can be closed; nothing waits for that deep link (the app routes
   it to ``on_return_link``, which closes the iOS in-app browser);
4. the waiter sees the callback and exchanges the code at once; or the user taps
   **Paste redirect URL / code** and ``complete_with_paste`` finishes from the saved
   pending state (works after the app or the loopback server was killed). Coming
   back to the app while the sign-in still waits (``on_lifecycle("resume")``) checks
   the loopback listener and shows the paste field with a ``notice``.

Account slots are the desktop's: slot 0 is ``<provider>_tokens.json``, slot N is
``<provider>_tokens_N.json`` (``get_store(N)``) and the routes ``authgptN/``,
``authgrokN/``, ``authcdN/``, ``authgemN/`` / ``authgem-vertexN/``; the pool routes
``authgpt0/``, ``authgrok0/`` and ``authgem-vertex0/`` use every saved slot.
``account_slots`` lists the slots the Accounts screen shows: slot 0, every slot with
a token file, the slot of the selected model, slots referenced by enabled key-pool
entries and slots added with "+ Add account" in this session.

Sign-out never deletes a token file outside the app's own data folder
(``safe_root``): a Windows dev run whose auth modules resolved ``~`` to the real
user profile must not log the desktop out.

``state`` is a ``SignInState`` (observable; listeners are marshalled with ``post``).
Pure Python (3.10); no Flet import. ``auth_module`` / ``auth_modules`` and
``opener`` are injectable.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import logging
import os
import re
import socket
import threading
import time
import webbrowser
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

__all__ = [
    "OAuthBridge",
    "PROVIDERS",
    "PROVIDER_INFO",
    "ProviderInfo",
    "SignInState",
    "STEP_LABELS",
    "pool_route_provider",
    "provider_for_model",
    "sign_in_satisfied",
    "sign_in_slot",
    "slot_key",
    "stall_notice",
]

log = logging.getLogger("glossarion.oauth")

DEFAULT_TIMEOUT = 300  # seconds (authgpt run_oauth_flow default)
SIGN_IN_HOLD = "sign-in"  # ServiceHolds name of a sign-in waiting in the browser
FGS_CHECK_DELAY = 2.0  # seconds after opening the browser: is the sign-in service still running?
RESUME_CHECK_DELAY = 1.0  # seconds after the app came back: a delayed callback lands first
_AUTO = object()


@dataclass(frozen=True)
class ProviderInfo:
    id: str  # route stem and token-file prefix: authgpt / authgrok / authcd / authgem
    label: str
    module: str  # backend auth module
    flow: str  # "loopback" | "device"
    pool_route: Optional[str]  # route that rotates through every saved slot
    rotation_note: str
    blurb: str
    paste_hint: str = ""
    icon: str = "ACCOUNT_CIRCLE"


#: UI_SPEC §4.13 provider cards, in card order (ChatGPT · Grok · Claude · Gemini).
PROVIDER_INFO: dict[str, ProviderInfo] = {
    "authgpt": ProviderInfo(
        "authgpt", "ChatGPT", "authgpt_auth", "loopback", "authgpt0/",
        "authgpt0/ uses every signed-in ChatGPT slot in turn; authgptN/ uses slot #N only.",
        "Your ChatGPT Plus/Pro subscription powers GPT-6 Luna, the default model.",
        "http://localhost:1455/auth/callback?code=…&state=…",
    ),
    "authgrok": ProviderInfo(
        "authgrok", "Grok", "authgrok_auth", "device", "authgrok0/",
        "authgrok0/ rotates through every saved Grok account; authgrokN/ uses slot #N only.",
        "Log in to xAI in the browser (Google sign-in works). No API key is needed.",
    ),
    "authcd": ProviderInfo(
        "authcd", "Claude", "authcd_auth", "loopback", None,
        "authcdN/ uses Claude slot #N; authcd/ uses slot #0.",
        "Log in with your Claude Pro/Max subscription in the browser. No API key is needed.",
        "Paste the redirect URL, or the code#state the page shows",
    ),
    "authgem": ProviderInfo(
        "authgem", "Gemini", "authgem_auth", "loopback", "authgem-vertex0/",
        "authgem-vertex0/ uses every Gemini slot (Vertex AI); authgemN/ and authgem-vertexN/ use slot #N.",
        "Log in with your Google account in the browser. No API key is needed.",
        "http://127.0.0.1:<port>/…?code=…&state=…",
    ),
}

#: ``(provider, label, reason)``: every provider signs in from U4 (reason None).
PROVIDERS = tuple((info.id, info.label, None) for info in PROVIDER_INFO.values())

STEP_LABELS = {
    "idle": "",
    "opening": "Opening browser…",
    "waiting": "Waiting for sign-in…",
    "exchanging": "Exchanging token…",
    "done": "Signed in",
    "error": "Sign-in failed",
    "cancelled": "Sign-in cancelled",
}

_PASTE_ADDRESS = "paste the address of that page (…?code=…&state=…) below"
_STALL_NOTICES = {
    # The "Signing in…" foreground service did not start, or is gone while the browser is in front.
    "service": ("Glossarion’s sign-in service is not running, so Android may pause Glossarion while the browser "
                "is open and the page there may hang after you sign in. If it does, come back here: the sign-in "
                f"usually finishes once Glossarion is open again. Otherwise {_PASTE_ADDRESS}."),
    # The app came back to the front and no callback has arrived yet.
    "waiting": f"Still waiting for the browser. If its page keeps loading or shows an error, {_PASTE_ADDRESS}.",
    # The app came back and its loopback listener no longer accepts connections.
    "listener": ("Glossarion stopped listening for the browser while it was in the background, so the page there "
                 f"cannot finish. {_PASTE_ADDRESS[0].upper()}{_PASTE_ADDRESS[1:]}, or tap Cancel and sign in again."),
}
_MANUAL_NOTICE = " Claude can also show a code: tap “Get a code to paste instead” and paste the code#state it shows."


def stall_notice(kind: str, provider: str = "authgpt", manual: bool = False) -> str:
    """The LoginPanel notice for a loopback sign-in that may be stuck (``service`` / ``waiting`` /
    ``listener``); *manual* adds Claude's code page."""
    text = _STALL_NOTICES.get(kind, _STALL_NOTICES["waiting"])
    return text + (_MANUAL_NOTICE if manual else "")

_MODEL_PATTERNS = (
    ("authgpt", re.compile(r"^authgpt(\d{0,4})$")),
    ("authgrok", re.compile(r"^authgrok(\d{0,4})$")),
    ("authcd", re.compile(r"^authcd(\d{0,4})$")),
    ("authgem", re.compile(r"^authgem(?:-vertex)?(\d{0,4})$")),
)

def provider_for_model(model: Optional[str]) -> Optional[tuple]:
    """``(provider, account_id)`` for an auth route (``authgpt/…`` -> ``("authgpt", 0)``,
    ``authgrok2/…`` -> ``("authgrok", 2)``, ``authgem-vertex3/…`` -> ``("authgem", 3)``), else None.

    ``authgem-key/`` uses an API key, not a sign-in, so it is not an account route.
    """
    prefix = str(model or "").split("/", 1)[0].strip().lower()
    for provider, pattern in _MODEL_PATTERNS:
        match = pattern.match(prefix)
        if match:
            return (provider, int(match.group(1)) if match.group(1) else 0)
    return None


def slot_key(provider: str, account_id: int = 0) -> str:
    """``"authgpt"`` for slot 0, ``"authgpt2"`` for slot 2 (``OAuthBridge.signed_in`` keys)."""
    return provider if not account_id else f"{provider}{int(account_id)}"


def pool_route_provider(model: Optional[str]) -> Optional[str]:
    """The provider whose every saved slot a pool route uses (``authgpt0/…`` -> ``"authgpt"``,
    ``authgrok0/…`` -> ``"authgrok"``, ``authgem-vertex0/…`` -> ``"authgem"``), else None."""
    text = str(model or "").strip().lower()
    for info in PROVIDER_INFO.values():
        if info.pool_route and text.startswith(info.pool_route):
            return info.id
    return None


def _key_slot(key: str, provider: str) -> Optional[int]:
    """Slot of a ``signed_in`` key of ``provider`` (``"authgpt"`` -> 0, ``"authgpt2"`` -> 2), else None."""
    if key == provider:
        return 0
    rest = key[len(provider):] if key.startswith(provider) else ""
    return int(rest) if rest.isdigit() else None


def sign_in_satisfied(model: Optional[str], signed_keys: Iterable[str]) -> bool:
    """Whether the account a sign-in route uses is signed in (True for models without one).

    ``authgptN/…`` needs slot #N (``slot_key``); the pool routes (``authgpt0/``,
    ``authgrok0/``, ``authgem-vertex0/``) rotate through every saved slot, so any signed-in
    slot of that provider will do.
    """
    route = provider_for_model(model)
    if route is None:
        return True
    provider, account_id = route
    keys = set(signed_keys or ())
    if pool_route_provider(model) == provider:
        return any(_key_slot(str(k), provider) is not None for k in keys)
    return slot_key(provider, account_id) in keys


def sign_in_slot(model: Optional[str], provider: str) -> int:
    """The slot a sign-in for ``model`` should target: the route's own slot (``authgpt2/`` -> 2),
    slot #0 for a pool route or a model of another provider."""
    route = provider_for_model(model)
    if route is None or route[0] != provider:
        return 0
    return int(route[1] or 0)


def _call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call ``fn`` with only the keyword arguments its signature accepts."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return fn(*args, **kwargs)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(*args, **kwargs)
    return fn(*args, **{k: v for k, v in kwargs.items() if k in params})


def _first_attr(obj: Any, names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        value = obj.get(name) if isinstance(obj, Mapping) else getattr(obj, name, None)
        if value not in (None, ""):
            return value
    return default


class _ConfigView:
    """``config.get`` over a ``MobileConfigStore.get``-style reader (what the shared rules read)."""

    def __init__(self, get: Callable[[str, Any], Any]) -> None:
        self._get = get

    def get(self, key: str, default: Any = None) -> Any:
        return self._get(key, default)


def _under(path: str, root: str) -> bool:
    try:
        path = os.path.normcase(os.path.realpath(path))
        root = os.path.normcase(os.path.realpath(root))
    except (OSError, ValueError):
        return False
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


@dataclass(frozen=True)
class SignInState:
    provider: str = "authgpt"
    step: str = "idle"  # idle | opening | waiting | exchanging | done | error | cancelled
    message: str = ""
    account_id: int = 0
    auth_url: str = ""
    email: str = ""
    plan: str = ""
    user_code: str = ""  # device-code flow (Grok)
    verification_uri: str = ""
    manual_url: str = ""  # Claude: the code page (code#state to paste)
    notice: str = ""  # waiting, but maybe stuck: the LoginPanel shows this and the paste field

    @property
    def busy(self) -> bool:
        return self.step in ("opening", "waiting", "exchanging")

    @property
    def label(self) -> str:
        return STEP_LABELS.get(self.step, self.step)

    @property
    def device(self) -> bool:
        return bool(self.user_code)


class OAuthBridge:
    def __init__(
        self,
        *,
        auth_module: Any = None,
        auth_modules: Optional[Mapping[str, Any]] = None,
        opener: Optional[Callable[[str], Any]] = None,
        close_browser: Optional[Callable[[], Any]] = None,
        native: Any = None,
        post: Optional[Callable[..., Any]] = None,
        run_io: Optional[Callable[..., Any]] = None,
        timeout: int = DEFAULT_TIMEOUT,
        is_android: bool = False,
        config_get: Optional[Callable[[str, Any], Any]] = None,
        safe_root: Any = _AUTO,
        fgs_check_delay: float = FGS_CHECK_DELAY,
        resume_check_delay: float = RESUME_CHECK_DELAY,
    ) -> None:
        self._modules: dict[str, Any] = dict(auth_modules or {})
        if auth_module is not None:
            self._modules["authgpt"] = auth_module
        self.opener = opener or webbrowser.open
        self.close_browser = close_browser
        self.native = native
        self.post = post
        self.run_io = run_io
        self.timeout = timeout
        self.is_android = is_android
        self.config_get = config_get  # MobileConfigStore.get (slots from the model / key pools)
        self._safe_root = safe_root
        self.fgs_check_delay = fgs_check_delay
        self.resume_check_delay = resume_check_delay
        self.state = SignInState()
        self._listeners: list = []
        self._session: Any = None
        self._cancel = threading.Event()
        self._lock = threading.RLock()
        self._lease: Any = None  # native.ServiceLease of the sign-in's foreground service (Android)
        self._fgs_failed = False  # Android: the sign-in service could not be started
        self._watch_task: Any = None  # checks the sign-in service once the browser is in front
        self._reopen_task: Any = None
        self._resume_task: Any = None
        self._pasting = False
        self._paste_claimed = False
        self.signed_in: set = set()
        self.pending_slots: dict[str, set] = {}  # "+ Add account" slots not signed in yet
        add_listener = getattr(native, "add_listener", None)
        if callable(add_listener):
            try:
                add_listener("foreground", self._on_foreground_event)
            except Exception:
                log.debug("listening for foreground-service events failed", exc_info=True)

    # ---- plumbing ------------------------------------------------------------------------

    @staticmethod
    def info(provider: str) -> ProviderInfo:
        try:
            return PROVIDER_INFO[provider]
        except KeyError:
            raise RuntimeError(f"Unknown sign-in provider {provider!r}") from None

    def module(self, provider: str = "authgpt") -> Any:
        module = self._modules.get(provider)
        if module is None:
            module = importlib.import_module(self.info(provider).module)  # backend (on sys.path after bootstrap)
            self._modules[provider] = module
        return module

    @property
    def auth(self) -> Any:
        """The ChatGPT auth module (U3 API)."""
        return self.module("authgpt")

    def subscribe(self, callback: Callable[[SignInState], Any]) -> Callable[[], None]:
        self._listeners.append(callback)

        def unsubscribe() -> None:
            try:
                self._listeners.remove(callback)
            except ValueError:
                pass

        return unsubscribe

    def _set(self, **changes: Any) -> SignInState:
        with self._lock:
            self.state = replace(self.state, **changes)
            state = self.state
        for callback in list(self._listeners):
            try:
                if self.post is not None:
                    self.post(callback, state)
                else:
                    callback(state)
            except Exception:
                log.exception("sign-in listener failed")
        return state

    async def _io(self, fn: Callable[..., Any], *args: Any) -> Any:
        if self.run_io is not None:
            return await self.run_io(fn, *args)
        return await asyncio.to_thread(fn, *args)

    def store(self, account_id: int = 0, provider: str = "authgpt") -> Any:
        return self.module(provider).get_store(int(account_id or 0) or None)

    @property
    def safe_root(self) -> Optional[str]:
        """Folder sign-out may delete token files in (the mobile HOME); None = no restriction."""
        root = self._safe_root
        if root is _AUTO:
            root = None
            try:
                from glossarion_mobile import runtime_bootstrap as rb

                paths = rb.get_paths() if rb.get_state() is not None else None
                if paths is not None:
                    root = str(paths.home)
            except Exception:
                root = None
        return str(root) if root else None

    # ---- status -------------------------------------------------------------------------------

    def status(self, account_id: int = 0, provider: str = "authgpt") -> dict:
        """Blocking (decrypts the token file): ``{signed_in, email, plan, name, source, expires_at}``."""
        account_id = int(account_id or 0)
        base = {"provider": provider, "account_id": account_id}
        try:
            store = self.store(account_id, provider)
            tokens = store.load_tokens() or {}
            if provider == "authgpt":
                info = store.account_info if tokens.get("id_token") else {}
            else:
                info = store.account_info if tokens else {}
        except Exception as exc:
            log.warning("reading the %s token store failed: %s", provider, exc)
            return {**base, "signed_in": False, "email": "", "plan": "", "name": "", "source": "",
                    "expires_at": None, "error": str(exc)}
        signed = bool(tokens.get("access_token") or tokens.get("refresh_token"))
        with self._lock:
            key = slot_key(provider, account_id)
            if signed:
                self.signed_in.add(key)
                self.pending_slots.get(provider, set()).discard(account_id)
            else:
                self.signed_in.discard(key)
        info = info if isinstance(info, Mapping) else {}
        return {
            **base,
            "signed_in": signed,
            "email": str(info.get("email") or ""),
            "plan": str(info.get("plan_type") or info.get("plan") or ""),
            "name": str(info.get("name") or ""),
            "source": str(info.get("source") or ""),
            "expires_at": tokens.get("expires_at"),
        }

    async def refresh_status(self, account_id: int = 0, provider: str = "authgpt") -> dict:
        return await self._io(self.status, account_id, provider)

    def statuses(self, provider: str, slots: Optional[Iterable[int]] = None) -> list:
        """Blocking: ``status`` of every slot (``account_slots`` when ``slots`` is None)."""
        ids = list(slots) if slots is not None else self.account_slots(provider)
        return [self.status(account_id, provider) for account_id in ids]

    def has_pending(self, account_id: int = 0, provider: str = "authgpt") -> bool:
        """Blocking: a sign-in of this slot can still be finished by a paste - the live session
        of this process, or the encrypted pending state saved next to the slot's token file
        (kept for an hour, also after the app or its loopback listener was killed)."""
        account_id = int(account_id or 0)
        with self._lock:
            live = (self._session is not None and self.state.provider == provider
                    and int(self.state.account_id or 0) == account_id)
        if live:
            return True
        try:
            store = self.store(account_id, provider)
            loader = getattr(store, "load_pending_oauth", None)
            return bool(loader()) if callable(loader) else False
        except Exception:
            return False

    async def pending_sign_in(self, account_id: int = 0, provider: str = "authgpt") -> bool:
        """``has_pending`` off the UI loop (decrypts the pending file)."""
        return bool(await self._io(self.has_pending, account_id, provider))

    # ---- account slots ------------------------------------------------------------------------

    def _token_dir(self, provider: str) -> Optional[str]:
        module = self.module(provider)
        directory = getattr(module, "_DEFAULT_TOKEN_DIR", None)
        if directory:
            return str(directory)
        try:
            return os.path.dirname(str(self.store(0, provider)._token_file))
        except Exception:
            return None

    def saved_slots(self, provider: str) -> set:
        """Numbered slots with a token file on disk (``<provider>_tokens_N.json``)."""
        found: set = set()
        module = self.module(provider)
        numbered = getattr(module, "_numbered_account_ids", None)  # authgrok's own helper
        if callable(numbered):
            try:
                found.update(int(i) for i in numbered())
            except Exception:
                pass
        directory = self._token_dir(provider)
        if directory:
            pattern = re.compile(rf"{re.escape(provider)}_tokens_(\d{{1,4}})\.json")
            try:
                for name in os.listdir(directory):
                    match = pattern.fullmatch(name)
                    if match:
                        found.add(int(match.group(1)))
            except OSError:
                pass
        found.discard(0)
        return found

    def configured_slots(self, provider: str, config_get: Optional[Callable[[str, Any], Any]] = None) -> set:
        """Slots named by the selected model and by enabled key-pool entries: the shared
        ``settings_rules.model_account_ids`` (desktop ``_collect_auth_account_ids_from_pools``
        + the model's own slot, as ``_refresh_auth_account_arrows`` adds it)."""
        get = config_get or self.config_get
        if get is None:
            return set()
        model = get("model", None)
        try:
            import settings_rules  # shared core (U4)

            ids = settings_rules.model_account_ids(model, _ConfigView(get))
            return set(int(i) for i in ids.get(provider, ()))
        except Exception:
            log.debug("settings_rules.model_account_ids unavailable", exc_info=True)
        route = provider_for_model(str(model or ""))
        return {route[1]} if route is not None and route[0] == provider else set()

    def account_slots(self, provider: str, config_get: Optional[Callable[[str, Any], Any]] = None) -> list:
        """Blocking (lists the token folder): the slots the Accounts screen shows, sorted."""
        slots = {0}
        slots.update(self.saved_slots(provider))
        slots.update(self.configured_slots(provider, config_get))
        with self._lock:
            slots.update(self.pending_slots.get(provider, set()))
        return sorted(slots)

    def next_slot(self, provider: str, slots: Optional[Iterable[int]] = None) -> int:
        """The next free slot for "+ Add account" (Grok asks ``get_next_account_id``, like desktop)."""
        known = set(int(s) for s in (slots if slots is not None else self.account_slots(provider)))
        if provider == "authgrok":
            allocate = getattr(self.module(provider), "get_next_account_id", None)
            if callable(allocate):
                return int(allocate(sorted(known)))
        return max(known | {0}) + 1

    def add_slot(self, provider: str, account_id: int) -> None:
        with self._lock:
            self.pending_slots.setdefault(provider, set()).add(int(account_id))

    # ---- sign in ------------------------------------------------------------------------------

    async def sign_in(self, provider: str = "authgpt", account_id: int = 0) -> dict:
        """Run the browser sign-in; returns the account status ({} when cancelled or
        finished by a paste). Raises on failure (the state then reads ``error``)."""
        info = self.info(provider)
        if self.state.busy:
            raise RuntimeError("A sign-in is already in progress")
        account_id = int(account_id or 0)
        self._cancel.clear()
        self._pasting = False
        self._paste_claimed = False
        self._set(provider=provider, step="opening", message="", account_id=account_id, auth_url="",
                  email="", plan="", user_code="", verification_uri="", manual_url="", notice="")
        if info.flow == "device":
            return await self._sign_in_device(provider, account_id)
        try:
            # A previous sign-in finished by a failed paste keeps its loopback server
            # until its watchdog fires; free the port before listening again. (The pending
            # state stays in the token store, so pasting still works after this.)
            await self._io(self._close_session)
            module, session = await self._io(self._begin_loopback, provider, account_id)
        except Exception as exc:
            self._set(step="error", message=str(exc))
            raise
        with self._lock:
            self._session = session
        await self._start_fgs(info)
        self._set(step="waiting", auth_url=str(getattr(session, "auth_url", "") or ""),
                  manual_url=str(getattr(session, "manual_auth_url", "") or ""))
        self._open(self.state.auth_url)
        if self._fgs_failed:
            self._flag_stalled("service")
        self._watch_task = asyncio.ensure_future(self._watch_fgs()) if self._fgs_held else None
        try:
            received = await self._io(self._wait_for_callback, session)
            if self._cancel.is_set():
                if self._paste_claimed:
                    return {}  # complete_with_paste finishes (and reports) this sign-in
                await self._io(self._close_session)  # off the UI loop: stopping a listener takes up to 0.5 s
                self._set(step="cancelled", message="")
                await self._stop_fgs()
                return {}
            if not received:
                raise RuntimeError("OAuth login timed out – no callback received.")
            self._set(step="exchanging")
            await self._io(module.complete_from_redirect, session)
        except Exception as exc:
            await self._io(self._close_session)
            self._set(step="error", message=str(exc))
            await self._stop_fgs()
            raise
        finally:
            watch, self._watch_task = self._watch_task, None
            if watch is not None:
                watch.cancel()
        return await self._finished(account_id, provider)

    def _begin_loopback(self, provider: str, account_id: int) -> tuple:
        """Worker thread: import the auth module, open the slot's store, ``begin_oauth``."""
        module = self.module(provider)
        store = self.store(account_id, provider)
        clear_flag = getattr(store, "clear_logout_flag", None)  # authcd: an explicit login ends the logout block
        if callable(clear_flag):
            clear_flag()
        return module, _call(module.begin_oauth, store=store, account_id=account_id or None, timeout=self.timeout)

    async def _sign_in_device(self, provider: str, account_id: int) -> dict:
        """Grok: RFC 8628 device code - show the code, open the verification page, poll."""
        info = self.info(provider)
        try:
            module, session = await self._io(self._begin_or_resume_device, provider, account_id)
        except Exception as exc:
            self._set(step="error", message=str(exc))
            raise
        with self._lock:
            self._session = session
        verification = str(_first_attr(session, ("verification_uri", "verification_url")) or "")
        open_url = str(_first_attr(session, ("open_url", "signed_out_url", "browser_url", "verification_uri_complete",
                                             "verification_url_complete")) or verification)
        await self._start_fgs(info)
        self._set(step="waiting", auth_url=open_url, verification_uri=verification or open_url,
                  user_code=str(_first_attr(session, ("user_code",)) or ""))
        self._open(open_url)
        try:
            tokens = await self._io(lambda: _call(module.poll_device_login, session, should_stop=self._cancel.is_set))
            if self._cancel.is_set():
                self._close_session()
                self._set(step="cancelled", message="")
                await self._stop_fgs()
                return {}
            self._set(step="exchanging")
            await self._io(self._save_device_tokens, provider, account_id, tokens)
        except Exception as exc:
            self._close_session()
            if self._cancel.is_set():
                self._set(step="cancelled", message="")
                await self._stop_fgs()
                return {}
            self._set(step="error", message=str(exc))
            await self._stop_fgs()
            raise
        return await self._finished(account_id, provider)

    def _begin_or_resume_device(self, provider: str, account_id: int) -> tuple:
        """Worker thread: the device login saved for this slot (app killed mid-poll), else a new one."""
        module = self.module(provider)
        store = self.store(account_id, provider)
        resume = getattr(module, "resume_device_login", None)
        if callable(resume):
            try:
                session = _call(resume, store=store, account_id=account_id, timeout=self.timeout)
            except Exception as exc:
                log.info("resuming the saved device login failed: %s", exc)
                session = None
            if session is not None:
                return module, session
        return module, _call(module.begin_device_login, store=store, account_id=account_id, timeout=self.timeout)

    def _save_device_tokens(self, provider: str, account_id: int, tokens: Any) -> None:
        """Desktop ``_authgrok_login_clicked``: ``validate_account_slot_tokens`` then ``save_tokens``
        (skipped when ``poll_device_login`` already saved these tokens to the slot)."""
        if not isinstance(tokens, Mapping) or not tokens.get("access_token"):
            return
        store = self.store(account_id, provider)
        saved = store.load_tokens() or {}
        if saved.get("access_token") == tokens.get("access_token"):
            return
        validate = getattr(self.module(provider), "validate_account_slot_tokens", None)
        if callable(validate):
            validate(account_id, dict(tokens))
        store.save_tokens(dict(tokens))

    def _wait_for_callback(self, session: Any) -> bool:
        deadline = time.monotonic() + self.timeout
        while not self._cancel.is_set():
            if session.wait_for_callback(0.25):
                return True
            if time.monotonic() >= deadline:
                return False
            if not getattr(session, "server_running", True) and not getattr(session, "callback_received", False):
                # Listener stopped (timeout watchdog / killed): only the paste path can finish now.
                return bool(getattr(session, "callback_received", False))
        return False

    async def _finished(self, account_id: int, provider: str = "authgpt") -> dict:
        await self._io(self._close_session)
        status = await self._io(self.status, account_id, provider)
        self._set(step="done", message="", email=status.get("email", "") or status.get("name", ""),
                  plan=status.get("plan", ""))
        await self._stop_fgs()
        return status

    def _open(self, url: str) -> None:
        if not url:
            return
        try:
            self.opener(url)
        except Exception as exc:
            log.warning("opening the sign-in page failed: %s", exc)

    def reopen_browser(self) -> bool:
        url = self.state.auth_url or getattr(self._session, "auth_url", "")
        if not url:
            return False
        if self._service_missing():
            # Android: the sign-in service is gone (the app is in front again, so it may start now):
            # restart it before the browser covers the app once more.
            try:
                self._reopen_task = asyncio.get_running_loop().create_task(self._restart_fgs_then_open(url))
                return True
            except RuntimeError:
                pass
        self._open(url)
        return True

    def _service_missing(self) -> bool:
        state = self.state
        return (self.is_android and self.native is not None and not self._fgs_held
                and state.step == "waiting" and not state.device)

    async def _restart_fgs_then_open(self, url: str) -> None:
        await self._start_fgs(self.info(self.state.provider))
        if self.state.step != "waiting":
            await self._stop_fgs()  # the sign-in ended meanwhile
            return
        if self._fgs_held:
            self._set(notice="")
            self._watch_task = asyncio.ensure_future(self._watch_fgs())
        self._open(url)

    def open_manual_page(self) -> bool:
        """Claude: open the code page (shows ``code#state``) of the sign-in in progress."""
        return self.open_url(self.state.manual_url)

    def open_url(self, url: str) -> bool:
        """Open a page in the in-app browser (Gemini verification URL, Grok verification page)."""
        if not url:
            return False
        self._open(url)
        return True

    async def complete_with_paste(self, text: str, account_id: Optional[int] = None,
                                  provider: Optional[str] = None) -> dict:
        """Finish from a pasted redirect URL, ``code#state`` or bare code (loopback providers)."""
        provider = provider or self.state.provider or "authgpt"
        info = self.info(provider)
        if info.flow != "loopback":
            raise RuntimeError(f"{info.label} signs in with a device code; there is nothing to paste.")
        value = str(text or "").strip()
        if not value:
            raise RuntimeError(f"Paste the redirect URL (or the code) to finish the {info.label} sign-in.")
        if account_id is None:
            account_id = self.state.account_id if self.state.provider == provider else 0
        with self._lock:
            session = self._session if self.state.provider == provider else None
            self._pasting = True
            self._paste_claimed = True
        self._cancel.set()  # stop the loopback waiter; the paste wins
        self._set(provider=provider, account_id=int(account_id or 0), step="exchanging", message="")
        try:
            await self._io(self._complete_paste, provider, session, int(account_id or 0), value)
        except Exception as exc:
            self._pasting = False
            self._set(step="error", message=str(exc))
            await self._stop_fgs()
            raise
        self._pasting = False
        return await self._finished(int(account_id or 0), provider)

    def _complete_paste(self, provider: str, session: Any, account_id: int, value: str) -> Any:
        """Worker thread: finish from the live session, else from the slot's saved pending state."""
        target = session if session is not None else self.store(account_id, provider)
        return self.module(provider).complete_from_redirect(target, value)

    def cancel(self) -> None:
        self._cancel.set()
        self._close_session()
        if self.state.busy:
            self._set(step="cancelled", message="")

    def _close_session(self) -> None:
        with self._lock:
            session, self._session = self._session, None
        if session is not None:
            close = getattr(session, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    def on_return_link(self, provider: Optional[str] = None) -> None:
        """``glossarion://app/oauth/return?p=<provider>``: close the iOS in-app browser.

        The waiter finishes on its own once the loopback server recorded the callback.
        """
        if self.close_browser is not None:
            try:
                result = self.close_browser()
                if asyncio.iscoroutine(result):
                    asyncio.ensure_future(result)
            except Exception:
                log.debug("closing the in-app browser failed", exc_info=True)

    # ---- sign out -----------------------------------------------------------------------------

    def _clear(self, provider: str, account_id: int) -> None:
        store = self.store(account_id, provider)
        root = self.safe_root
        token_file = getattr(store, "_token_file", None)
        if root and token_file and not _under(str(token_file), root):
            raise RuntimeError(
                f"Not signing out: the {self.info(provider).label} token file is outside the app's data folder "
                f"({token_file}). Set the token-file overrides before importing the backend."
            )
        store.clear_tokens()
        clear_pending = getattr(store, "clear_pending_oauth", None)
        if callable(clear_pending):
            clear_pending()

    async def sign_out(self, account_id: int = 0, provider: str = "authgpt") -> None:
        account_id = int(account_id or 0)
        await self._io(self._clear, provider, account_id)
        with self._lock:
            self.signed_in.discard(slot_key(provider, account_id))
        if self.state.provider == provider and self.state.account_id == account_id:
            self._set(step="idle", message="", email="", plan="", user_code="", verification_uri="", manual_url="")

    def sign_out_everywhere(self, config_get: Optional[Callable[[str, Any], Any]] = None) -> list:
        """Blocking: sign every slot of every provider out; returns the slot keys cleared."""
        cleared: list = []
        errors: list = []
        for provider in PROVIDER_INFO:
            try:
                slots = self.account_slots(provider, config_get)
            except Exception as exc:
                errors.append(f"{provider}: {exc}")
                continue
            for account_id in slots:
                try:
                    status = self.status(account_id, provider)
                    if not status.get("signed_in"):
                        continue
                    self._clear(provider, account_id)
                    cleared.append(slot_key(provider, account_id))
                except Exception as exc:
                    errors.append(f"{slot_key(provider, account_id)}: {exc}")
        with self._lock:
            for key in cleared:
                self.signed_in.discard(key)
        if errors:
            raise RuntimeError("; ".join(errors))
        return cleared

    # ---- Gemini extras ---------------------------------------------------------------------------

    def gemini_status(self, account_id: int = 0) -> dict:
        """Blocking: desktop 📊 (``authgem_auth.check_account_status``) for a signed-in slot."""
        module = self.module("authgem")
        store = self.store(account_id, "authgem")
        if not store.has_tokens:
            raise RuntimeError("Not logged in — sign in to Gemini first.")
        token = store.get_valid_access_token(auto_login=False)
        return dict(module.check_account_status(token, int(account_id or 0)) or {})

    def gemini_projects(self, account_id: int = 0) -> list:
        """Blocking: ``[(project_id, "billed"|"unknown"|"unbilled")]`` for the GCP project picker.

        The shared ``authgem_auth.list_gcp_projects`` (the desktop picker's listing: every
        active project, billing checked in parallel), in the desktop picker's order: billed,
        then unknown, then unbilled. ``[]`` when nothing was listed.
        """
        module = self.module("authgem")
        store = self.store(account_id, "authgem")
        token = store.get_valid_access_token(auto_login=False)
        result = module.list_gcp_projects(token)
        if not result:
            return []
        billed, unbilled, unknown = result
        return ([(str(pid), "billed") for pid in billed] + [(str(pid), "unknown") for pid in unknown]
                + [(str(pid), "unbilled") for pid in unbilled])

    @staticmethod
    def _split_projects(projects: Sequence[Any]) -> tuple:
        """``[(project_id, status)]`` -> the shared ``(billed, unbilled, unknown)`` lists."""
        billed: list = []
        unbilled: list = []
        unknown: list = []
        for pid, status in projects or ():
            target = billed if status == "billed" else unbilled if status == "unbilled" else unknown
            target.append(str(pid))
        return billed, unbilled, unknown

    def gemini_project_items(self, projects: Sequence[Any]) -> Optional[list]:
        """``[(label, project_id)]`` of the picker in the desktop dropdown order
        (``authgem_auth.authgem_project_items``: ✅ billed, ❔ unknown, ⚠️ … (no billing)), or None
        when this build's authgem_auth has no shared item builder."""
        items = getattr(self.module("authgem"), "authgem_project_items", None)
        if not callable(items):
            return None
        billed, unbilled, unknown = self._split_projects(projects)
        return [(str(label), str(pid)) for label, pid in items(billed, unbilled, unknown)]

    def gemini_project_choice(self, projects: Sequence[Any], saved: str = "") -> Optional[str]:
        """The project the picker selects after listing (desktop ``_authgem_projects_loaded``, shared
        as ``authgem_auth.choose_authgem_project_index``): the saved one unless it is known unbilled,
        else the first billed, else the first unknown. None: keep the current selection."""
        choose = getattr(self.module("authgem"), "choose_authgem_project_index", None)
        if not callable(choose):
            return None
        billed, unbilled, unknown = self._split_projects(projects)
        index = choose(str(saved or ""), billed, unbilled, unknown)
        ordered = billed + unknown + unbilled  # authgem_project_items order
        try:
            index = int(index)
        except (TypeError, ValueError):
            return None
        return ordered[index] if 0 <= index < len(ordered) else None

    def set_gemini_project(self, project_id: str, account_id: int = 0) -> None:
        """Desktop ``_authgem_project_changed`` minus the env write: the job's HeadlessOwner sets
        ``GOOGLE_CLOUD_PROJECT`` from ``authgem_project`` at job start (``_restore_authgem_project_selection``)."""
        project_id = str(project_id or "").strip()
        if not project_id:
            return
        module = self.module("authgem")
        account = int(account_id or 0)
        try:
            module._cached_project_id[account] = project_id
            module._project_set_by_gui[account] = True
        except Exception:
            log.debug("seeding the AuthGem project cache failed", exc_info=True)

    # ---- Android short sign-in FGS -------------------------------------------------------------
    #
    # While the browser is in front the app's activity is stopped. Without a foreground
    # service Android makes the process "previous", then cached (60 s, or at once when the
    # user switches to another app to read a code) and freezes it 10 s later: the loopback
    # socket still accepts the browser's connection but no thread answers, so the page
    # spins with no error. The service keeps the process perceptible (never frozen).
    # ``native.ServiceLease`` holds it (shared with a running job through ``ServiceHolds``).

    @property
    def _fgs_started(self) -> bool:
        """This sign-in started the foreground service."""
        return bool(self._lease is not None and self._lease.started)

    @property
    def _fgs_held(self) -> bool:
        """This sign-in relies on the foreground service (started, or shared with a job)."""
        return bool(self._lease is not None and self._lease.held)

    async def _start_fgs(self, info: Optional[ProviderInfo] = None) -> None:
        from glossarion_mobile.services.native import ServiceLease

        self._lease = None
        self._fgs_failed = False
        if not self.is_android or self.native is None:
            return
        label = info.label if info is not None else "ChatGPT"
        # Shares a running job's service (its end must not stop it while the browser is open).
        self._lease = ServiceLease(self.native, SIGN_IN_HOLD)
        if not await self._lease.acquire(f"Signing in to {label}", "Waiting for sign-in…") and self._lease.failed:
            log.warning("the sign-in foreground service did not start; Android may pause Glossarion while "
                        "the browser is open")
            self._fgs_failed = True

    async def _stop_fgs(self) -> None:
        lease, self._lease = self._lease, None
        if lease is not None and self.native is not None:
            await lease.release()

    async def _watch_fgs(self) -> None:
        """Once the browser is in front: is the sign-in service still running? (A build whose service
        stopped as soon as the app's activity paused left the loopback unprotected.)"""
        await asyncio.sleep(self.fgs_check_delay)
        if not self._fgs_held or self.state.step != "waiting":
            return
        try:
            running = bool(await self.native.is_job_service_running())
        except Exception:
            return
        if not running and self._fgs_held:
            self._service_lost("not running once the browser was open")

    def _service_lost(self, reason: str) -> None:
        log.warning("the sign-in foreground service is gone (%s): Android may freeze Glossarion while the "
                    "browser is open, and the browser then hangs on the sign-in redirect", reason)
        if self._lease is not None:
            self._lease.drop()
        self._flag_stalled("service")

    def _owns_service_alone(self) -> bool:
        return bool(self._lease is not None and self._lease.owns_alone())

    async def _on_foreground_event(self, event: Mapping[str, Any]) -> None:
        """Foreground-service events (UI loop): the service died, or its notification's Stop was tapped."""
        if not isinstance(event, Mapping):
            return
        kind = str(event.get("type") or "")
        if kind in ("destroyed", "timeout"):
            if not self._fgs_held:
                return
            # Every stop ends with "destroyed": a late one (an earlier sign-in's or job's service) must not
            # count against a service that runs again.
            try:
                running = bool(await self.native.is_job_service_running())
            except Exception:
                running = False
            if not running and self._fgs_held:
                self._service_lost(f"service {kind}")
        elif kind == "button" and str(event.get("button_id") or "") == "stop":
            if self.state.busy and self._owns_service_alone():
                log.info("sign-in cancelled from the notification")
                self.cancel()

    def _flag_stalled(self, kind: str) -> None:
        """Waiting, but maybe stuck: set the notice (the LoginPanel then shows the paste field)."""
        with self._lock:
            state = self.state
            session = self._session
        if state.step != "waiting" or state.device or self.info(state.provider).flow != "loopback":
            return
        if session is not None and getattr(session, "callback_received", False):
            return
        self._set(notice=stall_notice(kind, state.provider, bool(state.manual_url)))

    # ---- app lifecycle ---------------------------------------------------------------------------

    def on_lifecycle(self, name: Any) -> None:
        """App lifecycle change (UI loop). Back in the app while a loopback sign-in still waits:
        a moment later (a delayed callback lands first) check the listener and point at the paste."""
        if str(getattr(name, "value", name) or "").lower() != "resume":
            return
        state = self.state
        if state.step != "waiting" or state.device:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = self._resume_task
        if task is not None and not task.done():
            return
        self._resume_task = loop.create_task(self._after_resume())

    async def _after_resume(self) -> None:
        await asyncio.sleep(self.resume_check_delay)
        with self._lock:
            session = self._session
            step = self.state.step
        if session is None or step != "waiting" or getattr(session, "callback_received", False):
            return
        try:
            alive = bool(await self._io(self._listener_alive, session))
        except Exception:
            alive = True
        if not alive:
            log.warning("the sign-in loopback listener no longer accepts connections")
        self._flag_stalled("waiting" if alive else "listener")

    @staticmethod
    def _listener_alive(session: Any) -> bool:
        """Worker thread: does one of the session's loopback listeners still accept a connection?"""
        servers = list(getattr(session, "_servers", None) or ())
        if not servers:
            return True  # nothing to probe
        if not getattr(session, "server_running", True):
            return False
        for server in servers:
            try:
                family = server.address_family
                address = tuple(server.server_address)[:2]
            except Exception:
                continue
            try:
                with socket.socket(family, socket.SOCK_STREAM) as probe:
                    probe.settimeout(1.0)
                    probe.connect(address)
                return True
            except OSError:
                continue
        return False
