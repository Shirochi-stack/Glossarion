# authza_auth.py - Z.AI subscription access via Playwright browser automation
# chat.z.ai blocks all external/headless API requests.  This module uses a
# visible (minimized) Playwright browser to interact with the chat UI directly.
"""
Browser automation flow for Z.AI subscription access:

  1. First use: visible browser opens → user clicks "Continue with Google"
  2. Login cookies saved to persistent browser profile
  3. Subsequent calls: browser is kept alive (minimized), messages are
     typed into the chat input and responses are scraped from the DOM
  4. Streaming chunks are emitted as the response text grows

All Playwright operations run on a dedicated daemon thread to avoid
greenlet/thread-affinity issues.

Requires: pip install playwright && python -m playwright install chromium
"""
import os
import json
import time
import logging
import threading
import queue
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level cancellation flag
# ---------------------------------------------------------------------------
_cancel_event = threading.Event()


def cancel_stream():
    """Signal any active AuthZA stream to abort immediately."""
    _cancel_event.set()


def reset_cancel():
    """Clear the cancellation flag (call before starting a new request)."""
    _cancel_event.clear()


def is_cancelled() -> bool:
    return _cancel_event.is_set()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ZA_CHAT_URL = "https://chat.z.ai/"
_DEFAULT_TOKEN_DIR = os.path.join(os.path.expanduser("~"), ".glossarion")
_BROWSER_PROFILE_DIR = os.path.join(_DEFAULT_TOKEN_DIR, "authza_browser")


# ---------------------------------------------------------------------------
# Playwright availability check
# ---------------------------------------------------------------------------
def _check_playwright() -> bool:
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
        return True
    except ImportError:
        return False


PLAYWRIGHT_AVAILABLE = _check_playwright()


# ---------------------------------------------------------------------------
# Dedicated Playwright Thread
# ---------------------------------------------------------------------------
class _PlaywrightThread:
    """Runs all Playwright ops on a single daemon thread (greenlet-safe)."""

    def __init__(self):
        self._work_q: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._started = threading.Event()
        self._lock = threading.Lock()

    def _ensure_running(self):
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._started.clear()
                self._thread = threading.Thread(
                    target=self._run_loop, daemon=True, name="AuthZA-Playwright"
                )
                self._thread.start()
                self._started.wait(timeout=30)

    def _run_loop(self):
        self._started.set()
        while True:
            try:
                item = self._work_q.get(timeout=300)
            except queue.Empty:
                continue
            if item is None:
                break
            fn, result_q = item
            try:
                result = fn()
                result_q.put(("ok", result))
            except Exception as exc:
                result_q.put(("error", exc))

    def execute(self, fn, timeout: float = 600) -> Any:
        self._ensure_running()
        result_q: queue.Queue = queue.Queue()
        self._work_q.put((fn, result_q))
        try:
            status, value = result_q.get(timeout=timeout)
        except queue.Empty:
            raise RuntimeError("AuthZA: Playwright operation timed out")
        if status == "error":
            raise value
        return value


_pw_thread = _PlaywrightThread()


# ---------------------------------------------------------------------------
# Browser Manager
# ---------------------------------------------------------------------------
class _BrowserManager:
    """Manages a persistent visible Chromium browser for chat.z.ai."""

    def __init__(self, profile_dir: str, account_id: int = 0):
        self._profile_dir = profile_dir
        self._account_id = account_id
        self._pw = None
        self._context = None
        self._page = None
        self._chunk_fn_exposed = False
        self._chunk_queue: queue.Queue = queue.Queue()
        self._force_relogin = False
        os.makedirs(self._profile_dir, exist_ok=True)

    # -- internal (run ON _pw_thread) ----------------------------------------

    def _start_context_impl(self, headless: bool = False):
        from playwright.sync_api import sync_playwright

        if self._pw is None:
            self._pw = sync_playwright().start()

        if self._context is not None:
            try:
                self._context.close()
            except Exception:
                pass
            self._context = None
            self._page = None

        self._context = self._pw.chromium.launch_persistent_context(
            self._profile_dir,
            headless=headless,
            viewport={"width": 1024, "height": 768},
            args=["--disable-blink-features=AutomationControlled"],
            ignore_default_args=["--enable-automation"],
        )
        self._page = (
            self._context.pages[0]
            if self._context.pages
            else self._context.new_page()
        )
        self._chunk_fn_exposed = False

    def _close_impl(self):
        if self._context is not None:
            try:
                self._context.close()
            except Exception:
                pass
            self._context = None
            self._page = None
        if self._pw is not None:
            try:
                self._pw.stop()
            except Exception:
                pass
            self._pw = None

    def _is_logged_in_impl(self) -> bool:
        try:
            token = self._page.evaluate("localStorage.getItem('token')")
            return bool(token and len(str(token)) > 20)
        except Exception:
            return False

    def _navigate_impl(self):
        self._page.goto(ZA_CHAT_URL, wait_until="domcontentloaded", timeout=30000)
        try:
            self._page.wait_for_load_state("networkidle", timeout=10000)
        except Exception:
            pass

    def _minimize_window_impl(self):
        try:
            self._page.evaluate("window.moveTo(32000, 32000)")
            self._page.evaluate("window.resizeTo(1, 1)")
        except Exception:
            pass

    # -- public (dispatch to _pw_thread) -------------------------------------

    def close(self):
        try:
            _pw_thread.execute(self._close_impl, timeout=15)
        except Exception:
            pass

    def ensure_logged_in(self) -> bool:
        """Ensure valid session. Opens visible browser for login if needed."""
        acct = f" (Account #{self._account_id})" if self._account_id else ""

        # Reuse existing live context
        if not self._force_relogin and self._context is not None and self._page is not None:
            try:
                if _pw_thread.execute(self._is_logged_in_impl, timeout=10):
                    return True
            except Exception:
                pass

        # Try existing profile (visible but minimized)
        if not self._force_relogin:
            def _try_existing():
                self._start_context_impl(headless=False)
                self._minimize_window_impl()
                self._navigate_impl()
                return self._is_logged_in_impl()
            try:
                if _pw_thread.execute(_try_existing, timeout=60):
                    return True
            except Exception:
                pass
        else:
            self._force_relogin = False

        # Need login — show browser
        print(f"🔐 AuthZA{acct}: Opening browser for Google login…")
        print(f"   Please log in with Google. Browser will minimize after.")

        def _open_for_login():
            if self._context is not None and self._page is not None:
                try:
                    self._page.evaluate("window.moveTo(100, 100)")
                    self._page.evaluate("window.resizeTo(1024, 768)")
                    self._page.goto(ZA_CHAT_URL, wait_until="domcontentloaded", timeout=30000)
                    return
                except Exception:
                    pass
            self._start_context_impl(headless=False)
            self._navigate_impl()

        _pw_thread.execute(_open_for_login, timeout=60)

        # Poll for login
        start = time.time()
        while time.time() - start < 300:
            try:
                if _pw_thread.execute(self._is_logged_in_impl, timeout=10):
                    break
            except Exception:
                break
            time.sleep(1.5)
        else:
            raise RuntimeError(f"AuthZA{acct}: Login timed out.")

        print(f"✅ AuthZA{acct}: Login successful! Minimizing browser…")
        time.sleep(2)

        def _minimize():
            self._minimize_window_impl()
        _pw_thread.execute(_minimize, timeout=10)

        return True

    # -- send message via UI -------------------------------------------------

    def send_completion(
        self,
        messages: List[Dict],
        model: str,
        temperature: Optional[float],
        max_tokens: Optional[int],
        log_fn,
        log_stream: bool,
        timeout_ms: int = 600000,
    ) -> Dict:
        """Type the prompt into the chat UI, send it, and scrape the response."""
        t_start = time.time()

        # Combine messages into a single prompt
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"[System Instructions]\n{content}\n[End System Instructions]\n")
            elif role == "user":
                prompt_parts.append(content)
            elif role == "assistant":
                prompt_parts.append(f"[Previous Assistant Response]\n{content}\n")
        combined_prompt = "\n".join(prompt_parts)

        chunk_q = self._chunk_queue
        while not chunk_q.empty():
            try:
                chunk_q.get_nowait()
            except queue.Empty:
                break

        # JavaScript: type into chat, send, poll DOM for response
        js_chat = """
        async (prompt) => {
            // Helper: wait ms
            const wait = ms => new Promise(r => setTimeout(r, ms));

            // 1. Click "New Chat" to start fresh
            const newBtn = document.querySelector(
                'button[aria-label="New Chat"], [id="new-chat-button"]'
            ) || document.querySelector('nav a[href="/"]');
            if (newBtn) {
                newBtn.click();
                await wait(1500);
            }

            // 2. Find textarea
            let textarea = document.querySelector('textarea');
            if (!textarea) {
                // Try contenteditable
                textarea = document.querySelector('[contenteditable="true"]');
            }
            if (!textarea) {
                return {error: true, message: 'No chat input found. Title: ' + document.title + ' URL: ' + location.href};
            }

            // 3. Set value using Playwright-compatible method
            textarea.focus();
            if (textarea.tagName === 'TEXTAREA') {
                const setter = Object.getOwnPropertyDescriptor(
                    HTMLTextAreaElement.prototype, 'value'
                ).set;
                setter.call(textarea, prompt);
            } else {
                textarea.innerText = prompt;
            }
            textarea.dispatchEvent(new Event('input', {bubbles: true}));
            textarea.dispatchEvent(new Event('change', {bubbles: true}));
            await wait(500);

            // 4. Send — find the send/submit button
            let sent = false;
            // Try form submit button
            const submitBtn = textarea.closest('form')?.querySelector('button[type="submit"]');
            if (submitBtn && !submitBtn.disabled) {
                submitBtn.click();
                sent = true;
            }
            if (!sent) {
                // Try any nearby button with an SVG (icon button = send)
                const container = textarea.closest('div.relative, div.flex, form') || textarea.parentElement;
                const btns = container?.querySelectorAll('button') || [];
                for (const b of btns) {
                    if (b.querySelector('svg') && !b.disabled) {
                        b.click();
                        sent = true;
                        break;
                    }
                }
            }
            if (!sent) {
                // Last resort: Enter key
                textarea.dispatchEvent(new KeyboardEvent('keydown', {
                    key: 'Enter', code: 'Enter', keyCode: 13, bubbles: true, cancelable: true
                }));
            }

            // 5. Wait for response — poll DOM
            await wait(3000);

            let prevLen = 0;
            let stableCount = 0;
            let bestContent = '';

            for (let i = 0; i < 1200; i++) {  // max ~10 min
                await wait(500);

                // Find all rendered message blocks
                // Open WebUI uses .prose for markdown-rendered content
                const blocks = document.querySelectorAll('.prose');
                if (blocks.length < 2) continue;  // need at least user + assistant

                // Last .prose block is the assistant response
                const lastBlock = blocks[blocks.length - 1];
                const content = (lastBlock.innerText || '').trim();

                if (content.length === 0) continue;

                // Check if content is growing
                if (content.length > prevLen) {
                    // Emit the new portion
                    const delta = content.substring(prevLen);
                    try { await window._authza_emit(delta); } catch(e) {}
                    prevLen = content.length;
                    bestContent = content;
                    stableCount = 0;
                } else if (content.length === prevLen && content.length > 0) {
                    stableCount++;
                    bestContent = content;
                    // After 3 seconds of no change, consider it done
                    if (stableCount >= 6) {
                        return {content: bestContent, finish_reason: 'stop'};
                    }
                }
            }

            if (bestContent.length > 0) {
                return {content: bestContent, finish_reason: 'length'};
            }
            return {error: true, message: 'Timed out waiting for response'};
        }
        """

        def _do_send():
            page = self._page
            if page is None or page.is_closed():
                self._start_context_impl(headless=False)
                self._minimize_window_impl()
                self._navigate_impl()
                page = self._page

            if not self._chunk_fn_exposed:
                try:
                    page.expose_function(
                        "_authza_emit",
                        lambda data: chunk_q.put(data),
                    )
                    self._chunk_fn_exposed = True
                except Exception:
                    self._chunk_fn_exposed = True

            return page.evaluate(js_chat, combined_prompt)

        result_holder = {"result": None, "error": None}
        eval_done = threading.Event()

        def _run():
            try:
                result_holder["result"] = _do_send()
            except Exception as exc:
                result_holder["error"] = exc
            finally:
                eval_done.set()

        _pw_thread._ensure_running()
        rq: queue.Queue = queue.Queue()
        _pw_thread._work_q.put((_run, rq))

        # Stream chunks while waiting
        while not eval_done.is_set():
            if is_cancelled():
                break
            try:
                chunk = chunk_q.get(timeout=0.5)
                if log_stream and log_fn:
                    log_fn(chunk, end="", flush=True)
            except queue.Empty:
                continue

        # Drain
        while not chunk_q.empty():
            try:
                chunk = chunk_q.get_nowait()
                if log_stream and log_fn:
                    log_fn(chunk, end="", flush=True)
            except queue.Empty:
                break

        try:
            rq.get(timeout=10)
        except queue.Empty:
            pass

        elapsed = time.time() - t_start
        if log_stream:
            log_fn("")

        if result_holder["error"]:
            raise RuntimeError(f"AuthZA: Browser error: {result_holder['error']}")

        result = result_holder["result"]
        if result is None:
            raise RuntimeError("AuthZA: No response from browser")

        if result.get("error"):
            raise RuntimeError(f"AuthZA: {result.get('message', 'unknown')}")

        content = result.get("content", "")
        log_fn(f"✅ AuthZA: Response complete ({elapsed:.1f}s, {len(content)} chars)")

        return {
            "content": content,
            "finish_reason": result.get("finish_reason", "stop"),
            "usage": None,
        }

    def clear_session(self):
        self._force_relogin = True
        self.close()
        import shutil
        if os.path.exists(self._profile_dir):
            try:
                shutil.rmtree(self._profile_dir)
                os.makedirs(self._profile_dir, exist_ok=True)
            except Exception as exc:
                logger.warning("AuthZA: Failed to clear profile: %s", exc)


# ---------------------------------------------------------------------------
# Token Store — matches unified_api_client interface
# ---------------------------------------------------------------------------
class AuthZATokenStore:
    def __init__(self, account_id: int = 0):
        self._account_id = account_id
        profile_dir = (
            _BROWSER_PROFILE_DIR
            if account_id == 0
            else f"{_BROWSER_PROFILE_DIR}_{account_id}"
        )
        self._manager = _BrowserManager(profile_dir, account_id)

    @property
    def account_id(self) -> int:
        return self._account_id

    def account_info(self) -> str:
        return f"browser-session (account {self._account_id})"

    def get_valid_access_token(self, auto_login: bool = True) -> str:
        if not PLAYWRIGHT_AVAILABLE:
            raise RuntimeError(
                "AuthZA requires Playwright. Install it with:\n"
                "  pip install playwright && python -m playwright install chromium"
            )
        self._manager.ensure_logged_in()
        return "__browser_session__"

    def clear_tokens(self):
        self._manager.clear_session()

    @property
    def manager(self) -> _BrowserManager:
        return self._manager


# ---------------------------------------------------------------------------
# Module-level store singletons
# ---------------------------------------------------------------------------
_default_store: Optional[AuthZATokenStore] = None
_default_store_lock = threading.Lock()


def get_default_store() -> AuthZATokenStore:
    global _default_store
    if _default_store is None:
        with _default_store_lock:
            if _default_store is None:
                _default_store = AuthZATokenStore()
    return _default_store


_account_stores: Dict[int, AuthZATokenStore] = {}
_account_stores_lock = threading.Lock()


def get_store(account_id: Optional[int] = None) -> AuthZATokenStore:
    if account_id is None or account_id == 0:
        return get_default_store()
    with _account_stores_lock:
        if account_id in _account_stores:
            return _account_stores[account_id]
        store = AuthZATokenStore(account_id=account_id)
        _account_stores[account_id] = store
        return store


# ---------------------------------------------------------------------------
# Chat completion sender — public API
# ---------------------------------------------------------------------------
def send_chat_completion(
    access_token: str,
    messages: List[Dict],
    model: str = "glm-4-plus",
    temperature: Optional[float] = 0.7,
    max_tokens: Optional[int] = None,
    timeout: int = 600,
    base_url: Optional[str] = None,
    log_fn: Optional[Any] = None,
    connect_timeout: Optional[float] = None,
    account_id: int = 0,
) -> Dict:
    """Send a chat completion via Z.AI's chat UI (browser automation).

    ``access_token`` is ignored — auth is handled by browser cookies.
    """
    _log = log_fn or print

    log_stream = os.getenv("LOG_STREAM_CHUNKS", "1").lower() not in ("0", "false")
    if os.getenv("BATCH_TRANSLATION", "0") == "1":
        log_stream = os.getenv("ALLOW_AUTHZA_BATCH_STREAM_LOGS", "0").lower() not in (
            "0", "false",
        )

    store = get_store(account_id if account_id else None)
    return store.manager.send_completion(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        log_fn=_log,
        log_stream=log_stream,
        timeout_ms=timeout * 1000,
    )
