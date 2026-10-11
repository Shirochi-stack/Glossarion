"""ChatGPT web chat (chatgpt.com) through a dedicated, headless Chrome/Edge profile.

Works without an account: chatgpt.com answers logged-out visitors. The ChatGPT login is
optional and is the same one ``authgpt/`` uses (``authgpt_auth`` default token store): when
it is present, the page's backend calls carry that OAuth access token, so the chat runs on
the user's account and plan. The page itself produces the anti-abuse proofs (sentinel,
Turnstile, proof of work) for every request; Glossarion only replaces the prompt and the
model of the request the page sends, and reads the streamed answer.

Every completion opens a new tab and a new chat. Logged-in chats are temporary chats, so
translations do not fill the account's chat history. Desktop only.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import threading
import time
from contextlib import contextmanager

import cdp_browser
import mobile_runtime
from authds_auth import build_prompt as _build_prompt

BASE_URL = "https://chatgpt.com"
PREFIX = "authchatgpt/"
DEFAULT_MODEL = "auto"
# chatgpt.com serves logged-out visitors a different web app whose requests cannot carry
# the OAuth token; any session cookie makes it serve the regular app (still logged out
# until the token is added to its backend calls).
_APP_COOKIE = {"name": "__Secure-next-auth.session-token", "value": "glossarion",
               "url": BASE_URL + "/", "httpOnly": True, "secure": True}
_cancel = threading.Event()


class AuthChatGPTError(RuntimeError):
    def __init__(self, message, error_type="api_error"):
        super().__init__(message)
        self.error_type = error_type


def cancel_stream():
    _cancel.set()


def reset_cancel():
    _cancel.clear()


def _check(cancel_check=None):
    if _cancel.is_set() or (cancel_check and cancel_check()):
        raise AuthChatGPTError("AuthChatGPT: translation stopped by user", "cancelled")


def profile_dir():
    return Path(os.environ.get("AUTHCHATGPT_PROFILE_DIR") or
                (Path.home() / ".glossarion" / "authchatgpt_browser"))


def resolve_model(model):
    """The chatgpt.com model slug of ``authchatgpt/<slug>`` (``auto`` when empty)."""
    name = re.sub(r"^authchatgpt/", "", str(model).strip(), flags=re.IGNORECASE).strip().lower()
    if not name:
        return DEFAULT_MODEL
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", name):
        raise AuthChatGPTError(f"Unknown AuthChatGPT model '{name}'. Use a chatgpt.com model slug such as "
                               "auto, gpt-6 or gpt-6-thinking.", "config_error")
    return name


def build_prompt(messages):
    return _build_prompt(messages, AuthChatGPTError, "AuthChatGPT")


# ---------------------------------------------------------------------------
# Optional login: the authgpt/ ChatGPT token
# ---------------------------------------------------------------------------

def _token_store():
    import authgpt_auth
    return authgpt_auth.get_default_store()


def has_login():
    """Whether the authgpt/ ChatGPT login is saved (it may still need a refresh)."""
    try:
        return _token_store().has_tokens
    except Exception:
        return False


def _access_token(log_fn):
    """The saved ChatGPT access token, refreshed when due; None means logged out. Never
    opens a login: without a usable token the chat runs logged out."""
    try:
        store = _token_store()
        if not store.has_tokens:
            return None
        return store.get_valid_access_token(auto_login=False)
    except Exception:
        log_fn("AuthChatGPT: the saved ChatGPT login could not be refreshed; continuing logged out. "
               "Click ChatGPT Login to sign in again.")
        return None


def _refreshed_token(rejected):
    """A new access token after chatgpt.com rejected ``rejected``, or None."""
    try:
        store = _token_store()
        tokens = store.load_tokens() or {}
        if tokens.get("access_token") and tokens["access_token"] != rejected:
            return tokens["access_token"]  # another request already refreshed it
        refreshed = store._try_refresh(tokens) if tokens else None
        return (refreshed or {}).get("access_token")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Browser
# ---------------------------------------------------------------------------

class _Page(cdp_browser.CdpPage):
    command_failed = "AuthChatGPT browser command failed."
    command_timed_out = "AuthChatGPT browser command timed out."
    script_failed = "AuthChatGPT browser script failed. Reload ChatGPT and try again."

    def check(self, cancel_check):
        _check(cancel_check)

    def error(self, message, error_type):
        return AuthChatGPTError(message, error_type)


def _browser_binary():
    found = cdp_browser.find_browser_binary(
        "AUTHCHATGPT_BROWSER_BINARY", lambda message: AuthChatGPTError(message, "config_error"))
    if found:
        return found
    raise AuthChatGPTError("AuthChatGPT needs Google Chrome or Microsoft Edge installed. "
                           "You can also set AUTHCHATGPT_BROWSER_BINARY.", "config_error")


@contextmanager
def _launch_browser(*, visible=False, cancel_check=None):
    if not mobile_runtime.subprocesses_available():
        raise AuthChatGPTError("AuthChatGPT needs the desktop app and Chrome/Edge.", "config_error")
    with cdp_browser.launch(
            binary=_browser_binary(), profile=profile_dir(), start_url="about:blank",
            ready=lambda page, check: None, page_factory=lambda url, check: _Page(url, check),
            check=_check, error=AuthChatGPTError,
            closed_message="AuthChatGPT browser closed. Try again.",
            connect_message="AuthChatGPT could not connect to its browser.", visible=visible,
            extra_args=("--disable-blink-features=AutomationControlled", "--window-size=1400,1000"),
            cancel_check=cancel_check) as page:
        yield page


_READY = ("location.origin === 'https://chatgpt.com' && !!window.__glossarionGPT && "
          "!!document.querySelector('#prompt-textarea')")


def _tab_ready(page, cancel_check):
    cdp_browser.wait_until(
        page, _READY, timeout=60, check=_check, error=AuthChatGPTError, cancel_check=cancel_check,
        message="ChatGPT page did not finish loading. Check your connection and try again.")


_LEASES = cdp_browser.BrowserLeases(
    launch=lambda **kwargs: _launch_browser(**kwargs), page_factory=lambda url, check: _Page(url, check),
    check=_check, url=BASE_URL + "/", ready=_tab_ready)


def _prepare(config):
    def prepare(page):
        page.call("Page.enable")
        # Cloudflare stops the headless user agent; the browser is otherwise unchanged.
        agent = page.call("Browser.getVersion").get("userAgent", "").replace("HeadlessChrome", "Chrome")
        if agent:
            page.call("Emulation.setUserAgentOverride", {"userAgent": agent})
        page.call("Network.setCookie", _APP_COOKIE)
        page.call("Page.addScriptToEvaluateOnNewDocument",
                  {"source": _HOOK_SCRIPT.replace("__CONFIG__", json.dumps(config))})
    return prepare


@contextmanager
def _open_tab(config, cancel_check=None):
    """A new chatgpt.com tab of the shared browser, hooked with ``config``."""
    if not mobile_runtime.subprocesses_available():
        raise AuthChatGPTError("AuthChatGPT needs the desktop app and Chrome/Edge.", "config_error")
    with _LEASES.open(cancel_check=cancel_check, prepare=_prepare(config)) as page:
        yield page


# Runs before the page's own scripts. Logged in, every backend call goes to the account
# API with the token; the conversation request gets Glossarion's prompt and model, and its
# streams (the reply may continue on /resume) are copied into a queue Python drains.
_HOOK_SCRIPT = r"""(() => {
 if (window.__glossarionGPT) return;
 const cfg = __CONFIG__;
 const state = window.__glossarionGPT = {chunks:[], requested:false, streams:0, open:0, closedAt:0, status:0, error:null, sentinel:0};
 const nativeFetch = window.fetch;
 const backend = /^\/backend-(anon|api)\//, conversation = /^\/backend-(anon|api)\/f\/conversation(\/resume)?$/;
 window.fetch = async function(input, init) {
  // Building a Request from a Request consumes its body: inspect first, rebuild only to rewrite.
  const url = new URL(input instanceof Request ? input.url : String(input), location.href);
  if (url.origin !== location.origin || !backend.test(url.pathname)) return nativeFetch.call(this, input, init);
  const method = ((init && init.method) || (input instanceof Request ? input.method : 'GET')).toUpperCase();
  const isConversation = method === 'POST' && conversation.test(url.pathname);
  if (!cfg.token && !isConversation) return nativeFetch.call(this, input, init);
  const request = new Request(input, init), headers = new Headers(request.headers);
  if (cfg.token) {
   url.pathname = url.pathname.replace(/^\/backend-anon\//, '/backend-api/');
   headers.set('authorization', 'Bearer ' + cfg.token);
  }
  let body = ['GET', 'HEAD'].includes(request.method) ? undefined : await request.clone().arrayBuffer();
  if (isConversation && !url.pathname.endsWith('/resume')) {
   // One chat turn per tab: a repeated Send never becomes a second request.
   if (state.requested) return new Response(null, {status:409});
   state.requested = true;
   const payload = JSON.parse(new TextDecoder().decode(body));
   payload.messages[payload.messages.length - 1].content = {content_type:'text', parts:[cfg.prompt]};
   payload.model = cfg.model;
   if (cfg.temporary) payload.history_and_training_disabled = true;
   body = JSON.stringify(payload);
  }
  const response = await nativeFetch.call(this, url.toString(), {method:request.method, headers, body,
   signal:request.signal, credentials:'include', cache:request.cache, redirect:request.redirect,
   referrer:request.referrer, referrerPolicy:request.referrerPolicy});
  if (/\/sentinel\/chat-requirements/.test(url.pathname) && !response.ok) state.sentinel = response.status;
  if (!isConversation) return response;
  state.status = response.status;
  if (!response.ok || !response.body) {
   let detail = '';
   try { const j = await response.clone().json(); detail = typeof j.detail === 'string' ? j.detail : (j.detail && j.detail.message) || ''; } catch (_) {}
   state.error = ('HTTP ' + response.status + (detail ? ': ' + detail : '')).slice(0, 300);
   return response;
  }
  const [page, copy] = response.body.tee();
  state.streams += 1; state.open += 1;
  (async () => {
   const reader = copy.getReader(), decoder = new TextDecoder();
   try { while (true) { const {done, value} = await reader.read(); if (done) break; state.chunks.push(decoder.decode(value, {stream:true})); }
         state.chunks.push(decoder.decode() + '\n\n'); }
   catch (_) { state.chunks.push('\n\n'); }
   finally { state.open -= 1; state.closedAt = Date.now(); }
  })();
  return new Response(page, {status:response.status, statusText:response.statusText, headers:response.headers});
 };
})()"""

_POLL = """(() => {const s = window.__glossarionGPT; return s ? {chunks:s.chunks.splice(0), streams:s.streams,
 open:s.open, idle:s.closedAt ? Date.now() - s.closedAt : 0, status:s.status, error:s.error, sentinel:s.sentinel} : null;})()"""
_COMPOSER = ("(() => {const c = document.querySelector('#prompt-textarea'), b = document.querySelector('[data-testid=\"send-button\"]');"
             " return {text: c ? c.innerText.trim() : null, send: !!b && !b.disabled,"
             " requested: !!(window.__glossarionGPT && (window.__glossarionGPT.requested || window.__glossarionGPT.sentinel))};})()")
_SEND = "(() => {const b = document.querySelector('[data-testid=\"send-button\"]'); if (b) b.click(); return true;})()"


# ---------------------------------------------------------------------------
# Stream
# ---------------------------------------------------------------------------

_CODE_OPEN = re.compile(r'<CodeBlock(?:\s+language="([^"]*)")?[^>]*>\n?')


def clean_markup(text):
    """chatgpt.com's reply markup back to the model's markdown: ``<escape>`` wraps raw
    HTML and ``<CodeBlock language="x">`` is a fenced code block."""
    text = _CODE_OPEN.sub(lambda m: f"```{m.group(1) or ''}\n", text)
    text = re.sub(r"\n?</CodeBlock>", "\n```", text)
    return text.replace("<escape>", "").replace("</escape>", "")


class StreamParser:
    """Incremental parser of chatgpt.com's conversation stream (SSE, delta encoding v1).

    Only the assistant's visible text message is the answer; reasoning summaries, tool
    calls and system messages are kept out of it.
    """

    def __init__(self):
        self.buffer = ""
        self.message = None  # the message the patches apply to
        self.answer = None  # the latest visible assistant text message
        self.path = None
        self.operation = None
        self.reasoning = ""
        self.finished = False
        self.error = None
        self.model = None
        self.on_thinking = None

    @property
    def content(self):
        if not self.answer:
            return ""
        return clean_markup("".join(p for p in self.answer["content"].get("parts", []) if isinstance(p, str)))

    @staticmethod
    def _visible(message):
        content = message.get("content")
        content = content if isinstance(content, dict) else {}
        return ((message.get("author") or {}).get("role") == "assistant" and
                content.get("content_type") == "text" and message.get("recipient", "all") == "all" and
                message.get("channel") in (None, "final"))

    def feed(self, text):
        self.buffer = (self.buffer + text).replace("\r\n", "\n")
        outputs = []
        while "\n\n" in self.buffer:
            frame, self.buffer = self.buffer.split("\n\n", 1)
            values = [line[5:].lstrip() for line in frame.splitlines() if line.startswith("data:")]
            data = "\n".join(values)
            if not data:
                continue
            if data == "[DONE]":
                self.finished = True
                continue
            try:
                payload = json.loads(data)
            except ValueError as exc:
                raise AuthChatGPTError("AuthChatGPT returned an invalid stream frame.") from exc
            if isinstance(payload, dict):
                outputs.extend(self._event(payload))
        return outputs

    def _event(self, payload):
        if payload.get("type") == "message_stream_complete":
            self.finished = True
            return []
        if payload.get("error"):
            error = payload["error"]
            self.error = error if isinstance(error, str) else (error.get("message") if isinstance(error, dict) else None) or "error"
            return []
        if "type" in payload and "v" not in payload:
            return []  # markers, metadata, resume tokens
        value = payload.get("v")
        if isinstance(value, dict) and isinstance(value.get("message"), dict):
            return self._start(value["message"])
        if "p" in payload:
            self.path, self.operation = payload.get("p"), payload.get("o", self.operation)
        elif "o" in payload:
            self.operation = payload["o"]
        if isinstance(value, list) and (payload.get("o") == "patch" or "p" not in payload):
            outputs = []
            for patch in value:
                if isinstance(patch, dict) and "p" in patch:
                    self.path, self.operation = patch["p"], patch.get("o", "replace")
                    outputs.extend(self._apply(self.path, self.operation, patch.get("v")))
            return outputs
        if self.path is not None:
            return self._apply(self.path, self.operation or "append", value)
        return []

    def _start(self, message):
        self.message = message
        self.path = self.operation = None
        metadata = message.get("metadata") or {}
        self.model = metadata.get("resolved_model_slug") or metadata.get("model_slug") or self.model
        if not self._visible(message):
            return []
        message.setdefault("content", {}).setdefault("parts", [""])
        self.answer = message
        return [p for p in message["content"]["parts"] if isinstance(p, str) and p]

    def _apply(self, path, operation, value):
        if self.message is None or not isinstance(path, str) or not path.startswith("/message"):
            return []
        keys = [k for k in path.split("/")[2:] if k != ""]
        if not keys:
            return []
        node = self.message
        for key in keys[:-1]:
            if isinstance(node, list):
                if not key.isdigit() or int(key) >= len(node):
                    return []
                node = node[int(key)]
            elif isinstance(node, dict):
                node = node.setdefault(key, {})
            else:
                return []
        last = keys[-1]
        if isinstance(node, list):
            if not last.isdigit():
                return []
            index = int(last)
            while len(node) <= index:
                node.append("")
            current = node[index]
        elif isinstance(node, dict):
            current = node.get(last)
        else:
            return []
        if operation == "append" and isinstance(current, str) and isinstance(value, str):
            new = current + value
        elif operation == "append" and isinstance(current, list):
            new = current + (value if isinstance(value, list) else [value])
        elif operation == "append" and isinstance(current, dict) and isinstance(value, dict):
            new = {**current, **value}
        elif operation in ("append", "replace", "add"):
            new = value
        elif operation == "remove" and isinstance(node, dict):
            node.pop(last, None)
            return []
        else:
            return []
        if isinstance(node, list):
            node[int(last)] = new
        else:
            node[last] = new
        text = value if isinstance(value, str) and operation == "append" else ""
        if self._visible(self.message):
            self.answer = self.message
            return [text] if text and keys[:2] == ["content", "parts"] else []
        if text and "thoughts" in keys:
            self.reasoning += text
            if self.on_thinking:
                self.on_thinking(text)
        return []


def _kind(status):
    return {401: "auth_error", 403: "access_denied", 429: "rate_limit"}.get(status, "api_error")


def _submit(page, cancel_check, before_send_callback):
    """Send one message from the composer; the hook swaps in the real prompt and model.

    The composer needs some text before the page enables Send. While the app is still
    starting, a re-render can wipe that text or ignore a click, so this types and clicks
    again until the page has made its conversation request.
    """
    deadline = time.monotonic() + 45
    clicked = None
    while time.monotonic() < deadline:
        _check(cancel_check)
        state = page.evaluate(_COMPOSER) or {}
        if state.get("requested"):
            return
        waiting = clicked is not None and time.monotonic() - clicked < 8
        if state.get("text") == "" and not waiting:
            page.evaluate("document.querySelector('#prompt-textarea').focus()")
            page.call("Input.insertText", {"text": "."})
        elif state.get("text") and state.get("send") and not waiting:
            if clicked is None and before_send_callback:
                before_send_callback()
            page.evaluate(_SEND)
            clicked = time.monotonic()
        time.sleep(.3)
    raise AuthChatGPTError("ChatGPT did not accept the message. Try again.", "browser_error")


def _complete(page, timeout, cancel_check, on_delta, before_send_callback, on_thinking=None):
    _submit(page, cancel_check, before_send_callback)
    parser = StreamParser()
    parser.on_thinking = on_thinking
    started = time.monotonic()
    deadline = started + timeout
    while time.monotonic() < deadline:
        _check(cancel_check)
        state = page.evaluate(_POLL)
        if not state:
            raise AuthChatGPTError("AuthChatGPT page was reloaded during translation.", "browser_error")
        if state.get("error"):
            # Server detail text only: tokens and browser state never reach the log.
            raise AuthChatGPTError(f"AuthChatGPT: {state['error']}", _kind(state.get("status", 0)))
        for chunk in state.get("chunks", []):
            for delta in parser.feed(chunk):
                if on_delta:
                    on_delta(delta)
        if parser.error:
            raise AuthChatGPTError(f"AuthChatGPT: {parser.error}")
        if parser.finished and not state.get("open"):
            if not parser.content.strip():
                raise AuthChatGPTError("AuthChatGPT returned no answer text.")
            return {"content": parser.content, "reasoning_content": parser.reasoning,
                    "finish_reason": "stop", "usage": None, "model": parser.model}
        sentinel = state.get("sentinel") or 0
        if sentinel and not state.get("streams"):
            raise AuthChatGPTError(f"AuthChatGPT: ChatGPT refused the request check (HTTP {sentinel}).",
                                   _kind(sentinel))
        if not state.get("streams") and time.monotonic() - started > 60:
            raise AuthChatGPTError("AuthChatGPT: ChatGPT did not start a reply. It may be showing a "
                                   "usage limit; try again later or click ChatGPT Login.", "rate_limit")
        # A reply that moved to another stream (/resume) is picked up there.
        if state.get("streams") and not state.get("open") and state.get("idle", 0) > 30000 and not parser.finished:
            raise AuthChatGPTError("AuthChatGPT stream ended before its completion marker. "
                                   "Partial translation discarded.")
        time.sleep(.1)
    raise AuthChatGPTError("AuthChatGPT translation timed out; partial translation discarded.", "timeout")


def send_chat_completion(*, messages, model=DEFAULT_MODEL, timeout=300, log_fn=print,
                         cancel_check=None, on_delta=None, before_send_callback=None, on_thinking=None):
    slug = resolve_model(model)
    prompt = build_prompt(messages)
    token = _access_token(log_fn)
    for attempt in range(2):
        config = {"token": token, "prompt": prompt, "model": slug, "temporary": bool(token)}
        with _open_tab(config, cancel_check) as page:
            try:
                return _complete(page, float(timeout), cancel_check, on_delta, before_send_callback, on_thinking)
            except AuthChatGPTError as exc:
                if exc.error_type != "auth_error" or not token or attempt:
                    raise
        replacement = _refreshed_token(token)
        if not replacement:
            log_fn("AuthChatGPT: ChatGPT rejected the saved login; continuing logged out. "
                   "Click ChatGPT Login to sign in again.")
        token = replacement
    raise AuthChatGPTError("AuthChatGPT request failed.", "auth_error")
