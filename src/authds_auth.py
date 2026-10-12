"""DeepSeek web chat through a dedicated, persistent Chrome/Edge profile.

Login takes place on DeepSeek's own site (including its Google login option).
Credentials never leave the browser profile. This is a desktop web-session route,
not the paid DeepSeek API. Every completion uses a new chat and a fresh proof.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import subprocess
import threading
import time
import urllib.request
from contextlib import contextmanager

import browser_driver
import cdp_browser
import mobile_runtime

BASE_URL = "https://chat.deepseek.com"
POW_WORKER_URL = "https://fe-static.deepseek.com/chat/static/76608.8f2a9fa413.js"
_cancel = threading.Event()
_login_gate = threading.Lock()
_login_generation = 0


class AuthDSError(RuntimeError):
    def __init__(self, message, error_type="api_error", retry_after=None):
        super().__init__(message)
        self.error_type = error_type
        # seconds the limit lasts; math.inf for an account mute (a key pool rests it for the session)
        self.retry_after = retry_after


def cancel_stream():
    _cancel.set()
    browser_driver.cancel_pages(BROWSER_OWNER)


def reset_cancel():
    _cancel.clear()


def _check(cancel_check=None):
    if _cancel.is_set() or (cancel_check and cancel_check()):
        raise AuthDSError("AuthDS: translation stopped by user", "cancelled")


def profile_dir():
    return Path(os.environ.get("AUTHDS_PROFILE_DIR") or
                (Path.home() / ".glossarion" / "authds_browser"))


def has_session():
    # A hint only: the server is always checked before sending a chapter.
    return (profile_dir() / ".signed_in").exists()


def _browser_binary():
    found = cdp_browser.find_browser_binary(
        "AUTHDS_BROWSER_BINARY", lambda message: AuthDSError(message, "config_error"))
    if found:
        return found
    raise AuthDSError("AuthDS needs Google Chrome or Microsoft Edge installed. You can also set AUTHDS_BROWSER_BINARY.", "config_error")

class _Page(cdp_browser.CdpPage):
    command_failed = "AuthDS browser command failed."
    command_timed_out = "AuthDS browser command timed out."
    script_failed = "AuthDS browser script failed. Reload DeepSeek and sign in again."

    def check(self, cancel_check):
        _check(cancel_check)

    def error(self, message, error_type):
        return AuthDSError(message, error_type)

#: browser_driver page owner (Glossarion Mobile's in-app WebView): cancel_stream() closes only these pages.
BROWSER_OWNER = "authds"
#: Where the mobile app's own sign-in page is (its DeepSeek sign-in opens chat.deepseek.com in a
#: visible in-app WebView; the hidden pages that translate share that WebView storage).
MOBILE_SIGN_IN_HINT = "Sign in to DeepSeek in Settings › Accounts › DeepSeek first."


class _DriverPage:
    """A ``browser_driver`` page (Glossarion Mobile's hidden WebView) with the CDP page's ``evaluate``.
    It has no ``call``: worker discovery falls back to the known proof-worker asset."""

    driver = True

    def __init__(self, page):
        self.page = page

    def evaluate(self, expression):
        try:
            return self.page.run_js(expression, 30000)
        except browser_driver.BrowserCancelled as exc:
            raise AuthDSError("AuthDS: translation stopped by user", "cancelled") from exc
        except Exception as exc:
            raise AuthDSError("AuthDS browser script failed. Reload DeepSeek and sign in again.",
                              "browser_error") from exc


@contextmanager
def _driver_page(cancel_check=None):
    """A DeepSeek page of the registered browser driver, loaded and ready (mobile; no subprocess)."""
    try:
        driver = browser_driver.require_driver("AuthDS")
    except browser_driver.BrowserUnavailable as exc:
        raise AuthDSError("AuthDS needs an in-app browser here (Glossarion Mobile) or the desktop app with "
                          "Chrome/Edge.", "config_error") from exc

    def cancelled():
        return _cancel.is_set() or bool(cancel_check and cancel_check())

    page = driver.open_page(owner=BROWSER_OWNER, cancel_check=cancelled)
    try:
        page.load(BASE_URL)
        wrapped = _DriverPage(page)
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            _check(cancel_check)
            try:
                if wrapped.evaluate("document.readyState !== 'loading' && location.origin === 'https://chat.deepseek.com'"):
                    break
            except AuthDSError as exc:
                if exc.error_type == "cancelled":
                    raise
            time.sleep(.3)
        else:
            raise AuthDSError("DeepSeek page did not finish loading. Check your connection and try again.",
                              "browser_error")
        yield wrapped
    finally:
        page.close()


def driver_session_present(cancel_check=None):
    """Mobile sign-in check: whether the in-app WebView storage holds a DeepSeek session (marks it)."""
    with _driver_page(cancel_check) as page:
        present = bool(page.evaluate(_TOKEN_PRESENT))
    marker = profile_dir() / ".signed_in"
    if present:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch(mode=0o600)
    else:
        marker.unlink(missing_ok=True)
    return present


_READY = "document.readyState !== 'loading' && location.origin === 'https://chat.deepseek.com'"
_NOT_LOADED = "DeepSeek page did not finish loading. Check your connection and try again."


def _launch_ready(page, cancel_check):
    # navigation may destroy the old JS context
    cdp_browser.wait_until(page, _READY, timeout=30, check=_check, error=AuthDSError, message=_NOT_LOADED,
                           cancel_check=cancel_check, tolerate=("browser_error", "timeout", "api_error"))


@contextmanager
def _launch_browser(*, visible=False, cancel_check=None):
    if browser_driver.use_driver("AUTHDS_MODE"):
        # Glossarion Mobile: the in-app WebView driver (no Chrome/Edge process); its sign-in is the app's.
        with _driver_page(cancel_check) as page:
            yield page
        return
    # Keep the process-spawn gate in the same function for the mobile collector.
    if not mobile_runtime.subprocesses_available():
        raise AuthDSError("AuthDS web login currently needs the desktop app and Chrome/Edge. On mobile, use deepseek/ with an API key.", "config_error")
    with cdp_browser.launch(
            binary=_browser_binary(), profile=profile_dir(), start_url=BASE_URL, ready=_launch_ready,
            page_factory=lambda url, check: _Page(url, check), check=_check, error=AuthDSError,
            closed_message="AuthDS browser closed. Close any other AuthDS login window and try again.",
            connect_message="AuthDS could not connect to its browser.", visible=visible,
            # Remote debugging sets navigator.webdriver, which makes DeepSeek's captcha always fail.
            extra_args=("--disable-blink-features=AutomationControlled",),
            cancel_check=cancel_check) as page:
        yield page


def _tab_ready(page, cancel_check):
    cdp_browser.wait_until(page, _READY, timeout=30, check=_check, error=AuthDSError, message=_NOT_LOADED,
                           cancel_check=cancel_check)


_LEASES = cdp_browser.BrowserLeases(
    launch=lambda **kwargs: _launch_browser(**kwargs), page_factory=lambda url, check: _Page(url, check),
    check=_check, url=BASE_URL, ready=_tab_ready)
_gate = _LEASES.gate


@contextmanager
def _open_browser(*, visible=False, cancel_check=None):
    """Lease an independent tab; only browser/profile management is serialized."""
    if browser_driver.use_driver("AUTHDS_MODE") or not mobile_runtime.subprocesses_available():
        with _launch_browser(visible=visible, cancel_check=cancel_check) as page:
            yield page
        return
    with _LEASES.open(visible=visible, cancel_check=cancel_check) as page:
        yield page


def _serialized(cancel_check=None, gate=None):
    # Serialize profile startup and control commands, never completion streams.
    return _LEASES.serialized(cancel_check, gate)

_TOKEN_PRESENT = """(() => {
 if (location.origin !== 'https://chat.deepseek.com') return false;
 const raw = localStorage.getItem('userToken'); if (!raw) return false;
 try { const t = JSON.parse(raw); return !!(typeof t === 'string' ? t : t && t.value); }
 catch (_) { return !!raw; }
})()"""


def _wait_login(page, log_fn, cancel_check=None, timeout=300):
    if getattr(page, "driver", False):
        # a hidden in-app page cannot show DeepSeek's login: only an existing session counts
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            _check(cancel_check)
            if page.evaluate(_TOKEN_PRESENT):
                (profile_dir() / ".signed_in").touch(mode=0o600)
                return
            time.sleep(.3)
        (profile_dir() / ".signed_in").unlink(missing_ok=True)
        raise AuthDSError(f"AuthDS: not signed in. {MOBILE_SIGN_IN_HINT}", "auth_error")
    log_fn("AuthDS: Sign in on the DeepSeek browser window. You can choose Continue with Google; Glossarion does not receive your Google password.")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        _check(cancel_check)
        try:
            present = page.evaluate(_TOKEN_PRESENT)
        except AuthDSError as exc:
            if exc.error_type != "browser_error":
                raise
            present = False  # Google sign-in redirects replace the JS context.
        if present:
            (profile_dir() / ".signed_in").touch(mode=0o600)
            return
        time.sleep(.5)
    raise AuthDSError("AuthDS login timed out. Select DeepSeek Login and try again.", "auth_error")


def login(log_fn=print, cancel_check=None):
    global _login_generation
    with _open_browser(visible=True, cancel_check=cancel_check) as page:
        with _serialized(cancel_check, _login_gate):
            _wait_login(page, log_fn, cancel_check)
            _login_generation += 1
    log_fn("AuthDS: Browser session saved.")


def _clear_page_token(page):
    # An expired token remains in localStorage. Remove it before interactive
    # re-login, otherwise the presence check would close the window immediately.
    (profile_dir() / ".signed_in").unlink(missing_ok=True)
    page.evaluate("(() => {if (location.origin === 'https://chat.deepseek.com') {localStorage.removeItem('userToken'); location.reload();} return true;})()")


def resolve_model(model):
    name = re.sub(r"^authds/", "", str(model).lower()).strip()
    choices = {"flash": ("default", False), "auto": ("default", False), "pro": ("expert", False),
               "flash-thinking": ("default", True), "pro-thinking": ("expert", True),
               "deepseek-chat": ("default", False), "deepseek-reasoner": ("default", True)}
    if name not in choices:
        raise AuthDSError(f"Unknown AuthDS model '{name}'. Choose flash, pro, flash-thinking, or pro-thinking.", "config_error")
    return choices[name]


_IMAGE_TYPES = {"image/png": "png", "image/jpeg": "jpg", "image/jpg": "jpg", "image/webp": "webp", "image/gif": "gif"}


def _image(part, error, label, index):
    """``{mime, b64, name}`` of an OpenAI-style ``image_url`` part holding a base64 data URI."""
    url = part.get("image_url")
    url = url.get("url") if isinstance(url, dict) else url
    match = re.match(r"data:([\w/+.-]+);base64,(.+)", str(url or ""), re.S)
    if not match or match.group(1).lower() not in _IMAGE_TYPES:
        raise error(f"{label} can only send PNG, JPEG, WebP or GIF images given as data URLs.", "config_error")
    mime = match.group(1).lower()
    return {"mime": mime, "b64": re.sub(r"\s+", "", match.group(2)), "name": f"image{index}.{_IMAGE_TYPES[mime]}"}


def build_prompt(messages, error=AuthDSError, label="AuthDS", images=None):
    """One web-chat prompt from chat messages, keeping each role as a ``[role]`` header.

    Messages may carry OpenAI-style parts (``text`` / ``image_url``); with an ``images`` list the
    images are collected into it (the route uploads them), otherwise they are refused."""
    parts = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, list):
            texts = []
            for part in content:
                kind = part.get("type") if isinstance(part, dict) else None
                if isinstance(part, str):
                    texts.append(part)
                elif kind == "text":
                    texts.append(str(part.get("text") or ""))
                elif kind == "image_url" and images is not None:
                    images.append(_image(part, error, label, len(images) + 1))
                else:
                    raise error(f"{label} currently supports text and images only.", "config_error")
            content = "\n".join(text for text in texts if text)
        elif not isinstance(content, str):
            raise error(f"{label} currently supports text messages only.", "config_error")
        if content:
            parts.append(f"[{message.get('role', 'user')}]\n{content}")
    if not parts and not images:
        raise error(f"{label} cannot send an empty prompt.", "config_error")
    return "\n\n".join(parts)


class StreamParser:
    """Incremental SSE parser; thinking is never mistaken for translated output."""
    def __init__(self):
        self.buffer = ""
        self.channel = "THINK"
        self.content = ""
        self.reasoning = ""
        self.tokens = None
        self.finished = False
        self.on_thinking = None  # optional callback for live thinking text

    def feed(self, text):
        self.buffer = (self.buffer + text).replace("\r\n", "\n")
        outputs = []
        while "\n\n" in self.buffer:
            frame, self.buffer = self.buffer.split("\n\n", 1)
            event = ""
            values = []
            for line in frame.splitlines():
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    values.append(line[5:].lstrip())
            data = "\n".join(values)
            if event in ("close", "finish") or data == "[DONE]":
                self.finished = True
                continue
            if not data:
                continue
            try:
                payload = json.loads(data)
            except ValueError as exc:
                raise AuthDSError("AuthDS returned an invalid stream frame.") from exc
            if not isinstance(payload, dict):
                raise AuthDSError("AuthDS returned an invalid stream payload.")
            if event == "error" or payload.get("error") or payload.get("code", 0) != 0:
                raise AuthDSError("AuthDS rejected the chat request. Check your session or account limits.")
            if event in ("ready", "title"):
                continue
            outputs.extend(self._patch(payload))
        return outputs

    def _fragment(self, fragment):
        self.channel = fragment.get("type", self.channel)
        text = fragment.get("content", "")
        return self._append(text) if isinstance(text, str) else []

    def _append(self, text):
        if self.channel == "RESPONSE":
            self.content += text
            return [text] if text else []
        if self.channel == "THINK":
            self.reasoning += text
            if text and self.on_thinking:
                self.on_thinking(text)
        return []

    def _patch(self, payload):
        path, operation, value = payload.get("p"), payload.get("o"), payload.get("v")
        outputs = []
        if isinstance(value, dict) and isinstance(value.get("response"), dict):
            response = value["response"]
            for fragment in response.get("fragments", []):
                outputs.extend(self._fragment(fragment))
            self.tokens = response.get("accumulated_token_usage", self.tokens)
        elif path == "response/fragments" and operation == "APPEND" and isinstance(value, list):
            for fragment in value:
                outputs.extend(self._fragment(fragment))
        elif isinstance(value, str) and (path is None or path.endswith("/content")):
            outputs.extend(self._append(value))
        elif path == "response" and operation == "BATCH" and isinstance(value, list):
            for patch in value:
                patch = dict(patch)
                patch["p"] = "response/" + patch.get("p", "")
                outputs.extend(self._patch(patch))
        elif path == "response/accumulated_token_usage":
            self.tokens = value
        elif path == "response/status" and value in ("FINISHED", "finished"):
            self.finished = True
        return outputs


# Async work stays in the authenticated origin; polling drains a bounded chunk queue.
# The site-provided proof worker supplies its current hash implementation.
_START_SCRIPT = r"""(() => {
 const cfg = __CONFIG__;
 const state = window.__glossarionDS = {chunks:[], done:false, error:null, status:0};
 const controller = new AbortController(); state.abort = () => controller.abort();
 (async () => {
  const raw = localStorage.getItem('userToken'); let token = raw;
  try {const t=JSON.parse(raw); token=typeof t==='string'?t:t.value;} catch (_) {}
  if (!token) { state.status=401; throw Error('Sign in again'); }
  const headers = {'authorization':'Bearer '+token,'content-type':'application/json',
   'x-client-platform':'web','x-client-version':'2.2.0','x-client-locale':'en_US',
   'x-client-bundle-id':'com.deepseek.chat'};
  async function post(path, body) {
   const r=await fetch(path,{method:'POST',headers,body:JSON.stringify(body),signal:controller.signal});
   state.status=r.status;
   if (!r.ok) throw Error('HTTP '+r.status);
   const j=await r.json();
   if(j.code!==0 || j.data?.biz_code!==0) {
    const detail=String(j.msg || '')+' '+String(j.data?.biz_msg || '');
    if(/login|token|unauthorized|expired|登录|令牌/i.test(detail)) state.status=401;
    throw Error('Request rejected by DeepSeek');
   }
   return j.data.biz_data;
  }
  // A muted account (DeepSeek's limit after too many requests) fails every chat: say so up front.
  try {
   const me=await fetch('/api/v0/users/current',{headers,signal:controller.signal});
   const chat=me.ok ? (await me.json()).data?.biz_data?.chat : null;
   if(chat && chat.is_muted){
    state.status=429;
    if(chat.mute_until) state.retryAfter=Math.max(60, Math.ceil(chat.mute_until - Date.now()/1000));
    const until=chat.mute_until ? new Date(chat.mute_until*1000).toLocaleString() : 'later';
    throw Error('DeepSeek has muted this account until '+until+' (too many requests). Try again then.');
   }
  } catch(e) { if(state.status===429 || e.name==='AbortError') throw e; }
  const target='/api/v0/chat/completion', uploadTarget='/api/v0/file/upload_file';
  let workerSource=null;
  // Each completion and each upload needs its own proof of work for its path.
  async function proofFor(path) {
   const {challenge}=await post('/api/v0/chat/create_pow_challenge',{target_path:path});
   if(workerSource===null){
    const workerResponse=await fetch(cfg.worker || cfg.fallbackWorker, {signal:controller.signal});
    if(!workerResponse.ok) throw Error('Proof worker unavailable; update AUTHDS_POW_WORKER_URL');
    workerSource=await workerResponse.text();
   }
   const workerURL=URL.createObjectURL(new Blob([workerSource],{type:'application/javascript'}));
   const worker=new Worker(workerURL);
   let answer;
   try { answer=await new Promise((resolve,reject)=>{
    const timer=setTimeout(()=>reject(Error('Proof timed out')),120000);
    controller.signal.addEventListener('abort',()=>{clearTimeout(timer);reject(Error('Cancelled'));},{once:true});
    worker.onerror=()=>{clearTimeout(timer);reject(Error('Proof worker failed'));};
    worker.onmessage=e=>{clearTimeout(timer);
     if(e.data?.type==='pow-answer' && e.data.answer) resolve(e.data.answer);
     else reject(Error('Invalid proof response'));};
    worker.postMessage({type:'pow-challenge',challenge:{...challenge,expireAt:challenge.expire_at}});
   }); } finally {worker.terminate();URL.revokeObjectURL(workerURL);}
   const proof={algorithm:answer.algorithm,challenge:answer.challenge,salt:answer.salt,
    answer:answer.answer,signature:answer.signature,target_path:path};
   let binary=''; for(const byte of new TextEncoder().encode(JSON.stringify(proof))) binary+=String.fromCharCode(byte);
   return btoa(binary);
  }
  // Images: uploaded like the site's own attach button, then referenced by the completion
  // once DeepSeek has processed them.
  const fileIds=[];
  for(const image of cfg.images || []){
   const raw=atob(image.b64), bytes=new Uint8Array(raw.length);
   for(let i=0;i<raw.length;i++) bytes[i]=raw.charCodeAt(i);
   const form=new FormData(); form.append('file',new Blob([bytes],{type:image.mime}),image.name);
   const uploadHeaders={...headers,'x-ds-pow-response':await proofFor(uploadTarget),
    'x-thinking-enabled':cfg.thinking?'1':'0','x-model-type':cfg.model,'x-file-size':String(bytes.length)};
   delete uploadHeaders['content-type'];
   const uploaded=await fetch(uploadTarget,{method:'POST',headers:uploadHeaders,body:form,signal:controller.signal});
   state.status=uploaded.status;
   if(!uploaded.ok) throw Error('Image upload failed: HTTP '+uploaded.status);
   const u=await uploaded.json();
   if(u.code!==0 || u.data?.biz_code!==0) throw Error('DeepSeek refused the image: '+String(u.data?.biz_msg || u.msg || ''));
   const id=u.data.biz_data.id;
   const {'content-type':_ignored,...readHeaders}=headers;
   for(let attempt=0;;attempt++){
    const f=await fetch('/api/v0/file/fetch_files?file_ids='+encodeURIComponent(id),{headers:readHeaders,signal:controller.signal});
    const file=(await f.json()).data?.biz_data?.files?.[0];
    if(file && file.status==='SUCCESS') break;
    if(file && /FAIL|ERROR|EMPTY|UNSUPPORTED/i.test(String(file.status))) throw Error('DeepSeek could not read the image ('+file.status+')');
    if(attempt>=120) throw Error('DeepSeek took too long to process the image');
    await new Promise(r=>setTimeout(r,1000));
   }
   fileIds.push(id);
  }
  const session=await post('/api/v0/chat_session/create',{});
  headers['x-ds-pow-response']=await proofFor(target);
  const response=await fetch(target,{method:'POST',headers,signal:controller.signal,body:JSON.stringify({
   chat_session_id:session.chat_session.id,parent_message_id:null,model_type:cfg.model,
   prompt:cfg.prompt,ref_file_ids:fileIds,thinking_enabled:cfg.thinking,search_enabled:false,action:null,preempt:false})});
  state.status=response.status;
  if(!response.ok) throw Error('HTTP '+response.status);
  const reader=response.body.getReader(), decoder=new TextDecoder();
  while(true){const {done,value}=await reader.read();if(done)break;
   state.chunks.push(decoder.decode(value,{stream:true}));
   while(state.chunks.length>32){await new Promise(r=>setTimeout(r,50));if(controller.signal.aborted)throw Error('Cancelled');}}
  state.chunks.push(decoder.decode());
 })().catch(e=>{state.error=e.message;}).finally(()=>{state.done=true;});
 return true;
})()"""


def _discover_worker(page):
    """Read worker targets after login without modifying the page's Worker API."""
    from urllib.parse import urlparse
    call = getattr(page, "call", None)
    if not callable(call):
        return None
    try:
        targets = call("Target.getTargets").get("targetInfos", [])
    except AuthDSError as exc:
        if exc.error_type == "cancelled":
            raise
        return None
    for target in targets:
        if target.get("type") not in ("worker", "shared_worker"):
            continue
        url = target.get("url", "")
        parsed = urlparse(url)
        if (parsed.scheme == "https" and
                parsed.hostname in ("chat.deepseek.com", "fe-static.deepseek.com") and
                parsed.path.startswith("/chat/static/") and parsed.path.endswith(".js")):
            return url
    return None


def _complete(page, model, messages, timeout, cancel_check, on_delta, before_send_callback, on_thinking=None):
    model_type, thinking = resolve_model(model)
    images = []
    cfg = {"model": model_type, "thinking": thinking, "prompt": build_prompt(messages, images=images),
           "images": images,
           "worker": os.environ.get("AUTHDS_POW_WORKER_URL") or _discover_worker(page),
           "fallbackWorker": POW_WORKER_URL}
    if before_send_callback:
        before_send_callback()
    page.evaluate(_START_SCRIPT.replace("__CONFIG__", json.dumps(cfg)))
    parser = StreamParser()
    parser.on_thinking = on_thinking
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            _check(cancel_check)
            state = page.evaluate("(() => {const s=window.__glossarionDS; return s ? {chunks:s.chunks.splice(0),done:s.done,error:s.error,status:s.status,retryAfter:s.retryAfter||null} : null;})()")
            if not state:
                raise AuthDSError("AuthDS page was reloaded during translation.", "browser_error")
            if state.get("error"):
                status = state.get("status", 0)
                kind = "auth_error" if status == 401 else "access_denied" if status == 403 else "rate_limit" if status == 429 else "api_error"
                # No server bodies, tokens, or browser state are included in logs.
                raise AuthDSError(f"AuthDS: {state['error']}", kind,
                                  retry_after=math.inf if state.get("retryAfter") else None)
            for chunk in state.get("chunks", []):
                for delta in parser.feed(chunk):
                    if on_delta:
                        on_delta(delta)
            if state.get("done"):
                for delta in parser.feed("\n\n"):
                    if on_delta:
                        on_delta(delta)
                if not parser.finished:
                    raise AuthDSError("AuthDS stream ended before its completion marker. Partial translation discarded.")
                if not parser.content.strip():
                    raise AuthDSError("AuthDS returned no answer text. Thinking text was excluded.")
                return {"content": parser.content, "reasoning_content": parser.reasoning,
                        "finish_reason": "stop", "usage": {"total_tokens": parser.tokens} if parser.tokens is not None else None}
            time.sleep(.1)
        raise AuthDSError("AuthDS translation timed out; partial translation discarded.", "timeout")
    finally:
        try:
            page.evaluate("window.__glossarionDS?.abort()")
        except Exception:
            pass


def send_chat_completion(*, messages, model="flash", timeout=300, log_fn=print,
                         cancel_check=None, on_delta=None, before_send_callback=None, on_thinking=None):
    global _login_generation
    resolve_model(model)
    build_prompt(messages, images=[])
    login_generation = _login_generation
    # Try the saved session once, then permit an interactive re-login once.
    for attempt in range(2):
        visible = not has_session() or attempt == 1
        with _open_browser(visible=visible, cancel_check=cancel_check) as page:
            if visible:
                with _serialized(cancel_check, _login_gate):
                    if attempt and _login_generation == login_generation and not getattr(page, "driver", False):
                        _clear_page_token(page)
                    _wait_login(page, log_fn, cancel_check)
                    _login_generation += 1
                    login_generation = _login_generation
            else:
                # Wait for the initial document load without opening a login window.
                deadline = time.monotonic() + 20
                while not page.evaluate(_TOKEN_PRESENT) and time.monotonic() < deadline:
                    _check(cancel_check)
                    time.sleep(.2)
            try:
                return _complete(page, model, messages, float(timeout), cancel_check, on_delta, before_send_callback, on_thinking)
            except AuthDSError as exc:
                if exc.error_type != "auth_error" or attempt:
                    raise
                with _serialized(cancel_check, _login_gate):
                    if _login_generation == login_generation:
                        (profile_dir() / ".signed_in").unlink(missing_ok=True)
                log_fn("AuthDS: Session expired. Opening DeepSeek login again.")
    raise AuthDSError("AuthDS login failed.", "auth_error")


if __name__ == "__main__":
    login()
