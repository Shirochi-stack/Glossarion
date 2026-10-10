"""DeepSeek web chat through a dedicated, persistent Chrome/Edge profile.

Login takes place on DeepSeek's own site (including its Google login option).
Credentials never leave the browser profile. This is a desktop web-session route,
not the paid DeepSeek API. Every completion uses a new chat and a fresh proof.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from contextlib import contextmanager

import mobile_runtime

BASE_URL = "https://chat.deepseek.com"
POW_WORKER_URL = "https://fe-static.deepseek.com/chat/static/76608.8f2a9fa413.js"
_cancel = threading.Event()
_gate = threading.Lock()


class AuthDSError(RuntimeError):
    def __init__(self, message, error_type="api_error"):
        super().__init__(message)
        self.error_type = error_type


def cancel_stream():
    _cancel.set()


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
    configured = os.environ.get("AUTHDS_BROWSER_BINARY")
    if configured:
        if Path(configured).is_file():
            return configured
        raise AuthDSError("AUTHDS_BROWSER_BINARY does not point to a browser executable.", "config_error")
    candidates = []
    for root in (os.environ.get("PROGRAMFILES"), os.environ.get("PROGRAMFILES(X86)"),
                 os.environ.get("LOCALAPPDATA")):
        if root:
            candidates += [str(Path(root) / p) for p in
                           ("Google/Chrome/Application/chrome.exe", "Microsoft/Edge/Application/msedge.exe")]
    candidates += ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                   "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"]
    for candidate in candidates:
        if Path(candidate).is_file():
            return candidate
    for name in ("google-chrome", "chromium", "chromium-browser", "microsoft-edge"):
        found = shutil.which(name)
        if found:
            return found
    raise AuthDSError("AuthDS needs Google Chrome or Microsoft Edge installed. You can also set AUTHDS_BROWSER_BINARY.", "config_error")


class _Page:
    def __init__(self, url, cancel_check=None):
        import websocket
        self.ws = websocket.create_connection(url, timeout=1, suppress_origin=True,
                                              http_no_proxy=["127.0.0.1", "localhost"])
        self.sequence = 0
        self.cancel_check = cancel_check

    def call(self, method, params=None, timeout=20):
        import websocket
        self.sequence += 1
        request_id = self.sequence
        self.ws.send(json.dumps({"id": request_id, "method": method, "params": params or {}}))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            _check(self.cancel_check)
            try:
                message = json.loads(self.ws.recv())
            except websocket.WebSocketTimeoutException:
                continue
            if message.get("id") != request_id:
                continue
            if "error" in message:
                raise AuthDSError("AuthDS browser command failed.", "browser_error")
            return message.get("result", {})
        raise AuthDSError("AuthDS browser command timed out.", "timeout")

    def evaluate(self, expression):
        result = self.call("Runtime.evaluate", {"expression": expression, "returnByValue": True})
        if result.get("exceptionDetails"):
            raise AuthDSError("AuthDS browser script failed. Reload DeepSeek and sign in again.", "browser_error")
        return result.get("result", {}).get("value")


@contextmanager
def _open_browser(*, visible=False, cancel_check=None):
    # Keep the process-spawn gate in the same function for the mobile collector.
    if not mobile_runtime.subprocesses_available():
        raise AuthDSError("AuthDS web login currently needs the desktop app and Chrome/Edge. On mobile, use deepseek/ with an API key.", "config_error")
    directory = profile_dir()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    port_file = directory / "DevToolsActivePort"
    port_file.unlink(missing_ok=True)
    args = [_browser_binary(), "--remote-debugging-port=0", "--remote-debugging-address=127.0.0.1",
            f"--user-data-dir={directory}", "--no-first-run", "--no-default-browser-check", BASE_URL]
    if not visible:
        args.insert(1, "--headless=new")
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    page = None
    try:
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline:
            _check(cancel_check)
            if proc.poll() is not None:
                raise AuthDSError("AuthDS browser closed. Close any other AuthDS login window and try again.", "browser_error")
            try:
                port = int(port_file.read_text().splitlines()[0])
                # Bypass global HTTP proxy settings for the local debugging connection.
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(f"http://127.0.0.1:{port}/json/list", timeout=1) as response:
                    tabs = json.load(response)
                target = next(t for t in tabs if t.get("type") == "page" and
                              t.get("url", "").startswith(BASE_URL + "/"))
                page = _Page(target["webSocketDebuggerUrl"], cancel_check)
                ready_deadline = time.monotonic() + 30
                while time.monotonic() < ready_deadline:
                    _check(cancel_check)
                    try:
                        ready = page.evaluate("document.readyState !== 'loading' && location.origin === 'https://chat.deepseek.com'")
                    except AuthDSError:
                        ready = False  # navigation may destroy the old JS context
                    if ready:
                        break
                    time.sleep(.2)
                else:
                    raise AuthDSError("DeepSeek page did not finish loading. Check your connection and try again.", "browser_error")
                break
            except (OSError, ValueError, StopIteration, KeyError):
                time.sleep(.2)
        if page is None:
            raise AuthDSError("AuthDS could not connect to its browser.", "browser_error")
        yield page
    finally:
        if page:
            try:
                page.ws.send(json.dumps({"id": 999999, "method": "Browser.close"}))
            except Exception:
                pass
            page.ws.close()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


@contextmanager
def _serialized(cancel_check=None):
    # Only this dedicated profile is serialized; other providers remain independent.
    while not _gate.acquire(timeout=.2):
        _check(cancel_check)
    try:
        _check(cancel_check)
        yield
    finally:
        _gate.release()


_TOKEN_PRESENT = """(() => {
 if (location.origin !== 'https://chat.deepseek.com') return false;
 const raw = localStorage.getItem('userToken'); if (!raw) return false;
 try { const t = JSON.parse(raw); return !!(typeof t === 'string' ? t : t && t.value); }
 catch (_) { return !!raw; }
})()"""


def _wait_login(page, log_fn, cancel_check=None, timeout=300):
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
    with _serialized(cancel_check), _open_browser(visible=True, cancel_check=cancel_check) as page:
        _wait_login(page, log_fn, cancel_check)
    log_fn("AuthDS: Browser session saved.")


def _clear_page_token(page):
    # An expired token remains in localStorage. Remove it before interactive
    # re-login, otherwise the presence check would close the window immediately.
    (profile_dir() / ".signed_in").unlink(missing_ok=True)
    page.evaluate("(() => {if (location.origin === 'https://chat.deepseek.com') {localStorage.removeItem('userToken'); location.reload();} return true;})()")


def resolve_model(model):
    name = re.sub(r"^authds/", "", str(model).lower()).strip()
    choices = {"flash": ("default", False), "pro": ("expert", False),
               "flash-thinking": ("default", True), "pro-thinking": ("expert", True),
               "deepseek-chat": ("default", False), "deepseek-reasoner": ("default", True)}
    if name not in choices:
        raise AuthDSError(f"Unknown AuthDS model '{name}'. Choose flash, pro, flash-thinking, or pro-thinking.", "config_error")
    return choices[name]


def build_prompt(messages):
    parts = []
    for message in messages:
        content = message.get("content", "")
        if not isinstance(content, str):
            raise AuthDSError("AuthDS currently supports text messages only.", "config_error")
        if content:
            parts.append(f"[{message.get('role', 'user')}]\n{content}")
    if not parts:
        raise AuthDSError("AuthDS cannot send an empty prompt.", "config_error")
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
  const target='/api/v0/chat/completion';
  const {challenge}=await post('/api/v0/chat/create_pow_challenge',{target_path:target});
  const workerResponse=await fetch(cfg.worker || cfg.fallbackWorker, {signal:controller.signal});
  if(!workerResponse.ok) throw Error('Proof worker unavailable; update AUTHDS_POW_WORKER_URL');
  const workerURL=URL.createObjectURL(new Blob([await workerResponse.text()],{type:'application/javascript'}));
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
   answer:answer.answer,signature:answer.signature,target_path:target};
  let binary=''; for(const byte of new TextEncoder().encode(JSON.stringify(proof))) binary+=String.fromCharCode(byte);
  const session=await post('/api/v0/chat_session/create',{});
  headers['x-ds-pow-response']=btoa(binary);
  const response=await fetch(target,{method:'POST',headers,signal:controller.signal,body:JSON.stringify({
   chat_session_id:session.chat_session.id,parent_message_id:null,model_type:cfg.model,
   prompt:cfg.prompt,ref_file_ids:[],thinking_enabled:cfg.thinking,search_enabled:false,action:null,preempt:false})});
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


def _complete(page, model, messages, timeout, cancel_check, on_delta, before_send_callback):
    model_type, thinking = resolve_model(model)
    cfg = {"model": model_type, "thinking": thinking, "prompt": build_prompt(messages),
           "worker": os.environ.get("AUTHDS_POW_WORKER_URL") or _discover_worker(page),
           "fallbackWorker": POW_WORKER_URL}
    if before_send_callback:
        before_send_callback()
    page.evaluate(_START_SCRIPT.replace("__CONFIG__", json.dumps(cfg)))
    parser = StreamParser()
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            _check(cancel_check)
            state = page.evaluate("(() => {const s=window.__glossarionDS; return s ? {chunks:s.chunks.splice(0),done:s.done,error:s.error,status:s.status} : null;})()")
            if not state:
                raise AuthDSError("AuthDS page was reloaded during translation.", "browser_error")
            if state.get("error"):
                status = state.get("status", 0)
                kind = "auth_error" if status == 401 else "access_denied" if status == 403 else "rate_limit" if status == 429 else "api_error"
                # No server bodies, tokens, or browser state are included in logs.
                raise AuthDSError(f"AuthDS: {state['error']}", kind)
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
                         cancel_check=None, on_delta=None, before_send_callback=None):
    resolve_model(model)
    build_prompt(messages)
    with _serialized(cancel_check):
        # Try the saved session once, then permit an interactive re-login once.
        for attempt in range(2):
            visible = not has_session() or attempt == 1
            with _open_browser(visible=visible, cancel_check=cancel_check) as page:
                if visible:
                    if attempt:
                        _clear_page_token(page)
                    _wait_login(page, log_fn, cancel_check)
                else:
                    # Wait for the initial document load without opening a login window.
                    deadline = time.monotonic() + 20
                    while not page.evaluate(_TOKEN_PRESENT) and time.monotonic() < deadline:
                        _check(cancel_check)
                        time.sleep(.2)
                try:
                    return _complete(page, model, messages, float(timeout), cancel_check, on_delta, before_send_callback)
                except AuthDSError as exc:
                    if exc.error_type != "auth_error" or attempt:
                        raise
                    (profile_dir() / ".signed_in").unlink(missing_ok=True)
                    log_fn("AuthDS: Session expired. Opening DeepSeek login again.")
    raise AuthDSError("AuthDS login failed.", "auth_error")


if __name__ == "__main__":
    login()
