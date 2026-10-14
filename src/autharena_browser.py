"""The Arena bridge's async browser surface, backed by Qt6 WebEngine CDP.

Qt owns the profile, views and Chromium processes. This module only sends
DevTools messages through the application's Qt helper over standard I/O;
it neither installs nor launches a separate browser or automation driver.
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import inspect
import json
import re
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

class QtBrowserFactory:
    """Import placeholder replaced with the account's bound Qt context lease."""

    def __init__(self, **kwargs):
        raise RuntimeError("Arena's Qt browser factory must be bound to an account context.")


def _write_helper(process, message):
    # All writes run synchronously on the owning event loop, so window commands
    # and CDP messages cannot interleave within a JSON line.
    if process.poll() is not None:
        raise RuntimeError("Arena Qt browser exited.")
    try:
        process.stdin.write(json.dumps(message) + "\n")
        process.stdin.flush()
    except (AttributeError, OSError, ValueError) as exc:
        raise RuntimeError("Arena Qt browser command pipe is closed.") from exc


class _QtPipeSocket:
    """Async CDP stream relayed by the helper's bundled Qt WebSocket client."""

    _CLOSE_TIMEOUT = 3

    def __init__(self, process):
        self.process = process
        self._loop = asyncio.get_running_loop()
        self._incoming = asyncio.Queue()
        self._connected = self._loop.create_future()
        self._stopped = asyncio.Event()
        self._close_lock = asyncio.Lock()
        self._closed = False
        self._terminal = False
        self._disposed = False
        self._reader = threading.Thread(target=self._read_pipe, daemon=True,
                                        name="Arena Qt relay reader")
        self._reader.start()

    @classmethod
    async def connect(cls, process, endpoint):
        if process.stdout is None:
            raise RuntimeError("Arena Qt browser response pipe is unavailable.")
        self = cls(process)
        try:
            _write_helper(process, {"action": "connect", "endpoint": endpoint})
            await asyncio.wait_for(self._connected, 15)
            return self
        except BaseException:
            if not self._connected.done():
                self._connected.cancel()
            elif not self._connected.cancelled():
                self._connected.exception()
            await self.close()
            raise

    def _dispatch(self, callback, *args):
        # Shutdown must not leave a reader trying to use an already closed loop.
        with contextlib.suppress(RuntimeError):
            self._loop.call_soon_threadsafe(callback, *args)

    def _read_pipe(self):
        try:
            for line in self.process.stdout:
                try:
                    event = json.loads(line)
                    if not isinstance(event, dict):
                        raise ValueError("Expected a Qt relay event")
                    if event.get("event") not in ("connected", "message", "error", "closed"):
                        raise ValueError("Unknown Qt relay event")
                    if event["event"] == "message" and not isinstance(event.get("data"), str):
                        raise ValueError("Expected a CDP message string")
                except (TypeError, ValueError) as exc:
                    self._dispatch(self._finish, RuntimeError(f"Arena Qt browser sent an invalid relay event: {exc}"))
                    break
                self._dispatch(self._receive, event)
                if event.get("event") in ("error", "closed"):
                    break
        except (OSError, ValueError) as exc:
            self._dispatch(self._finish, RuntimeError(f"Arena Qt browser response pipe failed: {exc}"))
        finally:
            self._dispatch(self._finish)
            self._dispatch(self._stopped.set)

    def _finish(self, error=None):
        if self._terminal:
            return
        self._closed = self._terminal = True
        if not self._connected.done():
            self._connected.set_exception(error or RuntimeError("Arena Qt browser disconnected before connecting."))
        self._incoming.put_nowait(error)

    def _receive(self, event):
        if self._terminal:
            return
        kind = event.get("event")
        if kind == "connected":
            if not self._connected.done():
                self._connected.set_result(None)
        elif kind == "message":
            self._incoming.put_nowait(event["data"])
        elif kind == "error":
            self._finish(RuntimeError("Arena Qt browser relay failed: " + str(event.get("message", "unknown error"))))
        elif kind == "closed":
            self._finish()

    async def send(self, message):
        if self._closed:
            raise RuntimeError("Arena Qt browser connection is closed.")
        _write_helper(self.process, {"action": "cdp", "message": message})

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._terminal and self._incoming.empty():
            raise StopAsyncIteration
        message = await self._incoming.get()
        if message is None:
            raise StopAsyncIteration
        if isinstance(message, Exception):
            raise message
        return message

    async def close(self):
        async with self._close_lock:
            if self._disposed:
                return
            self._closed = True
            if not self._stopped.is_set():
                if not self._terminal:
                    with contextlib.suppress(RuntimeError):
                        _write_helper(self.process, {"action": "disconnect"})
                try:
                    await asyncio.wait_for(self._stopped.wait(), self._CLOSE_TIMEOUT)
                except TimeoutError:
                    # This helper belongs only to this context. If Qt no longer
                    # responds, ending it releases the blocked stdout reader.
                    if self.process.poll() is None:
                        self.process.kill()
                    await asyncio.to_thread(self.process.wait, timeout=self._CLOSE_TIMEOUT)
                    await asyncio.wait_for(self._stopped.wait(), self._CLOSE_TIMEOUT)
            self._finish()
            self._reader.join(timeout=self._CLOSE_TIMEOUT)
            self.process.stdout.close()
            self._disposed = True


class _CDP:
    def __init__(self, socket, event_handler, disconnected):
        self.socket = socket
        self._event_handler = event_handler
        self._disconnected = disconnected
        self._next_id = 0
        self._pending = {}
        self.closed = False
        self._reader = asyncio.create_task(self._read())

    async def send(self, method, params=None, session=None, timeout=30):
        if self.closed:
            raise RuntimeError("Arena Qt browser connection is closed.")
        self._next_id += 1
        call_id = self._next_id
        future = asyncio.get_running_loop().create_future()
        self._pending[call_id] = (future, session)
        message = {"id": call_id, "method": method, "params": params or {}}
        if session:
            message["sessionId"] = session
        try:
            await self.socket.send(json.dumps(message))
            return await asyncio.wait_for(future, timeout)
        finally:
            self._pending.pop(call_id, None)
            if not future.done():
                future.cancel()

    def fail_session(self, session):
        for future, pending_session in list(self._pending.values()):
            if session == pending_session and not future.done():
                future.set_exception(RuntimeError("Arena Qt page is closed."))

    async def _read(self):
        try:
            async for raw in self.socket:
                message = json.loads(raw)
                if "id" in message:
                    pending = self._pending.get(message["id"])
                    if pending and not pending[0].done():
                        future = pending[0]
                        if "error" in message:
                            future.set_exception(RuntimeError(message["error"].get("message", "Qt DevTools command failed")))
                        else:
                            future.set_result(message.get("result", {}))
                else:
                    self._event_handler(message.get("method"), message.get("params", {}), message.get("sessionId"))
        except asyncio.CancelledError:
            raise
        except Exception:
            # Pending callers receive a stable failure even if Qt disappears
            # while a response is streaming or a window is being closed.
            pass
        finally:
            self.closed = True
            for future, _ in list(self._pending.values()):
                if not future.done():
                    future.set_exception(RuntimeError("Arena Qt browser disconnected."))
            self._pending.clear()
            self._disconnected()

    async def close(self):
        await self.socket.close()
        if self._reader is not asyncio.current_task():
            self._reader.cancel()
            await asyncio.gather(self._reader, return_exceptions=True)


class QtArenaContext:
    """One off-the-record Qt profile and the pages belonging to one account."""

    def __init__(self, process):
        self.process = process
        self.context = self
        self._closed = False
        self._cdp = None
        self._pages = {}
        self._sessions = {}
        self._targets = {}
        self._closing_targets = set()
        self._attaching = {}
        self._tasks = set()
        self._scripts = []
        self._routes = []
        self._keeper = None
        self._new_lock = asyncio.Lock()

    @classmethod
    async def connect(cls, process, endpoint):
        self = cls(process)
        socket = await _QtPipeSocket.connect(process, endpoint)
        self._cdp = _CDP(socket, self._event, self._disconnected)
        try:
            result = await self._cdp.send("Target.getTargets")
            initial = [info for info in result["targetInfos"] if info["type"] == "page"]
            if not initial:
                raise RuntimeError("Arena Qt browser has no keeper page.")
            self._keeper = initial[0]["targetId"]
            for info in initial:
                self._targets[info["targetId"]] = info
                await self._attach(info)
            await self._cdp.send("Target.setDiscoverTargets", {"discover": True})
            return self
        except BaseException:
            await self._cdp.close()
            for task in list(self._tasks):
                task.cancel()
            await asyncio.gather(*list(self._tasks), return_exceptions=True)
            raise

    def _spawn(self, awaitable, tasks=None):
        task = asyncio.create_task(awaitable)
        collection = self._tasks if tasks is None else tasks
        collection.add(task)
        self._tasks.add(task)

        def finished(done):
            collection.discard(done)
            self._tasks.discard(done)
            if not done.cancelled():
                error = done.exception()
                if error and not self._closed:
                    asyncio.get_running_loop().call_exception_handler({
                        "message": "Arena Qt browser callback failed", "exception": error, "task": done})

        task.add_done_callback(finished)
        return task

    def _disconnected(self):
        for page in list(self._pages.values()):
            page._mark_closed()

    def _event(self, method, params, session):
        if method in ("Target.targetCreated", "Target.targetInfoChanged"):
            info = params["targetInfo"]
            if info["type"] != "page":
                return
            target = info["targetId"]
            if target in self._closing_targets:
                return
            self._targets[target] = info
            if target in self._pages:
                self._pages[target]._url = info.get("url", "about:blank")
            elif target not in self._attaching and not self._closed:
                self._attaching[target] = self._spawn(self._attach(info))
        elif method in ("Target.targetDestroyed", "Target.targetCrashed"):
            target = params["targetId"]
            self._targets.pop(target, None)
            page = self._pages.get(target)
            if page:
                page._mark_closed()
            self._closing_targets.discard(target)
        elif method == "Target.detachedFromTarget":
            page = self._sessions.get(params.get("sessionId"))
            if page:
                page._mark_closed()
        elif session in self._sessions:
            self._sessions[session]._event(method, params)

    async def _attach(self, info):
        target = info["targetId"]
        if target in self._pages:
            return self._pages[target]
        result = await self._cdp.send("Target.attachToTarget", {"targetId": target, "flatten": True})
        page = QtArenaPage(self, target, result["sessionId"], info.get("url", "about:blank"))
        self._pages[target] = page
        self._sessions[page._session] = page
        try:
            await page._initialize()
            page._ready.set()
            return page
        except BaseException:
            page._mark_closed()
            raise
        finally:
            self._attaching.pop(target, None)

    def is_connected(self):
        return not self._closed and self.process.poll() is None and self._cdp is not None and not self._cdp.closed

    @property
    def pages(self):
        return [page for target, page in self._pages.items()
                if target != self._keeper and not page.is_closed() and page._ready.is_set()]

    def command(self, action, target=None):
        _write_helper(self.process, {"action": action, "target": target})

    async def new_page(self):
        async with self._new_lock:
            before = set(self._targets)
            self.command("new")
            try:
                async with asyncio.timeout(15):
                    while True:
                        if not self.is_connected():
                            raise RuntimeError("Arena Qt browser disconnected while opening a page.")
                        for target in set(self._targets) - before:
                            pending = self._attaching.get(target)
                            if pending:
                                await asyncio.shield(pending)
                            page = self._pages.get(target)
                            if page and not page.is_closed():
                                await page._ready.wait()
                                return page
                        await asyncio.sleep(.025)
            except BaseException:
                # Qt may already have created a view whose CDP attachment has
                # not completed. Retire this request's profile so cancellation
                # cannot leave an unclaimed page in the reusable account pool.
                await self.close()
                raise

    async def add_init_script(self, script=None, path=None):
        script = Path(path).read_text(encoding="utf-8") if path is not None else script
        if not isinstance(script, str):
            raise TypeError("An initialization script or path is required.")
        self._scripts.append(script)
        for page in list(self._pages.values()):
            if not page.is_closed():
                await page._send("Page.addScriptToEvaluateOnNewDocument", {"source": script})

    async def add_cookies(self, cookies):
        await self._pages[self._keeper]._send("Network.setCookies", {"cookies": cookies})

    async def cookies(self, urls=None):
        if isinstance(urls, str):
            urls = [urls]
        result = await self._pages[self._keeper]._send(
            "Network.getCookies" if urls else "Network.getAllCookies", {"urls": urls} if urls else {})
        return result["cookies"]

    async def route(self, pattern, handler):
        self._routes.append((pattern, handler))
        for page in list(self._pages.values()):
            if not page.is_closed():
                await page._enable_routes()

    async def close(self):
        if self._closed:
            return
        self._closed = True
        for page in list(self._pages.values()):
            page._mark_closed()
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*list(self._tasks), return_exceptions=True)
        # Keep the response pipe open until Qt has finished emitting its
        # shutdown signals. Closing stdout first breaks windowed exe shutdown.
        if self.process.poll() is None:
            with contextlib.suppress(Exception):
                self.command("quit")
            try:
                await asyncio.to_thread(self.process.wait, timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                await asyncio.to_thread(self.process.wait)
        if self._cdp:
            await self._cdp.close()
        if self.process.stdin:
            with contextlib.suppress(OSError, ValueError):
                self.process.stdin.close()


def _expression(script, arg=None):
    # Evaluate both expressions and function-valued expressions as Playwright
    # does. JSON values are embedded as data, never as executable source.
    return "(() => {const value = (" + script.strip().rstrip(";") + "); return typeof value === 'function' ? value(" + json.dumps(arg) + ") : value;})()"


class _DOM:
    def __init__(self, page, context_id=None, frame_id=None):
        self._page = page
        self._context_id = context_id
        self._frame_id = frame_id

    async def _evaluate(self, expression, return_by_value=True):
        params = {"expression": expression, "awaitPromise": True,
                  "returnByValue": return_by_value, "userGesture": True}
        if self._context_id is not None:
            params["contextId"] = self._context_id
        result = await self._page._send("Runtime.evaluate", params, timeout=None)
        if "exceptionDetails" in result:
            details = result["exceptionDetails"]
            raise RuntimeError(details.get("exception", {}).get("description", details.get("text", "Arena page JavaScript failed")))
        remote = result.get("result", {})
        return remote.get("value") if return_by_value else remote

    async def evaluate(self, expression, arg=None):
        return await self._evaluate(_expression(expression, arg))

    def locator(self, selector):
        return _Locator(self, "Array.from(document.querySelectorAll(" + json.dumps(selector) + "))")

    def get_by_role(self, role, name=None, exact=False):
        selectors = {
            "button": 'button,input[type="button"],input[type="submit"],[role="button"]',
            "link": 'a[href],[role="link"]',
            "heading": 'h1,h2,h3,h4,h5,h6,[role="heading"]',
            "checkbox": 'input[type="checkbox"],[role="checkbox"]',
        }
        selector = selectors.get(role, '[role=' + json.dumps(role) + ']')
        selection = "Array.from(document.querySelectorAll(" + json.dumps(selector) + "))"
        if name is not None:
            if isinstance(name, re.Pattern):
                flags = "i" if name.flags & re.I else ""
                test = "new RegExp(" + json.dumps(name.pattern) + "," + json.dumps(flags) + ").test(label)"
            elif exact:
                test = "label === " + json.dumps(str(name))
            else:
                test = "label.toLowerCase().includes(" + json.dumps(str(name).lower()) + ")"
            selection += ".filter(el => {const ids = (el.getAttribute('aria-labelledby') || '').split(/\\s+/); const label = (el.getAttribute('aria-label') || ids.map(id=>document.getElementById(id)?.textContent || '').join(' ').trim() || el.innerText || el.value || el.textContent || '').trim(); return " + test + ";})"
        return _Locator(self, selection)

    async def query_selector(self, selector):
        locator = self.locator(selector)
        return locator.nth(0) if await locator.count() else None

    async def query_selector_all(self, selector):
        locator = self.locator(selector)
        return [locator.nth(index) for index in range(await locator.count())]

    async def wait_for_function(self, expression, arg=None, timeout=30000, polling=None):
        deadline = time.monotonic() + timeout / 1000 if timeout else None
        while True:
            remaining = max(0, deadline - time.monotonic()) if deadline else None
            if remaining == 0:
                raise TimeoutError("Timed out waiting for Arena page JavaScript.")
            try:
                value = await asyncio.wait_for(self.evaluate(expression, arg), remaining)
                if value:
                    return value
            except RuntimeError as exc:
                if self._page.is_closed() or not any(text in str(exc).lower() for text in (
                        "context was destroyed", "cannot find context", "inspected target navigated")):
                    raise
            await asyncio.sleep((polling / 1000) if isinstance(polling, (int, float)) else .05)


class QtArenaPage(_DOM):
    def __init__(self, context, target, session, url):
        super().__init__(self)
        self.context = context
        self.target = target
        self._session = session
        self._url = url
        self._closed = False
        self._ready = asyncio.Event()
        self._tasks = set()
        self._listeners = {}
        self._bindings = {}
        self._contexts = {}
        self._frames = {}
        self._main_frame = None
        self._lifecycle = set()
        self._routes = []
        self._routes_enabled = False
        self._fetching = {}
        self._responses = {}
        self._extra_headers = {}
        self.mouse = _Mouse(self)

    async def _send(self, method, params=None, timeout=30):
        if self.is_closed():
            raise RuntimeError("Arena Qt page is closed.")
        return await self.context._cdp.send(method, params, self._session, timeout)

    async def _initialize(self):
        await self._send("Page.enable")
        await self._send("Runtime.enable")
        await self._send("Network.enable")
        await self._send("Page.setLifecycleEventsEnabled", {"enabled": True})
        tree = await self._send("Page.getFrameTree")
        self._main_frame = tree["frameTree"]["frame"]["id"]
        for script in self.context._scripts:
            await self._send("Page.addScriptToEvaluateOnNewDocument", {"source": script})
        if self.context._routes:
            await self._enable_routes()

    @property
    def url(self):
        return self._url

    def is_closed(self):
        return self._closed or self.context._closed or self.context._cdp.closed

    def on(self, event, callback):
        self._listeners.setdefault(event, []).append(callback)

    def remove_listener(self, event, callback):
        callbacks = self._listeners.get(event, [])
        if callback in callbacks:
            callbacks.remove(callback)

    def _emit(self, event, *args):
        for callback in list(self._listeners.get(event, ())):
            try:
                result = callback(*args)
                if inspect.isawaitable(result):
                    self.context._spawn(result, self._tasks)
            except Exception as exc:
                asyncio.get_running_loop().call_exception_handler({
                    "message": "Arena Qt page event callback failed", "exception": exc})

    def _mark_closed(self):
        if self._closed:
            return
        self._closed = True
        self.context._cdp.fail_session(self._session)
        self.context._pages.pop(self.target, None)
        self.context._sessions.pop(self._session, None)
        self.context._closing_targets.add(self.target)
        self._ready.set()
        for task in list(self._tasks):
            if task is not asyncio.current_task():
                task.cancel()
        for future in self._fetching.values():
            if not future.done():
                future.set_exception(RuntimeError("Arena Qt page closed during an intercepted request."))
        for response in self._responses.values():
            response._finished.set()
        self._emit("close", self)

    async def _drain_tasks(self):
        tasks = [task for task in self._tasks if task is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def close(self):
        if not self._closed:
            self.context.command("close", self.target)
            self._mark_closed()
        await self._drain_tasks()

    async def bring_to_front(self):
        self.context.command("show", self.target)

    def _event(self, method, params):
        if method == "Runtime.executionContextCreated":
            ctx = params["context"]
            if ctx.get("auxData", {}).get("isDefault"):
                self._contexts[ctx["id"]] = ctx.get("auxData", {}).get("frameId")
        elif method == "Runtime.executionContextDestroyed":
            self._contexts.pop(params["executionContextId"], None)
        elif method == "Runtime.executionContextsCleared":
            self._contexts.clear()
        elif method == "Page.frameNavigated":
            frame = params["frame"]
            self._frames[frame["id"]] = frame
            if not frame.get("parentId"):
                self._main_frame = frame["id"]
                self._url = frame.get("url", self._url)
        elif method == "Page.navigatedWithinDocument" and params["frameId"] == self._main_frame:
            self._url = params["url"]
        elif method == "Page.lifecycleEvent":
            self._lifecycle.add((params.get("loaderId"), params["name"]))
        elif method == "Runtime.bindingCalled" and params["name"] in self._bindings:
            self.context._spawn(self._binding_called(params), self._tasks)
        elif method == "Runtime.consoleAPICalled":
            message = " ".join(str(arg.get("value", arg.get("description", ""))) for arg in params.get("args", ()))
            self._emit("console", SimpleNamespace(type=params["type"], text=message))
        elif method == "Network.responseReceived":
            response = _Response(self, params["requestId"], params["response"])
            self._responses[params["requestId"]] = response
            self._emit("response", response)
        elif method in ("Network.loadingFinished", "Network.loadingFailed"):
            response = self._responses.get(params["requestId"])
            if response:
                response._error = params.get("errorText")
                response._finished.set()
        elif method == "Network.requestWillBeSentExtraInfo":
            self._extra_headers[params["requestId"]] = params["headers"]
        elif method == "Fetch.requestPaused":
            if "responseStatusCode" in params or "responseErrorReason" in params:
                future = self._fetching.get(params["requestId"])
                if future and not future.done():
                    future.set_result(params)
                else:
                    self.context._spawn(self._send("Fetch.continueRequest", {"requestId": params["requestId"]}), self._tasks)
            else:
                self.context._spawn(self._handle_route(params), self._tasks)

    async def goto(self, url, wait_until="load", timeout=30000):
        self._lifecycle.clear()
        async with asyncio.timeout(timeout / 1000 if timeout else None):
            result = await self._send("Page.navigate", {"url": url})
            if result.get("errorText"):
                raise RuntimeError("Arena Qt navigation failed: " + result["errorText"])
            self._url = url
            loader = result.get("loaderId")
            if wait_until != "commit" and loader:
                state = {"domcontentloaded": "DOMContentLoaded", "load": "load", "networkidle": "networkIdle"}.get(wait_until)
                if not state:
                    raise ValueError("Unsupported navigation state: " + wait_until)
                while (loader, state) not in self._lifecycle:
                    if self.is_closed():
                        raise RuntimeError("Arena Qt page closed during navigation.")
                    await asyncio.sleep(.025)
        return next((response for response in reversed(list(self._responses.values())) if response.url == self._url), None)

    async def wait_for_load_state(self, state="load", timeout=30000):
        expression = "document.readyState === 'complete'" if state == "load" else "document.readyState !== 'loading'"
        await self.wait_for_function(expression, timeout=timeout)

    async def title(self):
        return await self.evaluate("document.title")

    async def content(self):
        return await self.evaluate("(document.doctype ? new XMLSerializer().serializeToString(document.doctype) : '') + document.documentElement.outerHTML")

    async def set_content(self, html, wait_until="load", timeout=30000):
        await self.evaluate("html => {document.open(); document.write(html); document.close();}", html)
        await self.wait_for_load_state(wait_until, timeout=timeout)

    async def expose_binding(self, name, callback):
        if name in self._bindings:
            raise RuntimeError("Arena page binding already exists: " + name)
        self._bindings[name] = callback
        await self._send("Runtime.addBinding", {"name": name})
        script = """(() => {
            const name = NAME;
            const native = window[name];
            if (!native || native.__arenaBinding) return;
            let sequence = 0;
            const pending = new Map();
            const binding = (...args) => new Promise((resolve, reject) => {
                const id = ++sequence;
                pending.set(id, {resolve, reject});
                try { native(JSON.stringify({id, args})); }
                catch (error) {pending.delete(id); reject(error);}
            });
            binding.__arenaBinding = true;
            binding.__settle = (id, ok, value) => {
                const item = pending.get(id); if (!item) return;
                pending.delete(id);
                if (ok) item.resolve(value); else item.reject(new Error(value));
            };
            window[name] = binding;
        })();""".replace("NAME", json.dumps(name))
        await self._send("Page.addScriptToEvaluateOnNewDocument", {"source": script})
        for context_id in list(self._contexts):
            await _DOM(self, context_id)._evaluate(script)

    async def _binding_called(self, params):
        payload = json.loads(params["payload"])
        context_id = params["executionContextId"]
        frame = _DOM(self, context_id, self._contexts.get(context_id))
        source = {"page": self, "context": self.context, "frame": frame}
        try:
            value = self._bindings[params["name"]](source, *payload.get("args", ()))
            if inspect.isawaitable(value):
                value = await value
            encoded = json.dumps(value)
            ok = True
        except Exception as exc:
            encoded = json.dumps(str(exc))
            ok = False
        expression = "window[" + json.dumps(params["name"]) + "]?.__settle(" + json.dumps(payload["id"]) + "," + str(ok).lower() + "," + encoded + ")"
        try:
            await frame._evaluate(expression)
        except RuntimeError:
            # Navigating/closing invalidates promises in the old document.
            if context_id in self._contexts and not self.is_closed():
                raise

    async def route(self, pattern, handler):
        self._routes.append((pattern, handler))
        await self._enable_routes()

    async def _enable_routes(self):
        if not self._routes_enabled:
            await self._send("Fetch.enable", {"patterns": [{"urlPattern": "*", "requestStage": "Request"}]})
            self._routes_enabled = True

    async def _handle_route(self, params):
        handlers = [handler for pattern, handler in reversed(self._routes) if _matches(pattern, params["request"]["url"])]
        handlers += [handler for pattern, handler in reversed(self.context._routes) if _matches(pattern, params["request"]["url"])]
        route = _Route(self, params, handlers)
        try:
            await route.fallback()
        except Exception:
            if not route._done and not self.is_closed():
                # Fail intercepted requests when user handlers fail. Letting
                # them escape would submit an unapproved browser request.
                with contextlib.suppress(Exception):
                    await self._send("Fetch.failRequest", {"requestId": route._id, "errorReason": "Aborted"})
            raise


def _matches(pattern, url):
    if isinstance(pattern, re.Pattern):
        return bool(pattern.search(url))
    # The bridge uses **/* and **/nextjs-api/stream/**. Preserve the distinction
    # between * (one path component) and ** (including slashes).
    pattern = re.escape(pattern).replace(r"\*\*", "\0").replace(r"\*", "[^/]*").replace("\0", ".*")
    return re.fullmatch(pattern, url) is not None


class _Request:
    def __init__(self, page, params):
        self._page = page
        self._data = params["request"]
        self._network_id = params.get("networkId")
        self.url = self._data["url"]
        self.method = self._data["method"]

    @property
    def post_data_json(self):
        data = self._data.get("postData")
        return json.loads(data) if data else None

    async def all_headers(self):
        data = dict(self._data.get("headers", {}))
        data.update(self._page._extra_headers.get(self._network_id, {}))
        headers = {name.lower(): value for name, value in data.items()}
        if "cookie" not in headers:
            cookies = await self._page.context.cookies(self.url)
            if cookies:
                headers["cookie"] = "; ".join(cookie["name"] + "=" + cookie["value"] for cookie in cookies)
        return headers


class _Response:
    def __init__(self, page, request_id, data, body=None):
        self._page, self._id = page, request_id
        self.url = data.get("url", "")
        self.status = data["status"]
        self.headers = {name.lower(): value for name, value in data.get("headers", {}).items()}
        self._body = body
        self._finished = asyncio.Event()
        self._error = None
        if body is not None:
            self._finished.set()

    async def body(self):
        if self._body is None:
            await self._finished.wait()
            if self._error:
                raise RuntimeError(self._error)
            result = await self._page._send("Network.getResponseBody", {"requestId": self._id})
            self._body = base64.b64decode(result["body"]) if result.get("base64Encoded") else result["body"].encode()
        return self._body

    async def text(self):
        return (await self.body()).decode("utf-8", errors="replace")


class _Route:
    def __init__(self, page, params, handlers):
        self._page, self._id = page, params["requestId"]
        self.request = _Request(page, params)
        self._handlers = iter(handlers)
        self._done = False
        self._fetched = None

    async def fallback(self):
        handler = next(self._handlers, None)
        if handler is None:
            await self.continue_()
        else:
            result = handler(self)
            if inspect.isawaitable(result):
                await result

    async def continue_(self):
        if self._done:
            return
        await self._page._send("Fetch.continueRequest", {"requestId": self._id})
        self._done = True

    async def fetch(self, timeout=30000):
        if self._fetched:
            return self._fetched
        future = asyncio.get_running_loop().create_future()
        self._page._fetching[self._id] = future
        try:
            await self._page._send("Fetch.continueRequest", {"requestId": self._id, "interceptResponse": True})
            response = await asyncio.wait_for(future, timeout / 1000 if timeout else None)
            if response.get("responseErrorReason"):
                raise RuntimeError(response["responseErrorReason"])
            result = await self._page._send("Fetch.getResponseBody", {"requestId": self._id})
            body = base64.b64decode(result["body"]) if result.get("base64Encoded") else result["body"].encode()
            self._fetched = _Response(self._page, self._id, {
                "url": self.request.url, "status": response["responseStatusCode"],
                "headers": {item["name"]: item["value"] for item in response.get("responseHeaders", [])}}, body)
            return self._fetched
        finally:
            self._page._fetching.pop(self._id, None)
            if not future.done():
                future.cancel()

    async def fulfill(self, status=None, headers=None, content_type=None, body=None, response=None):
        if self._done:
            return
        headers = dict(response.headers if response and headers is None else headers or {})
        if body is None:
            body = await response.body() if response else b""
        body = body.encode() if isinstance(body, str) else body
        headers = {name: str(value) for name, value in headers.items()
                   if name.lower() not in ("content-length", "content-encoding", "transfer-encoding")}
        if content_type:
            headers["Content-Type"] = content_type
        await self._page._send("Fetch.fulfillRequest", {
            "requestId": self._id, "responseCode": status if status is not None else (response.status if response else 200),
            "responseHeaders": [{"name": name, "value": value} for name, value in headers.items()],
            "body": base64.b64encode(body).decode("ascii")})
        self._done = True


class _Locator:
    def __init__(self, dom, selection, index=None):
        self._dom, self._selection, self._index = dom, selection, index

    def nth(self, index):
        return _Locator(self._dom, self._selection, index)

    async def count(self):
        return await self._dom._evaluate("(" + self._selection + ").length")

    @property
    def _element(self):
        return "(" + self._selection + ")[" + str(self._index or 0) + "]"

    async def is_visible(self):
        return bool(await self._dom._evaluate("(() => {const el = " + self._element + "; if (!el) return false; const r = el.getBoundingClientRect(); const s = getComputedStyle(el); return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none';})()"))

    async def _object(self):
        remote = await self._dom._evaluate(self._element, return_by_value=False)
        if "objectId" not in remote:
            raise RuntimeError("Arena page element is detached or missing.")
        return remote["objectId"]

    async def bounding_box(self):
        object_id = await self._object()
        try:
            result = await self._dom._page._send("DOM.getBoxModel", {"objectId": object_id})
            quad = result["model"]["border"]
            xs, ys = quad[0::2], quad[1::2]
            return {"x": min(xs), "y": min(ys), "width": max(xs) - min(xs), "height": max(ys) - min(ys)}
        except RuntimeError as exc:
            if "box model" in str(exc).lower():
                return None
            raise
        finally:
            with contextlib.suppress(RuntimeError):
                await self._dom._page._send("Runtime.releaseObject", {"objectId": object_id})

    async def click(self, timeout=30000, force=False):
        async with asyncio.timeout(timeout / 1000 if timeout else None):
            while True:
                if force or await self.is_visible():
                    await self._dom._evaluate("(() => {const el = " + self._element + "; if (el) el.scrollIntoView({block:'center',inline:'center'});})()")
                    box = await self.bounding_box()
                    if box and box["width"] > 0 and box["height"] > 0:
                        await self._dom._page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
                        return
                await asyncio.sleep(.05)

    async def content_frame(self):
        object_id = await self._object()
        try:
            result = await self._dom._page._send("DOM.describeNode", {"objectId": object_id})
            frame_id = result["node"].get("frameId")
            if frame_id:
                for context_id, candidate in self._dom._page._contexts.items():
                    if candidate == frame_id:
                        return _DOM(self._dom._page, context_id, frame_id)
            return None
        finally:
            with contextlib.suppress(RuntimeError):
                await self._dom._page._send("Runtime.releaseObject", {"objectId": object_id})


class _Mouse:
    def __init__(self, page):
        self._page = page
        self._x = self._y = 0

    async def move(self, x, y, steps=1):
        start_x, start_y = self._x, self._y
        for step in range(1, max(1, steps) + 1):
            self._x = start_x + (x - start_x) * step / max(1, steps)
            self._y = start_y + (y - start_y) * step / max(1, steps)
            await self._page._send("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": self._x, "y": self._y})

    async def wheel(self, delta_x, delta_y):
        await self._page._send("Input.dispatchMouseEvent", {"type": "mouseWheel", "x": self._x, "y": self._y,
                                                         "deltaX": delta_x, "deltaY": delta_y})

    async def click(self, x, y, button="left", click_count=1, delay=0):
        # Hidden Qt views can expose a subframe's DOM before Chromium publishes
        # its hit-test surface. A one-pixel compositor readback flushes that
        # surface before trusted input; no image is retained or written to disk.
        await self._page._send("Page.captureScreenshot", {
            "format": "png", "clip": {"x": 0, "y": 0, "width": 1, "height": 1, "scale": 1}})
        await self.move(x, y)
        params = {"x": x, "y": y, "button": button, "clickCount": click_count}
        await self._page._send("Input.dispatchMouseEvent", dict(params, type="mousePressed"))
        if delay:
            await asyncio.sleep(delay / 1000)
        await self._page._send("Input.dispatchMouseEvent", dict(params, type="mouseReleased"))
