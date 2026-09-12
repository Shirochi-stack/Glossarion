"""Run with the managed runtime's Python to exercise the pinned bridge offline.

Uses a simulated app-owned browser. Never reads browser sessions or contacts Arena.
"""
import asyncio
import ast
import base64
import contextlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import autharena_proxy as arena

RUNTIME = Path.home() / ".glossarion/autharena_proxy" / ("bridge-" + arena.REVISION)


class Page:
    def __init__(self, context):
        self.context = context
        self.token = f"upstream-test-token-{context.browser.page_sequence}"
        context.browser.page_sequence += 1

    async def expose_binding(self, name, callback):
        self.emit = callback

    async def goto(self, *args, **kwargs):
        self.context.browser.urls.append(args[0])
        if self.context.browser.refreshed_cookie:
            self.context.saved_cookies = [self.context.browser.refreshed_cookie]

    async def bring_to_front(self):
        pass

    async def title(self):
        if self.context.browser.challenge_pages:
            self.context.browser.challenge_pages -= 1
            return "Just a moment..."
        return "Arena"

    async def wait_for_function(self, *args, **kwargs):
        assert len(self.context.browser.sent) == self.context.browser.before_challenge
        self.context.browser.verifications += 1

    def get_by_role(self, role, name):
        page = self
        class Control:
            def nth(self, index):
                return self
            @property
            def first(self):
                return self
            async def count(self):
                return 1
            async def is_visible(self):
                return True
            async def click(self, **kwargs):
                page.context.browser.login_clicks += 1
                session = {"user": page.context.browser.login_user, "expires_at": int(time.time()) + 3600}
                page.context.saved_cookies = [{"name": "arena-auth-prod-v1", "domain": ".arena.ai", "path": "/",
                                               "value": "base64-" + base64.urlsafe_b64encode(json.dumps(session).encode()).decode()}]
        return Control()

    async def close(self):
        if self in self.context.pages:
            self.context.pages.remove(self)

    async def evaluate(self, script, args):
        if "LM_BRIDGE_MINT_RECAPTCHA_V3" in script:
            assert args["action"] == "chat_submit"
            if self.context.browser.mint_gate is not None:
                self.context.browser.mint_started.set()
                await self.context.browser.mint_gate.wait()
            return self.token
        assert args["payload"]["mode"] == "direct-battle"
        assert args["payload"]["modelAId"] == "test-model-id"
        self.context.browser.sent.append((self.context.saved_cookies, args["payload"]))
        if self.context.browser.response_handler is not None:
            await self.context.browser.response_handler(self, args["payload"])
            return
        if self.context.browser.reject_once:
            self.context.browser.reject_once = False
            await self.emit(None, {"status": 403, "headers": {}, "error_body": self.context.browser.rejection_body})
            return
        if self.context.browser.fail_request:
            await self.emit(None, {"status": 400, "headers": {}, "error_body": '{"message":"invalid test payload"}'})
            return
        await self.emit(None, {"status": 200, "headers": {}})
        await self.emit(None, {"line": 'ag:"reasoning"'})
        await asyncio.sleep(self.context.browser.delay_after_reasoning)
        await self.emit(None, {"line": 'a0:"hello"'})
        await asyncio.sleep(.01)
        await self.emit(None, {"line": 'ad:{"finishReason":"stop","usage":{"total_tokens":8}}'})


class Context:
    def is_connected(self):
        return self in self.browser.contexts

    def __init__(self, browser):
        self.browser = browser
        self.pages = []
        self.saved_cookies = []

    async def add_cookies(self, cookies):
        self.saved_cookies = cookies

    async def new_page(self):
        page = Page(self)
        self.pages.append(page)
        return page

    async def cookies(self, urls):
        return list(self.saved_cookies)

    async def close(self):
        self.browser.contexts.remove(self)


class Browser:
    version = "152.0.0.0"
    def __init__(self):
        self.page_sequence = 0
        self.contexts = [Context(self)]
        self.sent = []
        self.urls = []
        self.login_clicks = 0
        self.fail_request = False
        self.reject_once = False
        self.rejection_body = "session needs refresh"
        self.refreshed_cookie = None
        self.challenge_pages = 0
        self.before_challenge = 0
        self.verifications = 0
        self.mint_gate = None
        self.mint_started = asyncio.Event()
        self.delay_after_reasoning = 0
        self.response_handler = None
        self.login_user = {"id": "new-user", "email": "new@example.test"}

    async def close(self):
        self.contexts.clear()

    def is_connected(self):
        return True

    async def new_context(self):
        context = Context(self)
        self.contexts.append(context)
        return context


@unittest.skipUnless(importlib.util.find_spec("playwright") and importlib.util.find_spec("PySide6") and (RUNTIME / "bridge").exists(),
                     "Run with installed Arena managed Python for local browser tests")
class LoginNavigationTest(unittest.TestCase):
    def test_upstream_captcha_loader_and_token_generation(self):
        async def run():
            sys.path.insert(0, str(RUNTIME))
            from bridge.src.recaptcha import _mint_recaptcha_v3_token_in_page
            from playwright.async_api import async_playwright
            os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(RUNTIME.parent / "browsers")
            async with async_playwright() as p:
                browser = await arena._open_qt_browser(p)
                try:
                    page = await browser.new_page()
                    await page.route("**/recaptcha/**", lambda route: route.fulfill(
                        content_type="application/javascript",
                        body="window.grecaptcha={enterprise:{ready:fn=>fn(),execute:async(key,opts)=>{if(key!=='test-key'||opts.action!=='chat_submit') throw Error('wrong configuration');return 'upstream-token';}}};"))
                    await page.goto("about:blank")
                    token = await _mint_recaptcha_v3_token_in_page(page, sitekey="test-key", action="chat_submit")
                    self.assertEqual(token, "upstream-token")
                    await page.evaluate("grecaptcha.enterprise.execute=async()=>''")
                    self.assertEqual(await _mint_recaptcha_v3_token_in_page(page, sitekey="test-key", action="chat_submit"), "")
                finally:
                    await browser.close()
        asyncio.run(run())

    def test_qt_startup_failure_retries_before_opening_page(self):
        async def run():
            import io
            from playwright.async_api import async_playwright
            real_popen = arena.subprocess.Popen
            attempts = []
            class FailedProcess:
                returncode = 1
                stdin = io.StringIO()
                def poll(self): return 1
                def wait(self, **kwargs): return 1
            def popen(*args, **kwargs):
                attempts.append(kwargs.get("env", {}))
                return FailedProcess() if len(attempts) == 1 else real_popen(*args, **kwargs)
            async with async_playwright() as playwright:
                with patch.object(arena.subprocess, "Popen", popen):
                    context = await arena._open_qt_browser(playwright)
                    try:
                        self.assertEqual(len(attempts), 2)
                        page = await context.new_page()
                        self.assertEqual(await page.evaluate("2 + 3"), 5)
                    finally:
                        await context.close()
        asyncio.run(run())

    @unittest.skipUnless(os.name == "nt", "Windows desktop visibility check")
    def test_background_qt_pages_never_show_native_windows(self):
        async def run():
            import ctypes
            from ctypes import wintypes
            from playwright.async_api import async_playwright
            user32 = ctypes.WinDLL("user32")
            callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
            user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
            user32.IsWindowVisible.argtypes = [wintypes.HWND]
            async with async_playwright() as playwright:
                context = await arena._open_qt_browser(playwright)
                try:
                    seen = []
                    @callback_type
                    def inspect(hwnd, unused):
                        pid = wintypes.DWORD()
                        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                        if pid.value == context.process.pid and user32.IsWindowVisible(hwnd):
                            seen.append(hwnd)
                        return True
                    async def create_and_raise():
                        page = await context.new_page()
                        await page.bring_to_front()
                        await page.set_content('<button>test</button>')
                        await page.get_by_role('button').click()
                    operation = asyncio.create_task(create_and_raise())
                    while not operation.done():
                        user32.EnumWindows(inspect, 0)
                        await asyncio.sleep(.01)
                    await operation
                    user32.EnumWindows(inspect, 0)
                    self.assertEqual(seen, [])
                finally:
                    await context.close()
        asyncio.run(run())

    def test_qt_browser_profile_and_cleanup(self):
        async def run():
            from playwright.async_api import async_playwright
            async with async_playwright() as playwright:
                for attempt in range(2):
                    context = await arena._open_qt_browser(playwright)
                    try:
                        page = await context.new_page()
                        self.assertEqual(await context.cookies("https://arena.ai/"), [])
                        await context.add_cookies([{"name": "isolation-test", "value": "test", "url": "https://arena.ai/"}])
                        self.assertEqual(len(await context.cookies("https://arena.ai/")), 1)
                        await page.expose_binding("arenaTest", lambda source, value: value + 1)
                        self.assertEqual(await page.evaluate("arenaTest(4)"), 5)
                    finally:
                        await context.close()
                    self.assertIsNotNone(context.process.poll())
        asyncio.run(run())

    def test_sidebar_and_login_navigation(self):
        async def run():
            from playwright.async_api import async_playwright
            os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(RUNTIME.parent / "browsers")
            async with async_playwright() as playwright:
                browser = await arena._open_qt_browser(playwright)
                try:
                    page = await browser.new_page()
                    for expanded in (False, True):
                        await page.set_content('''
                            <script>window.sidebarClicks=0;window.loginClicks=0;</script>
                            <button aria-label="Toggle Sidebar" onclick="window.sidebarClicks++;document.querySelector('aside').hidden=false">Sidebar</button>
                            <button hidden>Log In</button>
                            <aside %s><button onclick="window.loginClicks++">Log In</button></aside>
                        ''' % ("" if expanded else "hidden"))
                        self.assertTrue(await arena._open_arena_login(page, {}))
                        self.assertEqual(await page.evaluate("window.sidebarClicks"), 0 if expanded else 1)
                        self.assertEqual(await page.evaluate("window.loginClicks"), 1)
                    # Icon-only trigger; login arrives after the sidebar animation/hydration.
                    await page.set_content('''
                        <script>window.sidebarClicks=0;window.loginClicks=0;</script>
                        <button data-sidebar="trigger" onclick="window.sidebarClicks++">Sidebar</button>
                    ''')
                    navigation = {}
                    self.assertFalse(await arena._open_arena_login(page, navigation))
                    self.assertFalse(await arena._open_arena_login(page, navigation))
                    self.assertEqual(await page.evaluate("window.sidebarClicks"), 1)
                    await page.evaluate("""document.body.insertAdjacentHTML('beforeend', '<a href="#" onclick="window.loginClicks++">Log In</a>')""")
                    self.assertTrue(await arena._open_arena_login(page, navigation))
                    self.assertEqual(await page.evaluate("window.loginClicks"), 1)
                finally:
                    await browser.close()
        asyncio.run(run())


@unittest.skipUnless(importlib.util.find_spec("camoufox") and (RUNTIME / "bridge").exists(),
                     "Run with installed Arena managed Python for the offline bridge integration test")
class WorkerTest(unittest.TestCase):
    def setUp(self):
        async def open_context(playwright, visible=False):
            browser = await playwright.chromium.launch(headless=not visible)
            return await browser.new_context()
        patcher = patch.object(arena, "_open_qt_browser", open_context)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_upstream_http_errors_are_prompt_and_preserve_status(self):
        import httpx
        browser = Browser()

        class Playwright:
            async def start(self):
                self.chromium = self
                return self

            async def launch(self, **kwargs):
                return browser

            async def stop(self):
                pass

        catalog = [{"id": "test-model-id", "publicName": "test-model", "organization": "test", "capabilities": {}}]

        async def discover_catalog(context):
            return catalog

        original_import = importlib.import_module

        def import_bridge(name, *args, **kwargs):
            module = original_import(name, *args, **kwargs)
            if name.startswith("arena_slot_") and name.endswith(".src.main"):
                async def discovery():
                    module.save_models(catalog)
                module.get_initial_data = discovery
            return module

        async def serve(server, sockets):
            headers = {"Authorization": "Bearer test-key"}
            payload = {"account_slot": 0, "model": "test-model", "messages": [{"role": "user", "content": "test"}]}
            try:
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.config.app), base_url="http://test", headers=headers) as client:
                    for status in (429, 404, 500):
                        with self.subTest(status=status):
                            async def reject(page, request_payload):
                                await page.emit(None, {"status": status, "headers": {"retry-after": "120"},
                                    "error_body": json.dumps({"error": "test rejection " + str(status)})})

                            browser.response_handler = reject
                            before = len(browser.sent)
                            # Retry-After must reach Glossarion, not become a hidden
                            # two-minute bridge sleep followed by another submission.
                            response = await asyncio.wait_for(client.post("/v1/chat/completions", json=payload), 5)
                            self.assertEqual(response.status_code, 200, response.text)
                            with self.assertRaisesRegex(arena.ArenaStreamError, "test rejection " + str(status)) as caught:
                                arena.consume_stream(response.text.splitlines(), log_stream=False)
                            self.assertEqual(caught.exception.http_status, status)
                            self.assertEqual(caught.exception.retry_after, "120")
                            self.assertFalse(caught.exception.partial_response)
                            self.assertEqual(len(browser.sent), before + 1)
                            self.assertTrue(all(not context.pages for context in browser.contexts))

                            contexts = list(browser.contexts)
                            browser.response_handler = None
                            recovered = await asyncio.wait_for(client.post("/v1/chat/completions", json=payload), 5)
                            self.assertEqual(arena.consume_stream(recovered.text.splitlines(), log_stream=False)["content"], "hello")
                            self.assertEqual(len(browser.sent), before + 2)
                            self.assertEqual(browser.contexts, contexts)
            finally:
                for sock in sockets:
                    sock.close()

        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {"AUTHARENA_PROXY_DATA_DIR": root}), \
                patch.object(arena, "__file__", str(RUNTIME / "autharena_proxy.py")), \
                patch.object(arena, "_discover_catalog", discover_catalog), \
                patch("playwright.async_api.async_playwright", Playwright), \
                patch("uvicorn.Server.serve", serve), patch("importlib.import_module", import_bridge):
            expiration = int(time.time()) + 3600
            token = "base64-" + base64.urlsafe_b64encode(json.dumps({"access_token": "test-access",
                "refresh_token": "test-refresh", "expires_at": expiration,
                "user": {"id": "user-0", "email": "saved@example.test"}}).encode()).decode()
            arena._save("accounts.enc", {"0": {"token": token, "user_id": "user-0", "expires_at": expiration,
                "cookies": [{"name": "test-account", "value": "slot-0", "domain": ".arena.ai", "path": "/"}]}})
            asyncio.run(arena._serve_worker("test-key"))

    def test_same_account_stream_overlap_cancel_and_reconnect(self):
        import httpx
        browser = Browser()

        @contextlib.asynccontextmanager
        async def login_browser(playwright):
            context = await browser.new_context()
            try:
                yield context
            finally:
                await context.close()

        class Playwright:
            async def start(self):
                self.chromium = self
                return self

            async def launch(self, **kwargs):
                return browser

            async def stop(self):
                pass

        catalog = [{"id": "test-model-id", "publicName": "test-model", "organization": "test", "capabilities": {}}]

        async def discover_catalog(context):
            return catalog

        original_import = importlib.import_module

        def import_bridge(name, *args, **kwargs):
            module = original_import(name, *args, **kwargs)
            if name.startswith("arena_slot_") and name.endswith(".src.main"):
                async def discovery():
                    module.save_models(catalog)
                module.get_initial_data = discovery
            return module

        async def serve(server, sockets):
            started, gates, pages, payloads = {}, {}, {}, {}
            requests = []

            async def respond(page, payload):
                label = payload["userMessage"]["content"].strip()
                pages[label], payloads[label] = page, payload
                self.assertEqual(payload["recaptchaV3Token"], page.token)
                await page.emit(None, {"status": 200, "headers": {}})
                await page.emit(None, {"line": "ag:" + json.dumps("thinking " + label)})
                started[label].set()
                await gates[label].wait()
                await page.emit(None, {"line": "a0:" + json.dumps("answer " + label)})
                await page.emit(None, {"line": 'ad:{"finishReason":"stop","usage":{"total_tokens":8}}'})

            browser.response_handler = respond
            headers = {"Authorization": "Bearer test-key"}
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.config.app), base_url="http://test", headers=headers) as client:
                def launch(label, complete=False):
                    started[label], gates[label] = asyncio.Event(), asyncio.Event()
                    if complete:
                        gates[label].set()
                    task = asyncio.create_task(client.post("/v1/chat/completions", json={
                        "request_id": label, "account_slot": 0, "stream_timeout": 30,
                        "model": "test-model", "messages": [{"role": "user", "content": label}]}))
                    requests.append(task)
                    return task

                async def result(task, label):
                    response = await asyncio.wait_for(task, 5)
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual(response.headers["X-Arena-Account-Slot"], "0")
                    parsed = arena.consume_stream(response.text.splitlines(), log_stream=False)
                    self.assertEqual(parsed["content"], "answer " + label)
                    self.assertEqual(parsed["reasoning_content"], "thinking " + label)
                    self.assertEqual(parsed["usage"], {"total_tokens": 8})

                try:
                    first = launch("first")
                    await asyncio.wait_for(started["first"].wait(), 5)
                    second = launch("second")
                    # Neither answer can finish until both requests have reached
                    # Arena. Serializing the account makes this assertion fail.
                    await asyncio.wait_for(started["second"].wait(), 5)
                    first_context, second_context = pages["first"].context, pages["second"].context
                    self.assertIsNot(first_context, second_context)
                    self.assertNotEqual(pages["first"].token, pages["second"].token)
                    self.assertNotEqual(payloads["first"]["id"], payloads["second"]["id"])
                    self.assertNotEqual(payloads["first"]["userMessageId"], payloads["second"]["userMessageId"])
                    self.assertEqual(first_context.saved_cookies[0]["value"], "slot-0")
                    self.assertEqual(second_context.saved_cookies[0]["value"], "slot-0")
                    first_context.saved_cookies.append({"name": "request-local", "value": "first", "domain": ".arena.ai", "path": "/"})
                    self.assertEqual(len(second_context.saved_cookies), 1)
                    self.assertEqual(browser.contexts[0].saved_cookies, [])

                    gates["first"].set()
                    await result(first, "first")
                    self.assertFalse(second.done())
                    cancelled = launch("cancelled")
                    await asyncio.wait_for(started["cancelled"].wait(), 5)
                    self.assertIs(pages["cancelled"].context, first_context)
                    await client.post("/cancel", json={"id": "cancelled"})
                    await asyncio.wait_for(asyncio.gather(cancelled, return_exceptions=True), 5)
                    self.assertFalse(second.done())
                    self.assertIn(pages["second"], second_context.pages)
                    gates["second"].set()
                    await result(second, "second")
                    self.assertNotIn(pages["cancelled"], first_context.pages)

                    contexts = list(browser.contexts)
                    await result(launch("reused", complete=True), "reused")
                    self.assertEqual(browser.contexts, contexts)

                    old = launch("old-session")
                    await asyncio.wait_for(started["old-session"].wait(), 5)
                    old_context = pages["old-session"].context
                    browser.login_user = {"id": "user-0", "email": "reconnected@example.test"}
                    login = await asyncio.wait_for(client.post("/login", json={"slot": 0}), 5)
                    self.assertEqual(login.status_code, 200, login.text)
                    self.assertIn(old_context, browser.contexts)
                    self.assertIn(pages["old-session"], old_context.pages)
                    self.assertFalse(old.done())
                    fresh = launch("new-session", complete=True)
                    await result(fresh, "new-session")
                    self.assertIsNot(pages["new-session"].context, old_context)
                    self.assertEqual(pages["new-session"].context.saved_cookies[0]["name"], "arena-auth-prod-v1")
                    gates["old-session"].set()
                    await result(old, "old-session")
                    self.assertNotIn(old_context, browser.contexts)
                    self.assertEqual(len(browser.sent), 6)
                    self.assertEqual(len({p["id"] for p in payloads.values()}), 6)
                finally:
                    for task in requests:
                        task.cancel()
                    await asyncio.gather(*requests, return_exceptions=True)
                    for sock in sockets:
                        sock.close()

        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {"AUTHARENA_PROXY_DATA_DIR": root}), \
                patch.object(arena, "__file__", str(RUNTIME / "autharena_proxy.py")), \
                patch.object(arena, "_regular_login_browser", login_browser), \
                patch.object(arena, "_discover_catalog", discover_catalog), \
                patch("playwright.async_api.async_playwright", Playwright), \
                patch("uvicorn.Server.serve", serve), patch("importlib.import_module", import_bridge):
            expiration = int(time.time()) + 3600
            token = "base64-" + base64.urlsafe_b64encode(json.dumps({
                "access_token": "test-access", "refresh_token": "test-refresh", "expires_at": expiration,
                "user": {"id": "user-0", "email": "saved@example.test"}}).encode()).decode()
            arena._save("accounts.enc", {"0": {"token": token, "user_id": "user-0", "expires_at": expiration,
                "cookies": [{"name": "test-account", "value": "slot-0", "domain": ".arena.ai", "path": "/"}]}})
            asyncio.run(arena._serve_worker("test-key"))

    def test_pinned_bridge_routing_isolation_and_stream(self):
        import httpx
        browser = Browser()

        @contextlib.asynccontextmanager
        async def login_browser(playwright):
            context = await browser.new_context()
            try:
                yield context
            finally:
                await context.close()

        class Playwright:
            chromium = None

            async def start(self):
                self.chromium = self
                return self

            async def launch(self, **kwargs):
                assert kwargs["headless"] is True
                return browser

            async def stop(self):
                pass

        original_import = importlib.import_module

        async def discover_catalog(context):
            return [{"id": "test-model-id", "publicName": "test-model", "organization": "test", "capabilities": {}}]

        def import_bridge(name, *args, **kwargs):
            module = original_import(name, *args, **kwargs)
            if name.startswith("arena_slot_") and name.endswith(".src.main"):
                async def discovery():
                    module.save_models([{"id": "test-model-id", "publicName": "test-model", "organization": "test", "capabilities": {}}])
                module.get_initial_data = discovery
            return module

        async def serve(server, sockets):
            headers = {"Authorization": "Bearer test-key"}
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.config.app), base_url="http://test", headers=headers) as client:
                assert (await client.get("/health", headers={"Authorization": "bad"})).status_code == 401
                for slot in (0, 1, None, None):
                    response = await client.post("/v1/chat/completions", json={"model": "test-model", "messages": [{"role": "user", "content": "test"}], "account_slot": slot})
                    assert response.status_code == 200, response.text
                    assert response.headers["X-Arena-Account-Slot"] in ("0", "1")
                    if slot is not None:
                        assert response.headers["X-Arena-Account-Slot"] == str(slot)
                    result = arena.consume_stream(response.text.splitlines(), log_stream=False)
                    assert result["content"] == "hello", response.text
                    assert result["usage"] == {"total_tokens": 8}, response.text
                    assert result["reasoning_content"] == "reasoning", response.text
                assert len(browser.sent) == 4
                cookie_ids = [cookies[0]["value"] for cookies, payload in browser.sent]
                assert cookie_ids == ["slot-0", "slot-1", "slot-0", "slot-1"]
                assert len({payload["id"] for cookies, payload in browser.sent}) == 4
                assert browser.contexts[0].saved_cookies == []
                assert len(browser.contexts) == 3
                assert not browser.contexts[1].pages
                assert not browser.contexts[2].pages
                responses = await asyncio.gather(*[
                    client.post("/v1/chat/completions", json={"model": "test-model", "messages": [{"role": "user", "content": "test"}], "account_slot": slot})
                    for slot in (0, 1, 0)])
                for response in responses:
                    assert arena.consume_stream(response.text.splitlines(), log_stream=False)["content"] == "hello"
                assert len(browser.sent) == 7
                assert len({payload["id"] for cookies, payload in browser.sent}) == 7
                await client.post("/cancel", json={"id": "cancel-before-dispatch"})
                cancelled = await client.post("/v1/chat/completions", json={"request_id": "cancel-before-dispatch"})
                assert cancelled.status_code == 409
                assert len(browser.sent) == 7
                login = await client.post("/login", json={"slot": 2})
                assert login.status_code == 200, login.text
                assert login.json()["slot"] == 2
                assert browser.login_clicks == 1
                assert all(url == "https://arena.ai/" for url in browser.urls)
                assert arena._load("accounts.enc")["2"]["user_id"] == "new-user"
                mismatch = await client.post("/login", json={"slot": 0})
                assert mismatch.status_code == 409, mismatch.text
                assert arena._load("accounts.enc")["0"]["user_id"] == "user-0"
                browser.fail_request = True
                before = len(browser.sent)
                failed = await client.post("/v1/chat/completions", json={"model": "test-model", "messages": [{"role": "user", "content": "test"}], "account_slot": 0})
                with self.assertRaisesRegex(RuntimeError, "invalid test payload"):
                    arena.consume_stream(failed.text.splitlines(), log_stream=False)
                self.assertEqual(len(browser.sent), before + 1)
                browser.fail_request = False
                browser.mint_gate = asyncio.Event()
                browser.mint_started.clear()
                before = len(browser.sent)
                interrupted = asyncio.create_task(client.post("/v1/chat/completions", json={
                    "request_id": "cancel-during-token", "dispatch_ack": True, "account_slot": 0,
                    "model": "test-model", "messages": [{"role": "user", "content": "test"}]}))
                await asyncio.wait_for(browser.mint_started.wait(), 5)
                await client.post("/cancel", json={"id": "cancel-during-token"})
                await asyncio.wait_for(asyncio.gather(interrupted, return_exceptions=True), 5)
                self.assertEqual(len(browser.sent), before)
                browser.mint_gate = None
                resumed = await asyncio.wait_for(client.post("/v1/chat/completions", json={
                    "account_slot": 0, "model": "test-model", "messages": [{"role": "user", "content": "test"}]}), 5)
                self.assertEqual(arena.consume_stream(resumed.text.splitlines(), log_stream=False)["content"], "hello")
                browser.fail_request = False
                refreshed = {"user": {"id": "user-0", "email": "restored@example.test"},
                             "expires_at": int(time.time()) + 3600, "refresh_token": "test-refresh"}
                browser.refreshed_cookie = {"name": "arena-auth-prod-v1", "domain": ".arena.ai", "path": "/",
                    "value": "base64-" + base64.urlsafe_b64encode(json.dumps(refreshed).encode()).decode()}
                browser.reject_once = True
                before = len(browser.sent)
                recovered = await client.post("/v1/chat/completions", json={"model": "test-model", "messages": [{"role": "user", "content": "test"}], "account_slot": 0})
                self.assertEqual(arena.consume_stream(recovered.text.splitlines(), log_stream=False)["content"], "hello")
                self.assertEqual(len(browser.sent), before + 2)
                self.assertEqual(arena._load("accounts.enc")["0"]["email"], "restored@example.test")
                browser.reject_once = True
                browser.rejection_body = '{"error":"recaptcha validation failed"}'
                contexts_before = list(browser.contexts)
                before = len(browser.sent)
                recovered = await client.post("/v1/chat/completions", json={"model": "test-model", "messages": [{"role": "user", "content": "test"}], "account_slot": 0})
                self.assertEqual(arena.consume_stream(recovered.text.splitlines(), log_stream=False)["content"], "hello")
                self.assertEqual(len(browser.sent), before + 2)
                self.assertEqual(browser.contexts, contexts_before)
                browser.challenge_pages = 2  # Headless page, then visible verification.
                browser.before_challenge = len(browser.sent)
                recovered = await client.post("/v1/chat/completions", json={"model": "test-model", "messages": [{"role": "user", "content": "test"}], "account_slot": 0})
                self.assertEqual(arena.consume_stream(recovered.text.splitlines(), log_stream=False)["content"], "hello")
                self.assertEqual(browser.verifications, 1)
                self.assertEqual(len(browser.sent), browser.before_challenge + 1)
                self.assertEqual(browser.contexts, contexts_before)
                before = len(browser.sent)
                delayed = asyncio.create_task(client.post("/v1/chat/completions", json={
                    "request_id": "delayed-ack", "dispatch_ack": True, "account_slot": 0,
                    "model": "test-model", "messages": [{"role": "user", "content": "test"}]}))
                await asyncio.sleep(31)  # Regression: old handshake expired after 30 seconds.
                self.assertEqual(len(browser.sent), before)
                self.assertFalse(delayed.done())
                ack = await client.post("/dispatch", json={"id": "delayed-ack"})
                self.assertEqual(ack.status_code, 200)
                recovered = await asyncio.wait_for(delayed, 5)
                self.assertEqual(arena.consume_stream(recovered.text.splitlines(), log_stream=False)["content"], "hello")
                self.assertEqual(len(browser.sent), before + 1)
                before = len(browser.sent)
                browser.delay_after_reasoning = 0.5
                stalled = await client.post("/v1/chat/completions", json={
                    "stream_timeout": 0.1, "account_slot": 0,
                    "model": "test-model", "messages": [{"role": "user", "content": "test"}]})
                with self.assertRaisesRegex(arena.ArenaStreamError, "no upstream data") as caught:
                    arena.consume_stream(stalled.text.splitlines(), log_stream=False)
                self.assertTrue(caught.exception.partial_response)
                self.assertEqual(len(browser.sent), before + 1)
                browser.delay_after_reasoning = 0
            for sock in sockets:
                sock.close()

        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {"AUTHARENA_PROXY_DATA_DIR": root}), \
                patch.object(arena, "__file__", str(RUNTIME / "autharena_proxy.py")), \
                patch.object(arena, "_regular_login_browser", login_browser), \
                patch.object(arena, "_discover_catalog", discover_catalog), \
                patch("playwright.async_api.async_playwright", Playwright), \
                patch("uvicorn.Server.serve", serve), patch("importlib.import_module", import_bridge):
            expiration = int(time.time()) + 3600
            b64 = lambda value: base64.urlsafe_b64encode(json.dumps(value).encode()).decode().rstrip("=")
            token = "base64-" + b64({"access_token": b64({"alg": "HS256"}) + "." + b64({"exp": expiration, "sub": "test"}) + ".signature", "refresh_token": "test", "expires_at": expiration, "user": {"id": "test", "email": "test@example.test"}})
            arena._save("accounts.enc", {str(i): {"token": token, "user_id": f"user-{i}", "expires_at": expiration, "cookies": [{"name": "test-account", "value": f"slot-{i}", "domain": ".arena.ai", "path": "/"}]} for i in range(2)})
            asyncio.run(arena._serve_worker("test-key"))


if __name__ == "__main__":
    unittest.main()
