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

    async def expose_binding(self, name, callback):
        self.emit = callback

    async def goto(self, *args, **kwargs):
        self.context.browser.urls.append(args[0])
        if self.context.browser.refreshed_cookie:
            self.context.saved_cookies = [self.context.browser.refreshed_cookie]

    async def bring_to_front(self):
        pass

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
                session = {"user": {"id": "new-user", "email": "new@example.test"}, "expires_at": int(time.time()) + 3600}
                page.context.saved_cookies = [{"name": "arena-auth-prod-v1", "domain": ".arena.ai", "path": "/",
                                               "value": "base64-" + base64.urlsafe_b64encode(json.dumps(session).encode()).decode()}]
        return Control()

    async def close(self):
        if self in self.context.pages:
            self.context.pages.remove(self)

    async def evaluate(self, script, args):
        assert args["payload"]["mode"] == "direct-battle"
        assert args["payload"]["modelAId"] == "test-model-id"
        self.context.browser.sent.append((self.context.saved_cookies, args["payload"]))
        await self.emit(None, {"dispatching": True})
        if self.context.browser.reject_once:
            self.context.browser.reject_once = False
            await self.emit(None, {"status": 403, "headers": {}, "error_body": self.context.browser.rejection_body})
            return
        if self.context.browser.fail_request:
            await self.emit(None, {"status": 400, "headers": {}, "error_body": '{"message":"invalid test payload"}'})
            return
        await self.emit(None, {"status": 200, "headers": {}})
        await self.emit(None, {"line": 'ag:"reasoning"'})
        await self.emit(None, {"line": 'a0:"hello"'})
        await asyncio.sleep(.01)
        await self.emit(None, {"line": 'ad:{"finishReason":"stop","usage":{"total_tokens":8}}'})


class Context:
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
        self.contexts = [Context(self)]
        self.sent = []
        self.urls = []
        self.login_clicks = 0
        self.fail_request = False
        self.reject_once = False
        self.rejection_body = "session needs refresh"
        self.refreshed_cookie = None

    async def close(self):
        self.contexts.clear()

    def is_connected(self):
        return True

    async def new_context(self):
        context = Context(self)
        self.contexts.append(context)
        return context


@unittest.skipUnless(importlib.util.find_spec("playwright") and (RUNTIME / "browser-ready").exists(),
                     "Run with installed Arena managed Python for local browser tests")
class LoginNavigationTest(unittest.TestCase):
    def test_captcha_readiness_and_fresh_tokens_in_browser(self):
        tree = ast.parse(Path(arena.__file__).read_text(encoding="utf-8"))
        script = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
                      and isinstance(n.value, str) and n.value.startswith("async ({url, method, payload, sitekey, action})"))
        async def run():
            from playwright.async_api import async_playwright
            os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(RUNTIME.parent / "browsers")
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                try:
                    page = await browser.new_page()
                    for enterprise in (False, True):
                        await page.goto("about:blank")
                        await page.evaluate("""enterprise => {
                            window.events=[]; window.calls=[]; window.counter=0;
                            window.arenaEmit = async item => events.push(item);
                            window.fetch = async (url, options) => {
                                calls.push(JSON.parse(options.body)); return new Response('a0:"ok"\\n', {status:200});
                            };
                            const s=document.createElement('script'); s.type='application/json';
                            s.src='https://www.google.com/recaptcha/'+(enterprise?'enterprise':'api')+'.js?render=page-site-key';
                            document.head.append(s);
                            setTimeout(() => {
                                const api={ready:fn=>fn(), execute:async (key,{action})=>{
                                    if(key!=='request-v3-key'||action!=='chat_submit') throw Error('wrong CAPTCHA configuration');
                                    return 'fresh-'+(++counter);
                                }};
                                window.grecaptcha=enterprise?{enterprise:api}:api;
                            },150);
                        }""", enterprise)
                        args = {"url": "https://arena.ai/test", "method": "POST", "payload": {"recaptchaV3Token": "stale", "recaptchaV2Token": "stale-v2"}, "sitekey": "request-v3-key", "action": "sign_up"}
                        await page.evaluate(script, args)
                        await page.evaluate(script, args)
                        calls = await page.evaluate("calls")
                        self.assertEqual(await page.evaluate("events.filter(e => 'line' in e).map(e => e.line)"), ['a0:"ok"', 'a0:"ok"'])
                        self.assertEqual([v["recaptchaV3Token"] for v in calls], ["fresh-1", "fresh-2"])
                        self.assertTrue(all("recaptchaV2Token" not in v for v in calls))
                        await page.evaluate("grecaptcha.enterprise ? grecaptcha.enterprise.execute = async()=>'' : grecaptcha.execute = async()=>''")
                        with self.assertRaisesRegex(Exception, "ARENA_CAPTCHA_EMPTY"):
                            await page.evaluate(script, args)
                        self.assertEqual(await page.evaluate("calls.length"), 2)
                finally:
                    await browser.close()
        asyncio.run(run())

    def test_regular_browser_profile_and_cleanup(self):
        executable = arena._regular_browser_executable()
        async def run():
            from playwright.async_api import async_playwright
            with tempfile.TemporaryDirectory(prefix="Arena browser test ") as root, \
                    patch.dict(os.environ, {"AUTHARENA_PROXY_DATA_DIR": root}), \
                    patch.object(arena, "_regular_browser_executable", return_value=executable):
                async with async_playwright() as playwright:
                    for attempt in range(2):
                        async with arena._regular_login_browser(playwright) as context:
                            self.assertIs(context, context.browser.contexts[0])
                            page = context.pages[0] if context.pages else await context.new_page()
                            self.assertFalse(await page.evaluate("navigator.webdriver"))
                            self.assertEqual(await context.cookies("https://arena.ai/"), [])
                            await context.add_cookies([{"name": "isolation-test", "value": "test", "url": "https://arena.ai/"}])
                            self.assertEqual(len(list((Path(root) / "login-profiles").iterdir())), 1)
                        self.assertEqual(list((Path(root) / "login-profiles").iterdir()), [])
        asyncio.run(run())

    def test_sidebar_and_login_navigation(self):
        async def run():
            from playwright.async_api import async_playwright
            os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(RUNTIME.parent / "browsers")
            async with async_playwright() as playwright:
                browser = await playwright.chromium.launch(headless=True)
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
            for sock in sockets:
                sock.close()

        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {"AUTHARENA_PROXY_DATA_DIR": root}), \
                patch.object(arena, "__file__", str(RUNTIME / "autharena_proxy.py")), \
                patch.object(arena, "_regular_login_browser", login_browser), \
                patch("playwright.async_api.async_playwright", Playwright), \
                patch("uvicorn.Server.serve", serve), patch("importlib.import_module", import_bridge):
            expiration = int(time.time()) + 3600
            b64 = lambda value: base64.urlsafe_b64encode(json.dumps(value).encode()).decode().rstrip("=")
            token = "base64-" + b64({"access_token": b64({"alg": "HS256"}) + "." + b64({"exp": expiration, "sub": "test"}) + ".signature", "refresh_token": "test", "expires_at": expiration, "user": {"id": "test", "email": "test@example.test"}})
            arena._save("accounts.enc", {str(i): {"token": token, "user_id": f"user-{i}", "expires_at": expiration, "cookies": [{"name": "test-account", "value": f"slot-{i}", "domain": ".arena.ai", "path": "/"}]} for i in range(2)})
            asyncio.run(arena._serve_worker("test-key"))


if __name__ == "__main__":
    unittest.main()
