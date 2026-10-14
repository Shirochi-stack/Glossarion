"""Native Qt/CDP regressions. All pages and network responses are local fixtures."""
import asyncio
import contextlib
import json
import subprocess
import sys

import pytest

from autharena_browser import _CDP, _QtPipeSocket


@pytest.fixture
def qt_relay_process():
    processes = []

    def start(mode="echo"):
        process = subprocess.Popen(
            [sys.executable, "-u", "-c", r'''
import json
import sys

mode = sys.argv[1]
def emit(event, **fields):
    print(json.dumps(dict(event=event, **fields)), flush=True)

for line in sys.stdin:
    command = json.loads(line)
    action = command["action"]
    if action == "connect":
        if mode == "connect_error":
            emit("error", message="fixture connection refused")
        elif mode == "connect_closed":
            emit("closed")
        elif mode == "connect_eof":
            break
        else:
            emit("connected")
    elif action == "cdp":
        if mode == "relay_error":
            emit("error", message="fixture socket failed")
        elif mode == "relay_closed":
            emit("closed")
        elif mode == "relay_eof":
            break
        else:
            emit("message", data=command["message"])
    elif action == "disconnect" and mode != "unresponsive":
        emit("closed")
    elif action == "quit":
        break
''', mode], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, encoding="utf-8",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        processes.append(process)
        return process

    yield start
    for process in processes:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        with contextlib.suppress(OSError, ValueError):
            process.stdin.close()
        process.stdout.close()


def test_qt_pipe_relays_unicode_and_disconnect_stops_reader(qt_relay_process):
    async def run():
        process = qt_relay_process()
        socket = await _QtPipeSocket.connect(process, "ws://127.0.0.1:9222/devtools/browser/test")
        message = json.dumps({"text": "مرحبا 世界\nnext line"}, ensure_ascii=False)
        await socket.send(message)
        assert await asyncio.wait_for(anext(socket), 2) == message
        await socket.close()
        await socket.close()
        assert not socket._reader.is_alive()
        assert process.stdout.closed
        with pytest.raises(StopAsyncIteration):
            await anext(socket)
        with pytest.raises(RuntimeError, match="closed"):
            await socket.send("{}")

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["connect_error", "connect_closed", "connect_eof"])
def test_qt_pipe_connect_failure_closes_response_pipe(qt_relay_process, mode):
    async def run():
        process = qt_relay_process(mode)
        with pytest.raises(RuntimeError, match="connection refused|before connecting"):
            await asyncio.wait_for(_QtPipeSocket.connect(process, "ws://127.0.0.1:1/"), 3)
        assert process.stdout.closed

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["relay_error", "relay_closed", "relay_eof"])
def test_qt_pipe_terminal_events_fail_pending_cdp_calls(qt_relay_process, mode):
    async def run():
        socket = await _QtPipeSocket.connect(qt_relay_process(mode), "ws://127.0.0.1:9222/")
        cdp = _CDP(socket, lambda *args: None, lambda: None)
        try:
            with pytest.raises(RuntimeError, match="disconnected"):
                await asyncio.wait_for(cdp.send("Runtime.evaluate", timeout=None), 3)
            assert not cdp._pending
            assert cdp.closed
        finally:
            await cdp.close()
        assert not socket._reader.is_alive()

    asyncio.run(run())


def test_qt_pipe_unresponsive_helper_is_reaped_on_close(qt_relay_process, monkeypatch):
    async def run():
        process = qt_relay_process("unresponsive")
        socket = await _QtPipeSocket.connect(process, "ws://127.0.0.1:9222/")
        monkeypatch.setattr(socket, "_CLOSE_TIMEOUT", .5)
        await asyncio.wait_for(socket.close(), 3)
        assert process.poll() is not None
        assert not socket._reader.is_alive()
        assert process.stdout.closed

    asyncio.run(run())


class Socket:
    def __init__(self):
        self.incoming = asyncio.Queue()
        self.outgoing = asyncio.Queue()

    def __aiter__(self):
        return self

    async def __anext__(self):
        message = await self.incoming.get()
        if message is None:
            raise StopAsyncIteration
        return json.dumps(message)

    async def send(self, message):
        await self.outgoing.put(json.loads(message))

    async def close(self):
        await self.incoming.put(None)


def test_disconnect_fails_pending_commands_and_rejects_later_calls():
    async def run():
        socket = Socket()
        disconnected = []
        cdp = _CDP(socket, lambda *args: None, lambda: disconnected.append(True))
        pending = asyncio.create_task(cdp.send("Runtime.evaluate", timeout=None))
        await socket.outgoing.get()
        await socket.incoming.put(None)
        with pytest.raises(RuntimeError, match="disconnected"):
            await pending
        with pytest.raises(RuntimeError, match="closed"):
            await cdp.send("Page.enable")
        assert not cdp._pending
        assert disconnected == [True]
        await cdp.close()
        assert cdp._reader.done()

    asyncio.run(run())


def test_cancelled_cdp_call_does_not_poison_later_commands():
    async def run():
        socket = Socket()
        cdp = _CDP(socket, lambda *args: None, lambda: None)
        try:
            first = asyncio.create_task(cdp.send("Runtime.evaluate", session="old", timeout=None))
            first_message = await socket.outgoing.get()
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            await socket.incoming.put({"id": first_message["id"], "result": {"ignored": True}})
            second = asyncio.create_task(cdp.send("Runtime.evaluate", session="new"))
            second_message = await socket.outgoing.get()
            await socket.incoming.put({"id": second_message["id"], "result": {"value": 42}})
            assert await second == {"value": 42}
            assert not cdp._pending
        finally:
            await cdp.close()

    asyncio.run(run())


def test_closing_one_page_fails_only_its_pending_cdp_calls():
    async def run():
        socket = Socket()
        cdp = _CDP(socket, lambda *args: None, lambda: None)
        try:
            first = asyncio.create_task(cdp.send("Runtime.evaluate", session="first", timeout=None))
            second = asyncio.create_task(cdp.send("Runtime.evaluate", session="second", timeout=None))
            messages = [await socket.outgoing.get(), await socket.outgoing.get()]
            cdp.fail_session("first")
            with pytest.raises(RuntimeError, match="page is closed"):
                await first
            assert not second.done()
            message = next(item for item in messages if item["sessionId"] == "second")
            await socket.incoming.put({"id": message["id"], "result": {"value": "other account"}})
            assert await second == {"value": "other account"}
        finally:
            await cdp.close()

    asyncio.run(run())


def test_real_qt_bindings_cross_origin_clicks_and_page_cleanup():
    pytest.importorskip("PySide6.QtWebEngineWidgets")
    import autharena_proxy as arena

    async def run():
        context = await arena._open_qt_browser()
        try:
            await context.add_init_script("window.initializedByQt = true;")
            await context.add_cookies([{"name": "account", "value": "test-account", "url": "https://arena.test/"}])
            seen = []

            async def fixture(route):
                seen.append(route.request.url)
                if "frame.test" in route.request.url:
                    body = '<input type="checkbox" onclick="window.trustedClick=event.isTrusted">'
                else:
                    body = '<title>Qt fixture</title><div style="height:200px"></div><iframe style="margin-left:250px" src="https://frame.test/widget"></iframe>'
                await route.fulfill(content_type="text/html", body=body)

            await context.route("**/*", fixture)
            page = await context.new_page()
            order = []

            async def observe(route):
                order.append("page")
                headers = await route.request.all_headers()
                assert "account=test-account" in headers["cookie"]
                await route.fallback()

            await page.route("https://arena.test/", observe)
            await page.expose_binding("double", lambda source, number: number * 2)

            async def failing(source):
                assert source["page"] is page
                assert source["context"] is context
                raise ValueError("fixture callback failed")

            await page.expose_binding("failing", failing)
            await page.goto("https://arena.test/", wait_until="load")
            assert await page.title() == "Qt fixture"
            assert await page.evaluate("window.initializedByQt") is True
            assert await page.evaluate("async () => await double(21)") == 42
            assert await page.evaluate("async () => {try {await failing()} catch(error) {return error.message}}") == "fixture callback failed"
            frame = await (await page.query_selector("iframe")).content_frame()
            assert frame is not None
            checkbox = await frame.query_selector("input[type=checkbox]")
            await checkbox.click(force=True)
            assert await frame.evaluate("window.trustedClick") is True
            assert order == ["page"]
            assert "https://frame.test/widget" in seen

            binding_started = asyncio.Event()
            binding_cancelled = asyncio.Event()

            async def blocked(source):
                binding_started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    binding_cancelled.set()

            await page.expose_binding("blocked", blocked)
            pending = asyncio.create_task(page.evaluate("async () => await blocked()"))
            await asyncio.wait_for(binding_started.wait(), 5)
            await page.close()
            with pytest.raises(RuntimeError, match="closed"):
                await pending
            assert binding_cancelled.is_set()
            assert not context.pages
            assert page.target not in context._pages
            assert page._session not in context._sessions
            assert not page._tasks
        finally:
            await context.close()
        assert not context._tasks
        assert not context._cdp._pending
        assert context.process.returncode == 0
        assert context.process.stdout.closed
        assert not context._cdp.socket._reader.is_alive()

    asyncio.run(run())


def test_real_qt_route_fetch_reads_browser_response_without_second_request():
    pytest.importorskip("PySide6.QtWebEngineWidgets")
    import autharena_proxy as arena

    async def run():
        requests = []

        async def serve(reader, writer):
            try:
                head = await reader.readuntil(b"\r\n\r\n")
                path = head.split(b" ")[1].decode()
                requests.append(path)
                if path == "/script.js":
                    body, kind = b"window.discovered = 42;", "text/javascript"
                else:
                    body, kind = b'<script src="/script.js"></script>', "text/html"
                writer.write((f"HTTP/1.1 200 OK\r\nContent-Type: {kind}\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n").encode() + body)
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()

        server = await asyncio.start_server(serve, "127.0.0.1", 0)
        context = await arena._open_qt_browser()
        captured = []
        try:
            page = await context.new_page()

            async def capture(route):
                if route.request.url.endswith("/script.js"):
                    response = await route.fetch()
                    captured.append(await response.text())
                    await route.fulfill(response=response, body=await response.body())
                else:
                    await route.continue_()

            await page.route("**/*", capture)
            port = server.sockets[0].getsockname()[1]
            await page.goto(f"http://127.0.0.1:{port}/")
            assert await page.evaluate("window.discovered") == 42
            assert captured == ["window.discovered = 42;"]
            assert requests.count("/script.js") == 1
        finally:
            await context.close()
            server.close()
            await server.wait_closed()

    asyncio.run(run())


def test_real_qt_cancelled_page_creation_retires_unclaimed_view(monkeypatch):
    pytest.importorskip("PySide6.QtWebEngineWidgets")
    import autharena_proxy as arena

    async def run():
        context = await arena._open_qt_browser()
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def delayed_attach(info):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        monkeypatch.setattr(context, "_attach", delayed_attach)
        pending = asyncio.create_task(context.new_page())
        try:
            await asyncio.wait_for(started.wait(), 5)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            assert cancelled.is_set()
            assert not context.is_connected()
            assert context.process.poll() is not None
            assert not context._tasks
            assert not context._cdp._pending
        finally:
            await context.close()

    asyncio.run(run())
