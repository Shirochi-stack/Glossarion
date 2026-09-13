import ast
import asyncio
import contextlib
import copy
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import autharena_proxy as arena


@pytest.mark.parametrize('status', [200, 403, 429])
def test_upstream_owns_captcha_fields_and_retry_policy(monkeypatch, status):
    tree = ast.parse(Path(arena.__file__).read_text(encoding='utf-8'))
    worker = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == '_serve_worker')
    configure = next(n for n in worker.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'configure_slot')
    sentinel = lambda *args: None
    discoveries = []
    async def discovery():
        discoveries.append(True)
    main = SimpleNamespace(click_turnstile=sentinel, refresh_recaptcha_token=sentinel, _cancel_background_task=sentinel, get_initial_data=discovery)
    recaptcha = SimpleNamespace(**{name: sentinel for name in (
        '_mint_recaptcha_v3_token_in_page', 'refresh_recaptcha_token',
        'get_cached_recaptcha_token', 'get_recaptcha_v3_token')})
    calls = []
    closed = []
    class Response:
        def __init__(self, status_code, headers, text=''):
            self.status_code, self.headers, self.text = status_code, headers, text
        async def aclose(self): closed.append(True)
    async def upstream(method, url, payload, auth_token, **kwargs):
        calls.append((copy.deepcopy(payload), kwargs))
        # Simulate the upstream v3 -> v2 decision. The adapter must not strip it.
        payload.pop('recaptchaV3Token', None)
        payload['recaptchaV2Token'] = 'upstream-v2'
        return Response(status, {'retry-after': '7'}, 'recaptcha validation failed' if status == 403 else '')
    transport = SimpleNamespace(fetch_lmarena_stream_via_camoufox=upstream, BrowserFetchStreamResponse=Response)
    modules = {'main': main, 'config': SimpleNamespace(_apply_config_defaults=lambda cfg: None),
               'auth': SimpleNamespace(), 'transport': transport, 'recaptcha': recaptcha}
    async def catalog(context): return [{'publicName': 'test'}]
    ns = dict(vars(arena), asyncio=asyncio, copy=copy, types=types, key='test',
              importlib=SimpleNamespace(import_module=lambda name: modules[name.rsplit('.', 1)[1]]),
              _prepare_qt_bridge_import=lambda namespace: None,
              _ensure_catalog=catalog)
    exec(compile(ast.Module(body=[configure], type_ignores=[]), 'configure', 'exec'), ns)
    async def run():
        async def cookies(urls): return []
        async def new_page(): raise AssertionError('Only upstream should create its request pages')
        context = SimpleNamespace(pages=[], cookies=cookies, new_page=new_page)
        state = await ns['configure_slot'](1, context, {'token': 'saved', 'cookies': []}, 'arena_test_upstream')
        state.update(submitted=False, events=asyncio.Queue(), dispatch_ack=None)
        assert discoveries == [True]
        assert main.get_initial_data is discovery
        assert recaptcha.find_chrome_executable() is None
        assert recaptcha.get_recaptcha_v3_token is sentinel
        assert main.click_turnstile is sentinel
        assert main.refresh_recaptcha_token is sentinel
        payload = {'mode': 'direct', 'recaptchaV3Token': 'upstream-v3', 'recaptchaV2Token': 'existing-v2'}
        response = await main.fetch_lmarena_stream_via_chrome(
            'POST', 'https://arena.ai/nextjs-api/stream/create-evaluation', payload, 'saved')
        assert len(calls) == 1
        assert calls[0] == ({'mode': 'direct-battle', 'recaptchaV3Token': 'upstream-v3', 'recaptchaV2Token': 'existing-v2'}, {})
        if status >= 400:
            assert state['upstream_error']['status_code'] == status
            assert state['upstream_error']['retry_after'] == '7'
        await response.aclose()
        if status == 200:
            # A retry chosen by the bridge must reach transport, not a local 400.
            retry = await main.fetch_lmarena_stream_via_chrome(
                'POST', 'https://arena.ai/nextjs-api/stream/create-evaluation', payload, 'saved')
            assert retry.status_code == 200
            assert len(calls) == 2
            await retry.aclose()
    try:
        asyncio.run(run())
    finally:
        sys.modules.pop('arena_test_upstream', None)


def test_bridge_cleanup_preserves_parent_cancellation():
    async def run():
        async def upstream_cancel(task):
            task.cancel()
            await task
        wrapped = arena._bridge_cancel_compat(upstream_cancel)
        child = asyncio.create_task(asyncio.sleep(60))
        await wrapped(child)
        assert child.cancelled()
        started = asyncio.Event()
        async def caller():
            async def blocked_cancel(task):
                started.set()
                await asyncio.sleep(60)
            await arena._bridge_cancel_compat(blocked_cancel)(child)
        parent = asyncio.create_task(caller())
        await started.wait()
        parent.cancel()
        with pytest.raises(asyncio.CancelledError):
            await parent
    asyncio.run(run())


@pytest.mark.parametrize('outcome', ['complete_before_dispatch', 'complete_after_dispatch', 'cancel', 'stall_after_dispatch'])
def test_idle_timeout_excludes_dispatch_wait_and_cleans_pending_tasks(outcome):
    async def run():
        existing = asyncio.all_tasks()
        entered, dispatch, finish, cleaned = (asyncio.Event() for _ in range(4))

        async def stream():
            try:
                entered.set()
                await finish.wait()
                return 'finished'
            finally:
                cleaned.set()

        monitored = asyncio.create_task(arena._run_stream_with_idle_timeout(
            stream(), asyncio.Event(), .05, started=dispatch))
        await entered.wait()
        # Preparation and dispatch approval can exceed the stream idle limit.
        await asyncio.sleep(.1)
        assert not monitored.done()
        if outcome == 'cancel':
            monitored.cancel()
            with pytest.raises(asyncio.CancelledError):
                await monitored
        elif outcome == 'stall_after_dispatch':
            dispatch.set()
            with pytest.raises(RuntimeError, match='no upstream data'):
                await asyncio.wait_for(monitored, 1)
        else:
            if outcome == 'complete_after_dispatch':
                dispatch.set()
            finish.set()
            assert await asyncio.wait_for(monitored, 1) == 'finished'
        assert cleaned.is_set()
        await asyncio.sleep(0)
        assert asyncio.all_tasks() == existing
    asyncio.run(run())


def test_pinned_bridge_v3_to_v2_fallback_in_real_qt(monkeypatch):
    import importlib.util
    import json
    if importlib.util.find_spec('PySide6') is None:
        pytest.skip('Requires Qt6 WebEngine')
    runtime = Path.home() / '.glossarion/autharena_proxy' / ('bridge-' + arena.REVISION)
    if not (runtime / 'bridge').exists():
        pytest.skip('Requires pinned bridge checkout')
    monkeypatch.syspath_prepend(str(runtime))
    with monkeypatch.context() as importing:
        importing.setattr(arena, '__file__', str(runtime / 'autharena_proxy.py'))
        arena._prepare_qt_bridge_import('bridge')
    from bridge.src import main, transport
    async def run():
        context = await arena._open_qt_browser(visible=False)
        original = context.new_page
        sent = []
        async def page():
            p = await original()
            arena._bridge_error_metadata(p)
            async def route(r):
                if '/nextjs-api/stream/' in r.request.url:
                    sent.append(r.request.post_data_json)
                    await r.fulfill(status=403 if len(sent) == 1 else 200,
                        content_type='application/json', body=json.dumps({'error': 'recaptcha validation failed'}) if len(sent) == 1 else 'ok\n')
                else:
                    await r.fulfill(content_type='text/html', body="<script>window.grecaptcha={enterprise:{ready:f=>f(),execute:async()=> 'upstream-v3',render:(el,opts)=>{setTimeout(()=>opts.callback('upstream-v2'),0);return 1;}}};</script>")
            await p.route('**/*', route)
            return p
        context.new_page = page
        class Factory:
            def __init__(self, **kwargs): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            async def new_context(self, **kwargs): return context
        monkeypatch.setattr(main, 'AsyncCamoufox', Factory)
        monkeypatch.setattr(main, '_cancel_background_task', arena._bridge_cancel_compat(main._cancel_background_task))
        monkeypatch.setattr(main, 'get_config', lambda: {'browser_cookies': {}, 'recaptcha_sitekey': 'test', 'recaptcha_action': 'chat_submit'})
        monkeypatch.setattr(main, 'debug_print', lambda *args, **kwargs: None)
        monkeypatch.setattr(main, 'save_config', lambda *args, **kwargs: None)
        try:
            response = await asyncio.wait_for(transport.fetch_lmarena_stream_via_camoufox(
                'POST', 'https://arena.ai/nextjs-api/stream/create-evaluation',
                {'recaptchaV3Token': ''}, 'test-auth', timeout_seconds=15), 40)
            assert response.status_code == 200
            assert len(sent) == 2
            assert 'recaptchaV3Token' in sent[0]
            assert 'recaptchaV2Token' in sent[1]
            assert 'recaptchaV3Token' not in sent[1]
            await response.aclose()
        finally:
            await context.close()
    asyncio.run(run())


def test_stream_response_lifetime_summary_and_empty_retry_input():
    async def run():
        closed = []
        class Response:
            status_code = 200
            async def __aenter__(self): return self
            async def aiter_lines(self):
                assert not closed
                yield 'a0:"hello"'
                yield 'ad:{"finishReason":"stop"}'
            async def aclose(self): closed.append('response')
        async def cleanup(): closed.append('pages')
        events = asyncio.Queue()
        wrapped = arena._ArenaBridgeResponse(Response(), cleanup, events)
        async with wrapped as stream:
            assert [line async for line in stream.aiter_lines()] == ['a0:"hello"', 'ad:{"finishReason":"stop"}']
            assert not closed
        assert closed == ['response', 'pages']
        await wrapped.aclose()
        assert closed == ['response', 'pages']
        summary = await events.get()
        assert summary == {'arena_progress': 'stream_summary:2 lines (a0=1, ad=1)'}
    asyncio.run(run())


def test_a3_error_message_is_reported_before_stream_summary():
    async def run():
        class Response:
            async def aiter_lines(self):
                yield 'a3:"Model temporarily unavailable"'
        events = asyncio.Queue()
        wrapped = arena._ArenaBridgeResponse(Response(), None, events)
        lines = [line async for line in wrapped.aiter_lines()]
        assert lines == ['a3:"Model temporarily unavailable"']
        assert await events.get() == {'arena_progress': 'stream_error:Model temporarily unavailable'}
        assert await events.get() == {'arena_progress': 'stream_summary:1 lines (a3=1)'}
    asyncio.run(run())


def test_real_qt_answer_reaches_pinned_bridge_and_client(monkeypatch, tmp_path):
    import importlib.util
    import json
    import base64
    import time
    if importlib.util.find_spec('PySide6') is None:
        pytest.skip('Requires Qt6 WebEngine')
    runtime = Path.home() / '.glossarion/autharena_proxy' / ('bridge-' + arena.REVISION)
    if not (runtime / 'bridge').exists():
        pytest.skip('Requires pinned bridge checkout')
    import uvicorn
    import httpx
    import importlib
    command = arena._qt_helper_command()
    original_browser = arena._open_qt_browser
    original_import = importlib.import_module
    async def discovery():
        pass
    def imported(name, *args, **kwargs):
        module = original_import(name, *args, **kwargs)
        if name.startswith('arena_slot_') and name.endswith('.src.main'):
            module.get_initial_data = discovery
        return module
    monkeypatch.setattr(importlib, 'import_module', imported)
    wire = 'ag:"thinking"\na0:"Hello world"\nad:{"finishReason":"stop"}\n'
    sent = []
    sent_cookies = []
    async def browser(*args, **kwargs):
        context = await original_browser(*args, **kwargs)
        async def route(r):
            if '/nextjs-api/stream/' in r.request.url:
                sent.append(r.request.post_data_json)
                sent_cookies.append((await r.request.all_headers()).get("cookie", ""))
                await r.fulfill(status=200, content_type='text/plain', body=wire)
            else:
                await r.fulfill(content_type='text/html', body='<script>window.grecaptcha={enterprise:{ready:f=>f(),execute:async()=>"test-v3"}};</script>')
        await context.context.route('**/*', route)
        return context
    async def serve(server, sockets):
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.config.app), base_url='http://test', headers={'Authorization':'Bearer diagnostic'}) as client:
                for number, slot in enumerate((0, 1, 0), 1):
                    response = await asyncio.wait_for(client.post('/v1/chat/completions', json={'account_slot':slot,'model':'test-model','messages':[{'role':'user','content':'Reply with OK'}],'stream':True}), 40)
                    assert response.status_code == 200, response.text
                    assert response.headers['X-Arena-Account-Slot'] == str(slot)
                    result = arena.consume_stream(response.text.splitlines(), log_stream=False)
                    assert result['content'] == 'Hello world', result
                    assert len(sent) == number
                    assert 'arena-auth-prod-v1=' + accounts[str(slot)]['token'] in sent_cookies[-1]
                    assert accounts[str(1-slot)]['token'] not in sent_cookies[-1]
        finally:
            for sock in sockets:
                sock.close()
    monkeypatch.setattr(arena, '_open_qt_browser', browser)
    monkeypatch.setattr(uvicorn.Server, 'serve', serve)
    monkeypatch.setenv('AUTHARENA_PROXY_DATA_DIR', str(tmp_path))
    accounts = {}
    for slot in (0, 1):
        email = f'test{slot}@example.test'
        token = 'base64-' + base64.urlsafe_b64encode(json.dumps({'user':{'id':str(slot),'email':email},'expires_at':time.time()+3600}).encode()).decode()
        accounts[str(slot)] = {'token':token,'cookies':[{'name':'arena-auth-prod-v1','value':token,'domain':'.arena.ai','path':'/'}],'email':email,'user_id':str(slot),'expires_at':time.time()+3600}
    arena._save('accounts.enc', accounts)
    arena._save('models.enc', {'models':[{'id':'test-id','publicName':'test-model','organization':'Test'}],'fetched_at':time.time()})
    monkeypatch.setattr(arena, '__file__', str(runtime/'autharena_proxy.py'))
    asyncio.run(arena._serve_worker('diagnostic', command))
