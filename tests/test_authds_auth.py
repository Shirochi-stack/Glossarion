import json
from contextlib import contextmanager

import pytest
import authds_auth as ds


def frame(value, event=None):
    return (f"event: {event}\n" if event else "") + "data: " + json.dumps(value) + "\n\n"


@pytest.fixture(autouse=True)
def reset():
    ds.reset_cancel()
    yield
    ds.reset_cancel()


@pytest.mark.parametrize("model,kind,thinking", [
    ("authds/flash", "default", False), ("authds/pro", "expert", False),
    ("authds/flash-thinking", "default", True), ("authds/pro-thinking", "expert", True),
])
def test_model(model, kind, thinking):
    assert ds.resolve_model(model) == (kind, thinking)


def test_unknown_model_and_images_rejected():
    with pytest.raises(ds.AuthDSError):
        ds.resolve_model("authds/fake-model")
    with pytest.raises(ds.AuthDSError):
        ds.build_prompt([{"content": [{"type": "image_url"}]}])


def test_roles_preserved():
    assert ds.build_prompt([{"role":"system", "content":"translate"},
                            {"role":"user", "content":"한국어"}]) == "[system]\ntranslate\n\n[user]\n한국어"


def test_split_sse_and_thinking_isolation():
    stream = frame({"v":{"response":{"fragments":[{"type":"THINK","content":"secret reasoning"}]}}})
    stream += frame({"o":"APPEND","v":" more reasoning"})
    stream += frame({"p":"response/fragments","o":"APPEND","v":[{"type":"RESPONSE","content":"Hello"}]})
    stream += frame({"p":"response/fragments/-1/content","o":"APPEND","v":" world"})
    stream += frame({}, "close")
    parser = ds.StreamParser()
    outputs=[]
    # One character at a time, including across CRLF boundaries.
    for character in stream.replace("\n", "\r\n"):
        outputs.extend(parser.feed(character))
    assert ''.join(outputs) == parser.content == "Hello world"
    assert parser.reasoning == "secret reasoning more reasoning"
    assert parser.finished


def test_nested_batch_and_usage():
    parser = ds.StreamParser()
    parser.feed(frame({"p":"response","o":"BATCH","v":[
        {"p":"fragments","o":"APPEND","v":[{"type":"RESPONSE","content":"answer"}]},
        {"p":"accumulated_token_usage","v":42}, {"p":"status","v":"FINISHED"}]}))
    assert parser.content == "answer" and parser.tokens == 42 and parser.finished


def test_error_never_returns_partial_answer():
    parser=ds.StreamParser()
    parser.feed(frame({"v":{"response":{"fragments":[{"type":"RESPONSE","content":"partial"}]}}}))
    with pytest.raises(ds.AuthDSError):
        parser.feed(frame({"error":{"message":"secret account data"}}, "error"))


class Page:
    def __init__(self, states):
        self.states=iter(states)
        self.start=None
    def evaluate(self, script):
        if 'const cfg =' in script:
            self.start=script
            return True
        if 'splice(0)' in script:
            return next(self.states)
        return None


def test_completion_uses_fresh_chat_and_omits_sampling(monkeypatch):
    monkeypatch.setenv('ENABLE_DEEPSEEK_THINKING','1')
    data = frame({"v":{"response":{"fragments":[{"type":"RESPONSE","content":"translated"}]}}}) + frame({}, "close")
    page=Page([{"chunks":[data], "done":True}])
    seen=[]
    result=ds._complete(page, "authds/pro-thinking", [{"role":"user","content":"source"}], 5, None, seen.append, lambda:seen.append("send"))
    assert result['content']=="translated" and seen==["send","translated"]
    assert 'parent_message_id:null' in page.start and "chat_session/create" in page.start
    assert '"model": "expert"' in page.start and '"thinking": true' in page.start
    assert 'temperature' not in page.start and 'max_tokens' not in page.start


@pytest.mark.parametrize('model', ['authds/flash', 'authds/pro', 'authds/flash-thinking',
                                  'authds/pro-thinking', 'authds/deepseek-reasoner'])
@pytest.mark.parametrize('enabled', [False, True])
def test_shared_deepseek_toggle_controls_web_deepthink(monkeypatch, model, enabled):
    monkeypatch.setenv('ENABLE_DEEPSEEK_THINKING', '1' if enabled else '0')
    data=frame({'v':{'response':{'fragments':[{'type':'RESPONSE','content':'answer'}]}}}) + frame({},'close')
    page=Page([{'chunks':[data],'done':True}])
    ds._complete(page,model,[{'content':'source'}],5,None,None,None)
    assert ('"thinking": true' in page.start) is enabled
    assert ('"thinking": false' in page.start) is (not enabled)


def test_deepthink_default_matches_shared_toggle_default(monkeypatch):
    monkeypatch.delenv('ENABLE_DEEPSEEK_THINKING',raising=False)
    data=frame({'v':{'response':{'fragments':[{'type':'RESPONSE','content':'answer'}]}}}) + frame({},'close')
    page=Page([{'chunks':[data],'done':True}])
    ds._complete(page,'authds/flash',[{'content':'source'}],5,None,None,None)
    assert '"thinking": true' in page.start


@pytest.mark.parametrize("chunks", [[], [frame({"v":{"response":{"fragments":[{"type":"RESPONSE","content":"partial"}]}}})],
    [frame({"v":{"response":{"fragments":[{"type":"THINK","content":"reasoning"}]}}}) + frame({}, "close")]])
def test_truncated_empty_and_thinking_only_fail(chunks):
    with pytest.raises(ds.AuthDSError):
        ds._complete(Page([{"chunks":chunks,"done":True}]), "flash", [{"content":"x"}], 5, None, None, None)


def test_expired_session_reopens_visible_browser(monkeypatch):
    visibility=[]
    scripts=[]
    @contextmanager
    def browser(**kwargs):
        visibility.append(kwargs['visible'])
        yield object()
    monkeypatch.setattr(ds, '_open_browser', browser)
    monkeypatch.setattr(ds, 'has_session', lambda:True)
    monkeypatch.setattr(ds, '_wait_login', lambda *a:None)
    class Profile:
        def __truediv__(self, other): return self
        def unlink(self, **kwargs): pass
    monkeypatch.setattr(ds, 'profile_dir', lambda:Profile())
    calls=[]
    def complete(*args):
        calls.append(1)
        if len(calls)==1:
            raise ds.AuthDSError("expired", "auth_error")
        return {'content':'ok'}
    # Avoid initial page-ready polling for this transport test.
    class BrowserPage:
        def evaluate(self, script):
            scripts.append(script)
            return True
    @contextmanager
    def ready_browser(**kwargs):
        visibility.append(kwargs['visible'])
        yield BrowserPage()
    monkeypatch.setattr(ds,'_open_browser',ready_browser)
    monkeypatch.setattr(ds,'_complete',complete)
    assert ds.send_chat_completion(messages=[{'content':'x'}],log_fn=lambda _:None)['content']=='ok'
    assert visibility==[False,True]
    assert any("removeItem('userToken')" in script for script in scripts)


def test_cancel_queued_work():
    ds.cancel_stream()
    with pytest.raises(ds.AuthDSError) as error:
        with ds._serialized(): pass
    assert error.value.error_type=='cancelled'
    assert ds._gate.acquire(blocking=False)
    ds._gate.release()


def test_mobile_rejected_before_process_spawn(monkeypatch):
    monkeypatch.setattr(ds.mobile_runtime,'subprocesses_available',lambda:False)
    monkeypatch.setattr(ds.subprocess,'Popen',lambda *a,**k:pytest.fail('must not launch'))
    with pytest.raises(ds.AuthDSError,match='desktop'):
        with ds._open_browser(): pass


def test_browser_script_protocol(tmp_path):
    import shutil
    import subprocess
    node=shutil.which('node')
    if not node:
        pytest.skip('Node is needed to execute the browser JavaScript test')
    config={'model':'expert','thinking':True,'prompt':'한국어 source',
            'worker':None,'fallbackWorker':ds.POW_WORKER_URL}
    setup=r'''
const assert = require('node:assert/strict');
global.window=global;
global.location={href:'https://chat.deepseek.com/'};
global.localStorage={getItem:()=>JSON.stringify({value:'test-token'})};
global.__glossarionDSWorkers=['https://fe-static.deepseek.com/chat/static/current-worker.js'];
const requests=[];
global.Worker=class {
 postMessage(input){assert.equal(input.type,'pow-challenge');
  queueMicrotask(()=>this.onmessage({data:{type:'pow-answer',answer:{
   algorithm:'DeepSeekHashV1',challenge:'c',salt:'s',signature:'sig',answer:7}}}));}
 terminate(){}
};
global.fetch=async (url, options={})=>{
 requests.push({url,options});
 if(url.endsWith('current-worker.js'))return new Response('worker');
 if(url.endsWith('create_pow_challenge'))return Response.json({code:0,data:{biz_code:0,biz_data:{
  challenge:{algorithm:'DeepSeekHashV1',challenge:'c',salt:'s',signature:'sig',difficulty:1,expire_at:9}}}});
 if(url.endsWith('chat_session/create'))return Response.json({code:0,data:{biz_code:0,biz_data:{chat_session:{id:'fresh'}}}});
 if(url.endsWith('chat/completion'))return new Response('data: {"v":{"response":{"fragments":[{"type":"RESPONSE","content":"answer"}]}}}\n\nevent: close\ndata: {}\n\n');
 throw Error('Unexpected endpoint');
};
'''
    checks=r'''
(async()=>{
 while(!window.__glossarionDS.done)await new Promise(r=>setTimeout(r,1));
 assert.equal(window.__glossarionDS.error,null);
 assert.ok(window.__glossarionDS.chunks.join('').includes('answer'));
 const completion=requests.find(r=>r.url.endsWith('chat/completion'));
 const body=JSON.parse(completion.options.body);
 assert.equal(body.chat_session_id,'fresh');assert.equal(body.parent_message_id,null);
 assert.equal(body.model_type,'expert');assert.equal(body.thinking_enabled,true);
 assert.equal(body.search_enabled,false);assert.equal(body.prompt,'한국어 source');
 assert.equal(completion.options.headers.authorization,'Bearer test-token');
 const proof=JSON.parse(Buffer.from(completion.options.headers['x-ds-pow-response'],'base64').toString());
 assert.equal(proof.answer,7);assert.equal(proof.target_path,'/api/v0/chat/completion');
 assert.equal('temperature' in body,false);assert.equal('max_tokens' in body,false);
 assert.equal(requests.filter(r=>r.url.endsWith('chat_session/create')).length,1);
})().catch(e=>{console.error(e);process.exitCode=1;});
'''
    script=tmp_path/'authds_protocol.cjs'
    script.write_text(setup+ds._START_SCRIPT.replace('__CONFIG__',json.dumps(config))+';\n'+checks,encoding='utf-8')
    result=subprocess.run([node,str(script)],capture_output=True,text=True,timeout=15)
    assert result.returncode==0, result.stderr


def test_unified_routing():
    from unified_api_client import UnifiedClient
    assert UnifiedClient._provider_from_model_name('authds/pro')=='authds'
    assert not UnifiedClient._model_needs_api_key('authds/flash')
    assert UnifiedClient._key_data_is_usable({'model':'authds/flash','api_key':''})


def test_client_initializes_without_api_key(monkeypatch):
    from unified_api_client import UnifiedClient
    monkeypatch.delenv('CUSTOM_OPENAI_PREFIX_ROUTES',raising=False)
    client=UnifiedClient.for_key_test(api_key='',model='authds/flash')
    assert client.client_type=='authds'
    assert client._get_actual_provider()=='authds'


def test_stop_lifecycle_resets_between_runs():
    from unified_api_client import set_stop_flag
    set_stop_flag(True)
    assert ds._cancel.is_set()
    set_stop_flag(False)
    assert not ds._cancel.is_set()


def test_unified_transport_dispatch(monkeypatch):
    from unified_api_client import UnifiedClient, UnifiedClientError
    from types import SimpleNamespace
    calls=[]
    tls=SimpleNamespace()
    client=object.__new__(UnifiedClient)
    client.request_timeout=60
    monkeypatch.setattr(client,'_get_thread_local_client',lambda:tls)
    monkeypatch.setattr(client,'_get_active_request_model',lambda:'authds/flash')
    monkeypatch.setattr(client,'_is_stop_requested',lambda:False)
    monkeypatch.setattr(client,'_streaming_enabled',lambda:False)
    def send(**kwargs):
        calls.append(kwargs)
        kwargs['before_send_callback']()
        return {'content':'answer','finish_reason':'stop','usage':None}
    monkeypatch.setattr(ds,'send_chat_completion',send)
    result=client._send_authds([{'content':'source'}],.5,8192,'chapter')
    assert result.content=='answer'
    assert calls[0]['model']=='authds/flash'
    assert 'temperature' not in calls[0] and 'max_tokens' not in calls[0]
    def rejected(**kwargs):
        raise ds.AuthDSError('limited','rate_limit')
    monkeypatch.setattr(ds,'send_chat_completion',rejected)
    with pytest.raises(UnifiedClientError) as exc:
        client._send_authds([{'content':'source'}],.5,8192,'chapter')
    assert exc.value.error_type=='rate_limit'
