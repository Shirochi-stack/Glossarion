import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import uuid

import pytest

import autharena as arena


MODEL_ID = "01a07d42-938f-7267-9398-4529e857491c"


@pytest.fixture(autouse=True)
def clean_cancel(monkeypatch, tmp_path):
    monkeypatch.delenv("TRANSLATION_CANCELLED", raising=False)
    arena._cancel_event.clear()
    monkeypatch.setattr(arena, "_profile_path", lambda account: tmp_path / str(account))
    yield
    arena.cancel_stream()
    arena._cancel_event.clear()


def model(**overrides):
    return {"id": MODEL_ID, "name": "internal-name", "publicName": "public-name",
            "displayName": "Public Model", "provider": "provider", "organization": "organization",
            "userSelectable": True,
            "capabilities": {"inputCapabilities": {"text": True}, "outputCapabilities": {"text": True}},
            **overrides}


def flight_page(models):
    data = '8:["$",{},{"initialModels":' + json.dumps(models) + "}]"
    # The real array can span multiple flight scripts. Decode first, then join.
    point = len(data) // 2
    return "".join("<script>self.__next_f.push(" + json.dumps([1, part]) + ")</script>"
                   for part in (data[:point], data[point:]))


def test_catalog_flight_chunks_selectability_aliases_and_stable_duplicates():
    entries = [
        model(name=None, displayName='Model "Quoted" \\ path'),
        model(displayName='Model "Quoted" \\ path', id=str(uuid.uuid4())),
        model(name="hidden", userSelectable=False),
        model(name="image", capabilities={"inputCapabilities": {"text": True},
                                         "outputCapabilities": {"image": True}}),
        model(name="retired", provider=None),
        model(name="missing-org", organization=None),
        model(displayName="bad-id", id="not-a-uuid"),
        model(displayName=None, publicName="public-fallback", id=str(uuid.uuid4())),
        model(displayName="Max", publicName="Max", name="boss-bandit", id=str(uuid.uuid4())),
    ]
    actual = arena._parse_model_catalog(flight_page(entries))
    assert set(actual) == {'Model "Quoted" \\ path', "public-fallback", "max"}
    assert actual['Model "Quoted" \\ path'] == MODEL_ID


@pytest.mark.parametrize("page", ["<html>Access denied</html>", '{"initialModels":[]}', '{"initialModels":null}'])
def test_catalog_missing_or_empty_is_an_error(page):
    with pytest.raises(RuntimeError, match="catalog"):
        arena._parse_model_catalog(page)


def test_model_polling_is_http_only_and_closes_response(monkeypatch):
    class Response:
        status_code = 200
        text = flight_page([model()])
        closed = False

        def close(self):
            self.closed = True

    response = Response()
    seen = []
    monkeypatch.setattr(arena.requests, "get", lambda url, **kw: seen.append((url, kw)) or response)
    monkeypatch.setattr(arena, "_run_browser_helper", lambda *a, **kw: pytest.fail("poll opened browser"))
    assert arena.fetch_available_models(timeout=7, account_id=2) == ["Public Model"]
    assert seen[0][0] == arena.CATALOG_URL
    assert seen[0][1]["timeout"] == 7
    assert "Cookie" not in seen[0][1]["headers"]
    assert response.closed
    response.status_code = 403
    with pytest.raises(RuntimeError, match="HTTP 403"):
        arena.fetch_available_models()


def test_payload_fresh_uuid7_identifiers_and_chat_modality():
    first = arena._build_payload(MODEL_ID, "hello")
    second = arena._build_payload(MODEL_ID, "hello")
    ids = [payload[key] for payload in (first, second)
           for key in ("id", "userMessageId", "modelAMessageId")]
    assert len(set(ids)) == 6
    assert all(uuid.UUID(value).version == 7 for value in ids)
    assert first["modality"] == "chat"
    assert first["mode"] == "direct"
    assert first["userMessage"]["content"] == "hello"
    assert "recaptchaV3Token" not in first


def test_messages_preserve_roles_and_text_without_silent_tools_or_images():
    assert arena._render_messages([{"role": "user", "content": "hello"}]) == "hello"
    messages = [{"role": "system", "content": "Translate faithfully."},
                {"role": "assistant", "name": "previous", "content": 'Answer "A"'},
                {"role": "user", "content": [{"type": "text", "text": "Line1"},
                                           {"type": "input_text", "text": "Line2"}]}]
    rendered = arena._render_messages(messages)
    conversation = json.loads(rendered.split("BEGIN CONVERSATION JSON\n", 1)[1].split("\nEND CONVERSATION JSON")[0])
    assert conversation == [messages[0], messages[1], {"role": "user", "content": "Line1\nLine2"}]
    for message in ({"role": "tool", "content": "secret"},
                    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]},
                    {"role": "assistant", "content": "", "tool_calls": [{"id": "x"}]}):
        with pytest.raises(ValueError, match="support"):
            arena._render_messages([message])
    with pytest.raises(ValueError, match="nonblank"):
        arena._render_messages([{"role": "user", "content": " \n"}])


def test_explicit_unknown_terminal_is_preserved():
    parser = arena._ArenaStreamParser()
    parser.feed('a0:"answer"\nad:{"finishReason":"unknown"}\n')
    assert parser.finish()["finish_reason"] == "unknown"


def test_stream_split_utf8_reasoning_metadata_usage_and_length_finish():
    events = []
    parser = arena._ArenaStreamParser(lambda *event: events.append(event))
    data = ('af:{"messageId":"m"}\nag:"Think…"\na0:"你好"\n'
            'a2:[{"type":"routed_model","organization":"openai"}]\n'
            'ad:{"finishReason":"length","usage":{"promptTokens":12,"completionTokens":3}}').encode()
    for byte in data:
        parser.feed(bytes([byte]))
    assert parser.finish() == {"content": "你好", "reasoning_content": "Think…",
                               "finish_reason": "length", "finish_reason_explicit": True,
                               "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}}
    assert events == [("reasoning", "Think…"), ("content", "你好")]


@pytest.mark.parametrize("data, match", [
    (b'a0:"partial"\n', "without a terminal"),
    (b'a0:"cut off', "malformed or truncated"),
    (b'a0:"\xe4', "truncated UTF-8"),
    (b'a3:"model overloaded"\n', "model overloaded"),
    (b'ad:{"finishReason":"error"}\n', "generation failed"),
    (b'ad:{"finishReason":null}\n', "finishReason"),
    (b'b0:"other model"\n', "second model"),
    (b'a2:[{"type":"image"}]\n', "nontext"),
    (b'<html>Blocked</html>', "unrecognized"),
    (b'ad:{"finishReason":"stop"}\na0:"late"\n', "after the terminal"),
])
def test_stream_errors_never_return_partial_success(data, match):
    with pytest.raises(RuntimeError, match=match):
        parser = arena._ArenaStreamParser()
        parser.feed(data)
        parser.finish()


def test_cancel_env_and_graceful_callback_semantics(monkeypatch):
    monkeypatch.setenv("TRANSLATION_CANCELLED", "1")
    assert arena._cancelled()
    assert not arena._cancelled(lambda: False)
    assert arena._cancelled(lambda: True)
    arena._cancel_event.set()
    arena.reset_cancel()
    assert arena._cancel_event.is_set()


def test_numbered_prefix_validation():
    assert arena._normalize_model("AuthArena2/gpt-example") == "gpt-example"
    assert arena._model_account("AuthArena2/gpt-example", 0) == 2
    with pytest.raises(ValueError, match="conflicts"):
        arena._model_account("autharena2/model", 3)
    for invalid in ("../test", -1, True):
        with pytest.raises(ValueError):
            arena._account_number(invalid)


def test_waiting_for_profile_is_cancellable(tmp_path):
    ready = threading.Event()
    release = threading.Event()

    def owner():
        with arena._profile_gate(0, time.monotonic() + 5):
            ready.set()
            release.wait(3)

    thread = threading.Thread(target=owner)
    thread.start()
    assert ready.wait(2)
    try:
        with pytest.raises(RuntimeError, match="cancelled"):
            with arena._profile_gate(0, time.monotonic() + 3, lambda: True):
                pytest.fail("concurrent profile use")
    finally:
        release.set()
        thread.join(2)


def child_script(monkeypatch, tmp_path, source):
    script = tmp_path / "helper.py"
    script.write_text(
        "import json,sys,time\n"
        "config=json.loads(sys.stdin.readline())\n"
        "def emit(event, **kw):\n"
        " print(json.dumps(dict(autharena=1,event=event,**kw)),flush=True)\n" + source,
        encoding="utf-8")
    monkeypatch.setattr(arena, "_helper_command", lambda: [sys.executable, str(script)])


def test_helper_drains_stderr_and_calls_ready_callback_before_dispatch(monkeypatch, tmp_path):
    marker = tmp_path / "ready"
    child_script(monkeypatch, tmp_path,
                 "sys.stderr.write('diagnostic\\n'*10000);sys.stderr.flush()\n"
                 "emit('ready')\n"
                 "command=json.loads(sys.stdin.readline())\n"
                 "from pathlib import Path\n"
                 "assert Path(config['marker']).exists()\n"
                 "assert command['command']=='dispatch'\n"
                 "emit('chunk',data='a0:\"ok\"\\nad:{\"finishReason\":\"stop\"}\\n')\n"
                 "emit('done')\n")
    result = arena._run_browser_helper({"marker": str(marker)}, time.monotonic() + 8,
                                       before_send_callback=lambda: marker.write_text("ready"))
    assert result["content"] == "ok"
    assert not arena._active_helpers


def test_helper_ready_callback_error_never_dispatches(monkeypatch, tmp_path):
    sent = tmp_path / "sent"
    child_script(monkeypatch, tmp_path,
                 "emit('ready')\n"
                 "if sys.stdin.readline():\n"
                 " from pathlib import Path\n"
                 " Path(config['sent']).touch()\n")

    def failed_callback():
        raise RuntimeError("rate limiter cancelled")

    with pytest.raises(RuntimeError, match="rate limiter cancelled"):
        arena._run_browser_helper({"sent": str(sent)}, time.monotonic() + 5, before_send_callback=failed_callback)
    assert not sent.exists()
    assert not arena._active_helpers


def test_helper_error_preserves_http_metadata(monkeypatch, tmp_path):
    child_script(monkeypatch, tmp_path, "emit('error',message='rate limited',status_code=429,retry_after='60')\n")
    with pytest.raises(arena.AuthArenaError) as caught:
        arena._run_browser_helper({}, time.monotonic() + 5)
    assert caught.value.status_code == 429
    assert caught.value.retry_after == "60"


@pytest.mark.parametrize("kind,exception", [("timeout", TimeoutError), ("configuration", ImportError)])
def test_helper_error_preserves_error_type(monkeypatch, tmp_path, kind, exception):
    child_script(monkeypatch, tmp_path, f"emit('error',message='classified failure',error_type={kind!r})\n")
    with pytest.raises(exception, match="classified failure"):
        arena._run_browser_helper({}, time.monotonic() + 5)


def test_helper_timeout_and_abrupt_exit_are_not_retried(monkeypatch, tmp_path):
    child_script(monkeypatch, tmp_path, "time.sleep(30)\n")
    with pytest.raises(TimeoutError):
        arena._run_browser_helper({}, time.monotonic() + 0.2)
    assert not arena._active_helpers
    child_script(monkeypatch, tmp_path, "emit('ready')\nsys.stdin.readline()\nsys.exit(2)\n")
    with pytest.raises(RuntimeError, match="after dispatch; the request was not retried"):
        arena._run_browser_helper({}, time.monotonic() + 5)


def test_helper_rejection_rearms_graceful_cancel(monkeypatch, tmp_path):
    child_script(monkeypatch, tmp_path,
                 "emit('ready')\nsys.stdin.readline()\nemit('rejected')\ntime.sleep(30)\n")
    stopped = []
    with pytest.raises(RuntimeError, match="cancelled"):
        arena._run_browser_helper({}, time.monotonic() + 5, cancel_check=lambda: bool(stopped),
                                  after_rejection_callback=lambda: stopped.append(True))
    assert stopped == [True]


def test_send_logs_options_once_and_uses_new_payloads(monkeypatch):
    configs = []
    logs = []
    callbacks = []
    arena._warned_options.clear()

    def fake_helper(config, deadline, **kwargs):
        configs.append(config)
        kwargs["before_send_callback"]()
        kwargs["on_chunk"]("content", "answer")
        return {"content": "answer", "reasoning_content": "", "finish_reason": "stop",
                "finish_reason_explicit": True, "usage": {}}

    monkeypatch.setattr(arena, "_run_browser_helper", fake_helper)
    for _ in range(2):
        arena.send_chat_completion(messages=[{"role": "user", "content": "hello"}], model="AuthArena2/test",
                                   max_tokens=100, log_fn=logs.append, progress_label="API call in progress",
                                   before_send_callback=lambda: callbacks.append("send"))
    assert configs[0]["payload"]["id"] != configs[1]["payload"]["id"]
    assert configs[0]["account_id"] == 2
    assert configs[0]["model"] == "test"
    assert sum("unsupported options" in log for log in logs) == 1
    assert logs.count("API call in progress") == 2
    assert logs.count("answer") == 2
    assert callbacks == ["send", "send"]


def test_cli_messages_stdin_json_and_no_browser_for_list(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["autharena.py", "--messages", "-", "--system", "Translate", "--json", "--quiet"])
    monkeypatch.setattr(sys, "stdin", io.StringIO('[{"role":"user","content":"你好"}]'))
    seen = []
    monkeypatch.setattr(arena, "send_chat_completion", lambda **kwargs: seen.append(kwargs) or {"content": "Hello"})
    assert arena._main() == 0
    assert json.loads(capsys.readouterr().out)["content"] == "Hello"
    assert seen[0]["messages"] == [{"role": "system", "content": "Translate"}, {"role": "user", "content": "你好"}]
    monkeypatch.setattr(sys, "argv", ["autharena.py", "--list-models", "--json"])
    monkeypatch.setattr(arena, "fetch_available_models", lambda **kwargs: ["model-a", "model-b"])
    assert arena._main() == 0
    assert json.loads(capsys.readouterr().out) == ["model-a", "model-b"]


def test_cli_invalid_inputs_and_frozen_command(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["autharena.py", "--prompt", "text", "--prompt-file", "file"])
    assert arena._main() == 1
    assert "not both" in capsys.readouterr().err
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert arena._helper_command() == [sys.executable, "--autharena-helper"]


@pytest.mark.skipif(not shutil.which("node"), reason="Node optional for browser JavaScript simulation")
@pytest.mark.parametrize("scenario", ["success", "anonymous", "terms", "transport", "rate", "challenge", "alias", "hydrated"])
def test_browser_script_runs_normal_protocol_without_exporting_tokens(tmp_path, scenario):
    entries = [model(displayName='Name "quoted" \\ backslash', publicName="public-model", name="internal-model")]
    wanted = "internal-model" if scenario == "alias" else entries[0]["displayName"]
    prompt = "__MODEL__ __PAYLOAD__ __TIMEOUT_MS__ \u2028 hello"
    user = None if scenario == "anonymous" else {"email": "user@example.test", "touConsentTimestamp": None if scenario == "terms" else "2026-01-01"}
    flight = json.dumps({"initialModels": entries, "user": user})
    source = arena._prepare_script(arena._build_payload(None, prompt), wanted, 3)
    script = r'''
const vm = require('vm');
const fs = require('fs');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
global.window = global;
global.location = {origin:'https://arena.ai'};
window.__next_f = [[1,input.flight]];
const requests = [];
const actions = [];
let challenge;
global.document = {
  scripts:[{src:'https://www.google.com/recaptcha/enterprise.js?render=public-v3-key'}],
  getElementById:()=>null,
  createElement:()=>({style:{}, append(){}, remove(){}}),
  body:{append(){}}
};
if(input.scenario==='hydrated') {
  window.__next_f=[];
  const middle=Math.floor(input.flight.length/2);
  for(const part of [input.flight.slice(0,middle),input.flight.slice(middle)]) {
    document.scripts.push({textContent:'self.__next_f.push('+JSON.stringify([1,part])+')'});
  }
}
window.grecaptcha = {enterprise:{
  ready:cb=>cb(),
  execute:async(key,args)=>{actions.push(args.action);return 'PRIVATE-V3-TOKEN';},
  render:(widget,opts)=>{challenge=opts.callback;}
}};
global.fetch = async(url, options)=>{
  requests.push({url, body:JSON.parse(options.body),credentials:options.credentials});
  if(input.scenario==='transport') throw Error('disconnected');
  if(input.scenario==='rate') return {ok:false,status:429,headers:{get:()=> '30'},json:async()=>({error:'Rate limit exceeded'})};
  if(input.scenario==='challenge' && requests.length===1) return {ok:false,status:403,headers:{get:()=>null},json:async()=>({error:'reCAPTCHA validation failed'})};
  let read=false;
  return {ok:true,status:200,headers:{get:()=> 'text/plain'},body:{getReader:()=>({
    read:async()=>{
      if(read)return {done:true};
      read=true;
      return {done:false,value:new TextEncoder().encode('ag:"thinking"\na0:"answer"\nad:{"finishReason":"stop"}\n')};
    }
  })}};
};
(async()=>{
  vm.runInThisContext(input.source);
  await new Promise(resolve=>setTimeout(resolve,50));
  let state=window.__glossarionArena;
  if(state.phase==='ready') await state.dispatch();
  if(challenge){challenge('PRIVATE-V2-TOKEN');await state.dispatch();}
  console.log(JSON.stringify({requests,actions,events:state.events,phase:state.phase}));
})().catch(error=>{console.error(error);process.exit(1)});
'''
    result = subprocess.run([shutil.which("node"), "-e", script], input=json.dumps({
        "flight": flight, "source": source, "scenario": scenario}), text=True,
        capture_output=True, timeout=10, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    actual = json.loads(result.stdout)
    assert "PRIVATE" not in json.dumps(actual["events"])
    if scenario in ("anonymous", "terms"):
        assert actual["requests"] == []
        assert actual["events"][-1]["event"] == "action"
        return
    assert actual["actions"] == ["chat_submit"]
    assert all(item["url"] == arena.CREATE_EVALUATION_PATH and item["credentials"] == "same-origin"
               and item["body"]["userMessage"]["content"] == prompt for item in actual["requests"])
    if scenario == "transport":
        assert len(actual["requests"]) == 1
        assert actual["events"][-1]["event"] == "error"
        assert "not retried" in actual["events"][-1]["message"]
    elif scenario == "rate":
        assert actual["events"][-1]["status_code"] == 429
        assert actual["events"][-1]["retry_after"] == "30"
    else:
        assert actual["phase"] == "done"
        parser = arena._ArenaStreamParser()
        for event in actual["events"]:
            if event["event"] == "chunk":
                parser.feed(event["data"])
        assert parser.finish()["content"] == "answer"
        if scenario == "challenge":
            assert len(actual["requests"]) == 2
            assert actual["requests"][1]["body"]["recaptchaV2Token"] == "PRIVATE-V2-TOKEN"
            assert "recaptchaV3Token" not in actual["requests"][1]["body"]
            assert any(event["event"] == "rejected" for event in actual["events"])
