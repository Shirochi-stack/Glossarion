import json

import pytest
import authchatgpt_auth as gpt


def frame(value, event=None):
    return (f"event: {event}\n" if event else "") + "data: " + (value if isinstance(value, str) else json.dumps(value)) + "\n\n"


def message(role, parts=None, content_type="text", text=None, **extra):
    content = {"content_type": content_type}
    if parts is not None:
        content["parts"] = parts
    if text is not None:
        content["content"] = text
    return {"message": {"id": role, "author": {"role": role}, "content": content, "recipient": "all",
                        "metadata": {"resolved_model_slug": "gpt-6"}, **extra}}


@pytest.fixture(autouse=True)
def reset():
    gpt.reset_cancel()
    yield
    gpt.reset_cancel()


@pytest.mark.parametrize("model,slug", [("authchatgpt/auto", "auto"), ("authchatgpt/GPT-6-Thinking", "gpt-6-thinking"),
                                        ("authchatgpt/", "auto"), ("authchatgpt/gpt-5.5", "gpt-5.5")])
def test_model_slug(model, slug):
    assert gpt.resolve_model(model) == slug


def test_bad_slug_and_images_rejected():
    with pytest.raises(gpt.AuthChatGPTError):
        gpt.resolve_model("authchatgpt/bad slug")
    with pytest.raises(gpt.AuthChatGPTError, match="AuthChatGPT"):
        gpt.build_prompt([{"content": [{"type": "image_url"}]}])


def test_markup_back_to_markdown():
    assert gpt.clean_markup("<escape><p>Hi</p></escape>") == "<p>Hi</p>"
    assert gpt.clean_markup('<CodeBlock language="html"><p>Hi</p></CodeBlock>') == "```html\n<p>Hi</p>\n```"


def test_stream_keeps_only_the_visible_answer():
    stream = (frame('"v1"', "delta_encoding")
              + frame({"p": "", "o": "add", "v": message("user", ["source"]), "c": 0}, "delta")
              + frame({"v": message("system", [""])}, "delta")
              + frame({"v": message("assistant", content_type="reasoning_recap", text="Worked")}, "delta")
              + frame({"v": message("assistant", ["<escape><p>Chap"], status="in_progress")}, "delta")
              + frame({"o": "patch", "v": [{"p": "/message/content/parts/0", "o": "append", "v": "ter</p>"},
                                           {"p": "/message/metadata/x", "o": "append", "v": {"y": 1}}]}, "delta")
              + frame({"p": "/message/content/parts/0", "o": "append", "v": "<p>One"}, "delta")
              + frame({"v": "</p></escape>"}, "delta")
              + frame({"v": [{"p": "/message/status", "o": "replace", "v": "finished_successfully"}]}, "delta")
              + frame({"type": "message_stream_complete", "conversation_id": "c"}) + frame("[DONE]"))
    parser = gpt.StreamParser()
    deltas = []
    for i in range(0, len(stream), 7):  # frames split anywhere
        deltas += parser.feed(stream[i:i + 7])
    assert parser.finished and parser.model == "gpt-6"
    assert parser.content == "<p>Chapter</p><p>One</p>"
    assert "".join(deltas) == "<escape><p>Chapter</p><p>One</p></escape>"
    assert "source" not in parser.content and "Worked" not in parser.content


def test_reply_continued_on_resume_stream():
    first = frame('"v1"', "delta_encoding") + frame({"v": message("user", ["x"])}, "delta") + "\n\n"
    resumed = (frame('"v1"', "delta_encoding") + frame({"v": message("assistant", ["Hel"])}, "delta")
               + frame({"p": "/message/content/parts/0", "o": "append", "v": "lo"}, "delta") + frame("[DONE]"))
    parser = gpt.StreamParser()
    parser.feed(first)
    assert not parser.finished
    parser.feed(resumed)
    assert parser.finished and parser.content == "Hello"


def test_stream_error_frame():
    parser = gpt.StreamParser()
    parser.feed(frame({"message": None, "error": "Something went wrong"}))
    assert parser.error == "Something went wrong"


class Page:
    """A tab whose composer accepts the message at once; polls replay ``states``."""

    def __init__(self, states):
        self.states = list(states)
        self.calls = []

    def call(self, method, params=None, timeout=20):
        self.calls.append(method)
        return {}

    def evaluate(self, script):
        if script == gpt._COMPOSER:
            return {"text": ".", "send": True, "requested": True}
        if script == gpt._POLL:
            return self.states.pop(0) if len(self.states) > 1 else self.states[0]
        return True


def poll(chunks=(), **state):
    return {"chunks": list(chunks), "streams": 1, "open": 0, "idle": 0, "status": 200, "error": None,
            "sentinel": 0, **state}


def test_complete_returns_answer_and_classifies_errors():
    done = frame({"v": message("assistant", ["ok"])}, "delta") + frame("[DONE]")
    page = Page([poll([done])])
    assert gpt._complete(page, 5, None, None, None)["content"] == "ok"
    for status, kind in ((429, "rate_limit"), (401, "auth_error"), (403, "access_denied")):
        page = Page([poll(error=f"HTTP {status}", status=status, streams=0)])
        with pytest.raises(gpt.AuthChatGPTError) as error:
            gpt._complete(page, 5, None, None, None)
        assert error.value.error_type == kind


def test_cancel_stops_queued_work():
    gpt.cancel_stream()
    with pytest.raises(gpt.AuthChatGPTError) as error:
        gpt._check()
    assert error.value.error_type == "cancelled"


def test_logged_out_without_token_and_never_prompts(monkeypatch):
    class Store:
        has_tokens = False

        def get_valid_access_token(self, auto_login=True):
            pytest.fail("no login may be attempted")
    monkeypatch.setattr(gpt, "_token_store", lambda: Store())
    assert gpt._access_token(lambda _: None) is None


def test_refresh_failure_falls_back_to_logged_out(monkeypatch):
    logs = []

    class Store:
        has_tokens = True

        def get_valid_access_token(self, auto_login=True):
            assert auto_login is False
            raise RuntimeError("refresh failed")
    monkeypatch.setattr(gpt, "_token_store", lambda: Store())
    assert gpt._access_token(logs.append) is None
    assert logs and "logged out" in logs[0]


def test_rejected_token_retries_logged_out(monkeypatch):
    from contextlib import contextmanager
    configs = []

    @contextmanager
    def tab(config, cancel_check=None):
        configs.append(dict(config))
        yield object()

    def complete(page, *args, **kwargs):
        if configs[-1]["token"]:
            raise gpt.AuthChatGPTError("HTTP 401", "auth_error")
        return {"content": "ok"}
    monkeypatch.setattr(gpt, "_access_token", lambda log: "stale")
    monkeypatch.setattr(gpt, "_refreshed_token", lambda rejected: None)
    monkeypatch.setattr(gpt, "_open_tab", tab)
    monkeypatch.setattr(gpt, "_complete", complete)
    assert gpt.send_chat_completion(messages=[{"content": "x"}], log_fn=lambda _: None)["content"] == "ok"
    assert [c["token"] for c in configs] == ["stale", None]
    assert configs[0]["temporary"] is True and configs[1]["temporary"] is False


def test_mobile_rejected_before_process_spawn(monkeypatch):
    monkeypatch.setattr(gpt.mobile_runtime, "subprocesses_available", lambda: False)
    monkeypatch.setattr(gpt.cdp_browser.subprocess, "Popen", lambda *a, **k: pytest.fail("must not launch"))
    with pytest.raises(gpt.AuthChatGPTError, match="desktop"):
        with gpt._open_tab({}):
            pass


def test_unified_routing_and_dispatch(monkeypatch):
    from types import SimpleNamespace
    from unified_api_client import UnifiedClient, UnifiedClientError
    assert UnifiedClient._provider_from_model_name("authchatgpt/auto") == "authchatgpt"
    assert not UnifiedClient._model_needs_api_key("authchatgpt/gpt-6")
    monkeypatch.delenv("CUSTOM_OPENAI_PREFIX_ROUTES", raising=False)
    assert UnifiedClient.for_key_test(api_key="", model="authchatgpt/auto").client_type == "authchatgpt"

    client = object.__new__(UnifiedClient)
    client.request_timeout = 60
    tls = SimpleNamespace()
    monkeypatch.setattr(client, "_get_thread_local_client", lambda: tls)
    monkeypatch.setattr(client, "_get_active_request_model", lambda: "authchatgpt/auto")
    monkeypatch.setattr(client, "_is_stop_requested", lambda: False)
    monkeypatch.setattr(client, "_streaming_enabled", lambda: False)

    def send(**kwargs):
        kwargs["before_send_callback"]()
        return {"content": "answer", "finish_reason": "stop", "usage": None}
    monkeypatch.setattr(gpt, "send_chat_completion", send)
    assert client._send_authchatgpt([{"content": "x"}], .5, 8192, "chapter").content == "answer"

    def limited(**kwargs):
        raise gpt.AuthChatGPTError("limited", "rate_limit")
    monkeypatch.setattr(gpt, "send_chat_completion", limited)
    with pytest.raises(UnifiedClientError) as exc:
        client._send_authchatgpt([{"content": "x"}], .5, 8192, "chapter")
    assert exc.value.error_type == "rate_limit"


def test_stop_flag_reaches_route():
    from unified_api_client import set_stop_flag
    set_stop_flag(True)
    assert gpt._cancel.is_set()
    set_stop_flag(False)
    assert not gpt._cancel.is_set()


def test_login_button_and_runs_on_mobile():
    import settings_rules
    assert settings_rules.authgpt_login_needed("authchatgpt/auto", {})
    assert settings_rules.excluded_route_reason("authchatgpt/auto", "mobile") is None


def token(email="reader@example.com", plan="pro"):
    import base64
    claims = {"https://api.openai.com/profile": {"email": email},
              "https://api.openai.com/auth": {"chatgpt_plan_type": plan}}
    body = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return f"e30.{body}.sig"


def test_login_identity_names_the_plan():
    assert gpt.login_identity(token(plan="prolite")) == ("reader@example.com", "Pro")  # ChatGPT shows prolite as Pro
    assert gpt.login_identity(token(plan="plus")) == ("reader@example.com", "Plus")


@pytest.fixture
def fresh_run(monkeypatch):
    monkeypatch.setenv("GLOSSARION_RUN_ID", "run-a")
    monkeypatch.setattr(gpt, "_announced", {})
    monkeypatch.setattr(gpt, "_rejected", set())


def fake_tabs(monkeypatch, complete):
    from contextlib import contextmanager
    configs = []

    @contextmanager
    def tab(config, cancel_check=None):
        configs.append(dict(config))
        yield object()
    monkeypatch.setattr(gpt, "_open_tab", tab)
    monkeypatch.setattr(gpt, "_complete", lambda page, *a, **k: complete(configs[-1]))
    return configs


def test_login_line_once_per_run(monkeypatch, fresh_run):
    logs = []
    fake_tabs(monkeypatch, lambda config: {"content": "ok"})
    monkeypatch.setattr(gpt, "_access_token", lambda log: token())
    for _ in range(3):
        gpt.send_chat_completion(messages=[{"content": "x"}], log_fn=logs.append)
    assert logs == ["🔐 AuthChatGPT: Logged in to ChatGPT as reader@example.com · Pro plan."]
    monkeypatch.setenv("GLOSSARION_RUN_ID", "run-b")
    gpt.send_chat_completion(messages=[{"content": "x"}], log_fn=logs.append)
    assert len(logs) == 2


def test_no_login_line_once_per_run(monkeypatch, fresh_run):
    class Store:
        has_tokens = False
    logs = []
    fake_tabs(monkeypatch, lambda config: {"content": "ok"})
    monkeypatch.setattr(gpt, "_token_store", lambda: Store())
    for _ in range(2):
        gpt.send_chat_completion(messages=[{"content": "x"}], log_fn=logs.append)
    assert len(logs) == 1 and "No ChatGPT login found" in logs[0]


def test_rejected_login_is_skipped_for_the_rest_of_the_run(monkeypatch, fresh_run):
    logs = []

    def complete(config):
        if config["token"]:
            raise gpt.AuthChatGPTError("AuthChatGPT: HTTP 403: Unusual activity has been detected", "access_denied")
        return {"content": "ok"}
    configs = fake_tabs(monkeypatch, complete)
    stale = token()

    class Store:
        has_tokens = True

        def get_valid_access_token(self, auto_login=True):
            return stale
    monkeypatch.setattr(gpt, "_token_store", lambda: Store())
    monkeypatch.setattr(gpt, "_refreshed_token", lambda rejected: pytest.fail("403 is not a refresh case"))
    for _ in range(3):
        assert gpt.send_chat_completion(messages=[{"content": "x"}], log_fn=logs.append)["content"] == "ok"
    assert [bool(c["token"]) for c in configs] == [True, False, False, False]
    assert len(logs) == 1 and "did not accept the saved login" in logs[0]


def test_a_limit_with_a_good_login_is_not_retried_logged_out(monkeypatch, fresh_run):
    def complete(config):
        raise gpt.AuthChatGPTError("AuthChatGPT: HTTP 429", "rate_limit")
    configs = fake_tabs(monkeypatch, complete)
    monkeypatch.setattr(gpt, "_access_token", lambda log: token())
    with pytest.raises(gpt.AuthChatGPTError):
        gpt.send_chat_completion(messages=[{"content": "x"}], log_fn=lambda _: None)
    assert len(configs) == 1


class DriverPage:
    """The mobile page contract only: load / run_js / close."""

    def __init__(self):
        self.loaded, self.scripts, self.closed, self.cookie = [], [], False, False

    def load(self, url):
        self.loaded.append(url)

    def run_js(self, script, js_timeout_ms=15000):
        self.scripts.append(script)
        if script == gpt._APP_COOKIE_JS:
            self.cookie = True
        if "next-auth.session-token=') >= 0" in script:
            return self.cookie
        return True

    def close(self):
        self.closed = True


def mobile_driver(monkeypatch, page):
    class Driver:
        def open_page(self, *, owner, cancel_check=None, **_):
            assert owner == gpt.BROWSER_OWNER
            return page
    monkeypatch.setattr(gpt.browser_driver, "use_driver", lambda env: env == "AUTHCHATGPT_MODE")
    monkeypatch.setattr(gpt.browser_driver, "require_driver", lambda route: Driver())
    monkeypatch.setattr(gpt._LEASES, "launch", lambda **k: pytest.fail("no desktop browser on mobile"))


def test_mobile_logged_out_runs_the_app_as_the_page(monkeypatch):
    page = DriverPage()
    mobile_driver(monkeypatch, page)
    with gpt._open_tab({"token": None, "prompt": "p", "model": "auto", "temporary": False}) as tab:
        assert tab.evaluate("1") is True
    # the app cookie is set once, then the page is loaded again and hooked after it started
    assert page.loaded == [gpt.BASE_URL + "/"] * 2
    assert any("})(window)" in s and "__glossarionGPT = " in s for s in page.scripts) and page.closed


def test_mobile_logged_in_hooks_the_chat_iframe_before_it_starts(monkeypatch):
    page = DriverPage()
    page.run_js = lambda script, timeout=0: (page.scripts.append(script), True)[1]  # cookies already set
    mobile_driver(monkeypatch, page)
    with gpt._open_tab({"token": "tok", "prompt": "p", "model": "auto", "temporary": True}) as tab:
        tab.evaluate("document.title")
    assert page.loaded == [gpt._HOST_URL]
    frame = next(s for s in page.scripts if "createElement('iframe')" in s)
    # the iframe is inserted and hooked in one script: its window is hooked before chatgpt.com loads
    assert frame.index("appendChild(frame)") < frame.index("})(frame.contentWindow)")
    assert page.scripts[-1].startswith("((window, document) => (document.title))(" + gpt._FRAME)


def run_hook(tmp_path, config, models, request_body):
    """Run the page hook in Node against a fake chatgpt.com page; returns what it sent."""
    import shutil
    import subprocess
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is needed to execute the browser JavaScript test")
    script = tmp_path / "hook.js"
    script.write_text(r"""
const sent = [];
global.location = {href:'https://chatgpt.com/', origin:'https://chatgpt.com'};
global.window = global;
const MODELS = %s;
global.fetch = async (url, init) => {
 sent.push({url:String(url), init});
 if (String(url).includes('/models')) return new Response(JSON.stringify(MODELS), {status:200});
 return new Response('data: [DONE]\n\n', {status:200});
};
%s;
(async () => {
 await window.fetch('https://chatgpt.com/backend-anon/f/conversation', {method:'POST', body:JSON.stringify(%s)});
 const conv = sent.find(s => s.url.includes('/f/conversation'));
 const s = window.__glossarionGPT;
 console.log(JSON.stringify({body: conv && JSON.parse(conv.init.body), url: conv && conv.url,
  auth: conv && new Headers(conv.init.headers).get('authorization'), error: s.error, kind: s.kind}));
})();
""" % (json.dumps(models), gpt._hook(config), json.dumps(request_body)), encoding="utf-8")
    import subprocess
    out = subprocess.run([node, str(script)], capture_output=True, text=True, timeout=20)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


PICKER = {"models": [{"slug": "gpt-6"}, {"slug": "auto"}], "versions": [
    {"id": "6", "display_text_for_intelligence": "6", "enabled": True, "intelligence_presets": [
        {"title": "Instant", "model_slug": "gpt-6-instant", "preset_type": "available"},
        {"title": "Medium", "model_slug": "gpt-6-thinking", "thinking_effort": "standard", "preset_type": "available"},
        {"title": "Extra High", "model_slug": "gpt-6-thinking", "thinking_effort": "xhigh", "preset_type": "available"},
        {"title": "Pro", "model_slug": "gpt-6-pro", "preset_type": "upgrade"}]},
    {"id": "5.6-sol", "display_text_for_intelligence": "5.6 Sol", "enabled": True, "intelligence_presets": [
        {"title": "High", "model_slug": "gpt-5-6-sol", "thinking_effort": "extended", "preset_type": "available"}]},
    {"id": "auto", "display_text_for_intelligence": "Auto", "enabled": True, "intelligence_presets": []}]}
CONVERSATION = {"model": "auto", "thinking_effort": "min", "messages": [{"content": {"parts": ["."]}}]}


@pytest.mark.parametrize("name,model,effort", [
    ("gpt-6-medium", "gpt-6-thinking", "standard"), ("gpt-6-extra-high", "gpt-6-thinking", "xhigh"),
    ("gpt-6-instant", "gpt-6-instant", None), ("gpt-5.6-sol-high", "gpt-5-6-sol", "extended"),
    ("auto", "auto", None)])
def test_picker_names_resolve_to_model_and_effort(tmp_path, name, model, effort):
    result = run_hook(tmp_path, {"token": "tok", "prompt": "[user]\nsource", "model": name, "temporary": True},
                      PICKER, CONVERSATION)
    body = result["body"]
    assert (body["model"], body.get("thinking_effort")) == (model, effort)
    assert body["messages"][-1]["content"]["parts"] == ["[user]\nsource"]
    assert body["history_and_training_disabled"] is True
    assert result["url"].endswith("/backend-api/f/conversation") and result["auth"] == "Bearer tok"


def test_locked_and_unknown_picker_names_are_refused(tmp_path):
    locked = run_hook(tmp_path, {"token": "tok", "prompt": "p", "model": "gpt-6-pro", "temporary": True},
                      PICKER, CONVERSATION)
    assert locked.get("body") is None and locked["kind"] == "config_error" and "plan" in locked["error"]
    unknown = run_hook(tmp_path, {"token": None, "prompt": "p", "model": "gpt-9-high", "temporary": False},
                       PICKER, CONVERSATION)
    assert unknown.get("body") is None and "gpt-6-medium" in unknown["error"] and "logged-out" in unknown["error"]


def test_thinking_streams_whole_thoughts():
    thoughts = [{"summary": "Reading the poem", "content": "Moonlight looks like frost.", "chunks": [], "finished": True}]
    stream = (frame({"v": message("assistant", content_type="thoughts", status="in_progress")}, "delta")
              + frame({"p": "/message/content/thoughts", "o": "append", "v": thoughts}, "delta")
              + frame({"v": [{"summary": "Polishing", "content": "Keep it simple.", "finished": True}]}, "delta")
              + frame({"v": message("assistant", ["Quiet night"])}, "delta") + frame("[DONE]"))
    parser = gpt.StreamParser()
    seen = []
    parser.on_thinking = seen.append
    parser.feed(stream)
    assert parser.content == "Quiet night"
    assert seen == ["**Reading the poem**\nMoonlight looks like frost.\n\n", "**Polishing**\nKeep it simple.\n\n"]


def test_pro_handoff_waits_and_reads_the_finished_answer():
    handoff = (frame('"v1"', "delta_encoding") + frame({"v": message("tool", [""], status="in_progress")}, "delta")
               + frame({"type": "stream_handoff", "conversation_id": "c", "options": []}) + frame("[DONE]"))
    parser = gpt.StreamParser()
    parser.feed(handoff)
    assert parser.handoff and not parser.finished  # the first stream's [DONE] is not the end

    class ProPage(Page):
        def __init__(self):
            super().__init__([poll([handoff])])
            self.polls = 0

        def evaluate(self, script):
            if script == gpt._POLL_ANSWER:
                self.polls += 1
                self.states = [poll(final={"text": "<escape><p>Done</p></escape>", "model": "gpt-6-pro"})]
                return True
            return super().evaluate(script)
    page, notes = ProPage(), []
    result = gpt._complete(page, 5, None, None, None, notes.append)
    assert result["content"] == "<p>Done</p>" and result["model"] == "gpt-6-pro" and page.polls == 1
    assert len(notes) == 1 and "background" in notes[0]


def test_images_keep_the_pages_uploaded_parts(tmp_path):
    pointer = {"content_type": "image_asset_pointer", "asset_pointer": "sediment://file_1", "width": 420, "height": 120}
    body = {"model": "auto", "messages": [{"content": {"content_type": "multimodal_text", "parts": [pointer, "."]}}]}
    result = run_hook(tmp_path, {"token": "tok", "prompt": "[user]\npage", "model": "auto", "temporary": True,
                                 "images": 1}, PICKER, body)
    assert result["body"]["messages"][-1]["content"] == {"content_type": "multimodal_text", "parts": [pointer, "[user]\npage"]}
    missing = run_hook(tmp_path, {"token": "tok", "prompt": "p", "model": "auto", "temporary": True, "images": 1},
                       PICKER, CONVERSATION)
    assert missing.get("body") is None and "did not attach" in missing["error"]


def test_images_need_a_login(monkeypatch):
    monkeypatch.setattr(gpt, "_access_token", lambda log: None)
    monkeypatch.setattr(gpt, "_open_tab", lambda *a, **k: pytest.fail("no page without a login"))
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}
    with pytest.raises(gpt.AuthChatGPTError) as error:
        gpt.send_chat_completion(messages=[{"content": [{"type": "text", "text": "x"}, image]}], log_fn=lambda _: None)
    assert error.value.error_type == "config_error" and "login" in str(error.value)


def test_every_mobile_script_is_one_expression(tmp_path, monkeypatch):
    """The WebView bridge evaluates ``var value = (<script>);`` (webview_bridge.page_script): a statement
    list there is a syntax error, which once left the cookie unset and the page "not loaded"."""
    import shutil
    import subprocess
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is needed to parse the scripts")
    scripts = []
    for config in ({"token": None, "prompt": "p", "model": "auto", "temporary": False},
                   {"token": "tok", "prompt": "p", "model": "auto", "temporary": True}):
        page = DriverPage()
        page.run_js = lambda script, timeout=0, page=page: (scripts.append(script), DriverPage.run_js(page, script))[1]
        mobile_driver(monkeypatch, page)
        with gpt._open_tab(config) as tab:
            for script in (gpt._POLL, gpt._COMPOSER, gpt._TYPE, gpt._SEND, gpt._LOAD_CATALOG, gpt._CATALOG,
                           gpt._POLL_ANSWER, gpt._ATTACH, gpt._ATTACHED, gpt._DOWNLOAD.replace("__POINTER__", '"x"')):
                tab.evaluate(script)
    source = tmp_path / "scripts.js"
    newline = "String.fromCharCode(10)"
    source.write_text("const s = " + json.dumps(scripts) + ";\n"
                      "for (const x of s) new Function('var value = (' + " + newline + " + x + " + newline + " + ');');\n"
                      "console.log(s.length);\n", encoding="utf-8")
    out = subprocess.run([node, str(source)], capture_output=True, text=True, timeout=20)
    assert out.returncode == 0, out.stderr[-2000:]
    assert int(out.stdout) == len(scripts) > 20


class NavPage:
    """A mobile page whose navigations commit a new document only a few polls after ``load``
    (a real WebView keeps the old document meanwhile). Frame scripts answer from ``frames``."""

    def __init__(self, frames=()):
        self.doc = {"id": 0, "url": None, "mark": None}
        self.pending = None
        self.cookie = False
        self.hooked_doc = None
        self.frames = list(frames)  # per inserted frame: a callable(poll) -> {"ready", "lost"}
        self.inserted = 0
        self.frame_polls = 0
        self.loaded = []
        self.closed = False

    def load(self, url):
        self.loaded.append(url)
        self.pending = [url, 5]

    def run_js(self, script, js_timeout_ms=15000):
        if self.pending:
            self.pending[1] -= 1
            if not self.pending[1]:
                self.doc = {"id": self.doc["id"] + 1, "url": self.pending[0], "mark": None}
                self.pending = None
        framed = script.startswith("((window, document) => (")
        inner = script[len("((window, document) => ("):script.rindex("))(" + gpt._FRAME)] if framed else script
        if framed:
            self.frame_polls += 1
            state = self.frames[self.inserted - 1](self.frame_polls) if self.inserted else {}
            if inner == gpt._READY:
                return state.get("ready", False)
            if inner == gpt._FRAME_LOST:
                return state.get("lost", False)
            return True
        if inner.startswith("(window.__glossarionNav = '"):
            self.doc["mark"] = inner.split("'")[1]
            return True
        if "__glossarionNav !== '" in inner:
            return self.doc["url"] is not None and self.doc["mark"] != inner.split("__glossarionNav !== '")[1][:32]
        if inner.startswith("document.readyState !== 'loading'"):  # a load wait without the mark
            return self.doc["url"] is not None
        if inner == gpt._APP_COOKIE_JS:
            self.cookie = True
            return True
        if "next-auth.session-token=') >= 0" in inner:
            return self.cookie
        if "createElement('iframe')" in inner:
            self.inserted += 1
            self.frame_polls = 0
            return True
        if "__glossarionGPT = " in inner:
            self.hooked_doc = self.doc["id"]
        return True

    def close(self):
        self.closed = True


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(gpt.time, "sleep", lambda seconds: None)


def test_mobile_reload_waits_for_the_new_document(monkeypatch, no_sleep):
    """The cookie reload loads the same URL: the old, already loaded document passed for the new
    one, so the hook went into a page about to be replaced ("did not finish loading")."""
    page = NavPage()
    mobile_driver(monkeypatch, page)
    with gpt._open_tab({"token": None, "prompt": "p", "model": "auto", "temporary": False}):
        pass
    assert page.loaded == [gpt.BASE_URL + "/"] * 2
    assert page.hooked_doc == page.doc["id"] == 2 and page.closed


def test_mobile_logged_in_first_visit_frames_the_host_page(monkeypatch, no_sleep):
    """A first visit loads the app, then the host page again: the frame belongs in that host page."""
    page = NavPage(frames=[lambda poll: {"ready": poll > 2}])
    real = page.run_js

    def run_js(script, js_timeout_ms=15000):
        if "oai-did=') >= 0" in script:
            return page.doc["url"] == gpt.BASE_URL + "/" or page.doc["id"] > 2
        if "createElement('iframe')" in script:
            page.frame_doc = page.doc["id"]
        return real(script, js_timeout_ms)
    page.run_js = run_js
    mobile_driver(monkeypatch, page)
    with gpt._open_tab({"token": "tok", "prompt": "p", "model": "auto", "temporary": True}):
        pass
    assert page.loaded == [gpt._HOST_URL, gpt.BASE_URL + "/", gpt._HOST_URL]
    assert page.frame_doc == 3 and page.inserted == 1


def test_mobile_frame_that_lost_its_hook_is_inserted_again(monkeypatch, no_sleep):
    """Cloudflare's check passes and reloads the chat iframe into a new window without the hook:
    waiting never helps (the next request, with the clearance cookie, worked), so the frame is
    inserted again at once."""
    page = NavPage(frames=[lambda poll: {"lost": poll > 4}, lambda poll: {"ready": poll > 2}])
    mobile_driver(monkeypatch, page)
    with gpt._open_tab({"token": "tok", "prompt": "p", "model": "auto", "temporary": True}):
        pass
    assert page.inserted == 2


def test_mobile_frame_gets_one_recovery_then_fails(monkeypatch, no_sleep):
    page = NavPage(frames=[lambda poll: {}, lambda poll: {}])
    mobile_driver(monkeypatch, page)
    monkeypatch.setattr(gpt, "_DRIVER_TIMEOUT", 0.05)
    with pytest.raises(gpt.AuthChatGPTError, match="did not finish loading"):
        with gpt._open_tab({"token": "tok", "prompt": "p", "model": "auto", "temporary": True}):
            pass
    assert page.inserted == 2 and page.closed
    lost = gpt._FRAME_LOST
    assert "__glossarionGPT" in lost and "about:blank" in lost and "Just a moment" in lost


def test_mobile_lost_frame_twice_fails_without_waiting(monkeypatch, no_sleep):
    page = NavPage(frames=[lambda poll: {"lost": True}, lambda poll: {"lost": True}])
    mobile_driver(monkeypatch, page)
    with pytest.raises(gpt.AuthChatGPTError, match="did not finish loading"):
        with gpt._open_tab({"token": "tok", "prompt": "p", "model": "auto", "temporary": True}):
            pass
    assert page.inserted == 2


class CdpCalls:
    def __init__(self, order):
        self.order = order

    def call(self, method, params=None, timeout=20):
        self.order.append(method)
        return {}


def test_desktop_cookie_reset_runs_once_for_concurrent_tabs():
    """Two tabs stuck on the same Cloudflare check: the second must not clear the clearance the
    first just earned; it only reloads."""
    resets = gpt._CookieReset()
    first, second = [], []
    seen = resets.generation
    resets.reset(CdpCalls(first), seen)
    resets.reset(CdpCalls(second), seen)
    assert first == ["Network.clearBrowserCookies", "Network.setCookie", "Page.reload"]
    assert second == ["Page.reload"]


def test_desktop_cookie_reset_waits_for_a_sending_tab():
    import threading
    resets = gpt._CookieReset()
    order = []
    sending, release, waiting = threading.Event(), threading.Event(), threading.Event()
    real_wait = resets._wait

    def wait(cancel_check):
        waiting.set()
        real_wait(cancel_check)
    resets._wait = wait

    def sender():
        with resets.sending_message():
            sending.set()
            release.wait(10)
            order.append("sent")
    threads = [threading.Thread(target=sender)]
    threads[0].start()
    assert sending.wait(10)
    threads.append(threading.Thread(target=resets.reset, args=(CdpCalls(order), resets.generation)))
    threads[1].start()
    assert waiting.wait(10)  # the reset is held while the other tab sends
    assert order == []
    release.set()
    for thread in threads:
        thread.join(10)
    assert order == ["sent", "Network.clearBrowserCookies", "Network.setCookie", "Page.reload"]


def test_concurrent_rejections_refresh_the_login_once(monkeypatch):
    """Batch requests all rejected with the same token: one refresh (a second one would reuse the
    rotated refresh token); the others take the new access token."""
    import threading
    inside, release, second_loaded = threading.Event(), threading.Event(), threading.Event()

    class Store:
        _lock = threading.RLock()
        tokens = {"access_token": "old", "refresh_token": "r1"}
        refreshes = 0
        loads = 0

        def load_tokens(self):
            Store.loads += 1
            if Store.loads == 2:
                second_loaded.set()
            return dict(Store.tokens)

        def _try_refresh(self, tokens):
            Store.refreshes += 1
            inside.set()
            release.wait(10)
            Store.tokens = {"access_token": "new" + str(Store.refreshes), "refresh_token": "r2"}
            return dict(Store.tokens)
    monkeypatch.setattr(gpt, "_token_store", lambda: Store())
    results = []
    first = threading.Thread(target=lambda: results.append(gpt._refreshed_token("old")))
    first.start()
    assert inside.wait(10)
    second = threading.Thread(target=lambda: results.append(gpt._refreshed_token("old")))
    second.start()
    second_loaded.wait(.5)  # without the lock the second request reads the old tokens here
    release.set()
    first.join(10)
    second.join(10)
    assert Store.refreshes == 1 and results == ["new1", "new1"]
