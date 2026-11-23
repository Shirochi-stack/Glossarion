"""authgem/ on the Gemini app (gemini.google.com), sent from inside a signed-in page.

Response frames below are trimmed copies of what gemini.google.com sent on 2026-11-23."""
import json
from contextlib import contextmanager

import pytest

import authgem_auth as auth
import settings_rules
import unified_api_client as api


@pytest.fixture(autouse=True)
def isolation(monkeypatch):
    for name in ("TRANSLATION_CANCELLED", "ENABLE_IMAGE_OUTPUT_MODE", "BATCH_TRANSLATION", "LOG_STREAM_CHUNKS",
                 "STREAM_THINKING_LOGS", "ALLOW_AUTHGPT_BATCH_STREAM_LOGS"):
        monkeypatch.delenv(name, raising=False)
    auth._cancel_event.clear()
    monkeypatch.setattr(api, "_save_outgoing_request", lambda *a, **kw: None, raising=False)
    monkeypatch.setattr(api, "_save_incoming_response", lambda *a, **kw: None, raising=False)
    auth.reset_web_session()
    yield
    auth._cancel_event.clear()
    auth.set_web_session_probe(None)
    auth.reset_web_session()


COMPACT = (",", ":")  # the Gemini app sends compact JSON


def answer_frame(text, chat_id="c_f25d9ecb07f4bcdd", model_id="56fdd199312815e2", label="4 Hydrogen",
                 thoughts=None):
    candidate = ["rc_1", [text], None, None, None, None, None, None, None, None, None, None, None, None,
                 None, None, model_id, None, None, label, True]
    if thoughts is not None:  # the thought summary rides in slot 37, as the Gemini app sends it
        candidate += [None] * (38 - len(candidate))
        candidate[37] = [thoughts, [[thoughts, "", "", ""]]]
    inner = [None, [chat_id, "r_1"], None, None, [candidate]]
    return json.dumps([["wrb.fr", None, json.dumps(inner, separators=COMPACT)]], separators=COMPACT)


KEEPALIVE = '[["wrb.fr",null,"[{\\"37\\":[0]}]"]]'
HEADLINE = ('[["wrb.fr",null,"[null,[\\"c_1\\",\\"r_1\\"],{\\"7\\":[null,null,null,null,null,'
            '[\\"Translating Korean to English\\"]],\\"44\\":true}]"]]')


# ---------------------------------------------------------------- parsing

def test_model_names_come_from_picker_labels():
    assert auth.web_model_name("4 Neon") == "gemini-4-neon"
    assert auth.web_model_name("Auto") == "auto"
    assert auth.web_model_name("gemini-4-carbon") == "gemini-4-carbon"


def test_picker_rpc_payload_lists_every_model():
    caps = [2, 140, 3]
    payload = [1, [False], False, None, None, None, True, None, None, None, None, None, False, False, None, [
        ["8c46e95b1a07cecc", "Auto", "Adapts to your needs", caps, 2, None, [], False, "", 1, "Auto"],
        ["56fdd199312815e2", "4 Hydrogen", "4 Hydrogen", caps, 2, None, [], False, "", 20, "4 Hydrogen"],
        ["e6fa609c3fa255c0", "Neon", "4 Neon", caps, 2, None, [], True, "Neon4p0", 4, "4 Neon"],
        ["95a4c0abcba24364", "Carbon", "4 Carbon", caps, 2, None, None, False, "", 4, "4 Carbon"],
    ]]
    assert auth._parse_web_models(payload) == {
        "auto": ("8c46e95b1a07cecc", "Auto"),
        "gemini-4-hydrogen": ("56fdd199312815e2", "4 Hydrogen"),
        "gemini-4-neon": ("e6fa609c3fa255c0", "4 Neon"),
        "gemini-4-carbon": ("95a4c0abcba24364", "4 Carbon"),
    }


def test_answer_frames_arrive_in_pieces_and_keep_text_chat_and_served_model():
    reader = auth._WebAnswer()
    stream = "\n".join([")]}'", "", "5470", answer_frame("o"), "120", answer_frame("ok"),
                        '[["di",294],["af.httprm",294,"-1",22]]'])
    for start in range(0, len(stream), 7):  # the page hands over arbitrary slices
        reader.feed(stream[start:start + 7])
    reader.close()
    assert reader.text == "ok"
    assert reader.chat_id == "c_f25d9ecb07f4bcdd"
    assert reader.served == "4 Hydrogen"


def test_empty_answer_frame_keeps_its_error_codes():
    reader = auth._WebAnswer()
    reader.feed(json.dumps([["wrb.fr", None, None, None, None, [8, None, None, [[1, [1037]]]]]]) + "\n")
    assert reader.text == "" and 1037 in reader.codes


# ---------------------------------------------------------------- the live log

def run_log(lines, **toggles):
    seen = []
    log = auth._WebStreamLog(seen.append, log_stream=toggles.get("log_stream", True),
                             stream_thinking=toggles.get("stream_thinking", True))
    reader = auth._WebAnswer(log)
    for line in lines:
        reader.feed(line + "\n")
    reader.close()
    return seen


def test_thinking_keepalives_and_headline_then_the_answer_line_by_line():
    seen = run_log([KEEPALIVE, KEEPALIVE, HEADLINE, answer_frame("<p>One.</p><p>Tw"),
                    answer_frame("<p>One.</p><p>Two.</p>\n<p>Three")])
    assert seen[0] == "🧠 [authgem] Thinking..."
    assert sum("still thinking" in s for s in seen) == 2
    assert "    **Translating Korean to English**" in seen
    assert any(s.startswith("🧠 [authgem] Thinking complete") for s in seen)
    assert seen[-3:] == ["<p>One.</p>", "<p>Two.</p>", "<p>Three"]


def test_thought_summary_is_logged_once_before_the_answer():
    summary = "**Evaluated prime divisibility**\n\nTested division of 221 by primes up to fifteen."
    seen = run_log([HEADLINE, answer_frame("", thoughts=[summary]),
                    answer_frame("<p>Not prime.</p>", thoughts=[summary]),
                    answer_frame("<p>Not prime.</p><p>13 × 17</p>", thoughts=[summary])])
    assert seen[:4] == ["🧠 [authgem] Thinking...", "    **Translating Korean to English**",
                        "    **Evaluated prime divisibility**", "    Tested division of 221 by primes up to fifteen."]
    assert seen[4].startswith("🧠 [authgem] Thinking complete")
    assert seen[-2:] == ["<p>Not prime.</p>", "<p>13 × 17</p>"]
    assert sum("Evaluated prime divisibility" in s for s in seen) == 1


def test_thought_summary_steps_and_the_thinking_toggle():
    seen = run_log([answer_frame("", thoughts=["**One**", "**Two**"]), answer_frame("<p>A</p>")])
    assert seen[:3] == ["🧠 [authgem] Thinking...", "    **One**", "    **Two**"]
    off = run_log([answer_frame("", thoughts=["**One**"]), answer_frame("<p>A</p>")], stream_thinking=False)
    assert off == ["<p>A</p>"]


def test_thinking_toggle_off_keeps_the_answer_lines():
    seen = run_log([KEEPALIVE, answer_frame("<p>A</p><p>B</p>")], stream_thinking=False)
    assert seen == ["<p>A</p>", "<p>B</p>"]


def test_stream_log_toggle_off_logs_nothing():
    assert run_log([KEEPALIVE, answer_frame("<p>A</p>")], log_stream=False) == []


def test_batch_mode_uses_the_forced_stream_batch_toggle(monkeypatch):
    monkeypatch.setenv("BATCH_TRANSLATION", "1")
    assert auth._web_stream_log(print).log_stream is False
    monkeypatch.setenv("ALLOW_AUTHGPT_BATCH_STREAM_LOGS", "1")
    assert auth._web_stream_log(print).log_stream is True


# ---------------------------------------------------------------- the page

class ScriptedPage:
    """``run_js`` of a gemini.google.com page: tokens, and fetch jobs whose body arrives in ``chunks``."""

    def __init__(self, chunks=(), status=200, tokens=None):
        self.chunks = list(chunks)
        self.status = status
        self.tokens = tokens if tokens is not None else {"at": "AT", "bl": "boq", "sid": "1", "push": "feeds/x",
                                                         "host": "gemini.google.com", "ready": "complete"}
        self.started = []
        self.aborted = 0

    def run_js(self, expression):
        if "WIZ_global_data" in expression:
            return json.dumps(self.tokens)
        if "AbortController" in expression:
            spec = json.loads(expression.split("const spec = ", 1)[1].split(";\n", 1)[0])
            self.started.append(spec)
            return "1"
        if ".splice(0)" in expression:
            piece = self.chunks.pop(0) if self.chunks else ""
            return json.dumps({"c": piece, "done": not self.chunks, "status": self.status, "error": ""})
        if "ctl.abort()" in expression:
            self.aborted += 1
            return True
        raise AssertionError(expression[:80])


def web_page(scripted):
    page = auth._WebPage(scripted.run_js, lambda ms: None)
    page.load_tokens()
    return page


def test_post_streams_the_job_and_sends_the_page_tokens():
    scripted = ScriptedPage(chunks=["a", "b", "c"])
    page = web_page(scripted)
    seen = []
    assert page.post("https://gemini.google.com/x?y=1", body="f=1", headers={"H": "1"}, on_text=seen.append) == "abc"
    assert seen == ["a", "b", "c"]
    (spec,) = scripted.started
    assert spec == {"url": "https://gemini.google.com/x?y=1", "body": "f=1", "headers": {"H": "1"}, "file": None}
    assert page.at == "AT" and "bl=boq" in page.query() and "f.sid=1" in page.query()


def test_post_maps_http_status_to_sign_in_and_rate_limit():
    with pytest.raises(auth.GeminiWebError) as caught:
        web_page(ScriptedPage(chunks=["x"], status=403)).post("https://gemini.google.com/x")
    assert caught.value.kind == "auth_error"
    with pytest.raises(RuntimeError, match="429"):
        web_page(ScriptedPage(chunks=["x"], status=429)).post("https://gemini.google.com/x")


def test_stop_aborts_the_job_in_the_page():
    scripted = ScriptedPage(chunks=["a", "b"])
    page = web_page(scripted)

    def stop(_chunk):
        auth.cancel_stream()

    with pytest.raises(RuntimeError, match="cancelled"):
        page.post("https://gemini.google.com/x", on_text=stop)
    assert scripted.aborted == 1


def test_a_signed_out_page_is_a_sign_in_error():
    scripted = ScriptedPage(tokens={"at": "", "bl": "boq", "host": "gemini.google.com", "ready": "complete"})
    with pytest.raises(auth.GeminiWebError) as caught:
        web_page(scripted)
    assert caught.value.kind == "auth_error"
    redirected = ScriptedPage(tokens={"at": "", "host": "accounts.google.com", "ready": "complete"})
    with pytest.raises(auth.GeminiWebError):
        web_page(redirected)


def test_a_browser_held_by_another_glossarion_is_an_authgem_error_that_is_not_retried(monkeypatch):
    import contextlib
    import gemini_free

    @contextlib.contextmanager
    def closed(*_a, **_kw):
        raise gemini_free.AIModeError("The Gemini Free browser closed.", gemini_free.KIND_BROWSER)
        yield

    monkeypatch.setattr(gemini_free, "signed_in_tab", closed)
    with pytest.raises(auth.GeminiWebError) as caught:
        with auth._open_web_page():
            pass
    assert caught.value.kind == "browser_error"
    assert "authgem/" in str(caught.value) and "Gemini Free" not in str(caught.value)


def test_image_upload_sends_the_file_from_the_page():
    scripted = ScriptedPage(chunks=["/contrib_service/ttl_1d/abc"])
    page = web_page(scripted)
    part = auth._upload_web_image(page, {"b64": "AAAA", "mime": "image/png", "name": "image1.png"}, 30)
    assert part == [["/contrib_service/ttl_1d/abc", 1], "image1.png"]
    spec = scripted.started[0]
    assert spec["file"] == {"b64": "AAAA", "mime": "image/png", "name": "image1.png"}
    assert spec["headers"]["Push-ID"] == "feeds/x"


# ---------------------------------------------------------------- sending

@pytest.fixture
def fake_app(monkeypatch):
    state = {"frames": [], "sent": [], "deleted": []}
    page = object()

    @contextmanager
    def open_page(cancel_check=None):
        yield page

    def generate(pg, prompt, images, model_id, timeout, log, thinking="high", expect=""):
        state["sent"].append({"prompt": prompt, "model_id": model_id, "images": images, "thinking": thinking})
        reader = auth._WebAnswer()
        for frame in state["frames"]:
            reader.feed(frame + "\n")
        reader.close()
        return reader

    monkeypatch.setattr(auth, "_open_web_page", open_page)
    monkeypatch.setattr(auth, "_web_models", lambda pg: dict(auth.GEMINI_WEB_KNOWN_MODELS))
    monkeypatch.setattr(auth, "_web_generate", generate)
    monkeypatch.setattr(auth, "_delete_web_chat", lambda pg, chat_id: state["deleted"].append(chat_id))
    return state


def test_send_flattens_messages_picks_the_model_and_deletes_the_chat(fake_app):
    fake_app["frames"] = [answer_frame("The weather is nice today.", model_id="e6fa609c3fa255c0", label="4 Neon")]
    reply = auth.send_chat_completion_web(
        [{"role": "system", "content": "You are a translator."}, {"role": "user", "content": "오늘은 날씨가 좋네요."}],
        model="gemini-4-neon")
    assert reply["content"] == "The weather is nice today." and reply["model"] == "4 Neon"
    assert fake_app["sent"][0]["model_id"] == "e6fa609c3fa255c0"
    assert fake_app["sent"][0]["prompt"] == "[system]\nYou are a translator.\n\n[user]\n오늘은 날씨가 좋네요."
    assert fake_app["deleted"] == ["c_f25d9ecb07f4bcdd"]


@pytest.mark.parametrize("env,expected", [
    ({}, "high"),  # the Gemini thinking level's default
    ({"GEMINI_THINKING_LEVEL": "medium"}, "medium"),
    ({"GEMINI_THINKING_LEVEL": "low"}, "low"),
    ({"GEMINI_THINKING_LEVEL": "minimal"}, "low"),  # the app has no minimal
    ({"ENABLE_GEMINI_THINKING": "0", "GEMINI_THINKING_LEVEL": "high"}, "low"),  # no off: its lightest level
    ({"THINKING_BUDGET": "0", "GEMINI_THINKING_LEVEL": "high"}, "low"),
    ({"GEMINI_THINKING_LEVEL": "nonsense"}, "high"),
])
def test_the_gemini_thinking_settings_pick_the_app_thinking_level(monkeypatch, fake_app, env, expected):
    for name in ("ENABLE_GEMINI_THINKING", "THINKING_BUDGET", "GEMINI_THINKING_LEVEL"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    fake_app["frames"] = [answer_frame("ok", model_id="e6fa609c3fa255c0", label="4 Neon")]
    auth.send_chat_completion_web([{"role": "user", "content": "hi"}], model="gemini-4-neon")
    assert fake_app["sent"][-1]["thinking"] == expected


@pytest.mark.parametrize("model, served, line", [
    ("gemini-4-neon", "4 Neon", "🧠 AuthGem: 4 Neon · thinking level High"),
    ("gemini-4-hydrogen", "4 Hydrogen", "🧠 AuthGem: 4 Hydrogen · thinking level High"),
    ("auto", "4 Hydrogen", "🧠 AuthGem: Auto · thinking level High (when Auto picks a thinking model)"),
])
def test_the_request_line_says_whether_the_model_thinks(monkeypatch, fake_app, model, served, line):
    monkeypatch.setenv("GEMINI_THINKING_LEVEL", "high")
    fake_app["frames"] = [answer_frame("ok", label=served)]
    seen = []
    auth.send_chat_completion_web([{"role": "user", "content": "hi"}], model=model, log_fn=seen.append)
    assert seen[0] == line


def test_a_swapped_model_stops_on_its_first_frame_with_a_log_line():
    seen = []
    log = auth._WebStreamLog(seen.append, log_stream=True, stream_thinking=True)
    reader = auth._WebAnswer(log, expect="4 Carbon", log=seen.append)
    with pytest.raises(auth.GeminiWebError) as caught:
        reader.feed(answer_frame("type,raw_name", label="3.5 Flash-Lite", chat_id="c_swap") + "\n")
    assert caught.value.kind == "model_unavailable" and "3.5 Flash-Lite instead of 4 Carbon" in str(caught.value)
    assert seen == ["⚠️ AuthGem: the Gemini app swapped 4 Carbon for 3.5 Flash-Lite; "
                    "stopping the request (its answer is not used)"]  # none of the swapped answer is logged
    assert reader.chat_id == "c_swap"


def test_the_swapped_request_is_aborted_in_the_page_and_its_chat_deleted(monkeypatch):
    deleted = []
    monkeypatch.setattr(auth, "_delete_web_chat", lambda pg, chat_id: deleted.append(chat_id))
    scripted = ScriptedPage(chunks=[answer_frame("a", label="3.5 Flash-Lite", chat_id="c_swap") + "\n",
                                    answer_frame("ab", label="3.5 Flash-Lite", chat_id="c_swap") + "\n"])
    page = web_page(scripted)
    with pytest.raises(auth.GeminiWebError):
        auth._web_generate(page, "hi", [], "95a4c0abcba24364", 60, lambda m: None, "high", expect="4 Carbon")
    assert deleted == ["c_swap"] and scripted.aborted
    reader = auth._web_generate(web_page(ScriptedPage(chunks=[answer_frame("ok", label="4 Carbon")])),
                                "hi", [], "95a4c0abcba24364", 60, lambda m: None, "high", expect="4 Carbon")
    assert reader.text == "ok"


def test_the_thinking_level_goes_to_the_request_slot_the_app_uses():
    scripted = ScriptedPage(chunks=[answer_frame("ok")])
    page = web_page(scripted)
    for level, code in (("low", 1), ("medium", 4), ("high", 2)):
        scripted.chunks = [answer_frame("ok")]
        auth._web_generate(page, "hi", [], "e6fa609c3fa255c0", 60, lambda m: None, level)
        body = scripted.started[-1]["body"]
        from urllib.parse import parse_qs
        inner = json.loads(json.loads(parse_qs(body)["f.req"][0])[1])
        assert len(inner) == 81 and inner[80] == code


def test_a_model_the_plan_lacks_is_refused_not_swapped(fake_app):
    fake_app["frames"] = [answer_frame("Hi!", label="4 Hydrogen")]
    with pytest.raises(auth.GeminiWebError) as caught:
        auth.send_chat_completion_web([{"role": "user", "content": "hi"}], model="gemini-4-carbon")
    assert caught.value.kind == "model_unavailable" and "4 Carbon" in str(caught.value)


def test_auto_accepts_whatever_model_answered(fake_app):
    fake_app["frames"] = [answer_frame("pong")]
    assert auth.send_chat_completion_web([{"role": "user", "content": "ping"}], model="")["content"] == "pong"


def test_cite_markers_are_removed(fake_app):
    fake_app["frames"] = [answer_frame("Red[cite: 1] and [cite_start]blue")]
    assert auth.send_chat_completion_web([{"role": "user", "content": "colors?"}], model="auto")["content"] == "Red and blue"


def test_usage_limit_reads_as_a_rate_limit(fake_app):
    fake_app["frames"] = [json.dumps([["wrb.fr", None, None, None, None, [8, None, None, [[1, [1037]]]]]])]
    with pytest.raises(RuntimeError, match="429"):
        auth.send_chat_completion_web([{"role": "user", "content": "hi"}], model="gemini-4-neon")


def test_unknown_model_lists_the_available_ones(fake_app):
    with pytest.raises(auth.GeminiWebError) as caught:
        auth.send_chat_completion_web([{"role": "user", "content": "hi"}], model="gemini-9-xenon")
    assert caught.value.kind == "config_error" and "authgem/gemini-4-neon" in str(caught.value)


def test_registered_probe_answers_has_session():
    auth.set_web_session_probe(lambda: False)
    assert auth.has_web_session() is False
    auth.set_web_session_probe(lambda: True)
    assert auth.has_web_session() is True


# ---------------------------------------------------------------- client side

def make_client(monkeypatch, tmp_path, model):
    client = api.UnifiedClient("", model, str(tmp_path))
    monkeypatch.setattr(client, "_get_max_retries", lambda: 3)
    monkeypatch.setattr(client, "_is_stop_requested", lambda: False)
    monkeypatch.setattr(client, "_get_active_request_model", lambda: model)
    return client


def test_client_sends_the_picker_name(monkeypatch, tmp_path):
    client = make_client(monkeypatch, tmp_path, "authgem/gemini-4-neon")
    sent = {}
    monkeypatch.setattr(api, "_authgem_send_web",
                        lambda **kw: sent.update(kw) or {"content": "done", "finish_reason": "stop", "usage": None})
    response = client._send_authgem([{"role": "user", "content": "x"}], 0.3, 4096, None)
    assert response.content == "done" and sent["model"] == "gemini-4-neon"


def test_client_empty_model_name_means_auto(monkeypatch, tmp_path):
    client = make_client(monkeypatch, tmp_path, "authgem/")
    sent = {}
    monkeypatch.setattr(api, "_authgem_send_web",
                        lambda **kw: sent.update(kw) or {"content": "done", "finish_reason": "stop", "usage": None})
    client._send_authgem([{"role": "user", "content": "x"}], None, None, None)
    assert sent["model"] == "auto"


@pytest.mark.parametrize("kind", ["auth_error", "model_unavailable", "config_error", "browser_error"])
def test_errors_a_retry_cannot_fix_stop_at_once(monkeypatch, tmp_path, kind):
    client = make_client(monkeypatch, tmp_path, "authgem/gemini-4-carbon")
    calls = []

    def fail(**kw):
        calls.append(1)
        raise auth.GeminiWebError("nope", kind)

    monkeypatch.setattr(api, "_authgem_send_web", fail)
    with pytest.raises(api.UnifiedClientError) as caught:
        client._send_authgem([{"role": "user", "content": "x"}], None, None, None)
    assert caught.value.error_type == kind and len(calls) == 1


def test_image_output_is_refused_on_the_text_route(monkeypatch, tmp_path):
    client = make_client(monkeypatch, tmp_path, "authgem/gemini-4-hydrogen")
    monkeypatch.setattr(client, "_authgem_image_output", lambda model: {"size": "1K", "aspect_ratio": "auto"})
    with pytest.raises(api.UnifiedClientError) as caught:
        client._send_authgem([{"role": "user", "content": "x"}], None, None, None)
    assert caught.value.error_type == "config_error"


# ---------------------------------------------------------------- login controls

def test_authgem_uses_the_google_login_and_vertex_keeps_oauth():
    assert settings_rules.google_login_needed("authgem/gemini-4-neon")
    assert settings_rules.google_login_needed("authgem2/auto")
    assert settings_rules.google_login_needed("search/gemini")
    assert not settings_rules.google_login_needed("authgem-vertex/gemini-2.5-pro")
    assert settings_rules.authgem_login_needed("authgem/gemini-4-neon", {}) == (False, False)
    assert settings_rules.authgem_login_needed("authgem-vertex/gemini-2.5-pro", {}) == (True, True)


def test_route_controls_name_the_google_login():
    assert "google" in settings_rules.route_controls("authgem/gemini-4-neon", {}).logins
    assert "authgem" not in settings_rules.route_controls("authgem/gemini-4-neon", {}).logins
    assert "authgem" in settings_rules.route_controls("authgem-vertex/gemini-2.5-pro", {}).logins


def test_enabled_pool_entry_shows_the_google_login():
    pool_key, toggle = next(iter(settings_rules.LOGIN_KEY_POOL_TOGGLES.items()))
    config = {toggle: True, pool_key: [{"model": "authgem/gemini-4-neon", "enabled": True}]}
    assert settings_rules.google_login_needed("gpt-5", config)
