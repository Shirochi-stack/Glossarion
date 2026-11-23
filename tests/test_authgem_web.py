"""authgem/ on the Gemini app (gemini.google.com) with the Google Login's session cookies.

Response frames below are trimmed copies of what gemini.google.com sent on 2026-11-23."""
import json

import pytest

import authgem_auth as auth
import settings_rules
import unified_api_client as api


@pytest.fixture(autouse=True)
def isolation(monkeypatch):
    monkeypatch.delenv("TRANSLATION_CANCELLED", raising=False)
    monkeypatch.delenv("ENABLE_IMAGE_OUTPUT_MODE", raising=False)
    auth._cancel_event.clear()
    monkeypatch.setattr(api, "_save_outgoing_request", lambda *a, **kw: None, raising=False)
    monkeypatch.setattr(api, "_save_incoming_response", lambda *a, **kw: None, raising=False)
    auth.reset_web_session()
    yield
    auth._cancel_event.clear()
    auth.set_web_cookie_provider(None)
    auth.reset_web_session()


def answer_frame(text, chat_id="c_f25d9ecb07f4bcdd", model_id="56fdd199312815e2", label="4 Hydrogen"):
    inner = [None, [chat_id, "r_1"], None, None, [["rc_1", [text], None, None, None, None, None, None,
                                                  None, None, None, None, None, None, None, None, model_id,
                                                  None, None, label, True]]]
    compact = (",", ":")  # the Gemini app sends compact JSON
    return json.dumps([["wrb.fr", None, json.dumps(inner, separators=compact)]], separators=compact)


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


def test_answer_frames_keep_the_longest_text_chat_and_served_model():
    reader = auth._WebAnswer()
    for line in (")]}'", "", "5470", answer_frame("o"), "120", answer_frame("ok"),
                 '[["di",294],["af.httprm",294,"-1",22]]'):
        reader.feed(line)
    assert reader.text == "ok"
    assert reader.chat_id == "c_f25d9ecb07f4bcdd"
    assert reader.served == "4 Hydrogen"


def test_empty_answer_frame_keeps_its_error_codes():
    reader = auth._WebAnswer()
    reader.feed(json.dumps([["wrb.fr", None, None, None, None, [8, None, None, [[1, [1037]]]]]]))
    assert reader.text == "" and 1037 in reader.codes


# ---------------------------------------------------------------- sending

class FakeSession:
    def __init__(self):
        self.models = dict(auth.GEMINI_WEB_KNOWN_MODELS)
        self.deleted = []


@pytest.fixture
def fake_app(monkeypatch):
    session = FakeSession()
    state = {"frames": [], "sent": []}

    def generate(sess, prompt, images, model_id, timeout, connect_timeout, log):
        state["sent"].append({"prompt": prompt, "model_id": model_id, "images": images})
        reader = auth._WebAnswer()
        for frame in state["frames"]:
            reader.feed(frame)
        return reader

    monkeypatch.setattr(auth, "_get_web_session", lambda **kw: session)
    monkeypatch.setattr(auth, "_web_generate", generate)
    monkeypatch.setattr(auth, "_delete_web_chat", lambda sess, chat_id: session.deleted.append(chat_id))
    return session, state


def test_send_flattens_messages_picks_the_model_and_deletes_the_chat(fake_app):
    session, state = fake_app
    state["frames"] = [answer_frame("The weather is nice today.", model_id="e6fa609c3fa255c0", label="4 Neon")]
    reply = auth.send_chat_completion_web(
        [{"role": "system", "content": "You are a translator."}, {"role": "user", "content": "오늘은 날씨가 좋네요."}],
        model="gemini-4-neon")
    assert reply["content"] == "The weather is nice today." and reply["model"] == "4 Neon"
    assert state["sent"][0]["model_id"] == "e6fa609c3fa255c0"
    assert state["sent"][0]["prompt"] == "[system]\nYou are a translator.\n\n[user]\n오늘은 날씨가 좋네요."
    assert session.deleted == ["c_f25d9ecb07f4bcdd"]


def test_a_model_the_plan_lacks_is_refused_not_swapped(fake_app):
    _session, state = fake_app
    state["frames"] = [answer_frame("Hi!", label="4 Hydrogen")]
    with pytest.raises(auth.GeminiWebError) as caught:
        auth.send_chat_completion_web([{"role": "user", "content": "hi"}], model="gemini-4-carbon")
    assert caught.value.kind == "model_unavailable" and "4 Carbon" in str(caught.value)


def test_auto_accepts_whatever_model_answered(fake_app):
    _session, state = fake_app
    state["frames"] = [answer_frame("pong")]
    assert auth.send_chat_completion_web([{"role": "user", "content": "ping"}], model="")["content"] == "pong"


def test_cite_markers_are_removed(fake_app):
    _session, state = fake_app
    state["frames"] = [answer_frame("Red[cite: 1] and [cite_start]blue")]
    assert auth.send_chat_completion_web([{"role": "user", "content": "colors?"}], model="auto")["content"] == "Red and blue"


def test_usage_limit_reads_as_a_rate_limit(fake_app):
    _session, state = fake_app
    state["frames"] = [json.dumps([["wrb.fr", None, None, None, None, [8, None, None, [[1, [1037]]]]]])]
    with pytest.raises(RuntimeError, match="429"):
        auth.send_chat_completion_web([{"role": "user", "content": "hi"}], model="gemini-4-neon")


def test_unknown_model_lists_the_available_ones(fake_app):
    with pytest.raises(auth.GeminiWebError) as caught:
        auth.send_chat_completion_web([{"role": "user", "content": "hi"}], model="gemini-9-xenon")
    assert caught.value.kind == "config_error" and "authgem/gemini-4-neon" in str(caught.value)


def test_session_needs_a_signed_in_cookie(monkeypatch):
    auth.set_web_cookie_provider(lambda cancel_check: [{"name": "NID", "value": "x", "domain": ".google.com"}],
                                 has_session=lambda: True)
    with pytest.raises(auth.GeminiWebError) as caught:
        auth._get_web_session()
    assert caught.value.kind == "auth_error"


def test_registered_provider_answers_has_session():
    auth.set_web_cookie_provider(lambda cancel_check: [], has_session=lambda: False)
    assert auth.has_web_session() is False
    auth.set_web_cookie_provider(lambda cancel_check: [], has_session=lambda: True)
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


@pytest.mark.parametrize("kind", ["auth_error", "model_unavailable", "config_error"])
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
    toggles = settings_rules.LOGIN_KEY_POOL_TOGGLES
    pool_key, toggle = next(iter(toggles.items()))
    config = {toggle: True, pool_key: [{"model": "authgem/gemini-4-neon", "enabled": True}]}
    assert settings_rules.google_login_needed("gpt-5", config)
