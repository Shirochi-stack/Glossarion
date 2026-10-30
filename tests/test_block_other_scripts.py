"""Universal adaptive script ban, "Block other scripts in translations (adaptive, all models)" (owner
2026-10-28; ``src/script_ban_core.py``, config ``block_other_scripts`` / env ``BLOCK_OTHER_SCRIPTS``).

Covers the shared decision (the target's own script is never banned, unknown targets ban CJK only,
glossary / single-pass never ban), the output guard (kept names from the request are ignored, small
amounts are tolerated), the OpenAI logit bias (official chat models only, capped, safe ids), the
UnifiedClient wiring (off by default and with Anti-Duplicate off; a leaking answer is kept and logged) and
the settings plumbing (Other Settings checkbox with a warning, owner_state, run_env, schema).

Run (repository root)::

    py -3.12 -m pytest -p no:cacheprovider -W ignore -q tests/test_block_other_scripts.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.environ.setdefault("GLOSSARION_HTTP_LOG", "0")

import script_ban_core as sbc  # noqa: E402

ENGLISH_BANS = sbc.banned_scripts("translation", "English")


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    for name in ("HOME", "USERPROFILE", "GLOSSARION_LIBRARY_DIR", "OUTPUT_DIRECTORY", "GLOSSARION_DATA_DIR"):
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    for name in ("ENABLE_ANTI_DUPLICATE", "BLOCK_OTHER_SCRIPTS", "OUTPUT_LANGUAGE", "AUTO_GLOSSARY_MODE",
                 "SINGLE_PASS_GLOSSARY_MODE", "RETRY_TRUNCATED", "SYSTEM_PROMPT_TO_USER"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


# ------------------------------------------------------------------------------------- decision

def test_mobile_module_uses_the_shared_table():
    sys.path.insert(0, str(SRC / "mobile" / "app"))
    try:
        from glossarion_mobile.services.local_llm import script_ban
    except ImportError as exc:  # numpy-free import; a missing optional dependency only
        pytest.skip(f"mobile package not importable: {exc}")
    assert script_ban.SCRIPTS is sbc.SCRIPTS
    assert script_ban.banned_scripts is sbc.banned_scripts
    assert script_ban.token_is_banned is sbc.token_is_banned


@pytest.mark.parametrize("target, own", [
    ("English", None), ("Hindi", "devanagari"), ("Russian", "cyrillic"), ("Korean", "cjk"),
    ("Chinese (Traditional)", "cjk"), ("Thai", "thai"), ("Arabic", "arabic"), ("हिन्दी", "devanagari"),
    ("sr-Latn", "cyrillic"), ("Punjabi", "gurmukhi"),
])
def test_the_target_language_script_is_never_banned(target, own):
    banned = sbc.banned_scripts("translation", target)
    assert banned
    if own:
        assert own not in banned
    for ch in {"Hindi": "न", "Russian": "Ж", "Korean": "한", "Thai": "ก", "Arabic": "ع"}.get(target, ""):
        assert not sbc.foreign_script_leak(ch * 50, banned)


def test_every_listed_target_keeps_its_own_script():
    from language_options import TARGET_LANGUAGES
    for target in TARGET_LANGUAGES:
        own = sbc.target_scripts(target)
        banned = sbc.banned_scripts("translation", target)
        if own is None:
            assert banned in ((), ("cjk",)), target
        else:
            assert not set(own) & set(banned), target


def test_unknown_targets_glossary_and_single_pass():
    assert sbc.banned_scripts("translation", "Klingon") == ("cjk",)
    assert sbc.banned_scripts("glossary", "English") == ()
    assert sbc.banned_scripts("translation", "English", single_pass=True) == ()
    assert sbc.banned_scripts("review", "English") == ()
    assert sbc.banned_scripts("translation", "") == ()


# ------------------------------------------------------------------------------------ output guard

def test_leak_counts_foreign_runs_but_not_names_from_the_request():
    source = [{"role": "system", "content": "Glossary: 김민준 -> Kim Min-jun"},
              {"role": "user", "content": "<p>김민준은 웃었다.</p>"}]
    kept = "<p>Kim Min-jun (김민준) smiled.</p>"
    assert sbc.foreign_script_leak(kept, ENGLISH_BANS, source) is None
    leaked = "<p>Kim Min-jun smiled 他笑了起来. नमस्ते</p>"
    leak = sbc.foreign_script_leak(leaked, ENGLISH_BANS, source)
    assert leak and leak["chars"] == 11
    assert leak["scripts"] == {"cjk": 5, "devanagari": 6}


def test_an_answer_that_copies_the_source_is_a_leak():
    """Owner 2026-10-30 (ollamapull/qwen3.5:2b): the model streamed the Chinese chapter back unchanged and
    no retry followed, because every copied run was "also in the request"."""
    chapter = ("<h2>第 25 章 寻找摩根勒菲的踪迹(三更)</h2>\n<p>某个小木屋中。</p>\n"
               "<p>“是么，你就是我妹妹之前提起过的那个岛对面的朋友么？”</p>\n<p>然后——</p>")
    source = [{"role": "system", "content": "Output only english translations with html tags"},
              {"role": "user", "content": chapter}]
    leak = sbc.foreign_script_leak(chapter, ENGLISH_BANS, source)
    assert leak and leak["chars"] > 40 and "cjk" in leak["scripts"]
    # one long copied line in an otherwise English answer is not a kept name either
    mixed = "<h2>Chapter 25</h2>\n<p>In a small cabin.</p>\n<p>“是么，你就是我妹妹之前提起过的那个岛对面的朋友么？”</p>"
    assert sbc.foreign_script_leak(mixed + "<p>Then —</p>" * 20, ENGLISH_BANS, source)
    # names kept in a real translation still pass, however often they occur
    names = [{"role": "user", "content": "<p>摩根勒菲和凯在小木屋中。</p>"}]
    english = "<p>Morgan (摩根勒菲) looked at Kay. Morgan (摩根勒菲) smiled and left the small cabin.</p>" * 5
    assert sbc.foreign_script_leak(english, ENGLISH_BANS, names) is None


def test_leak_threshold_tolerates_a_stray_character():
    assert sbc.foreign_script_leak("He said 你好 and left.", ENGLISH_BANS) is None  # 2 < LEAK_MIN_CHARS
    long_text = "word " * 2000 + "你好吗"  # 3 chars but under 0.1 % of ~8000
    assert sbc.foreign_script_leak(long_text, ENGLISH_BANS) is None
    assert sbc.foreign_script_leak("He said 你好吗 and left.", ENGLISH_BANS)


# ----------------------------------------------------------------------------------- logit bias

@pytest.mark.parametrize("model, name", [
    ("gpt-4o", "o200k_base"), ("gpt-4o-mini", "o200k_base"), ("gpt-4.1", "o200k_base"),
    ("gpt-4-turbo", "cl100k_base"), ("gpt-3.5-turbo", "cl100k_base"),
    ("o3-mini", None), ("gpt-5", None), ("o1", None), ("claude-sonnet-4", None), ("deepseek-chat", None),
])
def test_only_official_chat_models_take_a_logit_bias(model, name):
    assert sbc.openai_encoding_name(model) == name


def test_openai_logit_bias_is_capped_and_bans_only_banned_tokens():
    tiktoken = pytest.importorskip("tiktoken")
    try:
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception as exc:
        pytest.skip(f"tiktoken encoding unavailable offline: {exc}")
    bias = sbc.openai_logit_bias("gpt-4", ENGLISH_BANS)
    assert 0 < len(bias) <= sbc.OPENAI_LOGIT_BIAS_LIMIT
    assert set(bias.values()) == {-100}
    for key in bias:
        assert int(key) <= 100256
        assert sbc.token_is_banned(enc.decode_single_token_bytes(int(key)), ENGLISH_BANS)
    assert sbc.openai_logit_bias("gpt-5", ENGLISH_BANS) == {}
    assert sbc.openai_logit_bias("gpt-4", ()) == {}


# ------------------------------------------------------------------------------- UnifiedClient

@pytest.fixture
def client(isolated):
    import unified_api_client as api
    return api.UnifiedClient(api_key="sk-test", model="gpt-4o", output_dir=str(isolated / "out"))


def _turn_on(monkeypatch, target="English"):
    monkeypatch.setenv("ENABLE_ANTI_DUPLICATE", "1")
    monkeypatch.setenv("BLOCK_OTHER_SCRIPTS", "1")
    monkeypatch.setenv("OUTPUT_LANGUAGE", target)


def test_client_bans_nothing_unless_both_switches_are_on(client, monkeypatch):
    monkeypatch.setenv("OUTPUT_LANGUAGE", "English")
    assert client._script_ban_scripts("translation") == ()
    monkeypatch.setenv("BLOCK_OTHER_SCRIPTS", "1")
    assert client._script_ban_scripts("translation") == ()  # Anti-Duplicate off
    monkeypatch.setenv("ENABLE_ANTI_DUPLICATE", "1")
    assert client._script_ban_scripts("translation") == ENGLISH_BANS
    assert client._script_ban_scripts("review") == ()
    assert client._script_ban_scripts("glossary") == ()


def test_client_logit_bias_only_for_official_openai(client, monkeypatch):
    _turn_on(monkeypatch)
    monkeypatch.setattr(sbc, "openai_logit_bias", lambda model, scripts: {"1": -100})
    monkeypatch.setattr(client, "_server_tokenize", lambda *a: None)  # a server without /tokenize
    client.context = "translation"
    assert client._script_ban_logit_bias("openai", "gpt-4o", "https://api.openai.com/v1") == {"logit_bias": {"1": -100}}
    assert client._script_ban_logit_bias("openai", "gpt-4o", "https://my-proxy.example/v1") == {}
    assert client._script_ban_logit_bias("deepseek", "deepseek-chat", "https://api.deepseek.com/v1") == {}
    assert client._script_ban_logit_bias("openai", "gpt-4o", "https://api.openai.com/v1", True) == {}


# ----------------------------------------------- a server's own tokenizer (llama.cpp server, vLLM)

def _byte_level(text):
    spell = {byte: ch for ch, byte in sbc._byte_level_alphabet().items()}
    return "".join(spell[b] for b in text.encode("utf-8"))


def test_server_pieces_decode_in_every_tokenizer_spelling():
    # llama.cpp with_pieces: text, or byte values for a partial character
    assert sbc.server_logit_bias([(10, "人"), (11, " the"), (12, [0xE6]), (13, [0xC3])], ENGLISH_BANS) == \
        {"10": -100, "12": -100}
    # vLLM token_strs, byte-level BPE (Qwen, Llama 3): 人 is "äºº", which alone looks like Latin-1 text
    pairs = [(1, _byte_level("人")), (2, _byte_level(" 的")), (3, "Ġthe"), (4, "Ċ"), (5, _byte_level("é"))]
    assert sbc.server_logit_bias(pairs, ENGLISH_BANS) == {"1": -100, "2": -100}
    # vLLM token_strs, SentencePiece (Mistral, Gemma): "▁" spaces and <0xE6> byte tokens
    assert sbc.server_logit_bias([(20, "▁人"), (21, "<0xE6>"), (22, "▁the"), (23, "<0x41>")], ENGLISH_BANS) == \
        {"20": -100, "21": -100}


def test_server_probe_text_covers_each_banned_script():
    text = sbc.server_probe_text(ENGLISH_BANS)
    lines = set(text.split("\n"))
    for ch in "的人我是说你ไक्षаب":  # common Han, Thai, Devanagari, Cyrillic, Arabic characters, alone
        assert ch in lines or any(ch in line for line in lines), ch
    assert "的" in lines and "а" in lines
    assert sbc.server_probe_text(()) == ""


def test_client_asks_a_tokenizing_server_once_and_merges_nothing_else(client, monkeypatch):
    _turn_on(monkeypatch)
    client.context = "translation"
    import unified_api_client as api
    monkeypatch.setattr(api.UnifiedClient, "_SERVER_BIAS_CACHE", {})
    asked = []

    def tokenize(base_url, model, text):
        asked.append((base_url, model))
        return [(7, "人"), (8, " the"), (9, "ру")]

    monkeypatch.setattr(client, "_server_tokenize", tokenize)
    params = client._logit_bias_params("openai", "qwen2.5-7b", "http://127.0.0.1:8080/v1")
    assert params == {"logit_bias": {"7": -100, "9": -100}}
    assert client._logit_bias_params("openai", "qwen2.5-7b", "http://127.0.0.1:8080/v1") == params
    assert asked == [("http://127.0.0.1:8080/v1", "qwen2.5-7b")]  # cached per server and model
    # a cloud provider that is not an OpenAI-compatible custom endpoint is never probed
    assert client._logit_bias_params("deepseek", "deepseek-chat", "https://api.deepseek.com/v1") == {}
    assert len(asked) == 1
    monkeypatch.setenv("BLOCK_OTHER_SCRIPTS", "0")
    assert client._logit_bias_params("openai", "qwen2.5-7b", "http://127.0.0.1:8080/v1") == {}


def test_server_tokenize_reads_llamacpp_and_vllm_replies(client, monkeypatch):
    import unified_api_client as api
    replies = {"content": (200, {"tokens": [{"id": 5, "piece": "人"}, {"id": 6, "piece": [0xE6]}]}),
               "prompt": (200, {"tokens": [5, 6], "token_strs": ["äºº", "Ġthe"]})}
    sent = []

    class Reply:
        def __init__(self, status, data):
            self.status_code, self._data = status, data

        def json(self):
            return self._data

    def post(url, json=None, headers=None, timeout=None):
        sent.append((url, sorted(json)))
        kind = "content" if "content" in json else "prompt"
        status, data = replies[kind]
        return Reply(status, data)

    monkeypatch.setattr(api.requests, "post", post)
    assert client._server_tokenize("http://h:8080/v1", "m", "人") == [(5, "人"), (6, [0xE6])]
    assert sent[-1][0] == "http://h:8080/tokenize"
    replies["content"] = (404, {})  # vLLM: the llama.cpp body is refused, the vLLM one answers
    assert client._server_tokenize("http://h:8000/v1/", "m", "人") == [(5, "äºº"), (6, "Ġthe")]
    replies["prompt"] = (404, {})
    assert client._server_tokenize("http://h:8000/v1", "m", "人") is None


# ------------------------------------------------------------- word bias (Enable word bias presets)

def test_word_bias_presets_reach_official_openai_merged_with_the_script_ban(client, monkeypatch):
    client.context = "translation"
    monkeypatch.setenv("ENABLE_ANTI_DUPLICATE", "1")
    monkeypatch.setenv("LOGIT_BIAS_ENABLED", "1")
    monkeypatch.setenv("BIAS_COMMON_WORDS", "1")
    monkeypatch.setenv("LOGIT_BIAS_STRENGTH", "-0.5")  # an older config's signed value: same as 0.5
    import tiktoken
    enc = tiktoken.get_encoding("o200k_base")
    the = str(enc.encode(" the")[0])
    params = client._logit_bias_params("openai", "gpt-4o", "https://api.openai.com/v1")
    assert params["logit_bias"][the] == -0.5 and set(params["logit_bias"].values()) == {-0.5}
    # with the script ban on, both are sent in one map within OpenAI's 300 entries
    monkeypatch.setenv("BLOCK_OTHER_SCRIPTS", "1")
    monkeypatch.setenv("OUTPUT_LANGUAGE", "English")
    merged = client._logit_bias_params("openai", "gpt-4o", "https://api.openai.com/v1")["logit_bias"]
    assert merged[the] == -0.5 and -100 in merged.values() and len(merged) <= sbc.OPENAI_LOGIT_BIAS_LIMIT
    # never for other models / endpoints, nor with the switch, the presets or the strength off
    assert client._preset_logit_bias("openai", "gpt-5", "https://api.openai.com/v1") == {}
    assert client._preset_logit_bias("openai", "gpt-4o", "http://127.0.0.1:8080/v1") == {}
    monkeypatch.setenv("LOGIT_BIAS_STRENGTH", "0")
    assert client._preset_logit_bias("openai", "gpt-4o", "https://api.openai.com/v1") == {}
    monkeypatch.setenv("LOGIT_BIAS_STRENGTH", "1")
    monkeypatch.setenv("LOGIT_BIAS_ENABLED", "0")
    assert client._preset_logit_bias("openai", "gpt-4o", "https://api.openai.com/v1") == {}


def _scripted_responses(monkeypatch, client, answers, finish_reason="stop"):
    import unified_api_client as api
    seen = []

    def fake_get_response(messages, *args, **kwargs):
        seen.append([dict(m) for m in messages])
        return api.UnifiedResponse(content=answers[min(len(seen), len(answers)) - 1], finish_reason=finish_reason)

    monkeypatch.setattr(client, "_get_response", fake_get_response)
    return seen


MESSAGES = [{"role": "system", "content": "Translate to English."},
            {"role": "user", "content": "<p>그는 웃었다.</p>"}]


def test_a_leaking_answer_is_kept_and_the_log_says_so(client, monkeypatch, caplog):
    """Owner 2026-10-30: a retry with a language instruction only hides the problem; the request is never changed
    or sent again."""
    caplog.set_level("INFO")
    _turn_on(monkeypatch)
    monkeypatch.setenv("MAX_RETRIES", "3")
    seen = _scripted_responses(monkeypatch, client, ["<p>He 笑了起来 smiled.</p>", "<p>He smiled.</p>"])
    content, finish = client._send_internal(list(MESSAGES), temperature=0.3, max_tokens=1000, context="translation",
                                              request_id="sbtest01")
    assert content == "<p>He 笑了起来 smiled.</p>" and finish == "stop"
    assert len(seen) == 1 and seen[0][-1]["content"] == MESSAGES[-1]["content"]
    assert "characters of another script" in caplog.text


def test_a_truncated_answer_is_not_sent_again_for_a_leak(client, monkeypatch, caplog):
    caplog.set_level("INFO")
    _turn_on(monkeypatch)
    monkeypatch.setenv("MAX_RETRIES", "2")
    monkeypatch.setenv("RETRY_TRUNCATED", "0")
    seen = _scripted_responses(monkeypatch, client, ["<p>他笑了起来，然后转身离开了房间。</p>"], finish_reason="length")
    client._send_internal(list(MESSAGES), temperature=0.3, max_tokens=1000, context="translation",
                          request_id="sbtest01")
    assert len(seen) == 1
    assert "characters of another script" in caplog.text


def test_ollama_route_needs_only_the_block_switch(client, monkeypatch):
    """Ollama Settings has the same switch, and the route sends no anti-duplicate parameter."""
    monkeypatch.setenv("OUTPUT_LANGUAGE", "English")
    monkeypatch.setenv("BLOCK_OTHER_SCRIPTS", "1")
    assert client._script_ban_scripts("translation") == ()
    monkeypatch.setattr(client, "model", "ollamapull/qwen3.5:2b")
    assert client._script_ban_scripts("translation") == ENGLISH_BANS
    monkeypatch.setenv("BLOCK_OTHER_SCRIPTS", "0")
    assert client._script_ban_scripts("translation") == ()


def test_guard_is_off_by_default(client, monkeypatch):
    monkeypatch.setenv("OUTPUT_LANGUAGE", "English")
    monkeypatch.setenv("MAX_RETRIES", "3")
    seen = _scripted_responses(monkeypatch, client, ["<p>He 笑了起来.</p>"])
    content, _finish = client._send_internal(list(MESSAGES), temperature=0.3, max_tokens=1000, context="translation",
                                              request_id="sbtest01")
    assert content == "<p>He 笑了起来.</p>" and len(seen) == 1


def test_target_script_is_never_flagged(client, monkeypatch):
    _turn_on(monkeypatch, "Hindi")
    monkeypatch.setenv("MAX_RETRIES", "3")
    seen = _scripted_responses(monkeypatch, client, ["<p>वह हँसा।</p>"])
    content, _finish = client._send_internal(list(MESSAGES), temperature=0.3, max_tokens=1000, context="translation",
                                              request_id="sbtest01")
    assert content == "<p>वह हँसा।</p>" and len(seen) == 1


# ------------------------------------------------------------------------------------- settings

def test_setting_is_wired_through_desktop_and_schema():
    other = (SRC / "other_settings.py").read_text(encoding="utf-8")
    assert '"Block other scripts in translations (adaptive, all models)"' in other
    assert "Think before enabling" in other
    assert "self.block_other_scripts_var = self.config.get('block_other_scripts', False)" in \
        (SRC / "owner_state.py").read_text(encoding="utf-8")
    assert "'BLOCK_OTHER_SCRIPTS'" in (SRC / "run_env.py").read_text(encoding="utf-8")

    import settings_schema_data as data
    spec = data.SETTINGS["block_other_scripts"]
    assert spec["default"] is False and spec["type"] == "bool"
    assert spec["label"] == "Block other scripts in translations (adaptive, all models)"
    assert spec["tooltip"].splitlines()[0].startswith("⚠️ Mixed-language books")
    assert len(spec["tooltip"].splitlines()[0]) <= 140  # the mobile tile's help line shows it whole
    assert ("block_other_scripts", ("block_other_scripts_var",), ("value", False), ("bool",)) in \
        data.DESKTOP_SETTINGS_MAP


def test_new_module_is_in_every_spec():
    import ast
    for spec in sorted(SRC.glob("*.spec")):
        tree = ast.parse(spec.read_text(encoding="utf-8"))
        files, modules = None, None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id == "app_files":
                    files = ast.literal_eval(node.value)
                elif node.targets[0].id == "app_modules":
                    modules = ast.literal_eval(node.value)
        assert ("script_ban_core.py", ".") in files, spec.name
        assert "script_ban_core" in modules, spec.name
