"""Universal adaptive script ban, "Block other scripts in translations (adaptive, all models)" (owner
2026-10-28; ``src/script_ban_core.py``, config ``block_other_scripts`` / env ``BLOCK_OTHER_SCRIPTS``).

Covers the shared decision (the target's own script is never banned, unknown targets ban CJK only,
glossary / single-pass never ban), the output guard (kept names from the request are ignored, small
amounts are tolerated), the OpenAI logit bias (official chat models only, capped, safe ids), the
UnifiedClient wiring (off by default and with Anti-Duplicate off; a leaking answer is sent again with
the instruction on the client's own retry budget, the last answer is kept when the budget runs out) and
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


def test_retry_instruction_names_the_target_and_the_scripts():
    note = sbc.retry_instruction("English", ("cjk",))
    assert "English" in note and "Chinese, Japanese or Korean" in note


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
    client.context = "translation"
    assert client._script_ban_logit_bias("openai", "gpt-4o", "https://api.openai.com/v1") == {"logit_bias": {"1": -100}}
    assert client._script_ban_logit_bias("openai", "gpt-4o", "https://my-proxy.example/v1") == {}
    assert client._script_ban_logit_bias("deepseek", "deepseek-chat", "https://api.deepseek.com/v1") == {}
    assert client._script_ban_logit_bias("openai", "gpt-4o", "https://api.openai.com/v1", True) == {}


def _scripted_responses(monkeypatch, client, answers):
    import unified_api_client as api
    seen = []

    def fake_get_response(messages, *args, **kwargs):
        seen.append([dict(m) for m in messages])
        return api.UnifiedResponse(content=answers[min(len(seen), len(answers)) - 1], finish_reason="stop")

    monkeypatch.setattr(client, "_get_response", fake_get_response)
    return seen


MESSAGES = [{"role": "system", "content": "Translate to English."},
            {"role": "user", "content": "<p>그는 웃었다.</p>"}]


def test_leaking_answer_is_sent_again_with_the_instruction(client, monkeypatch):
    _turn_on(monkeypatch)
    monkeypatch.setenv("MAX_RETRIES", "3")
    seen = _scripted_responses(monkeypatch, client, ["<p>He 笑了起来 smiled.</p>", "<p>He smiled.</p>"])
    content, finish = client._send_internal(list(MESSAGES), temperature=0.3, max_tokens=1000, context="translation",
                                              request_id="sbtest01")
    assert content == "<p>He smiled.</p>" and finish == "stop"
    assert len(seen) == 2
    assert "Write the whole answer in English only" in seen[1][-1]["content"]
    assert "Write the whole answer" not in seen[0][-1]["content"]


def test_leak_guard_keeps_the_last_answer_when_the_budget_runs_out(client, monkeypatch):
    _turn_on(monkeypatch)
    monkeypatch.setenv("MAX_RETRIES", "2")
    seen = _scripted_responses(monkeypatch, client, ["<p>He 笑了起来.</p>"])
    content, _finish = client._send_internal(list(MESSAGES), temperature=0.3, max_tokens=1000, context="translation",
                                              request_id="sbtest01")
    assert content == "<p>He 笑了起来.</p>"
    assert len(seen) == 2


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
    assert spec["tooltip"].splitlines()[0].startswith("⚠️ Can cause errors or retries")
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
