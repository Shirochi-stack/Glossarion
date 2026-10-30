"""Opt-in emoticon exemption of the adaptive script ban, "Allow emoticons and symbols from other scripts
(kaomoji)" (owner 2026-10-28; ``src/script_ban_core.py``, config ``script_ban_allow_kaomoji`` / env
``SCRIPT_BAN_ALLOW_KAOMOJI``, off by default on desktop and mobile).

Covers the shared core (every kaomoji character the owner listed is blocked for English with the option
off and allowed with it on, either as exempt punctuation or as an emoticon character of the source;
Hindi / Korean prose stays blocked with it on; byte tails), the source reading on a Korean chapter, the
UnifiedClient guard / logit bias / on-device body field and the settings plumbing.

Run (repository root)::

    py -3.12 -m pytest -p no:cacheprovider -W ignore -q tests/test_script_ban_kaomoji.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.environ.setdefault("GLOSSARION_HTTP_LOG", "0")

import script_ban_core as sbc  # noqa: E402

ENGLISH = sbc.banned_scripts("translation", "English")

#: the owner's examples (2026-10-28)
EXEMPT = ["・", "｡", "）", "︵", "「", "」", "『", "』", "ー", "、", "〜", "（", "～", "￣", "ﾟ"]
FROM_SOURCE = ["ツ", "ノ", "ಠ", "٩", "۶"]
KAOMOJI = ["¯\\_(ツ)_/¯", "(ノ°▽°)ノ", "ಠ_ಠ", "٩(◕‿◕)۶", "(╯°□°）╯︵ ┻━┻", "「ok」", "(・_・)", "(ﾟ∀ﾟ)"]
HINDI = "बंद हो चुका विचारों को फिर से"
KOREAN = "그는 조용히 웃었다. 그리고 다시 걸었다."


def _banned(text, allow, scripts=ENGLISH):
    return sbc.token_is_banned(text.encode("utf-8"), scripts, allow_kaomoji=allow)


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    for name in ("HOME", "USERPROFILE", "GLOSSARION_LIBRARY_DIR", "OUTPUT_DIRECTORY", "GLOSSARION_DATA_DIR"):
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    for name in ("ENABLE_ANTI_DUPLICATE", "BLOCK_OTHER_SCRIPTS", "SCRIPT_BAN_ALLOW_KAOMOJI", "OUTPUT_LANGUAGE",
                 "AUTO_GLOSSARY_MODE", "SINGLE_PASS_GLOSSARY_MODE", "RETRY_TRUNCATED", "SYSTEM_PROMPT_TO_USER"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


# ------------------------------------------------------------------------------------ the table

def test_kaomoji_ranges_are_sorted_disjoint_and_inside_the_cjk_ban():
    ranges = sbc.KAOMOJI_RANGES
    for (lo, hi), (next_lo, _hi) in zip(ranges, ranges[1:]):
        assert lo <= hi < next_lo
    for lo, hi in ranges:
        for cp in range(lo, hi + 1):
            assert sbc.script_of(cp) == sbc.CJK, hex(cp)


@pytest.mark.parametrize("ch", EXEMPT)
def test_exempt_punctuation_is_blocked_off_and_allowed_on(ch):
    assert _banned(ch, False)
    assert not _banned(ch, True)
    assert sbc.is_kaomoji_char(ord(ch))


@pytest.mark.parametrize("ch", ["々", "〆", "〇", "〻", "０", "９", "Ａ", "ｚ", "ｱ", "ﾝ", "ﾡ", "あ", "ア", "中", "한"])
def test_language_characters_stay_blocked_with_the_option_on(ch):
    """Iteration / numeral marks, full-width digits and letters, half-width kana / Hangul, kana, Han and
    Hangul carry language: blocked either way."""
    assert _banned(ch, True)
    assert not sbc.is_kaomoji_char(ord(ch))


@pytest.mark.parametrize("ch", FROM_SOURCE)
def test_letters_of_other_scripts_need_the_source(ch):
    """ツ ノ ಠ ٩ ۶ are letters / digits of other scripts: still blocked as such with the option on; they
    pass on device when the source uses them in emoticons (``kaomoji_source_chars``)."""
    assert _banned(ch, False) and _banned(ch, True)
    assert ch in sbc.kaomoji_source_chars(f"He shrugged {KAOMOJI_SOURCE}", ENGLISH)


KAOMOJI_SOURCE = "그는 어깨를 으쓱했다. ¯\\_(ツ)_/¯ (ノ°▽°)ノ ಠ_ಠ ٩(◕‿◕)۶ 다시 걸었다."


@pytest.mark.parametrize("face", KAOMOJI)
def test_whole_kaomoji_off_vs_on(face):
    assert _banned(face, False)
    letters = [ch for ch in face if sbc.is_banned_char(ord(ch), ENGLISH, allow_kaomoji=True)]
    assert _banned(face, True) == bool(letters)
    assert set(letters) <= set(FROM_SOURCE)


def test_prose_stays_blocked_with_the_option_on():
    for text in (HINDI, KOREAN, "他笑了起来", "ひらがな カタカナ", "Привет"):
        assert _banned(text, True), text
        for ch in text.replace(" ", "").replace(".", ""):
            assert _banned(ch, True), (text, ch)


@pytest.mark.parametrize("piece, off, on", [
    (b"\xe3", True, True),            # U+3000-3FFF: 4,096 kana / CJK code points stay blocked
    (b"\xe3\x80", True, False),       # U+3000-303F: CJK punctuation
    (b"\xe3\x83", True, False),       # U+30C0-30FF: holds ・ and ー
    (b"\xe3\x81", True, True),        # U+3040-307F: hiragana only
    (b"\xe4\xb8", True, True),        # Han
    (b"\xef\xbc", True, False),       # U+FF00-FF3F: full-width forms, with （）！ among them
    (b"\xef", False, False),          # U+F000-FFFF: private use too, never banned
    (b"\xe0\xa4", True, True),        # Devanagari
])
def test_byte_tails(piece, off, on):
    assert sbc.token_is_banned(piece, ENGLISH) is off
    assert sbc.token_is_banned(piece, ENGLISH, allow_kaomoji=True) is on


def test_off_is_exactly_the_old_rule():
    pieces = [ch.encode() for ch in EXEMPT + FROM_SOURCE] + [HINDI.encode(), b"\xe3\x80", b"Hello"]
    for piece in pieces:
        assert sbc.token_masks(piece) == sbc.token_masks(piece, False)
    assert sbc.banned_token_ids(pieces, ENGLISH) == sbc.banned_token_ids(pieces, ENGLISH, allow_kaomoji=False)


# ----------------------------------------------------------------------------- reading the source

def test_source_chars_pick_emoticons_but_never_prose():
    got = sbc.kaomoji_source_chars(KAOMOJI_SOURCE, ENGLISH)
    assert got == frozenset(FROM_SOURCE)
    # Korean prose, its hanja and a lone syllable next to punctuation are never picked
    prose = "성(性)녀 입니다. 쾅―!! 네~ 5년 TV를 흣♡ 그 말"
    assert sbc.kaomoji_source_chars(prose, ENGLISH) == frozenset()
    # a Japanese source writes kana words: its ツ is a word letter
    assert "ツ" not in sbc.kaomoji_source_chars("ツールを使った。¯\\_(ツ)_/¯", ENGLISH)
    # Hindi prose with a Kannada face: Devanagari never, ಠ yes
    assert sbc.kaomoji_source_chars(f"{HINDI} ಠ_ಠ", ENGLISH) == frozenset({"ಠ"})
    # a script the target keeps is not banned, so it is never picked
    assert sbc.kaomoji_source_chars("ಠ_ಠ", sbc.banned_scripts("translation", "Kannada")) == frozenset()
    assert sbc.kaomoji_source_chars("plain ascii (^_^)", ENGLISH) == frozenset()
    assert sbc.kaomoji_source_chars([{"role": "user", "content": "ಠ_ಠ"}], ENGLISH) == frozenset({"ಠ"})


def test_source_keys_cover_byte_spelled_characters():
    keys = sbc.source_keys({"ಠ", "٩"})
    assert keys == {"ಠ", b"\xe0\xb2", "٩", b"\xd9"}  # never the lone lead byte E0 (all Indic scripts)
    assert sbc.token_requirements("ಠ_".encode()) == (("ಠ", sbc._BITS["kannada"], False),)
    tail = sbc.token_requirements(b"_\xe0\xb2")
    assert tail and tail[0][0] == b"\xe0\xb2" and tail[0][2] is True
    assert sbc.token_requirements("「」".encode()) == ()  # exempt
    assert sbc.token_requirements("「」".encode(), False)


def test_reading_a_long_source_is_cheap():
    text = (KOREAN + " 그녀는 대답하지 않았다. ") * 400 + KAOMOJI_SOURCE  # ~18k characters
    sbc.kaomoji_source_chars(text, ENGLISH)  # compile the patterns
    started = time.perf_counter()
    for _ in range(10):
        got = sbc.kaomoji_source_chars(text, ENGLISH)
    per_call = (time.perf_counter() - started) / 10
    assert got == frozenset(FROM_SOURCE)
    assert per_call < 0.05, per_call  # measured ~1 ms on the desktop; generous for CI runners


# ------------------------------------------------------------------------------- output guard

def test_guard_ignores_exempt_punctuation_only_when_on():
    answer = "<p>He shrugged 「fine」・・・ (・_・) ︵ ｡ ｡ ｡</p>"
    assert sbc.foreign_script_leak(answer, ENGLISH)
    assert sbc.foreign_script_leak(answer, ENGLISH, allow_kaomoji=True) is None
    # letters of other scripts still count (unless the request has them)
    leaked = "<p>He said 他笑了起来 「fine」</p>"
    assert sbc.foreign_script_leak(leaked, ENGLISH, allow_kaomoji=True)["chars"] == 5
    face = "<p>He shrugged ¯\\_(ツ)_/¯ ಠ_ಠ ツツ</p>"
    assert sbc.foreign_script_leak(face, ENGLISH, "¯\\_(ツ)_/¯ ಠ_ಠ ツツ", allow_kaomoji=True) is None


def test_logit_bias_leaves_exempt_tokens_out():
    tiktoken = pytest.importorskip("tiktoken")
    try:
        enc = tiktoken.get_encoding("o200k_base")
    except Exception as exc:
        pytest.skip(f"tiktoken encoding unavailable offline: {exc}")
    off = sbc.openai_logit_bias("gpt-4o", ENGLISH)
    on = sbc.openai_logit_bias("gpt-4o", ENGLISH, allow_kaomoji=True)
    assert 0 < len(on) <= sbc.OPENAI_LOGIT_BIAS_LIMIT
    exempt = [key for key in off if not sbc.token_is_banned(enc.decode_single_token_bytes(int(key)), ENGLISH,
                                                              allow_kaomoji=True)]
    assert exempt  # the most frequent banned tokens include CJK punctuation
    for key in on:
        assert sbc.token_is_banned(enc.decode_single_token_bytes(int(key)), ENGLISH, allow_kaomoji=True)
    assert not set(exempt) & set(on)


# ------------------------------------------------------------------------------- UnifiedClient

@pytest.fixture
def client(isolated):
    import unified_api_client as api
    return api.UnifiedClient(api_key="sk-test", model="gpt-4o", output_dir=str(isolated / "out"))


def _scripted(monkeypatch, client, answers):
    import unified_api_client as api
    seen = []

    def fake_get_response(messages, *args, **kwargs):
        seen.append([dict(m) for m in messages])
        return api.UnifiedResponse(content=answers[min(len(seen), len(answers)) - 1], finish_reason="stop")

    monkeypatch.setattr(client, "_get_response", fake_get_response)
    return seen


MESSAGES = [{"role": "system", "content": "Translate to English."},
            {"role": "user", "content": "<p>그는 웃었다.</p>"}]
FACE_ANSWER = "<p>He smiled 「quietly」・・・ ︵ ｡</p>"


def _guarded(monkeypatch):
    monkeypatch.setenv("ENABLE_ANTI_DUPLICATE", "1")
    monkeypatch.setenv("BLOCK_OTHER_SCRIPTS", "1")
    monkeypatch.setenv("OUTPUT_LANGUAGE", "English")
    monkeypatch.setenv("MAX_RETRIES", "3")


def test_client_warns_about_kaomoji_punctuation_while_off(client, monkeypatch, caplog):
    caplog.set_level("INFO")
    _guarded(monkeypatch)
    seen = _scripted(monkeypatch, client, [FACE_ANSWER, "<p>He smiled.</p>"])
    content, _finish = client._send_internal(list(MESSAGES), temperature=0.3, max_tokens=1000,
                                              context="translation", request_id="kaotest1")
    assert len(seen) == 1 and content == FACE_ANSWER  # kept, never sent again
    assert "characters of another script" in caplog.text


def test_client_accepts_kaomoji_punctuation_while_on(client, monkeypatch, caplog):
    caplog.set_level("INFO")
    _guarded(monkeypatch)
    monkeypatch.setenv("SCRIPT_BAN_ALLOW_KAOMOJI", "1")
    seen = _scripted(monkeypatch, client, [FACE_ANSWER, "<p>He smiled.</p>"])
    content, _finish = client._send_internal(list(MESSAGES), temperature=0.3, max_tokens=1000,
                                              context="translation", request_id="kaotest1")
    assert len(seen) == 1 and content == FACE_ANSWER
    assert "characters of another script" not in caplog.text
    # Chinese still is a leak with the option on
    seen = _scripted(monkeypatch, client, ["<p>He 笑了起来 smiled.</p>", "<p>He smiled.</p>"])
    client._send_internal(list(MESSAGES), temperature=0.3, max_tokens=1000, context="translation",
                          request_id="kaotest2")
    assert len(seen) == 1 and "characters of another script" in caplog.text


def test_client_logit_bias_and_on_device_field_follow_the_setting(client, monkeypatch):
    _guarded(monkeypatch)
    calls = []

    def fake_bias(model, scripts, limit=sbc.OPENAI_LOGIT_BIAS_LIMIT, **kwargs):
        calls.append(kwargs)
        return {"1": -100}

    monkeypatch.setattr(sbc, "openai_logit_bias", fake_bias)
    client.context = "translation"
    client._script_ban_logit_bias("openai", "gpt-4o", "https://api.openai.com/v1")
    assert "glossarion_allow_kaomoji" not in client._on_device_task_fields()
    monkeypatch.setenv("SCRIPT_BAN_ALLOW_KAOMOJI", "1")
    client._script_ban_logit_bias("openai", "gpt-4o", "https://api.openai.com/v1")
    assert calls == [{}, {"allow_kaomoji": True}]
    assert client._on_device_task_fields()["glossarion_allow_kaomoji"] is True


# ------------------------------------------------------------------------------------- settings

def test_setting_is_wired_through_desktop_and_schema():
    other = (SRC / "other_settings.py").read_text(encoding="utf-8")
    label = "Allow emoticons and symbols from other scripts (kaomoji)"
    assert f'"{label}"' in other
    assert other.index('"Block other scripts in translations (adaptive, all models)"') < other.index(f'"{label}"')
    assert "self.script_ban_allow_kaomoji_var = False" in other  # reset to defaults
    assert "self.script_ban_allow_kaomoji_var = self.config.get('script_ban_allow_kaomoji', False)" in \
        (SRC / "owner_state.py").read_text(encoding="utf-8")
    run_env = (SRC / "run_env.py").read_text(encoding="utf-8")
    assert run_env.count("SCRIPT_BAN_ALLOW_KAOMOJI") == 3

    import settings_schema_data as data
    spec = data.SETTINGS["script_ban_allow_kaomoji"]
    assert spec["default"] is False and spec["type"] == "bool" and spec["label"] == label
    assert spec["ui_tab"] == "Logit Bias"
    assert "ಠ_ಠ" in spec["tooltip"] and "slip by" in spec["tooltip"]
    assert {env for env, _phase, _x in spec["env"]} == {"SCRIPT_BAN_ALLOW_KAOMOJI"}
    assert ("script_ban_allow_kaomoji", ("script_ban_allow_kaomoji_var",), ("value", False), ("bool",)) in \
        data.DESKTOP_SETTINGS_MAP
    order = list(data.SETTINGS_MAP_ORDER)
    assert order.index("script_ban_allow_kaomoji") == order.index("block_other_scripts") + 1


def test_run_env_exports_the_variable_next_to_block_other_scripts():
    source = (SRC / "run_env.py").read_text(encoding="utf-8").replace("\r\n", "\n")
    block = "'BLOCK_OTHER_SCRIPTS': '1' if getattr(self, 'block_other_scripts_var', False) else '0',\n"
    line = "'SCRIPT_BAN_ALLOW_KAOMOJI': '1' if getattr(self, 'script_ban_allow_kaomoji_var', False) else '0',"
    assert block + "            " + line in source
    assert "('SCRIPT_BAN_ALLOW_KAOMOJI', '1' if getattr(self, 'script_ban_allow_kaomoji_var', False) else '0')," \
        in source
