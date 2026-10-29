"""v9.15.7 batch, issue 5 (owner 2026-10-28): Qwen3.5-2B on-device, Korean -> English, streamed Hindi
(Devanagari) lines; "Block CJK in translations (adaptive)" only banned Han / kana / Hangul. The adaptive
ban (``services.local_llm.script_ban``) now bans every script the target language does not use, keeps the
Prefs key, and the switch reads "Block other scripts in translations (adaptive)".

Run from src/mobile:
    .venv/Scripts/python.exe -m pytest -p no:cacheprovider -W ignore tests_host/test_mobile_batch_9157_script_ban.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
for entry in (str(MOBILE_DIR / "app"), str(MOBILE_DIR.parent)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

np = pytest.importorskip("numpy")

import local_llm_fakes as fakes  # noqa: E402
from glossarion_mobile.services.local_llm import downloads as dl  # noqa: E402
from glossarion_mobile.services.local_llm import engine as eng  # noqa: E402
from glossarion_mobile.services.local_llm import script_ban as sb  # noqa: E402
from glossarion_mobile.services.local_llm import service as svc  # noqa: E402
from language_options import TARGET_LANGUAGES  # noqa: E402

STEM = "Qwen3.5-2B-Q4_0"

#: what the owner's screenshot streamed, and the other drifts small models are known for
HINDI = "बंद हो चुका विचारों को फिर से एक बार"
FOREIGN = {
    "devanagari": HINDI, "thai": "สวัสดีครับ", "arabic": "مرحبا بالعالم", "cyrillic": "Привет, мир",
    "hebrew": "שלום עולם", "bengali": "আমি বাংলায়", "tamil": "வணக்கம்", "georgian": "გამარჯობა",
    "armenian": "Բարեւ", "ethiopic": "ሰላም", "myanmar": "မင်္ဂလာပါ", "khmer": "សួស្តី", "lao": "ສະບາຍດີ",
    "sinhala": "ආයුබෝවන්", "cjk": "中国 ひらがな カタカナ 한국어",
}
#: never banned for any target
ALWAYS_OK = ["Hello world!", "été — “quoted” …", "Ἀθῆναι α β Ω μ", "\U0001F600💕❤", "ﬁ ꞌ ẞ ơ ư", "→ ≤ ½ ™ ©"]

#: one sample per TARGET_LANGUAGES entry written in a non-Latin script
OWN_TEXT = {
    "Russian": "Привет", "Ukrainian": "Привіт", "Bulgarian": "Здравей", "Serbian": "Здраво", "Kazakh": "Сәлем",
    "Uzbek": "Салом", "Arabic": "مرحبا", "Urdu": "ہیلو", "Persian": "سلام", "Hindi": HINDI, "Marathi": "नमस्कार",
    "Nepali": "नमस्ते", "Bengali": "নমস্কার", "Punjabi": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ", "Gujarati": "નમસ્તે", "Tamil": "வணக்கம்",
    "Telugu": "నమస్కారం", "Kannada": "ನಮಸ್ಕಾರ", "Malayalam": "നമസ്കാരം", "Sinhala": "ආයුබෝවන්", "Thai": "สวัสดี",
    "Burmese": "မင်္ဂလာပါ", "Khmer": "សួស្តី", "Lao": "ສະບາຍດີ", "Hebrew": "שלום", "Greek": "Γειά σου",
    "Amharic": "ሰላም", "Chinese": "你好", "Chinese (Simplified)": "你好", "Chinese (Traditional)": "你好",
    "Japanese": "こんにちは 漢字 カタカナ", "Korean": "안녕하세요 漢字",
}


def _banned_text(text, scripts):
    return sb.token_is_banned(text.encode("utf-8"), scripts)


def test_scripts_table_is_sorted_disjoint_and_never_touches_latin_greek_or_punctuation():
    ranges = sorted(r for rs in sb.SCRIPTS.values() for r in rs)
    for (lo, hi), (next_lo, _next_hi) in zip(ranges, ranges[1:]):
        assert lo <= hi < next_lo, hex(lo)
    everything = tuple(sb.SCRIPTS)
    for cp in (list(range(0x0400)) + list(range(0x1D00, 0x2D00)) + list(range(0x2E00, 0x2E80))
               + list(range(0x1F000, 0x1FB00))):
        assert sb.script_of(cp) is None, hex(cp)
    for text in ALWAYS_OK:
        assert not _banned_text(text, everything), text


def test_english_target_bans_hindi_and_every_other_script():
    """The owner's case: Korean -> English bans Devanagari as well as the CJK scripts."""
    scripts = sb.banned_scripts("translation", "English")
    for name, text in FOREIGN.items():
        assert name in scripts and _banned_text(text, scripts), name
        for ch in text.replace(" ", "").replace(",", ""):
            assert _banned_text(ch, scripts), (name, ch)
    for text in ALWAYS_OK:
        assert not _banned_text(text, scripts), text
    assert sb.wants_ban("translation", "English")


@pytest.mark.parametrize("target", TARGET_LANGUAGES)
def test_every_listed_target_is_known_and_keeps_its_own_script(target):
    own = sb.target_scripts(target)
    assert own is not None, target
    scripts = sb.banned_scripts("translation", target)
    assert scripts and not own & set(scripts)
    if target in OWN_TEXT:
        assert not _banned_text(OWN_TEXT[target], scripts), target
    for text in ALWAYS_OK:  # Latin names, numbers and punctuation stay for every target
        assert not _banned_text(text, scripts), (target, text)
    if target not in ("Hindi", "Marathi", "Nepali"):
        assert _banned_text(HINDI, scripts), target


@pytest.mark.parametrize("target, keeps, bans", [
    ("Japanese", "cjk", "devanagari"), ("Korean", "cjk", "thai"), ("zh-TW", "cjk", "cyrillic"),
    ("Russian", "cyrillic", "cjk"), ("ru-RU", "cyrillic", "devanagari"), ("Hindi", "devanagari", "cjk"),
    ("हिन्दी", "devanagari", "arabic"), ("Русский", "cyrillic", "thai"), ("中文", "cjk", "devanagari"),
    ("Punjabi", "gurmukhi", "devanagari"), ("Punjabi", "arabic", "cjk"), ("Serbian (Latin)", "cyrillic", "cjk"),
    ("Bahasa Indonesia", None, "cjk"), ("Greek", None, "cyrillic"), ("thai", "thai", "lao"),
])
def test_targets_keep_their_scripts(target, keeps, bans):
    scripts = sb.banned_scripts("translation", target)
    if keeps:
        assert keeps not in scripts
    assert bans in scripts


def test_unknown_targets_fall_back_to_the_original_cjk_only_ban():
    for target in ("Other", "Klingon", "Elvish (Sindarin)"):
        assert sb.target_scripts(target) is None
        assert sb.banned_scripts("translation", target) == (sb.CJK,)
    # the adaptive request rules are unchanged
    assert sb.banned_scripts("glossary", "English") == ()
    assert sb.banned_scripts("translation", "English", single_pass=True) == ()
    assert sb.banned_scripts("translation", "English", enabled=False) == ()
    assert sb.banned_scripts("translation", "") == ()
    assert sb.banned_scripts("image_ocr", "English") == ()


@pytest.mark.parametrize("piece, english, hindi, russian", [
    (b"\xe0", True, False, True),          # U+0800-0FFF: Arabic ext, Indic, Thai, Lao, Tibetan
    (b"\xe0\xa4", True, False, True),      # Devanagari only
    (b" \xe0\xa4", True, False, True),
    ("ह".encode() + b"\xe0", True, False, True),  # a complete Devanagari letter
    (b"\xd0", True, True, False),          # Cyrillic
    (b"\xd8", True, True, True),           # Arabic
    (b"\xce", False, False, False),        # Greek
    (b"\xe1", False, False, False),        # also Latin Extended Additional / Greek Extended
    (b"\xe2", False, False, False),        # punctuation
    (b"\xf0", False, False, False),        # emoji
    (b"\xe4\xb8", True, True, True),       # Han
])
def test_byte_fragments_are_banned_only_when_every_completion_is(piece, english, hindi, russian):
    for target, banned in (("English", english), ("Hindi", hindi), ("Russian", russian)):
        assert sb.token_is_banned(piece, sb.banned_scripts("translation", target)) is banned, (piece, target)


VOCAB = [b"Hello", b" world", "été".encode(), "—“…".encode(), "\U0001F495".encode(), b"<|im_end|>", b"",
         HINDI.encode(), " विचार".encode(), "ा".encode(), b"\xe0\xa4", "中".encode(), "한국어".encode(),
         "สวัสดี".encode(), "Привет".encode(), "مرحبا".encode(), "שלום".encode(), b"\xe0", b"\xce"]


class _FakeLlama:
    """What LlamaEngine uses of llama_cpp.Llama (as in test_local_llm_script_ban)."""

    def __init__(self):
        self.kwargs = []
        self.detokenized = 0

    def n_vocab(self):
        return len(VOCAB)

    def detokenize(self, tokens, special=False):
        self.detokenized += 1
        return VOCAB[tokens[0]]

    def create_chat_completion(self, **kwargs):
        self.kwargs.append(kwargs)
        scores = np.zeros(len(VOCAB), dtype=np.single)
        for processor in kwargs.get("logits_processor") or []:
            scores = processor(np.array([], dtype=np.intc), scores)
        allowed = [i for i in range(len(VOCAB)) if not np.isneginf(scores[i])]
        yield {"choices": [{"delta": {"content": ",".join(map(str, allowed))}, "finish_reason": None}]}
        yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}

    def close(self):
        pass


def _allowed(engine, params):
    out = "".join(engine.chat([{"role": "user", "content": "안녕"}], {"max_tokens": 8, **params}, lambda: False))
    return [VOCAB[int(i)] for i in out.split(",") if i]


def test_service_and_engine_ban_hindi_for_english_and_keep_it_for_hindi(tmp_path):
    service = svc.LocalEngineService(prefs=fakes.FakePrefs(), registry=dl.Registry(str(tmp_path / "models")),
                                     engine_factory=eng.FakeEngine, watch=False, memory=lambda: (0, None))
    base = {"max_tokens": 64, "glossarion_task": "translation"}
    english = service._params({**base, "target_language": "English"}, 4096)
    hindi = service._params({**base, "target_language": "Hindi"}, 4096)
    assert "devanagari" in english["ban_scripts"] and "devanagari" not in hindi["ban_scripts"]

    engine = eng.LlamaEngine()
    engine.llm, engine.path, engine.n_ctx = _FakeLlama(), "x.gguf", 4096
    allowed = _allowed(engine, {"ban_scripts": english["ban_scripts"]})
    assert allowed == [b"Hello", b" world", "été".encode(), "—“…".encode(), "\U0001F495".encode(),
                       b"<|im_end|>", b"", b"\xce"]
    scanned = engine.llm.detokenized
    assert scanned == len(VOCAB)
    allowed = _allowed(engine, {"ban_scripts": hindi["ban_scripts"]})
    assert HINDI.encode() in allowed and " विचार".encode() in allowed and b"\xe0\xa4" in allowed
    assert b"\xe0" in allowed  # can still become Devanagari
    assert "中".encode() not in allowed and "Привет".encode() not in allowed
    first = engine.llm.kwargs[-1]["logits_processor"][0]
    _allowed(engine, {"ban_scripts": hindi["ban_scripts"]})
    assert engine.llm.kwargs[-1]["logits_processor"][0] is first  # one processor per script set
    assert engine.llm.detokenized == scanned  # the vocabulary is decoded once per loaded model
    engine.unload()
    assert engine._scripts is None


def test_vocab_scripts_matches_the_per_token_rule_and_caches():
    vocab = sb.VocabScripts(iter(VOCAB))
    assert vocab.n_vocab == len(VOCAB)
    for target in ("English", "Hindi", "Japanese", "Russian", "Other"):
        scripts = sb.banned_scripts("translation", target)
        assert vocab.ids(scripts) == sb.banned_token_ids(VOCAB, scripts), target
        assert vocab.processor(scripts) is vocab.processor(list(scripts))


def test_switch_is_relabelled_and_keeps_its_prefs_key(tmp_path):
    pytest.importorskip("flet")
    from glossarion_mobile.ui.screens.local_models_panel import LocalModelsPanel

    prefs = fakes.FakePrefs()
    prefs.set(svc.BLOCK_CJK_KEY, False)  # a choice saved under the old label survives
    service = svc.LocalEngineService(prefs=prefs, registry=dl.Registry(str(tmp_path / "models")),
                                     engine_factory=eng.FakeEngine, watch=False, memory=lambda: (0, None))
    assert svc.BLOCK_CJK_KEY == "local_llm_block_cjk"
    panel = LocalModelsPanel(service=service, prefs=prefs, current_model=lambda: None)
    panel.build()
    # the label wraps beside the switch (switch_row) rather than being the Switch's own label
    import flet as ft

    row = next(c for c in panel._sampling_section() if isinstance(c, ft.Row) and c.controls
               and isinstance(c.controls[0], ft.Switch) and c.controls[0].key == "local-block-cjk")
    assert row.controls[1].content.value == "Block other scripts in translations (adaptive)"
    assert panel.block_cjk_switch.value is False
    panel.dispose()
