"""Adaptive CJK ban of the on-device engine (owner 2026-10-19, ``services.local_llm.script_ban``): which
requests ban (translation-like kinds into a non-CJK target; never glossary, single-pass translation or
CJK targets), which token ids (byte-level BPE fragments included), the -inf processor, the engine passing
it to llama-cpp-python, and the Model manager switch.

Run from src/mobile:
    .venv/Scripts/python.exe -m pytest -p no:cacheprovider -W ignore tests_host/test_local_llm_script_ban.py
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

STEM = "Qwen3.5-0.8B-UD-Q4_K_XL"

#: a byte-level BPE vocabulary in miniature: (piece, banned?)
VOCAB = [
    (b"Hello", False), (b" world", False), (b"!", False), (b"\xc3\xa9t\xc3\xa9", False),  # été
    ("—“…".encode(), False),  # — “ …
    ("Привет".encode(), False), ("Ἀθῆναι".encode(), False), ("ꞌ".encode(), False),  # Latin Extended-D
    ("\U0001F600".encode(), False),  # emoji
    ("中".encode(), True), (" 中国".encode(), True), ("龍".encode(), True), ("㐀".encode(), True),  # Ext A
    ("日本".encode(), True), ("ひらがな".encode(), True), ("カタカナ".encode(), True),
    ("한국어".encode(), True), (" 그".encode(), True), ("ᄀ".encode(), True),  # Hangul jamo
    ("。".encode(), True), ("「".encode(), True), ("，".encode(), True), ("ｶ".encode(), True),
    ("\U00020000".encode(), True),  # Extension B
    (b"\xe4", True), (b"\xe9", True), (b"\xe3", True), (b"\xeb", True), (b"\xec", True), (b"\xed", True),
    (b" \xec", True), (b"\xe4\xb8", True), (b"\xea\xb0", True), (b"\xf0\xa0", True), (b"\xef\xbc", True),
    (b"\xea", False), (b"\xef", False), (b"\xf0", False), (b"\xe2", False), (b"\xc3", False),  # shared leads
    (b"\xb8\xad", False), (b"\x80", False),  # continuation bytes only
    (b"\xef\xbf", False),  # FFC0-FFFF: also the replacement character
    (b"<|im_end|>", False), (b"", False),
]


def _service(tmp_path, engine=None):
    engine = engine or eng.FakeEngine()
    service = svc.LocalEngineService(prefs=fakes.FakePrefs(), registry=dl.Registry(str(tmp_path / "models")),
                                     engine_factory=lambda: engine, watch=False, memory=lambda: (0, None))
    service.registry.add(STEM, {"repo": "unsloth/Qwen3.5-0.8B-GGUF", "file": f"{STEM}.gguf", "size": 558772480})
    Path(service.registry.file_path(STEM)).write_bytes(b"gguf")
    return service, engine


def _expected():
    return [i for i, (_piece, banned) in enumerate(VOCAB) if banned]


def test_banned_ids_include_byte_fragments_and_never_latin_or_punctuation():
    pieces = [piece for piece, _banned in VOCAB]
    assert sb.banned_token_ids(pieces) == _expected()
    for piece, banned in VOCAB:
        assert sb.token_is_banned(piece) is banned, piece
    # nothing below U+1100 and no general punctuation is a banned character
    assert not any(sb.is_banned_char(cp) for cp in range(0x1100))
    assert not any(sb.is_banned_char(cp) for cp in range(0x2000, 0x2070))


@pytest.mark.parametrize("target, banned", [
    ("English", True), ("Spanish", True), ("Vietnamese", True), ("Thai", True),
    ("Chinese", False), ("Chinese (Simplified)", False), ("Chinese (Traditional)", False),
    ("Cantonese", False), ("Mandarin", False), ("zh-TW", False), ("中文", False), ("繁體中文", False),
    ("Japanese", False), ("日本語", False), ("ja", False),
    ("Korean", False), ("한국어", False), ("ko-KR", False),
])
def test_cjk_targets_ban_nothing_others_ban_all_three_scripts(target, banned):
    """owner 2026-10-19: a Chinese / Japanese / Korean target gets the default logits (no processor);
    every other target bans Han, kana and Hangul alike."""
    assert sb.wants_ban("translation", target) is banned
    assert sb.is_cjk_target(target) is (not banned)


def test_request_kinds():
    for task in ("translation", "refinement", "metadata", "book_title", "batch_header_translation",
                 "batch_toc_translation", "image_translation", "summary"):
        assert sb.wants_ban(task, "English"), task
    for task in ("glossary", "glossary_refinement", "auto_glossary_(3/10)", "review", "image_ocr", "vision_ocr",
                 "manga_ocr", "image_scan", "truncation", "qa_truncation", "other", "", None):
        assert not sb.wants_ban(task, "English"), task
    # the single-pass auto glossary's answer carries raw source names: its translations keep CJK
    assert not sb.wants_ban("translation", "English", single_pass=True)
    assert sb.wants_ban("metadata", "English", single_pass=True)
    assert not sb.wants_ban("translation", "English", enabled=False)
    assert not sb.wants_ban("translation", "")  # no target given: never guess


def test_processor_sets_minus_infinity_only_for_banned_ids():
    processor = sb.BanLogitsProcessor([3, 1, 3, 99], n_vocab=6)
    assert list(processor.ids) == [1, 3] and len(processor) == 2
    scores = np.arange(6, dtype=np.single)
    out = processor(np.array([0], dtype=np.intc), scores)
    assert np.isneginf(out[[1, 3]]).all()
    assert list(out[[0, 2, 4, 5]]) == [0.0, 2.0, 4.0, 5.0]


def test_service_routes_the_request_kind(tmp_path):
    service, _engine = _service(tmp_path)
    base = {"temperature": 0.7, "max_tokens": 64, "target_language": "English"}
    assert service.block_cjk() is True  # on by default
    assert service._params({**base, "glossarion_task": "translation"}, 4096)["ban_cjk"] is True
    assert "ban_cjk" not in service._params({**base, "glossarion_task": "glossary"}, 4096)
    assert "ban_cjk" not in service._params({**base, "glossarion_task": "translation",
                                             "glossarion_single_pass": True}, 4096)
    assert "ban_cjk" not in service._params({**base, "glossarion_task": "translation",
                                             "target_language": "Japanese"}, 4096)
    assert "ban_cjk" not in service._params({"max_tokens": 64}, 4096)  # a plain OpenAI request (no fields)
    assert service.save_block_cjk(False) is False and service.block_cjk() is False
    assert "ban_cjk" not in service._params({**base, "glossarion_task": "translation"}, 4096)


def test_the_ban_reaches_the_engine_through_the_endpoint_path(tmp_path):
    service, engine = _service(tmp_path)
    body = {"messages": [{"role": "user", "content": "안녕"}], "glossarion_task": "translation",
            "target_language": "English", "max_tokens": 16}
    run = service.chat(f"local/{STEM}", body, lambda: False)
    list(run.deltas)
    run.close()
    assert engine.calls[-1]["params"]["ban_cjk"] is True
    run = service.chat(f"local/{STEM}", {**body, "glossarion_task": "glossary"}, lambda: False)
    list(run.deltas)
    run.close()
    assert "ban_cjk" not in engine.calls[-1]["params"]


class _FakeLlama:
    """What LlamaEngine uses of llama_cpp.Llama: the vocabulary, tokenize and a streaming chat."""

    def __init__(self):
        self.kwargs = []
        self.detokenized = 0

    def n_vocab(self):
        return len(VOCAB)

    def detokenize(self, tokens, special=False):
        self.detokenized += 1
        return VOCAB[tokens[0]][0]

    def tokenize(self, text, add_bos=False, special=True):
        return list(range(len(text.split())))

    def create_chat_completion(self, **kwargs):
        self.kwargs.append(kwargs)
        processors = kwargs.get("logits_processor") or []
        scores = np.zeros(len(VOCAB), dtype=np.single)
        for processor in processors:
            scores = processor(np.array([], dtype=np.intc), scores)
        best = [i for i in range(len(VOCAB)) if not np.isneginf(scores[i])]
        yield {"choices": [{"delta": {"content": f"{len(best)} allowed"}, "finish_reason": None}]}
        yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}

    def close(self):
        pass


def test_llama_engine_passes_the_processor_and_computes_the_ids_once():
    engine = eng.LlamaEngine()
    engine.llm, engine.path, engine.n_ctx = _FakeLlama(), "x.gguf", 4096
    messages = [{"role": "user", "content": "안녕하세요"}]
    params = {"max_tokens": 16, "temperature": 0.7}
    assert "".join(engine.chat(messages, params, lambda: False)) == f"{len(VOCAB)} allowed"
    assert "logits_processor" not in engine.llm.kwargs[-1]  # no ban asked: the default logits
    out = "".join(engine.chat(messages, {**params, "ban_cjk": True}, lambda: False))
    assert out == f"{len(VOCAB) - len(_expected())} allowed"
    (processor,) = engine.llm.kwargs[-1]["logits_processor"]
    assert list(processor.ids) == _expected()
    detokenized = engine.llm.detokenized
    "".join(engine.chat(messages, {**params, "ban_cjk": True}, lambda: False))
    assert engine.llm.detokenized == detokenized  # cached for the loaded model
    engine.unload()
    assert engine._ban is None


def test_llama_engine_hands_the_dry_settings_to_the_model():
    """create_chat_completion has no DRY parameters: the engine sets ``dry_sampling`` on the model for the
    patched _init_sampler (patches/mobile.patch 9.); the frequency penalty goes as a keyword."""
    engine = eng.LlamaEngine()
    engine.llm, engine.path, engine.n_ctx = _FakeLlama(), "x.gguf", 4096
    messages = [{"role": "user", "content": "hi"}]
    params = {"max_tokens": 16, "frequency_penalty": 0.3, "dry_multiplier": 0.8, "dry_base": 1.75,
              "dry_allowed_length": 2}
    "".join(engine.chat(messages, params, lambda: False))
    assert engine.llm.kwargs[-1]["frequency_penalty"] == 0.3
    assert engine.llm.dry_sampling == {"multiplier": 0.8, "base": 1.75, "allowed_length": 2,
                                       "penalty_last_n": -1, "seq_breakers": ["\n", ":", '"', "*"]}
    "".join(engine.chat(messages, {"max_tokens": 16, "dry_multiplier": 0}, lambda: False))
    assert engine.llm.dry_sampling["multiplier"] == 0.0  # off
    patch = (MOBILE_DIR / "ci" / "wheels" / "recipes" / "llama-cpp-python" / "patches" / "mobile.patch").read_text(
        encoding="utf-8")
    assert "diff --git a/llama_cpp/llama.py b/llama_cpp/llama.py" in patch
    assert '+        dry = getattr(self, "dry_sampling", None) or {}' in patch and "+            sampler.add_dry(" in patch


def test_engine_card_switch_turns_the_ban_off_and_on(tmp_path):
    pytest.importorskip("flet")
    from glossarion_mobile.ui.screens.local_models_panel import LocalModelsPanel

    service, _engine = _service(tmp_path)
    panel = LocalModelsPanel(service=service, prefs=service.prefs, current_model=lambda: f"local/{STEM}")
    panel.build()
    assert panel.block_cjk_switch.value is True
    # the label wraps beside the switch (switch_row), not as the Switch's own one-line label
    assert not panel.block_cjk_switch.label
    panel.block_cjk_switch.value = False
    assert panel.save_block_cjk() is False and service.prefs.get(svc.BLOCK_CJK_KEY) is False
    assert service.block_cjk() is False
    panel.block_cjk_switch.value = True
    assert panel.save_block_cjk() is True and service.block_cjk() is True
    panel.dispose()
