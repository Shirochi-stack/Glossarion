"""On-device side of "Allow emoticons and symbols from other scripts (kaomoji)" (owner 2026-10-28; config
``script_ban_allow_kaomoji``, off by default): the service passes the request's body field to the engine,
``VocabScripts`` lets the exempt punctuation and the source's emoticon characters through (a per-request
allow-set from an index of the vocabulary), Devanagari / Hangul prose tokens stay banned, the
Model manager › On-device panel shows the shared Settings tile right under the script-ban switch, and the
setting reaches the mobile Settings through the schema.

Run from src/mobile:
    .venv/Scripts/python.exe -m pytest -p no:cacheprovider -W ignore tests_host/test_script_ban_kaomoji_on_device.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
for entry in (str(MOBILE_DIR / "app"), str(MOBILE_DIR.parent)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

np = pytest.importorskip("numpy")

import local_llm_fakes as fakes  # noqa: E402
import script_ban_core  # noqa: E402
from glossarion_mobile.services.local_llm import downloads as dl  # noqa: E402
from glossarion_mobile.services.local_llm import engine as eng  # noqa: E402
from glossarion_mobile.services.local_llm import script_ban as sb  # noqa: E402
from glossarion_mobile.services.local_llm import service as svc  # noqa: E402

ENGLISH = sb.banned_scripts("translation", "English")
HINDI = "बंद हो चुका विचारों को"
SOURCE = "<p>그는 어깨를 으쓱했다. ¯\\_(ツ)_/¯ 그리고 ಠ_ಠ 했다. ٩(◕‿◕)۶ (ノ°▽°)ノ</p>"

#: a byte-level vocabulary: Latin, exempt punctuation, the source's face letters (whole and byte-spelled),
#: and prose of banned scripts
VOCAB = [b"Hello", b" world", "「".encode(), "」".encode(), "・".encode(), "｡".encode(), "︵".encode(),
         "（".encode(), "ツ".encode(), "(ツ)".encode(), "ノ".encode(), "ಠ".encode(), "ಠ_".encode(),
         b"\xe0\xb2", b"\xa0", "٩".encode(), "۶".encode(), HINDI.encode(), " विचार".encode(), b"\xe0\xa4",
         "한국어".encode(), "그는".encode(), "中".encode(), "ツール".encode(), "ಠ益".encode(), b"\xe3", b"\xe3\x80",
         "ア".encode(), b"<|im_end|>", b""]
EXEMPT = {"「".encode(), "」".encode(), "・".encode(), "｡".encode(), "︵".encode(), "（".encode(), b"\xe3\x80"}
FROM_SOURCE = {"ツ".encode(), "(ツ)".encode(), "ノ".encode(), "ಠ".encode(), "ಠ_".encode(), b"\xe0\xb2",
               "٩".encode(), "۶".encode()}
STILL_BANNED = {HINDI.encode(), " विचार".encode(), b"\xe0\xa4", "한국어".encode(), "그는".encode(), "中".encode(),
                "ツール".encode(), "ಠ益".encode(), b"\xe3", "ア".encode()}


class _FakeLlama:
    """What LlamaEngine uses of llama_cpp.Llama (as in test_mobile_batch_9157_script_ban)."""

    def __init__(self):
        self.kwargs = []

    def n_vocab(self):
        return len(VOCAB)

    def detokenize(self, tokens, special=False):
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


def _allowed(engine, params, source=SOURCE):
    messages = [{"role": "system", "content": "Translate. 中 ツール"}, {"role": "user", "content": source}]
    out = "".join(engine.chat(messages, {"max_tokens": 8, **params}, lambda: False))
    return {VOCAB[int(i)] for i in out.split(",") if i}


def _service(tmp_path):
    return svc.LocalEngineService(prefs=fakes.FakePrefs(), registry=dl.Registry(str(tmp_path / "models")),
                                  engine_factory=eng.FakeEngine, watch=False, memory=lambda: (0, None))


def test_service_passes_the_body_field_only_when_on(tmp_path):
    service = _service(tmp_path)
    base = {"max_tokens": 64, "glossarion_task": "translation", "target_language": "English"}
    assert "ban_allow_kaomoji" not in service._params(base, 4096)
    assert "ban_allow_kaomoji" not in service._params({**base, "glossarion_allow_kaomoji": False}, 4096)
    on = service._params({**base, "glossarion_allow_kaomoji": True}, 4096)
    assert on["ban_allow_kaomoji"] is True and on["ban_scripts"] == ENGLISH
    # nothing banned (review): nothing to allow either
    assert "ban_allow_kaomoji" not in service._params({**base, "glossarion_task": "review",
                                                       "glossarion_allow_kaomoji": True}, 4096)


def test_engine_off_is_the_old_ban_and_on_lets_faces_through():
    engine = eng.LlamaEngine()
    engine.llm, engine.path, engine.n_ctx = _FakeLlama(), "x.gguf", 4096
    off = _allowed(engine, {"ban_scripts": ENGLISH})
    assert off == {b"Hello", b" world", b"\xa0", b"<|im_end|>", b""}
    old = engine.llm.kwargs[-1]["logits_processor"][0]
    assert list(old.ids) == sb.banned_token_ids(VOCAB, ENGLISH)  # exactly the rule without the option

    on = _allowed(engine, {"ban_scripts": ENGLISH, "ban_allow_kaomoji": True})
    assert EXEMPT <= on and FROM_SOURCE <= on
    assert not STILL_BANNED & on, STILL_BANNED & on

    # a source without faces: only the exempt punctuation; the processor is the cached one
    plain = _allowed(engine, {"ban_scripts": ENGLISH, "ban_allow_kaomoji": True}, source="<p>그는 웃었다.</p>")
    assert EXEMPT <= plain and not FROM_SOURCE & plain
    first = engine.llm.kwargs[-1]["logits_processor"][0]
    _allowed(engine, {"ban_scripts": ENGLISH, "ban_allow_kaomoji": True}, source="<p>그녀는 걸었다.</p>")
    assert engine.llm.kwargs[-1]["logits_processor"][0] is first
    # another request with the same face characters: the processor kept for that set of characters
    _allowed(engine, {"ban_scripts": ENGLISH, "ban_allow_kaomoji": True})
    again = engine.llm.kwargs[-1]["logits_processor"][0]
    _allowed(engine, {"ban_scripts": ENGLISH, "ban_allow_kaomoji": True}, source="ಠ_ಠ ¯\\_(ツ)_/¯ ٩(◕‿◕)۶ (ノ_ノ)")
    assert engine.llm.kwargs[-1]["logits_processor"][0] is again
    # off again: the old processor
    _allowed(engine, {"ban_scripts": ENGLISH})
    assert engine.llm.kwargs[-1]["logits_processor"][0] is old


def test_vocab_allow_set_matches_the_per_token_rule():
    vocab = sb.VocabScripts(iter(VOCAB))
    chars = sb.kaomoji_source_chars(SOURCE, ENGLISH)
    assert chars == frozenset("ツಠ٩۶ノ")
    assert vocab.ids(ENGLISH, allow_kaomoji=True) == sb.banned_token_ids(VOCAB, ENGLISH, allow_kaomoji=True)
    assert vocab.ids(ENGLISH) == sb.banned_token_ids(VOCAB, ENGLISH)
    allowed = {VOCAB[i] for i in vocab.source_ids(ENGLISH, chars)}
    assert allowed == FROM_SOURCE
    # Hindi target: Devanagari is not banned, ಠ still is and is still allowed from the source
    hindi = sb.banned_scripts("translation", "Hindi")
    assert {VOCAB[i] for i in vocab.source_ids(hindi, chars)} == FROM_SOURCE
    assert HINDI.encode() not in {VOCAB[i] for i in vocab.ids(hindi, allow_kaomoji=True)}


def test_per_request_allow_set_is_cheap_on_a_large_vocabulary():
    """A 150k-token byte-level vocabulary (every CJK / Hangul / Indic character, byte fragments, words):
    the per-request part (source reading + allow-set + set difference) stays well under a few ms."""
    pieces = [bytes([b]) for b in range(256)]
    for lo, hi in ((0x3000, 0x30FF), (0x4E00, 0x9FFF), (0xAC00, 0xD7A3), (0x0900, 0x0DFF), (0x0600, 0x06FF),
                   (0x0400, 0x04FF), (0xFF00, 0xFFEF)):
        pieces.extend(chr(cp).encode() for cp in range(lo, hi + 1))
    words = [chr(0xAC00 + (i * 7) % 11172) + chr(0xAC00 + (i * 13) % 11172) for i in range(40000)]
    pieces.extend(w.encode() for w in words)
    pieces.extend(f"w{i}".encode() for i in range(150000 - len(pieces)))
    assert len(pieces) == 150000
    vocab = sb.VocabScripts(iter(pieces))
    base = vocab.processor(ENGLISH, allow_kaomoji=True)  # once per model: the base processor
    chapter = ("<p>그는 어깨를 으쓱했다. 그녀는 대답하지 않았다. 다시 걸었다.</p>" * 300) + SOURCE
    vocab.processor(ENGLISH, allow_kaomoji=True, source="ಠ_ಠ")  # once per model: the index
    timings = []
    for n in range(5):
        text = chapter + f" ({n})"  # a new request each time
        vocab._source_processors.clear()
        started = time.perf_counter()
        proc = vocab.processor(ENGLISH, allow_kaomoji=True, source=text)
        timings.append(time.perf_counter() - started)
    assert len(proc) < len(base)
    allowed = set(base.ids) - set(proc.ids)
    assert {pieces[i] for i in allowed} >= {"ツ".encode(), "ಠ".encode(), "٩".encode(), "۶".encode(), "ノ".encode()}
    assert not any("가" <= pieces[i].decode("utf-8", "ignore")[:1] <= "힣" for i in allowed)
    print(f"per-request allow-set: {min(timings) * 1000:.2f} ms (max {max(timings) * 1000:.2f} ms)")
    assert min(timings) < 0.05  # measured ~1 ms on the desktop; generous for CI runners


# ------------------------------------------------------------------------------- settings + panel

def _ctx(tmp_path, data=None):
    from glossarion_mobile.state.config_store import MobileConfigStore
    from glossarion_mobile.ui.settings.context import SettingsContext
    from glossarion_mobile.ui.settings.schema_access import SchemaAccess

    path = tmp_path / "config.json"
    path.write_text(json.dumps(data or {}), encoding="utf-8")
    schema = SchemaAccess()
    store = MobileConfigStore(path, debounce=10, defaults=schema.effective_default,
                              reader=lambda p, decrypt=True: json.loads(Path(p).read_text(encoding="utf-8")),
                              writer=lambda disk, p, backup=False: Path(p).write_text(json.dumps(disk), encoding="utf-8"))
    store.load()
    return SettingsContext(page=None, store=store, schema=schema)


def test_setting_is_in_the_mobile_schema_off_by_default():
    import settings_schema

    spec = settings_schema.spec("script_ban_allow_kaomoji")
    assert spec is not None
    assert settings_schema.effective_default("script_ban_allow_kaomoji") is False
    assert settings_schema.section_of("script_ban_allow_kaomoji").id == "other.anti_duplicate.logit_bias"
    assert "SCRIPT_BAN_ALLOW_KAOMOJI" in settings_schema.env_names("script_ban_allow_kaomoji")


def test_panel_shows_the_shared_tile_under_the_switch(tmp_path, monkeypatch):
    pytest.importorskip("flet")
    for name, sub in (("HOME", "home"), ("USERPROFILE", "home"), ("GLOSSARION_LIBRARY_DIR", "lib"),
                      ("OUTPUT_DIRECTORY", "out"), ("GLOSSARION_DATA_DIR", "data")):
        (tmp_path / sub).mkdir(exist_ok=True)
        monkeypatch.setenv(name, str(tmp_path / sub))
    import flet as ft
    from glossarion_mobile.ui.screens.local_models_panel import KAOMOJI_KEY, LocalModelsPanel

    ctx = _ctx(tmp_path)
    panel = LocalModelsPanel(service=_service(tmp_path), prefs=fakes.FakePrefs(), current_model=lambda: None)
    panel.settings_ctx = ctx
    # the tile's config observer must not stop the panel from following downloads and the engine (Build
    # Mobile 40776917992: test_local_models_panel rows stuck on "Working…" once a Settings context existed)
    calls = []
    real_downloads, real_engine = panel.downloads.subscribe, panel.service.subscribe
    panel.downloads.subscribe = lambda fn: (calls.append("downloads"), real_downloads(fn))[1]
    panel.service.subscribe = lambda fn: (calls.append("engine"), real_engine(fn))[1]
    panel.build()
    assert calls == ["downloads", "engine"]
    section = panel._sampling_section()
    switch_at = next(i for i, c in enumerate(section) if isinstance(c, ft.Row) and c.controls
                     and isinstance(c.controls[0], ft.Switch) and c.controls[0].key == "local-block-cjk")
    tile = panel.kaomoji_tile
    assert tile is not None and tile.key == KAOMOJI_KEY == "script_ban_allow_kaomoji"
    assert section.index(tile.control) == switch_at + 2  # the switch, its note, then the tile
    assert tile.label == "Allow emoticons and symbols from other scripts (kaomoji)"
    assert tile.switch.value is False
    tile.apply(True)  # the panel writes the shared config key...
    assert ctx.store.get(KAOMOJI_KEY) is True
    ctx.store.set(KAOMOJI_KEY, False)  # ...and follows a change made in Settings
    assert tile.switch.value is False
    panel.dispose()
    ctx.store.set(KAOMOJI_KEY, True)  # disposed: the tile no longer follows
    assert tile.switch.value is False


def test_panel_without_settings_context_has_no_tile(tmp_path):
    pytest.importorskip("flet")
    from glossarion_mobile.ui.screens.local_models_panel import LocalModelsPanel
    from glossarion_mobile.ui.sheets import model_sheet

    previous = model_sheet._SHEET_ENV
    model_sheet.install_sheet_env(None)
    try:
        panel = LocalModelsPanel(service=_service(tmp_path), prefs=fakes.FakePrefs(), current_model=lambda: None)
        panel.build()
        assert panel.kaomoji_tile is None
        panel.dispose()
    finally:
        model_sheet.install_sheet_env(previous)
