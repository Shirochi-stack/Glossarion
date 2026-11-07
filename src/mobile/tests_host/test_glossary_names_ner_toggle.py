"""Host tests for the owner's report (2026-11-07): Settings › QA Scanner Settings › "Check glossary names"

* did not mention NER (desktop's QA Scanner settings toggle says "Check glossary names with NER (Experimental)"
  or why NER is not used: not installed / not supported / off) - the schema's label override sat under the
  bare key while the setting is nested (qa_scanner_settings.check_glossary_names), and the tile had no
  runtime text;
* switching it on did not offer the NER model download - now the manga tools' model question
  (``ensure_models``: Download / Download in the run / Cancel) asks for the registry's NER model.

Real data is never touched: HOME / USERPROFILE / APPDATA / LOCALAPPDATA / GLOSSARION_LIBRARY_DIR /
OUTPUT_DIRECTORY / GLOSSARION_DATA_DIR point at pytest's tmp dir (autouse fixture), GLOSSARION_HTTP_LOG=0.

Run from src/mobile:
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_glossary_names_ner_toggle.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


needs_flet = pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet / msgpack not installed")

KEY = "qa_scanner_settings.check_glossary_names"
NER_KEY = "bert_base_ner_int8"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    for name, sub in (("HOME", "home"), ("USERPROFILE", "home"), ("APPDATA", "appdata"),
                      ("LOCALAPPDATA", "localappdata"), ("GLOSSARION_LIBRARY_DIR", "lib"),
                      ("OUTPUT_DIRECTORY", "out"), ("GLOSSARION_DATA_DIR", "data")):
        folder = tmp_path / "_iso" / sub
        folder.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(name, str(folder))
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    yield tmp_path / "_iso"


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


class FakeModels:
    """The manga_models registry with only the NER model (its companions download with it)."""

    NER_MODEL_KEY = NER_KEY
    KIND_DETECTOR = "detector"

    def __init__(self):
        self.calls: list = []
        self.installed: set = set()
        self.spec = types.SimpleNamespace(key=NER_KEY, kind="ner", title="English named-entity model (NER)",
                                          size=108952255, selector="ner")

    def specs(self, kind=None):
        return [self.spec] if kind in (None, "ner") else []

    def get_spec(self, key):
        if key != NER_KEY:
            raise KeyError(key)
        return self.spec

    def status(self, key):
        return types.SimpleNamespace(path=f"/m/{key}", installed=key in self.installed, partial_bytes=0,
                                     downloading=None)

    def download(self, key, progress=None, cancel=None):
        self.calls.append(("download", key))
        if progress is not None:
            progress(types.SimpleNamespace(fraction=0.5, phase="download"))
        self.installed.add(key)
        return f"/m/{key}"

    def is_installed(self, key, *a, **k):
        return key in self.installed

    def disk_usage(self):
        return 0


def _fake_ner(monkeypatch, models: FakeModels, runtime: bool = True):
    fake = types.ModuleType("ner_names")
    fake.runtime_available = lambda: runtime
    fake.model_installed = lambda: runtime and models.is_installed(NER_KEY)
    monkeypatch.setitem(sys.modules, "ner_names", fake)
    monkeypatch.setitem(sys.modules, "manga_models", models)


def test_schema_label_mentions_ner_like_desktop():
    """The nested setting's label is desktop's toggle text (it fell back to "Check glossary names")."""
    import glossary_name_check as name_check
    import settings_schema

    label = settings_schema.spec(KEY).label
    assert label == name_check.NER_TOGGLE_LABELS["installed"] == "Check glossary names with NER (Experimental)"
    assert KEY in [hit.key for hit in settings_schema.search("NER")]


def test_ner_toggle_label_follows_ner_status(monkeypatch):
    import glossary_name_check as name_check

    for state in ("installed", "missing", "unsupported", "off"):
        monkeypatch.setattr(name_check, "ner_status", lambda use_ner=True, s=state: s)
        assert name_check.ner_toggle_label(True) == name_check.NER_TOGGLE_LABELS[state]
    assert "NER not installed" in name_check.NER_TOGGLE_LABELS["missing"]


@needs_flet
def test_tile_text_says_whether_ner_does_the_check(tmp_path, monkeypatch):
    from glossarion_mobile.ui.settings.tiles import make_tile

    models = FakeModels()
    _fake_ner(monkeypatch, models)
    ctx = _ctx(tmp_path)
    tile = make_tile(ctx.schema.spec(KEY), ctx)
    assert tile.title_text.value == ("Check glossary names (Experimental) — NER not installed "
                                     "(downloads on first use)")
    models.installed.add(NER_KEY)
    ctx.store.set(("glossary_name_check_use_ner",), True)
    tile.refresh(push=False)
    assert tile.title_text.value == "Check glossary names with NER (Experimental)"
    ctx.store.set(("glossary_name_check_use_ner",), False)
    tile.refresh(push=False)
    assert tile.title_text.value == "Check glossary names (Experimental) — NER off"


@needs_flet
def test_switching_on_offers_the_ner_download(tmp_path, monkeypatch):
    from glossarion_mobile.ui.settings.tiles import make_tile

    models = FakeModels()
    _fake_ner(monkeypatch, models)

    async def scenario():
        ctx = _ctx(tmp_path)
        tile = make_tile(ctx.schema.spec(KEY), ctx)
        ctx.extras["answers"] = ["download"]
        tile.activate()  # the switch turned on
        for _ in range(50):
            await asyncio.sleep(0.01)
            if models.calls:
                break
        await asyncio.sleep(0.05)
        assert ctx.store.get(("qa_scanner_settings", "check_glossary_names")) is True
        title, body = ctx.extras["asked"][-1]
        assert title == "Download models"
        assert body.startswith("Check glossary names needs English named-entity model (NER)")
        assert models.calls == [("download", NER_KEY)]
        assert tile.title_text.value == "Check glossary names with NER (Experimental)"

        # off and on again with the model there: no question
        asked = len(ctx.extras["asked"])
        tile.activate()
        tile.activate()
        await asyncio.sleep(0.05)
        assert len(ctx.extras["asked"]) == asked

        # Cancel keeps the toggle on (the scan downloads the model on first use, like desktop)
        models.installed.clear()
        tile.activate()
        ctx.extras["answers"] = ["cancel"]
        tile.activate()
        await asyncio.sleep(0.05)
        assert len(ctx.extras["asked"]) == asked + 1 and models.calls == [("download", NER_KEY)]
        assert ctx.store.get(("qa_scanner_settings", "check_glossary_names")) is True

    asyncio.run(scenario())


@needs_flet
def test_no_question_when_ner_is_off_or_unsupported(tmp_path, monkeypatch):
    from glossarion_mobile.ui.settings.tiles import make_tile

    models = FakeModels()

    async def scenario(data, runtime):
        _fake_ner(monkeypatch, models, runtime=runtime)
        ctx = _ctx(tmp_path, data)
        tile = make_tile(ctx.schema.spec(KEY), ctx)
        ctx.extras["answers"] = []
        tile.activate()
        await asyncio.sleep(0.05)
        assert not ctx.extras.get("asked") and models.calls == []
        return tile.title_text.value

    assert asyncio.run(scenario({"glossary_name_check_use_ner": False}, True)).endswith("NER off")
    assert asyncio.run(scenario({}, False)).endswith("NER not supported in this build")
