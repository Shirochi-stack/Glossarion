"""Model manager › On-device: a download started off-screen brings its progress row into view, once.

Owner 2026-11-18 (phone): "Download doesn't scroll to the bottom, so the user has to manually scroll down to
see the hugging face model download." Add from Hugging Face's progress row lands in Other downloads, below
the card; a Custom pick appends a preset row. After the tap the panel's ListView ``scroll_to``s the row's
``scroll_anchor`` once the row is on the page (not before), never again on progress ticks, and the field's
keyboard is dropped first. A Download tapped on a preset row does not scroll: the progress is in that row.

No network: ``service.resolve`` and ``service.download`` are fakes that emit download states.

Run from src/mobile:
    .venv/Scripts/python.exe -m pytest -p no:cacheprovider -W ignore tests_host/test_local_hf_download_scroll.py
"""

from __future__ import annotations

import asyncio
import importlib.util

import pytest

import local_llm_fakes as fakes
from glossarion_mobile.services.local_llm import downloads as dl
from glossarion_mobile.services.local_llm import engine as eng
from glossarion_mobile.services.local_llm import hf, presets
from glossarion_mobile.services.local_llm import service as svc


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


pytestmark = pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet/msgpack not installed")

REPO = "tester/Tiny-GGUF"
FILE = "Tiny-Q4_K_M.gguf"
SIZE = 1000
TICKS = 6


def _keys(control) -> list:
    """Every key value under ``control`` (a ScrollKey counts by its value)."""
    out, stack = [], [control]
    while stack:
        c = stack.pop()
        key = getattr(c, "key", None)
        if key is not None:
            out.append(getattr(key, "value", key))
        for attr in ("content", "controls"):
            child = getattr(c, attr, None)
            if isinstance(child, list):
                stack.extend(child)
            elif child is not None and not isinstance(child, str) and hasattr(child, "__dict__"):
                stack.append(child)
    return out


def _panel(tmp_path, monkeypatch, preset_rows=()):
    from glossarion_mobile.ui.screens import local_models_panel as lmp

    prefs = fakes.FakePrefs({presets.PREF_KEY: list(preset_rows)})
    service = svc.LocalEngineService(prefs=prefs, registry=dl.Registry(str(tmp_path / "models")),
                                     engine_factory=eng.FakeEngine, watch=False, memory=lambda: (0, None))
    listing = hf.RepoFiles(commit="c0ffee", files=(hf.HfFile(FILE, SIZE),), projectors=())
    log: list = []  # ("tick", n) / ("scroll", key, key on the page then) / ("keyboard",)

    async def resolve(ref):
        await asyncio.sleep(0)  # the repo's file list arrives later
        return listing

    async def download(spec):
        state = dl.DownloadState(spec.stem, spec.repo, spec.file, bytes_total=SIZE)
        service.downloads.states[spec.stem] = state
        service.downloads._emit(state)
        for n in range(1, TICKS + 1):  # progress ticks, spread over the reveal's settle time
            await asyncio.sleep(getattr(lmp, "REVEAL_SETTLE", 0.05) / 2)
            state.phase, state.bytes_done = "downloading", n * SIZE // (TICKS + 1)
            log.append(("tick", n))
            service.downloads._emit(state)
        state.running, state.paused, state.status = False, True, "Paused"
        service.downloads._emit(state)
        return state

    monkeypatch.setattr(service, "resolve", resolve)
    monkeypatch.setattr(service, "download", download)
    monkeypatch.setattr(lmp, "dismiss_keyboard", lambda: log.append(("keyboard",)), raising=False)
    panel = lmp.LocalModelsPanel(service=service, prefs=prefs, confirm=lambda *a: True,
                                 choose=lambda repo, files, selected: selected or files[0])
    panel.build()
    calls: list = []

    async def scroll_to(**kwargs):
        key = kwargs.get("scroll_key")
        value = getattr(key, "value", key)
        calls.append(kwargs)
        log.append(("scroll", value, value in _keys(panel.control)))

    object.__setattr__(panel.control, "scroll_to", scroll_to)
    return panel, service, calls, log


def test_add_from_hugging_face_scrolls_to_its_progress_row_once(tmp_path, monkeypatch):
    async def scenario():
        panel, service, calls, log = _panel(tmp_path, monkeypatch)
        stem = service.registry.stem_for(REPO, FILE)
        _focus(panel.hf_field, True)  # the user typed the repo: its keyboard is up
        panel.hf_field.value = REPO
        assert await panel.download_typed() is None  # the fake download ends paused
        await asyncio.sleep(lmp_settle() * 2)
        anchor = f"local-dl-anchor-{stem}"
        assert [c["scroll_key"].value for c in calls] == [anchor], calls
        assert calls[0]["duration"] > 0
        scroll = next(entry for entry in log if entry[0] == "scroll")
        assert scroll[2], "scrolled before the row was on the page"
        # the keyboard went first; the scroll came once, while the download was still ticking, never per tick
        assert log[0] == ("keyboard",)
        index = log.index(scroll)
        assert any(e[0] == "tick" for e in log[index + 1:]), log
        assert [e for e in log if e[0] == "scroll"] == [scroll]
        # the row is in Other downloads, below the Add from Hugging Face card
        cards = [c.key for c in panel.control.controls]
        assert cards.index("local-downloaded") > cards.index("local-hf")
        assert anchor in _keys(panel.other_column)
        # later progress or renders of the same download never scroll again (the user may have scrolled away)
        state = service.downloads.states[stem]
        state.running, state.paused = True, False
        service.downloads._emit(state)
        panel.render()
        await asyncio.sleep(lmp_settle() * 2)
        assert len(calls) == 1
        panel.dispose()

    asyncio.run(scenario())


def test_keyboard_is_left_alone_when_the_field_has_no_focus(tmp_path, monkeypatch):
    async def scenario():
        panel, _service, calls, log = _panel(tmp_path, monkeypatch)
        _focus(panel.hf_field, True)
        _focus(panel.hf_field, False)
        panel.hf_field.value = REPO
        await panel.download_typed()
        await asyncio.sleep(lmp_settle() * 2)
        assert ("keyboard",) not in log and len(calls) == 1
        panel.dispose()

    asyncio.run(scenario())


def test_a_custom_pick_scrolls_to_its_new_preset_row(tmp_path, monkeypatch):
    async def scenario():
        rows = [{"name": "Other", "repo": "someone/Other-GGUF", "file": "Other-Q4_K_M.gguf", "note": "", "n_ctx": 0}]
        panel, _service, calls, log = _panel(tmp_path, monkeypatch, rows)
        row = presets.custom_row({"name": "Tiny 1B", "repo": REPO}, hf.HfFile(FILE, SIZE))
        assert await panel.get_custom(row) is None  # paused: the row stays with its Resume
        await asyncio.sleep(lmp_settle() * 2)
        assert [p["name"] for p in panel.presets()][-1] == row["name"]
        assert [c["scroll_key"].value for c in calls] == ["local-preset-anchor-1"]
        assert next(e for e in log if e[0] == "scroll")[2]
        panel.dispose()

    asyncio.run(scenario())


def test_download_on_a_preset_row_does_not_scroll(tmp_path, monkeypatch):
    """The progress shows in the row whose Download (or Retry) was just tapped: the page stays put, and the
    row still gets no anchor-driven scroll on its progress ticks."""
    async def scenario():
        rows = [{"name": "Tiny", "repo": REPO, "file": FILE, "note": "", "n_ctx": 0, "size": SIZE}]
        panel, _service, calls, _log = _panel(tmp_path, monkeypatch, rows)
        assert await panel.get_preset(0) is None
        await asyncio.sleep(lmp_settle() * 2)
        assert calls == []
        panel.dispose()

    asyncio.run(scenario())


def lmp_settle() -> float:
    from glossarion_mobile.ui.screens import local_models_panel as lmp

    return getattr(lmp, "REVEAL_SETTLE", 0.05) + 0.05


def _focus(field, focused: bool) -> None:
    handler = field.on_focus if focused else field.on_blur
    if handler is not None:
        handler(None)
