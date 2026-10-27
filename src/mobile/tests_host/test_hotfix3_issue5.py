"""Acceptance test for owner issue 5 of hotfix 3 (device report 2026-10-26): "Use gpu should automatically show
the gpu friendly quants, not require an extra tap".

On the phone (Android arm64, Qualcomm Adreno: the OpenCL GPU backend), Model manager › On-device › "Use GPU
(experimental)" switched on only revealed an "Add GPU-friendly presets (Q4_0)" button: the Q4_0 presets
(``presets.GPU_PRESETS``) joined the Models list after a second tap on it.

This file proves on the REAL app objects that switching the GPU on now shows them at once. The whole app runs
on a fake Flet session (``test_ui_foundations._start``) as a simulated Android arm64 phone
(``GLOSSARION_SIMULATE_PLATFORM=android`` and ``platform.machine() == "aarch64"``, so the app's own
``engine.gpu_offer`` picks OpenCL and ``ModelsKeysFeature`` builds its own ``LocalEngineService``). The test
goes through the real shell route ``/settings/models`` and the Model manager's "On-device" segment, and taps
the "Use GPU" switch through the session the way Flutter sends it (``host_tester.PyTester``: the value, then
``change``). Only llama.cpp itself is missing on the host: ``engine.engine_status`` says it is there, as in
the APK. Nothing loads or downloads, and the E2E network guard refuses every non-loopback connection.

Hotfix 6 (owner 2026-10-27: "gpu friendly presets are always shown regardless of toggle state"): opening the
Model manager already lists the Q4_0 presets with the GPU off (added once, without a note); the switch only
turns the GPU on and off.

``test_use_gpu_on_shows_the_q4_0_presets_at_once_and_keeps_every_model``:

1. Before the switch: the four defaults, the owner's own preset (downloaded) and the four Q4_0 rows are
   listed, in the Python tree and in the patches the client received. A Hugging Face file no preset names
   sits under "Other downloads". No "Add GPU-friendly presets" button and no note.
2. ONE tap on the switch, no other event: the GPU goes on, the list stays as it is, no note.
3. Off: every row stays (the Q4_0 ones too), the owner's preset is still "On this device", the other file is
   still listed, and both files and registry entries are untouched.
4. Repeated toggles (three on/off cycles, a three-tap burst, and a repeated "change true" while it is on):
   each Q4_0 file is in the list exactly once. There is never a second note or an "already in the list" one.
5. The Model manager closed and opened again rebuilds the panel without duplicates, with the switch on. The
   model picker's On-device tab (chat › model) lists the same nine presets, each Q4_0 one once.

Real data stays untouched: the app runs on temp FLET_APP_STORAGE_* dirs (HOME, OUTPUT_DIRECTORY,
GLOSSARION_LIBRARY_DIR, GLOSSARION_DATA_DIR, GLOSSARION_MODELS_DIR and CONFIG_FILE there; asserted).
USERPROFILE, APPDATA and LOCALAPPDATA point into tmp_path, and HTTP logging is off. Waits follow the app's
own state (a predicate poll), never a fixed sleep.

Run from src/mobile with the mobile env (on the dev box ``.venv``, Python 3.12 with the shell PYTHONPATH)::

    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic \\
        tests_host/test_hotfix3_issue5.py
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import os
import platform
import sys
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
for entry in (str(MOBILE_DIR / "app"), str(MOBILE_DIR.parent), str(MOBILE_DIR / "tests")):
    if entry not in sys.path:
        sys.path.insert(0, entry)


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


pytestmark = pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet/msgpack not installed")


def _load(alias: str, file_name: str):
    spec = importlib.util.spec_from_file_location(alias, Path(__file__).with_name(file_name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TF = _load("_glossarion_tf_hotfix3_issue5", "test_ui_foundations.py")
storage = TF._TB.storage
app_env = TF._TB.app_env

from glossarion_mobile.services.local_llm import downloads as dl  # noqa: E402
from glossarion_mobile.services.local_llm import engine as eng  # noqa: E402
from glossarion_mobile.services.local_llm import presets  # noqa: E402
from glossarion_mobile.services.local_llm import service as svc  # noqa: E402

DEFAULT_NAMES = [p["name"] for p in presets.DEFAULT_PRESETS]
GPU_NAMES = [p["name"] for p in presets.GPU_PRESETS]
GPU_FILES = [(p["repo"], p["file"]) for p in presets.GPU_PRESETS]
BUTTON = "Add GPU-friendly presets (Q4_0)"

# the owner's own models: a preset of their own (downloaded) and a Hugging Face file no preset names
MINE = {"name": "My Gemma", "repo": "owner/My-Gemma-GGUF", "file": "My-Gemma-Q8_0.gguf", "note": "mine", "n_ctx": 0}
MINE_STEM = "My-Gemma-Q8_0"
OTHER = ("someone/Other-GGUF", "Other-Q5_K_M.gguf")
OTHER_STEM = "Other-Q5_K_M"
EXPECTED = DEFAULT_NAMES + [MINE["name"]] + GPU_NAMES


@pytest.fixture
def phone(storage, monkeypatch, request, tmp_path):
    """A returning user on an Android arm64 phone (OpenCL offered), with every real-data path in tmp_path."""
    monkeypatch.setenv("GLOSSARION_SIMULATE_PLATFORM", "android")  # after ``storage`` cleared it
    monkeypatch.setattr(platform, "machine", lambda: "aarch64")  # what an arm64-v8a phone reports
    monkeypatch.setattr(eng, "engine_status", lambda: (True, None))  # the APK ships llama.cpp
    monkeypatch.setattr(svc, "_DEFAULT", None)  # never adopt another test's running engine
    monkeypatch.delenv(svc.URL_ENV, raising=False)  # the engine's start sets it; undone after the test
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    for name in ("USERPROFILE", "APPDATA", "LOCALAPPDATA"):
        folder = tmp_path / "win" / name
        folder.mkdir(parents=True)
        monkeypatch.setenv(name, str(folder))
    request.getfixturevalue("app_env")  # FLET_APP_STORAGE_* + bootstrap (HOME, OUTPUT_DIRECTORY, Library, config)
    assert eng.gpu_offer("android") == "OpenCL"
    root = os.path.normcase(str(tmp_path))
    for name in ("HOME", "OUTPUT_DIRECTORY", "GLOSSARION_LIBRARY_DIR", "GLOSSARION_DATA_DIR", "CONFIG_FILE",
                 "USERPROFILE", "APPDATA", "LOCALAPPDATA", dl.MODELS_ENV):
        value = os.environ.get(name)
        assert value and os.path.normcase(os.path.abspath(value)).startswith(root), (name, value)
    return tmp_path


@pytest.fixture
def offline():
    """The E2E's network guard (``diagnostics.e2e.ProcessGuards``): every non-loopback connect / DNS lookup
    fails at once and is recorded."""
    from glossarion_mobile.diagnostics.e2e import ProcessGuards

    guards = ProcessGuards()
    guards._install_network_guard()
    try:
        yield guards
    finally:
        guards.uninstall()


# ---- helpers ------------------------------------------------------------------------------------------------


def _instrument(conn) -> None:
    """Keep every message's encoded bytes at send time (what the phone receives)."""
    import msgpack
    from flet.controls.base_control import BaseControl
    from flet.messaging.protocol import configure_encode_object_for_msgpack

    encode = configure_encode_object_for_msgpack(BaseControl)
    conn.raw_sent = []
    original = conn.send_message

    def send_message(message):
        conn.raw_sent.append(msgpack.packb([message.action, message.body], default=encode))
        return original(message)

    conn.send_message = send_message


def _sent_strings(conn, start: int) -> list:
    """Every string in the messages sent to the client since ``start``."""
    import msgpack

    out: list = []
    for raw in conn.raw_sent[start:]:
        stack = [msgpack.unpackb(raw, strict_map_key=False)]
        while stack:
            cur = stack.pop()
            if isinstance(cur, str):
                out.append(cur)
            elif isinstance(cur, dict):
                stack.extend(cur.keys())
                stack.extend(cur.values())
            elif isinstance(cur, (list, tuple)):
                stack.extend(cur)
    return out


async def _until(predicate, timeout: float = 5.0) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return predicate()


def _seed_own_models(engine) -> None:
    """The owner's own preset (downloaded) and a Hugging Face file no preset names, before the screen opens."""
    presets.save_presets(engine.prefs, presets.load_presets(engine.prefs) + [MINE])
    registry = engine.registry
    os.makedirs(registry.root, exist_ok=True)
    for stem, (repo, file) in ((MINE_STEM, (MINE["repo"], MINE["file"])), (OTHER_STEM, OTHER)):
        payload = (stem * 64).encode()
        with open(registry.file_path(stem), "wb") as handle:
            handle.write(payload)
        registry.add(stem, {"repo": repo, "file": file, "revision": "r", "size": len(payload),
                            "sha256": hashlib.sha256(payload).hexdigest(), "downloaded_at": 1})


def _row_names(panel) -> list:
    """The preset names the Models card shows (each row's first Text is its name)."""
    from host_tester import _children

    names = []
    for row in panel.preset_column.controls:
        stack = [row]
        while stack:
            node = stack.pop(0)
            if type(node).__name__ == "Text":
                names.append(node.value)
                break
            stack[:0] = list(_children(node))
    return names


async def _count(tester, *, text=None, key=None) -> int:
    finder = await (tester.find_by_text(text) if text is not None else tester.find_by_key(key))
    return finder.count


async def _assert_list(tester, panel, prefs, *, gpu_on: bool) -> None:
    """The saved presets and what the page shows: every preset once, the owner's models intact."""
    assert [p["name"] for p in presets.load_presets(prefs)] == EXPECTED
    saved = [(p["repo"], p["file"]) for p in presets.load_presets(prefs)]
    assert all(saved.count(pair) == 1 for pair in GPU_FILES), saved
    assert _row_names(panel) == EXPECTED
    for name in EXPECTED:
        assert await _count(tester, text=name) == 1, name  # one visible row each, none duplicated
    assert await _count(tester, key="local-gpu-presets") == 0  # the extra button never shows
    assert await _count(tester, text=BUTTON) == 0
    switch = (await tester.find_by_key("local-gpu"))
    assert switch.count == 1 and bool(tester.control(switch.first).value) is gpu_on
    # the owner's own models: their preset still on this device, the other file still listed, files on disk
    mine = EXPECTED.index(MINE["name"])
    assert panel.row_state(presets.load_presets(prefs)[mine], MINE_STEM)[0] in ("use", "in_use")
    assert await _count(tester, key=f"local-preset-status-{mine}") == 1
    assert panel.other_card.visible and await _count(tester, key="local-downloaded") == 1
    assert any(OTHER_STEM in text for text in _texts_of(panel.other_column))
    registry = panel.service.registry
    assert set(registry.load()) == {MINE_STEM, OTHER_STEM}
    assert all(os.path.isfile(registry.file_path(stem)) for stem in (MINE_STEM, OTHER_STEM))


def _texts_of(control) -> list:
    from host_tester import _children, _texts

    out, stack = [], [control]
    while stack:
        node = stack.pop()
        out.extend(_texts(node))
        stack.extend(_children(node))
    return out


async def _open_on_device_tab(session, app, tester, *, route: bool = True):
    from glossarion_mobile.ui.screens.local_models_panel import LocalModelsPanel
    from glossarion_mobile.ui.screens.model_manager import ModelManagerScreen

    if route:  # the client opens /settings/models (as a link or the route the app pushed)
        await TF._TB._route(session, "/settings/models")
    screen = app.shell.top_screen
    assert isinstance(screen, ModelManagerScreen)
    segment = await tester.find_by_text("On-device")
    assert segment.count == 1
    await tester.tap(segment.first)
    assert screen.tab == "local" and isinstance(screen.local, LocalModelsPanel) and screen.local_box.visible
    return screen, screen.local


async def _tap_switch(tester) -> None:
    switch = await tester.find_by_key("local-gpu")
    assert switch.count == 1
    await tester.tap(switch.first)


# ---- the scenario ------------------------------------------------------------------------------------------


def test_use_gpu_on_shows_the_q4_0_presets_at_once_and_keeps_every_model(phone, offline):
    from host_tester import PyTester

    from glossarion_mobile.services import model_catalog as mc
    from glossarion_mobile.ui.sheets import model_sheet as ms

    async def scenario():
        _main, conn, session, page, app = await TF._start("android", width=412)
        _instrument(conn)
        TF._echo_push_routes(conn, session)  # the fake client follows push_route like Flutter does
        tester = PyTester(session, page)
        feature = app.models_keys
        engine = feature.local_engine
        said: list = []
        notify = app.notify

        def spy(message, *args, **kwargs):
            said.append(message)
            return notify(message, *args, **kwargs)

        app.notify = spy  # the Model manager's notes (its toast is the app's snack bar)
        try:
            # the app's own engine: Android, OpenCL offered, the GPU off by default, models under tmp_path
            assert engine is not None and engine.platform == "android" and engine.gpu_backend == "OpenCL"
            assert not engine.use_gpu() and engine.prefs is app.prefs
            assert os.path.normcase(os.path.abspath(engine.registry.root)).startswith(os.path.normcase(str(phone)))
            _seed_own_models(engine)
            prefs = app.prefs

            # 1. Model manager › On-device with the GPU off: every preset, the Q4_0 ones too (hotfix 6), no button
            start = len(conn.raw_sent)
            screen, panel = await _open_on_device_tab(session, app, tester)
            assert not engine.use_gpu()
            await _assert_list(tester, panel, prefs, gpu_on=False)
            # the phone got them: the patches sent for the open carry every Q4_0 row
            assert await _until(lambda: set(GPU_NAMES) <= set(_sent_strings(conn, start))), \
                [s for s in _sent_strings(conn, start) if "GPU" in s or "Q4_0" in s]
            assert await _count(tester, text="Use GPU (experimental)") == 1
            assert panel.other_card.visible and said == []  # added without a note

            # 2. ONE tap on the switch and nothing else: the GPU goes on, the list stays as it is
            events = len(tester.events)
            await _tap_switch(tester)
            assert tester.events[events:] == [("Switch", "local-gpu", "change")]
            assert prefs.get(svc.GPU_KEY) is True and engine.use_gpu()
            await _assert_list(tester, panel, prefs, gpu_on=True)
            assert said == []

            # 3. off: every preset, the owner's preset and the other file stay
            await _tap_switch(tester)
            assert prefs.get(svc.GPU_KEY) is False and not engine.use_gpu()
            await _assert_list(tester, panel, prefs, gpu_on=False)
            assert said == []

            # 4. repeated toggles: on/off three times, a three-tap burst, a repeated "change true"
            for _cycle in range(3):
                await _tap_switch(tester)
                await _assert_list(tester, panel, prefs, gpu_on=True)
                await _tap_switch(tester)
                await _assert_list(tester, panel, prefs, gpu_on=False)
            finder = (await tester.find_by_key("local-gpu")).first
            switch = tester.control(finder)
            await asyncio.gather(*(tester.tap(finder) for _ in range(3)))  # three quick taps: on, off, on
            assert switch.value is True and engine.use_gpu()
            await _assert_list(tester, panel, prefs, gpu_on=True)
            switch = tester.control((await tester.find_by_key("local-gpu")).first)
            await session.dispatch_event(switch._i, "change", "true")  # the client re-sends it while on
            await _assert_list(tester, panel, prefs, gpu_on=True)
            assert said == []  # no note in all; never "already in the list"

            # 5. closed and opened again: the rebuilt panel lists each preset once, the switch on
            # Android Back, then Settings › Model manager again (the hub tile, in-app navigation)
            await session.dispatch_event(page._i, "view_pop", {"route": page.views[-1].route})
            assert await _until(lambda: app.shell.current_route == "/settings" and page.route == "/settings")
            tile = await tester.find_by_key("hub-settings.models")
            assert tile.count == 1
            await tester.tap(tile.first)
            assert await _until(lambda: isinstance(app.shell.top_screen, type(screen))
                                and app.shell.top_screen is not screen)
            screen2, panel2 = await _open_on_device_tab(session, app, tester, route=False)
            assert screen2 is not screen and panel2 is not panel
            await _assert_list(tester, panel2, prefs, gpu_on=True)
            assert said == []
            # the model picker's On-device tab (chat › model) shows the same nine presets
            sheet = ms.ModelSheet(current_model=str(app.config_store.get("model", "") or ""), env=ms.sheet_env(),
                                  tab="ondevice")
            texts = [t for row in sheet.ondevice_list.controls for t in _texts_of(row)]
            assert f"Presets · {len(EXPECTED)}" in texts, [t for t in texts if t.startswith("Presets")]
            for name in GPU_NAMES:
                assert texts.count(name) == 1, (name, texts.count(name))
            assert MINE["name"] in texts
        finally:
            try:
                engine.stop()
            except Exception:
                pass
            if getattr(app, "jobs", None) is not None:
                app.jobs.close()
            ms.install_sheet_env(None)
            mc.set_default_service(None)
            await TF._stop(app)

        offline_events = [e for e in offline.network
                          if not any("_fetch_provider_catalog" in frame for frame in e.get("stack") or [])]
        assert not offline_events, offline_events[:3]

    asyncio.run(scenario())
