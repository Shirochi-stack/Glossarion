"""Host tests for Model manager › Ollama: Ollama in Termux, Hugging Face / library pulls and the
editable presets (``services/ollama_termux.py``, ``ui/screens/ollama_panel.py``, the shared
``ollamapull.normalize_pull_name``) and the Termux calls of the native bridge.

No network and no Termux: pulls, server probes and the native extension are fakes.

Run from src/mobile:
    .venv/Scripts/python.exe -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic \
        tests_host/test_ollama_termux.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
import threading
import time
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

os.environ.setdefault("GLOSSARION_HTTP_LOG", "0")

from glossarion_mobile.services import ollama_termux as ot  # noqa: E402


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


needs_flet = pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet/msgpack not installed")


class FakePrefs:
    def __init__(self, data=None):
        self.data = dict(data or {})

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value


class FakeNative:
    """NativeBridge stand-in: records Termux calls and answers ``termux_run`` with ``answer``."""

    is_stub = False

    def __init__(self, answer=None, opened=True):
        self.answer = answer if answer is not None else {"ok": True, "error": None}
        self.opened = opened
        self.runs = []
        self.opens = 0

    async def termux_run(self, command, *, label="", background=False):
        self.runs.append((command, label, background))
        return dict(self.answer)

    async def open_termux(self):
        self.opens += 1
        return self.opened


@pytest.fixture(autouse=True)
def _no_pull_url(monkeypatch):
    monkeypatch.setenv("OLLAMAPULL_BASE_URL", "")  # recorded, so a test's own address is undone afterwards
    monkeypatch.delenv("OLLAMAPULL_BASE_URL")


# ==========================================================================
# model names (shared ollamapull.normalize_pull_name)
# ==========================================================================


@pytest.mark.parametrize("typed,expected", [
    ("qwen3.5:2b", "qwen3.5:2b"),
    ("  Qwen3.5:9B ", "qwen3.5:9b"),
    ("ollamapull/qwen3.5:0.8b", "qwen3.5:0.8b"),
    ("llama3.2", "llama3.2"),
    ("huihui_ai/qwen3-abliterated:4b", "huihui_ai/qwen3-abliterated:4b"),
    ("hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_M", "hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_M"),
    ("https://huggingface.co/unsloth/Qwen3.5-2B-GGUF", "hf.co/unsloth/Qwen3.5-2B-GGUF"),
    ("huggingface.co/bartowski/Some-Model-GGUF/tree/main", "hf.co/bartowski/Some-Model-GGUF"),
    ("unsloth/Qwen3.5-0.8B-GGUF:Q8_0", "hf.co/unsloth/Qwen3.5-0.8B-GGUF:Q8_0"),
])
def test_pull_names_accept_library_tags_and_hugging_face_ids(typed, expected):
    import ollamapull

    assert ollamapull.normalize_pull_name(typed) == expected
    assert ot.normalize_model(typed) == (expected, None)


@pytest.mark.parametrize("typed", ["", "   ", "qwen 3", "hf.co/onlyuser", "hf.co/u/r:bad quant", "rm -rf /",
                                   "qwen3.5:2b; reboot", "a/b/c/d", "https://example.com/x"])
def test_pull_names_reject_garbage(typed):
    name, error = ot.normalize_model(typed)
    assert name is None and error


# ==========================================================================
# presets (data in Prefs)
# ==========================================================================


def test_default_presets_are_the_four_qwen35_small_models_as_data():
    prefs = FakePrefs()
    presets = ot.load_presets(prefs)
    assert [p["model"] for p in presets] == ["qwen3.5:0.8b", "qwen3.5:2b", "qwen3.5:4b", "qwen3.5:9b"]
    assert [p["name"] for p in presets] == ["Garbage", "Light", "Medium", "Heavy"]
    # the defaults are in the shared curated local list (model_options), not invented here
    import model_options

    assert {p["model"] for p in presets} <= set(model_options.LOCAL_CHAT_MODEL_TAGS)
    assert ot.PREF_KEY not in prefs.data  # reading never writes
    presets[0]["name"] = "changed"
    assert ot.load_presets(prefs)[0]["name"] == "Garbage"  # a copy, not the module data


def test_presets_rename_edit_remove_and_restore_persist():
    prefs = FakePrefs()
    presets = ot.load_presets(prefs)
    row, error = ot.validate_preset({"name": "Tiny", "model": "qwen3.5:0.8b"}, presets, 0)
    assert error is None and row["name"] == "Tiny"
    presets[0] = row
    ot.save_presets(prefs, presets)
    assert ot.load_presets(prefs)[0]["name"] == "Tiny"
    # duplicate names (any case) are refused, except the row's own name
    assert ot.validate_preset({"name": "light", "model": "x:1"}, presets, 0)[1] == "A preset is already called light"
    assert ot.validate_preset({"name": "Light", "model": "qwen3.5:2b"}, presets, 1)[1] is None
    assert ot.validate_preset({"name": "", "model": "x:1"}, presets)[1] == "Give the preset a name"
    assert "Hugging Face" in ot.validate_preset({"name": "Bad", "model": "hf.co/x"}, presets)[1]
    # an emptied list stays empty (never refilled with the defaults)
    ot.save_presets(prefs, [])
    assert ot.load_presets(prefs) == []
    assert len(ot.restore_defaults(prefs)) == 4 and len(ot.load_presets(prefs)) == 4
    # junk in the saved list is dropped
    prefs.set(ot.PREF_KEY, [{"name": "ok", "model": "m:1"}, {"name": ""}, "junk", None])
    assert ot.load_presets(prefs) == [{"name": "ok", "model": "m:1", "note": ""}]


def test_route_model_installed_and_loopback_helpers():
    assert ot.route_model("qwen3.5:2b") == "ollamapull/qwen3.5:2b"
    assert ot.route_model("ollamapull/qwen3.5:2b") == "ollamapull/qwen3.5:2b"
    assert ot.installed("qwen3.5:2b", ["qwen3.5:2b", "x"])
    assert ot.installed("llama3.2", ["llama3.2:latest"])
    assert not ot.installed("qwen3.5:4b", ["qwen3.5:2b"])
    assert ot.installed("hf.co/u/r-GGUF:Q4_K_M", ["hf.co/u/r-GGUF:Q4_K_M"])
    assert ot.is_loopback("") and ot.is_loopback("http://127.0.0.1:11434") and ot.is_loopback("localhost:11434")
    assert not ot.is_loopback("http://192.168.1.10:11434")


def test_manual_steps_cover_install_serve_pull_and_external_apps():
    commands = [cmd for _title, cmd in ot.manual_steps("qwen3.5:2b") if cmd]
    assert commands[:2] == ["pkg install -y ollama", "ollama serve"]
    assert "ollama pull qwen3.5:2b" in commands
    assert any("allow-external-apps=true" in c and "termux-reload-settings" in c for c in commands)
    assert "pkg install -y ollama" in ot.SETUP_COMMAND and ot.SETUP_COMMAND.endswith("ollama serve")
    # the pasted first-run setup also turns on one-tap setup, without duplicating the property line
    assert ot.FIRST_RUN_SETUP_COMMAND.startswith(ot.TERMUX_ALLOW_COMMAND)
    assert ot.FIRST_RUN_SETUP_COMMAND.endswith(ot.SETUP_COMMAND) and "grep -qs" in ot.TERMUX_ALLOW_COMMAND


def test_the_clipboard_fallback_pastes_the_first_run_command():
    copied = []
    native = FakeNative({"ok": False, "error": "permission"})
    outcome = asyncio.run(ot.run_in_termux(native, ot.SETUP_COMMAND, copy_text=copied.append,
                                           fallback_command=ot.FIRST_RUN_SETUP_COMMAND))
    assert outcome.mode == "clipboard" and copied == [ot.FIRST_RUN_SETUP_COMMAND]
    sent = FakeNative()  # one-tap works: the plain setup is sent, nothing is copied
    outcome = asyncio.run(ot.run_in_termux(sent, ot.SETUP_COMMAND, copy_text=copied.append,
                                           fallback_command=ot.FIRST_RUN_SETUP_COMMAND))
    assert outcome.mode == "intent" and sent.runs[0][0] == ot.SETUP_COMMAND and len(copied) == 1


# ==========================================================================
# Termux: RUN_COMMAND, else copy + open Termux
# ==========================================================================


def test_run_in_termux_uses_the_intent_and_opens_termux():
    native = FakeNative()
    copied = []
    outcome = asyncio.run(ot.run_in_termux(native, ot.SETUP_COMMAND, label="Ollama server", copy_text=copied.append))
    assert outcome.mode == "intent" and outcome.ok
    assert native.runs == [(ot.SETUP_COMMAND, "Ollama server", False)] and native.opens == 1 and copied == []


def test_run_in_termux_falls_back_to_the_clipboard():
    # no native extension (desktop / host / stub): copy and say how to paste
    copied = []
    outcome = asyncio.run(ot.run_in_termux(None, "ollama serve", copy_text=copied.append))
    assert outcome.mode == "clipboard" and copied == ["ollama serve"] and "Paste" in outcome.message
    # permission missing: the command is copied, Termux opened, and the message explains the permission
    native = FakeNative({"ok": False, "error": "permission"})
    copied.clear()

    async def acopy(text):
        copied.append(text)

    outcome = asyncio.run(ot.run_in_termux(native, "ollama serve", copy_text=acopy))
    assert outcome.mode == "clipboard" and outcome.error == "permission" and copied == ["ollama serve"]
    assert "Run commands in Termux" in outcome.message and native.opens == 1 and "Termux is open" in outcome.message
    # the permission prompt was just shown: Termux must not be opened over it
    native = FakeNative({"ok": False, "error": "permission", "requested": True})
    copied.clear()
    outcome = asyncio.run(ot.run_in_termux(native, "ollama serve", copy_text=acopy))
    assert outcome.error == "permission" and native.opens == 0 and copied == ["ollama serve"]
    assert "again" in outcome.message and "Termux is open" not in outcome.message
    # Termux missing: no copy, point at F-Droid
    native = FakeNative({"ok": False, "error": "not_installed"})
    outcome = asyncio.run(ot.run_in_termux(native, "ollama serve", copy_text=copied.append))
    assert outcome.mode == "missing" and "F-Droid" in outcome.message
    # a stub bridge is never asked
    stub = FakeNative()
    stub.is_stub = True
    outcome = asyncio.run(ot.run_in_termux(stub, "ollama serve", copy_text=None))
    assert stub.runs == [] and outcome.mode == "clipboard" and "type: ollama serve" in outcome.message


def test_native_bridge_termux_calls_and_stub_defaults():
    from glossarion_mobile.services.native import NativeBridge, NativeStub

    class Ext:
        def __init__(self):
            self.calls = []

        async def termux_run(self, command, label="", background=False):
            self.calls.append(("run", command, label, background))
            return {"ok": True}

        async def termux_status(self):
            return {"installed": True}

        async def open_termux(self):
            return True

    ext = Ext()
    bridge = NativeBridge(native=ext)
    assert asyncio.run(bridge.termux_run("ls", label="x", background=True)) == {"ok": True}
    assert ext.calls == [("run", "ls", "x", True)]
    assert asyncio.run(bridge.termux_status()) == {"installed": True, "permission": False}
    assert asyncio.run(bridge.open_termux()) is True
    stub = NativeBridge(native=NativeStub())
    assert asyncio.run(stub.termux_run("ls"))["error"] == "unavailable"
    assert asyncio.run(stub.termux_status()) == {"installed": False, "permission": False}
    assert asyncio.run(stub.open_termux()) is False


# ==========================================================================
# pulls: shared ollamapull.pull_model off the loop, progress back on it
# ==========================================================================


def test_pull_manager_reports_progress_on_the_loop_and_finishes():
    loop_threads = []
    seen = []

    def fake_pull(model, progress=None, should_stop=None):
        assert threading.current_thread() is not threading.main_thread()
        for pct in (0, 40, 100):
            progress(f"{model}: pulling 1a2b {pct}% · 3.0 MB/s · ETA 5s")
            time.sleep(0.01)
        progress(f"{model}: success")

    async def scenario():
        manager = ot.PullManager(pull=fake_pull, throttle=0)

        def listener(state):
            loop_threads.append(threading.current_thread())
            seen.append((state.status, state.fraction, state.running))

        manager.subscribe(listener)
        state = await manager.pull("qwen3.5:2b")
        await asyncio.sleep(0)
        return state

    state = asyncio.run(scenario())
    assert state.done and not state.running and state.error is None and state.fraction == 1.0
    assert all(t is threading.main_thread() for t in loop_threads)  # listeners run on the loop thread
    assert any(f == 0.4 and "40%" in s for s, f, _r in seen)
    assert seen[-1] == ("Downloaded ✓", 1.0, False)


def test_pull_manager_errors_and_stop():
    import ollamapull

    def failing(model, progress=None, should_stop=None):
        raise ollamapull.OllamaPullError("Ollama /api/pull failed (HTTP 404): file does not exist")

    async def fail():
        return await ot.PullManager(pull=failing).pull("hf.co/u/missing-GGUF")

    state = asyncio.run(fail())
    assert state.error and "404" in state.status and not state.running and not state.done

    started = threading.Event()

    def slow(model, progress=None, should_stop=None):
        started.set()
        while not should_stop():
            time.sleep(0.01)
        raise ollamapull.OllamaPullCancelled("Ollama operation cancelled")

    async def stop():
        manager = ot.PullManager(pull=slow)
        task = asyncio.ensure_future(manager.pull("qwen3.5:9b"))
        while not started.is_set():
            await asyncio.sleep(0.01)
        assert manager.running("qwen3.5:9b")
        again = await manager.pull("qwen3.5:9b")  # a second tap joins, never a second download
        assert again is manager.states["qwen3.5:9b"]
        assert manager.cancel("qwen3.5:9b")
        return await task

    state = asyncio.run(stop())
    assert state.cancelled and state.status == "Stopped" and state.error is None


def test_server_status_never_raises():
    assert asyncio.run(ot.server_status(probe=lambda: {"running": True, "version": "0.12", "models": ["a"]}))["running"]

    def boom():
        raise OSError("connection refused")

    status = asyncio.run(ot.server_status(probe=boom))
    assert status["running"] is False and "refused" in status["error"]


def test_server_status_reads_the_ollamapull_address(monkeypatch):
    import ollamapull

    monkeypatch.setenv("OLLAMAPULL_BASE_URL", "http://127.0.0.1:11434")
    monkeypatch.setattr(ollamapull, "_server_version", lambda: "0.12.5")
    monkeypatch.setattr(ollamapull, "_models", lambda: [{"name": "qwen3.5:2b"}, {"model": "llama3.2:latest"}])
    status = asyncio.run(ot.server_status())
    assert status == {"running": True, "version": "0.12.5", "models": ["qwen3.5:2b", "llama3.2:latest"],
                      "error": None, "url": "http://127.0.0.1:11434"}
    monkeypatch.setattr(ollamapull, "_server_version", lambda: None)
    status = asyncio.run(ot.server_status())
    assert not status["running"] and "127.0.0.1:11434" in status["error"]


# ==========================================================================
# the Model manager's Ollama tab
# ==========================================================================

_TB_SPEC = importlib.util.spec_from_file_location("_glossarion_tb_helpers_ollama", Path(__file__).with_name("test_bootstrap.py"))
_MK_SPEC = importlib.util.spec_from_file_location("_glossarion_mk_helpers_ollama", Path(__file__).with_name("test_models_keys.py"))


def _helpers():
    tb = importlib.util.module_from_spec(_TB_SPEC)
    _TB_SPEC.loader.exec_module(tb)
    mk = importlib.util.module_from_spec(_MK_SPEC)
    _MK_SPEC.loader.exec_module(mk)
    return tb, mk


def _texts(control) -> list:
    out = []

    def walk(c):
        if c is None:
            return
        value = getattr(c, "value", None)
        if isinstance(value, str) and type(c).__name__ == "Text":
            out.append(value)
        for attr in ("content", "controls"):
            child = getattr(c, attr, None)
            if isinstance(child, list):
                for item in child:
                    walk(item)
            elif child is not None and not isinstance(child, str):
                walk(child)

    walk(control)
    return out


@needs_flet
def test_model_manager_ollama_tab_presets_pull_and_use(tmp_path, monkeypatch):
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.screens.model_manager import TABS, ModelManagerScreen
    from glossarion_mobile.ui.screens.ollama_panel import OllamaPanel, OllamaPresetEditor

    tb, mk = _helpers()

    pulled = []

    def fake_pull(model, progress=None, should_stop=None):
        pulled.append(model)
        progress(f"{model}: pulling abc 50%")
        progress(f"{model}: success")

    async def scenario():
        conn, session = tb._fake_session("android")
        page = session.page
        store = mk._store(tmp_path, {"model": "gpt-6"})
        service = mk._service(store, mk.FakeOptions(catalog=["gpt-6"]))
        service.load_blocking()
        prefs = FakePrefs()
        native = FakeNative()
        notes, copied, used = [], [], []
        server = {"running": True, "version": "0.12.5", "models": ["qwen3.5:0.8b"], "error": None}
        pulls = ot.PullManager(pull=fake_pull, throttle=0)
        screen = ModelManagerScreen(parse_route("/settings/models"), catalog=service, page=page,
                                    notify=lambda m, a=None, cb=None: notes.append(m),
                                    spawn=lambda c: asyncio.ensure_future(c), prefs=prefs, native=native,
                                    copy_text=copied.append, use_model=used.append, ollama_pulls=pulls)
        page.views[0].controls.append(screen.get_body())
        screen.did_show()
        page.update()
        # four tabs; the Ollama panel is built only when its tab is first shown
        assert [s.value for s in screen.tabs.segments] == [v for v, _l in TABS] and screen.ollama is None
        monkeypatch.setattr(OllamaPanel, "did_show", lambda self: None)  # no probe through requests
        screen.set_tab("ollama")
        panel = screen.ollama
        assert isinstance(panel, OllamaPanel) and screen.ollama_box.visible and not screen.prefix_list.visible
        assert not screen.rows.control.visible and screen.add_fab.content == "Add preset"
        panel.probe = lambda: dict(server)
        await panel.check_server()
        assert panel.status_text.value.startswith("✓ Ollama 0.12.5 is running · 1 model")
        assert "this phone (Termux)" in panel.address_text.value and not panel.phone_button.visible
        texts = _texts(panel.preset_column)
        for name in ("Garbage", "Light", "Medium", "Heavy", "qwen3.5:9b"):
            assert name in texts, name
        assert panel.preset_column.controls[0].content.controls[1].key == "ollama-preset-have-0"  # 0.8b is there

        # rename a preset through the editor (FAB on this tab adds a preset)
        editor = panel.open_editor(1)
        assert isinstance(editor, OllamaPresetEditor) and editor.name.value == "Light"
        editor.name.value = "Fast"
        assert editor.save() is None
        assert ot.load_presets(prefs)[1]["name"] == "Fast" and "Fast" in _texts(panel.preset_column)
        new = screen.open_add()
        assert isinstance(new, OllamaPresetEditor)
        new.name.value, new.model.value = "Gemma", "not a model!"
        assert new.save() and new.error.visible
        new.model.value = "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF"
        assert new.save() is None
        assert ot.load_presets(prefs)[-1] == {"name": "Gemma", "model": "hf.co/unsloth/gemma-4-E2B-it-GGUF", "note": ""}

        # pull a typed id: validated first, then the shared pull with progress rows
        panel.pull_field.value = "qwen 3"
        assert await panel.pull_typed() is None and panel.pull_field.error
        panel.pull_field.value = "Qwen3.5:4B"
        server["models"] = ["qwen3.5:0.8b", "qwen3.5:4b"]
        state = await panel.pull_typed()
        assert state.done and pulled == ["qwen3.5:4b"] and panel.pull_field.error is None
        box, bar, text, stop = panel.download_rows["qwen3.5:4b"]
        assert bar.value == 1.0 and text.value == "Downloaded ✓" and stop.visible is False
        assert "Downloaded qwen3.5:4b" in notes
        assert panel.preset_column.controls[2].content.controls[1].key == "ollama-preset-have-2"

        # use a preset: ollamapull/<tag> becomes the model and joins the model list
        assert panel.use_preset(1) == "ollamapull/qwen3.5:2b"
        assert used == ["ollamapull/qwen3.5:2b"] and "ollamapull/qwen3.5:2b" in service.snapshot.models

        # remove + restore defaults (confirm dialogs)
        dialog = panel.confirm_remove(0)
        await dialog._on_confirm()
        assert [p["name"] for p in ot.load_presets(prefs)][:1] == ["Fast"]
        dialog = panel.confirm_restore()
        await dialog._on_confirm()
        assert [p["name"] for p in ot.load_presets(prefs)] == ["Garbage", "Light", "Medium", "Heavy"]

        # Set up in Termux sends the one-tap command through the native bridge
        outcome = await panel.setup_termux()
        assert outcome.mode == "intent" and native.runs[-1][0] == ot.SETUP_COMMAND and panel.termux_text.visible
        # a PC address: "Use this phone" points ollamapull/ back at Termux
        from glossarion_mobile.state import remote_routes

        remote_routes.apply_url(prefs, "ollamapull", "http://192.168.1.10:11434", save=True)
        panel._sync_server()
        assert panel.phone_button.visible and "your PC" in panel.address_text.value
        assert panel.use_phone_server() == ot.TERMUX_URL
        assert prefs.get("ollamapull_base_url") == ot.TERMUX_URL and os.environ["OLLAMAPULL_BASE_URL"] == ot.TERMUX_URL
        await asyncio.sleep(0.05)
        page.update()
        screen.set_tab("models")
        assert not screen.ollama_box.visible and screen.rows.control.visible
        screen.dispose()
        store._saver.close()

    asyncio.run(scenario())


@needs_flet
def test_model_manager_tabs_fit_a_phone():
    """Four segments fit 360 dp and 412 dp (no selected check, short labels); larger text scrolls the tab
    row instead of overflowing the page."""
    import flet as ft

    from glossarion_mobile.ui.screens.model_manager import TABS

    char_px, padding_px, gutter = 7.6, 24 + 2, 2 * 12  # labelLarge 14 sp average glyph, segment padding + border
    width = sum(len(label) * char_px + padding_px for _v, label in TABS)
    assert width <= 360 - gutter, width
    assert width <= 412 - gutter

    from glossarion_mobile.ui.screens import model_manager as mm

    source = Path(mm.__file__).read_text(encoding="utf-8")
    assert "show_selected_icon=False" in source and "ft.Row([self.tabs], scroll=ft.ScrollMode.AUTO)" in source
    assert ft.SegmentedButton  # the control the page uses


@needs_flet
def test_ollama_panel_layout_fits_360_dp():
    """Preset rows: text column expands, two 48 dp controls trail; commands wrap (no fixed widths)."""
    import flet as ft

    from glossarion_mobile.ui.screens.ollama_panel import OllamaPanel

    panel = OllamaPanel(prefs=FakePrefs(), pulls=ot.PullManager(pull=lambda *a, **k: None))
    control = panel.build()
    row = panel.preset_column.controls[0].content
    assert isinstance(row, ft.Row) and row.controls[0].expand
    trailing = 48 + 48
    assert 360 - 2 * 16 - 12 - trailing >= 200  # room left for the name / model / note column
    for text in row.controls[0].controls:
        assert text.max_lines and text.overflow == ft.TextOverflow.ELLIPSIS
    # no control in the panel has a fixed width wider than a 360 dp phone
    stack = [control]
    while stack:
        c = stack.pop()
        width = getattr(c, "width", None)
        assert width is None or width <= 320, (type(c).__name__, width)
        for attr in ("content", "controls"):
            child = getattr(c, attr, None)
            if isinstance(child, list):
                stack.extend(child)
            elif child is not None and not isinstance(child, str) and hasattr(child, "__dict__"):
                stack.append(child)
