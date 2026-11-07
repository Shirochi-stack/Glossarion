"""Compiled Windows builds have no console; console_window opens one at runtime.

Always when Other Settings > Show CMD is on, otherwise only while manga runs. Source runs and
helper processes never touch it.
"""

import ast
import glob
import io
import json
import os
import sys

import pytest

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import console_window  # noqa: E402

WINDOWS_SPECS = ("translator.spec", "translator_Heavy.spec", "translator_NoCuda.spec",
                 "translator_TurboLite.spec", "translator_lite.spec", "translatoronefileoff.spec")


@pytest.fixture
def win(monkeypatch, tmp_path):
    """Fresh module state with the Win32 calls replaced by a recorder."""
    calls = []
    monkeypatch.setattr(console_window, "_state",
                        {"started": False, "show_cmd": False, "manga": False, "allocated": False})
    monkeypatch.setattr(console_window, "_streams", [])
    monkeypatch.setattr(console_window, "_install_streams", lambda: calls.append("streams"))

    def allocate():
        if not console_window._state["allocated"]:
            calls.append("alloc")
            console_window._state["allocated"] = True
        return True

    monkeypatch.setattr(console_window, "_allocate", allocate)
    monkeypatch.setattr(console_window, "_show", lambda cmd: calls.append(cmd) or True)
    monkeypatch.setattr(sys, "platform", "win32")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"show_cmd": False}), encoding="utf-8")
    return calls, config


def _frozen(monkeypatch, argv=("Glossarion.exe",)):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "argv", list(argv))


def _set_show_cmd(config, value):
    config.write_text(json.dumps({"show_cmd": value}), encoding="utf-8")


SHOW, HIDE = console_window.SW_SHOWNOACTIVATE, console_window.SW_HIDE


def test_source_runs_never_touch_the_console(monkeypatch, win):
    calls, config = win
    monkeypatch.delattr(sys, "frozen", raising=False)
    _set_show_cmd(config, True)
    assert console_window.startup(config_path=str(config)) is False
    console_window.set_show_cmd(True)
    console_window.set_show_cmd(False)
    assert calls == []


def test_off_windows_nothing_is_managed(monkeypatch, win):
    _frozen(monkeypatch)
    monkeypatch.setattr(sys, "platform", "linux")
    assert console_window.managed() is False


def test_no_console_unless_show_cmd_is_on(monkeypatch, win):
    """Owner, 2026-11-08: manga runs no longer open the console; only the Show CMD toggle does."""
    calls, config = win
    _frozen(monkeypatch)
    assert console_window.startup(config_path=str(config)) is False
    assert calls == ["streams"]                       # nothing allocated at startup
    assert not hasattr(console_window, "set_manga_running")
    console_window.set_show_cmd(True)
    console_window.set_show_cmd(False)
    console_window.set_show_cmd(True)
    # allocated once, then hidden / shown again instead of freed
    assert calls == ["streams", "alloc", SHOW, HIDE, SHOW]


def test_show_cmd_opens_it_at_startup(monkeypatch, win):
    calls, config = win
    _frozen(monkeypatch)
    _set_show_cmd(config, True)
    assert console_window.startup(config_path=str(config)) is True
    assert calls == ["streams", "alloc", SHOW]


def test_toggling_show_cmd_takes_effect_immediately(monkeypatch, win):
    calls, config = win
    _frozen(monkeypatch)
    console_window.startup(config_path=str(config))
    console_window.set_show_cmd(True)
    console_window.set_show_cmd(False)
    assert calls == ["streams", "alloc", SHOW, HIDE]


def test_turning_show_cmd_off_before_any_console_does_nothing(monkeypatch, win):
    calls, config = win
    _frozen(monkeypatch)
    console_window.startup(config_path=str(config))
    console_window.set_show_cmd(False)
    assert calls == ["streams"]


def test_startup_runs_once(monkeypatch, win):
    calls, config = win
    _frozen(monkeypatch)
    _set_show_cmd(config, True)
    console_window.startup(config_path=str(config))
    assert console_window.startup(config_path=str(config)) is False
    assert calls.count("streams") == 1 and calls.count("alloc") == 1


@pytest.mark.parametrize("flag", ["--run-translation-instance", "--run-metadata-translation-worker",
                                  "--run-chapter-extraction", "--run-compress-worker", "--run-pdf-worker",
                                  "--autharena-qt-browser", "--authza-captcha", "--authnd-mint-token",
                                  "--gemini-free-search", "--multiprocessing-fork"])
def test_helper_processes_never_open_or_toggle_a_console(monkeypatch, win, flag):
    calls, config = win
    _frozen(monkeypatch, ["Glossarion.exe", flag, "job.json"])
    _set_show_cmd(config, True)
    assert console_window.startup(config_path=str(config)) is False
    console_window.set_show_cmd(True)
    assert calls == []


def test_failed_allocation_shows_nothing(monkeypatch, win):
    calls, config = win
    _frozen(monkeypatch)
    console_window.startup(config_path=str(config))
    monkeypatch.setattr(console_window, "_allocate", lambda: False)
    assert console_window.set_show_cmd(True) is False
    assert calls == ["streams"]


def test_show_cmd_setting_round_trip(tmp_path):
    config = tmp_path / "config.json"
    assert console_window.read_show_cmd(str(config)) is False          # missing file
    config.write_text("{not json", encoding="utf-8")
    assert console_window.read_show_cmd(str(config)) is False          # unreadable
    config.write_text(json.dumps({"other": 1}), encoding="utf-8")
    assert console_window.read_show_cmd(str(config)) is False          # default off
    for value in (True, False):
        data = json.loads(config.read_text(encoding="utf-8"))
        data[console_window.CONFIG_KEY] = value
        config.write_text(json.dumps(data), encoding="utf-8")
        assert console_window.read_show_cmd(str(config)) is value
    assert console_window.CONFIG_KEY == "show_cmd"


def test_stand_in_streams_replace_missing_stdio_and_follow_the_console(monkeypatch):
    monkeypatch.setattr(console_window, "_streams", [])
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "__stdout__", None)
    keep = sys.stderr
    console_window._install_streams()
    out = sys.stdout
    assert isinstance(out, console_window._ConsoleStream)
    assert sys.__stdout__ is out and sys.stderr is keep     # an existing stream is left alone
    assert console_window._streams == [out]
    # before a console exists text is dropped; reconfigure (translator_gui's UTF-8 setup) is fine
    assert out.write("lost\n") == 5
    out.reconfigure(encoding="utf-8", errors="replace")
    out.flush()
    assert out.isatty() is False and out.encoding == "utf-8"
    with pytest.raises(OSError):
        out.fileno()
    target = io.StringIO()
    out.target = target                                    # what _allocate does
    print("hello", file=out)
    assert target.getvalue() == "hello\n"


def test_ctrl_c_and_break_are_ignored_and_close_is_left_to_the_system():
    assert console_window._on_console_event(console_window.CTRL_C_EVENT) is True
    assert console_window._on_console_event(console_window.CTRL_BREAK_EVENT) is True
    assert console_window._on_console_event(console_window.CTRL_CLOSE_EVENT) is False


def test_no_console_window_means_nothing_to_show(monkeypatch):
    monkeypatch.setattr(console_window, "_console_hwnd", lambda: 0)
    assert console_window._show(console_window.SW_HIDE) is False
    assert console_window._disable_close_button(0) is False


class _AttachKernel:
    """kernel32 stand-in for attach_worker: records AttachConsole."""

    def __init__(self, has_console=0, attach_ok=True, redirected=False):
        self.attached, self.has_console, self.attach_ok, self.redirected = [], has_console, attach_ok, redirected

        def attach_console(pid):   # a plain function: the code under test sets .argtypes on it
            self.attached.append(pid)
            return self.attach_ok

        self.AttachConsole = attach_console

    def GetConsoleWindow(self):
        return self.has_console


@pytest.fixture
def attach(monkeypatch):
    kernel = _AttachKernel()
    calls = []
    monkeypatch.setattr(console_window, "_state",
                        {"started": False, "show_cmd": False, "manga": False, "allocated": False})
    monkeypatch.setattr(console_window, "_win", lambda name: kernel)
    monkeypatch.setattr(console_window, "_std_output_is_redirected", lambda k: k.redirected)
    monkeypatch.setattr(console_window, "_install_streams", lambda: calls.append("streams"))
    monkeypatch.setattr(console_window, "_connect", lambda k: calls.append("connect"))
    monkeypatch.setattr(console_window, "_allocate", lambda: calls.append("alloc") or True)
    monkeypatch.setattr(console_window, "_show", lambda cmd: calls.append(cmd) or True)
    monkeypatch.setattr(sys, "platform", "win32")
    return kernel, calls


MP_CHILD = ("Glossarion.exe", "--multiprocessing-fork", "parent_pid=4242", "pipe_handle=528")


def test_multiprocessing_child_attaches_to_the_gui_console(monkeypatch, attach):
    kernel, calls = attach
    _frozen(monkeypatch, MP_CHILD)
    assert console_window.attach_worker() is True
    assert kernel.attached == [4242]
    assert calls == ["streams", "connect"]          # never opens, shows or hides one


def test_multiprocessing_child_without_a_parent_console_stays_without(monkeypatch, attach):
    kernel, calls = attach
    kernel.attach_ok = False
    _frozen(monkeypatch, MP_CHILD)
    assert console_window.attach_worker() is False
    assert kernel.attached == [4242] and calls == []


@pytest.mark.parametrize("case", ["source", "gui", "run_helper", "no_pid", "has_console", "piped"])
def test_attach_worker_leaves_everything_else_alone(monkeypatch, attach, case):
    kernel, calls = attach
    argv = list(MP_CHILD)
    if case == "source":
        monkeypatch.setattr(sys, "argv", argv)
        monkeypatch.delattr(sys, "frozen", raising=False)
    else:
        if case == "gui":
            argv = ["Glossarion.exe"]
        elif case == "run_helper":
            argv = ["Glossarion.exe", "--run-pdf-worker", "parent_pid=4242"]
        elif case == "no_pid":
            argv = ["Glossarion.exe", "--multiprocessing-fork", "pipe_handle=528"]
        elif case == "has_console":
            kernel.has_console = 77
        elif case == "piped":
            kernel.redirected = True
        _frozen(monkeypatch, argv)
    assert console_window.attach_worker() is False
    assert kernel.attached == [] and calls == []


# --- wiring -----------------------------------------------------------------------------------

def _read(name, enc="utf-8-sig"):
    with open(os.path.join(SRC, name), encoding=enc) as f:
        return f.read()


def test_gui_starts_it_after_every_helper_dispatch_and_before_the_gui_imports():
    text = _read("translator_gui.py")
    start = text.index("_console_window.startup()")
    for flag in ("--run-metadata-translation-worker", "--run-translation-instance", "--autharena-qt-browser",
                 "--authza-captcha", "--authnd-mint-token", "--gemini-free-search"):
        assert text.index(flag) < start, flag
    assert start < text.index("import dpi_setup")
    assert "hide_at_startup" not in text
    # multiprocessing children attach to the GUI's console before freeze_support runs them
    assert text.index("_console_window.attach_worker()") < text.index("multiprocessing.freeze_support()")


def test_manga_runs_do_not_open_the_console():
    """Owner, 2026-11-08: "just remove it" (the console a manga run opened)."""
    source = _read("manga_integration.py")
    assert "_set_console_visible" not in source
    assert "set_manga_running" not in source


def test_show_cmd_is_a_toggle_button_next_to_debug_mode():
    """Owner, 2026-11-07: a Debug Mode-style ON/OFF button in the same row (second column), not a checkbox."""
    source = _read("other_settings.py")
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_create_debug_controls_section")
    body = ast.get_source_segment(source, fn)
    assert '_create_styled_checkbox("Show CMD")' not in body
    assert "show_cmd_btn.setCheckable(True)" in body
    assert '"🖥️ Show CMD: ON" if on else "🖥️ Show CMD: OFF"' in body
    assert "#dc3545" in body and "#6c757d" in body              # Debug Mode's ON / OFF colours
    assert ".get('show_cmd', False)" in body
    assert "self.config['show_cmd'] = bool(checked)" in body
    assert "self.save_config(show_message=False)" in body
    assert "console_window.set_show_cmd(bool(checked))" in body
    assert "Compiled builds only" in body
    # same row as Debug Mode, right after it with a small gap, before the layout is added
    i_debug = body.index("button_layout.addWidget(debug_toggle_btn)")
    i_gap = body.index("button_layout.addSpacing(12)")
    i_btn = body.index("button_layout.addWidget(show_cmd_btn)")
    assert i_debug < i_gap < i_btn < body.index("debug_layout.addLayout(button_layout)")


def test_show_cmd_button_toggles_the_setting_and_its_style(monkeypatch, tmp_path):
    """Offscreen: clicking the button saves show_cmd, tells console_window and restyles ON/OFF."""
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QGridLayout, QWidget
    app = QApplication.instance() or QApplication([])
    import other_settings
    calls = []
    monkeypatch.setattr(console_window, "set_show_cmd", lambda on: calls.append(on))

    class Owner:
        config = {"show_cmd": False}
        saved = 0

        def save_config(self, show_message=False):
            Owner.saved += 1

        def append_log(self, msg):
            pass

        def __getattr__(self, name):
            return getattr(other_settings, name).__get__(self) if hasattr(other_settings, name) else None

    owner = Owner()
    parent = QWidget()
    parent.setLayout(QGridLayout())   # the Other Settings page is a grid
    other_settings._create_debug_controls_section(owner, parent)
    btn = owner._show_cmd_btn
    assert btn.text() == "🖥️ Show CMD: OFF" and "#6c757d" in btn.styleSheet()
    btn.click()
    assert owner.config["show_cmd"] is True and calls == [True] and Owner.saved == 1
    assert btn.text() == "🖥️ Show CMD: ON" and "#dc3545" in btn.styleSheet()
    btn.click()
    assert owner.config["show_cmd"] is False and calls == [True, False]
    assert btn.text() == "🖥️ Show CMD: OFF"
    parent.deleteLater()
    app.processEvents()


def test_show_cmd_schema_is_a_desktop_only_debug_bool():
    import settings_schema
    spec = settings_schema.spec("show_cmd")
    assert spec.type == "bool" and spec.label == "Show CMD"
    assert settings_schema.section_of("show_cmd").id == "other.debug"
    assert settings_schema.is_available("show_cmd", "mobile")[0] is False
    assert settings_schema.is_available("show_cmd", "desktop")[0] is True


def test_windows_specs_build_without_a_console():
    for name in WINDOWS_SPECS:
        text = _read(name, enc="utf-8").replace("\r\n", "\n")
        assert "\nENABLE_CONSOLE = False  # No console in the exe: console_window" in text, name
        assert "console=True" not in text, name


def test_every_spec_bundles_the_module():
    for spec in glob.glob(os.path.join(SRC, "*.spec")):
        text = _read(os.path.basename(spec), enc="utf-8")
        assert "('console_window.py', '.')" in text and "'console_window'," in text, spec


@pytest.mark.skipif(sys.platform != "win32", reason="Windows console API")
def test_console_hwnd_is_harmless_in_a_real_process():
    assert isinstance(console_window._console_hwnd(), int)
