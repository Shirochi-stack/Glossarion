"""Fakes used to drive frozen (legacy) and extracted (new) owners without Qt.

* Fake widgets mirror only the Qt API surface the desktop code touches, so
  ``hasattr(widget, 'isChecked')`` / ``hasattr(widget, 'text')`` /
  ``hasattr(widget, 'get')`` answer exactly as the real QLineEdit / QCheckBox /
  QTextEdit / QComboBox would (``_bool_setting`` and ``_live_bool_setting``
  dispatch on those).
* ``FakeState`` is a plain object: an attribute exists only after init code
  sets it, so ``hasattr``/``getattr(..., default)`` branches behave like the
  desktop at the same point of its lifetime.
* GUI-only methods that are never frozen (``append_log``, watchdog widgets,
  executor) become recorders that append to a shared ``CallRecorder``.
* ``make_recording_unified_client`` returns a ``UnifiedClient`` subclass whose
  ``set_/clear_in_memory_*`` class methods and constructor only record.

``make_legacy_owner_factory(bundle)`` returns the ``owner_factory(scenario, ctx)``
used by ``capture_golden.capture``; U2 builds its own factory for the extracted
mixins / HeadlessOwner with the same signature.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parents[1]
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------


class CallRecorder:
    """Ordered log of calls into non-frozen GUI methods and client pools."""

    def __init__(self):
        self.calls: list = []

    def record(self, name: str, args=(), kwargs=None):
        self.calls.append([name, list(args), dict(sorted((kwargs or {}).items()))])

    def clear(self):
        self.calls.clear()


# ---------------------------------------------------------------------------
# Fake widgets (only the Qt surface the desktop code uses)
# ---------------------------------------------------------------------------


class _FakeWidgetBase:
    """Common QWidget-ish no-ops (visibility/enabled/signals/tooltips)."""

    def __init__(self):
        self._enabled = True
        self._visible = True
        self._signals_blocked = False

    def setEnabled(self, value):
        self._enabled = bool(value)

    def isEnabled(self):
        return self._enabled

    def setVisible(self, value):
        self._visible = bool(value)

    def isVisible(self):
        return self._visible

    def show(self):
        self._visible = True

    def hide(self):
        self._visible = False

    def blockSignals(self, value):
        previous = self._signals_blocked
        self._signals_blocked = bool(value)
        return previous

    def setToolTip(self, _text):
        return None

    def setStyleSheet(self, _text):
        return None

    def update(self):
        return None

    def objectName(self):
        return type(self).__name__

    def describe(self) -> dict:
        return {"__fake__": type(self).__name__}


class FakeWidget(_FakeWidgetBase):
    """Plain container/label/button/layout stand-in (frame, labels, buttons)."""

    def addWidget(self, *_args, **_kwargs):
        return None

    def setText(self, _text):
        return None


def _qt_text(text):
    """What a Qt text widget holds after ``setText(text)``: PySide6 turns ``None`` into ''."""
    return "" if text is None else str(text)


class FakeLineEdit(_FakeWidgetBase):
    """QLineEdit: text()/setText() (``None`` reads back as '', like Qt)."""

    def __init__(self, text=""):
        super().__init__()
        self._text = _qt_text(text)

    def text(self):
        return self._text

    def setText(self, text):
        self._text = _qt_text(text)

    def setPlaceholderText(self, _text):
        return None

    def clear(self):
        self._text = ""

    def describe(self):
        return {"__fake__": "FakeLineEdit", "text": self._text}


class FakeCheck(_FakeWidgetBase):
    """QCheckBox: isChecked()/setChecked(); text() is the label like Qt."""

    def __init__(self, checked=False, label=""):
        super().__init__()
        self._checked = bool(checked)
        self._label = str(label)

    def isChecked(self):
        return self._checked

    def setChecked(self, value):
        self._checked = bool(value)

    def text(self):
        return self._label

    def setText(self, text):
        self._label = str(text)

    def describe(self):
        return {"__fake__": "FakeCheck", "checked": self._checked}


class FakeTextEdit(_FakeWidgetBase):
    """QTextEdit / QPlainTextEdit: toPlainText()/setPlainText() (``None`` -> '', like Qt)."""

    def __init__(self, plain=""):
        super().__init__()
        self._plain = _qt_text(plain)

    def toPlainText(self):
        return self._plain

    def setPlainText(self, text):
        self._plain = _qt_text(text)

    def setText(self, text):
        self._plain = _qt_text(text)

    def describe(self):
        return {"__fake__": "FakeTextEdit", "plain": self._plain}


class FakeCombo(_FakeWidgetBase):
    """QComboBox with (text, data) items."""

    def __init__(self, items=(), index=0, text=None):
        super().__init__()
        self._items = [(str(t), d) for t, d in (items or [])]
        self._index = index if self._items else -1
        self._edit_text = text

    @classmethod
    def with_data(cls, items, data, fallback_index=0):
        combo = cls(items)
        idx = combo.findData(data)
        combo.setCurrentIndex(idx if idx >= 0 else fallback_index)
        return combo

    def count(self):
        return len(self._items)

    def currentIndex(self):
        return self._index

    def setCurrentIndex(self, index):
        self._index = int(index)
        self._edit_text = None

    def currentText(self):
        if self._edit_text is not None:
            return self._edit_text
        if 0 <= self._index < len(self._items):
            return self._items[self._index][0]
        return ""

    def setCurrentText(self, text):
        idx = self.findText(text)
        if idx >= 0:
            self._index = idx
            self._edit_text = None
        else:
            self._edit_text = str(text)

    def currentData(self):
        if self._edit_text is None and 0 <= self._index < len(self._items):
            return self._items[self._index][1]
        return None

    def findData(self, data):
        for i, (_t, d) in enumerate(self._items):
            if d == data:
                return i
        return -1

    def findText(self, text):
        for i, (t, _d) in enumerate(self._items):
            if t == text:
                return i
        return -1

    def itemText(self, i):
        return self._items[i][0]

    def itemData(self, i):
        return self._items[i][1]

    def describe(self):
        return {"__fake__": "FakeCombo", "text": self.currentText(), "data": self.currentData()}


class FakeSignal:
    """Qt Signal stand-in: emit() is recorded."""

    def __init__(self, recorder: CallRecorder, name: str):
        self._recorder = recorder
        self._name = name

    def emit(self, *args):
        self._recorder.record(f"signal:{self._name}", args)

    def connect(self, *_args, **_kwargs):
        return None

    def describe(self):
        return {"__fake__": "FakeSignal", "name": self._name}


FAKE_WIDGET_TYPES = (FakeWidget, FakeLineEdit, FakeCheck, FakeTextEdit, FakeCombo)

#: Signal class attributes of TranslatorGUI @ BASE_SHA (instances get FakeSignal recorders).
DESKTOP_SIGNALS = (
    "log_signal",
    "auth_status_ready_signal",
    "log_queue_ready_signal",
    "thread_complete_signal",
    "trigger_qa_scan_signal",
    "refresh_preview_signal",
    "open_progress_manager_signal",
    "input_files_updated_signal",
    "parallel_epub_restore_finished_signal",
    "model_catalog_updated_signal",
    "antigravity_proxy_started_signal",
    "glm_proxy_started_signal",
    "direct_text_glossary_approval_signal",
)


# ---------------------------------------------------------------------------
# FakeState + recorder methods
# ---------------------------------------------------------------------------


class FakeState:
    """Plain owner state. Internals live under ``_parity_*`` and are not snapshotted."""

    def __init__(self, recorder: CallRecorder):
        object.__setattr__(self, "_parity_recorder", recorder)
        for name in DESKTOP_SIGNALS:
            object.__setattr__(self, name, FakeSignal(recorder, name))


def _make_recorder_method(name: str):
    def recorder_method(self, *args, **kwargs):
        self._parity_recorder.record(name, args, kwargs)
        return None

    recorder_method.__name__ = name
    recorder_method.__qualname__ = f"FakeState.{name}"
    return recorder_method


def make_owner_class(class_name: str, bases: tuple, recorded_methods) -> type:
    """FakeState first, then the method providers; recorders for GUI-only methods."""
    namespace = {name: _make_recorder_method(name) for name in sorted(set(recorded_methods))}
    namespace["__module__"] = __name__
    return type(class_name, bases, namespace)


def install_widgets(owner, widgets: dict) -> None:
    for attr, widget in widgets.items():
        setattr(owner, attr, widget)


def bind_other_settings_methods(owner, bound: dict) -> None:
    """Mirror setup_other_settings_methods: bind frozen functions or recorders as instance methods."""
    import types

    for name, fn in bound.items():
        target = fn if fn is not None else _make_recorder_method(name)
        setattr(owner, name, types.MethodType(target, owner))


# ---------------------------------------------------------------------------
# UnifiedClient recorder
# ---------------------------------------------------------------------------


def make_recording_unified_client(recorder: CallRecorder):
    """Subclass of the live UnifiedClient whose pool setters and constructor only record."""
    import unified_api_client

    base = unified_api_client.UnifiedClient
    namespace = {"__module__": __name__}

    def _pool_method(name):
        def method(cls, *args, **kwargs):
            recorder.record(f"UnifiedClient.{name}", args, kwargs)
            return None

        method.__name__ = name
        return classmethod(method)

    for name in dir(base):
        if name.startswith(("set_in_memory_", "clear_in_memory_")):
            namespace[name] = _pool_method(name)

    def __init__(self, *args, **kwargs):  # noqa: N807 - recorder constructor
        recorder.record("UnifiedClient.__init__", args, kwargs)
        self._parity_init_kwargs = dict(kwargs)

    namespace["__init__"] = __init__
    return type("RecordingUnifiedClient", (base,), namespace)


# ---------------------------------------------------------------------------
# Legacy owner factory
# ---------------------------------------------------------------------------

#: Boot phases replayed in desktop __init__ order (translator_gui @ BASE_SHA):
#:   config load -> config block -> default prompts -> _init_variables ->
#:   setup_other_settings_methods (initialize_extraction_variables + method binding)
#:   -> _setup_gui (GUI-backed state, widgets, startup handlers incl. the glossary
#:   shortcut handler that runs save_config) -> watchdog dir ->
#:   initialize_environment_variables -> MetadataBatchTranslatorUI (3rd) + the
#:   auto-encryption save_config (config with a plain api_key / replicate_api_key).
#: Verified against a real offscreen TranslatorGUI (tests/parity/real_gui_probe.py, all
#: scenarios): identical attrs/config/translation env/startup env except normalised
#: cpu_count, sandbox paths, the executor and update-manager state.
#: Not replayed (GUI/process only): window/splash setup, UpdateManager
#: (config['last_update_check_time']), the real executor (_ensure_executor is a
#: recorder), _attach_gui_logging_handlers, restoring last input files /
#: Parallel EPUB pair, and _create_model_section's AuthGem project restore (module
#: cache + GOOGLE_CLOUD_PROJECT; no scenario sets authgem_project).
LEGACY_BOOT_PHASES = (
    "pre_config",
    "init_block",
    "default_prompts",
    "init_variables",
    "other_settings_methods",
    "gui_state",
    "widgets",
    "gui_handlers",
    "watchdog",
    "startup_env",
    "post_startup",
)


#: Live GUI-free modules whose path globals the owner factory points at the sandbox
#: (they resolve CONFIG_FILE / _APP_DIR / __file__ at import time).
SHARED_PATH_MODULES = ("app_paths", "owner_state", "run_env", "settings_persistence", "headless_owner")


def shared_mixin_classes(bundle) -> tuple:
    """The legacy oracle's shared mixin classes (U2+; empty before).

    These are the FROZEN copies freeze_legacy saved at the oracle's commit, never the
    live working-tree modules: otherwise a later milestone's edit to run_env /
    owner_state / settings_persistence would be compared against itself.
    """
    return tuple(bundle.mixin_classes())


def patch_frozen_mixin_paths(ctx, bundle) -> None:
    """Sandbox the path globals and backend entry points of the frozen mixin modules."""
    sb = ctx.sandbox
    for module_name, module in (getattr(bundle, "mixins", None) or {}).items():
        ns = vars(module)
        ctx.patch_dict(ns, "__file__", str(sb.src_file(f"{module_name}.py")))
        if "_APP_DIR" in ns:
            ctx.patch_dict(ns, "_APP_DIR", str(sb.app_dir))
        if "CONFIG_FILE" in ns:
            ctx.patch_dict(ns, "CONFIG_FILE", str(sb.config_file))
        for name, stub in ctx.backend_stubs.items():
            if name in ns:
                ctx.patch_dict(ns, name, stub)
        config_file = ns.get("CONFIG_FILE")
        if config_file is not None and not str(config_file).startswith(str(sb.root)):
            raise RuntimeError(
                f"refusing to run: frozen {module_name}.CONFIG_FILE {config_file!r} is outside the sandbox")


def patch_shared_module_paths(ctx) -> None:
    """Point the live shared modules' path globals at the capture sandbox."""
    import importlib

    sb = ctx.sandbox
    for module_name in SHARED_PATH_MODULES:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        ctx.patch(module, "__file__", str(sb.src_file(f"{module_name}.py")))
        if hasattr(module, "_APP_DIR"):
            ctx.patch(module, "_APP_DIR", str(sb.app_dir))
        if hasattr(module, "CONFIG_FILE"):
            ctx.patch(module, "CONFIG_FILE", str(sb.config_file))
        config_file = getattr(module, "CONFIG_FILE", None)
        if config_file is not None and not str(config_file).startswith(str(sb.root)):
            raise RuntimeError(f"refusing to run: {module_name}.CONFIG_FILE {config_file!r} is outside the sandbox")


def make_legacy_owner_factory(bundle):
    """``owner_factory(scenario, ctx)`` building a LegacyMethods owner in a sandbox.

    For a SHA where the shared GUI-free mixins exist (U2+) the frozen
    TranslatorGUI body is combined with the frozen copies of the mixin classes
    in the same precedence as the real class: ``(FakeState, <frozen TranslatorGUI
    body>, SettingsPersistenceMixin, RunEnvMixin, ConfigStateMixin)``.
    """
    from parity import scenarios as scenarios_mod

    mixin_bases = shared_mixin_classes(bundle)
    recorded = set(bundle.manifest["recorded_methods"])
    shadowed = sorted(name for name in recorded for base in mixin_bases if name in vars(base))
    if shadowed:
        raise RuntimeError(f"recorder methods would shadow shared mixin methods: {shadowed}")
    owner_cls = make_owner_class(
        "LegacyFake", (FakeState, bundle.methods) + mixin_bases, recorded
    )
    if mixin_bases:
        # Frozen code calls TranslatorGUI.<method>(self, ...) explicitly; moved
        # methods resolve through the mixins exactly like the real MRO.
        bundle.namespace["TranslatorGUI"] = owner_cls
    init_extraction_vars = bundle.external("other_settings", "initialize_extraction_variables")
    bound_methods = bundle.bound_methods()

    def other_settings_methods(owner):
        # other_settings.setup_other_settings_methods: initialize_extraction_variables first, then bind
        init_extraction_vars(owner)
        bind_other_settings_methods(owner, bound_methods)

    def factory(scenario: dict, ctx):
        ns = bundle.namespace
        sandbox = ctx.sandbox
        path_globals = {
            "_APP_DIR": str(sandbox.app_dir),
            "CONFIG_FILE": str(sandbox.config_file),
        }
        ctx.patch_dict(ns, "__file__", str(sandbox.src_file("translator_gui.py")))
        for name, value in path_globals.items():
            ctx.patch_dict(ns, name, value)
        for name, stub in ctx.backend_stubs.items():
            if name in ns:
                ctx.patch_dict(ns, name, stub)
        for module, ext_ns in bundle.externals.items():
            ctx.patch_dict(ext_ns, "__file__", str(sandbox.src_file(f"{module}.py")))
            for name, value in path_globals.items():
                if name in ext_ns:
                    ctx.patch_dict(ext_ns, name, value)
        for module, attr, obj in bundle.module_patches():
            ctx.patch_module_attr(module, attr, obj)
        for ext_ns in [ns, *bundle.externals.values()]:
            config_file = ext_ns.get("CONFIG_FILE")
            if config_file is not None and not str(config_file).startswith(str(sandbox.root)):
                raise RuntimeError(f"refusing to run: CONFIG_FILE {config_file!r} is outside the sandbox")
        # U1+ frozen code imports _get_app_dir/CONFIG_FILE from the live app_paths
        # (and U2+ mixins live in their own modules): sandbox their path globals too.
        patch_shared_module_paths(ctx)
        patch_frozen_mixin_paths(ctx, bundle)

        owner = owner_cls(ctx.recorder)
        steps = {
            "pre_config": owner.legacy_pre_config_block,
            "init_block": owner.legacy_init_block,
            "default_prompts": owner.legacy_default_prompts_block,
            "init_variables": owner._init_variables,
            "other_settings_methods": lambda: other_settings_methods(owner),
            "gui_state": owner.legacy_gui_state_block,
            "widgets": lambda: install_widgets(
                owner, scenarios_mod.startup_widgets(owner, scenario)
            ),
            "gui_handlers": owner.legacy_gui_handlers_block,
            "watchdog": owner.legacy_watchdog_block,
            "startup_env": owner.initialize_environment_variables,
            "post_startup": owner.legacy_post_startup_block,
        }
        for phase in LEGACY_BOOT_PHASES:
            result = steps[phase]()
            ctx.checkpoint(phase, owner, result=result)
        return owner

    factory.kind = f"legacy@{bundle.sha[:12]}"
    return factory
