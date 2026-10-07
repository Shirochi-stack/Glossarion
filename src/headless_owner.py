"""HeadlessOwner: a GUI-free stand-in for the desktop TranslatorGUI, built from a config dict.

Shared GUI-free core (Glossarion mobile rewrite, milestone U2). The mobile app (and
host tools/tests) never re-implement the desktop's settings -> environment logic:
they build a ``HeadlessOwner`` and call the same verbatim methods TranslatorGUI
inherits (``RunEnvMixin``, ``SettingsPersistenceMixin``, ``ConfigStateMixin``).

Construction replays a desktop start in ``TranslatorGUI.__init__`` order:

1. the plain attribute defaults ``__init__`` sets before loading config.json;
2. the config load: ``self.config`` = a deep copy of *config* (``config_store``
   only reads + decrypts; nothing here writes config.json), then
   ``_sanitize_config_prompts()`` inside the same try as the desktop;
3. ``_init_config_state()`` (the ``__init__`` config block), the metadata prompt
   defaults, ``_init_default_prompt_profiles()``, ``_init_variables()``;
4. ``initialize_extraction_variables`` (what ``setup_other_settings_methods`` runs
   first) and ``_init_gui_backed_state()``;
5. widget shims valued exactly as ``_setup_gui`` fills those widgets, then
   ``_replay_gui_startup_handlers()``: the AuthGem project restore, the temperature
   toggle, the glossary-mode shortcut handler with its startup ``save_config`` (in
   memory here), context mode, target language, active profile prompt and auto
   compression factor. This is why a fresh install runs with
   ``AUTO_GLOSSARY_MODE='off'`` exactly like the desktop (the ``_init_variables``
   default ``'balanced'`` is overridden);
6. ``_init_watchdog_dir()`` and ``initialize_environment_variables()``;
7. ``_auto_encrypt_api_keys()``: the desktop re-saves (here: in memory) when the
   decrypted config holds a plain ``api_key`` / ``replicate_api_key``, which
   re-exports the saved settings after the startup env.

Construction writes ~40 process-wide environment variables, so mobile builds an
owner only on its job thread (under ``job_runner.JOB_LOCK``), from a config snapshot
taken at job start. U3 adds the job code as extra bases placed before the shared
mixins, in TranslatorGUI's order: ``TranslationPipelineMixin`` (``_prepare_translation_run``
+ ``_translation_worker``: the desktop Run Translation; ``run_translation_direct``,
QA/multipass planning, ``run_glossary_extraction_direct``, glossary auto-loading and
auto-mapping), ``TextJobsMixin`` (``_process_text_file``,
``_extract_glossary_from_text_file``, ``_run_epub_compile`` / ``_run_pdf_compile``,
``_run_parallel_metadata_files``) and ``InputPreparationMixin`` (ZIP / HTML / subtitle
inputs); their hooks (``_backend_entry``, ``_ui_request``, ``_ui_message``,
``_notify_compile_result``, ...) use the GUI-free defaults of
``translation_pipeline.PipelineHooksMixin`` / ``job_runner.JobHooksMixin``, which report
to ``host.emit`` and ask ``host.ask`` (glossary approval). U7: ``TranslationPipelineMixin``
also brings the image / video and generative-only runners (``image_job.ImageJobMixin``) and
the RPG Maker runner (``rpgmaker_job.RpgMakerJobMixin``), exactly as TranslatorGUI gets them.

Rules: Python 3.10 compatible; never import PySide6, translator_gui or dpi_setup.
"""

import ast
import copy
import os
from dataclasses import dataclass, field
from pathlib import Path

from input_preparation import InputPreparationMixin
from owner_state import ConfigStateMixin, initialize_extraction_variables
from run_env import RunEnvMixin
from settings_persistence import SettingsPersistenceMixin
from text_jobs import TextJobsMixin
from translation_pipeline import TranslationPipelineMixin

# ---------------------------------------------------------------------------
# Widget shims: only the Qt surface the shared code reads. ``hasattr`` answers
# like the real widget (QLineEdit has no isChecked, QCheckBox.text() is the label,
# QTextEdit has toPlainText but no text(), QComboBox has neither), because
# save_config's ``_get_value`` and ``_live_bool_setting`` dispatch on them.
# ---------------------------------------------------------------------------


class _ShimBase:
    """No-op QWidget surface (enabled/visible/signals/tooltips/styles)."""

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

    def __repr__(self):
        return f"<{type(self).__name__} {self._shim_value()!r}>"

    def _shim_value(self):
        return None


def _qt_text(text):
    """Text a Qt text widget holds after ``setText(text)``: PySide6 turns ``None`` into ''."""
    return '' if text is None else str(text)


class TextShim(_ShimBase):
    """QLineEdit: ``text()`` / ``setText()`` (``None`` reads back as '', like Qt)."""

    def __init__(self, text=""):
        super().__init__()
        self._text = _qt_text(text)

    def text(self):
        return self._text

    def setText(self, text):
        self._text = _qt_text(text)

    def clear(self):
        self._text = ""

    def setPlaceholderText(self, _text):
        return None

    def _shim_value(self):
        return self._text


class CheckShim(_ShimBase):
    """QCheckBox: ``isChecked()`` / ``setChecked()``; ``text()`` is the label, like Qt."""

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

    def _shim_value(self):
        return self._checked


class PlainTextShim(_ShimBase):
    """QTextEdit: ``toPlainText()`` / ``setPlainText()`` / ``setText()`` (``None`` -> '', like Qt)."""

    def __init__(self, plain=""):
        super().__init__()
        self._plain = _qt_text(plain)

    def toPlainText(self):
        return self._plain

    def setPlainText(self, text):
        self._plain = _qt_text(text)

    def setText(self, text):
        self._plain = _qt_text(text)

    def _shim_value(self):
        return self._plain


class ComboShim(_ShimBase):
    """QComboBox with ``(text, data)`` items."""

    def __init__(self, items=(), index=0, text=None):
        super().__init__()
        self._items = [(str(t), d) for t, d in (items or [])]
        self._index = index if self._items else -1
        self._edit_text = text

    @classmethod
    def with_data(cls, items, data, fallback_index=0):
        """``setCurrentIndex(idx if idx >= 0 else fallback)`` of ``findData(data)``, like desktop."""
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

    def _shim_value(self):
        return (self.currentText(), self.currentData())


SHIM_TYPES = (TextShim, CheckShim, PlainTextShim, ComboShim)

# Combo items as the desktop section builders add them (translator_gui.py).
#: _create_settings_section: Context Mode combo (addItem(text, data)).
CONTEXT_MODE_ITEMS = (
    ("Off", "off"),
    ("Contextual History", "contextual_history"),
    ("Rolling Summary (Replace)", "rolling_summary_replace"),
    ("Rolling Summary (Append)", "rolling_summary_append"),
)
#: _create_settings_section: multipass refinement mode combo.
MULTIPASS_ITEMS = (
    ("Full", "full"),
    ("Full + raw", "full_with_raw"),
    ("Failed", "failed"),
    ("Partial", "partial"),
    ("Partial.b", "partial.b"),
    ("Partial.b2", "partial.b2"),
)
_MULTIPASS_INDEX = {data: i for i, (_text, data) in enumerate(MULTIPASS_ITEMS)}
#: _create_api_section: Remove AI Artifacts combo (restored with findData, index 0 otherwise).
REMOVE_ARTIFACTS_ITEMS = (("Off", "off"), ("Low", "low"), ("Medium", "medium"), ("High", "high"))
#: _create_settings_section: main-window glossary-mode shortcut combo (addItems; no data).
AUTO_GLOSSARY_SHORTCUT_ITEMS = (
    ("Off", None), ("Off (Fuzzy Mapping)", None), ("Manual Glossary Only", None), ("No Glossary", None),
    ("Minimal", None), ("Balanced", None), ("Full", None), ("Single Pass", None),
)

#: How the desktop fills each shimmed widget at startup (expression in the section
#: builder -> what HeadlessOwner uses). tests/test_headless_owner.py checks the
#: desktop expressions still read like this.
STARTUP_WIDGET_SOURCES = {
    "thread_delay_entry": "self.thread_delay_entry.setText(str(self.thread_delay_var))",
    "delay_entry": "self.delay_entry.setText(str(self.config.get('delay', 5)))",
    "api_queue_entry": "self.api_queue_entry.setText(str(self.api_queue_var))",
    "chapter_range_entry": "self.chapter_range_entry.setText(self.config.get('chapter_range', ''))",
    "use_spine_order_checkbox": "self.use_spine_order_checkbox.setChecked(self.config.get('use_spine_order', False))",
    "token_limit_entry": "self.token_limit_entry.setText(f\"{self.config.get('token_limit') or 200000:,}\")",
    "trans_history": "self.trans_history.setText(str(self.config.get('translation_history_limit', 2)))",
    "trans_temp": "self.trans_temp.setText(str(self.config.get('translation_temperature', 0.3)))",
    "api_key_entry": "self.api_key_entry.setText(initial_key)",
}


# ---------------------------------------------------------------------------
# Direct Text run options (the _input_output_run_active / _direct_text_* contract)
# ---------------------------------------------------------------------------


@dataclass
class DirectTextRunOptions:
    """Per-run Direct Text overrides set on the owner before a run.

    The desktop ``_InputOutputDialog`` builds one of these and calls ``apply_to(gui)``;
    mobile does the same on its HeadlessOwner. The attributes (read by
    ``_apply_direct_text_runtime_environment``, ``_current_auto_glossary_mode``,
    ``_export_multipass_runtime_env`` and the output-mode helpers) are set in the
    original dialog order.
    """

    selected_files: list = field(default_factory=list)
    force_stream_all: bool = True
    archive_conversion_dir: str = ''
    force_multipass_off: bool = True
    force_no_glossary: bool = True
    manual_glossary_path: str = ''
    skip_thinking: bool = False
    attachment_prompt: str = ''
    attachment_prompt_role: str = 'user'
    skip_prompt_profile: bool = False
    output_mode: str = 'text'

    #: Owner attributes apply_to() sets (the dialog saves/restores these around a run).
    OWNER_ATTRS = (
        'selected_files',
        '_force_stream_all',
        '_input_output_run_active',
        '_direct_text_archive_conversion_dir',
        '_direct_text_force_multipass_off',
        '_direct_text_force_no_glossary',
        '_direct_text_use_manual_glossary',
        '_direct_text_manual_glossary_path',
        '_direct_text_skip_thinking',
        '_direct_text_attachment_prompt',
        '_direct_text_attachment_prompt_role',
        '_direct_text_skip_prompt_profile',
        '_direct_text_output_mode',
        '_translation_run_output_mode_override',
        'output_mode_var',
        'enable_image_translation_var',
        'enable_image_output_mode_var',
        'enable_video_output_mode_var',
        'enable_audio_output_mode_var',
        'enable_refinement_output_mode_var',
    )
    #: Set only when a manual glossary is supplied.
    MANUAL_GLOSSARY_ATTRS = (
        'manual_glossary_path',
        'manual_glossary_manually_loaded',
        'auto_loaded_glossary_path',
        'auto_loaded_glossary_for_file',
        'manual_glossary_map',
    )

    def apply_to(self, owner):
        """Set the run attributes on *owner* (moved verbatim from _InputOutputDialog)."""
        owner.selected_files = self.selected_files
        owner._force_stream_all = self.force_stream_all
        owner._input_output_run_active = True
        owner._direct_text_archive_conversion_dir = self.archive_conversion_dir
        owner._direct_text_force_multipass_off = self.force_multipass_off
        owner._direct_text_force_no_glossary = self.force_no_glossary
        owner._direct_text_use_manual_glossary = bool(self.manual_glossary_path)
        owner._direct_text_manual_glossary_path = self.manual_glossary_path
        if self.manual_glossary_path:
            owner.manual_glossary_path = self.manual_glossary_path
            owner.manual_glossary_manually_loaded = True
            owner.auto_loaded_glossary_path = None
            owner.auto_loaded_glossary_for_file = None
            owner.manual_glossary_map = {}
        owner._direct_text_skip_thinking = self.skip_thinking
        owner._direct_text_attachment_prompt = self.attachment_prompt
        owner._direct_text_attachment_prompt_role = self.attachment_prompt_role
        owner._direct_text_skip_prompt_profile = self.skip_prompt_profile
        owner._direct_text_output_mode = self.output_mode
        owner._translation_run_output_mode_override = self.output_mode
        owner.output_mode_var = self.output_mode
        owner.enable_image_translation_var = self.output_mode in {
            'vision', 'image', 'video'
        }
        owner.enable_image_output_mode_var = self.output_mode == 'image'
        owner.enable_video_output_mode_var = self.output_mode == 'video'
        owner.enable_audio_output_mode_var = self.output_mode == 'audio'
        owner.enable_refinement_output_mode_var = (
            self.output_mode == 'refinement'
        )
        return owner

    @classmethod
    def from_owner(cls, owner):
        """Options currently applied to *owner* (inverse of apply_to for a run in progress)."""
        return cls(
            selected_files=list(getattr(owner, 'selected_files', []) or []),
            force_stream_all=bool(getattr(owner, '_force_stream_all', True)),
            archive_conversion_dir=getattr(owner, '_direct_text_archive_conversion_dir', ''),
            force_multipass_off=getattr(owner, '_direct_text_force_multipass_off', True),
            force_no_glossary=getattr(owner, '_direct_text_force_no_glossary', True),
            manual_glossary_path=getattr(owner, '_direct_text_manual_glossary_path', ''),
            skip_thinking=getattr(owner, '_direct_text_skip_thinking', False),
            attachment_prompt=getattr(owner, '_direct_text_attachment_prompt', ''),
            attachment_prompt_role=getattr(owner, '_direct_text_attachment_prompt_role', 'user'),
            skip_prompt_profile=getattr(owner, '_direct_text_skip_prompt_profile', False),
            output_mode=getattr(owner, '_direct_text_output_mode', 'text'),
        )


# ---------------------------------------------------------------------------
# HeadlessOwner
# ---------------------------------------------------------------------------


class HeadlessOwner(TranslationPipelineMixin, TextJobsMixin, InputPreparationMixin,
                    SettingsPersistenceMixin, RunEnvMixin, ConfigStateMixin):
    """GUI-free owner with the desktop's state machine (see module docstring).

    ``host`` (optional, a ``job_runner.JobHost``) receives log lines through
    ``host.log(message)`` and job events through ``host.emit(kind, **data)``.
    ``api_key`` fills the API key field (default: the config's ``api_key``);
    ``model`` behaves like a config.json whose ``model`` is that value.
    """

    def __init__(self, config, *, host=None, api_key='', model=None):
        self.host = host
        # ---- TranslatorGUI.__init__ plain attribute defaults (before the config load) ----
        self.max_output_tokens = 128000
        self.proc = self.glossary_proc = None
        self._modules_loaded = self._modules_loading = False
        self.stop_requested = False
        self.translation_thread = self.glossary_thread = self.qa_thread = self.epub_thread = self.pdf_thread = None
        self.translation_future = self.glossary_future = self.qa_future = self.epub_future = self.pdf_future = None
        self.executor = None
        self._executor_workers = None
        self.manual_glossary_path = None
        self.auto_loaded_glossary_path = None
        self.auto_loaded_glossary_for_file = None
        self.manual_glossary_manually_loaded = False
        self.manual_glossary_map = {}
        import app_paths
        self.config_file_path = app_paths.CONFIG_FILE
        self.payloads_dir = os.path.join(app_paths._get_app_dir(), "Payloads")

        # ---- config load (desktop: load_config(CONFIG_FILE) + sanitizer in one try) ----
        try:
            self.config = copy.deepcopy(config) if isinstance(config, dict) else {}
            if model is not None:
                self.config['model'] = model
            self._sanitize_config_prompts()
        except Exception:
            self.config = {}

        # ---- the rest of TranslatorGUI.__init__ in order ----
        self._init_config_state()
        self._hook_metadata_defaults()
        self._init_default_prompt_profiles()
        self._init_variables()
        initialize_extraction_variables(self)  # setup_other_settings_methods runs it first
        self._init_gui_backed_state()
        self._install_widget_shims(api_key)
        self._replay_gui_startup_handlers()
        self._init_watchdog_dir()
        self.initialize_environment_variables()
        self._auto_encrypt_api_keys()  # desktop's post-startup save when config holds plain keys

    @classmethod
    def from_config_store(cls, host=None, *, path=None, api_key='', model=None):
        """Owner for the config.json at *path* (default: app_paths.CONFIG_FILE), read + decrypted."""
        from config_store import load_config

        try:
            config = load_config(path)
        except Exception:
            config = {}
        return cls(config, host=host, api_key=api_key, model=model)

    # ---- desktop GUI surface the shared code calls -------------------------------------
    def append_log(self, message):
        """Desktop log panel -> ``host.log`` (stdout without a host)."""
        log = getattr(self.host, 'log', None)
        if callable(log):
            log(message)
        else:
            print(message)

    def _reset_api_watchdog_progress(self, *, clear_stale_external_files=True):
        """Desktop watchdog reset without its progress bar: counters + watchdog files."""
        from stop_control import reset_api_watchdog

        reset_api_watchdog(clear_stale_external_files=clear_stale_external_files)

    def _record_library_raw_inputs(self, files):
        """Run set-up (``_prepare_translation_run``) -> the Library raw-inputs registry.

        Desktop parity (TranslatorGUI._record_library_raw_inputs): every selected raw input
        that exists is recorded through the shared GUI-free ``library_core`` (the module
        ``epub_library`` re-exports it from), so the Library finds a translated book's raw
        source later. Builds without library_core skip it, like desktop builds without
        epub_library.
        """
        try:
            from library_core import record_library_raw_inputs
        except Exception:
            return None
        record_library_raw_inputs(files)
        return None

    def save_config(self, show_message=True):
        """The in-memory half of TranslatorGUI.save_config (no backup, dialogs or file write).

        Same steps and order as the desktop: refresh the context-mode flags, run the
        settings_map into self.config and export the saved settings to os.environ.
        Persisting is the caller's business (mobile writes config keys sparsely).
        """
        if getattr(self, '_config_restore_pending', False):
            return False
        try:
            debug_enabled = getattr(self, 'config', {}).get('show_debug_buttons', False)
            if hasattr(self, '_on_context_mode_changed'):
                self._on_context_mode_changed()
            self._apply_live_settings_to_config()
            self._export_settings_env(show_message=show_message, debug_enabled=debug_enabled)
            return True
        except Exception as e:
            print(f"Warning: Config save failed (silent): {e}")
            return False

    def _install_widget_shims(self, api_key=''):
        """Widgets the shared code reads, valued exactly as desktop _setup_gui fills them."""
        cfg = self.config
        # _create_settings_section (translator_gui.py)
        self.thread_delay_entry = TextShim(str(self.thread_delay_var))
        self.chunk_size_entry = TextShim("")
        self.delay_entry = TextShim(str(cfg.get('delay', 5)))
        self.api_queue_entry = TextShim(str(self.api_queue_var))
        self.chapter_range_entry = TextShim(cfg.get('chapter_range', ''))
        self.use_spine_order_checkbox = CheckShim(bool(cfg.get('use_spine_order', False)), "Spine Order")
        self.token_limit_entry = TextShim(f"{cfg.get('token_limit') or 200000:,}")
        self.context_mode_combo = ComboShim.with_data(CONTEXT_MODE_ITEMS, self.context_mode_var)
        self.trans_history = TextShim(str(cfg.get('translation_history_limit', 2)))
        self.rolling_summary_exchanges_edit = TextShim(str(self.rolling_summary_exchanges_var))
        self.rolling_summary_retain_edit = TextShim(str(self.rolling_summary_max_entries_var))
        self.trans_temp = TextShim(str(cfg.get('translation_temperature', 0.3)))
        self.disable_temperature_checkbox = CheckShim(bool(self.disable_temperature_var), "Disable temperature")
        self.batch_checkbox = CheckShim(bool(self.batch_translation_var), "Batch Translation")
        self.batch_size_entry = TextShim(str(self.batch_size_var))
        self.multipass_checkbox = CheckShim(bool(self.multipass_mode_var), "Multipass mode")
        self.multipass_refinement_mode_combo = ComboShim(
            MULTIPASS_ITEMS, index=_MULTIPASS_INDEX.get(self.multipass_refinement_mode_var, 0)
        )
        self.auto_glossary_shortcut_combo = ComboShim(
            AUTO_GLOSSARY_SHORTCUT_ITEMS, index=self._saved_auto_glossary_shortcut_index()
        )
        # create_file_section
        self.selected_files = []
        self.current_file_index = 0
        self.entry_epub = TextShim("No file selected")
        self.vertex_location_entry = TextShim(self.vertex_location_var)
        self.deep_scan_check = CheckShim(bool(self.deep_scan_var), "include subfolders")
        # _create_api_section: the key field shows config['api_key'] (decrypted)
        self.api_key_entry = TextShim(api_key or cfg.get('api_key', '') or '')
        # _create_api_section: Remove AI Artifacts level
        self.remove_artifacts_combo = ComboShim.with_data(
            REMOVE_ARTIFACTS_ITEMS,
            self.REMOVE_AI_ARTIFACTS_var if isinstance(self.REMOVE_AI_ARTIFACTS_var, str) else 'off',
        )
        # _create_prompt_section: the editor starts empty; _init_active_profile_prompt fills it
        self.prompt_text = PlainTextShim("")

    def apply_direct_text_options(self, options):
        """``options.apply_to(self)`` (DirectTextRunOptions)."""
        return options.apply_to(self)


# ---------------------------------------------------------------------------
# Owner contract: attributes the shared mixins read without a hasattr/getattr guard
# and never set themselves, i.e. what any owner (TranslatorGUI, HeadlessOwner) must
# provide. Derived from the mixin sources by AST on first access.
# ---------------------------------------------------------------------------

OWNER_CONTRACT_MODULES = (
    ("owner_state", "ConfigStateMixin"),
    ("run_env", "RunEnvMixin"),
    ("settings_persistence", "SettingsPersistenceMixin"),
    # U3: the pipelines, job runners and input preparation (+ their GUI-free hook defaults)
    ("translation_pipeline", "TranslationPipelineMixin"),
    ("translation_pipeline", "GlossaryPipelineMixin"),
    ("translation_pipeline", "PipelineHooksMixin"),
    ("text_jobs", "TextJobsMixin"),
    ("input_preparation", "InputPreparationMixin"),
    ("job_runner", "JobHooksMixin"),
    # U7: the image / generative-only and RPG Maker runners (TranslationPipelineMixin's bases)
    ("image_job", "ImageJobMixin"),
    ("rpgmaker_job", "RpgMakerJobMixin"),
)

#: U8: the duck-typed ``main_gui`` of the manga pipeline. ``MangaTranslator`` reads ``main_gui.X``
#: (its constructor argument) and ``self.main_gui.X``; the manga tab code moved out of
#: manga_integration (``MangaTranslationTab``'s mixins) reads ``self.main_gui.X``. Desktop passes
#: TranslatorGUI, mobile a HeadlessOwner: (module, class, owner expressions).
MANGA_OWNER_CONTRACT_MODULES = (
    ("manga_translator", "MangaTranslator", ("main_gui", "self.main_gui")),
    ("manga_env", "MangaEnvMixin", ("self.main_gui",)),
    ("manga_env", "MangaOcrSessionMixin", ("self.main_gui",)),
    ("manga_files_core", "MangaFilesMixin", ("self.main_gui",)),
    ("manga_runner", "MangaRunMixin", ("self.main_gui",)),
)

_SELF_OWNER = ('self',)


def _guarded_names(test, owners=_SELF_OWNER):
    """Names X guarded by ``hasattr(<owner>, 'X')`` in an if/elif/ternary/and test."""
    out = set()
    for node in ast.walk(test):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == 'hasattr'
            and len(node.args) >= 2
            and isinstance(node.args[0], (ast.Name, ast.Attribute))
            and ast.unparse(node.args[0]) in owners
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            out.add(node.args[1].value)
    return out


#: ``except`` types whose ``try`` body counts as guarded: a missing attribute there cannot
#: fail the run (same rule as tests/parity/owner_contract.py).
_GUARD_EXCEPTIONS = frozenset({'AttributeError', 'Exception', 'BaseException'})


class _AllNames(frozenset):
    """Guard set of a guarded ``try`` body: every name counts as guarded."""

    def __contains__(self, item):
        return True

    def __or__(self, other):
        return self

    __ror__ = __or__


_ALL_GUARDED = _AllNames()


def _try_guards(node):
    """True for a ``try`` with a bare ``except`` or one catching AttributeError/Exception/BaseException."""
    for handler in node.handlers:
        if handler.type is None:
            return True
        types_ = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
        if any(ast.unparse(t).split('.')[-1] in _GUARD_EXCEPTIONS for t in types_):
            return True
    return False


def _unguarded_reads(class_node, owners=_SELF_OWNER):
    """Attributes of the owner (``self``, or e.g. ``self.main_gui``) read unguarded / stored."""
    reads, stores = set(), set()

    def visit(node, guarded):
        if isinstance(node, ast.ClassDef) and node is not class_node:
            return  # a class nested in a method: its ``self`` is not the owner
        if isinstance(node, (ast.If, ast.IfExp, ast.While)):
            inner = guarded | _guarded_names(node.test, owners)
            visit(node.test, guarded | _guarded_names(node.test, owners))
            for child in (node.body if isinstance(node.body, list) else [node.body]):
                visit(child, inner)
            for child in (node.orelse if isinstance(node.orelse, list) else [node.orelse]):
                visit(child, guarded)
            return
        if isinstance(node, ast.Try) and _try_guards(node):
            for child in node.body:
                visit(child, _ALL_GUARDED)
            for part in (node.handlers, node.orelse, node.finalbody):
                for child in part:
                    visit(child, guarded)
            return
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
            seen = guarded
            for value in node.values:
                visit(value, seen)
                seen = seen | _guarded_names(value, owners)
            return
        if (isinstance(node, ast.Attribute) and isinstance(node.value, (ast.Name, ast.Attribute))
                and ast.unparse(node.value) in owners):
            if isinstance(node.ctx, ast.Store):
                stores.add(node.attr)
            elif isinstance(node.ctx, ast.Load) and node.attr not in guarded:
                reads.add(node.attr)
        for child in ast.iter_child_nodes(node):
            visit(child, guarded)

    visit(class_node, set())
    return reads, stores


def compute_owner_contract(src_dir=None, modules=None):
    """Attribute names the shared code reads unguarded from its owner and never sets (sorted tuple).

    *modules* defaults to :data:`OWNER_CONTRACT_MODULES` (``(module, class)``: the owner is
    ``self``, the class's own methods/attributes are provided). Entries may name the owner
    expressions as a third item, e.g. :data:`MANGA_OWNER_CONTRACT_MODULES`
    (``self.main_gui``): then nothing the class defines counts as provided.
    """
    src_dir = Path(src_dir) if src_dir else Path(__file__).resolve().parent
    reads, stores, provided = set(), set(), set()
    for entry in (OWNER_CONTRACT_MODULES if modules is None else modules):
        module, class_name = entry[0], entry[1]
        owners = tuple(entry[2]) if len(entry) > 2 else _SELF_OWNER
        tree = ast.parse((src_dir / f"{module}.py").read_text(encoding="utf-8-sig"))
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                if owners == _SELF_OWNER:
                    for stmt in node.body:
                        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            provided.add(stmt.name)
                        elif isinstance(stmt, ast.Assign):
                            provided.update(t.id for t in stmt.targets if isinstance(t, ast.Name))
                r, s = _unguarded_reads(node, owners)
                reads |= r
                stores |= s
    return tuple(sorted(name for name in reads - stores - provided if not name.startswith('__')))


def compute_manga_owner_contract(src_dir=None):
    """What the manga pipeline reads unguarded from its ``main_gui`` (MANGA_OWNER_CONTRACT)."""
    return compute_owner_contract(src_dir, MANGA_OWNER_CONTRACT_MODULES)


_OWNER_CONTRACT = None
_MANGA_OWNER_CONTRACT = None


def __getattr__(name):
    """``OWNER_CONTRACT`` / ``MANGA_OWNER_CONTRACT`` are computed from the sources on first access."""
    global _OWNER_CONTRACT, _MANGA_OWNER_CONTRACT
    if name == "OWNER_CONTRACT":
        if _OWNER_CONTRACT is None:
            try:
                _OWNER_CONTRACT = compute_owner_contract()
            except OSError:  # sources not shipped (bytecode-only bundle)
                _OWNER_CONTRACT = ()
        return _OWNER_CONTRACT
    if name == "MANGA_OWNER_CONTRACT":
        if _MANGA_OWNER_CONTRACT is None:
            try:
                _MANGA_OWNER_CONTRACT = compute_manga_owner_contract()
            except OSError:  # sources not shipped (bytecode-only bundle)
                _MANGA_OWNER_CONTRACT = ()
        return _MANGA_OWNER_CONTRACT
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AUTO_GLOSSARY_SHORTCUT_ITEMS",
    "CONTEXT_MODE_ITEMS",
    "CheckShim",
    "ComboShim",
    "DirectTextRunOptions",
    "HeadlessOwner",
    "MANGA_OWNER_CONTRACT",
    "MANGA_OWNER_CONTRACT_MODULES",
    "MULTIPASS_ITEMS",
    "OWNER_CONTRACT",
    "PlainTextShim",
    "REMOVE_ARTIFACTS_ITEMS",
    "SHIM_TYPES",
    "STARTUP_WIDGET_SOURCES",
    "TextShim",
    "compute_manga_owner_contract",
    "compute_owner_contract",
]
