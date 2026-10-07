"""Host tests for tools/schema_extract.py (the settings-schema generator).

Most tests build a small fake ``src/`` tree that uses the desktop patterns the generator
reads (settings_map tuples, bool_vars/str_vars, env builders, dialog widgets, nested
defaults) and check the generated data. Two layouts of the same code are compared: all
in ``translator_gui.py`` and moved verbatim into the GUI-free mixin modules
(``owner_state.py`` / ``run_env.py`` / ``settings_persistence.py``), as the shared-core
move does. The ``real_repo`` tests run the generator on the repository: determinism
across hash seeds and the drift check against the committed ``src/settings_schema_data.py``.

Run: python -m pytest -p no:cacheprovider -W ignore tests_host/test_schema_extract.py
"""
from __future__ import annotations

import ast
import importlib
import os
import runpy
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

MOBILE = Path(__file__).resolve().parents[1]
TOOLS = MOBILE / "tools"
SRC = MOBILE.parent
sys.path.insert(0, str(TOOLS))

se = importlib.import_module("schema_extract")


# --------------------------------------------------------------------------- fake tree
PROMPTS = '''
LONG_PROMPT = "{long}"
SHORT_LIST = ("a", "b")
'''.format(long="x" * 200)

# Shared method bodies (each starts at class-body indentation: 4 spaces).
INIT_BLOCK = '''
        self.max_output_tokens = 128000
        if 'force_ncx_only' not in self.config:
            self.config['force_ncx_only'] = True
        self.delay_hint_var = self.config.get('delay_hint', 7)
        self.max_output_tokens = self.config.get('max_output_tokens', self.max_output_tokens)
        os.environ['STARTUP_FLAG'] = '1' if self.config.get('startup_flag', False) else '0'
'''

INIT_VARIABLES = '''
    def _init_variables(self):
        self.vision_prompt = self.config.get('vision_prompt', LONG_PROMPT)

        def create_var(var_type, key, default):
            return self.config.get(key, default)

        bool_vars = [
            ('contextual_var', 'contextual', False),
            ('rolling_var', 'use_rolling', True),
        ]
        for var_name, key, default in bool_vars:
            setattr(self, var_name, create_var(bool, key, default))
        str_vars = [
            ('api_queue_var', 'api_queue', 4),
            ('mode_var', 'refine_mode', 'full'),
        ]
        for var_name, key, default in str_vars:
            setattr(self, var_name, create_var(str, key, str(default)))
        self.patterns = self.config.get('patterns', list(SHORT_LIST))
        self.top_p_var = self.config.get('top_p', 1.0)
'''

SAVE_CONFIG = '''
    def save_config(self, show_message=True):
        def safe_int(value, default):
            try: return int(value)
            except (ValueError, TypeError): return default

        def safe_float(value, default):
            try: return float(value)
            except (ValueError, TypeError): return default

        settings_map = [
            ('contextual', ['contextual_checkbox', 'contextual_var'], False, bool),
            ('delay', ['delay_entry'], 5.0, lambda v: safe_float(v, 5.0)),
            ('api_queue', ['api_queue_var'], 4, lambda v: max(-1, safe_int(v, 4))),
            ('clamped', ['clamped_var'], 3, lambda v: max(0, min(10, safe_int(v, 3)))),
            ('clamped2', ['clamped2_var'], 3, lambda v: min(10, max(0, safe_int(v, 3)))),
            ('refine_mode', ['mode_var'], 'full', lambda v: str(v).strip().lower() if str(v).strip().lower() in MULTIPASS_REFINEMENT_MODES else 'full'),
            ('budget', ['budget_var'], 0, lambda v: int(v) if str(v).lstrip('-').isdigit() else 0),
            ('provider', ['provider_var'], 'Auto', lambda v: (str(v).strip() if v is not None else '') or 'Auto'),
            ('weird', ['weird_var'], 0, lambda v: v * 2),
            ('use_rolling', ['rolling_var'], False, bool),
            ('use_rolling', ['rolling_var'], True, bool),
            ('decimal', ['decimal_var'], '0.0001', _format_plain_decimal_setting),
            ('top_p', ['top_p_var'], 1.0, float),
            ('rep_penalty', ['rep_var'], 1.0, float),
            ('translate_title', ['translate_title_var'], False, bool),
        ]
        for key, sources, default, converter in settings_map:
            pass
        self.config.setdefault('backup_enabled', True)

        def _update_env(key, new_val, is_bool=False):
            os.environ[key] = str(new_val)

        _update_env('ROLLING_FLAG', self.config.get('use_rolling'), is_bool=True)
        for env_key, env_value in self._glossary_env_mappings():
            _update_env(env_key, env_value)
'''

ENV_METHODS = '''
    def _get_environment_variables(self, epub_path, api_key):
        limit = self.config.get('glossary_max_sentences', 200)
        env_vars = {
            'CONTEXTUAL': '1' if self.contextual_var else '0',
            'API_QUEUE_SIZE': str(self.api_queue_var),
            'GLOSSARY_MAX_SENTENCES': str(limit),
            'DELAY_HINT': str(getattr(self, 'delay_hint_var', 3)),
            'MODE_FLAG': self._current_mode(),
        }
        return env_vars

    def _current_mode(self):
        return str(self.config.get('refine_mode', 'full'))

    def initialize_environment_variables(self):
        os.environ['GLOSSARY_MAX_SENTENCES'] = str(self.config.get('glossary_max_sentences', 10))

    def _glossary_env_mappings(self):
        return [
            ('GLOSSARY_FLAG', '1' if self.config.get('glossary_flag', False) else '0'),
        ]
'''

GUI_METHODS = '''
    def _setup_gui(self):
        self._create_settings_section()

    def _create_settings_section(self):
        delay_label = QLabel("API call delay (s):")
        self.delay_entry = QLineEdit()
        self.delay_entry.setText(str(self.config.get('delay', 5)))
        self.contextual_checkbox = QCheckBox("Contextual translation")
        self.contextual_checkbox.setToolTip("Send history")
        self.contextual_checkbox.setChecked(self.contextual_var)
'''

TRANSLATOR_GUI_A = (
    "# -*- coding: utf-8 -*-\n"
    "import os\n"
    "from PySide6.QtWidgets import QCheckBox, QLabel, QLineEdit, QMainWindow\n"
    "from fake_prompts import LONG_PROMPT, SHORT_LIST\n"
    "\n"
    'MULTIPASS_REFINEMENT_MODES = ("full", "partial")\n'
    "\n"
    "\n"
    "class TranslatorGUI(QMainWindow):\n"
    "    def __init__(self):\n"
    "        self.config = {}\n"
    + INIT_BLOCK.lstrip("\n")
    + "        self._init_variables()\n"
    "        self._setup_gui()\n"
    + INIT_VARIABLES + SAVE_CONFIG + ENV_METHODS + GUI_METHODS
)

# Layout B: the same bodies, moved into the GUI-free mixins.
TRANSLATOR_GUI_B = (
    "# -*- coding: utf-8 -*-\n"
    "import os\n"
    "from PySide6.QtWidgets import QCheckBox, QLabel, QLineEdit, QMainWindow\n"
    "from owner_state import ConfigStateMixin\n"
    "from run_env import RunEnvMixin, MULTIPASS_REFINEMENT_MODES\n"
    "from settings_persistence import SettingsPersistenceMixin\n"
    "\n"
    "\n"
    "class TranslatorGUI(SettingsPersistenceMixin, RunEnvMixin, ConfigStateMixin, QMainWindow):\n"
    "    def __init__(self):\n"
    "        self.config = {}\n"
    "        self._init_config_state()\n"
    "        self._init_variables()\n"
    "        self._setup_gui()\n"
    + GUI_METHODS
)
OWNER_STATE_B = (
    "import os\n"
    "from fake_prompts import LONG_PROMPT, SHORT_LIST\n"
    "\n"
    "\n"
    "class ConfigStateMixin:\n"
    "    def _init_config_state(self):\n"
    + INIT_BLOCK.lstrip("\n")
    + INIT_VARIABLES
)
RUN_ENV_B = (
    "import os\n"
    "\n"
    'MULTIPASS_REFINEMENT_MODES = ("full", "partial")\n'
    "\n"
    "\n"
    "class RunEnvMixin:\n"
    + ENV_METHODS.lstrip("\n")
)
SETTINGS_PERSISTENCE_B = (
    "import os\n"
    "from run_env import MULTIPASS_REFINEMENT_MODES\n"
    "\n"
    "\n"
    "class SettingsPersistenceMixin:\n"
    + SAVE_CONFIG.lstrip("\n")
)

OTHER_SETTINGS = '''
def _create_prompt_management_section(self, parent):
    section_box = QGroupBox("Meta Data")
    if not hasattr(self, 'translate_title_var'):
        self.translate_title_var = self.config.get('translate_title', True)
    cb = self._create_styled_checkbox("Translate book title")
    cb.setToolTip("Translates the title")
    cb.setChecked(bool(self.translate_title_var))

    def _toggle(checked):
        self.translate_title_var = bool(checked)
        self.config['translate_title'] = bool(checked)
    cb.toggled.connect(_toggle)


def _create_anti_duplicate_section(self, parent):
    notebook = QTabWidget()
    core_frame = QWidget()
    notebook.addTab(core_frame, "Core Parameters")

    def _create_slider_row(layout, label_text, holder, var_name, lo, hi):
        return None
    _create_slider_row(None, "Top-P (Nucleus Sampling):", self, 'top_p_var', 0.1, 1.0)
    advanced_frame = QWidget()
    notebook.addTab(advanced_frame, "Advanced")
    self.rep_var = self.config.get('rep_penalty', 1.0)
    rep_label = QLabel("Repetition Penalty:")
    rep_entry = QLineEdit()
    rep_entry.setText(str(self.rep_var))
'''

QA_RUNTIME = '''
def default_qa_scan_settings():
    return {
        "check_repetition": True,
        "min_file_length": 0,
        "word_count_multipliers": {"english": 1.0},
    }


def apply_qa_scan_env_from_settings(qa_settings):
    settings = qa_settings if isinstance(qa_settings, dict) else {}
    mappings = {
        "QA_CHECK_REPETITION": "1" if settings.get("check_repetition", True) else "0",
    }
    return mappings
'''

AI_HUNTER = '''
def default_ai_hunter_config():
    return {'enabled': True, 'thresholds': {'exact': 90}}
'''

MANGA_DEFAULTS = '''
def default_manga_settings():
    return {'ocr': {'provider': 'google'}}
'''

MANGA_DIALOG = '''
class MangaSettingsDialog(QDialog):
    def __init__(self, parent, main_gui, config):
        self.config = config
        self.default_settings = default_manga_settings()
        self.settings = self._merge_settings(config.get('manga_settings', {}))

    def _build(self):
        google_cb = QCheckBox("Use Google OCR")
        google_cb.setChecked(self.settings.get('ocr', {}).get('provider', 'google') == 'google')
'''


def write_tree(root: Path, layout: str = "A") -> Path:
    src = root / "src"
    src.mkdir(parents=True, exist_ok=True)
    files = {
        "fake_prompts.py": PROMPTS,
        "other_settings.py": OTHER_SETTINGS,
        "qa_scan_runtime.py": QA_RUNTIME,
        "ai_hunter_enhanced.py": AI_HUNTER,
        "manga_settings_dialog.py": MANGA_DIALOG,
        "manga_settings_defaults.py": MANGA_DEFAULTS,
    }
    if layout == "A":
        files["translator_gui.py"] = TRANSLATOR_GUI_A
    else:
        files["translator_gui.py"] = TRANSLATOR_GUI_B
        files["owner_state.py"] = OWNER_STATE_B
        files["run_env.py"] = RUN_ENV_B
        files["settings_persistence.py"] = SETTINGS_PERSISTENCE_B
    for name, text in files.items():
        encoding = "utf-8-sig" if name == "translator_gui.py" else "utf-8"   # the real file has a BOM
        (src / name).write_text(textwrap.dedent(text) if name != "translator_gui.py" else text,
                                encoding=encoding)
    for name in files:
        ast.parse((src / name).read_text(encoding="utf-8-sig"), filename=name)   # fixture sanity
    return src


def generate_data(src: Path) -> dict:
    text = se.generate(src)
    namespace = {}
    exec(compile(text, "settings_schema_data.py", "exec"), namespace)
    return namespace


@pytest.fixture(scope="module")
def fake_a(tmp_path_factory):
    src = write_tree(tmp_path_factory.mktemp("layout_a"), "A")
    return src, se.generate(src)


@pytest.fixture(scope="module")
def data_a(fake_a):
    namespace = {}
    exec(compile(fake_a[1], "settings_schema_data.py", "exec"), namespace)
    return namespace


# --------------------------------------------------------------------------- (a) settings_map
def test_translator_gui_bom_is_read(fake_a):
    src, _text = fake_a
    assert (src / "translator_gui.py").read_bytes().startswith(b"\xef\xbb\xbf")
    with pytest.raises(SyntaxError):
        ast.parse((src / "translator_gui.py").read_text(encoding="utf-8"))
    assert se.read_source(src / "translator_gui.py").startswith("# -*- coding")


def test_settings_map_order_and_duplicates(data_a):
    order = data_a["SETTINGS_MAP_ORDER"]
    assert order[:3] == ("contextual", "delay", "api_queue")
    assert order.count("use_rolling") == 2
    rolling = data_a["SETTINGS"]["use_rolling"]
    assert rolling["save_default"] is True                     # last occurrence wins
    assert "settings_map_duplicate" in rolling["flags"]
    assert "settings_map_duplicate_differs" in rolling["flags"]


@pytest.mark.parametrize("key, expected", [
    ("contextual", ("bool",)),
    ("delay", ("safe_float", 5.0, None, None, "")),
    ("api_queue", ("safe_int", 4, -1, None, "")),
    ("clamped", ("safe_int", 3, 0, 10, "max_min")),
    ("clamped2", ("safe_int", 3, 0, 10, "min_max")),
    ("refine_mode", ("choice", ("full", "partial"), "full")),   # module constant resolved
    ("budget", ("int_if_digits", "-", 0)),
    ("provider", ("str_or", "", "Auto")),
    ("decimal", ("call", "_format_plain_decimal_setting")),
    ("top_p", ("float",)),
])
def test_converter_pattern_table(data_a, key, expected):
    assert data_a["SETTINGS"][key]["converter"] == expected


def test_unknown_converter_is_flagged(data_a):
    weird = data_a["SETTINGS"]["weird"]
    assert weird["converter"][0] == "unknown"
    assert "unknown_converter" in weird["flags"]
    assert data_a["UNKNOWN_CONVERTERS"] == {"weird": "lambda v: v * 2"}


def test_sources_split_into_vars_and_widgets(data_a):
    contextual = data_a["SETTINGS"]["contextual"]
    assert contextual["widget_sources"] == ("contextual_checkbox",)
    assert contextual["var_names"][0] == "contextual_var"
    assert data_a["SETTINGS"]["delay"]["widget_sources"] == ("delay_entry",)


# --------------------------------------------------------------------------- (b) defaults
def test_effective_defaults(data_a):
    s = data_a["SETTINGS"]
    # str_vars stores str(default); the startup save_config converts it
    assert s["api_queue"]["init_default"] == "4"
    assert s["api_queue"]["default"] == 4
    # widget filled at startup from config.get('delay', 5) -> safe_float
    assert s["delay"]["default"] == 5.0 and s["delay"]["default_source"] == "init"
    assert s["contextual"]["default"] is False
    assert s["force_ncx_only"]["default"] is True                 # if 'k' not in config: ...
    assert s["max_output_tokens"]["default"] == 128000           # self attr literal
    assert s["patterns"]["default"] == ["a", "b"]                 # list(<imported tuple>)
    assert s["vision_prompt"]["default"] == {"$ref": "fake_prompts:LONG_PROMPT"}   # long text stays a ref
    assert s["backup_enabled"]["default"] is True                # save_config setdefault
    assert s["translate_title"]["dialog_default"] is True


def test_discrepancies_are_recorded_not_fixed(data_a):
    s = data_a["SETTINGS"]
    gms = s["glossary_max_sentences"]
    assert gms["default"] == 200
    text = " ".join(gms["discrepancies"])
    assert "=10" in text and "=200" in text
    hint = s["delay_hint"]
    assert hint["default"] == 7
    assert any("=3" in d for d in hint["discrepancies"])


# --------------------------------------------------------------------------- (c) env
def test_env_bindings(data_a):
    s = data_a["SETTINGS"]
    assert ("CONTEXTUAL", "translation", "<none>") in s["contextual"]["env"]
    assert ("GLOSSARY_MAX_SENTENCES", "startup", 10) in s["glossary_max_sentences"]["env"]
    assert ("GLOSSARY_MAX_SENTENCES", "translation", 200) in s["glossary_max_sentences"]["env"]
    assert ("DELAY_HINT", "translation", 3) in s["delay_hint"]["env"]
    assert ("STARTUP_FLAG", "startup", False) in s["startup_flag"]["env"]
    assert ("ROLLING_FLAG", "save", "<none>") in s["use_rolling"]["env"]
    assert ("GLOSSARY_FLAG", "save", False) in s["glossary_flag"]["env"]
    # a ref through a small helper method's return value
    assert ("MODE_FLAG", "translation", "full") in s["refine_mode"]["env"]


# --------------------------------------------------------------------------- (d) labels
def test_labels_tooltips_groups_tabs(data_a):
    s = data_a["SETTINGS"]
    assert s["delay"]["label"] == "API call delay (s):"           # nearest QLabel
    assert s["contextual"]["label"] == "Contextual translation"
    assert s["contextual"]["tooltip"] == "Send history"
    assert s["translate_title"]["label"] == "Translate book title"
    assert s["translate_title"]["tooltip"] == "Translates the title"
    assert s["translate_title"]["ui_group"] == "Meta Data"
    assert s["translate_title"]["ui_sites"][0] == "other.meta_data"
    assert s["top_p"]["label"] == "Top-P (Nucleus Sampling):"     # helper row
    assert s["top_p"]["ui_tab"] == "Core Parameters"
    assert s["rep_penalty"]["label"] == "Repetition Penalty:"
    assert s["rep_penalty"]["ui_tab"] == "Advanced"


# --------------------------------------------------------------------------- (e) nested
def test_nested_settings(data_a):
    s = data_a["SETTINGS"]
    rep = s["qa_scanner_settings.check_repetition"]
    assert rep["default"] is True and rep["parent"] == "qa_scanner_settings"
    assert ("QA_CHECK_REPETITION", "qa", True) in rep["env"]
    assert s["qa_scanner_settings.word_count_multipliers"]["default"] == {"english": 1.0}
    assert s["ai_hunter_config.thresholds.exact"]["default"] == 90
    provider = s["manga_settings.ocr.provider"]
    assert provider["default"] == "google" and provider["label"] == "Use Google OCR"
    for container in ("manga_settings", "manga_settings.ocr", "qa_scanner_settings", "ai_hunter_config.thresholds"):
        assert container not in s


# --------------------------------------------------------------------------- layouts, determinism, CLI
def test_layout_independent_output(tmp_path, fake_a):
    """A verbatim move into owner_state/run_env/settings_persistence changes nothing."""
    src_b = write_tree(tmp_path / "layout_b", "B")
    assert se.generate(src_b) == fake_a[1]


def test_generation_is_deterministic(tmp_path, fake_a):
    src, text = fake_a
    assert se.generate(src) == text
    outputs = set()
    for seed in ("0", "4242"):
        env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONIOENCODING="utf-8")
        result = subprocess.run([sys.executable, str(TOOLS / "schema_extract.py"), "--src", str(src), "--stdout"],
                                capture_output=True, text=True, encoding="utf-8", env=env, check=True)
        outputs.add(result.stdout)
    assert outputs == {text}


def test_output_is_pure_python310_data(fake_a):
    tree = ast.parse(fake_a[1], feature_version=(3, 10))
    for node in tree.body:
        assert isinstance(node, ast.Assign), ast.dump(node)[:80]
        ast.literal_eval(node.value)


def test_check_cli(tmp_path):
    src = write_tree(tmp_path, "A")
    tool = str(TOOLS / "schema_extract.py")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    assert subprocess.run([sys.executable, tool, "--src", str(src), "--check"], env=env,
                          capture_output=True).returncode == 1          # nothing written yet
    subprocess.run([sys.executable, tool, "--src", str(src)], env=env, capture_output=True, check=True)
    assert subprocess.run([sys.executable, tool, "--src", str(src), "--check"], env=env,
                          capture_output=True).returncode == 0
    gui = src / "translator_gui.py"
    gui.write_text(gui.read_text(encoding="utf-8-sig").replace("('delay', ['delay_entry'], 5.0", "('delay', ['delay_entry'], 6.0"),
                   encoding="utf-8-sig")
    assert subprocess.run([sys.executable, tool, "--src", str(src), "--check"], env=env,
                          capture_output=True).returncode == 1          # changed default -> stale


def test_missing_anchor_fails_loudly(tmp_path):
    src = write_tree(tmp_path, "A")
    gui = src / "translator_gui.py"
    gui.write_text(gui.read_text(encoding="utf-8-sig").replace("settings_map = [", "other_map = ["),
                   encoding="utf-8-sig")
    with pytest.raises(SystemExit, match="settings_map not found"):
        se.generate(src)


# --------------------------------------------------------------------------- real repository
@pytest.fixture(scope="module")
def real_generated():
    return se.generate(SRC)


def test_real_repo_data_is_fresh(real_generated):
    """Drift check: re-running the generator reproduces the committed data file."""
    committed = se.read_committed(SRC)
    assert committed is not None, "src/settings_schema_data.py is missing; run tools/schema_extract.py"
    assert committed == real_generated, (
        "src/settings_schema_data.py is stale: run python src/mobile/tools/schema_extract.py")


def test_real_repo_generation_is_deterministic(real_generated):
    env = dict(os.environ, PYTHONHASHSEED="12345", PYTHONIOENCODING="utf-8")
    result = subprocess.run([sys.executable, str(TOOLS / "schema_extract.py"), "--stdout"],
                            capture_output=True, text=True, encoding="utf-8", env=env, check=True)
    assert result.stdout == real_generated


def test_real_repo_covers_settings_map(real_generated):
    namespace = {}
    exec(compile(real_generated, "settings_schema_data.py", "exec"), namespace)
    settings, order = namespace["SETTINGS"], namespace["SETTINGS_MAP_ORDER"]
    assert len(order) > 300
    missing = [key for key in order if key not in settings]
    assert not missing
    assert namespace["UNKNOWN_CONVERTERS"] == {}, "extend the ConvSpec pattern table"
    assert settings["model"]["default"] == "authgpt/gpt-6-luna"
