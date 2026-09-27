import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(scope='module')
def gui_methods():
    tree = ast.parse((Path(__file__).parents[1] / 'src/translator_gui.py').read_text(encoding='utf-8-sig'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TranslatorGUI')
    names = {'_refresh_auth_account_arrows', '_get_authgpt_account_id', '_authgpt_pool_route_requested', '_authgem_vertex_control_model', '_get_authgem_account_id', '_update_authgem_login_status', '_on_auth_acct_combo_changed'}
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = {'_AUTHGROK_ADD_ACCOUNT_SENTINEL': 'new'}
    exec(compile(ast.Module(body=methods, type_ignores=[]), '<gui methods>', 'exec'), ns)
    return ns


@pytest.mark.parametrize('model,visible,slot', [
    ('authgpt/m', False, 0), ('authgpt0/m', True, 3),
    ('authgpt0', True, 3), ('authgpt1/m', False, 1),
    ('authgpt3/m', False, 3), ('authgrok0/m', False, 0),
])
def test_authgpt_only_pool_shows_selector(gui_methods, monkeypatch, tmp_path, model, visible, slot):
    import authgpt_auth
    monkeypatch.setattr(authgpt_auth, '_DEFAULT_TOKEN_DIR', str(tmp_path))
    class Combo:
        def __init__(self): self.items = []; self.visible = False
        def blockSignals(self, value): pass
        def clear(self): self.items.clear()
        def addItem(self, text, data): self.items.append((text, data))
        def setCurrentIndex(self, index): self.index = index
        def show(self): self.visible = True
        def hide(self): self.visible = False
    combo = Combo()
    gui = SimpleNamespace(
        model_var=model, config={}, authgpt_acct_combo=combo,
        authgpt_login_btn=SimpleNamespace(isVisible=lambda: False),
        _auth_account_ids={'authgpt': [0, 3]}, _auth_account_idx={'authgpt': 1},
        _collect_auth_account_ids_from_pools=lambda: {'authgpt': {0, 3}, 'authgrok': set(), 'authcd': set(), 'authgem': set()},
        _authgrok_pool_route_requested=lambda model: False,
        _authgem_vertex_control_model=lambda: '',
    )
    gui._authgpt_pool_route_requested = lambda model: gui_methods['_authgpt_pool_route_requested'](gui, model)
    assert gui_methods['_get_authgpt_account_id'](gui) == slot
    gui_methods['_refresh_auth_account_arrows'](gui)
    assert combo.visible == visible
    if visible:
        assert combo.items == [('#0', 0), ('#3', 3), ('+ N', '__authgpt_add_account__')]


def test_manager_hint_and_saved_pool_reveal_authgpt_controls(gui_methods):
    gui = SimpleNamespace(model_var='other/model', config={},
                          _multi_key_manager_authgpt_pool_hint=True,
                          _iter_enabled_key_pool_models=lambda: iter([]))
    requested = gui_methods['_authgpt_pool_route_requested']
    assert requested(gui)
    gui._multi_key_manager_authgpt_pool_hint = False
    assert not requested(gui)
    gui._iter_enabled_key_pool_models = lambda: iter([('glossary_keys', 'use_glossary_keys', 'authgpt0/model')])
    assert requested(gui)


@pytest.mark.parametrize('route,visible,slot', [
    ('authgem-vertex', False, 0), ('authgem-vertex/model', False, 0),
    ('authgem-vertex0/', True, 3), ('authgem-vertex0', True, 3),
    ('authgem-vertex1/model', False, 1), ('authgem-vertex3/model', False, 3),
])
@pytest.mark.parametrize('in_manager', [False, True])
def test_vertex_only_zero_shows_account_selector(gui_methods, monkeypatch, tmp_path, route, visible, slot, in_manager):
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    class Combo:
        def blockSignals(self, value): pass
        def clear(self): pass
        def addItem(self, text, data): pass
        def setCurrentIndex(self, index): pass
        def show(self): self.visible = True
        def hide(self): self.visible = False
    combo = Combo()
    gui = SimpleNamespace(
        model_var='other/model' if in_manager else route, config={},
        _multi_key_manager_authgem_vertex_model_hint=route if in_manager else '',
        _iter_enabled_key_pool_models=lambda: iter([]),
        authgem_acct_combo=combo, authgem_login_btn=SimpleNamespace(isVisible=lambda: False),
        _auth_account_ids={'authgem': [0, 3]}, _auth_account_idx={'authgem': 1},
        _collect_auth_account_ids_from_pools=lambda: {'authgem': {0, 3}, 'authgrok': set(), 'authcd': set(), 'authgpt': set()},
        _authgrok_pool_route_requested=lambda model: False,
        _authgpt_pool_route_requested=lambda model: False,
    )
    gui._authgem_vertex_control_model = lambda: gui_methods['_authgem_vertex_control_model'](gui)
    assert gui_methods['_get_authgem_account_id'](gui) == slot
    gui_methods['_refresh_auth_account_arrows'](gui)
    assert combo.visible == visible


@pytest.mark.parametrize('in_manager', [False, True])
@pytest.mark.parametrize('state,prefix,ending', [(None, '⏳', ''), (True, '✅', ''), (False, '🔐', ' Login')])
def test_gemini_numbered_label_during_and_after_account_load(gui_methods, in_manager, state, prefix, ending):
    class Button:
        def setText(self, text): self.text = text
        def setToolTip(self, text): pass
        def setStyleSheet(self, style): pass
    button = Button()
    gui = SimpleNamespace(
        model_var='other/model' if in_manager else 'authgem-vertex1/model', config={},
        _multi_key_manager_authgem_vertex_model_hint='authgem-vertex1/' if in_manager else '',
        _iter_enabled_key_pool_models=lambda: iter([]),
        authgem_login_btn=button,
        _auth_status_snapshot=lambda provider: None if state is None else SimpleNamespace(has_tokens=state, account_info={}),
    )
    gui._authgem_vertex_control_model = lambda: gui_methods['_authgem_vertex_control_model'](gui)
    gui._get_authgem_account_id = lambda: gui_methods['_get_authgem_account_id'](gui)
    gui_methods['_update_authgem_login_status'](gui)
    assert button.text == prefix + ' Gemini #1' + ending


def test_vertex_add_account_keeps_slot_and_opens_login(gui_methods, monkeypatch, tmp_path):
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    token_dir = tmp_path / '.glossarion'
    token_dir.mkdir()
    (token_dir / 'authgem_tokens_4.json').write_text('{}')
    class Combo:
        def __init__(self): self.items = []
        def blockSignals(self, value): pass
        def clear(self): self.items.clear()
        def addItem(self, text, data): self.items.append((text, data))
        def setCurrentIndex(self, index): self.index = index
        def itemData(self, index): return self.items[index][1]
        def show(self): pass
        def hide(self): pass
    combo = Combo()
    logged_in = []
    gui = SimpleNamespace(
        model_var='authgem-vertex0/model', config={}, authgem_acct_combo=combo,
        authgem_login_btn=SimpleNamespace(isVisible=lambda: False),
        _authgem_vertex_control_model=lambda: 'authgem-vertex0/model',
        _authgrok_pool_route_requested=lambda model: False,
        _authgpt_pool_route_requested=lambda model: False,
        _collect_auth_account_ids_from_pools=lambda: {p: set() for p in ('authgpt', 'authgrok', 'authgem', 'authcd')},
    )
    gui._refresh_auth_account_arrows = lambda: gui_methods['_refresh_auth_account_arrows'](gui)
    gui._authgem_login_clicked = lambda: logged_in.append(gui_methods['_get_authgem_account_id'](gui))
    monkeypatch.setitem(gui_methods, 'QTimer', SimpleNamespace(singleShot=lambda delay, callback: callback()))
    gui._refresh_auth_account_arrows()
    assert combo.items == [('#0', 0), ('#4', 4), ('+ N', '__authgem_add_account__')]
    for expected in (5, 6):
        gui_methods['_on_auth_acct_combo_changed'](gui, 'authgem', len(combo.items) - 1)
        assert combo.itemData(combo.index) == expected
        assert logged_in[-1] == expected
        gui._refresh_auth_account_arrows()
        assert combo.itemData(combo.index) == expected
        assert gui.model_var == 'authgem-vertex0/model'
    assert gui._authgem_pending_account_ids == {5, 6}


# Shared local Ollama settings and model-field controls.
@pytest.fixture(scope="module")
def ollama_qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _ollama_test_translator():
    from PySide6.QtWidgets import QWidget

    class Translator(QWidget):
        def __init__(self):
            super().__init__()
            self.config = {}
            self.saved = 0

        def save_config(self, show_message=True):
            self.saved += 1
            return True

    return Translator()


def test_ollamapull_model_name_and_defaults():
    from ollama_settings_dialog import (
        _reported_parameter_defaults, normalize_ollama_settings,
        is_ollamapull_route, ollamapull_model_name,
    )

    assert is_ollamapull_route("ollamapull")
    assert is_ollamapull_route("ollamapull/")
    assert is_ollamapull_route("OLLAMAPULL/llama3.2")
    assert not is_ollamapull_route("ollama/llama3.2")
    assert ollamapull_model_name("ollamapull/llama3.2:latest") == "llama3.2:latest"
    assert ollamapull_model_name("OLLAMAPULL/custom/model") == "custom/model"
    assert ollamapull_model_name("ollama/llama3.2") == ""
    assert normalize_ollama_settings(None) == {"auto_update": True, "models": {}}
    assert _reported_parameter_defaults({
        "parameters": 'num_ctx 8192\nPARAMETER temperature 0.4\nstop "<end>"',
    }) == {"num_ctx": "8192", "temperature": "0.4", "stop": '"<end>"'}


def test_ollamapull_settings_persist_native_options(ollama_qapp, monkeypatch):
    from ollama_settings_dialog import OllamaSettingsDialog

    monkeypatch.setattr(OllamaSettingsDialog, "refresh_status", lambda self: None)
    translator = _ollama_test_translator()
    translator.config["ollama_settings"] = {
        "models": {"other-model": {"options": {"seed": 4}}},
        "future_key": "keep",
    }
    dialog = OllamaSettingsDialog(translator, "ollamapull/llama3.2")
    dialog.option_fields["num_ctx"][0].setText("8192")
    dialog.option_fields["draft_num_predict"][0].setText("4")
    dialog.extra_options_edit.setPlainText('{"future_option": 123}')
    dialog.extra_request_edit.setPlainText('{"logprobs": true}')
    dialog.think_field.setText("high")
    dialog.keep_alive_field.setText("10m")
    dialog.format_field.setText("json")
    dialog.auto_update_checkbox.setChecked(False)
    dialog.save_settings()

    saved = translator.config["ollama_settings"]
    assert saved["auto_update"] is False
    assert saved["future_key"] == "keep"
    assert saved["models"]["other-model"]["options"] == {"seed": 4}
    selected = saved["models"]["llama3.2"]
    assert selected["options"] == {
        "future_option": 123, "num_ctx": 8192, "draft_num_predict": 4,
    }
    assert selected["request"] == {"logprobs": True}
    assert selected["think"] == "high"
    assert selected["keep_alive"] == "10m"
    assert selected["format"] == "json"
    assert translator.saved == 1
    assert json.loads(os.environ["OLLAMA_SETTINGS_JSON"]) == saved


def test_ollamapull_advanced_fields_validate_input(ollama_qapp, monkeypatch):
    from ollama_settings_dialog import OllamaSettingsDialog

    monkeypatch.setattr(OllamaSettingsDialog, "refresh_status", lambda self: None)
    dialog = OllamaSettingsDialog(_ollama_test_translator(), "ollamapull/test")
    dialog.extra_options_edit.setPlainText("[]")
    with pytest.raises(ValueError, match="JSON object"):
        dialog._collect_model_settings()
    dialog.extra_options_edit.setPlainText("{}")
    dialog.extra_request_edit.setPlainText('{"model": "other"}')
    with pytest.raises(ValueError, match="cannot override"):
        dialog._collect_model_settings()
    dialog.extra_request_edit.setPlainText("{}")
    dialog.option_fields["num_ctx"][0].setText("0")
    with pytest.raises(ValueError, match="greater than zero"):
        dialog._collect_model_settings()


def test_bare_ollamapull_route_opens_global_settings(ollama_qapp, monkeypatch):
    from ollama_settings_dialog import OllamaSettingsDialog

    monkeypatch.setattr(OllamaSettingsDialog, "refresh_status", lambda self: None)
    translator = _ollama_test_translator()
    dialog = OllamaSettingsDialog(translator, "ollamapull/")
    assert dialog.model_name == ""
    assert not dialog.tabs.isTabEnabled(0)
    dialog.auto_update_checkbox.setChecked(False)
    dialog.save_settings()
    assert translator.config["ollama_settings"]["auto_update"] is False
    assert translator.config["ollama_settings"]["models"] == {}


def test_multi_key_ollamapull_button_follows_model_field(ollama_qapp, monkeypatch):
    from PySide6.QtWidgets import QComboBox, QDialog, QPushButton
    from multi_api_key_manager import MultiAPIKeyDialog
    from ollama_settings_dialog import OllamaRouteButtonController

    monkeypatch.setattr(OllamaRouteButtonController, "_check_installation_passively", lambda self: None)

    manager = QDialog()
    manager.translator_gui = _ollama_test_translator()
    combo = QComboBox()
    combo.setEditable(True)
    container = MultiAPIKeyDialog._wrap_model_with_ollama_settings(manager, combo)
    button = combo._ollama_settings_button
    assert container.layout().itemAt(0).widget() is combo
    assert isinstance(button, QPushButton)
    assert button.isHidden()
    combo.setCurrentText("ollamapull")
    assert not button.isHidden()
    assert button.text() == "🦙 Download Ollama"
    combo.setCurrentText("ollamapull/")
    assert not button.isHidden()
    combo.setCurrentText("ollamapull/qwen3")
    assert not button.isHidden()
    combo.setCurrentText("openai/gpt-4.1")
    assert button.isHidden()


def test_main_ollamapull_button_follows_model_field(ollama_qapp):
    from PySide6.QtWidgets import QComboBox, QPushButton
    from translator_gui import TranslatorGUI

    fake = type("ModelView", (), {})()
    fake.ollama_settings_btn = QPushButton("Ollama Settings")
    fake.model_combo = QComboBox()
    fake.model_combo.setEditable(True)
    TranslatorGUI._update_ollama_settings_button(fake, "ollamapull")
    assert not fake.ollama_settings_btn.isHidden()
    TranslatorGUI._update_ollama_settings_button(fake, "ollamapull/")
    assert not fake.ollama_settings_btn.isHidden()
    TranslatorGUI._update_ollama_settings_button(fake, "ollamapull/gemma3")
    assert not fake.ollama_settings_btn.isHidden()
    TranslatorGUI._update_ollama_settings_button(fake, "authgpt/gpt-6-luna")
    assert fake.ollama_settings_btn.isHidden()


def test_ollamapull_button_rechecks_then_installs_or_opens_settings(ollama_qapp, monkeypatch):
    import ollama_settings_dialog as module
    from PySide6.QtWidgets import QPushButton

    monkeypatch.setattr(module.OllamaRouteButtonController, "_check_installation_passively", lambda self: None)
    translator = _ollama_test_translator()
    route = ["ollamapull/"]
    button = QPushButton()
    controller = module.OllamaRouteButtonController(button, translator, lambda: route[0], translator)
    controller.update_model()
    assert button.text() == "🦙 Download Ollama"
    operations = []
    monkeypatch.setattr(controller, "_start_job", lambda operation, fn: operations.append(operation))
    monkeypatch.setattr(controller, "_begin_install", lambda: operations.append("install"))
    controller._on_clicked()
    assert operations == ["click_status"]
    controller._on_job_finished((1, "click_status", {"installed": False}, None))
    assert operations[-1] == "install"

    controller._busy = False
    route[0] = "ollamapull/qwen3"
    opened = []
    monkeypatch.setattr(module, "open_ollama_settings", lambda *args: opened.append(args))
    controller.update_model()
    controller._on_clicked()
    controller._on_job_finished((2, "click_status", {"installed": True}, None))
    assert opened and opened[0][1] == "ollamapull/qwen3"
    assert button.text() == "Ollama Settings"


def test_ollamapull_button_status_runs_in_background(ollama_qapp, monkeypatch):
    import time
    import ollamapull
    from PySide6.QtWidgets import QPushButton
    from ollama_settings_dialog import OllamaRouteButtonController

    monkeypatch.setattr(ollamapull, "get_status", lambda _model="": {"installed": True})
    translator = _ollama_test_translator()
    button = QPushButton()
    controller = OllamaRouteButtonController(
        button, translator, lambda: "ollamapull", translator,
    )
    controller.update_model()
    assert button.text() == "🦙 Download Ollama"
    deadline = time.monotonic() + 3
    while button.text() != "Ollama Settings" and time.monotonic() < deadline:
        ollama_qapp.processEvents()
        time.sleep(0.01)
    assert button.text() == "Ollama Settings"
    assert not controller._jobs
