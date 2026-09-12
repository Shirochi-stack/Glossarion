import ast
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
