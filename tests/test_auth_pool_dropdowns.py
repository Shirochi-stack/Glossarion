import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(scope='module')
def gui_methods():
    tree = ast.parse((Path(__file__).parents[1] / 'src/translator_gui.py').read_text(encoding='utf-8-sig'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TranslatorGUI')
    names = {'_refresh_auth_account_arrows', '_get_authgpt_account_id', '_authgpt_pool_route_requested'}
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
