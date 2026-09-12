import ast
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).parents[1] / 'src'


def method(filename, cls, name):
    tree = ast.parse((ROOT / filename).read_text(encoding='utf-8-sig'))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    fn = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == name)
    fn.decorator_list = []
    import threading
    ns = {'threading': threading}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), filename, 'exec'), ns)
    return ns[name]


def test_account_status_does_not_block_ui_on_decryption(monkeypatch):
    import queue
    import sys
    import threading
    import time
    load = method('translator_gui.py', 'TranslatorGUI', '_auth_status_snapshot')
    receive = method('translator_gui.py', 'TranslatorGUI', '_receive_auth_status_snapshot')
    entered, release = threading.Event(), threading.Event()
    results = queue.Queue()
    calls = []
    def get_store(account):
        calls.append((account, threading.get_ident()))
        entered.set()
        assert release.wait(3)
        return SimpleNamespace(has_tokens=True, account_info={'email': 'test@example.test'})
    monkeypatch.setitem(sys.modules, 'authgpt_auth', SimpleNamespace(get_store=get_store))
    updates = []
    slot = [1]
    gui = SimpleNamespace(_get_authgpt_account_id=lambda: slot[0],
                          auth_status_ready_signal=SimpleNamespace(emit=lambda *args: results.put(args)),
                          _update_authgpt_login_status=lambda: updates.append(True))
    try:
        started = time.monotonic()
        for _ in range(20):
            assert load(gui, 'authgpt') is None
        assert time.monotonic() - started < 0.5
        assert entered.wait(1)
        assert len(calls) == 1 and calls[0][1] != threading.get_ident()
        slot[0] = 2
        release.set()
        receive(gui, *results.get(timeout=2))
        assert updates == []  # The old account cannot update the new selection.
        assert gui._auth_status_cache[('authgpt', 1)].has_tokens
    finally:
        release.set()


def test_proxy_status_decryption_runs_in_worker(monkeypatch):
    import queue
    import sys
    import threading
    load = method('translator_gui.py', 'TranslatorGUI', '_auth_status_snapshot')
    for provider, module, function in (
        ('antigravity', 'antigravity_proxy', 'get_stored_account_summary'),
        ('ocagy', 'ocagy_cli', 'get_account_summary'),
    ):
        results = queue.Queue()
        callers = []
        def summary():
            callers.append(threading.get_ident())
            return {'account_count': 3}
        monkeypatch.setitem(sys.modules, module, SimpleNamespace(**{function: summary}))
        gui = SimpleNamespace(auth_status_ready_signal=SimpleNamespace(emit=lambda *args: results.put(args)))
        assert load(gui, provider) is None
        key, result = results.get(timeout=2)
        assert key == (provider, 0) and result == {'account_count': 3}
        assert callers[0] != threading.get_ident()


def test_manager_does_not_install_arena_login_widgets():
    source = (ROOT / 'multi_api_key_manager.py').read_text(encoding='utf-8-sig')
    assert 'install_combo_login' not in source
    assert 'create_login_controls' not in source


def test_live_model_hint_tracks_edits_without_loading_accounts():
    fn = method('multi_api_key_manager.py', 'MultiAPIKeyDialog', '_pending_autharena_model')
    text = ['autharena3/model']
    manager = SimpleNamespace(_model_search_combos=[SimpleNamespace(currentText=lambda: text[0])])
    assert fn(manager) == 'autharena3/model'
    text[0] = 'authgpt/model'
    assert fn(manager) == ''


def test_authgpt_live_pool_hint_matches_authgrok_behavior():
    fn = method('multi_api_key_manager.py', 'MultiAPIKeyDialog', '_has_pending_authgrok_pool_model')
    text = ['authgpt0/']
    manager = SimpleNamespace(model_combo=SimpleNamespace(currentText=lambda: text[0]))
    assert fn(manager, 'authgpt')
    assert not fn(manager)
    text[0] = 'authgpt1/model'
    assert not fn(manager, 'authgpt')
    text[0] = 'authgrok0/'
    assert fn(manager)


def test_parent_arena_visibility_uses_enabled_pools_and_primary_precedence():
    fn = method('translator_gui.py', 'TranslatorGUI', '_autharena_control_model')
    entries = [('metadata_keys', 'use_metadata_keys', 'autharena2/model')]
    gui = SimpleNamespace(model_var='authgpt/model', config={},
                          _iter_enabled_key_pool_models=lambda: iter(entries),
                          _multi_key_manager_autharena_model_hint='autharena3/model')
    assert fn(gui) == 'autharena3/model'
    gui._multi_key_manager_autharena_model_hint = ''
    assert fn(gui) == 'autharena2/model'
    gui._multi_key_manager_autharena_model_hint = 'autharena0/model'
    assert fn(gui) == 'autharena0/model'
    gui._multi_key_manager_autharena_model_hint = 'autharena3/model'
    entries.clear()
    assert fn(gui) == 'autharena3/model'
    gui.model_var = 'autharena0/model'
    assert fn(gui) == 'autharena0/model'
    gui.model_var = 'authgpt/model'
    gui._multi_key_manager_autharena_model_hint = ''
    assert fn(gui) == 'authgpt/model'


def test_proxy_live_hints_refresh_parent_and_clear_after_edit():
    from types import MethodType
    pending = method('multi_api_key_manager.py', 'MultiAPIKeyDialog', '_has_pending_proxy_model')
    refresh = method('multi_api_key_manager.py', 'MultiAPIKeyDialog', '_refresh_parent_model_requirements')
    texts = [' Antigravity/model ', 'ocagy/model']
    observed = []
    gui = SimpleNamespace()
    gui.on_model_change = lambda: observed.append((
        gui._multi_key_manager_antigravity_hint, gui._multi_key_manager_ocagy_hint))
    manager = SimpleNamespace(
        translator_gui=gui,
        _model_search_combos=[SimpleNamespace(currentText=lambda i=i: texts[i]) for i in range(2)],
        _pending_autharena_model=lambda *args: '',
        _has_pending_authgrok_pool_model=lambda *args: False,
        _has_pending_google_creds_model=lambda: False,
    )
    manager._has_pending_proxy_model = MethodType(pending, manager)
    refresh(manager)
    assert observed[-1] == (True, True)
    texts[:] = ['authgpt/model', 'ocagy-unrelated/model']
    refresh(manager)
    assert observed[-1] == (False, False)


def test_main_gui_proxy_controls_follow_live_hints():
    tree = ast.parse((ROOT / 'translator_gui.py').read_text(encoding='utf-8-sig'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TranslatorGUI')
    change = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'on_model_change')
    class Button:
        def show(self): self.visible = True
        def hide(self): self.visible = False
    for provider in ('antigravity', 'ocagy'):
        # Execute the actual provider visibility block without constructing the full app.
        block = next(n for n in change.body if isinstance(n, ast.If)
                     and ast.unparse(n.test) == f"hasattr(self, '{provider}_login_btn')")
        gui = SimpleNamespace(**{
            provider + '_login_btn': Button(),
            '_multi_key_manager_' + provider + '_hint': True,
            '_has_' + provider + '_in_key_pools': lambda: False,
            '_update_' + provider + '_login_status': lambda: None,
        })
        code = compile(ast.Module(body=[block], type_ignores=[]), 'visibility', 'exec')
        exec(code, {'self': gui, 'model': 'authgpt/model'})
        assert getattr(gui, provider + '_login_btn').visible
        setattr(gui, '_multi_key_manager_' + provider + '_hint', False)
        exec(code, {'self': gui, 'model': 'authgpt/model'})
        assert not getattr(gui, provider + '_login_btn').visible


def test_vertex_live_model_hint_accepts_bare_and_numbered_prefixes():
    fn = method('multi_api_key_manager.py', 'MultiAPIKeyDialog', '_pending_autharena_model')
    text = ['authgem-vertex']
    manager = SimpleNamespace(_model_search_combos=[SimpleNamespace(currentText=lambda: text[0])])
    for value in ('authgem-vertex', 'authgem-vertex0/', 'authgem-vertex3/model'):
        text[0] = value
        assert fn(manager, 'authgem-vertex') == value
    text[0] = 'authgem/model'
    assert fn(manager, 'authgem-vertex') == ''
