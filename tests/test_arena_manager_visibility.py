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
