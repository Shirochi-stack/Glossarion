"""Model manager restores the last tab; an explicit navigation target takes precedence."""
import pytest

from local_llm_fakes import FakePrefs
from test_models_keys import _store, _service, FakeOptions, parse_route
from test_local_llm_server import make_service
from glossarion_mobile.ui.screens.model_manager import ModelManagerScreen


@pytest.mark.parametrize("tab", ["models", "auths", "prefixes", "local"])
def test_last_tab_survives_screen_recreation(tmp_path, tab):
    prefs = FakePrefs()
    store = _store(tmp_path, {"model": "gpt-6"})
    catalog = _service(store, FakeOptions())
    catalog.load_blocking()
    engine, _ = make_service(tmp_path)
    kwargs = dict(catalog=catalog, prefs=prefs, local_engine=engine)
    screen = ModelManagerScreen(parse_route("/settings/models"), **kwargs)
    try:
        screen.get_body()
        screen.set_tab(tab)
        assert prefs.get("model_manager_tab") == tab
        reopened = ModelManagerScreen(parse_route("/settings/models"), **kwargs)
        assert reopened.tab == tab
        reopened.get_body()
        assert reopened.tabs.selected == [tab]
        target = ModelManagerScreen(parse_route("/settings/models"), tab="local", **kwargs)
        assert target.tab == "local" and prefs.get("model_manager_tab") == "local"
        reopened.dispose()
        target.dispose()
    finally:
        screen.dispose()
        store._saver.close()


@pytest.mark.parametrize("saved", [None, "retired", ["local"]])
def test_invalid_saved_tab_falls_back_to_models(tmp_path, saved):
    store = _store(tmp_path, {})
    catalog = _service(store, FakeOptions())
    catalog.load_blocking()
    screen = ModelManagerScreen(None, catalog=catalog, prefs=FakePrefs({"model_manager_tab": saved}))
    try:
        assert screen.tab == "models"
    finally:
        screen.dispose()
        store._saver.close()
