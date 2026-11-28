"""Host test: a custom prefix endpoint corrected after a 404 reaches the mobile settings store.

Run from src/mobile:
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_custom_endpoint_listener.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

MOBILE_DIR = Path(__file__).resolve().parents[1]
for extra in (MOBILE_DIR / "app", MOBILE_DIR.parent):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import custom_endpoint_discovery as ced  # noqa: E402
from glossarion_mobile.state.config_store import MobileConfigStore  # noqa: E402
from glossarion_mobile.ui.settings.integration import custom_route_listener  # noqa: E402

ROUTE = {"prefix": "graf/", "routing": "https://status.5n7.net", "endpoint_type": "/chat/completions"}


def test_the_store_takes_the_new_route_and_keeps_the_old_url(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"model": "m", "custom_prefix_routes": [ROUTE]}), encoding="utf-8")
    store = MobileConfigStore(str(path), debounce=0.05)
    store.load()
    listener = custom_route_listener(store)
    assert listener("graf/", "https://status.5n7.net", "https://api.5n7.net/v1") is True
    assert store.get("custom_prefix_routes")[0]["routing"] == "https://api.5n7.net/v1"
    assert store.get(ced.PREVIOUS_ROUTING_KEY) == {"graf/": "https://status.5n7.net"}
    assert store.flush() in (True, False)
    store.wait_idle(2)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["model"] == "m"
    assert saved["custom_prefix_routes"][0]["routing"] == "https://api.5n7.net/v1"
    assert saved[ced.PREVIOUS_ROUTING_KEY] == {"graf/": "https://status.5n7.net"}
    store.close()


def test_an_unknown_prefix_changes_nothing(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"custom_prefix_routes": [ROUTE]}), encoding="utf-8")
    store = MobileConfigStore(str(path), debounce=0.05)
    store.load()
    assert custom_route_listener(store)("nope/", "a", "b") is False
    assert store.get("custom_prefix_routes")[0]["routing"] == "https://status.5n7.net"
    store.close()
