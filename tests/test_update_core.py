"""U9 update_core: the release-check logic moved out of update_manager.UpdateManager.

Covers
  * the moved members of ``update_core.UpdateCoreMixin`` are verbatim copies of the
    ``UpdateManager`` members at BASE_SHA (the parent of the move), the two splits included:
    ``_check_for_updates_core`` == the try-body of ``_check_for_updates_internal`` and
    ``_skip_latest_version`` == the config part of ``skip_version``; the desktop keeps the
    message-box handlers and calls the core;
  * desktop parity: the legacy ``UpdateManager`` methods (exec'd from ``git show BASE_SHA``) and
    today's ``UpdateManager`` give the same results, config writes, saved check times, release
    state and message boxes on scripted GitHub responses (cache window, newer / same / skipped
    release, history failure, timeout, connection error, 403/500, invalid JSON; silent and not);
  * ``HeadlessUpdateChecker`` (the GUI-free host the mobile Updates screen uses) gives the
    same core results as the desktop;
  * import hygiene: update_core imports without PySide6 / translator_gui and parses as 3.10.

Nothing reaches the network: ``requests.get`` is replaced by a scripted fake.
"""

import ast
import os
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import update_core  # noqa: E402

# The parent of the U9 move (HEAD when update_core was extracted).
BASE_SHA = "1cd681786382abed23069f4c06cb583948371d1a"

MOVED_MEMBERS = (
    "MIN_UPDATE_SIZE", "_eligible_update_asset", "_validate_update_file", "GITHUB_API_URL",
    "GITHUB_LATEST_URL", "_detect_build_variant", "_detect_arch", "_detect_platform",
    "_asset_platform", "_asset_arch", "fetch_multiple_releases", "_save_last_check_time",
)


def _git_show(relpath: str) -> str:
    try:
        data = subprocess.run(["git", "show", f"{BASE_SHA}:{relpath}"], cwd=str(ROOT),
                              check=True, capture_output=True).stdout
    except Exception as exc:  # pragma: no cover - shallow clone
        message = f"git show {BASE_SHA}:{relpath} unavailable: {exc}"
        if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
            pytest.fail(message + " (CI must fetch the base commit)")
        pytest.skip(message)
    return data.decode("utf-8-sig").replace("\r\n", "\n")


def _class_members(tree: ast.Module, name: str) -> dict:
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    out = {}
    for node in cls.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[node.name] = node
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            out[node.targets[0].id] = node
    return out


def _body_without_docstring(fn: ast.FunctionDef) -> list:
    body = list(fn.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    return body


def _unparse(nodes) -> str:
    return "\n".join(ast.unparse(n) for n in nodes)


@pytest.fixture(scope="module")
def legacy_source() -> str:
    return _git_show("src/update_manager.py")


# ---------------------------------------------------------------------------
# verbatim moves
# ---------------------------------------------------------------------------

def test_moved_members_are_verbatim(legacy_source):
    legacy = _class_members(ast.parse(legacy_source), "UpdateManager")
    core = _class_members(ast.parse((SRC / "update_core.py").read_text(encoding="utf-8-sig")), "UpdateCoreMixin")
    for name in MOVED_MEMBERS:
        assert name in legacy, name
        assert ast.unparse(core[name]) == ast.unparse(legacy[name]), f"update_core.{name} differs from {BASE_SHA[:12]}"


def test_check_split_is_the_legacy_try_body(legacy_source):
    legacy = _class_members(ast.parse(legacy_source), "UpdateManager")["_check_for_updates_internal"]
    legacy_try = next(n for n in _body_without_docstring(legacy) if isinstance(n, ast.Try))
    core_tree = ast.parse((SRC / "update_core.py").read_text(encoding="utf-8-sig"))
    core = _class_members(core_tree, "UpdateCoreMixin")["_check_for_updates_core"]
    assert _unparse(_body_without_docstring(core)) == _unparse(legacy_try.body)

    # Today's desktop method: same signature and handlers, its try-body is the core call.
    now_tree = ast.parse((SRC / "update_manager.py").read_text(encoding="utf-8-sig"))
    now = _class_members(now_tree, "UpdateManager")["_check_for_updates_internal"]
    assert ast.unparse(now.args) == ast.unparse(legacy.args)
    now_try = next(n for n in _body_without_docstring(now) if isinstance(n, ast.Try))
    assert _unparse(now_try.body) == "return self._check_for_updates_core(force_show)"
    assert _unparse(now_try.handlers) == _unparse(legacy_try.handlers)
    assert [ast.unparse(n) for n in now_try.orelse + now_try.finalbody] == \
           [ast.unparse(n) for n in legacy_try.orelse + legacy_try.finalbody]
    assert ast.get_docstring(now) == ast.get_docstring(legacy)


def test_skip_split_is_the_legacy_config_block(legacy_source):
    legacy = _class_members(ast.parse(legacy_source), "UpdateManager")["skip_version"]
    core = _class_members(ast.parse((SRC / "update_core.py").read_text(encoding="utf-8-sig")),
                          "UpdateCoreMixin")["_skip_latest_version"]
    now = _class_members(ast.parse((SRC / "update_manager.py").read_text(encoding="utf-8-sig")),
                         "UpdateManager")["skip_version"]
    old = _body_without_docstring(legacy)
    # legacy: [if not latest: close; return] + [config block (4 statements)] + [dialog.close(), message box ...]
    moved = old[1:5]
    assert _unparse(moved).startswith("if 'skipped_versions' not in self.main_gui.config")
    core_body = _body_without_docstring(core)
    assert _unparse(core_body[:-1]) == _unparse(moved)
    assert ast.unparse(core_body[-1]) == "return version_tag"
    new = _body_without_docstring(now)
    assert ast.unparse(new[1]) == "version_tag = self._skip_latest_version()"
    assert _unparse([new[0]] + new[2:]) == _unparse([old[0]] + old[5:])


def test_update_manager_lost_only_the_moved_members(legacy_source):
    legacy = set(_class_members(ast.parse(legacy_source), "UpdateManager"))
    now = set(_class_members(ast.parse((SRC / "update_manager.py").read_text(encoding="utf-8-sig")), "UpdateManager"))
    assert legacy - now == set(MOVED_MEMBERS)
    assert now - legacy == set()
    text = (SRC / "update_manager.py").read_text(encoding="utf-8-sig")
    assert "class UpdateManager(UpdateCoreMixin, QObject):" in text
    assert "from update_core import UpdateCoreMixin" in text


def test_sources_keep_uniform_line_endings():
    for name in ("update_core.py", "update_manager.py"):
        data = (SRC / name).read_bytes()
        assert data.count(b"\r\n") in (0, data.count(b"\n")), name


# ---------------------------------------------------------------------------
# import hygiene
# ---------------------------------------------------------------------------

def test_update_core_imports_without_the_desktop_gui():
    probe = textwrap.dedent(
        """
        import sys
        for name in ("PySide6", "shiboken6", "tkinter", "translator_gui", "dpi_setup"):
            sys.modules[name] = None
        sys.path.insert(0, sys.argv[1])
        import update_core
        assert "update_manager" not in sys.modules
        assert not [m for m, v in sys.modules.items() if m.startswith("PySide6") and v is not None]
        print("ok")
        """
    )
    out = subprocess.run([sys.executable, "-c", probe, str(SRC)], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr


def test_update_core_parses_as_python_310():
    ast.parse((SRC / "update_core.py").read_text(encoding="utf-8-sig"), feature_version=(3, 10))


# ---------------------------------------------------------------------------
# behaviour parity (scripted GitHub responses)
# ---------------------------------------------------------------------------

class _Response:
    def __init__(self, status=200, payload=None, bad_json=False):
        self.status_code = status
        self._payload = payload
        self._bad_json = bad_json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error", response=self)

    def json(self):
        if self._bad_json:
            raise ValueError("Expecting value: line 1 column 1 (char 0)")
        return self._payload


def _release(tag: str, body: str = "Notes\n\n\n\nmore") -> dict:
    return {"tag_name": tag, "body": body, "html_url": f"https://example.invalid/{tag}",
            "published_at": "2026-10-08T00:00:00Z", "assets": []}


LATEST = "https://api.github.com/repos/Shirochi-stack/Glossarion/releases/latest"
HISTORY = "https://api.github.com/repos/Shirochi-stack/Glossarion/releases?per_page=10"


def _scenario_get(name: str):
    def get(url, headers=None, timeout=None):
        assert headers == {'Accept': 'application/vnd.github.v3+json', 'User-Agent': 'Glossarion-Updater'}
        if name == "timeout":
            raise requests.Timeout("timed out")
        if name == "conn_github":
            raise requests.ConnectionError("HTTPSConnectionPool(host='api.github.com', port=443)")
        if name == "conn_other":
            raise requests.ConnectionError("Name or service not known")
        if name == "http403":
            return _Response(403)
        if name == "http500":
            return _Response(500)
        if name == "bad_json":
            return _Response(200, bad_json=True)
        if url == LATEST:
            tag = {"same": "v9.14.0", "older": "v9.13.0"}.get(name, "v9.15.0")
            return _Response(200, _release(tag))
        if url == HISTORY:
            if name == "history_fails":
                raise requests.ConnectionError("history offline")
            return _Response(200, [_release("v9.15.0"), _release("v9.14.0", "x")])
        raise AssertionError(url)
    return get


SCENARIOS = ("newer", "same", "older", "skipped", "history_fails", "timeout", "conn_github", "conn_other",
             "http403", "http500", "bad_json", "cached")


class _Gui:
    def __init__(self, config: dict):
        self.config = config
        self.saves = []

    def save_config(self, show_message=True):
        self.saves.append((show_message, dict(self.config)))


class _MessageBox:
    log: list = []
    Critical = "critical"
    Information = "information"

    def __init__(self, parent=None):
        self.record = {"parent": parent}
        _MessageBox.log.append(self.record)

    def setIcon(self, icon):
        self.record["icon"] = icon

    def setWindowTitle(self, title):
        self.record["title"] = title

    def setText(self, text):
        self.record["text"] = text

    def exec(self):
        self.record["shown"] = True


class _Dialog:
    def __init__(self):
        self.closed = 0

    def close(self):
        self.closed += 1


def _host(cls_dict: dict, base=object):
    members = ("_check_for_updates_internal", "skip_version", "fetch_multiple_releases", "_save_last_check_time",
               "GITHUB_API_URL", "GITHUB_LATEST_URL")
    ns = {name: cls_dict[name] for name in members if name in cls_dict}
    return type("Host", (base,), ns)


def _config_for(scenario: str) -> dict:
    config = {"last_update_check_time": 0}
    if scenario == "skipped":
        config["skipped_versions"] = ["v9.15.0"]
    if scenario == "cached":
        config["last_update_check_time"] = 1_000_000.0 - 60
    return config


def _new_obj(host_cls, scenario: str):
    obj = host_cls()
    config = _config_for(scenario)
    obj.main_gui = _Gui(config)
    obj.dialog = object()
    obj.CURRENT_VERSION = "9.14.0"
    obj._last_check_time = config["last_update_check_time"]
    obj._check_cache_duration = 1800
    obj.latest_release = None
    obj.all_releases = []
    obj.update_available = False
    return obj


def _state(obj, result):
    return {
        "result": result,
        "config": obj.main_gui.config,
        "saves": obj.main_gui.saves,
        "last_check": obj._last_check_time,
        "latest": obj.latest_release,
        "history": obj.all_releases,
        "available": obj.update_available,
        "boxes": list(_MessageBox.log),
    }


@pytest.fixture(scope="module")
def desktop_modules(legacy_source):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    legacy = types.ModuleType("legacy_update_manager_u9")
    legacy.__file__ = str(SRC / "update_manager.py")
    exec(compile(legacy_source, "legacy_update_manager_u9.py", "exec"), legacy.__dict__)
    import update_manager

    return legacy, update_manager


@pytest.fixture
def scripted(monkeypatch, desktop_modules):
    legacy, current = desktop_modules
    for module in (legacy, current):
        monkeypatch.setattr(module, "QMessageBox", _MessageBox)
    monkeypatch.setattr(update_core.time, "time", lambda: 1_000_000.0)
    monkeypatch.setattr(update_core.time, "sleep", lambda s: None)

    def use(name):
        monkeypatch.setattr(requests, "get", _scenario_get(name))
        _MessageBox.log = []

    return use


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("silent", (True, False))
@pytest.mark.parametrize("force_show", (False, True))
def test_check_matches_legacy(desktop_modules, scripted, capsys, scenario, silent, force_show):
    legacy_mod, current_mod = desktop_modules
    legacy_cls = _host(legacy_mod.UpdateManager.__dict__)
    current_cls = _host(dict(current_mod.UpdateManager.__dict__), base=update_core.UpdateCoreMixin)

    scripted(scenario)
    old = _new_obj(legacy_cls, scenario)
    old_state = _state(old, old._check_for_updates_internal(silent=silent, force_show=force_show))
    old_out = capsys.readouterr().out

    scripted(scenario)
    new = _new_obj(current_cls, scenario)
    new_state = _state(new, new._check_for_updates_internal(silent=silent, force_show=force_show))
    new_out = capsys.readouterr().out

    assert new_state == old_state
    assert new_out == old_out
    if not silent and scenario in ("timeout", "conn_github", "http403", "bad_json"):
        assert old_state["boxes"] and old_state["boxes"][0]["title"] == "Update Check Failed"


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("force_show", (False, True))
def test_headless_checker_matches_the_desktop_core(desktop_modules, scripted, capsys, scenario, force_show):
    legacy_mod, _current = desktop_modules
    legacy_cls = _host(legacy_mod.UpdateManager.__dict__)

    scripted(scenario)
    old = _new_obj(legacy_cls, scenario)
    old_result = old._check_for_updates_internal(silent=True, force_show=force_show)
    capsys.readouterr()

    scripted(scenario)
    gui = _Gui(_config_for(scenario))
    checker = update_core.HeadlessUpdateChecker(gui, "9.14.0", build_variant="Mobile")
    try:
        result = checker._check_for_updates_core(force_show=force_show)
    except (requests.Timeout, requests.ConnectionError, requests.HTTPError, ValueError):
        result = (False, None)  # what the desktop handlers return
    capsys.readouterr()
    assert result == old_result
    assert gui.config == old.main_gui.config
    assert gui.saves == old.main_gui.saves
    assert (checker.latest_release, checker.all_releases, checker.update_available) == \
           (old.latest_release, old.all_releases, old.update_available)


@pytest.mark.parametrize("has_latest", (True, False))
def test_skip_version_matches_legacy(desktop_modules, scripted, has_latest):
    legacy_mod, current_mod = desktop_modules
    states = []
    for module, base in ((legacy_mod, object), (current_mod, update_core.UpdateCoreMixin)):
        scripted("newer")
        obj = _new_obj(_host(dict(module.UpdateManager.__dict__), base=base), "newer")
        obj.main_gui.config["skipped_versions"] = ["v9.13.0"]
        obj.latest_release = _release("v9.15.0") if has_latest else None
        dialog = _Dialog()
        obj.skip_version(dialog)
        states.append((obj.main_gui.config, obj.main_gui.saves, dialog.closed, list(_MessageBox.log)))
    assert states[0] == states[1]
    if has_latest:
        assert states[1][0]["skipped_versions"] == ["v9.13.0", "v9.15.0"]


def test_headless_checker_skip_and_attributes():
    gui = _Gui({"last_update_check_time": 123.0})
    checker = update_core.HeadlessUpdateChecker(gui, "9.14.0", build_variant="Mobile")
    assert checker._last_check_time == 123.0 and checker._check_cache_duration == 1800
    assert checker.CURRENT_VERSION == "9.14.0" and checker._build_variant == "Mobile"
    checker.latest_release = _release("v9.15.0")
    assert checker._skip_latest_version() == "v9.15.0"
    assert checker._skip_latest_version() == "v9.15.0"
    assert gui.config["skipped_versions"] == ["v9.15.0"]
    assert [show for show, _cfg in gui.saves] == [False, False]
    # the asset helpers the mobile screen reuses (Android ABIs map onto the desktop arch names)
    assert update_core.UpdateCoreMixin._asset_arch("Glossarion_v9.15.0_Android_arm64-v8a.apk") == "arm64"
    assert update_core.UpdateCoreMixin._asset_arch("Glossarion_v9.15.0_Android_x86_64_debugsigned.apk") == "x64"
