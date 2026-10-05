"""Host tests: after bootstrap() no auth module touches the real user profile (U4).

The auth modules resolve ``~`` when they are imported. On Android/iOS
``expanduser`` follows the bootstrap's ``HOME=<data>/home``, but on a Windows
dev run (``flet run``) it reads ``USERPROFILE`` and ignores ``HOME``, so the app
would share - and on "Sign out" delete - the desktop app's token files. The
bootstrap therefore sets the token-file overrides (``GLOSSARION_TOKEN_DIR``,
``AUTH*_TOKEN_FILE`` and friends) under ``<data>/home/.glossarion``.

The proof runs a fresh interpreter whose profile (``USERPROFILE`` and ``HOME``)
is a decoy folder filled with token files, a Claude Code login and a Grok CLI
login. It bootstraps, imports the auth modules and exercises every store an
account slot uses (load, save, pending sign-in, sign-out, slot listing, login
caches) under an audit hook that records any open/list/create/remove inside
the decoy profile.

Run from src/mobile:  python -m pytest -p no:cacheprovider tests_host/test_auth_token_paths.py -q
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from glossarion_mobile import runtime_bootstrap as rb  # noqa: E402

TOKEN_ENV_KEYS = (
    "GLOSSARION_TOKEN_DIR",
    "AUTHGPT_TOKEN_FILE",
    "AUTHGEM_TOKEN_FILE",
    "AUTHGROK_TOKEN_FILE",
    "AUTHCD_TOKEN_FILE",
    "OPERA_ARIA_TOKEN_FILE",
    "AUTHARENA_PROXY_DATA_DIR",
)

DECOYS = (
    ".glossarion/authgpt_tokens.json",
    ".glossarion/authgpt_tokens_2.json",
    ".glossarion/authgem_tokens.json",
    ".glossarion/authgem_tokens_2.json",
    ".glossarion/authgrok_tokens.json",
    ".glossarion/authgrok_tokens_2.json",
    ".glossarion/authcd_tokens.json",
    ".glossarion/authcd_tokens_2.json",
    ".glossarion/authcd_client_version.json",
    ".glossarion/installation_id",
    ".glossarion/claude-code/.credentials.json",
    ".claude/.credentials.json",
    ".grok/auth.json",
)

PROBE = textwrap.dedent(
    r'''
    import json, os, sys, time

    app_dir, profile, out_path = sys.argv[1:4]
    sys.path.insert(0, app_dir)
    from glossarion_mobile import runtime_bootstrap as rb

    paths = rb.bootstrap(app_dir=app_dir, force=True, configure_logging=False)

    PROFILE = os.path.normcase(os.path.abspath(profile))
    EVENTS = {"open", "os.listdir", "os.scandir", "os.remove", "os.unlink", "os.rename", "os.replace",
              "os.mkdir", "os.rmdir", "os.chmod", "os.truncate", "shutil.copyfile", "shutil.rmtree"}
    current = ["import"]
    touched = []

    def hook(event, args):
        if event not in EVENTS or not args:
            return
        target = args[0]
        if isinstance(target, int) or target is None:
            return
        try:
            full = os.path.normcase(os.path.abspath(os.fsdecode(os.fspath(target))))
        except Exception:
            return
        if full == PROFILE or full.startswith(PROFILE + os.sep):
            touched.append([current[0], event, os.fsdecode(os.fspath(target))])

    sys.addaudithook(hook)

    import token_encryption
    import authgpt_auth, authgem_auth, authgrok_auth, authcd_auth

    results, files = {}, {}

    def check(label, fn):
        current[0] = label
        try:
            fn()
            results[label] = "ok"
        except Exception as exc:
            results[label] = f"{type(exc).__name__}: {exc}"
        current[0] = "-"

    def exercise(label, store):
        files[label] = store._token_file
        store.load_tokens()
        _ = store.has_tokens
        store.save_tokens({"access_token": "at-" + label, "refresh_token": "rt", "expires_at": time.time() + 3600})
        assert store.load_tokens()["access_token"] == "at-" + label
        store.save_pending_oauth({"provider": "x", "state": "s", "created_at": time.time()})
        assert store.load_pending_oauth()["state"] == "s"
        store.clear_pending_oauth()
        store.clear_tokens()

    for module, name in ((authgpt_auth, "authgpt"), (authgem_auth, "authgem"),
                         (authgrok_auth, "authgrok"), (authcd_auth, "authcd")):
        for slot in (0, 2):
            label = f"{name} slot {slot}"
            check(label, lambda module=module, slot=slot, label=label: exercise(label, module.get_store(slot)))

    def grok_extras():
        authgrok_auth.get_saved_account_ids()  # lists the token folder, reads the Grok CLI login (slot 0)
        try:
            authgrok_auth.get_store(0).get_valid_access_token(auto_login=False)
        except RuntimeError:
            pass

    def cd_extras():
        authcd_auth.claude_code_client_version()
        authcd_auth._adopt_client_version("999.0.0")
        authcd_auth.import_claude_code_login(authcd_auth.get_store(0))
        files["authcd client version"] = authcd_auth._CLIENT_VERSION_FILE

    def gem_extras():
        authgem_auth._get_installation_id()
        files["authgem installation id"] = os.path.join(authgem_auth._DEFAULT_TOKEN_DIR, "installation_id")

    check("authgrok extras", grok_extras)
    check("authcd extras", cd_extras)
    check("authgem extras", gem_extras)

    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump({"results": results, "touched": touched, "files": files,
                   "token_dir": str(paths.token_dir), "home": str(paths.home)}, handle)
    '''
)


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("auth_paths")
    profile = tmp / "profile"
    for rel in DECOYS:
        target = profile / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('{"decoy": true}', encoding="utf-8")
    storage = {name: tmp / "storage" / name for name in ("data", "cache", "temp")}
    for directory in storage.values():
        directory.mkdir(parents=True)

    env = {
        k: v for k, v in os.environ.items()
        if not k.startswith(("GLOSSARION_", "FLET_", "AUTHGPT_", "AUTHGEM_", "AUTHGROK_", "AUTHCD_"))
        and k not in ("CONFIG_FILE", "OPERA_ARIA_TOKEN_FILE", "AUTHARENA_PROXY_DATA_DIR", "PYTHONPATH")
    }
    env.update(
        HOME=str(profile),
        USERPROFILE=str(profile),  # Windows expanduser reads this, not HOME
        FLET_APP_STORAGE_DATA=str(storage["data"]),
        FLET_APP_STORAGE_CACHE=str(storage["cache"]),
        FLET_APP_STORAGE_TEMP=str(storage["temp"]),
        GLOSSARION_BACKEND_DIR=str(SRC_DIR),
        GLOSSARION_TOKEN_KEY_B64=base64.b64encode(bytes(range(32))).decode("ascii"),
        PYTHONIOENCODING="utf-8",
        PYTHONNOUSERSITE="1",
    )
    out = tmp / "probe.json"
    proc = subprocess.run(
        [sys.executable, "-c", PROBE, str(APP_DIR), str(profile), str(out)],
        env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300,
    )
    if proc.returncode != 0 or not out.exists():
        if "No module named 'requests'" in (proc.stderr or ""):
            pytest.skip("backend dependencies (requests) are not installed in this interpreter")
        raise AssertionError(f"probe failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
    data = json.loads(out.read_text(encoding="utf-8"))
    data["profile"] = profile
    data["storage"] = storage
    return data


def _under(path, root) -> bool:
    path = os.path.normcase(os.path.abspath(str(path)))
    root = os.path.normcase(os.path.abspath(str(root)))
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def test_env_contract_puts_every_token_override_under_home(tmp_path, monkeypatch):
    for name in ("data", "cache", "temp"):
        monkeypatch.setenv(f"FLET_APP_STORAGE_{name.upper()}", str(tmp_path / name))
    paths = rb.resolve_paths(APP_DIR, platform="android")
    env = paths.env_contract()
    token_dir = paths.home / ".glossarion"
    assert paths.token_dir == token_dir
    assert env["GLOSSARION_TOKEN_DIR"] == str(token_dir)
    assert set(TOKEN_ENV_KEYS) <= set(env)
    assert paths.token_env() == {key: env[key] for key in TOKEN_ENV_KEYS}
    for key in TOKEN_ENV_KEYS:
        assert _under(env[key], token_dir), key
    assert env["AUTHGPT_TOKEN_FILE"] == str(token_dir / "authgpt_tokens.json")
    assert env["AUTHCD_TOKEN_FILE"] == str(token_dir / "authcd_tokens.json")


def test_no_auth_store_touches_the_real_profile(probe):
    assert {label: result for label, result in probe["results"].items() if result != "ok"} == {}
    assert probe["touched"] == []
    for rel in DECOYS:  # nothing was removed or rewritten
        assert (probe["profile"] / rel).read_text(encoding="utf-8") == '{"decoy": true}', rel
    token_dir = probe["token_dir"]
    assert _under(token_dir, probe["storage"]["data"])
    for label, path in probe["files"].items():
        assert _under(path, token_dir), (label, path)
    assert {os.path.basename(probe["files"][f"{name} slot {slot}"])
            for name in ("authgpt", "authgem", "authgrok", "authcd") for slot in (0, 2)} == {
        "authgpt_tokens.json", "authgpt_tokens_2.json", "authgem_tokens.json", "authgem_tokens_2.json",
        "authgrok_tokens.json", "authgrok_tokens_2.json", "authcd_tokens.json", "authcd_tokens_2.json",
    }


def test_chatgpt_numbered_slots_stay_in_the_app_folder(probe):
    # authgpt_auth builds numbered slots (authgpt_tokens_N.json) from _DEFAULT_TOKEN_DIR, which honours
    # GLOSSARION_TOKEN_DIR (oauth_session.default_token_dir), like the other auth modules.
    assert [t for t in probe["touched"] if t[0] == "authgpt slot 2"] == []
    assert _under(probe["files"]["authgpt slot 2"], probe["token_dir"])
