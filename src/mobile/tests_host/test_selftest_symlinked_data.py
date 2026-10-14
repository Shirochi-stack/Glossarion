"""Self-test ``env_contract`` with the app data dir reached through a symlink.

Android's app storage ``/data/user/0/<pkg>`` is a symlink to ``/data/data/<pkg>``:
``FLET_APP_STORAGE_DATA`` (so ``paths.data``) keeps the symlinked spelling while the
process cwd reads back resolved. The check must accept that and still fail when the
cwd really is another directory.
"""

from __future__ import annotations

import os

import pytest

from test_bootstrap import APP_DIR, rb, storage  # noqa: F401  (storage: env + bootstrap teardown fixture)


def _link_dir(real, link) -> None:
    try:
        os.symlink(real, link, target_is_directory=True)
        return
    except (OSError, NotImplementedError) as exc:
        reason = f"cannot create a directory symlink here: {exc}"
    if os.name == "nt":  # no symlink privilege: a junction needs none and resolves the same way
        try:
            import _winapi

            _winapi.CreateJunction(str(real), str(link))
            return
        except (ImportError, AttributeError, OSError) as exc:
            reason += f"; junction failed too: {exc}"
    pytest.skip(reason)


@pytest.fixture
def symlinked_data(storage, tmp_path, monkeypatch):
    real = tmp_path / "real"
    (real / "files" / "data").mkdir(parents=True)
    link = tmp_path / "link"
    _link_dir(real, link)
    monkeypatch.setenv("FLET_APP_STORAGE_DATA", str(link / "files" / "data"))
    paths = rb.bootstrap(app_dir=APP_DIR, force=True)
    assert paths.data == link / "files" / "data"  # bootstrap keeps the platform's spelling
    # getcwd() reads back resolved on Linux/Android; Windows would keep the link, so force it.
    os.chdir(os.path.realpath(paths.data))
    return paths


def _env_contract() -> dict:
    from glossarion_mobile.diagnostics import selftest

    result = selftest.run_selftest("smoke", strict=False, emit=False, write_report=False, only={"env_contract"})
    assert [c["name"] for c in result["checks"]] == ["env_contract"]
    return result["checks"][0]


def test_env_contract_accepts_a_symlinked_data_dir(symlinked_data):
    assert os.path.normcase(os.getcwd()) != os.path.normcase(str(symlinked_data.data))
    check = _env_contract()
    assert check["status"] == "pass", check.get("error")


def test_env_contract_still_rejects_another_cwd(symlinked_data, storage):
    os.chdir(storage["cache"])
    check = _env_contract()
    assert check["status"] == "fail" and "cwd is" in check["error"], check
