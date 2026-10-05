"""Build a ``headless_owner.HeadlessOwner`` inside a scrubbed process sandbox (tests only).

HeadlessOwner replays a desktop start, so building one exports ~40 process-wide
environment variables, resolves ``app_paths.CONFIG_FILE`` and may create folders
relative to the working directory. Tests build it inside ``scrubbed_env`` with
``app_paths`` pointed at a temp dir, so nothing leaks into the test process, the
real config.json or the repository.

    from _headless_env import headless_owner, run_envs

    with headless_owner(tmp_path, monkeypatch, {"model": "gpt-4o"}) as owner:
        ...
    envs = run_envs(tmp_path, monkeypatch, {"enable_unified_glossary": True})
    envs["translation"]["ENABLE_UNIFIED_GLOSSARY"]   # run_env.build_translation_env
    envs["glossary"]["ENABLE_UNIFIED_GLOSSARY"]      # run_env.build_glossary_env().env_updates
    envs["startup"]["ENABLE_UNIFIED_GLOSSARY"]       # os.environ after the startup replay
"""

from __future__ import annotations

import contextlib
import os
import sys
import types
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

#: Variables kept from the real environment (Windows needs them to start processes).
KEEP_ENV = ("SYSTEMROOT", "PATH", "PATHEXT", "COMSPEC", "WINDIR")


class scrubbed_env:
    """Minimal scrubbed environment + cwd for a direct HeadlessOwner build; restored on exit."""

    def __init__(self, root):
        self.root = Path(root)

    def __enter__(self):
        self.saved = dict(os.environ)
        self.cwd = os.getcwd()
        keep = {k: os.environ[k] for k in KEEP_ENV if k in os.environ}
        os.environ.clear()
        os.environ.update(keep)
        os.environ.update({"TEMP": str(self.root), "TMP": str(self.root), "HOME": str(self.root),
                           "USERPROFILE": str(self.root)})
        os.chdir(self.root)
        _clear_large_env()
        return self

    def __exit__(self, *exc):
        os.chdir(self.cwd)
        os.environ.clear()
        os.environ.update(self.saved)
        _clear_large_env()
        return False


def _clear_large_env():
    try:
        import large_env
        large_env.clear_store()
    except Exception:
        pass


def _quiet_host():
    return types.SimpleNamespace(log=lambda _message: None)


@contextlib.contextmanager
def headless_owner(tmp_path, monkeypatch, config=None, **kwargs):
    """Yield ``HeadlessOwner(config, **kwargs)`` built and used inside ``scrubbed_env(tmp_path)``."""
    import app_paths
    from headless_owner import HeadlessOwner

    root = Path(tmp_path)
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(app_paths, "CONFIG_FILE", str(root / "config.json"))
    monkeypatch.setattr(app_paths, "__file__", str(root / "app_paths.py"))
    kwargs.setdefault("host", _quiet_host())
    with scrubbed_env(root):
        yield HeadlessOwner(dict(config or {}), **kwargs)


def run_envs(tmp_path, monkeypatch, config=None, *, source="Book.epub", api_key="sk-test", **kwargs):
    """The env a run of *source* gets from a HeadlessOwner built from *config*.

    Returns ``{'startup', 'translation', 'glossary'}``: os.environ right after the
    owner's startup replay, ``run_env.build_translation_env`` and
    ``run_env.build_glossary_env(...).env_updates``.
    """
    import run_env

    with headless_owner(tmp_path, monkeypatch, config, **kwargs) as owner:
        startup = dict(os.environ)
        path = str(Path(tmp_path) / source)
        translation = dict(run_env.build_translation_env(owner, path, api_key))
        glossary = dict(run_env.build_glossary_env(owner, path, api_key).env_updates)
    return {"startup": startup, "translation": translation, "glossary": glossary}


__all__ = ["KEEP_ENV", "headless_owner", "run_envs", "scrubbed_env"]
