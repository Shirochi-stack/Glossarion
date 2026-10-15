"""Shared updater contracts retained across the GitHub/GitLab provider change.

The old verbatim-extraction oracle was specific to the U9 refactor. Provider
and integrity behavior now has explicit offline regression coverage alongside
the GUI-free, configuration and platform contracts here.
"""
import ast
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
sys.path.insert(0, str(SRC))
from update_core import HeadlessUpdateChecker, UpdateCoreMixin


def test_core_remains_gui_free_and_python310_compatible():
    ast.parse((SRC / 'update_core.py').read_text(encoding='utf-8-sig'), feature_version=(3, 10))
    probe = "import sys; sys.modules['PySide6']=None; import update_core; assert 'update_manager' not in sys.modules"
    result = subprocess.run([sys.executable, '-c', probe], cwd=SRC, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_cache_skip_and_config_contract(monkeypatch):
    saves = []
    host = SimpleNamespace(config={'last_update_check_time': 123},
                           save_config=lambda **kw: saves.append(kw))
    checker = HeadlessUpdateChecker(host, '9.14.4', build_variant='Mobile')
    assert checker.CURRENT_VERSION == '9.14.4' and checker._build_variant == 'Mobile'
    assert checker._last_check_time == 123 and checker._check_cache_duration == 1800
    checker.latest_release = {'tag_name': 'v9.15.0'}
    assert checker._skip_latest_version() == 'v9.15.0'
    checker._skip_latest_version()
    assert host.config['skipped_versions'] == ['v9.15.0']
    assert saves == [{'show_message': False}] * 2
    monkeypatch.setattr('update_core.time.time', lambda: 500)
    monkeypatch.setattr(checker, '_json', lambda *a: pytest.fail('cached check made HTTP request'))
    assert checker._check_for_updates_core() == (False, None)


@pytest.mark.parametrize('name,platform,arch', [('Glossarion.exe','windows','unknown'),
    ('Glossarion_MAC.dmg','macos','arm64'), ('Glossarion_MAC_Intel.dmg','macos','x64'),
    ('Glossarion_Linux.tar.gz','linux','unknown'), ('Glossarion_Android_arm64-v8a.apk','unknown','arm64'),
    ('Glossarion_Android_x86_64_debugsigned.apk','unknown','x64')])
def test_asset_classification_contract(name, platform, arch):
    assert UpdateCoreMixin._asset_platform(name) == platform
    assert UpdateCoreMixin._asset_arch(name) == arch


@pytest.mark.parametrize('name,size,eligible', [('Glossarion.exe',60_000_000,True),
    ('Glossarion.exe',59_999_999,False), ('other.exe',60_000_000,False),
    ('../Glossarion.exe',60_000_000,False), ('Glossarion.exe','invalid',False)])
def test_requested_name_and_sixty_mb_gates(name, size, eligible):
    assert UpdateCoreMixin._eligible_update_asset({'name': name, 'size': size}) is eligible


def test_sources_keep_uniform_line_endings():
    for name in ('update_core.py', 'update_manager.py'):
        data = (SRC / name).read_bytes()
        assert data.count(b'\r\n') in (0, data.count(b'\n'))
