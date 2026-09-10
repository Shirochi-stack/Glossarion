import ast
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

import autharena_bridge as bridge


REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / 'assets' / 'autharena_extension'
FILES = ('manifest.json', 'background.js', 'connect.js', 'README.md')


def copy_bundle(path):
    source = path / 'autharena_extension'
    source.mkdir(parents=True)
    for name in FILES:
        shutil.copyfile(ASSETS / name, source / name)
    return source


def snapshot(path):
    return {item.name: item.read_bytes() for item in path.iterdir() if item.is_file()}


@pytest.fixture
def materializer(monkeypatch, tmp_path):
    bundle = tmp_path / '_MEI_first'
    source = copy_bundle(bundle)
    persistent = tmp_path / 'user-home' / '.glossarion'
    monkeypatch.setattr(sys, '_MEIPASS', str(bundle), raising=False)
    monkeypatch.setattr(bridge, '_root', lambda: persistent)
    monkeypatch.setitem(sys.modules, 'autharena', SimpleNamespace(
        _PREPARE_JS='return {payload: __PAYLOAD__, login: __LOGIN_ONLY__};',
    ))
    return source, persistent


def test_onefile_materialization_survives_extraction_cleanup_and_restart(materializer, monkeypatch, tmp_path):
    source, persistent = materializer
    assert bridge._extension_source() == source
    target = bridge.prepare_extension()
    assert target == persistent / 'autharena_extension'
    installed = snapshot(target)
    assert set(installed) == set(FILES) | {'arena_page.js'}
    assert b'config.payload' in installed['arena_page.js']
    assert b'__PAYLOAD__' not in installed['arena_page.js']

    # Simulate PyInstaller removing only its known temporary extraction tree.
    extraction = source.parent.resolve()
    extraction.relative_to(tmp_path.resolve())
    shutil.rmtree(extraction)
    assert snapshot(target) == installed

    second_source = copy_bundle(tmp_path / '_MEI_second')
    monkeypatch.setattr(sys, '_MEIPASS', str(second_source.parent))
    second_source.joinpath('background.js').write_bytes(installed['background.js'] + b'\n// update\n')
    assert bridge.prepare_extension() == target
    assert target.joinpath('background.js').read_bytes().endswith(b'// update\n')
    # An unpacked extension's stable absolute path and manifest are retained;
    # a new extraction directory never becomes its installed identity.
    assert target.joinpath('manifest.json').read_bytes() == installed['manifest.json']
    assert not any(item.suffix == '.tmp' for item in persistent.iterdir())


def test_unchanged_extension_is_not_rewritten(materializer):
    target = bridge.prepare_extension()
    mtimes = {path.name: path.stat().st_mtime_ns for path in target.iterdir()}
    assert bridge.prepare_extension() == target
    assert {path.name: path.stat().st_mtime_ns for path in target.iterdir()} == mtimes


@pytest.mark.parametrize('missing', FILES)
def test_missing_packaged_asset_leaves_installed_extension_untouched(materializer, missing):
    source, persistent = materializer
    target = bridge.prepare_extension()
    installed = snapshot(target)
    source.joinpath(missing).unlink()
    with pytest.raises(ImportError, match=missing.replace('.', r'\.')):
        bridge.prepare_extension()
    assert snapshot(target) == installed
    assert not list(persistent.glob('.arena-extension-*.tmp'))


@pytest.mark.parametrize('manifest', [
    b'{broken json', b'[]',
    json.dumps({'manifest_version': 3, 'background': {'service_worker': 'missing.js'}}).encode(),
    json.dumps({'manifest_version': 3, 'background': {'service_worker': 'background.js'},
                'content_scripts': ['invalid entry']}).encode(),
])
def test_invalid_manifest_fails_before_creating_installation(materializer, manifest):
    source, persistent = materializer
    source.joinpath('manifest.json').write_bytes(manifest)
    with pytest.raises(ImportError, match='manifest is invalid'):
        bridge.prepare_extension()
    assert not persistent.joinpath('autharena_extension').exists()


def test_empty_script_fails_before_creating_installation(materializer):
    source, persistent = materializer
    source.joinpath('connect.js').write_bytes(b'')
    with pytest.raises(ImportError, match='empty connect.js'):
        bridge.prepare_extension()
    assert not persistent.joinpath('autharena_extension').exists()


def test_frozen_missing_assets_do_not_fall_back_to_source_checkout(materializer, monkeypatch, tmp_path):
    _, persistent = materializer
    monkeypatch.setattr(sys, '_MEIPASS', str(tmp_path / 'incomplete-bundle'))
    assert bridge._extension_source() is None
    with pytest.raises(ImportError, match='missing from this installation'):
        bridge.prepare_extension()
    assert not persistent.joinpath('autharena_extension').exists()


def test_staging_failure_does_not_change_installed_files(materializer, monkeypatch):
    source, persistent = materializer
    target = bridge.prepare_extension()
    installed = snapshot(target)
    source.joinpath('background.js').write_bytes(b'// new background')
    source.joinpath('connect.js').write_bytes(b'// new connection')
    original_open = Path.open
    staged = []

    def failing_open(path, mode='r', *args, **kwargs):
        if mode == 'xb':
            staged.append(path)
            if len(staged) == 3:
                raise OSError('simulated disk full')
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', failing_open)
    with pytest.raises(OSError, match='disk full'):
        bridge.prepare_extension()
    assert snapshot(target) == installed
    assert not list(persistent.glob('.arena-extension-*.tmp'))


def test_replace_failure_rolls_back_installed_files(materializer, monkeypatch):
    source, persistent = materializer
    target = bridge.prepare_extension()
    installed = snapshot(target)
    source.joinpath('background.js').write_bytes(b'// new background')
    source.joinpath('connect.js').write_bytes(b'// new connection')
    original_replace = bridge.os.replace

    def failing_replace(src, destination):
        if Path(destination) == target / 'connect.js':
            raise PermissionError('simulated browser file lock')
        return original_replace(src, destination)

    monkeypatch.setattr(bridge.os, 'replace', failing_replace)
    with pytest.raises(PermissionError, match='browser file lock'):
        bridge.prepare_extension()
    assert snapshot(target) == installed
    assert not list(persistent.glob('.arena-extension-*.tmp'))


def test_desktop_specs_include_installer_and_complete_extension_assets():
    specs = sorted(REPO.joinpath('src').glob('translator*.spec'))
    assert len(specs) == 11
    for spec in specs:
        source = spec.read_text(encoding='utf-8')
        ast.parse(source, filename=str(spec))
        for module in ('autharena', 'autharena_browser', 'autharena_bridge', 'autharena_setup', 'streaming_log'):
            assert source.count(f"('{module}.py', '.')") == 1, (spec.name, module)
            assert source.count(f"'{module}',") == 1, (spec.name, module)
        assert "'assets', 'autharena_extension'), 'autharena_extension'" in source
    ignored = subprocess.run(
        ['git', 'check-ignore', '--no-index', 'assets/autharena_extension/manifest.json'],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    assert ignored.returncode == 1, ignored.stdout or ignored.stderr
