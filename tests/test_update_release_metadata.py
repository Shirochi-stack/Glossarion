"""Consumer metadata round trips and release wiring, without publishing anything."""
import hashlib
import importlib.util
import io
import json
import runpy
from pathlib import Path
import sys
from unittest.mock import Mock
from urllib.error import HTTPError

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from generate_update_metadata import metadata_bytes, main
CURRENT_TAG = runpy.run_path(str(ROOT / 'src/app_version.py'))['VERSION_TAG']
spec = importlib.util.spec_from_file_location('release_metadata_test', ROOT / '.gitlab/release.py')
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def test_generator_hashes_actual_bytes_and_binds_name_size_tag(tmp_path):
    artifact = tmp_path / 'Glossarion fixture.exe'
    artifact.write_bytes(b'final signed executable bytes')
    data = json.loads(metadata_bytes(artifact, CURRENT_TAG))
    assert data == {'schema_version': 1, 'tag': CURRENT_TAG, 'name': artifact.name,
                    'size': artifact.stat().st_size,
                    'sha256': hashlib.sha256(artifact.read_bytes()).hexdigest()}
    main([str(artifact), '--tag', CURRENT_TAG])
    assert json.loads(Path(str(artifact) + '.sha256.json').read_bytes()) == data
    with pytest.raises(ValueError, match='changed'):
        metadata_bytes(artifact, 'v9.14.4', '0' * 64)


def test_generator_rejects_no_files_or_wrong_package_version(tmp_path):
    with pytest.raises(ValueError, match='No artifacts'): main([str(tmp_path / '*.exe')])
    with pytest.raises(ValueError, match='differs'): main([str(tmp_path / '*.exe'), '--tag', 'v0.1.0'])


def test_gitlab_upload_preflights_binary_and_sidecar_before_writes(tmp_path):
    folder = tmp_path / 'artifacts/windows-lite'
    folder.mkdir(parents=True)
    artifact = folder / 'Glossarion.exe'
    artifact.write_bytes(b'fixture executable')
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    (folder / 'manifest.json').write_text(json.dumps({'label': 'windows-lite', 'source_commit': 'a' * 40,
                                                     'artifacts': [{'file': artifact.name, 'sha256': digest}]}))
    env = dict(GLOSSARION_PUBLISH='true', CI_COMMIT_REF_PROTECTED='true', CI_COMMIT_TAG='v9.14.4',
               CI_COMMIT_SHA='a' * 40, CI_JOB_TOKEN='offline-fixture-only',
               CI_API_V4_URL='https://gitlab.com/api/v4', CI_SERVER_URL='https://gitlab.com', CI_PROJECT_ID='87623444')
    calls, uploads, links = [], {}, []
    def open_request(req, **kwargs):
        calls.append(req.method)
        if req.method == 'GET':
            return io.BytesIO(json.dumps({'tag_name': env['CI_COMMIT_TAG'], 'commit': {'id': env['CI_COMMIT_SHA']},
                                         'assets': {'links': []}}).encode())
        if req.method == 'HEAD': raise HTTPError(req.full_url, 404, 'missing', {}, None)
        if req.method == 'PUT': uploads[req.full_url] = req.data.read()
        if req.method == 'POST': links.append(json.loads(req.data))
        return io.BytesIO(b'{}')
    release.publish(env, tmp_path, Mock(open=open_request))
    assert calls == ['GET', 'HEAD', 'HEAD', 'PUT', 'POST', 'PUT', 'POST']
    assert [link['name'] for link in links] == ['Glossarion.exe', 'Glossarion.exe.sha256.json']
    binary, sidecar = list(uploads.values())
    data = json.loads(sidecar)
    assert data['sha256'] == hashlib.sha256(binary).hexdigest()
    assert data['size'] == len(binary) and data['tag'] == env['CI_COMMIT_TAG']
