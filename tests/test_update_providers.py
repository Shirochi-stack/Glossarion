"""Offline provider, checksum and download regressions; never launch an installer."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from update_core import HeadlessUpdateChecker, UpdateCoreMixin, UpdateVerificationError, UpdateCancelled

GH = UpdateCoreMixin.GITHUB_LATEST_URL
GL = UpdateCoreMixin.GITLAB_API_URL
NAME = 'Glossarion v9.15.0.exe'
URL = 'https://github.com/Shirochi-stack/Glossarion/releases/download/v9.15.0/' + NAME
GLURL = GL + '/packages/generic/glossarion/v9.15.0/windows--' + NAME
DATA = b'actual update binary fixture'
SHA = hashlib.sha256(DATA).hexdigest()


class Response:
    def __init__(self, data=b'', status=200, headers=None, error=None):
        self.data = data if isinstance(data, bytes) else json.dumps(data).encode()
        self.status_code = status
        self.headers = headers or {}
        self.error = error
        self.closed = False

    def __enter__(self): return self
    def __exit__(self, *args): self.close()
    def close(self): self.closed = True
    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code), response=self)

    def iter_content(self, chunk_size):
        yield self.data[:10]
        if self.error: raise self.error
        yield self.data[10:]


def release(tag='v9.15.0', digest=True):
    asset = {'name': NAME, 'size': len(DATA), 'browser_download_url': URL}
    if digest: asset['digest'] = 'sha256:' + SHA
    return {'tag_name': tag, 'assets': [asset], 'body': 'Notes\n\n\nmore', 'html_url': 'https://github.com/x/y/releases/tag/' + tag}


def metadata(**updates):
    return dict(schema_version=1, tag='v9.15.0', name=NAME, size=len(DATA), sha256=SHA, **updates)


@pytest.fixture
def checker(monkeypatch):
    host = SimpleNamespace(config={}, save_config=lambda **kwargs: None)
    checker = HeadlessUpdateChecker(host, '9.14.4', build_variant='Standard')
    monkeypatch.setattr(UpdateCoreMixin, 'MIN_UPDATE_SIZE', 8)
    monkeypatch.setattr(checker, '_detect_platform', lambda: 'windows')
    monkeypatch.setattr(checker, '_detect_arch', lambda: 'x64')
    return checker


@pytest.fixture
def http(monkeypatch):
    routes, calls = {}, []
    def get(url, **kwargs):
        calls.append(url)
        assert kwargs['verify'] is True
        assert kwargs['allow_redirects'] is False
        assert kwargs.get('stream') is True
        value = routes.get(url, Response({}, 404))
        if isinstance(value, Exception): raise value
        return value
    monkeypatch.setattr(requests, 'get', get)
    return routes, calls


def gitlab(routes, tag='v9.15.0', name=NAME, data=DATA):
    url = GL + '/releases/' + tag
    routes[url] = Response({'tag_name': tag, 'description': 'GitLab notes', 'released_at': '2026-10-15',
                            'assets': {'sources': [{'url': 'ignore-source.zip'}], 'links': [
                                {'name': name, 'url': GLURL},
                                {'name': name + '.sha256.json', 'url': GLURL + '.sha256.json'}]}})
    routes[GLURL + '.sha256.json'] = Response({'schema_version': 1, 'tag': tag, 'name': name,
                                             'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
    return url


def tags_url(page=1):
    return GL + '/repository/tags?order_by=version&sort=desc&search=%5Ev&per_page=100&page=' + str(page)


def test_github_primary_and_history_cannot_change_selected_release(checker, http):
    routes, calls = http
    routes[GH] = Response(release())
    routes[checker.GITHUB_API_URL + '?per_page=10'] = Response([release('v10.0.0'), release('v9.15.0')])
    available, result = checker._check_for_updates_core()
    assert available and result['tag_name'] == 'v9.15.0'
    assert checker.all_releases[0] == result
    assert not any(url.startswith(GL) for url in calls)
    assert checker._check_for_updates_core() == (False, None)
    assert len(calls) == 2


@pytest.mark.parametrize('error', [requests.Timeout(), requests.ConnectionError(), Response({}, 403),
                                        Response({}, 404), Response({}, 429), Response({}, 503),
                                        Response(b'invalid JSON')])
def test_provider_outage_uses_tag_release_not_source_archive(checker, http, error):
    routes, calls = http
    routes[GH] = error
    routes[tags_url()] = Response([{'name': 'v9.15.0', 'release': {}}, {'name': 'v9.15.0', 'release': {'tag_name': 'v9.15.0'}}])
    gitlab(routes)
    available, result = checker._check_for_updates_core()
    assert available and result['_provider'] == 'gitlab'
    assert result['body'] == 'GitLab notes'
    assert [a['name'] for a in result['assets']] == [NAME, NAME + '.sha256.json']
    assert result['assets'][0]['size'] == len(DATA)


@pytest.mark.parametrize('error', [requests.exceptions.SSLError('certificate failed'), Response({}, 401)])
def test_security_or_auth_error_does_not_fallback(checker, http, error):
    routes, calls = http
    routes[GH] = error
    with pytest.raises(requests.RequestException): checker._check_for_updates_core()
    assert calls == [GH]


def test_numeric_version_stable_order_and_unreleased_tags(checker, http):
    routes, _ = http
    routes[GH] = Response({}, 404)
    values = ['v9.9.0', 'v9.10.0', 'v10.0.0rc1', 'v999.0.0', 'vbroken']
    routes[tags_url()] = Response([{'name': t, 'release': {'tag_name': t} if t != 'v999.0.0' else None} for t in values])
    for t in ('v9.9.0', 'v9.10.0'): gitlab(routes, tag=t)
    checker.CURRENT_VERSION = '9.8.0'
    assert checker._check_for_updates_core()[1]['tag_name'] == 'v9.10.0'


def test_skip_force_and_current_do_not_query_fallback(checker, http):
    routes, calls = http
    routes[GH] = Response(release())
    checker.main_gui.config['skipped_versions'] = ['v9.15.0']
    assert checker._check_for_updates_core() == (False, None)
    assert checker._check_for_updates_core(force_show=True)[0]
    checker.CURRENT_VERSION = '10.0.0'
    assert checker._check_for_updates_core(force_show=True) == (False, None)
    assert not checker.update_available
    assert all(not url.startswith(GL) for url in calls)


def test_missing_platform_build_uses_same_gitlab_tag(checker, http):
    routes, calls = http
    raw = release(); raw['assets'] = []
    routes[GH] = Response(raw)
    gitlab(routes)
    assert checker._check_for_updates_core()[1]['_provider'] == 'gitlab'
    assert tags_url() not in calls


def test_github_native_digest_download_real_hash(checker, http, tmp_path):
    routes, calls = http
    routes[URL] = Response(DATA)
    raw = release()
    path = str(tmp_path / NAME)
    result = checker._download_verified_update(raw['assets'][0], raw, path, lambda *args: None, lambda: False)
    assert Path(path).read_bytes() == DATA and result['_sha256'] == SHA
    assert not Path(path + '.part').exists()


def test_github_sidecar_and_conflicting_native_digest(checker, http):
    routes, _ = http
    raw = release(digest=False)
    raw['assets'].append({'name': NAME + '.sha256.json', 'browser_download_url': URL + '.sha256.json'})
    routes[URL + '.sha256.json'] = Response(metadata())
    assert checker._prepare_update_asset(raw['assets'][0], raw)['_sha256'] == SHA
    raw['assets'][0]['digest'] = 'sha256:' + '0' * 64
    with pytest.raises(UpdateVerificationError, match='conflicts'): checker._prepare_update_asset(raw['assets'][0], raw)


def test_missing_checksum_blocks_before_binary_request(checker, http, tmp_path):
    _, calls = http
    raw = release(digest=False)
    with pytest.raises(UpdateVerificationError, match='no published SHA-256'):
        checker._download_verified_update(raw['assets'][0], raw, str(tmp_path / NAME), lambda *a: None, lambda: False)
    assert calls == []


@pytest.mark.parametrize('updates', [{'tag': 'v0.0.1'}, {'name': 'other.exe'}, {'sha256': 'garbage'},
                                    {'size': 0}, {'size': True}, {'size': len(DATA) + 1}, {'schema_version': 2}])
def test_bad_sidecar_rejected(checker, http, updates):
    routes, _ = http
    raw = release(digest=False)
    raw['assets'].append({'name': NAME + '.sha256.json', 'browser_download_url': URL + '.sha256.json'})
    routes[URL + '.sha256.json'] = Response({**metadata(), **updates})
    with pytest.raises(UpdateVerificationError): checker._prepare_update_asset(raw['assets'][0], raw)


@pytest.mark.parametrize('binary', [b'corrupted fixture same size!', DATA[:-1], DATA + b'extra'])
def test_mismatch_blocks_without_mirror_and_cleans_partial(checker, http, tmp_path, binary):
    routes, calls = http
    routes[URL] = Response(binary)
    raw = release()
    path = str(tmp_path / NAME)
    with pytest.raises(UpdateVerificationError):
        checker._download_verified_update(raw['assets'][0], raw, path, lambda *a: None, lambda: False)
    assert calls == [URL]
    assert not Path(path).exists() and not Path(path + '.part').exists()


def test_download_network_failure_uses_identical_gitlab_file(checker, http, tmp_path):
    routes, calls = http
    routes[URL] = Response(DATA, error=requests.ConnectionError('stream interrupted'))
    gitlab(routes)
    routes[GLURL] = Response(DATA)
    raw = release()
    result = checker._download_verified_update(raw['assets'][0], raw, str(tmp_path / NAME), lambda *a: None, lambda: False)
    assert result['_provider'] == 'gitlab'
    assert GLURL in calls


def test_mirror_changed_hash_is_not_downloaded(checker, http, tmp_path):
    routes, calls = http
    routes[URL] = requests.Timeout()
    gitlab(routes, data=b'a different release file!!')
    raw = release()
    with pytest.raises(UpdateVerificationError, match='differs'):
        checker._download_verified_update(raw['assets'][0], raw, str(tmp_path / NAME), lambda *a: None, lambda: False)
    assert GLURL not in calls


@pytest.mark.parametrize('cancel_at', [0, 1, 2])
def test_cancel_never_installs_or_falls_back(checker, http, tmp_path, cancel_at):
    routes, calls = http
    routes[URL] = Response(DATA)
    raw = release()
    state = {'n': -1}
    def cancelled():
        state['n'] += 1
        return state['n'] >= cancel_at
    path = str(tmp_path / NAME)
    with pytest.raises(UpdateCancelled):
        checker._download_verified_update(raw['assets'][0], raw, path, lambda *a: None, cancelled)
    assert not Path(path).exists() and not Path(path + '.part').exists()
    assert not any(url.startswith(GL) for url in calls)


@pytest.mark.parametrize('name', ['../Glossarion.exe', 'Glossarion%PATH%.exe', 'Glossarion\\evil.exe', 'Glossarion.exe\n', 'other.exe'])
def test_unsafe_name_rejected(checker, http, name):
    raw = release(); raw['assets'][0]['name'] = name
    with pytest.raises(UpdateVerificationError): checker._prepare_update_asset(raw['assets'][0], raw)


@pytest.mark.parametrize('platform,arch,name,compatible', [('windows','x64',NAME,True),
    ('macos','arm64','Glossarion_MAC_Intel.dmg',False), ('macos','arm64','Glossarion_MAC.dmg',True),
    ('windows','x64','Glossarion_MAC.dmg',False), ('linux','x64','Glossarion_Linux.zip',False)])
def test_platform_and_architecture(checker, monkeypatch, platform, arch, name, compatible):
    monkeypatch.setattr(checker, '_detect_platform', lambda: platform)
    monkeypatch.setattr(checker, '_detect_arch', lambda: arch)
    assert checker._compatible_update_asset({'name': name}) is compatible


def test_https_redirect_and_downgrade_block(checker, http):
    routes, calls = http
    routes[URL] = Response(status=302, headers={'Location': 'http://insecure.example/file'})
    with pytest.raises(UpdateVerificationError): checker._get(URL, stream=True)
    assert calls == [URL]


def test_metadata_is_bounded(checker, http):
    routes, _ = http
    routes[URL] = Response(b'x' * 100)
    with pytest.raises(UpdateVerificationError, match='size'): checker._json(URL, limit=50)


def test_mirror_cannot_change_version_even_when_bytes_match(checker, http, tmp_path):
    routes, calls = http
    routes[URL] = requests.Timeout()
    routes[GL + '/releases/v9.15.0'] = Response({'tag_name': 'v9.16.0', 'assets': {'links': []}})
    raw = release()
    with pytest.raises(UpdateVerificationError, match='different release tag'):
        checker._download_verified_update(raw['assets'][0], raw, str(tmp_path / NAME), lambda *a: None, lambda: False)
    assert GLURL not in calls


@pytest.mark.parametrize('current,skipped', [('9.15.0', []), ('9.14.4', ['v9.15.0'])])
def test_current_or_skipped_missing_build_does_not_trigger_fallback(checker, http, current, skipped):
    routes, calls = http
    raw = release(); raw['assets'] = []
    routes[GH] = Response(raw)
    checker.CURRENT_VERSION = current
    checker.main_gui.config['skipped_versions'] = skipped
    assert checker._check_for_updates_core() == (False, None)
    assert not any(url.startswith(GL) for url in calls)


def test_tag_pagination_and_future_release_exclusion(checker, http):
    routes, calls = http
    routes[GH] = Response({}, 404)
    routes[tags_url()] = Response([{'name': 'v20.0.0', 'release': {'tag_name': 'v20.0.0'}}] * 100)
    routes[tags_url(2)] = Response([{'name': 'v9.15.0', 'release': {'tag_name': 'v9.15.0'}}])
    routes[GL + '/releases/v20.0.0'] = Response({'tag_name': 'v20.0.0', 'released_at': '2099-01-01T00:00:00Z', 'assets': {'links': []}})
    gitlab(routes)
    assert checker._check_for_updates_core()[1]['tag_name'] == 'v9.15.0'
    assert tags_url(2) in calls


def test_actual_60mb_threshold_and_rehash_before_install(tmp_path):
    path = tmp_path / 'Glossarion fixture.exe'
    with path.open('wb') as stream:
        stream.seek(60_000_000 - 1); stream.write(b'\0')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''): digest.update(chunk)
    asset = {'size': 60_000_000, '_sha256': digest.hexdigest()}
    assert UpdateCoreMixin._verify_downloaded_update(path, asset) == 60_000_000
    with path.open('r+b') as stream: stream.write(b'X')
    with pytest.raises(UpdateVerificationError, match='mismatch'): UpdateCoreMixin._verify_downloaded_update(path, asset)
    with path.open('r+b') as stream: stream.truncate(59_999_999)
    with pytest.raises(ValueError, match='60 MB'): UpdateCoreMixin._verify_downloaded_update(path, asset)
