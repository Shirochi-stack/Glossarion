"""Offline release receiver/deduplication and real disposable Git checkout tests."""
import base64
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('release_event', ROOT / '.gitlab/release_event.py')
event = importlib.util.module_from_spec(spec)
spec.loader.exec_module(event)


class API:
    def __init__(self, sha):
        self.sha = sha
        self.version = '9.14.5'
        self.pipelines = []
        self.triggers = []
        self.calls = []

    def get(self, path):
        self.calls.append(path)
        if path.startswith('/releases/'):
            return {'tag_name': 'v9.14.5', 'commit': {'id': self.sha}}
        if path.startswith('/repository/tags/'):
            return {'commit': {'id': self.sha}}
        if path.startswith('/repository/files/'):
            return {'content': base64.b64encode(f'APP_VERSION = {self.version!r}\n'.encode()).decode()}
        if path.startswith('/pipelines?'):
            query = parse_qs(urlsplit(path).query)
            return [p for p in self.pipelines if p['name'] == query['name'][0]]
        if path.startswith('/pipelines/'):
            return next(p for p in self.pipelines if str(p['id']) == path.rsplit('/', 1)[-1])
        raise AssertionError(path)

    def trigger(self, inputs):
        self.triggers.append(inputs)
        result = {'id': 123, 'source': 'pipeline', 'ref': 'main', 'name': f"Glossarion release {inputs['release_tag']} {inputs['release_source_sha']}"}
        self.pipelines.append(result)
        return result


class ReceiverTests(unittest.TestCase):
    def setUp(self):
        self.sha = 'a' * 40
        self.env = dict(CI_PROJECT_ID='87623444', CI_SERVER_URL='https://gitlab.com', CI_API_V4_URL='https://gitlab.com/api/v4',
                        CI_JOB_TOKEN='offline-job-token', CI_PIPELINE_SOURCE='trigger', CI_COMMIT_BRANCH='main', GLOSSARION_TASK='release-event')
        self.payload = {'object_kind': 'release', 'action': 'create', 'tag': 'v9.14.5', 'commit': {'id': self.sha},
                        'project': {'id': 87623444, 'path_with_namespace': 'shirochi-stack/Glossarion'}}
        self.api = API(self.sha)

    def test_one_build_for_existing_tag_and_duplicate_deliveries(self):
        result = event.dispatch(self.env, self.payload, self.api)
        self.assertEqual(result['action'], 'dispatched-release-build')
        self.assertEqual(self.api.triggers, [{'release_dispatch': True, 'release_tag': 'v9.14.5', 'release_source_sha': self.sha}])
        again = event.dispatch(self.env, self.payload, self.api)
        self.assertEqual(again['action'], 'reuse-existing-build')
        self.assertEqual(again['pipeline_id'], result['pipeline_id'])
        self.assertEqual(len(self.api.triggers), 1)

    def test_unknown_post_outcome_receiver_retry_discovers_pipeline(self):
        original = self.api.trigger
        def uncertain(inputs):
            original(inputs)
            raise TimeoutError('offline timeout')
        with patch.object(self.api, 'trigger', side_effect=uncertain), self.assertRaises(TimeoutError):
            event.dispatch(self.env, self.payload, self.api)
        self.assertEqual(event.dispatch(self.env, self.payload, self.api)['action'], 'reuse-existing-build')
        self.assertEqual(len(self.api.triggers), 1)

    def test_failed_build_is_not_automatically_rebuilt(self):
        event.dispatch(self.env, self.payload, self.api)
        self.api.pipelines[0]['status'] = 'failed'
        self.assertEqual(event.dispatch(self.env, self.payload, self.api)['action'], 'reuse-existing-build')
        self.assertEqual(len(self.api.triggers), 1)

    def test_trigger_response_without_name_is_verified_from_pipeline_details(self):
        original = self.api.trigger
        def nameless(inputs):
            result = dict(original(inputs)); result.pop('name')
            return result
        with patch.object(self.api, 'trigger', side_effect=nameless):
            self.assertEqual(event.dispatch(self.env, self.payload, self.api)['action'], 'dispatched-release-build')
        self.assertIn('/pipelines/123', self.api.calls)

    def test_release_notes_links_and_deletion_events_never_dispatch(self):
        for action in ('update', 'delete'):
            payload = {**self.payload, 'action': action}
            self.assertEqual(event.dispatch(self.env, payload, self.api)['action'], 'ignored-release-edit-or-delete')
        self.assertEqual(self.api.calls, [])
        self.assertEqual(self.api.triggers, [])

    def test_wrong_event_project_source_and_unsafe_tag_fail_closed(self):
        variants = [dict(self.payload, object_kind='pipeline'), dict(self.payload, action='unknown'),
                    dict(self.payload, project={'id': 1}), dict(self.payload, tag='../../main'),
                    dict(self.payload, object_attributes={'action': 'delete'}), dict(self.payload, commit={'id': 'bad'})]
        for payload in variants:
            with self.subTest(payload=payload), self.assertRaises(event.EventError):
                event.dispatch(self.env, payload, self.api)
        for key, value in (('CI_PROJECT_ID', '1'), ('CI_PIPELINE_SOURCE', 'push'), ('CI_COMMIT_BRANCH', 'other'), ('CI_JOB_TOKEN', '')):
            with self.subTest(key=key), self.assertRaises(event.EventError):
                event.dispatch({**self.env, key: value}, self.payload, self.api)
        self.assertEqual(self.api.triggers, [])

    def test_release_source_or_app_version_mismatch_prevents_build(self):
        for key, value in (('sha', 'b' * 40), ('version', '9.14.4')):
            self.api = API(self.sha)
            setattr(self.api, key, value)
            with self.subTest(key=key), self.assertRaises(event.EventError):
                event.dispatch(self.env, self.payload, self.api)
            self.assertEqual(self.api.triggers, [])

    def test_job_token_only_goes_to_same_project_trigger_body(self):
        api = event.GitLab(self.env)
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self): return b'{}'
        with patch.object(api.opener, 'open', return_value=Response()) as opened:
            api.get('/releases/v9.14.5')
            request = opened.call_args.args[0]
            self.assertFalse(request.headers)
            api.trigger({'release_dispatch': True})
            request = opened.call_args.args[0]
            self.assertEqual(request.full_url, event.API + '/trigger/pipeline')
            self.assertNotIn(self.env['CI_JOB_TOKEN'], request.full_url)
            self.assertEqual(json.loads(request.data)['token'], self.env['CI_JOB_TOKEN'])
            self.assertEqual(json.loads(request.data)['ref'], 'main')
        with self.assertRaises(event.EventError):
            event.NoRedirect().redirect_request(None, None, 302, 'redirect', {}, 'https://example.invalid')


class SourceCheckoutTests(unittest.TestCase):
    def git(self, root, *args):
        result = subprocess.run(['git', '-c', 'core.autocrlf=false', *args], cwd=root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout.strip()

    def test_historical_tag_builds_exact_app_source_retains_only_new_ci_helpers_and_incoming_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            original = base / 'origin'; original.mkdir()
            self.git(original, 'init', '-b', 'main')
            (original / 'src').mkdir(); (original / '.gitlab').mkdir()
            (original / '.github/workflows').mkdir(parents=True)
            (original / 'src/app_version.py').write_text("APP_VERSION = '9.14.5'\nVERSION_TAG = 'v' + APP_VERSION\n")
            (original / 'src/app.py').write_text('tagged app source')
            (original / '.gitlab/helper.py').write_text('old CI helper')
            (original / '.github/workflows/build-mobile.yml').write_text('original tag build steps')
            self.git(original, 'add', '.')
            self.git(original, '-c', 'user.name=Offline Test', '-c', 'user.email=test@example.invalid', 'commit', '--no-gpg-sign', '-m', 'tagged source')
            sha = self.git(original, 'rev-parse', 'HEAD')
            self.git(original, 'tag', 'v9.14.5')
            (original / 'src/app.py').write_text('later main app source must not be packaged')
            (original / '.gitlab/helper.py').write_text('current CI helper')
            (original / '.github/workflows/build-mobile.yml').write_text('later main build steps must not be used')
            self.git(original, 'add', '.')
            self.git(original, '-c', 'user.name=Offline Test', '-c', 'user.email=test@example.invalid', 'commit', '--no-gpg-sign', '-m', 'newer main')
            automation = self.git(original, 'rev-parse', 'HEAD')
            checkout = base / 'checkout'
            self.git(base, 'clone', str(original), str(checkout))
            artifact = checkout / 'artifacts/mobile-check/prepared.json'
            artifact.parent.mkdir(parents=True); artifact.write_text(json.dumps({'source_commit': sha, 'validated': True}))
            env = dict(CI='true', CI_PROJECT_ID='87623444', CI_SERVER_URL='https://gitlab.com', CI_API_V4_URL='https://gitlab.com/api/v4',
                       CI_JOB_TOKEN='offline-job-token', CI_PIPELINE_SOURCE='pipeline', GLOSSARION_RELEASE_EVENT_VALIDATED='true',
                       GLOSSARION_TASK='release-builds', GLOSSARION_RELEASE_TAG='v9.14.5', GLOSSARION_RELEASE_SOURCE_SHA=sha)
            result = event.checkout(env, checkout, API(sha))
            self.assertEqual(result, {'source_commit': sha, 'automation_commit': automation, 'tag': 'v9.14.5'})
            self.assertEqual(self.git(checkout, 'rev-parse', 'HEAD'), sha)
            self.assertEqual((checkout / 'src/app.py').read_text(), 'tagged app source')
            self.assertEqual((checkout / '.github/workflows/build-mobile.yml').read_text(), 'original tag build steps')
            self.assertEqual((checkout / '.gitlab/helper.py').read_text(), 'current CI helper')
            self.assertEqual(json.loads(artifact.read_text())['source_commit'], sha)
            self.assertEqual(self.git(original, 'rev-parse', 'HEAD'), automation)


if __name__ == '__main__':
    unittest.main(verbosity=2)
