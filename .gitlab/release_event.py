"""Validate release-created webhooks and dispatch one same-project build.

The persistent trigger token only reaches GitLab's main-branch event receiver.
Dispatch uses its ephemeral job token and typed inputs. The receiver job's
resource_group serializes lookup/create, including redeliveries and retries.
"""
from __future__ import annotations

import ast
import base64
import json
import os
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import quote, urlencode
from urllib.request import Request, build_opener, HTTPRedirectHandler
from urllib.error import HTTPError, URLError

API = 'https://gitlab.com/api/v4/projects/87623444'
PROJECT = 'https://gitlab.com/shirochi-stack/Glossarion'
VERSION = r'v[0-9]+(?:\.[0-9]+)+[A-Za-z0-9_.+-]*'
SHA = r'[a-f0-9]{40}'
ROOT = Path(__file__).resolve().parents[1]


class EventError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise EventError(message)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args):
        raise EventError('Release API redirected; refusing to forward credentials.')


class GitLab:
    def __init__(self, env):
        self.token = env.get('CI_JOB_TOKEN', '')
        self.opener = build_opener(NoRedirect())

    def get(self, path):
        # Public project metadata needs no credential. Never send the job token
        # to repository/pipeline metadata endpoints or through redirects.
        with self.opener.open(Request(API + path), timeout=30) as response:
            return json.load(response)

    def trigger(self, inputs):
        body = json.dumps({'token': self.token, 'ref': 'main', 'inputs': inputs}).encode()
        request = Request(API + '/trigger/pipeline', data=body,
                          headers={'Content-Type': 'application/json'}, method='POST')
        with self.opener.open(request, timeout=60) as response:
            return json.load(response)


def project_environment(env):
    require(env.get('CI_PROJECT_ID') == '87623444' and env.get('CI_SERVER_URL') == 'https://gitlab.com'
            and env.get('CI_API_V4_URL') == 'https://gitlab.com/api/v4', 'Unexpected project/server.')
    require(env.get('CI_JOB_TOKEN'), 'Missing running job token.')


def event_source(payload):
    require(payload.get('object_kind') == 'release', 'Only Release webhook payloads are accepted.')
    require(payload.get('project', {}).get('id') == 87623444
            and payload['project'].get('path_with_namespace') == 'shirochi-stack/Glossarion',
            'Release webhook belongs to another project.')
    # Current GitLab payloads use top-level action; accept the documented nested
    # spelling too, but never accept contradictory actions.
    action = payload.get('action', payload.get('object_attributes', {}).get('action'))
    nested = payload.get('object_attributes', {}).get('action', action)
    require(action == nested, 'Contradictory release actions.')
    if action in ('update', 'delete'):
        return None
    require(action == 'create', 'Unknown release action.')
    tag, commit = payload.get('tag', ''), payload.get('commit', {}).get('id', '')
    require(re.fullmatch(VERSION, tag) and re.fullmatch(SHA, commit), 'Invalid release tag/commit.')
    return tag, commit


def verify_release(api, tag, commit):
    current = api.get('/releases/' + quote(tag, safe=''))
    require(current.get('tag_name') == tag and current.get('commit', {}).get('id') == commit,
            'Release changed/deleted or no longer matches the event source.')
    actual = api.get('/repository/tags/' + quote(tag, safe=''))
    require(actual.get('commit', {}).get('id') == commit, 'Release tag moved; refusing this build.')
    source = api.get('/repository/files/src%2Fapp_version.py?ref=' + commit)
    version = None
    for node in ast.parse(base64.b64decode(source['content']).decode()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'APP_VERSION' for t in node.targets):
            version = ast.literal_eval(node.value)
    require(isinstance(version, str) and 'v' + version == tag, 'Release tag does not match app_version.py.')


def dispatch(env, payload, api=None):
    project_environment(env)
    require(env.get('CI_PIPELINE_SOURCE') == 'trigger' and env.get('CI_COMMIT_BRANCH') == 'main'
            and env.get('GLOSSARION_TASK') == 'release-event', 'Release receiver must run on triggered main.')
    source = event_source(payload)
    if source is None:
        return {'action': 'ignored-release-edit-or-delete'}
    tag, commit = source
    api = api or GitLab(env)
    verify_release(api, tag, commit)
    name = f'Glossarion release {tag} {commit}'
    # No automatic retry of POST: after an uncertain response, retry this
    # receiver; this lookup discovers an already-created downstream pipeline.
    pipelines = api.get('/pipelines?' + urlencode({'source': 'pipeline', 'ref': 'main', 'name': name, 'per_page': 100}))
    matches = [item for item in pipelines if item.get('name') == name and item.get('source') == 'pipeline'
               and item.get('ref') == 'main']
    if matches:
        pipeline = min(matches, key=lambda item: item['id'])
        return {'action': 'reuse-existing-build', 'pipeline_id': pipeline['id'],
                'web_url': PROJECT + '/-/pipelines/' + str(pipeline['id']), 'tag': tag, 'source_commit': commit}
    pipeline = api.trigger({'release_dispatch': True, 'release_tag': tag, 'release_source_sha': commit})
    require(pipeline.get('source') == 'pipeline' and pipeline.get('ref') == 'main'
            and not pipeline.get('yaml_errors'),
            'Unexpected downstream pipeline response; inspect before retrying the receiver.')
    if not pipeline.get('name'):
        pipeline = api.get('/pipelines/' + str(int(pipeline['id'])))
    require(pipeline.get('name') == name, 'Unexpected downstream pipeline name; inspect before retrying.')
    return {'action': 'dispatched-release-build', 'pipeline_id': pipeline['id'],
            'web_url': PROJECT + '/-/pipelines/' + str(pipeline['id']), 'tag': tag, 'source_commit': commit}


def checkout(env, root=ROOT, api=None):
    """In disposable Linux/macOS jobs, build exact tagged app source with current CI helpers.

    This also supports releases against historical tags whose old CI configuration
    cannot receive a release event. Only .gitlab automation comes from main;
    application source and original GitHub build steps come from the exact tag.
    """
    project_environment(env)
    automatic = env.get('CI_PIPELINE_SOURCE') == 'pipeline' and env.get('GLOSSARION_TASK') == 'release-builds'
    repair = (env.get('CI_PIPELINE_SOURCE') in ('api', 'web') and env.get('CI_COMMIT_BRANCH') == 'main'
              and env.get('GLOSSARION_TASK') == 'linux-release-repair' and env.get('GLOSSARION_PUBLISH') == 'true'
              and env.get('GLOSSARION_REPAIR_LINUX_VARIANT') in ('standard', 'nocuda', 'turbolite'))
    require(env.get('CI') == 'true' and (automatic or repair)
            and env.get('GLOSSARION_RELEASE_EVENT_VALIDATED') == 'true', 'Source checkout requires a validated CI build.')
    tag, commit = env.get('GLOSSARION_RELEASE_TAG', ''), env.get('GLOSSARION_RELEASE_SOURCE_SHA', '')
    require(re.fullmatch(VERSION, tag) and re.fullmatch(SHA, commit), 'Invalid dispatched source.')
    verify_release(api or GitLab(env), tag, commit)
    root = Path(root).resolve()

    def git(*args):
        result = subprocess.run(['git', *args], cwd=root, capture_output=True, text=True, timeout=180)
        require(result.returncode == 0, 'Release source checkout failed; no build started.')
        return result.stdout.strip()

    git('fetch', '--no-tags', 'origin', 'refs/tags/' + tag)
    require(git('rev-parse', 'FETCH_HEAD^{commit}') == commit, 'Fetched tag changed; no build started.')
    automation = git('rev-parse', 'HEAD')
    with tempfile.TemporaryDirectory(prefix='glossarion-release-ci-') as temporary:
        bridge = Path(temporary) / 'bridge'
        shutil.copytree(root / '.gitlab', bridge, ignore=shutil.ignore_patterns('__pycache__'))
        git('checkout', '--detach', commit)
        require(runpy.run_path(str(root / 'src/app_version.py'))['VERSION_TAG'] == tag,
                'Checked-out application version differs from the release.')
        shutil.copytree(bridge, root / '.gitlab', dirs_exist_ok=True)
    print(json.dumps({'checked_out_source': commit, 'tag': tag, 'automation_commit': automation}))
    return {'source_commit': commit, 'automation_commit': automation, 'tag': tag}


if __name__ == '__main__':
    try:
        if sys.argv[1:] == ['checkout']:
            checkout(dict(os.environ))
        else:
            require(sys.argv[1:] == [], 'Unknown release receiver command.')
            payload_path = Path(os.environ.get('TRIGGER_PAYLOAD', ''))
            require(payload_path.is_file() and payload_path.stat().st_size <= 1024 * 1024,
                    'Missing/oversized webhook payload file.')
            result = dispatch(dict(os.environ), json.loads(payload_path.read_text()))
            Path('artifacts/release-event').mkdir(parents=True, exist_ok=True)
            Path('artifacts/release-event/dispatch.json').write_text(json.dumps(result, indent=2))
            print(json.dumps(result))
    except HTTPError as exc:
        print(f'FAILED: GitLab release event API returned HTTP {exc.code}. No credentials/body logged.', file=sys.stderr)
        raise SystemExit(1)
    except (EventError, OSError, ValueError, KeyError, URLError, subprocess.TimeoutExpired):
        # urllib exceptions can contain URLs. Never print requests, headers,
        # webhook bodies or arbitrary API exception text into CI logs.
        print('FAILED: Release event validation/dispatch failed; inspect source and receiver before retrying.', file=sys.stderr)
        raise SystemExit(1)
