"""CI tests for ``.github/workflows/build-mobile.yml`` (U9): no publishing, triggers, device suites.

Pins the owner's decision (2026-10-09: "remove the publishing part of the workflow, we won't be
ever using it like this") and the U9 CI plumbing:

* the mobile app is never published by CI: build-mobile.yml has no release / publish job, no
  ``publish_release`` input, no ``MOBILE_RELEASE_ENABLED`` gate, no release upload (softprops,
  ``gh release``, upload-release actions, the GitHub API's releases), no ``GITHUB_TOKEN`` use
  and no write permission anywhere, so a ``workflow_call`` cannot publish either (a called
  workflow's token never exceeds its own ``contents: read``); the reusable wheels workflow and
  composite action it uses publish nothing; no other workflow uploads a mobile file; the
  release-only tools (``ci/release_assets.py``, ``tools/altstore_source.py``) are gone;
* the APK / AAB / IPA are downloadable run artifacts kept 7 days;
* triggers: only ``workflow_dispatch`` and ``workflow_call``; no push / tag / schedule trigger;
  ``build-all.yml`` and every other workflow leave ``build-mobile.yml`` alone;
* the prepare job computes the version, build number and artifact prefix (run under bash) and
  has no release / tag detection; every ``needs.prepare.outputs.*`` the jobs read exists;
* the optional device suites (iOS simulator smoke, Android UI tests): the prepare step's
  shell is run under bash for each input / policy combination; flipping a policy line to
  ``required`` is a one-line change that makes the suite default-on and blocking (the iOS
  smoke is required since U9, the Android UI tests stay optional); the iOS smoke keeps the
  launch-environment trigger (never ``simctl openurl``); a small GitHub-expression evaluator
  below dry-runs the jobs' ``if:`` / ``continue-on-error``;
* every APK / AAB / IPA name the build jobs write is one the in-app update check recognises;
* ``ci/apk_size_report.py`` and ``tools/build.py`` (dry runs).

Run from src/mobile:
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_ci_release.py
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
REPO = MOBILE_DIR.parents[1]
WORKFLOWS = REPO / ".github" / "workflows"
BUILD_MOBILE = WORKFLOWS / "build-mobile.yml"
MOBILE_WHEELS = WORKFLOWS / "mobile-wheels.yml"
WHEELHOUSE_ACTION = REPO / ".github" / "actions" / "mobile-wheelhouse" / "action.yml"
APP_DIR = MOBILE_DIR / "app"
for path in (MOBILE_DIR / "tools", MOBILE_DIR / "ci", APP_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import apk_size_report  # noqa: E402
from glossarion_mobile.services.updates import classify_asset  # noqa: E402

yaml = pytest.importorskip("yaml")

PREFIX = "Glossarion_v9.15.0"


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _triggers(workflow: dict) -> dict:
    # PyYAML (YAML 1.1) reads the bare key `on` as the boolean True.
    return workflow.get("on", workflow.get(True)) or {}


def _code(path: Path) -> str:
    """The file without its comment lines (comments may say what the workflow does not do)."""
    return "\n".join(line for line in path.read_text(encoding="utf-8").splitlines()
                     if not line.lstrip().startswith("#"))


@pytest.fixture(scope="module")
def wf() -> dict:
    return _load(BUILD_MOBILE)


# ---------------------------------------------------------------------------
# a small evaluator for GitHub Actions expressions (enough for this workflow's conditions)
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"\s*(?:(?P<str>'(?:[^']|'')*')|(?P<num>\d+(?:\.\d+)?)|(?P<op>&&|\|\||==|!=|<=|>=|[!<>(),])"
                    r"|(?P<name>[A-Za-z_][A-Za-z0-9_-]*(?:\.[A-Za-z_*][A-Za-z0-9_-]*)*))")


def _tokens(text: str) -> list:
    out, pos = [], 0
    text = text.strip()
    while pos < len(text):
        match = _TOKEN.match(text, pos)
        if not match or match.end() == pos:
            raise ValueError(f"cannot tokenize {text[pos:pos + 20]!r}")
        pos = match.end()
        kind = match.lastgroup
        out.append((kind, match.group(kind)))
    return out


def _num(value):
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip()) if value.strip() else 0.0
        except ValueError:
            return math.nan
    return math.nan


def _equal(a, b) -> bool:
    if isinstance(a, str) and isinstance(b, str):
        return a.casefold() == b.casefold()
    if type(a) is type(b) and not isinstance(a, (dict, list)):
        return a == b
    if isinstance(a, (dict, list)) or isinstance(b, (dict, list)):
        return a is b
    return _num(a) == _num(b)


def _truthy(value) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, (int, float)):
        return value != 0 and not math.isnan(value)
    if isinstance(value, str):
        return value != ""
    return True


class _Expr:
    """GitHub expression semantics: case-insensitive string ==, number coercion, && / || return
    operands, missing properties are null."""

    def __init__(self, text: str, ctx: dict, status: str = "success") -> None:
        text = text.strip()
        if text.startswith("${{") and text.endswith("}}"):
            text = text[3:-2]
        self.toks = _tokens(text)
        self.i = 0
        self.ctx = ctx
        self.status = status

    def value(self):
        result = self._or()
        if self.i != len(self.toks):
            raise ValueError(f"trailing tokens {self.toks[self.i:]}")
        return result

    def _peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else (None, None)

    def _take(self, op=None):
        tok = self._peek()
        if op is not None and tok[1] != op:
            raise ValueError(f"expected {op!r}, got {tok}")
        self.i += 1
        return tok

    def _or(self):
        left = self._and()
        while self._peek()[1] == "||":
            self._take()
            right = self._and()
            left = left if _truthy(left) else right
        return left

    def _and(self):
        left = self._cmp()
        while self._peek()[1] == "&&":
            self._take()
            right = self._cmp()
            left = right if _truthy(left) else left
        return left

    def _cmp(self):
        left = self._unary()
        while self._peek()[1] in ("==", "!=", "<", "<=", ">", ">="):
            op = self._take()[1]
            right = self._unary()
            if op == "==":
                left = _equal(left, right)
            elif op == "!=":
                left = not _equal(left, right)
            else:
                a, b = _num(left), _num(right)
                left = {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b}[op]
        return left

    def _unary(self):
        if self._peek()[1] == "!":
            self._take()
            return not _truthy(self._unary())
        return self._primary()

    def _primary(self):
        kind, text = self._take()
        if text == "(":
            value = self._or()
            self._take(")")
            return value
        if kind == "str":
            return text[1:-1].replace("''", "'")
        if kind == "num":
            return float(text)
        if kind == "name":
            if text in ("true", "false"):
                return text == "true"
            if text == "null":
                return None
            if self._peek()[1] == "(":
                self._take("(")
                args = []
                while self._peek()[1] != ")":
                    args.append(self._or())
                    if self._peek()[1] == ",":
                        self._take(",")
                self._take(")")
                return self._call(text, args)
            value = self.ctx
            for part in text.split("."):
                value = value.get(part) if isinstance(value, dict) else None
            return value
        raise ValueError(f"unexpected token {text!r}")

    def _call(self, name, args):
        lowered = name.lower()
        if lowered == "cancelled":
            return self.status == "cancelled"
        if lowered == "always":
            return True
        if lowered == "success":
            return self.status == "success"
        if lowered == "failure":
            return self.status == "failure"
        if lowered == "startswith":
            return str(args[0] or "").casefold().startswith(str(args[1] or "").casefold())
        raise ValueError(f"unknown function {name}")


def gh_eval(text: str, ctx: dict, status: str = "success"):
    return _Expr(str(text), ctx, status).value()


def test_expression_evaluator_semantics():
    ctx = {"inputs": {"flag": True, "name": "On"}, "needs": {"a": {"result": "success", "outputs": {"x": "true"}}}}
    assert gh_eval("inputs.flag == true", ctx) is True
    assert gh_eval("inputs.missing == true", ctx) is False
    assert gh_eval("inputs.name == 'on'", ctx) is True  # case-insensitive
    assert gh_eval("needs.a.outputs.x != 'true'", ctx) is False
    assert gh_eval("!cancelled() && (needs.a.result == 'success' || needs.b.result == 'skipped')", ctx) is True
    assert gh_eval("vars.UNSET == 'true'", ctx) is False
    assert gh_eval("${{ inputs.flag && 'yes' || 'no' }}", ctx) == "yes"


# ---------------------------------------------------------------------------
# no publishing (owner's decision, 2026-10-09)
# ---------------------------------------------------------------------------

BUILD_JOBS = {"prepare", "wheels-android", "wheels-ios", "android", "android-smoke", "android-ui-tests", "ios",
              "ios-signed", "ios-simulator-smoke"}
DISPATCH_INPUTS = {"targets", "emulator_smoke", "ios_simulator_smoke", "ui_tests", "android_legacy_packaging"}
CALL_INPUTS = {"targets", "emulator_smoke", "ios_simulator_smoke", "ui_tests"}
SIGNING_SECRETS = {"ANDROID_KEYSTORE_BASE64", "ANDROID_KEYSTORE_PASSWORD", "ANDROID_KEY_ALIAS", "ANDROID_KEY_PASSWORD",
                   "IOS_DIST_CERT_P12_BASE64", "IOS_DIST_CERT_PASSWORD", "IOS_PROVISIONING_PROFILE_BASE64",
                   "IOS_TEAM_ID"}

# Anything that creates a release, uploads to one, pushes a tag or talks to the releases API.
PUBLISH_PATTERNS = (
    r"action-gh-release", r"create-release", r"upload-release", r"release-action", r"automatic-releases",
    r"gh-action-pypi-publish", r"\bgh\s+(?:release|api)\b", r"/releases\b", r"\bgit\s+(?:push|tag)\b",
    r"GITHUB_TOKEN", r"github\.token", r"\bGH_TOKEN\b",
    # store / tester distribution (TestFlight, App Store Connect, Google Play, Firebase)
    r"\baltool\b", r"\bfastlane\b", r"testflight", r"\btransporter\b", r"upload-google-play", r"androidpublisher",
    r"appdistribution",
)
# What only the removed release job had.
RELEASE_JOB_LEFTOVERS = (r"publish_release", r"MOBILE_RELEASE_ENABLED", r"is_release", r"altstore", r"SHA256SUMS",
                         r"release_assets", r"github-release-", r"make_latest", r"overwrite_files")


def _uses(workflow: dict) -> list:
    out = []
    for job in (workflow.get("jobs") or {}).values():
        out.append(str(job.get("uses", "")))
        out += [str(step.get("uses", "")) for step in job.get("steps", []) or []]
    return [u for u in out if u]


def _permission_values(node, found=None) -> list:
    """Every value under any ``permissions`` key, anywhere in a loaded workflow."""
    found = [] if found is None else found
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "permissions":
                if isinstance(value, dict):
                    found += [str(v) for v in value.values()]
                else:
                    found.append(str(value))  # read-all / write-all
            else:
                _permission_values(value, found)
    elif isinstance(node, list):
        for item in node:
            _permission_values(item, found)
    return found


RELEASE_JOB = "update-release"


def _job_text(job_id: str) -> str:
    """The raw YAML block of one job (from its ``  <id>:`` line to the next job)."""
    text = BUILD_MOBILE.read_text(encoding="utf-8").replace("\r\n", "\n")
    start = text.index(f"\n  {job_id}:\n")
    nxt = re.search(r"\n  [A-Za-z0-9_-]+:\n", text[start + 1:])
    return text[start:start + 1 + nxt.start()] if nxt else text[start:]


def test_only_update_release_touches_a_release_and_never_creates_one(wf):
    """Owner (U13): like build-macos.yml, a v* tag run attaches the APK/AAB/IPA to the release that already
    exists for that tag. Only the update-release job may do it; nothing creates a release, pushes a tag
    or distributes to a store."""
    jobs = wf["jobs"]
    assert set(jobs) == BUILD_JOBS | {RELEASE_JOB}  # a new job is a deliberate change here
    for job_id, job in jobs.items():
        assert job.get("environment") is None, job_id
        if job_id == RELEASE_JOB:
            continue
        label = f"{job_id} {job.get('name', '')}".lower()
        assert "release" not in label and "publish" not in label and "deploy" not in label, job_id
    release = jobs[RELEASE_JOB]
    assert release["if"].replace(" ", "").startswith("${{always()&&startsWith(github.ref,'refs/tags/')")
    block = _job_text(RELEASE_JOB)
    assert 'gh release view "$TAG"' in block and "--clobber" in block and "gh release upload" in block
    code = _code(BUILD_MOBILE)
    assert not re.search(r"\bgh\s+release\s+(create|edit|delete)\b", code)
    for pattern in (r"action-gh-release", r"create-release", r"\bgit\s+(?:push|tag)\b", r"\baltool\b",
                    r"\bfastlane\b", r"testflight", r"upload-google-play", r"androidpublisher", r"appdistribution"):
        assert not re.search(pattern, code, re.I), pattern
    whole = BUILD_MOBILE.read_text(encoding="utf-8").replace("\r\n", "\n")
    outside = _code_of(whole.replace(block, "\n"))
    for pattern in PUBLISH_PATTERNS + RELEASE_JOB_LEFTOVERS:
        assert not re.search(pattern, outside, re.I), pattern  # only the update-release job uses a token / release
    header = BUILD_MOBILE.read_text(encoding="utf-8").split("\nname:", 1)[0]
    assert "# This workflow never creates a release." in header and "publish_release" not in header


def _code_of(block: str) -> str:
    return "\n".join(line.split("#", 1)[0].rstrip() for line in block.splitlines() if line.split("#", 1)[0].strip())


def test_no_publish_release_input(wf):
    triggers = _triggers(wf)
    dispatch = triggers["workflow_dispatch"]["inputs"]
    call = triggers["workflow_call"] or {}
    assert set(dispatch) == DISPATCH_INPUTS
    assert set(call.get("inputs") or {}) == CALL_INPUTS
    for name in set(dispatch) | set(call.get("inputs") or {}):
        assert "publish" not in name and "release" not in name, name
    text = BUILD_MOBILE.read_text(encoding="utf-8")
    assert "publish_release" not in text and "inputs.publish" not in text
    # repository variables: only the iOS signing knobs
    assert set(re.findall(r"\bvars\.([A-Za-z0-9_]+)", text)) == {"IOS_EXPORT_METHOD", "IOS_SIGNING_CERTIFICATE"}


def test_permissions_are_read_only_except_the_release_update(wf):
    assert wf["permissions"] == {"contents": "read"}
    for job_id, job in wf["jobs"].items():
        if job_id == RELEASE_JOB:
            assert job["permissions"] == {"contents": "write"}
        else:
            assert "permissions" not in job, job_id
    assert sorted(_permission_values(wf)) == ["read", "write"]
    assert "write-all" not in _code(BUILD_MOBILE)


def test_workflow_call_cannot_publish(wf):
    """A caller gets the same build jobs: the called workflow's token never exceeds its own
    ``contents: read``, it accepts only the signing secrets, and nothing it calls publishes."""
    call = _triggers(wf)["workflow_call"]
    assert set(call.get("inputs") or {}) == CALL_INPUTS
    assert set(call.get("secrets") or {}) == SIGNING_SECRETS
    assert all(not (spec or {}).get("required") for spec in (call.get("secrets") or {}).values())
    assert not (call.get("outputs") or {})  # nothing is handed back to a caller to upload
    assert wf["permissions"] == {"contents": "read"}
    secrets_used = set(re.findall(r"\bsecrets\.([A-Za-z0-9_]+)", BUILD_MOBILE.read_text(encoding="utf-8")))
    assert secrets_used <= SIGNING_SECRETS, secrets_used - SIGNING_SECRETS
    # the reusable wheels workflow: plain build inputs, no secrets, read-only, never fresh by default
    for job_id in ("wheels-android", "wheels-ios"):
        job = wf["jobs"][job_id]
        assert job["uses"] == "./.github/workflows/mobile-wheels.yml"
        assert set(job["with"]) == {"platform"} and "secrets" not in job, job_id
    wheels = _load(MOBILE_WHEELS)
    assert wheels["permissions"] == {"contents": "read"} and set(_permission_values(wheels)) == {"read"}
    assert _triggers(wheels)["workflow_call"]["inputs"]["fresh"]["default"] is False
    assert not (_triggers(wheels)["workflow_call"].get("secrets") or {})
    # neither it nor the composite action the build jobs use publishes anything
    for path in (MOBILE_WHEELS, WHEELHOUSE_ACTION):
        code = _code(path)
        for pattern in PUBLISH_PATTERNS:
            assert not re.search(pattern, code, re.I), (path.name, pattern)
    for uses in _uses(wheels) + [str(s.get("uses", "")) for s in _load(WHEELHOUSE_ACTION)["runs"].get("steps", [])]:
        assert "release" not in uses.lower() and "publish" not in uses.lower(), uses


def test_release_only_tools_are_gone():
    assert not (MOBILE_DIR / "ci" / "release_assets.py").exists()
    assert not (MOBILE_DIR / "tools" / "altstore_source.py").exists()
    for script in sorted((MOBILE_DIR / "ci").glob("*.sh")):
        text = script.read_text(encoding="utf-8")
        for pattern in PUBLISH_PATTERNS:
            assert not re.search(pattern, text, re.I), (script.name, pattern)


# ---------------------------------------------------------------------------
# triggers, other workflows, retention
# ---------------------------------------------------------------------------

def test_build_mobile_triggers_manual_call_and_version_tags(wf):
    triggers = _triggers(wf)
    assert set(triggers) == {"workflow_dispatch", "workflow_call", "push"}
    assert triggers["push"] == {"tags": ["v*"]}  # like build-macos.yml; no branch pushes
    text = BUILD_MOBILE.read_text(encoding="utf-8")
    assert len(re.findall(r"^on:\s*$", text, re.M)) == 1
    on_block = text.split("\non:", 1)[1].split("\npermissions:", 1)[0]
    assert not re.search(r"^  (pull_request|pull_request_target|schedule|release|create|repository_dispatch)\b",
                         on_block, re.M)


def test_no_other_workflow_calls_build_mobile_or_ships_mobile_files():
    mobile_file = re.compile(r"\.(?:apk|aab|ipa)\b|android|ios|mobile|altstore", re.I)
    for path in WORKFLOWS.glob("*.y*ml"):
        if path.name == BUILD_MOBILE.name:
            continue
        workflow = _load(path)
        assert not any("build-mobile" in uses for uses in _uses(workflow)), path.name
        code = _code(path)
        assert "MOBILE_RELEASE_ENABLED" not in code and "build-mobile.yml" not in code, path.name
        # nor names a mobile build artifact or file (e.g. a cross-run download + `gh release upload`)
        assert not re.search(r"Z_Glossarion-(?:Android|iOS)|\.(?:apk|aab|ipa)\b", code, re.I), path.name
        # the desktop tag workflows' release uploads carry desktop files only
        for job in (workflow.get("jobs") or {}).values():
            for step in job.get("steps", []) or []:
                if "release" in str(step.get("uses", "")).lower():
                    files = str((step.get("with") or {}).get("files", ""))
                    assert files and not mobile_file.search(files), (path.name, files)
    build_all = _load(WORKFLOWS / "build-all.yml")
    assert set(_triggers(build_all)) == {"workflow_dispatch"}
    assert not any("mobile" in str(job.get("uses", "")) for job in build_all["jobs"].values())


def test_device_artifacts_keep_seven_days(wf):
    seen = {}
    for job in wf["jobs"].values():
        for step in job.get("steps", []):
            if str(step.get("uses", "")).startswith("actions/upload-artifact"):
                name = step["with"]["name"]
                if str(name).startswith("Z_Glossarion-"):
                    seen[name] = step["with"].get("retention-days")
    assert set(seen) == {"Z_Glossarion-Android-APK", "Z_Glossarion-Android-AAB", "Z_Glossarion-iOS-Unsigned",
                         "Z_Glossarion-iOS-Signed", "Z_Glossarion-iOS-Simulator"}
    assert set(seen.values()) == {7}


def test_signed_builds_still_upload_their_artifacts(wf):
    """Signing stays: the keystore makes release-signed APKs + the AAB, the Apple secrets a signed IPA."""
    android = wf["jobs"]["android"]
    names = [step.get("name", "") for step in android["steps"]]
    assert "Configure release signing" in names and "Build AAB" in names and "Upload AAB" in names
    ios_signed = wf["jobs"]["ios-signed"]
    assert ios_signed["if"] == "needs.prepare.outputs.run_ios == 'true' && needs.prepare.outputs.has_ios_signing == 'true'"
    upload = next(step for step in ios_signed["steps"] if step.get("name") == "Upload signed IPA")
    assert upload["with"]["name"] == "Z_Glossarion-iOS-Signed" and upload["with"]["path"] == "dist/*.ipa"
    ctx = {"needs": {"prepare": {"outputs": {"run_ios": "true", "has_ios_signing": "true"}}}}
    assert gh_eval(ios_signed["if"], ctx) is True
    ctx["needs"]["prepare"]["outputs"]["has_ios_signing"] = "false"
    assert gh_eval(ios_signed["if"], ctx) is False


# ---------------------------------------------------------------------------
# the prepare job under bash: run options and build metadata
# ---------------------------------------------------------------------------

def _bash():
    if os.name == "nt":
        for candidate in (r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files (x86)\Git\bin\bash.exe"):
            if os.path.isfile(candidate):
                return candidate
        return None
    return shutil.which("bash")


def _step(workflow: dict, step_id: str) -> dict:
    return next(s for s in workflow["jobs"]["prepare"]["steps"] if s.get("id") == step_id)


def _run_step(tmp_path, workflow: dict, step_id: str, env_extra: dict) -> dict:
    """Run a prepare step's shell like the runner does; its $GITHUB_OUTPUT lines as a dict
    (``__summary__``: the job summary; ``__rc__`` / ``__err__`` on failure)."""
    bash = _bash()
    if bash is None:
        pytest.skip("bash not available")
    script = tmp_path / f"{step_id}.sh"
    script.write_bytes(_step(workflow, step_id)["run"].encode("utf-8"))
    out = tmp_path / f"{step_id}_output.txt"
    out.write_text("", encoding="utf-8")
    summary = tmp_path / f"{step_id}_summary.md"
    summary.write_text("", encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("IN_", "GITHUB_", "VI_"))}
    env.update({k: str(v) for k, v in workflow.get("env", {}).items()})
    env.update(env_extra)
    env["GITHUB_OUTPUT"] = out.as_posix()
    env["GITHUB_STEP_SUMMARY"] = summary.as_posix()
    proc = subprocess.run([bash, "--noprofile", "--norc", "-eo", "pipefail", script.as_posix()], env=env,
                          capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        return {"__rc__": proc.returncode, "__err__": proc.stdout + proc.stderr}
    result = dict(line.split("=", 1) for line in out.read_text(encoding="utf-8").splitlines() if "=" in line)
    result["__summary__"] = summary.read_text(encoding="utf-8")
    return result


def _run_opts(tmp_path, workflow: dict, **inputs) -> dict:
    env = {f"IN_{k.upper()}": ("true" if v is True else "false" if v is False else str(v)) for k, v in inputs.items()}
    result = _run_step(tmp_path, workflow, "opts", env)
    result.pop("__summary__", None)
    return result


def test_build_metadata_without_release_detection(tmp_path, wf):
    prepare = wf["jobs"]["prepare"]
    outputs = prepare["outputs"]
    assert {"build_version", "build_number", "artifact_prefix"} <= set(outputs)
    assert not any("release" in key or key == "tag" for key in outputs), sorted(outputs)
    # every prepare output a job reads exists
    read = set(re.findall(r"needs\.prepare\.outputs\.([A-Za-z0-9_]+)", BUILD_MOBILE.read_text(encoding="utf-8")))
    assert read and read <= set(outputs), read - set(outputs)
    meta = _step(wf, "meta")
    assert meta["name"] == "Build metadata"
    assert set(meta["env"]) == {"VI_BUILD_VERSION", "VI_BUILD_NUMBER", "VI_ARTIFACT_PREFIX"}
    # a tag run and a branch run produce the same three values, and nothing about a release
    for ref in ("refs/tags/v9.15.0", "refs/heads/main", "refs/tags/v1.0.0"):
        env = {"VI_BUILD_VERSION": "9.15.0", "VI_BUILD_NUMBER": "9150000", "VI_ARTIFACT_PREFIX": "",
               "GITHUB_REF": ref, "GITHUB_REF_NAME": ref.rsplit("/", 1)[-1]}
        result = _run_step(tmp_path, wf, "meta", env)
        summary = result.pop("__summary__")
        assert result == {"build_version": "9.15.0", "build_number": "9150000", "artifact_prefix": PREFIX}, ref
        assert "Build number" in summary and "Release" not in summary and "Tag" not in summary
    given = _run_step(tmp_path, wf, "meta", {"VI_BUILD_VERSION": "9.15.0", "VI_BUILD_NUMBER": "9150001",
                                             "VI_ARTIFACT_PREFIX": "Glossarion_v9.15.0"})
    assert given["build_number"] == "9150001" and given["artifact_prefix"] == PREFIX
    assert _run_step(tmp_path, wf, "meta", {"VI_BUILD_VERSION": "9.15.0", "VI_BUILD_NUMBER": "9.15"}).get("__rc__")
    assert _run_step(tmp_path, wf, "meta", {"VI_BUILD_VERSION": "", "VI_BUILD_NUMBER": "9150000"}).get("__rc__")


# ---------------------------------------------------------------------------
# optional device suites: the prepare step under bash
# ---------------------------------------------------------------------------

def test_suite_policies_today(wf):
    # U9: the iOS simulator smoke passed in Build Mobile run 37728461336 (96da1ec6) and is required; the
    # Android UI tests stay optional until `flet test android` has passed once in CI.
    assert wf["env"]["IOS_SIMULATOR_SMOKE_POLICY"] == "required"
    assert wf["env"]["ANDROID_UI_TESTS_POLICY"] == "optional"
    inputs = _triggers(wf)["workflow_dispatch"]["inputs"]
    for name in ("ios_simulator_smoke", "ui_tests"):
        assert inputs[name]["type"] == "choice" and inputs[name]["default"] == "auto"
        assert inputs[name]["options"] == ["auto", "on", "off"]


def test_suites_resolve_per_input(tmp_path, wf):
    base = _run_opts(tmp_path, wf)  # every input absent (a call without inputs)
    assert base["targets"] == "all" and base["emulator_smoke"] == "true"
    # the required iOS smoke runs by default and blocks; the optional UI tests stay off
    assert (base["ios_simulator_smoke"], base["ios_simulator_smoke_blocking"]) == ("true", "true")
    assert (base["ui_tests"], base["ui_tests_blocking"]) == ("false", "false")
    assert base["android_legacy_packaging"] == "false"
    on = _run_opts(tmp_path, wf, targets="all", ios_simulator_smoke="on", ui_tests="on", android_legacy_packaging=True)
    assert (on["ios_simulator_smoke"], on["ios_simulator_smoke_blocking"]) == ("true", "true")
    assert (on["ui_tests"], on["ui_tests_blocking"]) == ("true", "false")
    assert on["android_legacy_packaging"] == "true"
    off = _run_opts(tmp_path, wf, ios_simulator_smoke="off")  # one run may still skip a required suite
    assert (off["ios_simulator_smoke"], off["ios_simulator_smoke_blocking"]) == ("false", "false")
    # a suite never runs when its platform is not built
    android_only = _run_opts(tmp_path, wf, targets="android", ios_simulator_smoke="on", ui_tests="on")
    assert android_only["ios_simulator_smoke"] == "false" and android_only["ui_tests"] == "true"
    ios_only = _run_opts(tmp_path, wf, targets="ios", ios_simulator_smoke="on", ui_tests="on")
    assert ios_only["ios_simulator_smoke"] == "true" and ios_only["ui_tests"] == "false"
    assert ios_only["emulator_smoke"] == "false"
    # legacy boolean values from a caller still work
    legacy = _run_opts(tmp_path, wf, ios_simulator_smoke=True, ui_tests=False)
    assert legacy["ios_simulator_smoke"] == "true" and legacy["ui_tests"] == "false"
    bad = _run_opts(tmp_path, wf, ios_simulator_smoke="maybe")
    assert bad.get("__rc__") and "auto, on or off" in bad["__err__"]
    bad_targets = _run_opts(tmp_path, wf, targets="web")
    assert bad_targets.get("__rc__")


def _flip(text: str, name: str, to: str = "required") -> str:
    line = f"  {name}: {'optional' if to == 'required' else 'required'}"
    assert text.count(line) == 1, name
    return text.replace(line, f"  {name}: {to}")


@pytest.mark.parametrize("policy,suite,platform_input", [
    ("IOS_SIMULATOR_SMOKE_POLICY", "ios_simulator_smoke", "ios"),
    ("ANDROID_UI_TESTS_POLICY", "ui_tests", "android"),
])
def test_promoting_a_suite_is_one_line(tmp_path, wf, policy, suite, platform_input):
    text = BUILD_MOBILE.read_text(encoding="utf-8")
    # the iOS smoke is already promoted: its optional form is the one-line demotion of today's file
    optional_text = text if wf["env"][policy] == "optional" else _flip(text, policy, "optional")
    optional = _run_opts(tmp_path, yaml.safe_load(optional_text))
    assert (optional[suite], optional[f"{suite}_blocking"]) == ("false", "false")  # off unless asked
    flipped_text = _flip(optional_text, policy)
    changed = [(a, b) for a, b in zip(optional_text.splitlines(), flipped_text.splitlines()) if a != b]
    assert len(changed) == 1
    assert (flipped_text == text) is (wf["env"][policy] == "required")
    flipped = yaml.safe_load(flipped_text)
    out = _run_opts(tmp_path, flipped)
    assert (out[suite], out[f"{suite}_blocking"]) == ("true", "true")  # default-on and blocking
    assert _run_opts(tmp_path, flipped, **{suite: "off"})[suite] == "false"
    other = "android" if platform_input == "ios" else "ios"
    assert _run_opts(tmp_path, flipped, targets=other)[suite] == "false"
    # the job then fails the run (continue-on-error false)
    job = flipped["jobs"]["ios-simulator-smoke" if suite == "ios_simulator_smoke" else "android-ui-tests"]
    ctx = {"needs": {"prepare": {"outputs": {f"{suite}_blocking": out[f"{suite}_blocking"]}}}}
    assert gh_eval(job["continue-on-error"], ctx) is False
    assert gh_eval(job["continue-on-error"], {"needs": {"prepare": {"outputs": {f"{suite}_blocking": "false"}}}}) is True
    assert gh_eval(job["if"], {"needs": {"prepare": {"outputs": {suite: out[suite]}}}}) is True


def test_ios_simulator_smoke_keeps_the_launch_env_trigger(wf):
    job = wf["jobs"]["ios-simulator-smoke"]
    runs = "\n".join(str(step.get("run", "")) for step in job["steps"])
    assert "SIMCTL_CHILD_GLOSSARION_CI_SELFTEST" in runs and "simctl launch" in runs
    assert "simctl openurl" not in runs
    assert job["env"]["SELFTEST_SUITE"] == "smoke" and "GLOSSARION_SELFTEST_START" in runs
    assert int(job["env"]["SELFTEST_START_TIMEOUT"]) == 120


def test_android_ui_tests_job_runs_flet_test_on_the_emulator(wf):
    job = wf["jobs"]["android-ui-tests"]
    assert set(job["needs"]) == {"prepare", "wheels-android"}
    script = next(step["with"]["script"] for step in job["steps"]
                  if str(step.get("uses", "")).startswith("reactivecircus/android-emulator-runner")
                  and "android_ui_tests.sh" in str(step.get("with", {}).get("script", "")))
    assert "\n" not in script.strip()  # the runner runs each line with sh -c
    uses = [str(step.get("uses", "")) for step in job["steps"]]
    assert "./.github/actions/mobile-wheelhouse" in uses
    shell = (MOBILE_DIR / "ci" / "android_ui_tests.sh").read_bytes()
    assert b"\r\n" not in shell  # .gitattributes keeps *.sh LF
    text = shell.decode("utf-8")
    assert "set -euo pipefail" in text and "flet test android" in text and "--junitxml" in text
    assert (MOBILE_DIR / "tests" / "conftest.py").is_file()


# ---------------------------------------------------------------------------
# artifact names and the in-app update check
# ---------------------------------------------------------------------------

def test_workflow_asset_names_are_known_to_the_update_check(wf):
    """The names the build jobs write (Collect APKs, Collect AAB, IPA packaging) classify as
    mobile files in services/updates.py; nothing else the workflow writes does."""
    text = BUILD_MOBILE.read_text(encoding="utf-8")
    templates = set(re.findall(r'"?(?:dist/)?\$\{ARTIFACT_PREFIX\}(_[A-Za-z0-9_.${}-]+)"?', text))
    rendered = set()
    for template in templates:
        for abi in ("arm64-v8a", "x86_64"):
            for suffix in ("", "_debugsigned"):
                rendered.add(PREFIX + template.replace("${abi}", abi).replace("${suffix}", suffix).rstrip('"'))
    apks = {n for n in rendered if n.endswith(".apk")}
    assert apks and {classify_asset(n) for n in apks} == {"apk"}
    assert classify_asset(f"{PREFIX}_iOS_unsigned.ipa") == "ipa" and f"{PREFIX}_iOS_unsigned.ipa" in rendered
    assert f"{PREFIX}_iOS.ipa" in rendered and f"{PREFIX}_Android.aab" in rendered
    for name in rendered:
        if name.endswith((".apk", ".aab", ".ipa")):
            assert classify_asset(name) is not None, name
        else:
            assert classify_asset(name) is None, name  # e.g. the simulator zip
    assert classify_asset("altstore-source.json") is None and classify_asset(f"{PREFIX}_mobile_SHA256SUMS.txt") is None


# ---------------------------------------------------------------------------
# APK size report and the build driver
# ---------------------------------------------------------------------------

def _apk(path: Path) -> Path:
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w", zipfile.ZIP_STORED) as z:
        z.writestr("numpy/__init__.pyc", b"n" * 5000)
        z.writestr("openai/types/x.pyc", b"o" * 9000)
        z.writestr("openai-2.0.dist-info/METADATA", b"m" * 10)
    with zipfile.ZipFile(path, "w") as apk:
        apk.writestr(zipfile.ZipInfo("lib/x86_64/libpython3.13.so"), b"\x7fELF" + b"\0" * 200000)
        apk.writestr(zipfile.ZipInfo("lib/x86_64/libcv2.so"), bytes(range(256)) * 400)
        apk.writestr(zipfile.ZipInfo("assets/flutter_assets/app/sitepackages.zip"), inner.getvalue())
        apk.writestr("classes.dex", b"d" * 1000, compress_type=zipfile.ZIP_DEFLATED)
        apk.writestr("res/x.xml", b"r" * 100, compress_type=zipfile.ZIP_DEFLATED)
    return path


def test_apk_size_report(tmp_path):
    apk = _apk(tmp_path / f"{PREFIX}_Android_x86_64.apk")
    report = apk_size_report.analyze_apk(apk)
    cats = report["categories"]
    assert cats["native libraries (lib/<abi>/*.so)"]["files"] == 2
    assert cats["embedded Python zips (assets/**.zip)"]["files"] == 1
    assert report["native"][0]["name"] == "lib/x86_64/libpython3.13.so" and not report["native"][0]["compressed"]
    assert report["native_total_deflated"] < report["native_total_in_apk"]
    assert report["legacy_packaging_estimate"] < report["size"]
    assert [p["name"] for p in report["packages"]] == ["openai", "numpy", "*.dist-info"]
    summary = tmp_path / "summary.md"
    out = tmp_path / "r.json"
    assert apk_size_report.main([str(apk), "--json", str(out), "--summary", str(summary)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))[0]["apk"] == apk.name
    assert "--android-legacy-packaging" in summary.read_text(encoding="utf-8")
    (tmp_path / "broken.apk").write_bytes(b"nope")
    assert apk_size_report.main([str(tmp_path / "broken.apk"), "--no-legacy-estimate"]) == 1


def _build_dry(*args) -> str:
    proc = subprocess.run([sys.executable, str(MOBILE_DIR / "tools" / "build.py"), *args, "--dry-run",
                           "--flet", "flet"], capture_output=True, text=True, timeout=120,
                          env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_build_driver_dry_runs():
    apk = _build_dry("apk")
    flet_line = next(line for line in apk.splitlines() if line.startswith("$ flet build apk"))
    assert "--split-per-abi" in flet_line and "--build-number" in flet_line
    assert "--android-legacy-packaging" not in flet_line
    legacy = _build_dry("apk", "--legacy-packaging")
    assert "--android-legacy-packaging" in next(l for l in legacy.splitlines() if l.startswith("$ flet build apk"))
    ui = _build_dry("ui-tests", "--device-id", "emulator-5554")
    line = next(l for l in ui.splitlines() if " test android" in l)
    assert "--device-id emulator-5554" in line
    size = _build_dry("size")
    assert "apk_size_report.py" in size


def test_pyproject_cleanup_keeps_runtime_files():
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover
        pytest.skip("tomllib")
    data = tomllib.loads((MOBILE_DIR / "pyproject.toml").read_text(encoding="utf-8"))
    globs = data["tool"]["flet"]["cleanup"]["package_files"]

    def matches(path: str) -> bool:  # serious_python's glob semantics: ** spans directories
        for glob in globs:
            pattern = re.escape(glob).replace(r"\*\*", "\0").replace(r"\*", "[^/]*").replace("\0", ".*")
            if re.fullmatch(pattern, path):
                return True
        return False

    # dev-only trees (not imported at run time by Glossarion, onnxruntime, numpy or pymupdf) go
    for path in ("onnxruntime/transformers/models/bert/fusion.pyc", "numpy/f2py/__init__.pyc",
                 "pymupdf/mupdf-devel/include/mupdf/fitz.h", "cv2/data/haarcascade_eye.xml", "numpy/_core/tests/x.pyc"):
        assert matches(path), path
    # the runtime parts the app and its packages import stay
    for path in ("cv2/__init__.pyc", "cv2/cv2.abi3.so", "cv2/typing/__init__.pyc", "onnxruntime/capi/_pybind_state.pyc",
                 "onnxruntime/quantization/__init__.pyc", "onnxruntime/transformers/onnx_utils.pyc",
                 "onnxruntime/tools/symbolic_shape_infer.pyc", "numpy/typing/__init__.pyc", "numpy/_core/multiarray.pyc",
                 "rapidocr_onnxruntime/models/ch_PP-OCRv3_det_infer.onnx", "rapidocr_onnxruntime/config.yaml",
                 "tiktoken_ext/openai_public.pyc", "certifi/cacert.pem", "langdetect/profiles/ko",
                 "pymupdf/__init__.pyc", "pymupdf/_mupdf.so", "openai-2.32.0.dist-info/METADATA"):
        assert not matches(path), path
    assert data["tool"]["flet"]["android"]["target_arch"] == ["arm64-v8a", "x86_64"]
    assert "legacy_packaging" not in data["tool"]["flet"]["android"]  # evaluated per run (dispatch input)
