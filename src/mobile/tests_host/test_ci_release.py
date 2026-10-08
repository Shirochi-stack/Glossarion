"""CI and release tests for ``.github/workflows/build-mobile.yml`` (U9) and its release tools.

Pins the owner's no-release rule (2026-10-06) and the U9 CI plumbing:

* triggers: only ``workflow_dispatch`` and ``workflow_call``; no push / tag / schedule trigger;
  ``publish_release`` is a dispatch-only input; ``build-all.yml`` and every other workflow
  leave ``build-mobile.yml`` alone; APK / AAB / IPA artifacts keep 7-day retention;
* the release job's ``if:`` is evaluated (a small GitHub-expression evaluator below) for
  push, tag, workflow_call and dispatch contexts: it is reachable only from a manual run on a
  tag with ``publish_release`` ticked and ``vars.MOBILE_RELEASE_ENABLED == 'true'``, after the
  builds (and every *required* optional suite) passed;
* the optional device suites (iOS simulator smoke, Android UI tests): the prepare step's
  shell is run under bash for each input / policy combination; flipping a policy line to
  ``required`` is a one-line change that makes the suite default-on and blocking (the iOS
  smoke is required since U9, the Android UI tests stay optional); the iOS smoke keeps the
  launch-environment trigger (never ``simctl openurl``);
* ``tools/altstore_source.py`` writes what the old inline step wrote (a frozen copy of that
  heredoc is run on the same IPA) and points at the unsigned IPA asset;
* ``ci/release_assets.py``: complete mobile-only sets, sha256sum-format checksums, and the
  never-clobber guard against desktop assets of the same release;
* every asset name the workflow writes is one the in-app update check recognises;
* ``ci/apk_size_report.py`` and ``tools/build.py`` (dry runs).

Run from src/mobile:
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_ci_release.py
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import plistlib
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
APP_DIR = MOBILE_DIR / "app"
for path in (MOBILE_DIR / "tools", MOBILE_DIR / "ci", APP_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import altstore_source  # noqa: E402
import apk_size_report  # noqa: E402
import release_assets  # noqa: E402
from glossarion_mobile.services.updates import classify_asset  # noqa: E402

yaml = pytest.importorskip("yaml")


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _triggers(workflow: dict) -> dict:
    # PyYAML (YAML 1.1) reads the bare key `on` as the boolean True.
    return workflow.get("on", workflow.get(True)) or {}


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
# triggers, inputs, other workflows, retention
# ---------------------------------------------------------------------------

def test_build_mobile_has_only_manual_and_call_triggers(wf):
    triggers = _triggers(wf)
    assert set(triggers) == {"workflow_dispatch", "workflow_call"}
    assert "tags" not in json.dumps(triggers) and "branches" not in json.dumps(triggers)
    # the raw `on:` block too (a second `on` key would be merged away by the YAML loader)
    text = BUILD_MOBILE.read_text(encoding="utf-8")
    assert len(re.findall(r"^on:\s*$", text, re.M)) == 1
    on_block = text.split("\non:", 1)[1].split("\npermissions:", 1)[0]
    assert not re.search(r"^  (push|pull_request|pull_request_target|schedule|release|create|repository_dispatch)\b",
                         on_block, re.M)


def test_publish_release_is_dispatch_only_and_off_by_default(wf):
    triggers = _triggers(wf)
    dispatch = triggers["workflow_dispatch"]["inputs"]
    assert dispatch["publish_release"]["type"] == "boolean" and dispatch["publish_release"]["default"] is False
    call_inputs = (triggers["workflow_call"] or {}).get("inputs") or {}
    assert "publish_release" not in call_inputs
    # The wheels jobs only rebuild fresh for a publish run; nothing else reads the input.
    text = BUILD_MOBILE.read_text(encoding="utf-8")
    uses = re.findall(r"inputs\.publish_release[^\n]*", text)
    assert all("== true" in line for line in uses), uses


def _uses(workflow: dict) -> list:
    out = []
    for job in (workflow.get("jobs") or {}).values():
        out.append(str(job.get("uses", "")))
        out += [str(step.get("uses", "")) for step in job.get("steps", []) or []]
    return out


def test_no_other_workflow_calls_build_mobile():
    for path in WORKFLOWS.glob("*.y*ml"):
        if path.name == BUILD_MOBILE.name:
            continue
        workflow = _load(path)
        assert not any("build-mobile" in uses for uses in _uses(workflow)), path.name
        code = "\n".join(line for line in path.read_text(encoding="utf-8").splitlines()
                         if not line.lstrip().startswith("#"))
        assert "MOBILE_RELEASE_ENABLED" not in code and "build-mobile.yml" not in code, path.name
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


# ---------------------------------------------------------------------------
# release job reachability (dry run of its `if:`)
# ---------------------------------------------------------------------------

def _release_ctx(**over) -> dict:
    ctx = {
        "github": {"event_name": "workflow_dispatch", "ref": "refs/tags/v9.15.0"},
        "inputs": {"publish_release": True, "targets": "all"},
        "vars": {"MOBILE_RELEASE_ENABLED": "true"},
        "needs": {
            "prepare": {"result": "success", "outputs": {"is_release": "true", "ios_simulator_smoke_blocking": "false",
                                                          "ui_tests_blocking": "false"}},
            "android": {"result": "success"}, "android-smoke": {"result": "success"},
            "android-ui-tests": {"result": "skipped"}, "ios": {"result": "success"},
            "ios-signed": {"result": "skipped"}, "ios-simulator-smoke": {"result": "skipped"},
        },
    }
    for path, value in over.items():
        node = ctx
        parts = path.split("__")
        for part in parts[:-1]:
            node = node.setdefault(part.replace("_dash_", "-"), {})
        key = parts[-1].replace("_dash_", "-")
        if value is _DEL:
            node.pop(key, None)
        else:
            node[key] = value
    return ctx


_DEL = object()


def test_release_job_is_reachable_only_from_a_gated_manual_run(wf):
    release = wf["jobs"]["release"]
    cond = release["if"]
    assert set(release["needs"]) == {"prepare", "android", "android-smoke", "android-ui-tests", "ios", "ios-signed",
                                     "ios-simulator-smoke"}
    assert release["permissions"] == {"contents": "write"}
    assert gh_eval(cond, _release_ctx()) is True  # the one way in

    blocked = {
        "tag push": dict(github__event_name="push", inputs={}),
        "branch push": dict(github__event_name="push", github__ref="refs/heads/main", inputs={}),
        # workflow_call: the caller's event, and no publish_release input exists for a call
        "called from a tag push": dict(github__event_name="push", inputs={"targets": "all"}),
        "called from a manual run": dict(inputs={"targets": "all", "emulator_smoke": True}),
        "publish_release unticked": dict(inputs__publish_release=False),
        "variable unset": dict(vars={}),
        "variable not 'true'": dict(vars__MOBILE_RELEASE_ENABLED="1"),
        "not a tag": dict(needs__prepare__outputs__is_release="false"),
        "prepare failed": dict(needs__prepare__result="failure"),
        "android failed": dict(needs__android__result="failure"),
        "ios failed": dict(needs__ios__result="failure"),
        "emulator smoke failed": dict(needs__android_dash_smoke__result="failure"),
        "signed IPA failed": dict(needs__ios_dash_signed__result="failure"),
        "pull request": dict(github__event_name="pull_request"),
        "schedule": dict(github__event_name="schedule"),
    }
    for label, over in blocked.items():
        assert gh_eval(cond, _release_ctx(**over)) is False, label
    assert gh_eval(cond, _release_ctx(), status="cancelled") is False

    # optional suites never block; a required one must pass
    assert gh_eval(cond, _release_ctx(needs__ios_dash_simulator_dash_smoke__result="failure")) is True
    assert gh_eval(cond, _release_ctx(needs__android_dash_ui_dash_tests__result="failure")) is True
    required_ios = dict(needs__prepare__outputs__ios_simulator_smoke_blocking="true")
    assert gh_eval(cond, _release_ctx(**required_ios, needs__ios_dash_simulator_dash_smoke__result="failure")) is False
    assert gh_eval(cond, _release_ctx(**required_ios, needs__ios_dash_simulator_dash_smoke__result="success")) is True
    required_ui = dict(needs__prepare__outputs__ui_tests_blocking="true")
    assert gh_eval(cond, _release_ctx(**required_ui, needs__android_dash_ui_dash_tests__result="failure")) is False
    assert gh_eval(cond, _release_ctx(**required_ui, needs__android_dash_ui_dash_tests__result="success")) is True


def test_release_steps_use_the_tools_and_append_only(wf):
    steps = wf["jobs"]["release"]["steps"]
    runs = "\n".join(str(step.get("run", "")) for step in steps)
    assert "src/mobile/tools/altstore_source.py" in runs and "python3 - <<" not in runs
    assert "release_assets.py checksums " in runs and "release_assets.py check " in runs
    assert runs.index("release_assets.py checksums ") < runs.index("release_assets.py check ")
    upload = next(step for step in steps if str(step.get("uses", "")).startswith("softprops/action-gh-release@"))
    assert upload["uses"] == "softprops/action-gh-release@v3"
    assert upload["with"]["files"] == "dist/*" and upload["with"]["fail_on_unmatched_files"] is True
    assert upload["with"]["make_latest"] == "legacy" and "draft" not in upload["with"]
    assert "body" not in upload["with"] and "name" not in upload["with"]
    concurrency = wf["jobs"]["release"]["concurrency"]
    assert concurrency["group"] == "github-release-${{ github.ref }}" and concurrency["cancel-in-progress"] is False
    # the tools run before the upload
    names = [step.get("name", "") for step in steps]
    assert names.index("Check the asset set and never clobber desktop assets") < names.index(
        "Upload to Release (create or append)")


# ---------------------------------------------------------------------------
# optional device suites: the prepare step under bash
# ---------------------------------------------------------------------------

def _bash():
    if os.name == "nt":
        for candidate in (r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files (x86)\Git\bin\bash.exe"):
            if os.path.isfile(candidate):
                return candidate
        return None
    return shutil.which("bash")


def _opts_script(workflow: dict) -> str:
    step = next(s for s in workflow["jobs"]["prepare"]["steps"] if s.get("id") == "opts")
    return step["run"]


def _run_opts(tmp_path, workflow: dict, **inputs) -> dict:
    bash = _bash()
    if bash is None:
        pytest.skip("bash not available")
    script = tmp_path / "opts.sh"
    script.write_bytes(_opts_script(workflow).encode("utf-8"))
    out = tmp_path / "github_output.txt"
    out.write_text("", encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("IN_", "GITHUB_"))}
    env.update({k: str(v) for k, v in workflow.get("env", {}).items()})
    env.update({f"IN_{k.upper()}": ("true" if v is True else "false" if v is False else str(v))
                for k, v in inputs.items()})
    env["GITHUB_OUTPUT"] = out.as_posix()
    proc = subprocess.run([bash, "--noprofile", "--norc", "-eo", "pipefail", script.as_posix()], env=env,
                          capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        return {"__rc__": proc.returncode, "__err__": proc.stdout + proc.stderr}
    return dict(line.split("=", 1) for line in out.read_text(encoding="utf-8").splitlines() if "=" in line)


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
    # the job then fails the run (continue-on-error false) and the release needs it
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
# AltStore source
# ---------------------------------------------------------------------------

# The inline writer the release job ran before U9 (build-mobile.yml at 1cd68178), frozen.
LEGACY_ALTSTORE_WRITER = r'''
import datetime, json, os, pathlib, plistlib, zipfile

dist = pathlib.Path("dist")
prefix = os.environ["ARTIFACT_PREFIX"]
tag = os.environ["RELEASE_TAG"]
repo = os.environ["GITHUB_REPOSITORY"]
ipa = dist / f"{prefix}_iOS_unsigned.ipa"
with zipfile.ZipFile(ipa) as archive:
    plist_name = next(
        name for name in archive.namelist()
        if name.startswith("Payload/") and name.endswith(".app/Info.plist") and name.count("/") == 2
    )
    info = plistlib.loads(archive.read(plist_name))
releases = f"https://github.com/{repo}/releases"
privacy = {key: value for key, value in info.items()
           if key.startswith("NS") and key.endswith("UsageDescription")}
source = {
    "name": "Glossarion",
    "identifier": f"{info['CFBundleIdentifier']}.source",
    "sourceURL": f"{releases}/latest/download/altstore-source.json",
    "website": f"https://github.com/{repo}",
    "apps": [{
        "name": "Glossarion",
        "bundleIdentifier": info["CFBundleIdentifier"],
        "developerName": "Glossarion contributors",
        "subtitle": "AI novel and manga translator",
        "localizedDescription": "Translate novels, EPUBs, PDFs and manga with your own AI models.",
        "iconURL": f"https://raw.githubusercontent.com/{repo}/{tag}/src/mobile/app/assets/icon.png",
        "tintColor": "E18F98",
        "category": "utilities",
        "appPermissions": {"entitlements": [], "privacy": privacy},
        "versions": [{
            "version": info["CFBundleShortVersionString"],
            "buildVersion": str(info["CFBundleVersion"]),
            "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "localizedDescription": f"Glossarion {tag}",
            "downloadURL": f"{releases}/download/{tag}/{ipa.name}",
            "size": ipa.stat().st_size,
            "minOSVersion": str(info.get("MinimumOSVersion", "13.0")),
        }],
    }],
    "news": [],
}
(dist / "altstore-source.json").write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
'''

PREFIX = "Glossarion_v9.15.0"
INFO = {
    "CFBundleIdentifier": "com.glossarion.app", "CFBundleShortVersionString": "9.15.0", "CFBundleVersion": "9150000",
    "MinimumOSVersion": "13.0", "CFBundleExecutable": "Runner",
    "NSLocalNetworkUsageDescription": "Glossarion connects to model servers on your local network.",
    "NSCameraUsageDescription": "Not used", "UIFileSharingEnabled": True,
}


def _ipa(path: Path, info: dict, *, binary: bool = False) -> Path:
    fmt = plistlib.FMT_BINARY if binary else plistlib.FMT_XML
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("Payload/Runner.app/Info.plist", plistlib.dumps(info, fmt=fmt))
        archive.writestr("Payload/Runner.app/Frameworks/App.framework/Info.plist", plistlib.dumps({"x": 1}))
        archive.writestr("Payload/Runner.app/Runner", b"\0" * 4096)
    return path


def test_altstore_source_matches_the_legacy_inline_writer(tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    ipa = _ipa(dist / f"{PREFIX}_iOS_unsigned.ipa", INFO)
    env = dict(os.environ, ARTIFACT_PREFIX=PREFIX, RELEASE_TAG="v9.15.0", GITHUB_REPOSITORY="owner/repo")
    subprocess.run([sys.executable, "-I", "-c", LEGACY_ALTSTORE_WRITER], cwd=str(tmp_path), env=env, check=True,
                   timeout=60)
    legacy = json.loads((dist / "altstore-source.json").read_text(encoding="utf-8"))
    out = tmp_path / "new.json"
    assert altstore_source.main(["--ipa", str(ipa), "--tag", "v9.15.0", "--repo", "owner/repo", "--out", str(out)]) == 0
    new = json.loads(out.read_text(encoding="utf-8"))
    for doc in (legacy, new):
        date = doc["apps"][0]["versions"][0].pop("date")
        assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", date)
    assert new == legacy
    version = new["apps"][0]["versions"][0]
    assert version["downloadURL"] == f"https://github.com/owner/repo/releases/download/v9.15.0/{PREFIX}_iOS_unsigned.ipa"
    assert version["size"] == ipa.stat().st_size and version["buildVersion"] == "9150000"
    assert new["sourceURL"] == "https://github.com/owner/repo/releases/latest/download/altstore-source.json"
    assert set(new["apps"][0]["appPermissions"]["privacy"]) == {"NSLocalNetworkUsageDescription", "NSCameraUsageDescription"}
    assert out.read_bytes().endswith(b"}\n")


def test_altstore_source_options_and_errors(tmp_path, monkeypatch, capsys):
    ipa = _ipa(tmp_path / f"{PREFIX}_iOS_unsigned.ipa", dict(INFO, MinimumOSVersion=None) | {"MinimumOSVersion": "15.0"},
               binary=True)
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
    assert altstore_source.main(["--ipa", str(ipa), "--tag", "v9.15.0", "--date", "2026-10-09T00:00:00Z",
                                 "--source-url", "https://example.invalid/s.json"]) == 0
    doc = json.loads((tmp_path / "altstore-source.json").read_text(encoding="utf-8"))
    assert doc["sourceURL"] == "https://example.invalid/s.json" and doc["website"] == "https://github.com/o/r"
    assert doc["apps"][0]["versions"][0]["date"] == "2026-10-09T00:00:00Z"
    assert doc["apps"][0]["versions"][0]["minOSVersion"] == "15.0"
    empty = tmp_path / "empty.ipa"
    with zipfile.ZipFile(empty, "w") as archive:
        archive.writestr("Payload/Runner.app/Frameworks/X.framework/Info.plist", b"x")
    assert altstore_source.main(["--ipa", str(empty), "--tag", "v1", "--repo", "o/r"]) == 1
    (tmp_path / "bad.ipa").write_bytes(b"not a zip")
    assert altstore_source.main(["--ipa", str(tmp_path / "bad.ipa"), "--tag", "v1", "--repo", "o/r"]) == 1
    no_version = _ipa(tmp_path / "nov.ipa", {"CFBundleIdentifier": "x"})
    assert altstore_source.main(["--ipa", str(no_version), "--tag", "v1", "--repo", "o/r"]) == 1
    assert "no CFBundleShortVersionString" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# release assets
# ---------------------------------------------------------------------------

def _dist(tmp_path, names) -> Path:
    dist = tmp_path / "dist"
    dist.mkdir(exist_ok=True)
    for name in names:
        (dist / name).write_bytes(name.encode("utf-8") * 3)
    return dist


def test_expected_set_and_problems():
    names = release_assets.expected_names(PREFIX, debug_signed=True, aab=False, signed_ipa=False)
    assert names == {f"{PREFIX}_Android_arm64-v8a_debugsigned.apk", f"{PREFIX}_Android_x86_64_debugsigned.apk",
                     f"{PREFIX}_iOS_unsigned.ipa", "altstore-source.json", f"{PREFIX}_mobile_SHA256SUMS.txt"}
    assert release_assets.set_problems(names, PREFIX) == []
    full = release_assets.expected_names(PREFIX, debug_signed=False, aab=True, signed_ipa=True)
    assert release_assets.set_problems(full, PREFIX) == []
    assert any("missing" in p for p in release_assets.set_problems(names - {"altstore-source.json"}, PREFIX))
    assert any("not a mobile release asset" in p for p in release_assets.set_problems(names | {"notes.txt"}, PREFIX))
    assert any("does not start with" in p
               for p in release_assets.set_problems(names | {"Glossarion_v9.14.0_iOS.ipa"}, PREFIX))
    mixed = names | {f"{PREFIX}_Android_x86_64.apk"}
    assert any("mixed" in p for p in release_assets.set_problems(mixed, PREFIX))
    # every mobile name the release writes is one the app's update check recognises
    assert all(classify_asset(name) for name in full | names)


def test_never_clobber_desktop_assets():
    names = release_assets.expected_names(PREFIX, debug_signed=False, aab=False, signed_ipa=False)
    desktop = {"assets": [{"name": "Glossarion v9.15.0.exe"}, {"name": "Glossarion_v9.15.0_MAC.dmg"},
                          {"name": f"{PREFIX}_Android_arm64-v8a.apk"}]}  # our own asset from an earlier run
    assert release_assets.clobber_problems(names, desktop) == []
    assert release_assets.clobber_problems(names, {}) == [] and release_assets.clobber_problems(names, None) == []
    clash = release_assets.clobber_problems(names | {"Glossarion v9.15.0.exe"}, desktop)
    assert clash and "not a mobile asset" in clash[0]


def test_checksums_and_check_cli(tmp_path, capsys):
    names = sorted(release_assets.expected_names(PREFIX, debug_signed=True, aab=False, signed_ipa=False)
                   - {f"{PREFIX}_mobile_SHA256SUMS.txt"})
    dist = _dist(tmp_path, names)
    assert release_assets.main(["checksums", "--dist", str(dist), "--prefix", PREFIX]) == 0
    sums = (dist / f"{PREFIX}_mobile_SHA256SUMS.txt").read_bytes().decode("utf-8")
    expected = "".join(f"{hashlib.sha256((dist / n).read_bytes()).hexdigest()}  {n}\n" for n in names)
    assert sums == expected  # sha256sum format, LF, the sums file itself excluded
    release = tmp_path / "release.json"
    release.write_text(json.dumps({"assets": [{"name": "Glossarion v9.15.0.exe"}]}), encoding="utf-8")
    assert release_assets.main(["check", "--dist", str(dist), "--prefix", PREFIX, "--release-json", str(release)]) == 0
    release.write_text("{}", encoding="utf-8")
    assert release_assets.main(["check", "--dist", str(dist), "--prefix", PREFIX, "--release-json", str(release)]) == 0
    (dist / "Glossarion v9.15.0.exe").write_bytes(b"x")
    release.write_text(json.dumps({"assets": [{"name": "Glossarion v9.15.0.exe"}]}), encoding="utf-8")
    assert release_assets.main(["check", "--dist", str(dist), "--prefix", PREFIX, "--release-json", str(release)]) == 1
    assert "::error title=Mobile release assets::" in capsys.readouterr().out


def test_workflow_asset_names_are_known_to_the_update_check(wf):
    """The names the build jobs write (Collect APKs, Collect AAB, IPA packaging) classify as
    mobile assets in services/updates.py, and the release's own files too."""
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
