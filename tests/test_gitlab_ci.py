"""Offline checks for the GitLab bridge; no builds, credentials or live HTTP.

Run directly with Python 3.12+ and the pinned .gitlab/requirements.txt dependency.
"""
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import itertools
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location("glossarion_gitlab_" + name, ROOT / ".gitlab" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ci, release = load("ci"), load("release")


class BuildBridgeTests(unittest.TestCase):
    def test_every_mobile_build_is_noninteractive_and_uses_bundled_python_pin(self):
        for target, signed in (("apk", False), ("apk", True), ("aab", True), ("ipa", False), ("ipa", True), ("ios-simulator", False)):
            runner=ci.Runner("test",plan=True)
            with self.subTest(target=target,signed=signed), patch.object(runner,"command",wraps=runner.command) as command, redirect_stdout(io.StringIO()):
                ci.mobile_build(runner,target,signed)
                build=next(call.args[0] for call in command.call_args_list if call.kwargs.get("label","").startswith("Build "))
                self.assertEqual(build.count("--"),1)
                options=build[build.index("--")+1:]
                self.assertIn("--yes",options)
                self.assertIn("--no-rich-output",options)
                self.assertEqual(options[options.index("--python-version")+1],"3.13")
                if signed and target == "ipa": self.assertIn("--ios-team-id",options)

    def test_all_selected_build_plans_resolve_without_execution(self):
        commands = [["desktop", "--platform", platform, "--variant", variant, "--plan"]
                    for platform, variants in (("windows", ci.WINDOWS), ("linux", ci.LINUX),
                                               ("macos-arm", ("lite", "nocuda")), ("macos-intel", ("lite", "nocuda")))
                    for variant in variants]
        commands += [[task, "--plan"] for task in ("desktop-tests", "mobile-backend-check", "mobile-check", "android-ui", "android-smoke")]
        commands += [["wheels", "--platform", platform, "--plan"] for platform in ("android", "ios")]
        commands += [["mobile", "--target", target, "--plan"] + (["--signed"] if signed else [])
                     for target, signed in (("apk", False), ("apk", True), ("aab", True), ("ipa", False), ("ipa", True), ("ios-simulator", False))]
        with patch.object(ci.subprocess, "run", side_effect=AssertionError("Plan executed a process")), \
             patch.object(ci.Path, "mkdir", side_effect=AssertionError("Plan wrote a directory")):
            for command in commands:
                with self.subTest(command=command), redirect_stdout(io.StringIO()):
                    self.assertEqual(ci.main(command), 0)

    def test_unknown_workflow_expression_fails(self):
        with self.assertRaisesRegex(ci.PrerequisiteError, "Unsupported workflow expression"):
            ci.render("${{ secrets.UNDECLARED }}", {})

    def test_removed_or_renamed_step_fails(self):
        with self.assertRaisesRegex(ci.PrerequisiteError, "Expected one shell step"):
            ci.step_definition("build-mobile.yml", "prepare", "Nonexistent required step")

    def test_missing_suite_is_a_failure(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(ci, "ROOT", Path(directory)):
            with self.assertRaisesRegex(ci.PrerequisiteError, "Required tests missing"):
                ci.require_tests("tests")

    def test_missing_oracle_commit_is_a_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tests").mkdir()
            (root / "tests/test_parity.py").write_text("BASE_SHA = '" + "a" * 40 + "'\n")
            with patch.object(ci, "ROOT", root), patch.object(ci, "MOBILE", root / "src/mobile"), \
                 patch.object(ci.subprocess, "run", return_value=Mock(returncode=1)):
                with self.assertRaisesRegex(ci.PrerequisiteError, "Historical test oracles are missing"):
                    ci.check_oracle_history()

    def test_command_failure_propagates_without_printing_arguments(self):
        runner = ci.Runner("test", plan=True)
        runner.plan = False
        with patch.object(ci.subprocess, "run", return_value=Mock(returncode=9)), redirect_stdout(io.StringIO()) as output:
            with self.assertRaisesRegex(ci.PrerequisiteError, "exit 9") as caught:
                runner.command(["tool", "--password", "fixture-private-value"], label="Signing test")
        self.assertNotIn("fixture-private-value", str(caught.exception) + output.getvalue())

    def test_empty_artifact_collection_is_a_failure(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(ci, "ROOT", Path(directory)):
            runner = ci.Runner("build")
            with self.assertRaisesRegex(ci.PrerequisiteError, "no expected artifacts"):
                runner.collect(["dist/*.exe"])

    def test_build_manifest_records_actual_hash_and_commit(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(ci, "ROOT", Path(directory)):
            root = Path(directory)
            (root / "dist").mkdir()
            (root / "dist/app.exe").write_bytes(b"fixture executable")
            runner = ci.Runner("build")
            runner.env["CI_COMMIT_SHA"] = "a" * 40
            runner.collect(["dist/*.exe"])
            manifest = json.loads((runner.report / "manifest.json").read_text())
            self.assertEqual(manifest["source_commit"], "a" * 40)
            self.assertEqual(manifest["artifacts"][0]["sha256"], hashlib.sha256(b"fixture executable").hexdigest())

    def test_signing_needs_both_authorization_and_protected_ref(self):
        for env in ({}, {"CI_COMMIT_REF_PROTECTED": "true"}, {"GLOSSARION_SIGNING_APPROVED": "true"}):
            with self.subTest(env=env), self.assertRaises(ci.PrerequisiteError):
                ci.signing_gate(env, ())
        with self.assertRaisesRegex(ci.PrerequisiteError, "Signing prerequisite missing"):
            ci.signing_gate({"CI_COMMIT_REF_PROTECTED": "true", "GLOSSARION_SIGNING_APPROVED": "true"}, ["FIXTURE_KEY_FILE"])

    def test_prepared_bundle_requires_same_commit_pipeline_and_success(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(ci, "ROOT", Path(directory)):
            marker = Path(directory) / "artifacts/mobile-check/prepared.json"
            marker.parent.mkdir(parents=True)
            runner = ci.Runner("build", plan=True)
            runner.env.update(CI_COMMIT_SHA="a" * 40, CI_PIPELINE_ID="17")
            good = {"source_commit": "a" * 40, "pipeline_id": "17", "validated": True}
            for key, value in (("source_commit", "b" * 40), ("pipeline_id", "18"), ("validated", False)):
                marker.write_text(json.dumps({**good, key: value}))
                with self.subTest(key=key), self.assertRaises(ci.PrerequisiteError):
                    ci.prepared_mobile(runner)
            marker.write_text(json.dumps(good))
            with patch.object(runner, "command") as command:
                self.assertTrue(ci.prepared_mobile(runner))
                self.assertEqual(command.call_args_list[-1].args[0][-1], "--verify")


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        header, cls.config = yaml.safe_load_all((ROOT / ".gitlab-ci.yml").read_text())
        cls.inputs = header["spec"]["inputs"]
        for include in cls.config["include"]:
            cls.config.update(yaml.safe_load((ROOT / include["local"]).read_text()))

    def test_original_automatic_events_and_manual_dispatch_are_admitted(self):
        defaults = dict(CI_PIPELINE_SOURCE="", CI_COMMIT_BRANCH="", CI_COMMIT_TAG="", CI_MERGE_REQUEST_TARGET_BRANCH_NAME="")
        cases = [
            ({"CI_PIPELINE_SOURCE":"push", "CI_COMMIT_BRANCH":"main"}, "ci"),
            ({"CI_PIPELINE_SOURCE":"push", "CI_COMMIT_BRANCH":"feature"}, None),
            ({"CI_PIPELINE_SOURCE":"merge_request_event", "CI_MERGE_REQUEST_TARGET_BRANCH_NAME":"main"}, "ci"),
            ({"CI_PIPELINE_SOURCE":"merge_request_event", "CI_MERGE_REQUEST_TARGET_BRANCH_NAME":"develop"}, None),
            ({"CI_PIPELINE_SOURCE":"push", "CI_COMMIT_TAG":"v9.14.4"}, "tag-builds"),
            ({"CI_PIPELINE_SOURCE":"push", "CI_COMMIT_TAG":"v"}, "tag-builds"),
            ({"CI_PIPELINE_SOURCE":"push", "CI_COMMIT_TAG":"nightly"}, None),
            ({"CI_PIPELINE_SOURCE":"schedule", "CI_COMMIT_BRANCH":"main"}, None),
            ({"CI_PIPELINE_SOURCE":"web"}, "source-check"),
            ({"CI_PIPELINE_SOURCE":"api"}, "source-check"),
        ]
        for values, expected in cases:
            variables={**defaults, **values}
            actual=None
            for rule in self.config["workflow"]["rules"]:
                if rule.get("when") == "never": break
                expression=re.sub(r'\$(\w+) =~ /([^/]+)/', lambda m:str(bool(re.search(m[2],variables[m[1]]))), rule["if"])
                expression=re.sub(r'\$(\w+)',lambda m:repr(variables[m[1]]),expression).replace("&&","and").replace("||","or")
                if eval(expression,{"__builtins__":{}},{}):
                    actual=rule.get("variables",{}).get("GLOSSARION_TASK","source-check")
                    if actual in ("ci","tag-builds"):
                        self.assertEqual(rule["variables"]["GLOSSARION_PUBLISH"],"false")
                        self.assertEqual(rule["variables"]["GLOSSARION_SIGNING_APPROVED"],"false")
                    break
            with self.subTest(event=values): self.assertEqual(actual,expected)

    def test_trigger_mapping_matches_original_workflows_without_new_schedules(self):
        originals={p.name:yaml.safe_load(p.read_text()).get("on",yaml.safe_load(p.read_text()).get(True))
                   for p in (ROOT/".github/workflows").glob("*.yml")}
        self.assertTrue(all("schedule" not in trigger for trigger in originals.values()))
        self.assertEqual(originals["python-app.yml"]["push"],{"branches":["main"]})
        self.assertEqual(originals["python-app.yml"]["pull_request"],{"branches":["main"]})
        for name in ("build-linux.yml","build-macos.yml","build-mobile.yml"):
            self.assertEqual(originals[name]["push"],{"tags":["v*"]})
        for name in ("build-all.yml","build-windows.yml","mobile-wheels.yml"):
            self.assertNotIn("push",originals[name])

    def test_original_mobile_inputs_have_same_defaults_and_choices(self):
        workflow = ci.workflow("build-mobile.yml")
        original = workflow.get("on", workflow.get(True))["workflow_dispatch"]["inputs"]
        for name, value in original.items():
            with self.subTest(input=name):
                self.assertEqual(self.inputs[name]["default"], value["default"])
                if "options" in value:
                    self.assertEqual(self.inputs[name]["options"], value["options"])
                if value["type"] == "boolean":
                    self.assertEqual(self.inputs[name]["type"], "boolean")
        self.assertEqual(workflow["env"]["IOS_SIMULATOR_SMOKE_POLICY"], "required")
        self.assertEqual(workflow["env"]["ANDROID_UI_TESTS_POLICY"], "optional")

    def test_every_selection_has_complete_dependency_graph(self):
        jobs = {name: job for name, job in self.config.items() if isinstance(job, dict) and "script" in job}
        cases = [(task, "all", "true", "auto", "auto") for task in self.inputs["task"]["options"]]
        cases += [(task,"all","true","auto","auto") for task in ("ci","tag-builds")]
        cases += [("mobile", *values) for values in itertools.product(
            ("all", "android", "ios"), ("true", "false"), ("auto", "on", "off"), ("auto", "on", "off"))]
        for task, targets, smoke, simulator, ui in cases:
            variables = dict(GLOSSARION_TASK=task, GLOSSARION_TARGETS=targets,
                             GLOSSARION_EMULATOR_SMOKE=smoke, GLOSSARION_IOS_SIMULATOR_SMOKE=simulator,
                             GLOSSARION_UI_TESTS=ui, GLOSSARION_PUBLISH="false",
                             CI_COMMIT_TAG="", CI_COMMIT_REF_PROTECTED="true")
            selected = set()
            for name, job in jobs.items():
                for rule in job["rules"]:
                    expression = rule.get("if", "")
                    if not expression:
                        continue
                    expression = re.sub(r'\$(\w+) =~ /([^/]+)/',
                        lambda m: str(bool(re.search(m[2], variables[m[1]]))), expression)
                    expression = re.sub(r'\$(\w+)', lambda m: repr(variables[m[1]]), expression)
                    expression = expression.replace("&&", "and").replace("||", "or")
                    if eval(expression, {"__builtins__": {}}, {}):
                        selected.add(name)
                        break
            with self.subTest(task=task, targets=targets, smoke=smoke, simulator=simulator, ui=ui):
                self.assertTrue(selected)
                for name in selected:
                    for need in jobs[name].get("needs", []):
                        if isinstance(need, dict):
                            if need.get("optional") and need["job"] not in selected:
                                continue
                            need = need["job"]
                        self.assertIn(need, selected)
                        self.assertLess(self.config["stages"].index(jobs[need]["stage"]), self.config["stages"].index(jobs[name]["stage"]))
                if task == "mobile":
                    android, ios = targets in ("all", "android"), targets in ("all", "ios")
                    self.assertEqual("android-apk" in selected, android)
                    self.assertEqual("ios-unsigned" in selected, ios)
                    self.assertEqual("android-smoke" in selected, android and smoke == "true")
                    self.assertEqual("ios-simulator" in selected, ios and simulator != "off")
                    self.assertEqual("android-ui" in selected, android and ui == "on")
                    self.assertNotIn("android-signed", selected)
                    self.assertNotIn("ios-signed", selected)
                    self.assertNotIn("publish-existing-release", selected)
                if task == "ci":
                    self.assertEqual(selected,{"desktop-tests","mobile-backend-check"})
                if task == "tag-builds":
                    self.assertEqual(selected,{"mobile-prepare","wheels-android","wheels-ios","android-apk","ios-unsigned",
                                              "ios-simulator","android-smoke","linux-build","macos-arm-build","macos-intel-build"})

    def test_signing_and_publish_are_blocking_manual_jobs(self):
        for name in ("android-signed", "ios-signed", "publish-existing-release"):
            with self.subTest(job=name):
                job = self.config[name]
                self.assertEqual(job["rules"][0]["when"], "manual")
                self.assertIs(job["rules"][0]["allow_failure"], False)
                self.assertFalse(job["interruptible"])
                self.assertTrue(job["manual_confirmation"])

    def test_desktop_matrix_keeps_all_twelve_variants(self):
        variants = [variant for name in ("windows-build", "linux-build", "macos-arm-build", "macos-intel-build")
                    for matrix in self.config[name]["parallel"]["matrix"] for variant in matrix["VARIANT"]]
        self.assertEqual(len(variants), 12)


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.env = {"GLOSSARION_PUBLISH": "true", "CI_COMMIT_REF_PROTECTED": "true", "CI_COMMIT_TAG": "v1.0.0",
                    "CI_COMMIT_SHA": "a" * 40, "CI_JOB_TOKEN": "offline-test-token",
                    "CI_API_V4_URL": "https://gitlab.example.invalid/api/v4", "CI_SERVER_URL": "https://gitlab.example.invalid", "CI_PROJECT_ID": "7"}
        self.folder = self.root / "artifacts/build"
        self.folder.mkdir(parents=True)
        self.binary = self.folder / "app.exe"
        self.binary.write_bytes(b"fixture executable")
        self.manifest = {"label": "build", "source_commit": self.env["CI_COMMIT_SHA"], "artifacts": [{"file": "app.exe", "sha256": hashlib.sha256(self.binary.read_bytes()).hexdigest()}]}
        self.save_manifest()

    def save_manifest(self):
        (self.folder / "manifest.json").write_text(json.dumps(self.manifest))

    def test_invalid_gates_stop_before_network(self):
        for key, value in (("GLOSSARION_PUBLISH", "false"), ("CI_COMMIT_REF_PROTECTED", "false"),
                           ("CI_COMMIT_TAG", "main"), ("CI_JOB_TOKEN", ""), ("CI_API_V4_URL", "https://elsewhere.example.invalid/api/v4")):
            opener = Mock()
            with self.subTest(key=key), self.assertRaises(release.ReleaseError):
                release.publish({**self.env, key: value}, self.root, opener)
            opener.open.assert_not_called()

    def test_tampered_binary_stops_before_network(self):
        self.binary.write_bytes(b"changed")
        opener = Mock()
        with self.assertRaisesRegex(release.ReleaseError, "digest changed"):
            release.publish(self.env, self.root, opener)
        opener.open.assert_not_called()

    def test_wrong_commit_rejected(self):
        with self.assertRaisesRegex(release.ReleaseError, "source commit"):
            release.artifact_entries(self.root, "b" * 40)

    def test_traversal_and_debug_outputs_rejected(self):
        for name in ("../app.exe", "..\\app.exe", "app_debugsigned.apk", "app_simulator.zip"):
            with self.subTest(name=name):
                if not name.startswith(".."):
                    (self.folder / name).write_bytes(self.binary.read_bytes())
                self.manifest["artifacts"][0]["file"] = name
                self.save_manifest()
                with self.assertRaises(release.ReleaseError):
                    release.artifact_entries(self.root, self.env["CI_COMMIT_SHA"])

    def test_empty_manifest_is_not_success(self):
        self.manifest["artifacts"] = []
        self.save_manifest()
        with self.assertRaisesRegex(release.ReleaseError, "No verified build artifacts"):
            release.artifact_entries(self.root, self.env["CI_COMMIT_SHA"])

    def test_existing_release_and_package_never_overwritten(self):
        for existing_links in ([], [{"name": "already-released"}]):
            requests = []
            def open_request(req, **kwargs):
                requests.append(req.method)
                if req.method == "GET":
                    return io.BytesIO(json.dumps({"tag_name": self.env["CI_COMMIT_TAG"], "commit": {"id": self.env["CI_COMMIT_SHA"]}, "assets": {"links": existing_links}}).encode())
                if req.method == "HEAD":
                    return io.BytesIO(b"")
                self.fail("Attempted a write after finding existing assets")
            with self.subTest(existing_links=existing_links), self.assertRaises(release.ReleaseError):
                release.publish(self.env, self.root, Mock(open=open_request))
            self.assertNotIn("PUT", requests)
            self.assertNotIn("POST", requests)

    def test_release_commit_mismatch_stops_before_upload(self):
        opener = Mock()
        opener.open.return_value = io.BytesIO(json.dumps({"tag_name": self.env["CI_COMMIT_TAG"], "commit": {"id": "b" * 40}}).encode())
        with self.assertRaisesRegex(release.ReleaseError, "different commit"):
            release.publish(self.env, self.root, opener)
        self.assertEqual(opener.open.call_count, 1)

    def test_publish_only_after_preflight_with_mock_http(self):
        requests = []
        def open_request(req, **kwargs):
            requests.append(req.method)
            self.assertEqual(req.get_header("Job-token"), "offline-test-token")
            if req.method == "GET":
                return io.BytesIO(json.dumps({"tag_name": self.env["CI_COMMIT_TAG"], "commit": {"id": self.env["CI_COMMIT_SHA"]}, "assets": {"links": []}}).encode())
            if req.method == "HEAD":
                raise HTTPError(req.full_url, 404, "not found", {}, None)
            if req.method == "PUT":
                self.assertEqual(req.data.read(), b"fixture executable")
            if req.method == "POST":
                self.assertEqual(json.loads(req.data)["link_type"], "package")
            return io.BytesIO(b"{}")
        with redirect_stdout(io.StringIO()):
            release.publish(self.env, self.root, Mock(open=open_request))
        self.assertEqual(requests, ["GET", "HEAD", "PUT", "POST"])

    def test_redirect_does_not_forward_token(self):
        with self.assertRaisesRegex(release.ReleaseError, "refusing to forward"):
            release.NoRedirect().redirect_request(None, None, 302, "redirect", {}, "https://elsewhere.example.invalid")


if __name__ == "__main__":
    unittest.main(verbosity=2)
