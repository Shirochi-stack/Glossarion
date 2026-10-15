"""Offline release event, provenance, retry and preservation regressions."""
from contextlib import redirect_stdout
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import tempfile
import unittest
from urllib.error import HTTPError

import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("auto_release", ROOT / ".gitlab/auto_release.py")
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


class Response(io.BytesIO):
    def __init__(self, data=b"", headers=None):
        super().__init__(data)
        self.headers = headers or {}


class GitLab:
    def __init__(self, commit):
        self.release = {"tag_name": "v9.14.5", "commit": {"id": commit}, "name": "Owner's release title",
                        "description": "- updated glossarion auto updater for gitlab",
                        "assets": {"links": [{"name": "Windows executable", "url": "https://example.invalid/windows.exe"}]}}
        self.packages = {}
        self.calls = []
        self.fail_link_once = False

    def open(self, req, **kwargs):
        self.calls.append((req.method, req.full_url))
        assert req.get_header("Job-token") == "offline-job-token"
        if req.method == "GET" and "/releases/" in req.full_url:
            return Response(json.dumps(self.release).encode())
        if req.method == "HEAD":
            if req.full_url not in self.packages:
                raise HTTPError(req.full_url, 404, "missing", {}, None)
            data = self.packages[req.full_url]
            return Response(headers={"X-Checksum-SHA256": hashlib.sha256(data).hexdigest(), "Content-Length": str(len(data))})
        if req.method == "PUT":
            assert "/packages/generic/" in req.full_url
            assert req.full_url not in self.packages, "A retry attempted a duplicate package upload"
            self.packages[req.full_url] = req.data.read()
            return Response(b"{}")
        if req.method == "POST":
            assert req.full_url.endswith("/assets/links"), "Publisher attempted to create a release/pipeline"
            if self.fail_link_once:
                self.fail_link_once = False
                raise HTTPError(req.full_url, 500, "temporary failure", {}, None)
            self.release["assets"]["links"].append(json.loads(req.data))
            return Response(b"{}")
        raise AssertionError((req.method, req.full_url))


class PublisherTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "src").mkdir()
        (self.root / "src/app_version.py").write_text("VERSION_TAG = 'v9.14.5'\n")
        self.commit = "a" * 40
        self.env = dict(GLOSSARION_TASK="release-builds", GLOSSARION_PUBLISH="true", CI_PIPELINE_SOURCE="pipeline",
                        GLOSSARION_RELEASE_EVENT_VALIDATED="true",
                        CI_COMMIT_TAG="v9.14.5", CI_COMMIT_SHA=self.commit, CI_JOB_TOKEN="offline-job-token",
                        CI_PROJECT_ID="87623444", CI_SERVER_URL="https://gitlab.com", CI_API_V4_URL="https://gitlab.com/api/v4")
        self.api = GitLab(self.commit)
        self.binary = self.artifact("mobile-apk", "Glossarion_Android_arm64-v8a_debugsigned.apk", b"verified APK")

    def artifact(self, label, name, data):
        folder = self.root / "artifacts" / label
        folder.mkdir(parents=True)
        path = folder / name
        path.write_bytes(data)
        (folder / "manifest.json").write_text(json.dumps({"label": label, "source_commit": self.commit,
            "artifacts": [{"file": name, "sha256": hashlib.sha256(data).hexdigest()}]}))
        return path

    def publish(self):
        with redirect_stdout(io.StringIO()):
            return release.publish(self.env, self.root, self.api)

    def test_appends_outputs_and_checksums_preserving_owner_metadata_and_windows(self):
        self.artifact("desktop-linux-standard", "Glossarion_Linux_Standard.zip", b"verified Linux")
        self.artifact("mobile-ipa", "Glossarion_iOS_unsigned.ipa", b"verified IPA")
        self.artifact("mobile-ios-simulator", "diagnostic_simulator.zip", b"not a release asset")
        before = copy.deepcopy(self.api.release)
        result = self.publish()
        self.assertEqual(result["artifacts"], 3)
        self.assertEqual(self.api.release["name"], before["name"])
        self.assertEqual(self.api.release["description"], before["description"])
        self.assertEqual(self.api.release["assets"]["links"][0], before["assets"]["links"][0])
        added = self.api.release["assets"]["links"][1:]
        self.assertEqual(len(added), 6)
        self.assertTrue(all(row["name"].endswith(".sha256.json") for row in added[:3]))
        for url, data in self.api.packages.items():
            if url.endswith(".sha256.json"):
                metadata = json.loads(data)
                self.assertEqual(metadata["tag"], "v9.14.5")
                self.assertEqual(metadata["source_commit"], self.commit)
                binary = self.api.packages[url.removesuffix(".sha256.json")]
                self.assertEqual(metadata["sha256"], hashlib.sha256(binary).hexdigest())
                self.assertEqual(metadata["size"], len(binary))
        self.assertFalse(any("simulator" in url for url in self.api.packages))

    def test_repeat_publication_neither_uploads_nor_duplicates_links(self):
        self.publish()
        packages, links = copy.deepcopy(self.api.packages), copy.deepcopy(self.api.release["assets"]["links"])
        self.api.calls.clear()
        self.publish()
        self.assertEqual(self.api.packages, packages)
        self.assertEqual(self.api.release["assets"]["links"], links)
        self.assertFalse(any(method in ("PUT", "POST", "DELETE") for method, _ in self.api.calls))

    def test_link_failure_retry_reuses_uploads_without_rebuilding(self):
        self.api.fail_link_once = True
        with self.assertRaises(HTTPError):
            self.publish()
        self.assertEqual(len(self.api.packages), 2)
        self.api.calls.clear()
        self.publish()
        self.assertEqual(len(self.api.release["assets"]["links"]), 3)
        self.assertFalse(any(method == "PUT" for method, _ in self.api.calls))

    def test_conflicting_package_or_link_stops_before_any_write(self):
        url = "https://gitlab.com/api/v4/projects/87623444/packages/generic/glossarion/v9.14.5/mobile-apk--" + self.binary.name
        for kind in ("package", "link"):
            self.api = GitLab(self.commit)
            if kind == "package":
                self.api.packages[url] = b"different bytes"
            else:
                self.api.release["assets"]["links"].append({"name": self.binary.name, "url": "https://example.invalid/other"})
            with self.subTest(kind=kind), self.assertRaises(release.ReleaseError):
                self.publish()
            self.assertFalse(any(method in ("PUT", "POST", "DELETE") for method, _ in self.api.calls))

    def test_source_tamper_and_wrong_version_are_rejected_before_network(self):
        self.binary.write_bytes(b"tampered APK")
        with self.assertRaisesRegex(release.ReleaseError, "SHA-256"):
            self.publish()
        self.assertEqual(self.api.calls, [])
        for key, value in (("CI_COMMIT_TAG", "v9.14.4"), ("CI_PIPELINE_SOURCE", "merge_request_event"),
                           ("CI_PROJECT_ID", "99"), ("CI_JOB_TOKEN", ""), ("GLOSSARION_PUBLISH", "false")):
            with self.subTest(key=key), self.assertRaises(release.ReleaseError):
                release.publish({**self.env, key: value}, self.root, self.api)
            self.assertEqual(self.api.calls, [])

    def test_wrong_release_commit_and_missing_release_never_create_or_overwrite(self):
        self.api.release["commit"]["id"] = "b" * 40
        with self.assertRaisesRegex(release.ReleaseError, "source commit"):
            self.publish()
        self.assertEqual([method for method, _ in self.api.calls], ["GET"])
        class Missing:
            def open(self, req, **kwargs):
                assert req.method == "GET"
                raise HTTPError(req.full_url, 404, "release missing", {}, None)
        with self.assertRaises(HTTPError):
            release.publish(self.env, self.root, Missing())

    def test_only_exact_manifest_source_is_eligible(self):
        path = self.binary.parent / "manifest.json"
        data = json.loads(path.read_text()); data["source_commit"] = "b" * 40
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(release.ReleaseError, "source/label"):
            self.publish()
        self.assertEqual(self.api.calls, [])

    def test_token_never_follows_redirect(self):
        with self.assertRaisesRegex(release.ReleaseError, "refusing to forward"):
            release.NoRedirect().redirect_request(None, None, 302, "redirect", {}, "https://example.invalid")

    def test_linux_recovery_publishes_only_one_selected_variant(self):
        self.env.update(GLOSSARION_TASK="linux-release-repair", CI_PIPELINE_SOURCE="api", CI_COMMIT_BRANCH="main",
                        GLOSSARION_REPAIR_LINUX_VARIANT="standard")
        with self.assertRaisesRegex(release.ReleaseError, "only its one"):
            self.publish()  # existing APK fixture cannot leak into a Linux recovery
        self.assertEqual(self.api.calls, [])
        self.binary.unlink(); (self.binary.parent / "manifest.json").unlink()
        self.artifact("desktop-linux-standard", "Glossarion_Linux_Standard.zip", b"verified Linux")
        self.assertEqual(self.publish()["artifacts"], 1)
        self.assertEqual(len(self.api.release["assets"]["links"]), 3)


def evaluate(expression, variables):
    expression = re.sub(r'\$(\w+) =~ /([^/]+)/', lambda m: str(bool(re.search(m[2], variables.get(m[1], "")))), expression)
    expression = re.sub(r'\$(\w+)', lambda m: repr(variables.get(m[1], "")), expression)
    return bool(eval(expression.replace("&&", "and").replace("||", "or"), {"__builtins__": {}}, {}))


class ReleaseEventTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.header, cls.config = yaml.safe_load_all((ROOT / ".gitlab-ci.yml").read_text())
        for include in cls.config["include"]:
            cls.config.update(yaml.safe_load((ROOT / include["local"]).read_text()))

    def select(self, source, tag="", branch="", **extra):
        values = dict(CI_PIPELINE_SOURCE=source, CI_COMMIT_TAG=tag, CI_COMMIT_BRANCH=branch,
                      GLOSSARION_TASK="source-check", GLOSSARION_PUBLISH="false")
        values.update(extra)
        for rule in self.config["workflow"]["rules"]:
            if rule.get("when") == "never":
                return values, set()
            if evaluate(rule["if"], values):
                values.update(rule.get("variables", {})); break
        jobs = set()
        for name, job in self.config.items():
            if not isinstance(job, dict) or "script" not in job:
                continue
            for rule in job.get("rules", []):
                if rule.get("if") and evaluate(rule["if"], values):
                    if rule.get("when") != "never": jobs.add(name)
                    break
        return values, jobs

    def test_new_version_tag_is_cheap_until_release_creation(self):
        values, jobs = self.select("push", "v9.14.5")
        self.assertEqual(values["GLOSSARION_TASK"], "release-wait")
        self.assertEqual(jobs, {"release-tag-ready"})
        self.assertEqual(values["GLOSSARION_PUBLISH"], "false")

    def test_release_receiver_only_runs_dispatcher_on_main_with_payload(self):
        _, jobs = self.select("trigger", branch="main", TRIGGER_PAYLOAD="/tmp/release.json")
        self.assertEqual(jobs, {"release-event-dispatch"})
        self.assertEqual(self.config["release-event-dispatch"]["resource_group"], "glossarion-release-dispatch")
        self.assertFalse(self.config["release-event-dispatch"]["interruptible"])
        self.assertEqual(self.select("trigger", branch="main")[1], set())
        self.assertEqual(self.select("trigger", branch="feature", TRIGGER_PAYLOAD="/tmp/release.json")[1], set())

    def test_validated_release_dispatch_runs_requested_platforms_and_one_publisher(self):
        values, jobs = self.select("pipeline", branch="main", GLOSSARION_RELEASE_DISPATCH="true",
                                   GLOSSARION_RELEASE_TAG="v9.14.5", GLOSSARION_RELEASE_SOURCE_SHA="a" * 40)
        self.assertEqual(values["GLOSSARION_TASK"], "release-builds")
        self.assertEqual(values["GLOSSARION_PUBLISH"], "true")
        self.assertEqual(jobs, {"mobile-prepare", "wheels-android", "wheels-ios", "android-apk", "ios-unsigned",
                               "ios-simulator", "android-smoke", "linux-build", "publish-version-release"})
        self.assertNotIn("release-builds", self.header["spec"]["inputs"]["task"]["options"])
        publisher = self.config["publish-version-release"]
        self.assertEqual(publisher["dependencies"], ["linux-build", "android-apk", "ios-unsigned"])
        self.assertEqual(publisher["rules"][0]["when"], "always")
        self.assertNotIn("needs", publisher)  # waits for the existing device stage too
        for name in jobs:
            bootstrap = self.config[name]["before_script"][0]
            self.assertIn("release_event.py checkout", bootstrap)
            self.assertIn('CI_COMMIT_SHA="$GLOSSARION_RELEASE_SOURCE_SHA"', bootstrap)
        self.assertNotIn("release_event.py checkout", str(self.config["windows-build"]["before_script"]))

    def test_release_metadata_or_asset_edits_cannot_start_builds_or_self_trigger(self):
        for source in ("release", "webhook", "schedule"):
            with self.subTest(source=source):
                self.assertEqual(self.select(source, "v9.14.5")[1], set())
        self.assertNotIn("publish-version-release", self.select("push", branch="main")[1])
        self.assertNotIn("publish-version-release", self.select("api", "v9.14.5")[1])
        self.assertNotIn("publish-version-release", self.select("push", "v")[1])

    def test_explicit_recovery_selects_one_job_and_no_mobile_or_matrix_rebuild(self):
        values, jobs = self.select("api", branch="main", GLOSSARION_TASK="linux-release-repair",
                                   GLOSSARION_PUBLISH="true", GLOSSARION_RELEASE_TAG="v9.14.5",
                                   GLOSSARION_RELEASE_SOURCE_SHA="a" * 40)
        self.assertEqual(values["GLOSSARION_RELEASE_EVENT_VALIDATED"], "true")
        self.assertEqual(jobs, {"linux-release-repair"})
        self.assertNotIn("parallel", self.config["linux-release-repair"])
        self.assertEqual(self.config["linux-release-repair"]["before_script"][0], "python3 .gitlab/release_event.py checkout")


if __name__ == "__main__":
    unittest.main(verbosity=2)
