"""The historical fixture must work without importing refs or ancestry."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("oracle_package", ROOT / "tests/parity/oracle_package.py")
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


class OraclePackageTests(unittest.TestCase):
    def test_every_declared_object_is_verified(self):
        manifest = oracle.load_manifest()
        objects = list(oracle.verified_objects())
        self.assertEqual(len(objects), len(manifest["objects"]))
        self.assertEqual(len(manifest["commits"]), 13)
        self.assertEqual(sum(obj["type"] == "blob" for obj, _ in objects), 251)

    def test_tampered_object_fails(self):
        obj, raw = next(oracle.verified_objects())
        with self.assertRaises(oracle.OracleError):
            oracle.validate_object(obj, raw + b"changed")

    def test_tampered_manifest_fails_before_extraction(self):
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / "oracle_manifest.json").write_text('{"schema": 999}')
            with self.assertRaisesRegex(oracle.OracleError, "manifest integrity"):
                oracle.load_manifest(temp)

    def test_missing_archive_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / "oracle_manifest.json").write_bytes((oracle.HERE / "oracle_manifest.json").read_bytes())
            with self.assertRaises(FileNotFoundError):
                list(oracle.verified_objects(temp))

    def test_fresh_repository_uses_ephemeral_oracles_without_history_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "fresh"
            root.mkdir()
            clean = dict(os.environ)
            clean.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
            clean.pop("GIT_OBJECT_DIRECTORY", None)
            clean.pop("GIT_ALTERNATE_OBJECT_DIRECTORIES", None)
            def git(*args, env=clean, check=True):
                return subprocess.run(["git", "-C", str(root), *args], env=env, capture_output=True, check=check)
            git("init", "-q")
            git("-c", "user.name=Offline fixture", "-c", "user.email=fixture@example.invalid", "commit", "--allow-empty", "-qm", "Fresh fixture")
            head = git("rev-parse", "HEAD").stdout
            refs = git("show-ref").stdout
            before = {p.relative_to(root / ".git/objects").as_posix(): p.read_bytes()
                      for p in (root / ".git/objects").rglob("*") if p.is_file()}
            manifest = oracle.load_manifest()
            commit = manifest["commits"][0]
            self.assertNotEqual(git("cat-file", "-e", commit["commit"], check=False).returncode, 0)
            env = oracle.environment(root, Path(temp) / "scratch", clean)
            for pinned in manifest["commits"]:
                for alias in pinned["aliases"]:
                    self.assertEqual(git("rev-parse", alias + "^{commit}", env=env).stdout.decode().strip(), pinned["commit"])
                sample = pinned["files"][0]
                body = git("show", pinned["commit"] + ":" + sample["path"], env=env).stdout
                self.assertEqual(hashlib.sha1(f"blob {len(body)}\0".encode() + body).hexdigest(), sample["oid"])
            # Headless application tests intentionally scrub process variables;
            # frozen-source reads must retain the fixture's isolated Git lookup.
            with patch.dict(os.environ, env, clear=True):
                fixture_spec = importlib.util.spec_from_file_location("fixture_oracle", ROOT / "tests/parity/oracle_package.py")
                fixture = importlib.util.module_from_spec(fixture_spec)
                fixture_spec.loader.exec_module(fixture)
                progress_spec = importlib.util.spec_from_file_location("fixture_progress", ROOT / "tests/parity/progress_legacy.py")
                progress = importlib.util.module_from_spec(progress_spec)
                with patch.dict(sys.modules, {"parity.oracle_package": fixture}):
                    progress_spec.loader.exec_module(progress)
                    runner_spec = importlib.util.spec_from_file_location("fixture_runner", ROOT / "tests/parity/runner_parity.py")
                    runner = importlib.util.module_from_spec(runner_spec)
                    with patch.dict(sys.modules, {"fixture_runner": runner}):
                        runner_spec.loader.exec_module(runner)
                os.environ.pop("GIT_OBJECT_DIRECTORY", None)
                os.environ.pop("GIT_ALTERNATE_OBJECT_DIRECTORIES", None)
                expected = [(c["commit"] + ":" + f["path"], f["oid"]) for c in manifest["commits"] for f in c["files"]]
                probe = subprocess.run(["git", "cat-file", "--batch-check=%(objectname)"], cwd=root,
                                       env=fixture.oracle_git_environment(), input="\n".join(spec for spec, _ in expected) + "\n",
                                       text=True, capture_output=True, check=True)
                self.assertEqual(probe.stdout.splitlines(), [oid for _, oid in expected])
                with patch.object(progress, "REPO_ROOT", root):
                    source = progress._git_show("src/Retranslation_GUI.py")
                pinned = next(c for c in manifest["commits"] if c["commit"] == progress.BASE_SHA)
                declared = next(f for f in pinned["files"] if f["path"] == "src/Retranslation_GUI.py")
                body = source.encode("utf-8")
                self.assertEqual(hashlib.sha1(f"blob {len(body)}\0".encode() + body).hexdigest(), declared["oid"])
                sha = "9355bb5d376ca9eb8149a36b35b6028a95f86009"
                with patch.object(runner, "REPO_ROOT", root):
                    source = runner.git_show(sha, "src/manga_settings_dialog.py")
                self.assertIn("class MangaSettingsDialog", source)
            self.assertEqual(git("rev-parse", "HEAD").stdout, head)
            self.assertEqual(git("show-ref").stdout, refs)
            self.assertEqual(git("rev-list", "--all", "--count").stdout.strip(), b"1")
            self.assertEqual({p.relative_to(root / ".git/objects").as_posix(): p.read_bytes()
                             for p in (root / ".git/objects").rglob("*") if p.is_file()}, before)
            self.assertNotEqual(git("cat-file", "-e", commit["commit"], check=False).returncode, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
