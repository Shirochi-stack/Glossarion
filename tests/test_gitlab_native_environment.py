"""Native SDK environment regressions; no downloads, credentials or builds."""
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("native_ci", ROOT / ".gitlab/ci.py")
ci = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci)


class NativeEnvironmentTests(unittest.TestCase):
    def test_macos_keeps_bsd_tools_and_removes_test_object_overrides(self):
        runner = Mock(plan=False, env={"PATH":"/host/bin:/opt/homebrew/opt/coreutils/libexec/gnubin:/usr/bin:/bin",
                                     "GIT_OBJECT_DIRECTORY":"fixture", "GIT_ALTERNATE_OBJECT_DIRECTORIES":"checkout"})
        runner.command.return_value = "/usr/bin/stat\n"
        with patch.object(ci.sys, "platform", "darwin"), patch.object(ci.os, "pathsep", ":"):
            ci.native_build_environment(runner)
        self.assertEqual(runner.env["PATH"], "/host/bin:/usr/bin:/bin")
        self.assertNotIn("GIT_OBJECT_DIRECTORY", runner.env)
        self.assertNotIn("GIT_ALTERNATE_OBJECT_DIRECTORIES", runner.env)

    def test_linux_trust_is_exact_and_preserves_existing_process_configuration(self):
        home = ROOT / "fixture-home"
        sdk = home / "flutter/3.44.8"
        runner = Mock(plan=False, env={"HOME":str(home), "GIT_CONFIG_COUNT":"1", "GIT_CONFIG_KEY_0":"fixture.key",
                                     "GIT_CONFIG_VALUE_0":"preserved", "GIT_OBJECT_DIRECTORY":"fixture"})
        runner.command.return_value = json.dumps({"version":"3.44.8", "path":str(sdk)})
        with patch.object(ci.sys, "platform", "linux"):
            ci.native_build_environment(runner)
        self.assertEqual(runner.env["GIT_CONFIG_COUNT"], "2")
        self.assertEqual(runner.env["GIT_CONFIG_VALUE_0"], "preserved")
        self.assertEqual(runner.env["GIT_CONFIG_KEY_1"], "safe.directory")
        self.assertEqual(runner.env["GIT_CONFIG_VALUE_1"], str(sdk.resolve()))
        self.assertNotIn("GIT_OBJECT_DIRECTORY", runner.env)

    def test_unexpected_sdk_path_is_rejected(self):
        runner = Mock(plan=False, env={"HOME":str(ROOT / "fixture-home")})
        runner.command.return_value = json.dumps({"version":"3.44.8", "path":str(ROOT / "other")})
        with patch.object(ci.sys, "platform", "linux"), self.assertRaises(ci.PrerequisiteError):
            ci.native_build_environment(runner)


if __name__ == "__main__":
    unittest.main(verbosity=2)
