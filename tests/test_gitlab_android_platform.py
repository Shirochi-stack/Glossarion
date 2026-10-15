import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("android_platform", Path(__file__).resolve().parents[1] / ".gitlab/android_platform.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class AndroidPlatformTests(unittest.TestCase):
    def fixture(self, root, api="37.0", sdk="37", full="37.0"):
        source = Path(root) / "platforms/android-37.0"
        source.mkdir(parents=True)
        (source / "source.properties").write_text(f"AndroidVersion.ApiLevel={api}\nAndroidVersion.CodeName=\nPkg.Revision=2\n")
        (source / "build.prop").write_text(f"ro.build.version.sdk={sdk}\nro.build.version.sdk_full={full}\n")
        (source / "package.xml").write_text("official modern package metadata")
        (source / "android.jar").write_bytes(b"unchanged platform API bytes\x00")
        return source

    def test_only_integer_metadata_changes_and_official_installation_is_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            source = self.fixture(root)
            before = {p.name: p.read_bytes() for p in source.iterdir()}
            target = module.prepare_api37(root)
            self.assertEqual(before, {p.name: p.read_bytes() for p in source.iterdir()})
            self.assertEqual((target / "android.jar").read_bytes(), before["android.jar"])
            self.assertEqual((target / "build.prop").read_bytes(), before["build.prop"])
            self.assertIn("AndroidVersion.ApiLevel=37\n", (target / "source.properties").read_text())
            self.assertFalse((target / "package.xml").exists())
            with self.assertRaises(ValueError):
                module.prepare_api37(root)

    def test_rejects_nonzero_minor_api_and_mismatched_binary_metadata(self):
        for values in (("37.1", "37", "37.1"), ("37.0", "36", "37.0"), ("37.0", "37", "37.2")):
            with self.subTest(values=values), tempfile.TemporaryDirectory() as root:
                self.fixture(root, *values)
                with self.assertRaises(ValueError):
                    module.prepare_api37(root)
                self.assertFalse((Path(root) / "platforms/android-37").exists())


if __name__ == "__main__":
    unittest.main()
