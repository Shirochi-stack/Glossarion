"""Expose Google's API 37.0 platform to Gradle's integer-only SDK reader.

Only the disposable job SDK is changed. The official installation stays intact;
the additional legacy view contains identical binaries and integer API metadata.
"""
from pathlib import Path
import hashlib
import os
import shutil


def prepare_api37(sdk_root):
    sdk_root = Path(sdk_root).resolve()
    source = sdk_root / "platforms/android-37.0"
    target = sdk_root / "platforms/android-37"
    if source.is_symlink() or not source.is_dir() or target.exists():
        raise ValueError("Expected a fresh official Android 37.0 installation.")
    properties = (source / "source.properties").read_text()
    values = dict(line.split("=", 1) for line in properties.splitlines() if "=" in line)
    if values.get("AndroidVersion.ApiLevel") != "37.0" or values.get("AndroidVersion.CodeName"):
        raise ValueError("Only released API 37.0 can be exposed as integer API 37.")
    build = dict(line.split("=", 1) for line in (source / "build.prop").read_text().splitlines() if "=" in line)
    if build.get("ro.build.version.sdk") != "37" or build.get("ro.build.version.sdk_full") != "37.0":
        raise ValueError("Android platform build metadata does not match API 37.0.")
    if any(path.is_symlink() for path in source.rglob("*")):
        raise ValueError("Unexpected symlink in the official Android platform.")
    # Without package.xml, the Android repository reader loads the supported
    # legacy source.properties format. Preserve the original modern metadata.
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("package.xml"))
    (target / "source.properties").write_text(properties.replace("AndroidVersion.ApiLevel=37.0", "AndroidVersion.ApiLevel=37"))
    for path in source.rglob("*"):
        if path.is_file() and path.name not in ("source.properties", "package.xml"):
            copied = target / path.relative_to(source)
            with path.open("rb") as before, copied.open("rb") as after:
                if hashlib.file_digest(before, "sha256").digest() != hashlib.file_digest(after, "sha256").digest():
                    raise ValueError("Android platform binary changed while preparing the legacy view.")
    print("Prepared API 37 view from released 37.0; all platform binaries match SHA-256.")
    return target


if __name__ == "__main__":
    prepare_api37(os.environ["ANDROID_HOME"])
