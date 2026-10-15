"""Validate and expose a bounded source-oracle package without importing history.

Examples:
  python tests/parity/oracle_package.py verify
  python tests/parity/oracle_package.py run -- python -m pytest tests/test_mobile_auth_splits.py
  python tests/parity/oracle_package.py export --output /tmp/oracle_objects.zip

Export reads the exact pinned objects from a local original checkout. It never
fetches, stages, commits, changes refs, or uploads anything. Run uses a temporary
Git object directory and the checkout's existing objects as an alternate.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import zipfile
import zlib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MANIFEST_SHA256 = "36af019a573c45e926d116694b07eede3d219fd49fc945ec129e02040af21182"
MAX_OBJECTS = 400
MAX_TOTAL_BYTES = 110_000_000

# Application sandboxes clear os.environ, but parity reads still belong to the
# checkout's reviewed source fixture. Capture only its two Git lookup settings.
_ORACLE_GIT_ENV = {key: os.environ[key] for key in
                   ("GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES") if key in os.environ}


def oracle_git_environment():
    """Preserve the source fixture while inheriting other current process settings."""
    return {**os.environ, **_ORACLE_GIT_ENV}


class OracleError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise OracleError(message)


def load_manifest(directory=HERE):
    data = (Path(directory) / "oracle_manifest.json").read_bytes()
    require(hashlib.sha256(data).hexdigest() == MANIFEST_SHA256, "Oracle manifest integrity check failed.")
    manifest = json.loads(data)
    require(manifest.get("schema") == 1, "Unsupported oracle manifest.")
    objects = manifest["objects"]
    require(0 < len(objects) <= MAX_OBJECTS, "Oracle object count exceeds its bound.")
    require(sum(o["bytes"] for o in objects) <= MAX_TOTAL_BYTES, "Oracle package exceeds its size bound.")
    require(len({o["oid"] for o in objects}) == len(objects), "Duplicate oracle object.")
    for obj in objects:
        require(re.fullmatch(r"[0-9a-f]{40}", obj["oid"]) and obj["type"] in ("blob", "tree", "commit"), "Invalid oracle object declaration.")
    return manifest


def validate_object(obj, raw):
    require(len(raw) == obj["bytes"] and hashlib.sha256(raw).hexdigest() == obj["sha256"], "Oracle object SHA-256 check failed.")
    require(hashlib.sha1(raw).hexdigest() == obj["oid"], "Oracle Git object identity check failed.")
    header, body = raw.split(b"\0", 1)
    require(header == f'{obj["type"]} {len(body)}'.encode(), "Oracle Git object header check failed.")


def verified_objects(directory=HERE):
    directory = Path(directory)
    manifest = load_manifest(directory)
    archive = directory / "oracle_objects.zip"
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    require(digest.hexdigest() == manifest["archive_sha256"], "Oracle archive integrity check failed.")
    with zipfile.ZipFile(archive) as package:
        expected = {"objects/" + obj["oid"] for obj in manifest["objects"]}
        require(len(package.infolist()) == len(expected) and set(package.namelist()) == expected, "Unexpected or duplicate archive members.")
        for obj in manifest["objects"]:
            member = package.getinfo("objects/" + obj["oid"])
            require(member.file_size == obj["bytes"], "Oracle archive member exceeds its declared size.")
            raw = package.read(member)
            validate_object(obj, raw)
            yield obj, raw


def environment(root, work, env=None, directory=HERE):
    """Return a child-process environment; never modify checkout objects or refs."""
    root, work = Path(root).resolve(), Path(work).resolve()
    result = dict(os.environ if env is None else env)
    clean = dict(result)
    clean.pop("GIT_OBJECT_DIRECTORY", None)
    clean.pop("GIT_ALTERNATE_OBJECT_DIRECTORIES", None)
    probe = subprocess.run(["git", "rev-parse", "--git-path", "objects"], cwd=root, env=clean, capture_output=True, text=True)
    require(probe.returncode == 0, "Oracle tests require a Git checkout.")
    original = Path(probe.stdout.strip())
    if not original.is_absolute(): original = root / original
    original = original.resolve()
    target = work / "objects"
    require(target.resolve() != original and not target.is_symlink(), "Oracle scratch must be separate from repository objects.")
    target.mkdir(parents=True, exist_ok=True)
    for obj, raw in verified_objects(directory):
        path = target / obj["oid"][:2] / obj["oid"][2:]
        require(not path.is_symlink() and not path.parent.is_symlink(), "Oracle scratch contains a symlink.")
        path.parent.mkdir(exist_ok=True)
        encoded = zlib.compress(raw)
        if path.exists():
            require(path.read_bytes() == encoded, "Existing oracle scratch object differs; use a fresh scratch directory.")
        else:
            with path.open("xb") as stream: stream.write(encoded)
    result.update(GIT_OBJECT_DIRECTORY=str(target), GIT_ALTERNATE_OBJECT_DIRECTORIES=str(original), CI="true")
    return result


def export(output, root=ROOT, directory=HERE):
    """Reproduce the reviewed package from exact objects, without ancestor traversal."""
    output = Path(output)
    require(not output.exists(), "Refusing to overwrite an existing export.")
    manifest = load_manifest(directory)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as package:
        for obj in manifest["objects"]:
            process = subprocess.run(["git", "cat-file", obj["type"], obj["oid"]], cwd=root, capture_output=True)
            require(process.returncode == 0, "A reviewed object is missing from the local source repository.")
            raw = f'{obj["type"]} {len(process.stdout)}\0'.encode() + process.stdout
            validate_object(obj, raw)
            member = zipfile.ZipInfo("objects/" + obj["oid"], date_time=(2026, 10, 14, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = 0o100644 << 16
            package.writestr(member, raw, compresslevel=9)
    require(hashlib.sha256(output.read_bytes()).hexdigest() == manifest["archive_sha256"], "Export compression differs from the pinned archive; retain the original package and review the toolchain.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify")
    exporter = sub.add_parser("export")
    exporter.add_argument("--output", required=True)
    run = sub.add_parser("run")
    run.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command == "verify":
        count = sum(1 for _ in verified_objects())
        print(f"Verified {count} bounded oracle objects; no history imported.")
        return 0
    if args.command == "export":
        export(args.output)
        print("Reproduced the reviewed oracle package locally.")
        return 0
    command = args.args[1:] if args.args[:1] == ["--"] else args.args
    require(command, "Supply the test command after --.")
    with tempfile.TemporaryDirectory(prefix="glossarion-oracles-") as temp:
        return subprocess.run(command, cwd=ROOT, env=environment(ROOT, Path(temp))).returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OracleError, OSError, ValueError, zipfile.BadZipFile) as exc:
        print("FAILED: " + str(exc), file=sys.stderr)
        raise SystemExit(1)
