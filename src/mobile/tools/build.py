#!/usr/bin/env python3
"""Local build driver for the Glossarion mobile app (wraps the other tools and ``flet build``).

Run from anywhere; paths are relative to ``src/mobile``::

    python tools/build.py prepare            # collect backend + generate assets + verify bundle
    python tools/build.py apk                # prepare, then flet build apk --split-per-abi (debug-signed
                                             # unless FLET_ANDROID_SIGNING_* env vars are set)
    python tools/build.py aab                # prepare, then flet build aab
    python tools/build.py ipa                # macOS only: prepare, then flet build ipa (unsigned
                                             # unless --ios-* signing args are passed after --)
    python tools/build.py ios-simulator      # macOS only
    python tools/build.py version            # print version/build number (tools/version_info.py)
    python tools/build.py size [APK ...]     # where the APK bytes go (ci/apk_size_report.py; default
                                             # build/apk/*.apk)
    python tools/build.py ui-tests --device-id emulator-5554
                                             # prepare, then `flet test android` (tests/, the UI layer)

Every ``flet build`` gets ``--build-version`` (APP_VERSION from src/app_version.py) and
``--build-number`` (M*1_000_000 + m*10_000 + p*100). Arguments after ``--`` are passed
through to ``flet build`` unchanged, e.g.
``python tools/build.py apk -- --arch arm64-v8a -vv``.

Options:

* ``--skip-prepare``: reuse an existing bundle and assets. ``--verify`` and
  ``prepare_assets --check`` must still pass.
* ``--dry-run``: print the commands without running them.
* ``--no-split``: build one universal APK instead of per-ABI APKs.
* ``--legacy-packaging``: apk/aab: ``--android-legacy-packaging`` (native libraries compressed in
  the APK and extracted at install: a smaller download, a bigger install; README "APK size").
* ``--device-id``: ui-tests: the emulator/device (``adb devices``); ``ui-tests`` also takes the
  pytest arguments after ``--``.
* ``--flet``: the flet executable to use (default: ``flet`` on PATH, else ``python -m flet.cli``).

The tools run under the current interpreter (``sys.executable``). Use the host env
(``uv run python tools/build.py ...``) so ebooklib and the pins are available.
"""
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
MOBILE = TOOLS.parent
sys.path.insert(0, str(TOOLS))

import version_info  # noqa: E402  (sibling tool)

TARGETS = ("apk", "aab", "ipa", "ios-simulator")
CI = MOBILE / "ci"


def _fmt(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(c)) for c in cmd)


def run(cmd: list[str], *, dry_run: bool, cwd: Path = MOBILE, env: dict | None = None) -> None:
    print(f"$ {_fmt(cmd)}", flush=True)
    if dry_run:
        return
    result = subprocess.run(cmd, cwd=str(cwd), env=env)
    if result.returncode != 0:
        raise SystemExit(f"command failed with exit code {result.returncode}: {_fmt(cmd)}")


def prepare(dry_run: bool, skip: bool = False) -> None:
    py = sys.executable
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    if skip:
        run([py, str(TOOLS / "collect_backend.py"), "--verify"], dry_run=dry_run, env=env)
        run([py, str(TOOLS / "prepare_assets.py"), "--check"], dry_run=dry_run, env=env)
        return
    run([py, str(TOOLS / "collect_backend.py"), "--out", "app/backend", "--report", "build/backend_report.md"],
        dry_run=dry_run, env=env)
    run([py, str(TOOLS / "prepare_assets.py")], dry_run=dry_run, env=env)
    run([py, str(TOOLS / "collect_backend.py"), "--verify"], dry_run=dry_run, env=env)


def flet_command(explicit: str | None) -> list[str]:
    if explicit:
        return shlex.split(explicit)
    exe = shutil.which("flet")
    if exe:
        return [exe]
    return [sys.executable, "-m", "flet.cli"]


def build(target: str, args: argparse.Namespace, passthrough: list[str]) -> None:
    if target in ("ipa", "ios-simulator") and sys.platform != "darwin" and not args.dry_run:
        raise SystemExit(f"flet build {target} needs macOS with Xcode (use the CI workflow on other hosts)")
    info = version_info.compute()
    prepare(args.dry_run, skip=args.skip_prepare)
    cmd = flet_command(args.flet) + ["build", target,
                                     "--build-version", str(info["build_version"]),
                                     "--build-number", str(info["build_number"])]
    if target == "apk" and not args.no_split and "--split-per-abi" not in passthrough:
        cmd.append("--split-per-abi")
    if args.legacy_packaging and target in ("apk", "aab") and "--android-legacy-packaging" not in passthrough:
        cmd.append("--android-legacy-packaging")
    cmd += passthrough
    env = dict(os.environ, PYTHONIOENCODING="utf-8", FLET_CLI_NO_RICH_OUTPUT=os.environ.get(
        "FLET_CLI_NO_RICH_OUTPUT", "1"))
    run(cmd, dry_run=args.dry_run, env=env)
    if not args.dry_run:
        out = MOBILE / "build" / target
        print(f"build {target} {info['build_version']} ({info['build_number']}) done; output in {out}")


def size_report(apks: list[str], dry_run: bool) -> None:
    paths = apks or sorted(str(p) for p in (MOBILE / "build" / "apk").glob("*.apk"))
    if not paths and not dry_run:
        raise SystemExit("no APKs: pass paths or run `python tools/build.py apk` first")
    run([sys.executable, str(CI / "apk_size_report.py"), *(paths or ["build/apk/*.apk"]),
         "--json", str(MOBILE / "build" / "apk_size.json")], dry_run=dry_run)


def ui_tests(args: argparse.Namespace, passthrough: list[str]) -> None:
    """`flet test android` on a running emulator/device (the CI android-ui-tests job runs
    ci/android_ui_tests.sh, which does the same with logcat and logs)."""
    if not args.device_id:
        raise SystemExit("ui-tests needs --device-id (see `adb devices`)")
    prepare(args.dry_run, skip=args.skip_prepare)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", FLET_CLI_NO_RICH_OUTPUT=os.environ.get(
        "FLET_CLI_NO_RICH_OUTPUT", "1"), ANDROID_SERIAL=args.device_id)
    cmd = flet_command(args.flet) + ["test", "android", "--device-id", args.device_id, "--yes"]
    if passthrough:
        cmd += ["--", *passthrough]
    run(cmd, dry_run=args.dry_run, env=env)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    passthrough: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, passthrough = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description="Glossarion mobile build driver (see module docstring).")
    ap.add_argument("command", choices=("prepare", "version", "size", "ui-tests") + TARGETS)
    ap.add_argument("paths", nargs="*", help="size: the APKs to analyse (default build/apk/*.apk)")
    ap.add_argument("--skip-prepare", action="store_true", help="verify the existing bundle/assets instead")
    ap.add_argument("--dry-run", action="store_true", help="print commands only")
    ap.add_argument("--no-split", action="store_true", help="apk: one universal APK instead of --split-per-abi")
    ap.add_argument("--legacy-packaging", action="store_true",
                    help="apk/aab: --android-legacy-packaging (compressed native libraries)")
    ap.add_argument("--device-id", help="ui-tests: the emulator or device id")
    ap.add_argument("--flet", help="flet command to run (default: flet on PATH or python -m flet.cli)")
    args = ap.parse_args(argv)

    if args.command == "version":
        return version_info.main(["--json"])
    if args.command == "prepare":
        prepare(args.dry_run, skip=args.skip_prepare)
        return 0
    if args.command == "size":
        size_report(args.paths, args.dry_run)
        return 0
    if args.command == "ui-tests":
        ui_tests(args, passthrough)
        return 0
    build(args.command, args, passthrough)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
