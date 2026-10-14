"""Opt-in GitLab jobs that reuse Glossarion's checked-in build commands.

This is a curated command runner, not a GitHub Actions interpreter. Only the
named run steps below are used; marketplace actions are replaced by GitLab
artifacts or explicitly provisioned runner tools. Unknown expressions fail.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
MOBILE = ROOT / "src/mobile"
TEST_TREES = ("tests", "src/mobile/tests", "src/mobile/tests_host",
              "src/mobile/extensions/flet_glossarion_native/tests")
WINDOWS = {"lite": "translator_lite.spec", "nocuda": "translator_NoCuda.spec",
           "turbolite": "translator_TurboLite.spec", "main": "translator.spec", "heavy": "translator_Heavy.spec"}
LINUX = {"standard": ("translator_linux.spec", "requirements.txt", "Standard"),
         "nocuda": ("translator_linux_NoCuda.spec", "requirements_NoCuda.txt", "NoCuda"),
         "turbolite": ("translator_linux_TurboLite.spec", "requirements.txt", "TurboLite")}


class PrerequisiteError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise PrerequisiteError(message)


def workflow(name):
    return yaml.safe_load((ROOT / ".github/workflows" / name).read_text(encoding="utf-8"))


def step_definition(filename, job, name):
    matches = [s for s in workflow(filename)["jobs"][job]["steps"] if s.get("name") == name]
    require(len(matches) == 1 and "run" in matches[0], f"Expected one shell step: {filename}/{job}/{name}")
    return matches[0]


def render(text, context):
    def replace(match):
        key = match.group(1).strip()
        require(key in context, f"Unsupported workflow expression: {key}")
        return str(context[key])
    return re.sub(r"\$\{\{(.*?)\}\}", replace, str(text))


def tools_required(*names):
    for name in names:
        require(shutil.which(name), f"Runner prerequisite missing: {name}. See docs/gitlab_ci.md.")


def platform_required(expected, python_minor):
    require(sys.platform == expected, f"This job needs {expected}, not {sys.platform}.")
    require(sys.version_info[:2] == python_minor, f"This job needs Python {python_minor[0]}.{python_minor[1]}.")


def require_tests(*trees):
    for tree in trees:
        require(any((ROOT / tree).rglob("test_*.py")), f"Required tests missing: {tree}. Tests are included in Git; restore a complete checkout. No tests passed.")


def check_oracle_history(env=None):
    # The existing suites contain git-show parity oracles. Fresh history alone
    # cannot satisfy them; fail before a long dependency install/test run.
    commits = set()
    for tree in (ROOT / "tests", MOBILE / "tests_host"):
        for path in tree.rglob("*.py"):
            for match in re.finditer(r'(?m)^\w*(?:BASE|SHA)\w*\s*=\s*[\'"]([0-9a-f]{40})[\'"]', path.read_text(encoding="utf-8-sig", errors="replace")):
                commits.add(match.group(1))
    missing = [sha for sha in sorted(commits) if subprocess.run(
        ["git", "cat-file", "-e", sha + "^{commit}"], cwd=ROOT, env=env, capture_output=True).returncode]
    require(not missing, "Historical test oracles are missing: " + ", ".join(missing) +
            ". Verify the bounded oracle package; do not skip parity tests.")


def prepare_oracles(r):
    if r.plan or getattr(r, "oracles_prepared", False):
        return
    path = ROOT / "tests/parity/oracle_package.py"
    require(path.is_file(), "Required bounded oracle loader is missing.")
    spec = importlib.util.spec_from_file_location("glossarion_oracle_package", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    r.env = module.environment(ROOT, r.work / "oracles", r.env)
    r.oracles_prepared = True


class Runner:
    def __init__(self, label, plan=False):
        require(re.fullmatch(r"[a-z0-9-]+", label), "Invalid output label.")
        self.label, self.plan = label, plan
        self.env = dict(os.environ)
        self.env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", FLET_CLI_NO_RICH_OUTPUT="1",
                        CI="true", UV_LOCK_ARG="--locked", GITHUB_WORKSPACE=str(ROOT))
        self.context = {"github.workspace": str(ROOT)}
        self.work = ROOT / ".gitlab-work" / label
        self.report = ROOT / "artifacts" / label
        if not plan:
            self.work.mkdir(parents=True, exist_ok=True)
            self.report.mkdir(parents=True, exist_ok=True)
            self.env.update(RUNNER_TEMP=str(self.work), GITHUB_STEP_SUMMARY=str(self.report / "summary.md"))
        self.operations = []

    def command(self, args, cwd=ROOT, capture=False, label=None):
        title = label or " ".join(str(arg) for arg in args[:3])
        self.operations.append(title)
        print(title, flush=True)
        if self.plan:
            return ""
        # Never include full argv in exceptions: signing arguments may be private.
        result = subprocess.run([str(arg) for arg in args], cwd=cwd, env=self.env,
                                capture_output=capture, text=True)
        require(result.returncode == 0, f"{title} failed (exit {result.returncode}); see job output.")
        return result.stdout if capture else ""

    def step(self, filename, job, name, context=None):
        step = step_definition(filename, job, name)
        ctx = {**self.context, **{f"env.{k}": v for k, v in self.env.items()}, **(context or {})}
        script = render(step["run"], ctx)
        env = {k: render(v, ctx) for k, v in step.get("env", {}).items()}
        cwd = (ROOT / render(step.get("working-directory", "."), ctx)).resolve()
        require(cwd.is_relative_to(ROOT), "Workflow working directory leaves checkout.")
        shell = step.get("shell", "pwsh" if sys.platform == "win32" else "bash")
        require(shell in ("pwsh", "bash"), "Unsupported shell: " + shell)
        self.operations.append(name)
        print(name, flush=True)
        if self.plan:
            return
        tools_required(shell)
        self.env.update(env)
        number = str(len(self.operations))
        outputs = {key: self.work / (number + key) for key in ("GITHUB_ENV", "GITHUB_PATH", "GITHUB_OUTPUT")}
        for key, path in outputs.items():
            path.write_text("")
            self.env[key] = str(path)
        suffix = ".ps1" if shell == "pwsh" else ".sh"
        path = self.work / (number + suffix)
        prefix = ("$ErrorActionPreference = 'Stop'\n$PSNativeCommandUseErrorActionPreference = $true\n"
                  if shell == "pwsh" else "set -euo pipefail\n")
        path.write_text(prefix + script, encoding="utf-8", newline="\n")
        args = [shell, "-NoProfile", "-File", path] if shell == "pwsh" else [shell, str(path)]
        self.command(args, cwd, label=name)
        for line in outputs["GITHUB_ENV"].read_text(encoding="utf-8-sig").splitlines():
            require("=" in line and "<<" not in line, "Unsupported workflow env record.")
            key, value = line.split("=", 1)
            self.env[key] = value
        additions = outputs["GITHUB_PATH"].read_text(encoding="utf-8-sig").splitlines()
        if additions:
            self.env["PATH"] = os.pathsep.join(additions + [self.env.get("PATH", "")])
        for line in outputs["GITHUB_OUTPUT"].read_text(encoding="utf-8-sig").splitlines():
            require("=" in line, "Unsupported workflow output record.")
            key, value = line.split("=", 1)
            if "id" in step:
                self.context[f"steps.{step['id']}.outputs.{key}"] = value

    def collect(self, patterns):
        if self.plan:
            self.operations.append("Verify and collect build artifacts")
            return
        found = sorted({p for pattern in patterns for p in ROOT.glob(pattern) if p.is_file()})
        require(found, "Build produced no expected artifacts; refusing an empty success.")
        entries = []
        for path in found:
            require(not path.is_symlink() and path.resolve().is_relative_to(ROOT), "Artifact must be a regular file inside this checkout.")
            target = self.report / path.name
            require(not target.exists(), f"Artifact collision: {path.name}")
            shutil.copy2(path, target)
            entries.append({"file": path.name, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
        (self.report / "manifest.json").write_text(json.dumps({"label": self.label,
            "source_commit": self.env.get("CI_COMMIT_SHA", ""), "artifacts": entries}, indent=2))


def desktop(runner, target, variant):
    r = runner
    if target == "windows":
        require(variant in WINDOWS, "Unsupported Windows variant.")
        if not r.plan:
            platform_required("win32", (3, 12))
            tools_required("pwsh")
            msys = Path(r.env.get("MSYS2_ROOT", "C:/msys64"))
            binpath = msys / "mingw64/bin"
            for dll in ("libpango-1.0-0.dll", "libcairo-2.dll", "libgobject-2.0-0.dll",
                        "libglib-2.0-0.dll", "libharfbuzz-0.dll", "libfontconfig-1.dll"):
                require((binpath / dll).is_file(), "MSYS2/GTK prerequisite missing: " + dll)
            r.env.update(MSYS2_ROOT=str(msys), GTK_FOLDER=str(msys / "mingw64"), PATH=str(binpath) + os.pathsep + r.env["PATH"])
        filename, job = "build-windows.yml", "build-windows-" + variant
        for name in ("Install dependencies", "Build with PyInstaller"):
            r.step(filename, job, name)
        r.collect(["src/dist/*.exe"])
    elif target == "linux":
        require(variant in LINUX, "Unsupported Linux variant.")
        if not r.plan:
            platform_required("linux", (3, 12))
            tools_required("bash", "zip", "pkg-config", "cmake")
            r.command(["pkg-config", "--exists", "pango", "cairo", "gdk-pixbuf-2.0"])
        spec, requirements, display = LINUX[variant]
        r.env.update(BUILD_SPEC=spec, BUILD_VARIANT=display, REQUIREMENTS_FILE=requirements)
        filename, job = "build-linux.yml", "build-linux"
        names = ["Install dependencies"]
        if variant == "nocuda": names.append("Install NoCuda OCR models")
        if variant != "turbolite": names.append("Check Qt WebEngine imports")
        if variant == "nocuda": names.append("Check NoCuda dependencies")
        names += ["Build with PyInstaller", "Verify Linux package contents", "Create Linux ZIP"]
        for name in names: r.step(filename, job, name)
        r.collect(["src/dist/*.zip"])
    else:
        require(target in ("macos-arm", "macos-intel") and variant in ("lite", "nocuda"), "Unsupported macOS build.")
        arch = "arm" if target == "macos-arm" else "intel"
        if not r.plan:
            platform_required("darwin", (3, 12))
            require(platform.machine() in (("arm64", "aarch64") if arch == "arm" else ("x86_64", "AMD64")), "Wrong macOS Python architecture.")
            tools_required("bash", "create-dmg", "pkg-config", "cmake")
            r.command(["pkg-config", "--exists", "pango", "cairo", "gdk-pixbuf-2.0"])
        job = "build-macos-" + arch + ("-nocuda" if variant == "nocuda" else "")
        build_name = ("Build ARM" if arch == "arm" else "Build Intel") + (" NoCuda" if variant == "nocuda" else "") + (" (Apple Silicon)" if arch == "arm" else " (x86_64)")
        for name in ("Install dependencies", build_name): r.step("build-macos.yml", job, name)
        r.collect(["src/dist/*.dmg"])


def source_check(r):
    for path in (ROOT / "src").glob("*.py"):
        ast.parse(path.read_bytes(), filename=str(path))
    print("Top-level source syntax checked; this is not a test-suite pass.")


def desktop_tests(r):
    if not r.plan:
        platform_required("linux", (3, 10))
        require_tests("tests")
        prepare_oracles(r)
        check_oracle_history(r.env)
        tools_required("bash")
    r.env.update(QT_QPA_PLATFORM="offscreen", PYTHONPATH=str(ROOT / "src"))
    for name in ("Install dependencies", "Force CPU-only PyTorch wheels", "Check Python syntax", "Run basic smoke test",
                 "Mobile shared-module and backend patch tests (Python 3.10)",
                 "Import hygiene of the shared GUI-free modules (PySide6 blocked)"):
        r.step("python-app.yml", "test", name)


def mobile_prerequisites(r, target=None):
    if r.plan: return
    require(sys.version_info[:2] == (3, 13), "Mobile jobs require host Python 3.13.")
    tools_required("uv", "bash", "git")
    if target == "android":
        require(sys.platform == "linux", "Android jobs use the provisioned Linux runner.")
        tools_required("java", "keytool", "adb", "sdkmanager")
        require(Path(r.env.get("ANDROID_HOME", "__missing__")).is_dir(), "ANDROID_HOME must name the provisioned Android SDK.")
    elif target == "ios":
        require(sys.platform == "darwin", "iOS jobs require macOS/Xcode.")
        tools_required("xcodebuild", "xcrun", "pod", "ditto")


def mobile_prepare(r, tests=True):
    mobile_prerequisites(r)
    if tests and not r.plan:
        require_tests("src/mobile/tests_host", "src/mobile/extensions/flet_glossarion_native/tests")
        prepare_oracles(r)
        check_oracle_history(r.env)
    r.command(["uv", "sync", "--locked", "--group", "dev"], MOBILE)
    r.command(["uv", "run", "--locked", "python", "tools/build.py", "prepare"], MOBILE)
    r.step("build-mobile.yml", "prepare", "Host import smoke (Android and iOS simulation)")
    if tests:
        r.command(["uv", "run", "--locked", "pytest", "tests_host", "extensions/flet_glossarion_native/tests", "-q", "-m", "not online"], MOBILE)
    # Use a workspace-owned scratch folder; do not run the GitHub disk-cleanup steps.
    r.step("build-mobile.yml", "prepare", "Offline E2E on the collected bundle (fake OpenAI server)")


def wheels(r, target):
    mobile_prerequisites(r, target)
    if not r.plan:
        tools_required("rustup", "rustc", "cargo", "pkg-config")
        require(" " not in str(r.work), "Mobile wheel work path must not contain spaces.")
    r.env.update(WHEELCACHE=str(r.work / "wheelcache"), FORGE_DIR=str(r.work / "forge"))
    wheelhouse = ROOT / "artifacts" / ("wheelhouse-" + target)
    r.env["WHEELHOUSE"] = str(wheelhouse)
    base = ["uv", "run", "--no-project", "python", "src/mobile/ci/wheels/wheels.py"]
    pins = r.command(base + ["env", "--platform", target, "--format", "github"], capture=True)
    values = dict(line.split("=", 1) for line in pins.splitlines() if "=" in line)
    if not r.plan:
        rust_version = r.command(["rustc", "--version"], capture=True)
        require(values["rust_version"] in rust_version, "Provision the Rust version pinned in wheelhouse.toml before running this job.")
        installed = r.command(["rustup", "target", "list", "--installed"], capture=True).splitlines()
        require(set(values["rust_targets"].split(",")) <= set(installed), "Provision all Rust targets pinned for this mobile platform.")
    packages = values.get("packages", "cryptography pillow" if target == "android" else "cryptography").split()
    r.command(["bash", "src/mobile/ci/wheels/build_wheels.sh", target, *packages])
    r.command(base + ["assemble", "--platform", target, "--cache", r.env["WHEELCACHE"], "--out", wheelhouse])
    r.command(["uv", "run", "--no-project", "python", "src/mobile/tools/check_mobile_wheels.py", "--verify-wheelhouse", wheelhouse, "--platform", target,
               "--report", r.report / "wheelhouse-check.md"])
    r.command(base + ["verify-native", "--platform", target, wheelhouse])
    r.command(base + ["provenance", "--platform", target, "--wheelhouse", wheelhouse, "--cache", r.env["WHEELCACHE"], "--built", " ".join(packages)])


def signing_gate(env, required):
    require(env.get("CI_COMMIT_REF_PROTECTED") == "true" and env.get("GLOSSARION_SIGNING_APPROVED") == "true",
            "Signing needs a protected ref and explicit GLOSSARION_SIGNING_APPROVED=true.")
    for name in required:
        require(bool(env.get(name)), "Signing prerequisite missing: " + name)


def mobile_build(r, target, signed=False):
    target_platform = "android" if target in ("apk", "aab") else "ios"
    mobile_prerequisites(r, target_platform)
    if not r.plan:
        require_tests("src/mobile/tests_host", "src/mobile/extensions/flet_glossarion_native/tests")
        prepare_oracles(r)
        check_oracle_history(r.env)
        if signed:
            names = ("FLET_ANDROID_SIGNING_KEY_STORE", "FLET_ANDROID_SIGNING_KEY_STORE_PASSWORD", "FLET_ANDROID_SIGNING_KEY_PASSWORD", "FLET_ANDROID_SIGNING_KEY_ALIAS") if target_platform == "android" else ("IOS_TEAM_ID", "IOS_PROFILE_UUID", "IOS_SIGNING_CERTIFICATE", "IOS_EXPORT_METHOD")
            signing_gate(r.env, names)
            if target_platform == "android":
                require(Path(r.env["FLET_ANDROID_SIGNING_KEY_STORE"]).is_file(), "The protected keystore file variable is missing.")
        else:
            for name in list(r.env):
                if name.startswith("FLET_ANDROID_SIGNING_"): r.env.pop(name)
        if target == "aab": require(signed, "AAB requires an explicitly requested signed build.")
    wheelhouse = ROOT / "artifacts" / ("wheelhouse-" + target_platform)
    r.env["PIP_FIND_LINKS"] = str(wheelhouse)
    if not r.plan:
        require(wheelhouse.is_dir() and any(wheelhouse.glob("*.whl")), "Required verified wheelhouse artifact is missing.")
        require(" " not in str(wheelhouse), "PIP_FIND_LINKS path must not contain spaces.")
    mobile_prepare(r)
    r.command(["uv", "run", "--no-project", "python", "src/mobile/tools/check_mobile_wheels.py", "--platform", target_platform,
               "--no-host", "--wheelhouse", wheelhouse, "--report", r.report / "wheelhouse-selection.md"])
    info = r.command([sys.executable, "src/mobile/tools/version_info.py", "--json"], capture=True)
    version = json.loads(info) if not r.plan else {"build_version": "PLAN", "build_number": 1, "artifact_prefix": "PLAN"}
    r.env.update(BUILD_VERSION=str(version["build_version"]), BUILD_NUMBER=str(version["build_number"]),
                 ARTIFACT_PREFIX=version["artifact_prefix"], APP_PACKAGE="com.glossarion.app", HAS_KEYSTORE="true" if signed else "false")
    command = ["uv", "run", "--locked", "python", "tools/build.py", target, "--skip-prepare"]
    if signed and target_platform == "ios":
        # Identity/profile must already be provisioned in an isolated runner by its owner.
        # No importing certificates, creating keychains, or printing signing arguments.
        if not r.plan:
            tools_required("security")
            identities = r.command(["security", "find-identity", "-v", "-p", "codesigning"], capture=True)
            require(r.env["IOS_SIGNING_CERTIFICATE"] in identities, "Requested signing identity is not available in the runner keychain.")
        command += ["--", "--ios-team-id", r.env.get("IOS_TEAM_ID", "PLAN"), "--ios-provisioning-profile", r.env.get("IOS_PROFILE_UUID", "PLAN"),
                    "--ios-signing-certificate", r.env.get("IOS_SIGNING_CERTIFICATE", "PLAN"), "--ios-export-method", r.env.get("IOS_EXPORT_METHOD", "PLAN")]
    r.command(command, MOBILE, label="Build " + target + (" (signed)" if signed else ""))
    r.command(["uv", "run", "--no-project", "python", "src/mobile/tools/check_mobile_wheels.py", "--verify-installed", "src/mobile/build/site-packages", "--wheelhouse", wheelhouse, "--platform", target_platform])
    if target_platform == "android":
        r.context[f"steps.{target}.outputs.output-path"] = str(MOBILE / "build" / target)
        r.step("build-mobile.yml", "android", "Collect APKs" if target == "apk" else "Collect AAB")
        r.step("build-mobile.yml", "android", "Verify APK/AAB (package, versions, permissions, FGS, ABIs, signature)")
        r.collect(["dist/*." + target])
    elif target == "ipa":
        r.context["steps.build.outputs.output-path"] = str(MOBILE / "build/ipa")
        job = "ios-signed" if signed else "ios"
        r.step("build-mobile.yml", job, "Collect signed IPA" if signed else "Package unsigned IPA")
        r.step("build-mobile.yml", job, "Verify signed IPA" if signed else "Verify IPA (bundle id, versions, document types, background ids, URL scheme)")
        r.collect(["dist/*.ipa"])
    else:
        r.context["steps.build.outputs.output-path"] = str(MOBILE / "build/ios-simulator")
        r.env.update(READY_TIMEOUT="600", SELFTEST_TIMEOUT="900", SELFTEST_SUITE="smoke", SELFTEST_START_TIMEOUT="120")
        r.step("build-mobile.yml", "ios-simulator-smoke", "Package simulator app")
        try:
            r.step("build-mobile.yml", "ios-simulator-smoke", "Smoke test on the simulator")
        finally:
            r.step("build-mobile.yml", "ios-simulator-smoke", "Collect simulator logs")
        r.step("build-mobile.yml", "ios-simulator-smoke", "Self-built wheels on the simulator (cryptography + OpenSSL, Pillow)")
        r.collect(["dist/*simulator.zip"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("source-check", "desktop-tests", "mobile-check", "desktop", "wheels", "mobile", "android-ui", "android-smoke"))
    parser.add_argument("--platform", choices=("windows", "linux", "macos-arm", "macos-intel", "android", "ios"))
    parser.add_argument("--variant", default="lite")
    parser.add_argument("--target", choices=("apk", "aab", "ipa", "ios-simulator"))
    parser.add_argument("--signed", action="store_true")
    parser.add_argument("--plan", action="store_true", help="Inspect selected commands without executing or checking runner tools.")
    args = parser.parse_args(argv)
    label = "-".join(str(x) for x in (args.task, args.platform, args.target, args.variant if args.task == "desktop" else None) if x)
    runner = Runner(label, args.plan)
    if args.task == "source-check": source_check(runner)
    elif args.task == "desktop-tests": desktop_tests(runner)
    elif args.task == "desktop": desktop(runner, args.platform, args.variant)
    elif args.task == "mobile-check": mobile_prepare(runner)
    elif args.task == "wheels":
        require(args.platform in ("android", "ios"), "Select the wheel platform.")
        wheels(runner, args.platform)
    elif args.task == "mobile":
        require(args.target, "Select the mobile target.")
        mobile_build(runner, args.target, args.signed)
    else:
        mobile_prerequisites(runner, "android")
        if not args.plan:
            require(runner.env.get("ANDROID_SERIAL"), "ANDROID_SERIAL must identify a running emulator/device.")
            runner.command(["adb", "-s", runner.env["ANDROID_SERIAL"], "get-state"])
        if args.task == "android-ui":
            if not args.plan:
                require_tests("src/mobile/tests")
            mobile_prepare(runner)
            wheelhouse = ROOT / "artifacts/wheelhouse-android"
            runner.env["PIP_FIND_LINKS"] = str(wheelhouse)
            runner.command(["uv", "run", "--no-project", "python", "src/mobile/tools/check_mobile_wheels.py", "--platform", "android", "--no-host", "--wheelhouse", wheelhouse])
            runner.command(["bash", "src/mobile/ci/android_ui_tests.sh", str(runner.report / "ui-logs")])
        else:
            apks = list((ROOT / "artifacts/mobile-apk").glob("*x86_64*.apk"))
            if not args.plan: require(len(apks) == 1, "Expected one x86_64 APK from the mobile-apk job.")
            runner.env["E2E_TIMEOUT"] = "1200"
            runner.command(["bash", "src/mobile/ci/android_smoke.sh", apks[0] if apks else "PLAN.apk", ROOT / "mobile-smoke-logs"])
            runner.step("build-mobile.yml", "android-smoke", "Self-built wheels on the emulator (cryptography + OpenSSL, Pillow)")
    if args.plan: print(json.dumps({"task": args.task, "operations": runner.operations}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PrerequisiteError as exc:
        print("FAILED: " + str(exc), file=sys.stderr)
        raise SystemExit(1)
