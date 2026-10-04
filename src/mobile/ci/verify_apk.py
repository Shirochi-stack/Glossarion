#!/usr/bin/env python3
"""Verify Glossarion Android build outputs (APK / AAB) before they are uploaded.

APK checks (Android SDK build-tools: ``apksigner`` + ``aapt2``):
  * signature verifies (v2 or v3 scheme); with ``--expect-release`` the signer must not be the
    Android debug certificate;
  * package name, versionName, versionCode (Flutter's ``--split-per-abi`` adds ABI*1000 to the
    base code: arm64-v8a +2000, x86_64 +4000), minSdk;
  * permissions: required ones present, storage/media ones absent. Expectations are the built-in
    baseline merged with ``[tool.flet.android.permission]`` from pyproject.toml (true = required,
    false = forbidden);
  * a ``<service>`` declaring ``foregroundServiceType`` dataSync (the job foreground service);
  * the ``glossarion://app`` deep-link intent filter;
  * native libraries only for the allowed ABIs, and a split APK holds exactly its own ABI;
  * a size warning above ``--max-size-mb``.

AAB checks (zip level + ``jarsigner``): manifest present, ABIs allowed, signature verifies; with
``--expect-release`` the signer must not be the debug certificate.

Exit status 1 when any check fails. Findings are also written as GitHub annotations and, when
``GITHUB_STEP_SUMMARY`` is set, as a Markdown table.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - CI runs 3.13
    tomllib = None

ABI_VERSION_OFFSET = {"armeabi-v7a": 1, "arm64-v8a": 2, "x86_64": 4}  # Flutter FlutterPluginConstants.ABI_VERSION
DEFAULT_ALLOWED_ABIS = ("arm64-v8a", "x86_64")
BASELINE_REQUIRED = (
    "android.permission.INTERNET",
    "android.permission.POST_NOTIFICATIONS",
    "android.permission.FOREGROUND_SERVICE",
    "android.permission.FOREGROUND_SERVICE_DATA_SYNC",
    "android.permission.REQUEST_IGNORE_BATTERY_OPTIMIZATIONS",
)
BASELINE_FORBIDDEN = (
    "android.permission.READ_EXTERNAL_STORAGE",
    "android.permission.WRITE_EXTERNAL_STORAGE",
    "android.permission.MANAGE_EXTERNAL_STORAGE",
    "android.permission.READ_MEDIA_IMAGES",
    "android.permission.READ_MEDIA_VIDEO",
    "android.permission.READ_MEDIA_AUDIO",
    "android.permission.READ_MEDIA_VISUAL_USER_SELECTED",
)
FGS_TYPE_DATA_SYNC = 0x1  # android:foregroundServiceType="dataSync"
DEBUG_CERT_MARKER = "CN=Android Debug"
LIB_RE = re.compile(r"^(?:base/)?lib/([^/]+)/[^/]+\.so$")


@dataclass
class Report:
    path: Path
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    facts: dict[str, str] = field(default_factory=dict)

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)


# --------------------------------------------------------------------------- tools


def _version_key(name: str) -> tuple[int, ...]:
    parts = re.findall(r"\d+", name)
    return tuple(int(p) for p in parts) if parts and "rc" not in name else (-1,)


def find_build_tools(explicit: str | None) -> Path | None:
    """Return the newest build-tools dir that has both aapt2 and apksigner."""
    if explicit:
        path = Path(explicit)
        return path if path.is_dir() else None
    env_dir = os.environ.get("ANDROID_BUILD_TOOLS")
    if env_dir and Path(env_dir).is_dir():
        return Path(env_dir)
    for var in ("ANDROID_HOME", "ANDROID_SDK_ROOT"):
        sdk = os.environ.get(var)
        if not sdk:
            continue
        root = Path(sdk) / "build-tools"
        if not root.is_dir():
            continue
        candidates = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: _version_key(p.name), reverse=True)
        for candidate in candidates:
            if _tool(candidate, "aapt2") and _tool(candidate, "apksigner"):
                return candidate
    return None


def _tool(build_tools: Path | None, name: str) -> str | None:
    if build_tools is not None:
        for suffix in ("", ".exe", ".bat"):
            path = build_tools / f"{name}{suffix}"
            if path.is_file():
                return str(path)
    return shutil.which(name)


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)


# --------------------------------------------------------------------------- parsers


def parse_badging(text: str) -> dict:
    info: dict = {"permissions": set(), "native_code": set()}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("package:"):
            for key in ("name", "versionCode", "versionName"):
                match = re.search(rf"\b{key}='([^']*)'", line)
                if match:
                    info[key] = match.group(1)
        elif line.startswith(("minSdkVersion:", "sdkVersion:")):
            match = re.search(r"'(\d+)'", line)
            if match:
                info.setdefault("minSdk", match.group(1))
        elif line.startswith("targetSdkVersion:"):
            match = re.search(r"'(\d+)'", line)
            if match:
                info["targetSdk"] = match.group(1)
        elif line.startswith(("uses-permission:", "uses-permission-sdk-23:")):
            match = re.search(r"name='([^']+)'", line)
            if match:
                info["permissions"].add(match.group(1))
        elif line.startswith("native-code:") or line.startswith("alt-native-code:"):
            info["native_code"].update(re.findall(r"'([^']+)'", line))
        elif line.startswith("application-label:"):
            match = re.search(r"'([^']*)'", line)
            if match:
                info["label"] = match.group(1)
    return info


def parse_permissions(text: str) -> set[str]:
    perms = set()
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(("uses-permission:", "uses-permission-sdk-23:")):
            match = re.search(r"name='([^']+)'", line)
            if match:
                perms.add(match.group(1))
    return perms


@dataclass
class XmlElement:
    name: str
    depth: int
    attrs: dict[str, str] = field(default_factory=dict)
    children: list["XmlElement"] = field(default_factory=list)

    def iter(self, name: str):
        if self.name == name:
            yield self
        for child in self.children:
            yield from child.iter(name)


_ELEMENT_RE = re.compile(r"^(\s*)E: (\S+)")
_ATTR_RE = re.compile(r"^(\s*)A: (?:[^=]*?[:/])?([A-Za-z_][\w.-]*)(?:\(0x[0-9a-fA-F]+\))?=(.*)$")


def parse_xmltree(text: str) -> XmlElement:
    """Parse `aapt2 dump xmltree` output into a small element tree (indentation based)."""
    root = XmlElement("#root", -1)
    stack = [root]
    for raw in text.splitlines():
        elem = _ELEMENT_RE.match(raw)
        if elem:
            depth = len(elem.group(1))
            while stack and stack[-1].depth >= depth:
                stack.pop()
            node = XmlElement(elem.group(2), depth)
            stack[-1].children.append(node)
            stack.append(node)
            continue
        attr = _ATTR_RE.match(raw)
        if attr and len(stack) > 1:
            stack[-1].attrs[attr.group(2)] = attr.group(3).strip()
    return root


def attr_text(value: str | None) -> str:
    """Raw string value of an aapt2 attribute: '"x" (Raw: "x")' -> 'x'."""
    if value is None:
        return ""
    raw = re.search(r'\(Raw: "([^"]*)"\)', value)
    if raw:
        return raw.group(1)
    quoted = re.match(r'^"([^"]*)"', value)
    if quoted:
        return quoted.group(1)
    return value.strip()


def attr_int(value: str | None) -> int | None:
    """Integer/flag value of an aapt2 attribute: '0x00000001', '1', '(type 0x11)0x1', '"dataSync"'."""
    if value is None:
        return None
    text = attr_text(value)
    if "dataSync" in text:
        return FGS_TYPE_DATA_SYNC
    cleaned = re.sub(r"\(type 0x[0-9a-fA-F]+\)", "", value)
    match = re.search(r"0x[0-9a-fA-F]+|-?\d+", cleaned)
    if not match:
        return None
    token = match.group(0)
    return int(token, 16) if token.lower().startswith("0x") else int(token)


def data_sync_services(tree: XmlElement) -> list[str]:
    names = []
    for service in tree.iter("service"):
        fgs = attr_int(service.attrs.get("foregroundServiceType"))
        if fgs is not None and fgs & FGS_TYPE_DATA_SYNC:
            names.append(attr_text(service.attrs.get("name")) or "<unnamed>")
    return names


def has_deep_link(tree: XmlElement, scheme: str, host: str) -> bool:
    for activity in list(tree.iter("activity")) + list(tree.iter("activity-alias")):
        for intent_filter in activity.iter("intent-filter"):
            actions = {attr_text(a.attrs.get("name")) for a in intent_filter.iter("action")}
            if "android.intent.action.VIEW" not in actions:
                continue
            schemes = {attr_text(d.attrs.get("scheme")) for d in intent_filter.iter("data")} - {""}
            hosts = {attr_text(d.attrs.get("host")) for d in intent_filter.iter("data")} - {""}
            if scheme in schemes and (not host or host in hosts):
                return True
    return False


def manifest_flag(tree: XmlElement, element: str, attribute: str) -> str | None:
    for node in tree.iter(element):
        if attribute in node.attrs:
            return attr_text(node.attrs[attribute])
    return None


# --------------------------------------------------------------------------- expectations


def permission_expectations(pyproject: Path | None) -> tuple[set[str], set[str], list[str], list[str]]:
    """Return (required, forbidden, notes, config_errors)."""
    required, forbidden = set(BASELINE_REQUIRED), set(BASELINE_FORBIDDEN)
    notes: list[str] = []
    config_errors: list[str] = []
    if pyproject is None:
        return required, forbidden, notes, config_errors
    if tomllib is None:
        notes.append("tomllib unavailable; using the built-in permission baseline only")
        return required, forbidden, notes, config_errors
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    table = data.get("tool", {}).get("flet", {}).get("android", {}).get("permission", {})
    for name, value in table.items():
        if value is False:
            forbidden.add(name)
        elif value:
            required.add(name)
    conflict = required & forbidden
    if conflict:
        config_errors.append(
            f"{pyproject} enables or disables permissions against the baseline: " + ", ".join(sorted(conflict))
        )
    return required - forbidden, forbidden, notes, config_errors


def abis_in_zip(path: Path) -> set[str]:
    with zipfile.ZipFile(path) as archive:
        return {m.group(1) for m in (LIB_RE.match(n) for n in archive.namelist()) if m}


def abi_from_filename(name: str) -> str | None:
    for abi in ("arm64-v8a", "armeabi-v7a", "x86_64"):
        if abi in name:
            return abi
    return None


# --------------------------------------------------------------------------- checks


def check_signature_apk(report: Report, apk: Path, build_tools: Path | None, expect_release: bool) -> None:
    apksigner = _tool(build_tools, "apksigner")
    if not apksigner:
        report.error("apksigner not found (set ANDROID_HOME or --build-tools)")
        return
    result = run([apksigner, "verify", "--verbose", "--print-certs", str(apk)])
    out = result.stdout + result.stderr
    if result.returncode != 0:
        report.error("apksigner verify failed: " + " ".join(out.split())[:400])
        return
    schemes = [s for s in ("v1", "v2", "v3", "v3.1", "v4")
               if re.search(rf"Verified using {re.escape(s)} scheme[^:]*:\s*true", out)]
    report.facts["signature schemes"] = ",".join(schemes) or "?"
    if not ({"v2", "v3", "v3.1"} & set(schemes)):
        report.error("APK is not signed with the v2 or v3 signature scheme")
    dn = re.search(r"certificate DN:\s*(.+)", out)
    signer = dn.group(1).strip() if dn else "?"
    report.facts["signer"] = signer
    if DEBUG_CERT_MARKER in signer:
        if expect_release:
            report.error("signed with the Android debug certificate although a release keystore was configured")
        else:
            report.facts["signing"] = "debug"
    else:
        report.facts["signing"] = "release"


def check_apk(report: Report, args, build_tools: Path | None, required: set[str], forbidden: set[str]) -> None:
    apk = report.path
    aapt2 = _tool(build_tools, "aapt2")
    if not aapt2:
        report.error("aapt2 not found (set ANDROID_HOME or --build-tools)")
        return

    badging_run = run([aapt2, "dump", "badging", str(apk)])
    if badging_run.returncode != 0:
        report.error("aapt2 dump badging failed: " + " ".join((badging_run.stderr or badging_run.stdout).split())[:400])
        return
    badging = parse_badging(badging_run.stdout)

    package = badging.get("name", "")
    report.facts["package"] = package
    if package != args.package:
        report.error(f"package is {package!r}, expected {args.package!r}")

    version_name = badging.get("versionName", "")
    report.facts["versionName"] = version_name
    if args.version_name and version_name != args.version_name:
        report.error(f"versionName is {version_name!r}, expected {args.version_name!r}")

    abis = abis_in_zip(apk)
    report.facts["ABIs"] = ",".join(sorted(abis)) or "none"
    allowed = set(args.allowed_abis)
    if not abis:
        report.error("APK contains no native libraries (lib/<abi>/*.so)")
    if abis - allowed:
        report.error("native libraries for disallowed ABIs: " + ", ".join(sorted(abis - allowed)))
    file_abi = abi_from_filename(apk.name)
    if file_abi and abis and abis != {file_abi}:
        report.error(f"file name says {file_abi} but the APK contains {', '.join(sorted(abis))}")

    try:
        version_code = int(badging.get("versionCode", ""))
    except ValueError:
        version_code = None
    report.facts["versionCode"] = str(version_code)
    if args.version_code is not None:
        accepted = {args.version_code}
        if len(abis) == 1:
            (only_abi,) = tuple(abis)
            if only_abi in ABI_VERSION_OFFSET:
                accepted.add(ABI_VERSION_OFFSET[only_abi] * 1000 + args.version_code)
        if version_code not in accepted:
            report.error(f"versionCode is {version_code}, expected one of {sorted(accepted)}")

    min_sdk = badging.get("minSdk", "")
    report.facts["minSdk"] = min_sdk
    report.facts["targetSdk"] = badging.get("targetSdk", "?")
    if args.min_sdk is not None and min_sdk != str(args.min_sdk):
        report.error(f"minSdk is {min_sdk or '?'}, expected {args.min_sdk}")

    perm_run = run([aapt2, "dump", "permissions", str(apk)])
    permissions = parse_permissions(perm_run.stdout) if perm_run.returncode == 0 else set()
    permissions |= badging["permissions"]
    report.facts["permissions"] = str(len(permissions))
    missing = sorted(required - permissions)
    present_forbidden = sorted(forbidden & permissions)
    if missing:
        report.error("missing permissions: " + ", ".join(missing))
    if present_forbidden:
        report.error("forbidden permissions present: " + ", ".join(present_forbidden))

    tree_run = run([aapt2, "dump", "xmltree", "--file", "AndroidManifest.xml", str(apk)])
    if tree_run.returncode != 0:
        report.error("aapt2 dump xmltree failed: " + " ".join((tree_run.stderr or tree_run.stdout).split())[:400])
    else:
        tree = parse_xmltree(tree_run.stdout)
        services = data_sync_services(tree)
        report.facts["dataSync services"] = ", ".join(services) or "none"
        if not services:
            report.error("no <service> declares foregroundServiceType=dataSync (job foreground service missing from the merged manifest)")
        if not has_deep_link(tree, args.url_scheme, args.url_host):
            report.error(f"no VIEW intent filter for {args.url_scheme}://{args.url_host}")
        debuggable = manifest_flag(tree, "application", "debuggable")
        if debuggable and debuggable.lower() in ("true", "0xffffffff", "-1"):
            report.warn("application is debuggable")

    check_signature_apk(report, apk, build_tools, args.expect_release)


def check_aab(report: Report, args) -> None:
    aab = report.path
    with zipfile.ZipFile(aab) as archive:
        names = set(archive.namelist())
    if "base/manifest/AndroidManifest.xml" not in names:
        report.error("base/manifest/AndroidManifest.xml missing (not an Android App Bundle?)")
    abis = abis_in_zip(aab)
    report.facts["ABIs"] = ",".join(sorted(abis)) or "none"
    if not abis:
        report.error("bundle contains no native libraries (base/lib/<abi>/*.so)")
    disallowed = abis - set(args.allowed_abis)
    if disallowed:
        report.error("native libraries for disallowed ABIs: " + ", ".join(sorted(disallowed)))

    jarsigner = shutil.which("jarsigner")
    if not jarsigner:
        report.warn("jarsigner not found; AAB signature not verified")
        return
    result = run([jarsigner, "-verify", "-certs", str(aab)])
    out = result.stdout + result.stderr
    if result.returncode != 0 or "jar verified" not in out:
        report.error("jarsigner -verify failed: " + " ".join(out.split())[:400])
        return
    keytool = shutil.which("keytool")
    if keytool:
        cert = run([keytool, "-printcert", "-jarfile", str(aab)])
        owner = re.search(r"Owner:\s*(.+)", cert.stdout)
        signer = owner.group(1).strip() if owner else "?"
        report.facts["signer"] = signer
        if args.expect_release and "CN=Android Debug" in signer:
            report.error("bundle is signed with the Android debug certificate")


# --------------------------------------------------------------------------- output


def _size_mb(path: Path) -> str:
    try:
        return f"{path.stat().st_size / 1_048_576:.1f} MB"
    except OSError:
        return "missing"


def emit(reports: list[Report], notes: list[str], config_errors: list[str]) -> None:
    in_actions = os.environ.get("GITHUB_ACTIONS") == "true"
    for note in notes:
        print(("::warning title=verify_apk::" if in_actions else "warning: ") + note)
    for error in config_errors:
        print(("::error title=verify_apk::" if in_actions else "ERROR: ") + error)
    for report in reports:
        status = "FAIL" if report.errors else "OK"
        print(f"\n[{status}] {report.path.name} ({_size_mb(report.path)})")
        for key, value in report.facts.items():
            print(f"    {key}: {value}")
        for warning in report.warnings:
            print(f"::warning title=verify_apk {report.path.name}::{warning}" if in_actions else f"    warning: {warning}")
        for error in report.errors:
            print(f"::error title=verify_apk {report.path.name}::{error}" if in_actions else f"    ERROR: {error}")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as stream:
            stream.write("## Android package verification\n\n| File | Result | Details |\n|---|---|---|\n")
            for report in reports:
                details = "; ".join(f"{k}: {v}" for k, v in report.facts.items())
                problems = "<br>".join(report.errors + report.warnings)
                stream.write(
                    f"| `{report.path.name}` | {'FAIL' if report.errors else 'OK'} | {details}"
                    f"{'<br>' + problems if problems else ''} |\n"
                )
            stream.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", type=Path, help="APK and/or AAB files")
    parser.add_argument("--package", default="com.glossarion.app")
    parser.add_argument("--version-code", type=int, help="base versionCode (build number)")
    parser.add_argument("--version-name", help="expected versionName")
    parser.add_argument("--min-sdk", type=int, default=26)
    parser.add_argument("--allowed-abis", type=lambda s: [a for a in s.split(",") if a], default=list(DEFAULT_ALLOWED_ABIS))
    parser.add_argument("--pyproject", type=Path, help="src/mobile/pyproject.toml for permission expectations")
    parser.add_argument("--url-scheme", default="glossarion")
    parser.add_argument("--url-host", default="app")
    parser.add_argument("--expect-release", action="store_true", help="fail when signed with the debug certificate")
    parser.add_argument("--max-size-mb", type=float, default=350.0, help="warn above this size")
    parser.add_argument("--build-tools", help="Android SDK build-tools directory (default: newest under ANDROID_HOME)")
    args = parser.parse_args(argv)

    required, forbidden, notes, config_errors = permission_expectations(args.pyproject)
    build_tools = find_build_tools(args.build_tools)
    if build_tools:
        print(f"Using build-tools {build_tools}")

    reports: list[Report] = []
    for path in args.files:
        report = Report(path)
        reports.append(report)
        if not path.is_file():
            report.error("file not found")
            continue
        if path.stat().st_size > args.max_size_mb * 1_048_576:
            report.warn(f"size {path.stat().st_size / 1_048_576:.1f} MB exceeds the {args.max_size_mb:.0f} MB budget")
        suffix = path.suffix.lower()
        if suffix == ".apk":
            check_apk(report, args, build_tools, required, forbidden)
        elif suffix == ".aab":
            check_aab(report, args)
        else:
            report.error("unsupported file type (expected .apk or .aab)")

    emit(reports, notes, config_errors)
    return 1 if config_errors or any(r.errors for r in reports) else 0


if __name__ == "__main__":
    sys.exit(main())
