#!/usr/bin/env python3
"""Verify a Glossarion iOS build (IPA, .app bundle or .xcarchive) before it is uploaded.

Checks on the app's Info.plist:
  * CFBundleIdentifier, CFBundleShortVersionString (build version) and CFBundleVersion (build number);
  * the URL scheme ``glossarion`` (CFBundleURLTypes, host recorded as CFBundleURLName);
  * CFBundleDocumentTypes / UTImportedTypeDeclarations: every content type and imported UTI
    declared in ``[tool.flet.ios.info]`` of pyproject.toml, plus a built-in minimum
    (EPUB, PDF, plain text, ZIP);
  * UIBackgroundModes contains ``processing`` and BGTaskSchedulerPermittedIdentifiers holds a
    ``com.glossarion.app.job.`` identifier (iOS 26 BGContinuedProcessingTask);
  * UIFileSharingEnabled / LSSupportsOpeningDocumentsInPlace, NSLocalNetworkUsageDescription;
  * bundle layout: executable, Frameworks/Flutter.framework and Frameworks/App.framework;
  * signing state: ``--expect-signed`` needs _CodeSignature + embedded.mobileprovision,
    ``--expect-unsigned`` warns when a signature is present;
  * the engine's optional NPU backend (wheelhouse.toml [ane], ``--wheelhouse``): when enabled the bundle
    carries the framework serious-python makes from libggml-ane and the marker llama_cpp/npu-backend.txt,
    and when switched off neither; never a copy of Apple's private AppleNeuralEngine framework; and never in an
    App Store / TestFlight build (``--export-method`` app-store / app-store-connect, or an embedded profile without
    a device list): that fails, any other signed build only warns.

Info.plist is read with plistlib (XML or binary); on macOS ``plutil -lint`` also runs.
Exit status 1 when any check fails.
"""
from __future__ import annotations

import argparse
import io
import os
import plistlib
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - CI runs 3.13
    tomllib = None

MIN_CONTENT_TYPES = ("org.idpf.epub-container", "com.adobe.pdf", "public.plain-text", "public.zip-archive")
DEFAULT_WHEELHOUSE = Path(__file__).resolve().parent / "wheels" / "wheelhouse.toml"
NPU_MARKER = "llama_cpp/npu-backend.txt"
ANE_LIBRARY = "libggml-ane"
PRIVATE_ANE = "AppleNeuralEngine"


class Bundle:
    """Uniform read access to an .app inside an IPA, an .xcarchive or a directory."""

    def __init__(self, source: Path):
        self.source = source
        self._zip: zipfile.ZipFile | None = None
        self.app_root = ""  # posix path of the .app inside the zip, or filesystem path
        self.names: list[str] = []
        if source.is_file():
            self._zip = zipfile.ZipFile(source)
            apps = sorted({
                str(PurePosixPath(*PurePosixPath(n).parts[:2]))
                for n in self._zip.namelist()
                if n.startswith("Payload/") and len(PurePosixPath(n).parts) >= 2
                and PurePosixPath(n).parts[1].endswith(".app")
            })
            if len(apps) != 1:
                raise ValueError(f"expected exactly one Payload/*.app in {source.name}, found {len(apps)}")
            self.app_root = apps[0]
            prefix = self.app_root + "/"
            self.names = [n[len(prefix):] for n in self._zip.namelist() if n.startswith(prefix)]
        else:
            app = self._find_app_dir(source)
            if app is None:
                raise ValueError(f"no .app bundle found under {source}")
            self.app_root = str(app)
            self.names = [p.relative_to(app).as_posix() + ("/" if p.is_dir() else "") for p in app.rglob("*")]

    @staticmethod
    def _find_app_dir(source: Path) -> Path | None:
        if source.suffix == ".app":
            return source
        archived = sorted(source.glob("**/Products/Applications/*.app"))
        if archived:
            return archived[0]
        apps = sorted(source.glob("**/*.app"))
        return apps[0] if apps else None

    @property
    def app_name(self) -> str:
        return PurePosixPath(self.app_root).name

    def exists(self, rel: str) -> bool:
        rel = rel.rstrip("/")
        return any(n.rstrip("/") == rel or n.startswith(rel + "/") for n in self.names)

    def read(self, rel: str) -> bytes:
        if self._zip is not None:
            return self._zip.read(f"{self.app_root}/{rel}")
        return (Path(self.app_root) / rel).read_bytes()


def load_expectations(pyproject: Path | None) -> tuple[set[str], set[str], list[str]]:
    content_types = set(MIN_CONTENT_TYPES)
    imported: set[str] = set()
    notes: list[str] = []
    if pyproject is None:
        return content_types, imported, notes
    if tomllib is None:
        notes.append("tomllib unavailable; using the built-in document-type minimum only")
        return content_types, imported, notes
    info = (tomllib.loads(pyproject.read_text(encoding="utf-8"))
            .get("tool", {}).get("flet", {}).get("ios", {}).get("info", {}))
    for doc_type in info.get("CFBundleDocumentTypes", []):
        content_types.update(doc_type.get("LSItemContentTypes", []))
    for uti in info.get("UTImportedTypeDeclarations", []):
        if uti.get("UTTypeIdentifier"):
            imported.add(uti["UTTypeIdentifier"])
    return content_types, imported, notes


def ane_enabled(wheelhouse: Path | None) -> bool:
    """wheelhouse.toml [ane] enabled = true: the iOS engine carries the NPU backend (libggml-ane)."""
    if wheelhouse is None or tomllib is None or not wheelhouse.is_file():
        return False
    data = tomllib.loads(wheelhouse.read_text(encoding="utf-8"))
    return (data.get("ane") or {}).get("enabled") is True


#: Xcode export methods of an App Store / TestFlight upload (old and new names)
STORE_EXPORT_METHODS = ("app-store", "app-store-connect")


def profile_kind(data: bytes) -> str | None:
    """The kind of an embedded.mobileprovision (a CMS-signed XML plist): "store" (no device list: App Store /
    TestFlight), "enterprise" (ProvisionsAllDevices), "development" (devices and get-task-allow) or "ad-hoc"
    (devices); None when it cannot be read."""
    start, end = data.find(b"<?xml"), data.find(b"</plist>")
    if start < 0 or end < start:
        return None
    try:
        plist = plistlib.loads(data[start:end + len(b"</plist>")])
    except Exception:
        return None
    if not isinstance(plist, dict):
        return None
    if plist.get("ProvisionsAllDevices"):
        return "enterprise"
    if plist.get("ProvisionedDevices"):
        return "development" if (plist.get("Entitlements") or {}).get("get-task-allow") else "ad-hoc"
    return "store"


def _flat(name: str) -> str:
    """A name with case and the separators serious-python may change (-, _, .) removed."""
    return "".join(c for c in name.lower() if c not in "-_.")


def bundle_paths(bundle: "Bundle") -> list[str]:
    """Every file of the bundle, plus the members of each zip in it as ``<zip>!<member>`` (site-packages may be
    packed)."""
    out = [n for n in bundle.names if not n.endswith("/")]
    for name in [n for n in out if n.endswith(".zip")]:
        try:
            with zipfile.ZipFile(io.BytesIO(bundle.read(name))) as inner:
                out += [f"{name}!{m}" for m in inner.namelist() if not m.endswith("/")]
        except (zipfile.BadZipFile, KeyError, OSError):
            continue
    return out


def ane_problems(paths: list[str], enabled: bool) -> tuple[list[str], list[str], dict[str, str]]:
    """(errors, warnings, facts) of the iOS NPU backend in the bundle's ``paths`` (bundle_paths): with [ane] enabled,
    the framework made from libggml-ane (or its .fwork marker) and llama_cpp/npu-backend.txt; switched off, neither;
    never an AppleNeuralEngine binary (the private framework is the system's, dlopened at run time)."""
    errors: list[str] = []
    warnings: list[str] = []
    want = _flat(ANE_LIBRARY)
    library = sorted(p for p in paths if want in _flat(p.rsplit("!", 1)[-1])
                     and (".framework/" in p or p.endswith(".fwork") or p.endswith(".dylib")))
    markers = sorted(p for p in paths if p.rsplit("!", 1)[-1].endswith(NPU_MARKER))
    engine = any("llama_cpp/" in p for p in paths)
    private = sorted(p for p in paths if PRIVATE_ANE.lower() in p.rsplit("/", 1)[-1].lower()
                     or f"{PRIVATE_ANE}.framework/".lower() in p.lower())
    facts = {"NPU backend (wheelhouse.toml [ane])": ("enabled" if enabled else "switched off") +
             (f"; {len(library)} libggml-ane file(s), marker {'found' if markers else 'not found'}" if engine else "")}
    if private:
        errors.append(f"the bundle carries {private[:4]}: Apple's private AppleNeuralEngine framework may not ship "
                      f"(the NPU backend dlopens the system's at run time)")
    if not engine:
        warnings.append("no llama_cpp package found in the bundle (or its zips): the NPU backend was not checked")
        return errors, warnings, facts
    if enabled:
        if not library:
            errors.append("no framework made from libggml-ane in the bundle although wheelhouse.toml [ane] is enabled")
        if not markers:
            errors.append(f"no {NPU_MARKER} in the bundle although wheelhouse.toml [ane] is enabled")
    else:
        if library:
            errors.append(f"{library[:4]} ship but wheelhouse.toml [ane] is switched off")
        if markers:
            errors.append(f"{markers} ships but wheelhouse.toml [ane] is switched off")
    return errors, warnings, facts


def plutil_lint(data: bytes) -> str | None:
    plutil = shutil.which("plutil")
    if not plutil or sys.platform != "darwin":
        return None
    with tempfile.NamedTemporaryFile(suffix=".plist", delete=False) as handle:
        handle.write(data)
        path = handle.name
    try:
        result = subprocess.run([plutil, "-lint", path], capture_output=True, text=True, check=False)
        return None if result.returncode == 0 else (result.stdout + result.stderr).strip()
    finally:
        os.unlink(path)


def verify(args) -> tuple[list[str], list[str], dict[str, str]]:
    errors: list[str] = []
    warnings: list[str] = []
    facts: dict[str, str] = {}

    try:
        bundle = Bundle(args.path)
    except (ValueError, zipfile.BadZipFile, OSError) as exc:
        return [str(exc)], warnings, facts
    facts["app"] = bundle.app_name

    try:
        raw = bundle.read("Info.plist")
    except (KeyError, OSError):
        return [f"{bundle.app_name}/Info.plist is missing"], warnings, facts
    lint = plutil_lint(raw)
    if lint:
        errors.append(f"plutil -lint failed: {lint}")
    try:
        info = plistlib.loads(raw)
    except Exception as exc:  # plistlib raises several types for malformed input
        return errors + [f"Info.plist is not a valid property list: {exc}"], warnings, facts

    def expect_equal(key: str, expected: str | None) -> None:
        actual = info.get(key)
        facts[key] = str(actual)
        if expected is not None and str(actual) != expected:
            errors.append(f"{key} is {actual!r}, expected {expected!r}")

    expect_equal("CFBundleIdentifier", args.bundle_id)
    expect_equal("CFBundleShortVersionString", args.version)
    expect_equal("CFBundleVersion", args.build_number)
    facts["MinimumOSVersion"] = str(info.get("MinimumOSVersion", "?"))

    executable = info.get("CFBundleExecutable")
    if not executable or not bundle.exists(str(executable)):
        errors.append(f"CFBundleExecutable {executable!r} is not in the bundle")
    for framework in ("Frameworks/Flutter.framework", "Frameworks/App.framework"):
        if not bundle.exists(framework):
            errors.append(f"{framework} is missing from the bundle")

    # URL scheme (deep links glossarion://app/<route>)
    url_types = info.get("CFBundleURLTypes") or []
    matching = [t for t in url_types if args.url_scheme in (t.get("CFBundleURLSchemes") or [])]
    facts["URL schemes"] = ",".join(s for t in url_types for s in (t.get("CFBundleURLSchemes") or [])) or "none"
    if not matching:
        errors.append(f"CFBundleURLTypes has no {args.url_scheme!r} scheme")
    elif args.url_host and not any(t.get("CFBundleURLName") == args.url_host for t in matching):
        warnings.append(f"the {args.url_scheme!r} URL type is not named {args.url_host!r} (deep-link host)")

    # Document types and imported UTIs
    expected_types, expected_utis, notes = load_expectations(args.pyproject)
    warnings.extend(notes)
    declared_types = {ct for doc in (info.get("CFBundleDocumentTypes") or []) for ct in (doc.get("LSItemContentTypes") or [])}
    facts["document content types"] = str(len(declared_types))
    missing_types = sorted(expected_types - declared_types)
    if missing_types:
        errors.append("CFBundleDocumentTypes is missing content types: " + ", ".join(missing_types))
    declared_utis = {u.get("UTTypeIdentifier") for u in (info.get("UTImportedTypeDeclarations") or [])}
    missing_utis = sorted(expected_utis - declared_utis)
    if missing_utis:
        errors.append("UTImportedTypeDeclarations is missing: " + ", ".join(missing_utis))

    # Background processing (BGContinuedProcessingTask identifiers)
    modes = info.get("UIBackgroundModes") or []
    if "processing" not in modes:
        errors.append(f"UIBackgroundModes {modes!r} does not include 'processing'")
    bg_ids = info.get("BGTaskSchedulerPermittedIdentifiers") or []
    facts["BG task ids"] = ",".join(bg_ids) or "none"
    if not any(str(i).startswith(args.bg_prefix) for i in bg_ids):
        errors.append(f"BGTaskSchedulerPermittedIdentifiers {bg_ids!r} has no {args.bg_prefix}* identifier")
    foreign = [i for i in bg_ids if args.bundle_id and not str(i).startswith(args.bundle_id + ".")]
    if foreign:
        errors.append("BGTaskSchedulerPermittedIdentifiers outside the bundle id: " + ", ".join(foreign))

    # Files app integration and local network
    for key in ("UIFileSharingEnabled", "LSSupportsOpeningDocumentsInPlace"):
        if info.get(key) is not True:
            errors.append(f"{key} is {info.get(key)!r}, expected true")
    if not info.get("NSLocalNetworkUsageDescription"):
        errors.append("NSLocalNetworkUsageDescription is missing (LAN Ollama / LM Studio, OAuth loopback)")

    # Signing state
    signed = bundle.exists("_CodeSignature/CodeResources")
    provisioned = bundle.exists("embedded.mobileprovision")
    facts["signed"] = f"{signed} (provisioning profile: {provisioned})"
    if args.expect_signed and not (signed and provisioned):
        errors.append("expected a signed IPA with an embedded provisioning profile")
    if args.expect_unsigned and (signed or provisioned):
        warnings.append("an unsigned build was expected but the bundle carries a signature or profile")

    # The engine's optional NPU backend (Apple Neural Engine, wheelhouse.toml [ane])
    ane_errors, ane_warnings, ane_facts = ane_problems(
        bundle_paths(bundle), ane_enabled(getattr(args, "wheelhouse", DEFAULT_WHEELHOUSE)))
    errors.extend(ane_errors)
    warnings.extend(ane_warnings)
    facts.update(ane_facts)
    if ane_enabled(getattr(args, "wheelhouse", DEFAULT_WHEELHOUSE)) and (signed or args.expect_signed):
        method = str(getattr(args, "export_method", None) or "")
        kind = profile_kind(bundle.read("embedded.mobileprovision")) if provisioned else None
        facts["provisioning profile kind"] = kind or "unknown"
        if method in STORE_EXPORT_METHODS or kind == "store":
            errors.append(f"an App Store / TestFlight build (export method {method or '?'}, profile {kind or '?'}) "
                          f"carries the iOS NPU backend (wheelhouse.toml [ane]: Apple's private AppleNeuralEngine "
                          f"framework): never ship that signed for the store; switch [ane] off for this build")
        else:
            warnings.append("a signed build carries the iOS NPU backend (wheelhouse.toml [ane]: Apple's private "
                            "AppleNeuralEngine framework); App Store / TestFlight review refuses that")

    try:
        facts["size"] = f"{args.path.stat().st_size / 1_048_576:.1f} MB" if args.path.is_file() else "dir"
    except OSError:
        pass
    return errors, warnings, facts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path, help=".ipa, .app or .xcarchive (or a directory containing one)")
    parser.add_argument("--bundle-id", default="com.glossarion.app")
    parser.add_argument("--version", help="expected CFBundleShortVersionString")
    parser.add_argument("--build-number", help="expected CFBundleVersion")
    parser.add_argument("--pyproject", type=Path, help="src/mobile/pyproject.toml for document-type expectations")
    parser.add_argument("--url-scheme", default="glossarion")
    parser.add_argument("--url-host", default="app")
    parser.add_argument("--bg-prefix", default="com.glossarion.app.job.")
    parser.add_argument("--wheelhouse", type=Path, default=DEFAULT_WHEELHOUSE,
                        help="wheelhouse.toml: [ane] says whether the engine carries the iOS NPU backend")
    parser.add_argument("--export-method", help="the Xcode export method of a signed build: app-store / "
                                                "app-store-connect with [ane] enabled fails")
    signing = parser.add_mutually_exclusive_group()
    signing.add_argument("--expect-signed", action="store_true")
    signing.add_argument("--expect-unsigned", action="store_true")
    args = parser.parse_args(argv)

    errors, warnings, facts = verify(args)
    in_actions = os.environ.get("GITHUB_ACTIONS") == "true"
    status = "FAIL" if errors else "OK"
    print(f"[{status}] {args.path.name}")
    for key, value in facts.items():
        print(f"    {key}: {value}")
    for warning in warnings:
        print(f"::warning title=verify_ipa {args.path.name}::{warning}" if in_actions else f"    warning: {warning}")
    for error in errors:
        print(f"::error title=verify_ipa {args.path.name}::{error}" if in_actions else f"    ERROR: {error}")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as stream:
            stream.write(f"## iOS bundle verification: `{args.path.name}` {status}\n\n| Key | Value |\n|---|---|\n")
            for key, value in facts.items():
                stream.write(f"| {key} | `{value}` |\n")
            for problem in errors + warnings:
                stream.write(f"| problem | {problem} |\n")
            stream.write("\n")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
