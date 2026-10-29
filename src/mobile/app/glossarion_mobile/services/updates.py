"""Update checks for About › Updates (UI_SPEC §4.16 "Updates"; FEATURE_MAP "Update checker").

The release check is the desktop's GitHub-primary/GitLab-fallback checker:
``update_core.HeadlessUpdateChecker`` runs the shared
``UpdateManager`` logic (``_check_for_updates_core``: 30-minute cache, ``skipped_versions``,
``/releases/latest`` + history, version compare; ``_skip_latest_version``) against the mobile
``config.json`` through :class:`StoreConfigHost`, so desktop and mobile share
``auto_update_check``, ``last_update_check_time`` and ``skipped_versions``.

What is mobile-only is the asset choice. Mobile builds are never published (owner's rule,
2026-10-09): ``build-mobile.yml`` only keeps the APK/IPA as run artifacts, so GitHub releases
are desktop-only and "no file for this phone" is the ordinary answer, never an error. A release
asset is offered only when it carries a name the build jobs write:

    [X_]<prefix>_Android_<abi>[_debugsigned].apk   Android, one per ABI (arm64-v8a, x86_64)
    <prefix>_Android.aab                       Play bundle (never offered in the app)
    [X_]<prefix>_iOS_unsigned.ipa              unsigned IPA (sideloading)
    [X_]<prefix>_iOS.ipa                       signed IPA (devices in the provisioning profile)

A mobile hotfix (owner's process) adds ``X_Glossarion_v<release>_hotfixN_<platform>`` files to the
release it fixes: for this device the newest hotfix of that release is the file to install. The
installed build is recognised by its bytes: Android keeps the installed APK as ``base.apk`` (its
path is in ``/proc/self/maps``), and when its size and SHA-256 equal one of the release's APKs
(GitHub publishes each asset's ``digest``), the app knows which file it runs and offers a newer
hotfix of the same release.

Android installs in the app (:class:`ApkInstaller`): the APK is downloaded with the desktop's
verified downloader (``update_core._download_verified_update``: exact published size and SHA-256)
into ``<cache>/updates``, then handed to the system installer through the extension
(``install_package``: a FileProvider ``content://`` URI and ``ACTION_VIEW``). A sideloaded app can
never update itself silently: Android shows its own "Do you want to update this app?" screen, and the
first time it asks the user to allow installs from Glossarion (``open_install_settings``). iOS keeps
the download links: only the App Store, TestFlight or the sideloading tool can replace an iOS app.
Importing this module imports no backend module.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

__all__ = [
    "ApkInstaller", "MobileAsset", "MobileDownloads", "StoreConfigHost", "UPDATE_CONFIG_KEYS", "UPDATES_SUBDIR",
    "UpdateResult", "UpdateService", "classify_asset", "hotfix_number", "own_apk_path", "releases_page",
    "select_downloads",
]

log = logging.getLogger("glossarion.updates")

#: config.json keys the moved desktop methods read and write (shared with the desktop).
UPDATE_CONFIG_KEYS = ("auto_update_check", "last_update_check_time", "skipped_versions")
#: Under the app's cache folder; the extension's FileProvider shares only this folder
#: (``res/xml/glossarion_native_update_paths.xml``) and installs only APKs from it.
UPDATES_SUBDIR = "updates"
#: Next to a downloaded APK: what it was verified against (a later visit installs it without a new download).
VERIFIED_SUFFIX = ".verified.json"

_APK = re.compile(r"_android_(?P<abi>arm64-v8a|x86_64|armeabi-v7a|x86)(?P<debug>_debugsigned)?\.apk$", re.I)
_HOTFIX = re.compile(r"_hotfix(?P<n>\d+)_", re.I)
_DIGEST = re.compile(r"sha256:(?P<hex>[0-9a-f]{64})", re.I)


def hotfix_number(name: str) -> int:
    """``N`` of a ``..._hotfixN_...`` file (the owner's mobile hotfixes), 0 for the release's own file."""
    match = _HOTFIX.search(os.path.basename(str(name or "")))
    return int(match.group("n")) if match else 0


def own_apk_path(maps: str = "/proc/self/maps") -> Optional[str]:
    """The installed APK this process runs from (Android maps ``/data/app/.../base.apk``), else None."""
    try:
        with open(maps, encoding="utf-8", errors="replace") as stream:
            for line in stream:
                parts = line.split(None, 5)
                path = parts[5].strip() if len(parts) == 6 else ""
                if path.startswith("/data/app/") and path.endswith("/base.apk"):
                    return path
    except OSError:
        return None
    return None


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _core() -> Any:
    import update_core  # shared GUI-free core (U9; moved from update_manager.UpdateManager)

    return update_core


def classify_asset(name: str) -> Optional[str]:
    """``apk`` / ``aab`` / ``ipa`` (unsigned) / ``ipa_signed`` for a mobile build file (the names
    the build-mobile.yml build jobs write), else None (desktop files and anything else)."""
    lower = os.path.basename(str(name or "")).lower()
    if "glossarion" not in lower:
        return None
    if _APK.search(lower):
        return "apk"
    if lower.endswith("_android.aab"):
        return "aab"
    if lower.endswith("_ios_unsigned.ipa"):
        return "ipa"
    if lower.endswith("_ios.ipa"):
        return "ipa_signed"
    return None


@dataclass(frozen=True)
class MobileAsset:
    kind: str
    name: str
    url: str
    size: int = 0
    abi: str = ""  # Android ABI from the file name
    arch: str = "unknown"  # update_core naming: arm64 / x64 / unknown
    debug_signed: bool = False
    hotfix: int = 0  # N of an ``_hotfixN_`` file, 0 for the release's own file
    sha256: str = ""  # GitHub's published ``digest`` (lowercase hex), "" when the provider gave none

    @property
    def size_mb(self) -> float:
        return self.size / (1024 * 1024) if self.size else 0.0


@dataclass
class MobileDownloads:
    """What a release offers this device (``platform``: android / ios / other)."""

    platform: str
    apk: Optional[MobileAsset] = None  # the APK for this device's ABI (release-signed preferred)
    other_apks: list = field(default_factory=list)
    ipa: Optional[MobileAsset] = None
    ipa_signed: Optional[MobileAsset] = None

    @property
    def available(self) -> bool:
        if self.platform == "android":
            return self.apk is not None or bool(self.other_apks)
        if self.platform == "ios":
            return self.ipa is not None or self.ipa_signed is not None
        return False


def _asset(raw: dict) -> Optional[MobileAsset]:
    name = str(raw.get("name") or "")
    kind = classify_asset(name)
    url = str(raw.get("browser_download_url") or "")
    if kind is None or not url.startswith("https://"):
        return None
    try:
        size = int(raw.get("size") or 0)
    except (TypeError, ValueError):
        size = 0
    abi, debug = "", False
    match = _APK.search(name.lower())
    if match:
        abi, debug = match.group("abi"), bool(match.group("debug"))
    arch = _core().UpdateCoreMixin._asset_arch(name) if kind == "apk" else "unknown"
    digest = _DIGEST.fullmatch(str(raw.get("digest") or ""))
    return MobileAsset(kind=kind, name=name, url=url, size=size, abi=abi, arch=arch, debug_signed=debug,
                       hotfix=hotfix_number(name), sha256=digest.group("hex").lower() if digest else "")


def select_downloads(release: Optional[dict], platform: str, arch: str) -> MobileDownloads:
    """The mobile files of ``release`` for this device. ``arch`` is ``update_core``'s
    ``_detect_arch()`` (arm64 / x64 / unknown); an APK for another ABI is listed, never picked.
    Of this ABI's APKs the release-signed newest hotfix wins (the release's own file is hotfix 0)."""
    platform = str(platform or "").lower()
    out = MobileDownloads(platform=platform if platform in ("android", "ios") else "other")
    assets = [a for a in (_asset(raw) for raw in (release or {}).get("assets") or [] if isinstance(raw, dict)) if a]
    apks = [a for a in assets if a.kind == "apk"] if out.platform == "android" else []
    matching = [a for a in apks if arch != "unknown" and a.arch == arch]
    if matching:
        matching.sort(key=lambda a: (a.debug_signed, -a.hotfix, a.name))  # release-signed, newest hotfix first
        out.apk = matching[0]
    out.other_apks = [a for a in apks if a is not out.apk]
    for asset in assets:
        if asset.kind == "ipa" and out.ipa is None:
            out.ipa = asset
        elif asset.kind == "ipa_signed" and out.ipa_signed is None:
            out.ipa_signed = asset
    return out


def releases_page(provider: str = 'github') -> str:
    """``https://github.com/<owner>/<repo>/releases`` from the desktop's API URL."""
    if provider == 'gitlab':
        return 'https://gitlab.com/shirochi-stack/Glossarion/-/releases'
    api = _core().UpdateCoreMixin.GITHUB_API_URL  # https://api.github.com/repos/<owner>/<repo>/releases
    repo = api.split("/repos/", 1)[-1].rsplit("/releases", 1)[0]
    return f"https://github.com/{repo}/releases"


_CHECKER_CLASS: Any = None


def _mobile_checker(host: Any, current_version: str, arch: str) -> Any:
    """``update_core.HeadlessUpdateChecker`` whose downloader accepts this phone's APK (the desktop's
    ``_compatible_update_asset`` knows desktop installers only; the mobile check never calls it)."""
    global _CHECKER_CLASS
    if _CHECKER_CLASS is None:
        core = _core()

        class MobileUpdateChecker(core.HeadlessUpdateChecker):
            mobile_arch = "unknown"

            def _compatible_update_asset(self, asset: Any) -> bool:
                name = str((asset or {}).get("name") or "")
                return (classify_asset(name) == "apk" and self.mobile_arch != "unknown"
                        and core.UpdateCoreMixin._asset_arch(name) == self.mobile_arch)

        _CHECKER_CLASS = MobileUpdateChecker
    checker = _CHECKER_CLASS(host, current_version, build_variant="Mobile")
    checker.mobile_arch = arch
    return checker


class StoreConfigHost:
    """The ``main_gui`` the moved desktop methods use, over ``MobileConfigStore``: ``config`` holds
    the update keys present in config.json and ``save_config`` writes them back (debounced save;
    MobileConfigStore is thread-safe, the checks run on a worker thread)."""

    def __init__(self, store: Any) -> None:
        self.store = store
        self.config: dict = {}
        self.reload()

    def reload(self) -> None:
        config = {}
        for key in UPDATE_CONFIG_KEYS:
            try:
                if self.store is not None and self.store.has(key):
                    config[key] = self.store.get(key)
            except Exception:
                continue
        self.config = config

    def save_config(self, show_message: bool = False) -> None:
        if self.store is None:
            return
        values = {key: self.config[key] for key in UPDATE_CONFIG_KEYS if key in self.config}
        if values:
            self.store.set_many(values)


@dataclass
class UpdateResult:
    status: str  # update / hotfix / current / skipped / cached / error
    message: str = ""
    tag: str = ""
    notes: str = ""
    html_url: str = ""
    published: str = ""
    downloads: Optional[MobileDownloads] = None
    checked_at: float = 0.0
    #: The release file this phone runs (recognised by size + SHA-256), "" when unknown.
    installed: str = ""

    @property
    def has_release(self) -> bool:
        return bool(self.tag)

    @property
    def install_asset(self) -> Optional[MobileAsset]:
        """The APK worth installing: a newer release, a newer hotfix, or a hotfix of this release when
        the installed build is not one of the release's files (a CI build); None otherwise."""
        apk = self.downloads.apk if self.downloads is not None else None
        if apk is None or apk.name == self.installed:
            return None
        if self.status in ("update", "hotfix"):
            return apk
        return apk if apk.hotfix > 0 and not self.installed and self.status == "current" else None


class UpdateService:
    """One per app session: the desktop checker on the mobile config (blocking calls; run them
    on a worker thread)."""

    def __init__(self, store: Any, current_version: str, *, platform: str, arch: Optional[str] = None,
                 clock: Callable[[], float] = time.time,
                 installed_apk: Optional[Callable[[], Optional[str]]] = None) -> None:
        core = _core()
        self.host = StoreConfigHost(store)
        self.platform = str(platform or "").lower()
        self.arch = arch or core.UpdateCoreMixin._detect_arch()
        self.checker = _mobile_checker(self.host, current_version, self.arch)
        self.clock = clock
        #: Where the installed APK is (``own_apk_path`` on Android); None skips the hotfix recognition.
        self.installed_apk = installed_apk if installed_apk is not None else (
            own_apk_path if self.platform == "android" else (lambda: None))
        self._digests: dict = {}  # (path, size, mtime_ns) -> sha256 of the installed APK
        self.last: Optional[UpdateResult] = None

    @property
    def current_version(self) -> str:
        return self.checker.CURRENT_VERSION

    def startup_enabled(self) -> bool:
        self.host.reload()
        return bool(self.host.config.get("auto_update_check", True))  # the desktop default

    def set_startup(self, enabled: bool) -> None:
        self.host.config["auto_update_check"] = bool(enabled)
        self.host.save_config(show_message=False)

    def last_checked(self) -> float:
        self.host.reload()
        try:
            return float(self.host.config.get("last_update_check_time") or 0)
        except (TypeError, ValueError):
            return 0.0

    def skipped_versions(self) -> list:
        self.host.reload()
        return list(self.host.config.get("skipped_versions") or [])

    def check(self, *, manual: bool) -> UpdateResult:
        """Blocking. ``manual`` (Check now) ignores the 30-minute cache and the skipped versions,
        like the desktop's Help › Check for updates; the startup check honours both."""
        import requests

        self.host.reload()
        checker = self.checker
        checker._last_check_time = float(self.host.config.get("last_update_check_time") or 0)
        before = checker.latest_release
        checker.latest_release = None
        try:
            available, release = checker._check_for_updates_core(force_show=manual)
        except requests.Timeout:
            return self._remember(UpdateResult("error", "Connection timed out while checking for updates."))
        except requests.ConnectionError:
            return self._remember(UpdateResult("error", "Cannot reach update servers. Check the connection and try again."))
        except requests.HTTPError as exc:
            code = getattr(getattr(exc, "response", None), "status_code", None)
            text = ("Update API rate limit exceeded. Please try again later." if code == 403
                    else f"Update service returned error: {code}")
            return self._remember(UpdateResult("error", text))
        except ValueError as exc:
            return self._remember(UpdateResult("error", f"Invalid response from the update service: {exc}"))
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("update check failed")
            return self._remember(UpdateResult("error", f"An unexpected error occurred: {exc}"))
        latest = checker.latest_release
        if latest is None:  # inside the 30-minute window: no request was made
            checker.latest_release = before
            return self._remember(UpdateResult("cached", "Checked less than 30 minutes ago."), keep_last=True)
        result = self._describe(latest)
        if available:
            result.status = "update"
        elif str(latest.get("tag_name") or "") in self.skipped_versions() and not manual:
            result.status = "skipped"
        else:
            result.status = "current"
        if result.status == "current":
            self._match_hotfix(result)
        result.message = self._message(result)
        return self._remember(result)

    # ---- Android: hotfixes and the in-app download --------------------------------------------

    def _match_hotfix(self, result: UpdateResult) -> None:
        """This build's release: which of its APKs is installed, and is a newer hotfix out?"""
        downloads = result.downloads
        apk = downloads.apk if downloads is not None else None
        if apk is None or not self._same_version(result.tag):
            return
        installed = self.installed_asset([apk] + list(downloads.other_apks))
        if installed is not None:
            result.installed = installed.name
            if installed.hotfix < apk.hotfix:
                result.status = "hotfix"

    def _same_version(self, tag: str) -> bool:
        try:
            return self.checker._release_version(tag) == self.checker._release_version(self.current_version)
        except Exception:
            return False

    def installed_asset(self, apks: list) -> Optional[MobileAsset]:
        """The APK of ``apks`` whose bytes are the installed app's (blocking: hashes ~300 MB once)."""
        try:
            path = self.installed_apk()
            stat = os.stat(path) if path else None
        except OSError:
            stat = None
        if not path or stat is None:
            return None
        candidates = [a for a in apks if a.sha256 and a.size == stat.st_size]
        if not candidates:
            return None
        key = (path, stat.st_size, stat.st_mtime_ns)
        digest = self._digests.get(key)
        if digest is None:
            try:
                digest = _sha256_file(path)
            except OSError as exc:
                log.info("cannot read the installed APK: %s", exc)
                return None
            self._digests = {key: digest}
        return next((a for a in candidates if a.sha256 == digest), None)

    def ready_apk(self, asset: MobileAsset, folder: str) -> Optional[str]:
        """The verified download of ``asset`` already in ``folder`` (a visit before), else None."""
        path = os.path.join(folder, asset.name)
        try:
            with open(path + VERIFIED_SUFFIX, encoding="utf-8") as stream:
                marker = json.load(stream)
            if (os.path.getsize(path) == int(marker.get("size") or -1) and marker.get("name") == asset.name
                    and (not asset.sha256 or marker.get("sha256") == asset.sha256)):
                return path
        except (OSError, ValueError, TypeError, AttributeError):
            return None
        return None

    def prune_downloads(self, folder: str, keep: str = "") -> None:
        """Remove downloaded APKs (and partial files) other than ``keep`` from ``folder``."""
        try:
            names = os.listdir(folder)
        except OSError:
            return
        for name in names:
            if keep and name in (keep, keep + VERIFIED_SUFFIX):
                continue
            try:
                os.remove(os.path.join(folder, name))
            except OSError:
                pass

    def download_apk(self, asset: MobileAsset, folder: str, progress: Callable[[int, float, float], Any],
                     cancelled: Callable[[], bool]) -> str:
        """Blocking: download ``asset`` of the checked release into ``folder`` with the desktop's
        verified downloader (exact published size + SHA-256, never installed on a mismatch). The path."""
        release = self.checker.latest_release or {}
        raw = next((a for a in release.get("assets") or [] if isinstance(a, dict) and a.get("name") == asset.name), None)
        if raw is None:
            raise ValueError("Check for updates again: this file is not in the release Glossarion checked.")
        os.makedirs(folder, exist_ok=True)
        self.prune_downloads(folder)
        path = os.path.join(folder, asset.name)
        prepared = self.checker._download_verified_update(raw, release, path, progress, cancelled)
        with open(path + VERIFIED_SUFFIX, "w", encoding="utf-8") as stream:
            json.dump({"name": asset.name, "size": prepared["size"], "sha256": prepared["_sha256"],
                       "tag": prepared.get("_release_tag", "")}, stream)
        return path

    def skip(self) -> Optional[str]:
        """Skip the latest release (the desktop's Skip This Version); its tag, or None."""
        if not self.checker.latest_release:
            return None
        self.host.reload()
        return self.checker._skip_latest_version()

    # ---- helpers ----------------------------------------------------------------------------

    def _describe(self, release: dict) -> UpdateResult:
        html_url = str(release.get("html_url") or "")
        return UpdateResult(
            status="current",
            tag=str(release.get("tag_name") or ""),
            notes=str(release.get("body") or ""),
            html_url=html_url if html_url.startswith("https://") else releases_page(release.get('_provider', 'github')),
            published=str(release.get("published_at") or "")[:10],
            downloads=select_downloads(release, self.platform, self.arch),
        )

    def _message(self, result: UpdateResult) -> str:
        if result.status == "hotfix":
            return f"Hotfix {result.downloads.apk.hotfix} for Glossarion {result.tag} is available."
        if result.status == "current":
            offer = result.install_asset
            if offer is not None:
                return (f"You run {self.current_version}. Hotfix {offer.hotfix} for {result.tag} is out; "
                        "Glossarion cannot tell whether this build already has it.")
            return f"You are up to date ({self.current_version})."
        if result.status == "skipped":
            return f"{result.tag} is skipped. Check now to see it anyway."
        downloads = result.downloads
        if downloads is not None and downloads.available:
            return f"Glossarion {result.tag} is available."
        where = {"android": "Android", "ios": "iOS"}.get(self.platform, "mobile")
        return f"Glossarion {result.tag} is out, but it has no {where} build. See the release page."

    def _remember(self, result: UpdateResult, *, keep_last: bool = False) -> UpdateResult:
        result.checked_at = self.clock()
        if not keep_last or self.last is None:
            self.last = result
        return result


def _download_error(exc: BaseException) -> str:
    try:
        import requests
    except ImportError:  # pragma: no cover - requests ships with the app
        requests = None
    if requests is not None and isinstance(exc, requests.RequestException):
        return "The download stopped: the connection was lost. Try again."
    if isinstance(exc, OSError):
        return f"Could not save the update: {exc.strerror or exc}"
    return str(exc) or type(exc).__name__


class ApkInstaller:
    """Android's in-app update (one per app session, so a download goes on when the Updates screen
    closes). ``native`` is the app's ``NativeBridge``.

    States: ``idle``; ``downloading`` (``percent``, ``done_mb``, ``total_mb``); ``permission`` (the
    verified APK is ready but Android must first allow installs from Glossarion: ``allow_installs``
    opens that settings page, and ``resumed`` installs once the user comes back with it allowed);
    ``installing`` (the system installer is open: Android asks "update this app?", and the update
    restarts Glossarion; a user who backed out taps Install again); ``error`` (``error``)."""

    def __init__(self, service: Any, native: Any, *, folder: Optional[str] = None,
                 run_io: Optional[Callable[..., Awaitable[Any]]] = None) -> None:
        self.service = service
        self.native = native
        self.run_io = run_io or (lambda fn, *args: asyncio.to_thread(fn, *args))
        self._folder = folder
        self.state = "idle"
        self.asset: Optional[MobileAsset] = None
        self.path: Optional[str] = None
        self.percent = 0
        self.done_mb = 0.0
        self.total_mb = 0.0
        self.error = ""
        self._cancelled = False
        self._listeners: list = []

    # ---- observers ----------------------------------------------------------------------------

    def subscribe(self, callback: Callable[[bool], Any]) -> Callable[[], None]:
        """``callback(progress_only)`` on every change; returns the unsubscribe function."""
        self._listeners.append(callback)

        def unsubscribe() -> None:
            if callback in self._listeners:
                self._listeners.remove(callback)

        return unsubscribe

    def _emit(self, progress_only: bool = False) -> None:
        for callback in list(self._listeners):
            try:
                callback(progress_only)
            except Exception:
                log.exception("update installer listener failed")

    def _set(self, state: str, error: str = "") -> None:
        self.state, self.error = state, error
        self._emit()

    def busy_with(self, asset: Optional[MobileAsset]) -> bool:
        return asset is not None and self.asset is not None and self.asset.name == asset.name and self.state != "idle"

    # ---- steps --------------------------------------------------------------------------------

    async def folder(self) -> Optional[str]:
        """``<app cache>/updates`` (the only folder the extension installs from), None off-device."""
        if self._folder is None:
            info = await self.native.platform_info()
            cache = info.get("cache_dir") if isinstance(info, dict) else None
            if cache:
                self._folder = os.path.join(str(cache), UPDATES_SUBDIR)
        return self._folder

    async def adopt(self, asset: Optional[MobileAsset]) -> None:
        """After a check: a verified download of ``asset`` from before is ready to install; the files
        of anything else are removed (an installed or outdated APK is ~300 MB of cache)."""
        if self.state in ("downloading", "installing") and not self.busy_with(asset):
            return
        folder = await self.folder()
        if folder is None:
            return
        ready = await self.run_io(self.service.ready_apk, asset, folder) if asset is not None else None
        await self.run_io(self.service.prune_downloads, folder, asset.name if ready else "")
        if ready and self.state not in ("downloading", "installing"):
            self.asset, self.path = asset, ready
            self._set("permission" if not await self.native.can_install_packages() else "ready")

    async def download(self, asset: MobileAsset) -> str:
        """Download, verify and install ``asset``: the outcome of :meth:`install`, or ``cancelled`` /
        ``error`` / ``busy``."""
        if self.state == "downloading":
            return "busy"
        folder = await self.folder()
        if folder is None:
            self.asset = asset
            self._set("error", "Installing updates needs the Glossarion Android app. Use the browser link.")
            return "error"
        ready = await self.run_io(self.service.ready_apk, asset, folder)
        if ready:
            self.asset, self.path = asset, ready
            return await self.install()
        self.asset, self.path, self._cancelled = asset, None, False
        self.percent, self.done_mb, self.total_mb = 0, 0.0, asset.size_mb
        self._set("downloading")
        loop = asyncio.get_running_loop()
        last = [0.0]

        def apply(percent: int, done: float, total: float) -> None:
            if self.state == "downloading":
                self.percent, self.done_mb, self.total_mb = int(percent), float(done), float(total)
                self._emit(progress_only=True)

        def progress(percent: int, done: float, total: float) -> None:  # worker thread
            now = time.monotonic()
            if now - last[0] < 0.25 and percent < 100:
                return
            last[0] = now
            try:
                loop.call_soon_threadsafe(apply, percent, done, total)
            except RuntimeError:  # the loop is gone (app closing)
                pass

        try:
            path = await self.run_io(self.service.download_apk, asset, folder, progress, lambda: self._cancelled)
        except Exception as exc:
            if self._cancelled or type(exc).__name__ == "UpdateCancelled":
                self._set("idle")
                return "cancelled"
            log.info("update download failed: %s", type(exc).__name__)
            self._set("error", _download_error(exc))
            return "error"
        self.path = path
        return await self.install()

    def cancel(self) -> None:
        self._cancelled = True

    async def install(self) -> str:
        """Hand the verified APK to Android's installer: ``installer`` (its screen is open),
        ``permission`` (installs from Glossarion are not allowed yet), ``missing`` or ``error``."""
        if not self.path:
            self._set("error", "Download the update first.")
            return "missing"
        if not await self.native.can_install_packages():
            self._set("permission")
            return "permission"
        answer = await self.native.install_package(self.path)
        if answer.get("ok"):
            self._set("installing")
            return "installer"
        code = str(answer.get("error") or "error")
        if code == "permission":
            self._set("permission")
            return "permission"
        if code == "missing":
            self.path = None
            self._set("error", "The downloaded update is gone (Android cleared the cache). Download it again.")
            return "missing"
        self._set("error", str(answer.get("message") or "Android could not open its installer."))
        return "error"

    async def allow_installs(self) -> bool:
        """Open Android's "Install unknown apps" page for Glossarion (one switch)."""
        return await self.native.open_install_settings()

    async def resumed(self) -> Optional[str]:
        """The app is back (from the settings page): install when installs are allowed now."""
        if self.state == "permission" and self.path and await self.native.can_install_packages():
            return await self.install()
        return None
