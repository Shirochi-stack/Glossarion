"""Update checks for About › Updates (UI_SPEC §4.16 "Updates"; FEATURE_MAP "Update checker").

The release check is the desktop's: ``update_core.HeadlessUpdateChecker`` runs the moved
``UpdateManager`` logic (``_check_for_updates_core``: 30-minute cache, ``skipped_versions``,
``/releases/latest`` + history, version compare; ``_skip_latest_version``) against the mobile
``config.json`` through :class:`StoreConfigHost`, so desktop and mobile share
``auto_update_check``, ``last_update_check_time`` and ``skipped_versions``.

What is mobile-only is the asset choice. Mobile builds are never published (owner's rule,
2026-10-09): ``build-mobile.yml`` only keeps the APK/IPA as run artifacts, so GitHub releases
are desktop-only and "no file for this phone" is the ordinary answer, never an error. A release
asset is offered only when it carries a name the build jobs write:

    <prefix>_Android_<abi>[_debugsigned].apk   Android, one per ABI (arm64-v8a, x86_64)
    <prefix>_Android.aab                       Play bundle (never offered in the app)
    <prefix>_iOS_unsigned.ipa                  unsigned IPA (sideloading)
    <prefix>_iOS.ipa                           signed IPA (devices in the provisioning profile)

Nothing here installs anything: a download link opens in the browser. Importing this module
imports no backend module.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

__all__ = [
    "MobileAsset", "MobileDownloads", "StoreConfigHost", "UPDATE_CONFIG_KEYS", "UpdateResult", "UpdateService",
    "classify_asset", "releases_page", "select_downloads",
]

log = logging.getLogger("glossarion.updates")

#: config.json keys the moved desktop methods read and write (shared with the desktop).
UPDATE_CONFIG_KEYS = ("auto_update_check", "last_update_check_time", "skipped_versions")

_APK = re.compile(r"_android_(?P<abi>arm64-v8a|x86_64|armeabi-v7a|x86)(?P<debug>_debugsigned)?\.apk$", re.I)


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
    return MobileAsset(kind=kind, name=name, url=url, size=size, abi=abi, arch=arch, debug_signed=debug)


def select_downloads(release: Optional[dict], platform: str, arch: str) -> MobileDownloads:
    """The mobile files of ``release`` for this device. ``arch`` is ``update_core``'s
    ``_detect_arch()`` (arm64 / x64 / unknown); an APK for another ABI is listed, never picked."""
    platform = str(platform or "").lower()
    out = MobileDownloads(platform=platform if platform in ("android", "ios") else "other")
    assets = [a for a in (_asset(raw) for raw in (release or {}).get("assets") or [] if isinstance(raw, dict)) if a]
    apks = [a for a in assets if a.kind == "apk"] if out.platform == "android" else []
    matching = [a for a in apks if arch != "unknown" and a.arch == arch]
    if matching:
        matching.sort(key=lambda a: (a.debug_signed, a.name))  # release-signed first
        out.apk = matching[0]
    out.other_apks = [a for a in apks if a is not out.apk]
    for asset in assets:
        if asset.kind == "ipa" and out.ipa is None:
            out.ipa = asset
        elif asset.kind == "ipa_signed" and out.ipa_signed is None:
            out.ipa_signed = asset
    return out


def releases_page() -> str:
    """``https://github.com/<owner>/<repo>/releases`` from the desktop's API URL."""
    api = _core().UpdateCoreMixin.GITHUB_API_URL  # https://api.github.com/repos/<owner>/<repo>/releases
    repo = api.split("/repos/", 1)[-1].rsplit("/releases", 1)[0]
    return f"https://github.com/{repo}/releases"


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
    status: str  # update / current / skipped / cached / error
    message: str = ""
    tag: str = ""
    notes: str = ""
    html_url: str = ""
    published: str = ""
    downloads: Optional[MobileDownloads] = None
    checked_at: float = 0.0

    @property
    def has_release(self) -> bool:
        return bool(self.tag)


class UpdateService:
    """One per app session: the desktop checker on the mobile config (blocking calls; run them
    on a worker thread)."""

    def __init__(self, store: Any, current_version: str, *, platform: str, arch: Optional[str] = None,
                 clock: Callable[[], float] = time.time) -> None:
        core = _core()
        self.host = StoreConfigHost(store)
        self.checker = core.HeadlessUpdateChecker(self.host, current_version, build_variant="Mobile")
        self.platform = str(platform or "").lower()
        self.arch = arch or core.UpdateCoreMixin._detect_arch()
        self.clock = clock
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
            return self._remember(UpdateResult("error", "Cannot reach GitHub. Check the connection and try again."))
        except requests.HTTPError as exc:
            code = getattr(getattr(exc, "response", None), "status_code", None)
            text = ("GitHub API rate limit exceeded. Please try again later." if code == 403
                    else f"GitHub returned error: {code}")
            return self._remember(UpdateResult("error", text))
        except ValueError:
            return self._remember(UpdateResult("error", "Invalid response from GitHub. The update service may be "
                                                        "temporarily unavailable."))
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
        result.message = self._message(result)
        return self._remember(result)

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
            html_url=html_url if html_url.startswith("https://") else releases_page(),
            published=str(release.get("published_at") or "")[:10],
            downloads=select_downloads(release, self.platform, self.arch),
        )

    def _message(self, result: UpdateResult) -> str:
        if result.status == "current":
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
