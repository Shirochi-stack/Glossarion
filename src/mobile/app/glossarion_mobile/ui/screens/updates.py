"""About › Updates (``/settings/updates``; UI_SPEC §4.16 "Updates", FEATURE_MAP "Update checker").

**Check now** · **Check on startup** (the shared ``auto_update_check``) · **Skip this version** ·
release notes · Android: **Download and install** (the APK for this device's ABI, the newest
``_hotfixN`` of the release when there is one: downloaded in the app with progress, size and
SHA-256 verified, then handed to Android's installer by ``services.updates.ApkInstaller``; the
first time one tap opens "Install unknown apps" for Glossarion, and coming back installs) with
the browser link kept as a fallback · iOS: the IPA links and why the app cannot replace itself.
The checks are the desktop's (``services.updates.UpdateService`` over ``update_core``).

``await UpdatesFeature.install(app)`` wires the route into the shell's screen factory, marks
the page implemented on the Settings home and runs the quiet startup check: once per session,
after the backend is warm, only in the app on a phone or tablet (never in a desktop window,
under ``flet test`` or a host test, or with ``GLOSSARION_UPDATE_CHECK=0``), never while a
self-test runs, and it only speaks up (a snackbar with "View") when a newer release has a file
for this device.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import logging
import os
from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.router import RouteMatch
from glossarion_mobile.ui.screens.page_base import PageScreen, section

__all__ = ["ROUTE", "STARTUP_DELAY", "UpdatesFeature", "UpdatesScreen", "describe_checked"]

log = logging.getLogger("glossarion.updates")

ROUTE = "settings.updates"
STARTUP_DELAY = 20.0  # seconds after the backend is warm
#: ``GLOSSARION_UPDATE_CHECK=0`` turns the startup check off for automation (CI, UI tests).
DISABLE_ENV = "GLOSSARION_UPDATE_CHECK"


def describe_checked(timestamp: float, now: Optional[float] = None) -> str:
    """"Last checked: …" text for a ``last_update_check_time`` value."""
    try:
        value = float(timestamp or 0)
    except (TypeError, ValueError):
        value = 0.0
    if value <= 0:
        return "Never checked"
    try:
        when = _dt.datetime.fromtimestamp(value)
    except (OverflowError, OSError, ValueError):
        return "Never checked"
    return "Last checked " + when.strftime("%Y-%m-%d %H:%M")


class UpdatesScreen(PageScreen):
    title = "Updates"

    def __init__(self, match: Any, ctx: Any, *, service: Any, open_url: Optional[Callable[[str], Any]] = None,
                 installer: Any = None) -> None:
        super().__init__(match, ctx)
        self.service = service
        self.open_url = open_url
        #: ``services.updates.ApkInstaller`` (Android app); None: the APK opens in the browser.
        self.installer = installer
        self.busy = False
        self._builds = 0
        self._install_builds = 0
        self.install_box: Optional[ft.Column] = None
        self.install_bar: Optional[ft.ProgressBar] = None
        self.install_line: Optional[ft.Text] = None
        self._unsubscribe: Optional[Callable[[], None]] = None
        if installer is not None:
            self._unsubscribe = installer.subscribe(self._on_installer)
        self.result: Any = getattr(service, "last", None)

    # ---- layout --------------------------------------------------------------------------

    def build_body(self) -> ft.Control:
        service = self.service
        self.status_text = ft.Text(self._status_line(), theme_style=ft.TextThemeStyle.BODY_MEDIUM,
                                   key="updates-status")
        self.checked_text = ft.Text(describe_checked(service.last_checked()), theme_style=ft.TextThemeStyle.BODY_SMALL,
                                    color=ft.Colors.ON_SURFACE_VARIANT, key="updates-checked")
        self.ring = ft.ProgressRing(width=20, height=20, stroke_width=2, visible=False, key="updates-ring")
        self.check_button = ft.FilledButton(content="Check now", icon=ft.Icons.REFRESH, on_click=self._on_check,
                                            key="updates-check", tooltip="Check GitHub for a newer release")
        self.startup_switch = ft.Switch(label="Check on startup", value=service.startup_enabled(),
                                        on_change=self._on_startup, key="updates-startup")
        self.release_holder = ft.Container(key="updates-release")
        self._render(self.result)
        return self.scaffold([
            section("Glossarion " + str(service.current_version or "?"), [
                self.status_text,
                self.checked_text,
                ft.Row([self.check_button, self.ring], spacing=tokens.SPACING["md"],
                       vertical_alignment=ft.CrossAxisAlignment.CENTER),
                self.startup_switch,
            ], key="updates-version"),
            self.release_holder,
            ft.Text(self._note(), theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT,
                    key="updates-note"),
        ])

    def _note(self) -> str:
        platform = str(getattr(self.service, "platform", "") or "")
        if platform == "ios":
            return ("iOS does not let an app replace itself: only the App Store, TestFlight or the tool you "
                    "sideloaded Glossarion with (AltStore, SideStore, Sideloadly…) can install a new IPA over it. "
                    "Download the IPA here and install it with that tool.")
        if platform == "android":
            return ("Glossarion downloads the update and opens Android's installer. Android always asks you to "
                    "confirm an app update (only the Play Store or a managed phone can update silently), and the "
                    "first time it asks you to allow installs from Glossarion. Installing restarts the app: "
                    "a running translation stops and can be resumed.")
        return "Mobile builds are attached to some GitHub releases only; the release page has every file."

    def _status_line(self) -> str:
        result = self.result
        if self.busy:
            return "Checking GitHub releases…"
        if result is None:
            return "Check GitHub for a newer Glossarion release."
        return result.message or "Checked."

    def _render(self, result: Any) -> None:
        """Replace the release card (a new subtree per render: Flet freezes re-keyed controls)."""
        self._builds += 1
        n = self._builds
        self.install_box = None
        if result is None or not getattr(result, "has_release", False):
            self.release_holder.content = None
            return
        downloads = result.downloads
        rows: list[ft.Control] = []
        subtitle = f"Released {result.published}" if result.published else None
        if downloads is not None and downloads.platform == "android":
            rows += self._android_rows(downloads)
        elif downloads is not None and downloads.platform == "ios":
            rows += self._ios_rows(downloads)
        if downloads is None or not downloads.available:
            rows.append(ft.Text("This release has no file for this device.", theme_style=ft.TextThemeStyle.BODY_SMALL,
                                color=ft.Colors.ON_SURFACE_VARIANT))
        rows.append(self._link_tile("Release page", result.html_url, ft.Icons.OPEN_IN_NEW, f"updates-page-{n}"))
        if result.status == "update":
            rows.append(ft.TextButton(content="Skip this version", icon=ft.Icons.SKIP_NEXT, on_click=self._on_skip,
                                      key=f"updates-skip-{n}"))
        if result.notes:
            rows.append(ft.ExpansionTile(
                title="Release notes", expanded=result.status == "update", key=f"updates-notes-{n}",
                controls=[ft.Markdown(result.notes[:60000], selectable=True,
                                      extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
                                      on_tap_link=lambda e: self._open_web(getattr(e, "data", "") or ""))],
            ))
        self.release_holder.content = section(f"Glossarion {result.tag}", rows, subtitle=subtitle,
                                              key=f"updates-card-{n}")

    def _android_rows(self, downloads: Any) -> list:
        rows: list[ft.Control] = []
        apk = downloads.apk
        offer = getattr(self.result, "install_asset", None) if self.result is not None else None
        installed = getattr(self.result, "installed", "") if self.result is not None else ""
        if installed:
            rows.append(ft.Text(f"Installed: {installed}", theme_style=ft.TextThemeStyle.BODY_SMALL,
                                color=ft.Colors.ON_SURFACE_VARIANT, key=f"updates-installed-{self._builds}"))
        if apk is not None:
            if offer is not None and self.installer is not None:
                self.install_box = ft.Column(spacing=tokens.SPACING["sm"], key=f"updates-install-{self._builds}")
                self._fill_install_box(offer)
                rows.append(self.install_box)
            else:
                self.install_box = None
            label = f"Download APK ({apk.abi}, {apk.size_mb:.0f} MB)"
            if offer is not None and self.installer is not None:
                rows.append(self._link_tile(f"Download in the browser instead ({apk.size_mb:.0f} MB)", apk.url,
                                            ft.Icons.OPEN_IN_BROWSER, f"updates-apk-{self._builds}"))
            else:
                rows.append(ft.FilledButton(content=label, icon=ft.Icons.DOWNLOAD,
                                            on_click=lambda e, url=apk.url: self._open(url),
                                            key=f"updates-apk-{self._builds}"))
            if apk.debug_signed:
                rows.append(ft.Text("Debug-signed build: it updates only a debug-signed install.",
                                    theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT))
        for index, other in enumerate(downloads.other_apks):
            rows.append(self._link_tile(f"{other.name} ({other.size_mb:.0f} MB)", other.url, ft.Icons.ANDROID,
                                        f"updates-apk-other-{self._builds}-{index}"))
        if apk is not None or downloads.other_apks:
            rows.append(ft.Text("Android installs an update only when it is signed with the same key as the "
                                "installed app.", theme_style=ft.TextThemeStyle.BODY_SMALL,
                                color=ft.Colors.ON_SURFACE_VARIANT))
        return rows

    def _ios_rows(self, downloads: Any) -> list:
        rows: list[ft.Control] = []
        if downloads.ipa is not None:
            rows.append(self._link_tile(f"Unsigned IPA ({downloads.ipa.size_mb:.0f} MB)", downloads.ipa.url,
                                        ft.Icons.DOWNLOAD, f"updates-ipa-{self._builds}"))
        if downloads.ipa_signed is not None:
            rows.append(self._link_tile(f"Signed IPA ({downloads.ipa_signed.size_mb:.0f} MB)", downloads.ipa_signed.url,
                                        ft.Icons.DOWNLOAD, f"updates-ipa-signed-{self._builds}"))
        return rows

    # ---- Android in-app install (services.updates.ApkInstaller) ------------------------------------

    def _fill_install_box(self, asset: Any) -> None:
        """The install step for ``asset`` from the installer's state (new keys per fill)."""
        box, installer = self.install_box, self.installer
        if box is None or installer is None:
            return
        self._install_builds += 1
        n = f"{self._builds}-{self._install_builds}"
        mine = installer.asset is not None and installer.asset.name == asset.name
        state = installer.state if mine else "idle"
        what = f"hotfix {asset.hotfix}" if asset.hotfix else "update"
        controls: list[ft.Control] = []
        self.install_bar = self.install_line = None
        if state == "downloading":
            pct = installer.percent
            self.install_bar = ft.ProgressBar(value=pct / 100.0 if pct else None, key=f"updates-install-bar-{n}")
            self.install_line = ft.Text(self._progress_text(), theme_style=ft.TextThemeStyle.BODY_SMALL,
                                        key=f"updates-install-line-{n}")
            controls += [self.install_bar, self.install_line,
                         ft.TextButton(content="Cancel download", icon=ft.Icons.CLOSE,
                                       on_click=self._on_cancel_download, key=f"updates-install-cancel-{n}")]
        elif state == "permission":
            controls += [
                ft.Text("Downloaded and verified. Android needs your OK once: turn on \"Allow from this source\" "
                        "for Glossarion, then come back and the installer opens.",
                        theme_style=ft.TextThemeStyle.BODY_SMALL, key=f"updates-install-perm-text-{n}"),
                ft.FilledButton(content="Allow installs from Glossarion", icon=ft.Icons.SECURITY,
                                on_click=self._on_allow, key=f"updates-install-allow-{n}"),
                ft.TextButton(content="Install", icon=ft.Icons.INSTALL_MOBILE, on_click=self._on_install,
                              key=f"updates-install-retry-{n}"),
            ]
        elif state in ("ready", "installing"):
            text = ("Android's installer is open: tap Update there. Glossarion restarts when it is done."
                    if state == "installing" else "Downloaded and verified.")
            controls += [
                ft.Text(text, theme_style=ft.TextThemeStyle.BODY_SMALL, key=f"updates-install-ready-text-{n}"),
                ft.FilledButton(content="Install again" if state == "installing" else f"Install {what}",
                                icon=ft.Icons.INSTALL_MOBILE, on_click=self._on_install,
                                key=f"updates-install-go-{n}"),
            ]
        else:
            if state == "error" and installer.error:
                controls.append(ft.Text(installer.error, color=ft.Colors.ERROR, theme_style=ft.TextThemeStyle.BODY_SMALL,
                                        key=f"updates-install-error-{n}"))
            controls.append(ft.FilledButton(
                content=f"Download and install {what} ({asset.size_mb:.0f} MB)", icon=ft.Icons.SYSTEM_UPDATE,
                on_click=lambda e, a=asset: self._on_download(a), key=f"updates-install-start-{n}"))
        box.controls = controls

    def _progress_text(self) -> str:
        installer = self.installer
        if installer is None:
            return ""
        return f"Downloading… {installer.done_mb:.0f} of {installer.total_mb:.0f} MB ({installer.percent}%)"

    def _on_installer(self, progress_only: bool) -> None:
        """Installer listener (event loop): progress ticks move the bar, state changes refill the step."""
        offer = getattr(self.result, "install_asset", None) if self.result is not None else None
        if self.install_box is None or offer is None:
            return
        if progress_only and self.install_bar is not None and self.install_line is not None:
            pct = self.installer.percent
            self.install_bar.value = pct / 100.0 if pct else None
            self.install_line.value = self._progress_text()
            self.push(self.install_bar, self.install_line)
            return
        self._fill_install_box(offer)
        self.push(self.install_box)

    def _on_download(self, asset: Any) -> Any:
        return self.spawn(self.download_and_install(asset))

    async def download_and_install(self, asset: Any) -> Optional[str]:
        if self.installer is None:
            return None
        outcome = await self.installer.download(asset)
        if outcome == "error" and self.installer.error:
            self.say(self.installer.error)
        return outcome

    def _on_cancel_download(self, e: Any = None) -> None:
        if self.installer is not None:
            self.installer.cancel()

    def _on_install(self, e: Any = None) -> Any:
        return self.spawn(self.install()) if self.installer is not None else None

    async def install(self) -> Optional[str]:
        outcome = await self.installer.install()
        if outcome == "permission":
            await self.installer.allow_installs()  # one tap: straight to the switch
        elif outcome in ("error", "missing") and self.installer.error:
            self.say(self.installer.error)
        return outcome

    def _on_allow(self, e: Any = None) -> Any:
        return self.spawn(self.installer.allow_installs()) if self.installer is not None else None

    def app_resumed(self) -> None:
        """Back from Android's "Install unknown apps" page: install when it is allowed now."""
        if self.installer is not None and self.installer.state == "permission":
            self.spawn(self.installer.resumed())

    def dispose(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None

    def _link_tile(self, title: str, url: str, icon: Any, key: str) -> ft.Control:
        return ft.ListTile(leading=ft.Icon(icon), title=ft.Text(title), dense=True,
                           on_click=lambda e, u=url: self._open(u), key=key)

    # ---- actions ---------------------------------------------------------------------------

    def _open(self, url: str) -> None:
        if not url:
            return
        if self.open_url is not None:
            try:
                result = self.open_url(url)
                if asyncio.iscoroutine(result):
                    self.spawn(result)
            except Exception as exc:
                log.warning("opening %s failed: %s", url, exc)
                self.say(f"Could not open the link: {exc}")

    def _open_web(self, url: str) -> None:
        """A link in the release notes: web pages only (no intent:, file: or app schemes)."""
        if str(url).lower().startswith("https://"):
            self._open(url)

    def _on_check(self, e: Any = None) -> Any:
        return self.spawn(self.check_now())

    async def check_now(self) -> Any:
        if self.busy:
            return self.result
        self.busy = True
        self._set_busy(True)
        try:
            result = await self.io(lambda: self.service.check(manual=True))
        except Exception as exc:  # pragma: no cover - service.check catches its own errors
            log.exception("update check failed")
            result = None
            self.say(f"Update check failed: {exc}")
        finally:
            self.busy = False
        if result is not None:
            self.result = result
        self._set_busy(False)
        self._render(self.result)
        self.checked_text.value = describe_checked(self.service.last_checked())
        self.push(self.release_holder, self.checked_text)
        if self.installer is not None and result is not None and result.status != "error":
            # a verified APK from a visit before is offered as "Install"; any other download is removed
            self.spawn(self.installer.adopt(getattr(result, "install_asset", None)))
        return self.result

    def _set_busy(self, busy: bool) -> None:
        self.ring.visible = busy
        self.check_button.disabled = busy
        self.status_text.value = self._status_line()
        self.push(self.ring, self.check_button, self.status_text)

    def _on_startup(self, e: Any = None) -> None:
        value = bool(getattr(getattr(e, "control", None), "value", self.startup_switch.value))
        self.service.set_startup(value)

    def _on_skip(self, e: Any = None) -> Any:
        return self.spawn(self.skip())

    async def skip(self) -> Optional[str]:
        tag = await self.io(self.service.skip)
        if tag:
            if self.result is not None:
                self.result.status = "skipped"
                self.result.message = f"{tag} is skipped. Check now to see it anyway."
            self.status_text.value = self._status_line()
            self._render(self.result)
            self.push(self.status_text, self.release_holder)
        return tag


class UpdatesFeature:
    """Installs About › Updates into the app (see the module docstring)."""

    def __init__(self, app: Any, *, service: Any = None, startup_delay: float = STARTUP_DELAY,
                 installer: Any = None) -> None:
        self.app = app
        self._service = service
        self._installer = installer
        self.startup_delay = startup_delay
        self._fallback_factory: Optional[Callable[[RouteMatch], Any]] = None
        self._unsubs: list = []
        self.startup_task: Any = None
        self.startup_result: Any = None
        self.screens_built: list = []

    @classmethod
    async def install(cls, app: Any, **kwargs: Any) -> "UpdatesFeature":
        feature = cls(app, **kwargs)
        feature.attach()
        return feature

    def attach(self) -> None:
        app = self.app
        app.updates_feature = self
        shell = getattr(app, "shell", None)
        if shell is not None and self._fallback_factory is None:
            self._fallback_factory = shell.screen_factory
            shell.screen_factory = self.screen_factory
        backend = getattr(getattr(app, "state", None), "backend", None)
        if backend is not None and not self._unsubs and hasattr(backend, "subscribe"):
            self._unsubs.append(backend.subscribe(self._on_backend))
            self._on_backend(getattr(backend, "value", None))

    # ---- service ----------------------------------------------------------------------------

    @property
    def service(self) -> Any:
        if self._service is None:
            from glossarion_mobile.services.updates import UpdateService

            self._service = UpdateService(getattr(self.app, "config_store", None), self._current_version(),
                                          platform=self._platform())
        return self._service

    @property
    def installer(self) -> Any:
        """The in-app APK install (Android app only: the extension opens the system installer); None
        elsewhere, where the screen keeps the browser link."""
        if self._installer is None and self._platform() == "android":
            native = getattr(self.app, "native", None)
            if native is not None and not getattr(native, "is_stub", True):
                from glossarion_mobile.services.updates import ApkInstaller

                self._installer = ApkInstaller(self.service, native, run_io=self._run_io)
        return self._installer

    async def _run_io(self, fn: Callable[..., Any], *args: Any) -> Any:
        dispatcher = getattr(self.app, "dispatcher", None)
        if dispatcher is not None and getattr(dispatcher, "bound", False):
            return await dispatcher.run_in_thread(lambda: fn(*args), name="gl-updates")
        return await asyncio.to_thread(fn, *args)

    def _current_version(self) -> str:
        app = self.app
        version = getattr(getattr(app, "boot", None), "version", None) or {}
        if not version.get("version"):
            try:
                from glossarion_mobile import runtime_bootstrap as rb

                version = rb.app_version(getattr(getattr(app, "paths", None), "backend_dir", None))
            except Exception:
                version = {}
        return str(version.get("version") or "0.0.0")

    def _platform(self) -> str:
        page = getattr(self.app, "page", None)
        platform = getattr(getattr(page, "platform", None), "value", None)
        return str(platform or "desktop")

    def _ctx(self) -> Any:
        app = self.app
        pages = getattr(app, "pages_feature", None)
        ctx = getattr(pages, "ctx", None) if pages is not None else None
        if ctx is None:
            env = getattr(getattr(app, "chat_feature", None), "env", None)
            ctx = env if env is not None else getattr(getattr(app, "settings", None), "ctx", None)
        return ctx

    def open_url(self, url: str) -> Any:
        """External application: the browser downloads an APK or IPA, or shows the release page."""
        launcher = getattr(self.app, "url_launcher", None)
        if launcher is None:
            return None
        mode = ft.LaunchMode.EXTERNAL_APPLICATION if getattr(self.app, "is_mobile", False) else ft.LaunchMode.PLATFORM_DEFAULT
        return launcher.launch_url(url, mode=mode)

    # ---- routing ---------------------------------------------------------------------------

    def screen_factory(self, match: RouteMatch) -> Any:
        if match.name == ROUTE:
            screen = UpdatesScreen(match, self._ctx(), service=self.service, open_url=self.open_url,
                                   installer=self.installer)
            self.screens_built.append(match.name)
            return screen
        if self._fallback_factory is None:
            raise LookupError(f"no screen for {match.name}")
        screen = self._fallback_factory(match)
        implemented = getattr(screen, "implemented", None)
        if match.name == "settings" and isinstance(implemented, frozenset):
            screen.implemented = implemented | {ROUTE}
        return screen

    # ---- startup check ----------------------------------------------------------------------

    def _on_backend(self, result: Any) -> None:
        if not result or not result.get("ok") or self.startup_task is not None:
            return
        if not self._startup_allowed():
            self.startup_task = False
            return
        self.startup_task = self._spawn(self.startup_check())

    def _startup_allowed(self) -> bool:
        app = self.app
        if os.environ.get(DISABLE_ENV, "").strip() == "0":
            return False
        if os.environ.get("PYTEST_CURRENT_TEST"):
            return False  # host tests never reach GitHub
        if not getattr(app, "is_mobile", False):
            return False  # desktop dev window: no mobile files to offer
        if str(getattr(getattr(app, "paths", None), "platform", "") or "") not in ("android", "ios"):
            return False  # a phone-sized session on a desktop Python (host tests, `flet run`)
        if getattr(getattr(app, "page", None), "test", False):
            return False  # `flet test`
        return True

    def _spawn(self, coro: Any) -> Any:
        dispatcher = getattr(self.app, "dispatcher", None)
        if dispatcher is not None and getattr(dispatcher, "bound", False):
            return dispatcher.spawn(coro)
        try:
            return asyncio.ensure_future(coro)
        except RuntimeError:
            coro.close()
            return None

    async def startup_check(self) -> Any:
        await asyncio.sleep(self.startup_delay)
        state = getattr(self.app, "state", None)
        running = getattr(getattr(state, "selftest_running", None), "value", False)
        if running or not self.service.startup_enabled():
            return None
        dispatcher = getattr(self.app, "dispatcher", None)
        try:
            if dispatcher is not None and getattr(dispatcher, "bound", False):
                result = await dispatcher.run_in_thread(lambda: self.service.check(manual=False), name="gl-updates")
            else:
                result = await asyncio.to_thread(self.service.check, manual=False)
        except Exception as exc:
            log.info("startup update check failed: %s", exc)
            return None
        self.startup_result = result
        installer = self.installer
        if installer is not None and getattr(result, "status", "") in ("update", "hotfix", "current", "skipped"):
            try:
                await installer.adopt(getattr(result, "install_asset", None))  # drops an installed update's APK
            except Exception:
                log.exception("checking downloaded updates failed")
        downloads = getattr(result, "downloads", None)
        if getattr(result, "status", "") in ("update", "hotfix") and downloads is not None and downloads.available:
            notify = getattr(self.app, "notify", None)
            if callable(notify):
                notify(result.message, "View", lambda e=None: self._view())
        return result

    def _view(self) -> None:
        navigate = getattr(self.app, "navigate_to", None)
        if callable(navigate):
            navigate(ROUTE)
