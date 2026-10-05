"""About (``/settings/about``; UI_SPEC §4.16 About, §4.17 "re-runnable from About").

Version and build (``runtime_bootstrap.app_version``: ``app_version.APP_VERSION``,
build number M*1_000_000+m*10_000+p*100), the backend bundle manifest
(``app/backend/_bundle_info.py`` written by ``tools/collect_backend.py``: git sha,
dirty flag, module count, bundle sha256, collection time), Python / Flet versions,
platform, the bundled third-party packages with their licence metadata
(``importlib.metadata``, read on a worker thread), links, and **Run the Welcome guide
again** (``/welcome``).
"""

from __future__ import annotations

import ast
import platform as _platform
import sys
from pathlib import Path
from typing import Any, Optional

import flet as ft

from glossarion_mobile.ui.components.empty_state import HALGAKOS_ASSET
from glossarion_mobile.ui.screens.page_base import PageScreen, human_size, section

__all__ = ["AboutScreen", "BUNDLE_FIELDS", "bundle_info", "license_rows", "version_rows"]

BUNDLE_FIELDS = ("BUILD_VERSION", "GIT_SHA", "GIT_DIRTY", "GENERATED_AT", "COLLECTED_WITH_PYTHON", "BUNDLE_SHA256",
                 "MODULE_COUNT", "TOTAL_BYTES", "COLLECTOR_VERSION")
PROJECT_URL = "https://github.com/Shirochi-stack/Glossarion"


def bundle_info(backend_dir: Any) -> dict:
    """The scalar fields of ``<backend>/_bundle_info.py`` (source, or the built app's ``.pyc``); {} when absent."""
    if backend_dir is None:
        return {}
    path = Path(backend_dir) / "_bundle_info.py"
    out: dict = {}
    try:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    except (OSError, SyntaxError, ValueError):
        tree = None
    if tree is not None:
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in BUNDLE_FIELDS:
                    try:
                        out[name] = ast.literal_eval(node.value)
                    except (ValueError, SyntaxError):
                        pass
        return out
    try:  # built apps ship only legacy .pyc files
        from glossarion_mobile import runtime_bootstrap as rb

        for name in BUNDLE_FIELDS:
            value = rb._read_assignment(path, name)
            if value is not None:
                out[name] = value
    except Exception:
        pass
    return out


def version_rows(version: Optional[dict], info: dict, *, platform_name: str = "", flet_version: Optional[str] = None,
                 backend_source: str = "") -> list:
    """``[(label, value)]`` for the About card."""
    version = version or {}
    rows = [
        ("Version", str(version.get("version") or info.get("BUILD_VERSION") or "unknown")),
        ("Build", str(version.get("build") or "—")),
        ("Platform", platform_name or _platform.system()),
        ("Python", sys.version.split()[0]),
        ("Flet", flet_version or "—"),
    ]
    if backend_source:
        rows.append(("Backend", backend_source))
    if info:
        sha = str(info.get("GIT_SHA") or "")
        rows.append(("Commit", (sha[:12] + (" (modified)" if info.get("GIT_DIRTY") else "")) if sha else "—"))
        rows.append(("Bundle", f"{info.get('MODULE_COUNT', '?')} modules · {human_size(info.get('TOTAL_BYTES'))}"))
        bundle = str(info.get("BUNDLE_SHA256") or "")
        if bundle:
            rows.append(("Bundle sha256", bundle[:16] + "…"))
        if info.get("GENERATED_AT"):
            rows.append(("Collected", str(info["GENERATED_AT"])))
    return rows


def license_rows(limit: int = 400) -> list:
    """Blocking: ``[(name, version, licence)]`` of the installed distributions, sorted by name."""
    try:
        from importlib import metadata
    except ImportError:  # pragma: no cover
        return []
    rows = []
    for dist in metadata.distributions():
        try:
            meta = dist.metadata
            name = meta.get("Name") or ""
            if not name:
                continue
            licence = meta.get("License-Expression") or meta.get("License") or ""
            if not licence or len(licence) > 80:
                classifiers = [c.split("::")[-1].strip() for c in (meta.get_all("Classifier") or [])
                               if c.startswith("License ::")]
                licence = ", ".join(classifiers) or (licence.splitlines()[0][:80] if licence else "see package")
            rows.append((name, dist.version or "", licence))
        except Exception:
            continue
    rows.sort(key=lambda row: row[0].lower())
    unique, seen = [], set()
    for row in rows:
        if row[0].lower() in seen:
            continue
        seen.add(row[0].lower())
        unique.append(row)
    return unique[:limit]


class AboutScreen(PageScreen):
    title = "About"

    def __init__(self, match: Any, ctx: Any, *, boot: Any = None, paths: Any = None, platform_name: str = "",
                 open_url: Any = None) -> None:
        super().__init__(match, ctx)
        self.boot = boot
        self.paths = paths
        self.platform_name = platform_name
        self.open_url = open_url
        self.info = bundle_info(getattr(paths, "backend_dir", None))
        self.licenses: list = []
        self.license_column = ft.Column(spacing=0, tight=True)

    def build_body(self) -> ft.Control:
        try:
            from glossarion_mobile import runtime_bootstrap as rb

            flet_version = rb._flet_version()
        except Exception:
            flet_version = None
        version = getattr(self.boot, "version", None) if self.boot is not None else None
        if version is None:
            try:
                from glossarion_mobile import runtime_bootstrap as rb

                version = rb.app_version(getattr(self.paths, "backend_dir", None))
            except Exception:
                version = {}
        self.rows = version_rows(version, self.info, platform_name=self.platform_name, flet_version=flet_version,
                                 backend_source=str(getattr(self.paths, "backend_source", "") or ""))
        details = [
            ft.Row([ft.Text(label, theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT,
                            width=110), ft.Text(value, selectable=True, theme_style=ft.TextThemeStyle.BODY_SMALL,
                                                expand=True)])
            for label, value in self.rows
        ]
        self.license_tile = ft.ExpansionTile(title="Open-source licences", expanded=False, controls=[self.license_column],
                                             on_change=lambda e: self.spawn(self.load_licenses()),
                                             key="about-licenses")
        return self.scaffold([
            ft.Column([
                ft.Image(src=HALGAKOS_ASSET, width=88, height=88),
                ft.Text("Glossarion", theme_style=ft.TextThemeStyle.HEADLINE_SMALL, weight=ft.FontWeight.W_600),
                ft.Text("Translate novels, manga and documents with your own AI accounts and keys",
                        theme_style=ft.TextThemeStyle.BODY_SMALL, text_align=ft.TextAlign.CENTER),
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=6),
            section("Version", details, key="about-version"),
            section("Guides", [
                ft.ListTile(leading=ft.Icon(ft.Icons.WAVING_HAND), title=ft.Text("Run the Welcome guide again"),
                            on_click=lambda e: self.rerun_welcome(), key="about-welcome"),
                ft.ListTile(leading=ft.Icon(ft.Icons.OPEN_IN_NEW), title=ft.Text("Project page"),
                            subtitle=ft.Text(PROJECT_URL, theme_style=ft.TextThemeStyle.BODY_SMALL),
                            on_click=lambda e: self._open(PROJECT_URL), key="about-project"),
            ]),
            section("Licences", [self.license_tile]),
        ])

    def rerun_welcome(self) -> Optional[str]:
        return self.ctx.go("welcome")

    def _open(self, url: str) -> None:
        if self.open_url is not None:
            self.open_url(url)

    async def load_licenses(self) -> list:
        if self.licenses:
            return self.licenses
        self.licenses = await self.io(license_rows)
        self.license_column.controls = [
            ft.ListTile(title=ft.Text(f"{name} {version}", theme_style=ft.TextThemeStyle.BODY_SMALL),
                        subtitle=ft.Text(licence, theme_style=ft.TextThemeStyle.BODY_SMALL), dense=True)
            for name, version, licence in self.licenses
        ] or [ft.Text("No package metadata in this build.", theme_style=ft.TextThemeStyle.BODY_SMALL)]
        self.push(self.license_column)
        return self.licenses
