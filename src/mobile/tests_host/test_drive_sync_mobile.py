"""Host tests for Google Drive sync on the phone (``services/drive_sync.py`` over the shared ``drive_sync_core``)
and its two settings views: Library settings › Cloud (``LibraryCloudTab`` in the Library filter sheet) and the
Google Drive sync section at the top of Settings › Cloud sync & sharing.

* Nothing without sign-in: sync switched on but no Google account -> a finished compile creates no Drive client.
* Sign-in for sync reuses the share-link service's native Google sign-in WITHOUT turning on the Google Drive
  share-link site; Disconnect ends it.
* The DONE hook per mode: compiled output only (the reported EPUB; a recompile updates the same Drive file; a
  Compile PDF with no reported file uploads the book's PDF), full output folder (the tree, file by file, no zip),
  full output folder + raw source (``RAW/<source>``); a translation that compiled nothing is skipped in the
  compiled mode; sync off sends nothing while "Sync to Drive now" still works.
* The Cloud tab mirrors the settings page: one service, so the mode, folder and switch changed in one view show
  in the other at once; Google Drive sync is the first section of both; the card ⋯ menu has "Sync to Drive now"
  only while sync is on and connected.

Nothing contacts Google: the OAuth exchange / token / revoke calls are a recording stand-in and the Drive is an
in-memory stand-in of ``google_drive_core.DriveClient`` (the HTTP client itself is tested against a local fake
Google in tests/test_cloud_drive_core.py). Every path is under ``tmp_path``.

Run from src/mobile with the mobile venv:
    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_drive_sync_mobile.py
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import os
import sys
import threading
import types
from pathlib import Path
from typing import Any, Optional

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


pytestmark = pytest.mark.skipif(not (_has("flet") and _has("drive_sync_core")), reason="flet / drive core missing")

if _has("flet") and _has("drive_sync_core"):
    import flet as ft
    import google_drive_core as gdc

    from glossarion_mobile.services import share_links as sl
    from glossarion_mobile.services.drive_sync import STORE_KEY, MobileDriveSync
    from glossarion_mobile.ui.screens import cloud_sync as u10

_TB_SPEC = importlib.util.spec_from_file_location("_glossarion_tb_helpers_drive", Path(__file__).with_name(
    "test_bootstrap.py"))


def _tb():
    module = importlib.util.module_from_spec(_TB_SPEC)
    _TB_SPEC.loader.exec_module(module)
    return module


def run(coro):
    return asyncio.run(coro)


async def settle(rounds: int = 40) -> None:
    for _ in range(rounds):
        await asyncio.sleep(0)
    await asyncio.sleep(0.02)
    for _ in range(rounds):
        await asyncio.sleep(0)


def texts(control: Any, out: Optional[list] = None) -> list:
    out = out if out is not None else []
    if control is None:
        return out
    if isinstance(control, ft.Text) and control.value:
        out.append(str(control.value))
    for name in ("content", "controls", "label", "leading", "title", "subtitle", "trailing", "tabs"):
        value = getattr(control, name, None)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, ft.BaseControl):
                    texts(item, out)
        elif isinstance(value, ft.BaseControl):
            texts(value, out)
        elif isinstance(value, str) and name in ("content", "label"):
            out.append(value)
    return out


def chips(control: Any, out: Optional[list] = None) -> list:
    from glossarion_mobile.ui.components.reason_chip import ReasonChip

    out = out if out is not None else []
    if control is None:
        return out
    if isinstance(control, ReasonChip):
        out.append(control.reason)
    for name in ("content", "controls", "leading", "title", "subtitle", "trailing"):
        value = getattr(control, name, None)
        for item in value if isinstance(value, list) else [value]:
            if isinstance(item, ft.BaseControl):
                chips(item, out)
    return out


# ---------------------------------------------------------------------------
# stand-ins: secret box, Google OAuth, native sign-in, Drive, Library
# ---------------------------------------------------------------------------


class Fernet:  # the secret box checks the cipher type name; reversible stand-in
    pass


class _Handler:
    cipher = Fernet()

    @staticmethod
    def encrypt_value(value: str) -> str:
        return "ENC:" + value[::-1]

    @staticmethod
    def decrypt_value(token: str) -> str:
        return token[4:][::-1]


class FakeOAuth:
    """Records what would go to Google (exchange, token refresh, revoke); returns made-up values."""

    def __init__(self) -> None:
        self.calls: list = []

    def exchange(self, callback: str, nonce: str, verifier: str) -> str:
        self.calls.append(("exchange", callback))
        return "refresh-token-1"

    def token(self, refresh: str, rejected: str = "") -> str:
        self.calls.append(("token", refresh))
        return "access-1"

    def revoke(self, refresh: str) -> bool:
        self.calls.append(("revoke", refresh))
        return True


class FakeNative:
    """The iOS ``drive_authorize`` answer (a callback URL the OAuth stand-in exchanges)."""

    def __init__(self) -> None:
        self.calls: list = []

    async def call(self, name: str, arguments: Any = None, default: Any = None) -> Any:
        self.calls.append(name)
        return {"callback": "com.example:/oauth?code=c&state=s"}


class MemoryDrive:
    """An in-memory stand-in of ``google_drive_core.DriveClient`` (the surface ``DriveSyncRunner`` uses). Every
    call asks the service's ``access`` for a token, as the real client does."""

    def __init__(self, access: Any, world: dict) -> None:
        self.access = access
        self.world = world
        world["clients"] += 1

    def _token(self) -> None:
        self.world["tokens"].append(self.access(""))

    def _file(self, file_id: str) -> "gdc.DriveFile":
        item = self.world["files"][file_id]
        content = item.get("content")
        return gdc.DriveFile(file_id, item["name"], item["mime"],
                             hashlib.md5(content).hexdigest() if content is not None else "",
                             len(content) if content is not None else None, False, (item["parent"],))

    def get(self, file_id: str, cancel: Any = None) -> Any:
        self._token()
        return self._file(file_id) if file_id in self.world["files"] else None

    def list_children(self, folder_id: str, cancel: Any = None, *, name: Optional[str] = None,
                      folders_only: bool = False) -> list:
        self._token()
        found = [self._file(fid) for fid, item in sorted(self.world["files"].items()) if item["parent"] == folder_id
                 and (name is None or item["name"] == name)]
        return [f for f in found if f.is_folder or not folders_only]

    def find_child(self, folder_id: str, name: str, cancel: Any = None, *, folder: bool = False) -> Any:
        return next((f for f in self.list_children(folder_id, name=name) if f.is_folder == folder), None)

    def create_folder(self, name: str, parent: str = "root", cancel: Any = None) -> Any:
        self._token()
        fid = f"d{len(self.world['files']) + 1}"
        self.world["files"][fid] = {"name": name, "mime": gdc.FOLDER_MIME, "parent": parent, "content": None}
        return self._file(fid)

    def ensure_folder(self, name: str, parent: str = "root", cancel: Any = None) -> tuple:
        found = self.find_child(parent, name, folder=True)
        return (found, False) if found is not None else (self.create_folder(name, parent), True)

    def upload(self, path: str, *, name: str, mime: str = "", parent: Optional[str] = None,
               file_id: Optional[str] = None, size: Optional[int] = None, progress: Any = None, cancel: Any = None,
               fields: Any = None) -> Any:
        self._token()
        content = Path(path).read_bytes()
        if file_id:
            self.world["files"][file_id]["content"] = content
            self.world["updates"].append(file_id)
        else:
            file_id = f"f{len(self.world['files']) + 1}"
            self.world["files"][file_id] = {"name": name, "mime": mime, "parent": parent or "root", "content": content}
        return self._file(file_id)


def drive_world() -> dict:
    return {"files": {}, "tokens": [], "updates": [], "clients": 0}


def tree(world: dict) -> dict:
    files = world["files"]

    def path(fid: str) -> str:
        item = files[fid]
        parent = item["parent"]
        prefix = path(parent) + "/" if parent in files else ""
        return prefix + item["name"]

    return {path(fid) + ("/" if item["content"] is None else ""): item["content"] or b""
            for fid, item in files.items()}


class FakeLibrary:
    def __init__(self, root: Path, raw: Optional[Path] = None) -> None:
        self.root = root
        self.raw = raw
        self.snapshot = None

    def library_root(self) -> str:
        return str(self.root)

    def output_roots(self) -> list:
        return [str(self.root.parent / "Output")]

    def workspace_for(self, book: Any) -> str:
        folder = str(book.get("output_folder") or "")
        return folder if folder and os.path.isdir(folder) else ""

    def compiled_outputs_blocking(self, book: Any) -> list:
        folder = str(book.get("output_folder") or "")
        if not folder or not os.path.isdir(folder):
            return []
        return [(os.path.join(folder, n), n.rsplit(".", 1)[-1]) for n in sorted(os.listdir(folder))
                if n.lower().endswith((".epub", ".pdf"))]

    def raw_source(self, book: Any) -> str:
        return str(self.raw) if self.raw else ""


def write(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return str(path)


def snap(job_id: str, kind: str, folder: str, outputs: Any = (), state: str = "DONE") -> Any:
    spec = types.SimpleNamespace(kind=kind, params={"folder": folder}, origin={}, inputs=())
    return types.SimpleNamespace(id=job_id, state=state, spec=spec, outputs=list(outputs), output_dirs={},
                                 output_dir=None)


async def make_service(tmp_path: Path, *, platform: str = "ios", library: Any = None, loop_post: bool = True):
    shares = sl.ShareLinkService(tmp_path / "mobile_share_links.json", platform=platform, native=FakeNative(),
                                 cache_dir=str(tmp_path / "cache"), secrets=sl.SecretBox(handler=_Handler()))
    shares._drive_oauth = FakeOAuth()
    await shares.load()
    world = drive_world()
    loop = asyncio.get_running_loop()
    post = (lambda fn: loop.call_soon_threadsafe(fn)) if loop_post else None
    service = MobileDriveSync(shares, library=library, post=post,
                              client_factory=lambda access: MemoryDrive(access, world))
    return service, shares, world


def wait_idle(service: Any) -> None:
    assert service.core.worker.wait_idle(10)


# ---------------------------------------------------------------------------
# the service
# ---------------------------------------------------------------------------


def test_nothing_is_sent_without_a_google_account_and_sign_in_does_not_turn_on_drive_links(tmp_path):
    ws = tmp_path / "Library" / "Book"
    epub = write(ws / "Book.epub", b"compiled")

    async def scenario():
        service, shares, world = await make_service(tmp_path, library=FakeLibrary(tmp_path / "Library"))
        try:
            state = service.ui_state()
            assert not state["enabled"] and not state["signed_in"] and state["mode"] == "compiled"
            assert state["folder"] == "Glossarion" and state["supported"]
            service.on_job_transition(snap("j0", "compile_epub", str(ws), [epub]), "RUNNING")
            await service.set_enabled(True)  # switched on, but no account
            service.on_job_transition(snap("j1", "compile_epub", str(ws), [epub]), "RUNNING")
            assert not service.sync_book({"output_folder": str(ws), "name": "Book"})  # manual needs the account
            wait_idle(service)
            assert world["clients"] == 0 and not world["tokens"] and shares._drive_oauth.calls == []
            # sign in for sync: the share-link site stays off, the grant is sealed in the share-link sidecar
            await service.connect()
            assert service.signed_in() and shares._drive_oauth.calls == [("exchange", "com.example:/oauth?code=c&state=s")]
            gdrive = shares.provider_state("gdrive")
            assert gdrive.signed_in and not gdrive.enabled and not gdrive.consented
            raw = (tmp_path / "mobile_share_links.json").read_text(encoding="utf-8")
            assert "refresh-token-1" not in raw and STORE_KEY in raw  # sealed; settings beside it
            # the share-link path still asks for the site's consent
            await shares.disconnect_drive()
            assert not service.signed_in()
            with pytest.raises(sl.ShareError) as refused:
                await shares.connect_drive()
            assert refused.value.code == "consent" and not service.signed_in()
        finally:
            service.shutdown()

    run(scenario())


def test_done_hook_compiled_mode_uploads_the_compiled_file_and_updates_it_on_recompile(tmp_path):
    ws = tmp_path / "Library" / "Book"
    epub = write(ws / "Book.epub", b"first compile")
    write(ws / "response_001.html", b"<p>one</p>")
    pdf = write(ws / "Book.pdf", b"%PDF fake")

    async def scenario():
        service, shares, world = await make_service(tmp_path, library=FakeLibrary(tmp_path / "Library"))
        try:
            await service.connect()
            await service.set_enabled(True)
            await service.set_folder(" Books / Glossarion ")
            assert service.ui_state()["folder"] == "Books/Glossarion"
            service.on_job_transition(snap("a", "compile_epub", str(ws), [epub]), "RUNNING")
            wait_idle(service)
            assert tree(world) == {"Books/": b"", "Books/Glossarion/": b"", "Books/Glossarion/Book.epub": b"first compile"}
            assert world["tokens"] and set(world["tokens"]) == {"access-1"}  # the stored grant's token
            last = service.ui_state()["last_sync"]
            assert last["ok"] and last["uploaded"] == 1 and last["label"] == "Book"
            # the same job again (a repeated transition) is ignored; a recompile updates the same Drive file
            service.on_job_transition(snap("a", "compile_epub", str(ws), [epub]), "RUNNING")
            write(ws / "Book.epub", b"second compile, longer")
            service.on_job_transition(snap("b", "compile_epub", str(ws), [epub]), "RUNNING")
            wait_idle(service)
            assert tree(world)["Books/Glossarion/Book.epub"] == b"second compile, longer"
            assert len([f for f in world["files"].values() if f["name"] == "Book.epub"]) == 1
            assert len(world["updates"]) == 1 and service.ui_state()["last_sync"]["updated"] == 1
            # Compile PDF that reported nothing: the book's PDF (the resolver), not the EPUB again
            service.on_job_transition(snap("c", "compile_pdf", str(ws), []), "RUNNING")
            wait_idle(service)
            assert tree(world)["Books/Glossarion/Book.pdf"] == b"%PDF fake"
            assert "Books/Glossarion/response_001.html" not in tree(world)
            # a translation that compiled nothing changes nothing in the compiled mode; neither does a queued->done
            before = world["clients"]
            service.on_job_transition(snap("d", "translate", str(ws), [str(ws / "response_001.html")]), "RUNNING")
            service.on_job_transition(snap("e", "compile_epub", str(ws), [epub]), "QUEUED")
            service.on_job_transition(snap("f", "compile_epub", str(ws), [epub], state="FAILED"), "RUNNING")
            wait_idle(service)
            assert world["clients"] == before
            assert os.path.isfile(pdf)
        finally:
            service.shutdown()

    run(scenario())


def test_done_hook_full_modes_mirror_the_folder_and_send_the_raw_source_to_RAW(tmp_path):
    ws = tmp_path / "Library" / "My Novel"
    write(ws / "Book.epub", b"compiled")
    write(ws / "response_001.html", b"<p>one</p>")
    write(ws / "images" / "deep" / "x.png", b"png")
    write(ws / ".translation_progress.json.1.tmp", b"tmp")  # never uploaded
    raw = Path(write(tmp_path / "Library" / "Raw" / "Novel.epub", b"raw source"))

    async def scenario():
        service, _shares, world = await make_service(tmp_path, library=FakeLibrary(tmp_path / "Library", raw))
        try:
            await service.connect()
            await service.set_enabled(True)
            await service.set_mode("full")
            service.on_job_transition(snap("t1", "translate", str(ws)), "RUNNING")
            wait_idle(service)
            assert tree(world) == {
                "Glossarion/": b"", "Glossarion/My Novel/": b"", "Glossarion/My Novel/images/": b"",
                "Glossarion/My Novel/images/deep/": b"", "Glossarion/My Novel/Book.epub": b"compiled",
                "Glossarion/My Novel/response_001.html": b"<p>one</p>",
                "Glossarion/My Novel/images/deep/x.png": b"png"}
            # incremental: only the changed file goes again (updated in place)
            write(ws / "response_001.html", b"<p>one, edited</p>")
            service.on_job_transition(snap("t2", "translate", str(ws)), "RUNNING")
            wait_idle(service)
            last = service.ui_state()["last_sync"]
            assert (last["uploaded"], last["updated"], last["unchanged"]) == (0, 1, 2)
            await service.set_mode("full_raw")
            service.on_job_transition(snap("t3", "compile_epub", str(ws)), "RUNNING")
            wait_idle(service)
            assert tree(world)["Glossarion/My Novel/RAW/Novel.epub"] == b"raw source"
            assert not any(".tmp" in p for p in tree(world))
        finally:
            service.shutdown()

    run(scenario())


def test_sync_off_sends_nothing_but_sync_now_works_and_disconnect_stops_it(tmp_path):
    ws = tmp_path / "Library" / "Book"
    epub = write(ws / "Book.epub", b"compiled")

    async def scenario():
        service, shares, world = await make_service(tmp_path, library=FakeLibrary(tmp_path / "Library"))
        try:
            await service.connect()
            assert not service.active()
            service.on_job_transition(snap("a", "compile_epub", str(ws), [epub]), "RUNNING")
            wait_idle(service)
            assert world["clients"] == 0
            assert service.sync_book({"output_folder": str(ws), "name": "Book", "path": str(ws)})
            wait_idle(service)
            assert tree(world)["Glossarion/Book.epub"] == b"compiled"
            await service.set_enabled(True)
            assert service.active()
            await service.disconnect()
            assert not service.active() and not service.signed_in()
            assert ("revoke", "refresh-token-1") in shares._drive_oauth.calls
            before = world["clients"]
            service.on_job_transition(snap("b", "compile_epub", str(ws), [epub]), "RUNNING")
            wait_idle(service)
            assert world["clients"] == before
        finally:
            service.shutdown()

    run(scenario())


def test_a_drive_failure_is_logged_and_kept_never_raised(tmp_path, caplog):
    ws = tmp_path / "Library" / "Book"
    epub = write(ws / "Book.epub", b"compiled")

    class Broken(MemoryDrive):
        def ensure_folder(self, *args: Any, **kwargs: Any) -> tuple:
            raise gdc.ShareError("network", "Google Drive did not answer in time.")

    async def scenario():
        service, _shares, world = await make_service(tmp_path, library=FakeLibrary(tmp_path / "Library"))
        service.core.client = lambda: Broken(service.core.access(), world)
        try:
            await service.connect()
            await service.set_enabled(True)
            with caplog.at_level("INFO", logger="glossarion.drive.sync"):
                service.on_job_transition(snap("a", "compile_epub", str(ws), [epub]), "RUNNING")
                wait_idle(service)
            last = service.ui_state()["last_sync"]
            assert not last["ok"] and last["error_code"] == "network"
            assert "Drive sync failed: Book: Google Drive did not answer in time." in caplog.text
            assert "access-1" not in caplog.text and "refresh-token-1" not in caplog.text
            assert u10.drive_last_sync_text(last).startswith("⚠️ Last sync: Book · Google Drive did not answer")
        finally:
            service.shutdown()

    run(scenario())


# ---------------------------------------------------------------------------
# the two views: Library settings › Cloud and Settings › Cloud sync & sharing
# ---------------------------------------------------------------------------


class FakePage:
    width = 400
    height = 800

    def __init__(self) -> None:
        self.dialogs: list = []

    def show_dialog(self, dialog: Any) -> None:
        self.dialogs.append(dialog)

    def update(self, *args: Any) -> None:
        pass


class FakeCtx:
    def __init__(self, page: Any = None) -> None:
        self.page = page
        self.said: list = []
        self.went: list = []
        self.tablet = False
        self.dispatcher = None

    async def run_io(self, fn: Any, *args: Any) -> Any:
        return fn(*args)

    def spawn(self, coro: Any) -> Any:
        return asyncio.ensure_future(coro)

    def say(self, message: str, *args: Any) -> None:
        self.said.append(message)

    def go(self, name: str, params: Any = None, **kwargs: Any) -> None:
        self.went.append(name)


def test_cloud_tab_and_the_settings_page_show_the_same_drive_sync_state(tmp_path):
    from glossarion_mobile.ui.library.filter_sheet import FilterSheet
    from glossarion_mobile.ui.library.models import FilterState
    from glossarion_mobile.ui.router import parse_route

    async def scenario():
        service, shares, _world = await make_service(tmp_path)
        page = FakePage()
        ctx = FakeCtx(page)
        try:
            screen = u10.CloudSyncScreen(parse_route("/settings"), ctx, shares=shares, drive=service, platform="ios")
            body = screen.get_body()
            assert body.controls[0].key == "drive-card"  # Google Drive sync first on the page
            assert texts(body.controls[0])[0] == u10.DRIVE_SYNC_TITLE
            tab = u10.LibraryCloudTab(drive=service, shares=shares, platform="ios", say=ctx.say, go=ctx.go)
            sheet = FilterSheet(FilterState(), cloud_tab=tab, initial_tab=3)
            assert sheet.tabs.length == 4 and sheet.tabs.selected_index == 3
            bar = sheet.tabs.content.controls[0]
            assert [t.label for t in bar.tabs] == ["Filter", "Sort", "Display", "Cloud"]
            assert sheet.tabs.content.controls[1].controls[3] is tab.control
            assert texts(tab.control)[0] == u10.DRIVE_SYNC_TITLE  # Google Drive sync first in the tab too
            sheet.show(page)
            screen.did_show()
            await settle()
            for section in (tab.section, screen.drive_section):
                assert section.status.value == "Not connected"
                assert u10.DRIVE_NO_ACCOUNT_REASON in chips(section.reason_row)
                assert section.switch.disabled and section.mode_dropdown.value == "compiled"
                assert [o.text for o in section.mode_dropdown.options] == [
                    "Compiled output only", "Full output folder", "Full output folder + raw source"]
                assert section.folder_note.value.startswith("Glossarion can only see folders it made")
            assert "Copy to your cloud app: not available in this session" in tab.device_line.value
            assert tab.share_line.value == f"{u10.SHARE_LABEL}: every service is off"
            # switching on before signing in explains why and stays off
            assert not (await tab.section.set_enabled(True))["ok"] and not service.ui_state()["enabled"]
            # sign in from the tab: the consent dialog first, then the native sign-in
            dialog = await tab.section.account()
            assert isinstance(dialog, u10.ConfirmDialog) and dialog.title == u10.DRIVE_CONSENT_TITLE
            assert dialog.dialog in page.dialogs
            await dialog._on_confirm()
            await settle()
            for section in (tab.section, screen.drive_section):
                assert section.status.value == "Connected to Google Drive" and not section.switch.disabled
                assert section.account_button.content == "Disconnect" and not chips(section.reason_row)
            # a change in one view shows in the other at once (one service)
            await tab.section.set_mode("full_raw")
            await settle()
            assert screen.drive_section.mode_dropdown.value == "full_raw"
            screen.drive_section.folder_field.value = "Books/Novels"
            screen.drive_section._on_folder_typed()
            await screen.drive_section.save_folder()
            await settle()
            assert tab.section.folder_field.value == "Books/Novels"
            tab.section._on_switch(types.SimpleNamespace(control=types.SimpleNamespace(value=True)))
            await settle()
            assert screen.drive_section.switch.value is True and service.active()
            # a typed but unsaved folder survives a refresh caused by another change
            tab.section.folder_field.value = "Typing"
            tab.section._on_folder_typed()
            await screen.drive_section.set_mode("compiled")
            await settle()
            assert tab.section.folder_field.value == "Typing" and tab.section.mode_dropdown.value == "compiled"
            # disconnect from the page: asks first; both views follow
            dialog = await screen.drive_section.account()
            assert dialog.title == "Disconnect Google Drive?"
            await dialog._on_confirm()
            await settle()
            assert tab.section.status.value == "Not connected" and not service.active()
            # the full page's button closes the sheet (its subscriptions end) and goes there
            listeners = len(service._listeners)
            tab.open_full()
            assert ctx.went == [u10.ROUTE_NAME] and len(service._listeners) == listeners - 1
            screen.dispose()
            assert len(service._listeners) == listeners - 2
        finally:
            service.shutdown()

    run(scenario())


def test_cloud_tab_without_services_or_on_desktop_keeps_controls_disabled_with_reasons():
    from glossarion_mobile.ui.library.filter_sheet import FilterSheet
    from glossarion_mobile.ui.library.models import FilterState

    async def scenario():
        sheet = FilterSheet(FilterState())  # no services wired: the tab still shows, disabled
        tab = sheet.cloud_tab
        sheet.show(FakePage())
        await settle()
        assert u10.NO_SERVICE_REASON in chips(tab.section.reason_row)
        assert tab.section.switch.disabled and tab.section.account_button.disabled
        assert tab.section.mode_dropdown.disabled and tab.section.folder_field.disabled
        sheet._on_dismiss()

    run(scenario())


def test_desktop_platform_shows_phone_only(tmp_path):
    async def scenario():
        service, shares, _world = await make_service(tmp_path, platform="desktop")
        try:
            tab = u10.LibraryCloudTab(drive=service, shares=shares, platform="desktop")
            tab.attach()
            await settle()
            assert u10.PHONE_ONLY in chips(tab.section.reason_row) and tab.section.account_button.disabled
            assert not service.active() and not service.sync_book({"output_folder": str(tmp_path)})
            tab.dispose()
        finally:
            service.shutdown()

    run(scenario())


def test_card_menu_has_sync_to_drive_now_only_while_sync_is_on(tmp_path):
    from glossarion_mobile.services.library import LibraryService, SharedCore
    from glossarion_mobile.ui.library.common import LibraryContext
    from glossarion_mobile.ui.library.home import LibraryScreen
    from glossarion_mobile.ui.router import parse_route

    class Prefs:
        def __init__(self) -> None:
            self.data: dict = {}

        def file_ref(self, path: Any, *, kind: Any = None) -> str:
            return hashlib.sha1(os.fspath(path).encode()).hexdigest()[:12]

        def resolve_file_ref(self, fid: str, *, touch: bool = True) -> Any:
            return None

        def get(self, key: str, default: Any = None) -> Any:
            return self.data.get(key, default)

        def set(self, key: str, value: Any) -> None:
            self.data[key] = value

    ws = tmp_path / "Library" / "Book"
    write(ws / "Book.epub", b"compiled")
    book = {"name": "Book", "path": str(ws), "output_folder": str(ws), "type": "in_progress"}

    async def scenario():
        service, _shares, world = await make_service(tmp_path, library=FakeLibrary(tmp_path / "Library"))
        try:
            notes: list = []
            library = LibraryService(core=SharedCore({}), prefs=Prefs(), config={})
            ctx = LibraryContext(service=library, page=FakePage(), notify=lambda m, *a: notes.append(m),
                                 prefs=library.prefs, platform="ios")
            ctx.extras["drive_sync"] = lambda: service
            screen = LibraryScreen(parse_route("/library"), ctx)
            labels = [i.label for i in screen.card_actions(book).items]
            assert u10.DRIVE_SYNC_NOW_LABEL not in labels  # sync off
            await service.connect()
            assert u10.DRIVE_SYNC_NOW_LABEL not in [i.label for i in screen.card_actions(book).items]
            await service.set_enabled(True)
            items = {i.label: i for i in screen.card_actions(book).items}
            labels = list(items)
            assert labels.index(u10.DRIVE_SYNC_NOW_LABEL) == labels.index("↗ Share") + 1
            items[u10.DRIVE_SYNC_NOW_LABEL].on_select()
            wait_idle(service)
            assert tree(world)["Glossarion/Book.epub"] == b"compiled"
            assert notes[-1] == "Uploading “Book” to Google Drive"
            # the sheet's Cloud tab is built over the same service
            sheet = screen.open_filter_sheet(tab=3)
            assert sheet.cloud_tab.drive_facade.service is service
            sheet._on_dismiss()
            screen.dispose()
        finally:
            service.shutdown()

    run(scenario())


def test_the_library_settings_sheet_with_the_cloud_tab_serialises_in_a_flet_session(tmp_path):
    tb = _tb()
    from glossarion_mobile.ui.library.filter_sheet import FilterSheet
    from glossarion_mobile.ui.library.models import FilterState

    async def scenario():
        service, shares, _world = await make_service(tmp_path)
        try:
            conn, session = tb._fake_session("ios")
            page = session.page
            tab = u10.LibraryCloudTab(drive=service, shares=shares, platform="ios", page=page)
            sheet = FilterSheet(FilterState(), cloud_tab=tab, initial_tab=3)
            sheet.show(page)
            page.update()
            await settle()
            page.update()
            assert conn.bytes_sent > 0
            sheet._on_dismiss()
        finally:
            service.shutdown()

    run(scenario())


def test_a_folder_change_sends_the_recompile_to_the_new_folder(tmp_path):
    ws = tmp_path / "Library" / "Book"
    epub = write(ws / "Book.epub", b"first")

    async def scenario():
        service, _shares, world = await make_service(tmp_path, library=FakeLibrary(tmp_path / "Library"))
        try:
            await service.connect()
            await service.set_enabled(True)
            service.on_job_transition(snap("a", "compile_epub", str(ws), [epub]), "RUNNING")
            wait_idle(service)
            await service.set_folder("Books/Novels")
            write(ws / "Book.epub", b"second")
            service.on_job_transition(snap("b", "compile_epub", str(ws), [epub]), "RUNNING")
            wait_idle(service)
            files = tree(world)
            assert files["Books/Novels/Book.epub"] == b"second"
            assert files["Glossarion/Book.epub"] == b"first"  # the old copy is left alone, not updated
            assert service.ui_state()["last_sync"]["uploaded"] == 1
        finally:
            service.shutdown()

    run(scenario())


def test_a_chat_translation_uploads_when_the_book_moves_into_the_library(tmp_path):
    chat_ws = tmp_path / "Attachments" / "Novel"
    write(chat_ws / "Novel.epub", b"chat compile")
    lib_ws = tmp_path / "Library" / "Novel"
    epub = write(lib_ws / "Novel.epub", b"chat compile")
    plain = tmp_path / "Library" / "Plain"
    write(plain / "notes.txt", b"no compile")

    async def scenario():
        service, _shares, world = await make_service(tmp_path, library=FakeLibrary(tmp_path / "Library"))
        try:
            await service.connect()
            # off: the move sends nothing
            service.on_workspace_moved(str(chat_ws), str(lib_ws))
            wait_idle(service)
            assert world["clients"] == 0
            await service.set_enabled(True)
            # the chat job itself (direct_text, a chat book) is skipped by the DONE hook ...
            service.on_job_transition(snap("chat", "direct_text", str(chat_ws), [str(chat_ws / "Novel.epub")]),
                                      "RUNNING")
            wait_idle(service)
            assert world["clients"] == 0
            # ... and the auto-migrate's move into the Library queues it
            service.on_workspace_moved(str(chat_ws), str(lib_ws))
            wait_idle(service)
            assert tree(world)["Glossarion/Novel.epub"] == b"chat compile"
            # a moved book with nothing compiled is not queued in the compiled mode
            before = world["clients"]
            service.on_workspace_moved(str(tmp_path / "Attachments" / "Plain"), str(plain))
            wait_idle(service)
            assert world["clients"] == before and os.path.isfile(epub)
        finally:
            service.shutdown()

    run(scenario())


def test_a_done_triggered_upload_holds_the_keepalive_until_the_queue_is_empty(tmp_path):
    ws = tmp_path / "Library" / "Book"
    epub = write(ws / "Book.epub", b"compiled")

    class Keep:
        def __init__(self) -> None:
            self.calls: list = []

        def hold_now(self, text: str) -> bool:
            self.calls.append(("hold", text))
            return True

        async def release(self) -> None:
            self.calls.append(("release",))

    async def scenario():
        service, _shares, _world = await make_service(tmp_path, library=FakeLibrary(tmp_path / "Library"),
                                                      loop_post=False)
        keep = service.keepalive = Keep()
        try:
            await service.connect()
            await service.set_enabled(True)
            service.on_job_transition(snap("a", "compile_epub", str(ws), [epub]), "RUNNING")
            assert keep.calls[0][0] == "hold" and "Google Drive" in keep.calls[0][1]
            wait_idle(service)
            assert keep.calls == [keep.calls[0], ("release",)] and not service._held
            # a manual Sync now takes no hold
            assert service.sync_book({"output_folder": str(ws), "name": "Book", "path": str(ws)})
            wait_idle(service)
            assert len(keep.calls) == 2
        finally:
            service.shutdown()

    run(scenario())


def test_a_refused_disconnect_leaves_the_running_sync_alone(tmp_path):
    async def scenario():
        service, shares, _world = await make_service(tmp_path)
        cancelled: list = []
        service.core.worker.cancel_current = lambda: cancelled.append(True)
        busy = {"value": False}
        shares.__class__ = type("BusyShares", (type(shares),), {"uploading": property(lambda self: busy["value"])})
        try:
            await service.connect()
            busy["value"] = True  # a share-link upload is running
            with pytest.raises(sl.ShareError) as refused:
                await service.disconnect()
            assert refused.value.code == "busy" and not cancelled and service.signed_in()
            busy["value"] = False
            await service.disconnect()
            assert cancelled == [True] and not service.signed_in()
        finally:
            service.shutdown()

    run(scenario())


def test_sources_follow_the_mobile_rules():
    for rel in ("services/drive_sync.py", "ui/screens/cloud_sync.py", "ui/library/filter_sheet.py"):
        data = (APP_DIR / "glossarion_mobile" / rel).read_bytes()
        assert data.count(b"\r\n") in (0, data.count(b"\n")), rel  # one line ending, never mixed
        text = data.decode("utf-8")
        assert "import requests" not in text and "httpx" not in text, rel
        assert "pop_dialog(" not in text, rel
    service = (APP_DIR / "glossarion_mobile" / "services" / "drive_sync.py").read_text(encoding="utf-8")
    assert "import flet" not in service  # a pure service
    assert threading.current_thread() is threading.main_thread()
