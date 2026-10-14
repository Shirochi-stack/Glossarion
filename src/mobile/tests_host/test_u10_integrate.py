"""U10 Integrate: the cloud sync and share links are wired into the app.

* The route ``/settings/cloud`` (milestone U10) is implemented and grouped under Data in the Settings home.
* ``services/native``: the ``document`` event, the stub's document defaults, the long ``save_to_downloads``
  guard and ``ServiceLease`` (one foreground-service holder logic: sign-in, cloud save, uploads).
* iOS keeps a finished job's background grant while a cloud save holds it (``release_kept_background``).
* The cloud save's notification Stop stops only a save that holds the service alone.
* Deleting Library books forgets their cloud records and share links (never the cloud files).
* A share-link upload that fails while the app is hidden posts one notification with a route only.
* The self-test ``share_link_crypto`` passes on the host (the device smoke runs it in the real bundle).
* The real app on a fake Flet session installs both services, hands them to the Library, the Book page
  context and the Danger zone, and opens Settings › Cloud sync & sharing.

Run from src/mobile with the mobile venv:
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_u10_integrate.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from glossarion_mobile.services.background import BackgroundExecution  # noqa: E402
from glossarion_mobile.services.cloud_sync import CLOUD_HOLD, Keepalive  # noqa: E402
from glossarion_mobile.services.jobs import JobSnapshot, JobSpec, JobState  # noqa: E402
from glossarion_mobile.services.native import (  # noqa: E402
    DOCUMENT_METHODS,
    EVENTS,
    METHOD_TIMEOUTS,
    NativeBridge,
    NativeStub,
    ServiceHolds,
    ServiceLease,
)


def _has(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _load(name: str, alias: str):
    spec = importlib.util.spec_from_file_location(alias, Path(__file__).with_name(name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------------------------------
# route, Settings home
# ---------------------------------------------------------------------------------------------------


@pytest.mark.skipif(not _has("flet"), reason="flet not installed")
def test_cloud_route_is_a_shipped_data_page():
    from glossarion_mobile.ui.router import ROUTES_BY_NAME, parse_route
    from glossarion_mobile.ui.screens.base import SHIPPED_MILESTONES
    from glossarion_mobile.ui.screens.pages_feature import IMPLEMENTED_ROUTES, SCREEN_ROUTES
    from glossarion_mobile.ui.screens.cloud_sync import CLOUD_ROUTE, ROUTE_NAME
    from glossarion_mobile.ui.settings.settings_home import ROUTE_GROUPS
    from glossarion_mobile.services.cloud_sync import SETTINGS_ROUTE

    spec = ROUTES_BY_NAME["settings.cloud"]
    assert spec.milestone == "U10" and "U10" in SHIPPED_MILESTONES
    assert ROUTE_NAME == "settings.cloud" and CLOUD_ROUTE == SETTINGS_ROUTE == "/settings/cloud"
    assert parse_route(SETTINGS_ROUTE).name == "settings.cloud"
    assert "settings.cloud" in SCREEN_ROUTES and "settings.cloud" in IMPLEMENTED_ROUTES
    assert ROUTE_GROUPS["settings.cloud"] == "Data"


# ---------------------------------------------------------------------------------------------------
# native bridge
# ---------------------------------------------------------------------------------------------------


def test_document_events_and_stub_defaults():
    assert "document" in EVENTS
    stub = NativeStub("host", on_document=lambda e: None)
    assert stub.on_document is not None
    for name in DOCUMENT_METHODS:
        answer = asyncio.run(getattr(stub, name)())
        assert answer["ok"] is False and answer["error"] == "unavailable" and answer["retryable"] is False, name
    assert asyncio.run(stub.release("x")) is False and asyncio.run(stub.list_grants()) == []
    assert asyncio.run(stub.take_document_results()) == []
    bridge = NativeBridge(native=SimpleNamespace(is_stub=True))
    seen = []
    bridge.add_listener("document", seen.append)
    asyncio.run(bridge.native.on_document({"type": "progress", "op_id": "a", "written": 1, "total": 2}))
    assert seen and seen[0]["op_id"] == "a" and seen[0]["event"] == "document"
    assert METHOD_TIMEOUTS["save_to_downloads"] >= 900  # the extension's own limit for a whole-file copy


class _Native:
    """``NativeBridge``'s job-service surface."""

    def __init__(self, running: bool = False) -> None:
        self.service_holds = ServiceHolds()
        self.running = running
        self.calls: list = []

    async def is_job_service_running(self):
        return self.running

    async def start_job_service(self, title, text):
        self.calls.append(("start", title, text))
        self.running = True
        return True

    async def update_job_service(self, text=None, title=None):
        self.calls.append(("update", title, text))

    async def stop_job_service(self):
        self.calls.append(("stop",))
        self.running = False


def test_service_lease_join_remember_and_keep_service():
    async def scenario():
        native = _Native(running=True)
        lease = ServiceLease(native, "cloud")
        assert not lease.join("Glossarion", "Saving…", holder="jobs")  # nobody to join
        native.service_holds.hold("jobs", "Glossarion", "Chapter 9/9")
        assert lease.join("Glossarion", "Saving…", holder="jobs") and lease.held and not lease.started
        assert not await lease.update("Saving · 50%")  # the job's notification wins while it runs
        assert native.service_holds.release("jobs") == ("Glossarion", "Saving · 50%")  # remembered for later
        assert native.calls == []
        await lease.release(keep_service=lambda: True)  # a queued job takes the service over
        assert native.calls == [] and "cloud" not in native.service_holds
        alone = _Native()
        lease = ServiceLease(alone, "cloud")
        assert not await lease.acquire("Glossarion", "Saving…", may_start=False)  # background: no start
        assert not lease.failed and alone.calls == []
        assert await lease.acquire("Glossarion", "Saving…") and lease.started and lease.owns_alone()
        assert await lease.update("Saving · 10%")
        await lease.release(keep_service=lambda: False)
        assert alone.calls == [("start", "Glossarion", "Saving…"), ("update", "Glossarion", "Saving · 10%"), ("stop",)]

    asyncio.run(scenario())


# ---------------------------------------------------------------------------------------------------
# iOS: the finished job's background grant is kept for a cloud save
# ---------------------------------------------------------------------------------------------------


class _Bridge:
    def __init__(self) -> None:
        self.service_holds = ServiceHolds()
        self.log: list = []

    async def call(self, method, *args, default=None, **kwargs):
        self.log.append(method)
        if method == "begin_background_task":
            return 3
        return default


class _Jobs:
    active = None

    def view(self):
        return SimpleNamespace(active=self.active, queue=(), paused=False)

    def snapshot(self):
        return self.active


def _snap(state):
    return JobSnapshot(id="j", spec=JobSpec(kind="compile_epub", title="t"), state=state, created=0.0)


def test_ios_job_grant_is_kept_for_the_cloud_save_then_released():
    async def scenario():
        bridge, jobs = _Bridge(), _Jobs()
        background = BackgroundExecution(bridge, platform="ios", jobs=jobs)
        background.bg_task_id = 5
        background.continued_started = True
        keep = Keepalive(bridge, platform="ios", jobs=jobs, background=background)
        assert keep.hold_now("Saving…")  # synchronous, inside the transition callback
        await background.job_finished(_snap(JobState.DONE))
        assert "end_background_task" not in bridge.log and background.bg_task_id == 5  # kept for the save
        await keep.ensure("Saving…", may_start=False)
        await keep.release()
        assert bridge.log.count("end_background_task") == 2  # the cloud task and the kept job task
        assert "finish_continued_processing" in bridge.log and background.bg_task_id == -1
        assert CLOUD_HOLD not in bridge.service_holds

    asyncio.run(scenario())


def test_ios_without_a_cloud_save_ends_the_grant_at_once():
    async def scenario():
        bridge = _Bridge()
        background = BackgroundExecution(bridge, platform="ios", jobs=_Jobs())
        background.bg_task_id = 5
        await background.job_finished(_snap(JobState.DONE))
        assert bridge.log.count("end_background_task") == 1 and background.bg_task_id == -1
        await background.release_kept_background()  # nothing kept: no-op
        assert bridge.log.count("end_background_task") == 1

    asyncio.run(scenario())


# ---------------------------------------------------------------------------------------------------
# the cloud save's notification Stop
# ---------------------------------------------------------------------------------------------------


def test_notification_stop_stops_a_save_that_holds_the_service_alone(tmp_path):
    tc = _load("test_cloud_sync.py", "_glossarion_tc_helpers_u10integ")

    async def scenario():
        env = tc.Env(tmp_path)
        folder = env.book("BookStop", {"BookStop.epub": tc.EPUB1})
        env.cloud.gate = asyncio.Event()
        await env.service.pick_folder()
        await env.service.send_now(folder)
        for _ in range(100):
            await asyncio.sleep(0.01)
            if any(c[0] == "write_file" for c in env.cloud.calls):
                break
        assert env.bridge.service_holds.names() == [CLOUD_HOLD]
        # a job holding the service owns Stop: the save goes on
        env.bridge.service_holds.hold("jobs", "Glossarion", "Chapter 1/2")
        await env.service.on_foreground_event({"type": "button", "button_id": "stop"})
        assert not env.cloud.cancelled
        env.bridge.service_holds.release("jobs")
        await env.service.on_foreground_event({"type": "button", "button_id": "stop"})
        env.cloud.gate.set()
        await env.settle()
        assert env.cloud.cancelled  # the running write was stopped
        assert env.store.queued(folder) is not None  # kept for the next resume / job
        assert CLOUD_HOLD not in env.bridge.service_holds
        assert env.bridge.methods().count("stop_job_service") == 1

    asyncio.run(scenario())


# ---------------------------------------------------------------------------------------------------
# Library deletion
# ---------------------------------------------------------------------------------------------------


def test_deleted_books_drop_their_cloud_records_and_share_links(tmp_path):
    from glossarion_mobile.services.library import DeleteTarget, LibraryService

    gone_book = tmp_path / "Output" / "Gone"
    kept_book = tmp_path / "Output" / "Kept"
    kept_book.mkdir(parents=True)
    raw = tmp_path / "Library" / "Raw" / "Gone.epub"
    calls: dict = {}
    owner = SimpleNamespace(
        cloud_sync=SimpleNamespace(forget_books_threadsafe=lambda ids: calls.setdefault("cloud", list(ids))),
        share_links=SimpleNamespace(forget_books_blocking=lambda ids: calls.setdefault("share", list(ids))))
    targets = [DeleteTarget("Gone", str(gone_book), True, {"output_folder": str(gone_book)}),
               DeleteTarget("Gone.epub", str(raw), False, {"output_folder": str(gone_book), "path": str(raw)}),
               DeleteTarget("Kept", str(kept_book), True, {"output_folder": str(kept_book)})]
    LibraryService._forget_deleted_blocking(owner, targets)
    assert calls["cloud"] == [os.path.abspath(gone_book)]  # book identities only; the existing book stays
    assert sorted(calls["share"]) == sorted([os.path.abspath(gone_book), str(raw)])
    calls.clear()
    LibraryService._forget_deleted_blocking(SimpleNamespace(cloud_sync=None, share_links=None), targets)
    assert calls == {}


# ---------------------------------------------------------------------------------------------------
# share-link failure notification
# ---------------------------------------------------------------------------------------------------


@pytest.mark.skipif(not _has("flet"), reason="flet not installed")
def test_share_link_failure_notifies_once_while_hidden_with_a_route_only():
    from glossarion_mobile.app import GlossarionApp
    from glossarion_mobile.services.share_links import UploadState

    async def scenario():
        shown, spawned = [], []

        async def _show(nid, title, body, *, channel, route):
            shown.append((nid, title, body, channel, route))
            return True

        state = UploadState("failed", "gofile", "Novel.epub", message="Could not reach upload.gofile.io",
                            book="/x/Output/Novel")
        background = SimpleNamespace(app_visible=False)
        fake = SimpleNamespace(
            share_links=SimpleNamespace(state=state),  # ShareLinkService.state is a property, not a method
            jobs=SimpleNamespace(background=background, notifications=SimpleNamespace(_show=_show)),
            native=None,
            library=SimpleNamespace(bid_for=lambda row: "0123456789ab"),
            dispatcher=SimpleNamespace(spawn=lambda coro: spawned.append(asyncio.ensure_future(coro))),
        )
        GlossarionApp._on_share_link_change(fake, "links")  # not an upload change
        GlossarionApp._on_share_link_change(fake, "upload")
        await asyncio.gather(*spawned)
        assert len(shown) == 1
        _nid, title, body, channel, route = shown[0]
        assert title == "Upload to Gofile failed" and "Novel.epub" in body and channel == "jobs.action"
        assert route == "/library/book/0123456789ab?tab=output" and "/x/" not in route
        background.app_visible = True  # in front: the upload sheet shows the failure
        GlossarionApp._on_share_link_change(fake, "upload")
        await asyncio.gather(*spawned)
        assert len(shown) == 1

    asyncio.run(scenario())


# ---------------------------------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------------------------------


@pytest.mark.skipif(not (_has("cryptography") and _has("websockets")), reason="cryptography / websockets missing")
def test_selftest_share_link_crypto_passes_on_the_host():
    from glossarion_mobile.diagnostics import selftest

    assert ("share_link_crypto", selftest.check_share_link_crypto) in selftest.SUITES["smoke"]
    detail = selftest.check_share_link_crypto(selftest.Context(strict=True))
    assert detail["ece"] == "rfc8188" and detail["client"] == "websockets.sync.client"


# ---------------------------------------------------------------------------------------------------
# the real app
# ---------------------------------------------------------------------------------------------------

_TB = _load("test_bootstrap.py", "_glossarion_tb_helpers_u10integ")
storage = _TB.storage
app_env = _TB.app_env


@pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet/msgpack not installed")
@pytest.mark.skipif(not (_has("requests") and _has("bs4")), reason="the real features drive the backend")
def test_real_app_installs_cloud_sync_and_share_links(app_env):
    tf = _load("test_ui_foundations.py", "_glossarion_tf_helpers_u10integ")

    async def scenario():
        _m, _conn, _session, _page, app = await tf._start("android")
        try:
            await tf._wait(lambda: app.state.engine_ready, timeout=60)
            cloud, shares = app.cloud_sync, app.share_links
            assert cloud is not None and shares is not None
            assert app.library.cloud_sync is cloud and app.library.share_links is shares
            assert cloud.platform == "android" and cloud.docs.bridge is app.native
            extras = app.library_feature.context().extras
            assert extras["cloud_sync"]() is cloud and extras["share_links"]() is shares
            match = await app.navigate("/settings/cloud")
            assert match is not None and match.name == "settings.cloud"
            assert type(app.shell.top_screen).__name__ == "CloudSyncScreen"
            await app.navigate("/settings/danger")
            danger = app.shell.top_screen
            assert type(danger).__name__ == "DangerZoneScreen" and callable(danger.before_wipe)
            await app.navigate("/settings/storage")
            storage_screen = app.shell.top_screen
            assert storage_screen.files is app.files and storage_screen.cloud() is cloud
        finally:
            await tf._stop(app)
            cloud = getattr(app, "cloud_sync", None)
            if cloud is not None:
                cloud.close()

    asyncio.run(scenario())
