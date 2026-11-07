"""Owner issue 3 (2026-11-07): with the Translation keys on 2 instances, a chat batch (＋ › Files with several
EPUBs) showed no file -> instance mapping and queued one plain job, unlike the desktop's Run (assignment dialog,
one run per instance). The chat BatchPlanCard now lists each file with its instance (``InstancePickers``, shared
with the Library TranslateSheet), Start queues one job per instance and the "Batch started" card names which
file runs on which instance.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
for _entry in (str(APP_DIR), str(SRC_DIR)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _backend_error() -> str:
    try:
        import direct_text_stream  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on the environment
        return f"the shared backend is not importable here ({exc})"
    return ""


_BACKEND_ERROR = _backend_error()
pytestmark = [pytest.mark.skipif(bool(_BACKEND_ERROR), reason=_BACKEND_ERROR or "backend importable"),
              pytest.mark.skipif(not _has("flet"), reason="flet is not installed")]


def _load(filename: str, alias: str):
    spec = importlib.util.spec_from_file_location(alias, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_TC = _load("test_chat.py", "_glossarion_tc_helpers_batch_instances") if not _BACKEND_ERROR else None
if _TC is not None:
    desktop_store_cls = _TC.desktop_store_cls  # noqa: F811  (pytest fixtures)
    isolated_env = _TC.isolated_env

#: The owner's screenshot: a keyless authgpt/ row on instance 1 and an sk- key on instance 2.
TWO_INSTANCES = {"use_multi_api_keys": True,
                 "multi_api_keys": [{"api_key": "", "model": "authgpt/gpt-5.6-sol", "enabled": True},
                                    {"api_key": "placeholder", "model": "gpt-5.6-sol", "enabled": True,
                                     "instance": 2}]}


class _InstanceJobs(_TC.FakeJobService if _TC is not None else object):
    """The jobs feature with ``submit(spec, instance=)``: records which instance each spec was queued on."""

    def __init__(self):
        super().__init__()
        self.instances = []

    async def submit(self, spec, instance=None):
        self.instances.append(instance)
        return await super().submit(spec)


class _Library:
    """LibraryService's multi-instance calls over a config dict (the real ``instance_assignment``)."""

    def __init__(self, config):
        self.config = dict(config)
        self.saved = []

    def config_snapshot(self):
        return dict(self.config)

    def instance_assignment(self, sources):
        from glossarion_mobile.services.library import LibraryService

        return LibraryService.instance_assignment(self, sources)

    def save_instance_assignment(self, assignment):
        self.saved.append(dict(assignment))


def _books(root, *names):
    paths = []
    for name in names:
        path = root / name
        path.write_bytes(b"PK\x03\x04")
        paths.append(str(path))
    return paths


def test_chat_batch_maps_files_to_instances_and_starts_one_job_each(desktop_store_cls, isolated_env):
    async def scenario():
        library = _Library(TWO_INSTANCES)
        jobs = _InstanceJobs()
        chat = _TC._chat_view(desktop_store_cls, isolated_env, jobs=jobs, library_service=lambda: library)
        view = chat.view
        try:
            view.load_chat("2")
            files = _books(isolated_env, "a.epub", "b.epub", "c.epub")
            batch = view.set_batch(files)
            card = view._batch_control(batch)
            pickers = card.instances
            assert pickers is not None, "a two-instance batch shows the file -> instance mapping"
            assert pickers.assignment == {files[0]: 1, files[1]: 2, files[2]: 1}
            assert [pickers.pickers[f].value for f in files] == ["1", "2", "1"]
            assert pickers.text.visible and pickers.text.value == (
                "2 parallel jobs · Instance 1: 2 books · Instance 2: 1 book")
            pickers.set(files[2], "2")  # the user moves c.epub to instance 2
            card = view._batch_control(view._pending_batch())  # a rebuilt card keeps the pick
            assert card.instances.assignment[files[2]] == 2

            job_id = await view._start_batch(view._pending_batch())
            assert job_id == "job1"
            assert jobs.instances == [1, 2]
            assert [tuple(spec.inputs) for spec in jobs.submitted] == [(files[0],), (files[1], files[2])]
            assert library.saved == [{files[0]: 1, files[1]: 2, files[2]: 2}]
            # The history save (debounced, 10 ms here) externalises an assistant body into
            # ``Chat Messages/NNNNNN-response.md`` and blanks the inline text, so read the card's body the
            # way the transcript does (``message_text``), after a save so the file path is the one checked.
            chat.adapter.flush()
            text = chat.adapter.message_text("2", len(chat.adapter.messages("2")) - 1)
            assert "**Instance 1:** a.epub" in text and "**Instance 2:** b.epub, c.epub" in text
            assert "2 parallel jobs" in text
        finally:
            for task in chat.calls.tasks:
                task.cancel()
            _TC._close(chat)

    asyncio.run(scenario())


def test_chat_batch_on_one_instance_keeps_one_plain_job(desktop_store_cls, isolated_env):
    async def scenario():
        one = {"use_multi_api_keys": True, "multi_api_keys": [dict(TWO_INSTANCES["multi_api_keys"][1], instance=1)]}
        library = _Library(one)
        jobs = _InstanceJobs()
        chat = _TC._chat_view(desktop_store_cls, isolated_env, jobs=jobs, library_service=lambda: library)
        view = chat.view
        try:
            view.load_chat("2")
            files = _books(isolated_env, "a.epub", "b.epub")
            batch = view.set_batch(files)
            assert view._batch_control(batch).instances is None
            await view._start_batch(view._pending_batch())
            assert jobs.instances == [None] and [tuple(s.inputs) for s in jobs.submitted] == [tuple(files)]
            assert library.saved == []
        finally:
            for task in chat.calls.tasks:
                task.cancel()
            _TC._close(chat)

    asyncio.run(scenario())
