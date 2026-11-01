"""9.15.7 batch, issue 4 (owner, phone): "Why does clicking resume open a popup instead of just resuming. The
pop is essentially a duplicate page. Fix this."

The chat Result card's Resume / Retry failed of a book whose workspace moved into the Library used to open
the Library translate sheet ('Translate "<book>"': Model / Language / Glossary / All chapters chips, Run
options, Start). It now starts that book's translation run at once (``translate_sheet.resume_translation``)
with the chat's model / profile / language and the card's Run options, like the desktop's resume; the
sheet stays for a fresh Translate. What would block the run is one short message, never the sheet.

The Jobs page / Job page Resume (``JobService.resume``) never opened the sheet and is unchanged.

Run from src/mobile (mobile venv):
    python -m pytest -p no:cacheprovider -W ignore tests_host/test_mobile_batch_9157_resume.py -q
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from pathlib import Path

MOBILE_DIR = Path(__file__).resolve().parents[1]
for _path in (MOBILE_DIR / "app", MOBILE_DIR / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # its dataclasses look their module up
    spec.loader.exec_module(module)
    return module


RO = _load("_glossarion_run_options_batch_9157_resume", Path(__file__).with_name("test_run_options_resume.py"))
TC = RO.TC
desktop_store_cls = TC.desktop_store_cls
isolated_env = TC.isolated_env
needs_flet = TC.needs_flet


class _Page:
    def __init__(self) -> None:
        self.dialogs: list = []

    def show_dialog(self, control) -> None:
        self.dialogs.append(control)


def _library_ctx(raw: str, *, model: str = "authds/flash"):
    """A LibraryContext stand-in: a service whose ``translate_spec`` / ``submit`` are recorded."""
    from glossarion_mobile.services.library import LibraryService

    submitted: list = []
    said: list = []

    async def io(fn, *args):
        return fn(*args)

    async def submit(spec):
        submitted.append(spec)
        return "job-1"

    service = types.SimpleNamespace(
        jobs=object(), has_job_kind=lambda kind: kind == "translate",
        cfg=lambda key, default=None: {"model": model}.get(key, default),
        raw_source=lambda book: raw,
        origin_for=lambda book: {"type": "library", "bid": "bid7", "label": f"Library · {book.get('name')}"},
        submit=submit)
    service.translate_spec = lambda *a, **k: LibraryService.translate_spec(service, *a, **k)
    page = _Page()
    ctx = types.SimpleNamespace(service=service, page=page, io=io, haptic=lambda kind: None,
                                say=lambda message, *a: said.append(message))
    return ctx, submitted, said, page


def test_resume_starts_the_library_run_without_the_translate_sheet(tmp_path):
    from glossarion_mobile.ui.library import translate_sheet

    raw = tmp_path / "Book.epub"
    raw.write_bytes(b"PK")
    ctx, submitted, said, page = _library_ctx(str(raw))
    overrides = {"model": "authds/flash", "output_language": "English", "batch_size": 9}
    job_id = asyncio.run(translate_sheet.resume_translation(ctx, {"name": "Book"}, config_overrides=overrides))
    assert job_id == "job-1"
    assert page.dialogs == []  # no 'Translate "Book"' sheet, no other dialog
    (spec,) = submitted
    assert spec.kind == "translate" and spec.inputs == (str(raw),)
    assert spec.params == {"config_overrides": overrides} and spec.origin["bid"] == "bid7"
    assert said == ["Resuming “Book”"]  # at most a toast


def test_resume_guards_are_short_messages_not_the_sheet(tmp_path):
    from glossarion_mobile.ui.library import translate_sheet

    # no raw source: the desktop "missing raw" reason
    ctx, submitted, said, page = _library_ctx("")
    assert asyncio.run(translate_sheet.resume_translation(ctx, {"name": "Book"})) is None
    assert submitted == [] and page.dialogs == [] and said == [translate_sheet.MISSING_RAW_REASON]
    # a route excluded on mobile (the chat Send rule): one line, nothing queued
    raw = tmp_path / "Book.epub"
    raw.write_bytes(b"PK")
    ctx, submitted, said, page = _library_ctx(str(raw))
    assert asyncio.run(translate_sheet.resume_translation(ctx, {"name": "Book"},
                                                          config_overrides={"model": "ocz/some-model"})) is None
    assert submitted == [] and page.dialogs == [] and len(said) == 1 and said[0].startswith("Could not resume")
    # the job service refuses (e.g. its preflight): no toast claiming it resumed, and no sheet
    ctx, submitted, said, page = _library_ctx(str(raw))

    async def refused(spec):
        submitted.append(spec)
        return None

    ctx.service.submit = refused
    assert asyncio.run(translate_sheet.resume_translation(ctx, {"name": "Book"})) is None
    assert len(submitted) == 1 and said == [] and page.dialogs == []


def test_chat_feature_hands_the_chats_settings_and_run_options_to_the_library_run(tmp_path, monkeypatch):
    """``ChatFeature.library_translate`` (the chat env's hook): the card's Run options and the chat's model /
    language now, the chat's keys winning (``run_request.with_chat_overrides``, as a chat Resume)."""
    from glossarion_mobile.ui.chat.integration import ChatFeature
    from glossarion_mobile.ui.library import translate_sheet

    resumed: list = []

    async def fake_resume(ctx, book, config_overrides=None):
        resumed.append((ctx, dict(book), dict(config_overrides or {})))
        return "job-1"

    def forbidden(*a, **k):
        raise AssertionError("Resume must not open the translate sheet")

    monkeypatch.setattr(translate_sheet, "resume_translation", fake_resume)
    monkeypatch.setattr(translate_sheet, "open_translate_sheet", forbidden)
    book = {"name": "Book", "output_folder": str(tmp_path / "book")}
    library = types.SimpleNamespace(book_for_bid=lambda bid: book if bid == "bid7" else None)

    async def library_book(folder):
        return "bid7"

    feature = types.SimpleNamespace(
        app=types.SimpleNamespace(notify=lambda *a: None,
                                  library_feature=types.SimpleNamespace(context=lambda: "library-ctx")),
        library_service=lambda: library, library_book=library_book)
    job_id = asyncio.run(ChatFeature.library_translate(
        feature, str(tmp_path / "book"), "", run_config={"batch_size": 9, "model": "stale"},
        chat_overrides={"model": "authds/flash", "target_language": "English"}))
    assert job_id == "job-1"
    assert resumed == [("library-ctx", book, {"batch_size": 9, "model": "authds/flash",
                                              "output_language": "English"})]


@needs_flet
def test_result_card_resume_of_a_moved_book_hands_the_chats_overrides(desktop_store_cls, isolated_env):
    """The Result card's Resume (``ChatView._resume_turn``) hands the chat's run overrides with the card's Run
    options, so the Library run starts with the model / profile / language the card shows."""
    tmp_path = isolated_env
    handed: list = []

    async def library_translate(folder, source="", **extra):
        handed.append((folder, extra))
        return "job-1"

    async def scenario():
        chat = TC._chat_view(desktop_store_cls, tmp_path, library_translate=library_translate)
        view = chat.view
        try:
            await RO._settle(chat.calls)
            item = TC._job_item(chat.adapter)
            moved = str(tmp_path / "Output" / "book")

            async def state(fn, *args):
                return (moved, False) if fn == view._job_workspace_state else fn(*args)

            view.env.run_io = state
            view.run_overrides = lambda cid: {"model": "authds/flash"}
            await view._resume_turn(item)
            assert handed[-1] == (moved, {"chat_overrides": {"model": "authds/flash"}, "chat_cid": chat.cid})
            assert chat.jobs.submitted == []  # the Library runs the book, not the chat's run root
        finally:
            TC._close(chat)

    asyncio.run(scenario())
