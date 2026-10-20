"""Owner report (2026-10-20): after translating, from a chat, an EPUB the Library already held, the
Library showed TWO cards for one book ("IN PROGRESS ⏳ 1/107" and "NOT STARTED 🆕", same title and cover).

Root cause: the auto-migrate merges a finished chat book only into the Library folder of the SAME NAME
(``<output root>/<attachment stem>``). The chat's workspace is named after its own copy of the file,
which is not always the Library's: FileBridge names the Inbox copy "X (2).epub" when another "X.epub"
sits there, the Library's copy is "X (2).epub" when Library/Raw already had another "X.epub", and the
chat cuts the stem to 120 characters (``direct_text_store._attachment_output_subfolder``) while the
Library folder keeps the whole stem. The chat book then became a second output folder next to the
Library's import (``scan_output_folders`` lists one card per folder), and the import stayed "Not
started".

Fixed: the auto-migrate merges into the Library folder of the same book under another name (its
``source_epub.txt`` names the same bytes and its chapters match, ``workspace_sync.same_book``), keeping
the Library's raw pointer.

Real app objects as in ``test_devfix_issue4`` (fake LLM server, temp storage).
"""

from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location("_glossarion_devfix4_twin", Path(__file__).with_name("test_devfix_issue4.py"))
D = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(D)

iso = D.iso
migrate_calls = D.migrate_calls
app_env = D.app_env
storage = D.storage
pytestmark = D.pytestmark

NAME = "[TS] Alone Mecha Genius.epub"
LONG_NAME = "[TS] Alone Mecha Genius " + "x" * 110 + ".epub"  # the chat cuts the stem to 120 characters


def _registry() -> list:
    path = Path(os.environ["GLOSSARION_LIBRARY_DIR"]) / "library_raw_inputs.txt"
    return path.read_text(encoding="utf-8").splitlines() if path.is_file() else []


async def _books(app) -> list:
    await app.library.refresh(quiet=True, reason="test")
    return list(app.library.snapshot.all_books())


def _setup_variant(app, iso, variant: str, name: str) -> None:
    """Another book named like ours already sits in the Inbox / Library/Raw."""
    from glossarion_mobile.diagnostics import fixtures

    if variant in ("inbox_taken", "library_raw_taken"):
        other = fixtures.build_tiny_epub(iso.picks / "other" / name, chapters=3)
        if variant == "inbox_taken":
            D._import(app, other)
        else:
            shutil.copyfile(other, Path(app.files.library_raw_dir()) / name)


@pytest.mark.parametrize("variant", ["inbox_taken", "library_raw_taken", "long_name"])
def test_chat_book_merges_into_the_not_started_library_book(iso, migrate_calls, variant):
    name = LONG_NAME if variant == "long_name" else NAME
    source = iso.picks / name
    shutil.copyfile(D.SELFTEST_EPUB, source)

    async def scenario(server):
        tf, app, tester, driver = await D._start_app({})
        try:
            await D._until(lambda: getattr(app, "chat_feature", None) is not None and app.library is not None, 30)
            D._configure(app, server)
            notes = D._record_notes(app)
            _setup_variant(app, iso, variant, name)
            # Library ＋ Import: Library/Raw copy + its Not started workspace
            imported = app.files.import_paths([str(source)], target="library", names=[name])
            library_raw = imported[0].path
            before = await _books(app)
            assert [b.get("translation_state") for b in before] == ["not_started"], before
            library_folder = Path(before[0]["output_folder"])

            # the same EPUB from a chat: its Inbox copy (or its workspace) has another name
            raw = D._import(app, source)
            chats = app.chat_feature.chats
            cid = chats.new_chat()
            await D._finish(app, await D._send(app, driver, cid, raw))
            assert await D._until(lambda: chats.attachment_folders(cid) == [], 30), chats.attachment_folders(cid)
            call = [c for c in migrate_calls if c["chat"] == str(cid)]
            assert len(call) == 1 and call[0]["merge_callback"] and call[0]["result"].get("ok"), migrate_calls

            books = await _books(app)
            assert len(books) == 1, [(b.get("translation_state"), b.get("output_folder")) for b in books]
            book = books[0]
            assert D._same(book["output_folder"], library_folder)
            assert book.get("translation_state") == "completed"
            assert D._progress(library_folder) == {"total": D.CHAPTERS, "completed": D.CHAPTERS}
            out_root = Path(app.paths.output)
            assert sorted(p.name for p in out_root.iterdir() if p.name != "Direct Text") == [library_folder.name]
            # the Library book keeps its own raw; the chat's copy of the same bytes is no second raw input
            assert D._same((library_folder / "source_epub.txt").read_text(encoding="utf-8").strip(), library_raw)
            assert D._same(app.library.raw_source(book), library_raw)
            if not D._same(raw, library_raw):
                assert not any(D._same(line.strip(), raw) for line in _registry()), _registry()
            # the chat's stored paths followed the workspace into the Library folder
            folders = {m[4] for m in chats.messages(cid) if m[0] == "assistant" and len(m) > 4 and m[4]}
            assert any(D._same(f, library_folder) for f in folders), folders
            assert not any("Attachments" in Path(f).parts for f in folders), folders
            # silent (owner 2026-10-20): no "Added to the Library · Open book" snackbar
            assert not [n for n in notes if n[0] == "Added to the Library"], [n[:2] for n in notes]
        finally:
            await D._stop_app(tf, app)

    D._run_scenario(scenario)


def test_chat_book_merges_into_the_library_book_that_has_a_translation(iso, migrate_calls):
    """The Library book already holds a translation (the first chat sent the Library's own copy); the
    second chat's copy lands in the Inbox as "X (2).epub" and still merges into that book."""
    source = iso.picks / NAME
    shutil.copyfile(D.SELFTEST_EPUB, source)

    async def scenario(server):
        tf, app, tester, driver = await D._start_app({})
        try:
            await D._until(lambda: getattr(app, "chat_feature", None) is not None and app.library is not None, 30)
            D._configure(app, server)
            _setup_variant(app, iso, "inbox_taken", NAME)
            library_raw = app.files.import_paths([str(source)], target="library", names=[NAME])[0].path
            chats = app.chat_feature.chats
            cid1 = chats.new_chat()  # Library › Open in chat: the Library's own copy
            await D._finish(app, await D._send(app, driver, cid1, library_raw))
            assert await D._until(lambda: chats.attachment_folders(cid1) == [], 30)
            first = await _books(app)
            assert len(first) == 1 and first[0].get("translation_state") == "completed", first
            library_folder = Path(first[0]["output_folder"])

            raw = D._import(app, source)
            assert Path(raw).stem.endswith("(2)"), raw
            cid2 = chats.new_chat()
            await D._finish(app, await D._send(app, driver, cid2, raw))
            assert await D._until(lambda: chats.attachment_folders(cid2) == [], 30), chats.attachment_folders(cid2)
            books = await _books(app)
            assert len(books) == 1, [(b.get("translation_state"), b.get("output_folder")) for b in books]
            assert D._same(books[0]["output_folder"], library_folder)
            assert D._progress(library_folder) == {"total": D.CHAPTERS, "completed": D.CHAPTERS}
            assert not (Path(app.paths.output) / Path(raw).stem).exists()
        finally:
            await D._stop_app(tf, app)

    D._run_scenario(scenario)

