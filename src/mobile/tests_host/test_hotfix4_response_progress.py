"""Response progress opens its own source before output and never another turn's workspace."""
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))

from glossarion_mobile.ui.chat.chat_view import ChatView
from glossarion_mobile.ui.chat.cards import JobCard, CardPhase


def test_progress_button_and_bar_open_source_before_output(tmp_path):
    source = tmp_path / 'new-book.txt'
    source.write_text('Chapter one\n\nText.')
    opened, notes = [], []
    async def run_io(fn, *args):
        return fn(*args)
    view = ChatView.__new__(ChatView)
    view.cid = 'chat'
    view.env = NS(chats=NS(available=True), runs=NS(run_for=lambda cid: None),
                  run_io=run_io, open_progress=lambda *args, **kwargs: opened.append((args, kwargs)))
    view.notify = notes.append
    view._turn_source = lambda item: str(source)
    view._job_workspace = lambda item: ''
    view._last_output_folder = lambda: str(tmp_path / 'unrelated-old-book')
    item = NS(index=2)
    card = JobCard(phase=CardPhase('running'),
                   on_action=lambda action: asyncio.run(view._open_progress_async(item)))
    card._run_action('progress')
    card.progress_link.on_click(None)
    assert opened == [(('', str(source)), {'chapters': True})] * 2
    assert card.progress_link.visible and not notes


def test_glossary_stage_keeps_its_planned_workspace(tmp_path):
    opened = []
    async def run_io(fn, *args):
        return fn(*args)
    run = NS(output_dir=str(tmp_path / 'planned'), output_folder='')
    view = ChatView.__new__(ChatView)
    view.cid = 'chat'
    view.env = NS(chats=NS(available=True), runs=NS(run_for=lambda cid: run), run_io=run_io,
                  open_progress=lambda *args, **kwargs: opened.append((args, kwargs)))
    view._turn_source = lambda item: 'book.epub'
    view._job_workspace = lambda item: ''
    view._run_turn = lambda run: 2
    view.notify = lambda message: None
    asyncio.run(view._open_progress_async(NS(index=2)))
    assert opened == [((run.output_dir, 'book.epub'), {})]
    opened.clear()
    asyncio.run(view._open_progress_async(NS(index=0)))
    assert opened == [(('', 'book.epub'), {'chapters': True})]


def test_real_progress_router_opens_source_when_planned_folder_has_no_output(tmp_path):
    from glossarion_mobile.ui.chat.integration import ChatFeature

    source = tmp_path / 'book.txt'
    source.write_text('Chapter one\n\nText.')
    books, routes, notes = [], [], []
    async def run_io(fn, *args):
        return fn(*args)
    def bid_for(book):
        books.append(book)
        return 'source-book'
    feature = ChatFeature.__new__(ChatFeature)
    feature._run_io = run_io
    feature.app = NS(library=NS(bid_for=bid_for, output_roots=lambda: [str(tmp_path / 'output')]),
                     navigate_to=lambda *a, **kw: routes.append((a, kw)), notify=notes.append)
    result = asyncio.run(feature.open_progress(str(tmp_path / 'planned'), str(source)))
    assert result == 'source-book' and not notes
    assert books[0]['raw_source_path'] == str(source) and books[0]['progress_file'] == ''
    assert routes == [(('tools.progress', None, {'out': 'source-book'}), {'alone': True})]
