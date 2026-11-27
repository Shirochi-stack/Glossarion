"""Interactive Images progress file (stale detection, rows, stats, actions) and the runner end to end
with a fake model client (GUI-free)."""
import json
import os
import sys
import threading
from io import BytesIO
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("GLOSSARION_HTTP_LOG", "0")

import interactive_images_core as core  # noqa: E402
import interactive_images_progress_core as ipc  # noqa: E402
import interactive_images_runner as runner  # noqa: E402


def _png(color=(200, 20, 20)):
    from PIL import Image
    buffer = BytesIO()
    Image.new('RGB', (8, 8), color).save(buffer, format='PNG')
    return buffer.getvalue()


CH1 = ('<html><head><title>Ch 1</title></head><body>\n<h1>Chapter 1</h1>\n'
       '<p>Mira walked to the lantern gate.</p>\n<p>"Hello," said Tomas.</p>\n<p>[Status Window]</p>\n</body></html>')
CH2 = ('<html><head><title>Ch 2</title></head><body>\n<h1>Chapter 2</h1>\n'
       '<p>The rain kept falling on the river market.</p>\n<p>Nobody noticed the stranger.</p>\n</body></html>')


@pytest.fixture
def book(tmp_path):
    out = tmp_path / 'book'
    out.mkdir()
    (out / 'response_001_ch1.html').write_text(CH1, encoding='utf-8')
    (out / 'response_002_ch2.html').write_text(CH2, encoding='utf-8')
    (out / 'response_003_ch3.html').write_text('<p>half done</p>', encoding='utf-8')
    (out / 'translation_progress.json').write_text(json.dumps({'chapters': {
        'a': {'actual_num': 1, 'output_file': 'response_001_ch1.html', 'status': 'completed'},
        'b': {'actual_num': 2, 'output_file': 'response_002_ch2.html', 'status': 'completed'},
        'c': {'actual_num': 3, 'output_file': 'response_003_ch3.html', 'status': 'in_progress'},
        'd': {'actual_num': 4, 'output_file': 'response_002_ch2.html', 'status': 'merged'},
    }}), encoding='utf-8')
    (out / 'glossary.csv').write_text('type,raw_name,translated_name\ncharacter,미라,Mira\ncharacter,토마스,Tomas\n',
                                      encoding='utf-8')
    return out


class FakeClient:
    """Text replies from ``plans`` (by chapter title), images as PNG bytes in the thread-local slot."""

    def __init__(self, kind, plans, sent, fail_images=(), gate=None):
        self.kind = kind
        self.plans = plans
        self.sent = sent
        self.fail_images = set(fail_images)
        self.gate = gate
        self.tls = threading.local()
        self.count = 0

    def _get_thread_local_client(self):
        return self.tls

    def send(self, messages, temperature=None, max_tokens=None, context=None):
        self.sent.append((self.kind, context, messages))
        if self.kind == 'text':
            prompt = messages[-1]['content']
            for title, reply in self.plans.items():
                if f"Chapter: {title}" in prompt:
                    return reply, 'stop'
            return '{"images": [], "classes": []}', 'stop'
        self.count += 1
        if self.gate is not None:
            self.gate(self.count)
        if self.count in self.fail_images:
            raise RuntimeError("image model refused")
        self.tls._last_generated_image_bytes = _png()
        return '', 'stop'


def _factory(plans, sent, **kwargs):
    clients = {}

    def make(kind, model, config, output_dir):
        clients.setdefault(kind, FakeClient(kind, plans, sent, **kwargs))
        return clients[kind]
    return make


PLAN1 = ('```json\n{"images":[{"after_paragraph":3,"quote":"Mira walked","characters":["mira","Tomas"],'
         '"prompt":"A lantern gate at dusk"}],"classes":[{"paragraph":3,"quote":"\\"Hello,\\"","class":"glx-dialogue"},'
         '{"paragraph":4,"quote":"[Status Window]","class":"glx-status-window"}]}\n```')
PLAN2 = '{"images":[{"after_paragraph":2,"quote":"The rain kept","characters":[],"prompt":"Rain on a market"}]}'

CONFIG = {'model': 'gpt-test', 'interactive_images_image_model': 'gemini-3-pro-image-preview',
          'interactive_css_enabled': True, 'interactive_css_preset': 'LitRPG Status Window'}


def _refs(book):
    refs = core.empty_references()
    style = book / 'style.png'
    style.write_bytes(_png((0, 0, 200)))
    mira = book / 'mira.png'
    mira.write_bytes(_png((0, 200, 0)))
    core.add_reference_image(str(book), refs, str(mira), 'Mira')
    core.add_reference_image(str(book), refs, str(style), None)
    core.save_references(str(book), refs)


# ---- translated chapters / progress file ---------------------------------------------------------

def test_translated_chapters_and_rows_before_any_build(book):
    chapters = ipc.translated_chapters(str(book))
    assert [(c.key, c.number, c.status) for c in chapters] == [
        ('response_001_ch1.html', 1, 'completed'), ('response_002_ch2.html', 2, 'completed'),
        ('response_003_ch3.html', 3, 'in_progress')]
    model = ipc.open_progress(str(book))
    rows = ipc.image_rows(model)
    assert [r.status for r in rows] == ['not_built'] * 3
    assert ipc.image_stats(model, rows)['not_built'] == 3
    assert ipc.pending_keys(model) == [c.key for c in chapters]
    assert ipc.find_progress(str(book)) is None


def test_mutate_is_locked_and_atomic_across_threads(tmp_path):
    path = str(tmp_path / 'Interactive_Images' / core.PROGRESS_FILENAME)

    def worker(i):
        for j in range(15):
            ipc.mutate(path, lambda data, k=f"{i}-{j}": data['chapters'].__setitem__(k, {'status': 'pending'}))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    data = ipc.read_progress_file(path)
    assert len(data['chapters']) == 60
    assert not [n for n in os.listdir(os.path.dirname(path)) if n.endswith('.tmp')]
    before = ipc.progress_signature(path)
    ipc.set_status(path, '0-0', 'failed', 'x')
    assert ipc.progress_signature(path) != before


# ---- the runner ------------------------------------------------------------------------------------

def test_run_builds_images_classes_and_records_entries(book):
    _refs(book)
    sent = []
    result = runner.run_interactive_images(dict(CONFIG), str(book), client_factory=_factory(
        {'Ch 1': PLAN1, 'Ch 2': PLAN2}, sent))
    assert (result.completed, result.failed, result.rejected, result.skipped) == (2, 0, 0, 1)
    html1 = (book / 'response_001_ch1.html').read_text(encoding='utf-8')
    assert core.check_text_unchanged(CH1, html1) == (True, '')
    assert ('<p>Mira walked to the lantern gate.</p><figure class="glx-figure" data-glx="img"><img class="glx-illustration" '
            'src="Interactive_Images/Glossarion_ch001_1.png" alt=""/></figure>') in html1
    assert 'class="glx-dialogue" data-glx-cls="glx-dialogue"' in html1
    assert core.BOOK_WRAPPER_OPEN in html1
    folder = book / core.INTERACTIVE_FOLDER
    assert (folder / 'Glossarion_ch001_1.png').is_file() and (folder / 'Glossarion_ch002_1.png').is_file()
    assert (folder / 'glossarion_interactive.css').is_file()
    # the image request: the template filled with the scene, the matched names and the references in slot order
    image_requests = [m for kind, _c, m in sent if kind == 'image']
    first = image_requests[0][-1]['content']
    assert first[0]['text'] == "A lantern gate at dusk\nCharacters: Mira, Tomas\nImage 1: Mira\nImage 2: art style"
    assert [part['type'] for part in first] == ['text', 'image_url', 'image_url']
    assert all(c == runner.IMAGE_CONTEXT for kind, c, _m in sent if kind == 'image')
    assert all(c == runner.PLAN_CONTEXT for kind, c, _m in sent if kind == 'text')
    # the planner request is the profile prompt with its placeholders only
    planner = [m for kind, _c, m in sent if kind == 'text'][0]
    assert len(planner) == 1 and planner[0]['role'] == 'user'
    assert "Characters in this book:\nMira\nTomas\n" in planner[0]['content']
    assert "[2] Mira walked to the lantern gate." in planner[0]['content']
    model = ipc.open_progress(str(book))
    rows = {r.key: r for r in ipc.image_rows(model)}
    assert rows['response_001_ch1.html'].status == 'completed' and rows['response_001_ch1.html'].images == 1
    assert rows['response_003_ch3.html'].status == 'skipped'
    entry = model.data['chapters']['response_001_ch1.html']
    assert entry['images'][0]['refs'] == ['references/mira_1.png', 'references/style_1.png']
    assert entry['source_hash'] == core.visible_text_hash(CH1)
    assert ipc.tab_count_text(ipc.image_stats(model)) == '2/3'


def test_retranslated_chapter_shows_stale_and_a_rerun_replaces_images(book):
    sent = []
    factory = _factory({'Ch 1': PLAN1}, sent)
    runner.run_interactive_images(dict(CONFIG), str(book), ['response_001_ch1.html'], client_factory=factory)
    path = book / 'response_001_ch1.html'
    path.write_text(path.read_text(encoding='utf-8').replace('lantern gate', 'iron gate'), encoding='utf-8')
    model = ipc.open_progress(str(book))
    row = next(r for r in ipc.image_rows(model) if r.key == 'response_001_ch1.html')
    assert row.status == 'stale' and row.stale
    assert 'response_001_ch1.html' in ipc.pending_keys(model)
    (book / core.INTERACTIVE_FOLDER / 'Glossarion_ch001_2.png').write_bytes(_png())  # from an older run
    runner.run_interactive_images(dict(CONFIG), str(book), ['response_001_ch1.html'], client_factory=factory)
    html = path.read_text(encoding='utf-8')
    assert html.count('data-glx="img"') == 1 and html.count(core.BOOK_WRAPPER_OPEN) == 1
    assert core.chapter_images_on_disk(str(book), 1) == ['Glossarion_ch001_1.png']
    row = next(r for r in ipc.image_rows(ipc.open_progress(str(book))) if r.key == 'response_001_ch1.html')
    assert row.status == 'completed'


def test_a_plan_failure_never_touches_the_chapter(book):
    runner.run_interactive_images(dict(CONFIG), str(book), ['response_002_ch2.html'],
                                  client_factory=_factory({'Ch 2': 'I cannot help with that.'}, []))
    assert (book / 'response_002_ch2.html').read_text(encoding='utf-8') == CH2
    row = next(r for r in ipc.image_rows(ipc.open_progress(str(book))) if r.key == 'response_002_ch2.html')
    assert row.status == 'failed' and 'plan was not JSON' in row.error


def test_a_text_changing_plan_is_rejected_and_the_file_untouched(book, monkeypatch):
    real = core.apply_plan

    def tampering(html, *args, **kwargs):
        return real(html, *args, **kwargs).replace('rain kept', 'rain stopped')

    monkeypatch.setattr(core, 'apply_plan', tampering)
    result = runner.run_interactive_images(dict(CONFIG), str(book), ['response_002_ch2.html'],
                                           client_factory=_factory({'Ch 2': PLAN2}, []))
    assert result.rejected == 1
    assert (book / 'response_002_ch2.html').read_text(encoding='utf-8') == CH2
    assert core.chapter_images_on_disk(str(book), 2) == []
    row = next(r for r in ipc.image_rows(ipc.open_progress(str(book))) if r.key == 'response_002_ch2.html')
    assert row.status == 'rejected' and 'text' in row.error


def test_an_image_failure_gives_partial(book):
    plan = ('{"images":[{"after_paragraph":3,"quote":"Mira walked","prompt":"one"},'
            '{"after_paragraph":4,"quote":"\\"Hello","prompt":"two"}]}')
    runner.run_interactive_images(dict(CONFIG, interactive_images_per_chapter='2'), str(book),
                                  ['response_001_ch1.html'],
                                  client_factory=_factory({'Ch 1': plan}, [], fail_images={2}))
    html = (book / 'response_001_ch1.html').read_text(encoding='utf-8')
    assert html.count('data-glx="img"') == 1 and 'glx-pending' not in html
    entry = ipc.read_progress_file(core.progress_path(str(book)))['chapters']['response_001_ch1.html']
    assert entry['status'] == 'partial'
    assert [i['status'] for i in entry['images']] == ['completed', 'failed']


def test_stop_is_honoured_between_requests(book):
    stop = threading.Event()
    result = runner.run_interactive_images(
        dict(CONFIG), str(book), client_factory=_factory({'Ch 1': PLAN1, 'Ch 2': PLAN2}, [],
                                                         gate=lambda count: stop.set()),
        stop_event=stop)
    assert result.stopped
    assert (book / 'response_002_ch2.html').read_text(encoding='utf-8') == CH2


def test_css_only_profile_needs_no_image_model(book):
    sent = []
    config = {'model': 'gpt-test', 'interactive_images_per_chapter': '0', 'interactive_css_enabled': True}
    runner.run_interactive_images(config, str(book), ['response_001_ch1.html'],
                                  client_factory=_factory({'Ch 1': PLAN1}, sent))
    assert not [s for s in sent if s[0] == 'image']
    entry = ipc.read_progress_file(core.progress_path(str(book)))['chapters']['response_001_ch1.html']
    assert entry['status'] == 'css_only'
    assert 'data-glx="img"' not in (book / 'response_001_ch1.html').read_text(encoding='utf-8')


def test_delete_and_mark_actions(book):
    runner.run_interactive_images(dict(CONFIG), str(book), ['response_001_ch1.html'],
                                  client_factory=_factory({'Ch 1': PLAN1}, []))
    model = ipc.open_progress(str(book))
    row = next(r for r in ipc.image_rows(model) if r.key == 'response_001_ch1.html')
    targets = ipc.open_targets(str(book), row)
    assert [os.path.basename(p) for p in targets['images']] == ['Glossarion_ch001_1.png']
    assert 'Image 1 after paragraph 2' in ipc.plan_text(row)
    assert ipc.mark_for_rerun(core.progress_path(str(book)), [row]) == 1
    assert ipc.read_progress_file(core.progress_path(str(book)))['chapters'][row.key]['status'] == 'pending'
    assert ipc.delete_entries(str(book), [row]) == 1
    assert (book / 'response_001_ch1.html').read_text(encoding='utf-8') == CH1
    assert core.chapter_images_on_disk(str(book), 1) == []
    assert row.key not in ipc.read_progress_file(core.progress_path(str(book)))['chapters']


def test_missing_image_model_stops_before_any_request(book):
    sent = []
    result = runner.run_interactive_images({'model': 'gpt-test'}, str(book), client_factory=_factory({}, sent))
    assert result.errors == ['no image model'] and not sent


def test_generated_stylesheet_is_scoped(book):
    reply = 'Here:\n```css\nbody { color: red; } .glx-dialogue { color: blue; }\n```'

    class Client:
        def send(self, messages, **kwargs):
            Client.prompt = messages[-1]['content']
            return reply, 'stop'

    css, warnings = runner.generate_stylesheet({'model': 'm'}, 'pirate theme',
                                               client_factory=lambda *a: Client())
    assert '.glx-book {' in css and '.glx-dialogue {' in css and 'body' not in css
    assert warnings
    assert Client.prompt.startswith("Write a CSS stylesheet for an EPUB novel: pirate theme\n")


def test_shared_row_action_rules_and_owner_recompiler(tmp_path):
    from types import SimpleNamespace

    from interactive_images_runner import owner_recompiler

    def row(key, status, stored='', images=0):
        return ipc.ImagesRow('chapter', key, key, status, '', '', images=images, stored_status=stored)

    rows = [row('a', 'completed', 'completed', 1), row('b', 'failed', 'failed'), row('c', 'not_built')]
    assert [r.key for r in ipc.action_targets('retry', rows)] == ['b']
    assert [r.key for r in ipc.action_targets('delete', rows)] == ['a', 'b']
    assert ipc.action_label('build', rows) == "\U0001f5bc Build / Rebuild (3)"
    assert ipc.action_label('retry', rows) == "\U0001f501 Retry failed (1)"
    assert ipc.action_reason('build', rows) is None
    assert ipc.action_reason('build', rows, running=True) == "Interactive Images is running"
    assert ipc.action_reason('retry', rows[2:]) and ipc.action_reason('delete', rows[2:])
    assert ipc.action_reason('open_images', rows[1:]) == "No images yet"
    assert ipc.action_reason('open_images', rows) is None
    assert [a for a, _l in ipc.ROW_ACTIONS] == ['build', 'retry', 'mark', 'delete', 'open_images', 'open_chapter',
                                                'show_plan', 'open_folder']
    assert "3 chapter(s)" in ipc.delete_confirmation(3)
    assert [g for g, _t, _c in ipc.STAT_CHIPS] == list(ipc.STATUS_GROUPS)

    calls = []
    owner = SimpleNamespace(_run_epub_compile=lambda f: calls.append(('epub', f)),
                            _run_pdf_compile=lambda f: calls.append(('pdf', f)))
    recompile = owner_recompiler(owner)
    (tmp_path / 'book.pdf').write_bytes(b'%PDF')
    recompile(str(tmp_path))
    (tmp_path / 'book.epub').write_bytes(b'PK')
    recompile(str(tmp_path))
    assert calls == [('pdf', str(tmp_path)), ('epub', str(tmp_path))]
    assert owner_recompiler(SimpleNamespace()) is None