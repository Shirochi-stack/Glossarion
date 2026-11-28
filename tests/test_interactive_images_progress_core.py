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
    # the list: translated chapters whose translation completed; the in-progress one is not listed
    assert [r.key for r in rows] == ['response_001_ch1.html', 'response_002_ch2.html']
    assert [r.status for r in rows] == ['not_built'] * 2
    stats = ipc.image_stats(model, rows)
    assert stats['not_built'] == 2 and stats['total'] == 2
    assert ipc.tab_count_text(stats) == '0/2'
    assert ipc.pending_keys(model) == ['response_001_ch1.html', 'response_002_ch2.html']
    assert ipc.find_progress(str(book)) is None


def test_list_is_every_completed_translation_with_not_built_rows_and_a_built_count(book):
    """Every chapter whose translation completed is listed (a Not built row without an images entry); image-only,
    empty, skipped, failed and in-progress translations are not. The tab count is built/total over that list:
    stale, partial and failed chapters are not built."""
    chapters = {
        'a': ('response_001_ch1.html', 'completed'), 'b': ('response_002_ch2.html', 'completed'),
        'c': ('response_003_ch3.html', 'in_progress'), 'e': ('response_005_img.html', 'completed_image_only'),
        'f': ('response_006_empty.html', 'completed_empty'), 'g': ('response_007_skip.html', 'skipped'),
        'h': ('response_008_bad.html', 'failed'), 'i': ('response_009_ch9.html', 'completed'),
        'j': ('response_010_ch10.html', 'completed'),
    }
    for name, _status in chapters.values():
        if not (book / name).exists():
            (book / name).write_text(CH2, encoding='utf-8')
    (book / 'translation_progress.json').write_text(json.dumps({'chapters': {
        key: {'actual_num': index + 1, 'output_file': name, 'status': status}
        for index, (key, (name, status)) in enumerate(chapters.items())}}), encoding='utf-8')
    path = core.progress_path(str(book))
    ipc.set_entry(path, 'response_001_ch1.html', {
        'status': 'completed', 'source_hash': core.visible_text_hash(CH1), 'images': [], 'classes': [{'class': 'x'}]})
    ipc.set_entry(path, 'response_002_ch2.html', {'status': 'failed', 'error': 'boom'})
    ipc.set_entry(path, 'response_009_ch9.html', {'status': 'completed', 'source_hash': 'old text', 'images': []})
    model = ipc.open_progress(str(book))
    rows = {r.key: r for r in ipc.image_rows(model)}
    assert list(rows) == ['response_001_ch1.html', 'response_002_ch2.html', 'response_009_ch9.html',
                          'response_010_ch10.html']
    assert rows['response_010_ch10.html'].status == 'not_built' and not rows['response_010_ch10.html'].stored_status
    assert rows['response_009_ch9.html'].status == 'stale'
    assert rows['response_002_ch2.html'].status == 'failed'
    stats = ipc.image_stats(model)
    # chapter 1 has classes recorded but its file carries no Glossarion markup: stale as well
    assert (stats['total'], stats['stale'], stats['failed'], stats['not_built']) == (4, 2, 1, 1)
    # stale and failed are not built
    assert stats['built'] == 0
    assert ipc.tab_count_text(stats) == "0/4, 2 stale"
    # Not built rows can be built; there is nothing to delete or retry on them
    row = rows['response_010_ch10.html']
    assert ipc.action_reason('build', [row]) is None
    assert ipc.action_reason('delete', [row]) is not None and ipc.action_reason('retry', [row]) is not None


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
    assert (result.completed, result.failed, result.rejected, result.skipped) == (2, 0, 0, 0)
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
    assert 'response_003_ch3.html' not in rows  # its translation has not completed: not listed
    entry = model.data['chapters']['response_001_ch1.html']
    assert entry['images'][0]['refs'] == ['references/mira_1.png', 'references/style_1.png']
    assert entry['source_hash'] == core.visible_text_hash(CH1)
    assert ipc.tab_count_text(ipc.image_stats(model)) == '2/2'


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


def test_a_retranslation_with_the_same_text_shows_stale(book):
    runner.run_interactive_images(dict(CONFIG), str(book), ['response_001_ch1.html'],
                                  client_factory=_factory({'Ch 1': PLAN1}, []))
    (book / 'response_001_ch1.html').write_text(CH1, encoding='utf-8')  # same text, the insertions are gone
    row = next(r for r in ipc.image_rows(ipc.open_progress(str(book))) if r.key == 'response_001_ch1.html')
    assert row.status == 'stale' and row.stale


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
    """No image model, no pool, and a text model known to answer in text only (DeepSeek): refused before any request."""
    sent = []
    result = runner.run_interactive_images({'model': 'deepseek-chat'}, str(book), client_factory=_factory({}, sent))
    assert result.errors == ['no image model'] and not sent
    assert 'no image-capable model' in result.no_image_model and 'deepseek-chat' in result.no_image_model
    assert 'images per chapter to 0' in result.no_image_model


def test_image_model_falls_back_from_the_setting_to_the_pool_to_the_text_model():
    def choose(config, text='authgpt/gpt-6.2-sol'):
        return runner.resolve_image_model(config, core.settings_from_config(config), text)

    base = {'model': 'authgpt/gpt-6.2-sol'}
    pool = dict(base, use_inpainter_keys=True,
                inpainter_keys=[{'model': 'gemini-3-pro-image-preview', 'enabled': True}])
    own = dict(pool, interactive_images_image_model='dall-e-3')
    assert (choose(own).model, choose(own).source) == ('dall-e-3', 'the Interactive Images image model')
    assert (choose(pool).model, choose(pool).source) == ('gemini-3-pro-image-preview', 'the Image Gen / Edit pool')
    fallback = choose(base)
    assert (fallback.model, fallback.source, fallback.capability) == ('authgpt/gpt-6.2-sol', 'the text model', 'image')
    assert not fallback.probe                        # a route that returns images: no probe
    assert choose(base, 'gpt-image-1').capability == 'image'
    assert choose(base, 'custom/odd-model').probe     # unknown: the first image request is the test
    assert choose(base, 'deepseek-chat').model == '' and choose(base, 'claude-sonnet-4').capability == 'text_only'
    assert core.image_capability('qwen-image') == 'image'


def test_unknown_text_model_is_probed_once_and_stops_the_run_when_no_image_comes_back(book):
    _refs(book)
    sent = []
    result = runner.run_interactive_images(
        {'model': 'custom/odd-model'}, str(book), client_factory=_factory({'Ch 1': PLAN1, 'Ch 2': PLAN2}, sent,
                                                                            fail_images=(1,)))
    kinds = [kind for kind, _c, _m in sent]
    assert kinds.count('image') == 1 and kinds.count('text') == 1     # one probe, no planner for chapter 2
    assert result.stopped and result.errors == ['no image model']
    assert 'returned no image' in result.no_image_model and 'images per chapter to 0' in result.no_image_model
    assert all(c == runner.IMAGE_CONTEXT for kind, c, _m in sent if kind == 'image')
    model = ipc.open_progress(str(book))
    assert not any(r.status == 'completed' for r in ipc.image_rows(model))


def test_unknown_text_model_that_returns_a_picture_keeps_drawing(book):
    _refs(book)
    sent, logs = [], []
    result = runner.run_interactive_images(
        {'model': 'custom/odd-model'}, str(book), log=logs.append,
        client_factory=_factory({'Ch 1': PLAN1, 'Ch 2': PLAN2}, sent))
    assert (result.completed, result.failed) == (2, 0) and not result.no_image_model
    assert [kind for kind, _c, _m in sent].count('image') == 2
    assert any('custom/odd-model (the text model)' in line for line in logs)


def test_known_image_route_as_text_model_needs_no_probe(book):
    _refs(book)
    sent, logs = [], []
    result = runner.run_interactive_images(
        {'model': 'authgpt/gpt-6.2-sol'}, str(book), log=logs.append,
        client_factory=_factory({'Ch 1': PLAN1, 'Ch 2': PLAN2}, sent))
    assert result.completed == 2
    assert not any('first image request is the test' in line for line in logs)


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


class _SavingClient(FakeClient):
    """Like an authgpt/ image reply: the client saves its own copy in the output folder and returns the
    [GENERATED_IMAGE:path] marker (unified_api_client._save_route_image)."""

    def __init__(self, kind, plans, sent, out):
        super().__init__(kind, plans, sent)
        self.out = out

    def send(self, messages, temperature=None, max_tokens=None, context=None):
        if self.kind == 'text':
            return super().send(messages, temperature, max_tokens, context)
        self.count += 1
        path = os.path.join(self.out, f'image_generation_response_{self.count}_abc_A0.png')
        data = _png((10, 10, 200))
        with open(path, 'wb') as handle:
            handle.write(data)
        self.tls._last_generated_image_bytes = data
        self.tls._last_generated_image_path = path
        return f'[GENERATED_IMAGE:{path}]', 'stop'


def test_the_clients_own_saved_copy_of_an_image_is_removed(book):
    """Found in the live authgpt/ run: the reply's image_generation_response_*.png copy stayed in the book's
    output folder next to Interactive_Images/Glossarion_ch001_1.png (two copies of every image)."""
    clients = {}

    def make(kind, model, config, output_dir):
        return clients.setdefault(kind, _SavingClient(kind, {'Ch 1': PLAN1}, [], output_dir))

    result = runner.run_interactive_images(dict(CONFIG), str(book), ['response_001_ch1.html'], client_factory=make)
    assert result.completed == 1
    assert (book / core.INTERACTIVE_FOLDER / 'Glossarion_ch001_1.png').read_bytes() == _png((10, 10, 200))
    assert not list(book.glob('image_generation_response_*'))


# ---- the glossary: the one translation uses -----------------------------------------------------------

def _glossary_setup(tmp_path):
    out_root = tmp_path / 'outroot'
    output_dir = out_root / 'Lantern'
    output_dir.mkdir(parents=True)
    source = tmp_path / 'Lantern.epub'
    source.write_bytes(b'x')
    return out_root, output_dir, str(source)


def _csv(path, *names):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('type,raw_name,translated_name,gender\n' + ''.join(f'character,r{n},{n},\n' for n in names),
                    encoding='utf-8')
    return str(path)


def test_the_glossary_is_the_one_translation_resolves_for_each_mode(tmp_path, monkeypatch):
    from run_env import GlossaryResolver

    monkeypatch.delenv('OUTPUT_DIRECTORY', raising=False)
    out_root, output_dir, source = _glossary_setup(tmp_path)
    config = {'output_directory': str(out_root)}

    def resolve(mode='Off', manual=None, mapping=None, **extra):
        owner = GlossaryResolver(dict(config, auto_glossary_mode=mode, **extra), manual, mapping)
        return owner._resolve_book_glossary(str(output_dir), source), core.find_book_glossary(str(output_dir), owner)

    assert resolve() == (None, None)                                       # none
    in_output = _csv(output_dir / 'glossary.csv', 'Mira')
    assert resolve()[0] == in_output and resolve()[1] == in_output           # the output folder's glossary
    auto = _csv(out_root / 'Glossary' / 'Lantern' / 'Lantern_glossary.csv', 'Mira', 'Tomas')
    assert resolve('Balanced')[0] == auto and resolve('Full')[0] == auto     # the auto-glossary output is preferred
    assert resolve('Off')[0] == auto            # an output override prefers the shared folder in any mode, as translation does
    manual = _csv(tmp_path / 'mine' / 'my_glossary.csv', 'Zed')
    assert resolve('Balanced', manual)[0] == manual                         # Load Glossary wins in any mode
    assert resolve('Off', manual)[0] == manual
    other = _csv(tmp_path / 'mine' / 'other.csv', 'Yan')
    assert resolve('Off', None, {source: other})[0] == other                 # the per-book map (several books)
    assert resolve('No Glossary', manual) == (None, None)                    # no glossary: none, whatever was loaded
    # the characters the step uses come from that file
    names = core.character_names(core.load_glossary_entries(str(output_dir), owner=GlossaryResolver(
        dict(config, auto_glossary_mode='Balanced'))))
    assert names == ['Mira', 'Tomas']


def test_reference_rows_list_the_glossary_and_keep_names_with_images():
    refs = {'characters': {'Mira': ['references/mira_1.png'], 'Gone': ['references/gone_1.png'], 'Extra': []}}
    rows = {name: (paths, in_glossary) for name, paths, in_glossary in
            core.reference_rows_with_glossary(refs, ['Mira', 'Tomas'])}
    assert rows == {'Mira': (['references/mira_1.png'], True), 'Tomas': ([], True),
                    'Gone': (['references/gone_1.png'], False), 'Extra': ([], False)}
