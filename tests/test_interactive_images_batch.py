"""Interactive Images in batch mode: the translation's Batch Translation / batch size / threading delay run chapters and a
chapter's image requests in parallel (``run_interactive_images(..., batch=BatchSettings(...))``); batch off stays
sequential. A fake client records the start and end of every request, so overlap is measured, not assumed.

Python 3.10 compatible. Run:
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests/test_interactive_images_batch.py
"""
import base64
import json
import os
import re
import sys
import threading
import time
import types
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

CHAPTER = ('<html><head><title>Tale {n}</title></head><body>\n<h1>Tale {n}</h1>\n'
           '<p>Tale number {n} begins at the lantern gate.</p>\n'
           '<p>Mira walked to the gate in the rain.</p>\n'
           '<p>"Hello," said Tomas to the empty road.</p>\n</body></html>')
PLAN = ('{"images":[{"after_paragraph":2,"quote":"Mira walked","characters":[],"prompt":"A gate in the rain"},'
        '{"after_paragraph":3,"quote":"Hello,","characters":[],"prompt":"A road at night"}],"classes":[]}')
HOLD = 0.15   # how long a fake request takes


def _png():
    from PIL import Image
    buffer = BytesIO()
    Image.new('RGB', (8, 8), (10, 20, 30)).save(buffer, format='PNG')
    return buffer.getvalue()


class Fleet:
    """The fake clients of one run and everything they saw (thread-safe)."""

    def __init__(self, fail_chapter=None, on_request=None, hold=HOLD):
        self.lock = threading.Lock()
        self.requests = []      # {'kind', 'chapter', 'start', 'end', 'thread', 'env_batch'}
        self.made = {'text': 0, 'image': 0}
        self.fail_chapter = fail_chapter
        self.on_request = on_request
        self.hold = hold

    def factory(self, kind, model, config, output_dir):
        with self.lock:
            self.made[kind] += 1
        return _Client(self, kind)

    def overlap(self, kinds=('text', 'image')):
        events = []
        for r in self.requests:
            if r['kind'] in kinds:
                events += [(r['start'], 1), (r['end'], -1)]
        peak = now = 0
        for _t, delta in sorted(events, key=lambda e: (e[0], e[1])):
            now += delta
            peak = max(peak, now)
        return peak


class _Client:
    def __init__(self, fleet, kind):
        self.fleet = fleet
        self.kind = kind
        self.in_use = 0

    def send(self, messages, temperature=None, max_tokens=None, context=None):
        fleet = self.fleet
        self.in_use += 1
        assert self.in_use == 1, "one client was used by two requests at once"
        text = messages[-1]['content']
        chapter = None
        if self.kind == 'text':
            found = re.search(r'Tale number (\d+)', text)
            chapter = int(found.group(1)) if found else None
        record = {'kind': self.kind, 'chapter': chapter, 'start': time.monotonic(),
                  'thread': threading.current_thread().name, 'env_batch': os.environ.get('BATCH_TRANSLATION')}
        with fleet.lock:
            fleet.requests.append(record)
            count = len(fleet.requests)
        try:
            if fleet.on_request:
                fleet.on_request(count)
            time.sleep(fleet.hold)
            if self.kind == 'text' and fleet.fail_chapter is not None and chapter == fleet.fail_chapter:
                raise RuntimeError("the planner is down for this chapter")
            if self.kind == 'image':
                return "data:image/png;base64," + base64.b64encode(_png()).decode('ascii'), 'stop'
            return PLAN, 'stop'
        finally:
            record['end'] = time.monotonic()
            self.in_use -= 1


def _book(tmp_path, chapters=6):
    out = tmp_path / 'book'
    out.mkdir()
    entries = {}
    for n in range(1, chapters + 1):
        name = f'response_{n:03d}_ch{n}.html'
        (out / name).write_text(CHAPTER.format(n=n), encoding='utf-8')
        entries[str(n)] = {'actual_num': n, 'output_file': name, 'status': 'completed'}
    (out / 'translation_progress.json').write_text(json.dumps({'chapters': entries}), encoding='utf-8')
    return out


CONFIG = {'model': 'fake-text', 'interactive_images_image_model': 'fake-image', 'interactive_images_per_chapter': '2'}


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    monkeypatch.setattr(runner, 'configure_image_key_pool', lambda config, log: None)
    monkeypatch.setattr(runner, '_token_counter', lambda model: (lambda text: max(1, len(str(text)) // 4)))
    for name in ('BATCH_TRANSLATION', 'BATCH_SIZE', 'SEND_INTERVAL_SECONDS', 'SEND_INTERVAL_MAX_SECONDS',
                 'THREAD_SUBMISSION_DELAY_SECONDS'):
        monkeypatch.delenv(name, raising=False)


def _run(out, fleet, batch=None, **extra):
    logs = []
    result = runner.run_interactive_images(dict(CONFIG), str(out), log=logs.append, client_factory=fleet.factory,
                                           batch=batch, **extra)
    return result, logs


def _batch(size=3, delay='0'):
    return runner.BatchSettings(enabled=True, size=size, send_interval='0', send_interval_max='', thread_delay=delay)


def _assert_book_consistent(out, chapters, images_each=2):
    chapter_entries = ipc.read_progress_file(core.progress_path(str(out)))['chapters']
    names = set()
    for n in range(1, chapters + 1):
        html = (out / f'response_{n:03d}_ch{n}.html').read_text(encoding='utf-8')
        assert core.check_text_unchanged(CHAPTER.format(n=n), html)[0], f"chapter {n}: the text changed"
        entry = chapter_entries[f'response_{n:03d}_ch{n}.html']
        assert entry['status'] == 'completed', (n, entry)
        files = [img['file'] for img in entry['images'] if img['status'] == 'completed']
        assert len(files) == images_each and len(set(files)) == images_each
        for name in files:
            assert f'src="Interactive_Images/{name}"' in html
            assert (out / 'Interactive_Images' / name).is_file()
        names.update(files)
    on_disk = {p.name for p in (out / 'Interactive_Images').glob('Glossarion_*')}
    assert on_disk == names and len(names) == chapters * images_each     # no duplicate, no stray file
    assert not list((out / 'Interactive_Images').glob('*.tmp'))


def test_batch_size_3_over_6_chapters_overlaps_and_builds_every_chapter_once(tmp_path):
    out = _book(tmp_path)
    fleet = Fleet()
    result, logs = _run(out, fleet, _batch(3))
    assert (result.processed, result.completed, result.failed, result.stopped) == (6, 6, 0, False), result.errors
    assert sorted(result.changed_files) == sorted(f'response_{n:03d}_ch{n}.html' for n in range(1, 7))
    assert result.images == 12
    assert fleet.overlap() == 3                                        # parallel, and never above the batch size
    planners = [r for r in fleet.requests if r['kind'] == 'text']
    assert sorted(r['chapter'] for r in planners) == [1, 2, 3, 4, 5, 6]  # each chapter planned exactly once
    assert len([r for r in fleet.requests if r['kind'] == 'image']) == 12
    assert fleet.made['text'] <= 3 and fleet.made['image'] <= 3        # clients are reused, at most one per slot
    assert all(r['env_batch'] == '1' for r in fleet.requests)          # UnifiedClient sees the translation's batch env
    assert len({r['thread'] for r in fleet.requests}) > 1
    _assert_book_consistent(out, 6)
    assert os.environ.get('BATCH_TRANSLATION') is None                 # put back
    worker_lines = [line for line in logs if re.match(r'\[ii-worker_\d+\] ', line)]
    assert any('image 1 saved as' in line for line in worker_lines)    # per-chapter lines carry the worker tag
    assert any('✅' in line for line in worker_lines)


def test_a_chapters_images_run_at_the_same_time(tmp_path):
    out = _book(tmp_path, chapters=1)
    fleet = Fleet()
    result, _logs = _run(out, fleet, _batch(3))
    assert result.completed == 1
    images = [r for r in fleet.requests if r['kind'] == 'image']
    assert len(images) == 2
    assert fleet.overlap(('image',)) == 2
    planner = [r for r in fleet.requests if r['kind'] == 'text'][0]
    assert planner['end'] <= min(r['start'] for r in images) + 0.001   # the planner first, then its images
    _assert_book_consistent(out, 1)


def test_batch_off_or_size_one_stays_sequential(tmp_path):
    for n, batch in enumerate((None, runner.BatchSettings(enabled=False, size=5),
                               runner.BatchSettings(enabled=True, size=1))):
        (tmp_path / f'case{n}').mkdir()
        out = _book(tmp_path / f'case{n}', chapters=3)
        fleet = Fleet(hold=0.02)
        result, logs = _run(out, fleet, batch)
        assert result.completed == 3
        assert fleet.overlap() == 1
        assert [r['kind'] for r in fleet.requests] == ['text', 'image', 'image'] * 3     # chapter by chapter
        assert fleet.made == {'text': 1, 'image': 1}                                      # one client of each kind
        assert all(r['env_batch'] is None for r in fleet.requests)                        # the env is not touched
        assert not any(re.match(r'\[ii-', line) for line in logs)
        _assert_book_consistent(out, 3)


def test_the_batch_size_is_capped(tmp_path):
    assert runner.BatchSettings(enabled=True, size=500).workers == runner.MAX_BATCH_WORKERS
    assert runner.BatchSettings(enabled=True, size=0).workers == 1
    assert runner.BatchSettings(enabled=False, size=9).workers == 1
    assert runner.BatchSettings(enabled=True, size='x').workers == 1


def test_stop_mid_run_cancels_pending_chapters_and_leaves_no_half_built_one(tmp_path):
    out = _book(tmp_path, chapters=8)
    stop = threading.Event()
    marks = {}

    def on_request(count):
        if count == 5:
            marks['at'] = count
            stop.set()

    fleet = Fleet(on_request=on_request)
    result, _logs = _run(out, fleet, _batch(3), stop_event=stop)
    assert result.stopped is True
    assert len(fleet.requests) <= 5 + 3                                # only requests already past the gate finished
    entries = ipc.read_progress_file(core.progress_path(str(out)))['chapters']
    assert not any(e.get('status') in ('in_progress', 'failed') for e in entries.values())
    assert result.processed < 8                                        # the queued chapters never started
    for n in range(1, 9):
        html = (out / f'response_{n:03d}_ch{n}.html').read_text(encoding='utf-8')
        entry = entries.get(f'response_{n:03d}_ch{n}.html', {})
        if entry.get('status') == 'completed':
            assert html.count('Interactive_Images/Glossarion_') == 2     # a finished chapter is whole
        else:
            assert html == CHAPTER.format(n=n)                           # anything else is untouched
    assert not list((out / 'Interactive_Images').glob('*.tmp'))


def test_one_failing_chapter_never_touches_the_others(tmp_path):
    out = _book(tmp_path)
    fleet = Fleet(fail_chapter=3)
    result, logs = _run(out, fleet, _batch(3))
    assert (result.completed, result.failed) == (5, 1)
    assert [e.split(':')[0] for e in result.errors] == ['response_003_ch3.html']
    entries = ipc.read_progress_file(core.progress_path(str(out)))['chapters']
    assert entries['response_003_ch3.html']['status'] == 'failed'
    assert (out / 'response_003_ch3.html').read_text(encoding='utf-8') == CHAPTER.format(n=3)
    for n in (1, 2, 4, 5, 6):
        assert entries[f'response_{n:03d}_ch{n}.html']['status'] == 'completed'
    assert any('response_003_ch3.html' in line and '❌' in line for line in logs)


def test_one_failing_image_keeps_the_chapter_partial(tmp_path):
    out = _book(tmp_path, chapters=2)
    fleet = Fleet()
    original = _Client.send
    seen = {'n': 0}

    def flaky(self, messages, temperature=None, max_tokens=None, context=None):
        if self.kind == 'image':
            with fleet.lock:
                seen['n'] += 1
                bad = seen['n'] == 2
            if bad:
                raise RuntimeError("the image model refused")
        return original(self, messages, temperature, max_tokens, context)

    _Client.send = flaky
    try:
        result, _logs = _run(out, fleet, _batch(3))
    finally:
        _Client.send = original
    assert result.completed == 2 and result.failed == 0
    entries = ipc.read_progress_file(core.progress_path(str(out)))['chapters']
    assert sorted(e['status'] for e in entries.values()) == ['completed', 'partial']


def test_request_starts_are_spaced_by_the_threading_delay(tmp_path):
    out = _book(tmp_path, chapters=3)
    fleet = Fleet(hold=0.05)
    result, _logs = _run(out, fleet, _batch(3, delay='0.25'))
    assert result.completed == 3
    starts = sorted(r['start'] for r in fleet.requests)
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    assert gaps and min(gaps) >= 0.2, gaps                            # never closer than the delay (timer slack)


def test_the_translation_env_is_set_during_a_parallel_run_and_restored(tmp_path, monkeypatch):
    monkeypatch.setenv('SEND_INTERVAL_SECONDS', '7')
    monkeypatch.setenv('THREAD_SUBMISSION_DELAY_SECONDS', '9')
    out = _book(tmp_path, chapters=2)
    seen = {}

    def on_request(_count):
        seen.update({k: os.environ.get(k) for k in ('BATCH_TRANSLATION', 'BATCH_SIZE', 'SEND_INTERVAL_SECONDS',
                                                     'SEND_INTERVAL_MAX_SECONDS', 'THREAD_SUBMISSION_DELAY_SECONDS')})

    batch = runner.BatchSettings(enabled=True, size=2, send_interval='0', send_interval_max='0.01', thread_delay='0')
    result, _logs = _run(out, Fleet(on_request=on_request, hold=0.01), batch)
    assert result.completed == 2
    assert seen == {'BATCH_TRANSLATION': '1', 'BATCH_SIZE': '2', 'SEND_INTERVAL_SECONDS': '0',
                    'SEND_INTERVAL_MAX_SECONDS': '0.01', 'THREAD_SUBMISSION_DELAY_SECONDS': '0'}
    assert os.environ['SEND_INTERVAL_SECONDS'] == '7' and os.environ['THREAD_SUBMISSION_DELAY_SECONDS'] == '9'
    assert 'BATCH_TRANSLATION' not in os.environ and 'SEND_INTERVAL_MAX_SECONDS' not in os.environ


def test_progress_callbacks_never_overlap_and_cover_every_chapter(tmp_path):
    out = _book(tmp_path)
    active = {'now': 0, 'peak': 0}
    seen = []

    def on_progress(key, status):
        active['now'] += 1
        active['peak'] = max(active['peak'], active['now'])
        time.sleep(0.005)
        seen.append((key, status))
        active['now'] -= 1

    result, _logs = _run(out, Fleet(hold=0.03), _batch(3), on_progress=on_progress)
    assert result.completed == 6 and active['peak'] == 1
    assert sorted(k for k, s in seen if s == 'completed') == sorted(f'response_{n:03d}_ch{n}.html' for n in range(1, 7))


def test_batch_settings_follow_the_owners_translation_fields():
    owner = types.SimpleNamespace(
        config={'model': 'gpt-x'}, batch_translation_var=True, batch_size_var='4',
        delay_entry=types.SimpleNamespace(text=lambda: '3-7'),
        thread_delay_entry=types.SimpleNamespace(text=lambda: '0.5'))
    batch = runner.batch_settings_from_owner(owner)
    assert (batch.enabled, batch.size, batch.workers) == (True, 4, 4)
    assert (batch.send_interval, batch.send_interval_max, batch.thread_delay) == ('3.0', '7.0', '0.5')
    off = runner.batch_settings_from_owner(types.SimpleNamespace(config={}, batch_translation_var=False,
                                                                 batch_size_var=5))
    assert off.enabled is False and off.workers == 1
    saved = runner.batch_settings_from_owner(types.SimpleNamespace(
        config={'batch_translation': True, 'batch_size': 3, 'delay': 2.5, 'thread_submission_delay': '0.2'}))
    assert (saved.enabled, saved.size, saved.thread_delay) == (True, 3, '0.2')
    assert saved.send_interval == '2.5'
    bare = runner.batch_settings_from_owner(types.SimpleNamespace(config={}))
    assert bare.enabled is False                                       # nothing to follow: sequential
    web = runner.batch_settings_from_owner(types.SimpleNamespace(
        config={'model': 'authchatgpt/gpt-6.2-sol'}, batch_translation_var=True, batch_size_var=''))
    assert web.size == 1 and web.workers == 1                          # web-chat routes default to one at a time


def test_the_desktop_start_run_passes_the_translation_batch_fields_read_on_the_gui_thread(tmp_path, monkeypatch):
    pytest.importorskip('PySide6')
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    import interactive_images_gui as gui
    seen = {}
    done = threading.Event()

    def fake_run(config, output_dir, keys=None, **kwargs):
        seen.update(kwargs)
        seen['thread'] = threading.current_thread().name
        done.set()
        return runner.RunResult()

    monkeypatch.setattr(runner, 'run_interactive_images', fake_run)
    owner = types.SimpleNamespace(
        config={}, append_log=lambda message: None, batch_translation_var=True, batch_size_var='4',
        delay_entry=types.SimpleNamespace(text=lambda: '2-5'),
        thread_delay_entry=types.SimpleNamespace(text=lambda: '0.3'))
    assert gui.start_run(owner, str(tmp_path)) is True
    assert done.wait(10)
    owner._interactive_images_run['thread'].join(10)
    batch = seen['batch']
    assert (batch.enabled, batch.workers, batch.send_interval, batch.send_interval_max, batch.thread_delay) == (
        True, 4, '2.0', '5.0', '0.3')
    assert seen['thread'] == 'interactive-images'
    bridge = owner._interactive_images_bridge
    bridge.deleteLater()


@pytest.mark.parametrize('batch', [None, _batch(3)])
def test_a_client_that_cannot_be_made_fails_the_chapter_before_any_request(tmp_path, batch):
    out = _book(tmp_path, chapters=2)
    fleet = Fleet()

    def factory(kind, model, config, output_dir):
        if kind == 'image':
            raise RuntimeError("no image key")
        return fleet.factory(kind, model, config, output_dir)

    logs = []
    result = runner.run_interactive_images(dict(CONFIG), str(out), log=logs.append, client_factory=factory, batch=batch)
    assert (result.completed, result.failed) == (0, 2)
    assert fleet.requests == []          # no planner request was spent on a chapter whose images cannot be made
