"""Owner request (2026-11-01): the QA scanner stops at once when a stop flag is raised. A graceful stop too, unless
API calls are in flight (the AI truncation checker): those finish, and the forced stop (the second click) aborts them.

It used to run on: a stop shut the worker pool down waiting for every queued batch, the results loop only looked at the
flags when a batch finished, and reading the source EPUB, which takes the longest on a big book, had no check at all.
"""

import concurrent.futures
import os
import sys
import threading
import time
import zipfile
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.environ.setdefault("GLOSSARION_HTTP_LOG", "0")

import scan_html_folder as shf  # noqa: E402

CHAPTERS = 24
INSTANT = 1.0  # seconds: far more than the stop takes, little enough to fail on any wait for the work


@pytest.fixture(autouse=True)
def clean_stop_flags():
    def clean():
        for name in ("GRACEFUL_STOP", "TRANSLATION_CANCELLED", "GRACEFUL_STOP_COMPLETED"):
            os.environ.pop(name, None)
        shf._stop_flag = False
    clean()
    yield
    clean()


def _book(tmp_path, count=CHAPTERS):
    """A translated output folder and its source EPUB."""
    folder = tmp_path / "book"
    folder.mkdir()
    items = "".join(f'<item id="c{i}" href="chapter_{i:03d}.xhtml" media-type="application/xhtml+xml"/>'
                    for i in range(1, count + 1))
    spine = "".join(f'<itemref idref="c{i}"/>' for i in range(1, count + 1))
    epub = tmp_path / "book.epub"
    with zipfile.ZipFile(epub, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr("META-INF/container.xml", '<?xml version="1.0"?><container version="1.0" xmlns="urn:oasis:names:'
                         'tc:opendocument:xmlns:container"><rootfiles><rootfile full-path="OEBPS/content.opf" media-type='
                         '"application/oebps-package+xml"/></rootfiles></container>')
        archive.writestr("OEBPS/content.opf", '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version='
                         '"3.0" unique-identifier="id"><metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:identifier'
                         f' id="id">b</dc:identifier><dc:title>b</dc:title></metadata><manifest>{items}</manifest><spine>'
                         f'{spine}</spine></package>')
        for i in range(1, count + 1):
            archive.writestr(f"OEBPS/chapter_{i:03d}.xhtml", "<html><body><h1>Chapter</h1>" +
                             f"<p>The source text of chapter {i} goes on for a while here.</p>" * 20 + "</body></html>")
            (folder / f"response_chapter_{i:03d}.html").write_text(
                f"<html><head><title>Chapter {i}</title></head><body><h1>Chapter {i}</h1>" +
                f"<p>Translated text number {i} with enough words in it to be a chapter.</p>" * 20 + "</body></html>",
                encoding="utf-8")
    return folder, epub


class _Scan:
    """``scan_html_folder`` on its own thread, as the GUI and the job run it."""

    def __init__(self, folder, **kwargs):
        self.lines = []
        self.result = "not finished"
        self.ended_at = None
        kwargs.setdefault("qa_settings", {})

        def run():
            self.result = shf.scan_html_folder(str(folder), log=self.lines.append, **kwargs)
            self.ended_at = time.monotonic()

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def ended_within(self, seconds):
        self.thread.join(seconds)
        return not self.thread.is_alive()


def _thread_settings(**settings):
    return {"use_thread_executor": True, **settings}


# ---- the pool helpers ------------------------------------------------------------------------------------------


def test_leaving_a_stopped_process_pool_kills_its_workers():
    with pytest.raises(StopIteration):
        with shf._stoppable_pool(concurrent.futures.ProcessPoolExecutor, lambda: True, max_workers=2) as pool:
            for _ in range(6):
                pool.submit(time.sleep, 60)
            processes = list(pool._processes.values())
            assert processes
            raise StopIteration
    started = time.monotonic()
    for process in processes:
        process.join(5)
    assert not any(process.is_alive() for process in processes), "a worker survived"
    assert time.monotonic() - started < 5


def test_leaving_a_stopped_thread_pool_does_not_wait_and_drops_the_queue():
    release = threading.Event()
    calls = []

    def work(_):
        calls.append(1)
        release.wait(10)

    started = time.monotonic()
    with shf._stoppable_pool(concurrent.futures.ThreadPoolExecutor, lambda: True, max_workers=1) as pool:
        futures = [pool.submit(work, n) for n in range(5)]
    assert time.monotonic() - started < INSTANT
    time.sleep(0.05)
    assert len(calls) == 1 and all(future.cancelled() for future in futures[1:])
    release.set()


def test_a_pool_without_a_stop_waits_as_before():
    with shf._stoppable_pool(concurrent.futures.ThreadPoolExecutor, lambda: False, max_workers=2) as pool:
        futures = [pool.submit(time.sleep, 0.05) for _ in range(4)]
    assert all(future.done() and not future.cancelled() for future in futures)


def test_waiting_for_results_notices_a_stop_without_a_batch_finishing():
    flag = []
    release = threading.Event()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(release.wait, 10)
        threading.Timer(0.1, lambda: flag.append(1)).start()
        started = time.monotonic()
        assert list(shf._completed_until_stop([future], lambda: bool(flag))) == []
        assert time.monotonic() - started < INSTANT
        release.set()


# ---- a whole scan --------------------------------------------------------------------------------------------


def test_a_scan_with_no_stop_still_writes_its_report(tmp_path):
    folder, epub = _book(tmp_path, 6)
    scan = _Scan(folder, epub_path=str(epub), qa_settings=_thread_settings())
    assert scan.ended_within(120), "\n".join(scan.lines[-20:])
    assert list(folder.rglob("validation_results*"))


def test_a_graceful_stop_during_the_file_checks_ends_the_scan_at_once(tmp_path, monkeypatch):
    """Nothing is in flight here, so a graceful stop does not wait for the batches left."""
    folder, _epub = _book(tmp_path)
    real = shf.process_html_file_batch
    started = threading.Event()
    done = []

    def slow(args, should_stop=None):
        started.set()
        for item in args[0]:
            if should_stop is not None and should_stop():
                break
            time.sleep(0.5)
            done.append(item)
        return real((args[0][:0],) + tuple(args[1:]))

    monkeypatch.setattr(shf, "process_html_file_batch", slow)
    scan = _Scan(folder, qa_settings=_thread_settings(), mode="quick-scan")
    assert started.wait(30), "\n".join(scan.lines[-20:])
    from qa_scan_runtime import apply_qa_graceful_stop_flags
    apply_qa_graceful_stop_flags()
    assert scan.ended_within(INSTANT), "still running after the stop:\n" + "\n".join(scan.lines[-10:])
    assert scan.result is None and not list(folder.rglob("validation_results*"))
    assert any("interrupted" in line for line in scan.lines)
    # the workers it did not wait for stop at their next file: the book's other files are never read
    time.sleep(0.7)
    assert len(done) < CHAPTERS


def test_a_stop_while_reading_the_source_ends_the_scan_at_once(tmp_path, monkeypatch):
    """The first stage had no check: on a big book it ran for minutes after the click."""
    folder, epub = _book(tmp_path)
    workers = []
    monkeypatch.setattr(shf, "process_html_file_batch", lambda *a, **k: workers.append(a) or [])
    polls = []

    def stop_flag():
        polls.append(1)
        return len(polls) > 3  # in the first extractor's chapter loop

    scan = _Scan(folder, epub_path=str(epub), stop_flag=stop_flag, qa_settings=_thread_settings(
        check_word_count_ratio=True))
    assert scan.ended_within(30)
    assert scan.result is None and not workers and not list(folder.rglob("validation_results*"))
    assert getattr(shf._scan_thread, "should_stop", None) is None  # and outside a scan nothing is checked


def test_a_stop_during_duplicate_detection_writes_no_report(tmp_path, monkeypatch):
    folder, _epub = _book(tmp_path, 6)

    def detect(results, log, should_stop, config):
        shf.stop_scan()
        assert should_stop()
        return {}, {}, {}

    monkeypatch.setattr(shf, "detect_duplicates", detect)
    reports = []
    monkeypatch.setattr(shf, "generate_reports", lambda *a, **k: reports.append(a))
    scan = _Scan(folder, qa_settings=_thread_settings())
    assert scan.ended_within(60)
    assert scan.result is None and not reports


# ---- API calls in flight ----------------------------------------------------------------------------------------


class _Client:
    context = None
    _stop_callback = None

    def _get_thread_local_client(self):
        return threading.local()


@pytest.fixture
def ai_stage(tmp_path, monkeypatch):
    """A scan in its AI truncation stage with two API calls in flight (they block until ``release``)."""
    import extract_glossary_from_epub

    folder, epub = _book(tmp_path)
    client = _Client()
    monkeypatch.setattr(extract_glossary_from_epub, "create_client_with_multi_key_support", lambda *a, **k: client)
    release = threading.Event()
    in_flight = []
    both = threading.Event()

    def call(*args, **kwargs):
        in_flight.append(1)
        if len(in_flight) >= 2:
            both.set()
        release.wait(30)
        return {"flagged": False, "details": "fine", "cancelled": False}

    monkeypatch.setattr(shf, "run_ai_truncation_check", call)
    scan = _Scan(folder, epub_path=str(epub), mode="custom", qa_settings=_thread_settings(
        check_ai_truncation_detection=True,
        custom_mode_settings={"sample_size": 0},
        _live_config={"batch_translation": True, "batch_size": 2},
    ))
    assert both.wait(60), "\n".join(scan.lines[-20:])
    yield scan, client, release, in_flight, folder
    release.set()
    scan.thread.join(10)
    from qa_scan_runtime import clear_qa_stop_flags
    clear_qa_stop_flags()


def test_a_graceful_stop_lets_api_calls_in_flight_finish_and_starts_no_more(ai_stage):
    scan, client, release, in_flight, folder = ai_stage
    from qa_scan_runtime import apply_qa_graceful_stop_flags
    apply_qa_graceful_stop_flags()
    assert not scan.ended_within(0.6), "it did not wait for the calls in flight"
    assert client._stop_callback() is False  # the client must not abort them
    assert any("Waiting for 2 API call(s) in flight" in line and "Stop again" in line for line in scan.lines)
    release.set()
    assert scan.ended_within(INSTANT)
    assert scan.result is None and not list(folder.rglob("validation_results*"))
    assert len(in_flight) == 2, "a queued call started after the stop"


def test_the_forced_stop_does_not_wait_for_api_calls_in_flight(ai_stage):
    scan, client, _release, in_flight, _folder = ai_stage
    from qa_scan_runtime import apply_qa_force_stop_flags, apply_qa_graceful_stop_flags
    apply_qa_graceful_stop_flags()
    assert not scan.ended_within(0.3)
    forced_at = time.monotonic()
    apply_qa_force_stop_flags()  # the second click
    assert client._stop_callback() is True  # the client aborts them
    assert scan.ended_within(2.5), "\n".join(scan.lines[-10:])
    assert time.monotonic() - forced_at < 2.5 and len(in_flight) == 2


def test_a_force_stop_right_away_also_aborts_and_stays_stopped_after_the_flags_clear(ai_stage):
    """Graceful stop off: the first click forces. The GUI clears the flags soon after; a worker still running stays
    stopped, it must not carry on as if nothing had happened."""
    scan, client, _release, _in_flight, _folder = ai_stage
    from qa_scan_runtime import apply_qa_force_stop_flags, clear_qa_stop_flags
    apply_qa_force_stop_flags()
    assert scan.ended_within(2.5)
    clear_qa_stop_flags()
    os.environ.pop("GRACEFUL_STOP", None)
    shf._stop_flag = False
    assert client._stop_callback() is True


# ---- the glossary name check ----------------------------------------------------------------------------------------


def test_the_glossary_name_check_does_not_make_a_stop_wait(tmp_path, monkeypatch):
    import glossary_name_check as gnc

    release = threading.Event()
    calls = []
    sources = {f"response_{n}.html": "source" for n in range(6)}
    for name in sources:
        (tmp_path / name).write_text("text", encoding="utf-8")
    monkeypatch.setattr(gnc, "source_chapters_for_folder", lambda *a, **k: sources)

    def tagging(*args, **kwargs):
        calls.append(1)
        release.wait(30)

    monkeypatch.setattr(gnc, "check_chapter", tagging)
    flag = []
    threading.Timer(0.1, lambda: flag.append(1)).start()
    started = time.monotonic()
    found = gnc.scan_folder(str(tmp_path), [{"raw": "x"}], stop=lambda: bool(flag), workers=1)
    assert time.monotonic() - started < INSTANT
    assert found == [] and len(calls) == 1
    release.set()


def test_the_glossary_name_check_without_a_stop_still_checks_every_chapter(tmp_path, monkeypatch):
    import glossary_name_check as gnc

    sources = {f"response_{n}.html": "source" for n in range(4)}
    for name in sources:
        (tmp_path / name).write_text("text", encoding="utf-8")
    monkeypatch.setattr(gnc, "source_chapters_for_folder", lambda *a, **k: sources)
    seen = []
    monkeypatch.setattr(gnc, "check_chapter", lambda s, m, e, name, ner=None: seen.append(name))
    gnc.scan_folder(str(tmp_path), [{"raw": "x"}], stop=lambda: False, workers=1)
    assert sorted(seen) == sorted(sources)
    seen.clear()
    gnc.scan_folder(str(tmp_path), [{"raw": "x"}], workers=2)
    assert sorted(seen) == sorted(sources)

