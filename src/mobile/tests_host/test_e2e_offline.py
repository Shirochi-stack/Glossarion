"""Host tests for the offline end-to-end test (``diagnostics/e2e.py`` + ``diagnostics/fake_llm_server.py``).

Run from src/mobile (3.13 venv, or the desktop Python with the backend dependencies)::

    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_e2e_offline.py

* the fake OpenAI server: the OpenAI wire shapes through the real ``openai`` SDK (models, JSON,
  SSE with usage), the tagging "translation", glossary answers read by the shared parser, request
  classification on the real prompts, hold / release / abort;
* the E2E guards: write audit, process tripwire, loopback-only network, process-state diff;
* ``test_offline_e2e_suite``: the device suite ``e2e`` in a subprocess (bootstrap into temp
  ``FLET_APP_STORAGE_*`` dirs, then ``selftest.run_selftest("e2e")`` exactly as the deep link
  ``glossarion://app/__selftest__?suite=e2e`` does on a phone): a chat attachment with the Balanced
  glossary and its approval card, glossary Off + Compile EPUB, graceful stop + Resume, force stop
  + kill/relaunch + Resume, the Chapters tab's Retranslate (plan / apply) + Resolve QA and a Vision
  image + Generate from prompt in the chat (U7), and process hygiene, all through the real
  JobService, HeadlessOwner and shared pipeline. CI (build-mobile.yml prepare) also runs the suite
  on the collected bundle.
"""

from __future__ import annotations

import errno
import importlib.util
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
for entry in (str(APP_DIR), str(SRC_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from glossarion_mobile.diagnostics import e2e  # noqa: E402
from glossarion_mobile.diagnostics.fake_llm_server import (  # noqa: E402
    DEFAULT_GLOSSARY,
    FAKE_MARKER,
    FAKE_MODEL,
    FAKE_OCR_TEXT,
    FAKE_PNG,
    FakeLLMServer,
    applied_entries,
    chapter_numbers,
    classify_request,
    fake_translate,
    glossary_csv,
    has_image_part,
    png_bytes,
    romanize_hangul,
)

E2E_DEPENDENCIES = ("ebooklib", "openai", "httpx", "lxml", "bs4", "tiktoken")
SELFTEST_EPUB = APP_DIR / "assets" / "selftest" / "selftest_ko_12ch.epub"  # tools/prepare_assets.py (CI: prepare)
needs_selftest_epub = pytest.mark.skipif(not SELFTEST_EPUB.is_file(), reason="run tools/prepare_assets.py first")


def _missing(*modules: str) -> list:
    return [m for m in modules if importlib.util.find_spec(m) is None]


needs_openai = pytest.mark.skipif(bool(_missing("openai")), reason="openai not installed")

#: A real glossary-extraction system prompt (extract_glossary_from_epub, default settings) and the
#: reference glossary a translation prompt carries once a glossary is loaded.
EXTRACTION_SYSTEM = (
    "You are a novel glossary extraction assistant.\n\nYou must strictly return ONLY CSV format with columns "
    "separated by commas.\nColumns and entry types in this exact order provided:\n\nColumns:\n"
    "type, raw_name, translated_name, gender, description\\nEntry Types:\ncharacter, term"
)
TRANSLATION_SYSTEM = (
    "You are a professional novel translator. You MUST translate the following text to English.\n"
    "- Follow this reference glossary for consistent translation (Do not output any raw entries):\n"
    "Glossary Columns: raw_name, translated_name, gender, description\n\n=== CHARACTERS ===\n"
    "* 이서연 = Seo-yeon Lee [Female]: Young knight of the Silver Forest\n"
)
CHAPTER = "<body>\n<h1>제1화 은빛 숲의 소녀</h1>\n<p>새벽 안개가 은빛 숲을 덮고 있었다. 이서연은 낡은 검을 허리에 차고 숲길을 걸었다.</p>\n</body>"


# ==========================================================================
# Fake server: translation, glossary, classification
# ==========================================================================


def test_tagging_translation_keeps_markup_and_applies_an_injected_glossary():
    assert romanize_hangul("새벽 안개") == "saebyeok angae"
    applied = applied_entries(TRANSLATION_SYSTEM + CHAPTER, DEFAULT_GLOSSARY)
    assert [e.raw_name for e in applied] == ["이서연"]  # "Silver Forest" inside a description does not count
    out = fake_translate(CHAPTER, applied)
    assert out.startswith("<body>\n<h1>") and out.count("<p>") == 1 and out.endswith("</body>")
    assert "Seo-yeon Lee" in out and FAKE_MARKER in out
    assert not any("가" <= ch <= "힣" for ch in out)  # every Hangul run became Latin
    assert f"{FAKE_MARKER} je1hwa eunbit supui sonyeo" in out  # digits stay inside the run
    assert chapter_numbers(CHAPTER) == [1] and chapter_numbers("제2화 … 제12화") == [2, 12]
    assert fake_translate("<p>Already English.</p>") == "<p>Already English.</p>"


def test_glossary_answer_reads_back_with_the_shared_parser():
    pytest.importorskip("glossary_matching")
    from glossary_usage import parse_glossary_content

    text = "이서연과 강민호는 루미나 성의 언덕에서 해를 바라보았다."
    csv_text = glossary_csv(text, DEFAULT_GLOSSARY)
    entries = parse_glossary_content(csv_text, ".csv")
    assert [(e["raw_name"], e["translated_name"]) for e in entries] == [
        ("이서연", "Seo-yeon Lee"), ("강민호", "Min-ho Kang"), ("루미나 성", "Lumina Castle"),
    ]
    assert glossary_csv("no names here", DEFAULT_GLOSSARY).strip() == "type,raw_name,translated_name,gender,description"


def test_requests_are_classified_from_the_real_prompts():
    extraction = {"messages": [{"role": "system", "content": EXTRACTION_SYSTEM}, {"role": "user", "content": "제1화"}]}
    translation = {"messages": [{"role": "system", "content": TRANSLATION_SYSTEM}, {"role": "user", "content": CHAPTER}]}
    parts = {"messages": [{"role": "user", "content": [{"type": "text", "text": CHAPTER}]}]}
    assert classify_request(extraction) == "glossary"
    assert classify_request(translation) == "translation"  # an injected glossary is not an extraction request
    assert classify_request(parts) == "translation"


def test_server_answers_vision_image_generation_and_leaves_raw_text_once():
    """U7: an ``image_url`` part is a vision request (the OCR text back), ``/v1/images/generations``
    returns the fake PNG as ``b64_json``, ``leave_raw_once`` keeps Korean in one chapter answer once."""
    import base64

    picture = "data:image/png;base64," + base64.b64encode(png_bytes(4, 4)).decode("ascii")
    vision = {"model": FAKE_MODEL, "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Translate the text in this image."},
        {"type": "image_url", "image_url": {"url": picture}}]}]}
    assert has_image_part(vision) and classify_request(vision) == "vision"
    assert png_bytes(2, 2).startswith(b"\x89PNG\r\n\x1a\n") and FAKE_PNG.startswith(b"\x89PNG")
    with FakeLLMServer() as server:
        def post(path, payload):
            request = urllib.request.Request(server.url + path, data=json.dumps(payload).encode("utf-8"),
                                             headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read())

        assert post("/chat/completions", vision)["choices"][0]["message"]["content"] == FAKE_OCR_TEXT
        generated = post("/images/generations", {"model": FAKE_MODEL, "prompt": "a red fox", "n": 1})
        assert base64.b64decode(generated["data"][0]["b64_json"]) == FAKE_PNG
        server.leave_raw_once[1] = "남은 문장"
        chapter = {"model": FAKE_MODEL, "messages": [{"role": "user", "content": CHAPTER}]}
        first = post("/chat/completions", chapter)["choices"][0]["message"]["content"]
        again = post("/chat/completions", chapter)["choices"][0]["message"]["content"]
        assert "<p>남은 문장 " in first and not any("가" <= ch <= "힣" for ch in again)
        assert not server.leave_raw_once
        records = server.records()
        assert [(r.kind, r.status) for r in records] == [("vision", "ok"), ("image_generation", "ok"),
                                                         ("translation", "ok"), ("translation", "ok")]
        assert records[1].preview == "a red fox" and records[1].reply_chars == len(FAKE_PNG)


def test_the_server_only_listens_on_loopback():
    with pytest.raises(ValueError):
        FakeLLMServer(host="0.0.0.0")
    with pytest.raises(RuntimeError):
        _ = FakeLLMServer().url  # not started


@needs_openai
def test_server_speaks_the_openai_wire_format():
    import openai

    with FakeLLMServer() as server:
        client = openai.OpenAI(api_key="sk-dummy", base_url=server.url, max_retries=0)
        try:
            assert [m.id for m in client.models.list().data] == [FAKE_MODEL]
            reply = client.chat.completions.create(
                model=FAKE_MODEL, messages=[{"role": "system", "content": TRANSLATION_SYSTEM},
                                            {"role": "user", "content": CHAPTER}])
            assert "Seo-yeon Lee" in reply.choices[0].message.content
            assert reply.choices[0].finish_reason == "stop" and reply.usage.total_tokens > 0
            stream = client.chat.completions.create(
                model=FAKE_MODEL, stream=True, stream_options={"include_usage": True},
                messages=[{"role": "user", "content": "<p>새벽 안개가</p>"}])
            chunks = list(stream)
            text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
            assert text == f"<p>{FAKE_MARKER} saebyeok angaega</p>"
            assert chunks[-1].usage is not None and not chunks[-1].choices
            glossary = client.chat.completions.create(
                model=FAKE_MODEL, messages=[{"role": "system", "content": EXTRACTION_SYSTEM},
                                            {"role": "user", "content": "이서연은 마나석을 찾았다."}])
            assert glossary.choices[0].message.content.splitlines()[1:] == [
                "character,이서연,Seo-yeon Lee,Female,Young knight of the Silver Forest",
                "term,마나석,mana stone,,Stone that stores mana",
            ]
        finally:
            client.close()
        records = server.records()
        assert [(r.kind, r.stream, r.status) for r in records] == [
            ("translation", False, "ok"), ("translation", True, "ok"), ("glossary", False, "ok")]
        assert records[0].chapters == [1] and records[0].glossary_applied == ["이서연"]
        assert server.count("translation") == 2 and server.count(since=2) == 1


def _post(url: str, results: list, index: int) -> None:
    body = json.dumps({"model": FAKE_MODEL, "messages": [{"role": "user", "content": CHAPTER}]}).encode("utf-8")
    request = urllib.request.Request(url + "/chat/completions", data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            results[index] = ("ok", json.loads(response.read())["choices"][0]["message"]["content"])
    except Exception as exc:  # the aborted request: no response at all
        results[index] = ("error", type(exc).__name__)


def _wait(predicate, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def test_hold_parks_requests_until_released_or_aborted():
    with FakeLLMServer(hold_timeout=30) as server:
        responses: list = []
        server.on_response.append(lambda record: responses.append(record.id))
        server.hold()
        results = [None, None]
        first = threading.Thread(target=_post, args=(server.url, results, 0))
        first.start()
        assert _wait(lambda: server.parked == 1) and server.holding and results[0] is None
        server.release()
        first.join(10)
        assert results[0][0] == "ok" and FAKE_MARKER in results[0][1]
        assert _wait(lambda: responses == [1])  # on_response runs on the handler thread after the reply

        server.hold()
        second = threading.Thread(target=_post, args=(server.url, results, 1))
        second.start()
        assert _wait(lambda: server.parked == 1)
        server.release(abort=True)  # a stopped run never receives the parked answer
        second.join(10)
        assert results[1][0] == "error" and responses == [1]
        records = server.records()
        assert [(r.status, r.parked) for r in records] == [("ok", True), ("aborted", True)]
        assert not server.holding and server.parked == 0
        server.reset()
        assert server.records() == [] and server.on_response == []


# ==========================================================================
# Guards
# ==========================================================================


def test_write_audit_records_writes_outside_the_allowed_roots(tmp_path):
    allowed = tmp_path / "data"
    allowed.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "existing.txt").write_text("x", encoding="utf-8")
    audit = e2e.WriteAudit([allowed])
    audit.start()
    try:
        (allowed / "fine.txt").write_text("ok", encoding="utf-8")
        (allowed / "sub").mkdir()
        with open(os.devnull, "w", encoding="utf-8") as handle:
            handle.write("ignored")
        (outside / "existing.txt").read_text(encoding="utf-8")  # reading is not writing
        cache = outside / "__pycache__"
        cache.mkdir()  # bytecode caches are tolerated
        (outside / "stray.txt").write_text("no", encoding="utf-8")
        os.mkdir(outside / "made")
        os.replace(allowed / "fine.txt", outside / "moved.txt")
        fd = os.open(str(outside / "raw.bin"), os.O_WRONLY | os.O_CREAT)
        os.close(fd)
    finally:
        audit.stop()
    (outside / "after-stop.txt").write_text("not recorded", encoding="utf-8")
    seen = [(v["event"], Path(v["path"]).name) for v in audit.violations]
    assert seen == [("open", "stray.txt"), ("os.mkdir", "made"), ("os.rename", "moved.txt"), ("open", "raw.bin")]


def test_process_guards_refuse_spawns_and_remote_sockets():
    from concurrent.futures import ProcessPoolExecutor

    original_popen_init = subprocess.Popen.__init__
    original_connect = socket.socket.connect
    guards = e2e.ProcessGuards()
    guards.install()
    try:
        with pytest.raises(OSError) as spawn:
            subprocess.run([sys.executable, "-c", "pass"], check=False)
        assert spawn.value.errno == e2e.ENOTSUP
        with pytest.raises(OSError):
            ProcessPoolExecutor(max_workers=1)
        with pytest.raises(OSError):
            os.system("echo refused")
        remote = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with pytest.raises(OSError) as net:
                remote.connect(("203.0.113.9", 9))  # TEST-NET-3: refused before any packet is sent
            assert net.value.errno == errno.ENETUNREACH
        finally:
            remote.close()
        with pytest.raises(socket.gaierror):
            socket.getaddrinfo("glossarion.invalid", 443)
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            client.connect(listener.getsockname())  # loopback stays allowed (the fake server)
        finally:
            client.close()
            listener.close()
    finally:
        guards.uninstall()
    assert [e["api"] for e in guards.spawns] == ["subprocess.Popen", "ProcessPoolExecutor", "os.system"]
    assert [e["target"] for e in guards.network] == ["203.0.113.9:9", "glossarion.invalid"]
    assert subprocess.Popen.__init__ is original_popen_init and socket.socket.connect is original_connect


def test_process_state_diff_names_what_a_job_left_behind(tmp_path, monkeypatch):
    before = e2e.process_state()
    assert e2e.diff_process_state(before, e2e.process_state()) == []
    monkeypatch.setenv("GLOSSARION_E2E_LEAK", "1")
    monkeypatch.setattr(sys, "argv", ["leaked.py"])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "stdout", sys.__stdout__)
    problems = e2e.diff_process_state(before, e2e.process_state())
    text = " | ".join(problems)
    assert "os.environ gained ['GLOSSARION_E2E_LEAK']" in text and "sys.argv" in text and "cwd is" in text
    if sys.stdout is not before["stdout"]:
        assert "sys.stdout was not restored" in text


def test_epub_report_and_progress_rows(tmp_path):
    import zipfile

    book = tmp_path / "book.epub"
    with zipfile.ZipFile(book, "w") as archive:
        archive.writestr("EPUB/nav.xhtml", "<html><body>목차</body></html>")
        archive.writestr("EPUB/Text/chapter0001.xhtml", f"<html><body><p>{FAKE_MARKER} hello</p></body></html>")
        archive.writestr("EPUB/Text/chapter0002.xhtml", "<html><body><p>안녕</p></body></html>")
    report = e2e.epub_chapter_report(str(book), FAKE_MARKER)
    assert report == {"EPUB/Text/chapter0001.xhtml": {"marker": True, "hangul": 0},
                      "EPUB/Text/chapter0002.xhtml": {"marker": False, "hangul": 2}}
    (tmp_path / "response_chapter0001.html").write_text("x", encoding="utf-8")
    (tmp_path / "translation_progress.json").write_text(json.dumps({"chapters": {
        "1": {"actual_num": 1, "output_file": "response_chapter0001.html", "status": "completed"},
        "2": {"actual_num": 2, "output_file": "response_chapter0002.html", "status": "completed"},
        "3": {"actual_num": 3, "output_file": "response_chapter0003.html", "status": "in_progress"},
        "meta": {"output_file": "metadata.json", "status": "completed"},
    }}), encoding="utf-8")
    rows = e2e._progress_chapters(str(tmp_path))
    assert rows == {1: "completed", 2: "completed (file missing)", 3: "in_progress"}
    assert e2e._completed(rows) == {1}


def test_selftest_registers_the_e2e_suite_and_route():
    from glossarion_mobile.diagnostics import selftest
    from glossarion_mobile.ui.router import parse_route

    assert [name for name, _check in selftest.SUITES["e2e"]] == [name for name, _method in e2e.SCENARIOS]
    assert all(hasattr(e2e.E2ESession, method) for _name, method in e2e.SCENARIOS)
    match = parse_route("glossarion://app/__selftest__?suite=e2e")
    assert match is not None and match.name == "selftest" and match.get("suite") == "e2e"


@pytest.mark.skipif(bool(_missing("flet", "msgpack")), reason="flet/msgpack not installed")
def test_diagnostics_button_runs_the_e2e_suite():
    import asyncio

    from glossarion_mobile.state.app_state import AppState
    from glossarion_mobile.ui.screens.diagnostics import DiagnosticsScreen

    calls: list = []

    class Runner:  # SelfTestRunner surface the screen uses
        current_suite = None

        async def run(self, suite, *, source="button"):
            calls.append((suite, source))
            return {"suite": suite}

    screen = DiagnosticsScreen(None, page=None, state=AppState(), dispatcher=None, runner=Runner())
    screen.build_body()
    assert screen.e2e_button.content == "Run end-to-end test" and not screen.e2e_button.disabled
    asyncio.run(screen._on_run_e2e())
    asyncio.run(screen._on_run())
    assert calls == [("e2e", "diagnostics"), ("smoke", "diagnostics")]
    screen.runner.current_suite = "e2e"
    screen._render_running(True)
    assert screen.result_text.value == "Running suite 'e2e'…"
    assert screen.e2e_button.disabled and screen.run_button.disabled
    screen._render_running(False)
    assert not screen.e2e_button.disabled


@needs_selftest_epub
def test_e2e_refuses_to_start_while_an_app_job_runs(tmp_path):
    import types

    import job_runner

    held, release = threading.Event(), threading.Event()

    def app_job() -> None:
        with job_runner.JOB_LOCK:
            held.set()
            release.wait(20)

    thread = threading.Thread(target=app_job, name="gl-job")
    thread.start()
    try:
        assert held.wait(10)
        paths = types.SimpleNamespace(assets_dir=APP_DIR / "assets", temp=tmp_path, backend_dir=SRC_DIR,
                                      writable_dirs=lambda: {})
        session = e2e.E2ESession(paths)
        env_before = dict(os.environ)
        with pytest.raises(e2e.E2EFailure, match="A job is running in the app"):
            session.setup()
        with pytest.raises(e2e.E2EFailure):  # every later check reports the same setup error
            session.process_hygiene()
        session.close()
        assert dict(os.environ) == env_before and not session.ready
        assert not (tmp_path / "glossarion-e2e").exists()  # nothing was created
    finally:
        release.set()
        thread.join(10)


# ==========================================================================
# The suite itself (subprocess: bootstrap + selftest "e2e", the device path)
# ==========================================================================


@needs_selftest_epub
@pytest.mark.skipif(bool(_missing(*E2E_DEPENDENCIES)), reason=f"backend dependencies missing: {_missing(*E2E_DEPENDENCIES)}")
def test_offline_e2e_suite(tmp_path):
    storage = {name: tmp_path / name for name in ("data", "cache", "temp")}
    for directory in storage.values():
        directory.mkdir()
    # The child imports exactly like this interpreter; the app dir is its cwd (python -m) and the
    # backend resolves to the repo src/ (runtime_bootstrap's dev order).
    env = {k: v for k, v in os.environ.items() if not k.startswith(("FLET_", "GLOSSARION_"))}
    env.update({f"FLET_APP_STORAGE_{name.upper()}": str(path) for name, path in storage.items()})
    env.update(PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    result_file = tmp_path / "e2e_result.json"
    proc = subprocess.run(
        [sys.executable, "-m", "glossarion_mobile.diagnostics.e2e", "--json", str(result_file)],
        cwd=str(APP_DIR), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1500,
    )
    tail = (proc.stderr or "")[-6000:]
    assert result_file.is_file(), f"no result (exit {proc.returncode}):\n{tail}"
    result = json.loads(result_file.read_text(encoding="utf-8"))
    checks = {c["name"]: c for c in result["checks"]}
    failed = {n: c.get("error") or c.get("reason") for n, c in checks.items() if c["status"] != "pass"}
    assert not failed, json.dumps(failed, indent=1, ensure_ascii=False) + "\n" + tail
    assert proc.returncode == 0 and result["ok"] and list(checks) == [name for name, _ in e2e.SCENARIOS]
    markers = [ln for ln in proc.stderr.splitlines() if ln.startswith("GLOSSARION_SELFTEST ")]
    assert markers and markers[-1].startswith('GLOSSARION_SELFTEST PASS {"suite":"e2e"')

    # U5: the translated + compiled workspace opens in the Library, Book page, Chapters tab and Reader.
    library = checks["e2e_translate_glossary_off"]["detail"]["library"]
    assert library["library"]["shelf"] == "completed" and library["library"]["card"] == f"{e2e.CHAPTERS}/{e2e.CHAPTERS}"
    assert library["raw_inputs_registered"] >= 1 and library["book_page"]["chapters"] == e2e.CHAPTERS
    assert library["chapters_tab"]["statuses"].get("completed") == e2e.CHAPTERS
    assert library["reader"]["mode"] == "dual" and library["reader"]["translated"] == e2e.CHAPTERS
    chat = checks["e2e_chat_balanced_glossary"]["detail"]
    assert chat["requests"]["glossary"] >= 1 and chat["requests"]["chapters"] == e2e.CHAPTERS
    assert chat["approval_entries"] == len(DEFAULT_GLOSSARY) and chat["live_cards"] >= 1
    assert chat["book"]["progress"]["completed"] == e2e.CHAPTERS
    # U6: extract glossary -> edit -> save -> translate with the edit -> QA quick scan -> PDF via the shim.
    edited = checks["e2e_glossary_edit_qa_pdf"]["detail"]
    assert edited["edited"] == {e2e.E2ESession.EDIT_RAW: ["Seo-yeon Lee", e2e.E2ESession.EDIT_NAME]}
    assert edited["edited_prompts"] >= 1 and edited["backups"] >= 1
    assert edited["book"]["progress"]["completed"] == e2e.CHAPTERS and edited["qa"]["chapters"] == e2e.CHAPTERS
    assert edited["pdf"]["pages"] >= e2e.CHAPTERS and edited["pdf"]["outline"] >= e2e.CHAPTERS
    graceful = checks["e2e_graceful_stop_resume"]["detail"]
    assert graceful["stop_secs"] <= e2e.STOP_DEADLINE
    assert sorted(graceful["saved_before_resume"] + graceful["resumed"]) == list(range(1, e2e.CHAPTERS + 1))
    forced = checks["e2e_force_stop_kill_resume"]["detail"]
    assert forced["parked_at_stop"] >= 1  # the model was stalled when Stop was pressed
    assert sorted(forced["saved_before_resume"] + forced["resumed"]) == list(range(1, e2e.CHAPTERS + 1))
    # U7: Chapters tab Retranslate (only the two reset chapters are sent again) + Resolve QA (Partial.b).
    retranslated = checks["e2e_retranslate_resolve_qa"]["detail"]
    assert retranslated["retranslated"] == retranslated["resent"] == list(e2e.E2ESession.RETRANSLATE_CHAPTERS)
    assert retranslated["confirm_title"] == "Confirm Retranslation" and retranslated["resolve_requests"] >= 1
    assert retranslated["qa_flagged"] == e2e.E2ESession.RAW_QA_CHAPTER and retranslated["resolved_status"] == "completed"
    assert retranslated["book"]["completed"] == e2e.CHAPTERS
    # U7: Vision on an image attachment + Generate from prompt (Image) in the chat.
    media = checks["e2e_vision_and_generate"]["detail"]
    assert media["vision"]["requests"] == 1 and media["vision"]["responses"]
    assert media["generate"]["requests"] == 1 and media["generate"]["image"].startswith("Direct Text ")
    # U8: Tools › Manga Start on a 3-page CBZ (custom-api OCR + translation, inpainting skipped, CBZ at the end).
    manga = checks["e2e_manga_cbz"]["detail"]
    pages = e2e.E2ESession.MANGA_PAGES
    assert manga["pages"] == manga["ocr_requests"] == pages and manga["translation_requests"] >= 1
    assert manga["cbz"].endswith("_translated.cbz") and len(manga["cbz_members"]) == pages
    hygiene = checks["e2e_process_hygiene"]["detail"]
    assert hygiene["spawn_attempts"] == 0 and hygiene["writes_outside"] == 0 and hygiene["network_attempts"] == 0
    assert all(not job["process_diff"] for job in hygiene["jobs"]) and len(hygiene["jobs"]) == 19

    # The sandbox is gone after a pass; the user's own chats, settings, jobs and outputs were never touched.
    sandboxes = storage["temp"] / "glossarion-e2e"
    assert not sandboxes.exists() or not any(sandboxes.iterdir())
    data = storage["data"]
    for name in ("config.json", "direct_text_chats.json", "direct_text_chats.mobile.json", "jobs", "Inbox"):
        assert not (data / name).exists(), name
    assert not any((data / "Output").iterdir()) and not any((data / "Library").iterdir())
