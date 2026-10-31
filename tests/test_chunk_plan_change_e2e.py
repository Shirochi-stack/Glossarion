"""A chapter finished under one chunk plan must stay finished when the token budget would now split it
differently. Reported as "Ch.089 Chunk 3/2 ... 9/2 Pending" rows that kept re-sending their requests.

Real ``TransateKRtoEN.main()`` runs (a child process each: main() rewires print, the environment and module
state) on a real EPUB against a local fake OpenAI-compatible server, reading the real progress file back.
"""
import http.server
import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
PARAGRAPHS = 120
MODEL = "gpt-4o-mini"


class _Server(http.server.ThreadingHTTPServer):
    daemon_threads = True
    requests = []


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        marks = re.findall(r"PARA(\d{4})", json.dumps(body["messages"][-1], ensure_ascii=False))
        if marks:
            _Server.requests.append([marks[0], marks[-1]])
        text = "".join(f"<p>EN{mark}</p>\n" for mark in marks) or "<p>EN</p>"
        raw = json.dumps({"id": "x", "object": "chat.completion", "model": "m", "created": 1,
                          "choices": [{"index": 0, "finish_reason": "stop",
                                       "message": {"role": "assistant", "content": text}}],
                          "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


def _chapter_body():
    para = "그는 천천히 문을 열고 어두운 복도를 따라 걸어갔다. 아무도 그의 발소리를 듣지 못했다. " * 3
    return "\n".join(f"<p>PARA{index:04d} {para}</p>" for index in range(1, PARAGRAPHS + 1))


def _write_epub(path):
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_identifier("chunk-plan")
    book.set_title("Plan")
    book.set_language("ko")
    chapter = epub.EpubHtml(title="제1장", file_name="chapter_001.xhtml", lang="ko")
    chapter.content = f"<h1>제1장</h1>\n{_chapter_body()}"
    book.add_item(chapter)
    book.toc = (chapter,)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", chapter]
    epub.write_epub(str(path), book)


def _split_limits():
    """Output limits giving this chapter 2 chunks and many, whichever tokenizer the machine has."""
    from chapter_splitter import ChapterSplitter

    splitter = ChapterSplitter(model_name=MODEL)
    tokens = splitter.count_tokens(_chapter_body())
    return int(tokens * 0.75) + 500, max(tokens // 8, 1000) + 500


def _set_limit(via, limit):
    if via == "key":
        os.environ["USE_MULTI_API_KEYS"] = "1"
        os.environ["MULTI_API_KEYS"] = json.dumps([{
            "api_key": "sk-test", "model": MODEL, "cooldown": 0, "enabled": True,
            "individual_output_token_limit": limit}])
    else:
        os.environ["USE_MULTI_API_KEYS"] = "0"
        os.environ["MULTI_API_KEYS"] = "[]"
        os.environ["MAX_OUTPUT_TOKENS"] = str(limit)


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _chapter(prog):
    """The chapter with the ledger (the book's nav page and metadata have rows of their own)."""
    for key, info in prog["chapters"].items():
        if prog["chapter_chunks"].get(info.get("content_hash")):
            return key, info, prog["chapter_chunks"][info["content_hash"]]
    raise AssertionError(f"no chunk ledger: {sorted(prog['chapters'])}")


def _duplicated_output(out, info, count):
    """What a stuck run wrote: the old plan's whole chapter, then most of it again from the new plan."""
    from chapter_chunk_progress import wrap_chunk_html

    ledger_key = info["content_hash"]
    paragraphs = [f"<p>EN{index:04d}</p>" for index in range(1, PARAGRAPHS + 1)]
    half = len(paragraphs) // 2
    blocks = ["\n".join(paragraphs[:half]), "\n".join(paragraphs[half:])]
    blocks += ["\n".join(paragraphs[start:start + 8]) for start in range(20, 20 + 8 * (count - 2), 8)]
    html = "\n".join(wrap_chunk_html(ledger_key, index, count, block) for index, block in enumerate(blocks, 1))
    (out / info["output_file"]).write_text(html, encoding="utf-8")


def _apply_trigger(trigger, out, progress_file):
    prog = _read(progress_file)
    key, info, ledger = _chapter(prog)
    if trigger == "output_deleted":
        (out / info["output_file"]).unlink()
    elif trigger == "qa_failed":
        info["status"] = "qa_failed"
        info["qa_issues_found"] = ["MISSING_IMAGES"]
    elif trigger == "stopped_before_write":
        info["status"] = "in_progress"
        (out / info["output_file"]).unlink()
    elif trigger == "already_stuck":
        # A progress file from a build with the bug: rows 3-9 under a 2-chunk plan, parent pending.
        for index in range(3, 10):
            ledger["entries"][str(index)] = {"index": index, "status": "pending", "qa_issues_found": [],
                                             "qa_issue_previews": {}}
        info["status"] = "pending"
        _duplicated_output(out, info, 9)
    elif trigger == "rows_deleted_by_hand":
        # The manual workaround: rows gone and the parent called done, the duplicated output still there.
        info["status"] = "completed"
        _duplicated_output(out, info, 9)
    elif trigger == "partial_plan":
        # Nothing finished to keep: this plan really is stale once the budget changes.
        from chapter_chunk_progress import reset_chunks_for_retranslation

        reset_chunks_for_retranslation(ledger, [2])
        info["status"] = "pending"
        (out / info["output_file"]).unlink()
    progress_file.write_text(json.dumps(prog, ensure_ascii=False, indent=2), encoding="utf-8")


def _snapshot(out, progress_file):
    prog = _read(progress_file)
    _key, info, ledger = _chapter(prog)
    html = (out / info["output_file"]).read_text(encoding="utf-8")
    translated = re.findall(r"EN(\d{4})", html)
    snapshot = {
        "requests": list(_Server.requests),
        "status": info.get("status"),
        "total": ledger.get("total"),
        "rows": {key: row.get("status") for key, row in ledger.get("entries", {}).items()},
        "results": sorted(ledger.get("chunks", {}), key=int),
        "marker_totals": sorted({int(total) for total in re.findall(r"idx=\d+ total=(\d+)", html)}),
        "paragraphs": len(translated),
        "unique_paragraphs": len(set(translated)),
    }
    _Server.requests.clear()
    return snapshot


def _drive(work, batch, via, trigger):
    """Child process: translate, send the chapter back to the queue, change the budget, run twice more."""
    sys.path.insert(0, str(SRC))
    work.mkdir(parents=True, exist_ok=True)
    os.chdir(work)
    book = work / "book.epub"
    _write_epub(book)
    server = _Server(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    os.environ.update({
        "GLOSSARION_HTTP_LOG": "0", "OUTPUT_DIRECTORY": str(work), "input_path": str(book), "EPUB_PATH": str(book),
        "MODEL": MODEL,
        "API_KEY": "sk-test", "OPENAI_API_KEY": "sk-test", "USE_CUSTOM_OPENAI_ENDPOINT": "1",
        "OPENAI_CUSTOM_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1", "SYSTEM_PROMPT": "Translate.",
        "MAX_OUTPUT_TOKENS": "65536", "COMPRESSION_FACTOR": "1.0", "BATCH_TRANSLATION": batch, "BATCH_SIZE": "3",
        "SEND_INTERVAL_SECONDS": "0", "THREAD_SUBMISSION_DELAY_SECONDS": "0", "ENABLE_CHUNK_PROGRESS": "1",
        "ENABLE_AUTO_GLOSSARY": "0", "AUTO_GLOSSARY_MODE": "off", "CONTEXTUAL": "0", "RETRY_TRUNCATED": "0",
        "RETRY_TIMEOUT": "0", "ENABLE_POST_TRANSLATION_SCAN": "0", "SKIP_EPUB_BUILD": "1",
    })
    import TransateKRtoEN

    out = work / "book"
    progress_file = out / "translation_progress.json"
    big, small = _split_limits()
    results = {}

    def run(name, limit):
        _set_limit(via, limit)
        TransateKRtoEN.main(log_callback=lambda *a: None, stop_callback=lambda: False)
        results[name] = _snapshot(out, progress_file)

    run("first", big)
    _apply_trigger(trigger, out, progress_file)
    run("second", small)
    run("third", small)
    (work / "results.json").write_text(json.dumps(results), encoding="utf-8")
    os._exit(0)  # the app's pools and client threads are not this test's to wait for


def _results(tmp_path, batch, via, trigger):
    pytest.importorskip("ebooklib")
    env = dict(os.environ)
    for name in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "OUTPUT_DIRECTORY", "GLOSSARION_LIBRARY_DIR",
                 "GLOSSARION_DATA_DIR"):
        env[name] = str(tmp_path)
    env["PYTHONIOENCODING"] = "utf-8"
    work = tmp_path / "work"
    done = subprocess.run([sys.executable, __file__, str(work), batch, via, trigger], env=env,
                          capture_output=True, encoding="utf-8", errors="replace", timeout=400)
    assert (work / "results.json").exists(), (done.stdout + done.stderr)[-3000:]
    return _read(work / "results.json")


def _finished_plan(rows=("1", "2"), results=("1", "2"), **extra):
    return {
        "schema_version": 2,
        "total": 2,
        "completed": [1, 2],
        "chunks": {index: f"<p>result {index}</p>" for index in results},
        "chunk_metadata": {},
        "entries": {index: {"index": int(index), "status": "completed", "source": f"<p>source {index}</p>"}
                    for index in rows},
        "chapter_status": "completed",
        **extra,
    }


def test_a_plan_keeps_only_its_own_rows():
    from chapter_chunk_progress import (chunk_entry_is_fully_completed, chunk_entry_needs_translation,
                                        chunk_failure_summary, ensure_chunk_entry_schema)

    entry = _finished_plan()
    for index in range(3, 10):
        entry["entries"][str(index)] = {"index": index, "status": "pending"}
    entry["chunks"]["5"] = "<p>another split's chunk</p>"
    entry["chunk_metadata"]["5"] = {"model_name": "m"}

    assert chunk_entry_is_fully_completed(entry)
    assert ensure_chunk_entry_schema(entry) is True
    assert list(entry["entries"]) == ["1", "2"]
    assert list(entry["chunks"]) == ["1", "2"]
    assert entry["chunk_metadata"] == {}
    assert entry["completed"] == [1, 2]
    assert chunk_failure_summary(entry)["pending"] == 0
    assert chunk_entry_needs_translation(entry) is False
    assert ensure_chunk_entry_schema(entry) is False


def test_an_entry_with_no_known_plan_keeps_its_rows():
    from chapter_chunk_progress import ensure_chunk_entry_schema

    entry = {"schema_version": 2, "chunks": {"3": "x"}, "entries": {"3": {"index": 3, "status": "completed"}},
             "chunk_metadata": {}, "completed": [3]}
    ensure_chunk_entry_schema(entry)
    assert list(entry["entries"]) == ["3"] and list(entry["chunks"]) == ["3"]


def test_nothing_can_add_a_row_outside_the_plan():
    from chapter_chunk_progress import set_chunk_qa, set_chunk_runtime_status

    entry = _finished_plan()
    assert set_chunk_runtime_status(entry, 3, "in_progress", model_name="m") is False
    assert set_chunk_qa(entry, 3, ["TRUNCATED"]) is False
    assert list(entry["entries"]) == ["1", "2"]
    assert set_chunk_runtime_status(entry, 2, "in_progress", model_name="m") is True
    assert entry["entries"]["2"]["status"] == "in_progress"


def test_only_a_finished_plan_of_another_size_replaces_the_split():
    from chapter_chunk_progress import completed_plan_split

    fresh = [(f"<p>new {index}</p>", index, 3) for index in range(1, 4)]
    entry = _finished_plan()
    assert completed_plan_split(entry, fresh) == [("<p>source 1</p>", 1, 2), ("<p>source 2</p>", 2, 2)]

    same_size = [("<p>new 1</p>", 1, 2), ("<p>new 2</p>", 2, 2)]
    assert completed_plan_split(entry, same_size) is same_size
    assert completed_plan_split(None, fresh) is fresh
    partial = _finished_plan(results=("1",))
    assert completed_plan_split(partial, fresh) is fresh
    flagged = _finished_plan()
    flagged["entries"]["2"]["qa_issues_found"] = ["TRUNCATED"]
    assert completed_plan_split(flagged, fresh) is fresh
    no_source = _finished_plan()
    del no_source["entries"]["1"]["source"]
    assert completed_plan_split(no_source, fresh)[0] == ("", 1, 2)


def _assert_kept_two_chunk_plan(result):
    assert result["requests"] == []
    assert result["status"] == "completed"
    assert result["total"] == 2
    assert result["rows"] == {"1": "completed", "2": "completed"}
    assert result["marker_totals"] == [2]
    assert (result["paragraphs"], result["unique_paragraphs"]) == (PARAGRAPHS, PARAGRAPHS)


@pytest.mark.parametrize("batch, via, trigger", [
    ("1", "key", "output_deleted"),
    ("1", "global", "qa_failed"),
    ("1", "key", "stopped_before_write"),
    ("0", "key", "output_deleted"),
])
def test_finished_plan_survives_a_budget_that_splits_it_differently(tmp_path, batch, via, trigger):
    results = _results(tmp_path, batch, via, trigger)
    assert results["first"]["total"] == 2
    _assert_kept_two_chunk_plan(results["second"])
    _assert_kept_two_chunk_plan(results["third"])


@pytest.mark.parametrize("trigger", ["already_stuck", "rows_deleted_by_hand"])
def test_progress_left_by_the_bug_heals_without_a_request(tmp_path, trigger):
    results = _results(tmp_path, "1", "key", trigger)
    _assert_kept_two_chunk_plan(results["second"])
    _assert_kept_two_chunk_plan(results["third"])


def test_a_partial_plan_is_still_invalidated_and_translated_under_the_new_budget(tmp_path):
    results = _results(tmp_path, "1", "key", "partial_plan")
    second, third = results["second"], results["third"]
    assert second["total"] > 2
    assert len(second["requests"]) == second["total"]
    assert second["status"] == "completed"
    assert set(second["rows"].values()) == {"completed"}
    assert second["marker_totals"] == [second["total"]]
    assert second["unique_paragraphs"] == PARAGRAPHS
    assert second["paragraphs"] == PARAGRAPHS
    assert third["requests"] == []
    assert third["status"] == "completed"


if __name__ == "__main__":
    _drive(Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4])
