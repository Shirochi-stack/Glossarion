"""Payload Summary/logistics.html: readable summary of the saved API payloads (payload_summary)."""
import json
import os
import sys

import pytest

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import payload_summary as ps  # noqa: E402


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def payloads(tmp_path):
    root = tmp_path / "Payloads"
    new = write(root / "translation" / "MainThread_1" / "translation_payload_1_a_A0_120000_x.json", {
        "model": "claude-sonnet-5-5", "client_type": "anthropic", "timestamp": "2026-10-23T12:00:00",
        "messages": [{"role": "system", "content": "You are a translator."},
                     {"role": "user", "content": [{"type": "text", "text": "번역해 주세요"}, {"type": "image_url"}]},
                     {"role": "assistant", "content": "Understood."}],
        "request_params": {"temperature": 0.3, "max_tokens": 128000},
        "anti_duplicate": {"section": "translation", "enabled": True, "sent": {"top_p": 0.9}},
        "debug": {"chapter_info": {"chapter": 5, "chunk": "1", "total_chunks": "2"}},
        "key_identifier": "Key#1 (claude-sonnet-5-5)", "usage": {"input_tokens": 100, "output_tokens": 40}})
    # an old payload: only model and messages
    write(root / "glossary" / "Thread-1_2" / "glossary_payload_1_b_A0_110000_y.json",
          {"model": "gpt-4o", "messages": [{"role": "user", "content": "Extract names."}]})
    write(root / "failed_requests" / "failed_glossary_refinement_anthropic_20261023_193505.json",
          {"timestamp": "20261023_193505", "error": "HTTP 400: max_tokens too large", "model": "claude",
           "messages": []})
    write(root / "translation" / "MainThread_1" / "req_x_response.json", {"status": 200})  # not a request
    (root / "Logistics.json").write_text(json.dumps([{"payload": str(new), "api_key": "sk-ant-a...1234",
                                                      "output_dir": "C:/Books/Book"}]), encoding="utf-8")
    return root


def test_rows_fill_what_each_payload_has(payloads):
    rows = {}
    logistics = ps._load_logistics(str(payloads))
    for path in ps.collect_files(str(payloads)):
        row, _data = ps.read_row(path, str(payloads), logistics)
        rows[row["kind"] + ":" + str(row["context"])] = row
    assert set(rows) == {"Request:translation", "Request:glossary", "Failed:glossary_refinement"}
    new = rows["Request:translation"]
    assert new["chapter"] == "Ch 5 · chunk 1/2" and new["key"] == "Key#1 (claude-sonnet-5-5) · sk-ant-a...1234"
    assert new["user"] == "번역해 주세요 [image]" and new["assistant"] == "Understood."
    assert new["anti_duplicate"] == "translation: top_p=0.9" and new["tokens"] == "100 / 40"
    assert new["output_dir"] == "C:/Books/Book" and new["max_tokens"] == 128000
    old = rows["Request:glossary"]
    assert old["temperature"] is None and old["key"] is None and old["system"] is None  # N/A on the page
    failed = rows["Failed:glossary_refinement"]
    assert failed["time"] == "2026-10-23 19:35:05" and failed["error"].startswith("HTTP 400")


def test_build_writes_index_and_details_and_hides_empty_columns(payloads, tmp_path):
    out = tmp_path / "Payload Summary"
    seen = []
    index = ps.build(str(payloads), out_dir=str(out), workers=2, progress=lambda d, t: seen.append(t))
    assert index == str(out / "logistics.html") and seen == [3, 3, 3]
    page = (out / "logistics.html").read_text(encoding="utf-8")
    assert '"retry"' not in page.split("const COLS =")[1].split(";")[0]  # no request has a retry: column hidden
    assert "N/A" in page and "glossarion.payloadSummary.columns" in page
    details = sorted(os.listdir(out / "details"))
    assert len(details) == 3
    detail = (out / "details" / details[0]).read_text(encoding="utf-8")
    assert "Raw JSON" in detail and "All requests" in detail
    # cancelled: no index
    assert ps.build(str(payloads), out_dir=str(tmp_path / "x"), cancel=lambda: True) is None


@pytest.mark.parametrize("cpus,config,expected", [
    (16, {"enable_parallel_extraction": True, "extraction_workers": 8}, 8),
    (2, {"enable_parallel_extraction": True, "extraction_workers": 8}, 2),   # never more than the CPU has
    (16, {"enable_parallel_extraction": False, "extraction_workers": 8}, 1),
    (4, {}, 2),
])
def test_workers_follow_parallel_extraction(monkeypatch, cpus, config, expected):
    monkeypatch.setattr(os, "cpu_count", lambda: cpus)
    assert ps.default_workers(config) == expected


def test_summary_folder_is_beside_payloads_not_inside(tmp_path):
    # the Payloads cleanup trims that folder oldest-first; the summary must not count toward it
    assert ps.summary_dir(str(tmp_path / "Payloads")) == str(tmp_path / "Payload Summary")
