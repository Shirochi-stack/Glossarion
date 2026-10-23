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
    # header context menu, resizable columns, 250-row pages (all remembered in the browser)
    assert "oncontextmenu" in page and "resizer" in page and "pageSize : 250" in page and "renderPager" in page
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


def test_output_folder_and_context_come_from_the_payload(tmp_path):
    """Payloads saved since 2026-10-23 carry output_dir / context themselves (no Logistics.json needed)."""
    root = tmp_path / "Payloads"
    path = write(root / "glossary" / "T_1" / "glossary_payload_1_c_A0_130000_z.json", {
        "model": "gpt-5", "context": "glossary", "output_dir": "D:/Novels/Book A",
        "messages": [{"role": "user", "content": "Extract names."}]})
    row, _data = ps.read_row(str(path), str(root), {})
    assert row["output_dir"] == "D:/Novels/Book A" and row["context"] == "glossary"


def test_output_folder_is_a_link(tmp_path):
    root = tmp_path / "Payloads"
    book = tmp_path / "Books" / "[362697] 말랑쫀득"
    path = write(root / "translation" / "T_1" / "translation_payload_1_d_A0_140000_w.json",
                 {"model": "m", "output_dir": str(book), "messages": []})
    row, data = ps.read_row(str(path), str(root), {})
    assert row["output_dir_uri"].startswith("file:///") and row["output_dir_uri"].endswith("%EB%93%9D")
    assert f'<a href="{row["output_dir_uri"]}">' in ps._detail_page(row, data, str(path))
    assert ps._folder_uri("relative/folder") is None and ps._folder_uri(None) is None


def test_rows_carry_their_raw_json_and_folder_links(payloads, tmp_path):
    """Right-click a row: View raw JSON, Open payloads subfolder, Open output folder (when it has one)."""
    rows = [ps.read_row(p, str(payloads), {})[0] for p in ps.collect_files(str(payloads))]
    for row in rows:
        assert row["raw_uri"].startswith("file:///") and row["raw_uri"].endswith(".json")
        assert row["raw_uri"].startswith(row["folder_uri"] + "/")
    index = ps.build(str(payloads), out_dir=str(tmp_path / "S"))
    html = open(index, encoding="utf-8").read()
    assert "showRowMenu" in html and "View raw JSON" in html and "Open payloads subfolder" in html
    assert "if (r.output_dir_uri) item(\"Open output folder\"" in html
    assert "tr.ondblclick" in html and "tr.onclick" not in html  # double-click opens the details


def test_auto_update_rebuilds_after_a_stop_only_when_on(monkeypatch, tmp_path):
    """Payloads > Auto-update: the stop handlers call _auto_update_payload_summary once the worker returned."""
    import threading
    import other_settings
    import unified_api_client
    built, done = [], threading.Event()
    monkeypatch.setattr(unified_api_client, "_payloads_dir", lambda: str(tmp_path / "Payloads"))
    monkeypatch.setattr(ps, "build", lambda d, workers=1, **_k: (built.append((d, workers)), done.set())[0] or "x.html")

    class Gui:
        config = {"extraction_workers": 2}
        payload_summary_auto_update_var = False
        logs = []
        append_log = logs.append

    gui = Gui()
    other_settings._auto_update_payload_summary(gui)
    assert not built  # off by default
    gui.payload_summary_auto_update_var = True
    other_settings._auto_update_payload_summary(gui)
    assert done.wait(5) and built[0][0] == str(tmp_path / "Payloads")
    src = open(os.path.join(SRC, "translator_gui.py"), encoding="utf-8").read()
    assert src.count("getattr(self, '_auto_update_payload_summary', lambda: None)()") == 2


def test_no_payloads_folder_writes_nothing(tmp_path, monkeypatch):
    # payload saving off leaves a relative "Payloads": never write a summary into the start folder
    monkeypatch.chdir(tmp_path)
    for missing in ("Payloads", str(tmp_path / "nope"), None):
        with pytest.raises(FileNotFoundError):
            ps.build(missing)
    assert not os.listdir(tmp_path)


def test_unwritable_summary_folder_falls_back_to_temp(payloads, tmp_path, monkeypatch):
    blocked = str(tmp_path / "blocked")
    real_makedirs = os.makedirs

    def makedirs(path, *a, **k):
        if str(path).startswith(blocked):
            raise PermissionError(13, "Permission denied", path)
        return real_makedirs(path, *a, **k)

    monkeypatch.setattr(os, "makedirs", makedirs)
    monkeypatch.setattr(ps.tempfile, "gettempdir", lambda: str(tmp_path / "temp"))
    index = ps.build(str(payloads), out_dir=blocked)
    assert index == str(tmp_path / "temp" / ps.FALLBACK_DIR / "logistics.html") and os.path.exists(index)


def test_a_detail_page_that_cannot_be_written_does_not_stop_the_rest(payloads, tmp_path, monkeypatch):
    import builtins
    real_open = builtins.open

    def flaky_open(path, mode="r", *a, **k):
        if "w" in mode and str(path).endswith("00001.html"):
            raise PermissionError(13, "Permission denied", path)
        return real_open(path, mode, *a, **k)

    monkeypatch.setattr(builtins, "open", flaky_open)
    index = ps.build(str(payloads), out_dir=str(tmp_path / "S"))
    page = real_open(index, encoding="utf-8").read()
    assert '"detail": null' in page and page.count('"detail": "details/') == 2
    assert "if (r.detail) window.open" in page


def test_chapter_label_leads_with_the_spine_position():
    # like the progress manager: [009] is the chapter's place in the book ("Spine 9/108" in the log)
    assert ps._chapter_text({"chapter": "3", "chunk": "1", "total_chunks": "1", "spine": 9}) == "[009] Ch 3 · chunk 1/1"
    assert ps._chapter_text({"chapter": "7", "chunk": None, "spine": 12}) == "[012] Ch 7"
    assert ps._chapter_text({"chapter": "5", "chunk": "1", "total_chunks": "2"}) == "Ch 5 · chunk 1/2"  # older payloads
    assert ps._chapter_text({"chapter": None, "chunk": None, "total_chunks": None}) is None


def test_decode_images_shows_the_request_images(tmp_path):
    import base64
    png = base64.b64encode(b"\x89PNG\r\n\x1a\nfake").decode()
    root = tmp_path / "Payloads"
    write(root / "image" / "T_1" / "image_payload_1_e_A0_150000_v.json", {"model": "m", "messages": [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + png}},                 # OpenAI
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": png}},    # Anthropic
            {"inline_data": {"mime_type": "image/webp", "data": png}},                                  # Gemini
            {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,@@not base64@@"}}]}]})
    off = tmp_path / "off"
    ps.build(str(root), out_dir=str(off))  # Decode images is off by default: no image files, no <img>
    assert not (off / "details" / "images").exists()
    assert "class='shot'" not in (off / "details" / "00001.html").read_text(encoding="utf-8")
    out = tmp_path / "S"
    ps.build(str(root), out_dir=str(out), decode_images=True)
    assert sorted(os.listdir(out / "details" / "images")) == ["00001_1_1.png", "00001_1_2.jpg", "00001_1_3.webp"]
    assert (out / "details" / "images" / "00001_1_1.png").read_bytes() == b"\x89PNG\r\n\x1a\nfake"
    page = (out / "details" / "00001.html").read_text(encoding="utf-8")
    assert page.count("class='shot'") == 4 and 'src="https://example.test/a.png"' in page
