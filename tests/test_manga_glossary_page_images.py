"""Manga Glossary Workflow: "Include page images in glossary generation" (owner request, 2026-11-06).

Generate Glossary and a batch run's glossary step (both ``MangaRunMixin._generate_manga_glossary_from_ocr_pages``)
used to send one text-only request with the OCR text of every page. With the new toggle on (the default):

* the pages go in batches (the glossary Request Merging count, else one page per request), each request
  ``build_prompt()`` over the batch's OCR text plus the batch's page images; a tall page is cut into chunks
  with the image translator's chunk planner (``image_translator.image_glossary_page_parts``), a short one is
  its own file; with SINGLE_API_IMAGE_CHUNKS=0 every image is its own request;
* the entries of every batch are merged, validated and de-duplicated by the existing code and saved as
  before (the CSV equals what the text-only path saves for the same entries);
* a stop ends the run before the next request; a model that cannot take images falls back to the
  text-only request with one warning line (no retry loop);
* the debug dump notes per request which pages and images were sent.

With the toggle off the request is today's single text-only one.

Run:
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests/test_manga_glossary_page_images.py
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
for _p in (str(TESTS_DIR), str(SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("GLOSSARION_HTTP_LOG", "0")

pytest.importorskip("numpy")
from PIL import Image, ImageDraw  # noqa: E402

from _headless_env import headless_owner  # noqa: E402

HEADER = "type\x1Fraw_name\x1Ftranslated_name\x1Fgender"
KIM = "character\x1F김철수\x1FKim Cheolsu\x1FMale"
SEOUL = "term\x1F서울역\x1FSeoul Station\x1F"
YUNA = "character\x1F유나\x1FYuna\x1FFemale"
CHUNK_ENV = {"IMAGE_CHUNK_HEIGHT": "1000", "IMAGE_SMART_CHUNKING": "0", "IMAGE_CHUNK_OVERLAP_PERCENT": "3",
             "IMAGE_CHUNK_MIN_OVERLAP_PIXELS": "80", "SINGLE_API_IMAGE_CHUNKS": "1"}
#: chunk heights of a 2500 px page with CHUNK_ENV (ImageChunkPlanner, overlap 80 px)
TALL_CHUNKS = [1000, 1000, 660]


def _write_page(path, height, width=400, fmt=None):
    image = Image.new("RGB", (width, height), (255, 255, 255))
    ImageDraw.Draw(image).rectangle((20, 20, 60, 60), fill=(0, 0, 0))
    image.save(path, format=fmt)
    return str(path)


def _decode(url):
    header, data = url.split(",", 1)
    raw = base64.b64decode(data)
    with Image.open(io.BytesIO(raw)) as image:
        return header, raw, image.size


class _Host:
    def __init__(self):
        self.logs = []

    def log(self, message, **kw):
        self.logs.append((kw.get("level", "info"), str(message)))

    def emit(self, kind, **data):
        pass


class _FakeClient:
    """Records every request; answers from a list (an exception in it is raised)."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.requests = []

    def send(self, messages, context=None):
        self.requests.append({"messages": messages, "context": context})
        answer = self.answers.pop(0) if self.answers else HEADER
        if isinstance(answer, BaseException):
            raise answer
        return answer, "stop"


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A headless manga runner whose glossary requests go to a fake client, writing only under tmp_path."""
    import TransateKRtoEN
    import manga_env
    import manga_files_core
    import manga_runner

    for name in ("manga_files_core", "manga_env", "manga_runner"):
        module = sys.modules[name]
        if callable(vars(module).get("_get_app_dir")):
            monkeypatch.setattr(module, "_get_app_dir", lambda: str(tmp_path / "_app_dir"))
    pages_dir = tmp_path / "Book"
    pages_dir.mkdir()
    pages = [
        _write_page(pages_dir / "p1.png", 600),
        _write_page(pages_dir / "p2.png", 2500),
        _write_page(pages_dir / "p3.jpg", 500, fmt="JPEG"),
    ]
    ocr_pages = [
        {"index": 1, "path": pages[0], "texts": ["김철수 씨!"]},
        {"index": 2, "path": pages[1], "texts": ["서울역으로 가자", "응"]},
        {"index": 3, "path": pages[2], "texts": ["유나야"]},
    ]
    state = {}

    def fake_send_with_interrupt(messages, client, temperature, max_tokens, stop_check_fn, context=None, **_kw):
        from extract_glossary_from_epub import build_prompt
        runner = state["runner"]
        client.requests_extra = getattr(client, "requests_extra", [])
        # what today's single text-only request is, under the env the generation step set
        client.requests_extra.append({
            "temperature": temperature, "max_tokens": max_tokens, "stop_check": callable(stop_check_fn),
            "expected_text_only": build_prompt(runner._build_manga_glossary_input(ocr_pages)),
            "batch_prompts": {
                tuple(page["index"] for page in pages_subset): build_prompt(runner._build_manga_glossary_input(pages_subset))
                for pages_subset in ([ocr_pages[0], ocr_pages[1]], [ocr_pages[2]], [ocr_pages[0]], [ocr_pages[1]])
            },
        })
        text, finish = client.send(messages, context=context)
        if state.get("stop_after") == len(client.requests):
            runner.stop_flag.set()
        return text, finish, None

    monkeypatch.setattr(TransateKRtoEN, "send_with_interrupt", fake_send_with_interrupt)

    def make(config, answers):
        host = _Host()
        owner_cm = headless_owner(tmp_path / "owner", monkeypatch, dict(
            {"model": "gpt-4o-mini", "api_key": "sk", "output_directory": str(tmp_path / "out")}, **config))
        owner = owner_cm.__enter__()
        state.setdefault("cms", []).append(owner_cm)
        for key, value in dict(CHUNK_ENV, **state.get("env", {})).items():  # after the owner's startup env
            os.environ[key] = value
        owner.client = _FakeClient(answers)
        runner = manga_runner.HeadlessMangaRunner(owner, host=host, files=pages)
        state["runner"] = runner
        return runner, owner.client, host

    state["make"] = make
    state["pages"] = pages
    state["ocr_pages"] = ocr_pages
    state["tmp"] = tmp_path
    yield state
    for cm in reversed(state.get("cms", [])):
        cm.__exit__(None, None, None)


def _saved_csv(runner):
    path = runner.manga_generated_glossary_path
    assert path and path.endswith(".csv") and os.path.exists(path)
    return Path(path).read_text(encoding="utf-8")


def test_default_is_on_and_page_images_go_in_batches_with_tall_pages_chunked(env):
    runner, client, host = env["make"](
        {"glossary_request_merging_enabled": True, "glossary_request_merge_count": 2,
         "manga_glossary_debug_ocr_text": True},
        [HEADER + "\n" + KIM + "\n" + SEOUL, HEADER + "\n" + KIM + "\n" + YUNA],
    )
    assert runner._manga_glossary_include_images_enabled() is True  # the default
    text = runner._generate_manga_glossary_from_ocr_pages(env["ocr_pages"])
    assert text

    assert len(client.requests) == 2
    first, second = client.requests
    assert all(r["context"] == "glossary" for r in client.requests)
    extra = client.requests_extra
    # request 1: pages 1-2 (merge count 2); its text is build_prompt over those pages' OCR text only
    system, user = extra[0]["batch_prompts"][(1, 2)]
    assert first["messages"][0] == {"role": "system", "content": system}
    content = first["messages"][1]["content"]
    assert first["messages"][1]["role"] == "user"
    assert content[0] == {"type": "text", "text": user}
    assert "PAGE 1 START" in user and "PAGE 2 START" in user and "PAGE 3" not in user
    images = [_decode(part["image_url"]["url"]) for part in content[1:]]
    assert all(part["type"] == "image_url" for part in content[1:])
    # page 1 whole (its own file bytes), page 2 in the translator's chunks
    assert images[0][0] == "data:image/png;base64" and images[0][1] == Path(env["pages"][0]).read_bytes()
    assert [size[1] for _h, _r, size in images[1:]] == TALL_CHUNKS
    # request 2: page 3, a JPEG sent as itself
    system3, user3 = extra[1]["batch_prompts"][(3,)]
    assert second["messages"][1]["content"][0] == {"type": "text", "text": user3}
    (header3, raw3, _size3), = [_decode(part["image_url"]["url"]) for part in second["messages"][1]["content"][1:]]
    assert header3 == "data:image/jpeg;base64" and raw3 == Path(env["pages"][2]).read_bytes()

    # merged across batches and de-duplicated: Kim once
    names = [e["raw_name"] for e in runner.manga_generated_glossary_entries]
    assert sorted(names) == sorted(["김철수", "서울역", "유나"])
    csv_on = _saved_csv(runner)

    # the debug dump notes what each request carried
    debug_dir = Path(runner._manga_glossary_output_dir()) / "debug"
    batches = json.loads((debug_dir / "image_batches.json").read_text(encoding="utf-8"))
    assert [b["request"] for b in batches] == [1, 2]
    assert [[p["index"] for p in b["pages"]] for b in batches] == [[1, 2], [3]]
    assert [i["image"] for i in batches[0]["images"]] == [
        "whole page (400x600)", "chunk 1/3 (y: 0-1000)", "chunk 2/3 (y: 920-1920)", "chunk 3/3 (y: 1840-2500)"]
    assert "base64" not in (debug_dir / "image_batches.json").read_text(encoding="utf-8")
    manifest = json.loads((debug_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["image_requests"] == 2 and manifest["images_sent"] == 5
    assert manifest["parsed_undeduped_entries"] == 4
    raw = (debug_dir / "raw_undeduped_glossary_output.txt").read_text(encoding="utf-8")
    assert "=== REQUEST 1/2 ===" in raw and "=== REQUEST 2/2 ===" in raw

    # the saved glossary looks exactly as the text-only path saves the same entries
    runner_off, client_off, _host = env["make"](
        {"manga_glossary_include_images": False},
        [HEADER + "\n" + KIM + "\n" + SEOUL + "\n" + YUNA],
    )
    assert runner_off._generate_manga_glossary_from_ocr_pages(env["ocr_pages"]) == text
    assert _saved_csv(runner_off) == csv_on


def test_toggle_off_sends_exactly_todays_single_text_only_request(env):
    runner, client, host = env["make"]({"manga_glossary_include_images": False},
                                       [HEADER + "\n" + KIM])
    assert runner._manga_glossary_include_images_enabled() is False
    assert runner._generate_manga_glossary_from_ocr_pages(env["ocr_pages"])
    assert len(client.requests) == 1
    system, user = client.requests_extra[0]["expected_text_only"]
    assert client.requests[0]["messages"] == [{"role": "system", "content": system},
                                              {"role": "user", "content": user}]
    assert client.requests[0]["context"] == "glossary" and client.requests_extra[0]["stop_check"]
    assert not any("image" in message.lower() for _level, message in host.logs if "Manga glossary" in message
                   and "request" in message)


def test_one_page_per_request_without_request_merging(env):
    runner, client, _host = env["make"]({}, [HEADER + "\n" + KIM, HEADER, HEADER + "\n" + YUNA])
    assert runner._generate_manga_glossary_from_ocr_pages(env["ocr_pages"])
    image_counts = [len(r["messages"][1]["content"]) - 1 for r in client.requests]
    assert image_counts == [1, 3, 1]


def test_without_single_api_chunks_each_chunk_of_a_tall_page_is_its_own_request(env):
    """SINGLE_API_IMAGE_CHUNKS=0 (the default: "Direct image legacy mode" off): the batch's short pages share
    a request, each chunk of a tall page goes alone with that page's OCR text."""
    env["env"] = {"SINGLE_API_IMAGE_CHUNKS": "0"}
    runner, client, _host = env["make"]({"glossary_request_merging_enabled": True, "glossary_request_merge_count": 5},
                                        [HEADER + "\n" + KIM] * 4)
    assert runner._generate_manga_glossary_from_ocr_pages(env["ocr_pages"])
    assert [len(r["messages"][1]["content"]) - 1 for r in client.requests] == [2, 1, 1, 1]
    first_user = client.requests[0]["messages"][1]["content"][0]["text"]
    assert "PAGE 1 START" in first_user and "PAGE 3 START" in first_user and "PAGE 2" not in first_user
    chunk_users = [r["messages"][1]["content"][0]["text"] for r in client.requests[1:4]]
    assert all("PAGE 2 START" in u and "PAGE 1" not in u for u in chunk_users)
    heights = [_decode(r["messages"][1]["content"][1]["image_url"]["url"])[2][1] for r in client.requests[1:4]]
    assert heights == TALL_CHUNKS
    assert [e["raw_name"] for e in runner.manga_generated_glossary_entries] == ["김철수"]


def test_stop_ends_the_run_before_the_next_batch(env):
    runner, client, _host = env["make"]({}, [HEADER + "\n" + KIM] * 3)
    env["stop_after"] = 1
    assert runner._generate_manga_glossary_from_ocr_pages(env["ocr_pages"]) is None
    assert len(client.requests) == 1
    assert not getattr(runner, "manga_generated_glossary_entries", None)


def test_a_model_without_image_input_falls_back_to_text_only_once(env):
    from unified_api_client import UnifiedClientError

    refusal = UnifiedClientError("Invalid content type. image_url is only supported by certain models.")
    runner, client, host = env["make"]({}, [refusal, HEADER + "\n" + KIM])
    assert runner._generate_manga_glossary_from_ocr_pages(env["ocr_pages"])
    assert len(client.requests) == 2
    assert isinstance(client.requests[0]["messages"][1]["content"], list)
    system, user = client.requests_extra[1]["expected_text_only"]
    assert client.requests[1]["messages"] == [{"role": "system", "content": system},
                                              {"role": "user", "content": user}]
    warnings = [m for level, m in host.logs if level == "warning"]
    assert len(warnings) == 1 and "cannot take images" in warnings[0]
    assert [e["raw_name"] for e in runner.manga_generated_glossary_entries] == ["김철수"]


def test_another_api_error_is_not_retried_without_images(env):
    from unified_api_client import UnifiedClientError

    runner, client, _host = env["make"]({}, [UnifiedClientError("429 Too Many Requests"), HEADER + "\n" + KIM])
    assert runner._generate_manga_glossary_from_ocr_pages(env["ocr_pages"]) is None
    assert len(client.requests) == 1


@pytest.mark.parametrize("message, expected", [
    ("Invalid content type. image_url is only supported by certain models.", True),
    ("This model does not support image input", True),
    ("Model doesn't support images", True),
    ("Image input is not supported for this model", True),
    ("No endpoints found that support image input", True),
    ("unsupported content type: image_url", True),
    ("429 Too Many Requests", False),
    ("Request timed out", False),
    ("The image is too large", False),
])
def test_image_input_unsupported_errors(message, expected):
    from image_translator import is_image_input_unsupported_error

    assert is_image_input_unsupported_error(Exception(message)) is expected


def test_unreadable_page_sends_its_text_only(tmp_path):
    from image_translator import image_glossary_page_parts

    logs = []
    assert image_glossary_page_parts(str(tmp_path / "missing.png"), log=logs.append) == []
    assert logs


def test_desktop_checkbox_row_sits_under_generate_glossary_and_defaults_on():
    import manga_settings_defaults as msd

    assert msd.MANGA_TOP_LEVEL_DEFAULTS["manga_glossary_include_images"] is True
    source = (SRC_DIR / "manga_integration.py").read_text(encoding="utf-8").replace("\r\n", "\n")
    generate = source.index("manga_glossary_layout.addWidget(manga_glossary_toggle_frame)\n")
    row = source.index('self._create_styled_checkbox("Include page images in glossary generation")')
    compress = source.index("manga_glossary_compress_frame = QWidget()")
    assert generate < row < compress
    block = source[generate:compress]
    assert "setContentsMargins(20, 0, 0, 0)" in block and "setToolTip(" in block
    assert "self.main_gui.config.get('manga_glossary_include_images', True)" in block
    assert "Include page images in glossary generation" in source[source.index('"Manga Glossary Workflow",'):generate]
