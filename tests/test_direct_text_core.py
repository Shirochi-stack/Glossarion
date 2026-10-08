"""Direct Text shared core (Glossarion mobile rewrite, milestone U3 step 3).

The desktop Direct Text dialog (``translator_gui._InputOutputDialog``) kept its chat
persistence and its log-stream model inside Qt handlers. Both moved verbatim into
GUI-free mixins that the dialog now inherits first:

* ``direct_text_store.ChatStoreMixin``: ``direct_text_chats.json`` v2 (load / save /
  externalised bodies in ``Chat Messages/``), output folders (``Direct Text/<chat>``,
  ``Direct Text N``), attachment output persist / copy / glossary sync, attachment
  workspaces + Migrate (path rewrite), response edits; ``ChatStore`` is the GUI-free host
  (mobile ``ChatStoreAdapter``), ``apply_direct_text_run_environment`` the run env step.
* ``direct_text_stream.DirectTextStreamMixin``: the log-line classifier, request segments
  (spine order, payload markers, phases, token counting), commits and the run finish;
  ``DirectTextStream`` is the GUI-free host (mobile ``RunStream`` / JobService), the
  extraction report / attachment actions are data builders + the persisted card format.

Tiers checked here:

* I  - import hygiene: both modules import with PySide6 blocked; Python 3.10 syntax.
* V  - verbatim: every moved member equals the dialog member at ``BASE_SHA`` (the U2
  commit) after exactly the documented edits; the dialog no longer defines them; MRO.
* P  - dialog parity: a child process builds the REAL dialog offscreen
  (``QT_QPA_PLATFORM=offscreen``) from ``git show BASE_SHA:src/translator_gui.py`` and a
  second one from the working tree; both replay the same scenarios (history load /
  render / save on a copy of ``src/direct_text_chats.json`` and on a synthetic history,
  recorded log streams, full sends through ``_start_translation`` -> stream ->
  ``_finish_translation``, Migrate, response edits, pure helpers) and every observation
  (state, rendered transcript HTML, saved JSON, file trees) must be equal. The synthetic
  observations of the BASE_SHA dialog are also kept as a local golden
  (``tests/parity/golden/<sha12>/direct_text_dialog.json``, ``--capture-golden``).
* H  - GUI-free hosts: ``ChatStore`` / ``DirectTextStream`` reproduce the legacy dialog's
  observations for the same inputs (load/save, stream segments/messages, finish) in a child
  that never imports Qt.
* U  - unit tests of the public host API used by the mobile chat (ChatStoreBinding /
  RunStream / JobService entry points, ``finish_run``, Migrate, ``prepare_direct_text_input``,
  the shared run environment on a real HeadlessOwner, the report builders vs BASE_SHA).

Tiers V and U run without PySide6 (3.10 / 3.13 venvs); P and H need PySide6 and git.

Run (repository root)::

    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests/test_direct_text_core.py
    python tests/test_direct_text_core.py --capture-golden      # refresh the local golden

The probe children are this file run as a script (``--probe`` / ``--headless``).
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
TESTS = REPO_ROOT / "tests"
for _p in (str(TESTS), str(SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

#: Oracle: the U2 commit (the dialog is unchanged there since U0).
BASE_SHA = "1719fb59dcab56953ca0f32392d4f2159c703c2a"

TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAFklEQVR4nGP4z8DAwMDAxMDAwMDAAAANHQEDasKb6QAAAABJRU5ErkJggg=="
)

# ---------------------------------------------------------------------------
# Probe: fixtures, scenarios and normalisation (shared by parent and children)
# ---------------------------------------------------------------------------

SYNTH_FOLDER = "Synth chat - 20250101_101010_deadbeef"
OTHER_FOLDER = "Other - 20250101_000000_cafebabe"


def _write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, bytes):
        path.write_bytes(data)
    else:
        path.write_text(data, encoding="utf-8", newline="")


def _synthetic_history():
    synth = f"Direct Text/{SYNTH_FOLDER}"
    return {
        "version": 2,
        "current_chat_id": "7",
        "sessions": [
            {
                "id": 3,
                "title": "Synth   chat",
                "messages": [
                    ["user", "Hello"],
                    ["assistant", "", "", "Token summary  ·  Thinking 3  ·  Text 5", f"<HIST>/{synth}",
                     "Request 1", {"content_path": f"{synth}/Chat Messages/000002-response.md",
                                   "content_text_path": f"{synth}/Chat Messages/000002-response.txt",
                                   "content_html_path": f"{synth}/Chat Messages/000002-response.html",
                                   "content_xhtml_path": f"{synth}/Chat Messages/000002-response.xhtml",
                                   "thinking_path": f"{synth}/Chat Messages/000002-thinking.md",
                                   "content_chars": "17", "thinking_chars": 9,
                                   "created_at": "2025-01-02T03:04:05+00:00"}],
                    # cwd-relative marker (cwd = <sandbox>/cwd): the body length must not depend on the sandbox
                    ["assistant", "Inline **content**\n\n[GENERATED_IMAGE:../fx/hist_synth/images/gen.png]",
                     "inline thinking",
                     "Processing", "", "Request 2", {"created_at": "2024-12-31T23:59:59Z", "bogus": 1}],
                    ["user_file", "book.epub", "<HIST>/sources/book.epub", "2048", "Translate politely", "SYSTEM"],
                    ["assistant", "<h1>Chapter 1</h1>\n<p>Chapter text <b>bold</b></p>", "", "Completed",
                     f"<HIST>/{synth}/Attachments/book", "Chapter 1 · ch001.xhtml · Request 3",
                     {"created_at": "2025-06-01T12:00:00+02:00"}],
                    ["assistant", "## Attachment ready\n\n**Source:** book.epub", "", "Completed",
                     f"<HIST>/{synth}/Attachments/book", "Attachment actions"],
                    ["assistant", "", "", "Processing", "", "Request 4",
                     {"content_path": "Direct Text/nope/000009-response.md", "created_at": "bad-date"}],
                    ["user"], "not a list", ["weird", "x"], ["assistant", None, None],
                    ["user_file", "notes.md", "<HIST>/sources/notes.md", "bad", "", "narrator"],
                ],
                "draft": "draft text",
                "attachment": {"path": "<HIST>/sources/notes.md", "size": "x"},
                "output_folder": f"<HIST>/{synth}",
                "output_folder_name": SYNTH_FOLDER,
                "next_output_index": "2",
                "expanded": [1, 2, -1, "3"],
            },
            {"id": 3, "title": "   Many    spaces " + "x" * 200, "messages": [], "expanded": "bad"},
            {"id": "abc", "title": None, "messages": [["user", "orphan id"]], "next_output_index": "zz"},
            5,
            {
                "id": 7,
                "title": "Other",
                "messages": [
                    ["user", "Translate this please"],
                    ["assistant", "# Heading\n\nTranslated *text*.", "Some thinking\nmore", "Token summary  ·  "
                     "Thinking 4  ·  Text 6", f"<HIST>/Direct Text/{OTHER_FOLDER}", "Request 1",
                     {"created_at": "2025-03-04T05:06:07+00:00"}],
                ],
                "draft": "",
                "attachment": None,
                "output_folder": f"<HIST>/Direct Text/{OTHER_FOLDER}/Attachments/x",
                "output_folder_name": OTHER_FOLDER,
                "next_output_index": 1,
                "expanded": [1],
            },
        ],
    }


def build_fixtures(root):
    """Inputs shared by every child (copied, then ``<HIST>`` placeholders resolved)."""
    root = Path(root)
    hist = root / "hist_synth"
    _write(hist / "direct_text_chats.json", json.dumps(_synthetic_history(), ensure_ascii=False, indent=1))
    synth = hist / "Direct Text" / SYNTH_FOLDER
    _write(synth / "Chat Messages" / "000002-response.md", "Saved **response**")
    _write(synth / "Chat Messages" / "000002-response.txt", "Saved response")
    _write(synth / "Chat Messages" / "000002-response.html", "<p>Saved</p>")
    _write(synth / "Chat Messages" / "000002-response.xhtml", "<p>Saved</p>")
    _write(synth / "Chat Messages" / "000002-thinking.md", "Thinking!")
    _write(synth / "Direct Text 1.txt", "first output")
    book = synth / "Attachments" / "book"
    _write(book / "book.epub", b"PK\x03\x04new-epub")
    _write(book / "book_old.epub", b"PK\x03\x04old-epub")
    _write(book / "book.pdf", b"%PDF-1.4 fake")
    _write(book / "nested" / "inner.pdf", b"%PDF nested")
    _write(book / "glossary.json", "{}")
    _write(book / "response_ch001.html", "<html><body><h1>Chapter 1</h1><p>Chapter text <b>bold</b></p></body></html>")
    _write(book / "response_ch000_a.html", "<html><body><p>Cover page</p></body></html>")
    _write(book / "response_ch000_b.html", "<html><body><p>Information page about the book</p></body></html>")
    _write(book / "translation_progress.json", json.dumps({"chapters": {
        "a": {"actual_num": 1, "output_file": "response_ch001.html"},
        "b": {"actual_num": 0, "output_file": "response_ch000_a.html"},
        "c": {"actual_num": 0, "output_file": "response_ch000_b.html"},
        "d": {"actual_num": 2, "output_file": "../escape.html"},
    }}))
    os.utime(book / "book_old.epub", (1_600_000_000, 1_600_000_000))
    os.utime(book / "book.epub", (1_700_000_000, 1_700_000_000))
    _write(hist / "Direct Text" / OTHER_FOLDER / "Attachments" / "x" / "out.txt", "x")
    _write(hist / "images" / "gen.png", TINY_PNG)
    _write(hist / "sources" / "book.epub", b"PK\x03\x04source-epub")
    _write(hist / "sources" / "notes.md", "# Notes\n\nSome *markdown* to translate.\n")
    # migration destination conflicts (output root = <HIST>)
    _write(hist / "book" / "book.epub", b"PK\x03\x04dest-epub")
    _write(hist / "book" / "keep_me.txt", "kept")

    # Copy of the developer's real history (gitignored; skipped when absent).
    real_json = SRC / "direct_text_chats.json"
    if real_json.is_file():
        real = root / "hist_real"
        text = real_json.read_text(encoding="utf-8")
        src_prefix = str(SRC)
        text = text.replace(json.dumps(src_prefix)[1:-1], "<HIST>")
        _write(real / "direct_text_chats.json", text)
        if (SRC / "Direct Text").is_dir():
            shutil.copytree(SRC / "Direct Text", real / "Direct Text")

    # Inputs for full sends (attachments) and run trees the "pipeline" leaves behind.
    sources = root / "sources"
    _write(sources / "book.epub", b"PK\x03\x04epub-bytes")
    _write(sources / "notes.md", "# Notes\n\nTranslate *this* markdown.\n")
    _write(sources / "pic.png", TINY_PNG)
    _write(sources / "subs.srt", "1\n00:00:01,000 --> 00:00:02,000\n안녕\n")
    _write(sources / "manual_gloss.csv", "type,raw_name,translated_name\ncharacter,김,Kim\n")
    tree = root / "runs" / "book"
    _write(tree / "book.epub", b"PK\x03\x04compiled")
    _write(tree / "book.pdf", b"%PDF compiled")
    _write(tree / "translation_progress.json", json.dumps({"chapters": {"1": {"actual_num": 1, "output_file": "response_ch001.html"}}}))
    _write(tree / "response_ch001.html", "<html><body><p>Chapter one translated</p></body></html>")
    _write(tree / "glossary.json", "[]")
    _write(tree / "extraction_report.txt", (
        "EXTRACTION REPORT\n\nPOTENTIAL ISSUES:\n  • 2 chapters contain only images\n  • Missing NCX\n"
        "  • Encoding guessed\n  • Odd spine\n  • Fifth issue\n"))
    _write(tree / "metadata.json", json.dumps({
        "extraction_mode": "enhanced", "detected_language": "korean", "chapter_count": 4,
        "extracted_resources": {"images": ["a.png", "b.png"], "css": 1, "fonts": {"x": 1}},
    }))
    _write(tree / "chapters_full.json", json.dumps([
        {"body": "<p>x</p>", "file_size": 900, "has_images": False},
        {"body": "<p>y</p>", "file_size": 700, "has_images": True},
        {"body": None, "file_size": 10, "has_images": True, "is_image_only": True, "filename": "cover.xhtml"},
        {"body": "<p>z</p>", "file_size": 20, "has_images": True, "is_image_only": True, "title": "Illustration"},
    ]))
    _write(tree / "translated_headers.txt", "Chapter 1:\n  Original: 하나\n  Translated: One\nChapter 2:\n  Translated: Two\n")
    _write(tree / "Payloads" / "chunks" / "book_translated.txt", "chunk")


# Recorded log streams: (message, callback thread) in arrival order.
STREAM_TEXT = [
    ("🚀 [Thread-2 (api_call)] Sending API call now (Chapter 1, chunk 1/1)", "Thread-2 (api_call)"),
    ("🛰️ [gemini-native] Streaming ON (env=1)", "Thread-2 (api_call)"),
    ("🧠 [gemini-native] Thinking...", "Thread-2 (api_call)"),
    ("    The user wants a translation.", "Thread-2 (api_call)"),
    ("    Keep honorifics.\n    Second line", "Thread-2 (api_call)"),
    ("🧠 [gemini-native] Thinking complete.", "Thread-2 (api_call)"),
    ("📡 [gemini-native] Text streaming...", "Thread-2 (api_call)"),
    ("# Title", "Thread-2 (api_call)"),
    ("Hello **world**. 안녕하세요", "Thread-2 (api_call)"),
    ("", "Thread-2 (api_call)"),
    ("<p>Second paragraph</p>", "Thread-2 (api_call)"),
    ("🛰️ [gemini-native] Stream finished in 2.1s (42 chunks)", "Thread-2 (api_call)"),
    ("✅ Received translation from API", "Thread-2 (api_call)"),
    ("📂 Payloads directory: C:/x/Payloads", "MainThread"),
    ("Translation preview: <html><body>x</body></html>", "MainThread"),
    ("<html><body>preview</body></html>", "MainThread"),
    ("TRANSLATION_COMPLETE_SIGNAL", "MainThread"),
]

STREAM_EPUB = [
    ("[spine-order:2] Chapter 2 (chunk 1/2) · ch002.xhtml Direct Text dispatch", "Thread-5 (api_call)"),
    ("[spine-order:1] Chapter 1 · ch001.xhtml Direct Text dispatch", "Thread-6 (api_call)"),
    ("🚀 [Thread-5 (api_call)] Sending API call now: Chapter 2 (chunk 1/2) [File: OEBPS/Text/ch002.xhtml]",
     "TranslationWorker_1"),
    ("⏳ AuthND: NVIDIA queue / prefill — response headers received in 12.4s; waiting for first token",
     "AuthNDTransport[Thread-5 (api_call)]"),
    ("⏳ AuthND: NVIDIA queue / prefill — waiting", "AuthNDTransport[Thread-5 (api_call)]"),
    ("🧠 [authnd] Thinking...", "AuthNDTransport[Thread-5 (api_call)]"),
    ("    reasoning about chapter two", "AuthNDTransport[Thread-5 (api_call)]"),
    ("📡 AuthND: Text streaming...", "AuthNDTransport[Thread-5 (api_call)]"),
    ("<h1>Chapter 2</h1>", "AuthNDTransport[Thread-5 (api_call)]"),
    ("<p>Text of chapter two part one.</p>⏹️ Translation stopped by user", "AuthNDTransport[Thread-5 (api_call)]"),
    ("📡 AuthND: Stream finished in 9.0s", "AuthNDTransport[Thread-5 (api_call)]"),
    ("🚀 [Thread-6 (api_call)] Sending API call now: Chapter 1 · ch001.xhtml", "Thread-6 (api_call)"),
    ("API call in progress (Chapter 1)", "Thread-6 (api_call)"),
    ("📡 [Thread-6 (api_call)] Text streaming...", "Thread-6 (api_call)"),
    ("Chapter one text line", "Thread-6 (api_call)"),
    ("📊 not model text", "Thread-6 (api_call)"),
    ("📥 Received chapter 1 response", "Thread-6 (api_call)"),
    ('[DIRECT_TEXT_RESPONSE_PAYLOAD] {"label": "Chapter 2 (chunk 2/2) · ch002.xhtml", "content": "<p>Part two.</p>", '
     '"thinking": "think2", "order": 2, "request_number": 7, "source_thread": "Thread-7 (api_call)"}',
     "Thread-7 (api_call)"),
    ('[DIRECT_TEXT_RESPONSE_PAYLOAD] {"label": "Request 3", "content": "generic answer", "order": 99}', "Thread-8"),
    ('[DIRECT_TEXT_RESPONSE_PAYLOAD] not json', "Thread-8"),
    ('[DIRECT_TEXT_RESPONSE_PAYLOAD] {"label": "", "content": ""}', "Thread-8"),
    ("🚀 Sending API call now: Header batch 1/2", "Thread-9 (api_call)"),
    ("📡 [Thread-9 (api_call)] Text streaming...", "Thread-9 (api_call)"),
    ('{"1": "One"}', "Thread-9 (api_call)"),
    ("🛰️ [gemini-grpc] Stream finished in 1.0s", "Thread-9 (api_call)"),
    ("API call in progress Request (metadata)", "Thread-10 (api_call)"),
    ("🚀 Sending API call now: Merged 3-4 · chapter0003.xhtml", "Thread-11 (api_call)"),
    ("📥 Received merged response for Merged 3-4 response", "Thread-11 (api_call)"),
    ("📥 Received section 5 response", "Thread-12 (api_call)"),
    ("Traceback (most recent call last):", "MainThread"),
    ("✅ Text file translation complete", "MainThread"),
]

STREAM_GLOSSARY = [
    ('[DIRECT_TEXT_GLOSSARY_STREAM_START] {"label": "Glossary request 1/2", "source_thread": "Thread-21 (gloss)"}',
     "GlossaryWorker"),
    ("🧠 [openai] Thinking...", "Thread-21 (gloss)"),
    ("    gloss reasoning", "Thread-21 (gloss)"),
    ("🧠 [openai] Thinking complete.", "Thread-21 (gloss)"),
    ("type,raw_name,translated_name", "Thread-21 (gloss)"),
    ("character,김상현,Kim Sang-hyun", "Thread-21 (gloss)"),
    ("✅ Stream complete", "Thread-21 (gloss)"),
    ('[DIRECT_TEXT_GLOSSARY_STREAM_START] {"label": "Glossary request 2/2"}', "Thread-22 (gloss)"),
    ("#raw line that looks like status", "Thread-22 (gloss)"),
    ("- item", "Thread-22 (gloss)"),
    ('[DIRECT_TEXT_RESPONSE_PAYLOAD] {"label": "Glossary request 1/2", "content": "type,raw_name\\ncharacter,김", '
     '"thinking": "full thinking", "source_thread": "Thread-21 (gloss)"}', "Thread-21 (gloss)"),
    ("📤 [Thread-34 (api_call)] Chapter 25 (translation) Preparing Ollama request", "Thread-34 (api_call)"),
    ("🧠 [Ollama] Thinking...", "Thread-34 (api_call)"),
    ("    Checking wording", "Thread-34 (api_call)"),
    ("🧠 [Ollama] Thinking complete.", "Thread-34 (api_call)"),
    ("📡 [Ollama] Text streaming...", "Thread-34 (api_call)"),
    ("Translated sentence", "Thread-34 (api_call)"),
    ("Ollama: Ollama stream complete (3 generated tokens).", "Thread-34 (api_call)"),
]

STREAMS = {"text": STREAM_TEXT, "epub": STREAM_EPUB, "glossary": STREAM_GLOSSARY}

#: Pure helper corpus (static/class methods called on the dialog class).
HELPER_LINES = [line for line, _thread in STREAM_TEXT + STREAM_EPUB + STREAM_GLOSSARY] + [
    "Chapter 12 (chunk 2/3) [File: OEBPS/Text/ch012.xhtml] sending api call now",
    "Section 4.5 chunk 1/2", "merged 7 request", "[spine-order:x] bogus", "Header batch 2/3 · x",
    "toc batch 1 / 4", "request (metadata) whatever", "Merged chapters 1-3 response",
    "random text", "", "   ", "[DEBUG] x", "TRANSLATION_COMPLETE_SIGNAL", "Saved file", "chapter -3 x",
]
MARKUP_CORPUS = [
    "Plain **markdown** with a [link](http://x).\n\n- a\n- b",
    "```html\n<html><head><style>p{}</style></head><body><p onclick='x()' style='color:red'>Hi</p></body></html>\n```",
    "&lt;p&gt;escaped&lt;/p&gt; and &lt;script&gt;bad&lt;/script&gt;",
    "<div bgcolor='red'><script>alert(1)</script><p>ok</p></div>",
    "<!DOCTYPE html><html><body>line one\nline two</body></html>",
    "| a | b |\n|---|---|\n| 1 | 2 |",
    "text\nwith\nnewlines",
    "",
]


def _probe_normalizer(sandbox):
    roots = set()
    for base in (str(sandbox), os.path.realpath(str(sandbox))):
        for variant in (base, base.replace("\\", "/"), base.replace("\\", "\\\\")):
            roots.add(variant)
            roots.add(variant.lower())
    roots = sorted(roots, key=len, reverse=True)
    patterns = [
        (re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:[+-]\d{2}:\d{2}|Z)?"), "<TS>"),
        (re.compile(r"\d{8}_\d{6}_[0-9a-f]{8}"), "<STAMP>"),
        (re.compile(r"(glossarion_(?:input_output|direct_text_chat)_)[A-Za-z0-9_]{8}"), r"\1<R>"),
        (re.compile(r"·\s*\d{2}:\d{2}<"), "· <HM><"),
        (re.compile(r"Elapsed:\*\* \d+m \d+s"), "Elapsed:** <ELAPSED>"),
    ]

    def norm_text(value):
        for root in roots:
            if root and root in value:
                value = value.replace(root, "<SB>")
        for pattern, repl in patterns:
            value = pattern.sub(repl, value)
        return value

    def norm(value):
        if isinstance(value, str):
            return norm_text(value)
        if isinstance(value, dict):
            return {norm_text(str(k)): norm(v) for k, v in value.items()}
        if isinstance(value, (set, frozenset)):
            return sorted((norm(v) for v in value), key=repr)
        if isinstance(value, (list, tuple)):
            return [norm(v) for v in value]
        if isinstance(value, (bytes, bytearray)):
            return "bytes:" + hashlib.sha1(bytes(value)).hexdigest()[:12]
        if value is None or isinstance(value, (bool, int, float)):
            return value
        return norm_text(repr(value))

    return norm


def _tree(folder, norm):
    """Snapshot of a folder: normalised relative path -> text (normalised) or bytes hash."""
    folder = Path(folder)
    out = {}
    if not folder.exists():
        return None
    for path in sorted(folder.rglob("*")):
        rel = norm(path.relative_to(folder).as_posix())
        if path.is_dir():
            out[rel + "/"] = "dir"
            continue
        data = path.read_bytes()
        if path.suffix.lower() == ".json":
            try:
                out[rel] = norm(json.loads(data.decode("utf-8")))
                continue
            except Exception:
                pass
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            out[rel] = "bytes:" + hashlib.sha1(data).hexdigest()[:12]
            continue
        out[rel] = norm(text) if len(text) <= 4000 else "text:" + hashlib.sha1(norm(text).encode()).hexdigest()[:16]
    return out


def _resolve_placeholders(folder):
    hist = str(folder)
    for path in Path(folder).rglob("*.json"):
        text = path.read_text(encoding="utf-8")
        if "<HIST>" in text:
            path.write_text(text.replace("<HIST>", json.dumps(hist)[1:-1]), encoding="utf-8")


# ---------------------------------------------------------------------------
# Probe children: the real dialog offscreen (legacy / current) and the GUI-free hosts
# ---------------------------------------------------------------------------

GEN_MARKER = "[GENERATED_IMAGE:<TEMP>/gen/response_1_generated.png]"

#: Full sends: composer state, what the "pipeline" leaves in the temp root, the log stream.
SEND_CASES = [
    {"name": "text", "text": "Translate me please", "translated": "Translated text output",
     "stream": STREAM_TEXT},
    {"name": "text_stream_only", "text": "Stream only", "stream": STREAM_TEXT},
    {"name": "text_nothing", "text": "Nothing comes back", "stream": []},
    {"name": "epub", "attachment": "book.epub", "text": "Keep honorifics", "tree": "runs/book",
     "stream": STREAM_EPUB, "manual_source": {"kind": "path", "path": "sources/manual_gloss.csv"},
     "glossary": "manual"},
    {"name": "epub_no_glossary", "attachment": "book.epub", "tree": "runs/book", "glossary": "no_glossary",
     "stream": STREAM_EPUB[:12], "translator": {"_direct_text_force_no_glossary": True}},
    {"name": "markdown_pasted_glossary", "attachment": "notes.md", "glossary": "manual", "skip_thinking": True,
     "manual_source": {"kind": "content", "content": "type,raw_name\ncharacter,A", "extension": ".CSV"},
     "translated": "Notes translated", "stream": STREAM_GLOSSARY},
    {"name": "image_mode", "text": "Draw a cat", "mode": "image", "media": "gen/response_1_generated.png",
     "stream": [("📡 [Thread-3 (api_call)] Text streaming...", "Thread-3 (api_call)"),
                (GEN_MARKER, "Thread-3 (api_call)")]},
    {"name": "picture_vision", "attachment": "pic.png", "translated": "OCR text", "stream": STREAM_TEXT[:9],
     "run_mode": "vision"},  # the dialog switches to Vision for an image attachment
    {"name": "subtitle", "attachment": "subs.srt", "glossary": "none",
     "stream": [("📡 [Thread-4 (api_call)] Text streaming...", "Thread-4 (api_call)"), ("Hello", "Thread-4 (api_call)")]},
]

MIGRATE_VARIANTS = ("no_conflict", "merge", "cancel", "unmanaged", "active", "file_conflict")
EDIT_CALLS = ((4, "<h1>Chapter 1</h1>\n<p>Edited chapter</p>", {}),
              (2, "Edited **markdown**", {"text_source": "Edited plain"}),
              (0, "not assistant", {}), (99, "x", {}))
GATE_LINES = [("🚀 [Thread-90 (api_call)] Sending API call now: Chapter 9", "Thread-90 (api_call)"),
              ("📡 [Thread-90 (api_call)] Text streaming...", "Thread-90 (api_call)"),
              ("after gate", "Thread-90 (api_call)")]
COMPLETION = [("assistant", "done", "", "Completed", "", "Extraction report")]


def _child_setup(fixtures, sandbox):
    """Sandbox + scrubbed process state shared by every probe child; returns the normaliser."""
    sandbox = Path(sandbox)
    if sandbox.exists():
        shutil.rmtree(sandbox)
    shutil.copytree(fixtures, sandbox / "fx")
    for sub in ("hist_synth", "hist_real"):
        if (sandbox / "fx" / sub).exists():
            _resolve_placeholders(sandbox / "fx" / sub)
    for name in ("app", "tmp", "home", "cwd"):
        (sandbox / name).mkdir(parents=True, exist_ok=True)
    tiktoken_cache = os.path.join(tempfile.gettempdir(), "data-gym-cache")
    if os.path.isdir(tiktoken_cache):
        os.environ.setdefault("TIKTOKEN_CACHE_DIR", tiktoken_cache)
    for key in ("OUTPUT_DIRECTORY", "OUTPUT_DIR", "EPUB_OUTPUT_DIR", "MANUAL_GLOSSARY", "MODEL"):
        os.environ.pop(key, None)
    os.environ.update({
        "QT_QPA_PLATFORM": "offscreen",
        "GLOSSARION_APP_DIR": str(sandbox / "app"),
        "TEMP": str(sandbox / "tmp"), "TMP": str(sandbox / "tmp"),
        "HOME": str(sandbox / "home"), "USERPROFILE": str(sandbox / "home"),
        "GLOSSARION_DIRECT_TEXT_HISTORY": str(sandbox / "boot" / "direct_text_chats.json"),
        "GLOSSARION_HTTP_LOG": "0",
    })
    tempfile.tempdir = None
    # Deterministic mkdtemp names: run paths end up in streamed text that is token-counted.
    import random

    names = tempfile._get_candidate_names()
    names._rng = random.Random(20261005)
    names._rng_pid = os.getpid()
    # Run roots numbered per prefix, so atomic writes elsewhere cannot shift the sequence.
    real_mkdtemp = tempfile.mkdtemp
    counters = {}

    def numbered_mkdtemp(suffix=None, prefix=None, dir=None):
        if not str(prefix or "").startswith("glossarion_"):
            return real_mkdtemp(suffix, prefix, dir)
        counters[prefix] = counters.get(prefix, 0) + 1
        path = os.path.join(dir or tempfile.gettempdir(), f"{prefix}{counters[prefix]:08d}{suffix or ''}")
        os.makedirs(path)
        return path

    tempfile.mkdtemp = numbered_mkdtemp
    os.chdir(sandbox / "cwd")
    return _probe_normalizer(sandbox)


def _work_copy(sandbox, name):
    """A copy of the synthetic history whose absolute paths point at the copy."""
    work = Path(sandbox) / name
    shutil.copytree(Path(sandbox) / "fx" / "hist_synth", work)
    text = (work / "direct_text_chats.json").read_text(encoding="utf-8")
    old = json.dumps(str(Path(sandbox) / "fx" / "hist_synth"))[1:-1]
    (work / "direct_text_chats.json").write_text(text.replace(old, json.dumps(str(work))[1:-1]), encoding="utf-8")
    return work


def _env_keys():
    from direct_text_store import ChatStoreMixin
    from run_env import FORCED_STREAM_ENV_KEYS

    return sorted(set(ChatStoreMixin._OUTPUT_ENV_KEYS) | set(FORCED_STREAM_ENV_KEYS)
                  | set(ChatStoreMixin._DIRECT_TEXT_ENV_KEYS))


def _probe_child(tg_dir, fixtures, sandbox, out_path):
    """Run every scenario against the real ``_InputOutputDialog`` from *tg_dir* (or src/)."""
    norm = _child_setup(fixtures, sandbox)
    sandbox = Path(sandbox)
    if tg_dir:
        sys.path.insert(0, str(tg_dir))

    import translator_gui as tg
    from PySide6.QtWidgets import QApplication, QWidget

    app = QApplication.instance() or QApplication([])
    tg_path = norm(os.path.abspath(tg.__file__))
    obs = {"translator_gui": tg_path.replace(norm(str(tg_dir)), "<TG>") if tg_dir else tg_path}
    env_keys = sorted(set(tg._InputOutputDialog._OUTPUT_ENV_KEYS) | set(tg._InputOutputDialog._FORCED_STREAM_ENV_KEYS)
                      | set(tg._InputOutputDialog._DIRECT_TEXT_ENV_KEYS))

    class _Plain:
        def __init__(self, text=""):
            self._text = text

        def toPlainText(self):
            return self._text

    class FakeTranslator(QWidget):
        """The parts of TranslatorGUI the dialog touches; run_translation_thread records."""

        _create_styled_checkbox = tg.TranslatorGUI._create_styled_checkbox

        def __init__(self):
            super().__init__()
            self.config = {}
            self.model_var = "gpt-4o"
            self.selected_files = ["main.epub"]
            self.manual_glossary_path = ""
            self.prompt_text = _Plain("SYS PROMPT")
            self.listeners = []
            self.records = []
            self.translation_thread = None

        def _get_output_mode(self):
            return "text"

        def save_config(self, show_message=True):
            self.records.append(("save_config", bool(show_message)))

        def add_log_listener(self, callback):
            self.listeners.append(callback)

        def remove_log_listener(self, callback):
            self.records.append(("remove_log_listener",))

        def _is_any_process_running(self):
            return False

        def append_log(self, *args, **kwargs):
            pass

        def stop_translation(self):
            self.records.append(("stop_translation",))

        def _env_view(self):
            return {k: os.environ.get(k) for k in env_keys}

        def _apply_forced_streaming_environment(self):
            self.records.append(("forced_streaming", self._env_view()))

        def _apply_direct_text_runtime_environment(self):
            self.records.append(("direct_text_runtime", self._env_view()))

        def run_translation_thread(self):
            import threading

            attrs = {k: v for k, v in vars(self).items()
                     if k.startswith("_direct_text") or k in ("selected_files", "_force_stream_all",
                                                              "_input_output_run_active", "manual_glossary_path",
                                                              "manual_glossary_map", "_translation_run_output_mode_override")}
            self.records.append(("run_translation_thread", self._env_view(), attrs))
            self.translation_thread = threading.Thread(target=lambda: None)

    translator = FakeTranslator()
    dialog = tg._InputOutputDialog(translator)

    def html():
        dialog._render_output()
        return dialog.output_box.toHtml()

    def run(label, fn):
        print(f"[probe] {label}", file=sys.stderr, flush=True)  # last line before a native crash
        try:
            obs[label] = norm(fn())
        except Exception as exc:  # recorded, compared like any observation
            obs[label] = {"exception": f"{type(exc).__name__}: {norm(str(exc))}"}

    def use_history(path):
        dialog._chat_history_path = str(path)
        dialog._chat_sessions, current_id = dialog._load_chat_history()
        dialog._chat_session_counter = max((int(s.get("id", 0)) for s in dialog._chat_sessions), default=0)
        dialog._current_chat_index = next(
            (i for i, s in enumerate(dialog._chat_sessions) if s.get("id") == current_id), 0)
        dialog._chat_messages = dialog._chat_sessions[dialog._current_chat_index]["messages"]
        dialog._message_text_cache.clear()
        dialog._rendered_message_cache.clear()
        return current_id

    def message_views(session_messages):
        views = []
        for index, message in enumerate(session_messages):
            if not message or message[0] != "assistant":
                views.append(None)
                continue
            views.append({
                "content": dialog._assistant_message_text(message, "content", index),
                "thinking": dialog._assistant_message_text(message, "thinking", index),
                "chars": [dialog._assistant_message_char_count(message, k) for k in ("content", "thinking")],
                "has": [dialog._assistant_message_has_text(message, k) for k in ("content", "thinking")],
                "media": dialog._assistant_generated_media(message, message[1]),
                "ts": dialog._assistant_timestamp_label(message),
                "attachment_src": dialog._assistant_source_is_attachment(index),
                "artifact": dialog._response_output_artifact_path(message, index),
            })
        return views

    def history_scenario(hist_dir):
        out = {}
        current_id = use_history(Path(hist_dir) / "direct_text_chats.json")
        out["load"] = norm({"sessions": dialog._chat_sessions, "current": current_id})
        sessions = []
        for index, session in enumerate(dialog._chat_sessions):
            dialog._load_chat_session(index)
            view = {
                "render": html(),
                "status": dialog.status_label.text(),
                "pending_attachment": dialog._pending_attachment,
                "messages": message_views(session["messages"]),
                "next_request": dialog._next_conversation_request_number(),
                "managed": dialog._managed_conversation_output_folder(session),
                "attachments": dialog._conversation_attachment_folders(session),
            }
            try:
                view["validated"] = dialog._validated_chat_output_folder(session)
            except Exception as exc:
                view["validated"] = f"{type(exc).__name__}: {exc}"
            for folder in view["attachments"]:
                view.setdefault("workspaces", []).append({
                    "managed": dialog._is_managed_attachment_workspace(session, folder),
                    "epub": dialog._attachment_compiled_epub_path(folder),
                    "preferred": dialog._preferred_attachment_compiled_documents(folder),
                })
            sessions.append(norm(view))
        out["sessions"] = sessions
        current_index = next((i for i, s in enumerate(dialog._chat_sessions) if s.get("id") == current_id), 0)
        dialog._load_chat_session(current_index)
        dialog._save_chat_history()
        out["saved_tree"] = _tree(hist_dir, norm)
        out["reload"] = dialog._load_chat_history()
        return out

    # --- 1. history load / render / save -----------------------------------------------
    run("history_synth", lambda: history_scenario(sandbox / "fx" / "hist_synth"))
    if (sandbox / "fx" / "hist_real").exists():
        run("history_real", lambda: history_scenario(sandbox / "fx" / "hist_real"))
    broken = sandbox / "broken"
    _write(broken / "bad.json", "{not json")
    _write(broken / "empty.json", json.dumps({"version": 2, "sessions": []}))
    for name in ("bad.json", "empty.json", "missing.json"):
        run(f"history_{name}", lambda name=name: (setattr(dialog, "_chat_history_path", str(broken / name)),
                                                   dialog._load_chat_history())[1])

    # --- 2. pure helpers ---------------------------------------------------------------
    cls = tg._InputOutputDialog

    def helpers():
        out = {}
        out["order"] = [cls._request_order_from_log(line, 3) for line in HELPER_LINES]
        out["thread"] = [cls._thread_key_from_log(line, src) for line, src in
                         STREAM_TEXT + STREAM_EPUB + STREAM_GLOSSARY]
        out["status"] = [cls._looks_like_pipeline_status(line) for line in HELPER_LINES]
        out["split"] = [dialog._split_embedded_pipeline_log(line) for line in HELPER_LINES]
        out["markup"] = [cls._markup_to_html(text) for text in MARKUP_CORPUS]
        out["html_doc"] = [dialog._response_html_document(text) for text in MARKUP_CORPUS]
        out["xhtml_doc"] = [dialog._response_xhtml_document(text) for text in MARKUP_CORPUS]
        out["match_text"] = [cls._response_artifact_match_text(text) for text in MARKUP_CORPUS]
        out["sizes"] = [cls._format_attachment_size(v) for v in (0, 1023, 1024, 5 * 1024 * 1024, "x", -5)]
        out["limits"] = [cls._normalize_rendered_card_limit(v) for v in (None, 1, 50, 999, "x")]
        out["windows"] = [cls._history_window_bounds(*a) for a in ((0, 5), (10, 4), (10, 4, 0), (10, 4, 9), (3, 0))]
        out["glossary_mode"] = [cls._force_no_glossary_for_mode(m, a) for m in ("", "none", "no_glossary",
                                                                                "manual", "attachments_only", None)
                                for a in (True, False)]
        out["storage"] = [cls._normalize_message_storage(s) for s in (None, {}, {"content_path": " a ", "content_chars": "x",
                                                                               "created_at": "y" * 80, "media_kind": 3})]
        out["attachment_records"] = [cls._normalize_attachment_record(r) for r in (
            None, {"path": ""}, {"path": "a/b.TXT", "size": "12"}, {"path": "c.epub", "name": "N.EPUB", "size": -3})]
        out["supported"] = [cls._is_supported_dropped_text_file(p) for p in ("a.txt", "b.CBZ", "c.png", "d.zip", "")]
        out["vision"] = [(cls._is_image_attachment(p), cls._is_vision_attachment(p)) for p in ("a.png", "b.cbz", "c.txt")]
        out["media_refs"] = cls._generated_media_references_from_text(
            "[GENERATED_IMAGE:a.mp4] [GENERATED_AUDIO:'b.wav'] [GENERATED_VIDEO:c.unknown] [GENERATED_IMAGE:]")
        out["media_only"] = [(cls._is_generated_image_only_content(t), cls._is_generated_media_only_content(t))
                             for t in ("[GENERATED_IMAGE:x.png]", " [GENERATED_AUDIO:y.mp3] ", "text [GENERATED_IMAGE:x]", "")]
        out["cover"] = [cls._is_expected_cover_chapter(c) for c in (
            None, {"is_cover": True}, {"filename": "Text/Cover.xhtml"}, {"title": "Title Page"}, {"title": "Chapter"})]
        out["new_session"] = cls._new_chat_session(4)
        out["descendant"] = [cls._path_is_same_or_descendant(a, b) for a, b in (("a/b", "a"), ("a", "a/b"), ("", ""))]
        out["timestamps"] = [cls._assistant_timestamp_label(dialog, ("assistant", "", "", "", "", "", {"created_at": v}))
                             for v in ("2025-01-02T03:04:05+00:00", "2024-12-31T23:59:59Z", "bad", "")]
        return out

    run("helpers", helpers)

    # --- 3. recorded log streams ----------------------------------------------------------
    def stream_scenario(lines, attachment, commit):
        dialog._chat_history_path = str(sandbox / "streams" / "direct_text_chats.json")
        dialog._chat_sessions = [dialog._new_chat_session(1)]
        dialog._current_chat_index = 0
        dialog._load_chat_session(0)
        dialog._chat_messages.append(("user_file", "book.epub", "x.epub", 1, "", "user") if attachment else ("user", "hi"))
        dialog._reset_history_window()
        dialog._assistant_message_active = True
        dialog._active = True
        dialog._run_source_is_attachment = attachment
        dialog._active_request_next_number = dialog._next_conversation_request_number()
        dialog._active_response_timestamp = dialog._direct_response_timestamp()
        dialog._last_output_folder = ""
        hints = []
        steps = []
        half = len(lines) // 2
        for position, (line, thread) in enumerate(lines):
            hints.append(dialog._on_log_line(line, thread))
            if position == half:
                dialog._drain_log_queue(final=True)
                steps.append(norm({"segments": [dict(s) for s in dialog._active_request_segments],
                                   "label": dialog._processing_label_text}))
        dialog._drain_log_queue(final=True)
        out = norm({
            "hints": hints,
            "steps": steps,
            "segments": [dict(s) for s in dialog._active_request_segments],
            "by_thread": dict(dialog._request_segment_by_thread),
            "phases": dict(dialog._stream_phase_by_thread),
            "listener_phases": dict(dialog._listener_stream_phase_by_thread),
            "glossary_threads": set(dialog._listener_glossary_threads),
            "streamed": dialog._streamed_content,
            "tokens": [dialog._thinking_token_count, dialog._generation_token_count],
            "processing_text": dialog._processing_text,
            "thinking_text": dialog._thinking_stream_text,
            "label": dialog._processing_label_text,
            "flags": [dialog._in_thinking, dialog._streaming_text],
            "messages": [dialog._request_segment_message(s, "out") for s in dialog._active_request_segments],
            "render": html(),
        })
        if commit == "phase":
            dialog._commit_active_request_phase()
            for line, thread in GATE_LINES:
                dialog._on_log_line(line, thread)
            dialog._drain_log_queue(final=True)
        dialog._commit_assistant_message(completion_message=COMPLETION if commit == "completion" else None)
        out["committed"] = norm(list(dialog._chat_messages))
        out["committed_render"] = norm(html())
        out["saved"] = _tree(sandbox / "streams", norm)
        dialog._active = False
        return out

    for name, lines in STREAMS.items():
        for attachment in (False, True):
            for commit in ("plain", "phase", "completion"):
                run(f"stream_{name}_{'att' if attachment else 'txt'}_{commit}",
                    lambda lines=lines, attachment=attachment, commit=commit: stream_scenario(lines, attachment, commit))

    # --- 4. full sends: _start_translation -> stream -> _finish_translation ---------------------
    def send_scenario(case):
        hist = sandbox / "sends" / case["name"]
        dialog._chat_history_path = str(hist / "direct_text_chats.json")
        dialog._chat_sessions = [dialog._new_chat_session(1)]
        dialog._current_chat_index = 0
        dialog._load_chat_session(0)
        translator.records = []
        translator.config = dict(case.get("config", {}))
        translator.translation_thread = None
        translator.manual_glossary_path = ""
        for key, value in case.get("translator", {}).items():
            setattr(translator, key, value)
        dialog._saved_env = {}
        dialog._set_direct_output_mode(case.get("mode", "text"), persist=False)
        {"none": dialog.no_glossary_override_radio, "attachments_only": dialog.attachment_glossary_override_radio,
         "no_glossary": dialog.force_no_glossary_radio, "manual": dialog.manual_glossary_radio}[
            case.get("glossary", "attachments_only")].setChecked(True)
        dialog.skip_thinking_checkbox.setChecked(bool(case.get("skip_thinking")))
        manual = case.get("manual_source")
        if manual is not None:
            manual = dict(manual)
            if manual.get("path"):
                manual["path"] = str(sandbox / "fx" / manual["path"])
            dialog._request_direct_text_manual_glossary = lambda manual=manual: manual
        if case.get("attachment"):
            dialog._set_attachment(str(sandbox / "fx" / "sources" / case["attachment"]))
        dialog.input_box.setPlainText(case.get("text", ""))
        os.environ["OUTPUT_DIRECTORY"] = str(sandbox / "user_output")
        dialog._start_translation()
        out = {"after_start": norm({
            "temp_input_exists": bool(dialog._temp_input) and os.path.isfile(dialog._temp_input),
            "temp_input": dialog._temp_input,
            "temp_input_text": Path(dialog._temp_input).read_text(encoding="utf-8")
            if dialog._temp_input and os.path.isfile(dialog._temp_input) and dialog._temp_input.endswith(".txt") else None,
            "expected": dialog._expected_output,
            "source": [dialog._run_source_path, dialog._run_source_extension, dialog._run_source_is_attachment],
            "manual": dialog._run_manual_glossary_path,
            "mode": dialog._run_output_mode,
            "records": translator.records,
            "status": dialog.status_label.text(),
            "messages": list(dialog._chat_messages),
            "active": dialog._active,
        })}
        temp_root = dialog._temp_root
        _fill_run_tree(sandbox, case, temp_root, dialog._run_source_path, dialog._expected_output)
        for line, thread in case.get("stream", []):
            line = line.replace("<TEMP>", _temp_ref(temp_root))
            for callback in translator.listeners[-1:]:
                callback(line, thread)
        translator.records = []
        dialog._finish_translation()
        out["after_finish"] = norm({
            "messages": list(dialog._chat_messages),
            "status": dialog.status_label.text(),
            "last_output_folder": dialog._last_output_folder,
            "env": translator._env_view(),
            "translator_attrs": {k: getattr(translator, k, "<missing>") for k in (
                "selected_files", "_force_stream_all", "_input_output_run_active", "manual_glossary_path")},
            "records": translator.records,
            "temp_root_exists": bool(temp_root) and os.path.isdir(temp_root),
            "session": dialog._chat_sessions[0],
            "render": html(),
        })
        out["tree"] = _tree(hist, norm)
        out["cwd_tree"] = _tree(sandbox / "user_output", norm)
        shutil.rmtree(sandbox / "user_output", ignore_errors=True)
        os.environ.pop("OUTPUT_DIRECTORY", None)
        return out

    for case in SEND_CASES:
        run(f"send_{case['name']}", lambda case=case: send_scenario(case))

    # --- 5. Migrate (QMessageBox replaced by a recorder) -----------------------------------
    class FakeBox:
        Warning = "Warning"
        AcceptRole = "AcceptRole"
        Cancel = "Cancel"
        calls = []
        accept_merge = False

        def __init__(self, parent=None):
            self.buttons = []
            self.clicked = None
            FakeBox.calls.append(("dialog",))

        @classmethod
        def information(cls, parent, title, text):
            cls.calls.append(("information", title, text))

        @classmethod
        def warning(cls, parent, title, text, *args):
            cls.calls.append(("warning", title, text))

        def setIcon(self, icon):
            FakeBox.calls.append(("setIcon", icon))

        def setWindowTitle(self, title):
            FakeBox.calls.append(("setWindowTitle", title))

        def setText(self, text):
            FakeBox.calls.append(("setText", text))

        def setInformativeText(self, text):
            FakeBox.calls.append(("setInformativeText", text))

        def addButton(self, *args):
            button = ("button",) + tuple(args)
            self.buttons.append(button)
            FakeBox.calls.append(("addButton",) + tuple(args))
            return button

        def setDefaultButton(self, button):
            FakeBox.calls.append(("setDefaultButton", button))

        def exec(self):
            self.clicked = self.buttons[0] if FakeBox.accept_merge else self.buttons[-1]
            FakeBox.calls.append(("exec",))
            return 0

        def clickedButton(self):
            return self.clicked

    def migrate_scenario(variant):
        work = _work_copy(sandbox, f"migrate/{variant}")
        current_id = use_history(work / "direct_text_chats.json")
        index = next(i for i, s in enumerate(dialog._chat_sessions) if s.get("title") == "Synth chat")
        dialog._load_chat_session(index)
        session = dialog._chat_sessions[index]
        source = _migrate_setup(work, variant)
        translator.config = _migrate_config(work, variant)
        FakeBox.calls = []
        FakeBox.accept_merge = variant == "merge"
        dialog._active = variant == "active"
        saved = tg.QMessageBox
        tg.QMessageBox = FakeBox
        try:
            result = dialog._migrate_conversation_attachment(index, str(source))
        finally:
            tg.QMessageBox = saved
            dialog._active = False
        return {"result": result, "calls": list(FakeBox.calls), "current": current_id,
                "messages": session["messages"], "last_output": dialog._last_output_folder,
                "tree": _tree(work, norm), "render": html()}

    for variant in MIGRATE_VARIANTS:
        run(f"migrate_{variant}", lambda variant=variant: migrate_scenario(variant))

    # --- 6. response edits ---------------------------------------------------------------
    def edit_scenario():
        work = _work_copy(sandbox, "edit")
        use_history(work / "direct_text_chats.json")
        index = next(i for i, s in enumerate(dialog._chat_sessions) if s.get("title") == "Synth chat")
        dialog._load_chat_session(index)
        results = []
        for message_index, source, kwargs in EDIT_CALLS:
            try:
                results.append(dialog._save_response_output_edit(message_index, source, **kwargs))
            except Exception as exc:
                results.append(f"{type(exc).__name__}: {exc}")
        return {"results": results, "messages": dialog._chat_messages, "tree": _tree(work, norm),
                "paths": dialog._editable_response_paths(4)}

    run("edit", edit_scenario)

    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(obs, handle, ensure_ascii=False, indent=1, sort_keys=True)
    app.quit()
    return 0


def _temp_ref(temp_root):
    """The run root as a cwd-relative path: identical text (and token count) in every child sandbox."""
    return os.path.relpath(temp_root) if temp_root else ""


def _fill_run_tree(sandbox, case, temp_root, source_path, expected_output):
    """What the translation pipeline would have left in the run's temp root."""
    if temp_root and case.get("tree"):
        stem = os.path.splitext(os.path.basename(source_path))[0]
        shutil.copytree(Path(sandbox) / "fx" / case["tree"], Path(temp_root) / stem, dirs_exist_ok=True)
    if temp_root and case.get("translated"):
        Path(expected_output).parent.mkdir(parents=True, exist_ok=True)
        Path(expected_output).write_text(case["translated"], encoding="utf-8")
    if temp_root and case.get("media"):
        media_path = Path(temp_root) / case["media"]
        media_path.parent.mkdir(parents=True, exist_ok=True)
        media_path.write_bytes(TINY_PNG)


def _migrate_setup(work, variant):
    source = work / "Direct Text" / SYNTH_FOLDER / "Attachments" / "book"
    if variant == "unmanaged":
        source = work / "sources"
    if variant == "file_conflict":
        shutil.rmtree(work / "book")
        _write(work / "book", "a file in the way")
    return source


def _migrate_config(work, variant):
    return {"output_directory": str(work / "elsewhere" if variant == "no_conflict" else work)}


def _headless_child(fixtures, sandbox, out_path):
    """The same scenarios through ``ChatStore`` / ``DirectTextStream`` (never imports Qt)."""
    norm = _child_setup(fixtures, sandbox)
    sandbox = Path(sandbox)
    import direct_text_store as dts
    import direct_text_stream as dtm
    from headless_owner import DirectTextRunOptions

    obs = {}
    env_keys = _env_keys()

    def run(label, fn):
        print(f"[probe] {label}", file=sys.stderr, flush=True)  # last line before a native crash
        try:
            obs[label] = norm(fn())
        except Exception as exc:
            import traceback

            obs[label] = {"exception": f"{type(exc).__name__}: {norm(str(exc))}",
                          "traceback": traceback.format_exc()}

    def chip_record(path):
        """The attachment record the dialog's composer chip stores (``_set_attachment``)."""
        path = os.path.abspath(os.path.expanduser(str(path or "")))
        return {"path": path, "name": os.path.basename(path), "extension": os.path.splitext(path)[1].lower(),
                "size": os.path.getsize(path)}

    def history_scenario(hist_dir):
        out = {}
        store = dts.ChatStore(Path(hist_dir) / "direct_text_chats.json")
        sessions, current_id = store.load_chat_history()
        out["load"] = norm({"sessions": sessions, "current": current_id})
        views = []
        for session in sessions:
            # The dialog's _load_chat_session refreshes the composer chip record (or drops it).
            record = store._normalize_attachment_record(session.get("attachment"))
            session["attachment"] = chip_record(record["path"]) if record and os.path.isfile(record["path"]) else None
            messages = session["messages"]
            message_views = []
            for index, message in enumerate(messages):
                if not message or message[0] != "assistant":
                    message_views.append(None)
                    continue
                with store.session_scope(session):
                    message_views.append({
                        "content": store.assistant_message_text(session, index, "content"),
                        "thinking": store.assistant_message_text(session, index, "thinking"),
                        "chars": [store._assistant_message_char_count(message, k) for k in ("content", "thinking")],
                        "has": [store._assistant_message_has_text(message, k) for k in ("content", "thinking")],
                        "media": store._assistant_generated_media(message, message[1]),
                        "ts": dts.timestamp_label(store._assistant_storage_for(message).get("created_at", "")),
                        "attachment_src": store._assistant_source_is_attachment(index),
                        "artifact": store._response_output_artifact_path(message, index),
                    })
            with store.session_scope(session):
                next_request = store._next_conversation_request_number()
            view = {
                "messages": message_views,
                "next_request": next_request,
                "managed": store.managed_conversation_output_folder(session),
                "attachments": store.conversation_attachment_folders(session),
            }
            try:
                view["validated"] = store.validated_chat_output_folder(session)
            except Exception as exc:
                view["validated"] = f"{type(exc).__name__}: {exc}"
            for folder in view["attachments"]:
                view.setdefault("workspaces", []).append({
                    "managed": store.is_managed_attachment_workspace(session, folder),
                    "epub": store.attachment_compiled_epub_path(folder),
                    "preferred": store._preferred_attachment_compiled_documents(folder),
                })
            views.append(norm(view))
        out["sessions"] = views
        store.save_chat_history(sessions, current_id)
        out["saved_tree"] = _tree(hist_dir, norm)
        out["reload"] = dts.ChatStore(Path(hist_dir) / "direct_text_chats.json").load_chat_history()
        return out

    run("history_synth", lambda: history_scenario(sandbox / "fx" / "hist_synth"))
    if (sandbox / "fx" / "hist_real").exists():
        run("history_real", lambda: history_scenario(sandbox / "fx" / "hist_real"))
    broken = sandbox / "broken"
    _write(broken / "bad.json", "{not json")
    _write(broken / "empty.json", json.dumps({"version": 2, "sessions": []}))
    for name in ("bad.json", "empty.json", "missing.json"):
        run(f"history_{name}", lambda name=name: dts.ChatStore(broken / name)._load_chat_history())

    def stream_scenario(lines, attachment, commit):
        user_turn = ("user_file", "book.epub", "x.epub", 1, "", "user") if attachment else ("user", "hi")
        # bound to a store like the dialog: commits land in the chat and are saved (externalised)
        store = dts.ChatStore(sandbox / "streams" / "direct_text_chats.json")
        store.current_session()["messages"].append(user_turn)
        stream = dtm.DirectTextStream(store=store, source_is_attachment=attachment, model="gpt-4o")
        hints = []
        steps = []
        half = len(lines) // 2
        for position, (line, thread) in enumerate(lines):
            hints.append(stream.feed(line, thread))
            if position == half:
                steps.append(norm({"segments": stream.segments(), "label": stream.processing_label}))
        segments = stream.segments()
        out = norm({
            "hints": hints,
            "steps": steps,
            "segments": segments,
            "by_thread": dict(stream._request_segment_by_thread),
            "phases": dict(stream._stream_phase_by_thread),
            "listener_phases": dict(stream._listener_stream_phase_by_thread),
            "glossary_threads": set(stream._listener_glossary_threads),
            "streamed": stream.streamed_content,
            "tokens": [stream._thinking_token_count, stream._generation_token_count],
            "processing_text": stream._processing_text,
            "thinking_text": stream._thinking_stream_text,
            "label": stream.processing_label,
            "flags": [stream._in_thinking, stream._streaming_text],
            "messages": stream.messages("out"),
            "module_messages": [dtm.request_segment_message(s, "out") for s in segments],
        })
        if commit == "phase":
            stream.commit_active_request_phase()
            for line, thread in GATE_LINES:
                stream.feed(line, thread)
            stream.drain(final=True)
        stream._commit_assistant_message(completion_message=COMPLETION if commit == "completion" else None)
        out["committed"] = norm(list(stream.chat_messages))
        return out

    for name, lines in STREAMS.items():
        for attachment in (False, True):
            for commit in ("plain", "phase", "completion"):
                run(f"stream_{name}_{'att' if attachment else 'txt'}_{commit}",
                    lambda lines=lines, attachment=attachment, commit=commit: stream_scenario(lines, attachment, commit))

    class RecordingOwner:
        def __init__(self):
            self.records = []

        def _env_view(self):
            return {k: os.environ.get(k) for k in env_keys}

        def _apply_forced_streaming_environment(self):
            self.records.append(("forced_streaming", self._env_view()))

        def _apply_direct_text_runtime_environment(self):
            self.records.append(("direct_text_runtime", self._env_view()))

    def send_scenario(case):
        hist = sandbox / "sends" / case["name"]
        env_before = dict(os.environ)  # the mobile job restores it (job_runner.scoped_process_state)
        os.environ["OUTPUT_DIRECTORY"] = str(sandbox / "user_output")
        store = dts.ChatStore(hist / "direct_text_chats.json")
        session = store.current_session()
        text = case.get("text", "")
        attachment = chip_record(sandbox / "fx" / "sources" / case["attachment"]) if case.get("attachment") else None
        attachment = dts.normalize_attachment_record(attachment)
        store.title_chat_from_text(session, attachment["name"] if attachment else text)
        session["messages"].append(
            ("user_file", attachment["name"], attachment["path"], attachment["size"], text, "user")
            if attachment else ("user", text))
        manual = case.get("manual_source")
        if manual is not None:
            manual = dict(manual)
            if manual.get("path"):
                manual["path"] = str(sandbox / "fx" / manual["path"])
        run_state = dts.prepare_direct_text_input(text, attachment, manual)
        force_no_glossary = bool(
            dts.force_no_glossary_for_mode(case.get("glossary", "attachments_only"), run_state["is_attachment"])
            and not run_state["manual_glossary_path"])
        owner = RecordingOwner()
        DirectTextRunOptions(selected_files=[run_state["temp_input"]], force_no_glossary=force_no_glossary,
                             manual_glossary_path=run_state["manual_glossary_path"],
                             output_mode=case.get("run_mode", case.get("mode", "text"))).apply_to(owner)
        dts.apply_direct_text_run_environment(owner, run_state["temp_root"], run_state["is_attachment"])
        os.environ.clear()
        os.environ.update(env_before)
        os.environ["OUTPUT_DIRECTORY"] = str(sandbox / "user_output")
        out = {"after_start": norm({
            "temp_input_exists": os.path.isfile(run_state["temp_input"]),
            "temp_input": run_state["temp_input"],
            "temp_input_text": Path(run_state["temp_input"]).read_text(encoding="utf-8")
            if run_state["temp_input"].endswith(".txt") else None,
            "expected": run_state["expected_output"],
            "source": [run_state["source_path"], run_state["source_extension"], run_state["is_attachment"]],
            "manual": run_state["manual_glossary_path"],
            "records": owner.records,
        })}
        temp_root = run_state["temp_root"]
        _fill_run_tree(sandbox, case, temp_root, run_state["source_path"], run_state["expected_output"])
        stream = dtm.make_stream(source_is_attachment=run_state["is_attachment"], model="gpt-4o")
        for line, thread in case.get("stream", []):
            stream.feed(line.replace("<TEMP>", _temp_ref(temp_root)), thread)
        glossary_path = run_state["manual_glossary_path"] or dict(case.get("translator", {})).get("manual_glossary_path", "")
        result = store.finish_run(session, dict(run_state, output_mode=case.get("run_mode", case.get("mode", "text")),
                                                force_no_glossary=force_no_glossary, glossary_path=glossary_path),
                                  stream)
        out["after_finish"] = norm({
            "messages": list(session["messages"]),
            "status": result["status"],
            "last_output_folder": result["output_folder"],
            "temp_root_exists": bool(temp_root) and os.path.isdir(temp_root),
            "session": session,
            "result_messages": result["messages"],
        })
        out["tree"] = _tree(hist, norm)
        out["cwd_tree"] = _tree(sandbox / "user_output", norm)
        shutil.rmtree(sandbox / "user_output", ignore_errors=True)
        os.environ.pop("OUTPUT_DIRECTORY", None)
        return out

    for case in SEND_CASES:
        run(f"send_{case['name']}", lambda case=case: send_scenario(case))

    def migrate_scenario(variant):
        work = _work_copy(sandbox, f"migrate/{variant}")
        store = dts.ChatStore(work / "direct_text_chats.json", config=_migrate_config(work, variant))
        sessions, current_id = store.load_chat_history()
        session = next(s for s in sessions if s.get("title") == "Synth chat")
        record = store._normalize_attachment_record(session.get("attachment"))
        session["attachment"] = chip_record(record["path"]) if record and os.path.isfile(record["path"]) else None
        store.set_current_session(session)  # the dialog migrates from the open chat
        source = _migrate_setup(work, variant)
        store._active = variant == "active"
        outcome = store.migrate_attachment(session, str(source), confirm_merge=lambda target: variant == "merge")
        store._active = False
        return {"result": outcome["ok"],
                "notices": [(n["level"], n["title"], n["text"]) for n in outcome["notices"]],
                "current": current_id, "messages": session["messages"], "tree": _tree(work, norm)}

    for variant in MIGRATE_VARIANTS:
        run(f"migrate_{variant}", lambda variant=variant: migrate_scenario(variant))

    def edit_scenario():
        work = _work_copy(sandbox, "edit")
        store = dts.ChatStore(work / "direct_text_chats.json")
        sessions, _current = store.load_chat_history()
        session = next(s for s in sessions if s.get("title") == "Synth chat")
        record = store._normalize_attachment_record(session.get("attachment"))
        session["attachment"] = chip_record(record["path"]) if record and os.path.isfile(record["path"]) else None
        store.set_current_session(session)  # the dialog edits in the open chat
        results = []
        for message_index, source, kwargs in EDIT_CALLS:
            try:
                results.append(store.save_response_output_edit(session, message_index, source, **kwargs))
            except Exception as exc:
                results.append(f"{type(exc).__name__}: {exc}")
        return {"results": results, "messages": session["messages"], "tree": _tree(work, norm),
                "paths": store.editable_response_paths(session, 4)}

    run("edit", edit_scenario)

    obs["qt_loaded"] = sorted(name for name in sys.modules if name.startswith(("PySide6", "translator_gui")))
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(obs, handle, ensure_ascii=False, indent=1, sort_keys=True)
    return 0


def _run_probe(label, tg_dir, fixtures, root, headless=False, timeout=900):
    out_path = Path(root) / f"{label}.json"
    cmd = [sys.executable, str(Path(__file__).resolve()), "--probe", "--tg-dir", str(tg_dir or ""),
           "--fixtures", str(fixtures), "--sandbox", str(Path(root) / ("sb_" + hashlib.sha1(label.encode()).hexdigest()[:8])),
           "--out", str(out_path)]
    if headless:
        cmd.append("--headless")
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)
    if proc.returncode != 0 or not out_path.exists():
        raise AssertionError(f"probe {label} failed ({proc.returncode}):\n{proc.stdout[-4000:]}\n{proc.stderr[-6000:]}")
    return json.loads(out_path.read_text(encoding="utf-8"))


def _first_difference(a, b, path="$"):
    if type(a) is not type(b):
        return path, a, b
    if isinstance(a, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                return f"{path}.{key}", a.get(key, "<missing>"), b.get(key, "<missing>")
            found = _first_difference(a[key], b[key], f"{path}.{key}")
            if found:
                return found
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}[len]", len(a), len(b)
        for index, (x, y) in enumerate(zip(a, b)):
            found = _first_difference(x, y, f"{path}[{index}]")
            if found:
                return found
        return None
    return None if a == b else (path, a, b)


#: Observations the GUI-free hosts must reproduce, per scenario prefix (the rest is Qt UI state).
HEADLESS_FIELDS = {
    "history_": {"load": None, "saved_tree": None, "reload": None,
                 "sessions": ("messages", "next_request", "managed", "attachments", "validated", "workspaces")},
    "stream_": {"hints": None, "steps": None, "segments": None, "by_thread": None, "phases": None,
                "listener_phases": None, "glossary_threads": None, "streamed": None, "tokens": None,
                "processing_text": None, "thinking_text": None, "label": None, "flags": None, "messages": None,
                "committed": None},
    "send_": {"after_start": ("temp_input_exists", "temp_input", "temp_input_text", "expected", "source", "manual"),
              "after_finish": ("messages", "status", "last_output_folder", "temp_root_exists", "session"),
              "tree": None, "cwd_tree": None},
    "migrate_": {"result": None, "messages": None, "tree": None},
    "edit": {"results": None, "messages": None, "tree": None, "paths": None},
}


def headless_view(label, observation):
    """The part of a dialog observation the GUI-free hosts must reproduce (None: UI-only scenario)."""
    spec = next((fields for prefix, fields in HEADLESS_FIELDS.items() if label.startswith(prefix)), None)
    if spec is None:
        return None
    if not isinstance(observation, dict) or "exception" in observation:
        return observation
    view = {}
    for key, sub in spec.items():
        if key not in observation:
            continue
        value = observation[key]
        if sub is None:
            view[key] = value
        elif isinstance(value, list):
            view[key] = [{k: item.get(k) for k in sub if k in item} for item in value]
        else:
            view[key] = {k: value.get(k) for k in sub if k in value}
    return view


#: Local golden of the BASE_SHA dialog on the synthetic scenarios (gitignored like the other goldens;
#: the real-history scenario is never stored: it holds the developer's own chats).
GOLDEN_PATH = TESTS / "parity" / "golden" / BASE_SHA[:12] / "direct_text_dialog.json"


def golden_view(observations):
    return {k: v for k, v in observations.items() if k not in ("translator_gui", "history_real")}


def capture_golden():
    """Run the BASE_SHA dialog probe and store its synthetic observations as ``GOLDEN_PATH``."""
    data = subprocess.run(["git", "show", f"{BASE_SHA}:src/translator_gui.py"], cwd=str(REPO_ROOT),
                          capture_output=True, check=True, timeout=120).stdout
    root = Path(tempfile.mkdtemp(prefix="dtprobe_"))  # same prefix (path length) as the test run
    try:
        legacy_dir = root / "legacy_tg"
        legacy_dir.mkdir()
        (legacy_dir / "translator_gui.py").write_bytes(data)
        if (SRC / "Halgakos.ico").exists():
            shutil.copy2(SRC / "Halgakos.ico", legacy_dir / "Halgakos.ico")
        build_fixtures(root / "fixtures")
        observations = _run_probe("legacy", legacy_dir, root / "fixtures", root)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(golden_view(observations), ensure_ascii=False, indent=1, sort_keys=True),
                           encoding="utf-8")
    print(f"wrote {GOLDEN_PATH} ({len(observations)} scenarios)")
    return 0


def _main(argv=None):
    parser = argparse.ArgumentParser(description="Direct Text dialog probe (see the module docstring).")
    parser.add_argument("--capture-golden", action="store_true", help="store the BASE_SHA dialog golden")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--tg-dir", default="")
    parser.add_argument("--fixtures")
    parser.add_argument("--sandbox")
    parser.add_argument("--out")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args(argv)
    if args.probe or args.headless:
        import faulthandler

        faulthandler.enable()  # a native crash in a probe child prints its Python stack to stderr
    if args.capture_golden:
        return capture_golden()
    if args.headless:
        return _headless_child(args.fixtures, args.sandbox, args.out)
    return _probe_child(args.tg_dir or None, args.fixtures, args.sandbox, args.out)


# ---------------------------------------------------------------------------
# pytest: what moved where
# ---------------------------------------------------------------------------

#: _InputOutputDialog members now in direct_text_store.ChatStoreMixin.
MOVED_STORE = (
    "_OUTPUT_ENV_KEYS", "_VIDEO_OUTPUT_EXTENSIONS", "_AUDIO_OUTPUT_EXTENSIONS",
    "_VISION_ARCHIVE_ATTACHMENT_EXTENSIONS", "_DIRECT_TEXT_ENV_KEYS",
    "_DEFAULT_RENDERED_CARD_LIMIT", "_MIN_RENDERED_CARD_LIMIT", "_MAX_RENDERED_CARD_LIMIT",
    "_normalize_rendered_card_limit", "_history_window_bounds", "_new_chat_session", "_direct_response_timestamp",
    "_resolve_chat_history_path", "_load_chat_history", "_schedule_chat_history_save", "_normalize_message_storage",
    "_history_file_reference", "_resolve_history_file_reference", "_assistant_storage_for",
    "_generated_image_paths_from_text", "_media_kind_for_path", "_generated_media_references_from_text",
    "_assistant_generated_media", "_assistant_generated_image_path", "_is_generated_image_only_content",
    "_is_generated_media_only_content", "_promote_generated_media_reference", "_promote_generated_image_reference",
    "_response_generated_image_path", "_response_generated_media", "_assistant_message_with_timestamp",
    "_assistant_timestamp_label", "_next_conversation_request_number", "_assistant_message_char_count",
    "_assistant_message_has_text", "_assistant_message_text", "_response_html_document", "_response_xhtml_document",
    "_write_response_files", "_externalize_session_messages", "_save_chat_history", "_normalize_attachment_record",
    "_force_no_glossary_for_mode", "_is_supported_dropped_text_file", "_is_image_attachment", "_is_vision_attachment",
    "_format_attachment_size", "_current_chat_session", "_validated_chat_output_folder",
    "_managed_conversation_output_folder", "_conversation_attachment_folders", "_is_managed_attachment_workspace",
    "_direct_text_migration_output_root", "_path_is_same_or_descendant", "_relocate_session_attachment_paths",
    "_preferred_attachment_compiled_documents", "_remove_extra_attachment_compiled_documents",
    "_attachment_compiled_epub_path", "_migrate_conversation_attachment", "_title_current_chat_from_text",
    "_remember_output_folder", "_editable_response_paths", "_response_artifact_match_text",
    "_response_output_artifact_path", "_save_response_output_edit", "_markup_to_html",
    "_assistant_source_is_attachment", "_ensure_conversation_output_folder_for_session",
    "_conversation_output_folder", "_next_indexed_output_path", "_copy_indexed_output_file",
    "_attachment_output_subfolder", "_copy_attachment_output_tree", "_effective_run_glossary_path",
    "_sync_attachment_glossary", "_discover_generated_output", "_persist_output_folder", "_find_direct_run_artifact",
)
#: _InputOutputDialog members now in direct_text_stream.DirectTextStreamMixin.
MOVED_STREAM = (
    "_STATUS_FIRST_CHARS", "_PIPELINE_LOG_PHRASES", "_EMBEDDED_PIPELINE_MARKERS", "_STREAM_END_LOG_PHRASES",
    "_HIDDEN_STREAM_START_LOG_PHRASES", "_DIRECT_RESPONSE_PAYLOAD_PREFIX", "_DIRECT_GLOSSARY_STREAM_START_PREFIX",
    "_allocate_active_request_number", "_set_thinking_toggle_text", "_refresh_processing_label", "_get_token_encoder",
    "_count_tokens", "_set_thinking_spinner_active", "_request_label_from_log", "_request_segment_for_completion_log",
    "_apply_direct_response_payload", "_apply_direct_glossary_stream_start", "_request_order_from_log",
    "_sort_active_request_segments", "_thread_key_from_log", "_begin_request_segment", "_request_segment_for_thread",
    "_request_segment_message", "_on_log_line", "_looks_like_pipeline_status", "_classify_line",
    "_split_embedded_pipeline_log", "_drain_log_queue", "_commit_active_request_phase", "_commit_assistant_message",
    "_append_thinking", "_hydrate_header_toc_response", "_build_attachment_action_card",
    "_build_attachment_extraction_summary", "_is_expected_cover_chapter", "_finish_translation",
)
#: Split into data builders + card formatters (checked by behaviour, not text).
SPLIT_BUILDERS = ("_build_attachment_action_card", "_build_attachment_extraction_summary")
NEW_STORE_METHODS = ("_attachment_card_actions", "_init_chat_sessions", "_prepare_direct_text_input")
#: Mobile-only knobs of _prepare_direct_text_input (U3 Integrate); the defaults are the dialog's
#: behaviour: ``mkdtemp(dir=None)`` is the OS temp dir and no extra type is passed through.
MOBILE_PREPARE_KNOBS = ("_DIRECT_TEXT_TEMP_PARENT", "_EXTRA_PASS_THROUGH_EXTENSIONS")
PREPARE_KNOB_EDITS = (
    ('        self._temp_root = tempfile.mkdtemp(prefix="glossarion_input_output_")\n',
     '        self._temp_root = tempfile.mkdtemp(prefix="glossarion_input_output_", dir=self._DIRECT_TEXT_TEMP_PARENT)\n'),
    ("                or attached_extension in self._IMAGE_ATTACHMENT_EXTENSIONS\n",
     "                or attached_extension in self._IMAGE_ATTACHMENT_EXTENSIONS\n"
     "                or attached_extension in self._EXTRA_PASS_THROUGH_EXTENSIONS\n"),
)
#: Mobile-only knob of _effective_run_glossary_path (U3 fix pass): the run environment its
#: MANUAL_GLOSSARY comes from; None (the default) is the dialog's live os.environ read. Only the
#: GUI-free DirectTextStream.load_run sets it (the job's recorded values; the chat finishes a
#: run off the job thread, possibly while the next queued job runs).
MOBILE_FINISH_KNOBS = ("_RUN_ENVIRONMENT",)
RUN_ENV_KNOB_EDITS = (
    ("            os.environ.get('MANUAL_GLOSSARY', ''),\n",
     "            (os.environ if self._RUN_ENVIRONMENT is None else self._RUN_ENVIRONMENT).get('MANUAL_GLOSSARY', ''),\n"),
)

QMESSAGEBOX_EDITS = (
    ("            QMessageBox.information(\n                message_parent,\n",
     "            self._direct_text_notice(\n                \"information\",\n                message_parent,\n"),
    ("        QMessageBox.information(\n            message_parent,\n",
     "        self._direct_text_notice(\n            \"information\",\n            message_parent,\n"),
    ("            QMessageBox.warning(\n                message_parent,\n",
     "            self._direct_text_notice(\n                \"warning\",\n                message_parent,\n"),
)


#: U7: four rules that were inline in the dialog's Qt handlers moved to direct_text_store module
#: functions the dialog calls (source indentation; tests/parity/DISCREPANCIES.md "U7 image / RPG Maker").
U7_DIALOG_EDITS = {
    "__init__": [(
        "        configured_glossary_override = str(\n"
        "            translator.config.get('direct_text_glossary_override_mode', '') or ''\n"
        "        ).strip().lower()\n"
        "        if configured_glossary_override not in {\n"
        "            'none', 'attachments_only', 'no_glossary', 'manual'\n"
        "        }:\n"
        "            # Plain text keeps the safe No Glossary behavior, while attached\n"
        "            # documents inherit the main translator by default.\n"
        "            configured_glossary_override = 'attachments_only'\n",
        "        # Plain text keeps the safe No Glossary behavior, while attached\n"
        "        # documents inherit the main translator by default (direct_text_store).\n"
        "        configured_glossary_override = configured_glossary_override_mode(\n"
        "            translator.config.get('direct_text_glossary_override_mode', '')\n"
        "        )\n")],
    "_on_glossary_override_toggled": [(
        "        mode = str(mode or 'none').strip().lower()\n"
        "        if mode not in {\n"
        "            'none', 'attachments_only', 'no_glossary', 'manual'\n"
        "        }:\n"
        "            mode = 'attachments_only'\n"
        "        try:\n"
        "            config = self.translator.config\n"
        "            config['direct_text_glossary_override_mode'] = mode\n"
        "            # Keep the previous keys synchronized for backward compatibility\n"
        "            # with older builds that do not know about the enum setting.\n"
        "            config['direct_text_force_no_glossary'] = mode == 'no_glossary'\n"
        "            config['direct_text_manual_glossary'] = mode == 'manual'\n",
        "        # The enum plus the two legacy booleans (direct_text_store, shared with mobile)\n"
        "        updates = glossary_override_config_updates(mode)\n"
        "        try:\n"
        "            config = self.translator.config\n"
        "            config.update(updates)\n")],
    "_rename_chat": [(
        '        new_title = " ".join(str(new_title or "").split())[:120]\n',
        "        new_title = chat_rename_title(new_title)\n")],
    "_request_direct_text_manual_glossary": [
        ("        import json as json_lib\n", ""),
        ("        allowed_extensions = {'.csv', '.json', '.txt', '.md'}\n",
         "        allowed_extensions = set(MANUAL_GLOSSARY_EXTENSIONS)\n"),
        ("            content = editor.toPlainText()\n"
         "            if not content.strip():\n",
         "            content = editor.toPlainText()\n"
         "            # An unedited loaded file is used by path; edited/pasted contents\n"
         "            # get a sniffed extension (direct_text_store, shared with mobile).\n"
         "            record = manual_glossary_source_record(\n"
         "                content,\n"
         "                editor.source_path,\n"
         "                editor.source_text,\n"
         "                editor.source_extension,\n"
         "            )\n"
         "            if record is None:\n"),
        ("                return\n"
         "\n"
         "            if editor.source_path and content == editor.source_text:\n"
         "                result.update({\n"
         "                    'kind': 'path',\n"
         "                    'path': editor.source_path,\n"
         "                    'extension': editor.source_extension,\n"
         "                })\n"
         "                dialog.accept()\n"
         "                return\n"
         "\n"
         "            extension = '.txt'\n"
         "            stripped = content.lstrip()\n"
         "            if stripped.startswith(('{', '[')):\n"
         "                try:\n"
         "                    json_lib.loads(content)\n"
         "                    extension = '.json'\n"
         "                except (TypeError, ValueError):\n"
         "                    # Keep malformed/non-JSON structured text as plain text;\n"
         "                    # this avoids silently rewriting what the user pasted.\n"
         "                    extension = '.txt'\n"
         "            else:\n"
         "                nonempty_lines = [line for line in content.splitlines() if line.strip()]\n"
         "                if len(nonempty_lines) > 1 and ',' in nonempty_lines[0]:\n"
         "                    extension = '.csv'\n"
         "            result.update({\n"
         "                'kind': 'content',\n"
         "                'content': content,\n"
         "                'extension': extension,\n"
         "            })\n"
         "            dialog.accept()\n",
         "                return\n"
         "            result.update(record)\n"
         "            dialog.accept()\n"),
    ],
}


def _apply_u7_dialog_edits(name, text):
    for old, new in U7_DIALOG_EDITS.get(name, ()):
        assert text.count(_dedent4(old)) == 1, (name, old[:80])
        text = text.replace(_dedent4(old), _dedent4(new))
    return text


def _src_text(module):
    return (SRC / f"{module}.py").read_bytes().decode("utf-8-sig").replace("\r\n", "\n")


def _legacy_tg_text():
    try:
        data = subprocess.run(["git", "show", f"{BASE_SHA}:src/translator_gui.py"], cwd=str(REPO_ROOT),
                              capture_output=True, check=True, timeout=120).stdout
    except Exception as exc:  # shallow clone / no git
        import pytest

        pytest.skip(f"git show {BASE_SHA[:12]}:src/translator_gui.py unavailable: {exc}")
    return data.decode("utf-8-sig").replace("\r\n", "\n")


def _class_node(tree, name):
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


def _member_texts(text, class_name):
    """{member: dedented source incl. decorators} for a class body (methods and simple assigns)."""
    tree = ast.parse(text)
    lines = text.split("\n")
    out = {}
    for node in _class_node(tree, class_name).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = min([d.lineno for d in node.decorator_list] + [node.lineno])
            out[node.name] = textwrap.dedent("\n".join(lines[start - 1:node.end_lineno]))
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            out[node.targets[0].id] = textwrap.dedent("\n".join(lines[node.lineno - 1:node.end_lineno]))
    return out


def _function_text(text, name):
    tree = ast.parse(text)
    lines = text.split("\n")
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return "\n".join(lines[node.lineno - 1:node.end_lineno])


def _dedent4(text):
    return "\n".join(line[4:] if line.startswith("    ") else line for line in text.split("\n"))


def _expected_move(name, legacy_text):
    """The legacy member (dedented like the member texts) after the documented edits."""
    text = legacy_text
    if name in MOVED_STORE:
        text = text.replace("_InputOutputDialog.", "ChatStoreMixin.")
    if name == "_effective_run_glossary_path":
        for old, new in RUN_ENV_KNOB_EDITS:
            assert text.count(_dedent4(old)) == 1, old
            text = text.replace(_dedent4(old), _dedent4(new))
    if name == "_migrate_conversation_attachment":
        for old, new in QMESSAGEBOX_EDITS:
            assert _dedent4(old) in text
            text = text.replace(_dedent4(old), _dedent4(new))
        question = "        if warning.clickedButton() is not merge_button:\n"
        merge_start = text.index("        warning = QMessageBox(message_parent)\n")
        merge_end = text.index(question + "            return False\n")
        text = (text[:merge_start] + "        if not self._confirm_attachment_merge(message_parent, target_folder):\n"
                + text[merge_end + len(question):])
    return text


def test_moved_members_are_verbatim():
    legacy = _member_texts(_legacy_tg_text(), "_InputOutputDialog")
    store = _member_texts(_src_text("direct_text_store"), "ChatStoreMixin")
    stream = _member_texts(_src_text("direct_text_stream"), "DirectTextStreamMixin")
    assert set(MOVED_STORE) | set(NEW_STORE_METHODS) | {"_IMAGE_ATTACHMENT_EXTENSIONS"} | set(
        MOBILE_PREPARE_KNOBS) | set(MOBILE_FINISH_KNOBS) == set(store)
    # the mobile-only knobs default to the dialog's behaviour (OS temp dir, no extra pass-through
    # type, MANUAL_GLOSSARY read from the live os.environ)
    import direct_text_store

    assert direct_text_store.ChatStoreMixin._DIRECT_TEXT_TEMP_PARENT is None
    assert direct_text_store.ChatStoreMixin._EXTRA_PASS_THROUGH_EXTENSIONS == frozenset()
    assert direct_text_store.ChatStoreMixin._RUN_ENVIRONMENT is None
    assert set(MOVED_STREAM) == set(stream)
    for name in MOVED_STORE:
        assert store[name] == _expected_move(name, legacy[name]), f"ChatStoreMixin.{name} differs from {BASE_SHA[:12]}"
    for name in MOVED_STREAM:
        if name in SPLIT_BUILDERS:
            continue
        assert stream[name] == legacy[name], f"DirectTextStreamMixin.{name} differs from {BASE_SHA[:12]}"


def test_dialog_keeps_no_moved_member_and_defines_every_hook():
    import direct_text_store
    import direct_text_stream

    tree = ast.parse(_src_text("translator_gui"))
    dialog = _class_node(tree, "_InputOutputDialog")
    assert [ast.unparse(b) for b in dialog.bases] == ["DirectTextStreamMixin", "ChatStoreMixin", "QDialog"]
    body = _member_texts(_src_text("translator_gui"), "_InputOutputDialog")
    moved = set(MOVED_STORE) | set(MOVED_STREAM) | set(NEW_STORE_METHODS)
    assert not moved & set(body), sorted(moved & set(body))
    hooks = set(direct_text_store.CHAT_STORE_HOOKS) | set(direct_text_stream.STREAM_HOOKS)
    assert hooks <= set(body), sorted(hooks - set(body))
    # the mixins never define a hook (the dialog's Qt implementation must win)
    mixin_names = set(_member_texts(_src_text("direct_text_store"), "ChatStoreMixin")) | set(
        _member_texts(_src_text("direct_text_stream"), "DirectTextStreamMixin"))
    assert not hooks & mixin_names
    # both GUI-free hosts implement every hook
    for host in (direct_text_store.ChatStore, direct_text_stream.DirectTextStream):
        missing = [h for h in direct_text_store.CHAT_STORE_HOOKS if host is direct_text_store.ChatStore and h not in vars(host)]
        missing += [h for h in direct_text_stream.STREAM_HOOKS if host is direct_text_stream.DirectTextStream
                    and h not in vars(host)]
        assert not missing, (host, missing)
    # translator_gui keeps importing the moved helper name
    assert "from direct_text_store import ChatStoreMixin, _atomic_text_write, apply_direct_text_run_environment" in _src_text(
        "translator_gui")


def test_dialog_rewiring_is_exactly_the_documented_edits():
    legacy_text = _legacy_tg_text()
    legacy = _member_texts(legacy_text, "_InputOutputDialog")
    current = _member_texts(_src_text("translator_gui"), "_InputOutputDialog")
    store_text = _src_text("direct_text_store")
    store = _member_texts(store_text, "ChatStoreMixin")

    # the merge question is the _confirm_attachment_merge hook, verbatim
    merge = legacy["_migrate_conversation_attachment"]
    block = merge[merge.index("        warning = QMessageBox(message_parent)"):merge.index(
        "        if warning.clickedButton() is not merge_button:")]
    assert textwrap.indent(textwrap.dedent(block), "    ") in current["_confirm_attachment_merge"]
    assert "return warning.clickedButton() is merge_button" in current["_confirm_attachment_merge"]

    # __init__: the chat-loading block is _init_chat_sessions (minus the UI flag)
    init = legacy["__init__"]
    start = init.index("    self._chat_sessions, current_chat_id = self._load_chat_history()\n")
    end_marker = '    self._chat_messages = self._chat_sessions[self._current_chat_index]["messages"]\n'
    end = init.index(end_marker, start) + len(end_marker)
    legacy_block = init[start:end].replace("    self._switching_chat = False\n", "")
    assert legacy_block in store["_init_chat_sessions"]
    assert _apply_u7_dialog_edits(
        "__init__", init[:start] + "    self._init_chat_sessions()\n    self._switching_chat = False\n" + init[end:]
    ) == current["__init__"]

    # _start_translation: temp-input block -> _prepare_direct_text_input, env block -> the shared helper
    send = legacy["_start_translation"]
    temp_start = send.index('        self._temp_root = tempfile.mkdtemp(prefix="glossarion_input_output_")')
    temp_end = send.index("\n\n        gui = self.translator")
    prepare_block = textwrap.indent(textwrap.dedent(send[temp_start:temp_end]), "    ")
    for old, new in PREPARE_KNOB_EDITS:  # the mobile-only knobs (defaults = this exact code)
        assert prepare_block.count(_dedent4(old)) == 1, old
        prepare_block = prepare_block.replace(_dedent4(old), _dedent4(new))
    assert prepare_block in store["_prepare_direct_text_input"]
    env_start = send.index("        os.environ['OUTPUT_DIRECTORY'] = self._temp_root")
    env_end = send.index("gui._apply_direct_text_runtime_environment()\n", env_start) + len(
        "gui._apply_direct_text_runtime_environment()\n")
    env_block = textwrap.dedent(send[env_start:env_end]).replace("self._temp_root", "output_root").replace(
        "self._run_source_is_attachment", "is_attachment").replace("gui.", "owner.")
    assert textwrap.indent(env_block, "    ").rstrip("\n") in _function_text(
        store_text, "apply_direct_text_run_environment")
    expected_send = (
        send[:temp_start]
        + "        manual_glossary_path = self._prepare_direct_text_input(\n"
          "            text, attachment, manual_glossary_source\n        )"
        + send[temp_end:env_start]
        + "        apply_direct_text_run_environment(\n"
          "            gui, self._temp_root, self._run_source_is_attachment\n        )\n"
        + send[env_end:]
    ).replace("    import uuid\n    from datetime import datetime\n\n", "")
    assert expected_send == current["_start_translation"]

    # _render_output: the attachment links come from _attachment_card_actions (same order, same HTML)
    render = legacy["_render_output"]
    render = render.replace(
        "                if self._is_managed_attachment_workspace(\n"
        "                    session, output_folder\n"
        "                ):\n",
        "                attachment_actions = self._attachment_card_actions(\n"
        "                    session, output_folder\n"
        "                )\n"
        "                if 'migrate' in attachment_actions:\n")
    render = render.replace(
        "                compiled_epub = self._attachment_compiled_epub_path(\n"
        "                    output_folder\n"
        "                )\n"
        "                if compiled_epub:\n",
        "                if 'reader' in attachment_actions:\n")
    assert render == current["_render_output"]

    # U7: the inline rules are direct_text_store calls (exactly the documented edits)
    for name in sorted(set(U7_DIALOG_EDITS) - {"__init__"}):
        assert _apply_u7_dialog_edits(name, legacy[name]) == current[name], name

    # every other dialog member is unchanged
    changed = {"__init__", "_start_translation", "_render_output", "_direct_text_notice", "_confirm_attachment_merge"}
    changed |= set(U7_DIALOG_EDITS)
    moved = set(MOVED_STORE) | set(MOVED_STREAM)
    for name, text in current.items():
        if name in changed:
            continue
        assert legacy.get(name) == text or name == "_IMAGE_ATTACHMENT_EXTENSIONS", f"_InputOutputDialog.{name} changed"
    assert set(legacy) - moved - set(current) == set(), sorted(set(legacy) - moved - set(current))


def test_u7_dialog_rule_functions_behave_like_the_removed_lines():
    """direct_text_store's U7 rule functions give what the dialog's removed inline code gave."""
    import types

    import direct_text_store as dts

    legacy = _member_texts(_legacy_tg_text(), "_InputOutputDialog")

    # __init__'s read of direct_text_glossary_override_mode
    init_block = _dedent4(U7_DIALOG_EDITS["__init__"][0][0])
    assert init_block in legacy["__init__"]
    read_ns = {}
    exec("def read(translator):\n" + init_block + "    return configured_glossary_override\n", read_ns)
    values = ["", None, "none", "NONE ", " manual", "no_glossary", "attachments_only", "bogus", 0, 1, False, True]
    for value in values:
        translator = types.SimpleNamespace(config={} if value is None else {"direct_text_glossary_override_mode": value})
        assert dts.configured_glossary_override_mode(
            translator.config.get("direct_text_glossary_override_mode", "")) == read_ns["read"](translator), value

    # _on_glossary_override_toggled's config writes
    toggled_ns = {}
    exec(legacy["_on_glossary_override_toggled"], toggled_ns)
    for mode in values:
        config, saves = {"keep": 1}, []
        owner = types.SimpleNamespace(translator=types.SimpleNamespace(
            config=config, save_config=lambda show_message=True: saves.append(show_message)))
        toggled_ns["_on_glossary_override_toggled"](owner, mode, True)
        expected = {"keep": 1}
        expected.update(dts.glossary_override_config_updates(mode))
        assert config == expected and list(config) == list(expected) and saves == [False], mode

    # _rename_chat's title rule
    for title in ("  Renamed   chat  " + "x" * 200, "", None, "a\tb\nc", 42):
        assert dts.chat_rename_title(title) == " ".join(str(title or "").split())[:120]

    # the "Provide Manual Glossary" dialog's _accept (path record / sniffed content record)
    manual = legacy["_request_direct_text_manual_glossary"]
    start = manual.index("    def _accept():\n")
    end = manual.index("    browse_button.clicked.connect(_browse)")
    accept_src = textwrap.dedent(manual[start:end])
    contents = ["", "   ", "[1, 2]", "{\"a\": 1}", "{not json", "[broken", "a,b\nc,d", "a,b", "\n\nraw,x\n\nr2",
                "one line", "x\ny", "  {\"k\": [1]}  ", "名前,訳\n김,Kim"]
    for content in contents:
        for source_path, source_text, ext in (("", "", ".txt"), ("C:/g/book.csv", content, ".csv"),
                                              ("C:/g/book.json", "other", ".json")):
            ns = {"json_lib": json, "result": {}, "QMessageBox": types.SimpleNamespace(information=lambda *a: None)}
            ns["editor"] = types.SimpleNamespace(toPlainText=lambda c=content: c, source_path=source_path,
                                                 source_text=source_text, source_extension=ext)
            ns["dialog"] = types.SimpleNamespace(accept=lambda: None)
            exec(accept_src + "\n_accept()\n", ns)
            record = dts.manual_glossary_source_record(content, source_path, source_text, ext)
            assert (record or {}) == ns["result"], (content, source_path)
    assert dts.MANUAL_GLOSSARY_EXTENSIONS == {".csv", ".json", ".txt", ".md"}


# ---------------------------------------------------------------------------
# tier I: import hygiene
# ---------------------------------------------------------------------------

def test_modules_import_without_qt_or_the_gui():
    from parity import import_hygiene

    results = import_hygiene.check_imports(["direct_text_store", "direct_text_stream"],
                                           forbidden=("translator_gui", "dpi_setup", "epub_library"))
    for module, result in results.items():
        assert result.ok, result.describe()
        assert not result.qt_attempts, (module, result.qt_attempts)


def test_modules_parse_as_python_310():
    for module in ("direct_text_store", "direct_text_stream"):
        ast.parse(_src_text(module), feature_version=(3, 10))
        for node in ast.walk(ast.parse(_src_text(module))):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else (
                [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            for name in names:
                assert not name.startswith(("PySide6", "translator_gui", "dpi_setup", "epub_library")), (module, name)


# ---------------------------------------------------------------------------
# tiers P and H: the real dialog (BASE_SHA vs working tree) and the GUI-free hosts
# ---------------------------------------------------------------------------

def _probe_results():
    import pytest

    cached = getattr(_probe_results, "cache", None)
    if cached is not None:
        return cached
    pytest.importorskip("PySide6")
    legacy_text = _legacy_tg_text()
    root = Path(tempfile.mkdtemp(prefix="dtprobe_"))
    try:
        legacy_dir = root / "legacy_tg"
        legacy_dir.mkdir()
        (legacy_dir / "translator_gui.py").write_text(legacy_text, encoding="utf-8")
        if (SRC / "Halgakos.ico").exists():
            shutil.copy2(SRC / "Halgakos.ico", legacy_dir / "Halgakos.ico")
        fixtures = root / "fixtures"
        build_fixtures(fixtures)
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(3) as pool:
            jobs = {
                "legacy": pool.submit(_run_probe, "legacy", legacy_dir, fixtures, root),
                "current": pool.submit(_run_probe, "current", None, fixtures, root),
                "headless": pool.submit(_run_probe, "headless", None, fixtures, root, True),
            }
            results = {label: job.result() for label, job in jobs.items()}
    finally:
        shutil.rmtree(root, ignore_errors=True)
    _probe_results.cache = results
    return results


def test_dialog_parity_with_the_base_commit():
    results = _probe_results()
    legacy, current = dict(results["legacy"]), dict(results["current"])
    assert legacy.pop("translator_gui").replace("\\", "/") == "<TG>/translator_gui.py"
    assert current.pop("translator_gui").replace("\\", "/").endswith("src/translator_gui.py")
    assert sorted(legacy) == sorted(current)
    difference = _first_difference(legacy, current)
    assert difference is None, f"dialog differs from {BASE_SHA[:12]} at {difference[0]}:\n{difference[1]!r}\n!=\n{difference[2]!r}"


def test_dialog_matches_the_captured_golden():
    """The working-tree dialog equals the stored BASE_SHA golden (``--capture-golden``; local only)."""
    import pytest

    if not GOLDEN_PATH.is_file():
        pytest.skip(f"{GOLDEN_PATH.name} not captured (python tests/test_direct_text_core.py --capture-golden)")
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    current = golden_view(_probe_results()["current"])
    assert sorted(golden) == sorted(current)
    difference = _first_difference(golden, current)
    assert difference is None, f"dialog differs from the golden at {difference[0]}:\n{difference[1]!r}\n!=\n{difference[2]!r}"


def test_dialog_probe_exercises_the_moved_code():
    legacy = _probe_results()["legacy"]
    assert not [label for label, value in legacy.items() if isinstance(value, dict) and "exception" in value]
    expected = {"history_synth", "helpers", "edit"} | {f"migrate_{v}" for v in MIGRATE_VARIANTS} | {
        f"send_{case['name']}" for case in SEND_CASES} | {
        f"stream_{n}_{a}_{c}" for n in STREAMS for a in ("txt", "att") for c in ("plain", "phase", "completion")}
    assert expected <= set(legacy)
    epub = legacy["send_epub"]["after_finish"]["messages"]
    labels = [m[5] for m in epub if m[0] == "assistant"]
    assert labels[-2:] == ["Extraction report", "Attachment actions"]
    assert "Header / TOC translation · Request 4" in labels
    assert legacy["migrate_merge"]["result"] is True and legacy["migrate_cancel"]["result"] is False
    assert len(legacy["stream_epub_att_plain"]["segments"]) >= 5
    assert "Chat Messages/" in json.dumps(legacy["history_synth"]["saved_tree"], ensure_ascii=False)


def test_gui_free_hosts_match_the_dialog():
    results = _probe_results()
    legacy, headless = results["legacy"], results["headless"]
    errors = {label: value for label, value in headless.items() if isinstance(value, dict) and "exception" in value}
    assert not errors, json.dumps(errors, ensure_ascii=False, indent=1)[:4000]
    compared = set()
    for label, observation in legacy.items():
        expected = headless_view(label, observation)
        if expected is None:
            continue
        actual = headless_view(label, headless.get(label))
        difference = _first_difference(expected, actual)
        assert difference is None, f"{label}: GUI-free host differs at {difference[0]}:\n{difference[1]!r}\n!=\n{difference[2]!r}"
        compared.add(label)
    assert {"history_synth", "edit", "send_epub", "send_image_mode", "migrate_merge",
            "stream_glossary_att_phase"} <= compared
    # 4-5 histories + 18 streams + 9 sends + 6 migrations + edit (history_real only with a local history)
    assert len(compared) == len([label for label in legacy if headless_view(label, legacy[label]) is not None]) >= 38
    # Migrate notices are the dialog's message boxes
    for variant in MIGRATE_VARIANTS:
        boxes = [tuple(call) for call in legacy[f"migrate_{variant}"]["calls"] if call[0] in ("information", "warning")]
        assert [tuple(n) for n in headless[f"migrate_{variant}"]["notices"]] == boxes, variant


def test_gui_free_hosts_never_load_qt():
    assert _probe_results()["headless"]["qt_loaded"] == []


# ---------------------------------------------------------------------------
# tier U: the public host API the mobile chat uses
# ---------------------------------------------------------------------------

def _history(tmp_path):
    return tmp_path / "data" / "direct_text_chats.json"


def test_chat_store_round_trip_and_externalised_bodies(tmp_path, monkeypatch):
    import direct_text_store as dts

    monkeypatch.delenv("OUTPUT_DIRECTORY", raising=False)
    store = dts.ChatStore(_history(tmp_path), output_root=tmp_path / "out")
    sessions, current = store.load_chat_history()
    assert current == 1 and len(sessions) == 1 and sessions[0]["title"] == "New chat"
    session = sessions[0]
    assert store.title_chat_from_text(session, "  A   very long first message " + "x" * 60) == (
        "A very long first message " + "x" * 15 + "…")
    session["messages"].extend([("user", "hi"), ("assistant", "Hello **there**", "thought", "Processing", "", "Request 1",
                                                 {"created_at": "2025-01-02T03:04:05+00:00"})])
    store.save_chat_history(sessions, current)
    payload = json.loads(_history(tmp_path).read_text(encoding="utf-8"))
    assert payload["version"] == 2 and payload["current_chat_id"] == 1
    stored = payload["sessions"][0]["messages"][1]
    assert stored[1] == "" and stored[6]["content_path"].endswith("Chat Messages/000002-response.md")
    folder = Path(session["output_folder"])
    assert folder.parent == tmp_path / "out" / "Direct Text"
    assert (folder / "Chat Messages" / "000002-response.md").read_text(encoding="utf-8") == "Hello **there**"
    assert (folder / "Chat Messages" / "000002-thinking.md").read_text(encoding="utf-8") == "thought"
    assert {p.suffix for p in (folder / "Chat Messages").glob("000002-response.*")} == {".md", ".txt", ".html", ".xhtml"}
    assert store.assistant_message_text(session, 1) == "Hello **there**"
    assert store.assistant_message_text(session, 1, "thinking") == "thought"
    reloaded, current_again = dts.ChatStore(_history(tmp_path)).load_chat_history()
    assert current_again == 1 and reloaded[0]["messages"] == session["messages"]
    assert store.validated_chat_output_folder(session) == os.path.realpath(str(folder))
    assert dts.SIDECAR_NAME == "direct_text_chats.mobile.json"


def test_chat_store_names_the_mobile_binding_resolves(tmp_path):
    import inspect

    import direct_text_store as dts

    store = dts.ChatStore(_history(tmp_path))
    for name in ("load_chat_history", "save_chat_history", "new_chat_session", "resolve_history_file_reference",
                 "ensure_conversation_output_folder_for_session", "validated_chat_output_folder",
                 "conversation_attachment_folders", "save_response_output_edit", "finish_run", "migrate_attachment"):
        assert callable(getattr(store, name)), name
    for name in ("save_response_output_edit", "finish_run", "migrate_attachment", "conversation_attachment_folders",
                 "ensure_conversation_output_folder_for_session"):
        params = [p for p in inspect.signature(getattr(store, name)).parameters.values() if p.name != "self"]
        assert params[0].name == "session", name
    # direct_text_rules.shared_rule() looks these up as module functions
    assert dts.format_attachment_size(2048) == "2.0 KB"
    assert dts.force_no_glossary_for_mode("attachments_only", False) is True
    assert dts.history_window_bounds(10, 4, 0) == (0, 4)
    assert "<strong>x</strong>" in dts.markup_to_html("**x**")
    assert dts.normalize_rendered_card_limit(999) == 200
    assert dts.timestamp_label("bad") == ""


def test_stream_entry_points_the_mobile_app_looks_up(monkeypatch):
    import direct_text_stream as dtm

    stream = dtm.RequestStream()  # JobService: no-arg construction, feed(line), ordered_segments()
    stream.feed("🚀 [Thread-2 (api_call)] Sending API call now: Chapter 1")
    stream.feed("📡 [Thread-2 (api_call)] Text streaming...", "Thread-2 (api_call)")
    stream.feed("Hello", "Thread-2 (api_call)")
    segments = stream.ordered_segments()
    assert [s["content"] for s in segments] == ["Hello\n"] and segments[0]["label"] == "Request 1"
    model = dtm.make_stream(source_is_attachment=True, request_number=4, model="gpt-4o")
    assert model.on_log_line("[spine-order:2] Chapter 2 · ch002.xhtml Direct Text dispatch",
                             source_thread="Thread-9 (api_call)") is None
    assert model.drain(final=True) is True and model.drain(final=True) is False
    assert model.active_request_segments[0]["label"] == "Chapter 2 · ch002.xhtml"
    message = model.request_segment_message(model.active_request_segments[0], "out")
    assert message[:6] == ("assistant", "", "", "Processing", "out", "Chapter 2 · ch002.xhtml · Request 4")
    assert dtm.request_segment_message(model.active_request_segments[0], "out")[5] == message[5]
    payload_hint = model.feed('[DIRECT_TEXT_RESPONSE_PAYLOAD] {"label": "x", "content": "y"}', "Thread-9 (api_call)")
    assert payload_hint == "suppress-main-log"
    model.commit_active_request_phase()
    assert model.chat_messages and model.active_request_segments == []


def test_count_tokens_follows_the_desktop_encoder_fallbacks(monkeypatch):
    import direct_text_stream as dtm

    calls = []

    class Encoder:
        def __init__(self, name):
            self.name = name

        def encode(self, text, disallowed_special=()):
            return list(text)

    fake = types.ModuleType("tiktoken")

    def encoding_for_model(name):
        calls.append(("model", name))
        raise KeyError(name)

    def get_encoding(name):
        calls.append(("encoding", name))
        if name == "o200k_base":
            raise ValueError("no o200k")
        return Encoder(name)

    fake.encoding_for_model = encoding_for_model
    fake.get_encoding = get_encoding
    monkeypatch.setitem(sys.modules, "tiktoken", fake)
    monkeypatch.setattr(dtm, "_TOKEN_COUNTERS", {})
    assert dtm.count_tokens("abc​", "openai/gpt-x") == 3
    assert calls == [("model", "openai/gpt-x"), ("model", "gpt-x"), ("encoding", "o200k_base"),
                     ("encoding", "cl100k_base")]
    assert dtm.count_tokens("   ", "openai/gpt-x") == 0


def test_apply_direct_text_run_environment_matches_the_dialog_order(monkeypatch, tmp_path):
    import direct_text_store as dts

    for key in ("OUTPUT_DIRECTORY", "OUTPUT_DIR", "DIRECT_TEXT_ACTIVE", "DIRECT_TEXT_PRESERVE_MARKUP",
                "DIRECT_TEXT_ORDERED_BATCH", "ORDER_BATCH_REQUESTS_BY_SPINE"):
        monkeypatch.setenv(key, "sentinel")  # registers the original value (or absence) for restoring
        monkeypatch.delenv(key)
    calls = []
    owner = types.SimpleNamespace(
        _apply_forced_streaming_environment=lambda: calls.append(("forced", os.environ.get("DIRECT_TEXT_ORDERED_BATCH"))),
        _apply_direct_text_runtime_environment=lambda: calls.append(("runtime", os.environ.get("OUTPUT_DIR"))),
    )
    dts.apply_direct_text_run_environment(owner, str(tmp_path), False)
    assert calls == [("forced", "0"), ("runtime", str(tmp_path))]
    assert os.environ["OUTPUT_DIRECTORY"] == str(tmp_path) and os.environ["DIRECT_TEXT_ACTIVE"] == "1"
    assert "ORDER_BATCH_REQUESTS_BY_SPINE" not in os.environ
    dts.apply_direct_text_run_environment(owner, str(tmp_path), True)
    assert os.environ["DIRECT_TEXT_ORDERED_BATCH"] == "1" and os.environ["ORDER_BATCH_REQUESTS_BY_SPINE"] == "1"


def test_mobile_direct_text_job_uses_the_shared_options_and_env():
    """The mobile job kind applies the shared options, then the shared env helper (no local env copy)."""
    source = (SRC / "mobile" / "app" / "glossarion_mobile" / "job_kinds" / "direct_text.py").read_text(encoding="utf-8")
    import direct_text_store

    assert callable(direct_text_store.apply_direct_text_run_environment)
    assert "from direct_text_store import apply_direct_text_run_environment" in source
    assert "DirectTextRunOptions" in source and "options.apply_to(owner)" in source
    assert source.index("options.apply_to(owner)") < source.index("apply_run_environment(owner, params)")
    assert 'params.get("env")' not in source  # the dialog's env block is not mirrored any more


def test_prepare_direct_text_input(tmp_path, monkeypatch):
    import direct_text_store as dts

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    typed = dts.prepare_direct_text_input("Hello there")
    assert not typed["is_attachment"] and typed["source_extension"] == ".txt"
    assert re.fullmatch(r"direct_text_\d{8}_\d{6}_[0-9a-f]{8}\.txt", os.path.basename(typed["temp_input"]))
    assert Path(typed["temp_input"]).read_text(encoding="utf-8") == "Hello there"
    stem = Path(typed["temp_input"]).stem
    assert typed["expected_output"] == os.path.join(typed["temp_root"], stem, f"{stem}_translated.txt")

    notes = tmp_path / "notes.md"
    notes.write_text("﻿# Notes", encoding="utf-8")
    adapted = dts.prepare_direct_text_input("instr", {"path": str(notes)},
                                            {"kind": "content", "content": "a,b\nc,d", "extension": ".CSV"})
    assert adapted["is_attachment"] and adapted["temp_input"].endswith("notes.txt")
    assert Path(adapted["temp_input"]).read_text(encoding="utf-8") == "# Notes"
    assert adapted["manual_glossary_path"].endswith("direct_text_manual_glossary.csv")

    subs = tmp_path / "ep.srt"
    subs.write_text("1\n", encoding="utf-8")
    passthrough = dts.prepare_direct_text_input("", {"path": str(subs)})
    assert passthrough["temp_input"] == str(subs) and passthrough["expected_output"].endswith("ep_translated.srt")

    import pytest

    with pytest.raises(FileNotFoundError):
        dts.prepare_direct_text_input("x", None, {"kind": "path", "path": str(tmp_path / "missing.csv")})


def test_prepare_direct_text_input_mobile_knobs(tmp_path, monkeypatch):
    """The mobile chat's temp parent (resumable run roots) and extra pass-through types."""
    import direct_text_store as dts

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "os_tmp"))
    (tmp_path / "os_tmp").mkdir()
    runs = tmp_path / "app" / "direct_text_runs"
    typed = dts.prepare_direct_text_input("Hello", temp_parent=runs)
    assert Path(typed["temp_root"]).parent == runs and Path(typed["temp_root"]).name.startswith("glossarion_input_output_")
    book = tmp_path / "book.zip"
    book.write_bytes(b"PK\x03\x04")
    adapted = dts.prepare_direct_text_input("", {"path": str(book)})  # desktop: a .zip is read as text
    assert adapted["temp_input"].endswith("book.txt") and Path(adapted["temp_root"]).parent == tmp_path / "os_tmp"
    passed = dts.prepare_direct_text_input("", {"path": str(book)}, temp_parent=runs,
                                           extra_pass_through_extensions=(".ZIP", ".sdlxliff"))
    assert passed["temp_input"] == str(book) and passed["expected_output"].endswith(os.path.join("book", "book_translated.txt"))
    # the class defaults stay the dialog's (the knobs live on the throwaway host only)
    assert dts.ChatStoreMixin._DIRECT_TEXT_TEMP_PARENT is None and not dts.ChatStoreMixin._EXTRA_PASS_THROUGH_EXTENSIONS


def test_commit_request_phase_freezes_the_gate_cards_into_the_chat(tmp_path, monkeypatch):
    """ChatStore.commit_request_phase == the dialog's _commit_active_request_phase on the chat."""
    import direct_text_store as dts
    import direct_text_stream as dtm

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    store = dts.ChatStore(_history(tmp_path), output_root=tmp_path / "out")
    session = store.current_session()
    session["messages"].append(("user_file", "book.epub", str(tmp_path / "book.epub"), 4, "", "user"))
    lines = ("🚀 [Thread-2 (api_call)] Sending API call now", "📡 [Thread-2 (api_call)] Text streaming...",
             "type,raw_name,translated_name")
    stream = dtm.make_stream(source_is_attachment=True, model="gpt-4o")
    reference = dtm.make_stream(source_is_attachment=True, model="gpt-4o")
    for line in lines:
        stream.feed(line, "Thread-2 (api_call)")
        reference.feed(line, "Thread-2 (api_call)")
    expected_segments = reference.segments()
    assert expected_segments and any(str(s.get("content") or "").strip() for s in expected_segments)
    folder_at_gate = session.get("output_folder", "")  # the dialog's _last_output_folder at the gate
    committed = store.commit_request_phase(session, stream)
    expected = []
    for segment in expected_segments:
        segment = dict(segment, complete=True, phase="processing")
        expected.append(reference.request_segment_message(segment, folder_at_gate))
    assert [m[:6] for m in committed] == [m[:6] for m in expected]
    assert len(session["messages"]) == 1 + len(committed)
    assert stream.segments() == [] and stream.streamed_content == ""  # the next phase starts empty
    saved = json.loads(_history(tmp_path).read_text(encoding="utf-8"))
    assert len(saved["sessions"][0]["messages"]) == 1 + len(committed)


def test_extraction_report_and_action_builders_match_the_legacy_methods(tmp_path):
    import random

    import direct_text_stream as dtm

    legacy = _member_texts(_legacy_tg_text(), "_InputOutputDialog")
    namespace = {"os": os}
    for name in ("_build_attachment_action_card", "_build_attachment_extraction_summary",
                 "_is_expected_cover_chapter", "_preferred_attachment_compiled_documents"):
        exec(compile(legacy[name], f"<legacy {name}>", "exec"), namespace)
    rng = random.Random(7)
    for case in range(120):
        folder = tmp_path / f"case{case}"
        folder.mkdir()
        artifacts = {}
        if rng.random() < 0.7:
            artifacts["metadata.json"] = json.dumps({
                k: v for k, v in {
                    "extraction_mode": rng.choice(["enhanced", "standard", "", None]),
                    "detected_language": rng.choice(["korean", "", None]),
                    "chapter_count": rng.choice([0, 3, "x", None]),
                    "chapter_payloads_ready": rng.choice([3, 2, "bad", None]),
                    "extracted_resources": rng.choice([{"a": [1, 2], "b": 3, "c": -1}, [], None]),
                }.items() if rng.random() < 0.8})
        if rng.random() < 0.7:
            artifacts["chapters_full.json"] = json.dumps([
                {"body": rng.choice(["<p>x</p>", None]), "file_size": rng.choice([10, 600, "x", 49]),
                 "has_images": rng.random() < 0.5, "is_image_only": rng.random() < 0.3,
                 "filename": rng.choice(["cover.xhtml", "ch1.xhtml", "Title_Page.html"])}
                for _ in range(rng.randint(0, 5))] + [rng.choice(["junk", 3])])
        if rng.random() < 0.6:
            artifacts["extraction_report.txt"] = "X\nPOTENTIAL ISSUES:\n" + "\n".join(
                rng.choice(["  • 3 chapters contain only images", "  • None detected", "  • Odd", "  •"])
                for _ in range(rng.randint(0, 6)))
        for name, text in artifacts.items():
            (folder / name).write_text(text, encoding="utf-8")
        for name in rng.sample(["a.epub", "b.epub", "c.pdf"], rng.randint(0, 3)):
            (folder / name).write_bytes(b"x")
        segments = [{"thinking_tokens": rng.randint(0, 9), "text_tokens": rng.choice([1, "", None])}
                    for _ in range(rng.randint(0, 3))]
        state = dict(_run_source_is_attachment=rng.random() < 0.9, _run_source_path=rng.choice(["", "x/B.epub"]),
                     _active_request_segments=segments, _run_started_at=0.0)
        legacy_self = types.SimpleNamespace(
            **state, _is_expected_cover_chapter=namespace["_is_expected_cover_chapter"],
            _preferred_attachment_compiled_documents=namespace["_preferred_attachment_compiled_documents"],
            _find_direct_run_artifact=lambda name, folder=folder: str(folder / name) if (folder / name).exists() else "")
        new_self = types.SimpleNamespace(**vars(legacy_self))
        output = str(folder) if rng.random() < 0.9 else ""
        for name in ("_build_attachment_action_card", "_build_attachment_extraction_summary"):
            expected = namespace[name](legacy_self, output)
            actual = getattr(dtm.DirectTextStreamMixin, name)(new_self, output)
            assert expected == actual, (case, name)
        if state["_run_source_is_attachment"] and output:
            report = dtm.attachment_extraction_report(
                *(new_self._find_direct_run_artifact(n) for n in ("extraction_report.txt", "metadata.json",
                                                                  "chapters_full.json")),
                segments, 0.0, state["_run_source_path"])
            if report is not None:
                assert dtm.extraction_report_card(report, output) == namespace[
                    "_build_attachment_extraction_summary"](legacy_self, output)


def test_finish_run_persists_a_text_turn_like_the_dialog(tmp_path, monkeypatch):
    import direct_text_store as dts
    import direct_text_stream as dtm

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    store = dts.ChatStore(_history(tmp_path), output_root=tmp_path / "out")
    session = store.current_session()
    session["messages"].append(("user", "Hi"))
    run = dts.prepare_direct_text_input("Hi")
    Path(run["expected_output"]).parent.mkdir(parents=True)
    Path(run["expected_output"]).write_text("Bonjour", encoding="utf-8")
    stream = dtm.make_stream(model="gpt-4o")
    for line in ("🚀 [Thread-2 (api_call)] Sending API call now", "📡 [Thread-2 (api_call)] Text streaming...", "Bonj"):
        stream.feed(line, "Thread-2 (api_call)")
    result = store.finish_run(session, run, stream)
    assert result["status"] == "Ready"
    folder = Path(result["output_folder"])
    assert (folder / "Direct Text 1.txt").read_text(encoding="utf-8") == "Bonjour"
    assert session["next_output_index"] == 2
    message = session["messages"][-1]
    assert message[0] == "assistant" and message[5] == "Request 1" and message[3].startswith("Token summary")
    assert store.assistant_message_text(session, 1) == "Bonjour"
    assert not os.path.exists(run["temp_root"])  # cleaned like the dialog's _restore_run_context
    saved = json.loads(_history(tmp_path).read_text(encoding="utf-8"))
    assert saved["sessions"][0]["messages"][-1][6]["content_path"].endswith("000002-response.md")


def test_migrate_attachment_reports_the_dialog_notices(tmp_path, monkeypatch):
    import direct_text_store as dts

    monkeypatch.delenv("OUTPUT_DIRECTORY", raising=False)
    monkeypatch.delenv("OUTPUT_DIR", raising=False)
    root = tmp_path / "out"
    store = dts.ChatStore(_history(tmp_path), output_root=root, config={"output_directory": str(root)})
    session = store.current_session()
    folder = Path(store.ensure_conversation_output_folder_for_session(session))
    workspace = folder / "Attachments" / "book"
    (workspace / "nested").mkdir(parents=True)
    (workspace / "book.epub").write_bytes(b"new")
    (root / "book").mkdir(parents=True)
    (root / "book" / "book.epub").write_bytes(b"old")
    session["messages"].append(("assistant", "x", "", "Completed", str(workspace), "Attachment actions"))
    assert store.attachment_card_actions(session, str(workspace)) == ["migrate", "reader"]
    cancelled = store.migrate_attachment(session, str(workspace))
    assert cancelled == {"ok": False, "notices": []} and workspace.is_dir()
    asked = []
    merged = store.migrate_attachment(session, str(workspace), confirm_merge=lambda target: asked.append(target) or True)
    assert merged["ok"] and asked == [str(root / "book")]
    assert merged["notices"] == [{"level": "information", "title": "Attachment migrated",
                                  "text": f"The attachment workspace was moved to:\n{root / 'book'}"}]
    assert (root / "book" / "book.epub").read_bytes() == b"new" and not workspace.exists()
    assert session["messages"][-1][4] == os.path.normpath(str(root / "book"))
    again = store.migrate_attachment(session, str(workspace))
    assert again["notices"][0]["title"] == "Attachment unavailable"


def test_mobile_chat_run_env_on_a_headless_owner(tmp_path, monkeypatch):
    """The mobile direct_text job: DirectTextRunOptions + the shared env helper on a real HeadlessOwner
    give the env the dialog's own lines (BASE_SHA) give."""
    from _headless_env import headless_owner
    from headless_owner import DirectTextRunOptions

    import direct_text_store as dts
    from run_env import FORCED_STREAM_ENV_KEYS

    send = _member_texts(_legacy_tg_text(), "_InputOutputDialog")["_start_translation"]
    env_start = send.index("        os.environ['OUTPUT_DIRECTORY'] = self._temp_root")
    env_end = send.index("gui._apply_direct_text_runtime_environment()\n", env_start) + len(
        "gui._apply_direct_text_runtime_environment()\n")
    legacy_lines = compile(textwrap.dedent(send[env_start:env_end]), "<dialog env block>", "exec")
    keys = sorted(set(dts.ChatStoreMixin._OUTPUT_ENV_KEYS) | set(FORCED_STREAM_ENV_KEYS)
                  | set(dts.ChatStoreMixin._DIRECT_TEXT_ENV_KEYS))
    root = str(tmp_path / "run")
    config = {"model": "gpt-4o", "enable_gpt_thinking": True, "thinking_budget": 900,
              "direct_text_disable_thinking": True}
    views = []
    for attachment in (True, False):
        for use_helper in (False, True):
            with headless_owner(tmp_path / "owner", monkeypatch, config) as owner:
                DirectTextRunOptions(selected_files=[str(tmp_path / "in.txt")], force_no_glossary=not attachment,
                                     skip_thinking=True, attachment_prompt="Be literal" if attachment else "",
                                     attachment_prompt_role="system", output_mode="text").apply_to(owner)
                if use_helper:
                    dts.apply_direct_text_run_environment(owner, root, attachment)
                else:
                    dialog = types.SimpleNamespace(_temp_root=root, _run_source_is_attachment=attachment)
                    exec(legacy_lines, {"os": os, "self": dialog, "gui": owner})
                views.append({key: os.environ.get(key) for key in keys})
        assert views[-1] == views[-2], attachment
        assert views[-1]["DIRECT_TEXT_ACTIVE"] == "1" and views[-1]["OUTPUT_DIRECTORY"] == root
    # the run-time overrides really ran on the owner (_apply_direct_text_runtime_environment)
    assert views[0]["ENABLE_GPT_THINKING"] == "0" and views[0]["DIRECT_TEXT_ATTACHMENT_PROMPT_ROLE"] == "system"
    assert views[0]["ORDER_BATCH_REQUESTS_BY_SPINE"] == "1" and views[2]["DIRECT_TEXT_ORDERED_BATCH"] == "0"


if __name__ == "__main__":
    sys.exit(_main())
