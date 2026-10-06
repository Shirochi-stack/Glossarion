"""Tier T scenarios: desktop pipeline order and stop races driven through stubbed backends.

A scenario is plain data plus a few callables (see ``trace_harness`` for how it runs):

``config``        config.json the owner boots from (frozen legacy boot, like the goldens);
                  ``None`` = a true fresh install. Strings may contain ``<SANDBOX>``.
``files``         {path relative to the sandbox root: text | bytes} created before boot.
``run_attrs``     owner attributes set after boot (``selected_files``, stop settings, ...);
                  a dict or ``callable(sandbox) -> dict``.
``pre_run``       optional ``callable(owner, ctx)`` run on each side before the entry (the
                  Direct Text dialog's per-run setup).
``entry``         ``'translate'`` (desktop: the Run button = ``run_translation_thread``;
                  mobile: ``_prepare_translation_run`` + ``_translation_worker``) or
                  ``'glossary'`` (``run_glossary_extraction_direct``).
``plan``          per backend entry point, the behaviour of its 1st, 2nd, ... call
                  (:class:`CallPlan`); calls beyond the list use ``CallPlan()``.
``answers``       answers to the Direct Text glossary approval question, in order.
``post_actions``  actions after the run drained (e.g. a late second click).
``modes``         new-code modes this scenario is compared in (``desktop``: working-tree
                  TranslatorGUI MRO; ``mixins``: shared mixins + stop_control, the mobile
                  path). The legacy self-consistency tests run every scenario.
``expect``        ``callable(TraceView) -> [problem, ...]``: what the frozen desktop must do
                  in this scenario (proves the harness really reached the paths it claims).

Actions (``CallPlan.actions`` / ``post_actions``): ``('click',)`` presses Run/Stop (desktop:
``run_translation_thread()`` while the worker is alive, i.e. ``stop_translation``; mobile:
``stop_control`` with a ``StopClickTracker`` double-tap), ``('advance', seconds)`` moves the
fixed clock.
"""

from __future__ import annotations

import io
import json
import os
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parents[1]
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

TRANSLATE = "translate"
GLOSSARY = "glossary"

#: backend entry points stubbed by the harness (names used in plans and traces)
TRANSLATION_MAIN = "TransateKRtoEN.main"
GLOSSARY_MAIN = "extract_glossary_from_epub.main"
METADATA_JOB = "metadata_translation_worker.run_metadata_translation_job"
EPUB_COMPILE = "epub_converter.compile_epub"
PDF_COMPILE = "pdf_workspace_compiler.compile_pdf_workspace"
#: U7: the client calls of the image / generative-only / RPG Maker runners (image_job, rpgmaker_job)
CLIENT_SEND = "UnifiedClient.send"
IMAGE_SEND = "UnifiedClient.send_image"
SEND_INTERRUPT = "TransateKRtoEN.send_with_interrupt"
GAME_IMAGES = "rpgmaker_handler.translate_game_images"

ALL_MODES = ("desktop", "mixins")
DESKTOP_ONLY = ("desktop",)


@dataclass(frozen=True)
class CallPlan:
    """What one call of a stubbed backend entry point does.

    ``chunks``    simulated work units; before each one the stub asks ``stop_callback()``
                  and records the stop flags it sees, and stops when asked to.
    ``actions``   {chunk number: (action, ...)} fired after that chunk's log line.
    ``result``    return value of the call.
    ``glossary``  (glossary stub) write the generated glossary at ``OUTPUT_PATH``.
    ``progress``  (glossary stub) ``<book>_glossary_progress.json`` written next to it.
    """

    chunks: int = 2
    actions: dict = field(default_factory=dict)
    result: object = None
    glossary: bool = True
    progress: dict | None = None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_CONTAINER = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">\n'
    '  <rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles>\n'
    '</container>\n'
)


def build_epub(title: str, chapters: int = 3) -> bytes:
    """A small valid EPUB 3 (stored mimetype first, OPF spine, Korean chapter text)."""
    items, spine = [], []
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(zipfile.ZipInfo("mimetype"), "application/epub+zip")
        zf.writestr("META-INF/container.xml", _CONTAINER)
        for i in range(1, chapters + 1):
            name = f"chapter{i:04d}.xhtml"
            items.append(f'<item id="c{i}" href="Text/{name}" media-type="application/xhtml+xml"/>')
            spine.append(f'<itemref idref="c{i}"/>')
            zf.writestr(
                f"OEBPS/Text/{name}",
                '<?xml version="1.0" encoding="utf-8"?>\n'
                '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>'
                f"제{i}장</title></head><body><h1>제{i}장</h1>"
                f"<p>김상현은 {i}번째 문을 열었다.</p></body></html>",
            )
        opf = (
            '<?xml version="1.0" encoding="utf-8"?>\n'
            '<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="uid">'
            '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
            f'<dc:identifier id="uid">trace-{title}</dc:identifier><dc:title>{title}</dc:title>'
            '<dc:language>ko</dc:language></metadata>'
            f'<manifest>{"".join(items)}</manifest><spine>{"".join(spine)}</spine></package>'
        )
        zf.writestr("OEBPS/content.opf", opf)
    return buf.getvalue()


_GLOSSARY_CSV = "type,raw_name,translated_name,gender\ncharacter,김상현,Kim Sang-hyun,male\n"
OUT = "<SANDBOX>/outputs"


def _qa_progress_fixture() -> str:
    """translation_progress.json with one foreign-text QA failure and one other QA failure."""
    return json.dumps({
        "chapters": {
            "1": {
                "status": "qa_failed",
                "actual_num": 1,
                "output_file": "response_0001_chapter0001.html",
                "original_basename": "chapter0001.xhtml",
                "qa_issues_found": ["korean_text_found_12_chars_in_output"],
            },
            "2": {
                "status": "qa_failed",
                "actual_num": 2,
                "output_file": "response_0002_chapter0002.html",
                "original_basename": "chapter0002.xhtml",
                "qa_issues_found": ["TRUNCATED"],
            },
            "3": {
                "status": "completed",
                "actual_num": 3,
                "output_file": "response_0003_chapter0003.html",
                "original_basename": "chapter0003.xhtml",
            },
        },
    }, ensure_ascii=False, indent=2)


def _base_config(**overrides) -> dict:
    cfg = {
        "model": "gpt-4o-mini",
        "api_key": "sk-trace-0000",
        "output_directory": OUT,
        "auto_glossary_mode": "off",
        "auto_update_check": False,
        "batch_translation": False,
    }
    cfg.update(overrides)
    return cfg


def _files(*books, extra=None) -> dict:
    out = {f"inputs/{b}.epub": build_epub(b) for b in books}
    out.update(extra or {})
    return out


def _inputs(*books):
    def attrs(sandbox):
        return {"selected_files": [sandbox.path(f"inputs/{b}.epub") for b in books]}
    return attrs


def _with(attrs_fn, **extra):
    def attrs(sandbox):
        out = dict(attrs_fn(sandbox))
        out.update(sandbox.resolve(extra))
        return out
    return attrs


# ---------------------------------------------------------------------------
# Expectation helpers (operate on trace_harness.TraceView)
# ---------------------------------------------------------------------------


def _expect_entries(*names):
    def check(view):
        got = view.entries()
        return [] if got == list(names) else [f"backend entries {got} != {list(names)}"]
    return check


def _expect_logs(*fragments, absent=()):
    def check(view):
        logs = view.logs()
        problems = [f"log containing {f!r} missing" for f in fragments if not any(f in line for line in logs)]
        problems += [f"unexpected log containing {f!r}" for f in absent if any(f in line for line in logs)]
        return problems
    return check


def _all(*checks):
    def check(view):
        out = []
        for c in checks:
            out.extend(c(view))
        return out
    return check


def _expect_env(entry, index, **expected):
    def check(view):
        env = view.entry_env(entry, index)
        if env is None:
            return [f"no call #{index} of {entry}"]
        return [f"{entry}#{index} env {k}={env.get(k)!r}, expected {v!r}"
                for k, v in expected.items() if env.get(k) != v]
    return check


def _expect_env_endswith(entry, index, key, suffix):
    def check(view):
        env = view.entry_env(entry, index) or {}
        value = env.get(key)
        return [] if isinstance(value, str) and value.replace("\\", "/").endswith(suffix) else [
            f"{entry}#{index} env {key}={value!r} does not end with {suffix!r}"]
    return check


def _expect_after_click(**expected):
    """Stop flags the backend saw at its first stop check after the first click."""
    def check(view):
        seen = view.first_check_after_action()
        if seen is None:
            return ["no backend stop check after the first click"]
        problems = []
        for key, value in expected.items():
            got = view.flag(seen, key)
            if got != value:
                problems.append(f"after click: {key}={got!r}, expected {value!r}")
        return problems
    return check


def _expect_stop_api(*names, absent=(), after_action=True):
    def check(view):
        calls = view.stop_api_calls(after_action=after_action)
        problems = [f"stop API {n} not called" for n in names if n not in calls]
        problems += [f"stop API {n} unexpectedly called" for n in absent if n in calls]
        return problems
    return check


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

STOP_ATTRS = {"graceful_stop_var": True, "wait_for_chunks_var": True}


def _direct_text_pre_run(owner, ctx):
    """_InputOutputDialog._start_translation's per-run setup (translator_gui @ 1719fb59 9312-9348)."""
    from headless_owner import DirectTextRunOptions

    sandbox = ctx.sandbox
    temp_root = sandbox.path("outputs/direct")
    os.makedirs(temp_root, exist_ok=True)
    DirectTextRunOptions(
        selected_files=[sandbox.path("inputs/attachment.txt")],
        force_stream_all=True,
        archive_conversion_dir=os.path.join(temp_root, "_archive_input"),
        force_multipass_off=True,
        force_no_glossary=False,
        manual_glossary_path="",
        skip_thinking=False,
        attachment_prompt="Translate the attached chapter faithfully.",
        attachment_prompt_role="user",
        skip_prompt_profile=False,
        output_mode="text",
    ).apply_to(owner)
    os.environ["OUTPUT_DIRECTORY"] = temp_root
    os.environ["OUTPUT_DIR"] = temp_root
    os.environ["DIRECT_TEXT_ACTIVE"] = "1"
    os.environ["DIRECT_TEXT_PRESERVE_MARKUP"] = "1"
    os.environ["DIRECT_TEXT_ORDERED_BATCH"] = "1"
    os.environ["ORDER_BATCH_REQUESTS_BY_SPINE"] = "1"
    owner._apply_forced_streaming_environment()
    owner._apply_direct_text_runtime_environment()


_DIRECT_TEXT_FILES = {"inputs/attachment.txt": "첫 번째 문단.\n\n김상현은 문을 열었다.\n"}

T, G = TRANSLATION_MAIN, GLOSSARY_MAIN

# ---------------------------------------------------------------------------
# U7: image / video / audio inputs, generative-only runs, RPG Maker games
# ---------------------------------------------------------------------------

#: a 1x1 PNG and a tiny MP4 header
_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000d49444154789c6360000002000100054f6b2a0000000049454e44ae426082")
_MP4 = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + b"\x00" * 32


def _generated(rel, data=_PNG):
    """Client response: the provider wrote a media file into the sandbox and returns its sentinel."""
    def result(tracer, _payload):
        path = tracer.ctx.sandbox.path(rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(data)
        return f"[GENERATED_IMAGE:{path}]"
    return result


def _rpg_echo(_tracer, payload):
    """Client response for an RPG Maker chunk: ``[N] EN(text)`` for each numbered line."""
    import re

    user = next((m.get("content", "") for m in reversed(payload.get("messages") or [])
                 if isinstance(m, dict) and m.get("role") == "user"), "")
    parts = re.split(r"^\[(\d+)\]\s*", str(user), flags=re.M)
    return "\n".join(f"[{parts[i]}] EN({parts[i + 1].strip()})" for i in range(1, len(parts) - 1, 2))


def _mv_game_files(prefix="games/Hero"):
    data = f"{prefix}/www/data"
    return {
        f"{prefix}/Game.exe": b"MZ\x90\x00fake-exe",
        f"{prefix}/www/js/rpg_core.js": "// core\n",
        f"{data}/System.json": json.dumps({"gameTitle": "勇者の冒険", "terms": {
            "basic": ["レベル", "HP"], "commands": ["戦う", "逃げる"], "params": [],
            "messages": {"actionFailure": "%1には効かなかった！"}}}, ensure_ascii=False),
        f"{data}/Actors.json": json.dumps([None, {"id": 1, "name": "留奈", "nickname": "",
                                                 "profile": "元気な少女。"}], ensure_ascii=False),
        f"{data}/Map001.json": json.dumps({"displayName": "始まりの村", "events": [None, {
            "id": 1, "name": "村長", "pages": [{"list": [
                {"code": 401, "parameters": ["ようこそ、旅の方。"]},
                {"code": 102, "parameters": [["はい", "いいえ"], 1]},
                {"code": 0, "parameters": []}]}]}]}, ensure_ascii=False),
    }


def _no_file_selected(owner, _ctx):
    """The main window's input field with nothing selected (create_file_section; HeadlessOwner's shim)."""
    from parity import fakes

    owner.entry_epub = fakes.FakeLineEdit("No file selected")


def _media_inputs(*rels):
    def attrs(sandbox):
        return {"selected_files": [sandbox.path(r) for r in rels]}
    return attrs


_VISION_CFG = {"model": "gpt-4o", "output_mode": "vision", "enable_image_translation": True}
_IMAGE_CFG = {"model": "gpt-4o", "output_mode": "image", "enable_image_translation": True,
              "enable_image_output_mode": True, "image_output_resolution": "2K"}
IS, CS = IMAGE_SEND, CLIENT_SEND

SCENARIOS = {
    "fresh_install_translate": {
        "description": "Default settings (no config.json): authgpt/gpt-6-luna, startup glossary mode 'off'.",
        "config": None,
        "files": _files("Fresh Novel"),
        "run_attrs": _inputs("Fresh Novel"),
        "expect": _all(
            _expect_entries(T),
            _expect_env(T, 0, MODEL="authgpt/gpt-6-luna", AUTO_GLOSSARY_MODE="off"),
            _expect_logs("✅ Translation completed successfully!"),
        ),
    },
    "plain_translate": {
        "description": "One EPUB, glossary off, keyed OpenAI model, output override (mobile layout).",
        "config": _base_config(),
        "files": _files("Plain Novel"),
        "run_attrs": _inputs("Plain Novel"),
        "expect": _all(
            _expect_entries(T),
            _expect_env(T, 0, MODEL="gpt-4o-mini", USE_ASYNC_CHAPTER_EXTRACTION="1"),
            _expect_logs("🚀 Starting translation...", "✅ Translation completed successfully!"),
        ),
    },
    "balanced_pre_glossary": {
        "description": "Balanced auto glossary: forced request merging, glossary extraction, auto-load, translate.",
        "config": _base_config(auto_glossary_mode="balanced"),
        "files": _files("Balanced Novel"),
        "run_attrs": _inputs("Balanced Novel"),
        "expect": _all(
            _expect_entries(G, T),
            _expect_env(G, 0, GLOSSARY_REQUEST_MERGING_ENABLED="1"),
            _expect_env_endswith(T, 0, "MANUAL_GLOSSARY", "Balanced Novel_glossary.json"),
            _expect_logs("📑 Auto Glossary Mode: Balanced", "📑 Auto-loaded generated glossary"),
        ),
    },
    "balanced_glossary_retry": {
        "description": "Balanced extraction leaves a failed chapter; the worker retries it before translating.",
        "config": _base_config(auto_glossary_mode="balanced"),
        "files": _files("Retry Novel"),
        "run_attrs": _inputs("Retry Novel"),
        "plan": {G: [
            CallPlan(progress={"chapter_count": 3, "completed": [0, 2], "failed": [1],
                               "qa_issues_found": {"1": ["TRUNCATED"]}}),
            CallPlan(progress={"chapter_count": 3, "completed": [0, 1, 2]}),
        ]},
        "expect": _all(
            _expect_entries(G, G, T),
            _expect_logs("🔄 Retrying 1 failed glossary chapter(s)"),
        ),
    },
    "require_complete_gate_blocks": {
        "description": "Require-complete glossary gate: extraction left no progress file, translation blocked.",
        "config": _base_config(auto_glossary_mode="balanced",
                               glossary_require_complete_before_translation=True),
        "files": _files("Gate Novel"),
        "run_attrs": _inputs("Gate Novel"),
        "expect": _all(
            _expect_entries(G),
            _expect_env(G, 0, GLOSSARY_REQUIRE_COMPLETE_BEFORE_TRANSLATION="1"),
            _expect_logs("⏸️ Translation blocked for Gate Novel.epub"),
        ),
    },
    "require_complete_gate_passes": {
        "description": "Require-complete glossary gate with a complete progress file: translation proceeds.",
        "config": _base_config(auto_glossary_mode="balanced",
                               glossary_require_complete_before_translation=True),
        "files": _files("Gate Pass Novel"),
        "run_attrs": _inputs("Gate Pass Novel"),
        "plan": {G: [CallPlan(progress={"chapter_count": 3, "completed": [0, 1, 2]})]},
        "expect": _all(
            _expect_entries(G, T),
            _expect_logs(absent=("⏸️ Translation blocked",)),
        ),
    },
    "multipass_refinement_followup": {
        "description": "Partial.b multipass: foreign-text QA failure refined first, then the other QA failure retranslated.",
        "config": _base_config(multipass_mode=True, multipass_refinement_mode="partial.b"),
        "files": _files("Refine Novel", extra={
            "outputs/Refine Novel/translation_progress.json": _qa_progress_fixture(),
        }),
        "run_attrs": _inputs("Refine Novel"),
        "expect": _all(
            _expect_entries(T, T),
            _expect_env(T, 0, OUTPUT_MODE="refinement", MULTIPASS_MODE="1"),
            _expect_logs("running refinement instead of translation",
                         "🚀 Starting regular translation retry for skipped QA-failed entries"),
        ),
    },
    "metadata_only_batch": {
        "description": "Metadata-only run over two EPUBs in batch mode (in-process metadata worker pool).",
        "config": _base_config(auto_glossary_mode="balanced", batch_translation=True, batch_size="2"),
        "files": _files("Meta One", "Meta Two"),
        "run_attrs": _with(_inputs("Meta One", "Meta Two"), _metadata_only_run=True,
                           _metadata_output_roots={}),
        "expect": _all(
            _expect_entries(METADATA_JOB, METADATA_JOB),
            _expect_logs("🌐 Metadata-only mode: skipping auto glossary extraction",
                         "⚡ Metadata batch mode: translating 2 EPUBs"),
        ),
    },
    "single_chapter_filter": {
        "description": "Reader single-chapter run: glossary skipped, SINGLE_CHAPTER_FILTER, forced streaming.",
        "config": _base_config(auto_glossary_mode="balanced", chapter_range="1-3"),
        "files": _files("Reader Novel"),
        "run_attrs": _with(_inputs("Reader Novel"), _single_chapter_filter="OEBPS/Text/chapter0002.xhtml",
                           _force_stream_all=True),
        "expect": _all(
            _expect_entries(T),
            _expect_env(T, 0, SINGLE_CHAPTER_FILTER="OEBPS/Text/chapter0002.xhtml",
                        USE_ASYNC_CHAPTER_EXTRACTION="0", ENABLE_STREAMING="1"),
            _expect_logs("🎯 Single-chapter mode: skipping auto glossary extraction"),
        ),
    },
    "multi_epub_glossary_map": {
        "description": "Two EPUBs, each with its own mapped manual glossary (minimal mode, no pre-pass).",
        "config": _base_config(auto_glossary_mode="minimal"),
        "files": _files("Map One", "Map Two", extra={
            "inputs/map_one_glossary.csv": _GLOSSARY_CSV,
            "inputs/map_two_glossary.csv": _GLOSSARY_CSV,
        }),
        "run_attrs": lambda sb: {
            "selected_files": [sb.path("inputs/Map One.epub"), sb.path("inputs/Map Two.epub")],
            "manual_glossary_map": {
                os.path.normpath(sb.path("inputs/Map One.epub")): sb.path("inputs/map_one_glossary.csv"),
                os.path.normpath(sb.path("inputs/Map Two.epub")): sb.path("inputs/map_two_glossary.csv"),
            },
        },
        "expect": _all(
            _expect_entries(T, T),
            _expect_env_endswith(T, 0, "MANUAL_GLOSSARY", "map_one_glossary.csv"),
            _expect_env_endswith(T, 1, "MANUAL_GLOSSARY", "map_two_glossary.csv"),
            _expect_logs("📑 Glossary mapping enabled (2 file(s) mapped)"),
        ),
    },
    "graceful_stop": {
        "description": "Graceful stop mid-chunk (wait for chunks): queued sends cancelled, in-flight kept.",
        "config": _base_config(),
        "files": _files("Graceful Novel"),
        "run_attrs": _with(_inputs("Graceful Novel"), **STOP_ATTRS),
        "plan": {T: [CallPlan(chunks=3, actions={1: (("click",),)})]},
        "expect": _all(
            _expect_entries(T),
            _expect_after_click(stop_callback=True, GRACEFUL_STOP="1", WAIT_FOR_CHUNKS="1",
                                TRANSLATION_CANCELLED=None),
            _expect_stop_api("TransateKRtoEN.cancel_queued_translation_sends",
                             "unified_api_client._api_watchdog_clear_pending_requests",
                             "unified_api_client.reset_api_call_stagger"),
            _expect_logs("⏳ Graceful stop — waiting for in-flight API calls to complete..."),
        ),
    },
    "immediate_stop": {
        "description": "Immediate stop mid-chunk: cancel env, module stop flags, background hard cancel.",
        "config": _base_config(),
        "files": _files("Immediate Novel"),
        "run_attrs": _with(_inputs("Immediate Novel"), graceful_stop_var=False, wait_for_chunks_var=True),
        "plan": {T: [CallPlan(chunks=3, actions={1: (("click",),)})]},
        "expect": _all(
            _expect_entries(T),
            _expect_after_click(stop_callback=True, TRANSLATION_CANCELLED="1", GRACEFUL_STOP="0",
                                WAIT_FOR_CHUNKS="0", **{"module:TransateKRtoEN": True}),
            _expect_stop_api("unified_api_client.hard_cancel_all", "unified_api_client._api_watchdog_reset",
                             absent=("TransateKRtoEN.cancel_queued_translation_sends",)),
            _expect_logs("🛑 Force stop requested — aborting queued/in-flight API calls"),
        ),
    },
    "double_click_force_stop": {
        "description": "Graceful first click, second click 0.3 s later mid-chunk = force stop.",
        "config": _base_config(),
        "files": _files("Double Novel"),
        "run_attrs": _with(_inputs("Double Novel"), **STOP_ATTRS),
        "plan": {T: [CallPlan(chunks=3, actions={1: (("click",), ("advance", 0.3), ("click",))})]},
        "expect": _all(
            _expect_entries(T),
            _expect_after_click(stop_callback=True, TRANSLATION_CANCELLED="1", GRACEFUL_STOP="0",
                                WAIT_FOR_CHUNKS="0"),
            _expect_logs("⚡ Double-click detected — forcing immediate stop!"),
        ),
    },
    "click_after_graceful_finish": {
        "description": "Graceful stop, worker ends, Run clicked again within 2 s = forced hard cancel (desktop race).",
        "config": _base_config(),
        "files": _files("Late Click Novel"),
        "run_attrs": _with(_inputs("Late Click Novel"), **STOP_ATTRS),
        "plan": {T: [CallPlan(chunks=3, actions={1: (("click",),)})]},
        "post_actions": (("advance", 1.0), ("click",)),
        "modes": DESKTOP_ONLY,
        "expect": _all(
            _expect_entries(T),
            _expect_logs("⚡ Double-click detected after graceful stop — forcing hard cancel!"),
        ),
    },
    "immediate_stop_during_pre_glossary": {
        "description": "Immediate stop while the balanced pre-glossary runs: translation never starts.",
        "config": _base_config(auto_glossary_mode="balanced"),
        "files": _files("Early Stop Novel"),
        "run_attrs": _with(_inputs("Early Stop Novel"), graceful_stop_var=False, wait_for_chunks_var=True),
        "plan": {G: [CallPlan(chunks=3, actions={1: (("click",),)}, glossary=False)]},
        "expect": _all(
            _expect_entries(G),
            _expect_logs("⏹️ Translation cancelled during glossary extraction"),
        ),
    },
    # U6: Extract Glossary stopped mid-chunk. Desktop: the glossary button calls
    # stop_glossary_extraction; mobile: stop_control.request_glossary_stop (the same protocol).
    "glossary_immediate_stop": {
        "description": "Extract Glossary, immediate stop mid-chunk: cancel env, extractor/client stop flags, cleanup.",
        "entry": GLOSSARY,
        "config": _base_config(auto_glossary_mode="balanced"),
        "files": _files("Glossary Stop Novel"),
        "run_attrs": _with(_inputs("Glossary Stop Novel"), graceful_stop_var=False, wait_for_chunks_var=True),
        "plan": {G: [CallPlan(chunks=3, actions={1: (("click",),)}, glossary=False)]},
        "expect": _all(
            _expect_entries(G),
            _expect_after_click(stop_callback=True, TRANSLATION_CANCELLED="1", GRACEFUL_STOP="0",
                                GRACEFUL_STOP_COMPLETED="0"),
            _expect_stop_api("unified_api_client.hard_cancel_all"),
            _expect_logs("❌ Glossary extraction stop requested."),
        ),
    },
    "glossary_graceful_stop": {
        "description": "Extract Glossary, graceful stop mid-chunk: no cancel env, the extractor stops at its next check.",
        "entry": GLOSSARY,
        "config": _base_config(auto_glossary_mode="balanced"),
        "files": _files("Glossary Graceful Novel"),
        "run_attrs": _with(_inputs("Glossary Graceful Novel"), **STOP_ATTRS),
        "plan": {G: [CallPlan(chunks=3, actions={1: (("click",),)}, glossary=False)]},
        "expect": _all(
            _expect_entries(G),
            _expect_after_click(stop_callback=True, GRACEFUL_STOP="1", TRANSLATION_CANCELLED=None),
            _expect_stop_api(absent=("unified_api_client.hard_cancel_all",)),
            _expect_logs("🛑 Stop requested — cancelling glossary API calls (WAIT_FOR_CHUNKS=0)"),
        ),
    },
    "direct_text_glossary_approved": {
        "description": "Direct Text attachment run, balanced glossary generated, approval answered Yes.",
        "config": _base_config(auto_glossary_mode="balanced"),
        "files": dict(_DIRECT_TEXT_FILES),
        "pre_run": _direct_text_pre_run,
        "answers": (True,),
        "expect": _all(
            _expect_entries(G, T),
            _expect_logs("⏸️ Direct Text: glossary generation is complete; waiting for approval before translation"),
            lambda view: [] if view.asks() == 1 else [f"{view.asks()} approval questions, expected 1"],
        ),
    },
    "direct_text_glossary_rejected": {
        "description": "Direct Text attachment run, balanced glossary generated, approval answered No.",
        "config": _base_config(auto_glossary_mode="balanced"),
        "files": dict(_DIRECT_TEXT_FILES),
        "pre_run": _direct_text_pre_run,
        "answers": (False,),
        "expect": _all(
            _expect_entries(G),
            _expect_logs("⏹️ Direct Text translation cancelled at the glossary approval step"),
        ),
    },
    # U7: run_translation_direct dispatching to image_job / rpgmaker_job (the runners moved out of
    # TranslatorGUI); client calls are recorded with the environment they saw.
    "image_vision_translate": {
        "description": "One PNG in vision mode: progress file, payloads, translated page HTML.",
        "config": _base_config(**_VISION_CFG),
        "files": {"inputs/page01.png": _PNG},
        "run_attrs": _media_inputs("inputs/page01.png"),
        "plan": {IS: [CallPlan(result="<p>The translated page.</p>")]},
        "expect": _all(
            _expect_entries(IS),
            _expect_env(IS, 0, ENABLE_IMAGE_OUTPUT_MODE="0", IMAGE_OUTPUT_RESOLUTION="1K"),
            _expect_logs("🖼️ Processing image: page01.png", "✅ Translation saved to:"),
        ),
    },
    "image_batch_combined_folder": {
        "description": "Two PNGs: one combined output folder, both pages through the vision call.",
        "config": _base_config(**_VISION_CFG),
        "files": {"inputs/scans/p1.png": _PNG, "inputs/scans/p2.png": _PNG[:-1] + b"\x83"},
        "run_attrs": _media_inputs("inputs/scans/p1.png", "inputs/scans/p2.png"),
        "plan": {IS: [CallPlan(result="<p>one</p>"), CallPlan(result="<p>two</p>")]},
        "expect": _all(
            _expect_entries(IS, IS),
            _expect_logs("📁 Created combined output directory:"),
        ),
    },
    "image_output_generated": {
        "description": "Image output mode: the edited image comes back as a sentinel and is moved into the output.",
        "config": _base_config(**_IMAGE_CFG),
        "files": {"inputs/page01.png": _PNG},
        "run_attrs": _media_inputs("inputs/page01.png"),
        "plan": {IS: [CallPlan(result=_generated("Generated_Media/edit_page01.png"))]},
        "expect": _all(
            _expect_entries(IS),
            _expect_env(IS, 0, ENABLE_IMAGE_OUTPUT_MODE="1", IMAGE_OUTPUT_RESOLUTION="2K"),
            _expect_logs("✅ Generated media saved directly as: response_001_page01.png"),
        ),
    },
    "video_input_generated": {
        "description": "An MP4 input in video output mode: the source path reaches the client, video comes back.",
        "config": _base_config(model="gpt-4o", output_mode="video", enable_image_translation=True,
                               enable_video_output_mode=True),
        "files": {"inputs/clip.mp4": _MP4},
        "run_attrs": _media_inputs("inputs/clip.mp4"),
        "plan": {IS: [CallPlan(result=_generated("Generated_Media/clip_out.mp4", _MP4))]},
        "expect": _all(
            _expect_entries(IS),
            _expect_env(IS, 0, ENABLE_VIDEO_OUTPUT_MODE="1"),
            _expect_env_endswith(IS, 0, "NANOGPT_SOURCE_VIDEO_PATH", "inputs/clip.mp4"),
        ),
    },
    "audio_mode_image_input": {
        "description": "Audio output mode with an image input: the vision call sees the audio-mode environment.",
        "config": _base_config(model="gpt-4o", output_mode="audio", enable_audio_output_mode=True),
        "files": {"inputs/page01.png": _PNG},
        "run_attrs": _media_inputs("inputs/page01.png"),
        "plan": {IS: [CallPlan(result="<p>Narration</p>")]},
        "expect": _all(_expect_entries(IS), _expect_logs("✅ Translation saved to:")),
    },
    "generative_image_prompt": {
        "description": "Image-generation model, no input: the prompt-only generation run.",
        "config": _base_config(model="gpt-image-1", output_mode="image", enable_image_translation=True,
                               enable_image_output_mode=True),
        "run_attrs": {"selected_files": []},
        "pre_run": _no_file_selected,
        "plan": {CS: [CallPlan(result=_generated("Generated_Media/fox.png"))]},
        "expect": _all(
            _expect_entries(CS),
            _expect_env(CS, 0, ENABLE_IMAGE_OUTPUT_MODE="1"),
            _expect_logs("🎨 Generative mode: sending prompt to gpt-image-1", "📄 Media saved to:"),
        ),
    },
    "generative_video_prompt": {
        "description": "Video output mode, no input: generation with the video duration/resolution env.",
        "config": _base_config(model="veo-3", output_mode="video", enable_image_translation=True,
                               enable_video_output_mode=True),
        "run_attrs": {"selected_files": []},
        "pre_run": _no_file_selected,
        "plan": {CS: [CallPlan(result="video job accepted")]},
        "expect": _all(
            _expect_entries(CS),
            _expect_env(CS, 0, ENABLE_VIDEO_OUTPUT_MODE="1"),
            _expect_logs("📄 Saved to:"),
        ),
    },
    "generative_audio_prompt": {
        "description": "Audio output mode, the generative sentinel selected: prompt-only speech generation.",
        "config": _base_config(model="gpt-4o-mini-tts", output_mode="audio", enable_audio_output_mode=True),
        "run_attrs": {"selected_files": ["__generative_mode__"]},
        "plan": {CS: [CallPlan(result="[GENERATED_AUDIO:speech.mp3]")]},
        "expect": _all(_expect_entries(CS), _expect_logs("🎨 Generative mode: sending prompt to gpt-4o-mini-tts")),
    },
    "rpgmaker_exe_text": {
        "description": "RPG Maker MV game (.exe): extract, chunk, translate, apply into www/data.",
        "config": _base_config(batch_translation=True, batch_size="2"),
        "files": _mv_game_files(),
        "run_attrs": _media_inputs("games/Hero/Game.exe"),
        "plan": {CS: [CallPlan(result=_rpg_echo)] * 6},
        "expect": _all(
            _expect_logs("🎮 Detected RPG Maker MV", "🎮 GTool: Translation complete!"),
            lambda view: [] if view.entries() and set(view.entries()) == {CS} else [f"entries {view.entries()}"],
        ),
    },
    "rpgmaker_exe_image_mode": {
        "description": "RPG Maker MV game in image mode: the game image pipeline instead of text.",
        "config": _base_config(output_mode="vision", enable_image_translation=True),
        "files": _mv_game_files(),
        "run_attrs": _media_inputs("games/Hero/Game.exe"),
        "plan": {GAME_IMAGES: [CallPlan(result=2)]},
        "expect": _all(
            _expect_entries(GAME_IMAGES),
            _expect_logs("🖼️ Output mode: Image — translating game image assets only"),
        ),
    },
}

for _name, _scenario in SCENARIOS.items():
    _scenario["name"] = _name
    _scenario.setdefault("entry", TRANSLATE)
    _scenario.setdefault("plan", {})
    _scenario.setdefault("answers", ())
    _scenario.setdefault("post_actions", ())
    _scenario.setdefault("modes", ALL_MODES)

SCENARIO_NAMES = tuple(SCENARIOS)
STOP_SCENARIOS = tuple(n for n, s in SCENARIOS.items()
                       if any("click" in a[0] for p in s["plan"].values() for c in p
                              for acts in c.actions.values() for a in acts)
                       or any(a[0] == "click" for a in s["post_actions"]))


def get(name: str) -> dict:
    return SCENARIOS[name]


def modes(name: str) -> tuple:
    return tuple(SCENARIOS[name]["modes"])


__all__ = [
    "ALL_MODES",
    "CLIENT_SEND",
    "CallPlan",
    "DESKTOP_ONLY",
    "EPUB_COMPILE",
    "GAME_IMAGES",
    "GLOSSARY",
    "IMAGE_SEND",
    "SEND_INTERRUPT",
    "GLOSSARY_MAIN",
    "METADATA_JOB",
    "PDF_COMPILE",
    "SCENARIOS",
    "SCENARIO_NAMES",
    "STOP_SCENARIOS",
    "TRANSLATE",
    "TRANSLATION_MAIN",
    "build_epub",
    "get",
    "modes",
]
