"""image_job: the image / video runner and the prompt-only generation run (Glossarion mobile rewrite, U7).

``TranslatorGUI._process_image_file`` and ``_run_generative_prompt_mode`` moved verbatim into
``image_job.ImageJobMixin``; ``translation_pipeline.TranslationPipelineMixin`` inherits it in
place of the U3 placeholders, so TranslatorGUI and HeadlessOwner run the same code.

What is checked here:

* verbatim: each moved method equals its text at ``runner_parity.LEGACY_SHA`` (git show) after
  exactly the documented edits; TranslatorGUI no longer defines them; MRO / hooks;
* import hygiene (PySide6 blocked), Python 3.10 syntax, uniform line endings;
* runner-level trace parity (``tests/parity/runner_parity.py``): the legacy method and the
  mixin run the same scenario matrix (vision / image-edit / video / audio output modes,
  generated-media sentinels and data URIs, title translation, contextual history, appended
  glossaries, Direct Text prompt overrides, key pools, resume / skip / cover, errors and stops;
  generative-only image / video / audio runs and their prompt fallbacks) with recording
  backends; logs, backend calls with the environment they saw, the final environment, owner
  state and the whole output tree (``translation_progress.json``, HTML, payloads, media) must be
  equal;
* the GUI-free additions: the generative prompt override (mobile composer text) and the
  ``Generated_Media`` folder under ``GLOSSARION_DATA_DIR``;
* a HeadlessOwner reaches the real runners through the shared worker (no placeholder left).

The full pipeline (run_translation_thread / run_translation_direct dispatch, desktop and
mobile) is also traced against the frozen desktop in ``tests/parity/test_trace_parity.py``
(scenarios ``image_*`` / ``generative_*``).

Run (repository root)::

    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests/test_image_job.py
"""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
TESTS = REPO_ROOT / "tests"
for _p in (str(TESTS), str(SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from parity import runner_parity as rp  # noqa: E402
from parity.runner_parity import MP4_BYTES, PNG_BYTES, RunnerScenario, generated  # noqa: E402

MODULE = "image_job"

#: documented edits per moved method: (old, new, occurrences)
EDITS = {
    "_run_generative_prompt_mode": [
        ("            system_prompt = ''\n"
         "            try:\n"
         "                system_prompt = self.prompt_text.toPlainText().strip()\n"
         "            except Exception:\n"
         "                pass\n",
         "            system_prompt = self._generative_prompt_source()\n", 1),
        ("out_dir = pathlib.Path(os.path.dirname(os.path.abspath(__file__))) / 'Generated_Media'",
         "out_dir = pathlib.Path(data_dir(os.path.dirname(os.path.abspath(__file__)))) / 'Generated_Media'", 1),
    ],
    "_process_image_file": [],
}


def _legacy_or_skip(name):
    try:
        return rp.legacy_method_text(name)
    except rp.Unavailable as exc:  # pragma: no cover - shallow clone
        if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
            pytest.fail(str(exc))
        pytest.skip(str(exc))


def _apply(text, edits, where):
    for old, new, count in edits:
        assert text.count(old) == count, f"{where}: expected {count} x {old!r}, found {text.count(old)}"
        text = text.replace(old, new)
    return text


def _class_methods(module, cls):
    text = rp.module_text(module)
    tree = ast.parse(text)
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    return text, {n.name: n for n in node.body if isinstance(n, ast.FunctionDef)}


# ---------------------------------------------------------------------------
# verbatim / composition / hygiene
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(EDITS))
def test_moved_method_is_the_legacy_text_plus_documented_edits(name):
    legacy = _apply(_legacy_or_skip(name), EDITS[name], name)
    text, methods = _class_methods(MODULE, "ImageJobMixin")
    assert rp.node_text(text, methods[name]) == legacy


def test_prompt_source_hook_default_is_the_moved_lines():
    text, methods = _class_methods(MODULE, "ImageJobMixin")
    hook = rp.node_text(text, methods["_generative_prompt_source"])
    old = EDITS["_run_generative_prompt_mode"][0][0]
    assert "\n".join(line[4:] for line in old.rstrip("\n").split("\n")) in hook
    assert set(methods) == {"_generative_prompt_source", "_run_generative_prompt_mode", "_process_image_file"}


def test_translator_gui_no_longer_defines_the_runners():
    text = rp.module_text("translator_gui")
    tree = ast.parse(text)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TranslatorGUI")
    body = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
    for name in rp.RUNNERS + ("_generative_prompt_source", "_register_rpgmaker_game_input"):
        assert name not in body, name
    legacy = rp.legacy_translator_gui() if _git_ok() else None
    if legacy is not None:
        for name in rp.RUNNERS:
            assert f"    def {name}(self" in legacy


def _git_ok():
    try:
        rp.legacy_translator_gui()
        return True
    except rp.Unavailable:
        return False


#: Every U7 (image / RPG Maker, QA Stop) change of translator_gui.py vs its parent: (first old line, last old
#: line, new line count) of each non-equal difflib block. U9 gap closures moved more code out (pinned by
#: tests/parity/test_u9_extractions.py, test_u9_gap_moves.py and tests/test_translation_pipeline.py); their
#: spans are listed too, so the file still changes nowhere else.
TG_U7_SPANS = (
    (224, 324, 2),       # U9: _fmt_bytes / _sweep_size_capped_dir -> shutdown_utils (imported under the old names)
    (328, 328, 1),       #   _sweep_large_caches docstring
    (330, 415, 4),       #   ... its body calls shutdown_utils.sweep_large_caches(script_file=__file__)
    (1342, 1341, 8),     # import the Direct Text rule functions from direct_text_store
    (2751, 2759, 5),     # dialog __init__: configured_glossary_override_mode(...)
    (3560, 3564, 2),     # _on_glossary_override_toggled: glossary_override_config_updates(mode)
    (3567, 3571, 1),     #   ... config.update(updates)
    (3593, 3593, 0),     # _request_direct_text_manual_glossary: json_lib import gone
    (3596, 3596, 1),     #   allowed_extensions = set(MANUAL_GLOSSARY_EXTENSIONS)
    (3754, 3754, 9),     #   _accept: manual_glossary_source_record(...)
    (3761, 3789, 1),     #   ... result.update(record)
    (5700, 5700, 1),     # _rename_chat: chat_rename_title(new_title)
    (12277, 12276, 2),   # U9 review: select_google_credentials imports the settings_rules check + messages
    (12281, 12281, 1),   #   if is_google_service_account(creds_data):
    (12300, 12300, 1),   #   INVALID_GOOGLE_CREDENTIALS (tests/parity/test_u9_gap_round4.py runs it vs the frozen one)
    (12303, 12303, 1),   #   google_credentials_load_error(e)
    (13613, 13623, 8),   # _authgem_projects_loaded: authgem_auth.authgem_project_items(...)
    (13625, 13635, 1),   #   ... authgem_auth.choose_authgem_project_index(...)
    (14648, 14657, 1),   # U9: _show_model_info_dialog imports model_options.provider_info_html
    (14659, 14851, 1),   #   ... info_text = provider_info_html() (the HTML moved verbatim)
    (19821, 19845, 0),   # U9: _rename_input_for_existing_workspace_collision moved into run_env.RunEnvMixin
    (23411, 23468, 0),   # U9: _get_pdf_range_entries_for_preview moved into the GlossaryPipelineMixin
    (23577, 25279, 0),   # _run_generative_prompt_mode, _process_image_file, _process_rpgmaker_game moved
    # U7 Integrate: the QA Stop handlers call the qa_scan_runtime stop helpers (tests/test_u7_tool_cores.py
    # pins them against the frozen blocks)
    (26497, 26496, 2),   # stop_qa_scan: next_qa_stop_phase(current_phase, graceful_stop_enabled)
    (26500, 26500, 1),   #   if next_phase == 'graceful':
    (26506, 26517, 4),   #   apply_qa_graceful_stop_flags()
    (26552, 26553, 2),   #   elif next_phase == 'force':
    (26563, 26583, 5),   # _do_qa_force_stop: apply_qa_force_stop_flags()
    (26662, 26667, 2),   # _check_qa_stop_done._delayed_flag_cleanup: clear_qa_stop_flags()
)


def test_translator_gui_changed_only_in_the_u7_spans():
    import difflib

    try:
        legacy = rp.legacy_translator_gui().split("\n")
    except rp.Unavailable as exc:  # pragma: no cover - shallow clone
        if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
            pytest.fail(str(exc))
        pytest.skip(str(exc))
    current = rp.module_text("translator_gui").split("\n")
    spans = tuple((i1 + 1, i2, j2 - j1) for tag, i1, i2, j1, j2 in
                  difflib.SequenceMatcher(None, legacy, current, autojunk=False).get_opcodes() if tag != "equal")
    assert spans == TG_U7_SPANS


def test_pipeline_mixin_inherits_the_runners_in_place_of_the_placeholders():
    import headless_owner
    import image_job
    import rpgmaker_job
    import translation_pipeline as tp

    assert tp.TranslationPipelineMixin.__bases__ == (tp.GlossaryPipelineMixin, image_job.ImageJobMixin,
                                                     rpgmaker_job.RpgMakerJobMixin)
    assert not hasattr(tp, "U7_PLACEHOLDERS")
    for name in rp.RUNNERS:
        assert name not in vars(tp.PipelineHooksMixin) and name not in tp.PIPELINE_HOOKS, name
        owners = [k for k in headless_owner.HeadlessOwner.__mro__ if name in vars(k)]
        assert owners in ([image_job.ImageJobMixin], [rpgmaker_job.RpgMakerJobMixin]), (name, owners)
    assert image_job.GENERATIVE_MODE_SENTINEL == "__generative_mode__"
    assert ("image_job", "ImageJobMixin") in headless_owner.OWNER_CONTRACT_MODULES


def test_translator_gui_mro_resolves_the_runners_to_the_mixins():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from parity import normalize

    normalize.preimport_backend_modules()
    import image_job
    import translator_gui

    tg = translator_gui.TranslatorGUI
    for name in ("_run_generative_prompt_mode", "_process_image_file", "_generative_prompt_source"):
        assert [k for k in tg.__mro__ if name in vars(k)] == [image_job.ImageJobMixin], name


_HYGIENE_PROBE = r"""
import sys
sys.modules['PySide6'] = None
sys.path.insert(0, {src!r})
import importlib
importlib.import_module({module!r})
leaked = [n for n in ('translator_gui', 'dpi_setup', 'epub_library', 'PySide6.QtCore', 'rpgmaker_handler',
                      'TransateKRtoEN', 'unified_api_client') if sys.modules.get(n) is not None]
print('LEAKED=' + ','.join(leaked))
"""


@pytest.mark.parametrize("module", (MODULE, "rpgmaker_job"))
def test_module_imports_without_qt_the_gui_or_the_backends(module):
    env = {k: v for k, v in os.environ.items() if not k.startswith("GLOSSARION_")}
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run([sys.executable, "-c", _HYGIENE_PROBE.format(src=str(SRC), module=module)],
                          capture_output=True, text=True, encoding="utf-8", env=env, timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert proc.stdout.strip().splitlines()[-1] == "LEAKED=", proc.stdout[-2000:]


@pytest.mark.parametrize("module", (MODULE, "rpgmaker_job", "translation_pipeline"))
def test_module_parses_as_python_310_with_uniform_line_endings(module):
    raw = (SRC / f"{module}.py").read_bytes()
    ast.parse(raw.decode("utf-8"), feature_version=(3, 10))
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert raw.count(b"\r\n") in (0, raw.count(b"\n")), "mixed line endings"


# ---------------------------------------------------------------------------
# runner-level trace parity: _process_image_file
# ---------------------------------------------------------------------------

PROFILE = "Korean_BeautifulSoup"
PROFILE_PROMPT = "Translate the text in this image to {target_lang}.\n{split_marker_instruction}\nKeep formatting."


def _cfg(**extra):
    cfg = {"model": "gpt-4o", "api_key": "sk-image-0000", "active_profile": PROFILE,
           "prompt_profiles": {PROFILE: {"prompt": PROFILE_PROMPT, "book_title_prompt": ""}},
           "output_language": "English"}
    cfg.update(extra)
    return cfg


_IMG = {"inputs/page01.png": PNG_BYTES}
_OUT = {"OUTPUT_DIRECTORY": "@/out"}
_VISION = {"enable_image_translation_var": True, "output_mode_var": "vision"}
_PNG_HASH = hashlib.sha256(PNG_BYTES).hexdigest()


def _image(name, *, args=("@/inputs/page01.png",), files=None, config=None, attrs=None, env=None, **kw):
    return RunnerScenario(name, "_process_image_file", args=args, files=dict(_IMG if files is None else files),
                          config=_cfg(**(config or {})), attrs=dict(_VISION, **(attrs or {})),
                          env=dict(_OUT, **(env or {})), **kw)


def _done_progress(root_rel_output):
    def build(root):
        output = os.path.join(root, *root_rel_output.split("/"))
        os.makedirs(os.path.dirname(output), exist_ok=True)
        with open(output, "w", encoding="utf-8") as fh:
            fh.write("<p>done</p>")
        return json.dumps({"images": {_PNG_HASH: {"name": "page01.png", "status": "completed",
                                                  "output_file": output}},
                           "content_hashes": {}, "version": "1.0"})
    return build


_GLOSSARY_CSV = "type,raw_name,translated_name\ncharacter,김상현,Kim Sang-hyun\n"
_GLOSSARY_JSON_LIST = json.dumps([{"original_name": "김상현", "name": "Kim Sang-hyun", "title": "검사",
                                   "how_they_refer_to_others": {"이수": "형님"}}, "skip-me"], ensure_ascii=False)
_GLOSSARY_JSON_DICT = json.dumps({"entries": {"김상현": "Kim Sang-hyun"}, "metadata": {"v": 1}}, ensure_ascii=False)
_DATA_URI = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode("ascii")
_ALL_POOLS = {
    "use_multi_api_keys": True, "multi_api_keys": [{"api_key": "k1", "model": "gpt-4o"}],
    "force_key_rotation": False, "rotation_frequency": 3,
    "use_glossary_keys": True, "glossary_keys": [{"api_key": "g1"}],
    "use_glossary_refinement_keys": True, "glossary_refinement_keys": [{"api_key": "r1"}],
    "use_rolling_summary_keys": True, "rolling_summary_keys": [{"api_key": "s1"}],
    "use_truncation_retry_keys": True, "truncation_retry_keys": [],
    "use_inpainter_keys": True, "inpainter_keys": [{"api_key": "i1"}],
    "use_tts_keys": True, "tts_keys": [{"api_key": "t1"}],
}


def _stop(owner):
    owner.stop_requested = True


IMAGE_SCENARIOS = [
    _image("vision_text", send_image=("<p>Hello world</p>",)),
    _image("vision_title_translation", config={"skip_image_title_translation": False,
                                               "book_title_prompt": "Translate this title into {target_lang}:"},
           attrs={"skip_image_title_translation_var": False}, send_image=("<p>Body</p>",), send=("Page One",)),
    _image("vision_title_send_fails", config={"skip_image_title_translation": False},
           attrs={"skip_image_title_translation_var": False}, send_image=("<p>Body</p>",),
           send=(RuntimeError("title boom"),)),
    _image("vision_contextual_text", attrs={"contextual_var": True}, send_image=("<p>Context</p>",)),
    _image("image_generated_sentinel_contextual",
           attrs={"enable_image_output_mode_var": True, "contextual_var": True, "output_mode_var": "image",
                  "image_output_resolution_var": "2k"},
           send_image=(generated("Generated_Media/gen_page01.png"),)),
    _image("image_data_uri", attrs={"enable_image_output_mode_var": True, "output_mode_var": "image"},
           send_image=(_DATA_URI,)),
    _image("image_data_uri_broken", attrs={"enable_image_output_mode_var": True},
           send_image=("data:image/png;base64,@@not-base64@@",)),
    _image("video_input", args=("@/inputs/clip.mp4",), files={"inputs/clip.mp4": MP4_BYTES},
           attrs={"enable_video_output_mode_var": True, "output_mode_var": "video"},
           send_image=(generated("Generated_Media/clip_out.mp4", MP4_BYTES),)),
    _image("audio_mode", attrs={"enable_audio_output_mode_var": True, "output_mode_var": "audio"},
           send_image=("<p>Narration text</p>",)),
    _image("glossary_csv_appended", files=dict(_IMG, **{"glossary/book.csv": _GLOSSARY_CSV}),
           env={"MANUAL_GLOSSARY": "@/glossary/book.csv"}, config={"auto_glossary_mode": "off"},
           send_image=("<p>g</p>",)),
    _image("glossary_json_list_appended", files=dict(_IMG, **{"glossary/book.json": _GLOSSARY_JSON_LIST}),
           attrs={"manual_glossary_path": "@/glossary/book.json"}, config={"auto_glossary_mode": "balanced"},
           send_image=("<p>g</p>",)),
    _image("glossary_json_dict_appended", files=dict(_IMG, **{"glossary/book.json": _GLOSSARY_JSON_DICT}),
           env={"MANUAL_GLOSSARY": "@/glossary/book.json"}, send_image=("<p>g</p>",)),
    _image("glossary_empty_file", files=dict(_IMG, **{"glossary/empty.csv": ""}),
           env={"MANUAL_GLOSSARY": "@/glossary/empty.csv"}, attrs={"manual_glossary_path": "@/glossary/empty.csv"},
           config={"manual_glossary_path": "@/glossary/empty.csv"}, send_image=("<p>g</p>",)),
    _image("glossary_minimal_deferred", config={"auto_glossary_mode": "minimal"}, send_image=("<p>g</p>",)),
    _image("glossary_legacy_enable_auto_minimal", config={"enable_auto_glossary": True}, send_image=("<p>g</p>",)),
    _image("glossary_no_glossary_mode", files=dict(_IMG, **{"glossary/book.csv": _GLOSSARY_CSV}),
           env={"MANUAL_GLOSSARY": "@/glossary/book.csv"}, config={"auto_glossary_mode": "no_glossary"},
           send_image=("<p>g</p>",)),
    _image("glossary_append_disabled", attrs={"append_glossary_var": False}, env={"DEFER_GLOSSARY_APPEND": "1"},
           send_image=("<p>g</p>",)),
    _image("glossary_missing_file", env={"MANUAL_GLOSSARY": "@/glossary/missing.csv"}, send_image=("<p>g</p>",)),
    _image("direct_text_overrides",
           attrs={"_input_output_run_active": True, "_direct_text_skip_prompt_profile": True},
           env={"DIRECT_TEXT_ATTACHMENT_PROMPT": "Describe the picture first.",
                "DIRECT_TEXT_ATTACHMENT_PROMPT_ROLE": "user"}, send_image=("<p>dt</p>",)),
    _image("direct_text_profile_as_user", attrs={"_input_output_run_active": True, "system_prompt_to_user_var": True},
           env={"DIRECT_TEXT_PROFILE_USER_PROMPT": "Profile as user"}, send_image=("<p>dt</p>",)),
    _image("system_prompt_to_user_label", env={"SYSTEM_PROMPT_TO_USER": "1"}, send_image=("<p>u</p>",)),
    _image("split_marker_instruction", attrs={"request_merging_enabled_var": True}, send_image=("<p>s</p>",)),
    _image("old_format_profile_and_config_prompt",
           config={"prompt_profiles": {PROFILE: "Old style {target_lang}"}}, send_image=("<p>o</p>",)),
    _image("profile_missing_uses_config_key", config={"active_profile": "Custom", "prompt_profiles": {},
                                                      "Custom": "From config key"}, send_image=("<p>c</p>",)),
    _image("key_pools", config=_ALL_POOLS, send_image=("<p>k</p>",)),
    _image("key_pools_enabled_without_keys",
           config={"use_rolling_summary_keys": True, "use_truncation_retry_keys": True, "use_inpainter_keys": True,
                   "use_tts_keys": True}, send_image=("<p>k</p>",)),
    _image("resume_already_translated",
           files=dict(_IMG, **{"out/page01/translation_progress.json": _done_progress("out/page01/response_001_page01.html")})),
    _image("resume_output_deleted",
           files=dict(_IMG, **{"out/page01/translation_progress.json": json.dumps(
               {"images": {_PNG_HASH: {"status": "completed", "output_file": "missing.html"}},
                "content_hashes": {}, "version": "1.0"})}),
           send_image=("<p>again</p>",)),
    _image("resume_previous_error", files=dict(_IMG, **{"out/page01/translation_progress.json": json.dumps(
        {"images": {_PNG_HASH: {"status": "error"}}, "content_hashes": {}, "version": "1.0"})}),
           send_image=("<p>retry</p>",)),
    _image("corrupt_progress_file", files=dict(_IMG, **{"out/page01/translation_progress.json": "{not json"}),
           send_image=("<p>fresh</p>",)),
    _image("skip_marker_combined_output",
           args=("@/inputs/page01.png", "@/out/combined"),
           files=dict(_IMG, **{"out/combined/translation_progress.json": json.dumps(
               {"images": {}, "content_hashes": {}, "version": "1.0",
                "skip_page01.png": {"status": "skipped"}})})),
    _image("cover_combined_output", args=("@/inputs/cover.png", "@/out/combined"),
           files={"inputs/cover.png": PNG_BYTES}),
    _image("cover_own_folder", args=("@/inputs/cover.png",), files={"inputs/cover.png": PNG_BYTES}),
    _image("combined_output_text", args=("@/inputs/page01.png", "@/out/combined"), send_image=("<p>c</p>",)),
    _image("no_output_override_relative", env={"OUTPUT_DIRECTORY": ""}, send_image=("<p>rel</p>",)),
    _image("config_output_directory", env={"OUTPUT_DIRECTORY": ""}, config={"output_directory": "@/cfgout"},
           send_image=("<p>cfg</p>",)),
    _image("no_api_key", config={"api_key": ""}),
    _image("no_model", config={"model": ""}),
    _image("keyless_model", config={"model": "google-translate-free", "api_key": ""}, send_image=("<p>free</p>",)),
    _image("stop_before_start", before=_stop),
    _image("send_raises", send_image=(RuntimeError("boom"),)),
    _image("send_raises_interrupted", send_image=(RuntimeError("Request interrupted by user"),)),
    _image("empty_response", send_image=("",)),
    _image("failed_marker_response", send_image=("[IMAGE TRANSLATION FAILED]",)),
    _image("tuple_response", send_image=(("<p>tuple</p>", "length"),)),
    _image("current_file_index_numbering", attrs={"current_file_index": 4}, send_image=("<p>n</p>",)),
    _image("existing_progress_manager_reused",
           before=lambda owner: setattr(owner, "image_progress_manager", None), send_image=("<p>x</p>",)),
    _image("non_media_without_image_translation", args=("@/inputs/page01.txt",),
           files={"inputs/page01.txt": "text"}, attrs={"enable_image_translation_var": False}),
]


# ---------------------------------------------------------------------------
# runner-level trace parity: _run_generative_prompt_mode
# ---------------------------------------------------------------------------


def _gen(name, *, config=None, attrs=None, env=None, **kw):
    cfg = {"model": "gpt-image-1", "api_key": "sk-gen-0000", "prompt_text": "A red fox in fresh snow"}
    cfg.update(config or {})
    base = {"enable_image_output_mode_var": True, "enable_image_translation_var": True}
    base.update(attrs or {})
    return RunnerScenario(name, "_run_generative_prompt_mode", config=cfg, attrs=base, env=dict(env or {}), **kw)


GENERATIVE_SCENARIOS = [
    _gen("generative_image_text_result", send=("Here is your picture: a fox.",)),
    _gen("generative_image_sentinel", send=(generated("Generated_Media/fox.png"),)),
    _gen("generative_image_sentinel_missing_file", send=("[GENERATED_IMAGE:/nowhere/fox.png]",)),
    _gen("generative_video", config={"model": "veo-3"},
         attrs={"enable_image_output_mode_var": False, "enable_video_output_mode_var": True,
                "nanogpt_video_duration_var": "10", "nanogpt_video_resolution_var": "1080p"},
         send=(generated("Generated_Media/fox.mp4", MP4_BYTES),)),
    _gen("generative_audio", config={"model": "gpt-4o-mini-tts"},
         attrs={"enable_image_output_mode_var": False, "enable_audio_output_mode_var": True,
                "output_mode_var": "audio"}, send=("[GENERATED_AUDIO:speech.mp3]",)),
    _gen("generative_config_system_prompt_fallback", config={"prompt_text": "", "system_prompt": "Draw a castle"}),
    _gen("generative_image_chunk_prompt_fallback",
         config={"prompt_text": "", "image_chunk_prompt": "A lighthouse at dusk",
                 "translation_chunk_prompt": "{chunk_html}"}),
    _gen("generative_translation_chunk_prompt_fallback",
         config={"prompt_text": "", "translation_chunk_prompt": "A quiet harbour"},
         attrs={"image_chunk_prompt": "Template {chunk_idx}"}),
    _gen("generative_no_prompt", config={"prompt_text": "", "image_chunk_prompt": "{chunk_html}"}),
    _gen("generative_long_prompt", config={"prompt_text": "word " * 80}),
    _gen("generative_no_api_key", config={"api_key": ""}, send=("ok",)),
    _gen("generative_temperature_unreadable", attrs={"trans_temp": None}, send=("ok",)),
    _gen("generative_send_cancelled", send=(RuntimeError("Request cancelled by user"),)),
    _gen("generative_send_error", send=(ValueError("bad request"),)),
]


@pytest.fixture(scope="module", autouse=True)
def _legacy_available():
    _legacy_or_skip("_process_image_file")


@pytest.mark.parametrize("scenario", IMAGE_SCENARIOS + GENERATIVE_SCENARIOS, ids=lambda s: s.name)
def test_runner_matches_the_legacy_desktop_method(scenario, tmp_path):
    legacy, new = rp.run_pair(scenario, tmp_path)
    problems = rp.compare(legacy, new)
    assert not problems, "\n".join(problems)
    assert legacy["error"] is None, legacy["error"]


def test_the_matrix_reaches_its_paths(tmp_path):
    """The scenarios really exercise what their names claim (harness self-check on the legacy side)."""
    by_name = {s.name: s for s in IMAGE_SCENARIOS + GENERATIVE_SCENARIOS}

    def run(name):
        return rp.run_side("legacy", by_name[name], tmp_path / name)

    def logs(result):
        return [e[1] for e in result["events"] if e[0] == "log"]

    def calls(result, name):
        return [e[1] for e in result["events"] if e[0] == name]

    r = run("vision_text")
    assert r["result"] is True and "out/page01/response_001_page01.html" in r["fs"]
    progress = r["fs"]["out/page01/translation_progress.json"]["json"]
    assert progress["images"][_PNG_HASH]["status"] == "completed"
    r = run("image_generated_sentinel_contextual")
    assert "out/page01/response_001_page01.png" in r["fs"] and "out/page01/translation_history.json" in r["fs"]
    assert calls(r, "UnifiedClient.send_image")[0]["env"]["set"]["ENABLE_IMAGE_OUTPUT_MODE"] == "1"
    r = run("video_input")
    env = calls(r, "UnifiedClient.send_image")[0]["env"]["set"]
    assert env["ENABLE_VIDEO_OUTPUT_MODE"] == "1" and env["NANOGPT_SOURCE_VIDEO_PATH"].endswith("clip.mp4")
    r = run("glossary_csv_appended")
    assert "Kim Sang-hyun" in calls(r, "UnifiedClient.send_image")[0]["messages"][0]["content"]
    r = run("glossary_minimal_deferred")
    assert r["env"]["set"].get("DEFER_GLOSSARY_APPEND") == "1"
    r = run("key_pools")
    assert calls(r, "UnifiedClient.set_in_memory_multi_keys") and calls(r, "UnifiedClient.set_in_memory_tts_keys")
    r = run("resume_already_translated")
    assert r["result"] is True and any("Image already translated" in line for line in logs(r))
    assert not calls(r, "UnifiedClient.send_image")
    r = run("vision_title_translation")
    assert "<title>Page One</title>" in r["fs"]["out/page01/response_001_page01.html"]["text"]
    r = run("generative_image_text_result")
    assert r["result"] is True
    assert [k for k in r["fs"] if k.startswith("src/Generated_Media/generated_gpt-image-1_")]
    assert calls(r, "UnifiedClient.send")[0]["messages"] == [{"role": "user", "content": "A red fox in fresh snow"}]
    r = run("generative_video")
    env = calls(r, "UnifiedClient.send")[0]["env"]["set"]
    assert env["NANOGPT_VIDEO_DURATION"] == "10s" and env["NANOGPT_VIDEO_RESOLUTION"] == "1080p"
    r = run("generative_no_prompt")
    assert r["result"] is False and any("No prompt found" in line for line in logs(r))


# ---------------------------------------------------------------------------
# GUI-free additions (new side only)
# ---------------------------------------------------------------------------


def test_generative_prompt_override_is_the_composer_text(tmp_path):
    import image_job

    scenario = _gen("generative_override", send=("ok",),
                    attrs={image_job.GENERATIVE_PROMPT_ATTR: "  Composer: a koi pond  "})
    result = rp.run_side("new", scenario, tmp_path / "new")
    sends = [e[1] for e in result["events"] if e[0] == "UnifiedClient.send"]
    assert sends[0]["messages"] == [{"role": "user", "content": "Composer: a koi pond"}]
    # blank / non-text overrides fall back to the prompt editor (the desktop source)
    for value in ("   ", None, 42):
        scenario = _gen(f"generative_override_{value!r}", send=("ok",),
                        attrs={image_job.GENERATIVE_PROMPT_ATTR: value})
        result = rp.run_side("new", scenario, tmp_path / f"blank{len(str(value))}")
        sends = [e[1] for e in result["events"] if e[0] == "UnifiedClient.send"]
        assert sends[0]["messages"] == [{"role": "user", "content": "A red fox in fresh snow"}]


def test_generated_media_folder_follows_the_mobile_data_dir(tmp_path):
    data = tmp_path / "appdata"
    scenario = _gen("generative_data_dir", send=("text result",), env={"GLOSSARION_DATA_DIR": str(data)})
    result = rp.run_side("new", scenario, tmp_path / "new")
    assert result["result"] is True
    saved = list((data / "Generated_Media").glob("generated_gpt-image-1_*.txt"))
    assert len(saved) == 1 and saved[0].read_text(encoding="utf-8") == "text result"
    assert not [k for k in result["fs"] if "Generated_Media" in k]


def test_headless_owner_runs_the_generative_branch_end_to_end(tmp_path, monkeypatch):
    """Mobile composition: the shared worker reaches the real generative runner (no placeholder)."""
    import image_job
    import job_runner
    import stop_control
    from _headless_env import headless_owner

    calls = []

    class Client:
        def __init__(self, *args, **kwargs):
            calls.append(("init", kwargs))

        def send(self, messages, temperature=None, max_tokens=None, **_kw):
            calls.append(("send", messages, os.environ.get("ENABLE_AUDIO_OUTPUT_MODE")))
            return "spoken words", "stop"

        @classmethod
        def _model_needs_api_key(cls, model):
            return True

    import unified_api_client

    monkeypatch.setattr(unified_api_client, "UnifiedClient", Client)
    monkeypatch.setitem(sys.modules, "epub_library", None)
    host_logs = []
    host = type("H", (), {"log": lambda self, m, **k: host_logs.append(str(m)), "emit": lambda self, k, **d: None,
                          "ask": lambda self, k, **d: False, "is_stop_requested": lambda self: False,
                          "is_graceful_stop": lambda self: False})()
    cfg = {"model": "gpt-4o-mini-tts", "api_key": "sk-test-0000", "output_directory": str(tmp_path / "out"),
           "auto_update_check": False}
    with headless_owner(tmp_path / "app", monkeypatch, cfg, host=host) as owner:
        monkeypatch.setenv("GLOSSARION_DATA_DIR", str(tmp_path / "data"))
        from headless_owner import DirectTextRunOptions

        DirectTextRunOptions(selected_files=[], output_mode="audio").apply_to(owner)
        setattr(owner, image_job.GENERATIVE_PROMPT_ATTR, "Read this aloud: hello")
        with job_runner.job_process_state(host.log):
            stop_control.reset_for_new_run("translation")
            request = owner._prepare_translation_run([image_job.GENERATIVE_MODE_SENTINEL])
            assert request is not None
            outcome = owner._translation_worker(request)
    sends = [c for c in calls if c[0] == "send"]
    assert outcome is True, host_logs[-20:]
    assert sends and sends[0][1] == [{"role": "user", "content": "Read this aloud: hello"}]
    assert not any("not available in this build" in line for line in host_logs)
    assert list((tmp_path / "data" / "Generated_Media").glob("generated_*.txt"))
