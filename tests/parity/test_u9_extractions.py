"""U9: desktop code moved into shared GUI-free modules for Glossarion Mobile (gap closures).

The oracle is the desktop at ``U9_BASE_SHA`` (main 96da1ec6, the parent of the moves), read with
``git show``. Every moved body / literal must equal the frozen one (modulo the listed renames), and
the desktop callers now call the shared code:

* translator_gui ``_fmt_bytes`` / ``_sweep_size_capped_dir`` / ``_sweep_large_caches`` ->
  ``shutdown_utils.fmt_bytes`` / ``sweep_size_capped_dir`` / ``sweep_large_caches`` (public names;
  the script folder root comes from ``script_file``; ``extra_roots`` for mobile's data folders);
* translator_gui ``_show_model_info_dialog`` text -> ``model_options.PROVIDER_INFO_HTML``;
* other_settings ``test_api_connections`` endpoint collection / probe loop ->
  ``key_pool_service.collect_test_endpoints`` / ``run_endpoint_tests`` (``self`` -> ``owner``);
* other_settings ``HeaderTranslationHelpDialog`` sections -> ``metadata_defaults.HEADER_HELP_SECTIONS``;
* Retranslation_GUI's untranslated-rows closure -> ``progress_core.untranslated_manual_entries`` and
  its QA search-term extraction -> ``progress_actions.qa_issue_search_target``;
* manga_integration ``_show_model_info`` text -> ``manga_models.MODEL_INFO``, the Reset to Defaults
  values -> ``manga_settings_defaults.RENDERING_RESET_VALUES``, the custom image-edit endpoint test
  -> ``manga_env.normalize_custom_image_edit_url`` / ``probe_custom_image_edit_endpoint``;
* manga_settings_dialog's mask preset arguments -> ``manga_settings_defaults.MASK_PRESETS``.

(``_get_pdf_range_entries_for_preview`` -> GlossaryPipelineMixin is pinned by
tests/test_translation_pipeline.py; the Clear Boxes halves by tests/test_manga_editor_core.py.)
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

U9_BASE_SHA = "96da1ec6ccb34828c3ebd6e7f1c842796639a8e8"
NL = chr(10)


def frozen(relpath: str) -> str:
    raw = subprocess.check_output(["git", "show", f"{U9_BASE_SHA}:{relpath}"], cwd=str(REPO_ROOT))
    text = raw.decode("utf-8")
    return text.lstrip("﻿").replace("\r\n", "\n")


def current(name: str) -> str:
    return (SRC / name).read_bytes().decode("utf-8").lstrip("﻿").replace("\r\n", "\n")


def top_defs(source: str) -> dict:
    tree = ast.parse(source)
    lines = source.split("\n")
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            start = (node.decorator_list[0].lineno if node.decorator_list else node.lineno) - 1
            out[node.name] = "\n".join(lines[start:node.end_lineno]) + "\n"
    return out


def methods(source: str, class_name: str) -> dict:
    tree = ast.parse(source)
    lines = source.split("\n")
    cls = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name][0]
    return {n.name: "\n".join(lines[n.lineno - 1:n.end_lineno]) + "\n"
            for n in cls.body if isinstance(n, ast.FunctionDef)}


def method_node(source: str, class_name: str, name: str):
    """The method's AST node parsed in its module (string literals keep their own indentation)."""
    tree = ast.parse(source)
    cls = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name][0]
    return [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name][0]


def body_after_docstring(source: str) -> str:
    node = ast.parse(textwrap.dedent(source)).body[0]
    lines = textwrap.dedent(source).split("\n")
    first = node.body[0]
    is_doc = isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) \
        and isinstance(first.value.value, str)
    start = first.end_lineno if is_doc else first.lineno - 1
    return textwrap.dedent("\n".join(lines[start:node.end_lineno])) + "\n"


def module_assign(source: str, name: str):
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


# ---------------------------------------------------------------------------
# translator_gui -> shutdown_utils (debug cache sweep)
# ---------------------------------------------------------------------------

def _expected_sweep_block() -> str:
    tg = frozen("src/translator_gui.py")
    block = tg[tg.index("def _fmt_bytes(n: int) -> str:\n"):tg.index("def _log_mei_cleanup_on_exit():\n")]
    block = block.replace("def _fmt_bytes(n: int) -> str:", "def fmt_bytes(n: int) -> str:")
    block = block.replace('def _sweep_size_capped_dir(folder: str, max_bytes: int, label: str = "") -> tuple:',
                          'def sweep_size_capped_dir(folder: str, max_bytes: int, label: str = "") -> tuple:')
    block = block.replace("_fmt_bytes(", "fmt_bytes(").replace("_sweep_size_capped_dir(", "sweep_size_capped_dir(")
    block = block.replace(
        'def _sweep_large_caches(max_bytes: int = 400 * 1024 * 1024, phase: str = "startup") -> None:',
        'def sweep_large_caches(max_bytes: int = 400 * 1024 * 1024, phase: str = "startup", *,\n'
        '                       script_file: Optional[str] = None, extra_roots: Iterable[str] = ()) -> None:')
    block = block.replace("            roots.append(os.path.dirname(os.path.abspath(__file__)))\n",
                          "            roots.append(os.path.dirname(os.path.abspath(script_file or __file__)))\n")
    cwd = ("        try:\n            roots.append(os.path.abspath(os.getcwd()))\n"
           "        except Exception:\n            pass\n")
    assert block.count(cwd) == 1
    block = block.replace(cwd, cwd + (
        "        # Glossarion Mobile: the app data / logs folders where its Payloads and http_requests live.\n"
        "        for extra in extra_roots or ():\n"
        "            if extra:\n"
        "                roots.append(os.path.abspath(os.fspath(extra)))\n"))
    return block.rstrip("\n") + "\n"


def test_sweep_functions_are_the_frozen_bodies():
    su = current("shutdown_utils.py")
    moved = su[su.index("def fmt_bytes(n: int) -> str:\n"):].rstrip("\n") + "\n"
    assert moved == _expected_sweep_block()


def test_translator_gui_calls_the_shared_sweep():
    tg = current("translator_gui.py")
    defs = top_defs(tg)
    assert "_fmt_bytes" not in defs and "_sweep_size_capped_dir" not in defs
    assert "from shutdown_utils import fmt_bytes as _fmt_bytes" in tg
    assert "from shutdown_utils import sweep_size_capped_dir as _sweep_size_capped_dir" in tg
    assert "sweep_large_caches(max_bytes, phase, script_file=__file__)" in defs["_sweep_large_caches"]


def test_shared_sweep_caps_the_extra_roots(tmp_path):
    import shutdown_utils

    payloads = tmp_path / "data" / "Payloads"
    payloads.mkdir(parents=True)
    for i in range(4):
        (payloads / f"p{i}.json").write_bytes(b"x" * 1000)
        os.utime(payloads / f"p{i}.json", (1000 + i, 1000 + i))
    shutdown_utils.sweep_large_caches(2500, "test", script_file=str(tmp_path / "nowhere" / "x.py"),
                                      extra_roots=[str(tmp_path / "data")])
    left = sorted(p.name for p in payloads.iterdir())
    assert len(left) < 4 and "p3.json" in left  # the oldest go first
    assert shutdown_utils.fmt_bytes(2048) == frozen_fmt_bytes(2048)


def frozen_fmt_bytes(n):
    namespace: dict = {}
    source = top_defs(frozen("src/translator_gui.py"))["_fmt_bytes"]
    exec(compile(source, "frozen_fmt_bytes", "exec"), namespace)
    return namespace["_fmt_bytes"](n)


# ---------------------------------------------------------------------------
# translator_gui -> model_options (provider information)
# ---------------------------------------------------------------------------

def test_provider_info_is_the_frozen_text():
    import model_options

    node = method_node(frozen("src/translator_gui.py"), "TranslatorGUI", "_show_model_info_dialog")
    assign = next(n for n in node.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "info_text" for t in n.targets))
    assert model_options.PROVIDER_INFO_HTML == ast.literal_eval(assign.value)
    assert model_options.provider_info_html() == model_options.PROVIDER_INFO_HTML
    rewired = methods(current("translator_gui.py"), "TranslatorGUI")["_show_model_info_dialog"]
    assert "info_text = provider_info_html()" in rewired and "<h3>" not in rewired


# ---------------------------------------------------------------------------
# other_settings -> key_pool_service (Test Connections), metadata_defaults (header help)
# ---------------------------------------------------------------------------

def _frozen_other_settings_blocks():
    text = frozen("src/other_settings.py")
    cs = text.index("    # Collect all configured endpoints\n    endpoints_to_test = []\n")
    ce = text.index("    if not endpoints_to_test:\n        msg_box = QMessageBox()\n", cs)
    head = "    def run_tests_background():\n        results = []\n"
    ls = text.index(head + "        for endpoint_info in endpoints_to_test:\n")
    le = text.index("        if not cancel_event.is_set():\n            self.conn_test_bridge.finished.emit(results)\n", ls)
    return text[cs:ce], text[ls + len(head):le]


def test_endpoint_collection_and_probe_are_the_frozen_bodies():
    collect_block, loop_block = _frozen_other_settings_blocks()
    defs = top_defs(current("key_pool_service.py"))
    expected_collect = textwrap.dedent(collect_block).replace("self.", "owner.").replace("hasattr(self,", "hasattr(owner,")
    got_collect = body_after_docstring(defs["collect_test_endpoints"])
    assert got_collect == expected_collect.rstrip("\n") + "\nreturn endpoints_to_test\n"
    got_run = body_after_docstring(defs["run_endpoint_tests"])
    assert got_run == "results = []\n" + textwrap.dedent(loop_block).rstrip("\n") + "\nreturn results\n"


def test_other_settings_calls_the_shared_endpoint_code():
    text = current("other_settings.py")
    assert "endpoints_to_test = collect_test_endpoints(self)" in text
    assert "results = run_endpoint_tests(endpoints_to_test, api_key, openai, cancel_event)" in text
    assert "Collect all configured endpoints\n    endpoints_to_test = []" not in text


def test_collect_test_endpoints_reads_the_owner_vars():
    from key_pool_service import collect_test_endpoints

    owner = types.SimpleNamespace(
        use_custom_openai_endpoint_var=True, openai_base_url_var="http://localhost:1234/v1",
        azure_api_version_var="2024-02-01", model_var="gpt-4o", groq_base_url_var="",
        fireworks_base_url_var="", use_gemini_openai_endpoint_var=False, gemini_openai_endpoint_var="")
    endpoints = collect_test_endpoints(owner)
    assert any("localhost:1234" in str(item[1]) for item in endpoints)


def test_run_endpoint_tests_stops_on_cancel():
    import threading

    from key_pool_service import run_endpoint_tests

    cancel = threading.Event()
    cancel.set()
    assert run_endpoint_tests([("X", "http://127.0.0.1:9/v1", "m")], "key", None, cancel) == []


def test_header_help_sections_are_the_frozen_literal():
    import metadata_defaults

    text = frozen("src/other_settings.py")
    cls = text.index("class HeaderTranslationHelpDialog(QDialog):")
    marker = "        # Create sections with detailed explanations\n        sections = "
    start = text.index(marker, cls) + len(marker)
    end = text.index("\n        ]\n", start) + len("\n        ]")
    assert metadata_defaults.HEADER_HELP_SECTIONS == ast.literal_eval(textwrap.dedent("        " + text[start:end]))
    assert "sections = HEADER_HELP_SECTIONS" in current("other_settings.py")


# ---------------------------------------------------------------------------
# Retranslation_GUI -> progress_core / progress_actions
# ---------------------------------------------------------------------------

def test_untranslated_manual_entries_is_the_frozen_closure():
    text = frozen("src/Retranslation_GUI.py")
    start = text.index("            untranslated_entries = []\n            for entry in current_spine_chapters:\n")
    end = text.index("            return untranslated_entries\n", start) + len("            return untranslated_entries\n")
    expected = textwrap.dedent(text[start:end]).replace("self._progress_display_status", "owner._progress_display_status")
    expected = expected.replace("current_spine_chapters", "spine_chapters")
    got = body_after_docstring(top_defs(current("progress_core.py"))["untranslated_manual_entries"])
    assert got == expected
    assert "untranslated_manual_entries(\n                self, current_spine_chapters, status_data\n            )" \
        in current("Retranslation_GUI.py")


def test_untranslated_manual_entries_behaviour():
    from progress_core import untranslated_manual_entries

    owner = types.SimpleNamespace(_progress_display_status=lambda entry, data: entry.get("s"))
    rows = [{"s": "completed"}, {"s": "pending", "n": 1}, "junk", {"s": "not_translated", "n": 2}]
    got = untranslated_manual_entries(owner, rows, {})
    assert [r["n"] for r in got] == [1, 2] and all(r["status"] == "not_translated" for r in got)


def test_qa_issue_search_target_is_the_frozen_block():
    text = frozen("src/Retranslation_GUI.py")
    start = text.index("                if qa_issues:\n                    # Extract a meaningful search term from the QA issue strings\n")
    end = text.index("                    # Copy search term to clipboard\n                    if search_term:\n", start)
    block = text[start + len("                if qa_issues:\n"):end]
    got = body_after_docstring(top_defs(current("progress_actions.py"))["qa_issue_search_target"])
    expected = ("search_term = None\n_line_num = 1\n" + textwrap.dedent(block).rstrip("\n")
                + "\nreturn search_term, _line_num\n")
    assert got == expected
    assert "search_term, _line_num = qa_issue_search_target(qa_file_path, qa_issues)" in current("Retranslation_GUI.py")


def test_qa_issue_search_target_finds_the_quoted_term(tmp_path):
    from progress_actions import qa_issue_search_target

    path = tmp_path / "out.html"
    path.write_text("<p>one</p>\n<p>two 你好 three</p>\n", encoding="utf-8")
    term, line = qa_issue_search_target(str(path), ["Untranslated text: '你好'"])
    assert term == "你好" and line == 2


# ---------------------------------------------------------------------------
# manga_integration / manga_settings_dialog -> manga_models / manga_settings_defaults / manga_env
# ---------------------------------------------------------------------------

def _frozen_manga_method(name):
    return methods(frozen("src/manga_integration.py"), "MangaTranslationTab")[name]


def test_model_info_is_the_frozen_text():
    import manga_models

    node = method_node(frozen("src/manga_integration.py"), "MangaTranslationTab", "_show_model_info")
    info = next(n for n in ast.walk(node) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "info" for t in n.targets))
    assert manga_models.MODEL_INFO == ast.literal_eval(info.value)
    assert manga_models.model_info("nope") == "Please select a model type first"
    assert "from manga_models import MODEL_INFO as info" in current("manga_integration.py")


def test_rendering_reset_values_are_the_frozen_assignments():
    from manga_settings_defaults import RENDERING_RESET_VALUES

    method = textwrap.dedent(_frozen_manga_method("_reset_rendering_to_defaults"))
    cut = method[:method.index("# Update UI widgets")]
    pairs = []
    for node in ast.walk(ast.parse(cut + "\n    except Exception:\n        pass\n")):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                pairs.append((node.lineno, target.attr, ast.literal_eval(node.value)))
    assert [(a, v) for _l, a, v in sorted(pairs)] == list(RENDERING_RESET_VALUES.items())
    rewired = methods(current("manga_integration.py"), "MangaTranslationTab")["_reset_rendering_to_defaults"]
    assert "for _attr, _value in RENDERING_RESET_VALUES.items():" in rewired


def test_mask_presets_are_the_frozen_buttons():
    from manga_settings_defaults import MASK_PRESETS

    text = frozen("src/manga_settings_dialog.py")
    found = {}
    for var, args in re.findall(r"(\w+)_btn\.clicked\.connect\(lambda: self\._set_mask_preset\(([^)]*)\)\)", text):
        label = re.search(rf'{var}_btn = QPushButton\("([^"]+)"\)', text).group(1)
        found[var] = (label, tuple(ast.literal_eval(f"({args},)")))
    assert found == MASK_PRESETS
    rewired = current("manga_settings_dialog.py")
    for pid in MASK_PRESETS:
        assert f"self._set_mask_preset(*MASK_PRESETS['{pid}'][1])" in rewired


def _span_replace(text, start_marker, end_marker, new):
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    return text[:start] + new + text[end:]


def _replace_once(text, old, new):
    assert text.count(old) == 1, old[:60]
    return text.replace(old, new)


def test_manga_desktop_rewires_are_exact():
    """The four desktop methods U9 rewired equal their frozen text with exactly the listed
    replacements (tests/test_manga_env.py exempts them from its unchanged-method check)."""
    old = methods(frozen("src/manga_integration.py"), "MangaTranslationTab")
    new = methods(current("manga_integration.py"), "MangaTranslationTab")
    # _show_model_info: the info literal -> manga_models.MODEL_INFO
    text = old["_show_model_info"]
    start = text.index("        info = {" + NL)
    end = text.index(NL + "        }" + NL, start) + len(NL + "        }" + NL)
    expected = text[:start] + (
        "        # The model texts (manga_models.MODEL_INFO, moved in U9: the mobile Model manager ⓘ shows them)" + NL
        + "        from manga_models import MODEL_INFO as info" + NL) + text[end:]
    assert new["_show_model_info"] == expected
    # _reset_rendering_to_defaults: the assignments -> RENDERING_RESET_VALUES
    expected = _span_replace(old["_reset_rendering_to_defaults"],
                             "        try:" + NL + "            # Background settings" + NL,
                             "            # Update UI widgets" + NL, NL.join([
                                 "        try:",
                                 "            # The default values (manga_settings_defaults.RENDERING_RESET_VALUES, moved in U9; Glossarion",
                                 "            # Mobile's Rendering › Reset writes the same values)",
                                 "            from manga_settings_defaults import RENDERING_RESET_VALUES",
                                 "            for _attr, _value in RENDERING_RESET_VALUES.items():",
                                 "                setattr(self, _attr, _value)",
                                 "            ",
                                 "",
                             ]))
    assert new["_reset_rendering_to_defaults"] == expected
    # _test_custom_image_edit_endpoint: URL normalisation + probe -> manga_env
    text = old["_test_custom_image_edit_endpoint"]
    text = _replace_once(text, NL.join([
        "            if not url.startswith(('http://', 'https://')):",
        "                lower = url.lower()",
        "                url = ('http://' if lower.startswith(('localhost', '127.', '0.0.0.0', '[')) else 'https://') + url",
        "            url = url.rstrip('/')",
    ]), NL.join([
        "            # manga_env.normalize_custom_image_edit_url / probe_custom_image_edit_endpoint (moved in U9)",
        "            from manga_env import normalize_custom_image_edit_url, probe_custom_image_edit_endpoint",
        "            url = normalize_custom_image_edit_url(url)",
    ]))
    start = text.index("            headers = {" + NL)
    end = text.index("        except Exception as e:" + NL, start)
    text = text[:start] + NL.join([
        "            box, status, color, title, message = probe_custom_image_edit_endpoint(",
        "                url, self.main_gui.config, http_get=requests.get",
        "            )",
        "            self.local_model_status_label.setText(status)",
        "            self.local_model_status_label.setStyleSheet(f\"color: {color};\")",
        "            if box == 'warning':",
        "                QMessageBox.warning(self.dialog, title, message)",
        "            else:",
        "                QMessageBox.information(self.dialog, title, message)",
        "",
    ]) + text[end:]
    assert new["_test_custom_image_edit_endpoint"] == text
    # every other tab method is unchanged by U9
    for name in set(old) & set(new):
        if name not in ("_show_model_info", "_reset_rendering_to_defaults", "_test_custom_image_edit_endpoint"):
            assert new[name] == old[name], name
    # MangaSettingsDialog._create_inpainting_tab: the preset arguments -> MASK_PRESETS
    old_d = methods(frozen("src/manga_settings_dialog.py"), "MangaSettingsDialog")
    new_d = methods(current("manga_settings_dialog.py"), "MangaSettingsDialog")
    text = old_d["_create_inpainting_tab"]
    for pid, args in (("bw_manga", "15, False, 2, 2, 3, 0"), ("colored", "15, False, 2, 2, 3, 3"),
                      ("uniform", "0, True, 2, 2, 2, 0")):
        btn = "colored" if pid == "colored" else ("uniform" if pid == "uniform" else "bw_manga")
        text = _replace_once(text, f"{btn}_btn.clicked.connect(lambda: self._set_mask_preset({args}))",
                             f"{btn}_btn.clicked.connect(lambda: self._set_mask_preset(*MASK_PRESETS['{pid}'][1]))")
    assert new_d["_create_inpainting_tab"] == text
    for name in set(old_d) & set(new_d):
        if name != "_create_inpainting_tab":
            assert new_d[name] == old_d[name], name


def test_mask_preset_updates_mirror_the_dialog_save():
    from manga_settings_defaults import mask_preset_updates

    updates = mask_preset_updates("colored")
    assert updates["manga_settings.mask_dilation"] == 15
    assert updates["manga_settings.free_text_dilation_iterations"] == 3
    assert updates["manga_settings.bubble_dilation_iterations"] == updates["manga_settings.text_bubble_dilation_iterations"]
    assert mask_preset_updates("nope") == {}


class _Resp:
    def __init__(self, code):
        self.status_code = code


@pytest.mark.parametrize("code", [200, 401, 403, 500])
def test_image_edit_probe_keeps_the_desktop_texts(code):
    from manga_env import probe_custom_image_edit_endpoint

    calls = []

    def http_get(url, headers=None, timeout=None):
        calls.append((url, headers, timeout))
        return _Resp(code)

    box, status, color, title, message = probe_custom_image_edit_endpoint("https://e.test/v1", {"api_key": "k"},
                                                                          http_get=http_get)
    assert calls[0][0] == "https://e.test/v1/models" and calls[0][2] == 10
    frozen_text = textwrap.dedent(_frozen_manga_method("_test_custom_image_edit_endpoint"))
    for text in (status, title):
        assert text.split("{")[0] in frozen_text or text.split(str(code))[0] in frozen_text
    if code == 200:
        assert (box, color) == ("information", "green") and message == "Endpoint is reachable:\nhttps://e.test/v1"
    elif code in (401, 403):
        assert (box, color) == ("warning", "orange")
        assert message == f"Endpoint responded with authentication error ({code})."
    else:
        assert status == f"Endpoint responded: HTTP {code}" and message == f"Endpoint responded with HTTP {code}."


def test_image_edit_url_normalisation_is_the_frozen_rule():
    from manga_env import normalize_custom_image_edit_url, test_custom_image_edit_endpoint

    assert normalize_custom_image_edit_url("localhost:8080/v1/") == "http://localhost:8080/v1"
    assert normalize_custom_image_edit_url("api.example.test") == "https://api.example.test"
    assert normalize_custom_image_edit_url("https://x.test/") == "https://x.test"
    ok, status, _message = test_custom_image_edit_endpoint({"use_custom_image_edit_endpoint": False})
    assert ok and status == "Using current image provider/model"
    rewired = methods(current("manga_integration.py"), "MangaTranslationTab")["_test_custom_image_edit_endpoint"]
    assert "probe_custom_image_edit_endpoint(" in rewired and "normalize_custom_image_edit_url(url)" in rewired


def test_shared_modules_parse_as_python_310():
    for name in ("shutdown_utils.py", "model_options.py", "key_pool_service.py", "metadata_defaults.py",
                 "progress_core.py", "progress_actions.py", "manga_models.py", "manga_settings_defaults.py",
                 "manga_env.py", "manga_editor_core.py", "direct_text_stream.py", "translation_pipeline.py"):
        ast.parse(current(name), filename=name, feature_version=(3, 10))
