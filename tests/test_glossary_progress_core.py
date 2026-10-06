"""U5: glossary_progress_core (Glossary Progress panel core) parity and API tests.

* V: the lookup closures, the panel's data closures and the module helpers equal the
  frozen Retranslation_GUI (``tests/parity/progress_legacy.BASE_SHA``) modulo the
  documented edits;
* D: the frozen panel closures (RG 23524-24546 + 24872-24896, executed as a function)
  vs ``make_glossary_progress_model`` on >= 500 random progress files: status cache,
  row text/colour, Minimal-pass and refinement rows, legend statistics;
* F: the offscreen Glossary Progress dialog (legacy vs working tree): rows, colours,
  legend labels; Mark as Completed and Remove from progress through the context menu
  -> the progress JSON;
* C: both writes wait for the extractor's progress-file lock and keep its save;
* M: the mobile API (open / rows / stats / mark / remove / footnotes / summary).
"""

from __future__ import annotations

import ast
import copy
import json
import os
import random
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
for _p in (str(TESTS_DIR), str(SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from parity import progress_legacy as pl  # noqa: E402

import glossary_progress_core as gpc  # noqa: E402
import progress_core as pc  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("GLOSSARION_LIBRARY_DIR", str(tmp_path / "_library"))
    # Workspaces resolve under $OUTPUT_DIRECTORY / $OUTPUT_DIR before the config's output
    # directory: never let a caller's value send them outside tmp_path.
    monkeypatch.delenv("OUTPUT_DIRECTORY", raising=False)
    monkeypatch.delenv("OUTPUT_DIR", raising=False)
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


def _source(name):
    return (SRC_DIR / f"{name}.py").read_text(encoding="utf-8").replace("\r\n", "\n")


def _legacy_lines(start, end, strip, add=0):
    """Frozen RG lines re-indented like the extraction (whitespace-only lines keep
    whatever indentation is left after removing ``strip`` spaces)."""
    out = []
    for line in pl.legacy_source_lines()[start - 1:end]:
        if line.strip():
            assert line.startswith(" " * strip), (start, line)
            out.append(" " * add + line[strip:])
        elif line.startswith(" " * strip):
            out.append(line[strip:])
        else:
            out.append(line.lstrip(" "))
    return "\n".join(out)


def _apply(text, edits):
    for old, new in edits:
        assert text.count(old) == 1, (old, text.count(old))
        text = text.replace(old, new)
    return text


# ===========================================================================
# Tier V
# ===========================================================================

MODULE_HELPERS = {
    "_glossary_progress_filename_keys": (1403, 1412),
    "_filter_glossary_source_chapter_map": (1415, 1457),
    "_parallel_glossary_progress_filename_aliases": (1460, 1470),
    "_map_zero_based_glossary_progress_index": (1473, 1491),
    "_combine_glossary_progress_legend_stats": (1875, 1936),
    "_glossary_refinement_type_key": (1939, 1950),
    "_merge_glossary_refinement_row_info": (1953, 1994),
    "_glossary_refinement_row_detail": (1997, 2040),
    "_glossary_refinement_manual_completion_info": (2043, 2094),
    "_normalize_glossary_refinement_selection": (2097, 2111),
    "_find_matching_glossary_refinement_aggregate": (2114, 2136),
    "_derive_glossary_refinement_aggregate_status": (2152, 2169),
}


@pytest.mark.parametrize("name", sorted(MODULE_HELPERS))
def test_module_helpers_are_verbatim(name):
    start, end = MODULE_HELPERS[name]
    tree = ast.parse(_source("glossary_progress_core"))
    lines = _source("glossary_progress_core").split("\n")
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    assert "\n".join(lines[node.lineno - 1:node.end_lineno]) == _legacy_lines(start, end, 0)


#: (frozen RG range, strip, add, edits) blocks the two factories hold verbatim.
BLOCKS = [
    (22030, 22110, 4, 0, [("            from translator_gui import _get_app_dir\n",
                           "            from app_paths import _get_app_dir\n")]),
    (22136, 22159, 4, 0, []),
    (22543, 22649, 4, 0, []),
    (23524, 23574, 8, 0, []),
    (23575, 24546, 8, 0, [("(_d or gp_data).get('chapter_numbers', {})",
                           "(_d or _current_gp_data()).get('chapter_numbers', {})")]),
    (24872, 24896, 8, 0, []),
    (24932, 25139, 8, 0, [("        if changed:\n            with open(_rp, 'w', encoding='utf-8') as _f:\n"
                           "                json.dump(_d, _f, ensure_ascii=False, indent=2)\n",
                           "        if changed:\n            _write_glossary_progress_atomic(_rp, _d)\n")]),
    (25287, 25412, 8, 0, []),
    (25575, 25799, 8, 0, []),
    (25817, 25842, 8, 0, []),
    (26443, 26481, 8, 0, []),
    (26165, 26232, 8, 0, []),
]


@pytest.mark.parametrize("start,end,strip,add,edits", BLOCKS, ids=[f"{b[0]}-{b[1]}" for b in BLOCKS])
def test_factory_blocks_are_verbatim(start, end, strip, add, edits):
    block = _apply(_legacy_lines(start, end, strip, add), edits)
    assert block in _source("glossary_progress_core")


def test_desktop_panel_binds_the_shared_closures():
    source = _source("Retranslation_GUI")
    assert "_gp_model = make_glossary_progress_model(" in source
    assert "_gp_locator = glossary_progress_locator(" in source
    for name in ("_gp_status_cache", "_gp_display_for", "_gp_apply_mark_completed_to_progress",
                 "_gp_apply_remove_from_progress", "_find_glossary_file", "_gp_write_completed_summary"):
        assert f"            {name} = _gp_model.{name}" in source
        assert f"def {name}(" not in source.split("def _build_gp_panel(")[1].split("def _show_glossary_progress(")[0]


# ===========================================================================
# Tier D: frozen panel closures vs the factory
# ===========================================================================


def _legacy_model(owner, fp, gp_path, *, expected_entries, refinement_type_key, source_filenames=None):
    """The frozen ``_build_gp_panel`` data closures (RG 23524-24546 + 24872-24896)."""
    body = (_legacy_lines(23524, 24546, 12, 4) + "\n" + _legacy_lines(24872, 24896, 12, 4))
    source = (
        "def legacy_gp(self, fp, gp_path, pump_loading, glossary_progress_source_filenames,\n"
        "              _glossary_refinement_expected_entries, _refinement_type_key):\n"
        + body + "\n    return dict(locals())\n"
    )
    namespace = dict(vars(pl.legacy_rg()))
    exec(compile(source, "<legacy _build_gp_panel>", "exec"), namespace)
    return namespace["legacy_gp"](owner, fp, gp_path, None, source_filenames, expected_entries, refinement_type_key)


def _legacy_locator(owner, file_path):
    body = textwrap.indent(_legacy_lines(22543, 22649, 8, 0), "    ")
    source = ("def legacy_locator(self, file_path, glossary_progress_source_path):\n" + body
              + "\n    return dict(locals())\n")
    namespace = dict(vars(pl.legacy_rg()))
    exec(compile(source, "<legacy locator>", "exec"), namespace)
    return namespace["legacy_locator"](owner, file_path, None)


class _Owner:
    def __init__(self, config):
        self.config = config
        self.special_file_keywords_var = "title, notice"
        self.special_file_exact_var = "index"

    def _is_special_file(self, filename):
        from translation_pipeline import GlossaryPipelineMixin
        return GlossaryPipelineMixin._is_special_file(self, filename)


def _epub(path, count):
    names = ["title.xhtml"] + [f"chapter{i:04d}.xhtml" for i in range(1, count + 1)] + ["notice.xhtml"]
    pl.make_epub(path, [(name, f"<p>{name}</p>") for name in names])
    return names


STATUSES = ("completed", "failed", "qa_failed", "error", "merged", "in_progress", "skipped_empty",
            "skipped_image_only", "skipped_title_header_only", "", None)
REF_STATUSES = ("completed", "skipped", "in_progress", "failed", "error", "not_refined", "partially_in_progress")


def _random_gp(rng, count):
    data = {"book_title": rng.choice(["", "Book"])}
    if rng.random() < 0.7:
        data["indexing"] = "chapter_index_zero_based"
        data["chapter_filenames"] = {
            str(i): f"chapter{i + 1:04d}.xhtml" for i in range(count) if rng.random() < 0.9
        }
    if rng.random() < 0.3:
        data["chapter_positions"] = {str(i): i for i in range(count)}
    if rng.random() < 0.3:
        data["chapter_numbers"] = {str(i): i + 1 for i in range(count)}
    for key in ("completed", "skipped", "failed", "merged_indices", "in_progress"):
        if rng.random() < 0.6:
            data[key] = rng.sample(range(count + 2), rng.randint(0, min(count, 4)))
    chapters = {}
    for _ in range(rng.randint(0, count)):
        ci = rng.randrange(count + 1)
        entry = {"status": rng.choice(STATUSES)}
        if rng.random() < 0.6:
            entry["chapter_index"] = ci
        if rng.random() < 0.5:
            entry["actual_num"] = ci + 1
        if rng.random() < 0.4:
            entry["output_file"] = f"chapter{ci + 1:04d}.xhtml"
        if rng.random() < 0.5:
            entry["model_name"] = rng.choice(["g-1", "SKIPPED", ""])
        if rng.random() < 0.3:
            entry["qa_issues_found"] = rng.sample(["low", "dup", "x"], rng.randint(1, 3))
        if rng.random() < 0.2:
            entry["refinement_status"] = rng.choice(["refined", "failed"])
        if rng.random() < 0.2:
            entry["previous_progress_entry"] = {"status": "completed", "model_name": "g-old"}
        chapters[rng.choice([str(ci), f"x{ci}"])] = entry
    if chapters or rng.random() < 0.5:
        data["chapters"] = chapters
    if rng.random() < 0.4:
        data["qa_issues_found"] = {str(rng.randrange(count)): ["issue"] for _ in range(rng.randint(1, 3))}
    if rng.random() < 0.5:
        data["minimal_pass"] = {"status": rng.choice(["completed", "skipped", "failed", "error", "in_progress"]),
                                "reason": rng.choice(["", "no_entries", "stopped"]),
                                "entry_count": rng.choice([None, 0, 3, "x"]), "model_name": rng.choice(["", "m"]),
                                "error": rng.choice(["", "boom"])}
    if rng.random() < 0.6:
        refinement = {}
        for entry_type in rng.sample(["character", "terms", "surnames", "titles", "Term"], rng.randint(1, 3)):
            refinement[f"type::{entry_type}"] = {
                "entry_type": entry_type, "status": rng.choice(REF_STATUSES),
                "entry_count_before": rng.randint(0, 3), "entry_count_after": rng.randint(0, 3),
                "total_chunks": rng.choice([None, 2]), "completed_chunks": 1,
                "reason": rng.choice(["", "no_entries"]), "model_name": rng.choice(["", "r"]),
            }
        if rng.random() < 0.5:
            refinement["all::character,terms"] = {"status": rng.choice(REF_STATUSES)}
        data["refinement"] = refinement
    return data


def test_panel_closures_fuzz_match_the_frozen_panel(tmp_path):
    rng = random.Random(5201)
    count = 8
    fp = tmp_path / "Book.epub"
    _epub(fp, count)
    gdir = tmp_path / "Glossary" / "Book"
    gdir.mkdir(parents=True)
    (gdir / "Book_glossary.csv").write_text(pl.GLOSSARY_CSV, encoding="utf-8")
    gp_path = gdir / "Book_glossary_progress.json"
    checked = 0
    for case in range(500):
        data = _random_gp(rng, count)
        gp_path.write_text(json.dumps(data), encoding="utf-8")
        config = {"glossary_add_minimal_pass": rng.random() < 0.5,
                  "glossary_refinement_chunking_mode": rng.choice(["all", "separate"])}
        if rng.random() < 0.5:
            config["custom_entry_types"] = {"character": {"enabled": True}, "terms": {"enabled": rng.random() < 0.7},
                                            "places": {"enabled": True}}
        owner = _Owner(config)
        legacy_loc = _legacy_locator(owner, str(fp))
        legacy = _legacy_model(owner, str(fp), str(gp_path),
                               expected_entries=legacy_loc["_glossary_refinement_expected_entries"],
                               refinement_type_key=legacy_loc["_refinement_type_key"])
        locator = gpc.glossary_progress_locator(owner, str(fp))
        model = gpc.make_glossary_progress_model(
            owner, str(fp), str(gp_path),
            find_gp_for_file=lambda _fp, _p=str(gp_path): _p,
            glossary_refinement_expected_entries=locator._glossary_refinement_expected_entries,
            refinement_type_key=locator._refinement_type_key,
        )
        d = json.loads(json.dumps(data))
        assert model.panel_state["chapter_map"] == legacy["panel_state"]["chapter_map"], case
        assert model.panel_state["total"] == legacy["panel_state"]["total"], case
        cache = model._gp_status_cache(copy.deepcopy(d))
        legacy_cache = legacy["_gp_status_cache"](copy.deepcopy(d))
        assert {k: v for k, v in cache.items() if k != "entries_by_ci"} == \
            {k: v for k, v in legacy_cache.items() if k != "entries_by_ci"}, case
        for ci in range(model.panel_state["total"]):
            fname = model.panel_state["chapter_map"].get(ci, f"chapter {ci + 1}")
            row = model._gp_display_for(ci, fname, copy.deepcopy(d))
            assert row == legacy["_gp_display_for"](ci, fname, copy.deepcopy(d)), (case, ci)
            assert model._gp_color_for(row[1]) == legacy["_gp_color_for"](row[1])
        assert model._gp_minimal_pass_row(copy.deepcopy(d)) == legacy["_gp_minimal_pass_row"](copy.deepcopy(d)), case
        assert model._gp_refinement_rows(copy.deepcopy(d)) == legacy["_gp_refinement_rows"](copy.deepcopy(d)), case
        assert model._gp_stats_for_dict(copy.deepcopy(d)) == legacy["_gp_stats_for_dict"](copy.deepcopy(d)), case
        checked += 1
    assert checked == 500


# ===========================================================================
# Tier F: the real dialogs
# ===========================================================================


@pytest.fixture(scope="module")
def gp_fixture(tmp_path_factory):
    base = tmp_path_factory.mktemp("gp")
    source, _out, config = pl.epub_workspace(base)
    pl.glossary_progress_fixture(base, source)
    return base, Path(source).name, config


def _open_gp(module, fixture, work, *, remove_progress=False):
    base, source_name, config = fixture
    work = pl.copy_workspace(base, work)
    if remove_progress:
        (work / "out" / "Glossary" / "Book" / "Book_glossary_progress.json").unlink()
    host = pl.make_host(module, dict(config, output_directory=str(work / "out")))
    data = pl.open_progress_manager(host, work / source_name)
    data["dialog"]._show_glossary_progress()
    pl.pump(30, timeout=1.5)
    gp_dialog = data["dialog"]._glossary_progress_dialog
    return work, host, data, gp_dialog


def _gp_snapshot(gp_dialog, work):
    from PySide6.QtWidgets import QLabel, QListWidget

    rows = [[(lb.item(i).text(), lb.item(i).foreground().color().name()) for i in range(lb.count())]
            for lb in gp_dialog.findChildren(QListWidget)]
    labels = sorted(
        label.text().replace(str(work), "<ROOT>") + ("" if label.isVisibleTo(gp_dialog) else " [hidden]")
        for label in gp_dialog.findChildren(QLabel) if label.text()
    )
    return rows, labels


@pytest.mark.parametrize("remove_progress", (False, True))
def test_glossary_progress_dialog_matches_frozen_desktop(remove_progress, gp_fixture, tmp_path):
    snapshots = []
    for side, module in (("legacy", pl.legacy_rg()), ("current", pl.current_rg())):
        work, _host, data, gp_dialog = _open_gp(module, gp_fixture, tmp_path / side, remove_progress=remove_progress)
        snapshots.append(_gp_snapshot(gp_dialog, work))
        gp_dialog.hide()
        data["dialog"].hide()
    assert snapshots[0] == snapshots[1]
    assert snapshots[0][0][0], "no rows"


def _gp_context_action(gp_dialog, label_prefix, rows):
    """Select GP rows by index and pick a context-menu action (the menu is created via a
    local ``from PySide6.QtWidgets import QMenu``, so the class itself is patched)."""
    import PySide6.QtWidgets as widgets
    from PySide6.QtWidgets import QListWidget

    listbox = max(gp_dialog.findChildren(QListWidget), key=lambda lb: lb.count())
    listbox.clearSelection()
    for index in rows:
        listbox.item(index).setSelected(True)
    original = widgets.QMenu
    chosen = {}

    class AutoMenu(original):
        def exec(self, *args, **kwargs):
            for action in self.actions():
                if action.text().startswith(label_prefix):
                    chosen["text"] = action.text()
                    return action
            return None

    widgets.QMenu = AutoMenu
    try:
        listbox.customContextMenuRequested.emit(listbox.visualItemRect(listbox.item(rows[0])).center())
        pl.pump(20, timeout=0.3)
    finally:
        widgets.QMenu = original
    return chosen.get("text")


GP_ACTIONS = {
    "mark_completed": ("✅ Mark as Completed", [3, 5, 8, 11]),   # failed, in-progress, not completed, refinement
    "remove": ("🗑️ Remove", [0, 1, 3, 10]),                       # minimal pass, completed, failed, refinement
}


@pytest.mark.parametrize("action", sorted(GP_ACTIONS))
def test_glossary_progress_actions_match_frozen_desktop(action, gp_fixture, tmp_path):
    label, rows = GP_ACTIONS[action]
    results = []
    for side, module in (("legacy", pl.legacy_rg()), ("current", pl.current_rg())):
        work, _host, data, gp_dialog = _open_gp(module, gp_fixture, tmp_path / side)
        gp_file = work / "out" / "Glossary" / "Book" / "Book_glossary_progress.json"
        before = gp_file.read_bytes()
        chosen = _gp_context_action(gp_dialog, label, rows)
        pl.pump(30, until=lambda: gp_file.read_bytes() != before, timeout=5)
        pl.pump(30, timeout=0.5)
        snapshot = _gp_snapshot(gp_dialog, work)
        results.append((chosen, pl.normalize_progress(json.loads(gp_file.read_text(encoding="utf-8"))), snapshot))
        gp_dialog.hide()
        data["dialog"].hide()
    assert results[0][0] and results[0][0] == results[1][0]
    assert results[1][1] == results[0][1]
    assert results[1][2] == results[0][2]


# ===========================================================================
# Tier C: the extractor's lock
# ===========================================================================


@pytest.mark.parametrize("operation", ("mark", "remove"))
def test_writes_wait_for_the_extractor_lock_and_keep_its_save(operation, gp_fixture, tmp_path):
    from glossary_refinement import locked_progress_file

    base, source_name, config = gp_fixture
    work = pl.copy_workspace(base, tmp_path / "w")
    owner = pc.ProgressOwner(dict(config, output_directory=str(work / "out")))
    model = gpc.open_glossary_progress(owner, str(work / source_name), output_dir=str(work / "out" / "Book"))
    gp_file = Path(model.gp_path)
    rows = gpc.glossary_rows(model)
    target = next(row for row in rows if row.kind == "chapter" and row.status in ("failed", "qa_failed"))
    holding = threading.Event()
    release = threading.Event()

    def extractor():
        with locked_progress_file(str(gp_file)):
            holding.set()
            data = json.loads(gp_file.read_text(encoding="utf-8"))
            data["chapters"]["6"] = {"chapter_index": 6, "status": "completed", "model_name": "extractor"}
            data["completed"] = sorted(set(data.get("completed", [])) | {6})
            gp_file.write_text(json.dumps(data), encoding="utf-8")
            release.wait(5)

    thread = threading.Thread(target=extractor)
    thread.start()
    holding.wait(5)
    done = {}

    def run():
        if operation == "mark":
            done["result"] = gpc.mark_glossary_completed(model, [target])
        else:
            done["result"] = gpc.remove_glossary_progress(model, [target])

    worker = threading.Thread(target=run)
    worker.start()
    time.sleep(0.4)
    assert "result" not in done, "the write did not wait for the extractor's lock"
    release.set()
    worker.join(10)
    thread.join(10)
    saved = json.loads(gp_file.read_text(encoding="utf-8"))
    assert saved["chapters"]["6"]["model_name"] == "extractor"
    assert done["result"]["changed"]


# ===========================================================================
# Tier M: mobile API
# ===========================================================================


def test_mobile_glossary_rows_and_stats_match_the_dialog(gp_fixture, tmp_path):
    work, _host, data, gp_dialog = _open_gp(pl.legacy_rg(), gp_fixture, tmp_path / "desk")
    desk_rows, desk_labels = _gp_snapshot(gp_dialog, work)
    gp_dialog.hide()
    data["dialog"].hide()
    base, source_name, config = gp_fixture
    mobile = pl.copy_workspace(base, tmp_path / "mobile")
    owner = pc.ProgressOwner(dict(config, output_directory=str(mobile / "out")))
    assert gpc.find_glossary_progress(owner, str(mobile / source_name)).endswith("Book_glossary_progress.json")
    model = gpc.open_glossary_progress(owner, str(mobile / source_name))
    rows = gpc.glossary_rows(model)
    colours = {"#27ae60", "#94a3b8", "#17a2b8", "#f59e0b", "#7f5f00", "#e74c3c", "#5a9fd4"}
    assert [(row.display, row.color) for row in rows] == desk_rows[0]
    assert all(row.color in colours for row in rows)
    stats = gpc.glossary_stats(model)
    assert f"Total: {stats['total']} | " in desk_labels
    assert f"✅ Completed: {stats['completed']} | " in desk_labels
    assert f"⬜ Not Translated: {stats['remaining']} | " in desk_labels
    assert gpc.find_glossary_file(model).endswith("Book_glossary.csv")
    groups = gpc.GLOSSARY_STATUS_GROUPS
    assert sum(1 for row in rows if row.status in groups["failed"]) == stats["failed"]


def test_mobile_mark_and_remove_match_the_desktop(gp_fixture, tmp_path):
    base, source_name, config = gp_fixture
    desktop = {}
    for action in ("mark_completed", "remove"):
        label, rows = GP_ACTIONS[action]
        work, _host, data, gp_dialog = _open_gp(pl.legacy_rg(), gp_fixture, tmp_path / f"desk_{action}")
        gp_file = work / "out" / "Glossary" / "Book" / "Book_glossary_progress.json"
        before = gp_file.read_bytes()
        _gp_context_action(gp_dialog, label, rows)
        pl.pump(30, until=lambda: gp_file.read_bytes() != before, timeout=5)
        desktop[action] = pl.normalize_progress(json.loads(gp_file.read_text(encoding="utf-8")))
        gp_dialog.hide()
        data["dialog"].hide()
    for action, operation in (("mark_completed", gpc.mark_glossary_completed), ("remove", gpc.remove_glossary_progress)):
        work = pl.copy_workspace(base, tmp_path / f"mobile_{action}")
        owner = pc.ProgressOwner(dict(config, output_directory=str(work / "out")))
        model = gpc.open_glossary_progress(owner, str(work / source_name))
        rows = gpc.glossary_rows(model)
        targets = [rows[index] for index in GP_ACTIONS[action][1]]
        if action == "mark_completed":
            targets = [row for row in targets if row.status != "completed"]
        operation(model, targets)
        saved = pl.normalize_progress(json.loads(Path(model.gp_path).read_text(encoding="utf-8")))
        assert saved == desktop[action], action


def test_mobile_footnotes_summary_and_empty_state(gp_fixture, tmp_path):
    base, source_name, config = gp_fixture
    work = pl.copy_workspace(base, tmp_path / "m")
    owner = pc.ProgressOwner(dict(config, output_directory=str(work / "out")))
    model = gpc.open_glossary_progress(owner, str(work / source_name), output_dir=str(work / "out" / "Book"))
    notes = gpc.glossary_footnotes(model, [0, 6])
    assert notes["error"] is None
    assert notes["markdown"].strip()
    path, content = gpc.write_glossary_summary(model)
    assert Path(path).name == "Book_completed_glossary_footnotes.md"
    assert Path(path).read_text(encoding="utf-8") == content
    assert Path(path).parent == work / "out" / "Book" / "glossary_footnotes"
    (Path(model.gp_path)).unlink()
    assert gpc.open_glossary_progress(owner, str(work / source_name)) is None
    expected = gpc.glossary_refinement_expected(owner, [{"type": "character"}, {"type": "term"}])
    assert expected["all::character,terms,locations,nicknames,surnames,titles"]["current_entry_count"] == 2
    assert expected["type::character"]["status"] == "not_refined"
    assert expected["type::titles"]["status"] == "skipped"
