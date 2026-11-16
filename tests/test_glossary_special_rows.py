"""Mobile batch m5, request 3: one special-file decision for glossary extraction and the Glossary list.

* ``epub_special_files.glossary_special_file_settings`` / ``glossary_special_file_flags`` are the glossary
  extractor's own rule, moved out of ``extract_glossary_from_epub`` unchanged: pinned here against a frozen
  copy of the previous code for every setting combination, and the extractor's wrappers return the same.
* ``glossary_special_file_env`` / ``..._settings_for_owner`` build the same settings from an owner's start-up
  values (the config, like ``run_env.initialize_environment_variables``) instead of the process environment.
* ``glossary_progress_core`` without ``special_files`` (the desktop Glossary Progress) keeps its rows exactly;
  with them (the mobile list) the chapter rows are the extractor's documents in order and the skipped special
  files are listed at their spine position as "Special file (Skipped)", before and after an extraction ran.

Run from the repo root with ``src`` on PYTHONPATH::

    python -m pytest tests/test_glossary_special_rows.py -p no:cacheprovider -W ignore -o console_output_style=classic -q
"""

import json
import os
import re
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ebooklib import epub  # noqa: E402

import epub_special_files as esf  # noqa: E402
import extract_glossary_from_epub as extractor  # noqa: E402
import glossary_progress_core as gpc  # noqa: E402
import progress_core as pc  # noqa: E402

ENV_KEYS = ("TRANSLATE_SPECIAL_FILES", "SPECIAL_FILE_KEYWORDS", "SPECIAL_FILE_EXACT",
            "GLOSSARY_NEVER_CONSIDER_IN_BETWEEN_FILES_AS_SPECIAL")
FRONT = ["cover.html", "info.xhtml", "toc.xhtml", "title.xhtml", "chapter0001.xhtml", "chapter0002.xhtml",
         "chapter0003.xhtml", "chapter0004.xhtml", "chapter0005.xhtml", "copyright.xhtml"]
MIDDLE = ["cover.html", "info.xhtml", "chapter0001.xhtml", "chapter0002.xhtml", "notice.xhtml",
          "author_message.xhtml", "chapter0003.xhtml", "chapter0004.xhtml", "chapter0005.xhtml", "copyright.xhtml"]


# ---------------------------------------------------------------------------
# The previous extractor code, frozen (extract_glossary_from_epub before m5)
# ---------------------------------------------------------------------------


def _legacy_settings(*, include_special_files=False):
    keywords = os.getenv('SPECIAL_FILE_KEYWORDS', '')
    exact = os.getenv('SPECIAL_FILE_EXACT', '')
    translate_special = include_special_files or os.getenv('TRANSLATE_SPECIAL_FILES', '0') == '1'
    return {
        'parser': 'html.parser',
        'translate_special_files': translate_special,
        'protect_interior_special_files': not translate_special and os.getenv(
            'GLOSSARY_NEVER_CONSIDER_IN_BETWEEN_FILES_AS_SPECIAL', '1',
        ) == '1',
        'special_file_keywords': [] if translate_special else sorted(set(
            [k.strip().lower() for k in keywords.split(',') if k.strip()]
            if keywords else [
                'title', 'toc', 'copyright', 'preface', 'nav',
                'message', 'notice', 'colophon', 'dedication', 'epigraph',
                'foreword', 'acknowledgment', 'author', 'appendix', 'bibliography',
            ]
        )),
        'special_file_exact': [] if translate_special else sorted(set(
            [k.strip().lower() for k in exact.split(',') if k.strip()]
            if exact else ['index', 'glossary', 'glossary_extension', 'glossary_unified']
        )),
    }


def _legacy_flags(filenames, signature):
    def is_special(filename):
        if signature['translate_special_files']:
            return False
        name = os.path.splitext(os.path.basename(filename or ''))[0].lower()
        if not name:
            return False
        if name in signature['special_file_exact']:
            return True
        keywords = signature['special_file_keywords']
        if not any(keyword in name for keyword in keywords):
            return False
        stripped = re.sub(r'\d+$', '', name).rstrip('_- ')
        return not re.search(r'\d', name) or any(keyword in stripped for keyword in keywords)

    return esf.special_file_flags(filenames, is_special,
                                  protect_interior=signature['protect_interior_special_files'])


ENV_CASES = [
    {},
    {"TRANSLATE_SPECIAL_FILES": "1"},
    {"TRANSLATE_SPECIAL_FILES": "0", "GLOSSARY_NEVER_CONSIDER_IN_BETWEEN_FILES_AS_SPECIAL": "0"},
    {"SPECIAL_FILE_KEYWORDS": "Info, copyright , NOTICE", "SPECIAL_FILE_EXACT": "toc"},
    {"SPECIAL_FILE_KEYWORDS": "info, copyright", "GLOSSARY_NEVER_CONSIDER_IN_BETWEEN_FILES_AS_SPECIAL": "0"},
    {"SPECIAL_FILE_KEYWORDS": "", "SPECIAL_FILE_EXACT": ""},
]
NAME_LISTS = [FRONT, MIDDLE, ["title0900.xhtml", "notice0901.xhtml", "chapter0001.xhtml", "index.xhtml",
                             "chapter0002.xhtml", "notice0001.xhtml", "title0002.xhtml"]]


@pytest.fixture
def clean_env(monkeypatch):
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(extractor, "is_stop_requested", lambda: False)
    return monkeypatch


@pytest.mark.parametrize("env", ENV_CASES, ids=range(len(ENV_CASES)))
@pytest.mark.parametrize("include", [False, True])
def test_shared_rule_equals_the_extractors_previous_code(clean_env, env, include):
    for key, value in env.items():
        clean_env.setenv(key, value)
    legacy = _legacy_settings(include_special_files=include)
    assert extractor._glossary_epub_extraction_settings(include_special_files=include) == legacy
    assert esf.glossary_special_file_settings(include_special_files=include) == legacy
    assert esf.glossary_special_file_settings(dict(env), include_special_files=include) == legacy  # a mapping
    for names in NAME_LISTS:
        assert extractor._glossary_epub_special_file_flags(names, legacy) == _legacy_flags(names, legacy)
        assert esf.glossary_special_file_flags(names, legacy) == _legacy_flags(names, legacy)


CONFIG_CASES = {
    "defaults": {},
    "translate_special": {"translate_special_files": True},
    "protect_off": {"never_consider_in_between_files_as_special": False},
    "custom": {"special_file_keywords": "info, copyright, notice", "special_file_exact": "toc"},
    "custom_protect_off": {"special_file_keywords": "info, copyright, notice", "special_file_exact": "toc",
                           "never_consider_in_between_files_as_special": False},
}


def test_owner_env_is_the_runs_special_file_rows():
    """The start-up values (seeded from the config like the desktop's) as the run exports them."""
    owner = pc.ProgressOwner({})
    env = esf.glossary_special_file_env(owner)
    assert env["TRANSLATE_SPECIAL_FILES"] == "0" and env["GLOSSARY_NEVER_CONSIDER_IN_BETWEEN_FILES_AS_SPECIAL"] == "1"
    assert "copyright" in env["SPECIAL_FILE_KEYWORDS"] and "glossary_unified" in env["SPECIAL_FILE_EXACT"]
    owner = pc.ProgressOwner(CONFIG_CASES["custom_protect_off"] | {"translate_special_files": True})
    env = esf.glossary_special_file_env(owner)
    assert env == {"TRANSLATE_SPECIAL_FILES": "1", "SPECIAL_FILE_KEYWORDS": "info, copyright, notice",
                   "SPECIAL_FILE_EXACT": "toc", "GLOSSARY_NEVER_CONSIDER_IN_BETWEEN_FILES_AS_SPECIAL": "0"}
    assert esf.glossary_special_file_settings_for_owner(owner) == esf.glossary_special_file_settings(env)


# ---------------------------------------------------------------------------
# The Glossary Progress rows
# ---------------------------------------------------------------------------


def _write_epub(path, names):
    book = epub.EpubBook()
    book.set_identifier(path.stem)
    book.set_title("Special file rows")
    book.set_language("ko")
    image = epub.EpubItem(uid="cover-img", file_name="Images/cover.png", media_type="image/png",
                          content=b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    book.add_item(image)
    docs = []
    for index, name in enumerate(names):
        doc = epub.EpubHtml(uid=f"doc{index}", file_name=f"Text/{name}", lang="ko")
        doc.content = ('<html><body><img src="../Images/cover.png" alt=""/></body></html>' if name.startswith("cover")
                       else f"<html><body><p>Story prose of {name} with enough words for a glossary.</p></body></html>")
        book.add_item(doc)
        docs.append(doc)
    book.spine = docs
    book.add_item(epub.EpubNcx())
    epub.write_epub(str(path), book)
    return path


def _workspace(tmp_path, names, progress):
    source = _write_epub(tmp_path / "Book.epub", names)
    gp = tmp_path / "out" / "Glossary" / "Book" / "Book_glossary_progress.json"
    gp.parent.mkdir(parents=True)
    gp.write_text(json.dumps(progress), encoding="utf-8")
    return source, gp


def _extracted(names_env, source, monkeypatch):
    for key, value in names_env.items():
        monkeypatch.setenv(key, value)
    try:
        documents = extractor.extract_chapters_from_epub(str(source), return_document_metadata=True)
    finally:  # the list must not lean on this process's environment
        for key in ENV_KEYS:
            monkeypatch.delenv(key, raising=False)
    return [d["filename"] for d in documents]


def test_desktop_rows_are_unchanged_without_special_files(clean_env, tmp_path):
    """The desktop panel's rows: the env rule without protect-interior, no special rows (as before m5)."""
    source, gp = _workspace(tmp_path, FRONT, {"completed": [0]})
    owner = pc.ProgressOwner({"output_directory": str(tmp_path / "out")})
    model = gpc.open_glossary_progress(owner, str(source), str(gp))
    rows = gpc.glossary_rows(model)
    assert [r.filename for r in rows if r.kind == "chapter"] == [
        n for n in FRONT if n not in ("toc.xhtml", "title.xhtml", "copyright.xhtml")]
    assert not any(r.kind == "special" for r in rows) and "special_rows" not in model.panel_state
    assert model.panel_state["total"] == 7


@pytest.mark.parametrize("names", [FRONT, MIDDLE], ids=["front_back", "middle"])
@pytest.mark.parametrize("case", sorted(CONFIG_CASES))
@pytest.mark.parametrize("progress", [{"in_progress": [0]}, {"completed": [0, 1, 2], "skipped": [3]}],
                         ids=["running", "finished"])
def test_mobile_rows_follow_the_extractors_own_decision(clean_env, tmp_path, names, case, progress):
    owner = pc.ProgressOwner(dict(CONFIG_CASES[case], output_directory=str(tmp_path / "out")))
    source = _write_epub(tmp_path / "Book.epub", names)
    # what the run sends for these settings (the run's env), and the progress file it writes (zero-based)
    sent = _extracted(esf.glossary_special_file_env(owner), source, clean_env)
    progress = dict(progress, indexing="chapter_index_zero_based",
                    chapter_filenames={str(i): name for i, name in enumerate(sent)})
    _source, gp = _workspace(tmp_path, names, progress)
    settings = esf.glossary_special_file_settings_for_owner(owner)
    model = gpc.open_glossary_progress(owner, str(source), str(gp), special_files=settings)
    rows = gpc.glossary_rows(model)
    chapters = [r for r in rows if r.kind == "chapter"]
    special = [r for r in rows if r.kind == "special"]
    # row i is the extractor's chapter i
    assert [r.filename for r in chapters] == sent
    assert [r.chapter_index for r in chapters] == list(range(len(sent)))
    # every other spine file is listed as a skipped special file, in spine order
    assert sorted([r.filename for r in rows if r.kind in ("chapter", "special")], key=names.index) == \
        [r.filename for r in rows if r.kind in ("chapter", "special")]
    assert {r.filename for r in special} == set(names) - set(sent)
    for row in special:
        assert row.status == "skipped_special" and row.chapter_index is None
        assert gpc.GLOSSARY_SPECIAL_SKIPPED_LABEL in row.display and row.display.startswith(
            f"[{names.index(row.filename) + 1:03d}] ")
        assert row.status in gpc.GLOSSARY_STATUS_GROUPS["skipped"]
    # the special files are not counted (the total is the chapters plus the Minimal pass / refinement rows)
    others = [r for r in rows if r.kind not in ("chapter", "special")]
    assert gpc.glossary_stats(model)["total"] == len(chapters) + len(others)
    if "completed" in progress:
        assert [r.status for r in chapters[:3]] == ["completed"] * 3


def test_removing_a_special_row_touches_no_progress(clean_env, tmp_path):
    source, gp = _workspace(tmp_path, FRONT, {"completed": [0], "minimal_pass": {"status": "completed"}})
    owner = pc.ProgressOwner({"output_directory": str(tmp_path / "out")})
    model = gpc.open_glossary_progress(owner, str(source), str(gp),
                                       special_files=esf.glossary_special_file_settings_for_owner(owner))
    special = [r for r in gpc.glossary_rows(model) if r.kind == "special"]
    assert [r.filename for r in special] == ["copyright.xhtml"]  # protect-interior: only the trailing one
    before = gp.read_bytes()
    assert gpc.remove_glossary_progress(model, special)["changed"] is False
    assert gpc.mark_glossary_completed(model, special)["changed"] is False
    assert gp.read_bytes() == before


@pytest.mark.parametrize("names", [FRONT, MIDDLE], ids=["front_back", "middle"])
@pytest.mark.parametrize("case", sorted(CONFIG_CASES))
def test_special_rows_before_any_progress_file_match_the_rows_after(clean_env, tmp_path, names, case, capsys):
    """Before the first extraction (no progress file): the same special rows the list shows once the file
    exists, read from the EPUB alone; nothing is written and no "could not load" warning is printed."""
    owner = pc.ProgressOwner(dict(CONFIG_CASES[case], output_directory=str(tmp_path / "out")))
    settings = esf.glossary_special_file_settings_for_owner(owner)
    source, gp = _workspace(tmp_path, names, {"in_progress": [0]})
    after = [r for r in gpc.glossary_rows(gpc.open_glossary_progress(owner, str(source), str(gp),
                                                                     special_files=settings))
             if r.kind == "special"]
    gp.unlink()
    capsys.readouterr()
    before = gpc.glossary_special_preview_rows(owner, str(source), settings)
    assert [(r.key, r.display, r.status, r.filename) for r in before] == [
        (r.key, r.display, r.status, r.filename) for r in after]
    assert "Could not load" not in capsys.readouterr().out
    assert not gp.exists() and not list((tmp_path / "out").rglob("*.json"))
    assert gpc.glossary_special_preview_rows(owner, str(tmp_path / "Book.txt"), settings) == []
    assert gpc.glossary_special_preview_rows(owner, str(source), None) == []


def test_a_settings_change_re_maps_an_open_model(clean_env, tmp_path):
    """The model remembers the settings it was built with; new ones re-map the rows like a fresh model."""
    owner = pc.ProgressOwner({"output_directory": str(tmp_path / "out")})
    source, gp = _workspace(tmp_path, MIDDLE, {"in_progress": [0]})
    old = esf.glossary_special_file_settings_for_owner(owner)
    model = gpc.open_glossary_progress(owner, str(source), str(gp), special_files=old)
    assert not gpc.glossary_special_files_changed(model, old)
    new_owner = pc.ProgressOwner(dict(CONFIG_CASES["custom_protect_off"], output_directory=str(tmp_path / "out")))
    new = esf.glossary_special_file_settings_for_owner(new_owner)
    assert gpc.glossary_special_files_changed(model, new)
    gpc.apply_glossary_special_files(model, new)
    fresh = gpc.open_glossary_progress(new_owner, str(source), str(gp), special_files=new)

    def kinds(m):
        return [(r.kind, r.filename, r.chapter_index) for r in gpc.glossary_rows(m) if r.kind in ("chapter", "special")]

    assert kinds(model) == kinds(fresh) and not gpc.glossary_special_files_changed(model, new)
    # a desktop model (no special-file settings) is never re-mapped
    desktop = gpc.open_glossary_progress(owner, str(source), str(gp))
    assert not gpc.glossary_special_files_changed(desktop, new)
