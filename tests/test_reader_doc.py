"""U5 parity for the reader document builder and the live stream (reader_doc / live_stream).

What moved (milestone U5, plan section 2): the reader threads' ``run`` bodies (EPUB load,
cache load, overlay merge, image preload, workspace load, search), ``EpubReaderDialog``'s
page builder (themes, image materialisation, embedded CSS, the paged / scroll document)
and the live "Translate this chapter" stream routing, all byte-for-byte into GUI-free
mixins the Qt classes now inherit. Desktop parity is checked against ``epub_library`` at
``U5_BASE_SHA`` (``git show``, imported as a separate module so legacy and new code run
side by side; the oracle is shared with tests/test_library_core.py):

* verbatim: every moved method is identical to its source except the documented Qt
  replacements (tests/parity/DISCREPANCIES.md, U5); every other reader method is
  unchanged and the Phase-1 extractions equal the code they replaced;
* ``_url_scheme`` agrees with ``QUrl(src).scheme().lower()`` on a 40k-string corpus;
* differential fuzz (``PARITY_U5_READER_STATES``, default 40): ``_process_html``,
  ``_get_embedded_css`` and ``_wrap_html`` run through the legacy methods, the new desktop
  class and the plain ``ReaderDocument`` / ``wrap_reader_html`` on random EPUBs, CSS
  sources, images and options; the reader threads (load, cache, search, overlay merge,
  image preload, workspace load) against the plain API; the live stream (classify,
  drain, wrap, output folder, finish) against the legacy dialog methods;
* the mobile shell only adds the touch layer (the desktop page is recovered exactly) and
  its paging bridge runs in QtWebEngine (offscreen), posting ``GLRDR:`` console events
  and the ``/__ev`` fetch fallback;
* offscreen smoke: legacy vs new ``EpubReaderDialog`` render identical pages in every
  layout.

Run (repository root)::

    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic \\
        tests/test_reader_doc.py
"""

from __future__ import annotations

import ast
import collections
import copy
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import types
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
TESTS = REPO_ROOT / "tests"
for _p in (str(TESTS), str(SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# The legacy oracle and the EPUB / sandbox fixtures are shared with the Library tests
# (one ``legacy_epub_library_u5`` module per session).
import test_library_core as tlc  # noqa: E402
from test_library_core import (  # noqa: E402
    Sandbox,
    git_text,
    load_legacy_epub_library,
    norm,
    png_bytes,
    qapp,
    write_epub,
)

import library_core  # noqa: E402
import live_stream  # noqa: E402
import reader_doc  # noqa: E402

STATES = max(1, int(os.environ.get("PARITY_U5_READER_STATES", "40")))
LOADER_STATES = max(1, int(os.environ.get("PARITY_U5_LOADER_STATES", "12")))
SEED = int(os.environ.get("PARITY_U5_SEED", "5005"))

BRAIN = chr(0x1F9E0)
SATELLITE = chr(0x1F6F0)
CHECK = chr(0x2705)
ZWSP = chr(0x200B)

#: (legacy Qt class, member) -> (module, mixin) for every reader / live member that moved.
READER_DOC_METHODS = (
    "_get_theme", "_reader_chapter_display_number", "_ensure_reader_image_temp_dir",
    "_close_epub_image_zip", "_load_reader_image_resource", "_invalidate_processed_reader_cache",
    "_set_reader_images", "_processed_reader_html_key", "_chapter_image_preload_key",
    "_process_html", "_get_embedded_css", "_resolve_attach_css_to_chapters", "_wrap_html",
)
LIVE_METHODS = ("_LIVE_STATUS_CHARS", "_classify_live_line", "_wrap_live_html",
                "_resolve_live_output_folder")
THREAD_MIXINS = {
    "_EpubCacheLoaderThread": "EpubCacheLoaderMixin",
    "_OverlayMergeThread": "OverlayMergeMixin",
    "_ReaderImagePreloadThread": "ReaderImagePreloadMixin",
    "_WorkspaceReaderLoaderThread": "WorkspaceReaderLoaderMixin",
    "_EpubSearchThread": "EpubSearchMixin",
    "_EpubLoaderThread": "EpubLoaderMixin",
}
READER_MOVED = (
    [(cls, "run", "reader_doc", mixin) for cls, mixin in THREAD_MIXINS.items()]
    + [("EpubReaderDialog", name, "reader_doc", "ReaderDocMixin") for name in READER_DOC_METHODS]
    + [("EpubReaderDialog", name, "live_stream", "LiveStreamMixin") for name in LIVE_METHODS]
)

_QURL_SCHEME = ("QUrl(src).scheme().lower()", "_url_scheme(src)")
#: Documented Qt replacements inside moved methods: (mixin, member) -> [(old, new)].
METHOD_EDITS = {
    ("ReaderImagePreloadMixin", "run"): [_QURL_SCHEME],
    ("ReaderDocMixin", "_process_html"): [
        ("QUrl.fromLocalFile(warmed_path).toString()", "self._reader_file_url(warmed_path)"),
        ("QUrl.fromLocalFile(img_path).toString()", "self._reader_file_url(img_path)"),
        _QURL_SCHEME,
    ],
}

#: EpubReaderDialog methods that changed in place (Phase 1: they now call shared code).
READER_DIALOG_CHANGED = {"_finalize_post_load", "_render_current", "_open_google_translate",
                         "_open_web_define", "_drain_live_queue", "_finish_live_translation"}


# =============================================================================================
# fixtures
# =============================================================================================

@pytest.fixture(scope="session")
def legacy(tmp_path_factory):
    return load_legacy_epub_library(tmp_path_factory.mktemp("legacy_u5_reader"))


@pytest.fixture(scope="session")
def el():
    qapp()
    import epub_library
    return epub_library


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch, request):
    """Reader caches, image temp dirs, Library and output roots all live under tmp_path."""
    legacy_module = request.getfixturevalue("legacy") if "legacy" in request.fixturenames else None
    temp_root = tmp_path / "_tmp"
    temp_root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp_root))
    library = tmp_path / "_isolated" / "Library"
    output = tmp_path / "_isolated" / "Output"
    library.mkdir(parents=True)
    output.mkdir(parents=True)
    monkeypatch.setenv("GLOSSARION_LIBRARY_DIR", str(library))
    monkeypatch.setenv("OUTPUT_DIRECTORY", str(output))
    for var in ("EPUB_CSS_OVERRIDE_PATH", "ATTACH_CSS_TO_CHAPTERS", "PDF_PARAGRAPH_ALIGNMENT",
                "PDF_RTL_PARAGRAPH_LAYOUT", "TRANSLATE_SPECIAL_FILES", "SPECIAL_FILE_KEYWORDS",
                "SPECIAL_FILE_EXACT", "TRANSLATE_ALL_NUMBERED_HTML", "EXTRACTION_WORKERS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(library_core, "_default_output_root", lambda: str(output))
    monkeypatch.setattr(reader_doc, "_EPUB_CACHE_DIR_OVERRIDE", None)
    if legacy_module is not None:
        monkeypatch.setattr(legacy_module, "_default_output_root", lambda: str(output))
    yield


@pytest.fixture(scope="module")
def legacy_tree():
    return ast.parse(git_text("src/epub_library.py"))


def _module_tree(name):
    return ast.parse((SRC / f"{name}.py").read_text(encoding="utf-8-sig"))


def _class(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)


def _member(tree, cls, member):
    for node in _class(tree, cls).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == member:
            return node
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == member for t in node.targets):
            return node
    raise KeyError((cls, member))


def _function(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


# =============================================================================================
# 1. verbatim moves and Phase-1 extractions
# =============================================================================================

@pytest.mark.parametrize("cls,member,module_name,mixin", READER_MOVED,
                         ids=[f"{c}.{m}" for c, m, _mo, _mi in READER_MOVED])
def test_reader_members_are_verbatim_in_their_mixins(cls, member, module_name, mixin, legacy_tree):
    old_text = ast.unparse(_member(legacy_tree, cls, member))
    for before, after in METHOD_EDITS.get((mixin, member), []):
        assert before in old_text, (mixin, member, before)
        old_text = old_text.replace(before, after)
    new_text = ast.unparse(_member(_module_tree(module_name), mixin, member))
    assert new_text == old_text
    assert "QUrl" not in new_text and "QImage" not in new_text


def _class_members(cnode):
    out = {}
    for node in cnode.body:
        if isinstance(node, ast.FunctionDef):
            out[node.name] = ast.unparse(node)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out[target.id] = ast.unparse(node)
    return out


def test_qt_reader_classes_changed_only_the_documented_methods(legacy_tree, el):
    current = ast.parse((SRC / "epub_library.py").read_text(encoding="utf-8-sig"))
    moved_by_class = collections.defaultdict(set)
    for cls, member, _module, _mixin in READER_MOVED:
        moved_by_class[cls].add(member)
    for cls in list(THREAD_MIXINS) + ["EpubReaderDialog"]:
        old = _class_members(_class(legacy_tree, cls))
        new = _class_members(_class(current, cls))
        removed = set(old) - set(new)
        added = set(new) - set(old)
        changed = {name for name, text in new.items() if name in old and old[name] != text}
        assert removed == moved_by_class[cls], cls
        if cls == "EpubReaderDialog":
            assert added == {"_reader_file_url"}
            assert changed == READER_DIALOG_CHANGED
        else:
            assert not added and not changed, cls
        # the Qt class inherits the moved members from its mixin(s)
        qt_class = getattr(el, cls)
        for member in moved_by_class[cls]:
            assert member not in vars(qt_class)
            assert hasattr(qt_class, member)


def test_desktop_reader_file_url_is_the_qt_original(el):
    from PySide6.QtCore import QUrl

    node = _member(ast.parse((SRC / "epub_library.py").read_text(encoding="utf-8-sig")),
                   "EpubReaderDialog", "_reader_file_url")
    assert "QUrl.fromLocalFile(path).toString()" in ast.unparse(node)
    for path in ("C:/x/y.png", r"C:\Temp\Glossarion_EpubImages\ab12\cover image.png",
                 "/tmp/a b/델타 #1.png", r"\\server\share\x%20.png"):
        assert el.EpubReaderDialog._reader_file_url(None, path) == QUrl.fromLocalFile(path).toString()


def test_all_chapters_html_is_the_scroll_all_loop(legacy_tree):
    old = _member(legacy_tree, "EpubReaderDialog", "_render_current")
    new = _member(ast.parse((SRC / "epub_library.py").read_text(encoding="utf-8-sig")),
                  "EpubReaderDialog", "_render_current")
    extracted = _member(_module_tree("reader_doc"), "ReaderDocMixin", "_all_chapters_html")
    # the extracted body (docstring .. return) is the two statements it replaced
    expected_body = [ast.dump(s) for s in extracted.body[1:-1]]
    assert ast.unparse(extracted.body[-1]) == "return all_html"
    found = []
    replacement = ast.parse("all_html = self._all_chapters_html()").body[0]

    class Swap(ast.NodeTransformer):
        def generic_visit(self, node):
            super().generic_visit(node)
            for field in ("body", "orelse"):
                stmts = getattr(node, field, None)
                if not isinstance(stmts, list):
                    continue
                for index in range(len(stmts) - 1):
                    if [ast.dump(s) for s in stmts[index:index + 2]] == expected_body:
                        found.append(index)
                        stmts[index:index + 2] = [replacement]
                        break
            return node

    swapped = Swap().visit(copy.deepcopy(old))
    assert len(found) == 1
    assert ast.dump(swapped) == ast.dump(new)


def test_drain_live_lines_is_the_drain_loop(legacy_tree):
    old = _member(legacy_tree, "EpubReaderDialog", "_drain_live_queue")
    new = _member(ast.parse((SRC / "epub_library.py").read_text(encoding="utf-8-sig")),
                  "EpubReaderDialog", "_drain_live_queue")
    extracted = _member(_module_tree("live_stream"), "LiveStreamMixin", "_drain_live_lines")
    assert [ast.dump(s) for s in extracted.body[1:4]] == [ast.dump(s) for s in old.body[1:4]]
    # AST comparison: ast.unparse spells tuples differently across Python versions (3.10 CI)
    assert ast.dump(extracted.body[4]) == ast.dump(ast.parse("return (drained, content_added)").body[0])
    assert ast.dump(new.body[1]) == ast.dump(ast.parse("drained, content_added = self._drain_live_lines()").body[0])
    assert [ast.dump(s) for s in new.body[2:]] == [ast.dump(s) for s in old.body[4:]]


def test_chapter_display_numbers_is_the_post_load_expression(legacy_tree, legacy):
    old = _member(legacy_tree, "EpubReaderDialog", "_finalize_post_load")
    assign = next(n for n in ast.walk(old) if isinstance(n, ast.Assign)
                  and ast.unparse(n.targets[0]) == "self._chapter_display_numbers")
    helper = _function(_module_tree("reader_doc"), "_chapter_display_numbers")
    old_expr = ast.unparse(assign.value).replace(
        "getattr(self, '_config', None)", "config").replace("self._chapter_filenames", "filenames")
    assert ast.unparse(helper.body[-1].value) == old_expr
    # and the public wrapper equals the legacy expression evaluated on the dialog state
    rng = random.Random(SEED * 3)
    names = ["chapter0001.xhtml", "Chapter0002.XHTML", "Text/ch_3.html", "prologue.xhtml", "cover.xhtml",
             "nav.xhtml", "title.xhtml", "0004_section.xhtml", "notes.html", "chapter0010.xhtml",
             "Epilogue.xhtml", "toc.xhtml", "afterword_2.xhtml", "", "gallery.xhtml", "ch5.htm"]
    configs = [None, {}, {"special_file_keywords": "cover,nav,toc,note"},
               {"special_file_exact": "title,prologue"}, {"translate_all_numbered_html": False},
               {"special_file_keywords": "", "special_file_exact": ""}]
    code = compile(ast.Expression(assign.value), "<legacy _finalize_post_load>", "eval")
    for _ in range(300):
        filenames = [rng.choice(names) for _ in range(rng.randint(0, 12))]
        config = rng.choice(configs)
        lowered = [os.path.basename(f or "").lower() for f in filenames]
        state = types.SimpleNamespace(_chapter_filenames=lowered, _config=config)
        namespace = dict(vars(legacy))
        namespace["self"] = state  # a global: the generator expression body reads it too
        expected = eval(code, namespace)  # noqa: S307 - the legacy source expression
        assert reader_doc.chapter_display_numbers(filenames, config) == list(expected)
        assert reader_doc._chapter_display_numbers(lowered, config) == list(expected)


def test_url_scheme_agrees_with_qurl():
    from PySide6.QtCore import QUrl

    backslash = chr(92)
    corpus = ["http://x/a.png", "HTTPS://X/b.jpg", "https:foo", "http:/a", " http://x/a.png", "images/a.png",
              "../Images/a.png", "C:/x/a.png", "C:" + backslash + "x.png", "file:///C:/a.png", "data:image/png;base64,AA",
              "//cdn/x.png", "http://[::1", "http://[::1]/a.png", "http://exa mple.com/a.png", "ht tp://x",
              "1http://x", "a+b-c.d://x", "h%74tp://x", "", "  ", "\thttps://x\n", "https://例え.jp/a.png",
              "javascript:alert(1)", "#frag", "?q=1", "a?b:c", "a#b:c", ":foo", "Http://x", "h:", "http",
              "ħttp://x", "https://user:pw@host:443/a?x#y", backslash * 2 + "server" + backslash + "a.png"]
    rng = random.Random(SEED)
    alphabet = "htps:/" + backslash + ". %[]:#?@xX1-+_ \tħ"
    for _ in range(20000):
        corpus.append("".join(rng.choice(alphabet) for _ in range(rng.randint(0, 14))))
        corpus.append(rng.choice(["http", "https", "HTTP", "Https", "file", "data", "c", " http", "\thttps", "x+y"])
                      + rng.choice([":", ":/", "://", ": //", " ://", "://[", "://[::1"])
                      + "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 8))))
    for text in corpus:
        assert reader_doc._url_scheme(text) == QUrl(text).scheme().lower(), repr(text)


def test_google_and_define_urls_match_the_legacy_menu_actions(legacy, el, monkeypatch):
    from PySide6.QtGui import QDesktopServices

    opened = []
    monkeypatch.setattr(QDesktopServices, "openUrl", staticmethod(lambda url: opened.append(url.toString())))
    rng = random.Random(SEED * 5)
    texts = ["", "   ", "hello", "  two words  ", "한국어 문장", "a&b=c?d#e/f", "line\nbreak", "100% sure",
             "日本語テキスト", "quote \" and ' marks", "+plus+", "émigré"]
    languages = ["English", "Korean", "Japanese", "Chinese (Simplified)", "Spanish", "Klingon", ""]
    for _ in range(200):
        text = rng.choice(texts) + rng.choice(["", " ", "x"])
        language = rng.choice(languages)
        code = legacy._target_lang_to_google_code(language)
        assert reader_doc.target_lang_to_google_code(language) == code
        results = []
        for module in (legacy, el):
            del opened[:]
            module.EpubReaderDialog._open_google_translate(None, text, code)
            module.EpubReaderDialog._open_web_define(None, text)
            results.append(list(opened))
        assert results[0] == results[1]
        from PySide6.QtCore import QUrl
        public = [reader_doc.google_translate_url(text, language), reader_doc.define_url(text)]
        assert [QUrl(u).toString() for u in public if u] == results[0]


# =============================================================================================
# 2. the reader threads vs the plain API
# =============================================================================================

def _add_css_and_fonts(epub_path: Path, rng: random.Random):
    """Zip CSS / font members plus on-disk css/ fonts/ images/ folders next to the EPUB."""
    with zipfile.ZipFile(epub_path, "a") as archive:
        if rng.random() < 0.8:
            archive.writestr("OEBPS/Styles/style.css",
                             "@font-face { font-family: Book; src: url('../Fonts/book.ttf'); }\n"
                             "body { font-family: Book; } p { text-indent: 1em; }\n"
                             "h1 { src: url(\"missing.woff\"); }")
        if rng.random() < 0.7:
            archive.writestr("OEBPS/Fonts/book.ttf", b"TTF-FONT-DATA" * rng.randint(1, 4))
        if rng.random() < 0.3:
            archive.writestr("OEBPS/Styles/extra.CSS", "em { color: red; }")
    folder = epub_path.parent
    if rng.random() < 0.4:
        (folder / "css").mkdir(exist_ok=True)
        (folder / "css" / "disk.css").write_text(
            "@font-face { font-family: Disk; src: url(fonts/disk.woff2); }", encoding="utf-8")
    if rng.random() < 0.4:
        (folder / "fonts").mkdir(exist_ok=True)
        (folder / "fonts" / "disk.woff2").write_bytes(b"WOFF2DATA")
    if rng.random() < 0.5:
        (folder / "images").mkdir(exist_ok=True)
        (folder / "images" / "disk.png").write_bytes(png_bytes(240, 230))


def _image_set(rng):
    big_bytes = png_bytes(8, 8) + bytes(rng.randrange(256) for _ in range(6000))
    svg = (b'<svg xmlns="http://www.w3.org/2000/svg" width="400" height="300">'
           b'<rect width="400" height="300"/></svg>')
    return {"small.png": png_bytes(4, 4), "wide.png": png_bytes(300, 260), "big.png": big_bytes,
            "tall.png": png_bytes(100, 500), "pic.svg": svg}


_PIECES = (
    "<h1>Title {n}</h1>",
    "<p>Text {n} with <em>em</em> and &lt;b&gt;escaped bold&lt;/b&gt; words.</p>",
    "<p><img src='../Images/{img}' alt='a'/></p>",
    "<p><img src='../Images/{img}'/><br/><img src='../Images/{img2}'/></p>",
    "<div><img src='../Images/{img}'/><img src='../Images/{img2}'/><h2>Mixed</h2><p>text</p></div>",
    "<figure><img src='../Images/{img}'/><figcaption>cap</figcaption></figure>",
    "<h2>Header</h2><p>Lead in</p><p><img src='../Images/{img}'/></p>",
    "<h3>Only header</h3><div><img src='{img}'/></div>",
    "<p><img src='https://example.com/x.png'/></p>",
    "<p><img src='data:image/png;base64,iVBORw0KGgo='/></p>",
    "<p><img src='../Images/missing.png'/></p>",
    "<p><img src='images/disk.png'/></p>",
    "<p><img src='extra.png' decoding='sync'/></p>",
    "<svg xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'>"
    "<image xlink:href='../Images/{img}' width='10' height='10'/></svg>",
    "<svg><image href='../Images/{img}'/></svg>",
    "<a id='anchor{n}'></a><p><img src='../Images/{img}'/></p>",
    "<p>" + "long text " * 30 + "</p>",
    "<p>Para with image <img src='../Images/{img}'/> inline and " + "words " * 50 + "</p>",
    "<blockquote><p>Quote {n}</p></blockquote>",
)


def random_chapter_html(rng, n, images):
    names = list(images)
    parts = []
    for _ in range(rng.randint(1, 7)):
        parts.append(rng.choice(_PIECES).format(n=n, img=rng.choice(names), img2=rng.choice(names)))
    return "".join(parts)


def build_reader_book(root: Path, rng: random.Random):
    images = _image_set(rng)
    chapters = []
    if rng.random() < 0.4:
        chapters.append(("cover.xhtml", "Cover", "<p><img src='../Images/wide.png'/></p>"))
    for number in range(1, rng.randint(2, 6)):
        chapters.append((f"chapter{number:04d}.xhtml", f"Chapter {number}",
                         random_chapter_html(rng, number, images)))
    if rng.random() < 0.3:
        chapters.append(("nav.xhtml", "Contents", "<ol><li>One</li></ol>"))
    epub = Path(write_epub(root / "books" / f"Book {rng.randint(0, 99)}.epub", chapters,
                           title="Reader Book", images=images))
    _add_css_and_fonts(epub, rng)
    extra = root / "extra_images"
    extra.mkdir(parents=True, exist_ok=True)
    (extra / "extra.png").write_bytes(png_bytes(250, 250))
    return epub, chapters, [str(extra)]


def _load_legacy(legacy, epub, show_special, config):
    thread = legacy._EpubLoaderThread(str(epub), None, show_special_files=show_special, config=config)
    errors, done = [], []
    thread.error.connect(lambda message: errors.append(message))
    thread.done.connect(lambda: done.append(True))
    thread.run()
    assert not errors, errors
    assert done
    return legacy._load_epub_cache(str(epub), show_special_files=show_special, config=config)


def test_reader_threads_match_the_plain_api(legacy, tmp_path, monkeypatch):
    qapp()
    rng = random.Random(SEED * 7)
    for state in range(LOADER_STATES):
        root = tmp_path / f"s{state}"
        epub, chapters, extra_dirs = build_reader_book(root, rng)
        show_special = rng.random() < 0.6
        config = rng.choice([{}, {"special_file_keywords": "cover,nav"}, {"translate_special_files": True},
                             {"reader_worker_count": 1}])
        # --- EPUB load (separate cache folders; the plain call misses its cache) ---
        monkeypatch.setattr(tempfile, "tempdir", str(root / "legacy_tmp"))
        os.makedirs(tempfile.tempdir, exist_ok=True)
        legacy_loaded = _load_legacy(legacy, epub, show_special, config)
        monkeypatch.setattr(reader_doc, "_EPUB_CACHE_DIR_OVERRIDE", str(root / "plain_cache"))
        plain_loaded = reader_doc.load_epub_chapters(str(epub), show_special, config, use_cache=False)
        assert legacy_loaded is not None and plain_loaded is not None
        assert norm(list(legacy_loaded[0])) == norm(list(plain_loaded[0])), state
        assert norm(dict(legacy_loaded[1])) == norm(dict(plain_loaded[1]))
        assert list(legacy_loaded[2] or []) == plain_loaded[2]
        # --- cache loader thread (hit) vs load_epub_cache ---
        hits = []
        cache_thread = legacy._EpubCacheLoaderThread(str(epub), show_special, config)
        cache_thread.hit.connect(lambda c, i, f: hits.append((c, i, f)))
        cache_thread.miss.connect(lambda: hits.append(None))
        cache_thread.run()
        cached = reader_doc.load_epub_cache(str(epub), show_special_files=show_special, config=config)
        assert hits and hits[0] is not None and cached
        assert norm(list(hits[0][0])) == norm(list(cached[0]))
        assert list(hits[0][2]) == list(cached[2])
        assert reader_doc.load_epub_chapters(str(epub), show_special, config) == (
            cached[0], cached[1], list(cached[2]))
        raw_chapters, images, filenames = plain_loaded
        # --- search ---
        for query in ("Chapter", "text", "escaped bold", "zzz-none", "", "(", "  Para  "):
            rows = []
            search = legacy._EpubSearchThread(1, query, raw_chapters, config)
            search.results_batch_ready.connect(lambda _i, _q, batch, _d: rows.extend(batch))
            search.run()
            assert norm(reader_doc.search_chapters(raw_chapters, query, config)) == norm(rows), query
        # --- overlay merge ---
        workspace = root / "ws"
        workspace.mkdir(exist_ok=True)
        overlay = {}
        previous = [(f"Prev {i}", f"<p>prev {i}</p>") for i in range(len(raw_chapters))]
        for index, name in enumerate(filenames):
            roll = rng.random()
            response = workspace / f"response_{index}.html"
            key = os.path.basename(name).lower()
            if roll < 0.45:
                response.write_text(f"<html><head><title>T{index}</title></head><body><h1>Translated {index}"
                                    f"</h1><p>body</p></body></html>", encoding="utf-8")
                overlay[key] = {"path": str(response), "status": "completed",
                                **({"title": f"Given {index}"} if rng.random() < 0.4 else {})}
            elif roll < 0.6:
                response.write_text("   ", encoding="utf-8")
                overlay[key] = {"path": str(response)}
            elif roll < 0.7:
                overlay[key] = {"path": str(workspace / "missing.html")}
        results = []
        merge = legacy._OverlayMergeThread(raw_chapters, images, filenames, overlay, extra_dirs, config,
                                           None, previous)
        merge.done.connect(lambda c, i, applied: results.append((c, i, applied)))
        merge.run()
        merged = reader_doc.merge_overlay(raw_chapters, images, filenames, overlay, extra_dirs, config, previous)
        assert results and merged is not None
        assert norm(list(merged.chapters)) == norm(list(results[0][0]))
        assert merged.overlay_applied == results[0][2]
        assert merged.retry_required == merge._retry_required
        assert merged.read_signature == merge._read_signature
        # --- image preload (same temp folder, emptied between runs) ---
        temp_dir = str(root / "preload")
        for index, (_title, html) in enumerate(raw_chapters):
            shutil.rmtree(temp_dir, ignore_errors=True)
            os.makedirs(temp_dir)
            done = []
            preload = legacy._ReaderImagePreloadThread(f"k{index}", html, images, extra_dirs, str(epub), temp_dir)
            preload.done.connect(lambda key, resources: done.append((key, resources)))
            preload.run()
            legacy_tree_files = sorted(os.listdir(temp_dir))
            shutil.rmtree(temp_dir)
            os.makedirs(temp_dir)
            plain = reader_doc.preload_chapter_images(html, images, extra_dirs, str(epub), temp_dir, f"k{index}")
            assert done and done[0][0] == f"k{index}"
            assert norm(plain) == norm(done[0][1]), index
            assert sorted(os.listdir(temp_dir)) == legacy_tree_files
        # --- workspace loader ---
        entries = []
        for index in range(rng.randint(0, 4)):
            translated = workspace / f"section_{index}.html"
            if rng.random() < 0.6:
                translated.write_text(f"<html><body><h1>Sec {index} EN</h1><p>x</p></body></html>",
                                      encoding="utf-8")
            entries.append({"title": rng.choice([f"Section {index}", ""]), "filename": f"section_{index}.pdf",
                            "translated_path": str(translated)})
        manifest = {"entries": entries, "source_format": "html", "workspace": str(workspace)}
        loaded = []
        workspace_thread = legacy._WorkspaceReaderLoaderThread(manifest)
        workspace_thread.done.connect(lambda r, t, f: loaded.append((r, t, f)))
        workspace_thread.run()
        plain_ws = reader_doc.load_workspace_chapters(manifest)
        assert loaded and norm(list(plain_ws)) == norm(list(loaded[0]))


# =============================================================================================
# 3. the page builder: legacy methods vs the new desktop class vs ReaderDocument
# =============================================================================================

_STUB_METHODS = READER_DOC_METHODS + ("_reader_file_url", "_all_chapters_html")


def reader_stub(cls, state: dict):
    namespace = {}
    for name in _STUB_METHODS:
        member = getattr(cls, name, None)
        if member is not None:
            namespace[name] = member
    stub = type(f"{cls.__name__}DocStub", (), namespace)()
    stub.__dict__.update(copy.deepcopy(state))
    return stub


def qt_file_url(path):
    from PySide6.QtCore import QUrl
    return QUrl.fromLocalFile(path).toString()


def random_reader_options(rng, epub: Path, root: Path):
    translated_dirs = []
    overlay = {}
    if rng.random() < 0.5:
        out = root / "out"
        (out / "css").mkdir(parents=True, exist_ok=True)
        (out / "css" / "style.css").write_text(
            "@font-face { font-family: Out; src: url('../fonts/out.otf'); } body { margin: 0; }",
            encoding="utf-8")
        if rng.random() < 0.6:
            (out / "fonts").mkdir(exist_ok=True)
            (out / "fonts" / "out.otf").write_bytes(b"OTFDATA")
        translated_dirs = [str(out / "css")] if rng.random() < 0.7 else [str(out)]
    if rng.random() < 0.5:
        overlay = {"chapter0001.xhtml": {"path": str(root / "ws" / "response_0.html")}}
    return {
        "theme": rng.choice([0, 1, 2, 3, 4, 5, 9, -1]),
        "font_family": rng.choice(["Embedded CSS", "Embedded CSS", "Georgia", "Consolas", "Noto Sans KR",
                                   "It's Font", "", "  Embedded CSS  "]),
        "font_size": rng.choice([9, 12, 14, 18.5, 30]),
        "line_spacing": rng.choice([1.0, 1.5, 1.8, 2.25]),
        "show_raw": rng.random() < 0.3,
        "translated_overlay": overlay,
        "raw_epub_alt_path": rng.choice(["", "", str(root / "alt.epub")]),
        "translated_css_dirs": translated_dirs,
        "workspace_mode": rng.random() < 0.25,
        "config": rng.choice([{}, {"attach_css_to_chapters": True}, {"attach_css_to_chapters": False}]),
        "env": {
            "ATTACH_CSS_TO_CHAPTERS": rng.choice([None, None, "1", "off", "maybe"]),
            "EPUB_CSS_OVERRIDE_PATH": rng.choice([None, None, "override", "missing"]),
            "PDF_PARAGRAPH_ALIGNMENT": rng.choice([None, "source", "justify"]),
            "PDF_RTL_PARAGRAPH_LAYOUT": rng.choice([None, "0", "1"]),
        },
    }


def _apply_env(monkeypatch, env, root):
    for key, value in env.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        elif key == "EPUB_CSS_OVERRIDE_PATH" and value == "override":
            path = root / "override.css"
            path.write_text("@font-face { src: url(book.ttf); } body { color: #123; }", encoding="utf-8")
            monkeypatch.setenv(key, str(path))
        elif key == "EPUB_CSS_OVERRIDE_PATH":
            monkeypatch.setenv(key, str(root / "no-such.css"))
        else:
            monkeypatch.setenv(key, value)


def _doc_state(epub, images, extra_dirs, opts, filenames, chapters):
    return {
        "_epub_path": str(epub), "_images": images,
        "_image_resource_signature": reader_doc._reader_image_map_signature(images),
        "_extra_image_dirs": list(extra_dirs), "_config": opts["config"], "_theme_index": opts["theme"],
        "_font_family": opts["font_family"], "_font_size": opts["font_size"],
        "_line_spacing": opts["line_spacing"], "_show_raw": opts["show_raw"],
        "_translated_overlay": opts["translated_overlay"], "_raw_epub_alt_path": opts["raw_epub_alt_path"],
        "_translated_css_dirs": list(opts["translated_css_dirs"]), "_workspace_mode": opts["workspace_mode"],
        "_processed_html_cache": {}, "_image_sizeable_cache": {}, "_preloaded_image_resources": {},
        "_image_cache_generation": 0, "_epub_image_zip": None, "_epub_image_zip_path": "",
        "_epub_image_zip_names": {},
        "_chapter_display_numbers": reader_doc.chapter_display_numbers(filenames, opts["config"]),
        "_chapters": list(chapters),
    }


def _images_root():
    return os.path.join(tempfile.gettempdir(), "Glossarion_EpubImages")


def test_page_builder_matches_legacy_on_random_books(legacy, el, tmp_path, monkeypatch):
    qapp()
    rng = random.Random(SEED * 11)
    books = {}
    checked = collections.Counter()
    for state in range(STATES):
        book_key = state % max(1, min(STATES, 8))
        if book_key not in books:
            root = tmp_path / f"book{book_key}"
            epub, _chapters, extra_dirs = build_reader_book(root, rng)
            loaded = reader_doc.load_epub_chapters(str(epub), True, {})
            books[book_key] = (root, epub, extra_dirs, loaded)
        root, epub, extra_dirs, (chapters, images, filenames) = books[book_key]
        opts = random_reader_options(rng, epub, root)
        _apply_env(monkeypatch, opts["env"], root)
        state_dict = _doc_state(epub, images, extra_dirs, opts, filenames, chapters)
        outputs = {}
        for label in ("legacy", "desktop", "plain"):
            shutil.rmtree(_images_root(), ignore_errors=True)
            if label == "plain":
                doc = reader_doc.ReaderDocument(
                    str(epub), images=images, extra_image_dirs=extra_dirs, config=opts["config"],
                    theme=opts["theme"], font_family=opts["font_family"], font_size=opts["font_size"],
                    line_spacing=opts["line_spacing"], show_raw=opts["show_raw"],
                    translated_overlay=opts["translated_overlay"], raw_epub_alt_path=opts["raw_epub_alt_path"],
                    translated_css_dirs=opts["translated_css_dirs"], workspace_mode=opts["workspace_mode"],
                    image_url_for=qt_file_url, chapter_filenames=filenames)
                processed = [doc.process_html(html) for _title, html in chapters]
                css = doc.embedded_css()
                pages = [doc.wrap(body, paginated, spread) for body in processed[:2]
                         for paginated, spread in ((False, 1), (True, 1), (True, 2))]
                all_body = doc.all_chapters_body(chapters)
                doc.close()
            else:
                cls = legacy.EpubReaderDialog if label == "legacy" else el.EpubReaderDialog
                stub = reader_stub(cls, state_dict)
                processed = [stub._process_html(html) for _title, html in chapters]
                css = stub._get_embedded_css()
                pages = [stub._wrap_html(body, paginated, spread) for body in processed[:2]
                         for paginated, spread in ((False, 1), (True, 1), (True, 2))]
                all_body = stub._all_chapters_html() if label == "desktop" else None
                stub._close_epub_image_zip()
            outputs[label] = {"processed": processed, "css": css, "pages": pages, "all": all_body,
                              "files": sorted(p.relative_to(_images_root()).as_posix()
                                              for p in Path(_images_root()).rglob("*")
                                              if Path(_images_root()).exists())}
        legacy_out, desktop_out, plain_out = outputs["legacy"], outputs["desktop"], outputs["plain"]
        assert desktop_out["all"] == plain_out["all"]
        desktop_out["all"] = plain_out["all"] = None
        assert desktop_out == legacy_out, f"desktop differs from legacy in state {state}"
        assert plain_out == legacy_out, f"ReaderDocument differs from legacy in state {state}"
        # the one-shot helpers agree with the long-lived document
        if not opts["translated_overlay"] and not opts["translated_css_dirs"] and not opts["raw_epub_alt_path"]:
            assert reader_doc.get_embedded_css(str(epub), config=opts["config"], show_raw=opts["show_raw"],
                                               workspace_mode=opts["workspace_mode"]) == legacy_out["css"]
        for (paginated, spread), page in zip(((False, 1), (True, 1), (True, 2)), legacy_out["pages"][:3]):
            assert reader_doc.wrap_reader_html(
                legacy_out["processed"][0], opts["theme"], font_family=opts["font_family"],
                font_size=opts["font_size"], line_spacing=opts["line_spacing"], paginated=paginated,
                spread_pages=spread, embedded_css=legacy_out["css"], epub_path=str(epub), config=opts["config"],
                show_raw=opts["show_raw"], workspace_mode=opts["workspace_mode"]) == page
        joined = "".join(legacy_out["processed"])
        checked["full_page"] += "full-page-img" in joined
        checked["first"] += "full-page-img-first" in joined
        checked["file_url"] += "file:///" in joined
        checked["css_font"] += "data:font/" in legacy_out["css"]
        checked["embedded"] += bool(legacy_out["css"])
        checked["svg_image"] += "<image" in joined and "file:///" in joined
    # the fuzz reached the interesting branches
    if STATES >= 20:
        for key in ("full_page", "first", "file_url", "css_font", "embedded", "svg_image"):
            assert checked[key] > 0, key


def test_reader_document_api(tmp_path):
    rng = random.Random(SEED * 13)
    epub, chapters, extra_dirs = build_reader_book(tmp_path, rng)
    loaded_chapters, images, filenames = reader_doc.load_epub_chapters(str(epub))
    served = []

    def image_url_for(path):
        served.append(path)
        return "http://127.0.0.1:8765/img/" + str(len(served))

    doc = reader_doc.ReaderDocument(str(epub), images=images, extra_image_dirs=extra_dirs, theme="sepia",
                                    font_family="Georgia", image_url_for=image_url_for,
                                    chapter_filenames=filenames)
    try:
        html = "<p><img src='../Images/wide.png'/></p><p><img src='../Images/small.png'/></p>"
        processed = doc.process_html(html)
        assert "http://127.0.0.1:8765/img/1" in processed and "file:" not in processed
        assert all(os.path.isfile(p) for p in served)
        assert doc.chapter_page(html, paginated=True) == doc.wrap(processed, True)
        assert reader_doc.reader_theme("Sepia") == reader_doc.READER_THEMES[2]
        assert reader_doc.reader_theme(99) == reader_doc.READER_THEMES[0]
        assert reader_doc.reader_theme({"bg": "#000000"})["bg"] == "#000000"
        assert doc._get_theme()["name"] == "Sepia"
        assert reader_doc.READER_THEME_NAMES[:3] == ("Dark", "Light", "Sepia")
        mobile = doc.chapter_page(html, paginated=True, mobile=True, chapter=2, initial_page=3)
        assert '"GLRDR:"' in mobile and "var INITIAL_PAGE = 3" in mobile and "CHAPTER = 2" in mobile
        assert reader_doc.process_chapter_html(html, images=images, epub_path=str(epub),
                                               image_url_for=lambda p: "x://" + os.path.basename(p)).count("x://") == 2
        assert reader_doc.reader_image_temp_dir(str(epub)).startswith(_images_root())
    finally:
        doc.close()
    assert doc._epub_image_zip is None
    # cancellation: should_stop -> None (desktop drops the result)
    assert reader_doc.load_epub_chapters(str(epub), use_cache=False, should_stop=lambda: True) is None
    assert reader_doc.search_chapters(loaded_chapters, "Chapter", should_stop=lambda: True) == []
    batches = []
    rows = reader_doc.search_chapters(loaded_chapters, "Chapter", on_batch=lambda r, d: batches.append((len(r), d)))
    assert batches[-1][1] is True and sum(n for n, _d in batches) == len(rows)
    with pytest.raises(reader_doc.ReaderLoadError):
        broken = tmp_path / "broken.epub"
        broken.write_bytes(b"not a zip")
        reader_doc.load_epub_chapters(str(broken), use_cache=False)


# =============================================================================================
# 4. the live stream
# =============================================================================================

_LIVE_LINES = (
    f"{BRAIN} Thinking...", "    a thought", "    deeper thought", ZWSP, "Thinking complete",
    f"{SATELLITE} Text streaming started", "First text token received", "<p>Hello</p>", "<h1>T</h1>",
    "plain content line", "", "   ", "Stream complete", "Translation completed", "Traceback (most recent call last):",
    'File "x.py", line 1', "[DEBUG] x", "[WARNING] y", "═══════", f"{CHECK} done", "# hash line",
    "    indented content", "[INFO] info", "trailing words", "<div>frag</div>", "after thinking...",
)


def random_live_messages(rng, count=None):
    messages = []
    for _ in range(count if count is not None else rng.randint(0, 60)):
        messages.append("\n".join(rng.choice(_LIVE_LINES) for _ in range(rng.randint(1, 3))))
    return messages


class _Cursor:
    class MoveOperation:
        End = 11

    def __init__(self, view):
        self.view = view

    def movePosition(self, _op):
        return True

    def insertText(self, text):
        self.view.inserted.append(text)


class _ScrollBar:
    def __init__(self):
        self.values = []

    def maximum(self):
        return 100

    def setValue(self, value):
        self.values.append(value)


class _ThinkView:
    def __init__(self):
        self.inserted = []
        self.bar = _ScrollBar()

    def textCursor(self):
        return _Cursor(self)

    def setTextCursor(self, _cursor):
        pass

    def verticalScrollBar(self):
        return self.bar

    def blockCount(self):
        return "".join(self.inserted).count("\n") + 1


class _Toggle:
    def __init__(self, checked):
        self.checked = checked
        self.texts = []

    def isChecked(self):
        return self.checked

    def setText(self, text):
        self.texts.append(text)


class _Label:
    def __init__(self):
        self.texts = []

    def setText(self, text):
        self.texts.append(text)


def live_stub(cls, **state):
    namespace = {name: getattr(cls, name) for name in (
        "_classify_live_line", "_drain_live_queue", "_drain_live_lines", "_wrap_live_html",
        "_resolve_live_output_folder", "_finish_live_translation", "_live_outcome", "_LIVE_STATUS_CHARS",
        "_get_theme") if hasattr(cls, name)}
    stub = type(f"{cls.__name__}LiveStub", (), namespace)()
    stub.__dict__.update(state)
    return stub


def test_live_classify_and_drain_match_legacy(legacy, el):
    rng = random.Random(SEED * 17)
    for state in range(STATES * 3):
        messages = random_live_messages(rng, 450 if state % 25 == 7 else None)
        toggle_checked = rng.random() < 0.5
        runs = {}
        for label, cls in (("legacy", legacy.EpubReaderDialog), ("new", el.EpubReaderDialog)):
            renders = []
            stub = live_stub(
                cls, _live_log_queue=collections.deque(messages), _live_content_buf="", _live_think_pending="",
                _live_log_pending="", _live_in_thinking=False, _live_streaming_text=False,
                _live_think_view=_ThinkView(), _live_think_toggle=_Toggle(toggle_checked))
            stub._render_live_content = lambda s=stub: renders.append(len(s._live_content_buf))
            ticks = 0
            while True:
                before = len(stub._live_log_queue)
                stub._drain_live_queue()
                ticks += 1
                if not before:
                    break
            runs[label] = {"content": stub._live_content_buf, "inserted": stub._live_think_view.inserted,
                           "toggle": stub._live_think_toggle.texts, "renders": renders, "ticks": ticks,
                           "thinking": stub._live_in_thinking, "streaming": stub._live_streaming_text,
                           "bar": stub._live_think_view.bar.values}
        assert runs["new"] == runs["legacy"], state
        # the plain classifier routes the same text the same way
        classifier = live_stream.LiveLineClassifier()
        for message in messages:
            classifier.feed(message)
        drained = []
        while True:
            result = classifier.drain()
            drained.append(result)
            if not result["drained"]:
                break
        assert classifier.content == runs["legacy"]["content"]
        assert classifier.thinking_text == "".join(runs["legacy"]["inserted"])
        assert classifier.streaming == runs["legacy"]["streaming"]
        assert sum(1 for r in drained if r["content_added"]) == len(runs["legacy"]["renders"])
        # classify_live_lines == legacy _classify_live_line with fresh state
        lines = [line for message in messages[:20] for line in message.split("\n")]
        fresh = live_stub(legacy.EpubReaderDialog, _live_in_thinking=False, _live_streaming_text=False)
        assert live_stream.classify_live_lines(lines) == [(fresh._classify_live_line(line), line) for line in lines]


def test_live_wrap_and_output_folder_match_legacy(legacy, el, tmp_path, monkeypatch):
    rng = random.Random(SEED * 19)
    output = Path(os.environ["OUTPUT_DIRECTORY"])
    second = tmp_path / "SecondRoot"
    second.mkdir()
    for state in range(STATES * 2):
        theme = rng.choice([0, 1, 2, 3, 4, 5, 7])
        family = rng.choice(["Embedded CSS", "", None, "Georgia", "Noto Serif KR"])
        size = rng.choice([10, 14, 21.5])
        spacing = rng.choice([1.2, 1.8, 2.0])
        body = "".join(rng.choice(["line one\n", "<p>para</p>\n", "<h2>h</h2>", "two\nthree", "<br>", "x < y\n"])
                       for _ in range(rng.randint(0, 5)))
        legacy_stub = live_stub(legacy.EpubReaderDialog, _theme_index=theme, _font_family=family,
                                _font_size=size, _line_spacing=spacing)
        legacy_stub._get_theme = types.MethodType(legacy.EpubReaderDialog._get_theme, legacy_stub)
        expected = legacy_stub._wrap_live_html(body)
        assert live_stream.wrap_live_html(body, theme, family, size, spacing) == expected
        # output folder: overlay response folder first, else <root>/<epub stem> (progress first)
        stem = rng.choice(["Book A", "Book B", "Book C"])
        epub = tmp_path / "books" / f"{stem}.epub"
        chapter = rng.choice(["chapter0001.xhtml", "Chapter0002.XHTML", ""])
        for root in (output, second):
            folder = root / stem
            if rng.random() < 0.5:
                folder.mkdir(parents=True, exist_ok=True)
                if rng.random() < 0.5:
                    (folder / "translation_progress.json").write_text("{}", encoding="utf-8")
            elif folder.exists():
                shutil.rmtree(folder)
        overlay = {}
        if rng.random() < 0.4:
            response_dir = tmp_path / "overlay_ws"
            if rng.random() < 0.7:
                response_dir.mkdir(exist_ok=True)
            overlay[chapter.lower()] = {"path": str(response_dir / "response_x.html")}
        config = rng.choice([{}, {"output_directory_override": str(second)}, {"output_roots": [str(second)]}])
        if rng.random() < 0.3:
            monkeypatch.setenv("OUTPUT_DIRECTORY", str(second))
        else:
            monkeypatch.setenv("OUTPUT_DIRECTORY", str(output))
        results = []
        for cls in (legacy.EpubReaderDialog, el.EpubReaderDialog):
            stub = live_stub(cls, _live_chapter_file=chapter, _translated_overlay=overlay,
                             _live_epub_path=str(epub) if rng.random() < 0.8 else "",
                             _epub_path=str(epub), _config=config)
            results.append(stub._resolve_live_output_folder())
        assert results[0] == results[1]
        assert live_stream.resolve_live_output_folder(chapter, overlay, str(epub), config) == results[0]


class _Timer:
    calls = []

    @staticmethod
    def singleShot(ms, _callback):
        _Timer.calls.append(ms)


class _Gui:
    def __init__(self, stop_requested):
        self.stop_requested = stop_requested
        self.logs = []

    def append_log(self, message):
        self.logs.append(message)


def build_live_finish_case(root: Path, rng: random.Random) -> dict:
    output = root / "Output"
    output.mkdir(parents=True, exist_ok=True)
    (root / "Library").mkdir(exist_ok=True)
    stem = "Live Book"
    chapter = rng.choice(["chapter0001.xhtml", "chapter0002.xhtml", "Chapter0001.XHTML", ""])
    scenario = rng.choice(["no_folder", "no_progress", "completed", "completed_missing", "in_progress",
                           "failed", "other_chapter", "overlay"])
    workspace = output / stem
    if scenario != "no_folder":
        workspace.mkdir()
    if scenario not in ("no_folder", "no_progress"):
        entries = {}
        target = "chapter0001" if scenario != "other_chapter" else "chapter0009"
        status = {"completed": "completed", "completed_missing": "completed", "in_progress": "in_progress",
                  "failed": "failed", "other_chapter": "in_progress", "overlay": "in_progress"}[scenario]
        entries[f"1_{target}"] = {"status": status, "original_basename": f"{target}.xhtml",
                                  "output_file": f"response_{target}.html", "actual_num": 1}
        entries["2_chapter0003"] = {"status": "completed", "original_basename": "chapter0003.xhtml",
                                    "output_file": "response_chapter0003.html"}
        (workspace / "translation_progress.json").write_text(json.dumps({"chapters": entries}, indent=1),
                                                             encoding="utf-8")
        if scenario != "completed_missing":
            (workspace / f"response_{target}.html").write_text("<p>partial</p>", encoding="utf-8")
        (workspace / "response_chapter0003.html").write_text("<p>done</p>", encoding="utf-8")
    return {"stem": stem, "chapter": chapter, "scenario": scenario, "workspace": workspace,
            "stop": rng.random() < 0.5, "gui": rng.random() < 0.85}


def test_live_finish_matches_legacy(legacy, el, tmp_path, monkeypatch):
    rng = random.Random(SEED * 23)
    sandbox = Sandbox(tmp_path / "live", monkeypatch)
    monkeypatch.setattr(legacy, "QTimer", _Timer)
    monkeypatch.setattr(el, "QTimer", _Timer)
    for state in range(max(12, STATES)):
        case = sandbox.build(build_live_finish_case, rng)
        epub = str(tmp_path / "books" / f"{case['stem']}.epub")
        overlay = {}
        if case["scenario"] == "overlay" and case["chapter"]:
            overlay = {case["chapter"].lower(): {"path": str(case["workspace"] / "response_chapter0001.html")}}
        runs = {}
        for label, cls in (("legacy", legacy.EpubReaderDialog), ("new", el.EpubReaderDialog)):
            sandbox.reset()
            _Timer.calls = []
            gui = _Gui(case["stop"]) if case["gui"] else None
            calls = []
            stub = live_stub(cls, _live_translate_active=True, _live_poll_timer=None, _live_drain_timer=None,
                             _live_gui_ref=gui, _translate_btn=_Label(), _live_status_label=_Label(),
                             _live_chapter_file=case["chapter"], _translated_overlay=overlay,
                             _live_epub_path=epub, _epub_path=epub, _config={})
            stub._drain_live_queue = lambda c=calls: c.append("drain")
            stub._teardown_live_listener = lambda c=calls: c.append("teardown")
            stub._on_overlay_refresh_tick = lambda c=calls: c.append("refresh")
            stub._finish_live_translation()
            runs[label] = {"status": stub._live_status_label.texts, "button": stub._translate_btn.texts,
                           "logs": gui.logs if gui else None, "timer": list(_Timer.calls), "calls": calls,
                           "tree": sandbox.snapshot()}
        assert runs["new"] == runs["legacy"], (state, case["scenario"])
        # the plain classifier's outcome is the same decision
        sandbox.reset()
        outcome = live_stream.LiveLineClassifier(case["chapter"], epub, overlay, {}).outcome(
            was_stopped=bool(case["gui"] and case["stop"]))
        assert [outcome["status_text"]] == runs["legacy"]["status"]
        assert outcome["delay_ms"] == runs["legacy"]["timer"][0]
        assert bool(outcome["cleaned"]) == bool(runs["legacy"]["logs"]) or not case["gui"]
        assert sandbox.snapshot() == runs["legacy"]["tree"]


def test_live_constants_match_the_desktop_source():
    source = git_text("src/epub_library.py")
    for needle in ("setInterval(90)", "setInterval(700)", "QTimer.singleShot(2500",
                   "QTimer.singleShot(2200 if completed else 1200", "⏹ Stopping…",
                   live_stream.LIVE_FINISHED_TEXT, live_stream.LIVE_STOPPED_TEXT, live_stream.LIVE_INCOMPLETE_TEXT,
                   chr(92) + "U0001f6f0" + chr(0xFE0F) + " Translating “{os.path.basename(chapter_file)}”"
                   " — waiting for stream…"):
        assert needle in source, needle
    assert live_stream.LIVE_WAITING_TEXT == chr(0x1F6F0) + chr(0xFE0F) + " Translating “{name}” — waiting for stream…"
    assert live_stream.LIVE_DRAIN_INTERVAL_MS == 90 and live_stream.LIVE_POLL_INTERVAL_MS == 700
    assert live_stream.LIVE_POLL_START_DELAY_MS == 2500
    assert live_stream.LIVE_RETURN_DELAY_MS == {True: 2200, False: 1200}
    assert live_stream.LIVE_STATUS_CHARS is live_stream.LiveStreamMixin._LIVE_STATUS_CHARS


# =============================================================================================
# 5. the mobile shell
# =============================================================================================

def _mobile_pieces(paginated, event_url, chapter, initial_page):
    css = reader_doc._MOBILE_COMMON_CSS + (reader_doc._MOBILE_PAGED_CSS if paginated
                                           else reader_doc._MOBILE_SCROLL_CSS)
    script = "<script>" + reader_doc._MOBILE_BRIDGE_JS % {
        "prefix": json.dumps(reader_doc.MOBILE_EVENT_PREFIX),
        "event_url": json.dumps(event_url),
        "chapter": json.dumps(chapter),
        "initial_page": json.dumps(int(initial_page or 0)),
    } + "</script>"
    return css, script


def test_mobile_shell_only_adds_the_touch_layer():
    rng = random.Random(SEED * 29)
    for _ in range(60):
        paginated = rng.random() < 0.6
        kwargs = {"font_family": rng.choice(["Embedded CSS", "Georgia", "Consolas"]),
                  "font_size": rng.choice([11, 14, 20]), "line_spacing": rng.choice([1.4, 1.8]),
                  "paginated": paginated, "spread_pages": rng.choice([1, 2]),
                  "embedded_css": rng.choice(["", "p { color: red; }"]),
                  "workspace_mode": rng.random() < 0.2}
        theme = rng.choice([0, 2, "Midnight"])
        body = rng.choice(["<p>x</p>", "<h1>T</h1><p>a</p><p>b</p>", ""])
        chapter = rng.choice([None, 0, 7])
        initial = rng.choice([None, 0, 3])
        event_url = rng.choice([reader_doc.MOBILE_EVENT_PATH, "/x/__ev"])
        desktop = reader_doc.wrap_reader_html(body, theme, **kwargs)
        mobile = reader_doc.wrap_reader_html(body, theme, mobile=True, event_url=event_url, chapter=chapter,
                                             initial_page=initial, **kwargs)
        css, script = _mobile_pieces(paginated, event_url, chapter, initial)
        head = mobile.index("<head>") + len("<head>")
        assert mobile[head:head + len(reader_doc._MOBILE_VIEWPORT)] == reader_doc._MOBILE_VIEWPORT
        assert mobile.count(script) == 1 and mobile.index(script) < mobile.index("</head>")
        assert css in mobile and mobile.index(css) < mobile.index("</style>")
        stripped = mobile.replace(reader_doc._MOBILE_VIEWPORT, "", 1).replace(css, "", 1).replace(script, "", 1)
        assert stripped == desktop
        assert "viewport-fit=cover" in mobile and "env(safe-area-inset-" in mobile
        if paginated:
            assert "100dvh" in mobile and "-webkit-column-break-before" in mobile
            # the bridge drives the desktop paging functions; they must still exist
            for name in ("function _setupColumns", "function _pageCountFor", "function _pageWidthFor",
                         "_PAGE_GAP", "_CURRENT_PAGE", "id='columns'"):
                assert name in desktop, name
        assert json.dumps(reader_doc.MOBILE_EVENT_PREFIX) in mobile and json.dumps(event_url) in mobile
    assert reader_doc.MOBILE_EVENT_PREFIX == "GLRDR:" and reader_doc.MOBILE_EVENT_PATH == "/__ev"


_WEBENGINE_PROBE = r'''
import json, os, sys, threading, time
sys.path.insert(0, __SRC__)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu --no-sandbox --disable-logging")
try:
    from PySide6.QtCore import Qt, QCoreApplication, QUrl, QEventLoop
    QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    from PySide6.QtWidgets import QApplication
    from PySide6.QtWebEngineCore import QWebEnginePage
    from PySide6.QtWebEngineWidgets import QWebEngineView
except Exception as exc:
    print("SKIP " + repr(exc))
    sys.exit(0)
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import reader_doc

PAGES, POSTED, CONSOLE = {}, [], []

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass
    def do_GET(self):
        body = PAGES.get(self.path.split("?")[0].split("#")[0])
        if body is None:
            self.send_response(404); self.end_headers(); return
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def do_POST(self):
        data = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if self.path == reader_doc.MOBILE_EVENT_PATH:
            POSTED.append(json.loads(data.decode("utf-8")))
        self.send_response(204)
        self.end_headers()

server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
base = "http://127.0.0.1:%d" % server.server_address[1]
body = ("<p><a id='lnk' href='#note1'>note</a></p>"
        + "".join("<p>Paragraph %d %s</p>" % (i, "lorem ipsum dolor sit amet " * 30) for i in range(40)))
PAGES["/paged.html"] = reader_doc.wrap_reader_html(body, "Sepia", font_family="Georgia", paginated=True,
                                                   mobile=True, chapter=3, initial_page=2, embedded_css="")
PAGES["/scroll.html"] = reader_doc.wrap_reader_html(body, 0, font_family="Georgia", paginated=False,
                                                    mobile=True, chapter=4, embedded_css="")
app = QApplication([])

class Page(QWebEnginePage):
    def javaScriptConsoleMessage(self, level, message, line, source):
        if message.startswith(reader_doc.MOBILE_EVENT_PREFIX):
            CONSOLE.append(json.loads(message[len(reader_doc.MOBILE_EVENT_PREFIX):]))

view = QWebEngineView()
page = Page(view)
view.setPage(page)
view.resize(420, 720)
view.show()

def pump(seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents(QEventLoop.AllEvents, 50)
        time.sleep(0.01)

def wait(predicate, timeout=40.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.processEvents(QEventLoop.AllEvents, 50)
        if predicate():
            return True
        time.sleep(0.01)
    return False

def run_js(code):
    box = []
    page.runJavaScript(code, 0, lambda value: box.append(value))
    wait(lambda: bool(box), 10)
    return box[0] if box else None

def act(code, settle=0.5):
    start = len(CONSOLE)
    run_js(code)
    pump(settle)
    return CONSOLE[start:]

def click(x, target="document.body"):
    return ("%s.dispatchEvent(new MouseEvent('click', {bubbles: true, cancelable: true, "
            "clientX: %d, clientY: 200}))" % (target, x))

def swipe(x0, x1):
    return ("(function(){var t=document.body;function mk(x){return new Touch({identifier:1,target:t,"
            "clientX:x,clientY:300});}document.dispatchEvent(new TouchEvent('touchstart',{touches:[mk(%d)],"
            "changedTouches:[mk(%d)],bubbles:true}));document.dispatchEvent(new TouchEvent('touchend',"
            "{touches:[],changedTouches:[mk(%d)],bubbles:true}));})()" % (x0, x0, x1))

result = {}
view.setUrl(QUrl(base + "/paged.html"))
result["loaded"] = wait(lambda: any(e.get("type") == "ready" for e in CONSOLE))
pump(0.5)
result["start"] = list(CONSOLE)
result["goto0"] = act("GLRDR.goTo(0)")
result["prev_at_start"] = act("GLRDR.prev()")
result["goto_end"] = act("GLRDR.goTo(99999)")
result["next_at_end"] = act("GLRDR.next()")
result["tap_left"] = act(click(10))
result["tap_center"] = act(click(210))
result["tap_right"] = act(click(410))
result["link"] = act(click(210, "document.getElementById('lnk')"))
result["swipe_right"] = act(swipe(100, 300))
result["click_after_swipe"] = act(click(410))
result["swipe_left"] = act(swipe(300, 100))
result["api"] = json.loads(run_js("JSON.stringify([GLRDR.page(), GLRDR.count()])") or "null")
pump(1.0)
result["console"] = list(CONSOLE)
result["posted"] = list(POSTED)
mark_console, mark_posted = len(CONSOLE), len(POSTED)
run_js("GLRDR.setTransport('fetch'); GLRDR.goTo(1)")
pump(1.0)
result["fetch_only"] = {"console": CONSOLE[mark_console:], "posted": POSTED[mark_posted:]}
del CONSOLE[:]
del POSTED[:]
view.setUrl(QUrl(base + "/scroll.html#f=0.5"))
result["scroll_loaded"] = wait(lambda: any(e.get("type") == "ready" for e in CONSOLE))
pump(0.8)
result["scroll_start"] = list(CONSOLE)
result["scroll"] = act("window.scrollTo(0, 400)", 1.0)
result["scroll_tap"] = act(click(10))
print("RESULT " + json.dumps(result))
server.shutdown()
'''


def test_mobile_bridge_pages_in_webengine(tmp_path):
    if (os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS")) and os.environ.get("PARITY_U5_WEBENGINE") != "1":
        pytest.skip("QtWebEngine probe runs locally (set PARITY_U5_WEBENGINE=1 on CI)")
    script = tmp_path / "webengine_probe.py"
    script.write_text(_WEBENGINE_PROBE.replace("__SRC__", repr(str(SRC))), encoding="utf-8")
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONIOENCODING="utf-8")
    try:
        proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=240, env=env, cwd=str(tmp_path))
    except subprocess.TimeoutExpired:
        pytest.fail("QtWebEngine probe timed out")
    lines = [line for line in proc.stdout.splitlines() if line.startswith(("RESULT ", "SKIP "))]
    assert lines, (proc.returncode, proc.stdout[-2000:], proc.stderr[-3000:])
    if lines[-1].startswith("SKIP "):
        pytest.skip("QtWebEngine unavailable: " + lines[-1][5:])
    result = json.loads(lines[-1][len("RESULT "):])
    assert result["loaded"], result

    def only(events, **expected):
        assert len(events) == 1, events
        for key, value in expected.items():
            assert events[0].get(key) == value, (key, events)
        return events[0]

    initial = next(e for e in result["start"] if e.get("reason") == "initial")
    ready = next(e for e in result["start"] if e["type"] == "ready")
    assert initial["type"] == "page" and initial["page"] == 2 and initial["seq"] < ready["seq"]
    # a late window 'load' re-applies the same page; nothing else fires on open
    assert all(e["type"] == "ready" or (e["type"] == "page" and e["page"] == 2) for e in result["start"])
    assert ready["type"] == "ready" and ready["paginated"] is True and ready["page"] == 2
    count = ready["count"]
    assert count >= 6
    assert all(e["chapter"] == 3 for e in result["console"])
    only(result["goto0"], type="page", page=0, reason="api", count=count)
    only(result["prev_at_start"], type="edge", edge="start", page=0)
    only(result["goto_end"], type="page", page=count - 1)
    only(result["next_at_end"], type="edge", edge="end", page=count - 1)
    only(result["tap_left"], type="page", page=count - 2, reason="prev")
    only(result["tap_center"], type="tap", zone="center")
    only(result["tap_right"], type="page", page=count - 1, reason="next")
    only(result["link"], type="link", href="#note1")
    only(result["swipe_right"], type="page", page=count - 2, reason="prev")
    assert result["click_after_swipe"] == []  # the click that ends a swipe is swallowed
    only(result["swipe_left"], type="page", page=count - 1, reason="next")
    assert result["api"] == [count - 1, count]
    seqs = [e["seq"] for e in result["console"]]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    # transport 'both': every console event has its fetch('/__ev') twin
    assert sorted(result["posted"], key=lambda e: e["seq"]) == result["console"]
    assert result["fetch_only"]["console"] == []
    only(result["fetch_only"]["posted"], type="page", page=1)
    # scroll layout: no columns, fraction events, side taps are plain taps
    assert result["scroll_loaded"]
    scroll_ready = next(e for e in result["scroll_start"] if e["type"] == "ready")
    assert scroll_ready["paginated"] is False and scroll_ready["chapter"] == 4 and scroll_ready["count"] == 1
    fractions = [e["fraction"] for e in result["scroll_start"] + result["scroll"] if e["type"] == "scroll"]
    assert fractions and all(0 <= f <= 1 for f in fractions)
    assert any(0.3 <= f <= 0.7 for f in fractions)  # '#f=0.5' restored the position
    only(result["scroll_tap"], type="tap", zone="left")


# =============================================================================================
# 6. bilingual chapters and native blocks
# =============================================================================================

def test_bilingual_chapter_alignment():
    raw = "<html><body><h1>제목</h1><p>첫째</p><p>둘째</p><p></p><div><p>셋째</p></div></body></html>"
    translated = "<h1>Title</h1><p>First</p><p>Second</p><p>Third</p>"
    mode, pairs = reader_doc.bilingual_alignment(raw, translated)
    assert mode == "blocks"
    assert pairs == [("<h1>제목</h1>", "<h1>Title</h1>"), ("<p>첫째</p>", "<p>First</p>"),
                     ("<p>둘째</p>", "<p>Second</p>"), ("<p>셋째</p>", "<p>Third</p>")]
    page = reader_doc.build_bilingual_chapter(raw, translated)
    assert page.startswith("<style>") and page.count('class="glr-bi-pair"') == 4
    assert page.index("첫째") < page.index("First") < page.index("둘째")
    # counts too far apart -> whole sections, labels escaped
    mode, pairs = reader_doc.bilingual_alignment(raw, translated + "<p>4</p><p>5</p><p>6</p>")
    assert (mode, pairs) == ("sections", [])
    sections = reader_doc.build_bilingual_chapter(raw, "<p>only</p>" * 9, original_label="<Raw>")
    assert "&lt;Raw&gt;" in sections and sections.index("glr-bi-original") < sections.index("glr-bi-translated")
    # within the threshold the shorter side leaves blanks
    raw_ten = "".join(f"<p>r{i}</p>" for i in range(10))
    tr_nine = "".join(f"<p>t{i}</p>" for i in range(9))
    mode, pairs = reader_doc.bilingual_alignment(raw_ten, tr_nine)
    assert mode == "blocks" and pairs[-1] == ("<p>r9</p>", "")
    assert reader_doc.bilingual_alignment(raw_ten, tr_nine, threshold=0.05)[0] == "sections"
    assert reader_doc.bilingual_alignment("", "<p>x</p>") == ("sections", [])
    # escaped markup (older translations) is rehydrated before pairing
    assert reader_doc.bilingual_alignment("<p>a</p>", "&lt;p&gt;b&lt;/p&gt;") == ("blocks", [("<p>a</p>", "<p>b</p>")])
    # images outside blocks and nested lists keep document order
    mode, pairs = reader_doc.bilingual_alignment("<img src='a.png'/><ul><li>x<p>y</p></li></ul><hr/>",
                                                 "<img src='a.png'/><ul><li>X</li></ul><hr/>")
    assert mode == "blocks" and [p[0][:4] for p in pairs] == ["<img", "<li>", "<hr/"]


def test_html_to_blocks():
    html = ("<html><head><style>p{}</style><script>x()</script></head><body>"
            "<h2>Chapter  One</h2><p>Hello\n   world</p><p><img src='a.png' alt='A'/>Caption text</p>"
            "<pre>  keep\n  spacing</pre><blockquote><p>quoted</p></blockquote><ul><li>item 1</li></ul>"
            "<hr/><div>loose div text</div><img src='b.png'/><p>   </p>&lt;em&gt;rehydrated&lt;/em&gt;"
            "<dl><dt>term</dt><dd>definition</dd></dl></body></html>")
    blocks = reader_doc.html_to_blocks(html)
    kinds = [(b["kind"], b["text"] or b["src"]) for b in blocks]
    assert kinds == [
        ("heading", "Chapter One"), ("paragraph", "Hello world"), ("image", "a.png"),
        ("paragraph", "Caption text"), ("pre", "  keep\n  spacing"), ("quote", "quoted"),
        ("item", "item 1"), ("rule", ""), ("paragraph", "loose div text"), ("image", "b.png"),
        ("item", "term"), ("item", "definition"),
    ]
    assert blocks[0]["level"] == 2 and blocks[2]["alt"] == "A"
    assert all(set(b) == {"kind", "text", "level", "src", "alt"} for b in blocks)
    assert reader_doc.html_to_blocks("") == []


# =============================================================================================
# 7. offscreen smoke: legacy vs new EpubReaderDialog
# =============================================================================================

def test_reader_dialog_renders_like_legacy(legacy, el, tmp_path, monkeypatch):
    app = qapp()
    from PySide6.QtCore import QEventLoop

    rng = random.Random(SEED * 31)
    epub, chapters, _extra = build_reader_book(tmp_path, rng)

    def wait_for(predicate, timeout=20.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            app.processEvents(QEventLoop.AllEvents, 50)
            if predicate():
                return True
            time.sleep(0.02)
        return False

    snapshots = {}
    for label, module in (("legacy", legacy), ("new", el)):
        shutil.rmtree(os.path.join(tempfile.gettempdir(), "Glossarion_EpubCache"), ignore_errors=True)
        shutil.rmtree(_images_root(), ignore_errors=True)
        monkeypatch.setattr(module, "_HAS_WEBENGINE", False)
        dialog = module.EpubReaderDialog(str(epub), config={})
        dialog.show()
        try:
            assert wait_for(lambda: bool(dialog._chapters))
            wait_for(lambda: False, timeout=0.5)
            pages = []
            original = dialog._wrap_html

            def spy(*args, _original=original, _pages=pages, **kwargs):
                page = _original(*args, **kwargs)
                _pages.append(page)
                return page

            dialog._wrap_html = spy
            for layout in (module.LAYOUT_SCROLL, module.LAYOUT_SINGLE, module.LAYOUT_DOUBLE, module.LAYOUT_ALL):
                for row in range(min(2, len(dialog._chapters))):
                    dialog._layout_mode = layout
                    dialog._current_row = row
                    dialog._render_current()
            toc = [dialog._toc_list.item(i).text() for i in range(dialog._toc_list.count())]
            snapshots[label] = {"pages": pages, "toc": toc, "numbers": list(dialog._chapter_display_numbers),
                                "filenames": list(dialog._chapter_filenames),
                                "chapters": norm(list(dialog._chapters))}
        finally:
            dialog.close()
            dialog.deleteLater()
            wait_for(lambda: False, timeout=0.3)
    assert snapshots["new"]["pages"] and snapshots["new"]["toc"]
    assert snapshots["legacy"] == snapshots["new"]
