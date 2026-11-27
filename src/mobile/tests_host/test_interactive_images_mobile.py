"""Interactive Images on mobile: the Book page's Images tab, Tools › Interactive Images (profiles, models,
references, styles + the preset preview), the ``interactive_images`` job kind, the settings schema rows and
the fake model server's planner answers. Everything runs the shared modules (interactive_images_core,
interactive_images_progress_core, interactive_images_runner, interactive_css_presets).

Run from src/mobile:
    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic \
        tests_host/test_interactive_images_mobile.py
"""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import os
import sys
import types
import urllib.request
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
MOBILE_DIR = HERE.parent
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
for _path in (APP_DIR,):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

ft = pytest.importorskip("flet")
pytest.importorskip("interactive_images_core")

import interactive_css_presets as presets  # noqa: E402
import interactive_images_core as core  # noqa: E402
import interactive_images_progress_core as ipc  # noqa: E402
import interactive_images_runner as runner  # noqa: E402

CH = ('<html><head><title>Chapter {n}</title></head><body>\n<h1>Chapter {n}</h1>\n'
      '<p>Mira walked to the lantern gate in the rain of chapter {n}, carrying her brass lamp.</p>\n'
      '<p>"Hello," said Tomas, without looking up from the map.</p>\n</body></html>')


def _u7():
    name = "_glossarion_interactive_images_u7"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HERE / "test_tools_u7.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _png() -> bytes:
    from glossarion_mobile.diagnostics.fake_llm_server import png_bytes

    return png_bytes(8, 8)


def _book(tmp_path: Path, chapters: int = 2) -> Path:
    folder = tmp_path / "Output" / "Lantern"
    folder.mkdir(parents=True)
    entries = {}
    for n in range(1, chapters + 1):
        name = f"response_{n:03d}_Chapter {n}.html"
        (folder / name).write_text(CH.format(n=n), encoding="utf-8")
        entries[str(n)] = {"output_file": name, "status": "completed", "actual_num": n}
    (folder / "translation_progress.json").write_text(json.dumps({"chapters": entries}), encoding="utf-8")
    return folder


class FakeClient:
    """The planner answered by the fake model server's own ``planner_reply``; images as a PNG data URI."""

    def __init__(self, kind: str, sent: list) -> None:
        self.kind = kind
        self.sent = sent

    def send(self, messages, temperature=None, max_tokens=None, context=None):
        from glossarion_mobile.diagnostics.fake_llm_server import FAKE_STYLESHEET, planner_reply

        self.sent.append((self.kind, context, messages))
        if self.kind == "image":
            return "data:image/png;base64," + base64.b64encode(_png()).decode("ascii"), "stop"
        prompt = messages[-1]["content"]
        if context == runner.CSS_CONTEXT:
            return f"```css\n{FAKE_STYLESHEET}```", "stop"
        return planner_reply(prompt), "stop"


@pytest.fixture
def fake_models(monkeypatch):
    sent: list = []
    monkeypatch.setattr(runner, "_default_client_factory",
                        lambda kind, model, config, output_dir: FakeClient(kind, sent))
    monkeypatch.setattr(runner, "configure_image_key_pool", lambda config, log: None)
    return sent


class FakeService:
    """The LibraryService surface the Images tab and its model use."""

    def __init__(self, folder: Path, kinds=("interactive_images",)) -> None:
        from glossarion_mobile.services.library import SharedCore

        self.folder = str(folder)
        self.core = SharedCore()
        self.kinds = set(kinds)
        self.specs: list = []
        self.dirty = 0

    def workspace_for(self, book):
        return self.folder

    def has_job_kind(self, kind):
        return kind in self.kinds

    def origin_for(self, book):
        return {"type": "library", "bid": "b" * 12}

    async def submit(self, spec):
        self.specs.append(spec)
        return f"job{len(self.specs)}"

    async def io(self, fn, *args):
        return await asyncio.to_thread(fn, *args)

    def mark_dirty(self):
        self.dirty += 1


def _run_job(folder: Path, config: dict, keys=None, stop=False):
    from glossarion_mobile import job_kinds

    u7 = _u7()
    events: list = []
    ctx = u7.FakeCtx(types.SimpleNamespace(config=config), params={"keys": keys} if keys is not None else {},
                     inputs=(str(folder),), config=config,
                     host=types.SimpleNamespace(emit=lambda kind, **data: events.append((kind, data))))
    ctx.stop = stop
    result = job_kinds.get_kind("interactive_images").run(ctx)
    return ctx, result, events


CONFIG = {"model": "fake-text-model", "interactive_images_image_model": "gpt-image-1"}


# ==========================================================================
# The job kind and the Images tab model (shared cores)
# ==========================================================================


def test_job_kind_is_registered_with_the_job_service_vocabulary():
    from glossarion_mobile import job_kinds
    from glossarion_mobile.services.jobs import BOOK_JOB_KINDS, JobKind

    assert job_kinds.KIND_MODULES["interactive_images"] == "interactive_images"
    assert JobKind("interactive_images") is JobKind.INTERACTIVE_IMAGES
    info = job_kinds.get_kind("interactive_images")
    assert (info.verb, info.icon, info.stop_kind) == ("Illustrating", "IMAGE", "translation")
    assert "interactive_images" in BOOK_JOB_KINDS


def test_the_job_builds_the_book_and_the_images_view_follows(tmp_path, fake_models):
    from glossarion_mobile.ui.library import progress_model as pm

    folder = _book(tmp_path)
    service = FakeService(folder)
    view = pm.load_images_view(service, {"name": "Lantern"})
    assert [r.status for r in view.rows] == ["not_built", "not_built"] and view.tab_count == "0/2"
    assert {c.group: c.count for c in view.chips}["not_built"] == 2
    assert view.output_dir == str(folder) and not view.error

    ctx, result, events = _run_job(folder, dict(CONFIG))
    assert result == {"ok": True, "outputs": []}, ctx.logs
    assert ctx.phases[0] == "Illustrating" and ctx.output_dir == str(folder)
    assert ctx.results["interactive_images"]["completed"] == 2
    assert events[0] == ("progress", {"label": "Illustrating", "completed": 0, "total": 2})
    assert events[-1][1]["completed"] == 2
    for n in (1, 2):
        html = (folder / f"response_{n:03d}_Chapter {n}.html").read_text(encoding="utf-8")
        assert f'src="Interactive_Images/Glossarion_ch{n:03d}_1.png"' in html
        assert core.check_text_unchanged(CH.format(n=n), html)[0]
        assert (folder / "Interactive_Images" / f"Glossarion_ch{n:03d}_1.png").is_file()
    kinds = [kind for kind, _context, _messages in fake_models]
    assert kinds == ["text", "image", "text", "image"]

    view = pm.load_images_view(service, {"name": "Lantern"}, previous=view)
    assert [r.status for r in view.rows] == ["completed", "completed"] and view.tab_count == "2/2"
    assert all(r.images == 1 and r.palette == "completed" for r in view.rows)
    plan = pm.run_images_action(service, view, "show_plan", list(view.rows[:1]))
    assert "Image 1 after paragraph" in plan and "response_001" in plan
    targets = pm.run_images_action(service, view, "open_images", list(view.rows[:1]))
    assert targets[0]["images"] == [str(folder / "Interactive_Images" / "Glossarion_ch001_1.png")]

    # retranslating a chapter makes its row stale (the hash of the visible text it was built from)
    chapter = folder / "response_002_Chapter 2.html"
    chapter.write_text(chapter.read_text(encoding="utf-8").replace("Hello", "Hi"), encoding="utf-8")
    view = pm.load_images_view(service, {"name": "Lantern"}, previous=view)
    assert view.rows[1].status == "stale" and "stale" in view.tab_count

    assert pm.run_images_action(service, view, "mark", list(view.rows[:1])) == 1
    view = pm.load_images_view(service, {"name": "Lantern"}, previous=view)
    assert view.rows[0].status == "pending"
    assert pm.run_images_action(service, view, "delete", list(view.rows)) == 2
    view = pm.load_images_view(service, {"name": "Lantern"}, previous=view)
    assert [r.status for r in view.rows] == ["not_built", "not_built"]
    first = (folder / "response_001_Chapter 1.html").read_text(encoding="utf-8")
    assert "glx-" not in first and not list((folder / "Interactive_Images").glob("Glossarion_*.png"))


def test_the_job_honours_stop_and_the_css_request_mode(tmp_path, fake_models):
    folder = _book(tmp_path)
    ctx, result, _events = _run_job(folder, dict(CONFIG), stop=True)
    assert result == {"ok": None, "outputs": []} and ctx.results["interactive_images"]["stopped"] is True
    assert not list((folder / "Interactive_Images").glob("Glossarion_*.png"))

    from glossarion_mobile import job_kinds

    u7 = _u7()
    css_ctx = u7.FakeCtx(types.SimpleNamespace(config=dict(CONFIG)), params={"css_request": "soft blue notebook"},
                         config=dict(CONFIG))
    assert job_kinds.get_kind("interactive_images").run(css_ctx) == {"ok": True, "outputs": []}
    made = css_ctx.results["interactive_css"]
    assert made["request"] == "soft blue notebook" and ".glx-dialogue" in made["css"]
    # the request is the profile's CSS prompt with {user_request} and {css_classes} filled, nothing added
    _kind, context, messages = fake_models[-1]
    assert context == runner.CSS_CONTEXT
    assert messages[-1]["content"] == core.fill_placeholders(
        core.settings_from_config(CONFIG).css_prompt,
        {"user_request": "soft blue notebook", "css_classes": "\n".join(presets.CLASS_VOCABULARY)})


def test_the_job_refuses_a_book_without_an_output_folder(tmp_path):
    from glossarion_mobile import job_kinds
    from glossarion_mobile.services.jobs import JobError

    ctx = _u7().FakeCtx(types.SimpleNamespace(config={}), inputs=(str(tmp_path / "missing"),), config={})
    with pytest.raises(JobError):
        job_kinds.get_kind("interactive_images").run(ctx)


def test_fake_model_server_answers_the_real_planner_prompt():
    from glossarion_mobile.diagnostics.fake_llm_server import classify_request, planner_reply

    settings = core.settings_from_config({})
    blocks = core.number_blocks(CH.format(n=3))
    prompt = runner.build_planner_prompts(settings, blocks, "Chapter 3", [], [])[0][0]
    payload = {"messages": [{"role": "user", "content": prompt}]}
    assert classify_request(payload) == "planner"
    plan = core.parse_plan(planner_reply(prompt), len(blocks), [], 1)
    assert len(plan.images) == 1 and plan.images[0].after_paragraph == 2  # the first long paragraph
    css = {"messages": [{"role": "user", "content": core.fill_placeholders(settings.css_prompt, {
        "user_request": "x", "css_classes": "glx-book"})}]}
    assert classify_request(css) == "stylesheet"
    translation = {"messages": [{"role": "user", "content": "<p>새벽 안개</p>"}]}
    assert classify_request(translation) == "translation"


# ==========================================================================
# Routes, settings schema, curated settings page
# ==========================================================================


def test_routes_book_tab_and_hub_tile():
    from glossarion_mobile.ui.library.book_page import BOOK_TABS, WIDE_TABS
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.tools.feature import SCREEN_ROUTES
    from glossarion_mobile.ui.tools.hub import HUB_GROUPS

    assert BOOK_TABS == ("overview", "chapters", "glossary", "output", "images")  # appended: old indices keep
    assert WIDE_TABS[-1] == "images"
    assert parse_route("/library/book/0123456789ab?tab=images").get("tab") == "images"
    assert parse_route("/tools/interactive-images").name == "tools.interactive-images"
    assert parse_route("/tools/progress/images?out=0123456789ab").name == "tools.progress.images"
    assert "tools.interactive-images" in SCREEN_ROUTES
    tiles = {t.route for _group, group_tiles in HUB_GROUPS for t in group_tiles}
    assert "tools.interactive-images" in tiles


def test_settings_schema_rows_and_the_epub_output_group():
    import settings_schema as schema

    from glossarion_mobile.ui.settings.model import CURATED_SECTIONS
    from glossarion_mobile.ui.settings.section_page import STATIC_LINKS

    expected = {
        "interactive_images_text_model": ("str", ""), "interactive_images_image_model": ("str", ""),
        "interactive_images_max_references": ("int", 5), "interactive_images_auto_after_translation": ("bool", False),
        "interactive_images_recompile_after": ("bool", True), "interactive_css_enabled": ("bool", False),
        "interactive_images_resolution": ("choice", "1K"), "interactive_images_aspect_ratio": ("choice", "3:4"),
    }
    for key, (kind, default) in expected.items():
        spec = schema.spec(key)
        assert (spec.type, spec.default) == (kind, default), key
        assert default == core.DEFAULTS[key]
    assert schema.choice_values("interactive_images_resolution") == core.RESOLUTIONS
    assert schema.choice_values("interactive_images_aspect_ratio") == core.ASPECT_RATIOS
    groups = dict(dict((s[0], s[3]) for s in CURATED_SECTIONS)["epub_output"])
    assert set(groups["Interactive images"]) == set(expected)
    assert any(link[2] == "tools.interactive-images" for link in STATIC_LINKS["epub_output"])


# ==========================================================================
# Tools › Interactive Images
# ==========================================================================


def _screen(tmp_path, *, webview=False, store=None, files=None):
    u7 = _u7()
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.tools import interactive_images as ix

    _conn, session = u7._tb()._fake_session("android")
    page = session.page
    store = {} if store is None else store
    jobs = u7.U7Jobs()
    ctx = u7._tools_ctx(page, store=store, jobs=jobs, files=files, webview_ok=lambda: webview)
    screen = ix.InteractiveImagesScreen(parse_route("/tools/interactive-images"), ctx)
    body = screen.get_body()
    u7._tools()._mount(page, body)
    screen.did_show()
    return screen, ctx, store, jobs


def test_tools_screen_profile_and_models_write_the_desktop_keys(tmp_path):
    async def scenario():
        screen, ctx, store, _jobs = _screen(tmp_path)
        try:
            assert screen.current_tab == "profile"
            assert screen.bar.names == ["Default", "Illustrated", "Styles only"]
            assert screen.per_chapter.value == "1"
            assert [o.key for o in screen.per_chapter.options] == [str(n) for n in range(0, 11)]
            assert len(screen.prompt_cards.controls) == 4
            # images per chapter is the profile's per_chapter field and its runtime key
            assert screen.set_per_chapter("3") is None
            assert store["interactive_images_per_chapter"] == "3"
            assert store["interactive_images_prompt_profile_default"]["per_chapter"] == "3"
            screen.bar.select("Styles only")
            assert store["active_interactive_images_prompt_profile"] == "Styles only"
            assert store["interactive_images_per_chapter"] == "0" and screen.per_chapter.value == "0"
            assert store["interactive_images_planner_prompt"] == core.InteractiveImagesProfiles.DEFAULT_BUILTIN["planner"]
            # models
            screen.set_value("interactive_images_max_references", "99")
            assert store["interactive_images_max_references"] == core.MAX_REFERENCES
            screen.image_model.set_value("dall-e-3")
            assert store["interactive_images_image_model"] == "dall-e-3"
            assert screen.ref_note.visible and core.NO_REFERENCE_WARNING in screen.ref_note.value
            screen.image_model.set_value("gemini-3-pro-image-preview")
            assert "14 reference images" in screen.ref_note.value
            screen.text_model.set_value("")
            assert store["interactive_images_text_model"] == ""
            screen.set_value("interactive_images_aspect_ratio", "nonsense")
            assert store["interactive_images_aspect_ratio"] == core.DEFAULTS["interactive_images_aspect_ratio"]
            screen.set_tab("styles")
            assert screen.current_tab == "styles"
        finally:
            screen.dispose()

    asyncio.run(scenario())


def test_tools_screen_references_for_a_book(tmp_path):
    folder = _book(tmp_path)
    (folder / "glossary.csv").write_text("type,raw_name,translated_name,gender,description\n"
                                         "character,미라,Mira,Female,\ncharacter,토마스,Tomas,Male,\n", encoding="utf-8")
    good = tmp_path / "picks" / "mira.png"
    good.parent.mkdir()
    good.write_bytes(_png())
    bad = tmp_path / "picks" / "fake.png"
    bad.write_bytes(b"not an image")

    async def scenario():
        from glossarion_mobile.ui.tools import targets as tg

        screen, ctx, _store, _jobs = _screen(tmp_path, files=_u7()._tools().FakeFiles(picks=[[str(good), str(bad)]]))
        try:
            screen.set_target(tg.ToolTarget(title="Lantern", folder=str(folder)))
            await screen.load_references()
            assert await screen.load_names() == 2
            refs = json.loads((folder / "Interactive_Images" / "references.json").read_text(encoding="utf-8"))
            assert sorted(refs["characters"]) == ["Mira", "Tomas"]
            added = await screen.add_images("Mira")
            assert added == ["references/mira_1.png"]
            assert any("Not a PNG" in note for note in ctx.notes)
            refs = core.load_references(str(folder))
            assert refs["characters"]["Mira"] == ["references/mira_1.png"]
            assert (folder / "Interactive_Images" / "references" / "mira_1.png").read_bytes() == good.read_bytes()
            assert core.select_references(refs, ["Mira"], 5) == [("Mira", "references/mira_1.png")]
            await screen.clear_character("Mira")
            assert "Mira" not in core.load_references(str(folder))["characters"]
            assert not (folder / "Interactive_Images" / "references" / "mira_1.png").exists()
        finally:
            screen.dispose()

    asyncio.run(scenario())


def test_tools_screen_styles_presets_native_preview_and_new_from_prompt(tmp_path):
    folder = _book(tmp_path)

    async def scenario():
        from glossarion_mobile.ui.tools import interactive_images as ix
        from glossarion_mobile.ui.tools import targets as tg

        screen, ctx, store, jobs = _screen(tmp_path)
        try:
            assert [o.key for o in screen.style_dropdown.options] == list(presets.BUILTIN_PRESET_NAMES)
            assert len(presets.BUILTIN_PRESET_NAMES) == 12
            # nothing renders before the Styles tab is on screen; then, with no WebView on a host run, the native
            # view (note, the classes the style defines, its CSS)
            assert screen.preview.renderer == "pending"
            screen.set_tab("styles")
            assert screen.preview.renderer == "native"
            screen.select_style("LitRPG Status Window")
            assert store[presets.PRESET_KEY] == "LitRPG Status Window"
            slot = screen.preview.slot.content
            texts = [c.value for c in slot.controls]
            assert texts[0] == ix.NATIVE_NOTE and "glx-status-window" in texts[1]
            assert texts[2] == presets.BUILTIN_PRESETS["LitRPG Status Window"]
            assert screen.css_field.read_only and screen.save_style_button.disabled
            name = screen.duplicate_style()
            assert name == "LitRPG Status Window copy" and store[presets.PRESET_KEY] == name
            assert name in store[presets.USER_PRESETS_KEY] and screen.css_field.visible
            screen.css_field.value = ".glx-book p { color: red; }\nbody { margin: 0; }"
            assert await screen.save_style() == name
            saved = store[presets.USER_PRESETS_KEY][name]
            assert ".glx-book {" in saved and "body" not in saved  # scoped: body became .glx-book
            # Apply to book writes the book's stylesheet (never the CSS override)
            screen.set_target(tg.ToolTarget(title="Lantern", folder=str(folder)))
            path = await screen.apply_to_book()
            assert path == presets.book_stylesheet_path(str(folder)) and "color: red" in Path(path).read_text("utf-8")
            assert "epub_css_override_path" not in store
            # New from prompt: a job with css_request; its result becomes a user preset
            ctx.extras["answers"] = ["Cozy rainy night"]
            job_id = await screen.new_from_prompt()
            spec = jobs.specs[-1]
            assert spec.kind == "interactive_images" and spec.params == {"css_request": "Cozy rainy night"}
            jobs.finish(job_id, result={"interactive_css": {"css": ".glx-dialogue { color: blue; }",
                                                            "warnings": [], "request": "Cozy rainy night"}})
            assert store[presets.PRESET_KEY] == "Cozy Rainy Night"
            assert store[presets.USER_PRESETS_KEY]["Cozy Rainy Night"].strip().startswith(".glx-dialogue")
            ctx.extras["answers"] = [True]
            assert await screen.delete_style() is True
            assert "Cozy Rainy Night" not in store[presets.USER_PRESETS_KEY]
        finally:
            screen.dispose()

    asyncio.run(scenario())


def test_preview_webview_serves_the_shared_sample_and_falls_back(tmp_path):
    pytest.importorskip("flet_webview")

    async def scenario():
        screen, _ctx, _store, _jobs = _screen(tmp_path, webview=True)
        preview = screen.preview
        try:
            assert preview.renderer == "pending" and preview.webview is None  # built when the tab shows
            screen.set_tab("styles")
            assert preview.renderer == "webview"
            url = preview.webview.url
            assert url.startswith("http://127.0.0.1:")
            with urllib.request.urlopen(url, timeout=10) as response:
                html = response.read().decode("utf-8")
                csp = response.headers.get("Content-Security-Policy", "")
            assert html == presets.preview_document(presets.BUILTIN_PRESETS[presets.DEFAULT_PRESET])
            assert "script-src 'none'" in csp and "<script" not in html
            for part in ('class="glx-status-window"', "<details>", 'class="glx-reveal"', "data:image/svg+xml"):
                assert part in html
            # a CSS edit re-publishes after the debounce
            preview.update_css(".glx-book p { color: green; }")
            await asyncio.sleep(0.5)
            assert preview.webview.url != url and preview.version == 2
            # the page reported in: kept; a page that never loads: the native view
            preview._on_loaded()
            assert await preview.check_webview(0) is False and preview.renderer == "webview"
            preview.page_ready = False
            assert await preview.check_webview(0) is True and preview.renderer == "native"
            assert preview.server is None
        finally:
            screen.dispose()

    asyncio.run(scenario())


def test_converter_shows_interactive_styles_under_the_css_override(tmp_path):
    from glossarion_mobile.ui.tools import interactive_images as ix

    async def scenario():
        u7 = _u7()
        _conn, session = u7._tb()._fake_session("android")
        store = {"epub_css_override_path": "/x/override.css"}
        ctx = u7._tools_ctx(session.page, store=store)
        controls = ix.interactive_styles_controls(ctx, key="t")
        switch = controls[0].controls[0]
        switch.value = True
        switch.on_change(types.SimpleNamespace(control=switch))
        assert store[presets.ENABLED_KEY] is True
        clear = next(b for b in controls[2].controls if getattr(b, "key", None) == "t-clear")
        clear.on_click(None)
        assert store[presets.ENABLED_KEY] is False and store["epub_css_override_path"] == "/x/override.css"
        opened = next(b for b in controls[2].controls if getattr(b, "key", None) == "t-open")
        opened.on_click(None)
        assert ctx.navigated[-1] == ("tools.interactive-images", None, {"tab": "styles"})

    asyncio.run(scenario())


# ==========================================================================
# Book page › Images (the tab over the shared progress core)
# ==========================================================================


class FakeBookPage:
    def __init__(self, ctx, service, folder) -> None:
        self.ctx = ctx
        self.service = service
        self.bid = "0123456789ab"
        self.book = {"name": "Lantern", "output_folder": str(folder)}
        self.workspace = str(folder)
        self.job_view = None
        self.images = None
        self.reloads = 0
        self.readers: list = []

    async def reload_images(self, *, force=False):
        from glossarion_mobile.ui.library import progress_model as pm

        self.reloads += 1
        self.images = await self.ctx.io(lambda: pm.load_images_view(self.service, self.book, previous=self.images))
        self.images_tab.apply(self.images)
        return self.images

    def open_reader(self, **kwargs):
        self.readers.append(kwargs)
        return "reader"

    def open_files(self):
        pass


def test_images_tab_rows_chips_actions_and_build(tmp_path, fake_models):
    folder = _book(tmp_path, chapters=3)
    _run_job(folder, dict(CONFIG), keys=["response_001_Chapter 1.html"])

    async def scenario():
        from glossarion_mobile.ui.library.images_tab import ImagesTab

        u7 = _u7()
        _conn, session = u7._tb()._fake_session("android")
        service = FakeService(folder)
        ctx = u7._tools_ctx(session.page, service=service)
        page = FakeBookPage(ctx, service, folder)
        tab = ImagesTab(page)
        page.images_tab = tab
        u7._tools()._mount(session.page, tab.build())
        await page.reload_images()
        assert [r.status for r in tab.visible_rows()] == ["completed", "not_built", "not_built"]
        chips = {c.key: c for c in tab.stats_row.controls}
        assert set(chips) == {f"ix-chip-{g}" for g, _t, _c in ipc.STAT_CHIPS}
        tab.set_filter("not_built")
        assert [r.key for r in tab.visible_rows()] == ["response_002_Chapter 2.html", "response_003_Chapter 3.html"]
        tab.set_filter(None)
        # the row sheet: the desktop context menu's actions and rules
        sheet = tab.open_row_sheet(tab.view.rows[1])
        labels = {item.key: (item.label, item.disabled_reason) for item in sheet.items}
        assert labels["ix-act-build"][0] == "🖼 Build / Rebuild (1)" and labels["ix-act-build"][1] is None
        assert labels["ix-act-retry"][1] and labels["ix-act-delete"][1] and labels["ix-act-open_images"][1]
        sheet = tab.open_row_sheet(tab.view.rows[0])
        assert all(item.disabled_reason is None for item in sheet.items
                   if item.key in ("ix-act-delete", "ix-act-open_images", "ix-act-show_plan"))
        # Build pending / Build selected: interactive_images jobs on the workspace
        assert tab.build_reason() is None
        await tab.build_pending()
        await tab.run_action("build", [tab.view.rows[2]])
        assert [(s.kind, s.inputs, s.params) for s in service.specs] == [
            ("interactive_images", (str(folder),), {}),
            ("interactive_images", (str(folder),), {"keys": ["response_003_Chapter 3.html"]})]
        assert service.specs[0].title == "Lantern"
        # Open chapter: the Reader on that file; Mark for rebuild writes the shared progress file
        await tab.run_action("open_chapter", [tab.view.rows[0]])
        assert page.readers == [{"chapter_filename": "response_001_Chapter 1.html", "mode": "translated"}]
        await tab.run_action("mark", [tab.view.rows[0]])
        assert tab.view.rows[0].status == "pending" and page.reloads >= 2
        # selection mode: the bulk bar's primary actions
        tab.on_row_long_press(tab.view.rows[0])
        tab.toggle_select(tab.view.rows[1].key)
        primary, more = tab.bulk_actions(tab.selected_rows())
        assert [a.id for a in primary] == ["build", "retry", "delete"]
        assert [a.id for a in more] == ["mark", "open_images", "open_chapter", "show_plan"]
        tab.exit_selection()
        assert not tab.selecting
        # no job service: Build pending says why
        service.kinds.clear()
        assert tab.sync_build_button() == "The job service is not running" and tab.build_button.disabled

    asyncio.run(scenario())


def test_images_view_without_a_workspace_says_so(tmp_path):
    from glossarion_mobile.ui.library import progress_model as pm

    service = FakeService(tmp_path / "nowhere")
    view = pm.load_images_view(service, {"name": "x"})
    assert view.error == pm.IMAGES_NO_WORKSPACE and view.empty and not view.rows
    assert pm.images_action_reason(service, view, "build", []) == pm.IMAGES_NO_WORKSPACE
