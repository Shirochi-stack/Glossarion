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


def _run_job(folder: Path, config: dict, keys=None, stop=False, owner_fields=None):
    from glossarion_mobile import job_kinds

    u7 = _u7()
    events: list = []
    ctx = u7.FakeCtx(types.SimpleNamespace(config=config, **(owner_fields or {})),
                     params={"keys": keys} if keys is not None else {},
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
        "interactive_images_include_gender": ("bool", True), "interactive_images_subjects": ("choice", "characters"),
    }
    for key, (kind, default) in expected.items():
        spec = schema.spec(key)
        assert (spec.type, spec.default) == (kind, default), key
        assert default == core.DEFAULTS[key]
    assert schema.choice_values("interactive_images_resolution") == core.RESOLUTIONS
    assert schema.choice_values("interactive_images_aspect_ratio") == core.ASPECT_RATIOS
    assert schema.choice_values("interactive_images_subjects") == core.SUBJECTS
    # the Auto ratio checklist: a list stored by Tools > Interactive Images (read-only tile), default from the core
    choices = schema.spec("interactive_images_aspect_choices")
    assert choices.type == "list" and schema.effective_default("interactive_images_aspect_choices") == \
        core.DEFAULTS["interactive_images_aspect_choices"]
    groups = dict(dict((s[0], s[3]) for s in CURATED_SECTIONS)["epub_output"])
    assert set(groups["Interactive images"]) == set(expected) | {"interactive_images_aspect_choices"}
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
            assert len(presets.BUILTIN_PRESET_NAMES) == 22      # 12 + the fantasy / leveling-system presets
            assert {"Arcane Runes", "Level Up Burst", "Dungeon System", "Guild Card"} <= {
                o.key for o in screen.style_dropdown.options}
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


# ==========================================================================
# Round 2: gender / subjects / Auto ratio / placeholder texts / gallery / routing to the image key pool
# ==========================================================================


def test_tools_screen_gender_subjects_auto_ratio_and_placeholder_texts(tmp_path):
    async def scenario():
        screen, ctx, store, _jobs = _screen(tmp_path)
        try:
            # gender: default On, a switch of the desktop's key; the subjects dropdown lists the shared values
            assert screen.gender_switch.value is True
            screen.set_value("interactive_images_include_gender", False)
            assert store["interactive_images_include_gender"] is False
            assert [o.key for o in screen.subjects.options] == list(core.SUBJECTS) and screen.subjects.value == "characters"
            screen.set_value("interactive_images_subjects", "both")
            assert store["interactive_images_subjects"] == "both"
            screen.set_value("interactive_images_subjects", "nonsense")
            assert store["interactive_images_subjects"] == "characters"
            # the Auto ratio chips show only while the ratio is Auto, and tick / untick the pool
            assert "auto" in [o.key for o in screen.aspect.options] and "21:9" in [o.key for o in screen.aspect.options]
            assert screen.ratio_row.visible is False
            screen.set_value("interactive_images_aspect_ratio", "auto")
            assert screen.ratio_row.visible is True and screen.ratio_hint.visible is True
            chips = {c.key: c for c in screen.ratio_row.controls}
            assert set(chips) == {f"ix-ratio-{r}" for r in core.ASPECT_CHOICES}
            assert [k for k, c in chips.items() if c.selected] == [f"ix-ratio-{r}" for r in ("1:1", "3:4", "4:3", "16:9")]
            assert screen.toggle_ratio("21:9") == ["1:1", "3:4", "4:3", "16:9", "21:9"]
            assert store["interactive_images_aspect_choices"] == ["1:1", "3:4", "4:3", "16:9", "21:9"]
            screen.toggle_ratio("1:1")
            screen.toggle_ratio("3:4")
            assert store["interactive_images_aspect_choices"] == ["4:3", "16:9", "21:9"]
            assert {c.key: c.selected for c in screen.ratio_row.controls}["ix-ratio-21:9"] is True
            settings = core.settings_from_config(store)
            assert settings.aspect_ratio == "auto" and settings.aspect_choices == ("4:3", "16:9", "21:9")
            screen.set_value("interactive_images_aspect_ratio", "16:9")
            assert screen.ratio_row.visible is False
            # placeholder texts: every one has a row; Edit -> the shared setter; Reset restores the default
            names = [item["name"] for item in core.placeholder_text_defs()]
            assert [c.key for c in screen.placeholder_rows.controls] == [f"ix-ph-{n}" for n in names]
            assert screen.set_placeholder("character_gender", "{name} [{gender}]") is None
            assert store[core.PLACEHOLDER_TEXTS_KEY] == {"character_gender": "{name} [{gender}]"}
            assert core.format_character("Mira", "female", core.settings_from_config(store)) == "Mira"   # gender is Off
            screen.set_value("interactive_images_include_gender", True)
            assert core.format_character("Mira", "female", core.settings_from_config(store)) == "Mira [female]"
            assert screen.set_placeholder("reference_line", "Ref {n}: {name}") is None
            assert store["interactive_images_reference_line"] == "Ref {n}: {name}"
            assert screen.set_placeholder("character_gender", core.placeholder_default("character_gender")) is None
            assert store[core.PLACEHOLDER_TEXTS_KEY] == {}
            assert screen.edit_placeholder("gender_male") is None      # the host test ctx has no Settings context
            assert any("Settings context" in note for note in ctx.notes)
        finally:
            screen.dispose()

    asyncio.run(scenario())


def _gallery_book(tmp_path: Path) -> Path:
    folder = tmp_path / "Output" / "Gallery"
    folder.mkdir(parents=True)
    images = folder / core.INTERACTIVE_FOLDER
    images.mkdir()
    chapters, entries = {}, {}
    for n, who in ((1, ["Mira"]), (2, ["Mira", "Tomas"]), (3, [])):
        name = f"response_{n:03d}_Chapter {n}.html"
        image = f"Glossarion_ch{n:03d}_1.png"
        html = f"<html><body><p>Chapter {n} text.</p>{core.figure_html(core.image_src(image))}</body></html>"
        (folder / name).write_text(html, encoding="utf-8")
        (images / image).write_bytes(_png())
        chapters[str(n)] = {"output_file": name, "status": "completed", "actual_num": n}
        entries[name] = {"chapter": n, "status": "completed", "source_hash": core.visible_text_hash(html),
                         "images": [{"file": image, "status": "completed", "characters": who, "scene": who,
                                     "prompt": f"scene {n}"}], "classes": []}
    (folder / "translation_progress.json").write_text(json.dumps({"chapters": chapters}), encoding="utf-8")
    (folder / "glossary.csv").write_text("type,raw_name,translated_name\ncharacter,미라,Mira\n"
                                         "character,토마스,Tomas\ncharacter,하나,Hana\n", encoding="utf-8")
    for key, entry in entries.items():
        ipc.set_entry(core.progress_path(str(folder)), key, entry)
    return folder


def _gallery(tmp_path, **kwargs):
    from glossarion_mobile.ui.library.images_gallery import ImagesGallery

    u7 = _u7()
    _conn, session = u7._tb()._fake_session("android")
    ctx = u7._tools_ctx(session.page)
    ctx._keepalive = (_conn, session)   # the page only holds a weak reference to its session
    folder = _gallery_book(tmp_path)
    return ImagesGallery(ctx, str(folder), **kwargs), ctx, folder


def test_gallery_cards_filters_long_press_selection_and_multi_delete(tmp_path):
    async def scenario():
        from glossarion_mobile.ui.library.images_gallery import ALL_CHAPTERS, ALL_CHARACTERS

        changed: list = []
        gallery, ctx, folder = _gallery(tmp_path, on_changed=lambda: changed.append(1))
        await gallery.load()
        assert len(gallery.grid.controls) == 3 and gallery.count_text.value == "3 of 3 images"
        assert [o.text for o in gallery.character_dd.options] == [ALL_CHARACTERS, "Hana", "Mira", "Tomas"]
        assert [o.text for o in gallery.chapter_dd.options][0] == ALL_CHAPTERS and len(gallery.chapter_dd.options) == 4
        assert [i.caption for i in gallery.visible()] == ["Ch. 1 · Mira", "Ch. 2 · Mira, Tomas", "Ch. 3"]
        gallery.set_character("Tomas")
        assert [i.file for i in gallery.visible()] == ["Glossarion_ch002_1.png"] and len(gallery.grid.controls) == 1
        gallery.set_character("*")
        gallery.set_chapter("response_003_Chapter 3.html")
        assert len(gallery.visible()) == 1
        gallery.set_chapter("*")
        # a tap outside a selection opens the image; a long press starts selecting, then taps add / remove
        viewer = gallery.tap(gallery.images[0].id)
        assert viewer is not None and not gallery.selecting
        gallery.long_press(gallery.images[0].id)
        assert gallery.selecting and gallery.selected == {gallery.images[0].id}
        assert gallery.selection_bar.visible and gallery.select_text.value == "1 selected"
        gallery.tap(gallery.images[2].id)
        assert len(gallery.selected) == 2 and gallery.delete_button.disabled is False
        gallery.tap(gallery.images[0].id)
        gallery.tap(gallery.images[0].id)
        assert len(gallery.selected) == 2
        gallery.select_all()
        assert len(gallery.selected) == 3
        gallery.exit_selection()
        assert not gallery.selecting and not gallery.selected and not gallery.selection_bar.visible
        # Delete selected: confirmed, shared text-checked delete, files + figures + records, the view reloads
        gallery.long_press(gallery.images[0].id)
        gallery.tap(gallery.images[2].id)
        ctx.extras["answers"] = [False]
        assert await gallery.delete_selected() == 0
        assert (folder / core.INTERACTIVE_FOLDER / "Glossarion_ch001_1.png").exists()
        ctx.extras["answers"] = [True]
        assert await gallery.delete_selected() == 2
        assert changed == [1] and len(gallery.images) == 1 and len(gallery.grid.controls) == 1
        assert not gallery.selecting and gallery.images[0].file == "Glossarion_ch002_1.png"
        assert not (folder / core.INTERACTIVE_FOLDER / "Glossarion_ch001_1.png").exists()
        assert "data-glx" not in (folder / "response_001_Chapter 1.html").read_text(encoding="utf-8")
        assert "data-glx" in (folder / "response_002_Chapter 2.html").read_text(encoding="utf-8")
        # single image from the viewer: Go to chapter and Delete
        opened: list = []
        gallery.open_chapter = lambda image: opened.append(image.key)
        viewer = gallery.open_viewer(gallery.images[0])
        gallery.go_to_chapter(gallery.images[0])
        assert opened == ["response_002_Chapter 2.html"]
        ctx.extras["answers"] = [True]
        assert await viewer.delete() == 1 and gallery.images == []
        assert gallery.empty_text.visible and "No images" in gallery.empty_text.value

    asyncio.run(scenario())


def test_images_tab_opens_the_gallery_and_a_delete_reloads_the_rows(tmp_path, fake_models):
    folder = _book(tmp_path, chapters=2)
    _run_job(folder, dict(CONFIG))

    async def scenario():
        from glossarion_mobile.ui.library.images_gallery import ImagesGallery
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
        assert any(getattr(item, "key", "") == "ix-gallery" for item in tab.more_menu.items)
        gallery = tab.open_gallery()
        assert isinstance(gallery, ImagesGallery) and gallery.folder == str(folder)
        await gallery.load()
        assert len(gallery.images) == 2
        gallery.open_chapter(gallery.images[0])
        assert page.readers[-1] == {"chapter_filename": "response_001_Chapter 1.html", "mode": "translated"}
        before = page.reloads
        ctx.extras["answers"] = [True]
        assert await gallery.delete_images([gallery.images[0]]) == 1
        await asyncio.sleep(0)
        await asyncio.sleep(0.05)
        assert page.reloads > before

    asyncio.run(scenario())


def test_tools_screen_gallery_button_needs_a_book(tmp_path):
    folder = _gallery_book(tmp_path)

    async def scenario():
        from glossarion_mobile.ui.tools import targets as tg

        screen, ctx, _store, _jobs = _screen(tmp_path)
        try:
            assert screen.gallery_button.disabled is True and screen.open_gallery() is None
            screen.set_target(tg.ToolTarget(title="Gallery", folder=str(folder), bid="0123456789ab"))
            assert screen.gallery_button.disabled is False
            gallery = screen.open_gallery()
            await gallery.load()
            assert len(gallery.images) == 3
            gallery.open_chapter(gallery.images[0])
            assert ctx.navigated[-1][0] == "library.book" and ctx.navigated[-1][2] == {"tab": "images"}
        finally:
            screen.dispose()

    asyncio.run(scenario())


# ---- the job path: images use the Image Gen / Edit key pool (when on), text never does ------------------


@pytest.fixture
def routed(monkeypatch):
    """The real ``UnifiedClient`` behind the runner's own client factory, its provider call replaced by a recorder."""
    from unified_api_client import UnifiedClient

    for name in ("USE_INPAINTER_KEYS", "INPAINTER_API_KEYS", "ENABLE_IMAGE_OUTPUT_MODE", "USE_MULTI_API_KEYS",
                 "USE_MULTI_KEYS", "BATCH_TRANSLATION"):
        monkeypatch.delenv(name, raising=False)
    for attr in ("_in_memory_inpainter_keys", "_inpainter_key_pool"):
        monkeypatch.setattr(UnifiedClient, attr, None)
    for name, value in (("BATCH_TRANSLATION", "0"), ("THREAD_SUBMISSION_DELAY_SECONDS", "0"),
                        ("SEND_INTERVAL_SECONDS", "0"), ("USE_MULTI_API_KEYS", "0"), ("USE_MULTI_KEYS", "0")):
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(UnifiedClient, "_setup_client", lambda self: setattr(self, "client_type", "test"))
    calls: list = []

    def fake_send_internal(self, messages, *args, **kwargs):
        from glossarion_mobile.diagnostics.fake_llm_server import planner_reply

        image = bool(getattr(self, "_force_image_output_mode", False))
        calls.append({"api_key": self.api_key, "image": image, "text_only": bool(getattr(self, "_text_output_only", False))})
        if image:
            self._get_thread_local_client()._last_generated_image_bytes = _png()
            return "", "stop"
        return planner_reply(messages[-1]["content"]), "stop"

    monkeypatch.setattr(UnifiedClient, "_send_internal", fake_send_internal)
    monkeypatch.setattr(UnifiedClient, "reset_cleanup_state", lambda self, *a, **k: None)
    monkeypatch.setattr(UnifiedClient, "_log_pre_stagger", lambda self, *a: None)
    monkeypatch.setattr(UnifiedClient, "_apply_thread_submission_delay", lambda self: None)
    monkeypatch.setattr(UnifiedClient, "_should_abort_retry", lambda self: False)
    monkeypatch.setattr(UnifiedClient, "_extract_chapter_label", lambda self, _m: None)
    import unified_api_client as uac

    monkeypatch.setattr(uac, "_api_watchdog_started", lambda *a, **k: None)
    monkeypatch.setattr(uac, "_api_watchdog_finished", lambda *a, **k: None)
    yield calls
    UnifiedClient.clear_in_memory_inpainter_keys()


POOL = [{"api_key": "img-key-a", "model": "gemini-3-pro-image-preview", "enabled": True}]


def test_job_path_image_requests_use_the_image_pool_and_text_requests_never_do(tmp_path, routed):
    folder = _book(tmp_path, chapters=1)
    config = {"model": "fake-text-model", "api_key": "main-key", "interactive_images_image_model": "gpt-image-1",
              "use_inpainter_keys": True, "inpainter_keys": POOL}
    ctx, result, _events = _run_job(folder, config)
    assert result == {"ok": True, "outputs": []}, ctx.logs
    images = [c for c in routed if c["image"]]
    texts = [c for c in routed if not c["image"]]
    assert images and all(c["api_key"] == "img-key-a" for c in images)       # the pool key, not the main key
    assert texts and all(c["api_key"] == "main-key" and c["text_only"] for c in texts)


def test_job_path_with_the_pool_off_every_request_uses_the_normal_key(tmp_path, routed):
    folder = _book(tmp_path, chapters=1)
    config = {"model": "fake-text-model", "api_key": "main-key", "interactive_images_image_model": "gpt-image-1",
              "use_inpainter_keys": False, "inpainter_keys": POOL}
    ctx, result, _events = _run_job(folder, config)
    assert result == {"ok": True, "outputs": []}, ctx.logs
    assert routed and {c["api_key"] for c in routed} == {"main-key"}


def test_job_path_carries_gender_subjects_and_the_auto_ratio_to_the_requests(tmp_path, routed, monkeypatch):
    folder = _book(tmp_path, chapters=1)
    (folder / "glossary.csv").write_text("type,raw_name,translated_name,gender\ncharacter,미라,Mira,Female\n"
                                         "character,토마스,Tomas,Male\n", encoding="utf-8")
    seen: list = []
    from unified_api_client import UnifiedClient

    original = UnifiedClient._send_internal

    def spy(self, messages, *args, **kwargs):
        seen.append({"image": bool(getattr(self, "_force_image_output_mode", False)),
                     "ratio": getattr(self, "_forced_image_output_aspect_ratio", None),
                     "text": messages[-1]["content"]})
        return original(self, messages, *args, **kwargs)

    monkeypatch.setattr(UnifiedClient, "_send_internal", spy)
    config = {"model": "fake-text-model", "api_key": "main-key", "interactive_images_image_model": "gpt-image-1",
              "interactive_images_aspect_ratio": "auto", "interactive_images_aspect_choices": ["21:9"],
              "interactive_images_subjects": "both"}
    ctx, result, _events = _run_job(folder, config)
    assert result == {"ok": True, "outputs": []}, ctx.logs
    planner = [c for c in seen if not c["image"]][0]["text"]
    assert "Mira (female)" in planner and "Tomas (male)" in planner
    assert core.placeholder_default("image_subjects_both") in planner
    images = [c for c in seen if c["image"]]
    assert images and {c["ratio"] for c in images} == {"21:9"}


# ---- the job path in batch mode: the translation's Batch Translation / batch size run chapters in parallel -----


def _latency_models(monkeypatch, hold=0.12):
    """Fake clients (the planner / image answers of ``FakeClient``) that take ``hold`` seconds and record overlap."""
    import threading
    import time

    state = {"now": 0, "peak": 0, "sent": [], "threads": set()}
    lock = threading.Lock()

    class SlowClient(FakeClient):
        def send(self, messages, temperature=None, max_tokens=None, context=None):
            with lock:
                state["now"] += 1
                state["peak"] = max(state["peak"], state["now"])
                state["threads"].add(threading.current_thread().name)
            try:
                time.sleep(hold)
                return FakeClient.send(self, messages, temperature, max_tokens, context)
            finally:
                with lock:
                    state["now"] -= 1

    monkeypatch.setattr(runner, "_default_client_factory",
                        lambda kind, model, config, output_dir: SlowClient(kind, state["sent"]))
    monkeypatch.setattr(runner, "configure_image_key_pool", lambda config, log: None)
    return state


def _owner_fields(batch_on: bool, size):
    return {"batch_translation_var": batch_on, "batch_size_var": str(size),
            "delay_entry": types.SimpleNamespace(text=lambda: "0"),
            "thread_delay_entry": types.SimpleNamespace(text=lambda: "0")}


def test_job_path_with_batch_on_builds_chapters_in_parallel_and_counts_progress(tmp_path, monkeypatch):
    state = _latency_models(monkeypatch)
    folder = _book(tmp_path, chapters=6)
    ctx, result, events = _run_job(folder, dict(CONFIG), owner_fields=_owner_fields(True, 3))
    assert result == {"ok": True, "outputs": []}, ctx.logs
    assert ctx.results["interactive_images"]["completed"] == 6 and ctx.results["interactive_images"]["failed"] == 0
    assert 2 <= state["peak"] <= 3                                        # parallel, within the batch size
    assert len(state["threads"]) > 1
    assert sorted(kind for kind, _c, _m in state["sent"]) == ["image"] * 6 + ["text"] * 6
    progress = [data for kind, data in events if kind == "progress"]
    assert progress[0]["completed"] == 0 and progress[-1]["completed"] == 6   # each chapter counts, capped at the total
    assert all(p["completed"] <= p["total"] for p in progress)
    for n in range(1, 7):
        html = (folder / f"response_{n:03d}_Chapter {n}.html").read_text(encoding="utf-8")
        assert f'src="Interactive_Images/Glossarion_ch{n:03d}_1.png"' in html
        assert core.check_text_unchanged(CH.format(n=n), html)[0]
    assert any(line.startswith("[ii-worker_") for line in ctx.logs)
    assert os.environ.get("BATCH_TRANSLATION") != "1"


def test_job_path_with_batch_off_stays_one_request_at_a_time(tmp_path, monkeypatch):
    state = _latency_models(monkeypatch, hold=0.03)
    folder = _book(tmp_path, chapters=3)
    ctx, result, _events = _run_job(folder, dict(CONFIG), owner_fields=_owner_fields(False, 3))
    assert result == {"ok": True, "outputs": []}, ctx.logs
    assert state["peak"] == 1
    assert [kind for kind, _c, _m in state["sent"]] == ["text", "image"] * 3
    assert not any(line.startswith("[ii-worker_") for line in ctx.logs)


def test_job_path_with_batch_on_honours_stop(tmp_path, monkeypatch):
    state = _latency_models(monkeypatch, hold=0.03)
    folder = _book(tmp_path, chapters=6)
    ctx, result, _events = _run_job(folder, dict(CONFIG), stop=True, owner_fields=_owner_fields(True, 3))
    assert result == {"ok": None, "outputs": []} and ctx.results["interactive_images"]["stopped"] is True
    assert state["sent"] == [] and not list((folder / "Interactive_Images").glob("Glossarion_*.png"))