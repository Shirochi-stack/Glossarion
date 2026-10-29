"""Owner 2026-10-16 (phone ~412 dp wide):

1. "The welcome screen page 2 should be merged with page 1, just use a smaller font and better UI
   arrangement." Step 2 ("Other providers": Choose another model, API key + Save, Ollama / LM Studio)
   is part of step 1 now; there is no second page or dot for it, and step 1 fits a 360-412 dp phone
   at 100 % text without scrolling (it still scrolls at large text).
2. "Users shouldn't need to scroll to the right to see other tools. Make this section more compact by
   using a smaller font size." The drawer's destination row (Library, Jobs, Glossaries, Tools) never
   scrolls sideways: one row of equal cells (icon over a LABEL_SMALL label) that fits a 288-330 dp
   drawer (360-412 dp phones), each cell >= 40 dp tall.

3. (owner 2026-10-17) Page 1 gains "Run on this phone (no account needed)": the on-device presets as one
   row of one-tap cells (size + RAM hint, Light suggested, inline progress) through the Model manager's
   OnDeviceFlow; "authchatgpt/ and authnd/: no sign-in" went (the section title says it), and the page
   still fits 360 x 800 and 412 x 915 at 100 % text with the engine present.

The layout math is an estimate of Flutter's: Roboto advances (lowercase ~0.52 em, capitals ~0.66 em,
both above Roboto's real averages), the theme's TYPE_SCALE sizes and line heights, Material's 48 dp tap
target for buttons and the M3 dense text field.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
for entry in (str(MOBILE_DIR / "app"), str(MOBILE_DIR.parent)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

ft = pytest.importorskip("flet")

from glossarion_mobile.state.app_state import AppState  # noqa: E402
from glossarion_mobile.ui import tokens  # noqa: E402
from glossarion_mobile.ui.responsive import drawer_width  # noqa: E402
from glossarion_mobile.ui.router import parse_route  # noqa: E402
from glossarion_mobile.ui.screens.welcome import KEYLESS, ONDEVICE_LABELS, SIGN_INS, SMALL_LABELS, WelcomeScreen  # noqa: E402
from glossarion_mobile.ui.screens.welcome_flow import STEPS, WelcomeFlow  # noqa: E402
from glossarion_mobile.ui.shell import drawer as drawer_mod  # noqa: E402
from glossarion_mobile.ui.shell.drawer import DESTINATIONS, ChatDrawer  # noqa: E402

PHONES = (360, 412)  # dp widths
#: phone heights (dp) for those widths: a 360 x 800 phone and the owner's 412 x 915
HEIGHTS = {360: 800, 412: 915}
HIT = tokens.SIZES["hit_target"]


def walk(root):
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        for name in ("controls", "content"):
            child = getattr(node, name, None)
            if isinstance(child, list):
                stack.extend(reversed([c for c in child if isinstance(c, ft.Control)]))
            elif isinstance(child, ft.Control):
                stack.append(child)


def by_key(root) -> dict:
    return {getattr(c, "key", None): c for c in walk(root) if getattr(c, "key", None)}


def text_width(text: str, size: float) -> float:
    return sum((0.66 if ch.isupper() else 0.30 if ch == " " else 0.52) * size for ch in str(text))


def style_of(control) -> tokens.TypeStyle:
    name = getattr(getattr(control, "theme_style", None), "name", None) or "BODY_MEDIUM"
    return tokens.TYPE_SCALE[name.lower()]


def text_height(control, width: float, scale: float = 1.0) -> float:
    style = style_of(control)
    lines = max(1, math.ceil(text_width(control.value, style.size * scale) / max(1.0, width)))
    if control.max_lines:
        lines = min(lines, control.max_lines)
    return lines * style.line * scale


# ==========================================================================
# 1. Welcome: one page for sign-in + other providers
# ==========================================================================


def screen(**kw) -> WelcomeScreen:
    s = WelcomeScreen(parse_route("/welcome"), oauth=object(), navigate=lambda r: None, **kw)
    s.get_body()
    return s


def test_steps_have_no_separate_providers_page():
    assert "providers" not in STEPS and STEPS == ("sign_in", "language_glossary", "permissions", "done")
    flow = WelcomeFlow()
    assert flow.next() == "language_glossary" and flow.skipped_sign_in
    s = screen()
    assert len(s.dots.controls) == len(STEPS) == 4
    with pytest.raises(ValueError):
        WelcomeFlow().go("providers")


def test_page_one_has_every_former_page_two_action(monkeypatch):
    chosen, keys_saved, routes = [], [], []
    s = WelcomeScreen(parse_route("/welcome"), oauth=object(), navigate=routes.append,
                      on_set_model=chosen.append, on_api_key=keys_saved.append)
    s.get_body()
    rows = by_key(s.page_area.content)
    for provider, _label in SIGN_INS:
        if provider != "authds":  # DeepSeek needs the in-app browser (a ReasonChip without it)
            assert f"welcome-signin-{provider}" in rows
    for provider, _label in KEYLESS:
        assert f"welcome-keyless-{provider}" in rows
    assert s.api_key_field in list(walk(s.page_area.content))
    s.api_key_field.value = "sk-test"
    rows["welcome-save-key"].on_click(None)
    # owner 2026-10-21: a Model field (the desktop's model dropdown) above the API key; a pick sets the model
    keys = [getattr(c, "key", None) for c in walk(s.page_area.content)]
    assert keys.index("welcome-model") < keys.index("welcome-api-key")
    assert rows["welcome-model"].label == "Model" and rows["welcome-model"] is s.model_picker
    s.model_picker.set_value("openai/gpt-5.5")
    rows["welcome-local"].on_click(None)
    assert keys_saved == ["sk-test"] and chosen == ["openai/gpt-5.5"] and routes == ["settings.endpoints"]
    assert s.model_notes.get("openai") == "✓ Using openai/gpt-5.5"
    # smaller type: no headline on page 1 (it was HEADLINE_SMALL 22 sp over a 96 dp logo)
    texts = [c for c in walk(s.page_area.content) if isinstance(c, ft.Text)]
    assert not any(getattr(t, "theme_style", None) == ft.TextThemeStyle.HEADLINE_SMALL for t in texts)
    images = [c for c in walk(s.page_area.content) if isinstance(c, ft.Image)]
    assert images and all(i.height <= 64 for i in images)


def _grid_height(child, scale: float = 1.0) -> float:
    """A ResponsiveRow of cells: rows of 48 dp (xs=6: two per row on a phone, xs=3: four)."""
    cells = len(child.controls)
    per_row = max(1, 12 // int((child.controls[0].col or {}).get("xs", 6))) if cells else 1
    rows = math.ceil(cells / per_row)
    return rows * HIT * max(1.0, scale * 0.9) + (rows - 1) * (child.run_spacing or 0)


def _page_height(s: WelcomeScreen, width: float, scale: float = 1.0) -> float:
    """Estimated height of step 1 (its top-level children plus the column spacing)."""
    column = s.page_area.content
    total = 0.0
    for child in column.controls:
        if child.key == "welcome-ondevice-box":  # "Run on this phone": the preset cells (+ a note line)
            child = child.content
            if isinstance(child, ft.Column):
                # hidden controls take no room; the GPU switch / Q4_0 button (OpenCL) take one tap-target row
                shown = [c for c in child.controls if c.visible is not False]
                total += sum(_grid_height(c, scale) if isinstance(c, ft.ResponsiveRow)
                             else text_height(c, width, scale) if isinstance(c, ft.Text)
                             else HIT * max(1.0, scale * 0.9) for c in shown) + (len(shown) - 1) * child.spacing
                continue
        if isinstance(child, ft.Text):
            total += text_height(child, width, scale)
        elif isinstance(child, ft.ResponsiveRow):
            total += _grid_height(child, scale)
        elif child.key == "welcome-header":
            logo = child.controls[0].height
            text_w = width - logo - child.spacing
            title, tagline = child.controls[1].controls
            total += max(logo, text_height(title, text_w, scale) + 2 + text_height(tagline, text_w, scale))
        elif isinstance(child, ft.Row):
            total += HIT * max(1.0, scale * 0.9)
        else:
            raise AssertionError(f"unexpected step-1 child {type(child).__name__}")
    return total + (len(column.controls) - 1) * column.spacing


@pytest.mark.parametrize("width", PHONES)
@pytest.mark.parametrize("with_engine", [False, True])
def test_page_one_fits_the_phone_without_scrolling(width, with_engine, tmp_path):
    s = screen(**(_engine_kwargs(tmp_path)[0] if with_engine else {}))
    if with_engine:
        assert by_key(s.page_area.content)["welcome-ondevice"].controls[0].col == {"xs": 3}
    content_w = width - 32  # the Welcome body's 16 dp padding
    # chrome: status bar 24, app bar 56, 3-button navigation 48, body padding 32, dots 8, the
    # Skip/Back/Next row 48 and the body column's 2 x 10 spacing
    budget = HEIGHTS[width] - (24 + 56 + 48 + 32 + 8 + 48 + 20)
    height = _page_height(s, content_w)
    assert height <= budget, (width, height, budget)
    # the cells are half the row: the longest short label fits its button (8 dp padding each side)
    cell = (content_w - 8) / 2
    for provider, label in SIGN_INS + KEYLESS:
        short = label.replace("Sign in with ", "")
        size = tokens.TYPE_SCALE["label_small" if provider in SMALL_LABELS else "label_large"].size
        assert text_width(short, size) + 16 + 2 <= cell, (short, cell)
    # the on-device cells are a quarter of the row: name and size line fit (6 dp padding each side)
    quarter = (content_w - 3 * 6) / 4 - 12
    if with_engine:
        for cell in by_key(s.page_area.content)["welcome-ondevice"].controls:
            name, detail = cell.content.controls[:2]
            assert text_width(name.value, tokens.TYPE_SCALE["label_medium"].size) <= quarter, (name.value, quarter)
            assert text_width(detail.value, tokens.TYPE_SCALE["label_small"].size) <= quarter, (detail.value, quarter)
        # the other states' lines too, with room to spare: the estimate runs ~10 % short on a device
        for label in ONDEVICE_LABELS.values():
            assert text_width(label, tokens.TYPE_SCALE["label_small"].size) * 1.15 <= quarter, (label, quarter)
    # at 200 % text the page may scroll (it keeps ScrollMode.AUTO) rather than clip
    assert s.page_area.content.scroll == ft.ScrollMode.AUTO


def test_page_one_signed_in_button_keeps_its_half_row():
    # v9.15.7 hotfix 4 (owner: the button "disappeared" into a full-width row): signed in, it keeps its cell
    s = screen()
    s.provider_status["authgem"] = {"email": "gem@example.com"}
    s.render()
    grid = by_key(s.page_area.content)["welcome-sign-ins"]
    cols = {c.content.key: c.col for c in grid.controls if getattr(c.content, "key", None)}
    assert cols["welcome-signed-authgem"] == {"xs": 6}
    assert all(v == {"xs": 6} for v in cols.values())


# ==========================================================================
# 2. Drawer destinations: no sideways scroll
# ==========================================================================


def test_drawer_destinations_never_scroll_and_fit_360_and_412():
    drawer = ChatDrawer(state=AppState(), clock=lambda: 200.0)
    row = by_key(drawer.content)["drawer-destinations"]
    assert isinstance(row, ft.Row) and row.scroll is None and not row.wrap
    cells = row.controls
    assert [c.key for c in cells] == [f"dest-{d[0]}" for d in DESTINATIONS]
    assert all(c.expand for c in cells)  # equal shares of the row: the row is never wider than the drawer
    for width in PHONES:
        inner = drawer_width(width) - 2 * drawer_mod.DEST_ROW_PADDING - (len(cells) - 1) * drawer_mod.DEST_SPACING
        per_cell = inner / len(cells)
        for cell in cells:
            icon, label = cell.content.controls
            assert label.theme_style == ft.TextThemeStyle.LABEL_SMALL and label.max_lines == 1
            assert label.overflow == ft.TextOverflow.ELLIPSIS  # 200 % text: an ellipsis, never a scroll
            need = max(icon.size, text_width(label.value, tokens.TYPE_SCALE["label_small"].size)) + 4
            assert need <= per_cell, (width, label.value, need, per_cell)
            height = (2 * drawer_mod.DEST_PAD_V + icon.size + cell.content.spacing
                      + tokens.TYPE_SCALE["label_small"].line)
            assert height >= 40


def test_drawer_destination_cells_navigate_and_carry_the_jobs_badge():
    went = []
    state = AppState()
    drawer = ChatDrawer(state=state, on_navigate=went.append, clock=lambda: 200.0)
    for dest_id, _label, _icon, route in DESTINATIONS:
        drawer.destination_chips[dest_id].on_click(None)
    assert went == [d[3] for d in DESTINATIONS]
    state.jobs_badge.value = type(state.jobs_badge.value)(running=3)
    drawer.refresh()
    assert drawer.destination_chips["jobs"].badge is not None


# ==========================================================================
# 3. Welcome: "Run on this phone" (the Model manager's OnDeviceFlow)
# ==========================================================================

GB = 1_000_000_000


def _engine_kwargs(tmp_path, total=8 * 1024 ** 3, available=6 * 1024 ** 3, platform="android"):
    """WelcomeScreen kwargs with a LocalEngineService (FakeEngine, a phone with ``total`` / ``available`` RAM)."""
    import local_llm_fakes as fakes
    from glossarion_mobile.services.local_llm import downloads as dl
    from glossarion_mobile.services.local_llm import engine as eng
    from glossarion_mobile.services.local_llm import service as svc

    engine = eng.FakeEngine()
    prefs = fakes.FakePrefs()
    service = svc.LocalEngineService(prefs=prefs, registry=dl.Registry(str(tmp_path / "models")),
                                     engine_factory=lambda: engine, watch=False, platform=platform,
                                     memory=lambda: (total, available))
    return {"local_engine": lambda: service, "prefs": prefs}, service, engine


def test_page_one_lists_the_users_own_on_device_presets(tmp_path):
    """owner 2026-10-21: a preset added in the Model manager (Add from Hugging Face › Save as a preset)
    shows on the Welcome too: every preset gets a cell, four per row, and a tap picks that preset."""
    from glossarion_mobile.services.local_llm import presets as lp

    kwargs, _service, _engine = _engine_kwargs(tmp_path)
    custom = {"name": "Gemma E2B", "repo": "unsloth/gemma-4-E2B-it-GGUF", "file": "gemma-4-E2B-it-Q4_K_M.gguf",
              "size": 3_000_000_000}
    lp.save_presets(kwargs["prefs"], lp.load_presets(kwargs["prefs"]) + [custom])
    s = screen(**kwargs)
    rows = by_key(s.page_area.content)
    cells = rows["welcome-ondevice"].controls
    assert len(cells) == 5 and all(c.col == {"xs": 3} for c in cells)  # the fifth wraps onto a second row
    assert cells[-1] is rows["welcome-ondevice-gemma-e2b"]
    assert cells[-1].content.controls[0].value == "Gemma E2B"
    picked = []
    s.use_ondevice = lambda index: picked.append(index)
    cells[-1].on_click(None)
    assert picked == [4] and lp.load_presets(kwargs["prefs"])[4]["name"] == "Gemma E2B"


def test_page_one_offers_the_gpu_switch_where_the_build_has_a_gpu_backend(tmp_path):
    """owner 2026-10-22: "Use GPU (experimental)" on the Welcome page too: the Model manager's switch (same
    setting), under the on-device cells; switching it on adds the Q4_0 presets (Android; hotfix 3: no extra
    tap, the button is only the fallback). No backend: none."""
    from glossarion_mobile.services.local_llm import presets as lp
    from glossarion_mobile.services.local_llm import service as svc

    kwargs, service, _engine = _engine_kwargs(tmp_path / "x86")
    assert service.gpu_backend is None  # the host / emulator: no GPU backend
    assert "welcome-gpu" not in by_key(screen(**kwargs).page_area.content)

    kwargs, service, _engine = _engine_kwargs(tmp_path / "arm64")
    service.gpu_backend = "OpenCL"
    s = screen(**kwargs)
    rows = by_key(s.page_area.content)
    switch = rows["welcome-gpu"]
    assert switch.visible and switch.value is False and switch.label == "Use GPU (experimental)"
    assert "welcome-gpu-note" not in rows and "welcome-gpu-status" not in rows  # page 1 stays compact
    assert not rows["welcome-gpu-presets"].visible
    # the K-quant hint (test_local_llm_gpu) lives here too; hidden while no local/ model is in use
    assert not rows["welcome-gpu-swap"].visible
    switch.value = True
    switch.on_change(None)
    assert kwargs["prefs"].get(svc.GPU_KEY) is True and service.use_gpu()
    names = [p["name"] for p in lp.load_presets(kwargs["prefs"])]
    assert names[-len(lp.GPU_PRESETS):] == [p["name"] for p in lp.GPU_PRESETS]
    # the page re-rendered: the new presets have cells and the switch still shows the saved setting
    rows = by_key(s.page_area.content)
    assert rows["welcome-gpu"].value is True and len(rows["welcome-ondevice"].controls) == len(names)
    assert not rows["welcome-gpu-presets"].visible  # nothing missing: the fallback button stays hidden


@pytest.mark.parametrize("width", PHONES)
def test_page_one_fits_with_the_gpu_friendly_presets(width, tmp_path):
    """Hotfix 6 (owner 2026-10-27): on Android (OpenCL) the Q4_0 presets show with the GPU off, so page 1 has
    eight cells; their smaller "Garbage GPU" names fit a quarter row."""
    from glossarion_mobile.services.local_llm import presets as lp
    from glossarion_mobile.ui.screens import welcome

    kwargs, service, _engine = _engine_kwargs(tmp_path)
    service.gpu_backend = "OpenCL"
    s = screen(**kwargs)
    rows = by_key(s.page_area.content)
    assert rows["welcome-gpu"].value is False
    cells = rows["welcome-ondevice"].controls
    assert len(cells) == len(lp.DEFAULT_PRESETS) + len(lp.GPU_PRESETS)
    names = [cell.content.controls[0] for cell in cells]
    gpu = names[len(lp.DEFAULT_PRESETS):]
    assert [n.value for n in gpu] == ["Garbage GPU", "Light GPU", "Medium GPU", "Heavy GPU"]
    assert all(n.size == welcome.ONDEVICE_GPU_NAME_SIZE for n in gpu) and all(n.size is None for n in names[:4])
    assert cells[len(lp.DEFAULT_PRESETS)].tooltip.startswith("Garbage · GPU (Q4_0)")
    content_w = width - 32
    quarter = (content_w - 3 * 6) / 4 - 12
    for name in names:
        size = name.size or tokens.TYPE_SCALE["label_medium"].size
        assert text_width(name.value, size) <= quarter, (name.value, quarter)
    budget = HEIGHTS[width] - (24 + 56 + 48 + 32 + 8 + 48 + 20)
    if width >= 412:  # the owner's phone class: page 1 still fits
        assert _page_height(s, content_w) <= budget, (width, budget)
    else:  # 360 x 800: the second row of cells and the GPU switch need ~100 dp more than is left: it scrolls
        assert s.page_area.content.scroll == ft.ScrollMode.AUTO


def test_page_one_on_device_cells_show_size_ram_hint_and_suggest_light(tmp_path):
    kwargs, _service, _engine = _engine_kwargs(tmp_path, total=6 * 1024 ** 3, available=3 * 1024 ** 3)
    s = screen(**kwargs)
    rows = by_key(s.page_area.content)
    assert "welcome-local" in rows  # Ollama / LM Studio stays: the PC option
    texts = [c.value for c in walk(s.page_area.content) if isinstance(c, ft.Text)]
    assert "Run on this phone (no account needed)" in texts
    assert "authchatgpt/ and authnd/: no sign-in" not in texts
    cells = {k: rows[k] for k in ("welcome-ondevice-garbage", "welcome-ondevice-light", "welcome-ondevice-medium",
                                  "welcome-ondevice-heavy")}
    lines = {k: [t.value for t in c.content.controls if isinstance(t, ft.Text)] for k, c in cells.items()}
    # model + its image support; the engine's RAM gate on a 6 GB phone with 3 GB free
    assert lines["welcome-ondevice-garbage"] == ["Garbage", "764 MB ✓"]
    assert lines["welcome-ondevice-light"] == ["Light ⭐", "2.0 GB ✓"]
    assert lines["welcome-ondevice-medium"] == ["Medium", "3.6 GB ⚠"]
    assert lines["welcome-ondevice-heavy"] == ["Heavy", "6.6 GB ✕"]
    assert "Android may close Glossarion" in cells["welcome-ondevice-medium"].tooltip
    assert "Pick a smaller model" in cells["welcome-ondevice-heavy"].tooltip
    # a 2 GB phone: Light does not fit, the biggest preset that does is suggested instead
    kwargs, _service, _engine = _engine_kwargs(tmp_path / "small", total=2 * 1024 ** 3, available=1536 * 1024 ** 2)
    small = screen(**kwargs)
    names = [c.content.controls[0].value for c in by_key(small.page_area.content)["welcome-ondevice"].controls]
    assert names == ["Garbage ⭐", "Light", "Medium", "Heavy"]


def test_page_one_on_device_without_an_engine_points_to_the_model_manager():
    s = screen()
    row = by_key(s.page_area.content)["welcome-ondevice"]
    assert isinstance(row, ft.Row) and any("Model manager" in str(getattr(c, "reason", "")) for c in row.controls)


def test_page_one_on_device_one_tap_downloads_loads_and_selects(tmp_path, monkeypatch):
    """One tap on a cell: the Model manager's flow downloads the GGUF and its vision projector (inline
    progress in the cell), loads it and selects local/<stem>; the Welcome then counts as set up."""
    import asyncio
    import hashlib
    import os

    import local_llm_fakes as fakes
    from glossarion_mobile.services.local_llm import hf, presets

    payload = os.urandom(300_000)
    server = fakes.FakeHF()
    server.add("tester/Tiny-GGUF", {"Tiny-Q4_K_M.gguf": payload, "mmproj-F16.gguf": b"projector"})
    monkeypatch.setattr(hf, "HF_ENDPOINT", server.url)
    try:
        kwargs, service, engine = _engine_kwargs(tmp_path)
        kwargs["prefs"].set(presets.PREF_KEY, [
            {"name": "Light", "repo": "tester/Tiny-GGUF", "file": "Tiny-Q4_K_M.gguf", "note": "test", "n_ctx": 0,
             "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}])
        chosen = []
        s = screen(on_set_model=chosen.append, current_model=lambda: chosen[-1] if chosen else "", **kwargs)
        assert not s.flow.signed_in
        server.chunk_delay = 0.005
        seen = []
        service.downloads.throttle = 0

        def progress(state):
            if state.running and state.phase == "downloading" and not state.stem.endswith(".mmproj"):
                cell = by_key(s.page_area.content)["welcome-ondevice-light"]
                seen.append((cell.content.controls[1].value, isinstance(cell.content.controls[-1], ft.ProgressBar)))

        service.downloads.subscribe(progress)
        asked = []
        s.ondevice_flow().confirm_fn = lambda title, body, label: asked.append((title, body, label)) or True
        model = asyncio.run(s.use_ondevice(0))
        assert asked and asked[0][0] == "Download Light?" and asked[0][2] == "Download"  # every download asks first
        assert model == "local/Tiny-Q4_K_M" and chosen == ["local/Tiny-Q4_K_M"]
        assert any(text.endswith("% · ✕") and bar for text, bar in seen), seen  # tap again to cancel
        assert service.loaded_stem == "Tiny-Q4_K_M" and service.is_vision("Tiny-Q4_K_M")
        assert s.flow.signed_in and s.model_notes["local"] == "✓ Using local/Tiny-Q4_K_M · runs on this phone"
        cell = by_key(s.page_area.content)["welcome-ondevice-light"]
        assert cell.content.controls[1].value == "✓ In use"
        assert by_key(s.page_area.content)["welcome-model-local"].value.startswith("✓ Using local/Tiny-Q4_K_M")
        s.dispose()
    finally:
        server.close()


def test_page_one_on_device_second_tap_cancels_a_download(tmp_path, monkeypatch):
    """A mis-tap is undone with a second tap: the downloading cell asks "Cancel the Light download?" and
    Cancel download stops it and deletes the partial file; "Keep downloading" leaves it running."""
    import asyncio
    import hashlib
    import os

    import local_llm_fakes as fakes
    from glossarion_mobile.services.local_llm import hf, presets

    payload = os.urandom(4_000_000)
    server = fakes.FakeHF()
    server.add("tester/Tiny-GGUF", {"Tiny-Q4_K_M.gguf": payload})
    monkeypatch.setattr(hf, "HF_ENDPOINT", server.url)
    try:
        kwargs, service, _engine = _engine_kwargs(tmp_path)
        kwargs["prefs"].set(presets.PREF_KEY, [
            {"name": "Light", "repo": "tester/Tiny-GGUF", "file": "Tiny-Q4_K_M.gguf", "note": "test", "n_ctx": 0,
             "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}])
        s = screen(**kwargs)
        server.chunk_delay = 0.05
        service.downloads.throttle = 0
        asked = []
        answers = {"Download Light?": True, "Cancel the Light download?": False}
        s.ondevice_flow().confirm_fn = lambda title, body, label: asked.append(title) or answers[title]

        async def scenario():
            first = asyncio.create_task(s.use_ondevice(0))
            for _ in range(500):
                await asyncio.sleep(0.01)
                state = s.ondevice_flow().active_download(s.ondevice_flow().stem_of_preset(
                    s.ondevice_flow().presets()[0]))
                if state is not None and state.running and state.phase == "downloading":
                    break
            assert await s.use_ondevice(0) is None  # "Keep downloading": still running
            assert asked[-1] == "Cancel the Light download?" and not first.done()
            answers["Cancel the Light download?"] = True
            assert await s.use_ondevice(0) is None  # "Cancel download"
            return await asyncio.wait_for(first, 10)

        assert asyncio.run(scenario()) is None
        assert asked.count("Download Light?") == 1 and asked.count("Cancel the Light download?") == 2
        assert not service.loaded_stem
        assert not [p for p in tmp_path.rglob("*.part")]  # the partial file is gone
        s.dispose()
    finally:
        server.close()

def test_choosing_a_glossary_mode_keeps_the_page_and_its_scroll():
    """owner 2026-10-18: tapping a mode card on page 2 rebuilt the page and scrolled it back to the top."""
    s = screen()
    s.go("language_glossary")
    page = s.page_area.content
    first, second = list(s.mode_cards)[:2]
    s.select_mode(second)
    assert s.page_area.content is page  # the same page: its scroll position stays
    assert s.flow.glossary_mode == second
    assert s.mode_cards[second].border is not None and s.mode_cards[first].border is None
    assert s.mode_cards[second].bgcolor == ft.Colors.SECONDARY_CONTAINER
    s.select_mode(first)
    assert s.page_area.content is page and s.mode_cards[first].border is not None
    assert s.mode_cards[second].border is None
