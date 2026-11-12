"""Mobile v9.15.12 hotfix 1 (owner's device report):

1. The response (Plan) card's profile chip follows a profile chosen in Settings › Profiles, as the run does.
2. The SDLXLIFF reviewer remembers the workspace and piece it showed last, and the Tools hub opens the newest
   translated workspace instead when a job worked on it since.
3. The reviewer's buttons are short: "🌐 Translate", "🟣 Flag" and one row of OK / Tags / MT / Lost filters.
4. The on-device Custom › Gemma menu shows its sizes under "Gemma 3" and "Gemma 4".
5. On-device Gemma (sliding-window) models keep only their window in the SWA cache, the CPU runs flash
   attention, and every request on such a model starts on an empty cache (0.3.32's prefix reuse cannot tell
   that the SWA layers dropped the older tokens).

Run from src/mobile (mobile venv):
    python -m pytest -p no:cacheprovider -W ignore tests_host/test_mobile_hotfix1_9_15_12.py -q
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
import types
from pathlib import Path

MOBILE_DIR = Path(__file__).resolve().parents[1]
for _path in (MOBILE_DIR / "app", MOBILE_DIR / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # its dataclasses look their module up
    spec.loader.exec_module(module)
    return module


TC = _load("_glossarion_chat_helpers_hf1_91512", Path(__file__).with_name("test_chat.py"))
RO = _load("_glossarion_run_options_helpers_hf1_91512", Path(__file__).with_name("test_run_options_resume.py"))
U7 = _load("_glossarion_u7_helpers_hf1_91512", Path(__file__).with_name("test_tools_u7.py"))
desktop_store_cls = TC.desktop_store_cls
isolated_env = TC.isolated_env
needs_flet = TC.needs_flet


class Prefs:
    def __init__(self, data=None) -> None:
        self.data = dict(data or {})

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value


# ---- 1: the Plan card's profile chip follows Settings › Profiles -------------------------------------------


@needs_flet
def test_plan_profile_chip_follows_a_profile_chosen_outside_the_chat(desktop_store_cls, isolated_env):
    import flet as ft

    from glossarion_mobile.ui.screens.profiles import ProfileService

    tmp_path = isolated_env

    async def scenario():
        chat = TC._chat_view(desktop_store_cls, tmp_path)
        view = chat.view
        try:
            await RO._settle(chat.calls)
            book = tmp_path / "novel.epub"
            book.write_bytes(b"PK\x03\x04")
            assert view.attach_file(str(book))
            view.begin_send()

            def label() -> str:
                rows = [c for c in TC._plan_box(view) if isinstance(c, ft.Row)]
                chips = {c.key: c for row in rows for c in row.controls if isinstance(c, ft.Chip)}
                return str(chips["plan-profile"].label.value)

            service = ProfileService(chat.store)
            other = next(name for name in service.listing().names if name != label())
            renders = []
            original = view.render_transcript
            view.render_transcript = lambda *a, **kw: (renders.append(1), original(*a, **kw))[1]
            service.select(other)  # Settings › Profiles "Use this profile"
            assert chat.store.get("active_profile") == other
            assert label() == other and view.state.chat_context.value.profile == other
            count = len(renders)
            chat.store.set("active_profile", other)  # an unchanged value: no re-render
            chat.store.set("output_language", view.state.chat_context.value.target_language)
            assert len(renders) == count
        finally:
            TC._close(chat)

    asyncio.run(scenario())


# ---- 2: the reviewer remembers its workspace; a newer job's output wins -----------------------------------


def _workspace(root: Path, name: str) -> Path:
    folder = root / name
    (folder / "SDLXLIFF").mkdir(parents=True)
    return folder


def test_last_workspace_prefers_a_newer_jobs_output_over_the_remembered_one(tmp_path, monkeypatch):
    from glossarion_mobile.ui.tools import sdlxliff as sx
    from glossarion_mobile.ui.tools import targets as tg

    old = _workspace(tmp_path, "Old")
    new = _workspace(tmp_path, "New")
    bare = tmp_path / "NoSidecars"
    bare.mkdir()
    rows: list = []
    monkeypatch.setattr(tg, "recent_output_targets", lambda service, **kw: list(rows))
    monkeypatch.setattr(tg, "chat_workspace_targets", lambda root, **kw: [])
    row = lambda folder, mtime: types.SimpleNamespace(folder=str(folder), source="", mtime=mtime)  # noqa: E731

    assert sx.last_workspace(Prefs(), object()) is None
    prefs = Prefs({sx.LAST_PREF: {"folder": str(old), "source": "Old.epub", "piece": "ch2.html", "at": 100.0}})
    assert sx.last_workspace(prefs, None) == (str(old), "Old.epub", "ch2.html")
    # the newest translated workspace without sidecars is skipped; an older job keeps the remembered one
    rows[:] = [row(bare, 500.0), row(new, 50.0)]
    assert sx.last_workspace(prefs, object()) == (str(old), "Old.epub", "ch2.html")
    # a job worked on another book after the reviewer showed one: its workspace
    rows[:] = [row(new, 200.0), row(old, 150.0)]
    assert sx.last_workspace(prefs, object()) == (str(new), "", "")
    # the remembered workspace itself is the newest: it keeps its piece
    rows[:] = [row(old, 300.0)]
    assert sx.last_workspace(prefs, object()) == (str(old), "Old.epub", "ch2.html")
    # nothing remembered (or gone): the newest one
    rows[:] = [row(new, 1.0)]
    assert sx.last_workspace(Prefs({sx.LAST_PREF: {"folder": str(bare), "at": 9e9}}), object()) == (str(new), "", "")


@needs_flet
def test_tools_hub_reviewer_reopens_the_remembered_workspace_and_piece(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "sdlxliff_review_core", U7._fake_sdl_core())
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.tools import sdlxliff as sx

    out = _workspace(tmp_path / "Output", "Book")
    (out / "response_ch002.html").write_text("<p>Three</p>", encoding="utf-8")

    async def scenario():
        _conn, session = U7._tb()._fake_session("android")
        page = session.page
        ctx = U7._tools_ctx(page, store={"sdlxliff_machine_translation_provider": "auto"})
        ctx.prefs = Prefs({sx.LAST_PREF: {"folder": str(out), "source": str(tmp_path / "Book.epub"),
                                          "piece": "response_ch002.html", "at": 1.0}})
        screen = sx.SdlxliffScreen(parse_route("/tools/sdlxliff"), ctx)
        assert screen.folder == "" and screen.error
        U7._tools()._mount(page, screen.get_body())
        screen._visible = True
        binding = await screen.open_last()
        assert binding is not None and screen.folder == str(out) and screen.error is None
        assert binding.session.source_path == str(tmp_path / "Book.epub")
        assert screen.piece_index == 1  # the piece shown last
        saved = ctx.prefs.get(sx.LAST_PREF)
        assert saved["folder"] == str(out) and saved["at"] > 1.0  # opening moves its time on
        screen.step(-1)  # another piece: remembered for next time
        assert ctx.prefs.get(sx.LAST_PREF)["piece"] == binding.pieces[0]["output_name"]
        assert "sdl-workspace" in [a.key for a in screen.actions()]  # another workspace from the bar
        screen.dispose()

    asyncio.run(scenario())


# ---- 3: short reviewer buttons ------------------------------------------------------------------------------


@needs_flet
def test_reviewer_buttons_are_short_and_the_filters_share_one_row(tmp_path, monkeypatch):
    import flet as ft

    monkeypatch.setitem(sys.modules, "sdlxliff_review_core", U7._fake_sdl_core())
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.tools import sdlxliff as sx

    out = _workspace(tmp_path / "Output", "Book")

    async def scenario():
        _conn, session = U7._tb()._fake_session("android")
        page = session.page
        ctx = U7._tools_ctx(page, store={})
        fid = sx.open_reviewer(ctx, str(out))
        screen = sx.SdlxliffScreen(parse_route(f"/tools/sdlxliff?out={fid}"), ctx)
        U7._tools()._mount(page, screen.get_body())
        await screen.open()
        assert screen.mt_button.content == "🌐 Translate" and screen.flag_button.content == "🟣 Flag"
        assert screen.legend.scroll is None and screen.legend.wrap is False  # one row, no sideways scroll
        words = []
        for (status, legend_text), button in zip(sx.LEGEND, screen.legend.controls):
            assert button.expand and button.tooltip == legend_text and button.key == f"sdl-legend-{status}"
            words.append(button.content.controls[1].value.split()[0])
        assert words == ["OK", "Tags", "MT", "Lost"]
        screen.legend.controls[3].on_click(None)
        assert screen.filter_status == "red" and screen.legend.controls[3].bgcolor is not None
        screen.legend.controls[3].on_click(None)
        assert screen.filter_status is None
        assert isinstance(screen.mt_button, ft.FilledButton) and isinstance(screen.flag_button, ft.FilledButton)
        screen.dispose()

    asyncio.run(scenario())


# ---- 4: Gemma 3 and Gemma 4 in the Custom menu ------------------------------------------------------------------


def test_custom_gemma_menu_names_both_generations():
    from glossarion_mobile.services.local_llm import presets
    from glossarion_mobile.ui.screens.local_models_panel import custom_note, size_groups

    gemma = presets.custom_family("gemma")
    groups = size_groups(gemma)
    assert [(g, [s["label"] for _i, s in sizes]) for g, sizes in groups] == [
        ("Gemma 3", ["270M", "1B", "4B"]), ("Gemma 4", ["E2B", "E4B"])]
    assert all(s["repo"].startswith(f"unsloth/gemma-{g[-1]}-") for g, sizes in groups for _i, s in sizes)
    assert custom_note(gemma) == "Choose a size and a quant: Gemma 3 270M · 1B · 4B · Gemma 4 E2B · E4B"
    qwen = presets.custom_family("qwen35")
    assert size_groups(qwen)[0][0] == "" and custom_note(qwen) == "Choose a size and a quant: 0.8B · 2B · 4B · 9B"


@needs_flet
def test_custom_sheet_heads_the_gemma_sizes_with_their_generation():
    import flet as ft

    from glossarion_mobile.services.local_llm import presets
    from glossarion_mobile.ui.screens.local_models_panel import CustomModelSheet

    flow = types.SimpleNamespace(spawn=lambda coro: coro.close())
    sheet = CustomModelSheet(flow, presets.custom_family("gemma"))
    groups = sheet.size_row.controls
    assert [g.controls[0].value for g in groups] == ["Gemma 3", "Gemma 4"]
    assert [[b.content for b in g.controls[1].controls] for g in groups] == [["270M", "1B", "4B"], ["E2B", "E4B"]]
    gemma4 = groups[1].controls[1].controls[0]
    assert gemma4.key == "custom-size-3" and gemma4.tooltip == "Gemma 4 E2B"
    assert all(isinstance(b, ft.OutlinedButton) for g in groups for b in g.controls[1].controls)


# ---- 5: sliding-window models: small SWA cache, flash attention on the CPU, an empty cache per request -----------


class _SwaLlama:
    made: list = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.events: list = []
        self._model = types.SimpleNamespace(model=kwargs["model_path"])
        self._ctx = types.SimpleNamespace(kv_cache_clear=lambda: self.events.append("clear"))
        _SwaLlama.made.append(self)

    def reset(self):
        self.events.append("reset")

    def tokenize(self, text, add_bos=False, special=True):
        return list(range(len(text.split())))

    def create_chat_completion(self, **kwargs):
        self.events.append("chat")
        yield {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}

    def close(self):
        pass


def test_on_device_engine_limits_the_swa_cache_and_clears_it_per_request(monkeypatch):
    from glossarion_mobile.services.local_llm import engine as eng

    windows = {"gemma-4-E2B-it-Q4_0.gguf": 512, "Qwen3.5-2B-Q4_0.gguf": 0}
    module = types.SimpleNamespace(Llama=_SwaLlama, llama_model_n_swa=lambda model: windows[model])
    monkeypatch.setattr(eng.LlamaEngine, "_module", lambda self: module)
    monkeypatch.setattr(eng, "_make_formatter", lambda llm: None)
    monkeypatch.setattr(eng, "cpu_variant", lambda: "builtin")
    monkeypatch.setattr(eng, "_gpu_check", lambda llm: None)
    monkeypatch.setattr(eng, "gpu_state", lambda: {"backend": "OpenCL", "device": "Adreno", "error": None})
    messages = [{"role": "user", "content": "hi"}]

    engine = eng.LlamaEngine()
    engine.load("gemma-4-E2B-it-Q4_0.gguf", 4096, 4)
    llm = _SwaLlama.made[-1]
    assert llm.kwargs["swa_full"] is False and llm.kwargs["flash_attn"] is True and engine.sliding_window == 512
    for _ in range(2):
        assert "".join(engine.chat(messages, {"max_tokens": 8}, lambda: False)) == "ok"
    assert llm.events == ["reset", "clear", "chat"] * 2  # never a reused prefix

    engine.load("gemma-4-E2B-it-Q4_0.gguf", 4096, 4, gpu=True)  # the GPU also uses fused attention
    assert _SwaLlama.made[-1].kwargs["flash_attn"] is True and _SwaLlama.made[-1].kwargs["swa_full"] is False

    engine.load("Qwen3.5-2B-Q4_0.gguf", 4096, 4)  # no sliding window: prompt-prefix reuse stays
    qwen = _SwaLlama.made[-1]
    "".join(engine.chat(messages, {"max_tokens": 8}, lambda: False))
    assert engine.sliding_window == 0 and qwen.events == ["chat"]
    engine.unload()
    assert engine.sliding_window == 0
