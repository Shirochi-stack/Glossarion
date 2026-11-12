"""Sigma hotfix: exact-file policy, selection, migration and normal-setting restoration."""
import asyncio
from pathlib import Path

import pytest
from local_llm_fakes import FakePrefs
from glossarion_mobile.services.local_llm import presets, service as svc, downloads as dl, hf, engine
from glossarion_mobile.state import setting_writes as sw


def make_service(tmp_path, prefs=None, memory=(24 * 1024**3, 20 * 1024**3)):
    registry = dl.Registry(str(tmp_path / "models"))
    for stem, repo, file in (("sigma-q8", presets.SIGMA_REPO, presets.SIGMA_Q8_FILE),
                             ("sigma-q4", presets.SIGMA_REPO, "gguf/sft3-Q4_K_M.gguf"),
                             ("impostor", "other/sigma", presets.SIGMA_Q8_FILE)):
        registry.add(stem, {"repo": repo, "file": file, "size": 100})
        Path(registry.file_path(stem)).write_bytes(b"GGUF")
    return svc.LocalEngineService(prefs=prefs or FakePrefs(), registry=registry, watch=False,
                                  engine_factory=engine.FakeEngine, platform="android", memory=lambda: memory)


def test_sigma_q8_request_uses_editable_model_defaults(tmp_path):
    service = make_service(tmp_path)
    body = {"temperature": .23, "top_p": .76, "presence_penalty": .1, "max_tokens": 999,
            "glossarion_task": "translation", "target_language": "English"}
    params = service._params(body, 500, stem="sigma-q8")
    assert params["temperature"] == .23 and params["top_p"] == .95 and params["presence_penalty"] == 0
    assert params["max_tokens"] == 500
    assert not {"ban_scripts", "forced_truncation_seconds"} & params.keys()
    assert params["dry_multiplier"] == 0 and params["min_p"] == .05
    for stem in ("sigma-q4", "impostor"):
        small = service._params(body, 500, stem=stem)
        assert small["top_p"] == service.sampling()["top_p"]
        assert small["ban_scripts"] and small["forced_truncation_seconds"] == 3
    service._stem = "sigma-q8"
    assert not service.block_cjk() and not service.forced_truncation()
    service.save_forced_truncation(True, 7)
    assert service._params(body, 500, stem="sigma-q8")["forced_truncation_seconds"] == 7
    service._stem = "sigma-q4"
    assert service.forced_truncation() and service.block_cjk()


@pytest.mark.parametrize("old_context,expected", [(0, 24576), (8192, 8192), (32768, 24576)])
def test_context_migration_keeps_lower_values(tmp_path, old_context, expected):
    prefs = FakePrefs({svc.SETTINGS_KEY: {"n_ctx": old_context, "n_threads": 3}})
    service = make_service(tmp_path, prefs)
    assert service.context_for("sigma-q8") == expected
    assert service.threads() == 3
    prefs.set(presets.PREF_KEY, [{**presets.SIGMA_PRESET, "n_ctx": 4096}])
    assert service.context_for("sigma-q8") == 4096


def test_heavy_migration_keeps_identity_and_is_idempotent(tmp_path):
    old = {"name": "Heavy", "repo": "unsloth/Qwen3.5-9B-GGUF", "file": "Qwen3.5-9B-Q4_K_M.gguf",
           "quant": "", "n_ctx": 8192, "size": 5680522464, "note": "old", "mmproj": "mmproj-F16.gguf"}
    prefs = FakePrefs({presets.PREF_KEY: [old]})
    rows = presets.load_presets(prefs)
    custom = next(row for row in rows if row["repo"] == old["repo"])
    assert custom["name"] == "Qwen 3.5 9B · Q4_K_M"
    assert all(custom[key] == old[key] for key in ("repo", "file", "n_ctx", "size", "mmproj"))
    assert sum(presets.is_sigma(row["repo"]) for row in rows) == 1
    assert presets.load_presets(prefs) == rows
    registry = dl.Registry(str(tmp_path / "models"))
    registry.add("my-heavy", old)
    Path(registry.file_path("my-heavy")).write_bytes(b"GGUF")
    service = svc.LocalEngineService(prefs=FakePrefs(), registry=registry, watch=False)
    assert service.registry.load()["my-heavy"]["file"] == old["file"]
    assert service.preset_for("my-heavy")["name"] == custom["name"]


@pytest.mark.parametrize("available,expected", [(20 * 1024**3, presets.SIGMA_Q8_FILE),
                                               (6 * 1024**3, "gguf/sft3-Q4_K_M.gguf"), (None, "gguf/sft3-Q4_K_M.gguf")])
def test_sigma_popup_ram_selection_and_cancel(tmp_path, available, expected):
    from glossarion_mobile.ui.screens.local_models_panel import OnDeviceFlow, FileChooser
    service = make_service(tmp_path, memory=(24 * 1024**3, available) if available else (0, None))
    q4 = hf.HfFile("gguf/sft3-Q4_K_M.gguf", 5302273856)
    q8 = hf.HfFile(presets.SIGMA_Q8_FILE, 7972726592)
    async def resolve(ref):
        return hf.RepoFiles("main", (q4, q8), ())
    service.resolve = resolve
    calls = []
    async def choose(repo, files, selected):
        calls.append(selected.path)
        chooser = FileChooser(repo, files, selected)
        assert chooser.recommended is q8
        assert chooser.group.value == expected
        chooser.pick(None)
        return await chooser.future
    flow = OnDeviceFlow(service=service, prefs=service.prefs, choose=choose,
                        use_model=lambda model: pytest.fail("Cancelled choice activated a model"))
    assert asyncio.run(flow.get_preset(3)) is None
    assert calls == [expected] and not service.downloads.states


def test_hf_no_quant_radio_label():
    from glossarion_mobile.ui.screens.local_models_panel import FileChooser
    async def scenario():
        file = hf.HfFile("model-BF16.gguf", 900)
        chooser = FileChooser("owner/repo", [file], file)
        assert chooser.group.content.controls[0].title.value == "No quant (BF16)"
        assert chooser.chosen() is file
        chooser.pick(None)
    asyncio.run(scenario())


def test_sigma_quant_selection_persists_and_reopens_on_next_tap(tmp_path):
    from glossarion_mobile.ui.screens.local_models_panel import OnDeviceFlow, LocalModelsPanel
    service = make_service(tmp_path)
    q4 = hf.HfFile("gguf/sft3-Q4_K_M.gguf", 5302273856)
    q8 = hf.HfFile(presets.SIGMA_Q8_FILE, 7972726592)
    async def resolve(ref):
        return hf.RepoFiles("main", (q4, q8), ())
    service.resolve = resolve
    used, picks = [], iter((q8, q4))
    async def choose(repo, files, selected):
        return next(picks)
    async def scenario():
        flow = OnDeviceFlow(service=service, prefs=service.prefs, choose=choose, use_model=used.append)
        assert await flow.get_preset(3) == "local/sigma-q8"
        assert presets.load_presets(service.prefs)[3]["file"] == q8.path
        service.unload()
        panel = LocalModelsPanel(service=service, prefs=service.prefs,
                                 current_model=lambda: "local/sigma-q8")
        panel.build()
        assert not panel.block_cjk_switch.disabled and not panel.block_cjk_switch.value
        assert not panel.forced_truncation_switch.value
        assert all(not field.disabled for field in panel.sampling_fields.values())
        panel.dispose()
        fresh = OnDeviceFlow(service=service, prefs=service.prefs, choose=choose, use_model=used.append)
        assert await fresh.get_preset(3) == "local/sigma-q4"
        assert presets.load_presets(service.prefs)[3]["file"] == q4.path
        assert used == ["local/sigma-q8", "local/sigma-q4"]
    asyncio.run(scenario())


@pytest.mark.parametrize("saved_file,expected", [
    ("gguf/sft3-Q4_K_M.gguf", "gguf/sft3-Q4_K_M.gguf"),
    (presets.SIGMA_Q8_FILE, presets.SIGMA_Q8_FILE),
    ("gguf/removed.gguf", presets.SIGMA_Q8_FILE),
])
def test_sigma_saved_quant_wins_over_ram_suggestion_and_cancel_keeps_it(tmp_path, saved_file, expected):
    from glossarion_mobile.ui.screens.local_models_panel import OnDeviceFlow
    service = make_service(tmp_path)
    rows = presets.load_presets(service.prefs)
    rows[3].update(name="Sigma (Best)", file=saved_file, n_ctx=8192)
    presets.save_presets(service.prefs, rows)
    q4 = hf.HfFile("gguf/sft3-Q4_K_M.gguf", 5302273856)
    q8 = hf.HfFile(presets.SIGMA_Q8_FILE, 7972726592)
    async def resolve(ref):
        return hf.RepoFiles("main", (q4, q8), ())
    service.resolve = resolve
    async def choose(repo, files, selected):
        assert selected.path == expected
        return None
    flow = OnDeviceFlow(service=service, prefs=service.prefs, choose=choose)
    assert asyncio.run(flow.get_preset(3)) is None
    saved = presets.load_presets(service.prefs)[3]
    assert saved["name"] == "Sigma" and saved["file"] == saved_file and saved["n_ctx"] == 8192
    assert not service.downloads.states


def test_sigma_picker_marks_only_present_files_installed(tmp_path):
    from types import SimpleNamespace
    from glossarion_mobile.ui.screens.local_models_panel import OnDeviceFlow
    service = make_service(tmp_path)
    Path(service.registry.file_path("sigma-q4")).unlink()
    q4 = hf.HfFile("gguf/sft3-Q4_K_M.gguf", 5302273856)
    q8 = hf.HfFile(presets.SIGMA_Q8_FILE, 7972726592)
    async def scenario():
        flow = OnDeviceFlow(service=service, prefs=service.prefs,
                            page=SimpleNamespace(width=400, height=800, show_dialog=lambda dialog: None))
        task = asyncio.create_task(flow.choose(presets.SIGMA_REPO, [q4, q8], q4))
        await asyncio.sleep(0)
        chooser = flow.last_dialog
        assert chooser.group.value == q4.path
        assert "Installed" not in chooser.group.content.controls[0].subtitle.value
        assert "Installed" in chooser.group.content.controls[1].subtitle.value
        assert "Recommended" in chooser.group.content.controls[1].subtitle.value
        assert chooser.title.value == "Sigma · choose a quant"
        chooser.pick(None)
        assert await task is None
    asyncio.run(scenario())


def test_sigma_welcome_shows_installed_without_best(tmp_path):
    from test_welcome_drawer_compact import _engine_kwargs, screen, by_key
    kwargs, service, _engine = _engine_kwargs(tmp_path)
    Path(service.registry.root).mkdir(parents=True, exist_ok=True)
    Path(service.registry.file_path("sigma-q8")).write_bytes(b"GGUF")
    service.registry.add("sigma-q8", {"repo": presets.SIGMA_REPO, "file": presets.SIGMA_Q8_FILE})
    view = screen(**kwargs)
    cell = by_key(view.page_area.content)["welcome-ondevice-sigma"]
    assert [line.value for line in cell.content.controls] == ["Sigma", "Installed"]
    view.dispose()


def test_switching_sigma_q8_restores_prompts_and_temperature(tmp_path, monkeypatch):
    from test_on_device_profiles import _store, _existing_config
    service = make_service(tmp_path)
    monkeypatch.setattr(svc, "default_service", lambda: service)
    config = {**_existing_config(), "translation_temperature": .27, "batch_size": 6}
    store = _store(tmp_path, config)
    try:
        sw.write_setting(store, "model", "local/sigma-q8")
        assert store.get("active_profile") == config["active_profile"]
        assert store.get("translation_temperature") == .27
        sw.write_setting(store, "model", "local/sigma-q4")
        assert store.get("active_profile").startswith("On-device")
        assert store.get("translation_temperature") == .4
        sw.write_setting(store, "model", "local/sigma-q8")
        assert store.get("active_profile") == config["active_profile"]
        assert store.get("translation_temperature") == .27 and store.get("batch_size") == 1
        assert store.get("manual_glossary_prompt3") == config["manual_glossary_prompt3"]
        job = sw.with_mobile_streaming_defaults({"model": "local/sigma-q8", "batch_translation": True,
                                                "enable_parallel_extraction": True})
        assert not job["batch_translation"] and not job["enable_parallel_extraction"]
        assert job["batch_size"] == 1
        assert sw.default_batch_size("local/sigma-q8") == 1
        assert sw.default_batch_size("local/sigma-q4") == 1
        service.save_parallel_requests(True)
        job = sw.with_mobile_streaming_defaults({"model": "local/sigma-q8", "batch_translation": True,
                                                "enable_parallel_extraction": True, "batch_size": 3,
                                                "extraction_workers": 2})
        assert job["batch_translation"] and job["enable_parallel_extraction"]
        assert job["batch_size"] == 3 and job["extraction_workers"] == 2
        sw.write_setting(store, "model", "gpt-6")
        assert store.get("batch_size") == 6  # cloud setting survives the Q8/Q4 switches
        # Existing custom-HF Q8 users may still have an old on-device profile record.
        sw.write_setting(store, "model", "local/sigma-q4")
        store.set("model", "local/sigma-q8")
        sw.write_setting(store, "model", "local/sigma-q8")
        assert store.get("translation_temperature") == .27
        assert store.get("active_profile") == config["active_profile"]
    finally:
        store._saver.close()


def test_sigma_welcome_uses_normal_temperature_and_glossary(tmp_path, monkeypatch):
    from test_local_context import _welcome
    from glossarion_mobile.ui.screens.welcome_flow import WelcomeFlow, DEFAULT_GLOSSARY_MODE
    service = make_service(tmp_path)
    monkeypatch.setattr(svc, "default_service", lambda: service)
    welcome = _welcome(service, "local/sigma-q8")
    welcome._saved = lambda key, default=None: .27 if key == "translation_temperature" else default
    assert welcome.shown_temperature() == .27
    flow = WelcomeFlow()
    flow.default_mode_for_model("local/sigma-q4")
    assert flow.glossary_mode == "no_glossary"
    flow.default_mode_for_model("local/sigma-q8")
    assert flow.glossary_mode == DEFAULT_GLOSSARY_MODE


def test_sigma_engine_does_not_reintroduce_small_model_sampling(tmp_path):
    from test_local_llm_script_ban import _FakeLlama
    service = make_service(tmp_path)
    native = engine.LlamaEngine()
    native.llm, native.path, native.n_ctx = _FakeLlama(), "model.gguf", 24576
    params = service._params({"max_tokens": 8, "temperature": .23}, 100, stem="sigma-q8")
    list(native.chat([{"role": "user", "content": "hi"}], params, lambda: False))
    kwargs = native.llm.kwargs[-1]
    assert kwargs["temperature"] == .23
    assert kwargs["top_p"] == .95 and kwargs["top_k"] == 40 and kwargs["min_p"] == .05
    assert kwargs["presence_penalty"] == 0 and "logits_processor" not in kwargs
    assert native.llm.dry_sampling["multiplier"] == 0


def test_sigma_answer_not_trimmed_when_forced_truncation_is_off(tmp_path, monkeypatch):
    from unified_api_client import UnifiedClient
    from types import SimpleNamespace
    service = make_service(tmp_path)
    monkeypatch.setattr(svc, "default_service", lambda: service)
    client = UnifiedClient.__new__(UnifiedClient)
    client.model = "local/sigma-q8"
    monkeypatch.setattr(client, "_get_thread_local_client", lambda: SimpleNamespace())
    text = "This phrase is repeated. " * 50
    assert client._trim_forced_truncation(text, "length", "glossary") == (text, "length")


def test_active_sigma_reopens_quant_picker_in_both_model_menus(tmp_path):
    from glossarion_mobile.ui.screens.local_models_panel import LocalModelsPanel
    from glossarion_mobile.ui.sheets.model_sheet import ModelSheet, SheetEnv
    from test_local_models_panel import _find
    service = make_service(tmp_path)
    rows = presets.load_presets(service.prefs)
    rows[3]["file"] = presets.SIGMA_Q8_FILE
    presets.save_presets(service.prefs, rows)
    calls = []
    async def choose(index):
        calls.append(index)
    panel = LocalModelsPanel(service=service, prefs=service.prefs, current_model=lambda: "local/sigma-q8",
                             spawn=asyncio.run)
    panel.get_preset = choose
    panel.build()
    button = _find(panel.preset_column, "local-preset-action-3")
    assert button.content == "Choose quant" and not button.disabled
    button.on_click(None)
    panel.dispose()
    sheet = ModelSheet(current_model="local/sigma-q8", models=["local/sigma-q8"], tab="ondevice",
                       env=SheetEnv(prefs=service.prefs, local_engine=service))
    sheet._spawn = asyncio.run
    sheet.ondevice_flow().get_preset = choose
    button = _find(sheet.ondevice_list, "ondevice-preset-3-action")
    assert button.content == "Choose quant" and not button.disabled
    button.on_click(None)
    assert calls == [3, 3]


def test_q8_controls_save_apply_reset_and_stay_separate(tmp_path):
    from glossarion_mobile.ui.screens.local_models_panel import LocalModelsPanel
    service = make_service(tmp_path)
    service.save_sampling({"top_k": 9, "dry_multiplier": 1.2}, stem="sigma-q4")
    selected = ["local/sigma-q8"]
    panel = LocalModelsPanel(service=service, prefs=service.prefs, current_model=lambda: selected[0])
    panel.build()
    try:
        assert not panel.sampling_actions.disabled
        assert panel.sampling_reset.content == "Model defaults"
        assert panel.sampling_fields["top_k"].value == "40"
        assert not panel.mark_glossary_complete_switch.value
        panel.sampling_fields["top_k"].value = "60"
        panel.sampling_fields["dry_multiplier"].value = "0.4"
        assert panel.save_sampling() is None
        panel.block_cjk_switch.value = True
        assert panel.save_block_cjk()
        panel.forced_truncation_switch.value = True
        panel.forced_truncation_seconds.value = "9"
        assert panel.save_forced_truncation() == (True, 9)
        panel.mark_glossary_complete_switch.value = True
        panel.mark_glossary_complete_switch.on_change(None)
        body = {"temperature": .2, "glossarion_task": "translation", "target_language": "English",
                "glossarion_allow_kaomoji": True, "glossarion_whitelist_source": True}
        params = service._params(body, 200, stem="sigma-q8")
        assert params["top_k"] == 60 and params["dry_multiplier"] == .4
        assert params["ban_scripts"] and params["forced_truncation_seconds"] == 9
        assert params["ban_allow_kaomoji"] and params["ban_source_whitelist"]
        assert params["temperature"] == .2
        assert service._params({"glossarion_task": "glossary"}, 200, stem="sigma-q8")["forced_truncation_finish"] == "stop"
        fresh = make_service(tmp_path, service.prefs)
        assert fresh.sampling("sigma-q8")["top_k"] == 60
        assert fresh.block_cjk("sigma-q8") and fresh.mark_glossary_complete("sigma-q8")
        assert panel.reset_sampling() is None
        assert service.sampling("sigma-q8") == svc.DEFAULT_SIGMA_SAMPLING
        assert service.sampling("sigma-q4")["top_k"] == 9
        selected[0] = "local/sigma-q4"
        panel.render()
        assert panel.sampling_fields["top_k"].value == "9"
        assert panel.sampling_reset.content == "Qwen defaults"
        selected[0] = "local/sigma-q8"
        panel.render()
        assert panel.sampling_fields["top_k"].value == "40"
        assert panel.block_cjk_switch.value
    finally:
        panel.dispose()
