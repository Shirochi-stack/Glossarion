"""Device limits, picker persistence and mounted response-card updates."""
import asyncio
import struct
import types
from pathlib import Path

import flet as ft
import pytest

from local_llm_fakes import FakePrefs
from test_local_llm_server import make_service
from glossarion_mobile.services.local_llm import engine
from glossarion_mobile.services.local_llm.model_limits import gpu_layer_limit
from glossarion_mobile.ui.screens.local_models_panel import LocalModelsPanel, usage_color
from glossarion_mobile.ui.sheets.model_sheet import ModelSheet, SheetEnv
from test_run_options_resume import TC, desktop_store_cls, isolated_env


def gguf(path, blocks=32):
    def text(value):
        raw = value.encode()
        return struct.pack('<Q', len(raw)) + raw
    data = b'GGUF' + struct.pack('<IQQ', 3, 0, 3)
    data += text('tokenizer.ggml.tokens') + struct.pack('<IIQ', 9, 8, 2) + text('one') + text('two')
    data += text('general.architecture') + struct.pack('<I', 8) + text('test')
    data += text('test.block_count') + struct.pack('<II', 4, blocks)
    path.write_bytes(data)


@pytest.mark.parametrize('cpus,auto', [(1, 1), (2, 1), (4, 2), (8, 4)])
def test_threads_follow_cpu_affinity_and_clamp_saved_values(tmp_path, monkeypatch, cpus, auto):
    monkeypatch.setattr(engine.os, 'sched_getaffinity', lambda _pid: set(range(cpus)), raising=False)
    service, _ = make_service(tmp_path)
    assert service.thread_limit() == cpus and service.threads() == auto
    assert service.save_settings(2048, cpus)[1] is None
    assert service.save_settings(2048, cpus + 1)[1]
    monkeypatch.setattr(engine.os, 'sched_getaffinity', lambda _pid: {0}, raising=False)
    assert service.threads() == 1


def test_gguf_layers_are_detected_before_loading_and_cache_follows_replacement(tmp_path):
    path = tmp_path / 'model.gguf'
    gguf(path, 32)
    assert gpu_layer_limit(str(path)) == 33
    gguf(path, 40)
    assert gpu_layer_limit(str(path)) == 41
    path.write_bytes(b'GGUF' + b'\0' * 30)
    assert gpu_layer_limit(str(path)) is None


@pytest.mark.parametrize('used,total,color', [(5,10,None),(6,10,ft.Colors.AMBER),
                                             (9,10,ft.Colors.AMBER),(10,10,ft.Colors.ERROR),
                                             (4,8,None),(5,8,ft.Colors.AMBER),(8,8,ft.Colors.ERROR)])
def test_warning_thresholds(used,total,color):
    assert usage_color(used,total) == color


def test_controls_use_selected_model_layers_and_cpu_warning_colors(tmp_path, monkeypatch):
    monkeypatch.setattr(engine.os, 'sched_getaffinity', lambda _pid: set(range(8)), raising=False)
    service, _ = make_service(tmp_path, gpu_backend='OpenCL')
    selected = ['local/small']
    for stem, blocks in [('small', 9), ('large', 39)]:
        path = Path(service.registry.file_path(stem))
        path.parent.mkdir(parents=True, exist_ok=True)
        gguf(path, blocks)
        service.registry.add(stem, {'file': path.name, 'size': path.stat().st_size})
    panel = LocalModelsPanel(service=service, current_model=lambda: selected[0])
    panel.build()
    panel.gpu_switch.value = True
    panel.save_gpu()
    assert panel.gpu_layers_slider.max == 10 and not panel.gpu_layers_slider.disabled
    for value, color in [(5,None),(6,ft.Colors.AMBER),(9,ft.Colors.AMBER),(10,ft.Colors.ERROR)]:
        panel.gpu_layers_slider.value = value
        panel._gpu_layers_changed()
        assert panel.gpu_layers_slider.active_color == color
    assert panel.save_gpu_layers() == -1
    selected[0] = 'local/large'
    panel.render_engine(service.snapshot())
    assert panel.gpu_layers_slider.max == panel.gpu_layers_slider.value == 40
    assert panel.gpu_layers_label.value == 'GPU layers: 40 / 40'
    for value, color in [('4',None),('5',ft.Colors.AMBER),('8',ft.Colors.ERROR),('',None)]:
        panel.threads_field.value = value
        panel._threads_changed()
        assert panel.threads_field.border_color == color
    selected[0] = 'local/not-downloaded'
    panel.render_engine(service.snapshot())
    assert panel.gpu_layers_slider.disabled and panel.gpu_layers_slider.max == 1
    panel.dispose()


def test_picker_remembers_ondevice_when_reopened_from_model_entry_point(tmp_path):
    service, _ = make_service(tmp_path)
    prefs = FakePrefs()
    env = SheetEnv(prefs=prefs, local_engine=service)
    first = ModelSheet(env=env)
    first._on_tab(types.SimpleNamespace(control=types.SimpleNamespace(selected=['ondevice'])))
    assert ModelSheet(env=env, tab='model').tab == 'ondevice'
    assert ModelSheet(env=env, tab='profile').tab == 'profile'
    prefs.set('model_picker_tab', ['invalid'])
    assert ModelSheet(env=env).tab == 'model'


def test_settings_update_a_panel_after_its_card_is_rebuilt(tmp_path, monkeypatch):
    from test_run_options_resume import TC, _SettingsCtx
    from test_bootstrap import _fake_session
    from flet_glossarion_native import TapAway
    from glossarion_mobile.ui.components import tap_away
    from glossarion_mobile.ui.chat.plan_card import RunOptionsPanel
    monkeypatch.setattr(tap_away, '_region_class', TapAway)

    async def scenario():
        store, schema = TC._config_store(tmp_path)
        conn, session = _fake_session('android')
        panel = RunOptionsPanel(_SettingsCtx(store, schema), key='resume-options')
        def card(rows):
            return ft.Container(key='response', content=ft.Column(
                [ft.Text(r) for r in rows] + [tap_away.closes_on_tap_away(panel.control)]))
        slot = ft.Column([card(['first'])])
        session.page.add(slot)
        slot.controls = [card(['new', 'rows'])]
        slot.update()
        try:
            before = len(conn.messages)
            store.set_many({'batch_translation': True, 'batch_size': 7,
                            'translation_temperature': 0.8, 'delay': 3})
            assert 'Batch 7' in panel.summary_text.value
            assert 'Temp 0.8' in panel.summary_text.value
            assert 'Delay 3' in panel.summary_text.value
            assert panel.tiles['batch_size'].field.value == '7'
            assert len(conn.messages) > before, 'settings changed only in Python, not on the client'
        finally:
            panel.release()
            store._saver.close()
    asyncio.run(scenario())


def test_library_job_resume_applies_the_visible_run_options():
    from glossarion_mobile.ui.chat.chat_view import ChatView
    received = []
    original = {'input_files': ['book.epub'], 'config_overrides': {'model': 'local/test', 'batch_size': 2}}
    jobs = types.SimpleNamespace(
        snapshot_of=lambda jid: types.SimpleNamespace(spec=types.SimpleNamespace(params=original)),
        resume=lambda jid, params=None: received.append((jid, params)) or 'resumed')
    view = types.SimpleNamespace(env=types.SimpleNamespace(jobs=jobs),
        _take_resume_config=lambda item: ({'batch_size': 7}, 'job:library'),
        notify=lambda text: pytest.fail(text), transcript=types.SimpleNamespace(follow_tail=False),
        render_transcript=lambda **kw: None)
    assert ChatView._book_job_action(view, 'resume', 'library', types.SimpleNamespace(index=2))
    assert received == [('library', {'input_files': ['book.epub'],
                                    'config_overrides': {'model': 'local/test', 'batch_size': 7}})]
    assert original['config_overrides']['batch_size'] == 2


def test_resume_options_exist_without_direct_text_run_metadata(tmp_path):
    from test_run_options_resume import TC, _SettingsCtx
    from glossarion_mobile.ui.chat.chat_view import ChatView
    store, schema = TC._config_store(tmp_path)
    view = types.SimpleNamespace(env=_SettingsCtx(store, schema), cid='chat', run_panels={},
                                  _refresh_range_chips=lambda: None)
    try:
        item = types.SimpleNamespace(index=2)
        panel = ChatView._resume_panel(view, item, {'input_files': ['book.epub']}, job_id='library')
        assert panel is not None and not panel.only
        assert ('chat', 'resume', 2, 'job:library') in view.run_panels
        panel.release()
    finally:
        store._saver.close()


def test_response_card_follows_defaults_and_keeps_options_for_library_job(desktop_store_cls, isolated_env):
    from test_owner_card_settings import _ended_run
    from glossarion_mobile.ui.chat.cards import JobCard
    from glossarion_mobile.services.jobs import JobSnapshot, JobSpec, JobState

    async def scenario():
        chat = TC._chat_view(desktop_store_cls, isolated_env)
        view = chat.view
        try:
            await TC._settle(chat.calls)
            params = {'run': {'temp_root': str(isolated_env / 'root')}, 'user_index': 2}
            view.env.runs.run_for = lambda cid: _ended_run('stopped', params)
            view._run_turn = lambda run: 2
            view.render_transcript()
            view.env.store.set('chapter_range', '3-9')
            await TC._settle(chat.calls)
            card = next(c for c in TC._cards(view, JobCard) if c.title_text.value == 'book.epub')
            range_chip = next(c for c in card.setup_actions.controls if c.key == 'result-choose')
            assert '3-9' in range_chip.content.replace('\u2013', '-')

            other = JobSnapshot(id='library', spec=JobSpec('translate', 'Book',
                params={'input_files': ['book.epub'], 'config_overrides': {'batch_size': 2}}),
                state=JobState.CANCELLED, created=1.0, finished=2.0)
            view._book_job_elsewhere = lambda item, own: other
            view.render_transcript()
            card = next(c for c in TC._cards(view, JobCard) if c.title_text.value == 'book.epub')
            assert card.offers_resume and card.options_box.visible and card.options_box.controls
            panels = [p for k,p in view.run_panels.items() if k[1] == 'resume']
            assert len(panels) == 1
            panel = panels[0]
            panel.switch.value = True
            panel._on_switch()
            assert panel.tiles['batch_size'].apply(6)
            received = []
            view.env.jobs.snapshot_of = lambda jid: other
            view.env.jobs.resume = lambda jid, params=None: received.append(params) or 'new'
            view._on_job_action('resume', TC._job_item(chat.adapter))
            assert received[0]['config_overrides']['batch_size'] == 6
        finally:
            TC._close(chat)
    asyncio.run(scenario())

