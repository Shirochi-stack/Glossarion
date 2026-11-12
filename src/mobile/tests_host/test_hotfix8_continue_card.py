"""Compact continuation cards preserve jobs and resume with visible settings."""
import asyncio
import types

import flet as ft
import pytest

from test_run_options_resume import TC, _settle, _stop, _resume, desktop_store_cls, isolated_env
from glossarion_mobile.ui.chat.cards import CardPhase, JobCard


@pytest.mark.parametrize('phase,status', [('stopped','Stopped'), ('failed','Failed'),
                                         ('interrupted','Interrupted'), ('done','Done · 2 failed')])
def test_continue_card_has_only_translation_and_cancel_actions(phase, status):
    card = JobCard(phase=CardPhase(phase), on_action=lambda _: None)
    card.set_phase(CardPhase(phase), status=status)
    card.set_requests([{'content':'old answer','complete':True}])
    card.set_report('old report')
    card.set_outputs([('old.epub','epub')])
    card.set_failed(2)
    assert card.state_text.value == 'Continue Translation?'
    assert not card.result_details.visible
    assert [b.content for b in card.buttons.controls] == ['Translate','Cancel']
    assert isinstance(card.buttons.controls[0], ft.FilledButton)
    assert isinstance(card.buttons.controls[1], ft.TextButton)
    card.set_phase(CardPhase('done'), status='Done')
    assert card.result_details.visible and 'export' in card.action_buttons


def test_cancel_dismisses_only_the_response_and_survives_reopen(desktop_store_cls, isolated_env):
    from test_owner_card_settings import _ended_run
    async def scenario():
        chat = TC._chat_view(desktop_store_cls, isolated_env)
        view = chat.view
        try:
            await _settle(chat.calls)
            params = {'run':{'temp_root':str(isolated_env/'root')}, 'user_index':2}
            view.env.runs.run_for = lambda _: _ended_run('stopped', params)
            view._run_turn = lambda _:2
            view.render_transcript()
            await _settle(chat.calls)
            card = next(c for c in TC._cards(view, JobCard) if c.title_text.value == 'book.epub')
            assert card.state_text.value == 'Continue Translation?'
            assert card.plan_box.visible and card.resume_chips.visible and card.options_box.visible
            assert {c.key for c in card.resume_chips.controls} >= {'result-model','result-output'}
            assert not any(c.key in ('result-range','plan-destination') for c in card.resume_chips.controls)
            progress = next(c for c in card.setup_actions.controls if c.key == 'result-progress')
            assert isinstance(progress, ft.TextButton)
            opened=[];view._open_progress=lambda item:opened.append(item.index)
            progress.on_click(None);assert opened == [2]
            # A late library/share status update must not hide the compact setup.
            card.set_action_reason('library',None)
            assert card.plan_box.visible and not card.result_details.visible
            before=list(chat.adapter.messages(view.cid));submitted=len(chat.jobs.submitted)
            card.action_buttons['dismiss_resume'].on_click(None)
            assert list(chat.adapter.messages(view.cid)) == before
            assert len(chat.jobs.submitted) == submitted
            assert chat.adapter.meta(view.cid)['dismissed_response_cards']
            view.load_chat(view.cid)
            assert not any(c.title_text.value == 'book.epub' for c in TC._cards(view, JobCard))
        finally:TC._close(chat)
    asyncio.run(scenario())


def test_resume_output_mode_is_applied_without_changing_workspace(desktop_store_cls, isolated_env):
    async def scenario():
        chat=TC._chat_view(desktop_store_cls,isolated_env);view=chat.view
        try:
            await _settle(chat.calls)
            path=isolated_env/'novel.epub';path.write_bytes(b'PK\x03\x04')
            view.attach_file(str(path));view.begin_send();view.start_plan();await _settle(chat.calls)
            original=chat.jobs.submitted[-1].params
            await _stop(chat)
            card=next(c for c in TC._cards(view,JobCard) if c.offers_resume)
            choose=[]
            view.open_mode_options=lambda mode,on_switch:choose.append(on_switch)
            next(c for c in card.resume_chips.controls if c.key=='result-output').on_click(None)
            choose[0]('vision')
            config=await _resume(chat)
            resumed=chat.jobs.submitted[-1].params
            assert config['output_mode']=='vision' and config['enable_image_translation']
            assert resumed['run']['output_mode']==resumed['options']['output_mode']=='vision'
            assert resumed['run']['temp_root']==original['run']['temp_root']
            assert resumed['input_path']==original['input_path']
        finally:TC._close(chat)
    asyncio.run(scenario())
