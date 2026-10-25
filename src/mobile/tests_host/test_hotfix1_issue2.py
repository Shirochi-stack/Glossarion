"""Acceptance test for the owner's hotfix 1 report #2 (phone, 2026-10): "Clicking refine doesn't hide the
response card, which results in an issue where the user clicks yes during refinement, so we end up having
translation and refinement requests take place at the same time. The refinement is meant for the translation,
so it doesn't make sense for it to start before it. Solve this hiding the response card during glossary
refinement."

What the owner saw is replayed end to end on the REAL app objects (``test_devfix_issue12``'s harness): the app on
a fake Flet session, the real JobService on the app's UiDispatcher, ``HeadlessOwner``, the shared translation
pipeline (Balanced glossary pre-pass + the approval gate) and the shared manual refinement
(``glossary_progress_core.run_manual_glossary_refinement`` through ``JobService.refine_at_gate``), the real
ChatFeature / ChatRuns / ChatView. Only the model is the offline fake OpenAI server on 127.0.0.1 (a subclass that
recognises the refinement prompt, answers it with a refined glossary and records every prompt); the E2E network
guard refuses every non-loopback connection for the whole test. The owner's taps go through the UI: New chat,
＋ › Files (an EPUB), Send, Start, then the approval card's ✨ Refine and, once it is done, ✓ Yes.

``test_refine_hides_the_approval_card_and_the_translation_waits_for_it``:

1. The gate: the approval card is on screen with ✏️ Edit, ✨ Refine, ✓ Yes and ■ No (the client was sent them
   visible); no translation request was sent yet.
2. ✨ Refine, with the fake model slowed: the model's own ``on_request`` hook holds the first refinement request
   (it parks at the model, the refinement cannot end), so every check below happens while the refinement is
   provably still running - no sleeps decide anything. The card shows the refining state (the ring and
   "Refining the glossary…", the "REFINING THE GLOSSARY…" label) and NO Edit / Refine / Yes / No / Always
   accept, both in the Python tree and in what the session last sent to the client; the running card reads
   "Refining the glossary…", the composer is blocked with the refining reason, JobService reports
   ``refining`` while the job stays RUNNING at the same question.
3. Every way an answer could still arrive is refused and changes nothing: a ✓ Yes / ■ No / Always accept /
   ✏️ Edit tap already on its way from the client (dispatched to the hidden buttons), the chat's own answer path
   (``ChatView._answer_glossary`` / ``_always_accept_glossary``), the run controller
   (``ChatRuns.answer_glossary``), a direct ``JobService.answer`` (the notification / Book page / Jobs sheet
   path) and a second ``refine_at_gate``. No translation request reached the model. The app then goes to the
   background and comes back, and the owner opens another chat and comes back (the approval card is rebuilt):
   still the refining state, still no answers.
4. The model answers (slowed by a reply delay; one refinement request per entry type, so there are two): the
   refinement ends, the card brings Edit / ✓ Yes / ■ No back, Refine reads "✨ Refined" (disabled), the preview
   shows the refined file (one entry dropped, one renamed), the running card reads "Waiting for your glossary
   decision" again.
5. ✓ Yes starts the translation; the run ends "Done". Every translation request the model received arrived
   after the last refinement request had finished (by arrival order and by time), no refinement request came
   after ✓ Yes, and the translation used the refined glossary (its prompts carry the renamed entry, never the
   dropped one or the old name). No UI task failed on the way.

Without the fix (the base commit's package) the test fails at step 2: Edit / Refine / ✓ Yes / ■ No / Always
accept stay on screen while the refinement runs, and a ✓ Yes tapped then sends a translation request while the
refinement's request is still pending - the owner's report.

Real data stays untouched: the app runs in the harness's temp FLET_APP_STORAGE_* dirs (HOME, OUTPUT_DIRECTORY,
GLOSSARION_LIBRARY_DIR, GLOSSARION_DATA_DIR, CONFIG_FILE asserted inside tmp), USERPROFILE / APPDATA /
LOCALAPPDATA point into tmp and HTTP logging is off. A failing scenario force-stops its job before the app closes.

Run from src/mobile with the mobile env (on the dev box ``.venv``, Python 3.12, keep the shell PYTHONPATH)::

    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic \\
        tests_host/test_hotfix1_issue2.py
"""

from __future__ import annotations

import asyncio
import contextlib
import csv
import importlib.util
import io
import logging
import sys
import threading
import time
from pathlib import Path

MOBILE_DIR = Path(__file__).resolve().parents[1]
for entry in (str(MOBILE_DIR / "app"), str(MOBILE_DIR.parent), str(MOBILE_DIR / "tests")):
    if entry not in sys.path:
        sys.path.insert(0, entry)


def _load(alias: str, file_name: str):
    spec = importlib.util.spec_from_file_location(alias, Path(__file__).with_name(file_name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


I12 = _load("_glossarion_devfix12_hotfix1_issue2", "test_devfix_issue12.py")  # the real-pipeline harness
pytestmark = I12.pytestmark
storage = I12.storage
app_env = I12.app_env
iso = I12.iso
offline = I12.offline

CHAPTERS = 8
RUN_TIMEOUT = 300.0
#: the slowed model: every refinement reply waits this long once the hold is released
REFINE_DELAY = 0.8
#: the system prompt of every refinement request (``glossary_refinement.DEFAULT_GLOSSARY_REFINEMENT_SYSTEM_PROMPT``)
REFINE_MARK = "You are refining an already extracted translation glossary"
#: the glossary the fake model extracts from the test book (both raw names are in every chapter)
KEPT_RAW, OLD_NAME, REFINED_NAME = "마왕성", "Demon King's Castle", "Demon Lord's Castle"
DROPPED_RAW, DROPPED_NAME = "검", "Holy Blade"
CHARACTER_RAW, CHARACTER_NAME = "문", "Moon"
#: one refinement request per entry type (the shared ``glossary_refinement_chunking_mode`` setting), so the
#: refinement sends more than one request and "the last refinement request" is a real check
REFINEMENT_SETTINGS = {"glossary_refinement_chunking_mode": "separate"}
WAITING = "Waiting for your glossary decision"


# ==========================================================================
# The fake model that also refines
# ==========================================================================


def _server():
    from glossarion_mobile.diagnostics.fake_llm_server import FakeLLMServer, GlossaryEntry, _content_text

    class RefiningModel(FakeLLMServer):
        """The E2E's fake model, plus: a request with the refinement system prompt is kind ``refinement`` and is
        answered with the refined rows (``DROPPED_RAW`` removed, ``KEPT_RAW`` renamed ``REFINED_NAME``; header-less
        CSV in the request's own columns, as the prompt asks). Every request's whole prompt is kept in
        ``prompts`` (record id -> text)."""

        def __init__(self) -> None:
            super().__init__()
            self.prompts: dict = {}

        @staticmethod
        def _text(payload: dict) -> str:
            return "\n".join(_content_text(m.get("content")) for m in (payload.get("messages") or [])
                             if isinstance(m, dict))

        def _classify(self, payload: dict) -> str:
            if REFINE_MARK in self._text(payload):
                return "refinement"
            return super()._classify(payload)

        def _reply_for(self, payload: dict, record) -> str:
            full_text = self._text(payload)
            with self._lock:
                self.prompts[record.id] = full_text
            if record.kind != "refinement":
                return super()._reply_for(payload, record)
            messages = [m for m in (payload.get("messages") or []) if isinstance(m, dict)]
            last_user = next((_content_text(m.get("content")) for m in reversed(messages) if m.get("role") == "user"), "")
            record.prompt_chars = len(full_text)
            record.preview = " ".join(last_user.split())[:160]
            out = io.StringIO()
            writer = csv.writer(out, lineterminator="\n")
            for row in csv.reader(io.StringIO(last_user)):
                if len(row) < 3 or row[1].strip() == DROPPED_RAW:
                    continue
                if row[1].strip() == KEPT_RAW:
                    row = row[:2] + [REFINED_NAME] + row[3:]
                writer.writerow(row)
            return out.getvalue()

    server = RefiningModel()
    server.glossary = (GlossaryEntry("character", CHARACTER_RAW, CHARACTER_NAME, "Male", "The gatekeeper"),
                       GlossaryEntry("term", KEPT_RAW, OLD_NAME, "", "The demon's keep"),
                       GlossaryEntry("term", DROPPED_RAW, DROPPED_NAME, "", "A blade"))
    server.set_delay("refinement", REFINE_DELAY)  # the slowed model
    return server


class RefineHold:
    """The fake model's own request hook: the first refinement request makes the model hold, so that request
    parks (and the refinement cannot end) until ``release``. ``record`` is that request."""

    def __init__(self, server) -> None:
        self.server = server
        self.record = None
        self._lock = threading.Lock()
        server.on_request.append(self)

    def __call__(self, record) -> None:
        if record.kind != "refinement":
            return
        with self._lock:
            if self.record is not None:
                return
            self.record = record
        self.server.hold()  # runs before the request waits: this request parks

    def parked(self) -> bool:
        record = self.record
        return record is not None and record.parked and self.server.holding and self.server.parked >= 1


# ==========================================================================
# What the client was sent
# ==========================================================================


def _chain(control) -> list:
    """``control`` and its ancestors, up to the page."""
    chain, node = [], control
    while node is not None and len(chain) < 400:
        chain.append(node)
        try:
            node = node.parent
        except Exception:
            node = None
    return chain


class ClientView:
    """What the session last SENT the client about the approval card's controls (the phone shows that, not the
    Python tree). Flet sends a control's changed properties when it or one of its ancestors is patched, so after
    every ``Session.patch_control`` each watched control under the patched one records the ``visible`` of every
    node from it up to the patched control. A control is shown on the client when every node of its chain was
    last sent visible (a node never sent since the watch began - an ancestor mounted before the card - counts with
    its current value)."""

    def __init__(self) -> None:
        self.watched = lambda: []
        self.sent: dict = {}
        self.errors: list = []

    @contextlib.contextmanager
    def watching(self):
        from flet.messaging import session as fs

        original = fs.Session.patch_control
        view = self

        def patch_control(session_self, control, *args, **kwargs):
            result = original(session_self, control, *args, **kwargs)
            try:
                view._after(control)
            except Exception as exc:  # the mirror must never break the app's patch
                view.errors.append(f"{type(exc).__name__}: {exc}")
            return result

        fs.Session.patch_control = patch_control
        try:
            yield self
        finally:
            fs.Session.patch_control = original

    def _after(self, patched) -> None:
        for control in self.watched() or ():
            chain = _chain(control)
            if any(node is patched for node in chain):
                for node in chain:
                    self.sent[id(node)] = getattr(node, "visible", True) is not False
                    if node is patched:
                        break

    def sent_once(self, control) -> bool:
        return id(control) in self.sent

    def shown(self, control) -> bool:
        return all(self.sent.get(id(node), getattr(node, "visible", True) is not False) for node in _chain(control))


def _card_controls(card) -> dict:
    """The approval card's answer controls and (where the card has them) its refining state's."""
    if card is None:
        return {}
    names = ("edit_button", "refine_button", "yes_button", "no_button", "always_button", "refining_row", "refine_error")
    return {name: getattr(card, name) for name in names if getattr(card, name, None) is not None}


ANSWERS = ("edit_button", "refine_button", "yes_button", "no_button", "always_button")


# ==========================================================================
# Helpers
# ==========================================================================


def _visible_texts(probe) -> set:
    """Every text of the transcript as the user sees it (the request rows included)."""
    return probe.visible_texts(rows=True)


def _records(server, kind):
    return server.records(kind, status=None)


def _refine_again(service, job_id, path):
    """A second ``refine_at_gate`` while one runs: the refusal's message (None when it was not refused)."""
    try:
        service.refine_at_gate(job_id, path)
    except RuntimeError as exc:
        return str(exc)
    return None


def _names():
    """The fix's words, looked up so that this test also runs (and fails on its checks) against code without it."""
    from glossarion_mobile.ui.chat import cards, chat_view, job_binding

    return {
        "action_required": cards.ACTION_REQUIRED_LABEL,
        "refining_label": cards.REFINING_LABEL,
        "refining_text": getattr(cards, "REFINING_TEXT", "<cards.REFINING_TEXT is missing>"),
        "block": getattr(chat_view, "BLOCK_GLOSSARY_REFINING", "<chat_view.BLOCK_GLOSSARY_REFINING is missing>"),
        "running_line": getattr(job_binding, "GLOSSARY_REFINING_LABEL", "<job_binding.GLOSSARY_REFINING_LABEL is missing>"),
    }


# ==========================================================================
# The owner's flow
# ==========================================================================


def test_refine_hides_the_approval_card_and_the_translation_waits_for_it(iso, offline, monkeypatch, caplog):
    client = ClientView()
    configure = I12._configure

    def configure_with_refinement(app, server, glossary="balanced"):
        configure(app, server, glossary)  # the owner's settings, then the refinement's request split
        app.config_store.set_many(dict(REFINEMENT_SETTINGS))
        app.config_store.flush()
        assert app.config_store.save_error is None, app.config_store.save_error

    monkeypatch.setattr(I12, "_configure", configure_with_refinement)

    async def body(ctx):
        app, server, run, probe = ctx.app, ctx.server, ctx.run, ctx.probe
        view, service, runs = app.chat_view, app.job_service, app.chat_feature.runs
        client.watched = lambda: list(_card_controls(view.approval_card).values())
        words = _names()

        # ---- 1. the gate ---------------------------------------------------------------------------------
        assert await I12._until(lambda: run.awaiting_glossary, RUN_TIMEOUT), probe.describe()
        assert await I12._until(lambda: view.approval_card is not None and probe.on_screen(view.approval_card.yes_button)
                                and view.live_job_card is not None, 30), probe.describe()
        card = view.approval_card
        assert await I12._until(lambda: client.sent_once(card.yes_button), 10), "the approval card never reached the client"
        chain = _chain(card.yes_button)
        assert any(node is card for node in chain), "✓ Yes is not inside the approval card"
        inside = chain[:next(i for i, node in enumerate(chain) if node is card) + 1]
        assert all(client.sent_once(node) for node in inside), "the card's rows were never sent to the client"
        for name in ("edit_button", "refine_button", "yes_button", "no_button"):
            control = getattr(card, name)
            assert probe.on_screen(control) and client.shown(control) and not control.disabled, f"{name} at the gate"
        question = service.pending_question(run.job_id)
        assert question is not None and question.get("id") == (run.question or {}).get("id"), (question, run.question)
        assert server.count("glossary") >= 1 and not _records(server, "translation") and not _records(server, "refinement"), \
            [(r.id, r.kind) for r in server.records(status=None)]
        assert card.info.get("exists") and card.info.get("entries") == 3, card.info
        assert card.refine_button.content == "✨ Refine" and card.action_label.value == words["action_required"]
        assert view.live_job_card.state_text.value == WAITING, view.live_job_card.state_text.value
        before_views = len(probe.page.views)

        # ---- 2. ✨ Refine, the model slowed and holding the refinement's request ------------------------------------
        hold = RefineHold(server)
        await I12._tap(probe, card.refine_button)
        assert await I12._until(hold.parked, 120), \
            ("the refinement's request never reached the model", probe.describe(),
             [(r.id, r.kind, r.status) for r in server.records(status=None)])
        refine_record = hold.record
        await I12._until(lambda: not client.shown(card.yes_button), 5)  # the hide's patch (posted to the UI loop)

        def check_refining(where: str) -> None:
            shown = view.approval_card
            assert shown is not None, f"{where}: the approval card is gone"
            # what the owner saw: the answers are still there while it refines
            on_screen = [name for name in ANSWERS if probe.on_screen(getattr(shown, name))]
            assert not on_screen, f"{where}: {on_screen} on screen while the glossary is refined"
            on_client = [name for name in ANSWERS if client.shown(getattr(shown, name))]
            assert not on_client, f"{where}: the client still shows {on_client} while the glossary is refined"
            texts = _visible_texts(probe)
            for label in ("✓ Yes", "■ No", "✏️ Edit", "✨ Refine", "✨ Refining…", "Always accept"):
                assert label not in texts, f"{where}: {label!r} is visible while it refines"
            # the refining state instead
            row = getattr(shown, "refining_row", None)
            assert row is not None and probe.on_screen(row) and client.shown(row), f"{where}: no refining state on screen"
            assert getattr(shown, "refining", False), where
            assert words["refining_text"] in texts and words["refining_label"] in texts, \
                (where, sorted(t for t in texts if len(t) < 80))
            assert view.live_job_card.state_text.value == words["running_line"], \
                (where, view.live_job_card.state_text.value)
            assert view._chat_block() == words["block"], (where, view._chat_block())
            # the job still waits at the same question, its glossary being refined; nothing was translated
            snap = service.snapshot(run.job_id)
            assert getattr(snap, "refining", False) and snap.state.value == "RUNNING", (where, snap.state)
            pending = service.pending_question(run.job_id)
            assert pending is not None and pending.get("id") == question["id"], (where, pending)
            assert run.live and run.awaiting_glossary and not run.stop_requested, (where, run.state)
            assert refine_record.status == "pending" and server.holding, (where, refine_record.status)
            assert not _records(server, "translation"), f"{where}: a translation request reached the model"

        check_refining("after the Refine tap")

        # ---- 3. every answer that could still arrive is refused ---------------------------------------------------
        # taps already on their way from the client (sent before the buttons were hidden on the phone)
        for name in ("yes_button", "no_button", "always_button", "always_button", "edit_button", "refine_button"):
            await probe.tester._dispatch(getattr(card, name), "click")
            await asyncio.sleep(0)
        assert card.answered is None and not card.always_accepted, (card.answered, card.always_accepted)
        check_refining("after the late taps")
        assert len(probe.page.views) == before_views, "✏️ Edit opened the editor while it refines"
        # the chat's own answer paths
        view._answer_glossary(True)
        view._always_accept_glossary()
        view._answer_glossary(False)
        check_refining("after the chat's answer paths")
        # the run controller and JobService (the notification, Book page and Jobs sheet answer through it)
        assert runs.answer_glossary(view.cid, True) is False, "ChatRuns.answer_glossary took Yes while it refines"
        assert runs.answer_glossary(view.cid, False) is False, "ChatRuns.answer_glossary took No while it refines"
        assert service.answer(question["id"], True) is False, "JobService.answer took Yes while it refines"
        assert service.answer(question["id"], False) is False, "JobService.answer took No while it refines"
        second = await asyncio.wait_for(asyncio.to_thread(_refine_again, service, run.job_id, card.path), 30)
        assert second is not None and "already being refined" in second, second
        prefs = getattr(app, "prefs", None)
        if prefs is not None:
            from glossarion_mobile.ui.chat.direct_text_rules import AUTO_ACCEPT_GLOSSARY_PREF

            assert not prefs.get(AUTO_ACCEPT_GLOSSARY_PREF, False), "Always accept was stored while it refines"
        await asyncio.sleep(0.3)  # anything an answer would have set off (a repaint, a job start) has had its turn
        check_refining("after every refused answer")
        # Home and back to the app (Android lifecycle), then the chat opened again (its cards rebuilt): still hidden
        await probe.lifecycle(*I12.LEAVE)
        assert probe.page.app_visible is False
        await probe.lifecycle(*I12.COME_BACK)
        await asyncio.sleep(1.0)  # the catch-up repaint after the app came back
        check_refining("after the app came back")
        # another chat (the header's New chat), then back to this one (the drawer's chat row: ``_open_chat``)
        cid = view.cid
        await ctx.driver.tap(tooltip="New chat")
        assert await I12._until(lambda: view.cid != cid and view.approval_card is None, 10), "New chat did not open"
        await asyncio.sleep(0.3)  # the other chat's transcript reaches the client
        app._open_chat(cid)
        assert await I12._until(lambda: view.cid == cid and view.approval_card is not None
                                and view.approval_card is not card, 10), "the chat did not come back with its card"
        await asyncio.sleep(0.3)
        check_refining("after switching chats")
        assert service.answer(question["id"], True) is False, "JobService.answer took Yes while it refines"

        # ---- 4. the model answers (still slowed): the refinement ends, the answers come back -----------------------
        server.release()
        assert await I12._until(lambda: view.approval_card is not None and view.approval_card.refine_outcome is not None
                                and not view.approval_card.refining, RUN_TIMEOUT), probe.describe()
        card = view.approval_card
        assert card.refine_outcome == "refined", (card.refine_outcome, card.refine_error.value)
        back = ("edit_button", "yes_button", "no_button")
        assert await I12._until(lambda: all(probe.on_screen(getattr(card, n)) and client.shown(getattr(card, n))
                                            for n in back) and not client.shown(card.refining_row), 10), \
            ("the answers did not come back", {n: (probe.on_screen(c), client.shown(c)) for n, c in _card_controls(card).items()})
        assert not card.yes_button.disabled and not card.no_button.disabled and not card.edit_button.disabled
        refinements = _records(server, "refinement")
        assert len(refinements) >= 2 and all(r.status == "ok" and r.finished for r in refinements), \
            [(r.id, r.status) for r in refinements]  # one request per entry type (characters, terms)
        assert not _records(server, "translation"), "a translation request reached the model before ✓ Yes"
        assert card.refine_button.content == "✨ Refined" and card.refine_button.disabled
        assert card.action_label.value == words["action_required"] and not card.refine_error.visible
        assert card.info.get("entries") == 2, card.info  # the preview reads the refined file
        texts = _visible_texts(probe)
        assert f"{KEPT_RAW} → {REFINED_NAME}" in texts and "✨ Refined" in texts, sorted(t for t in texts if len(t) < 80)
        assert not any(DROPPED_NAME in t or OLD_NAME in t for t in texts)
        refined_text = Path(card.path).read_text(encoding="utf-8-sig")
        assert REFINED_NAME in refined_text and CHARACTER_NAME in refined_text and DROPPED_NAME not in refined_text, \
            refined_text[:400]
        assert await I12._until(lambda: view.live_job_card.state_text.value == WAITING, 10), \
            view.live_job_card.state_text.value
        snap = service.snapshot(run.job_id)
        assert not snap.refining and snap.state.value == "RUNNING" and run.awaiting_glossary
        pending = service.pending_question(run.job_id)
        assert pending is not None and pending.get("id") == question["id"], pending

        # ---- 5. ✓ Yes starts the translation, after the refinement -----------------------------------------------
        yes_mark = server.mark()
        await I12._tap(probe, card.yes_button)
        await I12._finished(app, run)
        assert run.state == "done" and run.error is None, (run.state, run.error)
        assert probe.job()["state"] == "DONE", probe.job()
        translations = _records(server, "translation")
        refinements = _records(server, "refinement")
        assert translations, "✓ Yes did not start the translation"
        assert all(r.status == "ok" for r in translations), [(r.id, r.status) for r in translations if r.status != "ok"]
        last_refinement_id = max(r.id for r in refinements)
        last_refinement_end = max(r.finished for r in refinements)
        early = [(r.id, r.started) for r in translations if r.id < last_refinement_id or r.started < last_refinement_end]
        assert not early, (f"translation requests sent before the last refinement request finished "
                           f"(it finished at {last_refinement_end}): {early}")
        assert min(r.id for r in translations) > yes_mark, "a translation request was sent before ✓ Yes"
        late = [r.id for r in refinements if r.id > yes_mark]
        assert not late, f"refinement requests after ✓ Yes: {late}"
        # the translation used the refined glossary
        prompts = [server.prompts.get(r.id, "") for r in translations]
        with_glossary = [p for p in prompts if KEPT_RAW in p and REFINED_NAME in p and CHARACTER_NAME in p]
        assert len(with_glossary) >= CHAPTERS, (len(with_glossary), len(prompts))
        assert not any(OLD_NAME in p or DROPPED_NAME in p for p in prompts), \
            "a translation prompt carries the unrefined glossary"
        assert not client.errors, client.errors

    with _server() as server, client.watching():
        asyncio.run(I12._owner_flow(iso, server, body, chapters=CHAPTERS))
    I12._assert_offline(offline)
    # no UI task failed on the way (a card that could not be updated leaves it hidden with no error on screen)
    failed = [f"{r.name}: {r.getMessage()} {r.exc_text or ''}"[:600] for r in caplog.records
              if r.levelno >= logging.ERROR and r.name.startswith("glossarion")]
    assert not failed, failed


def test_a_reload_of_the_open_chat_at_the_gate_keeps_the_approval_card_answerable(iso, offline, caplog):
    """Found by the issue-2 acceptance (also on the base 01ae89d6): ``ChatView.load_chat`` of the open chat while
    its approval card is mounted built a new card under the old card's key, which Flet 1.0.3 diffs as a frozen
    value - its ✓ Yes then did nothing (and a refinement's end could not show Yes again: "Frozen controls cannot
    be updated"). Each approval card now gets its own key: the rebuilt card is not frozen and ✓ Yes answers."""

    async def body(ctx):
        app, server, run, probe = ctx.app, ctx.server, ctx.run, ctx.probe
        view = app.chat_view
        assert await I12._until(lambda: run.awaiting_glossary, RUN_TIMEOUT), probe.describe()
        assert await I12._until(lambda: view.approval_card is not None and probe.on_screen(view.approval_card.yes_button),
                                30), probe.describe()
        old = view.approval_card
        await asyncio.sleep(0.3)  # the card reached the client
        view.load_chat(view.cid)
        assert await I12._until(lambda: view.approval_card is not None and view.approval_card is not old
                                and probe.on_screen(view.approval_card.yes_button), 30), probe.describe()
        card = view.approval_card
        frozen = [name for name in ("yes_button", "answer_row", "edit_row")
                  if getattr(card, name, None) is not None and hasattr(getattr(card, name), "_frozen")]
        assert not hasattr(card, "_frozen") and not frozen, ("the rebuilt approval card is frozen", frozen)
        await probe.tester._dispatch(card.yes_button, "click")
        assert await I12._until(lambda: not run.awaiting_glossary, 30), "✓ Yes on the rebuilt card did nothing"
        assert card.key != old.key, (card.key, old.key)  # a key per build
        await I12._finished(app, run)
        assert run.state == "done" and run.error is None, (run.state, run.error)
        assert _records(server, "translation"), "✓ Yes did not start the translation"

    with _server() as server:
        asyncio.run(I12._owner_flow(iso, server, body, chapters=CHAPTERS))
    I12._assert_offline(offline)
    failed = [f"{r.name}: {r.getMessage()} {r.exc_text or ''}"[:600] for r in caplog.records
              if r.levelno >= logging.ERROR and r.name.startswith("glossarion")]
    assert not failed, failed
