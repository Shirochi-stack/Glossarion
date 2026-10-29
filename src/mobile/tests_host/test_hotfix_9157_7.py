"""v9.15.7 hotfix 7, owner (2026-10-29, phone):

* "Why is there image support being downloaded to the on device model when image is not always used.
  Shouldn't it only be download when it is requested?" - a preset tap downloads the model only; the first
  image sent to it asks for its image support (``OnDeviceFlow.ensure_image_support``).
* "The requests dropdown is being shown by default while it is empty" - hidden until its first card.
* "rename the job name after the metadata.json gets translated" - ``JobService._take_translated_title``.
* (Stop, then Resume showed the job under Running and Finished) - see test_hotfix1_issue5.py.

Run from src/mobile with the mobile venv::

    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_hotfix_9157_7.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import threading
import types
from pathlib import Path

import pytest

_LP_SPEC = importlib.util.spec_from_file_location("_glossarion_hf9157_7_panel",
                                                  Path(__file__).with_name("test_local_models_panel.py"))
LP = importlib.util.module_from_spec(_LP_SPEC)
_LP_SPEC.loader.exec_module(LP)

pytestmark = LP.pytestmark


@pytest.fixture()
def fake_hf(monkeypatch):
    server = LP.fakes.FakeHF()
    server.add(LP.REPO, {LP.FILE: LP.PAYLOAD, "mmproj-F16.gguf": b"projector"})
    monkeypatch.setattr(LP.hf, "HF_ENDPOINT", server.url)
    yield server
    server.close()


def _flow(tmp_path, answers):
    from glossarion_mobile.ui.screens.local_models_panel import OnDeviceFlow

    prefs = LP.fakes.FakePrefs({LP.presets.PREF_KEY: LP._tiny_presets()[:1]})
    service, _engine = LP._engine_service(tmp_path, prefs)
    asked = []

    def confirm(title, body, label):
        asked.append((title, body))
        return answers.pop(0) if answers else True

    return OnDeviceFlow(service=service, prefs=prefs, confirm=confirm), service, asked


def test_a_preset_tap_downloads_the_model_without_its_image_support(tmp_path, fake_hf):
    flow, service, asked = _flow(tmp_path, [True])
    assert asyncio.run(flow.get_preset(0)) == "local/Tiny-Q4_K_M"
    assert [t for t, _b in asked] == ["Download Tiny?"] and "mmproj" not in asked[0][1]
    assert not service.is_vision("Tiny-Q4_K_M")
    assert LP.presets.download_size(LP._tiny_presets()[0]) == len(LP.PAYLOAD)
    assert flow.is_vision_preset(LP._tiny_presets()[0]) is False  # this preset names no projector


def test_the_first_image_asks_for_image_support_and_downloads_it(tmp_path, fake_hf):
    flow, service, asked = _flow(tmp_path, [True, False, True])
    asyncio.run(flow.get_preset(0))
    assert asyncio.run(flow.ensure_image_support("cloud/model")) is True  # not on-device: nothing to ask
    # declined: nothing downloaded, the image is not sent
    assert asyncio.run(flow.ensure_image_support("local/Tiny-Q4_K_M")) is False
    assert asked[-1][0] == "Download image support?" and "mmproj-F16.gguf (image support)" in asked[-1][1]
    assert not service.is_vision("Tiny-Q4_K_M")
    # accepted: only the projector downloads, and the model reads images from now on
    assert asyncio.run(flow.ensure_image_support("local/Tiny-Q4_K_M")) is True
    assert service.is_vision("Tiny-Q4_K_M")
    count = len(asked)
    assert asyncio.run(flow.ensure_image_support("local/Tiny-Q4_K_M")) is True and len(asked) == count


def test_the_chat_checks_only_images_for_text_only_on_device_models(tmp_path, fake_hf, monkeypatch):
    from glossarion_mobile.services.local_llm import service as svc
    from glossarion_mobile.ui.chat.chat_view import ChatView

    flow, service, _asked = _flow(tmp_path, [True])
    asyncio.run(flow.get_preset(0))
    monkeypatch.setattr(svc, "_DEFAULT", service)
    view = types.SimpleNamespace(state=types.SimpleNamespace(
        chat_context=types.SimpleNamespace(value=types.SimpleNamespace(model="local/Tiny-Q4_K_M"))))
    image, text = {"path": "/x/page.png"}, {"path": "/x/book.epub"}
    assert ChatView._image_model(view, image) == "local/Tiny-Q4_K_M"
    assert ChatView._image_model(view, text) == ""
    assert ChatView._image_model(view, image, {"model": "gpt-5"}) == ""
    view.state.chat_context.value = types.SimpleNamespace(model="gemini-3-pro")
    assert ChatView._image_model(view, image) == ""


def test_a_job_takes_its_translated_title_from_metadata_json(tmp_path):
    from glossarion_mobile.services.jobs import JobService, JobSpec, translated_book_title

    folder = tmp_path / "out"
    folder.mkdir()
    meta = folder / "metadata.json"
    meta.write_text(json.dumps({"title": "히어로 세계의 서큐버스", "original_title": "히어로 세계의 서큐버스"}),
                    encoding="utf-8")
    job = types.SimpleNamespace(id="j1", output_dir=str(folder), metadata_mtime=0.0,
                                spec=JobSpec(kind="translate", title="[213524] 히어로.epub"))
    fake = types.SimpleNamespace(_lock=threading.RLock())
    assert translated_book_title(str(meta)) == ""  # not translated yet
    assert JobService._take_translated_title(fake, job) is False and job.spec.title == "[213524] 히어로.epub"
    meta.write_text(json.dumps({"title": " The Hero  World's Succubus ", "title_translated": True}),
                    encoding="utf-8")
    import os
    os.utime(meta, (1, 2))
    assert JobService._take_translated_title(fake, job) is True
    assert job.spec.title == "The Hero World's Succubus"
    assert JobService._take_translated_title(fake, job) is False  # unchanged file: not read again


def test_the_requests_panel_is_hidden_until_its_first_card():
    from glossarion_mobile.ui.screens import job_detail as JD

    segments: list = []
    screen = types.SimpleNamespace(
        _last_requests=0.0, job_id="j1", snap=None,
        service=types.SimpleNamespace(request_segments=lambda jid: list(segments), has_request_stream=lambda jid: True),
        requests_card=types.SimpleNamespace(visible=False), requests_title=types.SimpleNamespace(value="Requests"),
        _show_requests=lambda rows, cards: None, _request_card=lambda card: card)
    JD.JobDetailScreen._refresh_requests(screen, force=True)
    assert screen.requests_card.visible is False
    segments.append({"label": "Chapter 1", "text": "hello"})
    JD.JobDetailScreen._refresh_requests(screen, force=True)
    assert screen.requests_card.visible is True
