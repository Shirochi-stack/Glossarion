"""Interactive Images and the Image Gen / Edit key pool, end to end with the real ``UnifiedClient``.

The runner's own client factory builds the clients, ``UnifiedClient._send_internal`` is replaced by a recorder, and
the pool is a real in-memory one (``set_in_memory_inpainter_keys``): the key that reaches the provider call is read
back. With the pool on, every IMAGE request of the step uses a pool key (and the image payload context); the planner
and CSS TEXT requests never do, whatever the main tab's Output mode is. With the pool off every request uses the
normal key.

Python 3.10 compatible. Run:
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests/test_interactive_images_routing.py
"""
import json
import os
import sys
from io import BytesIO
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("GLOSSARION_HTTP_LOG", "0")

import interactive_images_core as core  # noqa: E402
import interactive_images_runner as runner  # noqa: E402

POOL_ENV = ("USE_INPAINTER_KEYS", "INPAINTER_API_KEYS", "ENABLE_IMAGE_OUTPUT_MODE", "USE_MULTI_API_KEYS",
            "USE_MULTI_KEYS", "BATCH_TRANSLATION")

CH1 = ('<html><head><title>Ch 1</title></head><body>\n<h1>Chapter 1</h1>\n'
       '<p>Mira walked to the lantern gate.</p>\n<p>"Hello," said Tomas.</p>\n</body></html>')
PLAN = ('{"images":[{"after_paragraph":2,"quote":"Mira walked","characters":["Mira"],"prompt":"A gate"}],'
        '"classes":[]}')
MAIN_KEY = 'main-key'
POOL_KEYS = [{'api_key': 'img-key-a', 'model': 'gemini-3-pro-image-preview', 'enabled': True}]


def _png():
    from PIL import Image
    buffer = BytesIO()
    Image.new('RGB', (8, 8), (10, 20, 30)).save(buffer, format='PNG')
    return buffer.getvalue()


@pytest.fixture
def routed(monkeypatch, tmp_path):
    """A book, a recorder behind the real client, and a clean key-pool state. ``calls``: one dict per provider call."""
    from unified_api_client import UnifiedClient

    for name in POOL_ENV:
        monkeypatch.delenv(name, raising=False)
    for attr in ("_in_memory_inpainter_keys", "_inpainter_key_pool"):
        monkeypatch.setattr(UnifiedClient, attr, None)
    monkeypatch.setenv("BATCH_TRANSLATION", "0")
    monkeypatch.setenv("THREAD_SUBMISSION_DELAY_SECONDS", "0")
    monkeypatch.setenv("SEND_INTERVAL_SECONDS", "0")
    monkeypatch.setenv("USE_MULTI_API_KEYS", "0")
    monkeypatch.setenv("USE_MULTI_KEYS", "0")
    monkeypatch.setattr(UnifiedClient, "_setup_client", lambda self: setattr(self, "client_type", "test"))
    calls = []

    def fake_send_internal(self, messages, *args, **kwargs):
        context = kwargs.get('context') if 'context' in kwargs else (args[3] if len(args) > 3 else None)
        calls.append({'api_key': self.api_key, 'model': self.model, 'context': context,
                      'image_client': bool(getattr(self, '_force_image_output_mode', False)),
                      'text_only': bool(getattr(self, '_text_output_only', False))})
        if getattr(self, '_force_image_output_mode', False):
            self._get_thread_local_client()._last_generated_image_bytes = _png()
            return '', 'stop'
        return PLAN, 'stop'

    monkeypatch.setattr(UnifiedClient, "_send_internal", fake_send_internal)
    for method, value in (("reset_cleanup_state", lambda: None), ):
        monkeypatch.setattr(UnifiedClient, method, lambda self, *a, **k: None)
    monkeypatch.setattr(UnifiedClient, "_log_pre_stagger", lambda self, *a: None)
    monkeypatch.setattr(UnifiedClient, "_apply_thread_submission_delay", lambda self: None)
    monkeypatch.setattr(UnifiedClient, "_should_abort_retry", lambda self: False)
    monkeypatch.setattr(UnifiedClient, "_extract_chapter_label", lambda self, _m: None)
    import unified_api_client as uac
    monkeypatch.setattr(uac, "_api_watchdog_started", lambda *a, **k: None)
    monkeypatch.setattr(uac, "_api_watchdog_finished", lambda *a, **k: None)

    out = tmp_path / 'book'
    out.mkdir()
    (out / 'response_001_ch1.html').write_text(CH1, encoding='utf-8')
    (out / 'translation_progress.json').write_text(json.dumps({'chapters': {
        'a': {'actual_num': 1, 'output_file': 'response_001_ch1.html', 'status': 'completed'}}}), encoding='utf-8')
    (out / 'glossary.csv').write_text('type,raw_name,translated_name\ncharacter,미라,Mira\n', encoding='utf-8')
    yield out, calls, monkeypatch
    UnifiedClient.clear_in_memory_inpainter_keys()


def _config(**extra):
    config = {'model': 'gpt-text', 'api_key': MAIN_KEY, 'interactive_images_image_model': 'gemini-3-pro-image-preview',
              'interactive_images_per_chapter': '1'}
    config.update(extra)
    return config


def _run(out, config):
    result = runner.run_interactive_images(config, str(out), log=lambda *_a: None)
    assert result.completed == 1, result.errors
    return result


def test_image_requests_use_the_image_pool_and_text_requests_never_do(routed):
    out, calls, _mp = routed
    _run(out, _config(use_inpainter_keys=True, inpainter_keys=POOL_KEYS))
    images = [c for c in calls if c['image_client']]
    texts = [c for c in calls if not c['image_client']]
    assert len(images) == 1 and len(texts) == 1
    assert images[0]['api_key'] == 'img-key-a'                      # the pool key, not the main key
    assert images[0]['model'] == 'gemini-3-pro-image-preview'
    assert texts[0]['api_key'] == MAIN_KEY and texts[0]['text_only'] is True   # the planner: the normal route


def test_with_the_pool_off_every_request_uses_the_normal_key(routed):
    out, calls, _mp = routed
    _run(out, _config(use_inpainter_keys=False, inpainter_keys=POOL_KEYS))
    assert [c['api_key'] for c in calls] == [MAIN_KEY, MAIN_KEY]
    assert os.environ.get('USE_INPAINTER_KEYS') == '0'


def test_an_empty_or_all_disabled_pool_takes_the_normal_route(routed):
    out, calls, _mp = routed
    _run(out, _config(use_inpainter_keys=True, inpainter_keys=[dict(POOL_KEYS[0], enabled=False)]))
    assert [c['api_key'] for c in calls] == [MAIN_KEY, MAIN_KEY]


def test_the_main_tabs_image_output_mode_does_not_turn_the_planner_into_an_image_request(routed):
    out, calls, mp = routed
    mp.setenv("ENABLE_IMAGE_OUTPUT_MODE", "1")                     # the main tab's Output mode is "image"
    _run(out, _config(use_inpainter_keys=True, inpainter_keys=POOL_KEYS))
    texts = [c for c in calls if not c['image_client']]
    assert texts and all(c['api_key'] == MAIN_KEY for c in texts)
    from unified_api_client import UnifiedClient
    text_client = runner._default_client_factory('text', 'gpt-text', _config(), str(out))
    assert text_client._image_output_mode_enabled() is False       # even with the environment switch on
    assert text_client._uses_image_gen_edit_key_context(runner.PLAN_CONTEXT) is False
    assert text_client._uses_image_gen_edit_key_context(runner.CSS_CONTEXT) is False
    assert isinstance(text_client, UnifiedClient)


def test_context_names_decide_the_key_pool(routed):
    out, _calls, _mp = routed
    image_client = runner._default_client_factory('image', 'gemini-3-pro-image-preview', _config(), str(out))
    assert image_client._uses_image_gen_edit_key_context(runner.IMAGE_CONTEXT) is True
    assert runner.IMAGE_CONTEXT == 'image_generation'
    assert image_client._uses_image_gen_edit_key_context(runner.PLAN_CONTEXT) is False
    assert image_client._uses_image_gen_edit_key_context(runner.CSS_CONTEXT) is False
    assert image_client._force_image_output_mode is True
    assert image_client._get_image_output_resolution() == '1K'


def test_the_ratio_set_per_image_is_the_clients_api_parameter(routed):
    out, _calls, _mp = routed
    client = runner._default_client_factory('image', 'gemini-3-pro-image-preview', _config(), str(out))
    runner._set_image_aspect(client, '21:9')
    assert client._get_image_output_aspect_ratio() == '21:9'
    runner._set_image_aspect(client, '')
    assert client._get_image_output_aspect_ratio(default='auto') == os.getenv('IMAGE_OUTPUT_ASPECT_RATIO', 'auto')
    assert client._forced_image_output_aspect_ratio is None


def test_image_requests_land_in_the_image_payload_folder(routed, tmp_path):
    out, _calls, mp = routed
    import unified_api_client as uac
    mp.setattr(uac, "_payloads_dir", lambda: str(tmp_path / 'Payloads'))
    client = runner._default_client_factory('image', 'gemini-3-pro-image-preview', _config(), str(out))
    folder = client._generate_unique_thread_dir(runner.IMAGE_CONTEXT)
    assert Path(folder).parent.name == 'image_generation'
    text_folder = client._generate_unique_thread_dir(runner.PLAN_CONTEXT)
    assert Path(text_folder).parent.name == 'interactive_images_plan'
