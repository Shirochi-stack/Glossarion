"""Host tests for U9 gap closures that have no feature test file of their own.

* slash commands (UI_SPEC §2.7): the command table, the composer popover and Send / Enter running a
  complete command, the chat view's dispatch through the handlers the buttons use;
* the shared API-wait classifier behind the running card's issue chips and the safety-block card;
* keyboard zoom (Ctrl + / - / 0) and the passphrase config export / import;
* Logs & diagnostics: secret redaction of the shared logs bundle;
* Tools › Manga: the mask presets, the Model Information text and the measured Rendering › Reset
  values come from the shared modules (``manga_settings_defaults`` / ``manga_models`` /
  ``manga_env``), the reset waits for a running job;
* the Library TranslateSheet's run options reach the job (``config_overrides``);
* the schema label of ``max_output_tokens``.

Real data is never touched (the review runner isolates HOME / USERPROFILE / GLOSSARION_LIBRARY_DIR /
OUTPUT_DIRECTORY; nothing here writes outside pytest's tmp dirs).

Run from src/mobile:
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_u9_gaps.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
import types
import zipfile
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


needs_flet = pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet / msgpack not installed")


# ==========================================================================
# slash commands
# ==========================================================================


def test_slash_commands_match_and_parse():
    from glossarion_mobile.ui.chat import slash

    names = [c.name for c in slash.SLASH_COMMANDS]
    for spec_name in ("glossary", "qa", "compile epub", "compile pdf", "headers", "metadata", "manga", "review",
                      "async", "progress", "retranslate", "mode", "model", "profile", "lang", "policy", "scratch",
                      "export", "library", "jobs", "settings"):
        assert spec_name in names, spec_name
    assert len(slash.match_commands("/")) == len(slash.SLASH_COMMANDS)
    assert [c.name for c in slash.match_commands("/comp")] == ["compile epub", "compile pdf"]
    assert [c.name for c in slash.match_commands("/model gpt")] == ["model"]
    assert slash.match_commands("hello") == [] and slash.match_commands("/qa\nmore") == []
    assert slash.match_commands("/zzz") == []
    cmd, arg = slash.parse_command("/compile  PDF")
    assert cmd.name == "compile pdf" and arg == ""
    cmd, arg = slash.parse_command("/model gpt-4o mini")
    assert cmd.name == "model" and arg == "gpt-4o mini"
    assert slash.parse_command("/qa extra") is None  # /qa takes no argument
    assert slash.parse_command("/compile") is None and slash.parse_command("translate this") is None
    assert slash.completion(cmd) == "/model "
    assert slash.POLICY_ARGS["off"] == "no_glossary" and slash.POLICY_ARGS["attachments"] == "attachments_only"


@needs_flet
def test_composer_runs_complete_slash_commands():
    from glossarion_mobile.ui.chat import slash
    from glossarion_mobile.ui.chat.composer import Composer
    from glossarion_mobile.ui.chat.send_state import SendAction

    ran: list = []
    sent: list = []
    composer = Composer(on_slash=ran.append, on_send_action=sent.append)
    assert composer.slash.control not in composer.content.controls  # the chat view puts it above the card
    composer.handle_text("/")
    assert composer.slash.visible and len(composer.slash.commands) == len(slash.SLASH_COMMANDS)
    assert composer.slash.list.height == slash.ROW_HEIGHT * slash.MAX_VISIBLE
    composer.handle_text("/q")
    assert [c.name for c in composer.slash.commands] == ["qa"]
    # a tap on a command that takes an argument puts it in the field; nothing runs
    model = next(c for c in slash.SLASH_COMMANDS if c.name == "model")
    composer._on_slash_pick(model)
    assert composer.text == "/model " and ran == []
    composer.handle_text("/model gemini")
    composer._on_slash_pick(model)
    assert ran == ["/model gemini"] and composer.text == "" and not composer.slash.visible
    # Send on a complete command runs it instead of sending
    composer.handle_text("/qa")
    composer._on_send_action(SendAction.SEND)
    assert ran[-1] == "/qa" and sent == [] and composer.text == ""
    # anything else is sent as usual
    composer.handle_text("/not a command")
    composer._on_send_action(SendAction.SEND)
    assert sent == [SendAction.SEND]
    composer.clear()
    assert not composer.slash.visible
    # without a handler there is no popover and "/qa" is ordinary text
    plain = Composer(on_send_action=sent.append)
    plain.handle_text("/qa")
    assert not plain.slash.visible
    plain._on_send_action(SendAction.SEND)
    assert sent[-1] == SendAction.SEND


@needs_flet
def test_chat_view_runs_slash_commands_through_the_button_handlers():
    from glossarion_mobile.ui.chat.chat_view import ChatView

    calls: list = []
    overrides: dict = {}
    fake = types.SimpleNamespace(
        _on_tool=lambda tool: calls.append(("tool", tool)),
        notify=lambda message, **k: calls.append(("notify", message)),
        _spawn=lambda coro: (coro.close(), calls.append(("spawn", "retranslate"))),
        open_retranslate=lambda chapters=None: (calls.append(("retranslate", chapters)), _coro())[1],
        composer=types.SimpleNamespace(output_row=types.SimpleNamespace(select=lambda m: calls.append(("mode", m)))),
        open_model_sheet=lambda tab: calls.append(("sheet", tab)) or types.SimpleNamespace(
            set_query=lambda q: calls.append(("query", q))),
        _profiles=lambda: ["Korean", "Japanese"],
        env=object(),
        _on_model_sheet_select=lambda field, value, this_chat: calls.append(("select", field, value)),
        bound=True,
        cid="7",
        open_chat_settings=lambda: calls.append(("chat_settings",)),
        apply_settings_changed=lambda: calls.append(("applied",)),
        _on_new_scratch=lambda: calls.append(("scratch",)),
        open_export=lambda: calls.append(("export",)),
        navigate=lambda name, *a: calls.append(("nav", name)),
        open_settings_search=lambda q: calls.append(("search", q)),
    )
    fake.env = types.SimpleNamespace(chats=types.SimpleNamespace(
        set_override=lambda cid, key, value: overrides.__setitem__((cid, key), value)))
    run = lambda text: ChatView.run_slash(fake, text)  # noqa: E731

    assert run("/compile pdf") == "compile pdf" and calls[-1] == ("tool", "compile")
    assert run("/glossary") and calls[-1] == ("tool", "extract_glossary")
    assert run("/metadata") and calls[-1] == ("tool", "headers")
    run("/mode refine")
    assert calls[-1] == ("mode", "refine")
    run("/mode nonsense")
    assert calls[-1][0] == "notify"
    run("/model gpt-4o")
    assert calls[-2:] == [("sheet", "model"), ("query", "gpt-4o")]
    run("/profile japanese")
    assert ("select", "profile", "Japanese") in calls
    run("/profile Unknown")
    assert calls[-1] == ("sheet", "profile")
    run("/lang French")
    assert ("select", "language", "French") in calls
    run("/policy off")
    assert overrides[("7", "glossary_override_mode")] == "no_glossary" and ("applied",) in calls
    run("/policy manual")
    assert calls[-1] == ("chat_settings",)
    run("/scratch")
    assert calls[-1] == ("scratch",)
    run("/jobs")
    assert calls[-1] == ("nav", "jobs")
    run("/settings temperature")
    assert calls[-1] == ("search", "temperature")
    run("/settings")
    assert calls[-1] == ("nav", "settings")
    run("/retranslate 5-10")
    assert ("spawn", "retranslate") in calls and ("retranslate", (5, 10)) in calls  # U9: the range is passed on
    run("/retranslate five")
    assert calls[-1] == ("notify", "Use /retranslate N or /retranslate N-M (chapter numbers)")
    assert run("/unknown") is None and calls[-1][0] == "notify"


async def _coro():
    return None


# ==========================================================================
# API waits / safety blocks, keyboard zoom, config export, logs bundle
# ==========================================================================


@pytest.mark.skipif(not _has("direct_text_stream"), reason="direct_text_stream not importable")
def test_issue_classifier_and_safety_lines():
    from glossarion_mobile.services import jobs

    assert jobs.classify_issue("⏳ Rate limited (429), retrying in 12s") == ("rate_limited", 12.0)
    assert jobs.classify_issue("All keys are rate-limited; waiting for cooldown 30s")[0] == "key_cooling"
    assert jobs.classify_issue("Connection error: failed to establish a new connection")[0] == "network_wait"
    assert jobs.classify_issue("Content blocked: Google Generative AI Prohibited Use policy") == ("safety_block", None)
    assert jobs.classify_issue("Translated chapter 3") is None
    assert jobs.issue_label({"kind": "key_cooling"}, cooling_keys=2) == "Key cooling (2)"
    assert jobs.issue_label(None) == ""


def test_keyboard_zoom_steps():
    from glossarion_mobile.ui.keyboard import zoom_step

    assert zoom_step("=", ctrl=True) == 1 and zoom_step("Numpad Add", meta=True) == 1
    assert zoom_step("-", ctrl=True) == -1 and zoom_step("0", ctrl=True) == 0
    assert zoom_step("=", ctrl=False) is None and zoom_step("=", ctrl=True, alt=True) is None
    assert zoom_step("A", ctrl=True) is None


@pytest.mark.skipif(not (_has("cryptography") and _has("api_key_encryption")), reason="cryptography not installed")
def test_config_export_round_trip_with_passphrase(tmp_path):
    from glossarion_mobile.services import config_export as ce

    config = {"api_key": "sk-test-1234567890", "model": "gpt-4o", "output_language": "English"}
    document = ce.export_config(config, "correct horse")
    assert document["format"] == ce.FORMAT and "sk-test-1234567890" not in json.dumps(document)
    path = ce.write_export(str(tmp_path / "export.json"), document)
    restored = ce.import_config(ce.read_export(path), "correct horse")
    assert restored["api_key"] == "sk-test-1234567890" and restored["model"] == "gpt-4o"
    with pytest.raises(ce.WrongPassphrase):
        ce.import_config(document, "wrong passphrase")
    with pytest.raises(ValueError):
        ce.export_config(config, "short")
    with pytest.raises(ValueError):
        ce.import_config({"format": "other"}, "correct horse")


@pytest.mark.skipif(not (_has("cryptography") and _has("api_key_encryption") and _has("settings_schema")),
                    reason="cryptography / settings_schema not importable")
def test_config_exports_leave_no_credential_in_plain_text():
    """Both exports cover every credential in config.json: api_key_encryption's fields and pools, every
    settings_schema key typed 'secret' (nested qa_scanner_settings.ai_truncation_api_key too) and the Azure
    OCR keys. The passphrase export encrypts them (and imports them back); the share drops them."""
    from cryptography.fernet import Fernet

    import api_key_encryption
    import settings_schema
    from glossarion_mobile.services import config_export as ce

    api_key_encryption.set_key_material(Fernet.generate_key())  # never the key file beside the module
    try:
        paths, lists = ce.secret_fields()
        for spec in settings_schema.all_specs():  # a new credential setting must reach the export list
            if spec.type == "secret" or (spec.path[-1].endswith(("api_key", "_key")) and spec.type in ("str", "secret")):
                assert tuple(spec.path) in paths, spec.key
        expected = {("api_key",), ("replicate_api_key",), ("qa_scanner_settings", "ai_truncation_api_key"),
                    ("azure_vision_key",), ("azure_document_intelligence_key",), ("azure_key",)}
        assert expected <= set(paths) and {"multi_api_keys", "fallback_keys", "tts_keys"} <= set(lists)
        config: dict = {"model": "gpt-4o", "qa_scanner_settings": {"ai_truncation_model": "m"}}
        secrets = []
        for n, path in enumerate(paths):
            value = f"SECRET-PATH-{n:03d}-abcdefghijklmnop"
            node = config
            for part in path[:-1]:
                node = node.setdefault(part, {})
            node[path[-1]] = value
            secrets.append(value)
        for n, name in enumerate(lists):
            value = f"SECRET-POOL-{n:03d}-abcdefghijklmnop"
            config[name] = [{"api_key": value, "model": f"pool-{n}"}]
            secrets.append(value)
        original = json.loads(json.dumps(config))

        document = ce.export_config(config, "correct horse battery")
        text = json.dumps(document)
        assert [s for s in secrets if s in text] == []
        assert config == original  # the live config is never touched
        assert ce.import_config(document, "correct horse battery") == original

        stripped = ce.without_secrets(config)
        text = json.dumps(stripped)
        assert [s for s in secrets if s in text] == []
        assert stripped["model"] == "gpt-4o" and stripped["qa_scanner_settings"] == {"ai_truncation_model": "m"}
        assert stripped["multi_api_keys"] == [{"model": f"pool-{lists.index('multi_api_keys')}"}]
        assert config == original

        # fail closed: a value the handler cannot encrypt stops the export instead of shipping it
        real = api_key_encryption.APIKeyEncryption.encrypt_value
        try:
            api_key_encryption.APIKeyEncryption.encrypt_value = lambda self, value: value
            with pytest.raises(ValueError, match="Could not encrypt"):
                ce.export_config({"azure_vision_key": "SECRET-azure-1234567890"}, "correct horse battery")
        finally:
            api_key_encryption.APIKeyEncryption.encrypt_value = real
    finally:
        api_key_encryption.set_key_material(None)


@needs_flet
def test_logs_bundle_redacts_secrets(tmp_path):
    from glossarion_mobile.services import logs

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "run.log").write_text("calling with key sk-secret-abcdef123456 ok\n", encoding="utf-8")
    path = logs.build_logs_bundle(logs_dir, tmp_path / "out", config={"api_key": "sk-secret-abcdef123456"},
                                  env={"PATH": "x"}, extra_lines=["Glossarion Mobile test"])
    with zipfile.ZipFile(path) as archive:
        text = archive.read("run.log").decode("utf-8")
        env = archive.read("environment.txt").decode("utf-8")
    assert "sk-secret-abcdef123456" not in text and "<REDACTED>" in text
    assert "Glossarion Mobile test" in env
    assert logs.redact_text("a sk-secret-abcdef123456 b", ["sk-secret-abcdef123456", "x"]) == "a <REDACTED> b"


# ==========================================================================
# Tools › Manga: the shared presets / reset / model information
# ==========================================================================


@pytest.mark.skipif(not _has("manga_settings_defaults"), reason="manga_settings_defaults not importable")
def test_manga_mask_presets_come_from_the_shared_module():
    from glossarion_mobile.services import manga as svc

    assert svc.mask_preset_rows() == [("bw_manga", "B&W Manga"), ("colored", "Colored"), ("uniform", "Uniform")]
    updates = svc.mask_preset_updates("bw_manga")
    assert updates[("manga_settings", "mask_dilation")] == 15
    assert updates[("manga_settings", "empty_bubble_dilation_iterations")] == 3
    assert updates[("manga_settings", "dilation_iterations")] == 2
    assert svc.mask_preset_updates("nope") == {}
    assert svc.rendering_reset_available()


@pytest.mark.skipif(not _has("manga_models"), reason="manga_models not importable")
def test_manga_model_information_text():
    from glossarion_mobile.services import manga as svc

    import manga_models

    assert svc.model_info_text("aot_onnx") == manga_models.MODEL_INFO["aot_onnx"]
    assert svc.model_info_text("nope") == "Please select a model type first"


@pytest.mark.skipif(not (_has("manga_env") and _has("job_runner")), reason="manga_env / job_runner not importable")
def test_rendering_reset_is_measured_once_and_waits_for_a_running_job(monkeypatch):
    from glossarion_mobile.services import manga as svc

    import job_runner

    monkeypatch.setattr(svc, "_PRESET_CACHE", {})
    held, release = threading.Event(), threading.Event()

    def hold():
        with job_runner.JOB_LOCK:
            held.set()
            release.wait(10)

    worker = threading.Thread(target=hold)
    worker.start()
    try:
        assert held.wait(10)
        with pytest.raises(svc.PresetsBusy):
            svc.rendering_reset_updates()
    finally:
        release.set()
        worker.join(10)
    assert svc.cached_rendering_reset_updates() is None
    updates = svc.rendering_reset_updates()
    assert updates["manga_bg_style"] == "circle" and updates["manga_text_color"] == [102, 0, 0]
    assert updates[("manga_settings", "font_sizing", "algorithm")] == "smart"
    assert svc.cached_rendering_reset_updates() == updates


@pytest.mark.skipif(not _has("manga_env"), reason="manga_env not importable")
def test_image_edit_test_shows_the_desktop_status_and_message():
    from glossarion_mobile.services import manga as svc

    text = svc.test_image_edit_endpoint({"use_custom_image_edit_endpoint": False})
    assert text.startswith("Using current image provider/model\n") and "Blank URL" in text


# ==========================================================================
# Library TranslateSheet -> the job's config_overrides; schema label
# ==========================================================================


def test_translate_spec_carries_the_run_options():
    from glossarion_mobile.services.library import LibraryService

    fake = types.SimpleNamespace(origin_for=lambda book: {"type": "book", "label": book["name"]},
                                 raw_source=lambda book: "")
    spec = LibraryService.translate_spec(fake, [{"name": "Book"}], sources=["/x/book.epub"],
                                         config_overrides={"batch_size": 4, "chapter_range": "5-10"})
    assert spec.kind == "translate" and spec.inputs == ("/x/book.epub",)
    assert spec.params["config_overrides"] == {"batch_size": 4, "chapter_range": "5-10"}
    plain = LibraryService.translate_spec(fake, [{"name": "Book"}], sources=["/x/book.epub"])
    assert "config_overrides" not in plain.params


@pytest.mark.skipif(not _has("settings_schema"), reason="settings_schema not importable")
def test_max_output_tokens_has_a_readable_label():
    import settings_schema

    assert settings_schema.spec("max_output_tokens").label == "Max output tokens"
