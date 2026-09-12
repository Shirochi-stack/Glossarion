"""Exercise the real Arena/error handlers without starting provider clients or a GUI."""

import ast
import copy
from datetime import datetime
from functools import lru_cache
import logging
import os
from pathlib import Path
import random
import threading
from types import SimpleNamespace
import uuid

import pytest

import autharena_proxy as arena


SRC = Path(__file__).resolve().parents[1] / "src"


@lru_cache(maxsize=1)
def _unified_handlers_code():
    source = ast.parse((SRC / "unified_api_client.py").read_text(encoding="utf-8"))
    client_node = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "UnifiedClient")
    error_node = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "UnifiedClientError")
    wanted = {"_send_autharena", "_send_internal", "_is_rate_limit_error", "_parse_retry_after"}
    client_node.body = [node for node in client_node.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), error_node, client_node], type_ignores=[])
    return compile(ast.fix_missing_locations(module), str(SRC / "unified_api_client.py"), "exec")


@pytest.fixture
def handlers(tmp_path, monkeypatch):
    logs = []
    namespace = dict(os=os, uuid=uuid, threading=threading, random=random,
                     datetime=datetime, UnifiedResponse=SimpleNamespace,
                     time=SimpleNamespace(time=lambda: 1.0, sleep=lambda _delay: None),
                     logger=logging.getLogger(__name__),
                     print=lambda *args, **_kwargs: logs.append(" ".join(map(str, args))),
                     _payloads_dir=lambda: str(tmp_path),
                     _get_current_run_id=lambda: None,
                     _api_watchdog_record_retry=lambda *args, **kwargs: None,
                     _api_watchdog_update_model=lambda *args, **kwargs: None)
    exec(_unified_handlers_code(), namespace)
    client = namespace["UnifiedClient"]()
    tls = SimpleNamespace()
    client.client_type = "autharena"
    client.model = "autharena/test-model"
    client.request_timeout = 600
    client._multi_key_mode = False
    client.current_session_context = "translation"
    client.conversation_message_count = 0
    client._get_thread_local_client = lambda: tls
    client._get_active_request_model = lambda: client.model
    client._normalize_finish_reason = lambda value: value
    client._should_abort_retry = lambda: False
    client._is_stop_requested = lambda: False
    client._ensure_thread_client = lambda: None
    client._get_request_hash_with_request_id = lambda *_args: "test-request"
    client._get_max_retries = lambda: 3
    client._normalize_nanogpt_image_message_payloads = lambda value: value
    client._validate_request = lambda *_args: (True, "")
    client._get_file_names = lambda *args, **kwargs: ("request.json", "response.json")
    client._with_attempt_suffix = lambda *args, **kwargs: args[:2]
    client._save_payload = lambda *args, **kwargs: None
    client._set_idempotency_context = lambda *args: None
    client._should_retry_with_image_request_quality = lambda *args: False
    client._detect_safety_filter = lambda *args: False
    client._compute_backoff = lambda *args: 0
    client._save_failed_request = lambda *args: None
    client._track_stats = lambda *args: None
    client._handle_empty_result = lambda *args: ""
    client._failure_response_content = lambda *args, **kwargs: ""
    waits = []
    client._sleep_with_cancel = lambda delay, *_args: waits.append(delay) or True
    calls = []

    def fail_from_provider(error):
        def send(*args, **kwargs):
            calls.append(kwargs)
            raise error
        monkeypatch.setattr(arena, "send_message_stream", send)

    client._get_response = lambda messages, temperature, max_tokens, _mct, response_name, **kwargs: client._send_autharena(messages, temperature, max_tokens, response_name)
    monkeypatch.setattr(arena, "capture_cancel_generation", lambda: 1)
    monkeypatch.setattr(arena, "is_cancel_generation_cancelled", lambda _generation: False)
    for name in ("GRACEFUL_STOP", "GRACEFUL_STOP_COMPLETED", "SYSTEM_PROMPT_TO_USER", "INDEFINITE_RATE_LIMIT_RETRY", "USE_FALLBACK_KEYS"):
        monkeypatch.setenv(name, "0")
    return SimpleNamespace(client=client, error_type=namespace["UnifiedClientError"],
                           fail=fail_from_provider, calls=calls, waits=waits, logs=logs)


@pytest.mark.parametrize("status,kind", [(429, "rate_limit"), (401, "auth_error"),
    (403, "auth_error"), (500, "api_error"), (502, "api_error"),
    (503, "api_error"), (504, "api_error"), (400, "invalid_request_error"),
    (404, "invalid_request_error"), (422, "invalid_request_error")])
def test_arena_status_survives_provider_wrapper(handlers, status, kind):
    handlers.fail(arena.ArenaStreamError("provider rejected request", status, "17"))
    with pytest.raises(handlers.error_type) as caught:
        handlers.client._send_autharena([], 0.2, 1024, "test")
    assert caught.value.error_type == kind
    assert caught.value.http_status == status
    assert caught.value.details == {"retry_after": "17", "upstream_http_status": status, "partial_response": False}
    assert handlers.client._is_rate_limit_error(caught.value) == (status == 429)


def test_429_reaches_normal_single_key_retry_budget(handlers):
    handlers.fail(arena.ArenaStreamError("provider rejected request", 429))
    assert handlers.client._send_internal([], 0.2, 1024, request_id="rate-limit") == ("", "error")
    assert len(handlers.calls) == 3
    assert handlers.waits == [60, 60, 60]
    assert handlers.client._last_retry_error_type == "429_rate_limit"


def test_429_reaches_multi_key_rotation_handler(handlers):
    handlers.client._multi_key_mode = True
    handlers.fail(arena.ArenaStreamError("provider rejected request", 429))
    with pytest.raises(handlers.error_type) as caught:
        handlers.client._send_internal([], 0.2, 1024, request_id="rate-limit")
    assert caught.value.error_type == "rate_limit"
    assert handlers.client._is_rate_limit_error(caught.value)
    assert len(handlers.calls) == 1
    assert not handlers.waits


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_server_failures_use_normal_retry_budget(handlers, status):
    handlers.fail(arena.ArenaStreamError("provider rejected request", status))
    assert handlers.client._send_internal([], 0.2, 1024, request_id="server-error") == ("", "error")
    assert len(handlers.calls) == 3
    assert any("Server error" in line for line in handlers.logs)


@pytest.mark.parametrize("status", [401, 403])
def test_authentication_failures_request_login_without_replaying(handlers, status):
    handlers.fail(arena.ArenaStreamError("Arena Login required", status))
    with pytest.raises(handlers.error_type) as caught:
        handlers.client._send_internal([], 0.2, 1024, request_id="auth-error")
    assert caught.value.error_type == "auth_error"
    assert len(handlers.calls) == 1
    assert not handlers.waits


def test_bad_request_does_not_become_prohibited_content(handlers):
    handlers.fail(arena.ArenaStreamError("Arena HTTP 400: invalid conversation mode", 400))
    assert handlers.client._send_internal([], 0.2, 1024, request_id="bad-request") == ("", "error")
    assert not any("Prohibited content" in line for line in handlers.logs)


@pytest.mark.parametrize("status", [429, 503, None])
def test_partial_response_is_neither_retried_nor_rate_limit_rotated(handlers, status):
    handlers.fail(arena.ArenaStreamError("Arena HTTP 429: rate limit after partial reasoning", status, partial_response=True))
    with pytest.raises(handlers.error_type) as caught:
        handlers.client._send_internal([], 0.2, 1024, request_id="partial")
    assert caught.value.error_type == "autharena_stream_error"
    assert caught.value.http_status is None
    assert caught.value.details["upstream_http_status"] == status
    assert caught.value.details["partial_response"]
    assert not handlers.client._is_rate_limit_error(caught.value)
    assert len(handlers.calls) == 1
    assert not handlers.waits


def test_uncertain_stream_error_cannot_rotate_based_on_message_alone(handlers):
    handlers.fail(RuntimeError("stream ended after a 429-looking text fragment"))
    with pytest.raises(handlers.error_type) as caught:
        handlers.client._send_autharena([], 0.2, 1024, "test")
    assert caught.value.error_type == "autharena_stream_error"
    assert not handlers.client._is_rate_limit_error(caught.value)


@lru_cache(maxsize=2)
def _translation_error_code(queue_result):
    source = ast.parse((SRC / "TransateKRtoEN.py").read_text(encoding="utf-8"))
    if queue_result:
        func = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "send_with_interrupt")
        branch = next(node for node in ast.walk(func) if isinstance(node, ast.If)
                      and ast.unparse(node.test) == "isinstance(result, Exception)")
        body = copy.deepcopy(branch.body)
        arg = "result"
    else:
        cls = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "TranslationProcessor")
        func = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "translate_with_retry")
        handler = next(node for node in ast.walk(func) if isinstance(node, ast.ExceptHandler)
                       and isinstance(node.type, ast.Name) and node.type.id == "UnifiedClientError")
        attempt = ast.Try(body=[ast.Raise(exc=ast.Name(id="error", ctx=ast.Load()))], handlers=[copy.deepcopy(handler)], orelse=[], finalbody=[])
        body = [ast.For(target=ast.Name(id="attempt", ctx=ast.Store()),
                        iter=ast.Tuple(elts=[ast.Constant(value=0)], ctx=ast.Load()),
                        body=[attempt], orelse=[])]
        arg = "error"
    function = ast.FunctionDef(name="handle", args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=arg)], kwonlyargs=[], kw_defaults=[], defaults=[]), body=body, decorator_list=[])
    module = ast.Module(body=[function], type_ignores=[])
    return compile(ast.fix_missing_locations(module), str(SRC / "TransateKRtoEN.py"), "exec")


def _translation_error_handler(error_class, *, queue_result=False):
    namespace = {"UnifiedClientError": error_class, "_install_actual_request_metadata": lambda *args: None}
    exec(_translation_error_code(queue_result), namespace)
    return namespace["handle"]


@pytest.mark.parametrize("kind,status", [("autharena_stream_error", None), ("rate_limit", 429), ("api_error", 503)])
def test_translation_queue_preserves_structured_errors(handlers, kind, status):
    error = handlers.error_type("Arena HTTP 429: failure", error_type=kind, http_status=status,
                                details={"partial_response": kind == "autharena_stream_error", "retry_after": "17"})
    with pytest.raises(handlers.error_type) as caught:
        _translation_error_handler(handlers.error_type, queue_result=True)(error)
    assert caught.value is error


@pytest.mark.parametrize("message", ["Arena stream timed out", "Arena HTTP 429", "Arena stream cancelled before completion"])
def test_translation_retry_handler_preserves_no_replay_errors(handlers, message):
    error = handlers.error_type(message, error_type="autharena_stream_error", details={"partial_response": True})
    with pytest.raises(handlers.error_type) as caught:
        _translation_error_handler(handlers.error_type)(error)
    assert caught.value is error


def test_captcha_rejection_is_not_reported_as_lost_login(handlers, capsys):
    handlers.fail(arena.ArenaStreamError('Arena HTTP 403: recaptcha validation failed', 403))
    with pytest.raises(handlers.error_type) as caught:
        handlers.client._send_internal([], 0.2, 1024, request_id='captcha')
    assert caught.value.error_type == 'autharena_verification_error'
    assert caught.value.http_status == 403
    assert len(handlers.calls) == 1
    assert handlers.waits == []
    output = capsys.readouterr().out
    assert 'use Arena Login before retrying' not in output
