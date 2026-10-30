"""Owner 2026-10-30: "the whole point is to prevent the model from outputting any characters from the banned scripts
through logit bias". Ollama's API has no logit_bias, so a banned ollamapull/ request runs on the llama-server Ollama
ships, on the same model file, with the ban built from the file's vocabulary."""
from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import ollamapull  # noqa: E402
import script_ban_core as sbc  # noqa: E402

ENGLISH = tuple(sbc.banned_scripts("translation", "English"))


def _gguf_text(value: bytes) -> bytes:
    return struct.pack("<Q", len(value)) + value


def _write_gguf(path: Path, tokens: list[str]) -> str:
    """A GGUF header: other metadata of each kind before the vocabulary, as real files have."""
    kv = [
        _gguf_text(b"general.name") + struct.pack("<I", 8) + _gguf_text(b"test"),
        _gguf_text(b"general.alignment") + struct.pack("<II", 4, 32),
        _gguf_text(b"tokenizer.ggml.scores") + struct.pack("<IIQ", 9, 6, 2) + struct.pack("<ff", 0.0, 0.0),
        _gguf_text(b"tokenizer.ggml.merges") + struct.pack("<IIQ", 9, 8, 2) + _gguf_text(b"a b") + _gguf_text(b"c d"),
        _gguf_text(b"tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, len(tokens))
        + b"".join(_gguf_text(t.encode("utf-8")) for t in tokens),
    ]
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, len(kv)) + b"".join(kv))
    return str(path)


def test_vocabulary_is_read_from_the_model_file(tmp_path):
    tokens = ["<|im_end|>", "hello", "你好", "안녕"]
    assert ollamapull._gguf_tokens(_write_gguf(tmp_path / "m.gguf", tokens)) == tokens
    (tmp_path / "bad.gguf").write_bytes(b"nope")
    with pytest.raises(ollamapull.OllamaPullError):
        ollamapull._gguf_tokens(str(tmp_path / "bad.gguf"))


def test_bias_forbids_exactly_the_banned_tokens_and_is_kept(tmp_path, monkeypatch):
    tokens = ["hello", " the", "你好", "안녕", "こんにちは", "Привет", "—", "“", "<|im_end|>", "ツ"]
    path = _write_gguf(tmp_path / "m.gguf", tokens)
    bias = ollamapull._token_ban_bias(path, ENGLISH, False)
    assert all(flag is False for _id, flag in bias)  # llama-server: false = never
    assert {tokens[i] for i, _flag in bias} == {"你好", "안녕", "こんにちは", "Привет", "ツ"}
    monkeypatch.setattr(ollamapull, "_gguf_tokens", lambda _p: pytest.fail("read twice"))
    assert ollamapull._token_ban_bias(path, ENGLISH, False) is bias


def test_bias_lets_the_sources_tokens_through(tmp_path):
    tokens = ["hello", "김민", "김金", "金", "안"]
    path = _write_gguf(tmp_path / "w.gguf", tokens)
    bias = ollamapull._token_ban_bias(path, ENGLISH, False, frozenset("김민"))
    assert {tokens[i] for i, _flag in bias} == {"김金", "金", "안"}
    # the plain ban of the same file is untouched
    assert {tokens[i] for i, _flag in ollamapull._token_ban_bias(path, ENGLISH, False)} == {"김민", "김金", "金", "안"}


def _request(monkeypatch, payload, messages=None):
    seen = {}

    def start(name, options, vision, progress, should_stop):
        seen.update(name=name, options=options, vision=vision)
        return "http://127.0.0.1:5", "model.gguf"

    monkeypatch.setattr(ollamapull, "_start_ban_server", start)
    monkeypatch.setattr(ollamapull, "_token_ban_bias", lambda *a: [[7, False]])
    url, body = ollamapull._token_ban_request("m", messages or [{"role": "user", "content": "x"}], payload, ENGLISH,
                                              False, None, None)
    return url, body, seen


def test_request_carries_the_ban_and_ollamas_settings(monkeypatch):
    url, body, seen = _request(monkeypatch, {"stream": True, "options": {
        "temperature": 0.3, "top_k": 20, "min_p": 0.1, "repeat_penalty": 1.2, "num_predict": 500, "num_ctx": 8192,
        "stop": ["x"]}})
    assert url == "http://127.0.0.1:5/v1/chat/completions"
    assert body["logit_bias"] == [[7, False]]
    assert (body["temperature"], body["top_k"], body["min_p"], body["repeat_penalty"], body["stop"]) == (0.3, 20, 0.1, 1.2, ["x"])
    assert body["max_tokens"] == 500 and "num_ctx" not in body  # a load setting
    assert seen["options"]["num_ctx"] == 8192 and seen["vision"] is False
    assert body["stream_options"] == {"include_usage": True}


@pytest.mark.parametrize("think, enabled", [(None, False), (False, False), (True, True), ("high", True)])
def test_thinking_stays_off_unless_ollama_settings_asks(monkeypatch, think, enabled):
    payload = {"stream": False, "options": {}}
    if think is not None:
        payload["think"] = think
    assert _request(monkeypatch, payload)[1]["chat_template_kwargs"] == {"enable_thinking": enabled}


def test_an_image_request_loads_the_projector(monkeypatch):
    messages = [{"role": "user", "content": [{"type": "text", "text": "x"},
                                             {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]}]
    assert _request(monkeypatch, {"stream": False, "options": {}}, messages)[2]["vision"] is True


class _Stream:
    ok = True

    def __init__(self, lines):
        self.lines = lines

    def iter_lines(self):
        return iter(self.lines)


def test_llama_stream_reads_as_ollama_items():
    chunk = lambda **delta: b"data: " + json.dumps({"choices": [{"delta": delta, "finish_reason": None}]}).encode()  # noqa: E731
    items = list(ollamapull._iter_llama_stream(_Stream([
        chunk(reasoning_content="hm"), b"", chunk(content="He "), chunk(content="smiled."),
        b"data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "length"}]}).encode(),
        b"data: " + json.dumps({"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 4}}).encode(),
        b"data: [DONE]",
    ]), None))
    assert [i["message"]["content"] for i in items[:3]] == ["", "He ", "smiled."]
    assert items[0]["message"]["thinking"] == "hm"
    final = items[-1]
    assert final["done"] and final["done_reason"] == "length"
    assert (final["prompt_eval_count"], final["eval_count"]) == (3, 4)
    with pytest.raises(ollamapull.OllamaPullError):
        list(ollamapull._iter_llama_stream(_Stream([b'data: {"error": "boom"}']), None))


def test_backend_is_the_first_that_lists_a_device(tmp_path, monkeypatch):
    server = tmp_path / "llama-server.exe"
    server.write_text("")
    for variant in ("cuda_v13", "vulkan"):
        (tmp_path / variant).mkdir()
        (tmp_path / variant / "ggml-x.dll").write_text("")
    (tmp_path / "cuda_v12").mkdir()  # no library: skipped
    calls = []

    def run(args, env=None, **_kw):
        calls.append(env["GGML_BACKEND_PATH"])
        found = "vulkan" in env["GGML_BACKEND_PATH"]
        return type("Done", (), {"stdout": "Available devices:\n" + ("  Vulkan0: GPU\n" if found else "  (none)\n"), "stderr": ""})

    monkeypatch.setattr(ollamapull.subprocess, "run", run)
    monkeypatch.setattr(ollamapull, "_ban_backends", {})
    env = ollamapull._llama_backend_env(str(server))
    assert "vulkan" in env["GGML_BACKEND_PATH"] and env["PATH"].startswith(str(tmp_path / "vulkan"))
    assert len(calls) == 2 and ollamapull._llama_backend_env(str(server)) is env


def test_no_llama_server_is_an_error_not_a_silent_unbanned_run(monkeypatch):
    monkeypatch.setattr(ollamapull, "_llama_server_executable", lambda: None)
    with pytest.raises(ollamapull.OllamaPullError, match="Block other scripts"):
        ollamapull._start_ban_server("m", {}, False, None, None)


def test_a_remote_ollama_runs_unbanned_and_says_so(monkeypatch):
    monkeypatch.setenv("OLLAMAPULL_BASE_URL", "http://192.168.1.5:11434")
    monkeypatch.setattr(ollamapull, "model_settings", lambda _n: {})
    posted, status = [], []

    class Response:
        ok = True

        def json(self):
            return {"message": {"content": "hi"}, "done_reason": "stop"}

        def close(self):
            pass

    monkeypatch.setattr(ollamapull.requests, "post", lambda url, **kw: posted.append(url) or Response())
    result = ollamapull.chat("ollamapull/m", [{"role": "user", "content": "x"}], stream=False, ban_scripts=ENGLISH,
                             progress=status.append)
    assert result["content"] == "hi" and posted == ["http://192.168.1.5:11434/api/chat"]
    assert any("not applied" in line for line in status)


def test_client_passes_the_ban_to_the_ollama_route():
    source = (SRC / "unified_api_client.py").read_text(encoding="utf-8")
    assert "ban_scripts=ban_scripts, ban_exempt_chars=ban_exempt" in source
    assert os.path.exists(SRC / "script_ban_core.py")
