"""Glossary requests fit the model's context window (owner 2026-10-18).

On-device local/Qwen3.5-0.8B (8,192-token context, 4,096 output limit) was sent a ~20,000-token Minimal
glossary request three times: the chunk budget ignored the context window and the client retried the
"Context length exceeded" 400 as if it were transient. The chunk budget is now min(output-based,
context-based), a context-length error is never retried, and the glossary splits the chunk instead.
"""

from types import SimpleNamespace

import pytest

import GlossaryManager as GM
import extract_glossary_from_epub as egl
import unified_api_client as uac
from unified_api_client import UnifiedClientError

LOCAL = "local/Qwen3.5-0.8B-UD-Q4_K_XL"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in (uac.LOCAL_CONTEXT_ENV, "TRANSLATION_CANCELLED", "GRACEFUL_STOP", "GRACEFUL_STOP_COMPLETED"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(uac, "_KNOWN_CONTEXT_WINDOWS", {})
    monkeypatch.setattr(GM, "_stop_requested", False)
    monkeypatch.setattr(GM, "_get_stop_file_path", lambda: None)


# ---------------------------------------------------------------------------------------- classification

@pytest.mark.parametrize("message, expected", [
    ("Context length exceeded: max_context=8192, input≈20076.", (8192, 20076)),
    ("Error code: 400 - {'error': {'message': \"This model's maximum context length is 8192 tokens. However, "
     "your messages resulted in 20076 tokens. Please reduce the length of the messages.\", "
     "'code': 'context_length_exceeded'}}", (8192, 20076)),
    ("prompt is too long: 210,000 tokens > 200,000 maximum", (200000, 210000)),
    ("The input token count (1200000) exceeds the maximum number of tokens allowed (1048576).", (1048576, 1200000)),
    ("context_length_exceeded", (None, None)),
])
def test_context_length_errors_are_recognised_with_their_sizes(message, expected):
    info = uac.context_length_error_info(RuntimeError(message))
    assert (info["max_context"], info["input_tokens"]) == expected


@pytest.mark.parametrize("message", ["Error code: 429 - rate limit", "HTTP 500 internal server error",
                                     "Request timed out", "max_tokens: 64000 > 32000"])
def test_other_errors_are_not_context_errors(message):
    assert uac.context_length_error_info(RuntimeError(message)) is None
    assert uac.context_length_error(RuntimeError(message)) is None


def test_a_context_error_becomes_non_retryable_and_remembers_the_window():
    err = uac.context_length_error(RuntimeError("Context length exceeded: max_context=8192, input≈20076."), "gpt-x")
    assert isinstance(err, UnifiedClientError) and err.error_type == "context_length"
    assert err.details == {"max_context": 8192, "input_tokens": 20076}
    assert uac.model_context_window("GPT-X") == 8192


def test_local_models_read_their_context_from_the_mobile_job_env(monkeypatch):
    monkeypatch.setenv(uac.LOCAL_CONTEXT_ENV, f"local/Other=4096;{LOCAL}=8192")
    assert uac.model_context_window(LOCAL) == 8192
    assert uac.model_context_window("local/Other") == 4096
    assert uac.model_context_window("local/Missing") is None
    assert uac.model_context_window("gpt-x") is None


def _fake_client():
    return SimpleNamespace(
        model=LOCAL, _get_send_interval=lambda: 0, _is_stop_requested=lambda: False,
        _is_instance_cancel_requested=lambda: False, _debug_log=lambda *a, **k: None,
        _sleep_with_cancel=lambda *a, **k: True,
    )


def test_the_client_never_resends_a_request_that_overflows_the_context():
    calls = []

    def call():
        calls.append(1)
        raise RuntimeError("Context length exceeded: max_context=8192, input≈20076.")

    with pytest.raises(UnifiedClientError) as caught:
        uac.UnifiedClient._with_sdk_retries(_fake_client(), "local", 3, call)
    assert caught.value.error_type == "context_length" and len(calls) == 1


def test_other_errors_keep_the_normal_retry():
    calls = []

    def call():
        calls.append(1)
        raise RuntimeError("HTTP 500 internal server error")

    with pytest.raises(UnifiedClientError) as caught:
        uac.UnifiedClient._with_sdk_retries(_fake_client(), "local", 3, call)
    assert caught.value.error_type != "context_length" and len(calls) == 3


# ---------------------------------------------------------------------------------------- chunk budget

def test_chunk_budget_is_the_smaller_of_the_output_and_context_budgets(monkeypatch):
    monkeypatch.setenv(uac.LOCAL_CONTEXT_ENV, f"{LOCAL}=8192")
    output_based = egl._compute_safe_input_tokens(4096, 1.0)
    assert output_based == 3596
    # 8,192 context - 4,096 output - 500 prompt, less 10% for tokenizer differences
    assert egl._compute_glossary_input_budget(4096, 1.0, LOCAL, 500) == (3236, 8192, 3236)
    # a large window leaves the output-based budget in charge
    monkeypatch.setenv(uac.LOCAL_CONTEXT_ENV, f"{LOCAL}=131072")
    assert egl._compute_glossary_input_budget(4096, 1.0, LOCAL, 500)[0] == output_based
    # unknown window: output-based only
    assert egl._compute_glossary_input_budget(4096, 1.0, "gpt-x", 500) == (output_based, None, None)
    # an output limit as large as the window still leaves a usable minimum
    monkeypatch.setenv(uac.LOCAL_CONTEXT_ENV, f"{LOCAL}=4096")
    assert egl._compute_glossary_input_budget(4096, 1.0, LOCAL, 500)[0] == egl.CONTEXT_INPUT_MINIMUM


def test_minimal_glossary_budget_counts_its_prompt(monkeypatch):
    monkeypatch.setenv(uac.LOCAL_CONTEXT_ENV, f"{LOCAL}=8192")
    monkeypatch.setenv("MODEL", LOCAL)
    prompt = "Extract names. " * 100
    budget, window, context_budget = GM._minimal_chunk_budget(4096, 1.0, prompt)
    assert window == 8192 and budget == context_budget
    assert budget == int((8192 - 4096 - GM._glossary_prompt_tokens(prompt)) * 0.9)
    assert budget < egl._compute_safe_input_tokens(4096, 1.0)


def test_minimal_glossary_splits_to_the_context_window_even_with_chapter_split_off(monkeypatch, tmp_path):
    """The owner's log: chapter splitting off, a 4,096 output limit, an 8,192 context - the whole text was
    sent in one request. A known context window now forces token-based chunks that fit it."""
    for name in ("OUTPUT_DIRECTORY", "EPUB_PATH", "MANUAL_GLOSSARY", "GLOSSARY_ENABLE_CHAPTER_SPLIT",
                 "GLOSSARY_CHAPTER_SPLIT_THRESHOLD", "BATCH_TRANSLATION", "GLOSSARY_COMPRESSION_FACTOR",
                 "COMPRESSION_FACTOR", "GLOSSARY_DEFER_SAVE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(uac.LOCAL_CONTEXT_ENV, f"{LOCAL}=2048")
    monkeypatch.setenv("MODEL", LOCAL)
    monkeypatch.setenv("GLOSSARY_MAX_OUTPUT_TOKENS", "1024")
    monkeypatch.setenv("AUTO_GLOSSARY_PROMPT", "Extract names.")
    monkeypatch.setenv("GLOSSARY_USE_SMART_FILTER", "0")
    chunks = []

    def fake_extract(prompt, text, *args, **kwargs):
        chunks.append(text)
        return []

    monkeypatch.setattr(GM, "_extract_with_custom_prompt", fake_extract)
    chapters = [{"body": f"<p>Chapter {i}. " + "The knight rode through the long valley at dawn. " * 6 + "</p>"}
                for i in range(40)]
    GM.save_glossary(str(tmp_path), chapters, instructions="")
    budget = GM._minimal_chunk_budget(1024, 1.0, "Extract names.")[0]
    assert len(chunks) > 1
    assert max(GM._glossary_prompt_tokens(chunk) for chunk in chunks) <= budget


# ---------------------------------------------------------------------------------------- split on error

def _run_minimal(monkeypatch, send):
    monkeypatch.setenv("MODEL", LOCAL)
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("GLOSSARY_DEFER_SAVE", "1")
    monkeypatch.setenv("_CHUNK_ALREADY_FILTERED", "1")
    monkeypatch.setenv("THREAD_SUBMISSION_DELAY_SECONDS", "0")
    monkeypatch.setenv("GLOSSARY_MAX_OUTPUT_TOKENS", "4096")

    class FakeClient:
        def __init__(self, *a, **k):
            self.context = None

    monkeypatch.setattr(uac, "UnifiedClient", FakeClient)
    monkeypatch.setattr(GM, "send_with_interrupt", send)
    names = [f"Name{i:02d}" for i in range(8)]
    text = "\n".join(f"{name} walked into the hall and spoke at length about the weather." for name in names)
    lines = GM._extract_with_custom_prompt("Extract names.", text, "korean", 1, 50, 50, None, ".",
                                           raise_on_failure=True)
    return names, lines


def _respond_with_names(messages):
    user = messages[-1]["content"]
    rows = [f"character,{word},{word}-en" for word in user.split() if word.startswith("Name")]
    return SimpleNamespace(content="type,raw_name,translated_name\n" + "\n".join(rows)), "stop", None


def test_a_context_error_splits_the_chunk_and_merges_every_part(monkeypatch, capsys):
    sent = []

    def send(messages, **kwargs):
        user = messages[-1]["content"]
        sent.append(len(user))
        if len(user) > 300:
            raise UnifiedClientError("Context length exceeded: max_context=8192, input≈20076.",
                                     error_type="context_length",
                                     details={"max_context": 8192, "input_tokens": 20076})
        return _respond_with_names(messages)

    names, lines = _run_minimal(monkeypatch, send)
    assert GM._is_glossary_header(lines[0])
    assert sum(GM._is_glossary_header(line) for line in lines) == 1
    assert sorted(GM._gsplit(line)[1] for line in lines[1:]) == names  # entries from all parts, none lost
    assert len(sent) > 2 and max(sent[1:]) <= 300 < sent[0]
    out = capsys.readouterr().out
    assert "📐 Context limit hit (8,192): splitting chunk into 5 parts" in out  # 20,076 / (8,192 - 4,096)


def test_a_non_context_error_is_not_split(monkeypatch):
    sent = []

    def send(messages, **kwargs):
        sent.append(messages[-1]["content"])
        raise UnifiedClientError("HTTP 500 internal server error", error_type="api_error")

    with pytest.raises(UnifiedClientError):
        _run_minimal(monkeypatch, send)
    assert len(sent) == 1


def test_split_text_evenly_keeps_every_character_and_cuts_at_lines():
    text = "\n".join(f"line {i} " + "x" * 30 for i in range(20))
    pieces = GM._split_text_evenly(text, 4)
    assert len(pieces) == 4 and "".join(pieces) == text
    assert all(piece.endswith("\n") for piece in pieces[:-1])
