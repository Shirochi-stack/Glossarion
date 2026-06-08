import gemini_free
import pytest
import threading
import time


def _clear_gemini_free_budget_env(monkeypatch):
    for name in (
        "GEMINI_FREE_SUBCHUNK_PROMPT_CHARS",
        "GEMINI_FREE_SUBCHUNK_URL_CHARS",
        "GEMINI_FREE_SUBCHUNK_SAFETY_CHARS",
        "GEMINI_FREE_MIN_SUBCHUNK_BODY_CHARS",
    ):
        monkeypatch.delenv(name, raising=False)


def test_gemini_free_prompt_target_clamps_oversized_setting(monkeypatch):
    _clear_gemini_free_budget_env(monkeypatch)
    monkeypatch.setenv("GEMINI_FREE_SUBCHUNK_PROMPT_CHARS", "15000")
    monkeypatch.setenv("GEMINI_FREE_SUBCHUNK_URL_CHARS", "15500")
    messages = [{"role": "user", "content": "x" * 14000}]

    should_split, prompt_chars, url_chars = gemini_free._requires_adaptive_split(messages)

    assert gemini_free._subchunk_prompt_chars() == gemini_free.DEFAULT_SUBCHUNK_PROMPT_CHARS
    assert prompt_chars > 14000
    assert url_chars < gemini_free._subchunk_url_chars()
    assert should_split is True


def test_gemini_free_prompt_target_uses_visible_setting_within_ceiling(monkeypatch):
    _clear_gemini_free_budget_env(monkeypatch)
    monkeypatch.setenv("GEMINI_FREE_SUBCHUNK_PROMPT_CHARS", "6500")
    messages = [{"role": "user", "content": "x" * 6000}]

    should_split, prompt_chars, _url_chars = gemini_free._requires_adaptive_split(messages)

    assert gemini_free._subchunk_prompt_chars() == 6500
    assert prompt_chars < gemini_free._subchunk_prompt_chars()
    assert should_split is False


def test_gemini_free_preserves_large_fixed_prompt_and_warns(monkeypatch):
    _clear_gemini_free_budget_env(monkeypatch)
    warnings = []
    messages = [
        {"role": "system", "content": "rules " * 3000},
        {"role": "user", "content": "Translate this:\n" + ("body text " * 300)},
    ]

    prepared = gemini_free._prepare_ai_mode_messages(messages, log_fn=warnings.append)

    assert prepared[0]["content"] == messages[0]["content"]
    assert prepared[1]["content"] == messages[1]["content"]
    assert any("fixed/system prompt is larger" in warning for warning in warnings)


def test_gemini_free_runtime_enforces_user_budget_without_rewriting_fixed_prompt(monkeypatch):
    _clear_gemini_free_budget_env(monkeypatch)
    monkeypatch.setenv("GEMINI_FREE_SUBCHUNK_PROMPT_CHARS", "14000")
    calls = {"once": 0, "sequential": 0}
    system_prompt = "Translate Korean naturally. " * 300
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Translate:\n" + ("그는 문을 열었다. " * 500)},
    ]

    def fake_once(*args, **kwargs):
        calls["once"] += 1
        raise AssertionError("Non-ASCII AI Mode request should be split before submit")

    def fake_sequential(*args, **kwargs):
        calls["sequential"] += 1
        metadata = kwargs["split_metadata"]
        assert metadata["url_budget_enforced"] is True
        assert metadata["fixed_prompt_over_budget"] is True
        assert metadata["fixed_url_over_budget"] is False
        for chunk in kwargs["chunks"]:
            assert chunk[0]["content"] == system_prompt
            assert gemini_free._messages_to_search_url_chars(chunk) <= metadata["url_limit_chars"]
        return {"content": "translated", "finish_reason": "stop", "raw_response": {}}

    monkeypatch.setattr(gemini_free, "_run_search_subprocess_once", fake_once)
    monkeypatch.setattr(gemini_free, "_run_search_subprocess_sequential", fake_sequential)

    result = gemini_free._run_search_subprocess(
        messages=messages,
        model="gemini",
        timeout=30,
        max_tokens=None,
    )

    assert calls == {"once": 0, "sequential": 1}
    assert result["content"] == "translated"


def test_gemini_free_url_budget_can_use_encoded_url_length(monkeypatch):
    _clear_gemini_free_budget_env(monkeypatch)
    messages = [{"role": "user", "content": "界" * 6000}]

    should_split, prompt_chars, url_chars = gemini_free._requires_adaptive_split(messages)

    assert prompt_chars < gemini_free._subchunk_prompt_chars()
    assert url_chars > gemini_free._subchunk_url_chars()
    assert should_split is False

    should_split, prompt_chars, url_chars = gemini_free._requires_adaptive_split(
        messages,
        enforce_url_budget=True,
    )
    assert should_split is True


def test_gemini_free_html_budget_uses_text_node_transport(monkeypatch):
    _clear_gemini_free_budget_env(monkeypatch)
    monkeypatch.setenv("GEMINI_FREE_SUBCHUNK_PROMPT_CHARS", "15000")
    monkeypatch.setenv("GEMINI_FREE_SUBCHUNK_URL_CHARS", "2500")
    noisy_html = "<div>" + ("<span data-extra='abcdef'></span>" * 120) + "<p>Hello</p></div>"
    messages = [{"role": "user", "content": noisy_html}]

    raw_url_chars = gemini_free._messages_to_search_url_chars(messages)
    transport = gemini_free._build_html_text_node_transport(messages)
    assert transport is not None
    prompt_chars, url_chars = gemini_free._message_budget_usage(transport["messages"])

    assert raw_url_chars > gemini_free._subchunk_url_chars()
    assert url_chars < raw_url_chars
    assert prompt_chars < gemini_free._subchunk_prompt_chars()
    assert "data-extra" not in transport["messages"][-1]["content"]


def test_gemini_free_split_chunks_stay_under_prompt_and_url_budget(monkeypatch):
    _clear_gemini_free_budget_env(monkeypatch)
    filler = "This is neutral filler for transport budget testing. " * 800
    messages = [{"role": "user", "content": filler}]

    chunks, metadata = gemini_free._split_messages_for_search_budget(
        messages,
        gemini_free._subchunk_prompt_chars(),
        enforce_url_budget=True,
        return_metadata=True,
    )

    assert len(chunks) > 1
    assert metadata["url_budget_enforced"] is True
    for chunk in chunks:
        assert len(gemini_free._messages_to_prompt(chunk)) <= metadata["prompt_limit_chars"]
        assert gemini_free._messages_to_search_url_chars(chunk) <= metadata["url_limit_chars"]


def test_gemini_free_ai_mode_split_ignores_url_budget_and_limits_chunk_count(monkeypatch):
    _clear_gemini_free_budget_env(monkeypatch)
    monkeypatch.setenv("GEMINI_FREE_SUBCHUNK_PROMPT_CHARS", "14000")
    monkeypatch.setenv("GEMINI_FREE_SUBCHUNK_URL_CHARS", "14500")
    monkeypatch.setenv("GEMINI_FREE_SUBCHUNK_SAFETY_CHARS", "600")
    monkeypatch.setenv("GEMINI_FREE_MIN_SUBCHUNK_BODY_CHARS", "80")

    messages = [
        {"role": "system", "content": "s" * 14200},
        {"role": "user", "content": "Translate this:\n" + ("body text " * 600)},
    ]

    chunks, metadata = gemini_free._split_messages_for_search_budget(
        messages,
        gemini_free._subchunk_prompt_chars(),
        return_metadata=True,
    )

    assert gemini_free._subchunk_prompt_chars() == gemini_free.DEFAULT_SUBCHUNK_PROMPT_CHARS
    assert metadata["fixed_prompt_chars"] > gemini_free._subchunk_prompt_chars()
    assert metadata["url_budget_enforced"] is False
    assert metadata["body_budget_chars"] > 80
    assert len(chunks) <= 5
    for chunk in chunks:
        assert len(gemini_free._messages_to_prompt(chunk)) <= metadata["prompt_limit_chars"]


def test_gemini_free_accepts_plain_text_for_html_request():
    page_data = {
        "answerHtml": "<span>Hello tags.</span>",
        "answerText": "Hello tags.",
        "text": "You said:\n<p>Hello</p>\nHello tags.",
    }

    assert gemini_free._extract_rendered_content(page_data, prefer_html=True) == "Hello tags."


def test_gemini_free_rejects_generation_failure_answer_text_fallback():
    page_data = {
        "answerHtml": "",
        "answerText": "Something went wrong and the content wasn't generated.",
        "text": "You said:\n<p>Hello</p>\nSomething went wrong and the content wasn't generated.",
    }

    with pytest.raises(RuntimeError, match="generation failure"):
        gemini_free._extract_rendered_content(page_data, prefer_html=True)

    with pytest.raises(RuntimeError, match="generation failure"):
        gemini_free._extract_rendered_content(page_data, prefer_markdown=True)


def test_gemini_free_html2text_response_uses_browser_html(monkeypatch):
    monkeypatch.setenv("USE_HTML2TEXT", "1")
    page_data = {
        "answerHtml": "<div><h1>Chapter Title</h1><p><strong>Bold</strong> text.</p><ul><li>First item</li><li>Second item</li></ul></div>",
        "answerText": "Chapter Title\nBold text.\nFirst item\nSecond item",
        "text": "You said:\nTranslate markdown\nChapter Title\nBold text.\nFirst item\nSecond item",
    }

    content = gemini_free._extract_rendered_content(
        page_data,
        prompt="Translate markdown",
        prefer_markdown=gemini_free._messages_expect_markdown_response([
            {"role": "user", "content": "Translate this html2text markdown."}
        ]),
    )

    assert "# Chapter Title" in content
    assert "**Bold** text." in content
    assert "First item" in content
    assert content != page_data["answerText"]


def test_gemini_free_html2text_prefers_browser_markdown(monkeypatch):
    monkeypatch.setenv("USE_HTML2TEXT", "1")
    page_data = {
        "answerHtml": "<div><span>Chapter Title</span><span>Bold text.</span></div>",
        "answerMarkdown": "# Chapter Title\n\n**Bold** text.",
        "answerText": "Chapter Title\nBold text.",
        "text": "You said:\nTranslate markdown\nChapter Title\nBold text.",
    }

    content = gemini_free._extract_rendered_content(
        page_data,
        prompt="Translate markdown",
        prefer_markdown=True,
    )

    assert content == "# Chapter Title\n\n**Bold** text."


def test_gemini_free_html_split_keeps_instruction_prefix_with_body(monkeypatch):
    _clear_gemini_free_budget_env(monkeypatch)
    monkeypatch.setenv("GEMINI_FREE_SUBCHUNK_PROMPT_CHARS", "1000")
    monkeypatch.setenv("GEMINI_FREE_SUBCHUNK_URL_CHARS", "2400")
    monkeypatch.setenv("GEMINI_FREE_SUBCHUNK_SAFETY_CHARS", "150")
    html = "".join(f"<p>Line {index}: Hello <b>tagged</b> world.</p>" for index in range(1, 25))
    messages = [{
        "role": "user",
        "content": "Return ONLY valid HTML, preserving all tags.\n<div>" + html + "</div>",
    }]

    chunks, _ = gemini_free._split_messages_for_search_budget(
        messages,
        gemini_free._subchunk_prompt_chars(),
        return_metadata=True,
    )

    assert len(chunks) > 1
    for chunk in chunks:
        user_text = next(message["content"] for message in chunk if message["role"] == "user")
        assert "Return ONLY valid HTML" in user_text
        assert "<p>" in user_text


def test_gemini_free_html_subchunks_are_balanced_on_block_boundaries(monkeypatch):
    _clear_gemini_free_budget_env(monkeypatch)
    monkeypatch.setenv("GEMINI_FREE_SUBCHUNK_PROMPT_CHARS", "500")
    monkeypatch.setenv("GEMINI_FREE_SUBCHUNK_URL_CHARS", "50000")
    monkeypatch.setenv("GEMINI_FREE_SUBCHUNK_SAFETY_CHARS", "0")
    paragraph = "<p>" + ("word " * 20).strip() + "</p>"
    html = "<div>" + (paragraph * 9) + "</div>"

    chunks, metadata = gemini_free._split_messages_for_search_budget(
        [{"role": "user", "content": html}],
        gemini_free._subchunk_prompt_chars(),
        return_metadata=True,
    )

    paragraph_counts = [
        next(message["content"] for message in chunk if message["role"] == "user").count("<p>")
        for chunk in chunks
    ]
    assert len(chunks) == 3
    assert paragraph_counts == [3, 3, 3]
    assert metadata["subchunk_balancer"] == "balanced"
    for chunk in chunks:
        assert len(gemini_free._messages_to_prompt(chunk)) <= metadata["prompt_limit_chars"]


def test_gemini_free_always_uses_ai_mode_ui_submit(monkeypatch):
    monkeypatch.setenv("GEMINI_FREE_SUBMIT_MODE", "url")
    calls = {"url": 0, "ui": 0}

    def fail_url_load(*args, **kwargs):
        calls["url"] += 1
        raise AssertionError("URL submit mode should not be used")

    def ui_submit(*args, **kwargs):
        calls["ui"] += 1
        return {
            "submit_mode": "ui",
            "url": "https://www.google.com/search?udm=50",
            "title": "ok",
            "ready": "complete",
            "htmlLength": 100,
            "answerHtml": "",
            "text": "You said:\nhello\nfallback answer",
        }

    monkeypatch.setattr(gemini_free, "load_rendered_page_text", fail_url_load)
    monkeypatch.setattr(gemini_free, "load_ai_mode_prompt_text", ui_submit)

    result = gemini_free._send_chat_completion_qt_once(
        messages=[{"role": "user", "content": "hello"}],
        model="gemini",
        timeout=5,
    )

    assert calls == {"url": 0, "ui": 1}
    assert result["content"] == "fallback answer"
    assert result["raw_response"]["submit_mode"] == "ui"
    assert result["raw_response"]["url_load_fallback"] is False


def test_gemini_free_html_request_translates_text_nodes_and_restores_tags(monkeypatch):
    monkeypatch.setenv("GEMINI_FREE_SUBMIT_MODE", "url")
    calls = {"url": 0, "ui": 0}

    def fail_url_load(*args, **kwargs):
        calls["url"] += 1
        raise AssertionError("URL submit mode should not be used")

    def text_node_answer(prompt, *args, **kwargs):
        calls["ui"] += 1
        assert "<div" not in prompt
        assert "<p" not in prompt
        return {
            "submit_mode": "ui",
            "url": "https://www.google.com/search?q=segments",
            "title": "plain",
            "ready": "complete",
            "htmlLength": 100,
            "answerHtml": "",
            "answerText": "[1] Hello\n[2] world\n[3] again",
            "text": "You said:\nsegments\n[1] Hello\n[2] world\n[3] again",
        }

    monkeypatch.setattr(gemini_free, "load_rendered_page_text", fail_url_load)
    monkeypatch.setattr(gemini_free, "load_ai_mode_prompt_text", text_node_answer)

    result = gemini_free._send_chat_completion_qt_once(
        messages=[{"role": "user", "content": "<div><p>안녕 <b>세계</b>.</p></div>"}],
        model="gemini",
        timeout=5,
    )

    assert calls == {"url": 0, "ui": 1}
    assert result["content"] == "<div><p>Hello <b>world</b>again</p></div>"
    assert result["raw_response"]["submit_mode"] == "ui"
    assert result["raw_response"]["html_text_node_transport"] is True
    assert result["raw_response"]["html_text_node_count"] == 3
    assert result["raw_response"]["html_text_node_translated"] == 3


def test_gemini_free_text_node_parser_ignores_unnumbered_filler():
    parsed = gemini_free._parse_html_text_node_translations(
        "[1] Hello\n[2] world\nPlease send more text if needed.",
        2,
    )

    assert parsed == {1: "Hello", 2: "world"}


def test_gemini_free_subchunk_timeout_zero_disables_timeout(monkeypatch):
    monkeypatch.delenv("GEMINI_FREE_SUBCHUNK_TIMEOUT", raising=False)

    assert gemini_free._subchunk_timeout_seconds(90) == 0
    assert gemini_free._timeout_label(gemini_free._subchunk_timeout_seconds(90), 0) == "disabled"

    monkeypatch.setenv("GEMINI_FREE_SUBCHUNK_TIMEOUT", "180")
    assert gemini_free._subchunk_timeout_seconds(90) == 180


def test_gemini_free_subprocess_skips_adaptive_split_for_html_transport(monkeypatch):
    _clear_gemini_free_budget_env(monkeypatch)
    monkeypatch.setenv("GEMINI_FREE_SUBCHUNK_PROMPT_CHARS", "1000")
    monkeypatch.setenv("GEMINI_FREE_SUBCHUNK_URL_CHARS", "1000")
    calls = {"once": 0, "sequential": 0}
    noisy_html = "<div>" + ("<span data-extra='abcdef'></span>" * 120) + "<p>Hello</p></div>"

    def fake_once(*args, **kwargs):
        calls["once"] += 1
        return {"content": "<div><p>Hello</p></div>", "finish_reason": "stop", "raw_response": {}}

    def fake_sequential(*args, **kwargs):
        calls["sequential"] += 1
        raise AssertionError("HTML text-node transport should not use adaptive subchunks")

    monkeypatch.setattr(gemini_free, "_run_search_subprocess_once", fake_once)
    monkeypatch.setattr(gemini_free, "_run_search_subprocess_sequential", fake_sequential)

    result = gemini_free._run_search_subprocess(
        messages=[{"role": "user", "content": noisy_html}],
        model="gemini",
        timeout=5,
        max_tokens=None,
    )

    assert calls == {"once": 1, "sequential": 0}
    assert result["content"] == "<div><p>Hello</p></div>"


def test_gemini_free_subprocess_uses_html_split_when_text_node_transport_still_over_budget(monkeypatch):
    _clear_gemini_free_budget_env(monkeypatch)
    monkeypatch.setenv("GEMINI_FREE_SUBCHUNK_PROMPT_CHARS", "1000")
    monkeypatch.setenv("GEMINI_FREE_SUBCHUNK_URL_CHARS", "1000")
    calls = {"once": 0, "sequential": 0}
    html = "<div>" + "".join(
        f"<p>Line {index}: Hello <b>tagged</b> world.</p>"
        for index in range(1, 25)
    ) + "</div>"

    transport = gemini_free._build_html_text_node_transport([{"role": "user", "content": html}])
    assert transport is not None
    assert gemini_free._requires_adaptive_split(transport["messages"])[0] is True

    def fake_once(*args, **kwargs):
        calls["once"] += 1
        raise AssertionError("Oversized text-node transport should not bypass adaptive subchunks")

    def fake_sequential(*args, **kwargs):
        calls["sequential"] += 1
        assert kwargs["split_metadata"]["splitter"] == "beautifulsoup4"
        return {"content": "<div><p>split</p></div>", "finish_reason": "stop", "raw_response": {}}

    monkeypatch.setattr(gemini_free, "_run_search_subprocess_once", fake_once)
    monkeypatch.setattr(gemini_free, "_run_search_subprocess_sequential", fake_sequential)

    result = gemini_free._run_search_subprocess(
        messages=[{"role": "user", "content": html}],
        model="gemini",
        timeout=5,
        max_tokens=None,
    )

    assert calls == {"once": 0, "sequential": 1}
    assert result["content"] == "<div><p>split</p></div>"


def test_gemini_free_subchunks_use_authnd_token_concurrency_when_override_zero(monkeypatch):
    monkeypatch.setenv("AUTHND_TOKEN_CONCURRENCY_AUTO", "0")
    monkeypatch.setenv("AUTHND_TOKEN_CONCURRENCY", "2")
    monkeypatch.setenv("AUTHND_TOKEN_SUBPROCESS_CONCURRENCY", "8")
    monkeypatch.setenv("GEMINI_FREE_SUBCHUNK_START_DELAY", "0")
    active = 0
    max_active = 0
    lock = threading.Lock()

    def fake_once(messages, **kwargs):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        try:
            marker = messages[0]["content"]
            return {"content": marker, "finish_reason": "stop", "raw_response": {}}
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(gemini_free, "_run_search_subprocess_once", fake_once)
    chunks = [[{"role": "user", "content": f"part-{index}"}] for index in range(1, 4)]

    result = gemini_free._run_search_subprocess_sequential(
        source_messages=[{"role": "user", "content": "source"}],
        chunks=chunks,
        split_metadata={"target_prompt_chars": 1000},
        model="gemini",
        timeout=30,
        max_tokens=None,
    )

    assert max_active == 2
    assert result["content"] == "part-1\npart-2\npart-3"
    assert result["raw_response"]["submit_mode"] == "adaptive_split_parallel"
    assert result["raw_response"]["subchunk_concurrency"] == 2
    assert result["raw_response"]["subchunk_start_delay_seconds"] == 0.0


def test_gemini_free_subchunks_stagger_parallel_start(monkeypatch):
    monkeypatch.setenv("GEMINI_FREE_SUBCHUNK_CONCURRENCY", "3")
    monkeypatch.setenv("GEMINI_FREE_SUBCHUNK_START_DELAY", "0.05")
    starts = []
    lock = threading.Lock()

    def fake_once(messages, **kwargs):
        marker = messages[0]["content"]
        with lock:
            starts.append((marker, time.monotonic()))
        time.sleep(0.1)
        return {"content": marker, "finish_reason": "stop", "raw_response": {}}

    monkeypatch.setattr(gemini_free, "_run_search_subprocess_once", fake_once)
    chunks = [[{"role": "user", "content": f"part-{index}"}] for index in range(1, 4)]

    result = gemini_free._run_search_subprocess_sequential(
        source_messages=[{"role": "user", "content": "source"}],
        chunks=chunks,
        split_metadata={"target_prompt_chars": 1000},
        model="gemini",
        timeout=30,
        max_tokens=None,
    )

    ordered_starts = [started_at for _marker, started_at in sorted(starts)]
    assert len(ordered_starts) == 3
    assert ordered_starts[1] - ordered_starts[0] >= 0.04
    assert ordered_starts[2] - ordered_starts[1] >= 0.04
    assert result["content"] == "part-1\npart-2\npart-3"
    assert result["raw_response"]["subchunk_start_delay_seconds"] == 0.05


def test_gemini_free_subchunk_generation_failure_content_raises(monkeypatch):
    monkeypatch.setenv("AUTHND_TOKEN_CONCURRENCY_AUTO", "0")
    monkeypatch.setenv("AUTHND_TOKEN_CONCURRENCY", "1")
    monkeypatch.setenv("AUTHND_TOKEN_SUBPROCESS_CONCURRENCY", "1")

    def fake_once(messages, **kwargs):
        return {
            "content": "Something went wrong and the content wasn't generated.",
            "finish_reason": "stop",
            "raw_response": {},
        }

    monkeypatch.setattr(gemini_free, "_run_search_subprocess_once", fake_once)

    with pytest.raises(RuntimeError, match="generation failure"):
        gemini_free._run_search_subprocess_sequential(
            source_messages=[{"role": "user", "content": "source"}],
            chunks=[[{"role": "user", "content": "part-1"}]],
            split_metadata={"target_prompt_chars": 1000},
            model="gemini",
            timeout=30,
            max_tokens=None,
        )
