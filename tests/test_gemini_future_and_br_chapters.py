import ast
import hashlib
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from gemini_policy import uses_gemini_thinking_level, omit_gemini_legacy_parameters, gemini_ignores_sampling
from enhanced_text_extractor import EnhancedTextExtractor
from chapter_splitter import ChapterSplitter
import grpc_gemini_client as grpc_client


@pytest.mark.parametrize("model,expected", [
    ("gemini-3-flash", True), ("gemini-4-pro", True),
    ("or/google/gemini-10-flash", True), ("models/gemini-3.6-flash", True),
    ("gemini-2.5-pro", False), ("gemma-4", False), ("notgemini-4-pro", False),
])
def test_generation_detection(model, expected):
    assert uses_gemini_thinking_level(model) is expected


def test_nested_sampling_fields_are_omitted():
    body = {"temperature": 0.3, "max_tokens": 500,
            "extra_body": {"google": {"top_p": 0.9, "topK": 8}}}
    omit_gemini_legacy_parameters(body)
    assert body == {"max_tokens": 500, "extra_body": {"google": {}}}


@pytest.mark.parametrize("model,ignores", [
    ("gemini-2.5-pro", False), ("gemini-3-flash-preview", False), ("gemini-3.1-pro-preview", False),
    ("vertex/gemini-3.1-flash-lite", False), ("gemini-3.5-flash", True), ("gemini-3.8-flash", True),
    ("gemini-4-argon", True), ("authgem/gemini-3.6-flash", True), ("gemma-4", False),
])
def test_only_gemini_3_5_and_later_ignore_sampling(model, ignores):
    """owner 2026-10-18: Gemini 3 and 3.1 still use temperature / top_p / top_k; 3.5+ ignore them."""
    assert gemini_ignores_sampling(model) is ignores
    body = {"temperature": 0.3, "top_p": 0.9, "thinking_budget": 8192, "generation_config": {"topK": 8}}
    omit_gemini_legacy_parameters(body, model)
    assert "thinking_budget" not in body  # every Gemini 3+ payload uses levels
    assert ("temperature" in body and "top_p" in body and body["generation_config"] == {"topK": 8}) is not ignores


def test_grpc_gemini_3_1_keeps_the_temperature():
    client = grpc_client.GrpcGeminiClient.__new__(grpc_client.GrpcGeminiClient)
    config = client._build_generation_config(
        model="models/gemini-3.1-pro-preview", temperature=0.3, max_output_tokens=200,
        thinking_level="high", thinking_budget=8192, anti_dupe_params={"top_p": 0.9})
    data = type(config).to_dict(config)
    assert data.get("temperature") == pytest.approx(0.3) and data.get("top_p") == pytest.approx(0.9)


@pytest.mark.parametrize("stream_model", ["models/gemini-3.6-flash", "models/gemini-4-pro"])
def test_old_grpc_proto_uses_defaults_instead_of_numeric_budget(stream_model):
    client = grpc_client.GrpcGeminiClient.__new__(grpc_client.GrpcGeminiClient)
    config = client._build_generation_config(
        model=stream_model, temperature=0.3, max_output_tokens=200,
        thinking_level="high", thinking_budget=8192,
        anti_dupe_params={"top_p": 0.9, "top_k": 20})
    data = type(config).to_dict(config)
    assert "temperature" not in data and "top_p" not in data and "top_k" not in data
    assert "thinking_budget" not in data.get("thinking_config", {})
    assert data["max_output_tokens"] == 200


def test_grpc_can_encode_levels_when_library_supports_them(monkeypatch):
    class Thinking:
        _meta = SimpleNamespace(fields={"thinking_level": None, "thinking_budget": None})
        def __init__(self, **kwargs):
            self.kwargs = kwargs
    monkeypatch.setattr(grpc_client, "_ThinkingConfig", Thinking)
    monkeypatch.setattr(grpc_client, "GenerationConfig", lambda **kw: kw)
    client = grpc_client.GrpcGeminiClient.__new__(grpc_client.GrpcGeminiClient)
    data = client._build_generation_config(model="gemini-4-pro", thinking_level="medium")
    assert data["thinking_config"].kwargs["thinking_level"] == "MEDIUM"
    assert "thinking_budget" not in data["thinking_config"].kwargs


def test_grpc_25_still_accepts_budgets_and_sampling():
    client = grpc_client.GrpcGeminiClient.__new__(grpc_client.GrpcGeminiClient)
    config = client._build_generation_config(model="gemini-2.5-flash", temperature=0.3,
                                              thinking_budget=8192, anti_dupe_params={"top_p": 0.9})
    data = type(config).to_dict(config)
    assert data["thinking_config"]["thinking_budget"] == 8192
    assert data["temperature"] == pytest.approx(0.3)


def split_function():
    # Exercise the real routing helper without importing the application GUI.
    from html_duplicate_cleanup import prepare_html_for_model

    source = (Path(__file__).resolve().parents[1] / "src" / "TransateKRtoEN.py").read_text(encoding="utf-8")
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
                and n.name == "_split_chapter_for_translation")
    # The module-level names the function uses; 42488ba9 added prepare_html_for_model
    # (the "Remove excessive empty p and br tags" pass on the text sent to the model).
    scope = {"hashlib": hashlib, "re": re, "prepare_html_for_model": prepare_html_for_model}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "TransateKRtoEN.py", "exec"), scope)
    return scope[node.name]


@pytest.mark.parametrize("html", [
    "<body><h1>날 꼬셔줘</h1>이 세계는 로판이다.&#160;<br/><br/>나는 그 사실을 깨달았다.<br/><br/>마지막 문장이다.</body>",
    "<body><div><h1>Title</h1>First bare line.<br/>Second bare line.<p>Wrapped paragraph.</p>Final bare line.</div></body>",
    "<body>Before heading.<br/><br/><h1>Title</h1>After heading.<br/><br/><h2>Subtitle</h2>Last line.</body>",
    "<body>First line.<br/>Second line.<br/><br/>Last line.</body>",
])
def test_br_prose_survives_extraction_and_chunking(html, monkeypatch):
    monkeypatch.setenv("REMOVE_DUPLICATE_H1_P", "0")
    extractor = EnhancedTextExtractor()
    text, _, _ = extractor.extract_chapter_content(html, "comprehensive")
    blocks = extractor.last_html2text_blocks
    assert re.sub(r"\s+", "", "".join(blocks)) == re.sub(r"\s+", "", text)
    chapter = {"body": text, "enhanced_extraction": True, "html2text_blocks": blocks,
               "html2text_blocks_source_hash": hashlib.sha256(text.encode()).hexdigest()}
    chunks = split_function()(ChapterSplitter(), chapter, 30)
    combined = "".join(chunk[0] for chunk in chunks)
    assert re.sub(r"\s+", "", combined) == re.sub(r"\s+", "", text)
    assert "GLXH2TBLOCK" not in combined


def test_cached_header_only_blocks_fall_back_to_full_body():
    text = "# Title\n\nFirst body line.\n\nLast body line."
    chapter = {"body": text, "enhanced_extraction": True, "html2text_blocks": ["# Title"],
               "html2text_blocks_source_hash": hashlib.sha256(text.encode()).hexdigest()}
    chunks = split_function()(ChapterSplitter(), chapter, 100)
    assert "First body line." in chunks[0][0] and "Last body line." in chunks[0][0]
