"""Hugging Face references (``services/local_llm/hf.py``): every accepted form, the rejects, the file
listing (mmproj / imatrix / split shards left out) and the auto-pick order. The tree API is a fixture
shaped like the real unsloth/Qwen3.5-0.8B-GGUF listing, served by a local fake (no network).
"""

from __future__ import annotations

import pytest

import local_llm_fakes as fakes
from glossarion_mobile.services.local_llm import hf


@pytest.mark.parametrize("text,expected", [
    ("unsloth/Qwen3.5-2B-GGUF", hf.HfRef("unsloth/Qwen3.5-2B-GGUF")),
    ("unsloth/Qwen3.5-2B-GGUF:Q4_K_M", hf.HfRef("unsloth/Qwen3.5-2B-GGUF", quant="Q4_K_M")),
    ("hf.co/unsloth/Qwen3.5-2B-GGUF:q8_0", hf.HfRef("unsloth/Qwen3.5-2B-GGUF", quant="q8_0")),
    ("  hf.co/bartowski/Some.Model-GGUF ", hf.HfRef("bartowski/Some.Model-GGUF")),
    ("unsloth/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q4_K_M.gguf",
     hf.HfRef("unsloth/Qwen3.5-2B-GGUF", file="Qwen3.5-2B-Q4_K_M.gguf")),
    ("unsloth/big-GGUF/Q4_K_M/big-Q4_K_M-00001-of-00002.gguf",
     hf.HfRef("unsloth/big-GGUF", file="Q4_K_M/big-Q4_K_M-00001-of-00002.gguf")),
    ("https://huggingface.co/unsloth/Qwen3.5-2B-GGUF", hf.HfRef("unsloth/Qwen3.5-2B-GGUF")),
    ("https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/tree/dev", hf.HfRef("unsloth/Qwen3.5-2B-GGUF", revision="dev")),
    ("https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/blob/main/Qwen3.5-2B-Q4_K_M.gguf",
     hf.HfRef("unsloth/Qwen3.5-2B-GGUF", file="Qwen3.5-2B-Q4_K_M.gguf")),
    ("https://hf.co/unsloth/Qwen3.5-2B-GGUF/resolve/0123abc/Qwen3.5-2B-Q4_K_M.gguf?download=true",
     hf.HfRef("unsloth/Qwen3.5-2B-GGUF", file="Qwen3.5-2B-Q4_K_M.gguf", revision="0123abc")),
])
def test_parse_ref_accepts(text, expected):
    assert hf.parse_ref(text) == expected


@pytest.mark.parametrize("text,message", [
    ("", "Enter a Hugging Face GGUF repo"),
    ("qwen3.5:2b", "is not a Hugging Face repo id"),
    ("just-a-name", "is not a Hugging Face repo id"),
    ("a/b:", "is not a quant name"),
    ("a/b:Q4 K", "is not a quant name"),
    ("https://example.com/a/b", "Only huggingface.co links"),
    ("ftp://huggingface.co/a/b", "Only huggingface.co links"),
    ("https://huggingface.co/a", "Enter a Hugging Face GGUF repo"),
    ("https://huggingface.co/models/a/b", "Enter a Hugging Face GGUF repo"),
    ("https://huggingface.co/a/b/blob/main/model.safetensors", "Only .gguf files"),
    ("a/b/../../etc/passwd.gguf", "not allowed"),
    ("a/../b", "is not a Hugging Face repo id"),
    ("https://huggingface.co/a/b/resolve/../x.gguf", "not a branch or commit"),
])
def test_parse_ref_rejects(text, message):
    with pytest.raises(hf.HfError) as info:
        hf.parse_ref(text)
    assert message in str(info.value)


@pytest.fixture()
def fake_hf(monkeypatch):
    server = fakes.FakeHF()
    monkeypatch.setattr(hf, "HF_ENDPOINT", server.url)
    yield server
    server.close()


def test_list_files_leaves_out_projectors_imatrix_and_later_shards(fake_hf):
    fake_hf.trees["unsloth/Qwen3.5-0.8B-GGUF"] = fakes.QWEN_08B_TREE + [
        ("Q8/big-Q8_0-00001-of-00002.gguf", 10, "1" * 64), ("Q8/big-Q8_0-00002-of-00002.gguf", 10, "2" * 64)]
    commit, files = hf.list_files(hf.HfRef("unsloth/Qwen3.5-0.8B-GGUF"))
    assert commit == fakes.FakeHF.COMMIT
    names = [f.path for f in files]
    assert "Qwen3.5-0.8B-Q4_K_M.gguf" in names and "Q8/big-Q8_0-00001-of-00002.gguf" in names
    assert not any(n.startswith("mmproj") or "imatrix" in n or n.endswith(".md") or "00002-of" in n for n in names)
    q4 = next(f for f in files if f.name == "Qwen3.5-0.8B-Q4_K_M.gguf")
    assert q4.size == 532517120 and q4.sha256 == "bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517"
    assert next(f for f in files if f.path.startswith("Q8/")).split
    # the tree is read at the resolved commit, so a resumed download fetches the same bytes
    assert any(f"/tree/{fakes.FakeHF.COMMIT}" in path for path, _r in fake_hf.requests)


def test_list_files_errors(fake_hf):
    with pytest.raises(hf.HfError) as info:
        hf.list_files(hf.HfRef("nobody/missing-GGUF"))
    assert "gated or private" in str(info.value)
    fake_hf.status = 429
    with pytest.raises(hf.HfError) as info:
        hf.list_files(hf.HfRef("nobody/missing-GGUF"))
    assert "rate limiting" in str(info.value)


def _files():
    rows = [r for r in fakes.QWEN_08B_TREE if r[0].endswith(".gguf") and not r[0].startswith("mmproj")]
    return [hf.HfFile(p, s, h) for p, s, h in rows]


def test_pick_file_order_and_choices():
    files = _files()
    assert hf.pick_file(files).name == "Qwen3.5-0.8B-Q4_K_M.gguf"  # not UD-Q4_K_XL, not IQ4_XS
    assert hf.pick_file(files, quant="q8_0").name == "Qwen3.5-0.8B-Q8_0.gguf"
    assert hf.pick_file(files, quant="IQ4_XS").name == "Qwen3.5-0.8B-IQ4_XS.gguf"
    assert hf.pick_file(files, file="Qwen3.5-0.8B-Q5_K_M.gguf").size == 590057728
    without = [f for f in files if "Q4_K_M" not in f.name]
    assert hf.pick_file(without).name == "Qwen3.5-0.8B-Q4_K_S.gguf"
    assert hf.pick_file([f for f in files if f.name.endswith(("BF16.gguf", "Q3_K_M.gguf"))]) is None  # chooser
    assert hf.pick_file([files[0]]) is files[0]
    with pytest.raises(hf.HfError, match="No Q2_K file"):
        hf.pick_file(files, quant="Q2_K")
    with pytest.raises(hf.HfError, match="not in this repository"):
        hf.pick_file(files, file="nope.gguf")
    with pytest.raises(hf.HfError, match="no GGUF model files"):
        hf.pick_file([])
    with pytest.raises(hf.HfError, match="Split GGUF files are not supported yet"):
        hf.pick_file([hf.HfFile("big-Q4_K_M-00001-of-00003.gguf", 1, None, True)])


def test_resolve_url_pins_the_commit():
    assert hf.resolve_url("unsloth/Qwen3.5-2B-GGUF", "abc123", "sub dir/x.gguf") == (
        f"{hf.HF_ENDPOINT}/unsloth/Qwen3.5-2B-GGUF/resolve/abc123/sub%20dir/x.gguf")


def test_pagination_follows_only_hugging_face_api_pages():
    ok = f'<{hf.HF_ENDPOINT}/api/models/u/r/tree/abc?cursor=x>; rel="next"'
    assert hf._next_link({"Link": ok}) == f"{hf.HF_ENDPOINT}/api/models/u/r/tree/abc?cursor=x"
    assert hf._next_link({"Link": '<https://evil.example/api/models/u/r>; rel="next"'}) is None
    assert hf._next_link({"Link": f'<{hf.HF_ENDPOINT}.evil.example/api/models/x>; rel="next"'}) is None
    assert hf._next_link({}) is None
