"""On-device model presets (``services/local_llm/presets.py``): editable Prefs data, the verified defaults,
the one-time migration of the old Ollama presets, validation.

Run from src/mobile:
    .venv/Scripts/python.exe -m pytest -p no:cacheprovider -W ignore tests_host/test_local_llm_presets.py
"""

from __future__ import annotations

import local_llm_fakes as fakes  # noqa: F401  (sys.path)
from local_llm_fakes import FakePrefs

from glossarion_mobile.services.local_llm import presets


def test_defaults_are_the_four_verified_qwen35_ggufs():
    rows = presets.load_presets(FakePrefs())
    assert [(r["name"], r["repo"], r["file"], r["size"], r["sha256"]) for r in rows] == [
        ("Garbage", "unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-UD-Q4_K_XL.gguf", 558772480,
         "3177ebd67afe4438374da19e690bc1b98756f7e0fea9240e1be404336156a7b5"),
        ("Light", "unsloth/Qwen3.5-2B-GGUF", "Qwen3.5-2B-UD-Q4_K_XL.gguf", 1339752704,
         "0af96165ea615bea39a04118d63f0b6d35908aea850ee4a51aa6151d851b8b35"),
        ("Medium", "unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-UD-Q4_K_XL.gguf", 2912109728,
         "b252c5610a42ca82d20fe2a12813e9d069eed89292907e26c783eeb0bc961bc7"),
        ("Heavy", "unsloth/Qwen3.5-9B-GGUF", "Qwen3.5-9B-Q4_K_M.gguf", 5680522464,
         "03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8"),
    ]
    assert all(r["n_ctx"] == 0 and r["note"] for r in rows)
    rows[0]["name"] = "changed"  # a copy, never the module data
    assert presets.DEFAULT_PRESETS[0]["name"] == "Garbage"


def test_edit_rename_remove_restore_persist_in_prefs():
    prefs = FakePrefs()
    rows = presets.load_presets(prefs)
    rows[1]["name"] = "Fast"
    rows.pop(3)
    presets.save_presets(prefs, rows)
    assert [r["name"] for r in presets.load_presets(prefs)] == ["Garbage", "Fast", "Medium"]
    assert prefs.data[presets.PREF_KEY][1]["name"] == "Fast"
    presets.save_presets(prefs, [])
    assert presets.load_presets(prefs) == []  # an empty saved list wins over the defaults
    presets.restore_defaults(prefs)
    assert [r["name"] for r in presets.load_presets(prefs)] == ["Garbage", "Light", "Medium", "Heavy"]
    garbage = [{"name": "", "repo": "x/y"}, "nope", {"name": "a", "repo": ""}, {"name": "ok", "repo": "a/b", "n_ctx": 5}]
    assert presets.save_presets(prefs, garbage) == [
        {"name": "ok", "repo": "a/b", "file": "", "quant": "", "note": "", "n_ctx": 0}]


def test_legacy_ollama_presets_migrate_once_keeping_user_names():
    legacy = [
        {"name": "Trash", "model": "qwen3.5:0.8b", "note": "Qwen3.5 0.8B · fastest, rough translations"},
        {"name": "Mine", "model": "qwen3.5:4b", "note": "my own note"},
        {"name": "Gemma", "model": "hf.co/unsloth/gemma-4-E2B-it-GGUF:Q5_K_M", "note": ""},
        {"name": "Llama", "model": "llama3.2:3b", "note": ""},  # an Ollama library tag: no GGUF equivalent
    ]
    prefs = FakePrefs({presets.LEGACY_PREF_KEY: legacy})
    rows = presets.load_presets(prefs)
    assert [r["name"] for r in rows] == ["Trash", "Mine", "Gemma"]
    assert rows[0]["file"] == "Qwen3.5-0.8B-UD-Q4_K_XL.gguf" and rows[0]["note"].startswith("Qwen3.5 0.8B · fastest")
    assert rows[0]["note"].endswith("~1 GB RAM")  # the old default note became the new one
    assert rows[1]["repo"] == "unsloth/Qwen3.5-4B-GGUF" and rows[1]["note"] == "my own note"
    assert rows[1]["sha256"] == presets.DEFAULT_PRESETS[2]["sha256"]
    assert rows[2] == {"name": "Gemma", "repo": "unsloth/gemma-4-E2B-it-GGUF", "file": "", "quant": "Q5_K_M",
                       "note": "", "n_ctx": 0}
    assert prefs.data[presets.LEGACY_PREF_KEY] is None and prefs.data[presets.PREF_KEY] == rows
    # once migrated, the saved list wins
    prefs.data[presets.PREF_KEY] = []
    assert presets.load_presets(prefs) == []


def test_validation_messages_and_hf_forms():
    rows = presets.load_presets(FakePrefs())
    ok = lambda row, index=None: presets.validate_preset(row, rows, index)  # noqa: E731
    assert ok({"name": "", "repo": "a/b"})[1] == "Give the preset a name"
    assert ok({"name": "x" * 61, "repo": "a/b"})[1] == "Keep the name under 60 characters"
    assert ok({"name": "light", "repo": "a/b"})[1] == "A preset is already called light"
    assert ok({"name": "Light", "repo": "a/b"}, 1)[1] is None  # its own name may stay
    assert "not a Hugging Face repo id" in ok({"name": "n", "repo": "not a repo"})[1]
    assert ok({"name": "n", "repo": "a/b", "file": "x.bin"})[1] == "Only .gguf files can run on this device"
    assert ok({"name": "n", "repo": "a/b", "n_ctx": "abc"})[1].startswith("Context must be a whole number")
    assert ok({"name": "n", "repo": "a/b", "n_ctx": "100"})[1].startswith("Context must be between")
    clean, error = ok({"name": "Gemma", "repo": "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/blob/main/g-Q4_K_M.gguf",
                       "n_ctx": "4096", "note": " n "})
    assert error is None and clean == {"name": "Gemma", "repo": "unsloth/gemma-4-E2B-it-GGUF", "file": "g-Q4_K_M.gguf",
                                       "quant": "", "note": "n", "n_ctx": 4096}
    clean, _ = ok({"name": "Q", "repo": "hf.co/unsloth/Qwen3.5-2B-GGUF:Q8_0"})
    assert clean["repo"] == "unsloth/Qwen3.5-2B-GGUF" and clean["quant"] == "Q8_0" and clean["file"] == ""
    # renaming keeps the verified size / sha256; a new file drops them
    clean, _ = ok({"name": "Renamed", "repo": rows[0]["repo"], "file": rows[0]["file"]}, 0)
    assert clean["sha256"] == rows[0]["sha256"] and clean["size"] == rows[0]["size"]
    clean, _ = ok({"name": "Renamed", "repo": rows[0]["repo"], "file": "Qwen3.5-0.8B-Q8_0.gguf"}, 0)
    assert "sha256" not in clean and "size" not in clean
