"""On-device model presets (``services/local_llm/presets.py``): editable Prefs data, the verified defaults,
the one-time migration of the old Ollama presets, validation.

Run from src/mobile:
    .venv/Scripts/python.exe -m pytest -p no:cacheprovider -W ignore tests_host/test_local_llm_presets.py
"""

from __future__ import annotations

import local_llm_fakes as fakes  # noqa: F401  (sys.path)
from local_llm_fakes import FakePrefs

from glossarion_mobile.services.local_llm import presets


def test_defaults_are_three_verified_qwen35_ggufs_and_sigma_picker():
    """Owner 2026-11-25: uncensored (Heretic) Qwen3.5 i1-Q4_K_M; the projector from ``mmproj_repo`` (verified on
    Hugging Face that day: size and sha256 = the LFS oid)."""
    rows = presets.load_presets(FakePrefs())
    assert [(r["name"], r["repo"], r["file"], r["size"], r["sha256"]) for r in rows[:3]] == [
        ("Garbage", "mradermacher/Qwen3.5-0.8B-heretic-v3-i1-GGUF", "Qwen3.5-0.8B-heretic-v3.i1-Q4_K_M.gguf", 529298208,
         "3c1a21ee949303cbb65e8a8380ffa17161901680080e12e08c397528e110c0d1"),
        ("Light", "mradermacher/Qwen3.5-2B-heretic-i1-GGUF", "Qwen3.5-2B-heretic.i1-Q4_K_M.gguf", 1270809248,
         "66e55c4ccdb612daed6ff4149b46ab1fde736ad56f6de6903de6709dbe760c90"),
        ("Medium", "mradermacher/Qwen3.5-4B-heretic-i1-GGUF", "Qwen3.5-4B-heretic.i1-Q4_K_M.gguf", 2707515008,
         "a50ab7af6dabdb6b59b342c4aa89aaad4dc1b4d82bfc67286bb276a5c84f1018"),
    ]
    assert [(r["mmproj_repo"], r["mmproj"], r["mmproj_size"], r["mmproj_sha256"]) for r in rows[:3]] == [
        ("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf", 204987232,
         "56e4c6cfe73b0c82e3e82bc518d7591997e61d81f723fc41a586f4fa69ea2453"),
        ("unsloth/Qwen3.5-2B-GGUF", "mmproj-F16.gguf", 668227264,
         "7035e9cb8d7c6a9681d07eef9a364783e86ea4cd73faab2eabb4f43a101830c7"),
        ("mradermacher/Qwen3.5-4B-heretic-GGUF", "Qwen3.5-4B-heretic.mmproj-f16.gguf", 672423552,
         "e638dc8de3b75309a190092ba006307759343b62ae0d21ed8359df76b9b76c3b"),
    ]
    assert all("uncensored" in r["note"] for r in rows[:3])
    assert rows[3] == presets.SIGMA_PRESET
    assert all(r["n_ctx"] == 0 and r["note"] for r in rows)
    rows[0]["name"] = "changed"  # a copy, never the module data
    assert presets.DEFAULT_PRESETS[0]["name"] == "Garbage"


def _old_stock_row(name: str) -> dict:
    """A row the way earlier versions saved the stock default (Unsloth's censored Qwen3.5 UD-Q4_K_XL)."""
    _name, repo, file = next(row for row in presets.OLD_STOCK_PRESETS if row[0] == name)
    return {"name": name, "repo": repo, "file": file, "quant": "", "note": "Qwen3.5 · old note", "n_ctx": 0,
            "size": 1, "sha256": "0" * 64, "mmproj": "mmproj-F16.gguf", "mmproj_size": 2,
            "mmproj_sha256": "1" * 64}


def test_saved_stock_rows_become_the_uncensored_defaults_once():
    mine = {"name": "Gemma 4 E2B · Q4_0", "repo": "unsloth/gemma-4-E2B-it-GGUF", "file": "gemma-4-E2B-it-Q4_0.gguf",
            "quant": "", "note": "Custom · Gemma 4 E2B · Q4_0", "n_ctx": 0, "mmproj": "mmproj-F16.gguf"}
    renamed = {**_old_stock_row("Light"), "name": "My Light"}  # the user's own row: stays
    custom = {"name": "Qwen 3.5 0.8B · Q4_0", "repo": "unsloth/Qwen3.5-0.8B-GGUF", "file": "Qwen3.5-0.8B-Q4_0.gguf",
              "quant": "", "note": "Custom · Qwen 3.5 0.8B · Q4_0", "n_ctx": 0}
    old_medium = {**_old_stock_row("Medium"), "n_ctx": 8192, "backend": "cpu"}
    saved = [_old_stock_row("Garbage"), renamed, old_medium, dict(presets.SIGMA_PRESET), mine, custom]
    prefs = FakePrefs({presets.PREF_KEY: saved})
    rows = presets.load_presets(prefs)
    assert [r["name"] for r in rows] == ["Garbage", "My Light", "Medium", "Sigma", mine["name"], custom["name"]]
    assert rows[0] == presets.DEFAULT_PRESETS[0]
    assert rows[2] == {**presets.DEFAULT_PRESETS[2], "n_ctx": 8192, "backend": "cpu"}  # its own settings stay
    assert rows[1] == presets._clean(renamed) and rows[4] == presets._clean(mine) and rows[5] == presets._clean(custom)
    assert rows[3] == presets.SIGMA_PRESET
    assert prefs.data[presets.PREF_KEY] == rows  # saved once
    again = FakePrefs({presets.PREF_KEY: [dict(r) for r in rows]})
    assert presets.load_presets(again) == rows and again.data[presets.PREF_KEY] == rows


def test_a_stock_row_whose_replacement_is_already_listed_goes():
    prefs = FakePrefs({presets.PREF_KEY: [dict(presets.DEFAULT_PRESETS[0]), _old_stock_row("Garbage")]})
    assert [r["repo"] for r in presets.load_presets(prefs)] == [presets.DEFAULT_PRESETS[0]["repo"]]


def test_mmproj_repo_is_kept_cleaned_and_falls_back_to_the_custom_table():
    clean = presets._clean({"name": "n", "repo": "a/b", "mmproj_repo": " c/d ", "mmproj": "mmproj-F16.gguf"})
    assert clean["mmproj_repo"] == "c/d" and presets.projector_repo(clean) == "c/d"
    assert "mmproj_repo" not in presets._clean({"name": "n", "repo": "a/b", "mmproj_repo": "A/B"})  # its own repo
    assert "mmproj_repo" not in presets._clean({"name": "n", "repo": "a/b", "mmproj_repo": "not a repo"})
    assert presets.projector_repo({"name": "n", "repo": "a/b"}) == "a/b"  # a row without it: its own repository
    # a row (or a downloaded model) of a Custom size without the field: the size's projector repository
    assert presets.projector_repo({"repo": "mradermacher/qwen3.5-4b-heretic-i1-gguf"}) == \
        "mradermacher/Qwen3.5-4B-heretic-GGUF"
    assert presets.projector_repo(None, "mradermacher/Qwen3.5-2B-heretic-i1-GGUF") == "unsloth/Qwen3.5-2B-GGUF"
    # editing a row keeps where its projector is from while the repository stays
    rows = presets.load_presets(FakePrefs())
    clean, error = presets.validate_preset({"name": "Renamed", "repo": rows[0]["repo"], "file": rows[0]["file"]},
                                           rows, 0)
    assert error is None and clean["mmproj_repo"] == rows[0]["mmproj_repo"] and clean["mmproj"] == rows[0]["mmproj"]
    clean, _ = presets.validate_preset({"name": "Other", "repo": "x/y"}, rows, 0)
    assert "mmproj_repo" not in clean and "mmproj" not in clean


def test_lfm2_has_three_uncensored_sizes():
    lfm2 = presets.custom_family("lfm2")
    assert [(s["label"], s["name"]) for s in lfm2["sizes"]] == [
        ("350M", "LFM2.5 350M"), ("700M", "LFM2 700M"), ("1.2B", "LFM2.5 1.2B")]
    assert all("heretic" in s["repo"].lower() for s in lfm2["sizes"])


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
    assert [r["name"] for r in presets.load_presets(prefs)] == ["Garbage", "Light", "Medium", "Sigma"]
    garbage = [{"name": "", "repo": "x/y"}, "nope", {"name": "a", "repo": ""}, {"name": "ok", "repo": "a/b", "n_ctx": 5}]
    assert presets.save_presets(prefs, garbage) == [
        {"name": "ok", "repo": "a/b", "file": "", "quant": "", "note": "", "n_ctx": 0}]


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
