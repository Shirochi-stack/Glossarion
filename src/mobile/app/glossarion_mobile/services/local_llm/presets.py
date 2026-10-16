"""On-device model presets: editable data in Prefs (``PREF_KEY``), never code.

A row is ``{"name", "repo", "file", "quant", "note", "n_ctx"}`` plus, when known, the file's ``size``
and ``sha256`` (Hugging Face's LFS oid; the download re-reads both from Hugging Face). ``file`` may be
empty (the auto-pick of ``hf.pick_file`` applies), ``quant`` may name one (``Q5_K_M``), ``n_ctx`` 0
means the engine default. ``DEFAULT_PRESETS`` are only the first-run data; a saved list (even an empty
one) wins.

The Model manager's earlier Ollama presets (``LEGACY_PREF_KEY``, Ollama tags) are migrated
once: the four Qwen3.5 tags become the matching GGUF rows under the user's own names and notes, an
``hf.co/<user>/<repo>[:quant]`` entry becomes that repo; Ollama library tags with no GGUF equivalent
are dropped.
"""

from __future__ import annotations

import copy
import logging
import re
from typing import Any, Optional, Sequence

from glossarion_mobile.services.local_llm import hf

log = logging.getLogger("glossarion.local_llm")

__all__ = [
    "DEFAULT_PRESETS", "LEGACY_PREF_KEY", "MAX_PRESETS", "PREF_KEY", "load_presets", "migrate_legacy",
    "restore_defaults", "save_presets", "validate_preset",
]

PREF_KEY = "local_model_presets"
LEGACY_PREF_KEY = "ollama_presets"
MAX_PRESETS = 50
MAX_CTX = 262144

#: First-run presets (verified on Hugging Face 2026-10-16: size and sha256 = the LFS oid). Apache-2.0,
#: architecture ``qwen35``, 262,144 trained context.
DEFAULT_PRESETS: tuple = (
    {"name": "Garbage", "repo": "unsloth/Qwen3.5-0.8B-GGUF", "file": "Qwen3.5-0.8B-Q4_K_M.gguf", "quant": "",
     "note": "Qwen3.5 0.8B · fastest, rough translations · ~1 GB RAM", "n_ctx": 0, "size": 532517120,
     "sha256": "bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517"},
    {"name": "Light", "repo": "unsloth/Qwen3.5-2B-GGUF", "file": "Qwen3.5-2B-Q4_K_M.gguf", "quant": "",
     "note": "Qwen3.5 2B · quick drafts on most phones · ~2 GB RAM", "n_ctx": 0, "size": 1280835840,
     "sha256": "aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223"},
    {"name": "Medium", "repo": "unsloth/Qwen3.5-4B-GGUF", "file": "Qwen3.5-4B-Q4_K_M.gguf", "quant": "",
     "note": "Qwen3.5 4B · better quality · needs ~4 GB free RAM", "n_ctx": 0, "size": 2740937888,
     "sha256": "00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4"},
    {"name": "Heavy", "repo": "unsloth/Qwen3.5-9B-GGUF", "file": "Qwen3.5-9B-Q4_K_M.gguf", "quant": "",
     "note": "Qwen3.5 9B · best of the four · needs ~8 GB free RAM", "n_ctx": 0, "size": 5680522464,
     "sha256": "03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8"},
)
#: The old Ollama presets' tags and the default row each one became.
LEGACY_TAGS = {"qwen3.5:0.8b": 0, "qwen3.5:2b": 1, "qwen3.5:4b": 2, "qwen3.5:9b": 3}
_LEGACY_NOTES = {
    "Qwen3.5 0.8B · fastest, rough translations", "Qwen3.5 2B · quick drafts on most phones",
    "Qwen3.5 4B · better quality, needs about 4 GB free RAM", "Qwen3.5 9B · best of the four, needs about 8 GB free RAM",
}
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def _int(value: Any, low: int, high: int) -> int:
    try:
        number = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return number if low <= number <= high else 0


def _clean(row: Any) -> Optional[dict]:
    if not isinstance(row, dict):
        return None
    name = str(row.get("name") or "").strip()
    repo = str(row.get("repo") or "").strip()
    if not name or not repo:
        return None
    out = {"name": name[:60], "repo": repo, "file": str(row.get("file") or "").strip(),
           "quant": str(row.get("quant") or "").strip(), "note": str(row.get("note") or "").strip()[:200],
           "n_ctx": _int(row.get("n_ctx"), 512, MAX_CTX)}
    size = _int(row.get("size"), 1, 1 << 42)
    sha = str(row.get("sha256") or "").strip().lower()
    if size:
        out["size"] = size
    if _SHA_RE.match(sha):
        out["sha256"] = sha
    return out


def _defaults() -> list:
    return [copy.deepcopy(p) for p in DEFAULT_PRESETS]


def migrate_legacy(rows: Any) -> list:
    """The old Ollama preset rows as GGUF preset rows (user names and own notes kept)."""
    out: list = []
    for row in rows if isinstance(rows, list) else ():
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        model = str(row.get("model") or "").strip()
        note = str(row.get("note") or "").strip()
        if not name or not model:
            continue
        index = LEGACY_TAGS.get(model.lower())
        if index is not None:
            new = copy.deepcopy(DEFAULT_PRESETS[index])
            new["name"] = name
            if note and note not in _LEGACY_NOTES:
                new["note"] = note
            out.append(new)
            continue
        if model.lower().startswith(("hf.co/", "huggingface.co/")):
            try:
                ref = hf.parse_ref(model)
            except hf.HfError:
                continue
            out.append({"name": name, "repo": ref.repo, "file": ref.file or "", "quant": ref.quant or "",
                        "note": note, "n_ctx": 0})
            continue
        log.info("on-device presets: dropped the Ollama-only preset %s (%s)", name, model)
    return [row for row in (_clean(r) for r in out) if row is not None][:MAX_PRESETS]


def load_presets(prefs: Any) -> list:
    """The saved presets; on first use the migrated Ollama presets, else the defaults."""
    saved = None
    legacy = None
    try:
        if prefs is not None:
            saved = prefs.get(PREF_KEY, None)
            legacy = prefs.get(LEGACY_PREF_KEY, None)
    except Exception:
        saved = legacy = None
    if not isinstance(saved, list):
        if isinstance(legacy, list):
            rows = migrate_legacy(legacy)
            try:
                prefs.set(PREF_KEY, copy.deepcopy(rows))
                prefs.set(LEGACY_PREF_KEY, None)  # Ollama tags cannot run here any more
            except Exception:
                log.debug("saving the migrated presets failed", exc_info=True)
            return rows
        return _defaults()
    return [row for row in (_clean(r) for r in saved) if row is not None]


def save_presets(prefs: Any, presets: Sequence[dict]) -> list:
    rows = [row for row in (_clean(r) for r in presets) if row is not None][:MAX_PRESETS]
    if prefs is not None:
        prefs.set(PREF_KEY, copy.deepcopy(rows))
    return rows


def restore_defaults(prefs: Any) -> list:
    return save_presets(prefs, _defaults())


def validate_preset(row: dict, presets: Sequence[dict], index: Optional[int] = None) -> tuple:
    """(cleaned row, None) or (None, message). ``index``: the row being edited (its own name may stay;
    its known size / sha256 stay while repo and file are unchanged).

    ``repo`` accepts every ``hf.parse_ref`` form; a link or path that names a file fills ``file``."""
    name = str(row.get("name") or "").strip()
    if not name:
        return None, "Give the preset a name"
    if len(name) > 60:
        return None, "Keep the name under 60 characters"
    for i, other in enumerate(presets):
        if i != index and str(other.get("name") or "").strip().casefold() == name.casefold():
            return None, f"A preset is already called {name}"
    try:
        ref = hf.parse_ref(row.get("repo"))
        file = str(row.get("file") or "").strip() or (ref.file or "")
        if file:
            file = hf._check_file(file)
    except hf.HfError as exc:
        return None, str(exc)
    n_ctx_text = str(row.get("n_ctx") or "").strip()
    n_ctx = 0
    if n_ctx_text and n_ctx_text != "0":
        try:
            n_ctx = int(n_ctx_text)
        except ValueError:
            return None, "Context must be a whole number of tokens (or empty for the default)"
        if not 512 <= n_ctx <= MAX_CTX:
            return None, f"Context must be between 512 and {MAX_CTX:,} tokens"
    clean = {"name": name, "repo": ref.repo, "file": file, "quant": "" if file else (ref.quant or ""),
             "note": str(row.get("note") or "").strip()[:200], "n_ctx": n_ctx}
    if index is not None and 0 <= index < len(presets):
        old = presets[index]
        if old.get("repo") == clean["repo"] and old.get("file") == clean["file"] and clean["file"]:
            for key in ("size", "sha256"):
                if old.get(key):
                    clean[key] = old[key]
    return clean, None
