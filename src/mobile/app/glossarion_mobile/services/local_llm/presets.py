"""On-device model presets: editable data in Prefs (``PREF_KEY``), never code.

A row is ``{"name", "repo", "file", "quant", "note", "n_ctx"}`` plus, when known, the file's ``size``
and ``sha256`` (Hugging Face's LFS oid; the download re-reads both from Hugging Face), and for a vision
model its projector: ``mmproj`` (the ``mmproj-*.gguf`` file of the same repository, downloaded when
an image first needs it), ``mmproj_size`` and ``mmproj_sha256``. Without ``mmproj`` the repository's own projector is
used when it has one (``hf.pick_projector``). ``file`` may be
empty (the auto-pick of ``hf.pick_file`` applies), ``quant`` may name one (``Q5_K_M``), ``n_ctx`` 0
means the engine default. ``DEFAULT_PRESETS`` are only the first-run data; a saved list (even an empty
one) wins. ``CUSTOM_FAMILIES`` are the fixed Custom rows beside the saved list: a family's parameter sizes,
whose quants come from the repository's own files; a quant the user chooses becomes an ordinary saved row.
"""

from __future__ import annotations

import copy
import logging
import re
from typing import Any, Optional, Sequence

from glossarion_mobile.services.local_llm import hf

log = logging.getLogger("glossarion.local_llm")

__all__ = [
    "CUSTOM_FAMILIES", "DEFAULT_PRESETS", "GPU_PRESETS", "MAX_PRESETS", "NO_QUANT", "PREF_KEY", "add_row", "custom_family",
    "custom_row", "custom_size_for", "download_size", "file_quant", "is_gpu_quant", "is_no_quant", "is_old_gpu_row",
    "load_presets", "quant_label", "remove_row", "restore_defaults", "retire_old_gpu_rows", "save_presets",
    "validate_preset",
]

PREF_KEY = "local_model_presets"
MAX_PRESETS = 50
MAX_CTX = 262144

#: First-run presets (verified on Hugging Face 2026-10-16: size and sha256 = the LFS oid; the vision
#: projectors 2026-10-17). Garbage, Light and Medium use Unsloth's UD-Q4_K_XL (2026-10-18: lower KL
#: divergence than Q4_K_M for 2-6 % more bytes); Heavy keeps Q4_K_M, whose RAM is already tight on 8 GB phones. Apache-2.0, architecture ``qwen35``, 262,144 trained context. Qwen3.5 is
#: natively multimodal: with its ``mmproj-F16.gguf`` (llama.cpp's mtmd, libmtmd in our own
#: llama-cpp-python build) the same model reads images, for the chat's Vision mode and image jobs.
DEFAULT_PRESETS: tuple = (
    {"name": "Garbage", "repo": "unsloth/Qwen3.5-0.8B-GGUF", "file": "Qwen3.5-0.8B-UD-Q4_K_XL.gguf", "quant": "",
     "note": "Qwen3.5 0.8B · fastest, rough translations · ~1 GB RAM", "n_ctx": 0, "size": 558772480,
     "sha256": "3177ebd67afe4438374da19e690bc1b98756f7e0fea9240e1be404336156a7b5",
     "mmproj": "mmproj-F16.gguf", "mmproj_size": 204987232,
     "mmproj_sha256": "56e4c6cfe73b0c82e3e82bc518d7591997e61d81f723fc41a586f4fa69ea2453"},
    {"name": "Light", "repo": "unsloth/Qwen3.5-2B-GGUF", "file": "Qwen3.5-2B-UD-Q4_K_XL.gguf", "quant": "",
     "note": "Qwen3.5 2B · quick drafts on most phones · ~2 GB RAM", "n_ctx": 0, "size": 1339752704,
     "sha256": "0af96165ea615bea39a04118d63f0b6d35908aea850ee4a51aa6151d851b8b35",
     "mmproj": "mmproj-F16.gguf", "mmproj_size": 668227264,
     "mmproj_sha256": "7035e9cb8d7c6a9681d07eef9a364783e86ea4cd73faab2eabb4f43a101830c7"},
    {"name": "Medium", "repo": "unsloth/Qwen3.5-4B-GGUF", "file": "Qwen3.5-4B-UD-Q4_K_XL.gguf", "quant": "",
     "note": "Qwen3.5 4B · better quality · needs ~4 GB free RAM", "n_ctx": 0, "size": 2912109728,
     "sha256": "b252c5610a42ca82d20fe2a12813e9d069eed89292907e26c783eeb0bc961bc7",
     "mmproj": "mmproj-F16.gguf", "mmproj_size": 672423616,
     "mmproj_sha256": "cd88edcf8d031894960bb0c9c5b9b7e1fea6ebee02b9f7ce925a00d12891f864"},
    {"name": "Heavy", "repo": "unsloth/Qwen3.5-9B-GGUF", "file": "Qwen3.5-9B-Q4_K_M.gguf", "quant": "",
     "note": "Qwen3.5 9B · best of the four · needs ~8 GB free RAM", "n_ctx": 0, "size": 5680522464,
     "sha256": "03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8",
     "mmproj": "mmproj-F16.gguf", "mmproj_size": 918166080,
     "mmproj_sha256": "f70dc3509053962b0d0d3ee8a7eacebf5d60aa560cad78254ae8698516ae029f"},
)
#: The GPU-friendly variants of the four defaults that earlier versions added to the saved list on Android:
#: kept only to recognise those rows (``is_old_gpu_row``). Nothing adds them any more (owner 2026-10-31: the
#: Custom rows replace them); one whose file is on the phone stays an ordinary row, the others are dropped.
#: Android's OpenCL backend (Qualcomm Adreno) has its fast kernels for Q4_0 (Q6_K runs unoptimized, the other
#: K-quants may be slow or fall back to the CPU); Unsloth's Q4_0 files keep a few tensors (e.g. the output)
#: in higher quants, which run unoptimized. iOS (Metal) and the CPU run every quant alike.
#: Verified on Hugging Face 2026-10-20 (size and sha256 = the LFS oid); same repos, so same projectors.
GPU_PRESETS: tuple = (
    {"name": "Garbage · GPU (Q4_0)", "repo": "unsloth/Qwen3.5-0.8B-GGUF", "file": "Qwen3.5-0.8B-Q4_0.gguf",
     "quant": "", "note": "Qwen3.5 0.8B · Q4_0 for the Adreno GPU (OpenCL) · ~1 GB RAM", "n_ctx": 0,
     "size": 507154688, "sha256": "444406ddd926550c724ec18d5120a9d40ded44908a063b0e66e9a7e5464c652c",
     "mmproj": "mmproj-F16.gguf", "mmproj_size": 204987232,
     "mmproj_sha256": "56e4c6cfe73b0c82e3e82bc518d7591997e61d81f723fc41a586f4fa69ea2453"},
    {"name": "Light · GPU (Q4_0)", "repo": "unsloth/Qwen3.5-2B-GGUF", "file": "Qwen3.5-2B-Q4_0.gguf", "quant": "",
     "note": "Qwen3.5 2B · Q4_0 for the Adreno GPU (OpenCL) · ~2 GB RAM", "n_ctx": 0, "size": 1214873856,
     "sha256": "cd70221bebaee0503e0f6717e174250cd7825aa88438b3aabec9ad55731d9bb1",
     "mmproj": "mmproj-F16.gguf", "mmproj_size": 668227264,
     "mmproj_sha256": "7035e9cb8d7c6a9681d07eef9a364783e86ea4cd73faab2eabb4f43a101830c7"},
    {"name": "Medium · GPU (Q4_0)", "repo": "unsloth/Qwen3.5-4B-GGUF", "file": "Qwen3.5-4B-Q4_0.gguf", "quant": "",
     "note": "Qwen3.5 4B · Q4_0 for the Adreno GPU (OpenCL) · needs ~4 GB free RAM", "n_ctx": 0,
     "size": 2583221408, "sha256": "298fcb5fe7a77ccc79745ae24751560c5ac56874caff4bb39b1f2055bd72b8bb",
     "mmproj": "mmproj-F16.gguf", "mmproj_size": 672423616,
     "mmproj_sha256": "cd88edcf8d031894960bb0c9c5b9b7e1fea6ebee02b9f7ce925a00d12891f864"},
    {"name": "Heavy · GPU (Q4_0)", "repo": "unsloth/Qwen3.5-9B-GGUF", "file": "Qwen3.5-9B-Q4_0.gguf", "quant": "",
     "note": "Qwen3.5 9B · Q4_0 for the Adreno GPU (OpenCL) · needs ~8 GB free RAM", "n_ctx": 0,
     "size": 5379417312, "sha256": "17670346b4260ddcb0173965145155885024f3c9a4a24389a3370751edbcde24",
     "mmproj": "mmproj-F16.gguf", "mmproj_size": 918166080,
     "mmproj_sha256": "f70dc3509053962b0d0d3ee8a7eacebf5d60aa560cad78254ae8698516ae029f"},
)
#: The Custom rows (owner 2026-10-31): family -> parameter sizes. Tapping one lists the real files of the
#: chosen size's repository (``hf.list_repo``), so nothing here names a quant. Checked 2026-10-31 on
#: https://huggingface.co/api/models/<repo>?blobs=true: every repo exists, is not gated (no Hugging Face
#: login) and has a Q4_0 file. llama-cpp-python 0.3.32 vendors llama.cpp b3fed31b, whose llama-arch.cpp knows
#: all four architectures (qwen35, llama, gemma3 and gemma4, lfm2) and whose libmtmd knows the Qwen3.5 and
#: Gemma projectors. Gemma 3 4B and Gemma 4 ship an mmproj ("Reads images"); so does every Qwen 3.5 size.
#: LFM2 is the lightweight family: 350M to 1.2B parameters, 219-696 MB at Q4_0, and it was trained for
#: English, Arabic, Chinese, French, German, Japanese, Korean and Spanish (SmolLM2 is mostly English).
CUSTOM_FAMILIES: tuple = (
    {"id": "qwen35", "name": "Qwen 3.5", "sizes": (
        {"label": "0.8B", "name": "Qwen 3.5 0.8B", "repo": "unsloth/Qwen3.5-0.8B-GGUF"},
        {"label": "2B", "name": "Qwen 3.5 2B", "repo": "unsloth/Qwen3.5-2B-GGUF"},
        {"label": "4B", "name": "Qwen 3.5 4B", "repo": "unsloth/Qwen3.5-4B-GGUF"},
        {"label": "9B", "name": "Qwen 3.5 9B", "repo": "unsloth/Qwen3.5-9B-GGUF"},
    )},
    {"id": "llama", "name": "Llama", "sizes": (
        {"label": "1B", "name": "Llama 3.2 1B", "repo": "unsloth/Llama-3.2-1B-Instruct-GGUF"},
        {"label": "3B", "name": "Llama 3.2 3B", "repo": "unsloth/Llama-3.2-3B-Instruct-GGUF"},
        {"label": "8B", "name": "Llama 3.1 8B", "repo": "unsloth/Llama-3.1-8B-Instruct-GGUF"},
    )},
    {"id": "gemma", "name": "Gemma", "sizes": (
        {"label": "270M", "name": "Gemma 3 270M", "repo": "unsloth/gemma-3-270m-it-GGUF"},
        {"label": "1B", "name": "Gemma 3 1B", "repo": "unsloth/gemma-3-1b-it-GGUF"},
        {"label": "E2B", "name": "Gemma 4 E2B", "repo": "unsloth/gemma-4-E2B-it-GGUF"},
        {"label": "4B", "name": "Gemma 3 4B", "repo": "unsloth/gemma-3-4b-it-GGUF"},
        {"label": "E4B", "name": "Gemma 4 E4B", "repo": "unsloth/gemma-4-E4B-it-GGUF"},
    )},
    {"id": "lfm2", "name": "LFM2", "sizes": (
        {"label": "350M", "name": "LFM2 350M", "repo": "unsloth/LFM2-350M-GGUF"},
        {"label": "700M", "name": "LFM2 700M", "repo": "unsloth/LFM2-700M-GGUF"},
        {"label": "1.2B", "name": "LFM2.5 1.2B", "repo": "unsloth/LFM2.5-1.2B-Instruct-GGUF"},
    )},
)
#: The unquantized GGUF kinds: what "No quant" in the Custom menu stands for
NO_QUANT = ("F16", "BF16", "F32")
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
    mmproj = str(row.get("mmproj") or "").strip()
    if mmproj:
        try:
            out["mmproj"] = hf._check_file(mmproj)
        except hf.HfError:
            mmproj = ""
    if mmproj:
        mmproj_size = _int(row.get("mmproj_size"), 1, 1 << 40)
        mmproj_sha = str(row.get("mmproj_sha256") or "").strip().lower()
        if mmproj_size:
            out["mmproj_size"] = mmproj_size
        if _SHA_RE.match(mmproj_sha):
            out["mmproj_sha256"] = mmproj_sha
    return out


def download_size(row: dict) -> int:
    """Bytes one tap downloads for ``row`` when known: the model (its vision projector, ``mmproj_size``,
    downloads when an image first needs it)."""
    return int(row.get("size") or 0)


def _defaults() -> list:
    return [copy.deepcopy(p) for p in DEFAULT_PRESETS]


def load_presets(prefs: Any) -> list:
    """The saved presets, else the defaults."""
    saved = None
    try:
        if prefs is not None:
            saved = prefs.get(PREF_KEY, None)
    except Exception:
        saved = None
    if not isinstance(saved, list):
        return _defaults()
    return [row for row in (_clean(r) for r in saved) if row is not None]


def save_presets(prefs: Any, presets: Sequence[dict]) -> list:
    rows = [row for row in (_clean(r) for r in presets) if row is not None][:MAX_PRESETS]
    if prefs is not None:
        prefs.set(PREF_KEY, copy.deepcopy(rows))
    return rows


def restore_defaults(prefs: Any) -> list:
    return save_presets(prefs, _defaults())


def is_old_gpu_row(row: dict) -> bool:
    """A row an earlier version added from ``GPU_PRESETS``: its repo and file, under its stock name (a taken
    name got " 2"). A renamed row is the user's own, and a Custom pick of the same file has another name."""
    repo, file = str(row.get("repo") or "").casefold(), row.get("file")
    name = str(row.get("name") or "").casefold()
    return any(p["repo"].casefold() == repo and p["file"] == file and name.startswith(p["name"].casefold())
               for p in GPU_PRESETS)


def retire_old_gpu_rows(prefs: Any, stays: Any) -> list:
    """Drop the saved ``is_old_gpu_row`` rows that ``stays(row)`` does not keep (the model is not on the phone);
    the rest of the list, and its order, stay. Returns the names dropped; nothing is saved without one."""
    rows = load_presets(prefs)
    dropped = [row for row in rows if is_old_gpu_row(row) and not stays(row)]
    if dropped:
        save_presets(prefs, [row for row in rows if not any(row is d for d in dropped)])
    return [row["name"] for row in dropped]


def is_gpu_quant(file: Any) -> bool:
    """A Q4_0 GGUF: the quant llama.cpp's Adreno (OpenCL) kernels are fast for."""
    return "q4_0" in str(file or "").lower()


def file_quant(file: Any) -> str:
    """The quant a GGUF file name ends with (``Qwen3.5-2B-UD-Q4_K_XL.gguf`` -> ``Q4_K_XL``), else ""."""
    match = re.search(r"(?:^|[-_.])((?:I)?Q\d+(?:_[A-Z0-9]+)*|BF16|F16|F32)\.gguf$", str(file or ""), re.IGNORECASE)
    return match.group(1).upper() if match else ""


def is_no_quant(file: Any) -> bool:
    """An unquantized GGUF (F16, BF16 or F32): the full-precision weights."""
    return file_quant(file) in NO_QUANT


def quant_label(file: Any) -> str:
    """What the Custom menu calls a file: its quant, "No quant (BF16)", else the file name."""
    quant = file_quant(file)
    if quant in NO_QUANT:
        return f"No quant ({quant})"
    return quant or str(file or "").rsplit("/", 1)[-1]


def custom_family(family_id: Any) -> Optional[dict]:
    """The ``CUSTOM_FAMILIES`` entry called ``family_id``, else None."""
    return next((family for family in CUSTOM_FAMILIES if family["id"] == family_id), None)


def custom_size_for(repo: Any) -> Optional[tuple]:
    """``(family, size)`` of the Custom table entry for a repository, else None."""
    repo_cf = str(repo or "").casefold()
    for family in CUSTOM_FAMILIES:
        for size in family["sizes"]:
            if size["repo"].casefold() == repo_cf:
                return family, size
    return None


def custom_row(size: dict, file: Any, projector: Any = None) -> dict:
    """The preset row of a quant chosen in the Custom menu: ``file`` and ``projector`` are ``hf.HfFile``."""
    label = quant_label(file.path)
    row = {"name": f"{size['name']} · {label}", "repo": size["repo"], "file": file.path, "quant": "",
           "note": f"Custom · {size['name']} · {label}", "n_ctx": 0, "size": int(file.size or 0)}
    if file.sha256:
        row["sha256"] = file.sha256
    if projector is not None:
        row["mmproj"], row["mmproj_size"] = projector.path, int(projector.size or 0)
        if projector.sha256:
            row["mmproj_sha256"] = projector.sha256
    return row


def _same_file(row: dict, other: dict) -> bool:
    return str(row.get("repo") or "").casefold() == str(other.get("repo") or "").casefold() \
        and row.get("file") == other.get("file")


def add_row(prefs: Any, row: dict) -> tuple:
    """Make sure the presets list has ``row`` (same repo and file): ``(presets, index of that row, added)``,
    appending it under a free name (a taken one gets " 2") when missing. A full list gets nothing and answers
    the index -1."""
    rows = load_presets(prefs)
    for index, other in enumerate(rows):
        if _same_file(other, row):
            return rows, index, False
    if len(rows) >= MAX_PRESETS:
        return rows, -1, False
    taken = {str(r.get("name") or "").casefold() for r in rows}
    name, n = row["name"], 2
    while name.casefold() in taken:
        name, n = f"{row['name'][:56]} {n}", n + 1
    rows = save_presets(prefs, [*rows, {**row, "name": name}])
    return rows, len(rows) - 1, True


def remove_row(prefs: Any, row: dict) -> bool:
    """Remove the saved row of the same repo and file; True when there was one."""
    rows = load_presets(prefs)
    kept = [other for other in rows if not _same_file(other, row)]
    if len(kept) == len(rows):
        return False
    save_presets(prefs, kept)
    return True


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
        if old.get("repo") == clean["repo"]:  # the same repository keeps its vision projector
            for key in ("mmproj", "mmproj_size", "mmproj_sha256"):
                if old.get(key):
                    clean[key] = old[key]
    return clean, None
