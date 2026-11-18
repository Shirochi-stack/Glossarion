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

NPU (Snapdragon Hexagon): a row may name its ``backend`` ("npu", "gpu" or "cpu"; none = "auto": the NPU when
the phone's NPU check passed, "Use NPU" is on and the quant is one the backend accepts, else the GPU / CPU rule)
and its ``npu_layers`` (none = all). ``NPU_PRESETS`` (the Q4_0 files of ``GPU_PRESETS``, backend "npu") and
``NPU_FAMILY`` (every Custom size, quants marked by ``npu_quant_kind``) are fixed rows like ``CUSTOM_FAMILIES``,
shown only where the NPU check passed and saved only when tapped.
"""

from __future__ import annotations

import copy
import logging
import re
from typing import Any, Optional, Sequence

from glossarion_mobile.services.local_llm import hf

log = logging.getLogger("glossarion.local_llm")

__all__ = [
    "BACKENDS", "CUSTOM_FAMILIES", "DEFAULT_PRESETS", "GPU_PRESETS", "MAX_PRESETS", "NO_QUANT", "NPU_FAMILY",
    "NPU_PRESETS", "PREF_KEY", "add_row", "backend_of", "custom_family", "custom_row", "custom_size_for",
    "download_size", "file_quant", "is_gpu_quant", "is_no_quant", "is_old_gpu_row", "load_presets", "npu_quant_kind",
    "quant_label", "remove_row", "restore_defaults", "retire_old_gpu_rows", "save_presets", "validate_preset",
]

PREF_KEY = "local_model_presets"
MAX_PRESETS = 50
MAX_CTX = 262144
DEFAULT_CONTEXT = 24576
SIGMA_REPO = "sigma6196/e4b-ko-en-novel-translator-sft3"
SIGMA_Q8_FILE = "gguf/sft3-Q8_0.gguf"
SIGMA_PRESET = {"name": "Sigma", "repo": SIGMA_REPO, "file": "", "quant": "",
                "note": "Korean → English novels · choose a quant · Q8_0 recommended", "n_ctx": 0}


def is_sigma(repo: Any) -> bool:
    return str(repo or "").casefold() == SIGMA_REPO.casefold()


def uses_normal_settings(entry: dict) -> bool:
    """Sigma Q8_0 uses normal defaults; its on-device controls remain editable."""
    return is_sigma(entry.get("repo")) and entry.get("file") == SIGMA_Q8_FILE

#: First-run presets (verified on Hugging Face 2026-10-16: size and sha256 = the LFS oid; the vision
#: projectors 2026-10-17). Garbage, Light and Medium use Unsloth's UD-Q4_K_XL (2026-10-18: lower KL
#: divergence than Q4_K_M for 2-6 % more bytes). Sigma replaces Heavy and selects from its live HF files.
#: The Qwen rows are Apache-2.0, architecture ``qwen35``, 262,144 trained context. Qwen3.5 is
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
    SIGMA_PRESET,
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
#: login) and has a Q4_0 file. llama-cpp-python 0.3.37 vendors llama.cpp a956a852, whose llama-arch.cpp knows
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
    # mobile hotfix 1 #4: a ``group`` heads its sizes in the menu, so Gemma 4 reads as its own generation
    {"id": "gemma", "name": "Gemma", "sizes": (
        {"label": "270M", "name": "Gemma 3 270M", "repo": "unsloth/gemma-3-270m-it-GGUF", "group": "Gemma 3"},
        {"label": "1B", "name": "Gemma 3 1B", "repo": "unsloth/gemma-3-1b-it-GGUF", "group": "Gemma 3"},
        {"label": "4B", "name": "Gemma 3 4B", "repo": "unsloth/gemma-3-4b-it-GGUF", "group": "Gemma 3"},
        {"label": "E2B", "name": "Gemma 4 E2B", "repo": "unsloth/gemma-4-E2B-it-GGUF", "group": "Gemma 4"},
        {"label": "E4B", "name": "Gemma 4 E4B", "repo": "unsloth/gemma-4-E4B-it-GGUF", "group": "Gemma 4"},
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
#: a row's ``backend``: "auto" (also an empty or missing value) or one device
BACKENDS = ("auto", "npu", "gpu", "cpu")
BACKEND_LABELS = {"auto": "Auto", "npu": "NPU", "gpu": "GPU", "cpu": "CPU"}
#: one NPU session maps about 3.5 GB (llama.cpp docs/backend/snapdragon at the pin); bigger files are slower
NPU_SESSION_BYTES = 3_500_000_000
NPU_OVER_SESSION = "over the NPU's 3.5 GB: slower"


def _npu_row(gpu_row: dict, name: str, note: str) -> dict:
    """An NPU preset from a ``GPU_PRESETS`` row: the same verified Q4_0 file, size, sha256 and projector."""
    row = copy.deepcopy(gpu_row)
    row.update(name=name, note=note, backend="npu")
    return row


#: The built-in NPU rows (owner 2026-11-18): the Q4_0 files of the GPU table (Q4_0 is the Hexagon backend's best
#: weight type at the pin), backend "npu". Shown beside the saved list where the NPU check passed; a tap saves the
#: row (``OnDeviceFlow.get_npu_preset``), which is then edited like any other. The 9B file is over one NPU session's
#: 3.5 GB: it is only in the Custom · NPU list.
NPU_PRESETS: tuple = (
    _npu_row(GPU_PRESETS[0], "Garbage · NPU", "Qwen3.5 0.8B · Q4_0 on the NPU · ~1 GB RAM"),
    _npu_row(GPU_PRESETS[1], "Light · NPU", "Qwen3.5 2B · Q4_0 on the NPU · ~2 GB RAM"),
    _npu_row(GPU_PRESETS[2], "Medium · NPU", "Qwen3.5 4B · Q4_0 on the NPU · needs ~4 GB free RAM"),
)


def _npu_family() -> dict:
    sizes = []
    for family in CUSTOM_FAMILIES:
        for size in family["sizes"]:
            entry = dict(size)
            entry.setdefault("group", family["name"])
            if size["repo"].casefold() == "unsloth/qwen3.5-9b-gguf":
                entry["npu_note"] = NPU_OVER_SESSION
            sizes.append(entry)
    return {"id": "npu", "name": "NPU", "backend": "npu", "sizes": tuple(sizes)}


#: "Custom · NPU": every Custom size in one menu (grouped by family), its quants marked for the NPU
NPU_FAMILY: dict = _npu_family()
#: weight types MUL_MAT accepts on the Hexagon backend at the pin (ggml-hexagon htp-kparams.cpp); IQ4_XS and the
#: IQ3 types are not among them
_NPU_TYPES = {"Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "IQ4_NL", "IQ2_XXS", "Q1_0", "Q2_0", "F16", "BF16", "F32",
              "MXFP4", "NVFP4"}
_K_QUANT = re.compile(r"^Q[2-6]_K(?:_[A-Z]+)?$")


def npu_quant_kind(file: Any) -> str:
    """How a GGUF suits the NPU: "best" (Q4_0), "ok" (another weight type the backend accepts: Q4_1, Q5_x, Q8_0,
    the K-quants Q2_K-Q6_K with their size suffixes, IQ4_NL, IQ2_XXS, MXFP4, NVFP4, F16, BF16, F32) or "cpu" (not
    accepted or not recognisable: IQ4_XS, IQ3_*, ...; those layers would run on the CPU, so Auto skips the NPU)."""
    quant = file_quant(file)
    if quant == "Q4_0":
        return "best"
    if quant in _NPU_TYPES or _K_QUANT.match(quant):
        return "ok"
    name = str(file or "").upper()
    if not quant and ("MXFP4" in name or "NVFP4" in name):
        return "ok"
    return "cpu"


def backend_of(row: Optional[dict]) -> str:
    """The row's backend choice: "npu" / "gpu" / "cpu", else "auto"."""
    value = str((row or {}).get("backend") or "").strip().lower()
    return value if value in BACKENDS else "auto"


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
    backend = backend_of(row)
    if backend != "auto":
        out["backend"] = backend
    npu_layers = _int(row.get("npu_layers"), 1, 10_000)
    if npu_layers:
        out["npu_layers"] = npu_layers
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
    rows = [row for row in (_clean(r) for r in saved) if row is not None]
    repaired = _repair_projector_rows(rows)
    migrated = False
    renamed = repaired
    for row in rows:
        if is_sigma(row["repo"]) and row["name"] == "Sigma (Best)":
            row["name"] = "Sigma"
            renamed = True
        if row["repo"].casefold() == "unsloth/qwen3.5-9b-gguf" and (
                row["name"] == "Heavy" or row["name"].startswith("Heavy · GPU (Q4_0)")):
            row["name"] = f"Qwen 3.5 9B · {quant_label(row['file'])}"
            row["note"] = f"Custom · {row['name']}"
            migrated = True
    if migrated:
        if len(rows) < MAX_PRESETS and not any(is_sigma(row["repo"]) for row in rows):
            rows.insert(3 if len(rows) >= 3 else len(rows), copy.deepcopy(SIGMA_PRESET))
        return save_presets(prefs, rows)
    if renamed:
        return save_presets(prefs, rows)
    return rows


def _repair_projector_rows(rows: list) -> bool:
    """Rows whose ``file`` is a vision projector (``hf.is_projector_name``; "Save as a preset" of Add from Hugging
    Face saved ``…-mmproj-Q8_0.gguf`` as the model when the projector's name did not start with "mmproj") can never
    load. Each becomes the
    repository's file-less row again (its next Download lists the model files, or picks the usual quant), keeps
    that projector as its image support (``mmproj``) and loses a name that was the projector's file name; a row
    another one of the same repository already stands for is dropped. True when anything changed (in place)."""
    changed = False
    for row in list(rows):
        if not hf.is_projector_name(row.get("file")):
            continue
        changed = True
        projector = row["file"]
        if any(other is not row and not other.get("file") and other["repo"].casefold() == row["repo"].casefold()
               for other in rows):
            rows.remove(row)
            continue
        for key in ("size", "sha256"):
            row.pop(key, None)
        row["file"] = ""
        if not row.get("mmproj"):
            row["mmproj"] = projector
            row.pop("mmproj_size", None)
            row.pop("mmproj_sha256", None)
        if "mmproj" in row["name"].casefold():
            base = row["repo"].split("/", 1)[-1][:56]
            taken = {str(other.get("name") or "").casefold() for other in rows if other is not row}
            name, n = base, 2
            while name.casefold() in taken:
                name, n = f"{base} {n}", n + 1
            row["name"] = name
    if changed:
        log.info("on-device presets: rows that named a vision projector as their model were repaired")
    return changed


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
    """The ``CUSTOM_FAMILIES`` entry called ``family_id`` (``NPU_FAMILY`` for "npu"), else None."""
    if family_id == NPU_FAMILY["id"]:
        return NPU_FAMILY
    return next((family for family in CUSTOM_FAMILIES if family["id"] == family_id), None)


def custom_size_for(repo: Any) -> Optional[tuple]:
    """``(family, size)`` of the Custom table entry for a repository, else None."""
    repo_cf = str(repo or "").casefold()
    for family in CUSTOM_FAMILIES:
        for size in family["sizes"]:
            if size["repo"].casefold() == repo_cf:
                return family, size
    return None


def custom_row(size: dict, file: Any, projector: Any = None, backend: str = "") -> dict:
    """The preset row of a quant chosen in the Custom menu: ``file`` and ``projector`` are ``hf.HfFile``;
    ``backend`` "npu" (Custom · NPU) names the row's device."""
    label = quant_label(file.path)
    backend = backend_of({"backend": backend})
    if backend == "npu":
        row = {"name": f"{size['name']} · {label} · NPU", "repo": size["repo"], "file": file.path, "quant": "",
               "note": f"Custom · NPU · {size['name']} · {label}", "n_ctx": 0, "size": int(file.size or 0),
               "backend": "npu"}
    else:
        row = {"name": f"{size['name']} · {label}", "repo": size["repo"], "file": file.path, "quant": "",
               "note": f"Custom · {size['name']} · {label}", "n_ctx": 0, "size": int(file.size or 0)}
        if backend != "auto":
            row["backend"] = backend
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
            if hf.is_projector_name(file):
                raise hf.HfError(f"{file.rsplit('/', 1)[-1]} is a vision projector (image support), not a model: "
                                 "name a model file, or leave File empty")
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
    old = presets[index] if index is not None and 0 <= index < len(presets) else {}
    # Backend and NPU layers: the editor shows them only where the NPU check passed; elsewhere the row keeps its own
    if "backend" in row:
        backend = str(row.get("backend") or "").strip().lower() or "auto"
        if backend not in BACKENDS:
            return None, "Backend must be Auto, NPU, GPU or CPU"
    else:
        backend = backend_of(old)
    if backend != "auto":
        clean["backend"] = backend
    if "npu_layers" in row:
        layers_text = str(row.get("npu_layers") or "").strip().lower()
        if layers_text and layers_text not in ("0", "all", "-1"):
            try:
                layers = int(layers_text)
            except ValueError:
                return None, "NPU layers must be a whole number (or empty for all)"
            if not 1 <= layers <= 10_000:
                return None, "NPU layers must be between 1 and 10,000 (or empty for all)"
            clean["npu_layers"] = layers
    elif old.get("npu_layers"):
        clean["npu_layers"] = old["npu_layers"]
    if old:
        if old.get("repo") == clean["repo"] and old.get("file") == clean["file"] and clean["file"]:
            for key in ("size", "sha256"):
                if old.get(key):
                    clean[key] = old[key]
        if old.get("repo") == clean["repo"]:  # the same repository keeps its vision projector
            for key in ("mmproj", "mmproj_size", "mmproj_sha256"):
                if old.get(key):
                    clean[key] = old[key]
    return clean, None
