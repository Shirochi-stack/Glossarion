"""On-device model presets: editable data in Prefs (``PREF_KEY``), never code.

A row is ``{"name", "repo", "file", "quant", "note", "n_ctx"}`` plus, when known, the file's ``size``
and ``sha256`` (Hugging Face's LFS oid; the download re-reads both from Hugging Face), and for a vision
model its projector: ``mmproj`` (the ``mmproj-*.gguf`` file of the same repository, downloaded when
an image first needs it), ``mmproj_size`` and ``mmproj_sha256``. Without ``mmproj`` the repository's own projector is
used when it has one (``hf.pick_projector``). ``mmproj_repo`` (optional) names another repository the projector
comes from (listed, picked, downloaded and checked there: ``projector_repo``), for model repositories that carry
none, such as mradermacher's imatrix (``i1``) quants beside their static-quant sibling. ``file`` may be
empty (the auto-pick of ``hf.pick_file`` applies), ``quant`` may name one (``Q5_K_M``), ``n_ctx`` 0
means the engine default. ``DEFAULT_PRESETS`` are only the first-run data; a saved list (even an empty
one) wins. ``CUSTOM_FAMILIES`` are the fixed Custom rows beside the saved list: a family's parameter sizes,
whose quants come from the repository's own files; a quant the user chooses becomes an ordinary saved row.

NPU (Snapdragon Hexagon): a row may name its ``backend`` ("npu": NPU only, "gpu": CPU + GPU, or "cpu"; none =
"auto": the engine card's "Run on" mode, whose Auto is the NPU when the phone's NPU check passed and the quant is one
the backend accepts, else the CPU) and its ``npu_layers`` (none = all). There are no separate NPU rows (owner 2026-11-18): the default and Custom
rows are "auto", so they run on the NPU where it is offered; ``npu_quant_kind`` marks the Custom and Sigma quants.
"""

from __future__ import annotations

import copy
import logging
import re
from typing import Any, Optional, Sequence

from glossarion_mobile.services.local_llm import hf

log = logging.getLogger("glossarion.local_llm")

__all__ = [
    "BACKENDS", "CUSTOM_FAMILIES", "DEFAULT_PRESETS", "GPU_PRESETS", "MAX_PRESETS", "NO_QUANT", "NPU_OVER_SESSION",
    "OLD_STOCK_PRESETS", "PREF_KEY", "add_row", "backend_of", "custom_family", "custom_row", "custom_size_for",
    "download_size", "file_quant", "is_gpu_quant", "is_no_quant", "is_old_gpu_row", "load_presets", "npu_quant_kind",
    "npu_size_note", "projector_repo", "quant_label", "remove_row", "restore_defaults", "retire_old_gpu_rows",
    "save_presets", "validate_preset",
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

#: First-run presets. Owner 2026-11-25: every offered on-device model is an uncensored build, because the stock
#: models refused many novel passages. These are Heretic builds (p-e-w's automatic abliteration: it removes the
#: refusal direction from the text weights) in mradermacher's imatrix (``i1``) quants: Garbage, Light and Medium
#: use i1-Q4_K_M (the same quality tier as the Unsloth UD-Q4_K_XL files they replace). The i1 repositories hold no
#: vision projector, so ``mmproj_repo`` names where it comes from: the static-quant sibling of the same model
#: (4B), or, where no Heretic projector exists (0.8B, 2B), Unsloth's stock Qwen3.5 repository (Heretic leaves the
#: vision tower alone, so the stock projector fits). Verified on Hugging Face 2026-11-25
#: (``/api/models/<repo>?blobs=true``: not gated; size and sha256 = the LFS oid).
#: Sigma replaces Heavy, is already uncensored, and selects from its live HF files.
#: The Qwen rows are architecture ``qwen35``, 262,144 trained context. Qwen3.5 is natively multimodal: with its
#: projector (llama.cpp's mtmd, libmtmd in our own llama-cpp-python build) the same model reads images, for the
#: chat's Vision mode and image jobs.
DEFAULT_PRESETS: tuple = (
    {"name": "Garbage", "repo": "mradermacher/Qwen3.5-0.8B-heretic-v3-i1-GGUF",
     "file": "Qwen3.5-0.8B-heretic-v3.i1-Q4_K_M.gguf", "quant": "",
     "note": "Qwen3.5 0.8B uncensored · fastest, rough translations · ~1 GB RAM", "n_ctx": 0, "size": 529298208,
     "sha256": "3c1a21ee949303cbb65e8a8380ffa17161901680080e12e08c397528e110c0d1",
     "mmproj_repo": "unsloth/Qwen3.5-0.8B-GGUF", "mmproj": "mmproj-F16.gguf", "mmproj_size": 204987232,
     "mmproj_sha256": "56e4c6cfe73b0c82e3e82bc518d7591997e61d81f723fc41a586f4fa69ea2453"},
    {"name": "Light", "repo": "mradermacher/Qwen3.5-2B-heretic-i1-GGUF",
     "file": "Qwen3.5-2B-heretic.i1-Q4_K_M.gguf", "quant": "",
     "note": "Qwen3.5 2B uncensored · quick drafts on most phones · ~2 GB RAM", "n_ctx": 0, "size": 1270809248,
     "sha256": "66e55c4ccdb612daed6ff4149b46ab1fde736ad56f6de6903de6709dbe760c90",
     "mmproj_repo": "unsloth/Qwen3.5-2B-GGUF", "mmproj": "mmproj-F16.gguf", "mmproj_size": 668227264,
     "mmproj_sha256": "7035e9cb8d7c6a9681d07eef9a364783e86ea4cd73faab2eabb4f43a101830c7"},
    {"name": "Medium", "repo": "mradermacher/Qwen3.5-4B-heretic-i1-GGUF",
     "file": "Qwen3.5-4B-heretic.i1-Q4_K_M.gguf", "quant": "",
     "note": "Qwen3.5 4B uncensored · better quality · needs ~4 GB free RAM", "n_ctx": 0, "size": 2707515008,
     "sha256": "a50ab7af6dabdb6b59b342c4aa89aaad4dc1b4d82bfc67286bb276a5c84f1018",
     "mmproj_repo": "mradermacher/Qwen3.5-4B-heretic-GGUF", "mmproj": "Qwen3.5-4B-heretic.mmproj-f16.gguf",
     "mmproj_size": 672423552, "mmproj_sha256": "e638dc8de3b75309a190092ba006307759343b62ae0d21ed8359df76b9b76c3b"},
    SIGMA_PRESET,
)
#: The stock first-run rows of earlier versions (Unsloth's censored Qwen3.5 UD-Q4_K_XL): a saved row with one's
#: name, repo and file becomes the default row of that name (``load_presets``). Kept only to recognise them.
OLD_STOCK_PRESETS: tuple = (
    ("Garbage", "unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-UD-Q4_K_XL.gguf"),
    ("Light", "unsloth/Qwen3.5-2B-GGUF", "Qwen3.5-2B-UD-Q4_K_XL.gguf"),
    ("Medium", "unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-UD-Q4_K_XL.gguf"),
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
#: chosen size's repository (``hf.list_repo``), so nothing here names a quant. Owner 2026-11-25: every size is an
#: uncensored build: Heretic (p-e-w's automatic abliteration) where one has a Q4_0 file, else huihui-ai's
#: abliterated Gemma 3 1B. Checked 2026-11-25 on https://huggingface.co/api/models/<repo>?blobs=true: every repo
#: exists, is not gated (no Hugging Face login) and has a Q4_0 file. mradermacher's ``i1`` (imatrix) repositories
#: carry no projector: a size's ``mmproj_repo`` names the repository its vision projector is listed, downloaded
#: and checked in (the static-quant sibling of the same model; Unsloth's stock Qwen3.5 0.8B / 2B, which have no
#: Heretic projector). llama-cpp-python 0.3.37 vendors llama.cpp a956a852, whose llama-arch.cpp knows all four
#: architectures (qwen35, llama, gemma3 and gemma4, lfm2) and whose libmtmd knows the Qwen3.5 and Gemma
#: projectors. Gemma 3 4B and Gemma 4 read images; so does every Qwen 3.5 size. LFM2 is the lightweight family:
#: 350M to 1.2B parameters, 220-698 MB at Q4_0, trained for English, Arabic, Chinese, French, German, Japanese,
#: Korean and Spanish (LFM2.5 also Portuguese).
CUSTOM_FAMILIES: tuple = (
    {"id": "qwen35", "name": "Qwen 3.5", "sizes": (
        {"label": "0.8B", "name": "Qwen 3.5 0.8B", "repo": "mradermacher/Qwen3.5-0.8B-heretic-v3-i1-GGUF",
         "mmproj_repo": "unsloth/Qwen3.5-0.8B-GGUF"},
        {"label": "2B", "name": "Qwen 3.5 2B", "repo": "mradermacher/Qwen3.5-2B-heretic-i1-GGUF",
         "mmproj_repo": "unsloth/Qwen3.5-2B-GGUF"},
        {"label": "4B", "name": "Qwen 3.5 4B", "repo": "mradermacher/Qwen3.5-4B-heretic-i1-GGUF",
         "mmproj_repo": "mradermacher/Qwen3.5-4B-heretic-GGUF"},
        {"label": "9B", "name": "Qwen 3.5 9B", "repo": "mradermacher/Qwen3.5-9B-heretic-v2-i1-GGUF",
         "mmproj_repo": "mradermacher/Qwen3.5-9B-heretic-v2-GGUF"},
    )},
    {"id": "llama", "name": "Llama", "sizes": (
        {"label": "1B", "name": "Llama 3.2 1B", "repo": "mradermacher/Llama-3.2-1B-Instruct-heretic-i1-GGUF"},
        {"label": "3B", "name": "Llama 3.2 3B", "repo": "mradermacher/Llama-3.2-3B-Instruct-heretic-i1-GGUF"},
        {"label": "8B", "name": "Llama 3.1 8B", "repo": "bartowski/p-e-w_Llama-3.1-8B-Instruct-heretic-GGUF"},
    )},
    # mobile hotfix 1 #4: a ``group`` heads its sizes in the menu, so Gemma 4 reads as its own generation
    {"id": "gemma", "name": "Gemma", "sizes": (
        {"label": "270M", "name": "Gemma 3 270M", "repo": "ZuzeTt/gemma-3-270m-it-heretic-GGUF", "group": "Gemma 3"},
        {"label": "1B", "name": "Gemma 3 1B", "repo": "bartowski/huihui-ai_gemma-3-1b-it-abliterated-GGUF",
         "group": "Gemma 3"},
        {"label": "4B", "name": "Gemma 3 4B", "repo": "mradermacher/gemma-3-4b-it-heretic-v1.2-i1-GGUF",
         "mmproj_repo": "mradermacher/gemma-3-4b-it-heretic-v1.2-GGUF", "group": "Gemma 3"},
        {"label": "E2B", "name": "Gemma 4 E2B", "repo": "mradermacher/gemma-4-E2B-it-ultra-uncensored-heretic-i1-GGUF",
         "mmproj_repo": "mradermacher/gemma-4-E2B-it-ultra-uncensored-heretic-GGUF", "group": "Gemma 4"},
        {"label": "E4B", "name": "Gemma 4 E4B", "repo": "mradermacher/gemma-4-E4B-it-ultra-uncensored-heretic-i1-GGUF",
         "mmproj_repo": "mradermacher/gemma-4-E4B-it-ultra-uncensored-heretic-GGUF", "group": "Gemma 4"},
    )},
    {"id": "lfm2", "name": "LFM2", "sizes": (
        {"label": "350M", "name": "LFM2.5 350M", "repo": "mradermacher/LFM2.5-350M-heretic-i1-GGUF"},
        {"label": "700M", "name": "LFM2 700M", "repo": "mradermacher/LFM2-700M-Heretic-i1-GGUF"},
        {"label": "1.2B", "name": "LFM2.5 1.2B", "repo": "mradermacher/LFM2.5-1.2B-Instruct-Heretic-i1-GGUF"},
    )},
)
#: The unquantized GGUF kinds: what "No quant" in the Custom menu stands for
NO_QUANT = ("F16", "BF16", "F32")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
#: a row's ``backend``: "auto" (also an empty or missing value) or one device
BACKENDS = ("auto", "npu", "gpu", "cpu")
BACKEND_LABELS = {"auto": "Auto", "npu": "NPU", "gpu": "GPU", "cpu": "CPU"}
#: one NPU session maps about 3.5 GB (llama.cpp docs/backend/snapdragon at the pin); bigger files are slower
NPU_OVER_SESSION = "over the NPU's 3.5 GB: slower"
#: the Custom sizes whose files are all over one NPU session (the Custom sheet notes it where the NPU is offered)
_NPU_OVER_SESSION_REPOS = {"mradermacher/qwen3.5-9b-heretic-v2-i1-gguf"}


def npu_size_note(size: Optional[dict]) -> str:
    """The NPU note of a Custom size (``NPU_OVER_SESSION`` for Qwen 3.5 9B), else ""."""
    return NPU_OVER_SESSION if str((size or {}).get("repo") or "").casefold() in _NPU_OVER_SESSION_REPOS else ""


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
    mmproj_repo = _projector_repo_field(row.get("mmproj_repo"), repo)
    if mmproj_repo:
        out["mmproj_repo"] = mmproj_repo
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


def _projector_repo_field(value: Any, repo: Any) -> str:
    """A row's ``mmproj_repo`` as stored: a Hugging Face repo id other than the row's own ``repo``, else ""."""
    text = str(value or "").strip()
    if not text or text.casefold() == str(repo or "").strip().casefold():
        return ""
    try:
        return hf._check_repo(text)
    except hf.HfError:
        return ""


def projector_repo(row: Optional[dict], repo: Any = None) -> str:
    """The repository a row's vision projector comes from: its ``mmproj_repo``, else the ``mmproj_repo`` of the
    Custom size of its repository (a downloaded model whose preset row is gone, or a row saved before the field
    existed), else the row's own ``repo``. ``repo`` stands in for a missing ``row["repo"]``."""
    row = row or {}
    own = str(row.get("repo") or repo or "").strip()
    named = _projector_repo_field(row.get("mmproj_repo"), own)
    if named:
        return named
    found = custom_size_for(own)
    return _projector_repo_field((found[1] if found else {}).get("mmproj_repo"), own) or own


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
    renamed = _uncensor_stock_rows(rows) or repaired
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


def _uncensor_stock_rows(rows: list) -> bool:
    """Owner 2026-11-25: the saved stock Garbage / Light / Medium rows of earlier versions (``OLD_STOCK_PRESETS``:
    stock name, repo and file) become today's uncensored default rows of the same name, in place. Renamed or edited
    rows and Custom picks are the user's own and stay; a downloaded old file stays on the phone (the Model manager
    lists and deletes it). A stock row whose replacement is already in the list is dropped. True when anything
    changed (in place)."""
    changed = False
    for row in list(rows):
        stock = next((name for name, repo, file in OLD_STOCK_PRESETS
                      if row["name"] == name and row["repo"].casefold() == repo.casefold() and row["file"] == file), None)
        if stock is None:
            continue
        new = next(p for p in DEFAULT_PRESETS if p["name"] == stock)
        changed = True
        if any(other is not row and _same_file(other, new) for other in rows):
            rows.remove(row)
            continue
        replacement = copy.deepcopy(new)
        for key in ("n_ctx", "backend", "npu_layers"):  # the user's own settings of the row stay
            if row.get(key):
                replacement[key] = row[key]
        row.clear()
        row.update(replacement)
    if changed:
        log.info("on-device presets: the stock Qwen3.5 rows now use the uncensored models")
    return changed


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
    """The preset row of a quant chosen in the Custom menu: ``file`` and ``projector`` are ``hf.HfFile`` (the
    projector one of ``projector_repo(size)``, which the row keeps as ``mmproj_repo`` when it is another one)."""
    label = quant_label(file.path)
    row = {"name": f"{size['name']} · {label}", "repo": size["repo"], "file": file.path, "quant": "",
           "note": f"Custom · {size['name']} · {label}", "n_ctx": 0, "size": int(file.size or 0)}
    if file.sha256:
        row["sha256"] = file.sha256
    mmproj_repo = _projector_repo_field(size.get("mmproj_repo"), size["repo"])
    if mmproj_repo:
        row["mmproj_repo"] = mmproj_repo
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
        if old.get("repo") == clean["repo"]:  # the same repository keeps its vision projector (and where it is from)
            for key in ("mmproj_repo", "mmproj", "mmproj_size", "mmproj_sha256"):
                if old.get(key):
                    clean[key] = old[key]
    return clean, None
