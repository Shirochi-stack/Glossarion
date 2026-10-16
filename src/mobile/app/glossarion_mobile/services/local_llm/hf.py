"""Hugging Face references for on-device models: parse what the user typed, list a repo's GGUF files,
pick one, and build the pinned download URL.

Accepted (``parse_ref``):

* ``user/repo``, ``user/repo:Q4_K_M``, ``hf.co/user/repo[:quant]`` (the Ollama-style form);
* ``user/repo/path/file.gguf``;
* ``https://huggingface.co/user/repo``, ``…/tree/<rev>``, ``…/blob/<rev>/<file>.gguf``,
  ``…/resolve/<rev>/<file>.gguf``.

``list_repo`` lists a repository's runnable GGUF files and, apart, its vision projectors
(``mmproj-*.gguf``); ``pick_projector`` chooses the one downloaded with a model (F16 first).

The API base is ``HF_ENDPOINT`` (env ``HF_ENDPOINT``, default ``https://huggingface.co``) so tests and
mirrors can point it elsewhere. Downloads always use the resolved commit, so a resumed transfer fetches
the same bytes.
"""

from __future__ import annotations

import dataclasses
import os
import re
from typing import Any, Iterable, Optional, Sequence
from urllib.parse import quote, unquote, urlsplit

__all__ = [
    "HF_ENDPOINT", "HfError", "HfFile", "HfRef", "PROJECTOR_ORDER", "QUANT_ORDER", "RepoFiles", "list_files",
    "list_repo", "parse_ref", "pick_file", "pick_projector", "repo_url", "resolve_url",
]

HF_ENDPOINT = (os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")
TIMEOUT = 30.0
#: Auto-pick order when the reference names no quant (Q4_K_M: the usual phone trade-off).
QUANT_ORDER = ("Q4_K_M", "Q4_K_S", "Q4_0", "IQ4_XS", "Q5_K_M", "Q8_0")

_PART = r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}"
_REPO_RE = re.compile(rf"^{_PART}/{_PART}$")
_QUANT_RE = re.compile(r"^[A-Za-z0-9_.-]{1,32}$")
_SPLIT_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)
_HOSTS = ("huggingface.co", "www.huggingface.co", "hf.co")
USAGE = "Enter a Hugging Face GGUF repo (user/repo, user/repo:Q4_K_M) or a huggingface.co link"


class HfError(ValueError):
    """A reference that cannot be used, or a Hugging Face answer that says why not (the message is
    what the screen shows)."""


@dataclasses.dataclass(frozen=True)
class HfRef:
    repo: str
    file: Optional[str] = None
    revision: str = "main"
    quant: Optional[str] = None

    @property
    def owner(self) -> str:
        return self.repo.split("/", 1)[0]

    def text(self) -> str:
        """The canonical short form (what a preset stores in ``repo``)."""
        if self.file:
            return f"{self.repo}/{self.file}"
        return f"{self.repo}:{self.quant}" if self.quant else self.repo


@dataclasses.dataclass(frozen=True)
class HfFile:
    path: str
    size: int
    sha256: Optional[str] = None
    split: bool = False

    @property
    def name(self) -> str:
        return self.path.rsplit("/", 1)[-1]


def _check_repo(repo: str) -> str:
    if not _REPO_RE.match(repo) or ".." in repo:
        raise HfError(f"“{repo}” is not a Hugging Face repo id (user/repo). {USAGE}.")
    return repo


def _check_file(path: str) -> str:
    path = path.strip().strip("/")
    if not path or path.startswith(("/", "\\")) or "\\" in path or any(p in ("", ".", "..") for p in path.split("/")):
        raise HfError("That file path is not allowed")
    if not path.lower().endswith(".gguf"):
        raise HfError("Only .gguf files can run on this device")
    return path


def _split_quant(text: str) -> tuple:
    repo, sep, quant = text.partition(":")
    if not sep:
        return text, None
    quant = quant.strip()
    if not quant or not _QUANT_RE.match(quant):
        raise HfError(f"“{quant}” is not a quant name (for example Q4_K_M)")
    return repo, quant


def parse_ref(text: Any) -> HfRef:
    """``HfRef`` for what the user typed; ``HfError`` (with a precise message) for anything else."""
    raw = str(text or "").strip()
    if not raw:
        raise HfError(USAGE)
    if re.match(r"^[a-z][a-z0-9+.-]*://", raw, re.IGNORECASE):
        parts = urlsplit(raw)
        if parts.scheme.lower() not in ("http", "https") or (parts.hostname or "").lower() not in _HOSTS:
            raise HfError("Only huggingface.co links are supported")
        segments = [unquote(s) for s in parts.path.split("/") if s]
        if segments and segments[0] in ("models", "api"):
            raise HfError(USAGE)
        if len(segments) < 2:
            raise HfError(USAGE)
        repo = _check_repo(f"{segments[0]}/{segments[1]}")
        rest = segments[2:]
        if not rest:
            return HfRef(repo)
        kind = rest[0]
        if kind == "tree" and len(rest) >= 2:
            return HfRef(repo, revision=_check_revision(rest[1]))
        if kind in ("blob", "resolve") and len(rest) >= 3:
            return HfRef(repo, file=_check_file("/".join(rest[2:])), revision=_check_revision(rest[1]))
        raise HfError(USAGE)
    lowered = raw.lower()
    for prefix in ("hf.co/", "huggingface.co/"):
        if lowered.startswith(prefix):
            raw = raw[len(prefix):]
            break
    if raw.count("/") >= 2:  # user/repo/path/file.gguf
        owner, name, path = raw.split("/", 2)
        return HfRef(_check_repo(f"{owner}/{name}"), file=_check_file(path))
    repo, quant = _split_quant(raw)
    return HfRef(_check_repo(repo), quant=quant)


def _check_revision(rev: str) -> str:
    if not re.match(r"^[A-Za-z0-9._-]{1,96}$", rev) or ".." in rev:
        raise HfError(f"“{rev}” is not a branch or commit")
    return rev


# ---- the API ---------------------------------------------------------------------------------------


def repo_url(repo: str) -> str:
    return f"{HF_ENDPOINT}/{repo}"


def resolve_url(repo: str, revision: str, file: str) -> str:
    return f"{HF_ENDPOINT}/{repo}/resolve/{quote(revision, safe='')}/{quote(file, safe='/')}"


def _http_error(status: int, repo: str) -> HfError:
    if status in (401, 403):
        return HfError(f"Hugging Face did not give access to {repo}: it does not exist, or it is gated or "
                       "private (Glossarion cannot sign in to Hugging Face)")
    if status == 404:
        return HfError(f"{repo} was not found on Hugging Face")
    if status == 429:
        return HfError("Hugging Face is rate limiting this device; try again in a minute")
    return HfError(f"Hugging Face answered HTTP {status}")


def _get_json(session: Any, url: str, repo: str) -> tuple:
    import requests

    try:
        response = (session or requests).get(url, timeout=TIMEOUT, headers={"Accept": "application/json"})
    except requests.RequestException as exc:
        raise HfError(f"Could not reach Hugging Face ({type(exc).__name__})") from exc
    if response.status_code != 200:
        raise _http_error(response.status_code, repo)
    try:
        return response.json(), response
    except ValueError as exc:
        raise HfError("Hugging Face sent an unreadable answer") from exc


def _is_projector(path: str) -> bool:
    """A vision projector (``mmproj-*.gguf``, unsplit): the image encoder of a multimodal model."""
    name = path.rsplit("/", 1)[-1].lower()
    return name.endswith(".gguf") and name.startswith("mmproj") and not _SPLIT_RE.search(name)


def _is_candidate(path: str) -> bool:
    name = path.rsplit("/", 1)[-1].lower()
    if not name.endswith(".gguf"):
        return False
    return not (name.startswith("mmproj") or "imatrix" in name)


@dataclasses.dataclass(frozen=True)
class RepoFiles:
    """One listing of a repository at a commit: runnable GGUF files and vision projectors."""
    commit: str
    files: tuple
    projectors: tuple


def list_repo(ref: HfRef, *, session: Any = None) -> RepoFiles:
    """The repo's runnable GGUF files at ``ref.revision`` (imatrix files and split shards after the first
    are left out; a first shard has ``split=True``) and, apart, its vision projectors (``mmproj-*``)."""
    info, _response = _get_json(session, f"{HF_ENDPOINT}/api/models/{ref.repo}/revision/{quote(ref.revision, safe='')}",
                                ref.repo)
    commit = str((info or {}).get("sha") or ref.revision)
    url: Optional[str] = f"{HF_ENDPOINT}/api/models/{ref.repo}/tree/{quote(commit, safe='')}?recursive=1"
    files: list = []
    projectors: list = []
    pages = 0
    while url and pages < 20:
        pages += 1
        entries, response = _get_json(session, url, ref.repo)
        for entry in entries if isinstance(entries, list) else ():
            if not isinstance(entry, dict) or entry.get("type", "file") != "file":
                continue
            path = str(entry.get("path") or "")
            projector = _is_projector(path)
            if not projector and not _is_candidate(path):
                continue
            split = _SPLIT_RE.search(path)
            if split and split.group(1) != "00001":
                continue
            lfs = entry.get("lfs") if isinstance(entry.get("lfs"), dict) else {}
            sha = str(lfs.get("oid") or "").lower() or None
            size = int(lfs.get("size") or entry.get("size") or 0)
            item = HfFile(path, size, sha if sha and re.fullmatch(r"[0-9a-f]{64}", sha) else None, bool(split))
            (projectors if projector else files).append(item)
        url = _next_link(getattr(response, "headers", {}) or {})
    files.sort(key=lambda f: f.path.lower())
    projectors.sort(key=lambda f: f.path.lower())
    return RepoFiles(commit, tuple(files), tuple(projectors))


def list_files(ref: HfRef, *, session: Any = None) -> tuple:
    """``(commit, [HfFile])``: the repo's runnable GGUF files (``list_repo`` without the projectors)."""
    listing = list_repo(ref, session=session)
    return listing.commit, list(listing.files)


#: Projector precision order: F16 (what llama.cpp's own conversions ship; the CPU build runs it),
#: then BF16, Q8_0 (smaller, slightly worse), F32 (twice the size).
PROJECTOR_ORDER = ("F16", "BF16", "Q8_0", "F32")


def pick_projector(projectors: Iterable[HfFile], file: Optional[str] = None) -> Optional[HfFile]:
    """The vision projector to download with a model: ``file`` when the preset names one (None when the
    repository does not have it), else the first of ``PROJECTOR_ORDER``, else the smallest; None when
    the repository has none (a text-only model)."""
    projectors = [p for p in projectors if not p.split]
    if file:
        wanted = file.strip("/").lower()
        return next((p for p in projectors if p.path.lower() == wanted or p.name.lower() == wanted), None)
    for name in PROJECTOR_ORDER:
        found = _matches(projectors, name)
        if found:
            return found[0]
    return min(projectors, key=lambda p: (p.size or 0, p.path)) if projectors else None


def _next_link(headers: Any) -> Optional[str]:
    link = str(headers.get("Link") or headers.get("link") or "")
    match = re.search(r'<([^>]+)>;\s*rel="next"', link)
    if not match:
        return None
    url = match.group(1)
    # follow only Hugging Face's own API pages (never a host a header names)
    return url if url.startswith(f"{HF_ENDPOINT}/api/models/") else None


def _quant_re(quant: str) -> re.Pattern:
    return re.compile(rf"(?<![A-Za-z0-9]){re.escape(quant)}(?![A-Za-z0-9])", re.IGNORECASE)


def _matches(files: Sequence[HfFile], quant: str) -> list:
    pattern = _quant_re(quant)
    return sorted((f for f in files if pattern.search(f.name)), key=lambda f: (len(f.name), f.name))


def pick_file(files: Iterable[HfFile], quant: Optional[str] = None, file: Optional[str] = None) -> Optional[HfFile]:
    """The file to download: ``file`` exactly, else the named ``quant``, else ``QUANT_ORDER``; None when
    the user should choose (several files, none of the usual quants). ``HfError`` when the named file /
    quant is not there or the choice is a split GGUF."""
    files = list(files)
    chosen: Optional[HfFile] = None
    if file:
        wanted = file.strip("/").lower()
        chosen = next((f for f in files if f.path.lower() == wanted), None)
        if chosen is None:
            raise HfError(f"{file} is not in this repository (or is not a runnable GGUF file)")
    elif quant:
        found = _matches(files, quant)
        if not found:
            raise HfError(f"No {quant} file in this repository")
        chosen = found[0]
    else:
        if not files:
            raise HfError("This repository has no GGUF model files")
        if len(files) == 1:
            chosen = files[0]
        else:
            for name in QUANT_ORDER:
                found = _matches(files, name)
                if found:
                    chosen = found[0]
                    break
    if chosen is not None and chosen.split:
        raise HfError("Split GGUF files are not supported yet")
    return chosen
