"""Chat transcript operations without Flet (pure Python, Python 3.10).

* **Versions** (UI_SPEC §2.10): an edit-and-resend or a retranslation records the new user
  turn as a version of the original (sidecar ``versions``: ``{anchor fp: {members, selected}}``);
  ``version_view`` hides the turns (and their responses) of the unselected members and says
  where the "‹ 2/3 ›" switcher goes. Desktop, which knows nothing of versions, shows every
  member as consecutive cards.
* **Jump-to / search** (§2.18): the Input / Output lists (desktop navigators) and the match
  list of "Search in chat" over the message bodies.
* **Export chat** (§2.18): a Markdown transcript, or a ZIP of the chat folder plus the v2 JSON
  subset of this session with its file references rebased into the ZIP.
"""

from __future__ import annotations

import json
import os
import zipfile
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

__all__ = [
    "JumpEntry",
    "VersionView",
    "build_chat_export_zip",
    "jump_entries",
    "safe_file_stem",
    "search_matches",
    "transcript_markdown",
    "turn_span",
    "version_view",
    "workspace_summary",
]

STORAGE_PATH_KEYS = (
    "content_path", "content_text_path", "content_html_path", "content_xhtml_path",
    "thinking_path", "image_path", "media_path",
)


def _role(message: Any) -> str:
    return str(message[0]) if isinstance(message, (list, tuple)) and message else ""


def turn_span(messages: Sequence[Any], user_index: int) -> list:
    """The user turn at ``user_index`` and its responses (assistant messages up to the next user turn)."""
    if not (0 <= user_index < len(messages)) or _role(messages[user_index]) not in ("user", "user_file"):
        return []
    span = [user_index]
    for index in range(user_index + 1, len(messages)):
        if _role(messages[index]) != "assistant":
            break
        span.append(index)
    return span


# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------


@dataclass
class VersionView:
    hidden: set = field(default_factory=set)  # message indices not rendered
    switchers: dict = field(default_factory=dict)  # visible member's user index -> (anchor fp, selected, total)


def version_view(messages: Sequence[Any], fingerprints: Sequence[str], groups: Mapping[str, Any]) -> VersionView:
    """Which messages the transcript hides for the version groups, and where the switchers go."""
    view = VersionView()
    index_of = {fp: i for i, fp in enumerate(fingerprints)}
    for anchor, group in (groups or {}).items():
        members = [fp for fp in (group or {}).get("members") or () if fp in index_of]
        if len(members) < 2:
            continue
        try:
            selected = int((group or {}).get("selected", len(members) - 1))
        except (TypeError, ValueError):
            selected = len(members) - 1
        selected = max(0, min(selected, len(members) - 1))
        for position, fp in enumerate(members):
            span = turn_span(messages, index_of[fp])
            if position == selected:
                if span:
                    view.switchers[span[0]] = (anchor, selected, len(members))
            else:
                view.hidden.update(span)
    return view


# ---------------------------------------------------------------------------
# Jump-to and search
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JumpEntry:
    index: int
    label: str


def _preview(text: Any, limit: int = 80) -> str:
    value = " ".join(str(text or "").split())
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "…"


def jump_entries(messages: Sequence[Any], hidden: Iterable[int] = ()) -> tuple:
    """``(inputs, outputs)``: the desktop Input / Output navigators as rows ("1. preview…",
    "📎 name — prompt"; outputs use their request label, else the inline text)."""
    hidden = set(hidden)
    inputs: list = []
    outputs: list = []
    for index, message in enumerate(messages):
        if index in hidden:
            continue
        role = _role(message)
        if role == "user":
            inputs.append(JumpEntry(index, f"{len(inputs) + 1}. {_preview(message[1] if len(message) > 1 else '')}"))
        elif role == "user_file":
            name = str(message[1] if len(message) > 1 else "")
            prompt = _preview(message[4] if len(message) > 4 else "", 60)
            inputs.append(JumpEntry(index, f"📎 {name}" + (f" — {prompt}" if prompt else "")))
        elif role == "assistant":
            label = str(message[5] if len(message) > 5 else "") or _preview(message[1] if len(message) > 1 else "")
            outputs.append(JumpEntry(index, f"{len(outputs) + 1}. {label or 'Response'}"))
    return inputs, outputs


def search_matches(messages: Sequence[Any], query: str, body: Callable[[int], str],
                   hidden: Iterable[int] = ()) -> list:
    """Indices of messages whose text contains ``query`` (case-insensitive; assistant bodies are
    read through ``body(index)``, i.e. the lazy ``Chat Messages/`` files). Blocking."""
    needle = str(query or "").strip().casefold()
    if not needle:
        return []
    hidden = set(hidden)
    found = []
    for index, message in enumerate(messages):
        if index in hidden:
            continue
        role = _role(message)
        if role == "user":
            texts = [message[1] if len(message) > 1 else ""]
        elif role == "user_file":
            texts = [message[1] if len(message) > 1 else "", message[4] if len(message) > 4 else ""]
        elif role == "assistant":
            texts = [body(index), message[5] if len(message) > 5 else ""]
        else:
            continue
        if any(needle in str(text or "").casefold() for text in texts):
            found.append(index)
    return found


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def safe_file_stem(title: Any, fallback: str = "Chat") -> str:
    import re

    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", " ".join(str(title or "").split())).rstrip(" .")[:60]
    return text or fallback


def transcript_markdown(title: str, messages: Sequence[Any], body: Callable[[int], str]) -> str:
    """The chat as a Markdown transcript (Export chat › Markdown). Blocking (reads bodies)."""
    lines = [f"# {title or 'Chat'}", ""]
    for index, message in enumerate(messages):
        role = _role(message)
        if role == "user":
            lines += ["**You:**", "", str(message[1] if len(message) > 1 else ""), ""]
        elif role == "user_file":
            name = str(message[1] if len(message) > 1 else "")
            prompt = str(message[4] if len(message) > 4 else "")
            lines += [f"**You attached** `{name}`", ""]
            if prompt:
                lines += [prompt, ""]
        elif role == "assistant":
            label = str(message[5] if len(message) > 5 else "")
            lines += ["### GLOSSARION" + (f" · {label}" if label else ""), "", str(body(index) or ""), ""]
    return "\n".join(lines).rstrip() + "\n"


def _under(path: str, folder: str) -> bool:
    try:
        path = os.path.normcase(os.path.abspath(path))
        folder = os.path.normcase(os.path.abspath(folder))
        return os.path.commonpath([path, folder]) == folder
    except (TypeError, ValueError):
        return False


def build_chat_export_zip(session: Mapping[str, Any], resolve: Callable[[str], str], destination: str) -> str:
    """Blocking: ZIP of the chat folder + ``direct_text_chats.json`` holding only this session.

    The chat folder goes to ``Direct Text/<folder name>/`` inside the ZIP and the session's file
    references (``output_folder``, assistant output folders and storage paths, resolved with
    ``resolve`` = the store's ``_resolve_history_file_reference``) are rewritten relative to the
    ZIP root, so the archive is self-contained; paths outside the folder stay absolute.
    """
    folder = os.path.abspath(str(session.get("output_folder") or ""))
    has_folder = bool(session.get("output_folder")) and os.path.isdir(folder)
    inner = "Direct Text/" + (os.path.basename(folder) if has_folder else safe_file_stem(session.get("title")))

    def rebase(path: str) -> str:
        if has_folder and path and _under(path, folder):
            rel = os.path.relpath(os.path.abspath(path), folder).replace("\\", "/")
            return inner if rel == "." else f"{inner}/{rel}"
        return path

    messages = []
    for message in session.get("messages") or []:
        values = list(message)
        if values and values[0] == "assistant":
            while len(values) < 7:
                values.append({} if len(values) == 6 else "")
            values[4] = rebase(os.path.abspath(str(values[4]))) if values[4] else values[4]
            storage = dict(values[6] or {}) if isinstance(values[6], dict) else {}
            for key in STORAGE_PATH_KEYS:
                if storage.get(key):
                    storage[key] = rebase(resolve(str(storage[key])))
            values[6] = storage
        messages.append(values)
    exported = {key: value for key, value in session.items() if key != "messages"}
    exported["messages"] = messages
    exported["output_folder"] = inner if has_folder else ""
    exported["expanded"] = sorted(int(i) for i in (session.get("expanded") or ()))
    payload = {"version": 2, "current_chat_id": session.get("id"), "sessions": [exported]}
    os.makedirs(os.path.dirname(os.path.abspath(destination)) or ".", exist_ok=True)
    temp = destination + ".part"
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("direct_text_chats.json", json.dumps(payload, ensure_ascii=False, indent=2))
        if has_folder:
            for root, _dirs, files in os.walk(folder):
                for name in files:
                    path = os.path.join(root, name)
                    rel = os.path.relpath(path, folder).replace("\\", "/")
                    archive.write(path, f"{inner}/{rel}")
    os.replace(temp, destination)
    return destination


# ---------------------------------------------------------------------------
# Attachments manager
# ---------------------------------------------------------------------------


def workspace_summary(folder: str) -> str:
    """Blocking, read-only: "48/48 · EPUB ready" for an ``Attachments/<stem>`` workspace card.

    Counts the chapter entries of its ``translation_progress.json`` (completed / all, the
    pipeline's own statuses; nothing is written) and names the compiled EPUB / PDF the
    desktop Migrate would keep (``ChatStoreMixin._preferred_attachment_compiled_documents``).
    """
    parts = []
    progress = os.path.join(folder, "translation_progress.json")
    try:
        with open(progress, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        chapters = [c for c in (data.get("chapters") or {}).values() if isinstance(c, dict)]
        if chapters:
            done = sum(1 for c in chapters if str(c.get("status") or "") == "completed")
            parts.append(f"{done}/{len(chapters)}")
    except (OSError, ValueError, AttributeError):
        pass
    try:
        from direct_text_store import ChatStoreMixin  # shared (U3)

        preferred = ChatStoreMixin._preferred_attachment_compiled_documents(folder)
        for extension in (".epub", ".pdf"):
            if preferred.get(extension):
                parts.append(f"{extension[1:].upper()} ready")
    except Exception:
        pass
    return " · ".join(parts) or "No progress yet"
