"""
gemini_free.py - browser-backed Google Search / Gemini route.

This module is intentionally not an official Gemini API client. It uses Qt
WebEngine to open Google Search's AI/search page and extract the rendered page
text. The public entry point mirrors the other browser-backed auth modules:
send_chat_completion(...) returns a dict with content/finish_reason/raw_response.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import queue
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional
from urllib.parse import parse_qsl, urlencode, urlparse


SEARCH_BASE_URL = "https://www.google.com/search"
DEFAULT_SEARCH_PARAMS = {
    "udm": "50",
    "aep": "46",
    "source": "25q2-US-SearchSites-Site-CTA",
    "hl": "en",
}
DEFAULT_URL = f"{SEARCH_BASE_URL}?{urlencode(DEFAULT_SEARCH_PARAMS)}"
DEFAULT_MODEL = "gemini"
DEFAULT_TIMEOUT = 90
DEFAULT_SUBMIT_MODE = "url"
DEFAULT_SUBCHUNK_PROMPT_CHARS = 1200
DEFAULT_SUBCHUNK_URL_CHARS = 3500
DEFAULT_SUBCHUNK_SAFETY_CHARS = 300
DEFAULT_MIN_SUBCHUNK_BODY_CHARS = 80
DEFAULT_SUBCHUNK_TIMEOUT = 90
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
GENERATION_FAILURE_MARKERS = (
    "something went wrong and the content wasn't generated",
    "something went wrong and the content was not generated",
    "content wasn't generated",
    "content was not generated",
)
PAGE_SNAPSHOT_SCRIPT = """
JSON.stringify({
  url: location.href,
  title: document.title || "",
  ready: document.readyState || "",
  text: document.body ? document.body.innerText || "" : "",
  htmlLength: document.documentElement ? document.documentElement.outerHTML.length : 0,
  busy: Array.from(document.querySelectorAll('button,[role="button"]')).some((button) => {
    const style = window.getComputedStyle(button);
    const rect = button.getBoundingClientRect();
    const visible = style.visibility !== "hidden" && style.display !== "none" && rect.width > 0 && rect.height > 0;
    const label = [
      button.getAttribute("aria-label") || "",
      button.getAttribute("title") || "",
      button.innerText || ""
    ].join(" ").toLowerCase();
    return visible && label.includes("stop") && !button.disabled && button.getAttribute("aria-disabled") !== "true";
  })
})
"""


def _page_snapshot_script(prompt: str = "") -> str:
    prompt_json = json.dumps(str(prompt or ""), ensure_ascii=False)
    return f"""
(() => {{
  const promptText = {prompt_json};
  const normalize = (value) => String(value || "").replace(/\\s+/g, " ").trim();
  const bodyText = document.body ? document.body.innerText || "" : "";
  const promptLines = new Set(
    String(promptText || "")
      .replace(/\\r/g, "\\n")
      .split("\\n")
      .map((line) => line.trim())
      .filter(Boolean)
  );
  const roleMarkers = new Set(["user:", "assistant:", "system instructions:"]);
  const endMarkers = [
    "ai can make mistakes",
    "ai mode response is ready",
    "search results",
    "people also ask",
    "related searches"
  ];

  const lines = bodyText.replace(/\\r/g, "\\n").split("\\n").map((line) => line.trim()).filter(Boolean);
  let start = -1;
  for (let i = 0; i < lines.length; i++) {{
    if (lines[i].toLowerCase() === "you said:") {{
      start = i + 1;
      break;
    }}
  }}
  if (start >= 0) {{
    while (start < lines.length) {{
      const current = lines[start] || "";
      const lower = current.toLowerCase();
      if (!current || roleMarkers.has(lower) || promptLines.has(current)) {{
        start += 1;
        continue;
      }}
      break;
    }}
  }}
  let end = start >= 0 ? lines.length : 0;
  for (let i = Math.max(0, start); i < lines.length; i++) {{
    const lower = (lines[i] || "").toLowerCase();
    if (endMarkers.some((marker) => lower.includes(marker)) || /^\\d+\\s+sites?$/.test(lower)) {{
      end = i;
      break;
    }}
  }}
  const answerLines = start >= 0
    ? lines.slice(start, end).filter((line) => line && !/^\\+\\d+$/.test(line))
    : [];
  const answerText = answerLines.join("\\n").trim();

  const visible = (element) => {{
    if (!element || !element.getBoundingClientRect) return false;
    const style = window.getComputedStyle(element);
    const rect = element.getBoundingClientRect();
    return style && style.display !== "none" && style.visibility !== "hidden" && rect.width > 0 && rect.height > 0;
  }};
  const firstNeedle = normalize(answerLines[0] || "").slice(0, 100);
  const lastNeedle = normalize(answerLines[answerLines.length - 1] || "").slice(0, 100);
  const promptNeedle = normalize(String(promptText || "").split("\\n").find((line) => line.trim()) || "").slice(0, 100);
  const candidates = Array.from(document.body ? document.body.querySelectorAll("article, main, section, div, [role='article'], [data-md], [data-attrid]") : []);
  let best = null;
  for (const element of candidates) {{
    if (!visible(element)) continue;
    const text = normalize(element.innerText || "");
    if (!text) continue;
    let score = 0;
    if (firstNeedle && text.includes(firstNeedle)) score += 80;
    if (lastNeedle && text.includes(lastNeedle)) score += 80;
    if (answerText && text.includes(normalize(answerText).slice(0, Math.min(160, normalize(answerText).length)))) score += 30;
    if (/\\byou said:\\b/i.test(text)) score -= 120;
    if (promptNeedle && text.includes(promptNeedle)) score -= 80;
    if (/ai can make mistakes|search results|people also ask|related searches/i.test(text)) score -= 35;
    const answerLength = Math.max(1, normalize(answerText).length);
    score -= Math.abs(text.length - answerLength) / Math.max(60, answerLength / 10);
    score += Math.min(40, element.querySelectorAll("p,h1,h2,h3,h4,h5,h6,li,blockquote,pre,table,br").length * 3);
    if (!best || score > best.score || (score === best.score && text.length < best.textLength)) {{
      best = {{
        score,
        textLength: text.length,
        html: element.innerHTML || "",
        text: element.innerText || "",
        tag: element.tagName || ""
      }};
    }}
  }}

  return JSON.stringify({{
    url: location.href,
    title: document.title || "",
    ready: document.readyState || "",
    text: bodyText,
    answerText,
    answerHtml: best ? best.html : "",
    answerContainerText: best ? best.text : "",
    answerContainerTag: best ? best.tag : "",
    answerContainerScore: best ? best.score : null,
    htmlLength: document.documentElement ? document.documentElement.outerHTML.length : 0,
    busy: Array.from(document.querySelectorAll('button,[role="button"]')).some((button) => {{
      const style = window.getComputedStyle(button);
      const rect = button.getBoundingClientRect();
      const visibleButton = style.visibility !== "hidden" && style.display !== "none" && rect.width > 0 && rect.height > 0;
      const label = [
        button.getAttribute("aria-label") || "",
        button.getAttribute("title") || "",
        button.innerText || ""
      ].join(" ").toLowerCase();
      return visibleButton && label.includes("stop") && !button.disabled && button.getAttribute("aria-disabled") !== "true";
    }})
  }});
}})()
"""

_cancel_event = threading.Event()
_active_helper_processes: set = set()
_active_helper_lock = threading.Lock()


def _repo_src_dir() -> Path:
    return Path(__file__).resolve().parent


def _debug_enabled() -> bool:
    return os.getenv("GEMINI_FREE_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")


def _log(log_fn: Optional[Callable[[str], None]], message: str, *, debug_only: bool = False) -> None:
    if not log_fn:
        return
    if debug_only and not _debug_enabled():
        return
    log_fn(message)


def _short_error(error: Any, limit: int = 1200) -> str:
    text = str(error or "").replace("\r", " ").replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _safe_url_for_log(url: Any, limit: int = 500) -> str:
    text = str(url or "").strip()
    if not text:
        return ""
    try:
        parsed = urlparse(text)
        if parsed.query:
            params = []
            for key, value in parse_qsl(parsed.query, keep_blank_values=True):
                if key == "q":
                    params.append((key, f"<{len(value)} chars omitted>"))
                else:
                    params.append((key, value))
            text = parsed._replace(query=urlencode(params)).geturl()
    except Exception:
        pass
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def cancel_stream() -> None:
    """Cancel an active helper subprocess if the translation run is stopped."""
    _cancel_event.set()
    _terminate_active_helper_processes(kill=True)


def _terminate_active_helper_processes(*, kill: bool = False) -> None:
    with _active_helper_lock:
        helpers = list(_active_helper_processes)
    for proc in helpers:
        _terminate_process_tree(proc, kill=kill)


def reset_cancel() -> None:
    _cancel_event.clear()


def _is_cancelled() -> bool:
    return _cancel_event.is_set()


def _terminate_process_tree(proc: Any, *, kill: bool = False) -> None:
    try:
        if proc is None or proc.poll() is not None:
            return
    except Exception:
        return
    if os.name == "nt":
        try:
            args = ["taskkill", "/T", "/PID", str(proc.pid)]
            if kill:
                args.insert(1, "/F")
            subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3)
            return
        except Exception:
            pass
    try:
        if kill:
            proc.kill()
        else:
            proc.terminate()
    except Exception:
        pass


def _qtwebengine_chromium_flags(existing: str = "") -> str:
    try:
        sys.path.insert(0, str(_repo_src_dir()))
        from authnd_auth import _qtwebengine_chromium_flags as authnd_flags

        return authnd_flags(existing)
    except Exception:
        pass

    try:
        tokens = shlex.split(existing or "")
    except ValueError:
        tokens = (existing or "").split()

    required_flags = [
        "--disable-gpu",
        "--disable-dev-shm-usage",
        "--no-sandbox",
        "--disable-setuid-sandbox",
    ]
    seen = set()
    merged: List[str] = []
    for flag in [*tokens, *required_flags]:
        if flag in seen:
            continue
        seen.add(flag)
        merged.append(flag)
    return " ".join(merged)


def _default_user_agent() -> str:
    try:
        sys.path.insert(0, str(_repo_src_dir()))
        from authnd_auth import USER_AGENT

        return USER_AGENT
    except Exception:
        return DEFAULT_USER_AGENT


def _origin_url(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return url
    return f"{parsed.scheme}://{parsed.netloc}/"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _parse_headers(values: Optional[List[str]]) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    for value in values or []:
        if ":" not in value:
            raise ValueError(f"header must be in 'Name: value' form: {value}")
        name, header_value = value.split(":", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"header name is empty: {value}")
        headers[name] = header_value.strip()
    return headers


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _write_text(path: str, value: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(value)


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                item_type = str(item.get("type") or "").lower()
                if item_type in ("", "text", "input_text"):
                    parts.append(str(item.get("text") or ""))
                elif item_type in ("image_url", "input_image", "image"):
                    parts.append("[image omitted]")
        return "\n".join(part for part in parts if part)
    return str(content)


def _messages_to_prompt(messages: Iterable[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "user").strip().lower() or "user"
        text = _content_to_text(message.get("content")).strip()
        if not text:
            continue
        if role == "system":
            parts.append(f"System instructions:\n{text}")
        elif role == "assistant":
            parts.append(f"Assistant:\n{text}")
        else:
            parts.append(f"User:\n{text}")
    return "\n\n".join(parts).strip()


def _messages_to_search_url_chars(messages: Iterable[Dict[str, Any]]) -> int:
    return len(_build_search_url(_messages_to_prompt(messages)))


def _generation_failure_error(error: Any) -> bool:
    text = str(error or "").lower()
    return (
        "generation failure" in text
        or "content wasn't generated" in text
        or "content was not generated" in text
    )


def _adaptive_split_enabled() -> bool:
    return os.getenv("GEMINI_FREE_ADAPTIVE_SPLIT", "1").strip().lower() not in ("0", "false", "no", "off")


def _subchunk_prompt_chars() -> int:
    return max(300, _env_int("GEMINI_FREE_SUBCHUNK_PROMPT_CHARS", DEFAULT_SUBCHUNK_PROMPT_CHARS))


def _subchunk_url_chars() -> int:
    return max(1000, _env_int("GEMINI_FREE_SUBCHUNK_URL_CHARS", DEFAULT_SUBCHUNK_URL_CHARS))


def _subchunk_safety_chars() -> int:
    return max(0, _env_int("GEMINI_FREE_SUBCHUNK_SAFETY_CHARS", DEFAULT_SUBCHUNK_SAFETY_CHARS))


def _min_subchunk_body_chars() -> int:
    return max(1, _env_int("GEMINI_FREE_MIN_SUBCHUNK_BODY_CHARS", DEFAULT_MIN_SUBCHUNK_BODY_CHARS))


def _subchunk_timeout_seconds(request_timeout: int) -> int:
    configured = max(30, _env_int("GEMINI_FREE_SUBCHUNK_TIMEOUT", DEFAULT_SUBCHUNK_TIMEOUT))
    return max(30, min(max(30, int(request_timeout or DEFAULT_TIMEOUT)), configured))


def _split_user_instruction_prefix(text: str) -> tuple[str, str]:
    raw = str(text or "")
    if not raw:
        return "", ""
    html_match = re.search(r"<(?:\?xml|!doctype|html|head|body|h[1-6]|p|div|img|ruby|br|ul|ol|li|table|span)\b", raw, re.IGNORECASE)
    if html_match and html_match.start() > 0:
        lead = raw[: html_match.start()].strip()
        if lead.startswith("[") and len(lead) <= 200:
            return raw[: html_match.start()].rstrip(), raw[html_match.start():].lstrip()
    first_line = re.match(r"^([^\r\n]{1,200})(\r\n|\n|\r)(.*)$", raw, flags=re.DOTALL)
    if first_line:
        lead = first_line.group(1).strip()
        if lead.startswith("[") and lead.endswith("]"):
            return first_line.group(1).rstrip(), first_line.group(3).lstrip()
    return "", raw


def _configured_text_extraction_method() -> str:
    method = os.getenv("TEXT_EXTRACTION_METHOD", "").strip().lower()
    extraction_mode = os.getenv("EXTRACTION_MODE", "").strip().lower()
    use_html2text = os.getenv("USE_HTML2TEXT", "").strip().lower() in ("1", "true", "yes", "on")
    if method in ("enhanced", "html2text", "markdown") or extraction_mode == "enhanced" or use_html2text:
        return "enhanced"
    if method in ("standard", "beautifulsoup", "bs"):
        return "standard"
    return ""


def _looks_like_html_payload(text: str) -> bool:
    raw = str(text or "")
    return bool(re.search(r"<[A-Za-z][A-Za-z0-9:_-]*(?:\s[^<>]*)?/?>|</[A-Za-z][A-Za-z0-9:_-]*>", raw))


def _payload_format_for_split(text: str) -> str:
    raw = str(text or "")
    configured_method = _configured_text_extraction_method()
    if configured_method == "enhanced":
        return "text"
    if configured_method == "standard" and _looks_like_html_payload(raw):
        return "html"
    if re.search(r"<(?:!doctype|html|head|body|article|section|main|div|p|h[1-6]|ul|ol|li|table|tr|td|blockquote)\b|</(?:div|p|h[1-6]|ul|ol|li|table|tr|td|blockquote)>", raw, re.IGNORECASE):
        return "html"
    return "text"


def _splitter_name_for_payload(payload_format: str) -> str:
    return "beautifulsoup4" if payload_format == "html" else "text"


def _split_plain_text_units(text: str) -> List[str]:
    raw = str(text or "")
    if not raw:
        return []
    units = re.findall(r".*?(?:\n\s*\n|$)", raw, flags=re.DOTALL)
    units = [unit for unit in units if unit]
    if len(units) <= 1:
        units = [line + "\n" for line in raw.splitlines() if line.strip()]
    return units or [raw]


def _split_html_units_beautifulsoup(text: str) -> List[str]:
    raw = str(text or "")
    if not raw:
        return []
    try:
        from bs4 import BeautifulSoup
        from bs4.element import NavigableString, Tag

        soup = BeautifulSoup(raw, "html.parser")
        container = soup.body if soup.body else soup
        block_names = {
            "article", "aside", "blockquote", "br", "center", "dd", "div", "dl", "dt",
            "figure", "figcaption", "footer", "h1", "h2", "h3", "h4", "h5", "h6",
            "header", "hr", "li", "main", "nav", "ol", "p", "pre", "ruby", "section",
            "table", "tbody", "td", "tfoot", "th", "thead", "tr", "ul",
        }

        def child_units(parent: Any) -> List[str]:
            found: List[str] = []
            for child in getattr(parent, "children", []):
                if isinstance(child, NavigableString):
                    text_value = str(child)
                    if text_value.strip():
                        found.append(text_value)
                    continue
                if isinstance(child, Tag):
                    if child.name and child.name.lower() in block_names:
                        found.append(str(child))
                    elif str(child).strip():
                        found.append(str(child))
            return found

        units = child_units(container)
        if len(units) == 1:
            only = BeautifulSoup(units[0], "html.parser")
            only_tags = [child for child in only.contents if isinstance(child, Tag)]
            if len(only_tags) == 1:
                nested_units = child_units(only_tags[0])
                if len(nested_units) > 1:
                    units = nested_units
        return units or [raw]
    except Exception:
        return _split_html_units_regex(raw)


def _split_html_units_regex(text: str) -> List[str]:
    raw = str(text or "")
    if not raw:
        return []
    html_pattern = re.compile(
        r".*?</(?:p|div|h[1-6]|li|tr|table|blockquote)>\s*|<img\b[^>]*>\s*|<br\s*/?>\s*|.+$",
        re.IGNORECASE | re.DOTALL,
    )
    if re.search(r"</(?:p|div|h[1-6]|li|tr|table|blockquote)>|<img\b|<br\b", raw, re.IGNORECASE):
        return [match.group(0) for match in html_pattern.finditer(raw) if match.group(0)] or [raw]
    return _split_plain_text_units(raw)


def _split_text_units(text: str, *, payload_format: Optional[str] = None) -> List[str]:
    raw = str(text or "")
    if not raw:
        return []
    if (payload_format or _payload_format_for_split(raw)) == "html":
        return _split_html_units_beautifulsoup(raw)
    return _split_plain_text_units(raw)


def _avoid_split_inside_tag(raw: str, end: int) -> int:
    if end <= 0 or end >= len(raw):
        return end
    last_lt = raw.rfind("<", 0, end)
    last_gt = raw.rfind(">", 0, end)
    if last_lt <= last_gt:
        return end
    next_gt = raw.find(">", end)
    if next_gt >= 0 and next_gt - end <= 80:
        return min(len(raw), next_gt + 1)
    if last_lt > 0:
        return last_lt
    return end


def _split_large_unit(unit: str, max_chars: int) -> List[str]:
    raw = str(unit or "")
    if len(raw) <= max_chars:
        return [raw]
    pieces: List[str] = []
    start = 0
    while start < len(raw):
        end = min(len(raw), start + max_chars)
        if end < len(raw):
            window = raw[start:end]
            split_at = max(
                window.rfind("\n"),
                window.rfind(". "),
                window.rfind("。"),
                window.rfind("! "),
                window.rfind("? "),
                window.rfind("</"),
            )
            if split_at > max_chars * 0.45:
                end = start + split_at + 1
        end = _avoid_split_inside_tag(raw, end)
        if end <= start:
            end = min(len(raw), start + max_chars)
        pieces.append(raw[start:end])
        start = end
    return pieces


def _split_messages_for_search_budget(
    messages: Iterable[Dict[str, Any]],
    max_prompt_chars: int,
    *,
    return_metadata: bool = False,
) -> Any:
    source_messages = [dict(message) for message in (messages or []) if isinstance(message, dict)]
    if not source_messages:
        return ([], {}) if return_metadata else []
    user_indices = [
        idx for idx, message in enumerate(source_messages)
        if str(message.get("role") or "user").strip().lower() not in ("system", "assistant")
    ]
    if not user_indices:
        return ([source_messages], {}) if return_metadata else [source_messages]

    split_idx = user_indices[-1]
    original_user = source_messages[split_idx]
    user_text = _content_to_text(original_user.get("content"))
    prefix, body = _split_user_instruction_prefix(user_text)
    if not body.strip():
        return ([source_messages], {}) if return_metadata else [source_messages]

    payload_format = _payload_format_for_split(body)
    splitter_name = _splitter_name_for_payload(payload_format)
    fixed_messages = [dict(message) for idx, message in enumerate(source_messages) if idx != split_idx]
    def make_messages(chunk_text: str) -> List[Dict[str, Any]]:
        content_parts = [part for part in (prefix, chunk_text.strip()) if part]
        chunk_user = dict(original_user)
        chunk_user["content"] = "\n".join(content_parts)
        return fixed_messages + [chunk_user]

    def fits_budget(chunk_text: str) -> bool:
        probe_messages = make_messages(chunk_text)
        return (
            len(_messages_to_prompt(probe_messages)) <= prompt_limit_chars
            and _messages_to_search_url_chars(probe_messages) <= url_limit_chars
        )

    units: List[str] = []
    fixed_probe = "x"
    fixed_prompt_chars = max(0, len(_messages_to_prompt(make_messages(fixed_probe))) - len(fixed_probe))
    fixed_url_chars = max(0, _messages_to_search_url_chars(make_messages(fixed_probe)) - len(fixed_probe))
    safety_chars = _subchunk_safety_chars()
    url_limit_chars = _subchunk_url_chars()
    available_body_chars = max_prompt_chars - fixed_prompt_chars - safety_chars
    min_body_chars = _min_subchunk_body_chars()
    if available_body_chars <= 0:
        body_budget = min_body_chars
    else:
        body_budget = max(1, available_body_chars)
    prompt_limit_chars = max(1, max_prompt_chars - safety_chars)
    if fixed_prompt_chars + body_budget > prompt_limit_chars:
        prompt_limit_chars = fixed_prompt_chars + body_budget

    def split_piece_to_budget(piece: str) -> List[str]:
        raw_piece = str(piece or "")
        if not raw_piece or fits_budget(raw_piece):
            return [raw_piece] if raw_piece else []
        if len(raw_piece) <= 1:
            return [raw_piece]

        split_at = len(raw_piece) // 2
        window = raw_piece[:split_at]
        boundary = max(window.rfind("\n"), window.rfind(" "), window.rfind(">"))
        if boundary > max(1, split_at // 3):
            split_at = boundary + 1
        split_at = _avoid_split_inside_tag(raw_piece, split_at)
        if split_at <= 0 or split_at >= len(raw_piece):
            split_at = len(raw_piece) // 2
        left = raw_piece[:split_at]
        right = raw_piece[split_at:]
        return [*split_piece_to_budget(left), *split_piece_to_budget(right)]

    for unit in _split_text_units(body, payload_format=payload_format):
        if payload_format == "html":
            units.append(unit)
            continue
        for piece in _split_large_unit(unit, body_budget):
            units.extend(split_piece_to_budget(piece))

    chunks: List[str] = []
    current = ""
    for unit in units:
        candidate = current + unit
        if current and not fits_budget(candidate):
            chunks.append(current)
            current = unit
        else:
            current = candidate
    if current:
        chunks.append(current)

    result = [source_messages] if len(chunks) <= 1 else [make_messages(chunk) for chunk in chunks]
    metadata = {
        "target_prompt_chars": max_prompt_chars,
        "fixed_prompt_chars": fixed_prompt_chars,
        "safety_chars": safety_chars,
        "prompt_limit_chars": prompt_limit_chars,
        "url_limit_chars": url_limit_chars,
        "available_body_chars": available_body_chars,
        "body_budget_chars": body_budget,
        "fixed_url_chars": fixed_url_chars,
        "body_chars": len(body),
        "prefix_chars": len(prefix),
        "subchunk_count": len(result),
        "payload_format": payload_format,
        "splitter": splitter_name,
    }
    return (result, metadata) if return_metadata else result


def _strip_search_prefix(model: str) -> str:
    raw = str(model or "").strip()
    match = re.match(r"^search\d{0,4}/", raw, flags=re.IGNORECASE)
    if match:
        raw = raw[match.end():]
    elif raw.lower().startswith("search"):
        raw = raw[len("search"):].lstrip("/")
    return raw.strip("/") or DEFAULT_MODEL


def _search_params() -> Dict[str, str]:
    params = dict(DEFAULT_SEARCH_PARAMS)
    extra = os.getenv("GEMINI_FREE_SEARCH_PARAMS", "").strip()
    if extra:
        for key, value in parse_qsl(extra.lstrip("?"), keep_blank_values=True):
            if key:
                params[key] = value
    return params


def _build_search_base_url() -> str:
    base = os.getenv("GEMINI_FREE_SEARCH_BASE_URL", SEARCH_BASE_URL).strip() or SEARCH_BASE_URL
    params = _search_params()
    params.pop("q", None)
    return f"{base}?{urlencode(params)}"


def _build_search_url(prompt: str) -> str:
    base = os.getenv("GEMINI_FREE_SEARCH_BASE_URL", SEARCH_BASE_URL).strip() or SEARCH_BASE_URL
    params = _search_params()
    params["q"] = prompt
    return f"{base}?{urlencode(params)}"


def _google_blocked(page_data: Dict[str, Any]) -> bool:
    url = str(page_data.get("url") or "").lower()
    text = str(page_data.get("text") or "")
    text_l = text.lower()
    return (
        "google.com/sorry/" in url
        or "our systems have detected unusual traffic" in text_l
        or ("about this page" in text_l and "unusual traffic" in text_l)
        or "to continue, please type the characters" in text_l
    )


def _contains_generation_failure(lines: Iterable[str]) -> bool:
    for line in lines or []:
        current = str(line or "").strip().lower()
        if any(marker in current for marker in GENERATION_FAILURE_MARKERS):
            return True
    return False


def _messages_expect_html_response(messages: Iterable[Dict[str, Any]]) -> bool:
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "user").strip().lower() or "user"
        if role in ("system", "assistant"):
            continue
        if _payload_format_for_split(_content_to_text(message.get("content"))) == "html":
            return True
    return False


def _html_has_block_structure(value: str) -> bool:
    return bool(re.search(r"<(?:html|body|article|section|main|div|p|h[1-6]|ul|ol|li|table|tr|blockquote|br)\b|</(?:p|div|h[1-6]|li|tr|blockquote)>", str(value or ""), re.IGNORECASE))


def _normalize_visible_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _html_visible_text(value: str) -> str:
    raw = str(value or "")
    if not raw:
        return ""
    try:
        from bs4 import BeautifulSoup

        return BeautifulSoup(raw, "html.parser").get_text(" ")
    except Exception:
        return re.sub(r"<[^>]+>", " ", raw)


def _answer_html_matches_text(answer_html: str, answer_text: str) -> bool:
    if not _html_has_block_structure(answer_html):
        return False
    html_text = _normalize_visible_text(_html_visible_text(answer_html))
    expected_text = _normalize_visible_text(answer_text)
    if not html_text:
        return False
    ui_markers = (
        "copy share public link",
        "good response",
        "bad response",
        "thanks for letting us know",
        "google may use account",
        "ai can make mistakes",
    )
    html_text_l = html_text.lower()
    if any(marker in html_text_l for marker in ui_markers):
        return False
    if expected_text:
        expected_l = expected_text.lower()
        if expected_l not in html_text_l and html_text_l not in expected_l:
            return False
        if len(html_text) > max(250, len(expected_text) * 3):
            return False
    return True


def _clean_answer_html(answer_html: str) -> str:
    raw = str(answer_html or "").strip()
    if not raw:
        return ""
    if "&lt;" in raw and not _html_has_block_structure(raw):
        raw = html.unescape(raw)
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "button", "svg", "form", "input", "textarea", "noscript"]):
            tag.decompose()
        for tag in soup.find_all(True):
            for attr in list(tag.attrs):
                if attr not in ("href", "src", "alt", "title", "colspan", "rowspan"):
                    del tag.attrs[attr]
        container = soup.body if soup.body else soup
        cleaned = "".join(str(child) for child in container.children if str(child).strip()).strip()
        return cleaned
    except Exception:
        return raw


def _extract_rendered_content(
    page_data: Dict[str, Any],
    *,
    prompt: str = "",
    prefer_html: bool = False,
) -> str:
    if _google_blocked(page_data):
        raise RuntimeError(
            "Google Search returned a verification page instead of a Gemini/Search response. "
            "Open Google in a regular browser/profile and clear the verification, or retry later."
        )
    if prefer_html:
        answer_text = str(page_data.get("answerText") or "").strip()
        html_content = _clean_answer_html(str(page_data.get("answerHtml") or ""))
        if _answer_html_matches_text(html_content, answer_text):
            failure_probe = html_content
            try:
                from bs4 import BeautifulSoup

                failure_probe = BeautifulSoup(html_content, "html.parser").get_text("\n")
            except Exception:
                pass
            if _contains_generation_failure(failure_probe.splitlines()):
                raise RuntimeError(
                    "Google Search AI Mode returned a generation failure: "
                    "Something went wrong and the content wasn't generated."
                )
            return html_content
        if _html_has_block_structure(answer_text):
            return answer_text
        if answer_text:
            raise RuntimeError(
                "Google Search AI Mode returned plain text for an HTML request; "
                "refusing to accept a response that would destroy BeautifulSoup HTML structure."
            )

    text = str(page_data.get("text") or "")
    lines = [line.strip() for line in text.replace("\r", "\n").split("\n")]
    lines = [line for line in lines if line]
    content_lines = _extract_ai_answer_lines(lines, prompt)
    if _contains_generation_failure(content_lines or lines):
        raise RuntimeError(
            "Google Search AI Mode returned a generation failure: "
            "Something went wrong and the content wasn't generated."
        )
    content = "\n".join(content_lines or lines).strip()
    if not content:
        title = str(page_data.get("title") or "").strip()
        raise RuntimeError(f"Google Search returned an empty rendered page. title={title!r}")
    if prefer_html and not _html_has_block_structure(content):
        raise RuntimeError(
            "Google Search AI Mode returned plain text for an HTML request; "
            "refusing to accept a response that would destroy BeautifulSoup HTML structure."
        )
    return content


def _extract_ai_answer_lines(lines: List[str], prompt: str = "") -> List[str]:
    """Best-effort cleanup for Google AI mode rendered page text."""
    if not lines:
        return []

    start = 0
    for idx, line in enumerate(lines):
        if line.strip().lower() == "you said:":
            start = idx + 1
            break
    else:
        return []

    prompt_lines = {
        line.strip()
        for line in str(prompt or "").replace("\r", "\n").split("\n")
        if line.strip()
    }
    role_markers = {"user:", "assistant:", "system instructions:"}
    while start < len(lines):
        current = lines[start].strip()
        current_l = current.lower()
        if not current:
            start += 1
            continue
        if current_l in role_markers or current in prompt_lines:
            start += 1
            continue
        break

    end = len(lines)
    end_markers = (
        "ai can make mistakes",
        "ai mode response is ready",
        "search results",
        "people also ask",
        "related searches",
    )
    for idx in range(start, len(lines)):
        current_l = lines[idx].strip().lower()
        if any(marker in current_l for marker in end_markers):
            end = idx
            break
        if re.match(r"^\d+\s+sites?$", current_l):
            end = idx
            break

    extracted = []
    for line in lines[start:end]:
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^\+\d+$", stripped):
            continue
        extracted.append(stripped)
    return extracted


def _setup_qt_environment() -> None:
    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = _qtwebengine_chromium_flags(
        os.environ.get("QTWEBENGINE_CHROMIUM_FLAGS", "")
    )
    os.environ.setdefault("QTWEBENGINE_DISABLE_SANDBOX", "1")


def _create_profile(app: Any, *, user_agent: Optional[str] = None) -> Any:
    from PySide6.QtWebEngineCore import QWebEngineProfile

    profile_root = Path.home() / ".glossarion" / "gemini_free_browser"
    if os.getenv("GEMINI_FREE_EPHEMERAL_PROFILE", "0").lower() in ("1", "true", "yes"):
        profile_root = profile_root / uuid.uuid4().hex
    profile_root.mkdir(parents=True, exist_ok=True)

    profile = QWebEngineProfile(f"gemini-free-{uuid.uuid4().hex}", app)
    profile.setHttpUserAgent(user_agent or _default_user_agent())
    try:
        profile.setPersistentStoragePath(str(profile_root))
        profile.setCachePath(str(profile_root / "cache"))
    except Exception:
        pass
    try:
        profile.setHttpAcceptLanguage(os.getenv("GEMINI_FREE_ACCEPT_LANGUAGE", "en-US,en;q=0.9"))
    except Exception:
        pass
    return profile


def _set_default_page_viewport(page: Any) -> None:
    try:
        from PySide6.QtCore import QSize

        width = max(320, _env_int("GEMINI_FREE_VIEWPORT_WIDTH", 1280))
        height = max(480, _env_int("GEMINI_FREE_VIEWPORT_HEIGHT", 900))
        page.setViewportSize(QSize(width, height))
    except Exception:
        pass


def run_qtwebengine_request(
    *,
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    body: Optional[str] = None,
    bootstrap_url: Optional[str] = None,
    credentials: str = "include",
    mode: str = "same-origin",
    timeout: int = 60,
    user_agent: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a browser-side fetch() from a Qt WebEngine page."""
    _setup_qt_environment()

    from PySide6.QtCore import QEventLoop, QTimer, QUrl
    from PySide6.QtWebEngineCore import QWebEnginePage
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication(["gemini-free-request"])

    profile = _create_profile(app, user_agent=user_agent)
    page = QWebEnginePage(profile, app)
    _set_default_page_viewport(page)

    def run_js(script: str, js_timeout_ms: int = 15000) -> Any:
        holder: Dict[str, Any] = {}
        loop = QEventLoop()

        def _done(value: Any) -> None:
            holder["value"] = value
            loop.quit()

        page.runJavaScript(script, _done)
        QTimer.singleShot(js_timeout_ms, loop.quit)
        loop.exec()
        return holder.get("value")

    try:
        load_loop = QEventLoop()
        load_state: Dict[str, Any] = {"ok": False, "error": "", "last_event": ""}

        def _loaded(ok: bool) -> None:
            load_state["ok"] = bool(ok)
            load_loop.quit()

        page.loadFinished.connect(_loaded)
        page.load(QUrl(bootstrap_url or _origin_url(url)))
        QTimer.singleShot(max(1000, int(timeout * 1000)), load_loop.quit)
        load_loop.exec()
        if not load_state.get("ok"):
            raise RuntimeError(f"Qt WebEngine failed to load bootstrap page {bootstrap_url}")

        payload = {
            "url": url,
            "method": method.upper(),
            "headers": headers or {},
            "body": body,
            "credentials": credentials,
            "mode": mode,
            "timeoutMs": max(1000, int(timeout * 1000)),
        }
        script = f"""
(() => {{
  const request = {json.dumps(payload, ensure_ascii=False)};
  window.__geminiFreeRequestResult = {{pending: true}};
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), request.timeoutMs);
  (async () => {{
    try {{
      const init = {{
        method: request.method,
        credentials: request.credentials,
        redirect: "follow",
        headers: request.headers || {{}},
        signal: controller.signal
      }};
      if (request.mode) init.mode = request.mode;
      if (!["GET", "HEAD"].includes(request.method) && request.body !== null && request.body !== undefined) {{
        init.body = String(request.body);
      }}
      const response = await fetch(request.url, init);
      const responseHeaders = {{}};
      try {{
        response.headers.forEach((value, key) => {{ responseHeaders[key] = value; }});
      }} catch (error) {{}}
      const text = await response.text();
      window.__geminiFreeRequestResult = {{
        pending: false,
        ok: response.ok,
        status: response.status,
        statusText: response.statusText,
        url: response.url,
        redirected: response.redirected,
        type: response.type,
        headers: responseHeaders,
        body: text,
        bodyLength: text.length,
        truncated: false,
        error: null
      }};
    }} catch (error) {{
      window.__geminiFreeRequestResult = {{
        pending: false,
        ok: false,
        status: 0,
        statusText: "Qt WebEngine fetch failed",
        url: request.url,
        redirected: false,
        type: "",
        headers: {{}},
        body: "",
        bodyLength: 0,
        truncated: false,
        error: String(error && (error.stack || error.message || error))
      }};
    }} finally {{
      clearTimeout(timer);
    }}
  }})();
  return true;
}})();
"""
        run_js(script, js_timeout_ms=10000)

        deadline = time.time() + max(1, int(timeout))
        last_result: Dict[str, Any] = {}
        while time.time() < deadline:
            if _is_cancelled():
                raise RuntimeError("stream cancelled")
            raw = run_js("JSON.stringify(window.__geminiFreeRequestResult || {pending:true})", js_timeout_ms=5000)
            try:
                result = json.loads(raw or "{}")
            except Exception:
                result = {}
            last_result = result
            if result and not result.get("pending", True):
                result["bootstrap_url"] = bootstrap_url
                return result

            wait_loop = QEventLoop()
            QTimer.singleShot(100, wait_loop.quit)
            wait_loop.exec()

        raise RuntimeError(f"Qt WebEngine fetch timed out: {last_result}")
    finally:
        try:
            page.deleteLater()
            profile.deleteLater()
            cleanup_loop = QEventLoop()
            QTimer.singleShot(100, cleanup_loop.quit)
            cleanup_loop.exec()
        except Exception:
            pass


def load_rendered_page_text(
    url: str,
    *,
    prompt: str = "",
    timeout: int = DEFAULT_TIMEOUT,
    wait_after_load_ms: Optional[int] = None,
    user_agent: Optional[str] = None,
) -> Dict[str, Any]:
    """Load a page in Qt WebEngine and return rendered body text."""
    _setup_qt_environment()

    from PySide6.QtCore import QEventLoop, QTimer, QUrl
    from PySide6.QtWebEngineCore import QWebEnginePage
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication(["gemini-free-page"])

    profile = _create_profile(app, user_agent=user_agent)
    page = QWebEnginePage(profile, app)
    _set_default_page_viewport(page)

    def run_js(script: str, js_timeout_ms: int = 15000) -> Any:
        holder: Dict[str, Any] = {}
        loop = QEventLoop()

        def _done(value: Any) -> None:
            holder["value"] = value
            loop.quit()

        page.runJavaScript(script, _done)
        QTimer.singleShot(js_timeout_ms, loop.quit)
        loop.exec()
        return holder.get("value")

    try:
        load_loop = QEventLoop()
        load_state: Dict[str, Any] = {"ok": False}

        def _loaded(ok: bool) -> None:
            load_state["ok"] = bool(ok)
            load_loop.quit()

        page.loadFinished.connect(_loaded)
        page.load(QUrl(url))
        QTimer.singleShot(max(1000, int(timeout * 1000)), load_loop.quit)
        load_loop.exec()
        if not load_state.get("ok"):
            raise RuntimeError(f"Qt WebEngine failed to load {url}")

        wait_ms = wait_after_load_ms
        if wait_ms is None:
            wait_ms = _env_int("GEMINI_FREE_WAIT_AFTER_LOAD_MS", 12000)
        deadline = time.time() + max(1, int(timeout))
        min_ready_at = time.time() + max(0, int(wait_ms)) / 1000.0
        last_result: Dict[str, Any] = {}

        while time.time() < deadline:
            if _is_cancelled():
                raise RuntimeError("stream cancelled")
            raw = run_js(_page_snapshot_script(prompt), js_timeout_ms=5000)
            try:
                result = json.loads(raw or "{}")
            except Exception:
                result = {}
            last_result = result
            text = str(result.get("text") or "")
            if _google_blocked(result):
                return result
            if text.strip() and time.time() >= min_ready_at:
                return result

            wait_loop = QEventLoop()
            QTimer.singleShot(500, wait_loop.quit)
            wait_loop.exec()

        return last_result
    finally:
        try:
            page.deleteLater()
            profile.deleteLater()
            cleanup_loop = QEventLoop()
            QTimer.singleShot(100, cleanup_loop.quit)
            cleanup_loop.exec()
        except Exception:
            pass


def load_ai_mode_prompt_text(
    prompt: str,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    wait_after_load_ms: Optional[int] = None,
    user_agent: Optional[str] = None,
) -> Dict[str, Any]:
    """Open AI Mode, submit the prompt through the page UI, and return rendered text."""
    _setup_qt_environment()

    from PySide6.QtCore import QEventLoop, QTimer, QUrl
    from PySide6.QtWebEngineCore import QWebEnginePage
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication(["gemini-free-ai-mode"])

    profile = _create_profile(app, user_agent=user_agent)
    page = QWebEnginePage(profile, app)
    _set_default_page_viewport(page)
    base_url = _build_search_base_url()

    def run_js(script: str, js_timeout_ms: int = 15000) -> Any:
        holder: Dict[str, Any] = {}
        loop = QEventLoop()

        def _done(value: Any) -> None:
            holder["value"] = value
            loop.quit()

        page.runJavaScript(script, _done)
        QTimer.singleShot(js_timeout_ms, loop.quit)
        loop.exec()
        return holder.get("value")

    def run_json(script: str, js_timeout_ms: int = 15000) -> Dict[str, Any]:
        raw = run_js(script, js_timeout_ms=js_timeout_ms)
        try:
            return json.loads(raw or "{}")
        except Exception:
            return {}

    def wait_ms(ms: int) -> None:
        wait_loop = QEventLoop()
        QTimer.singleShot(max(1, int(ms)), wait_loop.quit)
        wait_loop.exec()

    try:
        load_loop = QEventLoop()
        load_state: Dict[str, Any] = {"ok": False}

        def _loaded(ok: bool) -> None:
            load_state["ok"] = bool(ok)
            load_loop.quit()

        page.loadFinished.connect(_loaded)
        page.load(QUrl(base_url))
        QTimer.singleShot(max(1000, int(timeout * 1000)), load_loop.quit)
        load_loop.exec()
        if not load_state.get("ok"):
            raise RuntimeError(f"Qt WebEngine failed to load {base_url}")

        set_prompt_script = f"""
(() => {{
  const promptText = {json.dumps(prompt, ensure_ascii=False)};
  const visibleScore = (element) => {{
    if (!element) return 0;
    const style = window.getComputedStyle(element);
    const rect = element.getBoundingClientRect();
    if (style.visibility === "hidden" || style.display === "none") return 0;
    if (rect.width <= 0 || rect.height <= 0) return 0;
    return rect.width * rect.height;
  }};
  const textareas = Array.from(document.querySelectorAll("textarea"));
  const ranked = textareas
    .map((element) => ({{element, score: visibleScore(element)}}))
    .sort((a, b) => b.score - a.score);
  const textarea =
    ranked.find((item) => item.score > 0 && /ask/i.test(item.element.placeholder || ""))?.element ||
    ranked.find((item) => item.score > 0)?.element ||
    textareas.find((element) => /ask/i.test(element.placeholder || "")) ||
    textareas[0];
  if (!textarea) {{
    return JSON.stringify({{ok: false, error: "AI Mode textarea not found", textareaCount: textareas.length}});
  }}
  textarea.scrollIntoView({{block: "center", inline: "center"}});
  textarea.focus();
  const valueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value")?.set;
  if (valueSetter) {{
    valueSetter.call(textarea, promptText);
  }} else {{
    textarea.value = promptText;
  }}
  textarea.dispatchEvent(new Event("focus", {{bubbles: true}}));
  try {{
    textarea.dispatchEvent(new InputEvent("input", {{
      bubbles: true,
      inputType: "insertText",
      data: promptText
    }}));
  }} catch (error) {{
    textarea.dispatchEvent(new Event("input", {{bubbles: true}}));
  }}
  textarea.dispatchEvent(new Event("change", {{bubbles: true}}));
  return JSON.stringify({{
    ok: true,
    textareaCount: textareas.length,
    valueLength: textarea.value.length,
    placeholder: textarea.placeholder || ""
  }});
}})()
"""
        set_state = run_json(set_prompt_script, js_timeout_ms=15000)
        if not set_state.get("ok"):
            detail = str(set_state.get("error") or "AI Mode textarea not found")
            raise RuntimeError(detail)

        click_send_script = """
(() => {
  const visibleScore = (element) => {
    if (!element) return 0;
    const style = window.getComputedStyle(element);
    const rect = element.getBoundingClientRect();
    if (style.visibility === "hidden" || style.display === "none") return 0;
    if (rect.width <= 0 || rect.height <= 0) return 0;
    return rect.width * rect.height;
  };
  const labelOf = (element) => [
    element.getAttribute("aria-label") || "",
    element.getAttribute("title") || "",
    element.innerText || "",
    element.value || ""
  ].join(" ").replace(/\\s+/g, " ").trim();
  const buttons = Array.from(document.querySelectorAll('button,[role="button"],input[type="submit"]'));
  const ranked = buttons
    .map((element) => ({element, label: labelOf(element), score: visibleScore(element)}))
    .sort((a, b) => b.score - a.score);
  const exactSend = (item) => /^send$/i.test(item.element.getAttribute("aria-label") || "");
  const enabled = (item) => !item.element.disabled && item.element.getAttribute("aria-disabled") !== "true";
  const candidate =
    ranked.find((item) => exactSend(item) && enabled(item)) ||
    ranked.find((item) => /\\bsend\\b/i.test(item.label) && enabled(item)) ||
    ranked.find((item) => exactSend(item));
  if (!candidate) {
    return JSON.stringify({
      ok: false,
      error: "AI Mode send button not ready",
      buttonCount: buttons.length,
      labels: ranked.slice(0, 8).map((item) => item.label)
    });
  }
  candidate.element.click();
  return JSON.stringify({ok: true, label: candidate.label, buttonCount: buttons.length});
})()
"""
        click_state: Dict[str, Any] = {}
        click_deadline = time.time() + min(10, max(2, int(timeout)))
        while time.time() < click_deadline:
            if _is_cancelled():
                raise RuntimeError("stream cancelled")
            wait_ms(250)
            click_state = run_json(click_send_script, js_timeout_ms=10000)
            if click_state.get("ok"):
                break
        if not click_state.get("ok"):
            detail = str(click_state.get("error") or "AI Mode send button not ready")
            raise RuntimeError(detail)

        wait_after_ms = wait_after_load_ms
        if wait_after_ms is None:
            wait_after_ms = _env_int("GEMINI_FREE_WAIT_AFTER_LOAD_MS", 12000)
        stable_ms = _env_int("GEMINI_FREE_STABLE_MS", 2500)
        deadline = time.time() + max(1, int(timeout))
        min_ready_at = time.time() + max(0, int(wait_after_ms)) / 1000.0
        last_result: Dict[str, Any] = {}
        last_answer = ""
        last_change_at = time.time()

        while time.time() < deadline:
            if _is_cancelled():
                raise RuntimeError("stream cancelled")
            result = run_json(_page_snapshot_script(prompt), js_timeout_ms=5000)
            result["submit_mode"] = "ui"
            result["search_base_url"] = base_url
            result["submit_state"] = {"set": set_state, "click": click_state}
            last_result = result
            if _google_blocked(result):
                return result

            text = str(result.get("text") or "")
            lines = [line.strip() for line in text.replace("\r", "\n").split("\n") if line.strip()]
            answer = "\n".join(_extract_ai_answer_lines(lines, prompt)).strip()
            if _contains_generation_failure(answer.splitlines() or lines):
                return result
            if answer and answer != last_answer:
                last_answer = answer
                last_change_at = time.time()
            if (
                answer
                and time.time() >= min_ready_at
                and not result.get("busy")
                and (time.time() - last_change_at) * 1000 >= max(0, int(stable_ms))
            ):
                return result

            wait_ms(500)

        title = str(last_result.get("title") or "").strip()
        url = _safe_url_for_log(last_result.get("url") or base_url)
        raise RuntimeError(f"Google Search AI Mode did not produce a response after UI submit. title={title!r} url={url}")
    finally:
        try:
            page.deleteLater()
            profile.deleteLater()
            cleanup_loop = QEventLoop()
            QTimer.singleShot(100, cleanup_loop.quit)
            cleanup_loop.exec()
        except Exception:
            pass


def _send_chat_completion_qt_once(
    *,
    messages: Iterable[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
    timeout: int = DEFAULT_TIMEOUT,
    max_tokens: Optional[int] = None,
    log_fn: Optional[Callable[[str], None]] = None,
    user_agent: Optional[str] = None,
) -> Dict[str, Any]:
    actual_model = _strip_search_prefix(model)
    if actual_model.lower() != DEFAULT_MODEL:
        raise RuntimeError(f"Gemini Free only supports search/{DEFAULT_MODEL}; got search/{actual_model}")

    prompt = _messages_to_prompt(messages)
    if not prompt:
        raise RuntimeError("Gemini Free request has an empty prompt")
    prefer_html = _messages_expect_html_response(messages)

    submit_mode = os.getenv("GEMINI_FREE_SUBMIT_MODE", DEFAULT_SUBMIT_MODE).strip().lower() or DEFAULT_SUBMIT_MODE
    if submit_mode in ("url", "query", "q"):
        search_url = _build_search_url(prompt)
        _log(log_fn, f"Gemini Free: opening Google Search browser route (model={actual_model}, mode=url)")
        _log(log_fn, f"Gemini Free debug URL: {search_url[:1000]}", debug_only=True)
        page_data = load_rendered_page_text(
            search_url,
            prompt=prompt,
            timeout=timeout,
            user_agent=user_agent,
        )
    else:
        search_url = _build_search_base_url()
        _log(log_fn, f"Gemini Free: opening Google Search browser route (model={actual_model}, mode=ui)")
        _log(log_fn, f"Gemini Free debug URL: {search_url[:1000]}", debug_only=True)
        page_data = load_ai_mode_prompt_text(
            prompt,
            timeout=timeout,
            user_agent=user_agent,
        )
    content = _extract_rendered_content(page_data, prompt=prompt, prefer_html=prefer_html)
    return {
        "content": content,
        "finish_reason": "stop",
        "usage": {
            "prompt_chars": len(prompt),
            "completion_chars": len(content),
        },
        "raw_response": {
            "model": actual_model,
            "submit_mode": page_data.get("submit_mode") or submit_mode,
            "search_url": _safe_url_for_log(search_url),
            "page_url": _safe_url_for_log(page_data.get("url")),
            "page_url_chars": len(str(page_data.get("url") or "")),
            "title": page_data.get("title"),
            "ready": page_data.get("ready"),
            "html_length": page_data.get("htmlLength"),
            "answer_html_length": len(str(page_data.get("answerHtml") or "")),
            "prefer_html_response": prefer_html,
        },
    }


def _send_chat_completion_split(
    *,
    messages: Iterable[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
    timeout: int = DEFAULT_TIMEOUT,
    max_tokens: Optional[int] = None,
    log_fn: Optional[Callable[[str], None]] = None,
    user_agent: Optional[str] = None,
) -> Dict[str, Any]:
    source_messages = [dict(message) for message in (messages or []) if isinstance(message, dict)]
    max_prompt_chars = _subchunk_prompt_chars()
    chunks, split_metadata = _split_messages_for_search_budget(
        source_messages,
        max_prompt_chars,
        return_metadata=True,
    )
    if len(chunks) <= 1:
        return _send_chat_completion_qt_once(
            messages=source_messages,
            model=model,
            timeout=timeout,
            max_tokens=max_tokens,
            log_fn=log_fn,
            user_agent=user_agent,
        )

    _log(
        log_fn,
        f"🧩 Gemini Free: adaptive subchunking {len(chunks)} browser requests "
        f"(target prompt chars: {max_prompt_chars}, "
        f"payload format: {split_metadata.get('payload_format')}, "
        f"splitter: {split_metadata.get('splitter')}, "
        f"limit chars: {split_metadata.get('prompt_limit_chars')}, "
        f"url limit chars: {split_metadata.get('url_limit_chars')}, "
        f"fixed prompt chars: {split_metadata.get('fixed_prompt_chars')}, "
        f"fixed url chars: {split_metadata.get('fixed_url_chars')}, "
        f"body budget chars: {split_metadata.get('body_budget_chars')})"
    )
    contents: List[str] = []
    raw_parts: List[Dict[str, Any]] = []
    prompt_chars_total = 0
    completion_chars_total = 0
    for index, chunk_messages in enumerate(chunks, start=1):
        if _is_cancelled():
            raise RuntimeError("stream cancelled")
        chunk_prompt_chars = len(_messages_to_prompt(chunk_messages))
        prompt_chars_total += chunk_prompt_chars
        _log(log_fn, f"🧩 Gemini Free: subchunk {index}/{len(chunks)} ({chunk_prompt_chars:,} prompt chars)")
        result = _send_chat_completion_qt_once(
            messages=chunk_messages,
            model=model,
            timeout=timeout,
            max_tokens=max_tokens,
            log_fn=log_fn,
            user_agent=user_agent,
        )
        content = str(result.get("content") or "")
        contents.append(content)
        completion_chars_total += len(content)
        raw_response = dict(result.get("raw_response") or {})
        raw_response.pop("content", None)
        raw_parts.append({
            "index": index,
            "prompt_chars": chunk_prompt_chars,
            "completion_chars": len(content),
            "raw_response": raw_response,
        })

    combined = "\n".join(part for part in contents if part).strip()
    return {
        "content": combined,
        "finish_reason": "stop",
        "usage": {
            "prompt_chars": len(_messages_to_prompt(source_messages)),
            "subchunk_prompt_chars": prompt_chars_total,
            "completion_chars": completion_chars_total,
        },
        "raw_response": {
            "model": _strip_search_prefix(model),
            "submit_mode": "adaptive_split",
            "subchunk_count": len(chunks),
            "subchunk_prompt_target_chars": max_prompt_chars,
            "subchunk_fixed_prompt_chars": split_metadata.get("fixed_prompt_chars"),
            "subchunk_fixed_url_chars": split_metadata.get("fixed_url_chars"),
            "subchunk_prompt_limit_chars": split_metadata.get("prompt_limit_chars"),
            "subchunk_url_limit_chars": split_metadata.get("url_limit_chars"),
            "subchunk_body_budget_chars": split_metadata.get("body_budget_chars"),
            "subchunk_safety_chars": split_metadata.get("safety_chars"),
            "subchunk_payload_format": split_metadata.get("payload_format"),
            "subchunk_splitter": split_metadata.get("splitter"),
            "parts": raw_parts,
        },
    }


def _send_chat_completion_qt(
    *,
    messages: Iterable[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
    timeout: int = DEFAULT_TIMEOUT,
    max_tokens: Optional[int] = None,
    log_fn: Optional[Callable[[str], None]] = None,
    user_agent: Optional[str] = None,
) -> Dict[str, Any]:
    source_messages = [dict(message) for message in (messages or []) if isinstance(message, dict)]
    prompt_chars = len(_messages_to_prompt(source_messages))
    if _adaptive_split_enabled() and prompt_chars > _subchunk_prompt_chars():
        return _send_chat_completion_split(
            messages=source_messages,
            model=model,
            timeout=timeout,
            max_tokens=max_tokens,
            log_fn=log_fn,
            user_agent=user_agent,
        )
    try:
        return _send_chat_completion_qt_once(
            messages=source_messages,
            model=model,
            timeout=timeout,
            max_tokens=max_tokens,
            log_fn=log_fn,
            user_agent=user_agent,
        )
    except RuntimeError as exc:
        if _adaptive_split_enabled() and _generation_failure_error(exc):
            return _send_chat_completion_split(
                messages=source_messages,
                model=model,
                timeout=timeout,
                max_tokens=max_tokens,
                log_fn=log_fn,
                user_agent=user_agent,
            )
        raise


def _is_frozen_app() -> bool:
    return bool(getattr(sys, "frozen", False))


def _extract_json_from_process(stdout: str) -> Dict[str, Any]:
    for line in reversed((stdout or "").splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise RuntimeError("Gemini Free helper did not return JSON")


def _run_search_subprocess_once(
    *,
    messages: Iterable[Dict[str, Any]],
    model: str,
    timeout: int,
    max_tokens: Optional[int],
    log_fn: Optional[Callable[[str], None]] = None,
    wait_log_event: Optional[threading.Event] = None,
) -> Dict[str, Any]:
    message_list = list(messages or [])
    helper_timeout = max(30, int(timeout))
    temp_dir = tempfile.mkdtemp(prefix="glossarion_gemini_free_")
    prompt_path = os.path.join(temp_dir, "messages.json")
    try:
        with open(prompt_path, "w", encoding="utf-8") as handle:
            json.dump({"messages": message_list}, handle, ensure_ascii=False)

        if _is_frozen_app():
            cmd = [
                sys.executable,
                "--gemini-free-search",
                "--messages",
                prompt_path,
                "--model",
                model,
                "--timeout",
                str(helper_timeout),
            ]
        else:
            cmd = [
                sys.executable,
                os.path.abspath(__file__),
                "--search-helper",
                "--messages",
                prompt_path,
                "--model",
                model,
                "--timeout",
                str(helper_timeout),
            ]
        if max_tokens:
            cmd.extend(["--max-tokens", str(int(max_tokens))])

        env = os.environ.copy()
        env["GEMINI_FREE_HELPER"] = "1"
        env["QTWEBENGINE_CHROMIUM_FLAGS"] = _qtwebengine_chromium_flags(env.get("QTWEBENGINE_CHROMIUM_FLAGS", ""))
        env.setdefault("QTWEBENGINE_DISABLE_SANDBOX", "1")

        creationflags = 0
        if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
            creationflags = subprocess.CREATE_NO_WINDOW

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            creationflags=creationflags,
        )
        with _active_helper_lock:
            _active_helper_processes.add(proc)
        deadline = time.time() + helper_timeout + 20
        stdout = ""
        stderr = ""
        result_queue: "queue.Queue[tuple[str, str]]" = queue.Queue(maxsize=1)

        def _drain_output() -> None:
            try:
                out, err = proc.communicate()
            except Exception as exc:
                out, err = "", str(exc)
            try:
                result_queue.put_nowait((out or "", err or ""))
            except Exception:
                pass

        output_thread = threading.Thread(
            target=_drain_output,
            name="GeminiFreeHelperOutput",
            daemon=True,
        )
        output_thread.start()
        try:
            next_wait_log = time.time() + 10
            wait_log_printed = False
            while True:
                try:
                    stdout, stderr = result_queue.get(timeout=0.1)
                    break
                except queue.Empty:
                    pass
                if _is_cancelled():
                    _terminate_process_tree(proc, kill=True)
                    raise RuntimeError("stream cancelled")
                if time.time() >= deadline:
                    _terminate_process_tree(proc, kill=True)
                    raise RuntimeError(f"Gemini Free helper timed out after {helper_timeout}s")
                if log_fn and not wait_log_printed and time.time() >= next_wait_log:
                    should_log_wait = wait_log_event is None or not wait_log_event.is_set()
                    if wait_log_event is not None:
                        wait_log_event.set()
                    if should_log_wait:
                        _log(log_fn, "⏳ Gemini Free: still waiting for Qt WebEngine helper...")
                    wait_log_printed = True
        finally:
            with _active_helper_lock:
                _active_helper_processes.discard(proc)

        result = _extract_json_from_process(stdout)
        if proc.returncode != 0:
            error = str(result.get("error") or "").strip()
            if error:
                raise RuntimeError(f"Gemini Free helper failed ({proc.returncode}): {error}")
            detail = (stderr or stdout or "").strip()
            raise RuntimeError(f"Gemini Free helper failed ({proc.returncode}): {detail[-1200:]}")
        if result.get("error"):
            raise RuntimeError(str(result.get("error")))
        return result
    finally:
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass


def _run_search_subprocess_sequential(
    *,
    source_messages: List[Dict[str, Any]],
    chunks: List[List[Dict[str, Any]]],
    split_metadata: Dict[str, Any],
    model: str,
    timeout: int,
    max_tokens: Optional[int],
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    max_prompt_chars = _subchunk_prompt_chars()
    subchunk_timeout = _subchunk_timeout_seconds(timeout)
    _log(
        log_fn,
        f"🧩 Gemini Free: adaptive subchunking {len(chunks)} browser requests "
        f"(sequential helpers, subchunk timeout: {subchunk_timeout}s, "
        f"target prompt chars: {max_prompt_chars}, "
        f"payload format: {split_metadata.get('payload_format')}, "
        f"splitter: {split_metadata.get('splitter')}, "
        f"limit chars: {split_metadata.get('prompt_limit_chars')}, "
        f"url limit chars: {split_metadata.get('url_limit_chars')}, "
        f"fixed prompt chars: {split_metadata.get('fixed_prompt_chars')}, "
        f"fixed url chars: {split_metadata.get('fixed_url_chars')}, "
        f"body budget chars: {split_metadata.get('body_budget_chars')})"
    )

    wait_log_event = threading.Event()
    completed_results: List[Dict[str, Any]] = []
    prompt_chars_total = 0
    completion_chars_total = 0
    for index, chunk_messages in enumerate(chunks, start=1):
        if _is_cancelled():
            raise RuntimeError("stream cancelled")
        chunk_prompt_chars = len(_messages_to_prompt(chunk_messages))
        prompt_chars_total += chunk_prompt_chars
        chunk_started_at = time.time()
        _log(log_fn, f"🧩 Gemini Free: subchunk {index}/{len(chunks)} helper start ({chunk_prompt_chars:,} prompt chars)")
        try:
            result = _run_search_subprocess_once(
                messages=chunk_messages,
                model=model,
                timeout=subchunk_timeout,
                max_tokens=max_tokens,
                log_fn=log_fn,
                wait_log_event=wait_log_event,
            )
        except Exception as exc:
            elapsed = time.time() - chunk_started_at
            _log(log_fn, f"❌ Gemini Free: subchunk {index}/{len(chunks)} failed after {elapsed:.1f}s: {_short_error(exc)}")
            raise
        content = str(result.get("content") or "")
        completion_chars_total += len(content)
        raw_response = dict(result.get("raw_response") or {})
        raw_response.pop("content", None)
        elapsed = time.time() - chunk_started_at
        _log(log_fn, f"✅ Gemini Free: subchunk {index}/{len(chunks)} complete ({len(content):,} chars, {elapsed:.1f}s)")
        completed_results.append({
            "index": index,
            "content": content,
            "prompt_chars": chunk_prompt_chars,
            "completion_chars": len(content),
            "raw_response": raw_response,
        })

    _log(log_fn, f"✅ Gemini Free: all {len(completed_results)}/{len(chunks)} subchunks complete; merging responses")
    contents = [str(result.get("content") or "") for result in completed_results]
    raw_parts = [
        {
            "index": int(result.get("index") or 0),
            "prompt_chars": int(result.get("prompt_chars") or 0),
            "completion_chars": int(result.get("completion_chars") or 0),
            "raw_response": dict(result.get("raw_response") or {}),
        }
        for result in completed_results
    ]
    combined = "\n".join(part for part in contents if part).strip()
    return {
        "content": combined,
        "finish_reason": "stop",
        "usage": {
            "prompt_chars": len(_messages_to_prompt(source_messages)),
            "subchunk_prompt_chars": prompt_chars_total,
            "completion_chars": completion_chars_total,
        },
        "raw_response": {
            "model": _strip_search_prefix(model),
            "submit_mode": "adaptive_split_sequential",
            "subchunk_count": len(chunks),
            "subchunk_sequential": True,
            "subchunk_timeout_seconds": subchunk_timeout,
            "subchunk_prompt_target_chars": max_prompt_chars,
            "subchunk_fixed_prompt_chars": split_metadata.get("fixed_prompt_chars"),
            "subchunk_fixed_url_chars": split_metadata.get("fixed_url_chars"),
            "subchunk_prompt_limit_chars": split_metadata.get("prompt_limit_chars"),
            "subchunk_url_limit_chars": split_metadata.get("url_limit_chars"),
            "subchunk_body_budget_chars": split_metadata.get("body_budget_chars"),
            "subchunk_safety_chars": split_metadata.get("safety_chars"),
            "subchunk_payload_format": split_metadata.get("payload_format"),
            "subchunk_splitter": split_metadata.get("splitter"),
            "parts": raw_parts,
        },
    }


def _run_search_subprocess(
    *,
    messages: Iterable[Dict[str, Any]],
    model: str,
    timeout: int,
    max_tokens: Optional[int],
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    message_list = [dict(message) for message in (messages or []) if isinstance(message, dict)]
    prompt_chars = len(_messages_to_prompt(message_list))
    if _adaptive_split_enabled() and prompt_chars > _subchunk_prompt_chars():
        target_chars = _subchunk_prompt_chars()
        planned_chunks, split_metadata = _split_messages_for_search_budget(
            message_list,
            target_chars,
            return_metadata=True,
        )
        estimated_subchunks = max(1, len(planned_chunks))
        _log(
            log_fn,
            f"🧩 Gemini Free: large prompt will use adaptive browser subchunks "
            f"({estimated_subchunks}, {prompt_chars:,} prompt chars, "
            f"payload format: {split_metadata.get('payload_format')}, "
            f"splitter: {split_metadata.get('splitter')}, "
            f"limit chars: {split_metadata.get('prompt_limit_chars')}, "
            f"url limit chars: {split_metadata.get('url_limit_chars')}, "
            f"fixed prompt chars: {split_metadata.get('fixed_prompt_chars')}, "
            f"fixed url chars: {split_metadata.get('fixed_url_chars')}, "
            f"body budget chars: {split_metadata.get('body_budget_chars')})"
        )
        if len(planned_chunks) > 1:
            return _run_search_subprocess_sequential(
                source_messages=message_list,
                chunks=planned_chunks,
                split_metadata=split_metadata,
                model=model,
                timeout=timeout,
                max_tokens=max_tokens,
                log_fn=log_fn,
            )

    return _run_search_subprocess_once(
        messages=message_list,
        model=model,
        timeout=timeout,
        max_tokens=max_tokens,
        log_fn=log_fn,
    )


def send_chat_completion(
    *,
    messages: Iterable[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout: Optional[int] = None,
    log_fn: Optional[Callable[[str], None]] = None,
    **_: Any,
) -> Dict[str, Any]:
    """Send a chat-style prompt through the browser-backed Search/Gemini route."""
    del temperature
    timeout_value = int(timeout or _env_int("GEMINI_FREE_TIMEOUT", DEFAULT_TIMEOUT))
    mode = os.getenv("GEMINI_FREE_MODE", "subprocess").strip().lower()
    if mode == "inline":
        return _send_chat_completion_qt(
            messages=messages,
            model=model,
            timeout=timeout_value,
            max_tokens=max_tokens,
            log_fn=log_fn,
        )
    _log(log_fn, "🌐 Gemini Free: starting Qt WebEngine helper subprocess")
    return _run_search_subprocess(
        messages=messages,
        model=model,
        timeout=timeout_value,
        max_tokens=max_tokens,
        log_fn=log_fn,
    )


def _load_cli_messages(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.messages:
        with open(args.messages, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict) and isinstance(data.get("messages"), list):
            data = data["messages"]
        if not isinstance(data, list):
            raise ValueError("--messages must contain a JSON list or an object with a messages list")
        return data
    prompt = ""
    if args.prompt_file:
        prompt = _read_text(args.prompt_file)
    elif args.prompt == "-":
        prompt = sys.stdin.read()
    elif args.prompt:
        prompt = args.prompt
    return [{"role": "user", "content": prompt}]


def _print_summary(result: Dict[str, Any]) -> None:
    if "content" in result:
        print(result.get("content") or "")
        return
    body = str(result.get("body") or "")
    print(f"{result.get('status')} {result.get('statusText')}")
    print(f"url: {result.get('url')}")
    print(f"ok: {result.get('ok')} redirected: {result.get('redirected')} type: {result.get('type')}")
    print(f"bodyLength: {result.get('bodyLength')} truncated: {result.get('truncated')}")
    if result.get("error"):
        print(f"error: {result.get('error')}")
    if body:
        print("\nbody:")
        print(body)


def _main() -> int:
    for stream_name in ("stdout", "stderr"):
        stream_obj = getattr(sys, stream_name, None)
        if hasattr(stream_obj, "reconfigure"):
            try:
                stream_obj.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    parser = argparse.ArgumentParser(
        description="Use Qt WebEngine for Google Search/Gemini browser-backed requests.",
    )
    parser.add_argument("url", nargs="?", default=DEFAULT_URL, help="Raw request URL for fetch mode.")
    parser.add_argument("--search-helper", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", help="Prompt text. Use '-' to read stdin.")
    parser.add_argument("--prompt-file", help="UTF-8 file containing prompt text.")
    parser.add_argument("--messages", help="JSON file containing messages list or {'messages': [...]}.")
    parser.add_argument("--method", default="GET", help="Raw fetch HTTP method. Default: GET.")
    parser.add_argument("--header", action="append", help="Raw fetch request header in 'Name: value' form.")
    parser.add_argument("--data", help="Raw fetch request body text.")
    parser.add_argument("--data-file", help="Read raw fetch request body text from a UTF-8 file.")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Timeout in seconds.")
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--output", help="Write response body/content to a UTF-8 file.")
    parser.add_argument("--json", action="store_true", help="Print the full result object as JSON.")
    args = parser.parse_args()

    try:
        if args.search_helper or args.prompt or args.prompt_file or args.messages:
            result = _send_chat_completion_qt(
                messages=_load_cli_messages(args),
                model=args.model,
                timeout=args.timeout,
                max_tokens=args.max_tokens,
            )
        else:
            body = _read_text(args.data_file) if args.data_file else args.data
            result = run_qtwebengine_request(
                url=args.url,
                method=args.method,
                headers=_parse_headers(args.header),
                body=body,
                bootstrap_url=_origin_url(args.url),
                timeout=args.timeout,
            )
        if args.output:
            _write_text(args.output, str(result.get("content") or result.get("body") or ""))
        if args.json or args.search_helper:
            print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
        else:
            _print_summary(result)
        return 0 if not result.get("error") else 1
    except Exception as exc:
        error = {"content": "", "finish_reason": "error", "error": _short_error(exc)}
        if args.json or args.search_helper:
            print(json.dumps(error, ensure_ascii=False, separators=(",", ":")))
        else:
            print(f"Gemini Free request failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(_main())
