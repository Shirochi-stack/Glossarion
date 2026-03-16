# review_generator.py
"""
EPUB Review Generator — Extracts text from an EPUB using html2text,
fits as many complete chapters as possible within the input token limit,
and sends a single API call to generate a review/summary.
"""

import os
import sys
import json
import time
import zipfile
import threading
import queue
from typing import List, Tuple, Optional, Callable

import tiktoken

try:
    import html2text
except ImportError:
    html2text = None

try:
    import ebooklib
    from ebooklib import epub
except ImportError:
    ebooklib = None

from bs4 import BeautifulSoup


# ─── tokenizer setup ────────────────────────────────────────────────────
_enc = None
_enc_lock = threading.Lock()

def _get_encoder():
    global _enc
    if _enc is not None:
        return _enc
    with _enc_lock:
        if _enc is not None:
            return _enc
        try:
            # Use cl100k_base directly — it works well for token estimation
            # and avoids hangs from tiktoken trying to download BPE files
            # for unrecognized model names (e.g. gemini-2.0-flash)
            _enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _enc = None
    return _enc


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken, with a crude fallback."""
    enc = _get_encoder()
    if enc:
        return len(enc.encode(text))
    return max(1, len(text) // 4)


# ─── EPUB reading helpers ───────────────────────────────────────────────

# Special file patterns to skip (nav, toc, cover)
_SPECIAL_PATTERNS = [
    'nav', 'toc', 'cover', 'titlepage', 'copyright',
    'colophon', 'halftitle', 'frontmatter',
]


def _is_special_file(filename: str) -> bool:
    """Check if a filename is a special/metadata file that should be skipped."""
    base = os.path.splitext(os.path.basename(filename))[0].lower()
    return any(pat in base for pat in _SPECIAL_PATTERNS)


def _html_to_plaintext(html_content: str) -> str:
    """Convert HTML to plain text using html2text."""
    if html2text:
        h = html2text.HTML2Text()
        h.body_width = 0
        h.unicode_snob = True
        h.images_as_html = False
        h.images_to_alt = True
        h.ignore_images = True
        h.ignore_links = True
        h.protect_links = False
        return h.handle(html_content)
    else:
        # Fallback: strip tags with BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        return soup.get_text(separator='\n')


def _get_spine_ordered_html_files(epub_path: str) -> List[Tuple[str, str]]:
    """
    Read an EPUB and return a list of (filename, html_content) tuples
    in spine reading order, filtering out special files.
    """
    results = []

    try:
        # Try ebooklib first
        if ebooklib:
            book = epub.read_epub(epub_path, options={'ignore_ncx': True})
            spine_ids = [item_id for item_id, _ in book.spine]
            id_to_item = {item.get_id(): item for item in book.get_items()}

            for item_id in spine_ids:
                item = id_to_item.get(item_id)
                if item is None:
                    continue
                fname = item.get_name()
                if _is_special_file(fname):
                    continue
                try:
                    content = item.get_content().decode('utf-8', errors='replace')
                except Exception:
                    continue
                results.append((fname, content))

            if results:
                return results
    except Exception:
        pass

    # Fallback: manual zipfile + OPF parsing
    try:
        with zipfile.ZipFile(epub_path, 'r') as zf:
            # Find OPF
            opf_name = None
            for name in zf.namelist():
                if name.lower().endswith('.opf'):
                    opf_name = name
                    break

            if not opf_name:
                # No OPF — just grab all xhtml/html files in order
                html_files = sorted(
                    n for n in zf.namelist()
                    if n.lower().endswith(('.xhtml', '.html', '.htm'))
                    and not _is_special_file(n)
                )
                for fname in html_files:
                    content = zf.read(fname).decode('utf-8', errors='replace')
                    results.append((fname, content))
                return results

            # Parse OPF for spine order
            opf_content = zf.read(opf_name).decode('utf-8', errors='replace')
            opf_dir = os.path.dirname(opf_name)

            try:
                soup = BeautifulSoup(opf_content, 'xml')
            except Exception:
                soup = BeautifulSoup(opf_content, 'html.parser')

            # Build manifest id → href map
            manifest = {}
            for item in soup.find_all('item'):
                item_id = item.get('id', '')
                href = item.get('href', '')
                if item_id and href:
                    manifest[item_id] = href

            # Read spine order
            spine_refs = []
            spine_tag = soup.find('spine')
            if spine_tag:
                for itemref in spine_tag.find_all('itemref'):
                    idref = itemref.get('idref', '')
                    if idref and idref in manifest:
                        spine_refs.append(manifest[idref])

            for href in spine_refs:
                # Resolve path relative to OPF
                full_path = os.path.normpath(os.path.join(opf_dir, href)).replace('\\', '/')
                if _is_special_file(href):
                    continue
                try:
                    content = zf.read(full_path).decode('utf-8', errors='replace')
                    results.append((href, content))
                except KeyError:
                    # Try without OPF dir prefix
                    try:
                        content = zf.read(href).decode('utf-8', errors='replace')
                        results.append((href, content))
                    except KeyError:
                        continue

    except Exception as e:
        print(f"⚠️ Failed to read EPUB: {e}")

    return results


def _extract_epub_chapters(file_path: str, log_fn: Callable = print) -> List[Tuple[str, str]]:
    """
    Extract all chapters from an EPUB as plain text.
    Returns list of (chapter_name, plain_text) tuples in spine order.
    """
    html_files = _get_spine_ordered_html_files(file_path)
    chapters = []

    for fname, html_content in html_files:
        text = _html_to_plaintext(html_content).strip()
        if text:
            chapters.append((fname, text))

    log_fn(f"📖 Extracted {len(chapters)} chapters from EPUB")
    return chapters


def _extract_text_file(file_path: str, log_fn: Callable = print) -> List[Tuple[str, str]]:
    """Read a plain text or HTML file and return as a single-chapter list."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
    except Exception as e:
        log_fn(f"⚠️ Failed to read file: {e}")
        return []

    # If it looks like HTML, convert it
    if content.strip().startswith('<') and ('<html' in content.lower() or '<body' in content.lower()):
        content = _html_to_plaintext(content)

    if content.strip():
        log_fn(f"📄 Read text file: {os.path.basename(file_path)}")
        return [(os.path.basename(file_path), content.strip())]
    return []


def _extract_pdf_file(file_path: str, log_fn: Callable = print) -> List[Tuple[str, str]]:
    """Extract text from a PDF file."""
    try:
        from txt_processor import TextFileProcessor
        processor = TextFileProcessor()
        text = processor.extract_text_from_file(file_path)
        if text and text.strip():
            log_fn(f"📄 Extracted text from PDF: {os.path.basename(file_path)}")
            return [(os.path.basename(file_path), text.strip())]
    except Exception as e:
        log_fn(f"⚠️ Failed to extract PDF text: {e}")
    return []


def extract_chapter_texts(file_path: str, log_fn: Callable = print) -> List[Tuple[str, str]]:
    """
    Extract text content from any supported file type.
    Returns list of (section_name, plain_text) tuples.
    Supported: .epub, .txt, .html, .htm, .xhtml, .pdf, .md
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.epub':
        return _extract_epub_chapters(file_path, log_fn)
    elif ext == '.pdf':
        return _extract_pdf_file(file_path, log_fn)
    else:
        # Text-based files: .txt, .html, .htm, .xhtml, .md, etc.
        return _extract_text_file(file_path, log_fn)


# ─── Token counting ─────────────────────────────────────────────────────

def count_epub_tokens(epub_path: str, log_fn: Callable = print) -> int:
    """
    Count total tokens in a file using html2text + tiktoken.
    Uses parallel threading for speed on large files.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time

    print(f"[TokenCount] Starting for: {os.path.basename(epub_path)}")
    t0 = time.time()

    print(f"[TokenCount] Extracting chapters...")
    chapters = extract_chapter_texts(epub_path, log_fn=lambda *_: None)
    print(f"[TokenCount] Extracted {len(chapters)} chapters in {time.time() - t0:.1f}s")
    if not chapters:
        return 0

    # Eagerly initialize encoder before threading
    print(f"[TokenCount] Initializing encoder...")
    enc = _get_encoder()
    if enc is None:
        print(f"[TokenCount] WARNING: tiktoken encoder is None, using fallback")
    print(f"[TokenCount] Encoder ready in {time.time() - t0:.1f}s")

    # Count tokens in parallel across chapters
    print(f"[TokenCount] Counting tokens across {len(chapters)} chapters...")
    with ThreadPoolExecutor(max_workers=min(8, len(chapters))) as pool:
        futures = [pool.submit(count_tokens, text) for _, text in chapters]
        total = sum(f.result() for f in as_completed(futures))

    print(f"[TokenCount] Done: {total:,} tokens in {time.time() - t0:.1f}s")
    return total


# ─── Chapter chunking helper ────────────────────────────────────────────

import re

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')


def _chunk_chapter(name: str, text: str, budget: int) -> Tuple[str, str]:
    """
    Truncate a single chapter's text so it fits within *budget* tokens.

    Strategy:
      1. Split on sentence boundaries and greedily accumulate.
      2. If even the first sentence exceeds the budget, fall back to a raw
         character slice (~4 chars per token heuristic, then trim to budget).

    Returns (name_with_suffix, truncated_text).
    """
    sentences = _SENTENCE_SPLIT_RE.split(text)
    accumulated = []
    used = 0
    for sent in sentences:
        t = count_tokens(sent)
        if used + t > budget:
            break
        accumulated.append(sent)
        used += t

    if accumulated:
        return (name, " ".join(accumulated))

    # Even the first sentence is too large — hard-truncate by characters
    approx_chars = budget * 4  # rough estimate
    chunk = text[:approx_chars]
    # Trim to fit exactly
    while count_tokens(chunk) > budget and len(chunk) > 10:
        chunk = chunk[:int(len(chunk) * 0.9)]
    return (name, chunk)


# ─── Chapter fitting logic ──────────────────────────────────────────────

def _fit_chapters(
    chapters: List[Tuple[str, str]],
    token_budget: int,
    spoiler_mode: bool,
) -> Tuple[List[Tuple[str, str]], str]:
    """
    Select which chapters to include based on the token budget.

    If no whole chapter fits, falls back to chunking:
      - Normal mode:  chunk the first chapter.
      - Spoiler mode:  chunk the first AND last chapter (each gets half budget).

    Returns:
        (selected_chapters, info_message)
    """
    # Count tokens per chapter
    chapter_tokens = [(name, text, count_tokens(text)) for name, text in chapters]
    total_chapters = len(chapter_tokens)

    if not spoiler_mode:
        # Normal mode: greedily include chapters from the start
        selected = []
        used = 0
        for name, text, tokens in chapter_tokens:
            if used + tokens > token_budget:
                break
            selected.append((name, text))
            used += tokens

        if not selected:
            # Fallback: chunk the first chapter to fit
            name, text, _ = chapter_tokens[0]
            chunked_name, chunked_text = _chunk_chapter(name, text, token_budget)
            chunked_tokens = count_tokens(chunked_text)
            info = (
                f"📊 Budget too small for whole chapters — using partial first chapter "
                f"({chunked_tokens:,} tokens, budget: {token_budget:,})"
            )
            return [(chunked_name, chunked_text)], info

        info = f"📊 Fitted {len(selected)}/{total_chapters} chapters ({used:,} tokens, budget: {token_budget:,})"
        return selected, info

    else:
        # Spoiler mode: split budget 50/50 between first and last chapters
        half_budget = token_budget // 2

        # Single-chapter content (1-chapter EPUB, .txt, .pdf, etc.)
        # Split the TEXT itself into first-half and last-half portions.
        if total_chapters == 1:
            name, text, tokens = chapter_tokens[0]
            # Split text roughly in half by characters, then chunk each half to fit
            mid = len(text) // 2
            first_text = text[:mid]
            last_text = text[mid:]
            fn, ft = _chunk_chapter(f"{name} [first half]", first_text, half_budget)
            ln, lt = _chunk_chapter(f"{name} [last half]", last_text, half_budget)
            tok = count_tokens(ft) + count_tokens(lt)
            info = (
                f"📊 Spoiler mode (single file): first half + last half "
                f"({tok:,} tokens, budget: {token_budget:,})"
            )
            return [(fn, ft), (ln, lt)], info

        # Multi-chapter: greedy from start
        first_half = []
        used_first = 0
        for name, text, tokens in chapter_tokens:
            if used_first + tokens > half_budget:
                break
            first_half.append((name, text))
            used_first += tokens

        # Last half: greedy from end (reversed)
        last_half = []
        used_last = 0
        first_count = len(first_half)
        # Only pick from chapters AFTER the first half to avoid overlap
        remaining = chapter_tokens[first_count:]
        for name, text, tokens in reversed(remaining):
            if used_last + tokens > half_budget:
                break
            last_half.insert(0, (name, text))  # Maintain order
            used_last += tokens

        # Fallback: if either half is empty, chunk the relevant chapter
        if not first_half and not last_half:
            # Nothing fits at all — chunk both first and last
            n1, t1, _ = chapter_tokens[0]
            n2, t2, _ = chapter_tokens[-1]
            cn1, ct1 = _chunk_chapter(n1, t1, half_budget)
            cn2, ct2 = _chunk_chapter(n2, t2, half_budget)
            tok = count_tokens(ct1) + count_tokens(ct2)
            info = (
                f"📊 Spoiler mode: partial first + partial last chapter "
                f"({tok:,} tokens, budget: {token_budget:,})"
            )
            return [(cn1, ct1), (cn2, ct2)], info
        elif not first_half:
            # Only the first half couldn't fit — chunk the first chapter
            n, t, _ = chapter_tokens[0]
            cn, ct = _chunk_chapter(n, t, half_budget)
            first_half = [(cn, ct)]
            used_first = count_tokens(ct)
        elif not last_half and remaining:
            # Only the last half couldn't fit — chunk the last chapter
            n, t, _ = chapter_tokens[-1]
            cn, ct = _chunk_chapter(n, t, half_budget)
            last_half = [(cn, ct)]
            used_last = count_tokens(ct)

        selected = first_half + last_half
        total_used = used_first + used_last
        skipped = total_chapters - len(selected)
        info = (
            f"📊 Spoiler mode: {len(first_half)} first + {len(last_half)} last chapters "
            f"({total_used:,} tokens, {skipped} middle chapters skipped, budget: {token_budget:,})"
        )
        return selected, info


# ─── Default system prompt ──────────────────────────────────────────────

DEFAULT_REVIEW_PROMPT = """You are a literary critic and book reviewer. You will be given the text content of a novel (or part of it).

Write a comprehensive review ENTIRELY in {target_lang} that includes:
1. **Summary** — A concise plot summary covering the main storyline
2. **Characters** — Key characters and their roles
3. **Themes** — Major themes explored in the work
4. **Writing Style** — Assessment of the author's writing style, pacing, and narrative technique
5. **Strengths & Weaknesses** — What the work does well and where it falls short
6. **Originality Score** — Create a markdown comparison table scoring how unique vs generic the novel is across these categories: Plot, Setting/World-Building, Characters, Power System/Magic, Themes, Prose Style, and Overall. Use a 1-10 scale (1 = completely generic/derivative, 10 = highly original/unique). Include a brief note for each score explaining your reasoning. Example format:

| Category | Score | Notes |
|----------|:-----:|-------|
| Plot | 6/10 | ... |

7. **Overall Assessment** — Your overall rating and recommendation

Use markdown ### headers for each section title (e.g. ### 1. Summary) to make them stand out. Use **bold** for emphasis within sections.

IMPORTANT: Your entire output must be in {target_lang}. Do NOT include any raw/untranslated text from the source language. All character names, place names, titles, and terms must be transliterated or translated into {target_lang}. Write in a professional but engaging tone. Be specific with examples from the text when possible."""


# ─── Main review generation ─────────────────────────────────────────────

def generate_review(
    epub_path: str,
    output_dir: str,
    api_key: str,
    model: str,
    endpoint: str,
    system_prompt: str,
    input_token_limit: int,
    spoiler_mode: bool,
    temperature: float,
    config: dict,
    log_fn: Callable = print,
    stop_check_fn: Callable = None,
) -> Optional[str]:
    """
    Generate a review of an EPUB by sending content in a single API call.
    
    Args:
        epub_path: Path to the EPUB file
        output_dir: Output directory (review saved under review/ subfolder)
        api_key: API key for the model
        model: Model name
        endpoint: API endpoint URL
        system_prompt: System prompt for the review
        input_token_limit: Maximum input tokens
        spoiler_mode: If True, split 50/50 between first and last chapters
        temperature: API temperature
        config: Full config dict (for multi-key support etc.)
        log_fn: Logging callback
        stop_check_fn: Stop check callback
        
    Returns:
        The generated review text, or None on failure.
    """
    if stop_check_fn and stop_check_fn():
        return None

    # 1. Extract chapters
    log_fn("📖 Extracting EPUB content with html2text...")
    chapters = extract_chapter_texts(epub_path, log_fn=log_fn)

    if not chapters:
        log_fn("❌ No chapters found in EPUB")
        return None

    # 2. Count system prompt tokens
    prompt_tokens = count_tokens(system_prompt)
    available_budget = input_token_limit - prompt_tokens
    if available_budget <= 0:
        log_fn(f"❌ System prompt ({prompt_tokens:,} tokens) exceeds input token limit ({input_token_limit:,})")
        return None

    log_fn(f"📊 Input token limit: {input_token_limit:,} | System prompt: {prompt_tokens:,} tokens | Available for content: {available_budget:,} tokens")

    # 3. Fit chapters
    selected_chapters, fit_info = _fit_chapters(chapters, available_budget, spoiler_mode)
    log_fn(fit_info)

    if not selected_chapters:
        log_fn("❌ No chapters fit within the token budget")
        return None

    if stop_check_fn and stop_check_fn():
        log_fn("🛑 Stopped by user")
        return None

    # 4. Build the concatenated content
    content_parts = []
    for i, (name, text) in enumerate(selected_chapters):
        content_parts.append(f"--- Chapter: {name} ---\n{text}")

    full_content = "\n\n".join(content_parts)
    content_tokens = count_tokens(full_content)

    # 5. Build messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": full_content},
    ]

    # 6. Make API call via UnifiedClient
    try:
        from unified_api_client import UnifiedClient
        from extract_glossary_from_epub import create_client_with_multi_key_support

        # Set environment variables for the client
        if endpoint:
            os.environ['ENDPOINT'] = endpoint
        os.environ['MODEL'] = model

        client = create_client_with_multi_key_support(api_key, model, output_dir, config)

        # Show token/key info (model is logged by UnifiedClient after key selection)
        is_multi = getattr(client, '_multi_key_mode', False)
        key_count = 1
        if is_multi:
            try:
                pool = getattr(client, '_api_key_pool', None) or client.__class__._api_key_pool
                if pool and hasattr(pool, 'keys'):
                    key_count = len(pool.keys)
                elif pool and hasattr(pool, '_keys'):
                    key_count = len(pool._keys)
            except Exception:
                key_count = 2
        if key_count > 1:
            log_fn(f"📤 Sending {content_tokens:,} tokens ({key_count} keys available)...")
        else:
            log_fn(f"📤 Sending {content_tokens:,} tokens to {model}...")

        log_fn("🚀 Sending API request (single call)...")
        start_time = time.time()

        result = client.send(messages, temperature=temperature, max_tokens=None, context='review')
        elapsed = time.time() - start_time

        # Check if force-stopped while waiting for API
        if stop_check_fn and stop_check_fn():
            log_fn("🛑 Stopped by user — discarding API response")
            return None

        # Extract content from result
        if isinstance(result, tuple):
            review_text, finish_reason = result
        else:
            review_text = result
            finish_reason = 'stop'

        if not review_text or not review_text.strip():
            log_fn("❌ Empty response from API")
            return None

        log_fn(f"✅ Review generated in {elapsed:.1f}s (finish_reason: {finish_reason})")

        # 7. Save review
        review_dir = os.path.join(output_dir, "review")
        os.makedirs(review_dir, exist_ok=True)
        review_path = os.path.join(review_dir, "review.md")

        with open(review_path, 'w', encoding='utf-8') as f:
            f.write(review_text.strip())

        log_fn(f"💾 Review saved to: {review_path}")
        return review_text.strip()

    except Exception as e:
        # Silently handle user-initiated cancellation (force stop)
        err_str = str(e).lower()
        if 'cancelled' in err_str or 'canceled' in err_str:
            log_fn("🛑 Review cancelled by user")
            return None
        log_fn(f"❌ API call failed: {e}")
        import traceback
        traceback.print_exc()
        return None
