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

# Special file patterns to skip — mirrors TransateKRtoEN.py special_keywords
_SPECIAL_PATTERNS = [
    'title', 'toc', 'cover', 'index', 'copyright', 'preface', 'nav',
    'message', 'info', 'notice', 'colophon', 'dedication', 'epigraph',
    'foreword', 'acknowledgment', 'author', 'appendix', 'glossary',
    'bibliography', 'titlepage', 'halftitle', 'frontmatter', 'backmatter',
]


def _is_special_file(filename: str) -> bool:
    """Check if a filename is a special/metadata file that should be skipped.
    Mirrors the heuristic in translator_gui._is_special_file:
    1) Known keyword patterns
    2) Filenames with no digits (e.g. 'info.xhtml', 'about.xhtml')
    """
    import re
    base = os.path.splitext(os.path.basename(filename))[0].lower()
    # Check known special-file keywords
    if any(pat in base for pat in _SPECIAL_PATTERNS):
        return True
    # Heuristic: filenames with no digits are often special/metadata files
    if not re.search(r'\d', base):
        return True
    return False


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


def _get_spine_ordered_html_files(epub_path: str, log_fn: Callable = print) -> List[Tuple[str, str]]:
    """
    Read an EPUB and return a list of (filename, html_content) tuples
    in spine reading order, filtering out special files unless
    TRANSLATE_SPECIAL_FILES is enabled.
    """
    results = []
    # Respect the translate_special_files toggle — when ON, include special files
    include_special = os.environ.get('TRANSLATE_SPECIAL_FILES', '0') == '1'

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
                if not include_special and _is_special_file(fname):
                    log_fn(f"⏭️ Skipping special file: {fname}")
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
                    and (include_special or not _is_special_file(n))
                )
                # Log skipped special files
                if not include_special:
                    for n in zf.namelist():
                        if n.lower().endswith(('.xhtml', '.html', '.htm')) and _is_special_file(n):
                            log_fn(f"⏭️ Skipping special file: {n}")
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
                if not include_special and _is_special_file(href):
                    log_fn(f"⏭️ Skipping special file: {href}")
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
    html_files = _get_spine_ordered_html_files(file_path, log_fn=log_fn)
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
    """Extract text from a PDF file using fast sequential extraction."""
    # Use fitz (PyMuPDF) directly in sequential mode to avoid ProcessPoolExecutor
    # overhead that extract_text_from_pdf uses for large PDFs (spawning 8+ worker
    # processes on Windows is very slow and causes log spam).
    try:
        import fitz
        doc = fitz.open(file_path)
        text_parts = []
        for page in doc:
            text = page.get_text()
            if text.strip():
                text_parts.append(text)
        doc.close()
        text = "\n\n".join(text_parts)
        if text.strip():
            log_fn(f"📄 Extracted text from PDF: {os.path.basename(file_path)}")
            return [(os.path.basename(file_path), text.strip())]
    except ImportError:
        pass
    except Exception as e:
        log_fn(f"⚠️ Failed to extract PDF with PyMuPDF: {e}")

    # Fallback to pdf_extractor (may use ProcessPoolExecutor for large files)
    try:
        from pdf_extractor import extract_text_from_pdf
        text = extract_text_from_pdf(file_path)
        if text and text.strip():
            log_fn(f"📄 Extracted text from PDF: {os.path.basename(file_path)}")
            return [(os.path.basename(file_path), text.strip())]
    except ImportError:
        log_fn("⚠️ No PDF library available (install PyMuPDF, pypdf, or pdfplumber)")
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
        # If the whole thing fits, just use it all — no split needed.
        if total_chapters == 1:
            name, text, tokens = chapter_tokens[0]
            if tokens <= token_budget:
                info = f"📊 Spoiler mode: entire file included ({tokens:,} tokens, budget: {token_budget:,})"
                return [(name, text)], info
            # Doesn't fit whole — split the TEXT into first-half and last-half portions.
            mid = len(text) // 2
            first_text = text[:mid]
            last_text = text[mid:]
            fn, ft = _chunk_chapter(f"{name} [first half]", first_text, half_budget)
            ln, lt = _chunk_chapter(f"{name} [last half]", last_text, half_budget)
            tok_first = count_tokens(ft)
            tok_last = count_tokens(lt)
            tok = tok_first + tok_last
            info = (
                f"📊 Spoiler mode (single file): first half ({tok_first:,}) + last half ({tok_last:,}) "
                f"= {tok:,} tokens (budget: {token_budget:,})"
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

DEFAULT_CHUNK_PROMPT = DEFAULT_REVIEW_PROMPT  # Per-chunk prompt is the same as the main review prompt

DEFAULT_FINAL_REVIEW_PROMPT = """You are an expert literary critic and book reviewer. Below you will find multiple individual review segments, each covering a different section (chunk) of the same novel. These chunk reviews were generated independently by reviewing consecutive chapter ranges.

Your task is to synthesize ALL of the chunk reviews below into a single, comprehensive, cohesive, and polished final review. Follow these guidelines carefully:

1. **Merge and Deduplicate** — Combine overlapping information. If multiple chunks mention the same characters, themes, or plot points, merge them into unified descriptions rather than repeating.
2. **Resolve Contradictions** — If chunk reviews disagree on quality assessments or interpretations, weigh the evidence and provide a balanced final judgment. Note significant shifts in quality across the novel if relevant.
3. **Maintain Chronological Awareness** — Since chunks cover sequential portions of the novel, your summary should reflect the full narrative arc from beginning to end.
4. **Comprehensive Coverage** — Ensure no significant plot points, characters, or themes from any chunk review are lost in the synthesis.

Your final review MUST include these sections, using markdown ### headers:

### 1. Summary
A complete plot summary covering the entire novel's storyline from start to finish, synthesized from all chunk summaries.

### 2. Characters
All key characters mentioned across all chunks, with their roles and development throughout the story.

### 3. Themes
All major themes explored across the entire work, noting how they evolve or deepen across different sections.

### 4. Writing Style
A holistic assessment of the author's writing style, pacing, and narrative technique across the full novel. Note any changes in quality or style between early and later sections.

### 5. Strengths & Weaknesses
A consolidated list of what the work does well and where it falls short, drawn from all chunk assessments.

### 6. Originality Score
Create a markdown comparison table scoring how unique vs generic the novel is across these categories: Plot, Setting/World-Building, Characters, Power System/Magic, Themes, Prose Style, and Overall. Use a 1-10 scale (1 = completely generic/derivative, 10 = highly original/unique). Include a brief note for each score explaining your reasoning.

| Category | Score | Notes |
|----------|:-----:|-------|
| Plot | ?/10 | ... |
| Setting/World-Building | ?/10 | ... |
| Characters | ?/10 | ... |
| Power System/Magic | ?/10 | ... |
| Themes | ?/10 | ... |
| Prose Style | ?/10 | ... |
| **Overall** | ?/10 | ... |

### 7. Overall Assessment
Your final rating and recommendation for the novel as a whole, considering all sections reviewed.

IMPORTANT: Your entire output must be in {target_lang}. Do NOT include any raw/untranslated text. All names, places, and terms must be transliterated or translated into {target_lang}. Write in a professional but engaging tone. Be specific with examples from the text when possible."""


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
    log_fn(f"📖 Extracting content from {os.path.basename(epub_path)}...")
    chapters = extract_chapter_texts(epub_path, log_fn=log_fn)

    if not chapters:
        log_fn("❌ No text content found in file")
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

        try:
            from TransateKRtoEN import send_with_interrupt
            result_tuple = send_with_interrupt(
                messages, client, temperature=temperature, max_tokens=None,
                stop_check_fn=stop_check_fn, context='review',
            )
            elapsed = time.time() - start_time
            if isinstance(result_tuple, tuple) and len(result_tuple) == 3:
                review_text, finish_reason, raw_obj = result_tuple
            elif isinstance(result_tuple, tuple) and len(result_tuple) == 2:
                review_text, finish_reason = result_tuple
            else:
                review_text = result_tuple
                finish_reason = 'stop'
        except ImportError:
            result = client.send(messages, temperature=temperature, max_tokens=None, context='review')
            elapsed = time.time() - start_time
            if isinstance(result, tuple):
                review_text, finish_reason = result
            else:
                review_text = result
                finish_reason = 'stop'

        # Check if force-stopped while waiting for API
        if stop_check_fn and stop_check_fn():
            log_fn("🛑 Stopped by user — discarding API response")
            return None

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
        # Silently handle user-initiated cancellation / stop
        err_str = str(e).lower()
        if 'cancelled' in err_str or 'canceled' in err_str or 'stopped by user' in err_str:
            log_fn("🛑 Review stopped by user")
            return None
        log_fn(f"❌ API call failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ─── Chunked review generation ──────────────────────────────────────────

def _split_into_chunks(
    chapters: List[Tuple[str, str]],
    token_budget: int,
    spoiler_mode: bool,
) -> List[List[Tuple[str, str]]]:
    """
    Split a full chapter list into multiple chunk groups, each fitting
    within the token budget. In spoiler mode, each chunk applies 50/50
    first/last splitting logic.

    Returns a list of chunk groups, where each group is a list of
    (chapter_name, chapter_text) tuples.
    """
    if not chapters:
        return []

    # Count tokens per chapter
    chapter_tokens = [(name, text, count_tokens(text)) for name, text in chapters]

    chunks = []
    remaining = list(chapter_tokens)

    while remaining:
        current_chunk = []
        used = 0

        if not spoiler_mode:
            # Greedy: pack as many consecutive chapters as possible
            while remaining:
                name, text, tokens = remaining[0]
                if used + tokens > token_budget and current_chunk:
                    break  # This chapter doesn't fit, start a new chunk
                if used + tokens > token_budget and not current_chunk:
                    # Single chapter too large — chunk it down
                    cn, ct = _chunk_chapter(name, text, token_budget)
                    current_chunk.append((cn, ct))
                    remaining.pop(0)
                    break
                current_chunk.append((name, text))
                used += tokens
                remaining.pop(0)
        else:
            # Spoiler mode: take chapters from front and back of remaining
            half_budget = token_budget // 2

            # Front half
            front = []
            front_used = 0
            while remaining:
                name, text, tokens = remaining[0]
                if front_used + tokens > half_budget and front:
                    break
                if front_used + tokens > half_budget and not front:
                    cn, ct = _chunk_chapter(name, text, half_budget)
                    front.append((cn, ct))
                    remaining.pop(0)
                    break
                front.append((name, text))
                front_used += tokens
                remaining.pop(0)

            # Back half (from end of remaining)
            back = []
            back_used = 0
            while remaining:
                name, text, tokens = remaining[-1]
                if back_used + tokens > half_budget and back:
                    break
                if back_used + tokens > half_budget and not back:
                    cn, ct = _chunk_chapter(name, text, half_budget)
                    back.insert(0, (cn, ct))
                    remaining.pop()
                    break
                back.insert(0, (name, text))
                back_used += tokens
                remaining.pop()

            current_chunk = front + back

        if current_chunk:
            chunks.append(current_chunk)

    return chunks


def generate_chunked_review(
    epub_path: str,
    output_dir: str,
    api_key: str,
    model: str,
    endpoint: str,
    system_prompt: str,
    final_review_prompt: str,
    input_token_limit: int,
    spoiler_mode: bool,
    wrap_chunks: bool,
    temperature: float,
    config: dict,
    batch_size: int = 1,
    log_fn: Callable = print,
    stop_check_fn: Callable = None,
) -> Optional[str]:
    """
    Generate a review by splitting content into chunks, reviewing each
    chunk separately, then synthesizing all chunk reviews into one final review.

    Args:
        epub_path: Path to the file
        output_dir: Output directory
        api_key: API key
        model: Model name
        endpoint: API endpoint URL
        system_prompt: System prompt for per-chunk reviews
        final_review_prompt: System prompt for the final synthesis review
        input_token_limit: Maximum input tokens per API call
        spoiler_mode: If True, apply 50/50 first/last logic per chunk
        wrap_chunks: If True, wrap chunk reviews with header/footer markers
        temperature: API temperature
        config: Full config dict
        batch_size: Number of parallel chunk workers (1 = sequential)
        log_fn: Logging callback
        stop_check_fn: Stop check callback

    Returns:
        The final synthesized review text, or None on failure.
    """
    if stop_check_fn and stop_check_fn():
        return None

    # 1. Extract chapters
    log_fn(f"📖 Extracting content from {os.path.basename(epub_path)}...")
    chapters = extract_chapter_texts(epub_path, log_fn=log_fn)

    if not chapters:
        log_fn("❌ No text content found in file")
        return None

    # 2. Compute token budget for content (subtract system prompt tokens)
    prompt_tokens = count_tokens(system_prompt)
    available_budget = input_token_limit - prompt_tokens
    if available_budget <= 0:
        log_fn(f"❌ System prompt ({prompt_tokens:,} tokens) exceeds input token limit ({input_token_limit:,})")
        return None

    log_fn(f"📊 Input token limit: {input_token_limit:,} | System prompt: {prompt_tokens:,} tokens | Available per chunk: {available_budget:,} tokens")

    # 3. Split into chunks
    chunk_groups = _split_into_chunks(chapters, available_budget, spoiler_mode)
    total_chunks = len(chunk_groups)

    if total_chunks == 0:
        log_fn("❌ No chunks could be created from the content")
        return None

    # If only 1 chunk, fall back to normal single-call review
    if total_chunks == 1:
        log_fn("📦 Only 1 chunk needed — falling back to single-call review")
        return generate_review(
            epub_path=epub_path,
            output_dir=output_dir,
            api_key=api_key,
            model=model,
            endpoint=endpoint,
            system_prompt=system_prompt,
            input_token_limit=input_token_limit,
            spoiler_mode=spoiler_mode,
            temperature=temperature,
            config=config,
            log_fn=log_fn,
            stop_check_fn=stop_check_fn,
        )

    log_fn(f"📦 Chunk Mode: splitting into {total_chunks} chunks (batch_size={batch_size})")

    if stop_check_fn and stop_check_fn():
        log_fn("🛑 Stopped by user")
        return None

    # 4. Client factory — each thread creates its own for true parallelism
    def _make_client():
        from unified_api_client import UnifiedClient
        from extract_glossary_from_epub import create_client_with_multi_key_support

        if endpoint:
            os.environ['ENDPOINT'] = endpoint
        os.environ['MODEL'] = model

        return create_client_with_multi_key_support(api_key, model, output_dir, config)

    # Verify we can create at least one client before spawning threads
    try:
        _test_client = _make_client()
        del _test_client
    except Exception as e:
        log_fn(f"❌ Failed to create API client: {e}")
        return None

    # 5. Send chunk API calls (parallel when batch_size > 1)
    import threading as _threading
    _log_lock = _threading.Lock()

    def _safe_log(msg):
        with _log_lock:
            log_fn(msg)

    def _review_chunk(ci, chunk_chapters):
        """Process a single chunk — returns (index, review_text_or_None, elapsed)."""
        if stop_check_fn and stop_check_fn():
            return ci, None, 0.0

        # Per-thread client for true parallelism
        try:
            thread_client = _make_client()
        except Exception as e:
            _safe_log(f"❌ Chunk {ci+1}: Failed to create API client: {e}")
            return ci, None, 0.0

        first_ch = chunk_chapters[0][0]
        last_ch = chunk_chapters[-1][0]
        range_label = first_ch if first_ch == last_ch else f"{first_ch} → {last_ch}"

        _safe_log(f"\n{'─'*40}\n📄 Chunk {ci+1}/{total_chunks}: {range_label} ({len(chunk_chapters)} chapter(s))\n{'─'*40}")

        content_parts = [f"--- Chapter: {name} ---\n{text}" for name, text in chunk_chapters]
        full_content = "\n\n".join(content_parts)
        content_tokens = count_tokens(full_content)

        _safe_log(f"📤 Chunk {ci+1}: Sending {content_tokens:,} tokens...")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_content},
        ]

        try:
            start_time = time.time()

            try:
                from TransateKRtoEN import send_with_interrupt
                result_tuple = send_with_interrupt(
                    messages, thread_client, temperature=temperature, max_tokens=None,
                    stop_check_fn=stop_check_fn, context='review',
                )
                elapsed = time.time() - start_time
                if isinstance(result_tuple, tuple) and len(result_tuple) == 3:
                    chunk_text, finish_reason, _ = result_tuple
                elif isinstance(result_tuple, tuple) and len(result_tuple) == 2:
                    chunk_text, finish_reason = result_tuple
                else:
                    chunk_text = result_tuple
                    finish_reason = 'stop'
            except ImportError:
                result = thread_client.send(messages, temperature=temperature, max_tokens=None, context='review')
                elapsed = time.time() - start_time
                if isinstance(result, tuple):
                    chunk_text, finish_reason = result
                else:
                    chunk_text = result
                    finish_reason = 'stop'

            if not chunk_text or not chunk_text.strip():
                _safe_log(f"⚠️ Chunk {ci+1}: Empty response, skipping")
                return ci, None, elapsed

            _safe_log(f"✅ Chunk {ci+1} done in {elapsed:.1f}s (finish_reason: {finish_reason})")

            chunk_review = chunk_text.strip()
            if wrap_chunks:
                header = f"=== CHUNK REVIEW: {range_label} ==="
                footer = f"=== END CHUNK REVIEW: {range_label} ==="
                chunk_review = f"{header}\n\n{chunk_review}\n\n{footer}"

            return ci, chunk_review, elapsed

        except Exception as e:
            err_str = str(e).lower()
            if 'cancelled' in err_str or 'canceled' in err_str or 'stopped by user' in err_str:
                return ci, '__CANCELLED__', 0.0
            _safe_log(f"⚠️ Chunk {ci+1} failed: {e}")
            return ci, None, 0.0

    # Use ThreadPoolExecutor for parallel chunk processing
    from concurrent.futures import ThreadPoolExecutor, as_completed

    workers = max(1, batch_size)
    chunk_results = [None] * total_chunks  # Preserve order
    total_elapsed = 0.0
    cancelled = False

    # Signal batch mode to the API client so streaming log toggles are respected
    _prev_batch_translation = os.environ.get('BATCH_TRANSLATION')
    if workers > 1:
        os.environ['BATCH_TRANSLATION'] = '1'

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ChunkReview") as pool:
        futures = {
            pool.submit(_review_chunk, ci, chunk_chapters): ci
            for ci, chunk_chapters in enumerate(chunk_groups)
        }
        for future in as_completed(futures):
            ci = futures[future]
            try:
                idx, review_text, elapsed = future.result()
                total_elapsed += elapsed

                if review_text == '__CANCELLED__':
                    log_fn("🛑 Chunk review cancelled by user")
                    cancelled = True
                    # Cancel remaining futures
                    for f in futures:
                        f.cancel()
                    break

                chunk_results[idx] = review_text
            except Exception as e:
                log_fn(f"⚠️ Chunk {ci+1} future failed: {e}")

            if stop_check_fn and stop_check_fn():
                log_fn("🛑 Stopped by user")
                cancelled = True
                for f in futures:
                    f.cancel()
                break

    # Restore BATCH_TRANSLATION env var
    if _prev_batch_translation is None:
        os.environ.pop('BATCH_TRANSLATION', None)
    else:
        os.environ['BATCH_TRANSLATION'] = _prev_batch_translation

    if cancelled:
        return None

    # Filter out None results (failed/empty chunks) while preserving order
    chunk_reviews = [r for r in chunk_results if r is not None]

    if not chunk_reviews:
        log_fn("❌ No chunk reviews were generated")
        return None

    log_fn(f"\n{'═'*40}\n📊 {len(chunk_reviews)}/{total_chunks} chunk reviews collected ({total_elapsed:.1f}s total)\n{'═'*40}")

    if stop_check_fn and stop_check_fn():
        log_fn("🛑 Stopped by user")
        return None

    # 6. Final synthesis call
    log_fn(f"\n🔄 Sending Final Review synthesis prompt...")
    combined_chunks = "\n\n".join(chunk_reviews)
    combined_tokens = count_tokens(combined_chunks)
    log_fn(f"📤 Sending {combined_tokens:,} tokens of chunk reviews for synthesis...")

    final_messages = [
        {"role": "system", "content": final_review_prompt},
        {"role": "user", "content": combined_chunks},
    ]

    try:
        synthesis_client = _make_client()
        start_time = time.time()

        try:
            from TransateKRtoEN import send_with_interrupt
            result_tuple = send_with_interrupt(
                final_messages, synthesis_client, temperature=temperature, max_tokens=None,
                stop_check_fn=stop_check_fn, context='review',
            )
            elapsed = time.time() - start_time
            if isinstance(result_tuple, tuple) and len(result_tuple) == 3:
                final_text, finish_reason, _ = result_tuple
            elif isinstance(result_tuple, tuple) and len(result_tuple) == 2:
                final_text, finish_reason = result_tuple
            else:
                final_text = result_tuple
                finish_reason = 'stop'
        except ImportError:
            result = synthesis_client.send(final_messages, temperature=temperature, max_tokens=None, context='review')
            elapsed = time.time() - start_time
            if isinstance(result, tuple):
                final_text, finish_reason = result
            else:
                final_text = result
                finish_reason = 'stop'

        if not final_text or not final_text.strip():
            log_fn("❌ Empty response from final synthesis")
            # Fall back to concatenated chunk reviews
            final_text = combined_chunks
            log_fn("📝 Using concatenated chunk reviews as fallback")

        log_fn(f"✅ Final review synthesized in {elapsed:.1f}s (finish_reason: {finish_reason})")

    except Exception as e:
        err_str = str(e).lower()
        if 'cancelled' in err_str or 'canceled' in err_str or 'stopped by user' in err_str:
            log_fn("🛑 Final review stopped by user")
            return None
        log_fn(f"⚠️ Final synthesis failed: {e}")
        log_fn("📝 Using concatenated chunk reviews as fallback")
        final_text = combined_chunks

    # 7. Save
    review_dir = os.path.join(output_dir, "review")
    os.makedirs(review_dir, exist_ok=True)
    review_path = os.path.join(review_dir, "review.md")

    with open(review_path, 'w', encoding='utf-8') as f:
        f.write(final_text.strip())

    log_fn(f"💾 Review saved to: {review_path}")
    return final_text.strip()
