# TransateKRtoEN.py
# -*- coding: utf-8 -*-
import json
import logging
import shutil
import threading
import queue
import uuid
import inspect
import os, sys, io, zipfile, time, re, mimetypes, subprocess, tiktoken
import builtins
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup, NavigableString
try:
    from bs4 import XMLParsedAsHTMLWarning
    import warnings
    # Suppress the warning since we handle both HTML and XHTML content
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
except ImportError:
    # Older versions of BeautifulSoup might not have this warning
    pass
from collections import Counter
from unified_api_client import UnifiedClient, UnifiedClientError
import hashlib
import tempfile
import unicodedata
from difflib import SequenceMatcher
import unicodedata
import re
import time
from history_manager import HistoryManager
from chapter_splitter import ChapterSplitter
from image_translator import ImageTranslator
from typing import Dict, List, Tuple 
from txt_processor import TextFileProcessor
from ai_hunter_enhanced import ImprovedAIHunterDetection
import GlossaryManager  # Module with glossary functions
import csv
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# Module-level functions for ProcessPoolExecutor compatibility
from tqdm import tqdm

class ProgressBar:
    """Simple in-place progress bar for terminal output"""
    _last_line_length = 0
    
    @classmethod
    def update(cls, current, total, prefix="Progress", bar_length=30):
        """Update progress bar in-place
        
        Args:
            current: Current progress value
            total: Total value for 100% completion
            prefix: Text to show before the bar
            bar_length: Length of the progress bar in characters
        """
        if total == 0:
            return
            
        percent = min(100, int(100 * current / total))
        filled = int(bar_length * current / total)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        # Build the line
        line = f"\r{prefix}: [{bar}] {current}/{total} ({percent}%)"
        
        # Pad with spaces to clear previous line if it was longer
        if len(line) < cls._last_line_length:
            line += ' ' * (cls._last_line_length - len(line))
        
        cls._last_line_length = len(line)
        
        # Print without newline
        print(line, end='', flush=True)
    
    @classmethod
    def finish(cls):
        """Finish progress bar and move to next line"""
        print()  # Move to next line
        cls._last_line_length = 0

def is_traditional_translation_api(model: str) -> bool:
    """Check if the model is a traditional translation API"""
    return model in ['deepl', 'google-translate', 'google-translate-free'] or model.startswith('deepl/') or model.startswith('google-translate/')
    
def get_chapter_terminology(is_text_file, chapter_data=None):
    """Get appropriate terminology (Chapter/Section) based on source type"""
    if is_text_file:
        return "Section"
    if chapter_data:
        if chapter_data.get('filename', '').endswith('.txt') or chapter_data.get('is_chunk', False):
            return "Section"
    return "Chapter"


def extract_text_from_raw_content(raw_obj) -> str:
    """
    Safely extract human-readable text from a Gemini raw_content_object.
    Skips reasoning-only parts (thought=True) but preserves normal text.
    """
    try:
        parts = []
        if hasattr(raw_obj, 'parts'):
            parts = raw_obj.parts or []
        elif isinstance(raw_obj, dict):
            parts = raw_obj.get('parts', []) or []

        texts = []
        for p in parts:
            is_thought = False
            text_val = None

            if hasattr(p, 'thought'):
                is_thought = bool(getattr(p, 'thought', False))
            elif isinstance(p, dict):
                is_thought = bool(p.get('thought', False))

            if hasattr(p, 'text'):
                text_val = getattr(p, 'text', None)
            elif isinstance(p, dict):
                text_val = p.get('text')

            if text_val and not is_thought:
                texts.append(str(text_val))

        return "\n".join(texts).strip()
    except Exception:
        return ""


def build_gemini_model_message(content: str = "", raw_obj=None) -> dict:
    """
    Build a Gemini 3-compatible assistant-role message with parts:
      - text part (when available)
      - thought_signature part (when available)
    Using assistant keeps roles valid while preserving parts for Gemini 3.
    """
    import base64

    parts = []

    # Prefer text from raw_obj parts if present; else use provided content
    text_added = False
    if raw_obj:
        candidate_parts = []
        if hasattr(raw_obj, "parts"):
            candidate_parts = raw_obj.parts or []
        elif isinstance(raw_obj, dict):
            candidate_parts = raw_obj.get("parts", []) or []

        for p in candidate_parts:
            if hasattr(p, "text") and getattr(p, "text", None):
                parts.append({"text": str(getattr(p, "text"))})
                text_added = True
            elif isinstance(p, dict) and p.get("text"):
                parts.append({"text": str(p.get("text"))})
                text_added = True

    if content and not text_added:
        parts.append({"text": str(content)})

    # Find thought signature (snake or camel case, bytes or dict)
    sig_bytes = None
    if raw_obj:
        def _extract_sig_from_part(part):
            ts = None
            if hasattr(part, "thought_signature"):
                ts = getattr(part, "thought_signature", None)
            elif hasattr(part, "thoughtSignature"):
                ts = getattr(part, "thoughtSignature", None)
            elif isinstance(part, dict):
                ts = part.get("thought_signature") or part.get("thoughtSignature")
            return ts

        # Check top-level then parts
        top_ts = None
        if isinstance(raw_obj, dict):
            top_ts = raw_obj.get("thought_signature") or raw_obj.get("thoughtSignature")
        if hasattr(raw_obj, "thought_signature"):
            top_ts = getattr(raw_obj, "thought_signature", None)
        if hasattr(raw_obj, "thoughtSignature"):
            top_ts = getattr(raw_obj, "thoughtSignature", None)
        if top_ts is not None:
            sig_bytes = top_ts
        else:
            cand_parts = []
            if hasattr(raw_obj, "parts"):
                cand_parts = raw_obj.parts or []
            elif isinstance(raw_obj, dict):
                cand_parts = raw_obj.get("parts", []) or []
            for p in cand_parts:
                ts = _extract_sig_from_part(p)
                if ts is not None:
                    sig_bytes = ts
                    break

    if sig_bytes is not None:
        if isinstance(sig_bytes, dict) and sig_bytes.get("_type") == "bytes" and sig_bytes.get("data"):
            data_b64 = sig_bytes.get("data")
        elif isinstance(sig_bytes, (bytes, bytearray)):
            data_b64 = base64.b64encode(sig_bytes).decode("utf-8")
        else:
            # If provided as string (already b64) keep as-is
            data_b64 = str(sig_bytes)
        parts.append({"thought_signature": {"_type": "bytes", "data": data_b64}})

    # Fallback to text-only part if nothing found
    if not parts and content:
        parts.append({"text": str(content)})

    return {"role": "assistant", "parts": parts} if parts else {"role": "assistant", "parts": []}

def _merge_split_paragraphs(html_body: str) -> str:
    """Merge paragraphs that were artificially split across PDF pages.
    
    PDFs are extracted page-by-page, which can split paragraphs mid-sentence.
    This function merges consecutive justified paragraphs that don't end with
    sentence-ending punctuation, creating more natural paragraph breaks.
    
    Only affects PDFs, not EPUBs.
    """
    from bs4 import BeautifulSoup
    
    soup = BeautifulSoup(html_body, 'html.parser')
    
    # Find all <p> tags
    paragraphs = soup.find_all('p')
    
    if len(paragraphs) < 2:
        return html_body  # Nothing to merge
    
    # Process paragraphs and merge when appropriate
    i = 0
    while i < len(paragraphs) - 1:
        current_p = paragraphs[i]
        next_p = paragraphs[i + 1]
        
        # Skip if either is None or not a tag
        if not current_p or not next_p:
            i += 1
            continue
        
        # Get paragraph classes - only merge justified paragraphs
        current_class = current_p.get('class', [])
        next_class = next_p.get('class', [])
        
        current_is_justified = 'align-justify' in current_class if current_class else False
        next_is_justified = 'align-justify' in next_class if next_class else False
        
        # Only merge if both are justified (regular body text)
        if not (current_is_justified and next_is_justified):
            i += 1
            continue
        
        # Get text content of current paragraph
        current_text = current_p.get_text().strip()
        
        # Check if current paragraph ends with sentence-ending punctuation
        ends_with_sentence = bool(re.search(r'[.!?]\s*$', current_text))
        
        # Check if next paragraph looks like continuation (doesn't start with capital)
        next_text = next_p.get_text().strip()
        starts_with_capital = bool(re.match(r'^[A-Z"\(]', next_text)) if next_text else False
        
        # Merge if:
        # - Current doesn't end with sentence punctuation, OR
        # - Current ends with sentence but next doesn't start with capital (likely continuation)
        should_merge = not ends_with_sentence or (ends_with_sentence and not starts_with_capital)
        
        if should_merge:
            # Merge next paragraph's content into current
            # Add a space between them
            current_p.append(' ')
            for content in list(next_p.contents):
                try:
                    current_p.append(content.extract())
                except Exception:
                    current_p.append(content)
            
            # Remove the next paragraph
            next_p.decompose()
            
            # Update list and continue without increment to consider further merges
            paragraphs = soup.find_all('p')
            continue
        else:
            # Can't merge, move to next pair
            i += 1
    
    # Use decode() instead of str() to preserve original formatting and attributes
    return soup.decode(formatter='minimal')

def _merge_image_only_pages(html_body: str) -> str:
    """Merge image-only extracted PDF page containers into the previous container.

    Motivation: When PDFs are extracted page-by-page, some pages contain only a single image.
    Keeping them as a standalone container often produces large wasted whitespace in the final
    PDF/HTML output. This pass moves the image(s) into the previous page container.

    We treat a container as "image-only" if:
      - it contains at least one <img>
      - its visible text (after stripping whitespace/nbsp) is empty

    This is a best-effort layout hint; the renderer may still paginate based on available space.
    """
    try:
        from bs4 import BeautifulSoup
        import re as _re

        soup = BeautifulSoup(html_body, 'html.parser')

        # Common page wrapper IDs produced by our pipeline / MuPDF
        id_pat = _re.compile(r'^(?:mupdf-page0-\d+|page\d+|page0)$')

        def _is_image_only(div) -> bool:
            if not div:
                return False
            imgs = div.find_all('img')
            if not imgs:
                return False
            txt = (div.get_text(' ', strip=True) or '').replace('\xa0', '').strip()
            return txt == ''

        changed = True
        while changed:
            changed = False
            divs = soup.find_all('div', id=id_pat)
            for idx in range(1, len(divs)):
                div = divs[idx]
                if not _is_image_only(div):
                    continue
                prev = div.find_previous('div', id=id_pat)
                if not prev:
                    continue

                # Move children into previous container
                for child in list(div.contents):
                    try:
                        prev.append(child.extract())
                    except Exception:
                        prev.append(child)

                div.decompose()
                changed = True
                break  # restart scan since tree changed

        return soup.decode(formatter='minimal')
    except Exception:
        return html_body


def _keep_text_with_following_image(html_body: str, *, min_text_chars: int = 40) -> str:
    """Reduce image-only PDF pages by keeping the last text block together with the following image.

    If an image doesn't fit at the bottom of a page, renderers will push it to the next page,
    sometimes resulting in a page that contains only the image. By wrapping the last text block
    immediately before an image together with that image in a container that avoids page breaks
    inside, the renderer will move BOTH to the next page when needed.

    This intentionally trades some extra whitespace on the previous page to avoid image-only pages.
    """
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html_body, 'html.parser')

        # Target <p><img ...></p> blocks (most of your extracted images are in this shape)
        for p in soup.find_all('p'):
            imgs = p.find_all('img')
            if len(imgs) != 1:
                continue
            # Ensure this <p> is basically image-only
            txt = (p.get_text(' ', strip=True) or '').replace('\xa0', '').strip()
            if txt:
                continue

            # Find a preceding text block sibling (h1-h6 or p with text)
            prev = p.find_previous_sibling(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p'])
            if not prev:
                continue
            prev_txt = (prev.get_text(' ', strip=True) or '').replace('\xa0', '').strip()
            if len(prev_txt) < min_text_chars:
                continue

            # Wrap prev + image-paragraph together
            wrapper = soup.new_tag('div')
            wrapper['class'] = (wrapper.get('class', []) or []) + ['keep-with-image']
            wrapper['style'] = 'break-inside:avoid; page-break-inside:avoid;'

            prev.insert_before(wrapper)
            wrapper.append(prev.extract())
            wrapper.append(p.extract())

        return soup.decode(formatter='minimal')
    except Exception:
        return html_body


def _generate_and_replace_toc(html_body: str) -> str:
    """Generate a proper table of contents from headers and replace any existing broken TOC.
    
    Only affects PDFs, not EPUBs.
    """
    from bs4 import BeautifulSoup
    
    soup = BeautifulSoup(html_body, 'html.parser')
    
    # Find all h1 and h2 headers (skip those in first 3 pages/divs as they're likely title page)
    headers = []
    all_divs = soup.find_all('div', id=lambda x: x and x.startswith('page'))
    
    # Start collecting headers after the first 3 pages
    for div in all_divs[3:] if len(all_divs) > 3 else []:
        for header in div.find_all(['h1', 'h2']):
            header_text = header.get_text().strip()
            if header_text and len(header_text) > 2:  # Skip very short headers
                # Create anchor ID
                if not header.get('id'):
                    anchor_id = re.sub(r'[^a-zA-Z0-9]+', '-', header_text[:50].lower()).strip('-')
                    header['id'] = anchor_id
                else:
                    anchor_id = header['id']
                
                headers.append({
                    'text': header_text,
                    'id': anchor_id,
                    'level': int(header.name[1])  # h1 -> 1, h2 -> 2
                })
    
    # If we found headers, generate TOC
    if headers:
        # Build TOC HTML
        toc_html = '<div class="toc" style="text-align:center;margin:2em 0;">\n'
        toc_html += '<h1 style="text-align:center!important;">Table of Contents</h1>\n'
        
        for h in headers:
            indent = '' if h['level'] == 1 else '&nbsp;&nbsp;&nbsp;&nbsp;'
            toc_html += f'<p style="text-align:center!important;margin:0.5em 0;"><a href="#{h["id"]}">{indent}{h["text"]}</a></p>\n'
        
        toc_html += '</div>\n'
        
        # Search for existing TOC by looking for "Table of Contents" or "Contents" text
        toc_replaced = False
        
        # Method 1: Search for any element containing "Table of Contents" text
        for element in soup.find_all(string=re.compile(r'table of contents|^contents$', re.IGNORECASE)):
            # Find the containing page div
            page_div = element.find_parent('div', id=lambda x: x and x.startswith('page'))
            if page_div:
                page_div.clear()
                page_div.append(BeautifulSoup(toc_html, 'html.parser'))
                toc_replaced = True
                print(f"   • Replaced broken TOC with generated TOC ({len(headers)} entries)")
                break
        
        # Method 2: If not found by text, check page divs for TOC-like content
        if not toc_replaced:
            for i, div in enumerate(all_divs[:10]):  # Check first 10 pages
                div_text = div.get_text().lower().strip()
                # Check if this looks like a TOC page (has "contents" early in the page)
                if ('table of contents' in div_text or 
                    (div_text.startswith('contents') or 'contents' in div_text[:100])):
                    # Replace entire div content with new TOC
                    div.clear()
                    div.append(BeautifulSoup(toc_html, 'html.parser'))
                    toc_replaced = True
                    print(f"   • Replaced broken TOC on page {i+1} with generated TOC ({len(headers)} entries)")
                    break
    
    # Use decode() instead of str() to preserve original formatting and attributes
    return soup.decode(formatter='minimal')
# =====================================================
# CONFIGURATION AND ENVIRONMENT MANAGEMENT
# =====================================================
class TranslationConfig:
    """Centralized configuration management"""
    def __init__(self):
        self.MODEL = os.getenv("MODEL", "gemini-1.5-flash")
        self.input_path = os.getenv("input_path", "default.epub")
        self.PROFILE_NAME = os.getenv("PROFILE_NAME", "korean").lower()
        self.CONTEXTUAL = os.getenv("CONTEXTUAL", "1") == "1"
        self.DELAY = float(os.getenv("SEND_INTERVAL_SECONDS", "1"))
        self.SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "").strip()
        self.REQUEST_MERGING_ENABLED = os.getenv("REQUEST_MERGING_ENABLED", "0") == "1"
        
        # Handle split marker instruction placeholder
        if self.SYSTEM_PROMPT:
            import re
            if self.REQUEST_MERGING_ENABLED:
                split_instr = "- CRITICAL Requirement: If you see any HTML tags containing 'SPLIT MARKER' (Example: <h1 id=\"split-1\">SPLIT MARKER: Do Not Remove This Tag</h1>), you MUST preserve them EXACTLY as they appear. Do not translate, modify, or remove these markers."
                # Replace placeholder with instruction (handling potential newlines if user added them, though usually we want to ensure it has its own line)
                # We simply replace the placeholder. The prompt template likely has newlines around it.
                self.SYSTEM_PROMPT = self.SYSTEM_PROMPT.replace("{split_marker_instruction}", split_instr)
            else:
                # Strip placeholder if merging is disabled
                self.SYSTEM_PROMPT = re.sub(r'\s*\{split_marker_instruction\}\s*', '', self.SYSTEM_PROMPT)
            
        self.REMOVE_AI_ARTIFACTS = os.getenv("REMOVE_AI_ARTIFACTS", "0") == "1"
        self.TEMP = float(os.getenv("TRANSLATION_TEMPERATURE", "0.3"))
        self.HIST_LIMIT = int(os.getenv("TRANSLATION_HISTORY_LIMIT", "20"))
        self.MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "8192"))
        self.EMERGENCY_RESTORE = os.getenv("EMERGENCY_PARAGRAPH_RESTORE", "1") == "1"
        self.BATCH_TRANSLATION = os.getenv("BATCH_TRANSLATION", "0") == "1"  
        self.BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))
        self.BATCHING_MODE = os.getenv("BATCHING_MODE", "aggressive")
        self.BATCH_GROUP_SIZE = int(os.getenv("BATCH_GROUP_SIZE", os.getenv("CONSERVATIVE_BATCH_GROUP_SIZE", "3")))
        self.REQUEST_MERGING_ENABLED = os.getenv("REQUEST_MERGING_ENABLED", "0") == "1"
        self.REQUEST_MERGE_COUNT = int(os.getenv("REQUEST_MERGE_COUNT", "3"))
        # Synthetic header injection for merged requests (Split-the-Merge helper)
        self.SYNTHETIC_MERGE_HEADERS = os.getenv("SYNTHETIC_MERGE_HEADERS", "1") == "1"
        self.ENABLE_IMAGE_TRANSLATION = os.getenv("ENABLE_IMAGE_TRANSLATION", "1") == "1"
        
        # Auto-disable image translation for html2text and BeautifulSoup profiles
        # These profiles are designed for text extraction and don't need image translation
        if self.ENABLE_IMAGE_TRANSLATION and self.PROFILE_NAME:
            profile_lower = self.PROFILE_NAME.lower()
            if 'html2text' in profile_lower or 'beautifulsoup' in profile_lower:
                self.ENABLE_IMAGE_TRANSLATION = False
                print(f"ℹ️  Image translation disabled for {self.PROFILE_NAME} profile")
        
        self.TRANSLATE_BOOK_TITLE = os.getenv("TRANSLATE_BOOK_TITLE", "1") == "1"
        self.DISABLE_ZERO_DETECTION = os.getenv("DISABLE_ZERO_DETECTION", "0") == "1"
        self.ENABLE_AUTO_GLOSSARY = os.getenv("ENABLE_AUTO_GLOSSARY", "0") == "1"
        self.COMPREHENSIVE_EXTRACTION = os.getenv("COMPREHENSIVE_EXTRACTION", "0") == "1"
        self.MANUAL_GLOSSARY = os.getenv("MANUAL_GLOSSARY")
        self.RETRY_TRUNCATED = os.getenv("RETRY_TRUNCATED", "1") == "1"
        try:
            self.TRUNCATION_RETRY_ATTEMPTS = int(os.getenv("TRUNCATION_RETRY_ATTEMPTS", "1"))
        except Exception:
            self.TRUNCATION_RETRY_ATTEMPTS = 1
        self.RETRY_SPLIT_FAILED = os.getenv("RETRY_SPLIT_FAILED", "0") == "1"
        try:
            self.SPLIT_FAILED_RETRY_ATTEMPTS = int(os.getenv("SPLIT_FAILED_RETRY_ATTEMPTS", "1"))
        except Exception:
            self.SPLIT_FAILED_RETRY_ATTEMPTS = 1
        self.RETRY_DUPLICATE_BODIES = os.getenv("RETRY_DUPLICATE_BODIES", "1") == "1"
        self.RETRY_TIMEOUT = os.getenv("RETRY_TIMEOUT", "0") == "1"
        self.CHUNK_TIMEOUT = int(os.getenv("CHUNK_TIMEOUT", "900"))
        self.DISABLE_MERGE_FALLBACK = os.getenv("DISABLE_MERGE_FALLBACK", "0") == "1"
        self.MAX_RETRY_TOKENS = int(os.getenv("MAX_RETRY_TOKENS", "16384"))
        self.DUPLICATE_LOOKBACK_CHAPTERS = int(os.getenv("DUPLICATE_LOOKBACK_CHAPTERS", "3"))
        self.USE_ROLLING_SUMMARY = os.getenv("USE_ROLLING_SUMMARY", "0") == "1"
        self.ROLLING_SUMMARY_EXCHANGES = int(os.getenv("ROLLING_SUMMARY_EXCHANGES", "5"))
        self.ROLLING_SUMMARY_MODE = os.getenv("ROLLING_SUMMARY_MODE", "replace")
        # New: maximum number of rolling summary entries to retain when in append mode (0 = unlimited)
        self.ROLLING_SUMMARY_MAX_ENTRIES = int(os.getenv("ROLLING_SUMMARY_MAX_ENTRIES", "10"))
        self.DUPLICATE_DETECTION_MODE = os.getenv("DUPLICATE_DETECTION_MODE", "basic")
        self.AI_HUNTER_THRESHOLD = int(os.getenv("AI_HUNTER_THRESHOLD", "75"))
        self.TRANSLATION_HISTORY_ROLLING = os.getenv("TRANSLATION_HISTORY_ROLLING", "0") == "1"
        self.API_KEY = (os.getenv("API_KEY") or 
                       os.getenv("OPENAI_API_KEY") or 
                       os.getenv("OPENAI_OR_Gemini_API_KEY") or
                       os.getenv("GEMINI_API_KEY"))
        # NEW: Simple chapter number offset
        self.CHAPTER_NUMBER_OFFSET = int(os.getenv("CHAPTER_NUMBER_OFFSET", "0"))
        self.ENABLE_WATERMARK_REMOVAL = os.getenv("ENABLE_WATERMARK_REMOVAL", "1") == "1"
        self.SAVE_CLEANED_IMAGES = os.getenv("SAVE_CLEANED_IMAGES", "1") == "1"
        self.EMERGENCY_IMAGE_RESTORE = os.getenv("EMERGENCY_IMAGE_RESTORE", "0") == "1"
        self.WATERMARK_PATTERN_THRESHOLD = int(os.getenv("WATERMARK_PATTERN_THRESHOLD", "10"))
        self.WATERMARK_CLAHE_LIMIT = float(os.getenv("WATERMARK_CLAHE_LIMIT", "3.0"))
        self.COMPRESSION_FACTOR = float(os.getenv("COMPRESSION_FACTOR", "2.0"))
        
        # Multi API key support
        self.use_multi_api_keys = os.environ.get('USE_MULTI_API_KEYS', '0') == '1'
        self.multi_api_keys = []
        
        if self.use_multi_api_keys:
            multi_keys_json = os.environ.get('MULTI_API_KEYS', '[]')
            try:
                self.multi_api_keys = json.loads(multi_keys_json)
                print(f"Loaded {len(self.multi_api_keys)} API keys for multi-key mode")
            except Exception as e:
                print(f"Failed to load multi API keys: {e}")
                self.use_multi_api_keys = False
        
        # Fallback keys (for direct fallback retries)
        self.use_fallback_keys = os.environ.get('USE_FALLBACK_KEYS', '0') == '1'
        self.fallback_keys = []
        if self.use_fallback_keys:
            fk_json = os.environ.get('FALLBACK_KEYS', '[]')
            try:
                self.fallback_keys = json.loads(fk_json)
            except Exception as e:
                print(f"Failed to load fallback keys: {e}")
                self.use_fallback_keys = False

    def get_effective_output_limit(self) -> int:
        """Return the effective output token limit, considering per-key overrides.

        - Start from the global MAX_OUTPUT_TOKENS.
        - Check if the model has a discovered limit (from auto-adjustment)
        - If multi-key mode is enabled, intersect with any per-key
          individual_output_token_limit values (min of all >0 limits).
        - If fallback keys are enabled, also intersect with their per-key
          individual_output_token_limit values.
        """
        effective = self.MAX_OUTPUT_TOKENS
        
        # Check if we've discovered a model limit via auto-adjustment
        try:
            from unified_api_client import UnifiedClient
            with UnifiedClient._model_limits_lock:
                cached_limit = UnifiedClient._model_token_limits.get(self.MODEL)
                if cached_limit and cached_limit < effective:
                    effective = cached_limit
        except Exception:
            pass

        # Collect per-key limits from multi-key pool (only from enabled keys)
        per_key_limits = []
        try:
            for idx, key_data in enumerate(self.multi_api_keys or []):
                if not isinstance(key_data, dict):
                    continue
                # Skip disabled keys
                if not key_data.get('enabled', True):
                    continue
                raw = key_data.get('individual_output_token_limit')
                if raw in (None, "", 0):
                    continue
                try:
                    val = int(raw)
                    if val > 0:
                        per_key_limits.append(val)
                except Exception:
                    continue
        except Exception:
            pass

        # Collect per-key limits from fallback keys (only from enabled keys)
        try:
            for idx, fb in enumerate(self.fallback_keys or []):
                if not isinstance(fb, dict):
                    continue
                # Skip disabled keys
                if not fb.get('enabled', True):
                    continue
                raw = fb.get('individual_output_token_limit')
                if raw in (None, "", 0):
                    continue
                try:
                    val = int(raw)
                    if val > 0:
                        per_key_limits.append(val)
                except Exception:
                    continue
        except Exception:
            pass

        if per_key_limits:
            effective = min(effective, min(per_key_limits))

        return effective


# =====================================================
# REQUEST MERGING UTILITIES
# =====================================================
class RequestMerger:
    """Handles merging multiple chapters into a single request"""
    
    @classmethod
    def merge_chapters(cls, chapters_data, log_injections=True):
        """Merge multiple chapters into a single content block.

        This is used both for request-size estimation and for the actual
        merged request that is sent to the API.

        Before concatenating, we inject an invisible split marker at the
        beginning of each chapter. This greatly improves the reliability of
        Split-the-Merge, because the splitter can simply find these markers
        instead of carefully parsing headers.

        Args:
            chapters_data: List of tuples (chapter_num, content, chapter_obj)
            log_injections: If False, perform marker injection silently
                (no console logging). Used for size-estimation previews to
                avoid duplicate log lines.

        Returns:
            Merged content string
        """
        if not chapters_data:
            return ""

        # Split markers are only needed when split-the-merge is enabled
        # Check if the feature is turned on
        split_the_merge_enabled = os.getenv('SPLIT_THE_MERGE', '0') == '1'
        split_markers_enabled = split_the_merge_enabled

        merged_parts = []

        for chapter_num, content, chapter_obj in chapters_data:
            # Defensive: if something goes wrong in the marker injection
            # logic, fall back to the original content rather than breaking
            # the whole merge.
            try:
                if isinstance(content, str):
                    # Only add split markers if split-the-merge is enabled
                    if split_markers_enabled:
                        # Use H1 tag as split marker - AI will preserve visible HTML elements
                        split_marker = f'<h1 id="split-{chapter_num}">SPLIT MARKER: Do Not Remove This Tag</h1>\n'
                        marked_content = split_marker + content
                        
                        if log_injections:
                            preview = marked_content[:120].replace('\n', ' ')
                            print(
                                f"   ℹ️ Request Merging: Injected H1 split marker for "
                                f"chapter {chapter_num}: {preview}..."
                            )
                        
                        merged_parts.append(marked_content)
                    else:
                        # No split markers - just append content as-is
                        merged_parts.append(content)
                else:
                    # Non-string content, just append as-is
                    merged_parts.append(content)
                
            except Exception as e:
                # Fallback: append original content if anything goes wrong
                if log_injections:
                    print(f"   ⚠️ Request Merging: Failed to inject split marker for chapter {chapter_num}: {e}")
                merged_parts.append(content)

        return "\n\n".join(merged_parts)
    
    @classmethod
    def create_merge_groups(cls, chapters_to_translate, merge_count):
        """Group chapters into merge groups, keeping only nearby chapters together.

        This prevents cases like chapter 7 being merged with chapter 29 just
        because chapters 8–28 were already translated or merged earlier.

        Args:
            chapters_to_translate: List of tuples. Supported shapes:
                - (idx, chapter_obj)
                - (idx, chapter_obj, actual_num, ...)
            merge_count: Maximum number of chapters to merge per request.

        Returns:
            List of merge groups, each group is a list of chapter tuples taken
            from ``chapters_to_translate`` in order.
        """
        if merge_count <= 1 or not chapters_to_translate:
            # No merging, return each chapter as its own group
            return [[ch] for ch in chapters_to_translate]

        def _get_actual_num(item):
            """Best-effort extraction of the logical chapter number for grouping.

            This is primarily used as a *display* / fallback value. For actual
            proximity checks we prefer OPF spine order when available (see
            ``_get_proximity_key`` below).

            We try, in order:
            1. Explicit ``actual_num`` in position 2 (non-text merge path).
            2. ``chapter_obj['actual_chapter_num']`` if present.
            3. ``chapter_obj['num']``.
            4. Fallback to idx (position 0).
            """
            # Shape: (idx, chapter_obj, actual_num, ...)
            try:
                if len(item) >= 3 and isinstance(item[2], (int, float)):
                    return item[2]
            except Exception:
                pass

            # Shape: (idx, chapter_obj)
            try:
                chapter_obj = item[1]
                if isinstance(chapter_obj, dict):
                    if 'actual_chapter_num' in chapter_obj:
                        return chapter_obj.get('actual_chapter_num')
                    return chapter_obj.get('num')
            except Exception:
                pass

            # Fallback: idx
            try:
                return item[0]
            except Exception:
                return None

        def _get_proximity_key(item):
            """Return a numeric key representing *reading order* proximity.

            We want proximity to reflect where chapters sit in the *book* rather
            than their logical numbering, so that multiple files with the same
            chapter number (e.g. notice pages vs. main text) don't get merged
            just because their labels are "4, 5, 4".

            Strategy (in order):
            1. Use ``spine_order`` or ``opf_spine_position`` if present on the
               chapter object (true reading order from content.opf).
            2. Fall back to the chapter index ``idx`` (position 0 in the tuple),
               which preserves the original ordering of the ``chapters`` list.
            3. As a last resort, fall back to ``_get_actual_num``.
            """
            # 1) Prefer explicit spine-based order from OPF if available
            try:
                chapter_obj = item[1]
                if isinstance(chapter_obj, dict):
                    spine_pos = chapter_obj.get('spine_order')
                    if spine_pos is None:
                        spine_pos = chapter_obj.get('opf_spine_position')
                    if spine_pos is not None:
                        return float(spine_pos)
            except Exception:
                pass

            # 2) Fall back to the chapter's index in the master chapter list.
            # ``idx`` is stored in position 0 in all supported shapes.
            try:
                return float(item[0])
            except Exception:
                pass

            # 3) Ultimate fallback – use the logical chapter number.
            return _get_actual_num(item)

        groups = []
        current_group = []
        prev_num = None

        for ch in chapters_to_translate:
            # Use proximity key (spine order when available) instead of the
            # logical chapter number alone. This prevents far‑apart chapters
            # with the same numeric label (e.g. multiple "Ch.004" entries in
            # different parts of the book) from being merged together when
            # there are many intervening chapters in the OPF spine.
            current_num = _get_proximity_key(ch)

            if not current_group:
                # Start the first group
                current_group = [ch]
                prev_num = current_num
                continue

            # If we've hit the per-request limit, start a new group
            if len(current_group) >= merge_count:
                groups.append(current_group)
                current_group = [ch]
                prev_num = current_num
                continue

            # If we can't safely determine chapter numbers, be conservative and
            # start a new group so we never merge far‑apart chapters by accident.
            if current_num is None or prev_num is None:
                groups.append(current_group)
                current_group = [ch]
                prev_num = current_num
                continue

            # Only merge if chapters are numerically adjacent (or effectively so).
            # This means sequences like 1→2→3 will merge, but 1→4 will not.
            try:
                gap = abs(float(current_num) - float(prev_num))
            except Exception:
                gap = None

            if gap is not None and gap <= 1:
                # Close enough in chapter numbering, keep in same group
                current_group.append(ch)
            else:
                # Too far apart (e.g. 7 then 29) → start a new group
                groups.append(current_group)
                current_group = [ch]

            prev_num = current_num

        if current_group:
            groups.append(current_group)

        return groups
    
    @classmethod
    def split_by_markers(cls, content, expected_count):
        """
        Split merged translation output by split markers.
        
        This method is robust to broken or missing split tags:
        - Handles partial marker tags (e.g., missing closing tag)
        - Handles malformed id attributes
        - Falls back to ANY h1 tag if split markers are missing
        - Works even if some markers are completely missing
        
        Args:
            content: The translated HTML content
            expected_count: Expected number of sections (should match merged chapter count)
            
        Returns:
            List of content sections if we can reliably split,
            or None if splitting is not possible (fallback to normal merged behavior)
        """
        import re
        from bs4 import BeautifulSoup
        
        # Try multiple strategies in order of reliability:
        # 1. Perfect split markers with proper id="split-N"
        # 2. Any h1 tag with "split" in the id (even broken)
        # 3. Any h1 tag containing "SPLIT MARKER" text
        # 4. Any h1 tag at all
        
        # Strategy 1: Perfect markers
        perfect_pattern = r'<h1[^>]*id="split-\d+"[^>]*>.*?</h1>'
        perfect_markers = list(re.finditer(perfect_pattern, content, flags=re.DOTALL | re.IGNORECASE))
        
        if len(perfect_markers) == expected_count:
            print(f"   ✓️ Split the Merge: Found {len(perfect_markers)} perfect split markers")
            return cls._split_by_positions(content, [m.start() for m in perfect_markers])
        
        print(f"   ⚠️ Split the Merge: Found {len(perfect_markers)} perfect markers, expected {expected_count}. Trying fallback strategies...")
        
        # Strategy 2: Broken markers with "split" in id (handles broken closing tags, etc.)
        try:
            soup = BeautifulSoup(content, 'html.parser')
            h1_tags = soup.find_all('h1')
            
            # Try markers with "split" in id
            split_id_tags = [tag for tag in h1_tags if tag.get('id') and 'split' in tag.get('id', '').lower()]
            
            if len(split_id_tags) == expected_count:
                print(f"   ✓️ Split the Merge: Found {len(split_id_tags)} h1 tags with 'split' in id (broken marker format)")
                positions = []
                for tag in split_id_tags:
                    # Find position of this tag in original content
                    tag_str = str(tag)
                    # Search for the opening tag
                    opening_tag = re.escape(tag_str.split('>')[0] + '>')
                    match = re.search(opening_tag, content, flags=re.IGNORECASE)
                    if match:
                        positions.append(match.start())
                
                if len(positions) == expected_count:
                    return cls._split_by_positions(content, sorted(positions))
        except Exception as e:
            print(f"   ⚠️ Split the Merge: BeautifulSoup fallback failed: {e}")
        
        # Strategy 3: H1 tags containing "SPLIT MARKER" text
        try:
            soup = BeautifulSoup(content, 'html.parser')
            h1_tags = soup.find_all('h1')
            
            marker_text_tags = [tag for tag in h1_tags if 'split marker' in tag.get_text().lower()]
            
            if len(marker_text_tags) == expected_count:
                print(f"   ✓️ Split the Merge: Found {len(marker_text_tags)} h1 tags with 'SPLIT MARKER' text")
                positions = []
                for tag in marker_text_tags:
                    tag_str = str(tag)
                    opening_tag = re.escape(tag_str.split('>')[0] + '>')
                    match = re.search(opening_tag, content, flags=re.IGNORECASE)
                    if match:
                        positions.append(match.start())
                
                if len(positions) == expected_count:
                    return cls._split_by_positions(content, sorted(positions))
        except Exception as e:
            print(f"   ⚠️ Split the Merge: Text marker fallback failed: {e}")
        
        # All strategies failed
        print(f"   ❌ Split the Merge: Could not reliably split content (found varying marker counts across strategies)")
        return None
    
    @classmethod
    def _split_by_positions(cls, content, positions):
        """
        Helper to split content at specific character positions.
        
        Args:
            content: Full content string
            positions: List of character positions where splits should occur (sorted)
            
        Returns:
            List of content sections
        """
        if not positions:
            return [content]
        
        sections = []
        
        # First section is before the first marker (usually empty/whitespace)
        first_section = content[:positions[0]].strip()
        if first_section:  # Only include if non-empty
            sections.append(first_section)
        
        # Middle sections between markers
        for i in range(len(positions) - 1):
            # Find where the actual content starts (after the marker tag)
            start_pos = positions[i]
            # Skip past the h1 tag
            marker_end = content.find('</h1>', start_pos)
            if marker_end != -1:
                content_start = marker_end + 5  # len('</h1>')
            else:
                # Broken closing tag, try to skip past the opening tag at least
                next_close_bracket = content.find('>', start_pos)
                content_start = next_close_bracket + 1 if next_close_bracket != -1 else start_pos
            
            section = content[content_start:positions[i + 1]].strip()
            sections.append(section)
        
        # Last section after the last marker
        last_marker_pos = positions[-1]
        marker_end = content.find('</h1>', last_marker_pos)
        if marker_end != -1:
            content_start = marker_end + 5
        else:
            next_close_bracket = content.find('>', last_marker_pos)
            content_start = next_close_bracket + 1 if next_close_bracket != -1 else last_marker_pos
        
        last_section = content[content_start:].strip()
        sections.append(last_section)
        
        print(f"   ✓️ Split the Merge: Successfully split into {len(sections)} sections")
        return sections


# =====================================================
# UNIFIED PATTERNS AND CONSTANTS
# =====================================================
class PatternManager:
    """Centralized pattern management"""
    
    CHAPTER_PATTERNS = [
        # English patterns
        (r'chapter[\s_-]*(\d+)', re.IGNORECASE, 'english_chapter'),
        (r'\bch\.?\s*(\d+)\b', re.IGNORECASE, 'english_ch'),
        (r'part[\s_-]*(\d+)', re.IGNORECASE, 'english_part'),
        (r'episode[\s_-]*(\d+)', re.IGNORECASE, 'english_episode'),
        # Chinese patterns
        (r'第\s*(\d+)\s*[章节話话回]', 0, 'chinese_chapter'),
        (r'第\s*([一二三四五六七八九十百千万]+)\s*[章节話话回]', 0, 'chinese_chapter_cn'),
        (r'(\d+)[章节話话回]', 0, 'chinese_short'),
        # Japanese patterns
        (r'第\s*(\d+)\s*話', 0, 'japanese_wa'),
        (r'第\s*(\d+)\s*章', 0, 'japanese_chapter'),
        (r'その\s*(\d+)', 0, 'japanese_sono'),
        (r'(\d+)話目', 0, 'japanese_wame'),
        # Korean patterns
        (r'제\s*(\d+)\s*[장화권부편]', 0, 'korean_chapter'),
        (r'(\d+)\s*[장화권부편]', 0, 'korean_short'),
        (r'에피소드\s*(\d+)', 0, 'korean_episode'),
        # Generic numeric patterns
        (r'^\s*(\d+)\s*[-–—.\:]', re.MULTILINE, 'generic_numbered'),
        (r'_(\d+)\.x?html?$', re.IGNORECASE, 'filename_number'),
        (r'/(\d+)\.x?html?$', re.IGNORECASE, 'path_number'),
        (r'(\d+)', 0, 'any_number'),
    ]
    
    FILENAME_EXTRACT_PATTERNS = [
        # IMPORTANT: More specific patterns MUST come first
        r'^\d{3}(\d)_(\d{2})_\.x?html?$', # Captures both parts for decimal: group1.group2
        r'^\d{4}_(\d+)\.x?html?$',  # "0000_1.xhtml" - extracts 1, not 0000
        r'^\d+_(\d+)[_\.]',         # Any digits followed by underscore then capture next digits
        r'^(\d+)[_\.]',             # Standard: "0249_" or "0249."
        r'response_(\d+)_',         # Standard pattern: response_001_
        r'response_(\d+)\.',        # Pattern: response_001.
        r'(\d{3,5})[_\.]',          # 3-5 digit pattern with padding
        r'[Cc]hapter[_\s]*(\d+)',   # Chapter word pattern
        r'[Cc]h[_\s]*(\d+)',        # Ch abbreviation
        r'No(\d+)Chapter',          # No prefix with Chapter - matches "No00013Chapter.xhtml"
        r'No(\d+)Section',          # No prefix with Section - matches "No00013Section.xhtml"
        r'No(\d+)(?=\.|_|$)',       # No prefix followed by end, dot, or underscore (not followed by text)
        r'第(\d+)[章话回]',          # Chinese chapter markers
        r'_(\d+)(?:_|\.|$)',        # Number between underscores or at end
        r'^(\d+)(?:_|\.|$)',        # Starting with number
        r'(\d+)',                   # Any number (fallback)
    ]
    
    CJK_HONORIFICS = {
        'korean': [
            # Modern honorifics
            '님', '씨', '선배', '후배', '동기', '형', '누나', '언니', '오빠', '동생',
            '선생님', '교수님', '박사님', '사장님', '회장님', '부장님', '과장님', '대리님',
            '팀장님', '실장님', '이사님', '전무님', '상무님', '부사장님', '고문님',
            
            # Classical/formal honorifics
            '공', '옹', '군', '양', '낭', '랑', '생', '자', '부', '모', '시', '제', '족하',
            
            # Royal/noble address forms
            '마마', '마노라', '대감', '영감', '나리', '도령', '낭자', '아씨', '규수',
            '각하', '전하', '폐하', '저하', '합하', '대비', '대왕', '왕자', '공주',
            
            # Buddhist/religious
            '스님', '사부님', '조사님', '큰스님', '화상', '대덕', '대사', '법사',
            '선사', '율사', '보살님', '거사님', '신부님', '목사님', '장로님', '집사님',
            
            # Confucian/scholarly
            '부자', '선생', '대인', '어른', '어르신', '존자', '현자', '군자', '대부',
            '학사', '진사', '문하생', '제자',
            
            # Kinship honorifics
            '어르신', '할아버님', '할머님', '아버님', '어머님', '형님', '누님',
            '아주버님', '아주머님', '삼촌', '이모님', '고모님', '외삼촌', '장인어른',
            '장모님', '시아버님', '시어머님', '처남', '처형', '매형', '손님',
            
            # Verb-based honorific endings and speech levels
            '습니다', 'ㅂ니다', '습니까', 'ㅂ니까', '시다', '세요', '셔요', '십시오', '시오',
            '이에요', '예요', '이예요', '에요', '어요', '아요', '여요', '해요', '이세요', '으세요',
            '으시', '시', '으십니다', '십니다', '으십니까', '십니까', '으셨', '셨',
            '드립니다', '드려요', '드릴게요', '드리겠습니다', '올립니다', '올려요',
            '사옵니다', '사뢰', '여쭙니다', '여쭤요', '아뢰', '뵙니다', '뵈요', '모십니다',
            '시지요', '시죠', '시네요', '시는군요', '시는구나', '으실', '실',
            '드시다', '잡수시다', '주무시다', '계시다', '가시다', '오시다',
            
            # Common verb endings with 있다/없다/하다
            '있어요', '있습니다', '있으세요', '있으십니까', '없어요', '없습니다', '없으세요',
            '해요', '합니다', '하세요', '하십시오', '하시죠', '하시네요', '했어요', '했습니다',
            '되세요', '되셨어요', '되십니다', '됩니다', '되요', '돼요',
            '이야', '이네', '이구나', '이군', '이네요', '인가요', '인가', '일까요', '일까',
            '거예요', '거에요', '겁니다', '건가요', '게요', '을게요', '을까요', '었어요', '었습니다',
            '겠습니다', '겠어요', '겠네요', '을겁니다', '을거예요', '을거에요',
            
            # Common endings
            '요', '죠', '네요', '는데요', '거든요', '니까', '으니까', '는걸요', '군요', '구나',
            '는구나', '는군요', '더라고요', '더군요', '던데요', '나요', '가요', '까요',
            '라고요', '다고요', '냐고요', '자고요', '란다', '단다', '냔다', '잔다',
            
            # Formal archaic endings
            '나이다', '사옵나이다', '옵니다', '오', '소서', '으오', '으옵소서', '사이다',
            '으시옵니다', '시옵니다', '으시옵니까', '시옵니까', '나이까', '리이까', '리이다',
            '옵소서', '으소서', '소이다', '로소이다', '이옵니다', '이올시다', '하옵니다'
        ],
        'japanese': [
            # Modern honorifics
            'さん', 'ちゃん', '君', 'くん', '様', 'さま', '先生', 'せんせい', '殿', 'どの', '先輩', 'せんぱい',
            # Classical/historical
            '氏', 'し', '朝臣', 'あそん', '宿禰', 'すくね', '連', 'むらじ', '臣', 'おみ', '君', 'きみ',
            '真人', 'まひと', '道師', 'みちのし', '稲置', 'いなぎ', '直', 'あたい', '造', 'みやつこ',
            # Court titles
            '卿', 'きょう', '大夫', 'たいふ', '郎', 'ろう', '史', 'し', '主典', 'さかん',
            # Buddhist titles
            '和尚', 'おしょう', '禅師', 'ぜんじ', '上人', 'しょうにん', '聖人', 'しょうにん',
            '法師', 'ほうし', '阿闍梨', 'あじゃり', '大和尚', 'だいおしょう',
            # Shinto titles
            '大宮司', 'だいぐうじ', '宮司', 'ぐうじ', '禰宜', 'ねぎ', '祝', 'はふり',
            # Samurai era
            '守', 'かみ', '介', 'すけ', '掾', 'じょう', '目', 'さかん', '丞', 'じょう',
            # Keigo (honorific language) verb forms
            'です', 'ます', 'ございます', 'いらっしゃる', 'いらっしゃいます', 'おっしゃる', 'おっしゃいます',
            'なさる', 'なさいます', 'くださる', 'くださいます', 'いただく', 'いただきます',
            'おります', 'でございます', 'ございません', 'いたします', 'いたしました',
            '申す', '申します', '申し上げる', '申し上げます', '存じる', '存じます', '存じ上げる',
            '伺う', '伺います', '参る', '参ります', 'お目にかかる', 'お目にかかります',
            '拝見', '拝見します', '拝聴', '拝聴します', '承る', '承ります',
            # Respectful prefixes/suffixes
            'お', 'ご', '御', 'み', '美', '貴', '尊'
        ],
        'chinese': [
            # Modern forms
            '先生', '小姐', '夫人', '公子', '大人', '老师', '师父', '师傅', '同志', '同学',
            # Ancient/classical forms
            '子', '丈', '翁', '公', '侯', '伯', '叔', '仲', '季', '父', '甫', '卿', '君', '生',
            # Imperial court
            '陛下', '殿下', '千岁', '万岁', '圣上', '皇上', '天子', '至尊', '御前', '爷',
            # Nobility/officials
            '阁下', '大人', '老爷', '相公', '官人', '郎君', '娘子', '夫子', '足下',
            # Religious titles
            '上人', '法师', '禅师', '大师', '高僧', '圣僧', '神僧', '活佛', '仁波切',
            '真人', '天师', '道长', '道友', '仙长', '上仙', '祖师', '掌教',
            # Scholarly/Confucian
            '夫子', '圣人', '贤人', '君子', '大儒', '鸿儒', '宗师', '泰斗', '巨擘',
            # Martial arts
            '侠士', '大侠', '少侠', '女侠', '英雄', '豪杰', '壮士', '义士',
            # Family/kinship
            '令尊', '令堂', '令郎', '令爱', '贤弟', '贤侄', '愚兄', '小弟', '家父', '家母',
            # Humble forms
            '在下', '小人', '鄙人', '不才', '愚', '某', '仆', '妾', '奴', '婢',
            # Polite verbal markers
            '请', '请问', '敢问', '恭请', '敬请', '烦请', '有请', '请教', '赐教',
            '惠顾', '惠赐', '惠存', '笑纳', '雅正', '指正', '斧正', '垂询',
            '拜', '拜见', '拜访', '拜读', '拜托', '拜谢', '敬上', '谨上', '顿首'
        ],
        'english': [
            # Modern Korean romanizations (Revised Romanization of Korean - 2000)
            '-nim', '-ssi', '-seonbae', '-hubae', '-donggi', '-hyeong', '-nuna', 
            '-eonni', '-oppa', '-dongsaeng', '-seonsaengnim', '-gyosunim', 
            '-baksanim', '-sajangnim', '-hoejangnim', '-bujangnim', '-gwajangnim',
            '-daerim', '-timjangnim', '-siljangnim', '-isanim', '-jeonmunim',
            '-sangmunim', '-busajangnim', '-gomunnim',
            
            # Classical/formal Korean romanizations  
            '-gong', '-ong', '-gun', '-yang', '-nang', '-rang', '-saeng', '-ja',
            '-bu', '-mo', '-si', '-je', '-jokha',
            
            # Royal/noble Korean romanizations
            '-mama', '-manora', '-daegam', '-yeonggam', '-nari', '-doryeong',
            '-nangja', '-assi', '-gyusu', '-gakha', '-jeonha', '-pyeha', '-jeoha',
            '-hapka', '-daebi', '-daewang', '-wangja', '-gongju',
            
            # Buddhist/religious Korean romanizations
            '-seunim', '-sabunim', '-josanim', '-keunseunim', '-hwasang',
            '-daedeok', '-daesa', '-beopsa', '-seonsa', '-yulsa', '-bosalnim',
            '-geosanim', '-sinbunim', '-moksanim', '-jangnonim', '-jipsanim',
            
            # Confucian/scholarly Korean romanizations
            '-buja', '-seonsaeng', '-daein', '-eoreun', '-eoreusin', '-jonja', 
            '-hyeonja', '-gunja', '-daebu', '-haksa', '-jinsa', '-munhasaeng', '-jeja',
            
            # Kinship Korean romanizations
            '-harabeonim', '-halmeonim', '-abeonim', '-eomeonim', '-hyeongnim', 
            '-nunim', '-ajubeonim', '-ajumeonim', '-samchon', '-imonim', '-gomonim',
            '-oesamchon', '-jangineoreun', '-jangmonim', '-siabeonim', '-sieomeonim',
            '-cheonam', '-cheohyeong', '-maehyeong', '-sonnim',
            
            # Korean verb endings romanized (Revised Romanization)
            '-seumnida', '-mnida', '-seumnikka', '-mnikka', '-sida', '-seyo', 
            '-syeoyo', '-sipsio', '-sio', '-ieyo', '-yeyo', '-iyeyo', '-eyo', 
            '-eoyo', '-ayo', '-yeoyo', '-haeyo', '-iseyo', '-euseyo',
            '-eusi', '-si', '-eusimnida', '-simnida', '-eusimnikka', '-simnikka',
            '-eusyeot', '-syeot', '-deurimnida', '-deuryeoyo', '-deurilgeyo',
            '-deurigesseumnida', '-ollimnida', '-ollyeoyo', '-saomnida', '-saroe',
            '-yeojjumnida', '-yeojjwoyo', '-aroe', '-boemnida', '-boeyo', '-mosimnida',
            '-sijiyo', '-sijyo', '-sineyo', '-sineungunyo', '-sineunguna', '-eusil', '-sil',
            '-deusida', '-japsusida', '-jumusida', '-gyesida', '-gasida', '-osida',
            
            # Common Korean verb endings romanized
            '-isseoyo', '-isseumnida', '-isseuseyo', '-isseusimnikka', 
            '-eopseoyo', '-eopseumnida', '-eopseuseyo', '-hamnida', '-haseyo', 
            '-hasipsio', '-hasijyo', '-hasineyo', '-haesseoyo', '-haesseumnida',
            '-doeseyo', '-doesyeosseoyo', '-doesimnida', '-doemnida', '-doeyo', '-dwaeyo',
            '-iya', '-ine', '-iguna', '-igun', '-ineyo', '-ingayo', '-inga', 
            '-ilkkayo', '-ilkka', '-geoyeyo', '-geoeyo', '-geomnida', '-geongayo',
            '-geyo', '-eulgeyo', '-eulkkayo', '-eosseoyo', '-eosseumnida',
            '-gesseumnida', '-gesseoyo', '-genneyo', '-eulgeommida', '-eulgeoyeyo', '-eulgeoeyo',
            
            # Common Korean endings romanized
            '-yo', '-jyo', '-neyo', '-neundeyo', '-geodeunyo', '-nikka', 
            '-eunikka', '-neungeolyo', '-gunyo', '-guna', '-neunguna', '-neungunyo',
            '-deoragoyo', '-deogunyo', '-deondeyo', '-nayo', '-gayo', '-kkayo',
            '-ragoyo', '-dagoyo', '-nyagoyo', '-jagoyo', '-randa', '-danda', 
            '-nyanda', '-janda',
            
            # Formal archaic Korean romanized
            '-naida', '-saomnaida', '-omnida', '-o', '-soseo', '-euo', 
            '-euopsoseo', '-saida', '-eusiomnida', '-siomnida', '-eusiomnikka', 
            '-siomnikka', '-naikka', '-riikka', '-riida', '-opsoseo', '-eusoseo',
            '-soida', '-rosoida', '-iomnida', '-iolsida', '-haomnida',
            
            # Japanese keigo romanized (keeping existing)
            '-san', '-chan', '-kun', '-sama', '-sensei', '-senpai', '-dono', 
            '-shi', '-tan', '-chin', '-desu', '-masu', '-gozaimasu', 
            '-irassharu', '-irasshaimasu', '-ossharu', '-osshaimasu',
            '-nasaru', '-nasaimasu', '-kudasaru', '-kudasaimasu', '-itadaku', 
            '-itadakimasu', '-orimasu', '-degozaimasu', '-gozaimasen', 
            '-itashimasu', '-itashimashita', '-mousu', '-moushimasu', 
            '-moushiageru', '-moushiagemasu', '-zonjiru', '-zonjimasu',
            '-ukagau', '-ukagaimasu', '-mairu', '-mairimasu', '-haiken', 
            '-haikenshimasu',
            
            # Chinese romanizations (keeping existing)
            '-xiong', '-di', '-ge', '-gege', '-didi', '-jie', '-jiejie', 
            '-meimei', '-shixiong', '-shidi', '-shijie', '-shimei', '-gongzi', 
            '-guniang', '-xiaojie', '-daren', '-qianbei', '-daoyou', '-zhanglao', 
            '-shibo', '-shishu', '-shifu', '-laoshi', '-xiansheng', '-daxia', 
            '-shaoxia', '-nvxia', '-jushi', '-shanren', '-dazhang', '-zhenren',
            
            # Ancient Chinese romanizations
            '-zi', '-gong', '-hou', '-bo', '-jun', '-qing', '-weng', '-fu', 
            '-sheng', '-lang', '-langjun', '-niangzi', '-furen', '-gege', 
            '-jiejie', '-yeye', '-nainai',
            
            # Chinese politeness markers romanized
            '-qing', '-jing', '-gong', '-hui', '-ci', '-bai', '-gan', '-chui',
            'qingwen', 'ganwen', 'gongjing', 'jingjing', 'baijian', 'baifang', 
            'baituo'
        ]
    }

    TITLE_PATTERNS = {
        'korean': [
            # Modern titles
            r'\b(왕|여왕|왕자|공주|황제|황후|대왕|대공|공작|백작|자작|남작|기사|장군|대장|원수|제독|함장|대신|재상|총리|대통령|시장|지사|검사|판사|변호사|의사|박사|교수|신부|목사|스님|도사)\b',
            r'\b(폐하|전하|각하|예하|님|대감|영감|나리|도련님|아가씨|부인|선생)\b',
            # Historical/classical titles
            r'\b(대왕|태왕|왕비|왕후|세자|세자빈|대군|군|옹주|공주|부마|원자|원손)\b',
            r'\b(영의정|좌의정|우의정|판서|참판|참의|정승|판사|사또|현령|군수|목사|부사)\b',
            r'\b(대제학|제학|대사간|사간|대사헌|사헌|도승지|승지|한림|사관|내시|환관)\b',
            r'\b(병조판서|이조판서|호조판서|예조판서|형조판서|공조판서)\b',
            r'\b(도원수|부원수|병마절도사|수군절도사|첨절제사|만호|천호|백호)\b',
            r'\b(정일품|종일품|정이품|종이품|정삼품|종삼품|정사품|종사품|정오품|종오품)\b',
            # Korean honorific verb endings patterns
            r'(습니다|ㅂ니다|습니까|ㅂ니까|세요|셔요|십시오|시오)$',
            r'(이에요|예요|이예요|에요|어요|아요|여요|해요)$',
            r'(으시|시)(었|겠|ㄹ|을|는|던)*(습니다|ㅂ니다|어요|아요|세요)',
            r'(드립니다|드려요|드릴게요|드리겠습니다|올립니다|올려요)$',
            r'(사옵니다|여쭙니다|여쭤요|뵙니다|뵈요|모십니다)$',
            r'(나이다|사옵나이다|옵니다|으오|으옵소서|사이다)$'
        ],
        'japanese': [
            # Modern titles
            r'\b(王|女王|王子|姫|皇帝|皇后|天皇|皇太子|大王|大公|公爵|伯爵|子爵|男爵|騎士|将軍|大将|元帥|提督|艦長|大臣|宰相|総理|大統領|市長|知事|検事|裁判官|弁護士|医者|博士|教授|神父|牧師|僧侶|道士)\b',
            r'\b(陛下|殿下|閣下|猊下|様|大人|殿|卿|君|氏)\b',
            # Historical titles
            r'\b(天皇|皇后|皇太子|親王|内親王|王|女王|太政大臣|左大臣|右大臣|内大臣|大納言|中納言|参議)\b',
            r'\b(関白|摂政|征夷大将軍|管領|執権|守護|地頭|代官|奉行|与力|同心)\b',
            r'\b(太政官|神祇官|式部省|治部省|民部省|兵部省|刑部省|大蔵省|宮内省)\b',
            r'\b(大僧正|僧正|大僧都|僧都|律師|大法師|法師|大禅師|禅師)\b',
            r'\b(正一位|従一位|正二位|従二位|正三位|従三位|正四位|従四位|正五位|従五位)\b',
            r'\b(大和守|山城守|摂津守|河内守|和泉守|伊賀守|伊勢守|尾張守|三河守|遠江守)\b',
            # Japanese keigo (honorific language) patterns
            r'(です|ます|ございます)$',
            r'(いらっしゃ|おっしゃ|なさ|くださ)(います|いました|る|った)$',
            r'(いただ|お|ご|御)(き|きます|きました|く|ける|けます)',
            r'(申し上げ|申し|存じ上げ|存じ|伺い|参り)(ます|ました|る)$',
            r'(拝見|拝聴|承り|承)(します|しました|いたします|いたしました)$',
            r'お[^あ-ん]+[になる|になります|くださる|くださいます]'
        ],
        'chinese': [
            # Modern titles
            r'\b(王|女王|王子|公主|皇帝|皇后|大王|大公|公爵|伯爵|子爵|男爵|骑士|将军|大将|元帅|提督|舰长|大臣|宰相|总理|大总统|市长|知事|检察官|法官|律师|医生|博士|教授|神父|牧师|和尚|道士)\b',
            r'\b(陛下|殿下|阁下|大人|老爷|夫人|小姐|公子|少爷|姑娘|先生)\b',
            # Imperial titles
            r'\b(天子|圣上|皇上|万岁|万岁爷|太上皇|皇太后|太后|皇后|贵妃|妃|嫔|贵人|常在|答应)\b',
            r'\b(太子|皇子|皇孙|亲王|郡王|贝勒|贝子|公主|格格|郡主|县主|郡君|县君)\b',
            # Ancient official titles
            r'\b(丞相|相国|太师|太傅|太保|太尉|司徒|司空|大司马|大司农|大司寇)\b',
            r'\b(尚书|侍郎|郎中|员外郎|主事|知府|知州|知县|同知|通判|推官|巡抚|总督)\b',
            r'\b(御史大夫|御史中丞|监察御史|给事中|都察院|翰林院|国子监|钦天监)\b',
            r'\b(大学士|学士|侍读|侍讲|编修|检讨|庶吉士|举人|进士|状元|榜眼|探花)\b',
            # Military ranks
            r'\b(大元帅|元帅|大将军|将军|都督|都指挥使|指挥使|千户|百户|总兵|副将|参将|游击|都司|守备)\b',
            r'\b(提督|总兵官|副总兵|参将|游击将军|都司|守备|千总|把总|外委)\b',
            # Religious titles
            r'\b(国师|帝师|法王|活佛|堪布|仁波切|大和尚|方丈|住持|首座|维那|知客)\b',
            r'\b(天师|真人|道长|掌教|监院|高功|都讲|总理|提点|知观)\b',
            # Nobility ranks
            r'\b(公|侯|伯|子|男|开国公|郡公|国公|郡侯|县侯|郡伯|县伯|县子|县男)\b',
            r'\b(一品|二品|三品|四品|五品|六品|七品|八品|九品|正一品|从一品|正二品|从二品)\b',
            # Chinese politeness markers
            r'(请|敢|恭|敬|烦|有)(问|请|赐|教|告|示)',
            r'(拜|惠|赐|垂|雅|笑)(见|访|读|托|谢|顾|赐|存|纳|正|询)',
            r'(敬|谨|顿)(上|呈|启|白|首)'
        ],
        'english': [
            # Western titles
            r'\b(King|Queen|Prince|Princess|Emperor|Empress|Duke|Duchess|Marquis|Marquess|Earl|Count|Countess|Viscount|Viscountess|Baron|Baroness|Knight|Lord|Lady|Sir|Dame|General|Admiral|Captain|Major|Colonel|Commander|Lieutenant|Sergeant|Minister|Chancellor|President|Mayor|Governor|Judge|Doctor|Professor|Father|Reverend|Master|Mistress)\b',
            r'\b(His|Her|Your|Their)\s+(Majesty|Highness|Grace|Excellency|Honor|Worship|Lordship|Ladyship)\b',
            # Romanized historical titles
            r'\b(Tianzi|Huangdi|Huanghou|Taizi|Qinwang|Junwang|Beile|Beizi|Gongzhu|Gege)\b',
            r'\b(Chengxiang|Zaixiang|Taishi|Taifu|Taibao|Taiwei|Situ|Sikong|Dasima)\b',
            r'\b(Shogun|Daimyo|Samurai|Ronin|Ninja|Tenno|Mikado|Kampaku|Sessho)\b',
            r'\b(Taewang|Wangbi|Wanghu|Seja|Daegun|Gun|Ongju|Gongju|Buma)\b'
        ]
    }

    # Expanded Chinese numbers including classical forms
    CHINESE_NUMS = {
        # Basic numbers
        '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
        '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
        '十一': 11, '十二': 12, '十三': 13, '十四': 14, '十五': 15,
        '十六': 16, '十七': 17, '十八': 18, '十九': 19, '二十': 20,
        '二十一': 21, '二十二': 22, '二十三': 23, '二十四': 24, '二十五': 25,
        '三十': 30, '四十': 40, '五十': 50, '六十': 60,
        '七十': 70, '八十': 80, '九十': 90, '百': 100,
        # Classical/formal numbers
        '壹': 1, '贰': 2, '叁': 3, '肆': 4, '伍': 5,
        '陆': 6, '柒': 7, '捌': 8, '玖': 9, '拾': 10,
        '佰': 100, '仟': 1000, '萬': 10000, '万': 10000,
        # Ordinal indicators
        '第一': 1, '第二': 2, '第三': 3, '第四': 4, '第五': 5,
        '首': 1, '次': 2, '初': 1, '末': -1,
    }

    # Common words - keeping the same for filtering
    COMMON_WORDS = {
        '이', '그', '저', '우리', '너희', '자기', '당신', '여기', '거기', '저기',
        '오늘', '내일', '어제', '지금', '아까', '나중', '먼저', '다음', '마지막',
        '모든', '어떤', '무슨', '이런', '그런', '저런', '같은', '다른', '새로운',
        '하다', '있다', '없다', '되다', '하는', '있는', '없는', '되는',
        '것', '수', '때', '년', '월', '일', '시', '분', '초',
        '은', '는', '이', '가', '을', '를', '에', '의', '와', '과', '도', '만',
        '에서', '으로', '로', '까지', '부터', '에게', '한테', '께', '께서',
        'この', 'その', 'あの', 'どの', 'これ', 'それ', 'あれ', 'どれ',
        'わたし', 'あなた', 'かれ', 'かのじょ', 'わたしたち', 'あなたたち',
        'きょう', 'あした', 'きのう', 'いま', 'あとで', 'まえ', 'つぎ',
        'の', 'は', 'が', 'を', 'に', 'で', 'と', 'も', 'や', 'から', 'まで',
        '这', '那', '哪', '这个', '那个', '哪个', '这里', '那里', '哪里',
        '我', '你', '他', '她', '它', '我们', '你们', '他们', '她们',
        '今天', '明天', '昨天', '现在', '刚才', '以后', '以前', '后来',
        '的', '了', '在', '是', '有', '和', '与', '或', '但', '因为', '所以',
        '一', '二', '三', '四', '五', '六', '七', '八', '九', '十',
        '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
    }
# =====================================================
# CHUNK CONTEXT MANAGER (unchanged - already optimal)
# =====================================================
class ChunkContextManager:
    """Manage context within a chapter separate from history"""
    def __init__(self):
        self.current_chunks = []
        self.chapter_num = None
        self.chapter_title = None
        
    def start_chapter(self, chapter_num, chapter_title):
        """Start a new chapter context"""
        self.current_chunks = []
        self.chapter_num = chapter_num
        self.chapter_title = chapter_title
        
    def add_chunk(self, user_content, assistant_content, chunk_idx, total_chunks):
        """Add a chunk to the current chapter context"""
        self.current_chunks.append({
            "user": user_content,
            "assistant": assistant_content,
            "chunk_idx": chunk_idx,
            "total_chunks": total_chunks
        })
    
    def get_context_messages(self, limit=3):
        """Get last N chunks as messages for API context"""
        context = []
        for chunk in self.current_chunks[-limit:]:
            context.extend([
                {"role": "user", "content": chunk["user"]},
                {"role": "assistant", "content": chunk["assistant"]}
            ])
        return context
    
    def get_summary_for_history(self):
        """Create a summary representation for the history"""
        if not self.current_chunks:
            return None, None
            
        total_chunks = len(self.current_chunks)
        
        user_summary = f"[Chapter {self.chapter_num}: {self.chapter_title}]\n"
        user_summary += f"[{total_chunks} chunks processed]\n"
        if self.current_chunks:
            first_chunk = self.current_chunks[0]['user']
            if len(first_chunk) > 500:
                user_summary += first_chunk[:500] + "..."
            else:
                user_summary += first_chunk
        
        assistant_summary = f"[Chapter {self.chapter_num} Translation Complete]\n"
        assistant_summary += f"[Translated in {total_chunks} chunks]\n"
        if self.current_chunks:
            samples = []
            first_trans = self.current_chunks[0]['assistant']
            samples.append(f"Beginning: {first_trans[:200]}..." if len(first_trans) > 200 else f"Beginning: {first_trans}")
            
            if total_chunks > 2:
                mid_idx = total_chunks // 2
                mid_trans = self.current_chunks[mid_idx]['assistant']
                samples.append(f"Middle: {mid_trans[:200]}..." if len(mid_trans) > 200 else f"Middle: {mid_trans}")
            
            if total_chunks > 1:
                last_trans = self.current_chunks[-1]['assistant']
                samples.append(f"End: {last_trans[:200]}..." if len(last_trans) > 200 else f"End: {last_trans}")
            
            assistant_summary += "\n".join(samples)
        
        return user_summary, assistant_summary
    
    def clear(self):
        """Clear the current chapter context"""
        self.current_chunks = []
        self.chapter_num = None 
        self.chapter_title = None

# =====================================================
# UNIFIED UTILITIES
# =====================================================
class FileUtilities:
    """Utilities for file and path operations"""
    
    @staticmethod
    def extract_actual_chapter_number(chapter, patterns=None, config=None):
        """Extract actual chapter number from filename using improved logic"""
        
        # IMPORTANT: Check if this is a pre-split TEXT FILE chunk first
        if (chapter.get('is_chunk', False) and 
            'num' in chapter and 
            isinstance(chapter['num'], float) and
            chapter.get('filename', '').endswith('.txt')):
            # For text file chunks only, preserve the decimal number
            return chapter['num']  # This will be 1.1, 1.2, etc.
        
        # Get filename for extraction (broadened to match GUI/spine data)
        filename = (
            chapter.get('original_basename')
            or chapter.get('original_filename')
            or chapter.get('filename')
            or chapter.get('source_filename')
            or chapter.get('href')
            or chapter.get('idref')
            or chapter.get('id')
            or chapter.get('name')
            or chapter.get('key')
            or ''
        )
        
        opf_spine_position = chapter.get('spine_order')
        if opf_spine_position is None:
            opf_spine_position = chapter.get('opf_spine_position')
        actual_num, method = extract_chapter_number_from_filename(filename, opf_spine_position=opf_spine_position)

        # If extraction failed (no digits and no special), fall back to spine/file data
        if actual_num is None and opf_spine_position is not None:
            actual_num = opf_spine_position
            method = 'opf_spine_fallback'

        # Only fall back to file_chapter_num when we still have no number
        if actual_num is None and chapter.get('file_chapter_num') is not None:
            actual_num = chapter['file_chapter_num']
            method = 'file_chapter_num_fallback'

       # Prefer OPF spine position when available (ensures range selection follows content.opf)
       # opf_spine_position = chapter.get('spine_order')
       # opf_spine_data = chapter.get('opf_spine_data')
        
       # Use our improved extraction function
       # actual_num, method = extract_chapter_number_from_filename(
       #     filename,
       #     opf_spine_position=opf_spine_position,
       #     opf_spine_data=opf_spine_data
       # )
        
        # If extraction succeeded, return the result
        if actual_num is not None:
            #print(f"[DEBUG] Extracted {actual_num} from '{filename}' using method: {method}")
            return actual_num
        
        # Fallback to original complex logic for edge cases
        actual_num = None
        
        if patterns is None:
            patterns = PatternManager.FILENAME_EXTRACT_PATTERNS
        
        # Try to extract from original basename first
        if chapter.get('original_basename'):
            basename = chapter['original_basename']
            
            # Check if decimal chapters are enabled for EPUBs
            enable_decimal = os.getenv('ENABLE_DECIMAL_CHAPTERS', '0') == '1'
            
            # For EPUBs, only check decimal patterns if the toggle is enabled
            if enable_decimal:
                # Check for standard decimal chapter numbers (e.g., Chapter_1.1, 1.2.html)
                decimal_match = re.search(r'(\d+)\.(\d+)', basename)
                if decimal_match:
                    actual_num = float(f"{decimal_match.group(1)}.{decimal_match.group(2)}")
                    return actual_num
                
                # Check for the XXXX_YY pattern where it represents X.YY decimal chapters
                decimal_prefix_match = re.match(r'^(\d{4})_(\d{1,2})(?:_|\.)?(?:x?html?)?$', basename)
                if decimal_prefix_match:
                    first_part = decimal_prefix_match.group(1)
                    second_part = decimal_prefix_match.group(2)
                    
                    if len(second_part) == 2 and int(second_part) > 9:
                        chapter_num = int(first_part[-1])
                        decimal_part = second_part
                        actual_num = float(f"{chapter_num}.{decimal_part}")
                        return actual_num
            
            # Standard XXXX_Y format handling (existing logic)
            prefix_suffix_match = re.match(r'^(\d+)_(\d+)', basename)
            if prefix_suffix_match:
                second_part = prefix_suffix_match.group(2)
                
                if not enable_decimal:
                    actual_num = int(second_part)
                    return actual_num
                else:
                    if len(second_part) == 1 or (len(second_part) == 2 and int(second_part) <= 9):
                        actual_num = int(second_part)
                        return actual_num
            
            # Check other patterns if no match yet
            for pattern in patterns:
                if pattern in [r'^(\d+)[_\.]', r'(\d{3,5})[_\.]', r'^(\d+)_']:
                    continue
                match = re.search(pattern, basename, re.IGNORECASE)
                if match:
                    actual_num = int(match.group(1))
                    break
        
        # Final fallback to chapter num
        if actual_num is None:
            actual_num = chapter.get("num", 0)
            print(f"[DEBUG] No pattern matched, using chapter num: {actual_num}")
        
        return actual_num
    
    @staticmethod
    def create_chapter_filename(chapter, actual_num=None):
        """Create consistent chapter filename"""
        # Check if we should use header as output name
        use_header_output = os.getenv("USE_HEADER_AS_OUTPUT", "0") == "1"
        
        # Check if this is for a text file
        is_text_file = chapter.get('filename', '').endswith('.txt') or chapter.get('is_chunk', False)
        
        # Respect toggle: retain source extension and remove 'response_' prefix
        retain = should_retain_source_extension()
        
        # Helper to compute full original extension chain (e.g., '.html.xhtml')
        def _full_ext_from_original(ch):
            fn = ch.get('original_filename')
            if not fn:
                return '.html'
            bn = os.path.basename(fn)
            root, ext = os.path.splitext(bn)
            if not ext:
                return '.html'
            full_ext = ''
            while ext:
                full_ext = ext + full_ext
                root, ext = os.path.splitext(root)
            return full_ext or '.html'
        
        if use_header_output and chapter.get('title'):
            chapter_num_for_name = actual_num or chapter.get('num', 0)
            safe_title = make_safe_filename(chapter['title'], chapter_num_for_name)
            # For comparison, handle both int and float chapter numbers
            if isinstance(chapter_num_for_name, float):
                major = int(chapter_num_for_name)
                minor = int(round((chapter_num_for_name - major) * 100))
                if minor > 0:
                    comparison_name = f"chapter_{major:03d}_{minor:02d}"
                else:
                    comparison_name = f"chapter_{major:03d}"
            else:
                comparison_name = f"chapter_{chapter_num_for_name:03d}"
            if safe_title and safe_title != comparison_name:
                if is_text_file:
                    return f"{safe_title}.txt" if retain else f"response_{safe_title}.txt"
                else:
                    # If retaining, use full original ext chain; else default .html
                    if retain:
                        return f"{safe_title}{_full_ext_from_original(chapter)}"
                    return f"response_{safe_title}.html"
        
        # Check if decimal chapters are enabled
        enable_decimal = os.getenv('ENABLE_DECIMAL_CHAPTERS', '0') == '1'
        
        # For EPUBs with decimal detection enabled
        if enable_decimal and 'original_basename' in chapter and chapter['original_basename']:
            basename = chapter['original_basename']
            
            # Check for standard decimal pattern (e.g., Chapter_1.1)
            decimal_match = re.search(r'(\d+)\.(\d+)', basename)
            if decimal_match:
                # Create a modified basename that preserves the decimal
                base = os.path.splitext(basename)[0]
                # Replace dots with underscores for filesystem compatibility
                base = base.replace('.', '_')
                # Use .txt extension for text files
                if is_text_file:
                    return f"{base}.txt" if retain else f"response_{base}.txt"
                else:
                    if retain:
                        return f"{base}{_full_ext_from_original(chapter)}"
                    return f"response_{base}.html"
            
            # Check for the special XXXX_YY decimal pattern
            decimal_prefix_match = re.match(r'^(\d{4})_(\d{1,2})(?:_|\.)?(?:x?html?)?$', basename)
            if decimal_prefix_match:
                first_part = decimal_prefix_match.group(1)
                second_part = decimal_prefix_match.group(2)
                
                # If this matches our decimal pattern (e.g., 0002_33 -> 2.33)
                if len(second_part) == 2 and int(second_part) > 9:
                    chapter_num = int(first_part[-1])
                    decimal_part = second_part
                    # Create filename reflecting the decimal interpretation
                    if is_text_file:
                        return f"{chapter_num:03d}_{decimal_part}.txt" if retain else f"response_{chapter_num:03d}_{decimal_part}.txt"
                    else:
                        return f"{chapter_num:03d}_{decimal_part}{_full_ext_from_original(chapter)}" if retain else f"response_{chapter_num:03d}_{decimal_part}.html"
        
        # Standard EPUB handling - use original basename
        if 'original_basename' in chapter and chapter['original_basename']:
            base = os.path.splitext(chapter['original_basename'])[0]
            # Use .txt extension for text files
            if is_text_file:
                return f"{base}.txt" if retain else f"response_{base}.txt"
            else:
                if retain:
                    # Preserve the full original extension chain
                    return f"{base}{_full_ext_from_original(chapter)}"
                return f"response_{base}.html"
        else:
            # Text file handling (no original basename)
            if actual_num is None:
                actual_num = chapter.get('actual_chapter_num', chapter.get('num', 0))
            
            # Handle decimal chapter numbers from text file splitting
            if isinstance(actual_num, float):
                major = int(actual_num)
                minor = int(round((actual_num - major) * 10))  # Use *10 to get 0, 1, 2, etc. from 1.0, 1.1, 1.2
                
                # PDF CHUNK FIX: Check if the chunk has a specific filename with extension
                # For PDF chunks, preserve the .html or .md extension from the original filename
                chunk_filename = chapter.get('filename', '')
                if chunk_filename and (chunk_filename.endswith('.html') or chunk_filename.endswith('.md')):
                    # Use the extension from the chunk's original filename
                    file_ext = '.html' if chunk_filename.endswith('.html') else '.md'
                    if retain:
                        return f"section_{major}_{minor}{file_ext}"
                    else:
                        return f"response_section_{major}_{minor}{file_ext}"
                elif is_text_file:
                    return f"section_{major}_{minor}.txt" if retain else f"response_section_{major}_{minor}.txt"
                else:
                    return f"{major:03d}_{minor:02d}.html" if retain else f"response_{major:03d}_{minor:02d}.html"
            else:
                # For integer chapter numbers, use standard formatting
                if is_text_file:
                    return f"section_{actual_num}.txt" if retain else f"response_section_{actual_num}.txt"
                else:
                    return f"{actual_num:03d}.html" if retain else f"response_{actual_num:03d}.html"

# =====================================================
# UNIFIED PROGRESS MANAGER
# =====================================================
class ProgressManager:
    """Unified progress management"""
    
    def __init__(self, payloads_dir):
        self.payloads_dir = payloads_dir
        self.PROGRESS_FILE = os.path.join(payloads_dir, "translation_progress.json")
        self.prog = self._init_or_load()
        # Disable auto-dedup unless explicitly enabled; dedup can drop distinct chapters sharing filenames
        if os.getenv("ENABLE_PROGRESS_DEDUP", "0") == "1":
            self._dedup_by_output()
        
    def _init_or_load(self):
        """Initialize or load progress tracking with improved structure"""
        if os.path.exists(self.PROGRESS_FILE):
            try:
                with open(self.PROGRESS_FILE, "r", encoding="utf-8") as pf:
                    prog = json.load(pf)
            except json.JSONDecodeError as e:
                print(f"⚠️ Warning: Progress file is corrupted: {e}")
                print("🔧 Attempting to fix JSON syntax...")
                
                try:
                    with open(self.PROGRESS_FILE, "r", encoding="utf-8") as pf:
                        content = pf.read()
                    
                    content = re.sub(r',\s*\]', ']', content)
                    content = re.sub(r',\s*\}', '}', content)
                    
                    prog = json.loads(content)
                    
                    with open(self.PROGRESS_FILE, "w", encoding="utf-8") as pf:
                        json.dump(prog, pf, ensure_ascii=False, indent=2)
                    print("✅ Successfully fixed and saved progress file")
                    
                except Exception as fix_error:
                    print(f"❌ Could not fix progress file: {fix_error}")
                    print("🔄 Creating backup and starting fresh...")
                    
                    backup_name = f"translation_progress_backup_{int(time.time())}.json"
                    backup_path = os.path.join(self.payloads_dir, backup_name)
                    try:
                        shutil.copy(self.PROGRESS_FILE, backup_path)
                        print(f"📁 Backup saved to: {backup_name}")
                    except:
                        pass
                    
                    prog = {
                        "chapters": {},
                        "chapter_chunks": {},
                        "version": "2.0"
                    }
            
            if "chapters" not in prog:
                prog["chapters"] = {}
                
                for idx in prog.get("completed", []):
                    prog["chapters"][str(idx)] = {
                        "status": "completed",
                        "timestamp": None
                    }
            
            if "chapter_chunks" not in prog:
                prog["chapter_chunks"] = {}
                
        else:
            prog = {
                "chapters": {},
                "chapter_chunks": {},
                "image_chunks": {},
                "version": "2.1"
            }
        
        return prog

    def _dedup_by_output(self):
        """Keep a single entry per normalized output filename; priority: qa_failed > pending > failed > in_progress > completed."""
        def _norm_out(fname: str):
            if not fname:
                return None
            base = os.path.basename(fname)
            if base.startswith("response_"):
                base = base[len("response_"):]
            return os.path.splitext(base)[0]
        def _infer_num(fname: str):
            if not fname:
                return None
            nums = re.findall(r"\d+", fname)
            if not nums:
                return None
            nums = list(map(int, nums))
            if nums[0] == 0 and nums[-1] > 0:
                return nums[-1]
            return nums[0]
        # Prefer completed over failed/pending/in_progress, but keep qa_failed highest
        severity = {'qa_failed': 6, 'completed': 5, 'merged': 5, 'pending': 4, 'failed': 3, 'in_progress': 2, 'unknown': 0}
        dedup = {}
        for key, info in list(self.prog.get("chapters", {}).items()):
            out = info.get("output_file")
            norm = _norm_out(out) or key
            if (info.get("actual_num") in (None, 0)) and out:
                hint = _infer_num(out)
                if hint is not None:
                    info["actual_num"] = hint
            current = dedup.get(norm)
            if current:
                cur_rank = severity.get(current.get("status", "unknown"), 0)
                new_rank = severity.get(info.get("status", "unknown"), 0)
                if (new_rank > cur_rank) or (new_rank == cur_rank and info.get("last_updated", 0) > current.get("last_updated", 0)):
                    dedup[norm] = info
            else:
                dedup[norm] = info
        new_chapters = {}
        for norm, info in dedup.items():
            new_key = str(info["actual_num"]) if info.get("actual_num") is not None else norm
            if new_key in new_chapters:
                cur_rank = severity.get(new_chapters[new_key].get("status", "unknown"), 0)
                new_rank = severity.get(info.get("status", "unknown"), 0)
                if (new_rank > cur_rank) or (new_rank == cur_rank and info.get("last_updated", 0) > new_chapters[new_key].get("last_updated", 0)):
                    new_chapters[new_key] = info
            else:
                new_chapters[new_key] = info
        self.prog["chapters"] = new_chapters
        # NOTE: caller is responsible for saving after dedup
    
    def _get_chapter_key(self, actual_num, output_file=None, chapter_obj=None, content_hash=None):
        """Generate consistent chapter key, handling collisions with composite keys.
        
        Returns the key that should be used for this chapter in the progress dict.
        """
        def _normalize_fname(fname):
            """Normalize filename for comparison regardless of response_ prefix or extension."""
            if not fname:
                return None
            base = os.path.basename(fname)
            if base.startswith('response_'):
                base = base[len('response_'):]
            # Strip extension only for comparison so .html vs .xhtml don't diverge
            return os.path.splitext(base)[0]
        def _make_spine_key(num, spine_pos):
            if spine_pos is None:
                return None
            return f"{num}@{spine_pos}"

        spine_pos = None
        if chapter_obj:
            spine_pos = chapter_obj.get('spine_order')
            if spine_pos is None:
                spine_pos = chapter_obj.get('opf_spine_position')
        # CHUNK FIX: For decimal chapter numbers (e.g., 1.0, 1.1), use the full decimal in the key
        # This prevents collisions when multiple chunks share the same integer part
        if isinstance(actual_num, float) and actual_num != int(actual_num):
            # Convert to string preserving decimal: "1.0", "1.1", etc.
            chapter_key = str(actual_num)
        else:
            chapter_key = str(actual_num)
        
        # Determine the output filename
        if output_file:
            filename = output_file
        elif chapter_obj:
            from TransateKRtoEN import FileUtilities
            filename = FileUtilities.create_chapter_filename(chapter_obj, actual_num)
        else:
            # No way to determine filename, use simple key
            return chapter_key
        
        # SPECIAL FILES FIX: Check if there's an in-progress entry with matching content_hash
        # This allows us to update the same entry when completing a special file
        if content_hash and chapter_key in self.prog["chapters"]:
            existing_info = self.prog["chapters"][chapter_key]
            existing_hash = existing_info.get("content_hash")
            existing_file = existing_info.get("output_file")
            
            # If hashes match and it's in-progress (no output file yet), keep using simple key
            if existing_hash == content_hash and not existing_file:
                return chapter_key
        
        # If a spine key already exists, prefer it
        spine_key = _make_spine_key(actual_num, spine_pos)
        if spine_key and spine_key in self.prog["chapters"]:
            existing_info = self.prog["chapters"][spine_key]
            existing_file = existing_info.get("output_file")
            # Require exact filename match to avoid mixing notice/chapter files with same number
            if existing_file == filename:
                return spine_key

        # Check if simple key exists and matches this file
        if chapter_key in self.prog["chapters"]:
            existing_info = self.prog["chapters"][chapter_key]
            existing_file = existing_info.get("output_file")
            existing_status = existing_info.get("status")
            
            # If the existing entry is for the same file, use simple key
            if existing_file == filename:
                return chapter_key

            # NEW: tolerate retain-source toggle changes (response_ prefix / extension)
            existing_norm = _normalize_fname(existing_file)
            new_norm = _normalize_fname(filename)
            if existing_norm and new_norm and existing_norm == new_norm:
                return chapter_key
            
            # MERGED STATUS FIX: If existing entry is merged, always use simple key
            # Merged chapters point to parent's output_file, so filename won't match
            # but we still want to use the same key to find the merged status
            if existing_status == "merged":
                return chapter_key
            
            # Different file with same chapter number - prefer spine-based composite, else filename-based
            if spine_key:
                return spine_key
            file_basename = os.path.splitext(os.path.basename(filename))[0]
            file_basename = file_basename.replace("response_", "")
            composite_key = f"{actual_num}_{file_basename}"
            # NEW: if existing entry is pending and for a different file, don't overwrite it
            if existing_status and str(existing_status).lower().startswith("pending"):
                if existing_file and existing_file != filename:
                    return composite_key
            return composite_key
        
        # Check if composite key already exists for this file
        file_basename = os.path.splitext(os.path.basename(filename))[0]
        file_basename = file_basename.replace("response_", "")
        composite_key = f"{actual_num}_{file_basename}"
        spine_composite = spine_key
        
        if spine_composite and spine_composite in self.prog["chapters"]:
            return spine_composite
        if composite_key in self.prog["chapters"]:
            return composite_key
        
        # No existing entry - use simple key for new entries
        return spine_key or chapter_key
    
    def save(self):
        """Save progress to file"""
        try:
            self.prog["completed_list"] = []
            for chapter_key, chapter_info in self.prog.get("chapters", {}).items():
                if chapter_info.get("status") == "completed" and chapter_info.get("output_file"):
                    actual_num = chapter_info.get("actual_num", 0)
                    self.prog["completed_list"].append({
                        "num": actual_num,
                        "idx": 0,  # idx is not used anymore
                        "title": f"Chapter {actual_num}",
                        "file": chapter_info.get("output_file", ""),
                        "key": chapter_key
                    })
            
            if self.prog.get("completed_list"):
                self.prog["completed_list"].sort(key=lambda x: x["num"])
            
            temp_file = self.PROGRESS_FILE + '.tmp'
            with open(temp_file, "w", encoding="utf-8") as pf:
                json.dump(self.prog, pf, ensure_ascii=False, indent=2)
            
            if os.path.exists(self.PROGRESS_FILE):
                os.remove(self.PROGRESS_FILE)
            os.rename(temp_file, self.PROGRESS_FILE)
        except Exception as e:
            print(f"⚠️ Warning: Failed to save progress: {e}")
            temp_file = self.PROGRESS_FILE + '.tmp'
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
    
    def update(self, idx, actual_num, content_hash, output_file, status="in_progress", ai_features=None, raw_num=None, chapter_obj=None, merged_chapters=None, qa_issues_found=None):
        """Update progress for a chapter"""
        # Use helper method to get consistent key
        chapter_key = self._get_chapter_key(actual_num, output_file, chapter_obj, content_hash)
        
        # Log if we're using a composite key
        if "_" in chapter_key and chapter_key != str(actual_num):
            print(f"📌 Using composite key for chapter {actual_num}: {chapter_key}")
        
        # MERGED CHAPTERS FIX: If this chapter has merged children and status changes to failed/pending,
        # clear the merged status from all child chapters so they can be retranslated
        if status in ["qa_failed", "failed", "pending", "error"] and chapter_key in self.prog["chapters"]:
            existing_info = self.prog["chapters"][chapter_key]
            merged_child_nums = existing_info.get("merged_chapters", [])
            
            if merged_child_nums:
                print(f"🔓 Clearing merged status from {len(merged_child_nums)} child chapters due to parent status: {status}")
                
                # Find and clear merged status from all child chapters
                for child_chapter_key, child_info in list(self.prog["chapters"].items()):
                    if child_info.get("status") == "merged" and child_info.get("merged_parent_chapter") == actual_num:
                        child_actual_num = child_info.get("actual_num")
                        print(f"   🔓 Clearing merged status for chapter {child_actual_num}")
                        # Delete the merged child entry so it will be retranslated
                        del self.prog["chapters"][child_chapter_key]
        
        chapter_info = {
            "actual_num": actual_num,
            "content_hash": content_hash,
            "output_file": output_file,
            "status": status,
            "last_updated": time.time()
        }
        
        # Add raw number tracking
        if raw_num is not None:
            chapter_info["raw_chapter_num"] = raw_num
        
        # Check if zero detection was disabled
        if hasattr(builtins, '_DISABLE_ZERO_DETECTION') and builtins._DISABLE_ZERO_DETECTION:
            chapter_info["zero_adjusted"] = False
        else:
            chapter_info["zero_adjusted"] = (raw_num != actual_num) if raw_num is not None else False
        
        # FIXED: Store AI features if provided
        if ai_features is not None:
            chapter_info["ai_features"] = ai_features
        
        # Preserve existing AI features if not overwriting
        elif chapter_key in self.prog["chapters"] and "ai_features" in self.prog["chapters"][chapter_key]:
            chapter_info["ai_features"] = self.prog["chapters"][chapter_key]["ai_features"]
        
        # Add merged chapters list if provided (for parent chapters in request merging)
        if merged_chapters is not None:
            chapter_info["merged_chapters"] = merged_chapters
        
        # Add QA issues if provided (for qa_failed status)
        if qa_issues_found is not None:
            chapter_info["qa_issues"] = True
            chapter_info["qa_timestamp"] = time.time()
            chapter_info["qa_issues_found"] = qa_issues_found
        # IMPORTANT: When changing to in_progress or failed status, explicitly clear QA fields
        # This ensures old qa_failed markers don't persist
        elif status in ["in_progress", "failed"]:
            # Don't add QA fields - they will be excluded from chapter_info
            pass
        
        self.prog["chapters"][chapter_key] = chapter_info
    
    def mark_as_merged(self, idx, actual_num, content_hash, parent_chapter_num, chapter_obj=None, parent_output_file=None):
        """Mark a chapter as merged into a parent chapter"""
        chapter_key = self._get_chapter_key(actual_num, output_file=None, chapter_obj=chapter_obj, content_hash=content_hash)
        
        merged_info = {
            "actual_num": actual_num,
            "content_hash": content_hash,
            "output_file": parent_output_file,  # Point to parent's output file
            "status": "merged",
            "merged_parent_chapter": parent_chapter_num,
            "last_updated": time.time()
        }
        
        # Add original_basename so GUI can match by source filename
        if chapter_obj and 'original_basename' in chapter_obj:
            merged_info["original_basename"] = chapter_obj['original_basename']
        elif chapter_obj and 'filename' in chapter_obj:
            merged_info["original_basename"] = chapter_obj['filename']
        
        self.prog["chapters"][chapter_key] = merged_info
    
    def update_merged_chapters_list(self, parent_chapter_num, merged_chapter_nums, parent_content_hash=None, parent_chapter_obj=None):
        """Update the parent chapter to track which chapters were merged into it"""
        chapter_key = self._get_chapter_key(parent_chapter_num, output_file=None, chapter_obj=parent_chapter_obj, content_hash=parent_content_hash)
        
        if chapter_key in self.prog["chapters"]:
            self.prog["chapters"][chapter_key]["merged_chapters"] = merged_chapter_nums
        
    def check_chapter_status(self, chapter_idx, actual_num, content_hash, output_dir, chapter_obj=None):
        """Check if a chapter needs translation"""
        # If caller passed 0/None, recompute from filename/spine to avoid collapsing to chapter 0
        if (actual_num is None or actual_num <= 0) and chapter_obj:
            try:
                from TransateKRtoEN import FileUtilities
                recomputed = FileUtilities.extract_actual_chapter_number(chapter_obj, patterns=None, config=None)
                if recomputed is not None:
                    actual_num = recomputed
            except Exception:
                pass
        # Use helper method to get consistent key
        chapter_key = self._get_chapter_key(actual_num, output_file=None, chapter_obj=chapter_obj, content_hash=content_hash)
        
        # Check if we have tracking for this chapter
        if chapter_key in self.prog["chapters"]:
            chapter_info = self.prog["chapters"][chapter_key]
            status = chapter_info.get("status")
            status_l = status.lower() if isinstance(status, str) else status or ""
            # Failed statuses ALWAYS trigger retranslation
            if status in ["qa_failed", "failed", "error", "file_missing"]:
                return True, None, None
            
            # Merged status - skip translation, content is in parent chapter
            if status == "merged":
                parent_chapter = chapter_info.get("merged_parent_chapter")
                return False, f"Chapter {actual_num} merged into chapter {parent_chapter}", None
            
            # Completed - check file exists
            if status in ["completed", "completed_empty", "completed_image_only"]:
                output_file = chapter_info.get("output_file")
                if output_file:
                    output_path = os.path.join(output_dir, output_file)
                    if os.path.exists(output_path):
                        return False, f"Chapter {actual_num} already translated: {output_file}", output_file

                    # Fallback: look for any file with same base name (ignore extensions)
                    expected_norm = _norm(output_file)
                    try:
                        for f in os.listdir(output_dir):
                            if _norm(f) == expected_norm:
                                alt_path = os.path.join(output_dir, f)
                                if os.path.exists(alt_path):
                                    # Update stored filename to the discovered one
                                    self.prog["chapters"][chapter_key]["output_file"] = f
                                    self.save()
                                    return False, f"Chapter {actual_num} already translated: {f}", f
                    except Exception:
                        pass

                # File missing - retranslate
                del self.prog["chapters"][chapter_key]
                if chapter_key in self.prog.get("chapter_chunks", {}):
                    del self.prog["chapter_chunks"][chapter_key]
                self.save()
                return True, None, None
            
            # Any other status - retranslate
            return True, None, None
        
        # No entry in progress tracking - check if file exists on disk
        # This handles the case where progress file was deleted but translated files remain
        if chapter_obj:
            from TransateKRtoEN import FileUtilities
            output_filename = FileUtilities.create_chapter_filename(chapter_obj, actual_num)
            output_path = os.path.join(output_dir, output_filename)

            # If a differently-keyed entry already tracks this file, reuse it instead of auto-discovering
            def _norm(fname: str):
                """
                Normalize a filename for comparison:
                - drop leading response_ prefix
                - strip *all* extensions (handle .html.xhtml, .md.html, etc.)
                - lowercase for case-insensitive matching on Windows
                """
                if not fname:
                    return ""
                base = os.path.basename(fname)
                if base.startswith("response_"):
                    base = base[len("response_"):]
                # Strip all extensions, not just the last one
                while True:
                    base, ext = os.path.splitext(base)
                    if not ext:
                        break
                return base.lower()

            expected_norm = _norm(output_filename)
            for k, info in self.prog.get("chapters", {}).items():
                if _norm(info.get("output_file")) == expected_norm:
                    status = info.get("status")
                    if status in ["completed", "completed_empty", "completed_image_only"]:
                        if info.get("output_file"):
                            if os.path.exists(os.path.join(output_dir, info["output_file"])):
                                return False, f"Chapter {info.get('actual_num', actual_num)} already translated: {info['output_file']}", info["output_file"]
                    # If tracked with other status, treat as tracked (will retranslate if non-completed)
                    return True, None, info.get("output_file")
            
            # Check if file exists for auto-discovery
            if os.path.exists(output_path):
                print(f"📁 Found existing file for chapter {actual_num}: {output_filename}")
                
                self.prog["chapters"][chapter_key] = {
                    "actual_num": actual_num,
                    "content_hash": content_hash,
                    "output_file": output_filename,
                    "status": "completed",
                    "last_updated": os.path.getmtime(output_path),
                    "auto_discovered": True
                }
                
                self.save()
                return False, f"Chapter {actual_num} already exists: {output_filename}", output_filename
        
        # No entry and no file - needs translation
        return True, None, None
        
    def cleanup_missing_files(self, output_dir):
        """Remove missing files and clear merged children of missing parents"""
        cleaned_count = 0
        deleted_parents = set()  # Track which parent chapters were deleted
        parents_with_missing_files = set()  # Track parents with missing files (for merged children clearing)
        
        # First pass: Remove entries for missing files (except merged children and certain non-final states)
        for chapter_key, chapter_info in list(self.prog["chapters"].items()):
            output_file = chapter_info.get("output_file")
            status = chapter_info.get("status")
            status_l = status.lower().strip() if isinstance(status, str) else (str(status).lower().strip() if status is not None else "")
            # MERGED CHAPTERS FIX: Don't delete merged children in first pass
            # They will be handled in second pass if their parent was deleted
            if status == "merged":
                continue
            
            # QA_FAILED / FAILED / IN_PROGRESS / PENDING FIX:
            # Don't delete entries that are meant to be visible in the retranslation UI
            # even when their output file is missing.
            # - qa_failed/failed: should remain visible for investigation/retry
            # - in_progress: file doesn't exist yet because translation is ongoing
            # - pending: user explicitly marked for retranslation; file may have been deleted on purpose
            if status_l.startswith("pending") or status_l in ["qa_failed", "failed", "in_progress"]:
                continue
            
            if output_file:
                output_path = os.path.join(output_dir, output_file)
                if not os.path.exists(output_path):
                    
                    actual_num = chapter_info.get("actual_num")
                    if actual_num is not None:
                        # Track if this was a parent of merged chapters
                        deleted_parents.add(actual_num)
                        
                        # Also track if this chapter has merged children (for later clearing)
                        if chapter_info.get("merged_chapters"):
                            parents_with_missing_files.add(actual_num)
                    
                    # Delete the entry
                    del self.prog["chapters"][chapter_key]
                    
                    # Remove chunk data
                    if chapter_key in self.prog.get("chapter_chunks", {}):
                        del self.prog["chapter_chunks"][chapter_key]
                    
                    cleaned_count += 1
        
        # Second pass: Clear merged children whose parents were deleted OR have missing files
        if deleted_parents or parents_with_missing_files:
            all_affected_parents = deleted_parents | parents_with_missing_files
            for chapter_key, chapter_info in list(self.prog["chapters"].items()):
                if chapter_info.get("status") == "merged":
                    parent_num = chapter_info.get("merged_parent_chapter")
                    if parent_num in all_affected_parents:
                        actual_num = chapter_info.get("actual_num")
                        print(f"🔓 Clearing merged child chapter {actual_num} (parent {parent_num} file is missing)")
                        del self.prog["chapters"][chapter_key]
                        cleaned_count += 1
        
        if cleaned_count > 0:
            print(f"🔄 Removed {cleaned_count} missing file entries")
    
    def migrate_to_content_hash(self, chapters):
        """Change keys to match actual_num values for proper mapping and sort by chapter number"""
        
        def _normalize_out(fname: str):
            if not fname:
                return None
            base = os.path.basename(fname)
            if base.startswith('response_'):
                base = base[len('response_'):]
            return os.path.splitext(base)[0]

        def _infer_num_from_filename(fname: str):
            if not fname:
                return None
            nums = re.findall(r'\\d+', fname)
            if not nums:
                return None
            nums = list(map(int, nums))
            if nums[0] == 0 and nums[-1] > 0:
                return nums[-1]
            return nums[0]

        # Priority: qa_failed > pending > failed > in_progress > completed
        severity_rank = {'qa_failed': 6, 'completed': 5, 'merged': 5, 'pending': 4, 'failed': 3, 'in_progress': 2, 'unknown': 0}

        # First, deduplicate by normalized output filename choosing highest severity then latest timestamp
        dedup = {}
        for old_key, chapter_info in self.prog["chapters"].items():
            out = chapter_info.get("output_file")
            norm = _normalize_out(out)
            if not norm:
                norm = old_key  # fallback to key to avoid losing entry

            # Fix actual_num if missing or zero using filename hint
            actual_num = chapter_info.get("actual_num")
            if (actual_num in (None, 0)) and out:
                hint = _infer_num_from_filename(out)
                if hint is not None:
                    chapter_info["actual_num"] = hint
                    actual_num = hint

            current_best = dedup.get(norm)
            if current_best:
                best_sev = severity_rank.get(current_best.get("status", "unknown"), 0)
                cur_sev = severity_rank.get(chapter_info.get("status", "unknown"), 0)
                if (cur_sev > best_sev) or (cur_sev == best_sev and chapter_info.get("last_updated", 0) > current_best.get("last_updated", 0)):
                    dedup[norm] = chapter_info
            else:
                dedup[norm] = chapter_info

        new_chapters = {}
        migrated_count = 0
        
        for norm, chapter_info in dedup.items():
            actual_num = chapter_info.get("actual_num")
            key_candidate = None
            # Prefer numeric key when available
            if actual_num is not None:
                key_candidate = str(actual_num)
            else:
                key_candidate = norm

            # If non-numeric key, keep as-is
            if not key_candidate.isdigit():
                new_key = key_candidate
            else:
                new_key = key_candidate

            # Handle collisions by severity and timestamp
            if new_key in new_chapters:
                existing = new_chapters[new_key]
                best_sev = severity_rank.get(existing.get("status", "unknown"), 0)
                cur_sev = severity_rank.get(chapter_info.get("status", "unknown"), 0)
                if (cur_sev > best_sev) or (cur_sev == best_sev and chapter_info.get("last_updated", 0) > existing.get("last_updated", 0)):
                    new_chapters[new_key] = chapter_info
            else:
                new_chapters[new_key] = chapter_info
            migrated_count += 1
        
        # Sort chapters by actual_num field, then by key as fallback
        def sort_key(item):
            key, chapter_info = item
            actual_num = chapter_info.get("actual_num")
            if actual_num is not None:
                return actual_num
            else:
                # Fallback to key if no actual_num
                try:
                    return int(key)
                except ValueError:
                    # For non-numeric keys, sort them at the end
                    return float('inf')
        
        sorted_chapters = dict(sorted(new_chapters.items(), key=sort_key))
        
        if migrated_count > 0:
            # Also migrate and sort chapter_chunks if they exist
            if "chapter_chunks" in self.prog:
                new_chunks = {}
                for old_key, chunk_data in self.prog["chapter_chunks"].items():
                    if not str(old_key).isdigit():
                        new_chunks[old_key] = chunk_data
                    elif old_key in self.prog["chapters"] and "actual_num" in self.prog["chapters"][old_key]:
                        new_key = str(self.prog["chapters"][old_key]["actual_num"])
                        new_chunks[new_key] = chunk_data
                    else:
                        new_chunks[old_key] = chunk_data
                
                # Sort chapter_chunks using the same sorting logic
                sorted_chunks = dict(sorted(new_chunks.items(), key=sort_key))
                self.prog["chapter_chunks"] = sorted_chunks
            
            self.prog["chapters"] = sorted_chapters
            self.save()
            print(f"✅ Migrated {migrated_count} entries to use actual_num as key and sorted by chapter number")
        else:
            # Even if no migration occurred, still apply sorting
            self.prog["chapters"] = sorted_chapters
            if "chapter_chunks" in self.prog:
                sorted_chunks = dict(sorted(self.prog["chapter_chunks"].items(), key=sort_key))
                self.prog["chapter_chunks"] = sorted_chunks
            self.save()
            print("✅ Sorted chapters by chapter number")
    
    def get_stats(self, output_dir):
        """Get statistics about translation progress"""
        stats = {
            "total_tracked": len(self.prog["chapters"]),
            "completed": 0,
            "missing_files": 0,
            "in_progress": 0
        }
        
        for chapter_info in self.prog["chapters"].values():
            status = chapter_info.get("status")
            output_file = chapter_info.get("output_file")
            
            if status == "completed" and output_file:
                output_path = os.path.join(output_dir, output_file)
                if os.path.exists(output_path):
                    stats["completed"] += 1
                else:
                    stats["missing_files"] += 1
            elif status == "in_progress":
                stats["in_progress"] += 1
            elif status == "file_missing":
                stats["missing_files"] += 1
        
        return stats

# =====================================================
# UNIFIED CONTENT PROCESSOR
# =====================================================
class ContentProcessor:
    """Unified content processing"""
    
    @staticmethod
    def clean_ai_artifacts(text, remove_artifacts=True):
        """Remove AI response artifacts from text - but ONLY when enabled"""
        if not remove_artifacts:
            return text
        
        # IMPORTANT: Protect split markers used by request merging
        # These must NEVER be removed as they're critical for split-the-merge
        split_marker_pattern = r'<h1[^>]*id="split-\d+"[^>]*>.*?SPLIT MARKER.*?</h1>'
        has_split_markers = bool(re.search(split_marker_pattern, text, re.DOTALL | re.IGNORECASE))
        
        if has_split_markers:
            # Extract and preserve split markers temporarily
            split_markers = []
            def preserve_marker(match):
                marker_id = f"__SPLIT_MARKER_{len(split_markers)}__"
                split_markers.append(match.group(0))
                return marker_id
            
            text = re.sub(split_marker_pattern, preserve_marker, text, flags=re.DOTALL | re.IGNORECASE)
        
        # First, remove thinking tags if they exist
        text = ContentProcessor._remove_thinking_tags(text)
        
        # After removing thinking tags, re-analyze the text structure
        # to catch AI artifacts that may now be at the beginning
        lines = text.split('\n')
        
        # Clean up empty lines at the beginning
        while lines and not lines[0].strip():
            lines.pop(0)
        
        if not lines:
            # Restore split markers before returning
            if has_split_markers:
                for i, marker in enumerate(split_markers):
                    text = text.replace(f"__SPLIT_MARKER_{i}__", marker)
            return text
        
        # Check the first non-empty line for AI artifacts
        first_line = lines[0].strip()
        
        ai_patterns = [
            r'^(?:Sure|Okay|Understood|Of course|Got it|Alright|Certainly|Here\'s|Here is)',
            r'^(?:I\'ll|I will|Let me) (?:translate|help|assist)',
            r'^(?:System|Assistant|AI|User|Human|Model)\s*:',
            r'^\[PART\s+\d+/\d+\]',
            r'^(?:Translation note|Note|Here\'s the translation|I\'ve translated)',
            r'^```(?:html|xml|text)?\s*$',  # Enhanced code block detection
            r'^<!DOCTYPE',
        ]
        
        for pattern in ai_patterns:
            if re.search(pattern, first_line, re.IGNORECASE):
                remaining_lines = lines[1:]
                remaining_text = '\n'.join(remaining_lines)
                
                if remaining_text.strip():
                    # More lenient conditions: if we detect AI artifact patterns and there's meaningful content
                    if (re.search(r'<h[1-6]', remaining_text, re.IGNORECASE) or 
                        re.search(r'Chapter\s+\d+', remaining_text, re.IGNORECASE) or
                        re.search(r'第\s*\d+\s*[章節話话回]', remaining_text) or
                        re.search(r'제\s*\d+\s*[장화]', remaining_text) or
                        re.search(r'<p>', remaining_text, re.IGNORECASE) or
                        len(remaining_text.strip()) > 50):  # Reduced from 100 to 50
                        
                        print(f"✂️ Removed AI artifact: {first_line[:50]}...")
                        return remaining_text.lstrip()
        
        if first_line.lower() in ['html', 'text', 'content', 'translation', 'output']:
            remaining_lines = lines[1:]
            remaining_text = '\n'.join(remaining_lines)
            if remaining_text.strip():
                print(f"✂️ Removed single word artifact: {first_line}")
                result = remaining_text.lstrip()
                # Restore split markers
                if has_split_markers:
                    for i, marker in enumerate(split_markers):
                        result = result.replace(f"__SPLIT_MARKER_{i}__", marker)
                return result
        
        result = '\n'.join(lines)
        
        # Restore split markers before returning
        if has_split_markers:
            for i, marker in enumerate(split_markers):
                result = result.replace(f"__SPLIT_MARKER_{i}__", marker)
        
        return result
    
    @staticmethod
    def _remove_thinking_tags(text):
        """Remove thinking tags that some AI models produce"""
        if not text:
            return text
        
        # Common thinking tag patterns used by various AI models
        thinking_patterns = [
            # XML-style thinking tags
            (r'<thinking>.*?</thinking>', 'thinking'),
            (r'<think>.*?</think>', 'think'),
            (r'<thoughts>.*?</thoughts>', 'thoughts'),
            (r'<reasoning>.*?</reasoning>', 'reasoning'),
            (r'<analysis>.*?</analysis>', 'analysis'),
            (r'<reflection>.*?</reflection>', 'reflection'),
            # OpenAI o1-style reasoning blocks - fix the regex escaping
            (r'<\|thinking\|>.*?</\|thinking\|>', 'o1-thinking'),
            # Claude-style thinking blocks
            (r'\[thinking\].*?\[/thinking\]', 'claude-thinking'),
            # Generic bracketed thinking patterns
            (r'\[THINKING\].*?\[/THINKING\]', 'bracketed-thinking'),
            (r'\[ANALYSIS\].*?\[/ANALYSIS\]', 'bracketed-analysis'),
        ]
        
        original_text = text
        removed_count = 0
        
        for pattern, tag_type in thinking_patterns:
            # Use DOTALL flag to match across newlines
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            if matches:
                text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
                removed_count += len(matches)
        
        # Also remove standalone code block markers that might be artifacts
        # But preserve all actual content - only remove the ``` markers themselves
        code_block_removed = 0
        code_block_patterns = [
            (r'^```\w*\s*\n', '\n'),                # Opening code blocks - replace with newline
            (r'\n```\s*$', ''),                     # Closing code blocks at end - remove entirely
            (r'^```\w*\s*$', ''),                   # Standalone ``` on its own line - remove entirely
        ]
        
        for pattern, replacement in code_block_patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            if matches:
                text = re.sub(pattern, replacement, text, flags=re.MULTILINE)
                code_block_removed += len(matches)
        
        # Clean up any extra whitespace or empty lines left after removing thinking tags
        total_removed = removed_count + code_block_removed
        if total_removed > 0:
            # Remove multiple consecutive newlines
            text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
            # Remove leading/trailing whitespace
            text = text.strip()
            if removed_count > 0 and code_block_removed > 0:
                print(f"🧠 Removed {removed_count} thinking tag(s) and {code_block_removed} code block marker(s)")
            elif removed_count > 0:
                print(f"🧠 Removed {removed_count} thinking tag(s)")
            elif code_block_removed > 0:
                print(f"📝 Removed {code_block_removed} code block marker(s)")
        
        return text
    
    @staticmethod
    def clean_memory_artifacts(text):
        """Remove any memory/summary artifacts that leaked into the translation"""
        text = re.sub(r'\[MEMORY\].*?\[END MEMORY\]', '', text, flags=re.DOTALL)
        
        lines = text.split('\n')
        cleaned_lines = []
        skip_next = False
        
        for line in lines:
            if any(marker in line for marker in ['[MEMORY]', '[END MEMORY]', 'Previous context summary:', 
                                                  'memory summary', 'context summary', '[Context]']):
                skip_next = True
                continue
            
            if skip_next and line.strip() == '':
                skip_next = False
                continue
                
            skip_next = False
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    @staticmethod
    def emergency_restore_paragraphs(text, original_html=None, verbose=True):
        """Emergency restoration when AI returns wall of text without proper paragraph tags"""
        def log(message):
            if verbose:
                print(message)
        
        if text.count('</p>') >= 3:
            return text
        
        if original_html:
            original_para_count = original_html.count('<p>')
            current_para_count = text.count('<p>')
            
            if current_para_count < original_para_count / 2:
                log(f"⚠️ Paragraph mismatch! Original: {original_para_count}, Current: {current_para_count}")
                log("🔧 Attempting emergency paragraph restoration...")
        
        if '</p>' not in text and len(text) > 300:
            log("❌ No paragraph tags found - applying emergency restoration")
            
            if '\n\n' in text:
                parts = text.split('\n\n')
                paragraphs = ['<p>' + part.strip() + '</p>' for part in parts if part.strip()]
                return '\n'.join(paragraphs)
            
            dialogue_pattern = r'(?<=[.!?])\s+(?=[""\u201c\u201d])'
            if re.search(dialogue_pattern, text):
                parts = re.split(dialogue_pattern, text)
                paragraphs = []
                for part in parts:
                    part = part.strip()
                    if part:
                        if not part.startswith('<p>'):
                            part = '<p>' + part
                        if not part.endswith('</p>'):
                            part = part + '</p>'
                        paragraphs.append(part)
                return '\n'.join(paragraphs)
            
            sentence_boundary = r'(?<=[.!?])\s+(?=[A-Z\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af])'
            sentences = re.split(sentence_boundary, text)
            
            if len(sentences) > 1:
                paragraphs = []
                current_para = []
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                        
                    current_para.append(sentence)
                    
                    should_break = (
                        len(current_para) >= 3 or
                        sentence.rstrip().endswith(('"', '"', '"')) or
                        '* * *' in sentence or
                        '***' in sentence or
                        '---' in sentence
                    )
                    
                    if should_break:
                        para_text = ' '.join(current_para)
                        if not para_text.startswith('<p>'):
                            para_text = '<p>' + para_text
                        if not para_text.endswith('</p>'):
                            para_text = para_text + '</p>'
                        paragraphs.append(para_text)
                        current_para = []
                
                if current_para:
                    para_text = ' '.join(current_para)
                    if not para_text.startswith('<p>'):
                        para_text = '<p>' + para_text
                    if not para_text.endswith('</p>'):
                        para_text = para_text + '</p>'
                    paragraphs.append(para_text)
                
                result = '\n'.join(paragraphs)
                log(f"✅ Restored {len(paragraphs)} paragraphs from wall of text")
                return result
            
            words = text.split()
            if len(words) > 100:
                paragraphs = []
                words_per_para = max(100, len(words) // 10)
                
                for i in range(0, len(words), words_per_para):
                    chunk = ' '.join(words[i:i + words_per_para])
                    if chunk.strip():
                        paragraphs.append('<p>' + chunk.strip() + '</p>')
                
                return '\n'.join(paragraphs)
        
        elif '<p>' in text and text.count('<p>') < 3 and len(text) > 1000:
            log("⚠️ Very few paragraphs for long text - checking if more breaks needed")
            
            soup = BeautifulSoup(text, 'html.parser')
            existing_paras = soup.find_all('p')
            
            new_paragraphs = []
            for para in existing_paras:
                para_text = para.get_text()
                if len(para_text) > 500:
                    sentences = re.split(r'(?<=[.!?])\s+', para_text)
                    if len(sentences) > 5:
                        chunks = []
                        current = []
                        for sent in sentences:
                            current.append(sent)
                            if len(current) >= 3:
                                chunks.append('<p>' + ' '.join(current) + '</p>')
                                current = []
                        if current:
                            chunks.append('<p>' + ' '.join(current) + '</p>')
                        new_paragraphs.extend(chunks)
                    else:
                        new_paragraphs.append(str(para))
                else:
                    new_paragraphs.append(str(para))
            
            return '\n'.join(new_paragraphs)
        
        return text
    
    @staticmethod
    def emergency_restore_images(text, original_html=None, verbose=True):
        """Emergency restoration of images lost during translation - Filename Pattern Search"""
        if not original_html or not text:
            return text
            
        def log(message):
            if verbose:
                print(message)
                
        try:
            import re
            import os
            
            # Parse both documents
            soup_orig = BeautifulSoup(original_html, 'html.parser')
            soup_text = BeautifulSoup(text, 'html.parser')
            
            # Extract images from source
            orig_images = soup_orig.find_all('img')
            if not orig_images:
                return text
                
            # Extract images from translation
            text_images = soup_text.find_all('img')
            
            # If counts match, nothing to do
            if len(orig_images) == len(text_images):
                return text
                
            # If translation has fewer images, try to restore them
            if len(text_images) < len(orig_images):
                log(f"🖼️ Image mismatch! Source: {len(orig_images)}, Translation: {len(text_images)}")
                log("🔧 Attempting emergency image restoration (filename search method)...")
                
                # Get the set of image sources present in translation
                present_srcs = set()
                for img in text_images:
                    src = img.get('src')
                    if src:
                        present_srcs.add(src)
                
                # Collect missing images
                missing_images = []
                for img in orig_images:
                    src = img.get('src')
                    if src and src not in present_srcs:
                        missing_images.append((src, img))
                
                if not missing_images:
                    return text
                
                # Convert both to strings for searching
                source_str = str(original_html)
                text_str = str(text)
                inserted_count = 0
                
                # For each missing image, find where it appears in source and insert at same relative position in output
                for src, orig_img in missing_images:
                    # Extract just the filename from the path
                    filename = os.path.basename(src)
                    
                    # Search for the filename in the SOURCE HTML to find its position
                    pattern = re.escape(filename)
                    source_matches = list(re.finditer(pattern, source_str, re.IGNORECASE))
                    
                    if source_matches:
                        # Found the filename in source! Calculate its relative position
                        source_pos = source_matches[0].start()
                        source_len = len(source_str)
                        
                        # Calculate proportional position (0.0 to 1.0)
                        relative_pos = source_pos / source_len if source_len > 0 else 0.5
                        
                        # Calculate corresponding position in translation
                        text_len = len(text_str)
                        insert_pos = int(relative_pos * text_len)
                        
                        # Find a good insertion point (after a tag close, not in the middle of text)
                        # Search backwards for the nearest '>' to insert after a complete tag
                        while insert_pos > 0 and text_str[insert_pos] != '>':
                            insert_pos -= 1
                        insert_pos += 1  # Insert after the '>'
                        
                        # Create the image tag HTML
                        img_html = f'<p><img src="{src}"'
                        for attr, val in orig_img.attrs.items():
                            if attr != 'src':
                                img_html += f' {attr}="{val}"'
                        img_html += '/></p>'
                        
                        # Insert the image HTML at the calculated position
                        text_str = text_str[:insert_pos] + img_html + text_str[insert_pos:]
                        inserted_count += 1
                    else:
                        # Filename not found in source - append to end as fallback
                        soup_text = BeautifulSoup(text_str, 'html.parser')
                        body = soup_text.find('body')
                        if not body:
                            body = soup_text
                        
                        new_p = soup_text.new_tag('p')
                        new_img = soup_text.new_tag('img', src=src)
                        for attr, val in orig_img.attrs.items():
                            if attr != 'src':
                                new_img[attr] = val
                        new_p.append(new_img)
                        body.append(new_p)
                        text_str = str(soup_text)
                        inserted_count += 1
                
                log(f"✅ Restored {inserted_count} missing images using filename search")
                return text_str
                
        except Exception as e:
            log(f"⚠️ Failed to restore images: {e}")
            import traceback
            traceback.print_exc()
            return text
            
        return text

    @staticmethod
    def get_content_hash(html_content):
        """Create a stable hash of content"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            for tag in soup(['script', 'style', 'meta', 'link']):
                tag.decompose()
            
            text_content = soup.get_text(separator=' ', strip=True)
            text_content = ' '.join(text_content.split())
            
            return hashlib.sha256(text_content.encode('utf-8')).hexdigest()
            
        except Exception as e:
            print(f"[WARNING] Failed to create hash: {e}")
            return hashlib.sha256(html_content.encode('utf-8')).hexdigest()
    
    @staticmethod
    def is_meaningful_text_content(html_content):
        """Check if chapter has meaningful text beyond just structure"""
        try:
            # Check if this is plain text from enhanced extraction (html2text output)
            # html2text output characteristics:
            # - Often starts with # for headers
            # - Contains markdown-style formatting
            # - Doesn't have HTML tags
            content_stripped = html_content.strip()
            
            # Quick check for plain text/markdown content
            is_plain_text = False
            if content_stripped and (
                not content_stripped.startswith('<') or  # Doesn't start with HTML tag
                content_stripped.startswith('#') or      # Markdown header
                '\n\n' in content_stripped[:500] or      # Markdown paragraphs
                not '<p>' in content_stripped[:500] and not '<div>' in content_stripped[:500]  # No common HTML tags
            ):
                # This looks like plain text or markdown from html2text
                is_plain_text = True
                
            if is_plain_text:
                # For plain text, just check the length
                text_length = len(content_stripped)
                # Be more lenient with plain text since it's already extracted
                return text_length > 50  # Much lower threshold for plain text
            
            # Original HTML parsing logic
            soup = BeautifulSoup(html_content, 'html.parser')
            
            soup_copy = BeautifulSoup(str(soup), 'html.parser')
            
            for img in soup_copy.find_all('img'):
                img.decompose()
            
            text_elements = soup_copy.find_all(['p', 'div', 'span'])
            text_content = ' '.join(elem.get_text(strip=True) for elem in text_elements)
            
            headers = soup_copy.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            header_text = ' '.join(h.get_text(strip=True) for h in headers)
            
            if headers and len(text_content.strip()) > 1:
                return True
            
            if len(text_content.strip()) > 200:
                return True
            
            if len(header_text.strip()) > 100:
                return True
                
            return False
            
        except Exception as e:
            print(f"Warning: Error checking text content: {e}")
            return True
        
# =====================================================
# UNIFIED TRANSLATION PROCESSOR
# =====================================================
STOP_LOGGED = False

def log_stop_once(message="❌ Translation stopped by user request."):
    """Print a single stop message per run."""
    global STOP_LOGGED
    if not STOP_LOGGED:
        print(message)
        STOP_LOGGED = True
    
class TranslationProcessor:
    """Handles the translation of individual chapters"""
    
    def __init__(self, config, client, out_dir, log_callback=None, stop_callback=None, uses_zero_based=False, is_text_file=False):
        self.config = config
        self.client = client
        self.out_dir = out_dir
        self.log_callback = log_callback
        self.stop_callback = stop_callback
        self.chapter_splitter = ChapterSplitter(model_name=config.MODEL)
        self.uses_zero_based = uses_zero_based
        self.is_text_file = is_text_file
        
        # Check and log multi-key status
        if hasattr(self.client, 'use_multi_keys') and self.client.use_multi_keys:
            stats = self.client.get_stats()
            self._log(f"🔑 Multi-key mode active: {stats.get('total_keys', 0)} keys")
            self._log(f"   Active keys: {stats.get('active_keys', 0)}")
    
    def _log(self, message):
        """Log a message"""
        if self.log_callback:
            self.log_callback(message)
        else:
            print(message)

    def report_key_status(self):
        """Report multi-key status if available"""
        if hasattr(self.client, 'get_stats'):
            stats = self.client.get_stats()
            if stats.get('multi_key_mode', False):
                self._log(f"\n📊 API Key Status:")
                self._log(f"   Active Keys: {stats.get('active_keys', 0)}/{stats.get('total_keys', 0)}")
                self._log(f"   Success Rate: {stats.get('success_rate', 0):.1%}")
                self._log(f"   Total Requests: {stats.get('total_requests', 0)}\n")
        
    def check_stop(self):
        """Check if translation should stop"""
        if self.stop_callback and self.stop_callback():
            log_stop_once()
            return True
    
    def check_duplicate_content(self, result, idx, prog, out, actual_num=None):
        """Check if translated content is duplicate - with mode selection"""
        
        # Get detection mode from config
        detection_mode = getattr(self.config, 'DUPLICATE_DETECTION_MODE', 'basic')
        print(f"    🔍 DEBUG: Detection mode = '{detection_mode}'")
        print(f"    🔍 DEBUG: Lookback chapters = {self.config.DUPLICATE_LOOKBACK_CHAPTERS}")
        
        # Extract content_hash if available from progress
        content_hash = None
        if detection_mode == 'ai-hunter':
            # Try to get content_hash from the current chapter info
            # Use actual_num if provided, otherwise fallback to idx+1
            if actual_num is not None:
                chapter_key = str(actual_num)
            else:
                chapter_key = str(idx + 1)
            if chapter_key in prog.get("chapters", {}):
                chapter_info = prog["chapters"][chapter_key]
                content_hash = chapter_info.get("content_hash")
                print(f"    🔍 DEBUG: Found content_hash for chapter {idx}: {content_hash}")
        
        if detection_mode == 'ai-hunter':
            print("    🤖 DEBUG: Routing to AI Hunter detection...")
            # Check if AI Hunter method is available (injected by the wrapper)
            if hasattr(self, '_check_duplicate_ai_hunter'):
                return self._check_duplicate_ai_hunter(result, idx, prog, out, content_hash)
            else:
                print("    ⚠️ AI Hunter method not available, falling back to basic detection")
                return self._check_duplicate_basic(result, idx, prog, out)
        elif detection_mode == 'cascading':
            print("    🔄 DEBUG: Routing to Cascading detection...")
            return self._check_duplicate_cascading(result, idx, prog, out)
        else:
            print("    📋 DEBUG: Routing to Basic detection...")
            return self._check_duplicate_basic(result, idx, prog, out)

    def _check_duplicate_basic(self, result, idx, prog, out):
        """Original basic duplicate detection"""
        try:
            result_clean = re.sub(r'<[^>]+>', '', result).strip().lower()
            result_sample = result_clean[:1000]
            
            lookback_chapters = self.config.DUPLICATE_LOOKBACK_CHAPTERS
            
            for prev_idx in range(max(0, idx - lookback_chapters), idx):
                prev_key = str(prev_idx)
                if prev_key in prog["chapters"] and prog["chapters"][prev_key].get("output_file"):
                    prev_file = prog["chapters"][prev_key]["output_file"]
                    prev_path = os.path.join(out, prev_file)
                    
                    if os.path.exists(prev_path):
                        try:
                            with open(prev_path, 'r', encoding='utf-8') as f:
                                prev_content = f.read()
                                prev_clean = re.sub(r'<[^>]+>', '', prev_content).strip().lower()
                                prev_sample = prev_clean[:1000]
                                
                                # Use SequenceMatcher for similarity comparison
                                similarity = SequenceMatcher(None, result_sample, prev_sample).ratio()
                                
                                if similarity >= 0.85:  # 85% threshold
                                    print(f"    🚀 Basic detection: Duplicate found ({int(similarity*100)}%)")
                                    return True, int(similarity * 100)
                                    
                        except Exception as e:
                            print(f"    Warning: Failed to read {prev_path}: {e}")
                            continue
            
            return False, 0
            
        except Exception as e:
            print(f"    Warning: Failed to check duplicate content: {e}")
            return False, 0

       
    def _check_duplicate_cascading(self, result, idx, prog, out):
        """Cascading detection - basic first, then AI Hunter for borderline cases"""
        # Step 1: Basic 
        is_duplicate_basic, similarity_basic = self._check_duplicate_basic(result, idx, prog, out)
        
        if is_duplicate_basic:
            return True, similarity_basic
        
        # Step 2: If basic detection finds moderate similarity, use AI Hunter
        if similarity_basic >= 60:  # Configurable threshold
            print(f"    🤖 Moderate similarity ({similarity_basic}%) - running AI Hunter analysis...")
            if hasattr(self, '_check_duplicate_ai_hunter'):
                is_duplicate_ai, similarity_ai = self._check_duplicate_ai_hunter(result, idx, prog, out)
                if is_duplicate_ai:
                    return True, similarity_ai
            else:
                print("    ⚠️ AI Hunter method not available for cascading analysis")
        
        return False, max(similarity_basic, 0)

    def _extract_text_features(self, text):
        """Extract multiple features from text for AI Hunter analysis"""
        features = {
            'semantic': {},
            'structural': {},
            'characters': [],
            'patterns': {}
        }
        
        # Semantic fingerprint
        lines = text.split('\n')
        
        # Character extraction (names that appear 3+ times)
        words = re.findall(r'\b[A-Z][a-z]+\b', text)
        word_freq = Counter(words)
        features['characters'] = [name for name, count in word_freq.items() if count >= 3]
        
        # Dialogue patterns
        dialogue_patterns = re.findall(r'"([^"]+)"', text)
        features['semantic']['dialogue_count'] = len(dialogue_patterns)
        features['semantic']['dialogue_lengths'] = [len(d) for d in dialogue_patterns[:10]]
        
        # Speaker patterns
        speaker_patterns = re.findall(r'(\w+)\s+(?:said|asked|replied|shouted|whispered)', text.lower())
        features['semantic']['speakers'] = list(set(speaker_patterns[:20]))
        
        # Number extraction
        numbers = re.findall(r'\b\d+\b', text)
        features['patterns']['numbers'] = numbers[:20]
        
        # Structural signature
        para_lengths = []
        dialogue_count = 0
        for para in text.split('\n\n'):
            if para.strip():
                para_lengths.append(len(para))
                if '"' in para:
                    dialogue_count += 1
        
        features['structural']['para_count'] = len(para_lengths)
        features['structural']['avg_para_length'] = sum(para_lengths) / max(1, len(para_lengths))
        features['structural']['dialogue_ratio'] = dialogue_count / max(1, len(para_lengths))
        
        # Create structural pattern string
        pattern = []
        for para in text.split('\n\n')[:20]:  # First 20 paragraphs
            if para.strip():
                if '"' in para:
                    pattern.append('D')  # Dialogue
                elif len(para) > 300:
                    pattern.append('L')  # Long
                elif len(para) < 100:
                    pattern.append('S')  # Short
                else:
                    pattern.append('M')  # Medium
        features['structural']['pattern'] = ''.join(pattern)
        
        return features

    def _calculate_exact_similarity(self, text1, text2):
        """Calculate exact text similarity"""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def _calculate_smart_similarity(self, text1, text2):
        """Smart similarity with length-aware sampling"""
        # Check length ratio first
        len_ratio = len(text1) / max(1, len(text2))
        if len_ratio < 0.7 or len_ratio > 1.3:
            return 0.0
        
        # Smart sampling for large texts
        if len(text1) > 10000:
            sample_size = 3000
            samples1 = [
                text1[:sample_size],
                text1[len(text1)//2 - sample_size//2:len(text1)//2 + sample_size//2],
                text1[-sample_size:]
            ]
            samples2 = [
                text2[:sample_size],
                text2[len(text2)//2 - sample_size//2:len(text2)//2 + sample_size//2],
                text2[-sample_size:]
            ]
            similarities = [SequenceMatcher(None, s1.lower(), s2.lower()).ratio() 
                           for s1, s2 in zip(samples1, samples2)]
            return sum(similarities) / len(similarities)
        else:
            # Use first 2000 chars for smaller texts
            return SequenceMatcher(None, text1[:2000].lower(), text2[:2000].lower()).ratio()

    def _calculate_semantic_similarity(self, sem1, sem2):
        """Calculate semantic fingerprint similarity"""
        score = 0.0
        max_score = 0.0
        
        # Compare dialogue counts
        if 'dialogue_count' in sem1 and 'dialogue_count' in sem2:
            max_score += 1.0
            ratio = min(sem1['dialogue_count'], sem2['dialogue_count']) / max(1, max(sem1['dialogue_count'], sem2['dialogue_count']))
            score += ratio * 0.3
        
        # Compare speakers
        if 'speakers' in sem1 and 'speakers' in sem2:
            max_score += 1.0
            if sem1['speakers'] and sem2['speakers']:
                overlap = len(set(sem1['speakers']) & set(sem2['speakers']))
                total = len(set(sem1['speakers']) | set(sem2['speakers']))
                score += (overlap / max(1, total)) * 0.4
        
        # Compare dialogue lengths pattern
        if 'dialogue_lengths' in sem1 and 'dialogue_lengths' in sem2:
            max_score += 1.0
            if sem1['dialogue_lengths'] and sem2['dialogue_lengths']:
                # Compare dialogue length patterns
                len1 = sem1['dialogue_lengths'][:10]
                len2 = sem2['dialogue_lengths'][:10]
                if len1 and len2:
                    avg1 = sum(len1) / len(len1)
                    avg2 = sum(len2) / len(len2)
                    ratio = min(avg1, avg2) / max(1, max(avg1, avg2))
                    score += ratio * 0.3
        
        return score / max(1, max_score)

    def _calculate_structural_similarity(self, struct1, struct2):
        """Calculate structural signature similarity"""
        score = 0.0
        
        # Compare paragraph patterns
        if 'pattern' in struct1 and 'pattern' in struct2:
            pattern_sim = SequenceMatcher(None, struct1['pattern'], struct2['pattern']).ratio()
            score += pattern_sim * 0.4
        
        # Compare paragraph statistics
        if all(k in struct1 for k in ['para_count', 'avg_para_length', 'dialogue_ratio']) and \
           all(k in struct2 for k in ['para_count', 'avg_para_length', 'dialogue_ratio']):
            
            # Paragraph count ratio
            para_ratio = min(struct1['para_count'], struct2['para_count']) / max(1, max(struct1['para_count'], struct2['para_count']))
            score += para_ratio * 0.2
            
            # Average length ratio
            avg_ratio = min(struct1['avg_para_length'], struct2['avg_para_length']) / max(1, max(struct1['avg_para_length'], struct2['avg_para_length']))
            score += avg_ratio * 0.2
            
            # Dialogue ratio similarity
            dialogue_diff = abs(struct1['dialogue_ratio'] - struct2['dialogue_ratio'])
            score += (1 - dialogue_diff) * 0.2
        
        return score

    def _calculate_character_similarity(self, chars1, chars2):
        """Calculate character name similarity"""
        if not chars1 or not chars2:
            return 0.0
        
        # Find overlapping characters
        set1 = set(chars1)
        set2 = set(chars2)
        overlap = len(set1 & set2)
        total = len(set1 | set2)
        
        return overlap / max(1, total)

    def _calculate_pattern_similarity(self, pat1, pat2):
        """Calculate pattern-based similarity"""
        score = 0.0
        
        # Compare numbers (they rarely change in translations)
        if 'numbers' in pat1 and 'numbers' in pat2:
            nums1 = set(pat1['numbers'])
            nums2 = set(pat2['numbers'])
            if nums1 and nums2:
                overlap = len(nums1 & nums2)
                total = len(nums1 | nums2)
                score = overlap / max(1, total)
        
        return score
    
    def generate_rolling_summary(
        self,
        history_manager,
        actual_num,
        base_system_content=None,
        source_text=None,
        previous_summary_text=None,
        previous_summary_chapter_num=None,
        prefer_translations_only_user=False,
    ):
        """Generate rolling summary after a chapter for context continuity.
        Uses a dedicated summary system prompt (with glossary) distinct from translation.
        Writes the summary to rolling_summary.txt and returns the summary string.

        IMPORTANT: The SUMMARY_ROLE setting controls what is sent to the summary API:
          - system: send system prompt + user message containing ONLY the translated text
          - user:   send ONLY a user message (configured prompt template + translated text)
          - both:   send system + user (current/legacy behavior)

        Optional:
          - previous_summary_text: when provided, it is sent as an assistant message for context.
          - prefer_translations_only_user: when True, the user message will be ONLY the translated text
            (even if SUMMARY_ROLE would otherwise use the configured user template).
        """
        if not self.config.USE_ROLLING_SUMMARY:
            return None

        current_history = history_manager.load_history()
        messages_to_include = self.config.ROLLING_SUMMARY_EXCHANGES * 2

        # Prefer directly provided source text (e.g., just-translated chapter) when available
        assistant_responses = []
        if source_text and isinstance(source_text, str) and source_text.strip():
            assistant_responses = [source_text]
        else:
            if len(current_history) >= 2:
                recent_messages = current_history[-messages_to_include:] if messages_to_include > 0 else current_history
                for h in recent_messages:
                    if h.get("role") == "assistant":
                        assistant_responses.append(h["content"])

        # If still empty, skip quietly
        if not assistant_responses:
            return None

        # Build a dedicated summary system prompt (do NOT reuse main translation system prompt)
        # Append glossary to keep terminology consistent
        summary_system_template = os.getenv("ROLLING_SUMMARY_SYSTEM_PROMPT", "You create concise summaries for continuity.").strip()
        try:
            glossary_path = find_glossary_file(self.out_dir)
        except Exception:
            glossary_path = None

        # Rolling summary generation is a summarization-only call; do NOT append glossary here.
        # (This keeps prompts smaller and avoids glossary-compression logic for summaries.)
        _prev_append_glossary_env = os.environ.get("APPEND_GLOSSARY")
        try:
            os.environ["APPEND_GLOSSARY"] = "0"
            system_prompt = build_system_prompt(summary_system_template, glossary_path, source_text=source_text)
        finally:
            if _prev_append_glossary_env is None:
                os.environ.pop("APPEND_GLOSSARY", None)
            else:
                os.environ["APPEND_GLOSSARY"] = _prev_append_glossary_env

        # Add explicit instruction for clarity (glossary usage instructions come from APPEND_GLOSSARY_PROMPT).
        system_prompt += "\n\n[Instruction: Update the rolling summary using any prior summary context provided, plus the newly provided translated text. Do not include warnings or explanations.]"

        user_prompt_template = os.getenv(
            "ROLLING_SUMMARY_USER_PROMPT",
            "Summarize the key events, characters, tone, and important details from these translations. "
            "Focus on: character names/relationships, plot developments, and any special terminology used.\n\n"
            "{translations}"
        )

        translations_text = "\n---\n".join(assistant_responses)
        user_prompt = user_prompt_template.replace("{translations}", translations_text)

        # Optional: provide the previous rolling summary as an assistant message for context.
        # IMPORTANT: This MUST NOT be duplicated into the user message.
        prev_summary_msg = None
        if previous_summary_text and isinstance(previous_summary_text, str) and previous_summary_text.strip():
            prev_summary_msg = {
                "role": "assistant",
                "content": (
                    "[PREVIOUS ROLLING SUMMARY — UPDATE THIS]\n"
                    + previous_summary_text.strip()
                    + "\n[END PREVIOUS ROLLING SUMMARY]"
                ),
            }

        # SUMMARY_ROLE also controls the rolling-summary generation payload.
        # Default to 'both' to preserve legacy behavior when the env var isn't set.
        summary_role = (os.getenv("SUMMARY_ROLE", "both") or "both").strip().lower()

        # When requested, force the user message to be ONLY the translated text.
        if prefer_translations_only_user:
            summary_role = "system"  # ensures we include system prompt + translations-only user message

        if summary_role == "system":
            # System prompt + user content containing ONLY the translated text
            summary_msgs = [{"role": "system", "content": system_prompt}]
            if prev_summary_msg:
                summary_msgs.append(prev_summary_msg)
            summary_msgs.append({"role": "user", "content": translations_text})
        elif summary_role == "user":
            # User prompt only (as configured) with translated text inside it
            summary_msgs = []
            if prev_summary_msg:
                summary_msgs.append(prev_summary_msg)
            summary_msgs.append({"role": "user", "content": user_prompt})
        else:
            # both (current behavior)
            summary_msgs = [{"role": "system", "content": system_prompt}]
            if prev_summary_msg:
                summary_msgs.append(prev_summary_msg)
            summary_msgs.append({"role": "user", "content": f"[Rolling Summary of Chapter {actual_num}]\n" + user_prompt})

        try:
            # Get configurable rolling summary token limit
            # -1 means: use the main MAX_OUTPUT_TOKENS value
            raw_max = os.getenv('ROLLING_SUMMARY_MAX_TOKENS', '-1')
            try:
                rolling_summary_max_tokens = int(str(raw_max).strip())
            except Exception:
                rolling_summary_max_tokens = -1
            
            if rolling_summary_max_tokens == -1:
                rolling_summary_max_tokens = int(getattr(self.config, 'MAX_OUTPUT_TOKENS', 8192))

            send_result = send_with_interrupt(
                summary_msgs, self.client, self.config.TEMP,
                min(int(rolling_summary_max_tokens), self.config.MAX_OUTPUT_TOKENS),
                self.check_stop,
                context='summary'
            )
            
            # send_with_interrupt may return:
            # - a plain string (content)
            # - (content, finish_reason)
            # - (content, finish_reason, raw_obj)
            # We only need the content for rolling summaries.
            if isinstance(send_result, tuple) and len(send_result) >= 1:
                summary_resp = send_result[0]
            else:
                summary_resp = send_result
            
            # Save the summary to the output folder
            summary_file = os.path.join(self.out_dir, "rolling_summary.txt")

            mode = "a" if self.config.ROLLING_SUMMARY_MODE == "append" else "w"

            # Header formatting:
            # - append mode: each appended block corresponds to a specific chapter → keep chapter-specific header
            # - replace mode: file is overwritten and represents the current rolling window → label as "Last N Chapters"
            if mode == "a":
                header_title = f"=== Rolling Summary of Chapter {actual_num} ==="
            else:
                try:
                    _n = int(getattr(self.config, 'ROLLING_SUMMARY_MAX_ENTRIES', 0) or 0)
                except Exception:
                    _n = 0
                header_title = f"=== Rolling Summary of Last {_n} Chapters ===" if _n > 0 else "=== Rolling Summary ==="

            header = header_title + "\n"

            with open(summary_file, mode, encoding="utf-8") as sf:
                if mode == "a":
                    sf.write("\n\n")
                sf.write(header)
                sf.write(summary_resp.strip())

            # If in append mode, trim to retain only the last N entries if configured
            try:
                if self.config.ROLLING_SUMMARY_MODE == "append":
                    max_entries = int(getattr(self.config, "ROLLING_SUMMARY_MAX_ENTRIES", 0) or 0)
                    if max_entries > 0:
                        with open(summary_file, 'r', encoding='utf-8') as rf:
                            content = rf.read()
                        # Find the start of each summary block by header line
                        headers = [m.start() for m in re.finditer(r"(?m)^===\s*Rolling Summary.*$", content)]
                        if len(headers) > max_entries:
                            # Keep only the last max_entries blocks
                            keep_starts = headers[-max_entries:]
                            blocks = []
                            for i, s in enumerate(keep_starts):
                                e = keep_starts[i + 1] if i + 1 < len(keep_starts) else len(content)
                                block = content[s:e].strip()
                                if block:
                                    blocks.append(block)
                            trimmed_content = ("\n\n".join(blocks) + "\n") if blocks else ""
                            with open(summary_file, 'w', encoding='utf-8') as wf:
                                wf.write(trimmed_content)
                            # Optional log showing retained count
                            try:
                                self._log(f"📚 Total summaries in memory: {len(blocks)} (trimmed to last {max_entries})")
                            except Exception:
                                pass
            except Exception as _trim_err:
                try:
                    self._log(f"⚠️ Failed to trim rolling summaries: {_trim_err}")
                except Exception:
                    pass
            
            # Log to GUI if available, otherwise console
            try:
                self._log(f"📝 Generated rolling summary for Chapter {actual_num} ({'append' if mode=='a' else 'replace'} mode)")
                self._log(f"   ➜ Saved to: {summary_file} ({len(summary_resp.strip())} chars)")
            except Exception:
                print(f"📝 Generated rolling summary for Chapter {actual_num} ({'append' if mode=='a' else 'replace'} mode)")
                print(f"   ➜ Saved to: {summary_file} ({len(summary_resp.strip())} chars)")
            return summary_resp.strip()
            
        except Exception as e:
            try:
                self._log(f"⚠️ Failed to generate rolling summary: {e}")
            except Exception:
                print(f"⚠️ Failed to generate rolling summary: {e}")
            return None
    
    def translate_with_retry(self, msgs, chunk_html, c, chunk_idx, total_chunks, merge_group_len=None):
        """Handle translation with retry logic"""
        
        # CRITICAL FIX: Reset client state for each chunk
        if hasattr(self.client, 'reset_cleanup_state'):
            self.client.reset_cleanup_state()
        
        # Also ensure we're not in cleanup mode from previous operations
        if hasattr(self.client, '_in_cleanup'):
            self.client._in_cleanup = False
        if hasattr(self.client, '_cancelled'):
            self.client._cancelled = False
    

        truncation_retry_count = 0
        split_failed_retry_count = 0
        
        # Get retry attempts from AI Hunter config if available
        ai_config = {}
        try:
            # Try to get AI Hunter config from environment variable first
            ai_hunter_config_str = os.getenv('AI_HUNTER_CONFIG')
            if ai_hunter_config_str:
                ai_config = json.loads(ai_hunter_config_str)
            else:
                # Fallback to config attribute
                ai_config = getattr(self.config, 'ai_hunter_config', {})
        except (json.JSONDecodeError, AttributeError):
            ai_config = {}
        
        if isinstance(ai_config, dict):
            max_retries = ai_config.get('retry_attempts', 3)
            max_duplicate_retries = ai_config.get('retry_attempts', 6)  # Use same setting for duplicate retries
        else:
            max_retries = 3
            max_duplicate_retries = 6

        try:
            truncation_retry_limit = int(getattr(self.config, 'TRUNCATION_RETRY_ATTEMPTS', 1))
        except Exception:
            truncation_retry_limit = 1
        try:
            split_failed_retry_limit = int(getattr(self.config, 'SPLIT_FAILED_RETRY_ATTEMPTS', 2))
        except Exception:
            split_failed_retry_limit = 2

        disable_merge_fallback_flag = os.getenv("DISABLE_MERGE_FALLBACK", "0") == "1" or getattr(self.config, 'DISABLE_MERGE_FALLBACK', False)
        truncation_retry_enabled = (os.getenv("RETRY_TRUNCATED", "0") == "1") or bool(getattr(self.config, "RETRY_TRUNCATED", False))
        split_retry_enabled = (os.getenv("RETRY_SPLIT_FAILED", "0") == "1") or bool(getattr(self.config, "RETRY_SPLIT_FAILED", False))
        
        duplicate_retry_count = 0
        timeout_retry_count = 0
        max_timeout_retries = 2
        history_purged = False
        
        original_max_tokens = self.config.MAX_OUTPUT_TOKENS
        original_temp = self.config.TEMP
        original_user_prompt = msgs[-1]["content"]

        # Determine stable chapter number for this chunk (used for payload metadata)
        idx = c.get('__index', 0)
        actual_num = c.get('actual_chapter_num', c.get('num', idx + 1))
        
        # Determine chunk timeout respecting runtime env overrides.
        # If RETRY_TIMEOUT is "0"/false/blank, disable chunk timeouts entirely.
        env_retry = os.getenv("RETRY_TIMEOUT")
        if env_retry is not None:
            retry_timeout_enabled = env_retry.strip().lower() not in ("0", "false", "off", "")
        else:
            retry_timeout_enabled = bool(getattr(self.config, "RETRY_TIMEOUT", False))

        chunk_timeout = None
        if retry_timeout_enabled:
            env_ct = os.getenv("CHUNK_TIMEOUT")
            if env_ct and str(env_ct).strip().lower() not in ("", "none", "0"):
                try:
                    chunk_timeout = int(float(env_ct))
                except Exception:
                    chunk_timeout = getattr(self.config, "CHUNK_TIMEOUT", None)
            else:
                chunk_timeout = getattr(self.config, "CHUNK_TIMEOUT", None)

            # Treat non-positive timeouts as disabled
            try:
                if chunk_timeout is not None and float(chunk_timeout) <= 0:
                    chunk_timeout = None
            except Exception:
                chunk_timeout = None
        
        result = None
        finish_reason = None

        # Fallback stop callback (overridden later for chunked chapters)
        def local_stop_cb():
            return self.check_stop() if hasattr(self, "check_stop") else False
        
        while True:
            if local_stop_cb():
                return None, None, None
            
            try:
                current_max_tokens = self.config.MAX_OUTPUT_TOKENS
                current_temp = self.config.TEMP
                
                # Compute token counts, separating assistant (memory/context) tokens when present
                total_tokens = 0
                assistant_tokens = 0
                for m in msgs:
                    content = m.get("content", "")
                    tokens = self.chapter_splitter.count_tokens(content)
                    total_tokens += tokens
                    if m.get("role") == "assistant":
                        assistant_tokens += tokens
                non_assistant_tokens = total_tokens - assistant_tokens

                # Determine file reference
                if c.get('is_chunk', False):
                    # Handle float chapter numbers in file reference
                    chapter_num_for_ref = c['num']
                    if isinstance(chapter_num_for_ref, float):
                        # Keep decimal notation for display (e.g., "Section_1.0")
                        file_ref = f"Section_{chapter_num_for_ref}"
                    else:
                        file_ref = f"Section_{chapter_num_for_ref}"
                else:
                    # Check if this is a text file - need to access from self
                    is_text_source = self.is_text_file or c.get('filename', '').endswith('.txt')
                    terminology = "Section" if is_text_source else "Chapter"
                    chapter_num_for_ref = c['num']
                    if isinstance(chapter_num_for_ref, float):
                        file_ref = c.get('original_basename', f'{terminology}_{chapter_num_for_ref}')
                    else:
                        file_ref = c.get('original_basename', f'{terminology}_{chapter_num_for_ref}')

                # When contextual translation is enabled and we have assistant-role
                # context (memory, summaries, etc.), surface its token share explicitly.
                if getattr(self.config, 'CONTEXTUAL', False) and assistant_tokens > 0:
                    print(
                        f"💬 Chunk {chunk_idx}/{total_chunks} combined prompt: "
                        f"{total_tokens:,} tokens (system + user: {non_assistant_tokens:,}, "
                        f"assistant/memory: {assistant_tokens:,}) / {self.get_token_budget_str()} [File: {file_ref}]"
                    )
                else:
                    print(
                        f"💬 Chunk {chunk_idx}/{total_chunks} combined prompt: "
                        f"{total_tokens:,} tokens (system + user) / {self.get_token_budget_str()} [File: {file_ref}]"
                    )
                
                self.client.context = 'translation'

                # Generate filename for chunks
                if chunk_idx and total_chunks > 1:
                    # This is a chunk - use chunk naming format
                    # Handle float chapter numbers (e.g., 1.0, 2.5) properly
                    chapter_num = c['num']
                    if isinstance(chapter_num, float):
                        # For decimal chapters like 1.5, use format like "response_001_5_chunk_1.html"
                        major = int(chapter_num)
                        minor = int(round((chapter_num - major) * 100))  # 1.5 -> 50, 1.1 -> 10
                        if minor > 0:
                            fname = f"response_{major:03d}_{minor:02d}_chunk_{chunk_idx}.html"
                        else:
                            # It's like 1.0, just use the integer part
                            fname = f"response_{major:03d}_chunk_{chunk_idx}.html"
                    else:
                        fname = f"response_{chapter_num:03d}_chunk_{chunk_idx}.html"
                else:
                    # Not a chunk - use regular naming
                    fname = FileUtilities.create_chapter_filename(c, c.get('actual_chapter_num', c['num']))

                # Set output filename BEFORE the API call
                if hasattr(self.client, 'set_output_filename'):
                    self.client.set_output_filename(fname)
                
                # Track the filename so truncation logs know which file this is
                if hasattr(self.client, '_current_output_file'):
                    self.client._current_output_file = fname

                # Generate unique request ID for this chunk
                #request_id = f"{c['num']:03d}_chunk{chunk_idx}_{uuid.uuid4().hex[:8]}"

                chapter_ctx = {
                    'chapter': actual_num,
                    'chunk': chunk_idx,
                    'total_chunks': total_chunks,
                }
                
                result, finish_reason, raw_obj = send_with_interrupt(
                    msgs,
                    self.client,
                    current_temp,
                    current_max_tokens,
                    local_stop_cb,
                    chunk_timeout,
                    context='translation',
                    chapter_context=chapter_ctx,
                )
                
                # Enhanced mode workflow:
                # 1. Original HTML -> html2text -> Markdown/plain text (during extraction)
                # 2. Markdown sent to translation API (better for translation quality)
                # 3. Translated markdown -> HTML conversion (here)
                if result and c.get("enhanced_extraction", False):
                    print(f"🔄 Converting translated markdown back to HTML...")
                    result = convert_enhanced_text_to_html(result, c)
                
                # Emergency Image Restoration (if enabled)
                if result and self.config.EMERGENCY_IMAGE_RESTORE:
                    result = ContentProcessor.emergency_restore_images(result, chunk_html)
                    
                retry_needed = False
                retry_reason = ""
                retry_limit_for_reason = None
                is_duplicate_retry = False
                
                # Debug logging to verify the toggle state
                #print(f"    DEBUG: finish_reason='{finish_reason}', truncation_enabled={truncation_retry_enabled}, split_retry_enabled={split_retry_enabled}")
                if finish_reason == "length":
                    if truncation_retry_enabled and truncation_retry_count < truncation_retry_limit:
                        # Always attempt a truncation retry, even if token limits are equal
                        new_token_limit = self.config.MAX_RETRY_TOKENS
                        retry_needed = True
                        retry_reason = "truncated output"
                        retry_limit_for_reason = truncation_retry_limit
                        old_limit = self.config.MAX_OUTPUT_TOKENS
                        self.config.MAX_OUTPUT_TOKENS = new_token_limit
                        truncation_retry_count += 1
                        print(f"    🔄 TRUNCATION RETRY: Attempt {truncation_retry_count}/{truncation_retry_limit} — tokens {old_limit} → {new_token_limit}")
                    elif truncation_retry_enabled:
                        print(f"    ⚠️ TRUNCATION DETECTED: Max truncation retries ({truncation_retry_limit}) reached - accepting truncated response")
                    else:
                        print(f"    ⏭️ TRUNCATION DETECTED: Auto-retry is DISABLED - accepting truncated response")

                # Treat split failures like truncation for auto-retry
                split_failed_in_finish = bool(finish_reason and 'split' in str(finish_reason).lower())
                split_failed_in_body = bool(isinstance(result, str) and 'SPLIT_FAILED' in result)
                
                # Check for split markers if this is a merged request
                split_validation_failed = False
                if merge_group_len and merge_group_len > 1 and result and isinstance(result, str):
                    # We need to import RequestMerger here or assume it's available in module scope
                    # RequestMerger is defined at module level
                    try:
                        # Clean artifacts first? No, we want to check raw result usually, 
                        # but split_by_markers is robust. 
                        # However, translate_with_retry doesn't clean artifacts yet.
                        # Let's try splitting.
                        split_sections = RequestMerger.split_by_markers(result, merge_group_len)
                        if not split_sections or len(split_sections) != merge_group_len:
                            print(f"    ⚠️ Split validation failed: Expected {merge_group_len} sections")
                            split_validation_failed = True
                    except Exception as e:
                        print(f"    ⚠️ Split validation error: {e}")
                        split_validation_failed = True

                if not retry_needed and (split_failed_in_finish or split_failed_in_body or split_validation_failed) and split_retry_enabled:
                    if split_failed_retry_count < split_failed_retry_limit:
                        retry_needed = True
                        retry_reason = "split failed"
                        retry_limit_for_reason = split_failed_retry_limit
                        split_failed_retry_count += 1
                        print(f"    🔄 Split failed — retrying merged request (attempt {split_failed_retry_count}/{split_failed_retry_limit})")
                    else:
                        print(f"    ⚠️ SPLIT FAILED: Max split-failed retries ({split_failed_retry_limit}) reached - accepting response")
                
                if not retry_needed:
                    # Force re-read the environment variable to ensure we have current setting
                    duplicate_enabled = os.getenv("RETRY_DUPLICATE_BODIES", "0") == "1"
                    
                    if duplicate_enabled and duplicate_retry_count < max_duplicate_retries:
                        idx = c.get('__index', 0)
                        prog = c.get('__progress', {})
                        print(f"    🔍 Checking for duplicate content...")
                        # Get actual chapter number for duplicate detection
                        actual_num = c.get('actual_chapter_num', c.get('num', idx + 1))
                        is_duplicate, similarity = self.check_duplicate_content(result, idx, prog, self.out_dir, actual_num)
                        
                        if is_duplicate:
                            retry_needed = True
                            is_duplicate_retry = True
                            retry_reason = f"duplicate content (similarity: {similarity}%)"
                            duplicate_retry_count += 1
                            
                            # Check if temperature change is disabled
                            disable_temp_change = ai_config.get('disable_temperature_change', False) if isinstance(ai_config, dict) else False
                            
                            if duplicate_retry_count >= 3 and not history_purged:
                                print(f"    🧹 Clearing history after 3 attempts...")
                                if 'history_manager' in c:
                                    c['history_manager'].save_history([])
                                history_purged = True
                                if not disable_temp_change:
                                    self.config.TEMP = original_temp
                                else:
                                    print(f"    🌡️ Temperature change disabled - keeping current temp: {self.config.TEMP}")
                            
                            elif duplicate_retry_count == 1:
                                if disable_temp_change:
                                    print(f"    🔄 First duplicate retry - temperature change disabled")
                                else:
                                    print(f"    🔄 First duplicate retry - same temperature")
                            
                            elif history_purged:
                                if not disable_temp_change:
                                    attempts_since_purge = duplicate_retry_count - 3
                                    self.config.TEMP = min(original_temp + (0.1 * attempts_since_purge), 1.0)
                                    print(f"    🌡️ Post-purge temp: {self.config.TEMP}")
                                else:
                                    print(f"    🌡️ Temperature change disabled - keeping temp: {self.config.TEMP}")
                            
                            else:
                                if not disable_temp_change:
                                    self.config.TEMP = min(original_temp + (0.1 * (duplicate_retry_count - 1)), 1.0)
                                    print(f"    🌡️ Gradual temp increase: {self.config.TEMP}")
                                else:
                                    print(f"    🌡️ Temperature change disabled - keeping temp: {self.config.TEMP}")
                            
                            if duplicate_retry_count == 1:
                                user_prompt = f"[RETRY] Chapter {c['num']}: Ensure unique translation.\n{chunk_html}"
                            elif duplicate_retry_count <= 3:
                                user_prompt = f"[ATTEMPT {duplicate_retry_count}] Translate uniquely:\n{chunk_html}"
                            else:
                                user_prompt = f"Chapter {c['num']}:\n{chunk_html}"
                            
                            msgs[-1] = {"role": "user", "content": user_prompt}
                    
                if retry_needed:
                    if is_duplicate_retry:
                        print(f"    🔄 Duplicate retry {duplicate_retry_count}/{max_duplicate_retries}")
                    
                    time.sleep(2)
                    continue
                
                break
                
            except UnifiedClientError as e:
                error_msg = str(e)
                
                if "stopped by user" in error_msg:
                    print("❌ Translation stopped by user during API call")
                    return None, None, None
                
                if "took" in error_msg and "timeout:" in error_msg:
                    if timeout_retry_count < max_timeout_retries:
                        timeout_retry_count += 1
                        print(f"    ⏱️ Chunk took too long, retry {timeout_retry_count}/{max_timeout_retries}")
                        print(f"    🔄 Retrying")
                        time.sleep(2)
                        continue
                    else:
                        print(f"    ❌ Max timeout retries reached")
                        raise UnifiedClientError("Translation failed after timeout retries")
                
                elif "timed out" in error_msg and "timeout:" not in error_msg:
                    print(f"⚠️ {error_msg}, retrying...")
                    time.sleep(5)
                    continue
                
                elif getattr(e, "error_type", None) == "rate_limit" or getattr(e, "http_status", None) == 429:
                    # Rate limit errors - clean handling without traceback
                    print("⚠️ Rate limited, sleeping 60s…")
                    for i in range(60):
                        if self.check_stop():
                            print("❌ Translation stopped during rate limit wait")
                            return None, None, None
                        time.sleep(1)
                    continue
                
                else:
                    # For unexpected errors, show the error message but suppress traceback in most cases
                    if getattr(e, "error_type", None) in ["api_error", "validation", "prohibited_content"]:
                        print(f"❌ API Error: {error_msg}")
                        raise UnifiedClientError(f"API Error: {error_msg}")
                    else:
                        raise
            
            except Exception as e:
                print(f"❌ Unexpected error during API call: {e}")
                import traceback
                print(f"Full traceback:\n{traceback.format_exc()}")
                raise
        
        self.config.MAX_OUTPUT_TOKENS = original_max_tokens
        self.config.TEMP = original_temp
        
        total_simple_retries = truncation_retry_count + split_failed_retry_count
        if total_simple_retries > 0 or duplicate_retry_count > 0 or timeout_retry_count > 0:
            if duplicate_retry_count > 0:
                print(f"    🔄 Restored original temperature: {self.config.TEMP} (after {duplicate_retry_count} duplicate retries)")
            elif timeout_retry_count > 0:
                print(f"    🔄 Restored original settings after {timeout_retry_count} timeout retries")
            elif total_simple_retries > 0:
                print(f"    🔄 Restored original settings after {total_simple_retries} retries")
        
        if duplicate_retry_count >= max_duplicate_retries:
            print(f"    ⚠️ WARNING: Duplicate content issue persists after {max_duplicate_retries} attempts")
        
        return result, finish_reason, raw_obj
    
    def get_token_budget_str(self):
        """Get token budget as string"""
        _tok_env = os.getenv("MAX_INPUT_TOKENS", "1000000").strip()
        max_tokens_limit, budget_str = parse_token_limit(_tok_env)
        return budget_str

# =====================================================
# BATCH TRANSLATION PROCESSOR
# =====================================================
class BatchTranslationProcessor:
    """Handles batch/parallel translation processing"""
    
    def __init__(self, config, client, base_msg, out_dir, progress_lock, 
                 save_progress_fn, update_progress_fn, check_stop_fn, 
                 image_translator=None, is_text_file=False, history_manager=None):
        self.config = config
        self.client = client
        self.base_msg = base_msg
        self.out_dir = out_dir
        self.progress_lock = progress_lock
        self.save_progress_fn = save_progress_fn
        self.update_progress_fn = update_progress_fn
        self.check_stop_fn = check_stop_fn
        self.image_translator = image_translator
        self.chapters_completed = 0
        self.chunks_completed = 0
        self.is_text_file = is_text_file
        # Optional shared HistoryManager for contextual translation across chapters
        self.history_manager = history_manager

        # Rolling summary support (batch mode): inject a snapshot per batch.
        # This is updated by the main thread between batches.
        import threading
        self._batch_rolling_summary_lock = threading.Lock()
        self._batch_rolling_summary_text = ""  # exact rolling_summary.txt contents for current batch
        
       # Optionally log multi-key status
        if hasattr(self.client, 'use_multi_keys') and self.client.use_multi_keys:
            stats = self.client.get_stats()
            print(f"🔑 Batch processor using multi-key mode: {stats.get('total_keys', 0)} keys")

    def set_batch_rolling_summary_text(self, text: str) -> None:
        """Set the rolling summary snapshot to be injected for the current batch."""
        try:
            if text is None:
                text = ""
        except Exception:
            text = ""
        with self._batch_rolling_summary_lock:
            self._batch_rolling_summary_text = text

    def get_batch_rolling_summary_text(self) -> str:
        """Get the rolling summary snapshot (thread-safe)."""
        with self._batch_rolling_summary_lock:
            return self._batch_rolling_summary_text
    
    def process_single_chapter(self, chapter_data):
        """Process a single chapter (runs in thread)"""
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # APPLY INTERRUPTIBLE THREADING DELAY FIRST
        thread_delay = float(os.getenv("THREAD_SUBMISSION_DELAY_SECONDS", "0.5"))
        if thread_delay > 0:
            # Check if we need to wait (same logic as unified_api_client)
            if hasattr(self.client, '_thread_submission_lock') and hasattr(self.client, '_last_thread_submission_time'):
                with self.client._thread_submission_lock:
                    current_time = time.time()
                    time_since_last = current_time - self.client._last_thread_submission_time
                    
                    if time_since_last < thread_delay:
                        sleep_time = thread_delay - time_since_last
                        thread_name = threading.current_thread().name
                        
                        # PRINT BEFORE THE DELAY STARTS
                        idx, chapter = chapter_data  # Extract chapter info for better logging
                        print(f"🧵 [{thread_name}] Applying thread delay: {sleep_time:.1f}s for Chapter {idx+1}")
                        
                        # Interruptible sleep - check stop flag every 0.1 seconds
                        elapsed = 0
                        check_interval = 0.1
                        while elapsed < sleep_time:
                            if self.check_stop_fn():
                                print(f"🛑 Threading delay interrupted by stop flag")
                                raise Exception("Translation stopped by user during threading delay")
                            
                            sleep_chunk = min(check_interval, sleep_time - elapsed)
                            time.sleep(sleep_chunk)
                            elapsed += sleep_chunk
                    
                    self.client._last_thread_submission_time = time.time()
                    if not hasattr(self.client, '_thread_submission_count'):
                        self.client._thread_submission_count = 0
                    self.client._thread_submission_count += 1
        
        idx, chapter = chapter_data
        chap_num = chapter["num"]
        
        # Use the pre-calculated actual_chapter_num from the main loop
        actual_num = chapter.get('actual_chapter_num')
        
        # Fallback if not set (common in batch mode where first pass might be skipped)
        if actual_num is None:
            # CHUNK FIX: For split text/PDF chunks with decimal numbering, use chap_num directly
            # Chunks have 'is_chunk' flag and decimal 'num' values (1.0, 1.1, etc.)
            if chapter.get('is_chunk', False) and isinstance(chap_num, float):
                actual_num = chap_num
            else:
                # Try to extract it using the same logic as non-batch mode
                raw_num = FileUtilities.extract_actual_chapter_number(chapter, patterns=None, config=self.config)
                
                # Apply offset if configured
                offset = self.config.CHAPTER_NUMBER_OFFSET if hasattr(self.config, 'CHAPTER_NUMBER_OFFSET') else 0
                raw_num += offset
                
                # Check if zero detection is disabled
                if hasattr(self.config, 'DISABLE_ZERO_DETECTION') and self.config.DISABLE_ZERO_DETECTION:
                    actual_num = raw_num
                elif hasattr(self.config, '_uses_zero_based') and self.config._uses_zero_based:
                    # This is a 0-based novel, adjust the number
                    actual_num = raw_num + 1
                else:
                    # Default to raw number (1-based or unknown)
                    actual_num = raw_num
                
                print(f"    📖 Extracted actual chapter number: {actual_num} (from raw: {raw_num})")
        
        # Initialize variables that might be needed in except block
        content_hash = None
        ai_features = None
        
        try:
            # Check if this is from a text file
            is_text_source = self.is_text_file or chapter.get('filename', '').endswith('.txt') or chapter.get('is_chunk', False)
            terminology = "Section" if is_text_source else "Chapter"
            print(f"🔄 Starting #{idx+1} (Internal: {terminology} {chap_num}, Actual: {terminology} {actual_num})  (thread: {threading.current_thread().name}) [File: {chapter.get('original_basename', f'{terminology}_{chap_num}')}]")
                      
            content_hash = chapter.get("content_hash") or ContentProcessor.get_content_hash(chapter["body"])
            
            # Determine output filename early so we can track it in progress
            fname = FileUtilities.create_chapter_filename(chapter, actual_num)
            
            with self.progress_lock:
                self.update_progress_fn(idx, actual_num, content_hash, fname, status="in_progress")
                self.save_progress_fn()
            
            chapter_body = chapter["body"]
            if chapter.get('has_images') and self.image_translator and self.config.ENABLE_IMAGE_TRANSLATION:
                print(f"🖼️ Processing images for Chapter {actual_num}...")
                self.image_translator.set_current_chapter(actual_num)
                chapter_body, image_translations = process_chapter_images(
                    chapter_body, 
                    actual_num, 
                    self.image_translator,
                    self.check_stop_fn
                )
                if image_translations:
                    # Create a copy of the processed body
                    from bs4 import BeautifulSoup 
                    c = chapter
                    soup_for_text = BeautifulSoup(c["body"], 'html.parser')
                    
                    # Remove all translated content
                    for trans_div in soup_for_text.find_all('div', class_='translated-text-only'):
                        trans_div.decompose()
                    
                    # Use this cleaned version for text translation
                    text_to_translate = str(soup_for_text)
                    final_body_with_images = c["body"]
                else:
                    text_to_translate = c["body"]
                    image_translations = {}
                    print(f"✅ Processed {len(image_translations)} images for Chapter {actual_num}")
            
            # Build chapter-specific system prompt with glossary compression
            glossary_path = find_glossary_file(self.out_dir)
            
            # Capture compression stats if enabled
            compress_glossary_enabled = os.getenv("COMPRESS_GLOSSARY_PROMPT", "0") == "1"
            if compress_glossary_enabled and glossary_path and os.path.exists(glossary_path):
                try:
                    # Load glossary to get original size
                    with open(glossary_path, 'r', encoding='utf-8') as f:
                        if glossary_path.lower().endswith(('.csv', '.md', '.txt')):
                            original_glossary = f.read()
                        else:
                            try:
                                glossary_data = json.load(f)
                                original_glossary = json.dumps(glossary_data, ensure_ascii=False, indent=2)
                            except json.JSONDecodeError:
                                # If JSON parsing fails, treat as text
                                f.seek(0)
                                original_glossary = f.read()
                    
                    original_length = len(original_glossary)
                    
                    # Build system prompt with compression
                    chapter_system_prompt = build_system_prompt(self.config.SYSTEM_PROMPT, glossary_path, source_text=chapter_body)
                    
                    # Extract compressed glossary from system prompt to measure compression
                    # The glossary is appended after the prompt, so we can estimate the size
                    prompt_without_glossary = self.config.SYSTEM_PROMPT
                    glossary_in_prompt = len(chapter_system_prompt) - len(prompt_without_glossary) if len(chapter_system_prompt) > len(prompt_without_glossary) else 0
                    
                    if glossary_in_prompt > 0 and original_length > glossary_in_prompt:
                        reduction_pct = ((original_length - glossary_in_prompt) / original_length * 100)
                        
                        # Calculate token savings
                        try:
                            import tiktoken
                            try:
                                enc = tiktoken.encoding_for_model(self.config.MODEL)
                            except:
                                enc = tiktoken.get_encoding('cl100k_base')
                            
                            original_tokens = len(enc.encode(original_glossary))
                            compressed_tokens = len(enc.encode(chapter_system_prompt)) - len(enc.encode(prompt_without_glossary))
                            token_reduction_pct = ((original_tokens - compressed_tokens) / original_tokens * 100) if original_tokens > 0 else 0
                            
                            print(f"🗜️ Glossary: {original_length:,}→{glossary_in_prompt:,} chars ({reduction_pct:.1f}%), {original_tokens:,}→{compressed_tokens:,} tokens ({token_reduction_pct:.1f}%)")
                        except ImportError:
                            print(f"🗜️ Glossary compressed: {original_length:,} → {glossary_in_prompt:,} chars ({reduction_pct:.1f}% reduction)")
                    
                except Exception as e:
                    print(f"⚠️ Failed to measure glossary compression: {e}")
                    chapter_system_prompt = build_system_prompt(self.config.SYSTEM_PROMPT, glossary_path, source_text=chapter_body)
            else:
                chapter_system_prompt = build_system_prompt(self.config.SYSTEM_PROMPT, glossary_path, source_text=chapter_body)
            
            # Check if chapter needs chunking
            from chapter_splitter import ChapterSplitter
            chapter_splitter = ChapterSplitter(model_name=self.config.MODEL)
            
            # Get token budget
            token_env = os.getenv("MAX_INPUT_TOKENS", "1000000").strip()
            if not token_env or token_env.lower() == "unlimited":
                max_input_tokens = 1000000
                budget_str = "unlimited"
            elif token_env.isdigit():
                max_input_tokens = int(token_env)
                budget_str = f"{max_input_tokens:,}"
            else:
                max_input_tokens = 1000000
                budget_str = "unlimited"
            
            # Calculate available tokens for content based on effective OUTPUT limit (same as calculation phase)
            # Use output token limit with compression factor, not input limit
            max_output_tokens = self.config.get_effective_output_limit()
            safety_margin_output = 500
            compression_factor = self.config.COMPRESSION_FACTOR
            available_tokens = int((max_output_tokens - safety_margin_output) / compression_factor)
            available_tokens = max(available_tokens, 1000)  # Ensure minimum
            
            # Split into chunks if needed
            # Get filename for content type detection
            chapter_filename = chapter.get('filename') or chapter.get('original_basename', '')
            chunks = chapter_splitter.split_chapter(chapter_body, available_tokens, filename=chapter_filename)
            total_chunks = len(chunks)
            
            file_ref = chapter.get('original_basename', f'{terminology}_{chap_num}')
            
            # Initialize shared structures for chunk processing (works for 1 or many chunks)
            translated_chunks = [None] * total_chunks  # Pre-allocate to maintain order
            chunks_lock = threading.Lock()
            chunk_abort = False

            if total_chunks > 1:
                print(f"✂️ Chapter {actual_num} requires {total_chunks} chunks - processing in parallel")
            
            def process_chunk(chunk_data):
                """Process a single chunk in parallel"""
                chunk_html, chunk_idx, chunk_total = chunk_data
                
                if local_stop_cb():
                    return None, chunk_idx
                
                # Build user prompt for this chunk
                if total_chunks > 1:
                    chunk_prompt_template = os.getenv("TRANSLATION_CHUNK_PROMPT", "[PART {chunk_idx}/{total_chunks}]\\n{chunk_html}")
                    user_prompt = chunk_prompt_template.format(
                        chunk_idx=chunk_idx,
                        total_chunks=total_chunks,
                        chunk_html=chunk_html
                    )
                else:
                    user_prompt = chunk_html
                
                # Build history-based memory when contextual translation is enabled
                memory_msgs = []
                if (
                    self.config.CONTEXTUAL
                    and self.history_manager is not None
                    and getattr(self.config, 'HIST_LIMIT', 0) > 0
                ):
                    try:
                        # Thread-safe history access - load_history() already has internal locking
                        history = self.history_manager.load_history()
                        hist_limit = getattr(self.config, 'HIST_LIMIT', 0)
                        trimmed = history[-hist_limit * 2:]
                        include_source = os.getenv("INCLUDE_SOURCE_IN_HISTORY", "0") == "1"

                        model_lower = getattr(self.config, 'MODEL', '').lower()
                        is_gemini_3 = ('gemini-3' in model_lower) or ('gemini-exp-' in model_lower)

                        if is_gemini_3:
                            # Preserve raw content (thought signatures) and reconstruct text when missing
                            for h in trimmed:
                                if not isinstance(h, dict):
                                    continue
                                role = h.get('role', 'user')
                                raw_obj = h.get('_raw_content_object')
                                content = h.get('content') or ""

                                if (not content) and raw_obj:
                                    content = extract_text_from_raw_content(raw_obj)

                                # Skip empty entries unless raw content exists
                                if not content and raw_obj is None:
                                    continue

                                if role == 'user' and not include_source:
                                    continue

                                msg = {'role': role}
                                if content:
                                    msg['content'] = content
                                if raw_obj is not None:
                                    msg['_raw_content_object'] = raw_obj
                                memory_msgs.append(msg)
                        else:
                            # Original memory block approach for non-Gemini 3 models
                            memory_blocks = []
                            for h in trimmed:
                                if not isinstance(h, dict):
                                    continue
                                role = h.get('role', 'user')
                                content = h.get('content', '')
                                if not content:
                                    continue
                                # Optionally skip previous source text when disabled
                                if role == 'user' and not include_source:
                                    continue
                                if role == 'user':
                                    prefix = (
                                        "[MEMORY - PREVIOUS SOURCE TEXT]\\n"
                                        "This is prior source content provided for context only.\\n"
                                        "Do NOT translate or repeat this text directly in your response.\\n\\n"
                                    )
                                else:
                                    prefix = (
                                        "[MEMORY - PREVIOUS TRANSLATION]\\n"
                                        "This is prior translated content provided for context only.\\n"
                                        "Do NOT repeat or re-output this translation.\\n\\n"
                                    )
                                footer = "\\n\\n[END MEMORY BLOCK]\\n"
                                memory_blocks.append(prefix + content + footer)

                            if memory_blocks:
                                combined_memory = "\\n".join(memory_blocks)
                                # Present history as an assistant message so the model
                                # treats it as prior context, not a new user instruction.
                                memory_msgs = [{
                                    'role': 'assistant',
                                    'content': combined_memory
                                }]
                    except Exception as e:
                        print(f"⚠️ Failed to build contextual memory for batch chunk: {e}")
                        memory_msgs = []
                
                # Build messages for this chunk (system + optional rolling summary + optional memory + user)
                rolling_summary_msgs = []
                if getattr(self.config, 'USE_ROLLING_SUMMARY', False):
                    try:
                        rs_text = self.get_batch_rolling_summary_text()
                    except Exception:
                        rs_text = ""
                    if isinstance(rs_text, str) and rs_text:
                        # Do not strip/parse the file content. Only wrap to prevent accidental translation.
                        rolling_summary_msgs = [{
                            "role": "assistant",
                            "content": (
                                "CONTEXT ONLY - DO NOT INCLUDE IN TRANSLATION:\n"
                                "[MEMORY] Previous context summary:\n\n"
                                + rs_text + "\n\n"
                                "[END MEMORY]\n"
                                "END OF CONTEXT - BEGIN ACTUAL CONTENT TO TRANSLATE:"
                            )
                        }]

                chapter_msgs = (
                    [{"role": "system", "content": chapter_system_prompt}]
                    + rolling_summary_msgs
                    + memory_msgs
                    + [{"role": "user", "content": user_prompt}]
                )

                # Abort immediately if a prior chunk triggered stop/prohibition
                if local_stop_cb():
                    raise UnifiedClientError("Chunk aborted due to earlier failure", error_type="cancelled")
                
                # Log combined prompt token count, including assistant/memory tokens when present
                total_tokens = 0
                assistant_tokens = 0
                for msg in chapter_msgs:
                    content = msg.get("content", "")
                    tokens = chapter_splitter.count_tokens(content)
                    total_tokens += tokens
                    if msg.get("role") == "assistant":
                        assistant_tokens += tokens
                non_assistant_tokens = total_tokens - assistant_tokens

                if self.config.CONTEXTUAL and assistant_tokens > 0:
                    print(
                        f"💬 Chunk {chunk_idx}/{total_chunks} combined prompt: "
                        f"{total_tokens:,} tokens (system + user: {non_assistant_tokens:,}, "
                        f"assistant/memory: {assistant_tokens:,}) / {budget_str} [File: {file_ref}]"
                    )
                else:
                    print(
                        f"💬 Chunk {chunk_idx}/{total_chunks} combined prompt: "
                        f"{total_tokens:,} tokens (system + user) / {budget_str} [File: {file_ref}]"
                    )
                
                # Generate filename before API call
                if chunk_idx < total_chunks:
                    # This is a chunk - use chunk naming format
                    # Handle float chapter numbers (e.g., 1.0, 2.5) properly
                    if isinstance(actual_num, float):
                        # For decimal chapters like 1.5, use format like "response_001_5_chunk_1.html"
                        major = int(actual_num)
                        minor = int(round((actual_num - major) * 100))  # 1.5 -> 50, 1.1 -> 10
                        if minor > 0:
                            fname = f"response_{major:03d}_{minor:02d}_chunk_{chunk_idx}.html"
                        else:
                            # It's like 1.0, just use the integer part
                            fname = f"response_{major:03d}_chunk_{chunk_idx}.html"
                    else:
                        fname = f"response_{actual_num:03d}_chunk_{chunk_idx}.html"
                else:
                    # Last chunk or single chunk - use regular naming
                    fname = FileUtilities.create_chapter_filename(chapter, actual_num)
                
                if hasattr(self.client, 'set_output_filename'):
                    self.client.set_output_filename(fname)

                if hasattr(self.client, '_current_output_file'):
                    self.client._current_output_file = fname

                # Set thread-local label so downstream logs include chapter/chunk
                try:
                    tls = self.client._get_thread_local_client()
                    tls.current_request_label = f"Chapter {actual_num} (chunk {chunk_idx}/{total_chunks})"
                except Exception:
                    pass

                print(f"📤 Sending Chapter {actual_num}, Chunk {chunk_idx}/{total_chunks} to API...")
                chapter_ctx = {
                    'chapter': actual_num,
                    'chunk': chunk_idx,
                    'total_chunks': total_chunks,
                }
                result, finish_reason, raw_obj_from_send = send_with_interrupt(
                    chapter_msgs,
                    self.client,
                    self.config.TEMP,
                    self.config.MAX_OUTPUT_TOKENS,
                    local_stop_cb,
                    context='translation',
                    chapter_context=chapter_ctx,
                )
                
                # Use the raw object directly from send_with_interrupt
                raw_obj = raw_obj_from_send
                # if raw_obj:
                #     print(f"🧠 Captured thought signature for chunk {chunk_idx}/{total_chunks}")
                
                print(f"📥 Received Chapter {actual_num}, Chunk {chunk_idx}/{total_chunks} response, finish_reason: {finish_reason}")
                
                # Treat truncation retries exhaustion as truncation even if finish_reason changed
                # In batch mode each worker has its own thread-local client; check that flag too
                try:
                    tls_client = self.client._get_thread_local_client()
                except Exception:
                    tls_client = None

                truncation_exhausted = False
                if tls_client is not None:
                    truncation_exhausted = getattr(tls_client, "_truncation_retries_exhausted", False)
                if not truncation_exhausted:
                    truncation_exhausted = getattr(self.client, "_truncation_retries_exhausted", False)

                # Clear the flag on whichever client had it so it doesn't bleed into later calls
                try:
                    if tls_client is not None and getattr(tls_client, "_truncation_retries_exhausted", False):
                        tls_client._truncation_retries_exhausted = False
                except Exception:
                    pass
                try:
                    if getattr(self.client, "_truncation_retries_exhausted", False):
                        self.client._truncation_retries_exhausted = False
                except Exception:
                    pass
                if finish_reason in ["length", "max_tokens"] or truncation_exhausted:
                    print(f"    ⚠️ Chunk {chunk_idx}/{total_chunks} response was TRUNCATED!")
                    # Track truncation status
                    is_truncated = True
                else:
                    is_truncated = False
                
                if result:
                    # Remove chunk markers from result
                    result = re.sub(r'\[PART \d+/\d+\]\s*', '', result, flags=re.IGNORECASE)
                    return result, chunk_idx, raw_obj, is_truncated, finish_reason
                else:
                    raise Exception(f"Empty result for chunk {chunk_idx}/{total_chunks}")
            
            # Use ThreadPoolExecutor to process chunks in parallel
            # Use same batch size as chapter-level parallelism
            max_chunk_workers = min(total_chunks, self.config.BATCH_SIZE)

            # Shared abort flag for this chapter's chunks (set when a chunk hits prohibited content)
            chunk_abort_event = threading.Event()

            # Stop callback that also checks the per-chapter abort flag
            def local_stop_cb():
                return (self.check_stop_fn() if hasattr(self, "check_stop_fn") else False) or chunk_abort_event.is_set()

            last_chunk_raw_obj = None
            chapter_truncated = False  # Track if any chunk was truncated

            with ThreadPoolExecutor(max_workers=max_chunk_workers, thread_name_prefix=f"Ch{actual_num}Chunk") as chunk_executor:
                # Submit all chunks
                future_to_chunk = {chunk_executor.submit(process_chunk, chunk_data): chunk_data[1] 
                                  for chunk_data in chunks}
                
                # Collect results as they complete
                completed_chunks = 0
                for future in as_completed(future_to_chunk):
                    if local_stop_cb():
                        print("❌ Translation stopped during chunk processing")
                        chunk_executor.shutdown(wait=False, cancel_futures=True)
                        raise Exception("Translation stopped by user")
                    
                    try:
                        result, chunk_idx, raw_obj, is_truncated, finish_reason = future.result()

                        # Immediate QA fail: stop remaining chunks and mark chapter
                        if finish_reason in ("content_filter", "prohibited_content", "error"):
                            # Signal other chunk workers to abort quickly (chapter-local only)
                            chunk_abort_event.set()
                            fname = FileUtilities.create_chapter_filename(chapter, actual_num)
                            with self.progress_lock:
                                self.update_progress_fn(
                                    idx, actual_num, content_hash, fname,
                                    status="qa_failed",
                                    qa_issues_found=["PROHIBITED_CONTENT"],
                                    chapter_obj=chapter
                                )
                                self.save_progress_fn()
                            chunk_executor.shutdown(wait=False, cancel_futures=True)
                            return False, actual_num, None, None, None
                        if result:
                            # Store result at correct index to maintain order
                            with chunks_lock:
                                translated_chunks[chunk_idx - 1] = result  # chunk_idx is 1-based
                                self.chunks_completed += 1
                                completed_chunks += 1
                                # Store the raw object if it's the last chunk (or the only chunk)
                                if chunk_idx == total_chunks:
                                    last_chunk_raw_obj = raw_obj
                                # Track if any chunk was truncated
                                if is_truncated:
                                    chapter_truncated = True
                            
                            print(f"✅ Chunk {chunk_idx}/{total_chunks} completed ({completed_chunks}/{total_chunks})")
                    except Exception as e:
                        chunk_idx = future_to_chunk[future]
                        print(f"❌ Chunk {chunk_idx}/{total_chunks} failed: {e}")
                        raise
            if chunk_abort:
                print(f"⚠️ Chapter {actual_num}: aborted due to prohibited content; skipping remaining chunks")
                return False, actual_num, None, None, None

            # Verify all chunks completed
            if None in translated_chunks:
                missing = [i+1 for i, chunk in enumerate(translated_chunks) if chunk is None]
                raise Exception(f"Failed to translate chunks: {missing}")
            
            # Combine all chunks
            if total_chunks > 1:
                result = '\n'.join(translated_chunks)
                print(f"🔗 Combined {total_chunks} chunks for Chapter {actual_num}")
            else:
                result = translated_chunks[0] if translated_chunks else None
            
            if not result:
                raise Exception("No translation result produced")

            # Enhanced mode workflow (same as non-batch):
            # 1. Original HTML -> html2text -> Markdown/plain text (during extraction)
            # 2. Markdown sent to translation API (better for translation quality)
            # 3. Translated markdown -> HTML conversion (here)
            if result and chapter.get("enhanced_extraction", False):
                print(f"🔄 Converting translated markdown back to HTML...")
                result = convert_enhanced_text_to_html(result, chapter)
            
            if self.config.REMOVE_AI_ARTIFACTS:
                result = ContentProcessor.clean_ai_artifacts(result, True)
                
            result = ContentProcessor.clean_memory_artifacts(result)
            
            cleaned = re.sub(r"^```(?:html)?\s*\n?", "", result, count=1, flags=re.MULTILINE)
            cleaned = re.sub(r"\n?```\s*$", "", cleaned, count=1, flags=re.MULTILINE)
            cleaned = ContentProcessor.clean_ai_artifacts(cleaned, remove_artifacts=self.config.REMOVE_AI_ARTIFACTS)
            
            # Check for empty or failed response BEFORE writing to disk
            if not cleaned or not str(cleaned).strip():
                print(f"❌ Batch: Translation empty for chapter {actual_num} — skipping file write")
                with self.progress_lock:
                    self.update_progress_fn(idx, actual_num, content_hash, None, status="qa_failed", qa_issues_found=["EMPTY_OUTPUT"])
                    self.save_progress_fn()
                return False, actual_num, None, None, None

            if is_qa_failed_response(cleaned):
                failure_reason = get_failure_reason(cleaned)
                print(f"❌ Batch: Translation failed for chapter {actual_num} - marked as failed, no output file created")
                with self.progress_lock:
                    fname = FileUtilities.create_chapter_filename(chapter, actual_num)
                    self.update_progress_fn(idx, actual_num, content_hash, fname, status="qa_failed", ai_features=ai_features)
                    self.save_progress_fn()
                return False, actual_num, None, None, None

            # NOTE: We no longer append to translation history here in the worker thread.
            # History is now written in the main thread per batch, in a stable order.
            fname = FileUtilities.create_chapter_filename(chapter, actual_num)
            
            if self.is_text_file:
                # For text files, save as plain text
                fname_txt = fname.replace('.html', '.txt') if fname.endswith('.html') else fname
                
                # Extract text from HTML
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(cleaned, 'html.parser')
                text_content = soup.get_text(strip=True)
                
                # Merge image translations back with text translation
                if 'final_body_with_images' in locals() and image_translations:
                    # Parse both versions
                    soup_with_images = BeautifulSoup(final_body_with_images, 'html.parser')
                    soup_with_text = BeautifulSoup(cleaned, 'html.parser')
                    
                    # Get the translated text content (without images)
                    body_content = soup_with_text.body
                    
                    # Add image translations to the translated content
                    for trans_div in soup_with_images.find_all('div', class_='translated-text-only'):
                        body_content.insert(0, trans_div)
                    
                    final_html = str(soup_with_text)
                    cleaned = final_html

                with open(os.path.join(self.out_dir, fname), 'w', encoding='utf-8') as f:
                    f.write(cleaned)
                
                # Update with .txt filename
                with self.progress_lock:
                    self.update_progress_fn(idx, actual_num, content_hash, fname_txt, status="completed", ai_features=ai_features)
                    self.save_progress_fn()
            else:
                # Original code for EPUB files
                with open(os.path.join(self.out_dir, fname), 'w', encoding='utf-8') as f:
                    f.write(cleaned)
            
            print(f"💾 Saved Chapter {actual_num}: {fname} ({len(cleaned)} chars)")
            
            # Initialize ai_features at the beginning to ensure it's always defined
            if ai_features is None:
                ai_features = None
            
            # Extract and save AI features for future duplicate detection
            if (self.config.RETRY_DUPLICATE_BODIES and 
                hasattr(self.config, 'DUPLICATE_DETECTION_MODE') and 
                self.config.DUPLICATE_DETECTION_MODE in ['ai-hunter', 'cascading']):
                try:
                    # Extract features from the translated content
                    cleaned_text = re.sub(r'<[^>]+>', '', cleaned).strip()
                    # Note: self.translator doesn't exist, so we can't extract features here
                    # The features will need to be extracted during regular processing
                    print(f"    ⚠️ AI features extraction not available in batch mode")
                except Exception as e:
                    print(f"    ⚠️ Failed to extract AI features: {e}")
            
            with self.progress_lock:
                # Check for truncation first
                if chapter_truncated:
                    chapter_status = "qa_failed"
                    print(f"⚠️ Batch: Chapter {actual_num} marked as qa_failed: Response was truncated")
                    # Update progress to qa_failed status with TRUNCATED issue
                    self.update_progress_fn(idx, actual_num, content_hash, fname, status=chapter_status, ai_features=ai_features, qa_issues_found=["TRUNCATED"])
                    self.save_progress_fn()
                    # DO NOT increment chapters_completed for qa_failed
                    # Return False to indicate failure (return 5 values to match successful return)
                    return False, actual_num, None, None, None
                else:
                    chapter_status = "completed"
                    # Update progress to completed status
                    self.update_progress_fn(idx, actual_num, content_hash, fname, status=chapter_status, ai_features=ai_features)
                    self.save_progress_fn()
                    # Only increment chapters_completed for successful chapters
                    self.chapters_completed += 1
                    # Note: chunks_completed is already incremented in the loop above
            
            print(f"✅ Chapter {actual_num} completed successfully")
            # Return chapter body and final cleaned translation so the main thread
            # can append to translation history in a stable batch order.
            return True, actual_num, chapter_body, cleaned, last_chunk_raw_obj
            
        except Exception as e:
            print(f"❌ Chapter {actual_num} failed: {e}")
            with self.progress_lock:
                # Use the same output filename so we can track failed chapters properly
                fname = FileUtilities.create_chapter_filename(chapter, actual_num)
                self.update_progress_fn(idx, actual_num, content_hash, fname, status="failed")
                self.save_progress_fn()
            # No history for failed chapters
            return False, actual_num, None, None, None
    
    def process_merged_group(self, merge_group, progress_manager):
        """
        Process a merge group (multiple chapters merged into a single API request).
        
        Args:
            merge_group: List of (idx, chapter) tuples to merge
            progress_manager: ProgressManager instance for updating merged chapter status
            
        Returns:
            List of results, each in format: (success, actual_num, hist_user, hist_assistant, raw_obj)
        """
        import threading
        
        if len(merge_group) == 1:
            # Single chapter, process normally
            result = self.process_single_chapter(merge_group[0])
            return [result]
        
        # Get info for all chapters in the group
        chapters_data = []  # List of (chapter_num, content, idx, chapter_obj, content_hash)
        parent_idx, parent_chapter = merge_group[0]
        parent_actual_num = parent_chapter.get('actual_chapter_num', parent_chapter['num'])
        
        thread_name = threading.current_thread().name
        print(f"\n🔗 [{thread_name}] Processing MERGED group: Chapters {[c.get('actual_chapter_num', c['num']) for _, c in merge_group]}")
        
        # Check ignore settings for filtering
        batch_translate_active = os.getenv('BATCH_TRANSLATE_HEADERS', '0') == '1'
        ignore_title_tag = os.getenv('IGNORE_TITLE', '0') == '1' and batch_translate_active
        ignore_header_tags = os.getenv('IGNORE_HEADER', '0') == '1' and batch_translate_active
        remove_duplicate_h1_p = os.getenv('REMOVE_DUPLICATE_H1_P', '0') == '1'
        
        for idx, chapter in merge_group:
            actual_num = chapter.get('actual_chapter_num', chapter['num'])
            content_hash = chapter.get("content_hash") or ContentProcessor.get_content_hash(chapter["body"])
            
            # Get chapter body and apply ignore filters if needed
            chapter_body = chapter["body"]
            
            if (ignore_title_tag or ignore_header_tags or remove_duplicate_h1_p) and chapter_body:
                from bs4 import BeautifulSoup
                body_soup = BeautifulSoup(chapter_body, 'html.parser')
                
                # Remove title tags if ignored (including those in <head>)
                if ignore_title_tag:
                    for title_tag in body_soup.find_all('title'):
                        title_tag.decompose()
                
                # Remove header tags if ignored
                if ignore_header_tags:
                    for header_tag in body_soup.find_all(['h1', 'h2', 'h3']):
                        header_tag.decompose()
                
                # Remove duplicate H1+P pairs (where P immediately follows H1 with same text)
                if remove_duplicate_h1_p:
                    for h1_tag in body_soup.find_all('h1'):
                        # Skip split marker H1 tags
                        h1_id = h1_tag.get('id', '')
                        if h1_id and h1_id.startswith('split-'):
                            continue
                        h1_text = h1_tag.get_text(strip=True)
                        if 'SPLIT MARKER' in h1_text:
                            continue
                        
                        # Get the next sibling (skipping whitespace/text nodes)
                        next_sibling = h1_tag.find_next_sibling()
                        if next_sibling and next_sibling.name == 'p':
                            # Compare text content (stripped)
                            p_text = next_sibling.get_text(strip=True)
                            if h1_text == p_text:
                                # Remove the duplicate paragraph
                                next_sibling.decompose()
                
                chapter_body = str(body_soup)
            
            chapters_data.append((actual_num, chapter_body, idx, chapter, content_hash))
        
        try:
            # Mark all chapters as in_progress
            for actual_num, _, idx, chapter, content_hash in chapters_data:
                with self.progress_lock:
                    # Determine output filename for tracking (consistent with process_single_chapter)
                    fname = FileUtilities.create_chapter_filename(chapter, actual_num)
                    self.update_progress_fn(idx, actual_num, content_hash, fname, status="in_progress", chapter_obj=chapter)
                    self.save_progress_fn()
            
            # Merge chapter contents
            merge_input = [(cn, content, ch) for cn, content, _, ch, _ in chapters_data]
            merged_content = RequestMerger.merge_chapters(merge_input)
            
            expected_chapters = [cn for cn, _, _, _, _ in chapters_data]
            print(f"   📊 Merged {len(merge_group)} chapters ({len(merged_content):,} chars total)")
            
            # Build system prompt with glossary
            glossary_path = find_glossary_file(self.out_dir)
            chapter_system_prompt = build_system_prompt(
                self.config.SYSTEM_PROMPT, 
                glossary_path, 
                source_text=merged_content
            )
            
            # Build messages
            rolling_summary_msgs = []
            if getattr(self.config, 'USE_ROLLING_SUMMARY', False):
                try:
                    rs_text = self.get_batch_rolling_summary_text()
                except Exception:
                    rs_text = ""
                if isinstance(rs_text, str) and rs_text:
                    rolling_summary_msgs = [{
                        "role": "assistant",
                        "content": (
                            "CONTEXT ONLY - DO NOT INCLUDE IN TRANSLATION:\n"
                            "[MEMORY] Previous context summary:\n\n"
                            + rs_text + "\n\n"
                            "[END MEMORY]\n"
                            "END OF CONTEXT - BEGIN ACTUAL CONTENT TO TRANSLATE:"
                        )
                    }]

            memory_msgs = []
            if (self.config.CONTEXTUAL 
                and self.history_manager is not None 
                and getattr(self.config, 'HIST_LIMIT', 0) > 0):
                try:
                    history = self.history_manager.load_history()
                    hist_limit = getattr(self.config, 'HIST_LIMIT', 0)
                    trimmed = history[-hist_limit * 2:]
                    include_source = os.getenv("INCLUDE_SOURCE_IN_HISTORY", "0") == "1"
                    for h in trimmed:
                        if not isinstance(h, dict):
                            continue
                        role = h.get('role', 'user')
                        raw_obj = h.get('_raw_content_object')
                        content = h.get('content') or ""

                        if role == 'user' and not include_source:
                            continue

                        if (not content) and raw_obj is None:
                            continue

                        msg = {'role': role}
                        if content:
                            msg['content'] = content
                        if raw_obj is not None:
                            msg['_raw_content_object'] = raw_obj
                        memory_msgs.append(msg)
                except Exception as e:
                    print(f"   ⚠️ Failed to load history for merged group: {e}")
            
            msgs = [{"role": "system", "content": chapter_system_prompt}] + rolling_summary_msgs + memory_msgs + [
                {"role": "user", "content": merged_content}
            ]

            # Prepare split-failed retry controls
            try:
                split_retry_limit = int(getattr(self.config, 'SPLIT_FAILED_RETRY_ATTEMPTS', 2))
            except Exception:
                split_retry_limit = 2
            disable_fallback_flag = (os.getenv('DISABLE_MERGE_FALLBACK', '0') == '1') or bool(getattr(self.config, 'DISABLE_MERGE_FALLBACK', False))
            # Use toggle/config for split retries (works in batch and non-batch)
            split_retry_enabled = (os.getenv('RETRY_SPLIT_FAILED', '0') == '1') or bool(getattr(self.config, 'RETRY_SPLIT_FAILED', False))
            split_retry_attempts = 0
            print(f"   [DEBUG] Split retry enabled={split_retry_enabled}, limit={split_retry_limit}, disable_fallback={disable_fallback_flag}")

            # Log combined prompt token count for merged request (treated as Chunk 1/1).
            try:
                # Use the same token counter as regular batch splitting.
                # Instantiate a lightweight ChapterSplitter here for counting only.
                chapter_splitter = ChapterSplitter(model_name=self.config.MODEL)
                
                # Count tokens for system+assistant(user/memory) messages
                total_tokens = 0
                assistant_tokens = 0
                for m in msgs:
                    content = m.get("content", "")
                    tokens = chapter_splitter.count_tokens(content)
                    total_tokens += tokens
                    if m.get("role") == "assistant":
                        assistant_tokens += tokens
                non_assistant_tokens = total_tokens - assistant_tokens

                # Determine a stable file reference based on parent chapter
                parent_file_ref = (
                    parent_chapter.get('original_basename')
                    or parent_chapter.get('filename')
                    or f"Chapter_{parent_actual_num}"
                )

                # Get budget string from MAX_INPUT_TOKENS
                token_env = os.getenv("MAX_INPUT_TOKENS", "1000000").strip()
                _, budget_str = parse_token_limit(token_env)

                if self.config.CONTEXTUAL and assistant_tokens > 0:
                    print(
                        f"💬 Chunk 1/1 combined prompt: "
                        f"{total_tokens:,} tokens (system + user: {non_assistant_tokens:,}, "
                        f"assistant/memory: {assistant_tokens:,}) / {budget_str} [File: {parent_file_ref}]"
                    )
                else:
                    print(
                        f"💬 Chunk 1/1 combined prompt: "
                        f"{total_tokens:,} tokens (system + user) / {budget_str} [File: {parent_file_ref}]"
                    )
            except Exception as e:
                # Never break translation due to logging issues.
                print(f"   ⚠️ Failed to log combined prompt tokens for merged group: {e}")
            
            # Get max output tokens
            env_max_output = os.getenv("MAX_OUTPUT_TOKENS", "")
            if env_max_output.isdigit() and int(env_max_output) > 0:
                mtoks = int(env_max_output)
            else:
                mtoks = self.config.MAX_OUTPUT_TOKENS
            
            # Finite retry loop to avoid infinite re-requests when Split‑the‑Merge keeps failing.
            max_merge_attempts = (max(1, split_retry_limit) + 1) if split_retry_enabled else 1
            split_retry_attempts = 0
            while split_retry_attempts < max_merge_attempts:
                # Call API for merged content
                print(f"   🌐 Sending merged request to API...")
                
                merged_response, finish_reason, raw_obj = send_with_interrupt(
                    msgs,
                    self.client,
                    self.config.TEMP,
                    mtoks,
                    self.check_stop_fn,
                    context='translation'
                )
                # Preserve the finish reason from the merged API call for later status decisions.
                merged_finish_reason = finish_reason
                truncation_exhausted = getattr(self.client, "_truncation_retries_exhausted", False)
                if truncation_exhausted:
                    try:
                        self.client._truncation_retries_exhausted = False
                    except Exception:
                        pass
                
                if self.check_stop_fn():
                    raise Exception("Translation stopped by user")
                
                if not merged_response:
                    raise Exception("Empty response from API for merged request")
                
                # Check for truncation (use preserved finish reason so retries/merges don't lose the flag)
                merged_truncated = merged_finish_reason in ["length", "max_tokens"] or truncation_exhausted
                if merged_truncated:
                    print(f"   ⚠️ Merged response was TRUNCATED!")
                
                print(f"   ✅ Received merged response ({len(merged_response):,} chars)")
                
                # Clean the merged response
                cleaned = merged_response
                if self.config.REMOVE_AI_ARTIFACTS:
                    cleaned = ContentProcessor.clean_ai_artifacts(cleaned, True)
                cleaned = ContentProcessor.clean_memory_artifacts(cleaned)
                cleaned = re.sub(r"^```(?:html)?\s*\n?", "", cleaned, count=1, flags=re.MULTILINE)
                cleaned = re.sub(r"\n?```\s*$", "", cleaned, count=1, flags=re.MULTILINE)
                
                # Get parent chapter info
                parent_actual_num, parent_content, parent_idx, parent_chapter, parent_content_hash = chapters_data[0]
                merged_child_nums = [cn for cn, _, _, _, _ in chapters_data[1:]]
                
                # Check if enhanced extraction was used
                try:
                    enhanced_group = any(bool(ch.get('enhanced_extraction')) for _, _, _, ch, _ in chapters_data)
                except Exception:
                    enhanced_group = False
                
                # Check if Split the Merge is enabled
                split_the_merge = os.getenv('SPLIT_THE_MERGE', '0') == '1'
                
                # If Split the Merge is enabled, SKIP markdown→HTML conversion here
                # We'll do it AFTER splitting so markers are preserved
                if not split_the_merge and enhanced_group and isinstance(cleaned, str):
                    print("   🔄 Converting merged enhanced text back to HTML...")
                    try:
                        cleaned = convert_enhanced_text_to_html(cleaned, parent_chapter)
                    except Exception as conv_err:
                        print(f"   ⚠️ Enhanced HTML conversion failed: {conv_err} — saving raw content")
                    
                    # Emergency Image Restoration (if enabled)
                    if self.config.EMERGENCY_IMAGE_RESTORE:
                        cleaned = ContentProcessor.emergency_restore_images(cleaned, merged_content)
                    
                    # Optionally restore paragraphs if the output lacks structure
                    if getattr(self.config, 'EMERGENCY_RESTORE', False):
                        try:
                            if cleaned and cleaned.count('<p>') < 3 and len(cleaned) > 300:
                                cleaned = ContentProcessor.emergency_restore_paragraphs(cleaned)
                        except Exception:
                            pass
            
                # Check for truncation / QA failures first
                results = []
                if is_qa_failed_response(cleaned):
                    # Only save file for debugging if it contains meaningful content beyond error markers
                    cleaned_stripped = cleaned.strip()
                    is_only_error_marker = cleaned_stripped in [
                        "[TRANSLATION FAILED]",
                        "[Content Blocked]",
                        "[IMAGE TRANSLATION FAILED]",
                        "[EXTRACTION FAILED]",
                        "[RATE LIMITED]",
                        "[]"
                    ] or cleaned_stripped.startswith("[TRANSLATION FAILED - ORIGINAL TEXT PRESERVED]") or cleaned_stripped.startswith("[CONTENT BLOCKED - ORIGINAL TEXT PRESERVED]")
                    
                    if not is_only_error_marker and cleaned_stripped:
                        parent_fname = FileUtilities.create_chapter_filename(parent_chapter, parent_actual_num)
                        try:
                            cleaned_to_save = cleaned
                            if split_the_merge:
                                cleaned_to_save = re.sub(
                                    r'<h1[^>]*id=\"split-\\d+\"[^>]*>.*?</h1>\\s*',
                                    '',
                                    cleaned_to_save,
                                    flags=re.IGNORECASE | re.DOTALL,
                                )
                            with open(os.path.join(self.out_dir, parent_fname), 'w', encoding='utf-8') as f:
                                f.write(cleaned_to_save)
                        except Exception:
                            pass
                    # Use each chapter's own expected filename so we overwrite the existing in_progress entry
                    for actual_num, _, idx, chapter, content_hash in chapters_data:
                        chapter_fname = FileUtilities.create_chapter_filename(chapter, actual_num)
                        with self.progress_lock:
                            self.update_progress_fn(
                                idx,
                                actual_num,
                                content_hash,
                                chapter_fname,
                                status="qa_failed",
                                chapter_obj=chapter,
                            )
                            self.save_progress_fn()
                        results.append((False, actual_num, None, None, None))
                    return results

                # Now handle split-the-merge
                disable_fallback = disable_fallback_flag
                split_sections = None
                
                if split_the_merge and len(chapters_data) > 1:
                    # Try to split by invisible markers
                    split_sections = RequestMerger.split_by_markers(cleaned, len(chapters_data))
                
                # If split failed, optionally retry; if retries exhausted, mark qa_failed when fallback disabled
                if split_the_merge and (not split_sections or len(split_sections) != len(chapters_data)):
                    if split_retry_enabled and split_retry_attempts + 1 < max_merge_attempts:
                        split_retry_attempts += 1
                        print(f"   🔄 Split failed — retrying merged request (attempt {split_retry_attempts}/{max_merge_attempts - 1})")
                        continue

                    if disable_fallback:
                        print(f"   ⚠️ Split failed and fallback disabled - marking merged group as qa_failed")
                    
                    # Only save file for debugging if it contains meaningful content beyond error markers
                    cleaned_stripped = cleaned.strip()
                    is_only_error_marker = cleaned_stripped in [
                        "[TRANSLATION FAILED]",
                        "[Content Blocked]",
                        "[IMAGE TRANSLATION FAILED]",
                        "[EXTRACTION FAILED]",
                        "[RATE LIMITED]",
                        "[]"
                    ] or cleaned_stripped.startswith("[TRANSLATION FAILED - ORIGINAL TEXT PRESERVED]") or cleaned_stripped.startswith("[CONTENT BLOCKED - ORIGINAL TEXT PRESERVED]")
                    
                    if not is_only_error_marker and cleaned_stripped:
                        # Save for debugging - contains actual translation attempt that failed split
                        parent_fname = FileUtilities.create_chapter_filename(parent_chapter, parent_actual_num)
                        try:
                            cleaned_to_save = cleaned
                            if split_the_merge:
                                cleaned_to_save = re.sub(
                                    r'<h1[^>]*id=\"split-\\d+\"[^>]*>.*?</h1>\\s*',
                                    '',
                                    cleaned_to_save,
                                    flags=re.IGNORECASE | re.DOTALL,
                                )
                            with open(os.path.join(self.out_dir, parent_fname), 'w', encoding='utf-8') as f:
                                f.write(cleaned_to_save)
                        except Exception:
                            pass

                    # IMPORTANT:
                    # Use each chapter's own expected filename so we overwrite the
                    # existing in_progress entry instead of creating composite keys.
                    for actual_num, _, idx, chapter, content_hash in chapters_data:
                        chapter_fname = FileUtilities.create_chapter_filename(chapter, actual_num)
                        with self.progress_lock:
                            self.update_progress_fn(
                                idx,
                                actual_num,
                                content_hash,
                                chapter_fname,
                                status="qa_failed",
                                qa_issues_found=["SPLIT_FAILED"],
                                chapter_obj=chapter,
                            )
                            self.save_progress_fn()
                        results.append((False, actual_num, None, None, None))
                    return results

                # If split failed and fallback is allowed, optionally retry merged translation
                if split_the_merge and (not split_sections or len(split_sections) != len(chapters_data)) and split_retry_enabled:
                    if split_retry_attempts < split_retry_limit:
                        split_retry_attempts += 1
                        attempt_no = split_retry_attempts
                        print(f"   🔄 Split failed retry {attempt_no}/{split_retry_limit} — requesting new merged translation")
                        time.sleep(1)
                        # Try a fresh merged request on next loop iteration
                        continue
                    else:
                        print(f"   ⚠️ Split failed after {split_retry_limit} retries, falling back to merged output")
                
                if split_sections and len(split_sections) == len(chapters_data):
                    # Split successful - save each section as individual file
                    print(f"   ✂️ Splitting merged content into {len(split_sections)} individual files")
                    
                    saved_files = []
                    for i, (actual_num, content, idx, chapter, content_hash) in enumerate(chapters_data):
                        section_content = split_sections[i]
                        
                        # NOW convert markdown→HTML for each section if enhanced extraction was used
                        if enhanced_group and isinstance(section_content, str):
                            try:
                                section_content = convert_enhanced_text_to_html(section_content, chapter)
                            except Exception as conv_err:
                                print(f"   ⚠️ Enhanced HTML conversion failed for chapter {actual_num}: {conv_err}")
                        
                        # Generate filename for this chapter using content.opf naming
                        fname = FileUtilities.create_chapter_filename(chapter, actual_num)
                        
                        # Handle text file mode
                        if getattr(self, 'is_text_file', False):
                            fname = fname.replace('.html', '.txt')
                            from bs4 import BeautifulSoup
                            soup = BeautifulSoup(section_content, 'html.parser')
                            section_content = soup.get_text(strip=True)
                        
                        # Save the section
                        with open(os.path.join(self.out_dir, fname), 'w', encoding='utf-8') as f:
                            f.write(section_content)
                        
                        saved_files.append((actual_num, fname, idx, chapter, content_hash))
                        print(f"      💾 Saved Chapter {actual_num}: {fname} ({len(section_content)} chars)")
                    
                    # Mark all chapters as completed or qa_failed (for truncated)
                    with self.progress_lock:
                        for actual_num, fname, idx, chapter, content_hash in saved_files:
                            chapter_status = "qa_failed" if merged_truncated else "completed"
                            qa_issues = ["TRUNCATED"] if merged_truncated else None
                            self.update_progress_fn(
                                idx, actual_num, content_hash, fname,
                                status=chapter_status, qa_issues_found=qa_issues, chapter_obj=chapter
                            )
                            self.chapters_completed += 1
                        
                        # Save once after all updates
                        self.save_progress_fn()
                    
                    # Build results - if truncated, treat as failure for all chapters
                    if merged_truncated:
                        for actual_num, _, idx, chapter, content_hash in chapters_data:
                            results.append((False, actual_num, None, None, None))
                    else:
                        results.append((True, chapters_data[0][0], merged_content, merged_response, raw_obj))
                        for actual_num, _, idx, chapter, content_hash in chapters_data[1:]:
                            results.append((True, actual_num, None, None, None))
                    
                    print(f"   ✅ Split the Merge complete: {len(saved_files)} files created")
                    return results
                
                # Normal merged behavior (split not enabled or header count mismatch)
                # Save entire merged response to parent chapter's file
                fname = FileUtilities.create_chapter_filename(parent_chapter, parent_actual_num)

                # If Split-the-Merge was enabled but we couldn't split reliably, remove injected markers
                cleaned_to_save = cleaned
                if split_the_merge and len(chapters_data) > 1:
                    cleaned_to_save = re.sub(
                        r'<h1[^>]*id=\"split-\\d+\"[^>]*>.*?</h1>\\s*',
                        '',
                        cleaned_to_save,
                        flags=re.IGNORECASE | re.DOTALL,
                    )
                
                # If translating a plain text source, mirror non-merged behavior and write .txt
                if getattr(self, 'is_text_file', False):
                    parent_fname = fname.replace('.html', '.txt')
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(cleaned_to_save, 'html.parser')
                    text_content = soup.get_text(strip=True)
                    with open(os.path.join(self.out_dir, parent_fname), 'w', encoding='utf-8') as f:
                        f.write(text_content)
                    saved_name = parent_fname
                else:
                    with open(os.path.join(self.out_dir, fname), 'w', encoding='utf-8') as f:
                        f.write(cleaned_to_save)
                    saved_name = fname
                
                print(f"   💾 Saved merged content to Chapter {parent_actual_num}: {saved_name} ({len(cleaned_to_save)} chars)")
                
                with self.progress_lock:
                    if merged_truncated:
                        # Truncated merged response: mark ALL chapters as qa_failed
                        # Check if we can retry this truncation failure as a general merge failure
                        if split_retry_enabled and split_retry_attempts + 1 < max_merge_attempts:
                            split_retry_attempts += 1
                            print(f"   🔄 Truncated merged response — retrying request (attempt {split_retry_attempts}/{max_merge_attempts - 1})")
                            time.sleep(2)
                            continue
                        
                        # Check if we can retry this truncation failure as a general merge failure
                        if split_retry_enabled and split_retry_attempts + 1 < max_merge_attempts:
                            split_retry_attempts += 1
                            print(f"   🔄 Truncated merged response — retrying request (attempt {split_retry_attempts}/{max_merge_attempts - 1})")
                            time.sleep(2)
                            continue
                        
                        qa_issues = ["TRUNCATED"]
                        self.update_progress_fn(
                            parent_idx, parent_actual_num, parent_content_hash, saved_name,
                            status="qa_failed", qa_issues_found=qa_issues, chapter_obj=parent_chapter
                        )
                        for actual_num, _, idx, chapter, content_hash in chapters_data[1:]:
                            self.update_progress_fn(
                                idx, actual_num, content_hash, None,
                                status="qa_failed", qa_issues_found=qa_issues, chapter_obj=chapter
                            )
                        self.chapters_completed += len(chapters_data)
                    else:
                        # Normal success path: parent completed, children merged
                        self.update_progress_fn(
                            parent_idx, parent_actual_num, parent_content_hash, saved_name,
                            status="completed",
                            merged_chapters=merged_child_nums,
                            chapter_obj=parent_chapter
                        )
                        self.chapters_completed += 1
                        
                        # Then mark all child chapters as merged (only after parent is completed)
                        for actual_num, _, idx, chapter, content_hash in chapters_data[1:]:
                            progress_manager.mark_as_merged(idx, actual_num, content_hash, parent_actual_num, chapter, parent_output_file=saved_name)
                            self.chapters_completed += 1
                    
                    # Save once after all updates
                    self.save_progress_fn()
                
                # Build results based on truncation status
                if merged_truncated:
                    for actual_num, _, idx, chapter, content_hash in chapters_data:
                        results.append((False, actual_num, None, None, None))
                else:
                    results.append((True, parent_actual_num, merged_content, merged_response, raw_obj))
                    for actual_num, _, idx, chapter, content_hash in chapters_data[1:]:
                        results.append((True, actual_num, None, None, None))
                
                return results

            # Should never hit this line; guard to prevent infinite loop
            raise RuntimeError("Merged translation exited retry loop without returning a result")
            
        except Exception as e:
            print(f"❌ Merged group failed: {e}")
            # Mark all chapters as failed
            results = []
            for actual_num, _, idx, chapter, content_hash in chapters_data:
                with self.progress_lock:
                    fname = FileUtilities.create_chapter_filename(chapter, actual_num)
                    self.update_progress_fn(idx, actual_num, content_hash, fname, status="failed", chapter_obj=chapter)
                    self.save_progress_fn()
                results.append((False, actual_num, None, None, None))
            return results


# =====================================================
# UNIFIED UTILITIES
# =====================================================
def sanitize_resource_filename(filename):
    """Sanitize resource filenames for filesystem compatibility"""
    filename = unicodedata.normalize('NFC', filename)
    
    replacements = {
        '/': '_', '\\': '_', ':': '_', '*': '_',
        '?': '_', '"': '_', '<': '_', '>': '_',
        '|': '_', '\0': '', '\n': '_', '\r': '_'
    }
    
    for old, new in replacements.items():
        filename = filename.replace(old, new)
    
    filename = ''.join(char for char in filename if ord(char) >= 32)
    
    name, ext = os.path.splitext(filename)
    
    if not name:
        name = 'resource'
    
    return name + ext

def should_retain_source_extension():
    """Read GUI toggle for retaining original extension and no 'response_' prefix.
    This is stored in config or env by the GUI; we read env as bridge.
    """
    return os.getenv('RETAIN_SOURCE_EXTENSION', os.getenv('retain_source_extension', '0')) in ('1', 'true', 'True')

def make_safe_filename(title, actual_num):
    """Create a safe filename that works across different filesystems"""
    if not title:
        return f"chapter_{actual_num:03d}"
    
    title = unicodedata.normalize('NFC', str(title))
    
    dangerous_chars = {
        '/': '_', '\\': '_', ':': '_', '*': '_', '?': '_',
        '"': '_', '<': '_', '>': '_', '|': '_', '\0': '',
        '\n': ' ', '\r': ' ', '\t': ' '
    }
    
    for old, new in dangerous_chars.items():
        title = title.replace(old, new)
    
    title = ''.join(char for char in title if ord(char) >= 32)
    title = re.sub(r'\s+', '_', title)
    title = title.strip('_.• \t')
    
    if not title or title == '_' * len(title):
        title = f"chapter_{actual_num:03d}"
    
    return title

def get_content_hash(html_content):
    """Create a stable hash of content"""
    return ContentProcessor.get_content_hash(html_content)

def clean_ai_artifacts(text, remove_artifacts=True):
    """Remove AI response artifacts from text"""
    return ContentProcessor.clean_ai_artifacts(text, remove_artifacts)

def find_glossary_file(output_dir):
    """Return path to glossary file preferring CSV/MD/TXT over JSON, or None if not found"""
    candidates = [
        os.path.join(output_dir, "glossary.csv"),
        os.path.join(output_dir, "glossary.md"),
        os.path.join(output_dir, "glossary.txt"),
        os.path.join(output_dir, "glossary.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def clean_memory_artifacts(text):
    """Remove any memory/summary artifacts"""
    return ContentProcessor.clean_memory_artifacts(text)

def emergency_restore_paragraphs(text, original_html=None, verbose=True):
    """Emergency restoration when AI returns wall of text"""
    return ContentProcessor.emergency_restore_paragraphs(text, original_html, verbose)

def is_meaningful_text_content(html_content):
    """Check if chapter has meaningful text beyond just structure"""
    return ContentProcessor.is_meaningful_text_content(html_content)

# =====================================================
# GLOBAL SETTINGS AND FLAGS
# =====================================================
logging.basicConfig(level=logging.DEBUG)

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
except AttributeError:
    if sys.stdout is None:
        devnull = open(os.devnull, "wb")
        sys.stdout = io.TextIOWrapper(devnull, encoding='utf-8', errors='ignore')
    elif hasattr(sys.stdout, 'buffer'):
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
        except:
            pass

_stop_requested = False

def set_stop_flag(value):
    """Set the global stop flag"""
    global _stop_requested
    _stop_requested = value

def is_stop_requested():
    """Check if stop was requested"""
    global _stop_requested
    return _stop_requested

def set_output_redirect(log_callback=None):
    """Redirect print statements to a callback function for GUI integration"""
    if log_callback:
        import threading
        
        class CallbackWriter:
            def __init__(self, callback):
                self.callback = callback
                self.main_thread = threading.main_thread()
                
            def write(self, text):
                if text.strip():
                    # The callback (append_log) is already thread-safe - it handles QTimer internally
                    # So we can call it directly from any thread
                    self.callback(text.strip())
                    
            def flush(self):
                pass
                
        sys.stdout = CallbackWriter(log_callback)

# =====================================================
# EPUB AND FILE PROCESSING
# =====================================================

def extract_chapter_number_from_filename(filename, opf_spine_position=None, opf_spine_data=None):
    """Extract chapter number from filename.

    Preference order:
    1) Rightmost digits in the filename (0 if all zeros)
    2) Special keywords with no digits -> 0
    3) Legacy fallback patterns
    """
    # Normalize: strip directory, extension, and response_ prefix for parsing
    basename = os.path.basename(filename)
    base_no_ext = os.path.splitext(basename)[0]
    if base_no_ext.lower().startswith('response_'):
        base_no_ext = base_no_ext[len('response_'):]
    base_no_ext_lower = base_no_ext.lower()

    # Priority 1: digits in filename (use rightmost match to mirror GUI column)
    numbers = re.findall(r'[0-9]+', base_no_ext)
    if numbers:
        last_num = int(numbers[-1])
        if last_num == 0:
            return 0, 'filename_zero'
        return last_num, 'filename_digits'
    # Priority 2: special keyword files with no digits -> chapter 0
    # Priority 3: special keyword files with no digits -> chapter 0
    special_keywords = ['title', 'toc', 'cover', 'index', 'copyright', 'preface', 'nav', 'message', 'info', 'notice', 'colophon', 'dedication', 'epigraph', 'foreword', 'acknowledgment', 'author', 'appendix', 'glossary', 'bibliography']
    if any(name in base_no_ext_lower for name in special_keywords):
        return 0, 'special_file'
    # Priority 3: legacy fallback patterns
    name_without_ext = base_no_ext
    fallback_patterns = [
        (r'^response_(\d+)[_\.]', 'response_prefix'),
        (r'[Cc]hapter[_\s]*(\d+)', 'chapter_word'),
        (r'[Cc]h[_\s]*(\d+)', 'ch_abbreviation'),
        (r'No(\d+)', 'no_prefix'),
        (r'第(\d+)[章话回]', 'chinese_chapter'),
        (r'-h-(\d+)', 'h_suffix'),              # For your -h-16 pattern
        (r'_(\d+)', 'underscore_suffix'),
        (r'-(\d+)', 'dash_suffix'),
        (r'(\d+)', 'trailing_number'),
    ]
    
    for pattern, method in fallback_patterns:
        match = re.search(pattern, name_without_ext, re.IGNORECASE)
        if match:
            return int(match.group(1)), method
    return None, None

def process_chapter_images(chapter_html: str, actual_num: int, image_translator: ImageTranslator, 
                         check_stop_fn=None) -> Tuple[str, Dict[str, str]]:
    """Process and translate images in a chapter"""
    from bs4 import BeautifulSoup
    images = image_translator.extract_images_from_chapter(chapter_html)

    if not images:
        return chapter_html, {}
        
    print(f"🖼️ Found {len(images)} images in chapter {actual_num}")
    
    soup = BeautifulSoup(chapter_html, 'html.parser')
    
    image_translations = {}
    translated_count = 0
    
    max_images_per_chapter = int(os.getenv('MAX_IMAGES_PER_CHAPTER', '10'))
    if len(images) > max_images_per_chapter:
        print(f"   ⚠️ Chapter has {len(images)} images - processing first {max_images_per_chapter} only")
        images = images[:max_images_per_chapter]
    
    for idx, img_info in enumerate(images, 1):
        if check_stop_fn and check_stop_fn():
            print("❌ Image translation stopped by user")
            break
            
        img_src = img_info['src']
        original_img_src = img_src  # keep for DOM matching
        img_path = None

        # Handle inline data URI images (e.g., PDF image render mode)
        if img_src.startswith('data:image'):
            try:
                import base64, uuid, mimetypes
                header, b64data = img_src.split(',', 1)
                mime = 'image/png'
                if ':' in header and ';' in header:
                    mime = header.split(';')[0].split(':')[1] or mime
                ext = mimetypes.guess_extension(mime) or '.png'
                os.makedirs(image_translator.images_dir, exist_ok=True)
                temp_name = f"datauri_{actual_num}_{idx}_{uuid.uuid4().hex}{ext}"
                img_path = os.path.join(image_translator.images_dir, temp_name)
                with open(img_path, 'wb') as f:
                    f.write(base64.b64decode(b64data))
                # Keep img_src pointing to original so DOM match works; translator uses img_path
            except Exception as e:
                print(f"   ❌ Failed to decode data URI image: {e}")
                continue

        if img_path is None and img_src.startswith('../'):
            img_path = os.path.join(image_translator.output_dir, img_src[3:])
        elif img_path is None and img_src.startswith('./'):
            img_path = os.path.join(image_translator.output_dir, img_src[2:])
        elif img_path is None and img_src.startswith('/'):
            img_path = os.path.join(image_translator.output_dir, img_src[1:])
        elif img_path is None:
            possible_paths = [
                os.path.join(image_translator.images_dir, os.path.basename(img_src)),
                os.path.join(image_translator.output_dir, img_src),
                os.path.join(image_translator.output_dir, 'images', os.path.basename(img_src)),
                os.path.join(image_translator.output_dir, os.path.basename(img_src)),
                os.path.join(image_translator.output_dir, os.path.dirname(img_src), os.path.basename(img_src))
            ]
            
            img_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    img_path = path
                    print(f"   ✅ Found image at: {path}")
                    break
            
            if not img_path:
                print(f"   ❌ Image not found in any location for: {img_src}")
                print(f"   Tried: {possible_paths}")
                continue
        
        img_path = os.path.normpath(img_path)
        
        if not os.path.exists(img_path):
            print(f"   ⚠️ Image not found: {img_path}")
            print(f"   📁 Images directory: {image_translator.images_dir}")
            print(f"   📁 Output directory: {image_translator.output_dir}")
            print(f"   📁 Working directory: {os.getcwd()}")
            
            if os.path.exists(image_translator.images_dir):
                files = os.listdir(image_translator.images_dir)
                print(f"   📁 Files in images dir: {files[:5]}...")
            continue
        
        print(f"   🔍 Processing image {idx}/{len(images)}: {os.path.basename(img_path)}")
        
        context = ""
        if img_info.get('alt'):
            context += f", Alt text: {img_info['alt']}"
            
        if translated_count > 0:
            delay = float(os.getenv('IMAGE_API_DELAY', '1.0'))
            time.sleep(delay)
            
        translation_result = image_translator.translate_image(img_path, context, check_stop_fn)
        
        print(f"\n🔍 DEBUG: Image {idx}/{len(images)}")
        print(f"   Translation result: {'Success' if translation_result and '[Image Translation Error:' not in translation_result else 'Failed'}")
        if translation_result and "[Image Translation Error:" in translation_result:
            print(f"   Error message: {translation_result}")
        
        if translation_result:
            img_tag = None
            for img in soup.find_all('img'):
                if img.get('src') == original_img_src:
                    img_tag = img
                    break
            
            if img_tag:
                hide_label = os.getenv("HIDE_IMAGE_TRANSLATION_LABEL", "0") == "1"
                
                print(f"   🔍 DEBUG: Integration Phase")
                print(f"   🏷️ Hide label mode: {hide_label}")
                src_display = img_tag.get('src', '')
                if src_display.startswith('data:image'):
                    src_display = src_display[:80] + '...'
                print(f"   📍 Found img tag: {src_display}")
                
                # Store the translation result in the dictionary FIRST
                image_translations[img_path] = translation_result
                
                # Parse the translation result to integrate into the chapter HTML
                if '<div class="image-translation">' in translation_result:
                    trans_soup = BeautifulSoup(translation_result, 'html.parser')
                    
                    # Try to get the full container first
                    full_container = trans_soup.find('div', class_=['translated-text-only', 'image-with-translation'])
                    
                    if full_container:
                        # Clone the container to avoid issues
                        new_container = BeautifulSoup(str(full_container), 'html.parser').find('div')
                        img_tag.replace_with(new_container)
                        print(f"   ✅ Replaced image with full translation container")
                    else:
                        # Fallback: manually build the structure
                        trans_div = trans_soup.find('div', class_='image-translation')
                        if trans_div:
                            container = soup.new_tag('div', **{'class': 'translated-text-only' if hide_label else 'image-with-translation'})
                            img_tag.replace_with(container)
                            
                            if not hide_label:
                                new_img = soup.new_tag('img', src=img_src)
                                if img_info.get('alt'):
                                    new_img['alt'] = img_info.get('alt')
                                container.append(new_img)
                            
                            # Clone the translation div content
                            new_trans_div = soup.new_tag('div', **{'class': 'image-translation'})
                            # Copy all children from trans_div to new_trans_div
                            for child in trans_div.children:
                                if hasattr(child, 'name'):
                                    new_trans_div.append(BeautifulSoup(str(child), 'html.parser'))
                                else:
                                    new_trans_div.append(str(child))
                            
                            container.append(new_trans_div)
                            print(f"   ✅ Built container with translation div")
                        else:
                            print(f"   ⚠️ No translation div found in result")
                            continue
                else:
                    # Plain text translation - build structure manually
                    container = soup.new_tag('div', **{'class': 'translated-text-only' if hide_label else 'image-with-translation'})
                    img_tag.replace_with(container)
                    
                    if not hide_label:
                        new_img = soup.new_tag('img', src=img_src)
                        if img_info.get('alt'):
                            new_img['alt'] = img_info.get('alt')
                        container.append(new_img)
                    
                    # Create translation div with content
                    translation_div = soup.new_tag('div', **{'class': 'image-translation'})
                    if not hide_label:
                        label_p = soup.new_tag('p')
                        label_em = soup.new_tag('em')
                        #label_em.string = "[Image text translation:]"
                        label_p.append(label_em)
                        translation_div.append(label_p)
                    
                    trans_p = soup.new_tag('p')
                    trans_p.string = translation_result
                    translation_div.append(trans_p)
                    container.append(translation_div)
                    print(f"   ✅ Created plain text translation structure")
                
                translated_count += 1
                
                # Save to translated_images folder
                trans_filename = f"ch{actual_num:03d}_img{idx:02d}_translation.html"
                trans_filepath = os.path.join(image_translator.translated_images_dir, trans_filename)
                
                # Extract just the translation content for saving
                save_soup = BeautifulSoup(translation_result, 'html.parser')
                save_div = save_soup.find('div', class_='image-translation')
                if not save_div:
                    # Create a simple div for plain text
                    save_div = f'<div class="image-translation"><p>{translation_result}</p></div>'
                
                with open(trans_filepath, 'w', encoding='utf-8') as f:
                    f.write(f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8"/>
    <title>Chapter {actual_num} - Image {idx} Translation</title>
</head>
<body>
    <h2>Chapter {actual_num} - Image {idx}</h2>
    <p>Original: {os.path.basename(img_path)}</p>
    <hr/>
    {save_div}
</body>
</html>""")
                
                print(f"   ✅ Saved translation to: {trans_filename}")
            else:
                print(f"   ⚠️ Could not find image tag in HTML for: {img_src}")
    
    if translated_count > 0:
        print(f"   🖼️ Successfully translated {translated_count} images")
        
        # Debug output
        final_html = str(soup)
        trans_count = final_html.count('<div class="image-translation">')
        print(f"   📊 Final HTML has {trans_count} translation divs")
        print(f"   📊 image_translations dict has {len(image_translations)} entries")
        
        prog = image_translator.load_progress()
        if "image_chunks" in prog:
            completed_images = []
            for img_key, img_data in prog["image_chunks"].items():
                if len(img_data["completed"]) == img_data["total"]:
                    completed_images.append(img_key)
            
            for img_key in completed_images:
                del prog["image_chunks"][img_key]
                
            if completed_images:
                image_translator.save_progress(prog)
                print(f"   🧹 Cleaned up progress for {len(completed_images)} completed images")
        
        image_translator.save_translation_log(actual_num, image_translations)
        
        return str(soup), image_translations
    else:
        print(f"   ℹ️ No images were successfully translated")
        
    return chapter_html, {}

def detect_novel_numbering(chapters):
    """Detect if the novel uses 0-based or 1-based chapter numbering with improved accuracy"""
    print("[DEBUG] Detecting novel numbering system...")
    
    if not chapters:
        return False
    
    if isinstance(chapters[0], str):
        print("[DEBUG] Text file detected, skipping numbering detection")
        return False
    
    patterns = PatternManager.FILENAME_EXTRACT_PATTERNS
    
    # Special check for prefix_suffix pattern like "0000_1.xhtml"
    prefix_suffix_pattern = r'^(\d+)_(\d+)[_\.]'
    
    # Track chapter numbers from different sources
    filename_numbers = []
    content_numbers = []
    has_prefix_suffix = False
    prefix_suffix_numbers = []
    
    for idx, chapter in enumerate(chapters):
        extracted_num = None
        
        # Check filename patterns
        if 'original_basename' in chapter and chapter['original_basename']:
            filename = chapter['original_basename']
        elif 'filename' in chapter:
            filename = os.path.basename(chapter['filename'])
        else:
            continue
            
        # First check for prefix_suffix pattern
        prefix_match = re.search(prefix_suffix_pattern, filename, re.IGNORECASE)
        if prefix_match:
            has_prefix_suffix = True
            # Use the SECOND number (after underscore)
            suffix_num = int(prefix_match.group(2))
            prefix_suffix_numbers.append(suffix_num)
            extracted_num = suffix_num
            print(f"[DEBUG] Prefix_suffix pattern matched: {filename} -> Chapter {suffix_num}")
        else:
            # Try other patterns
            for pattern in patterns:
                match = re.search(pattern, filename)
                if match:
                    extracted_num = int(match.group(1))
                    #print(f"[DEBUG] Pattern '{pattern}' matched: {filename} -> Chapter {extracted_num}")
                    break
        
        if extracted_num is not None:
            filename_numbers.append(extracted_num)
        
        # Also check chapter content for chapter declarations
        if 'body' in chapter:
            # Look for "Chapter N" in the first 1000 characters
            content_preview = chapter['body'][:1000]
            content_match = re.search(r'Chapter\s+(\d+)', content_preview, re.IGNORECASE)
            if content_match:
                content_num = int(content_match.group(1))
                content_numbers.append(content_num)
                print(f"[DEBUG] Found 'Chapter {content_num}' in content")
    
    # Decision logic with improved heuristics
    
    # 1. If using prefix_suffix pattern, trust those numbers exclusively
    if has_prefix_suffix and prefix_suffix_numbers:
        min_suffix = min(prefix_suffix_numbers)
        if min_suffix >= 1:
            print(f"[DEBUG] ✅ 1-based novel detected (prefix_suffix pattern starts at {min_suffix})")
            return False
        else:
            print(f"[DEBUG] ✅ 0-based novel detected (prefix_suffix pattern starts at {min_suffix})")
            return True
    
    # 2. If we have content numbers, prefer those over filename numbers
    if content_numbers:
        min_content = min(content_numbers)
        # Check if we have a good sequence starting from 0 or 1
        if 0 in content_numbers and 1 in content_numbers:
            print(f"[DEBUG] ✅ 0-based novel detected (found both Chapter 0 and Chapter 1 in content)")
            return True
        elif min_content == 1:
            print(f"[DEBUG] ✅ 1-based novel detected (content chapters start at 1)")
            return False
    
    # 3. Fall back to filename numbers
    if filename_numbers:
        min_filename = min(filename_numbers)
        max_filename = max(filename_numbers)
        
        # Check for a proper sequence
        # If we have 0,1,2,3... it's likely 0-based
        # If we have 1,2,3,4... it's likely 1-based
        
        # Count how many chapters we have in sequence starting from 0
        zero_sequence_count = 0
        for i in range(len(chapters)):
            if i in filename_numbers:
                zero_sequence_count += 1
            else:
                break
        
        # Count how many chapters we have in sequence starting from 1
        one_sequence_count = 0
        for i in range(1, len(chapters) + 1):
            if i in filename_numbers:
                one_sequence_count += 1
            else:
                break
        
        print(f"[DEBUG] Zero-based sequence length: {zero_sequence_count}")
        print(f"[DEBUG] One-based sequence length: {one_sequence_count}")
        
        # If we have a better sequence starting from 1, it's 1-based
        if one_sequence_count > zero_sequence_count and min_filename >= 1:
            print(f"[DEBUG] ✅ 1-based novel detected (better sequence match starting from 1)")
            return False
        
        # If we have any 0 in filenames and it's part of a sequence
        if 0 in filename_numbers and zero_sequence_count >= 3:
            print(f"[DEBUG] ✅ 0-based novel detected (found 0 in sequence)")
            return True
    
    # 4. Default to 1-based if uncertain
    print(f"[DEBUG] ✅ Defaulting to 1-based novel (insufficient evidence for 0-based)")
    return False
    
def validate_chapter_continuity(chapters):
    """Validate chapter continuity and warn about issues"""
    if not chapters:
        print("No chapters to translate")
        return
    
    issues = []
    
    # Get all chapter numbers
    chapter_nums = [c['num'] for c in chapters]
    actual_nums = [c.get('actual_chapter_num', c['num']) for c in chapters]
    
    # Check for duplicates
    duplicates = [num for num in chapter_nums if chapter_nums.count(num) > 1]
    if duplicates:
        issues.append(f"Duplicate chapter numbers found: {set(duplicates)}")
    
    # Check for gaps in sequence
    min_num = min(chapter_nums)
    max_num = max(chapter_nums)
    expected = set(range(min_num, max_num + 1))
    actual = set(chapter_nums)
    missing = expected - actual
    
    if missing:
        issues.append(f"Missing chapter numbers: {sorted(missing)}")
        # Show gaps more clearly
        gaps = []
        sorted_missing = sorted(missing)
        if sorted_missing:
            start = sorted_missing[0]
            end = sorted_missing[0]
            for num in sorted_missing[1:]:
                if num == end + 1:
                    end = num
                else:
                    gaps.append(f"{start}-{end}" if start != end else str(start))
                    start = end = num
            gaps.append(f"{start}-{end}" if start != end else str(start))
            issues.append(f"Gap ranges: {', '.join(gaps)}")
    
    # Check for duplicate titles
    title_map = {}
    for c in chapters:
        title_lower = c['title'].lower().strip()
        if title_lower in title_map:
            title_map[title_lower].append(c['num'])
        else:
            title_map[title_lower] = [c['num']]
    
    for title, nums in title_map.items():
        if len(nums) > 1:
            issues.append(f"Duplicate title '{title}' in chapters: {nums}")
    
    # Print summary
    print("\n" + "="*60)
    print("📚 CHAPTER VALIDATION SUMMARY")
    print("="*60)
    print(f"Total chapters: {len(chapters)}")
    print(f"Chapter range: {min_num} to {max_num}")
    print(f"Expected count: {max_num - min_num + 1}")
    print(f"Actual count: {len(chapters)}")
    
    if len(chapters) != (max_num - min_num + 1):
        print(f"⚠️  Chapter count mismatch - missing {(max_num - min_num + 1) - len(chapters)} chapters")
    
    if issues:
        print("\n⚠️  Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ No continuity issues detected")
    
    print("="*60 + "\n")

def validate_epub_structure(output_dir):
    """Validate that all necessary EPUB structure files are present"""
    print("🔍 Validating EPUB structure...")
    
    required_files = {
        'container.xml': 'META-INF container file (critical)',
        '*.opf': 'OPF package file (critical)',
        '*.ncx': 'Navigation file (recommended)'
    }
    
    found_files = {}
    missing_files = []
    
    container_path = os.path.join(output_dir, 'container.xml')
    if os.path.exists(container_path):
        found_files['container.xml'] = 'Found'
        print("   ✅ container.xml - Found")
    else:
        missing_files.append('container.xml')
        print("   ❌ container.xml - Missing (CRITICAL)")
    
    opf_files = []
    ncx_files = []
    
    for file in os.listdir(output_dir):
        if file.lower().endswith('.opf'):
            opf_files.append(file)
        elif file.lower().endswith('.ncx'):
            ncx_files.append(file)
    
    if opf_files:
        found_files['opf'] = opf_files
        print(f"   ✅ OPF file(s) - Found: {', '.join(opf_files)}")
    else:
        missing_files.append('*.opf')
        print("   ❌ OPF file - Missing (CRITICAL)")
    
    if ncx_files:
        found_files['ncx'] = ncx_files
        print(f"   ✅ NCX file(s) - Found: {', '.join(ncx_files)}")
    else:
        missing_files.append('*.ncx')
        print("   ⚠️ NCX file - Missing (navigation may not work)")
    
    html_files = [f for f in os.listdir(output_dir) if f.lower().endswith('.html') and f.startswith('response_')]
    if html_files:
        print(f"   ✅ Translated chapters - Found: {len(html_files)} files")
    else:
        print("   ⚠️ No translated chapter files found")
    
    critical_missing = [f for f in missing_files if f in ['container.xml', '*.opf']]
    
    if not critical_missing:
        print("✅ EPUB structure validation PASSED")
        print("   All critical files present for EPUB reconstruction")
        return True
    else:
        print("❌ EPUB structure validation FAILED")
        print(f"   Missing critical files: {', '.join(critical_missing)}")
        print("   EPUB reconstruction may fail without these files")
        return False

def check_epub_readiness(output_dir):
    """Check if the output directory is ready for EPUB compilation"""
    print("📋 Checking EPUB compilation readiness...")
    
    issues = []
    
    if not validate_epub_structure(output_dir):
        issues.append("Missing critical EPUB structure files")
    
    html_files = [f for f in os.listdir(output_dir) if f.lower().endswith('.html') and f.startswith('response_')]
    if not html_files:
        issues.append("No translated chapter files found")
    else:
        print(f"   ✅ Found {len(html_files)} translated chapters")
    
    metadata_path = os.path.join(output_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        print("   ✅ Metadata file present")
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            if 'title' not in metadata:
                issues.append("Metadata missing title")
        except Exception as e:
            issues.append(f"Metadata file corrupted: {e}")
    else:
        issues.append("Missing metadata.json file")
    
    resource_dirs = ['css', 'fonts', 'images']
    found_resources = 0
    for res_dir in resource_dirs:
        res_path = os.path.join(output_dir, res_dir)
        if os.path.exists(res_path):
            files = [f for f in os.listdir(res_path) if os.path.isfile(os.path.join(res_path, f))]
            if files:
                found_resources += len(files)
                print(f"   ✅ Found {len(files)} {res_dir} files")
    
    if found_resources > 0:
        print(f"   ✅ Total resources: {found_resources} files")
    else:
        print("   ⚠️ No resource files found (this may be normal)")
    
    if not issues:
        print("🎉 EPUB compilation readiness: READY")
        print("   All necessary files present for EPUB creation")
        return True
    else:
        print("⚠️ EPUB compilation readiness: ISSUES FOUND")
        for issue in issues:
            print(f"   • {issue}")
        return False

def cleanup_previous_extraction(output_dir):
    """Clean up any files from previous extraction runs (preserves CSS files)"""
    # Remove 'css' from cleanup_items to preserve CSS files
    cleanup_items = [
         'images',  # Removed 'css' from this list
        '.resources_extracted'
    ]
    
    epub_structure_files = [
        'container.xml', 'content.opf', 'toc.ncx'
    ]
    
    cleaned_count = 0
    
    # Clean up directories (except CSS)
    for item in cleanup_items:
        if item.startswith('.'):
            continue
        item_path = os.path.join(output_dir, item)
        try:
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
                print(f"🧹 Removed directory: {item}")
                cleaned_count += 1
        except Exception as e:
            print(f"⚠️ Could not remove directory {item}: {e}")
    
    # Clean up EPUB structure files
    for epub_file in epub_structure_files:
        file_path = os.path.join(output_dir, epub_file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"🧹 Removed EPUB file: {epub_file}")
                cleaned_count += 1
        except Exception as e:
            print(f"⚠️ Could not remove {epub_file}: {e}")
    
    # Clean up any loose .opf and .ncx files
    try:
        for file in os.listdir(output_dir):
            if file.lower().endswith(('.opf', '.ncx')):
                file_path = os.path.join(output_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"🧹 Removed EPUB file: {file}")
                    cleaned_count += 1
    except Exception as e:
        print(f"⚠️ Error scanning for EPUB files: {e}")
    
    # Remove extraction marker
    marker_path = os.path.join(output_dir, '.resources_extracted')
    try:
        if os.path.isfile(marker_path):
            os.remove(marker_path)
            print(f"🧹 Removed extraction marker")
            cleaned_count += 1
    except Exception as e:
        print(f"⚠️ Could not remove extraction marker: {e}")
    
    # Check if CSS files exist and inform user they're being preserved
    css_path = os.path.join(output_dir, 'css')
    if os.path.exists(css_path):
        try:
            css_files = [f for f in os.listdir(css_path) if os.path.isfile(os.path.join(css_path, f))]
            if css_files:
                print(f"📚 Preserving {len(css_files)} CSS files")
        except Exception:
            pass
    
    if cleaned_count > 0:
        print(f"🧹 Cleaned up {cleaned_count} items from previous runs (CSS files preserved)")
    
    return cleaned_count

# =====================================================
# API AND TRANSLATION UTILITIES
# =====================================================
def send_with_interrupt(messages, client, temperature, max_tokens, stop_check_fn,
                       chunk_timeout=None, request_id=None, context=None,
                       chapter_context=None):
    """Send API request with interrupt capability and optional timeout retry.
    Optional context parameter is passed through to the client to improve payload labeling.

    chapter_context (dict) may contain "chapter", "chunk", and "total_chunks".
    When provided and the client supports set_chapter_context, it will be applied
    inside the API thread so that thread-local payload metadata is accurate.
    """
    # Import UnifiedClientError at function level to avoid scoping issues
    from unified_api_client import UnifiedClientError
    
    # The client.send() call will handle multi-key rotation automatically
    
    result_queue = queue.Queue()

    # Honor RETRY_TIMEOUT toggle: when off, disable chunk timeout entirely
    retry_env = os.getenv("RETRY_TIMEOUT")
    retry_timeout_enabled = retry_env is None or retry_env.strip().lower() not in ("0", "false", "off", "")
    if not retry_timeout_enabled:
        chunk_timeout = None
    
    def api_call():
        try:
            start_time = time.time()

            # Apply chapter/chunk context in THIS thread so UnifiedClient's
            # thread-local chapter_info is visible to payload saving.
            if chapter_context and hasattr(client, 'set_chapter_context'):
                try:
                    client.set_chapter_context(
                        chapter=chapter_context.get('chapter'),
                        chunk=chapter_context.get('chunk'),
                        total_chunks=chapter_context.get('total_chunks'),
                    )
                except Exception:
                    # Context is best-effort and should never break the call
                    pass
            
            # Build send parameters (context is optional)
            send_params = {
                'messages': messages,
                'temperature': temperature,
                'max_tokens': max_tokens,
            }
            sig = inspect.signature(client.send)
            if 'context' in sig.parameters and context is not None:
                send_params['context'] = context
            
            result = client.send(**send_params)
            
            # Capture raw response object for thought signatures (if available)
            raw_obj = None
            if hasattr(client, 'get_last_response_object'):
                resp_obj = client.get_last_response_object()
                if resp_obj and hasattr(resp_obj, 'raw_content_object'):
                    raw_obj = resp_obj.raw_content_object
                    # print("🧠 Captured thought signature for history in send_with_interrupt")
            
            elapsed = time.time() - start_time
            # Include raw_obj in the result tuple
            result_queue.put((result, elapsed, raw_obj))
        except Exception as e:
            result_queue.put(e)
    
    api_thread = threading.Thread(target=api_call)
    api_thread.daemon = True
    api_thread.start()
    
    timeout = chunk_timeout
    check_interval = 0.5
    elapsed = 0
    
    while True:
        try:
            result = result_queue.get(timeout=check_interval)
            if isinstance(result, Exception):
                # For expected errors like rate limits, preserve the error type without extra traceback
                if hasattr(result, 'error_type') and result.error_type == "rate_limit":
                    raise result
                elif "429" in str(result) or "rate limit" in str(result).lower():
                    # Convert generic exceptions to UnifiedClientError for rate limits
                    raise UnifiedClientError(str(result), error_type="rate_limit")
                else:
                    raise result
            if isinstance(result, tuple):
                # Unpack the tuple (now includes raw_obj)
                if len(result) == 3:
                    api_result, api_time, raw_obj = result
                    # Store raw_obj as an attribute for later retrieval
                    if hasattr(api_result, '__class__'):
                        # If api_result is a tuple, return a new tuple with raw_obj
                        if isinstance(api_result, tuple):
                            return (*api_result, raw_obj)
                        else:
                            # Store as attribute for retrieval
                            api_result._raw_obj = raw_obj
                else:
                    # Backward compatibility for old format
                    api_result, api_time = result
                    
                if chunk_timeout is not None and api_time > chunk_timeout:
                    # Set cleanup flag when chunk timeout occurs
                    if hasattr(client, '_in_cleanup'):
                        client._in_cleanup = True
                    if hasattr(client, 'cancel_current_operation'):
                        client.cancel_current_operation()
                    raise UnifiedClientError(f"API call took {api_time:.1f}s (timeout: {chunk_timeout}s)")
                return api_result
            return result
        except queue.Empty:
            if stop_check_fn():
                # Set cleanup flag when user stops
                if hasattr(client, '_in_cleanup'):
                    client._in_cleanup = True
                if hasattr(client, 'cancel_current_operation'):
                    client.cancel_current_operation()
                raise UnifiedClientError("Translation stopped by user")
            elapsed += check_interval
            if chunk_timeout is not None and elapsed >= chunk_timeout:
                if hasattr(client, '_in_cleanup'):
                    client._in_cleanup = True
                if hasattr(client, 'cancel_current_operation'):
                    client.cancel_current_operation()
                raise UnifiedClientError(f"API call timed out after {chunk_timeout} seconds")

def handle_api_error(processor, error, chunk_info=""):
    """Handle API errors with multi-key support"""
    error_str = str(error)
    
    # Check for rate limit
    if "429" in error_str or "rate limit" in error_str.lower():
        if processor.config.use_multi_api_keys:
            print(f"⚠️ Rate limit hit {chunk_info}, client should rotate to next key")
            stats = processor.client.get_stats()
            print(f"📊 API Stats - Active keys: {stats.get('active_keys', 0)}/{stats.get('total_keys', 0)}")
            
            if stats.get('active_keys', 0) == 0:
                print("⏳ All API keys are cooling down - will wait and retry")
            print(f"🔄 Multi-key error handling: Rate limit processed, preparing for key rotation...")
            time.sleep(0.1)  # Brief pause after rate limit detection for stability
            return True  # Always retry
        else:
            print(f"⚠️ Rate limit hit {chunk_info}, waiting before retry...")
            time.sleep(60)
            print(f"🔄 Single-key error handling: Rate limit wait completed, ready for retry...")
            time.sleep(0.1)  # Brief pause after rate limit wait for stability
            return True  # Always retry
    
    # Other errors
    print(f"❌ API Error {chunk_info}: {error_str}")
    return False
    
def parse_token_limit(env_value):
    """Parse token limit from environment variable"""
    if not env_value or env_value.strip() == "":
        return None, "unlimited"
    
    env_value = env_value.strip()
    if env_value.lower() == "unlimited":
        return None, "unlimited"
    
    if env_value.isdigit() and int(env_value) > 0:
        limit = int(env_value)
        return limit, str(limit)
    
    return 1000000, "1000000 (default)"

def build_system_prompt(user_prompt, glossary_path=None, source_text=None):
    """Build the system prompt with glossary - TRUE BRUTE FORCE VERSION"""
    append_glossary = os.getenv("APPEND_GLOSSARY", "1") == "1"
    actual_glossary_path = glossary_path
    
    # Replace {target_lang} placeholder if present
    target_lang = os.getenv("OUTPUT_LANGUAGE", "English")
    if user_prompt and "{target_lang}" in user_prompt:
        user_prompt = user_prompt.replace("{target_lang}", target_lang)
    
    system = user_prompt if user_prompt else ""
    
    if append_glossary and actual_glossary_path and os.path.exists(actual_glossary_path):
        try:
            print(f"✅ Loading glossary from: {os.path.abspath(actual_glossary_path)}")
            
            # Try to load as JSON first
            try:
                with open(actual_glossary_path, "r", encoding="utf-8") as gf:
                    glossary_data = json.load(gf)
                glossary_text = json.dumps(glossary_data, ensure_ascii=False, indent=2)
                print(f"Loaded as JSON")
            except json.JSONDecodeError:
                # If JSON fails, just read as raw text
                with open(actual_glossary_path, "r", encoding="utf-8") as gf:
                    glossary_text = gf.read()
            
            # Apply glossary compression if enabled and source text is provided
            compress_glossary_enabled = os.getenv("COMPRESS_GLOSSARY_PROMPT", "0") == "1"
            if compress_glossary_enabled and source_text:
                try:
                    from glossary_compressor import compress_glossary
                    original_glossary_text = glossary_text  # Store original for token counting
                    original_length = len(glossary_text)
                    glossary_text = compress_glossary(glossary_text, source_text, glossary_format='auto')
                    compressed_length = len(glossary_text)
                    reduction_pct = ((original_length - compressed_length) / original_length * 100) if original_length > 0 else 0
                    
                    # Also calculate token savings if tiktoken is available
                    try:
                        import tiktoken
                        try:
                            enc = tiktoken.encoding_for_model(os.getenv("MODEL", "gpt-4"))
                        except:
                            enc = tiktoken.get_encoding("cl100k_base")
                        
                        # Count tokens for original and compressed glossary
                        original_tokens = len(enc.encode(original_glossary_text))
                        compressed_tokens = len(enc.encode(glossary_text))
                        token_reduction = original_tokens - compressed_tokens
                        token_reduction_pct = (token_reduction / original_tokens * 100) if original_tokens > 0 else 0
                        
                        print(f"🗜️ Glossary: {original_length:,}→{compressed_length:,} chars ({reduction_pct:.1f}%), {original_tokens:,}→{compressed_tokens:,} tokens ({token_reduction_pct:.1f}%)")
                    except ImportError:
                        # If tiktoken is not available, just show character reduction
                        print(f"🗜️ Glossary compressed: {original_length:,} → {compressed_length:,} chars ({reduction_pct:.1f}% reduction)")
                except Exception as e:
                    print(f"⚠️ Glossary compression failed: {e}")
                    # Continue with uncompressed glossary
            
            if system:
                system += "\n\n"
            
            custom_prompt = os.getenv("APPEND_GLOSSARY_PROMPT", "").strip()
            if not custom_prompt:
                raise ValueError(
                    "APPEND_GLOSSARY_PROMPT environment variable is not set!\n"
                    "Please configure your glossary append format in:\n"
                    "Glossary Manager → Automatic Glossary → Glossary Append Format"
                )
            
            system += f"{custom_prompt}\n{glossary_text}"
            
            print(f"✅ Glossary appended ({len(glossary_text):,} characters)")
            
            # Check for glossary extension file (only if ADD_ADDITIONAL_GLOSSARY is enabled)
            add_additional_glossary = os.getenv("ADD_ADDITIONAL_GLOSSARY", "0") == "1"
            if add_additional_glossary:
                glossary_dir = os.path.dirname(actual_glossary_path)
                # Check for extension with any supported format
                additional_glossary_path = None
                for ext in ['.csv', '.md', '.txt', '.json']:
                    candidate = os.path.join(glossary_dir, f"glossary_extension{ext}")
                    if os.path.exists(candidate):
                        additional_glossary_path = candidate
                        break
                
                if additional_glossary_path:
                    try:
                        print(f"✅ Loading glossary extension from: {os.path.basename(additional_glossary_path)}")
                        with open(additional_glossary_path, "r", encoding="utf-8") as af:
                            additional_glossary_text = af.read()
                        
                        # Apply same compression logic if enabled
                        if compress_glossary_enabled and source_text:
                            try:
                                from glossary_compressor import compress_glossary
                                original_add_length = len(additional_glossary_text)
                                additional_glossary_text = compress_glossary(additional_glossary_text, source_text, glossary_format='auto')
                                compressed_add_length = len(additional_glossary_text)
                                add_reduction_pct = ((original_add_length - compressed_add_length) / original_add_length * 100) if original_add_length > 0 else 0
                                print(f"🗃️ Glossary extension compressed: {original_add_length:,} → {compressed_add_length:,} chars ({add_reduction_pct:.1f}% reduction)")
                            except Exception as e:
                                print(f"⚠️ Glossary extension compression failed: {e}")
                        
                        # Append glossary extension
                        system += f"\n\n{additional_glossary_text}"
                        print(f"✅ Glossary extension appended ({len(additional_glossary_text):,} characters)")
                        
                    except Exception as e:
                        print(f"⚠️ Failed to load glossary extension: {e}")
                
        except Exception as e:
            print(f"[ERROR] Could not load glossary: {e}")
            import traceback
            print(f"[ERROR] Full traceback: {traceback.format_exc()}")
    else:
        if not append_glossary:
            #print(f"[DEBUG] ❌ Glossary append disabled")
            pass
        elif not actual_glossary_path:
            # Check if we're translating CSV/JSON files (they typically don't need glossaries)
            input_path = os.getenv('EPUB_PATH', '')
            if not input_path.lower().endswith(('.csv', '.json')):
                print(f"[DEBUG] ❌ No glossary path provided")
        elif not os.path.exists(actual_glossary_path):
            print(f"[DEBUG] ❌ Glossary file does not exist: {actual_glossary_path}")
    
    # Calculate token count for system prompt
    try:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(os.getenv("MODEL", "gpt-4"))
        except:
            enc = tiktoken.get_encoding("cl100k_base")
        system_tokens = len(enc.encode(system))
        print(f"🎯 Final system prompt: {len(system):,} chars, {system_tokens:,} tokens")
    except ImportError:
        print(f"🎯 Final system prompt length: {len(system)} characters")
    
    return system

def translate_title(title, client, system_prompt, user_prompt, temperature=0.3):
    """Translate the book title using the configured settings"""
    if not title or not title.strip():
        return title
        
    print(f"📚 Processing book title: {title}")
    
    try:
        if os.getenv("TRANSLATE_BOOK_TITLE", "1") == "0":
            print(f"📚 Book title translation disabled - keeping original")
            return title
        
        # Check if we're using a translation service (not AI)
        client_type = getattr(client, 'client_type', '')
        is_translation_service = client_type in ['deepl', 'google_translate']
        
        if is_translation_service:
            # For translation services, send only the text without AI prompts
            print(f"📚 Using translation service ({client_type}) - sending text directly")
            messages = [
                {"role": "user", "content": title}
            ]
            max_tokens = int(os.getenv("MAX_OUTPUT_TOKENS", "8192"))
            translated_title, _ = client.send(messages, temperature=temperature, max_tokens=max_tokens)
        else:
            # For AI services, use prompts as before
            book_title_prompt = os.getenv("BOOK_TITLE_PROMPT", 
                "Translate this book title to English while retaining any acronyms:")
            
            # Get the system prompt for book titles, with fallback to default
            book_title_system_prompt = os.getenv("BOOK_TITLE_SYSTEM_PROMPT", 
                "You are a translator. Respond with only the translated text, nothing else. Do not add any explanation or additional content.")
            
            # Replace {target_lang} variable with output language
            output_lang = os.getenv("OUTPUT_LANGUAGE", "English")
            book_title_prompt = book_title_prompt.replace("{target_lang}", output_lang)
            book_title_system_prompt = book_title_system_prompt.replace("{target_lang}", output_lang)
            
            messages = [
                {"role": "system", "content": book_title_system_prompt},
                {"role": "user", "content": f"{book_title_prompt}\n\n{title}"}
            ]
            max_tokens = int(os.getenv("MAX_OUTPUT_TOKENS", "8192"))
            translated_title, _ = client.send(messages, temperature=temperature, max_tokens=max_tokens)
        
        print(f"[DEBUG] Raw API response: '{translated_title}'")
        print(f"[DEBUG] Response length: {len(translated_title)} (original: {len(title)})")
        newline = '\n'
        print(f"[DEBUG] Has newlines: {repr(translated_title) if newline in translated_title else 'No'}")
        
        translated_title = translated_title.strip()
        
        if ((translated_title.startswith('"') and translated_title.endswith('"')) or 
            (translated_title.startswith("'") and translated_title.endswith("'"))):
            translated_title = translated_title[1:-1].strip()
        
        if '\n' in translated_title:
            print(f"⚠️ API returned multi-line content, keeping original title")
            return title           
            
        # Check for JSON-like structured content, but allow simple brackets like [END]
        if (any(char in translated_title for char in ['{', '}']) or 
            '"role":' in translated_title or 
            '"content":' in translated_title or
            ('[[' in translated_title and ']]' in translated_title)):  # Only flag double brackets
            print(f"⚠️ API returned structured content, keeping original title")
            return title
            
        if any(tag in translated_title.lower() for tag in ['<p>', '</p>', '<h1>', '</h1>', '<html']):
            print(f"⚠️ API returned HTML content, keeping original title")
            return title
        
        print(f"✅ Processed title: {translated_title}")
        return translated_title
        
    except Exception as e:
        print(f"⚠️ Failed to process title: {e}")
        return title

# =====================================================
# FAILURE RESPONSES
# =====================================================
def is_qa_failed_response(content):
    """
    Comprehensive check for API failure markers based on research of major AI providers
    (OpenAI, Anthropic, Google Gemini, Azure OpenAI, etc.)
    """
    if not content:
        return True
    
    content_str = str(content).strip()
    content_lower = content_str.lower()
    
    # 1. EXPLICIT FAILURE MARKERS from unified_api_client fallback responses
    explicit_failures = [
        "[TRANSLATION FAILED - ORIGINAL TEXT PRESERVED]",
        "[IMAGE TRANSLATION FAILED]",
        "[Content Blocked]",
        "API response unavailable",
        "[]",  # Empty JSON response from glossary context
        "[API_ERROR]",
        "[TIMEOUT]",
        "[RATE_LIMIT_EXCEEDED]",
        "All Google Translate endpoints failed"  # Free Google Translate failures
    ]
    
    for marker in explicit_failures:
        if marker in content_str:
            return True
    
    # 2. HTTP ERROR STATUS MESSAGES
    http_errors = [
        "400 - invalid_request_error",
        "401 - authentication_error", 
        "403 - permission_error",
        "404 - not_found_error",
        "413 - request_too_large",
        "429 - rate_limit_error",
        "500 - api_error",
        "529 - overloaded_error",
        "invalid x-api-key",
        "authentication_error",
        "permission_error",
        "rate_limit_error",
        "api_error",
        "overloaded_error"
    ]
    
    for error in http_errors:
        if error in content_lower:
            return True
    
    # 3. CONTENT FILTERING / SAFETY BLOCKS
    content_filter_markers = [
        "content_filter",  # OpenAI finish_reason
        "content was blocked",
        "response was blocked",
        "safety filter",
        "content policy",
        "harmful content",
        "content filtering",
        "blocked by safety",
        "harm_category_harassment",
        "harm_category_hate_speech", 
        "harm_category_sexually_explicit",
        "harm_category_dangerous_content",
        "block_low_and_above",
        "block_medium_and_above",
        "block_only_high"
    ]
    
    for marker in content_filter_markers:
        if marker in content_lower:
            return True
    
    # 4. TIMEOUT AND NETWORK ERRORS
    timeout_markers = [
        "timed out",
        "request timeout",
        "connection timeout",
        "read timeout",
        "apitimeouterror",
        "network error",
        "connection refused",
        "connection reset",
        "socket timeout"
    ]
    
    for marker in timeout_markers:
        if marker in content_lower:
            return True
    
    # 6. EMPTY OR MINIMAL RESPONSES INDICATING FAILURE
    if len(content_str) <= 10:
        # Very short responses that are likely errors
        short_error_indicators = [
            "error", "fail", "null", "none", "empty", 
            "unavailable", "timeout", "blocked", "denied"
        ]
        if any(indicator in content_lower for indicator in short_error_indicators):
            return True
    
    # 7. COMMON REFUSAL PATTERNS (AI refusing to generate content)
    # Be more precise: look for AI refusal patterns, not natural dialogue
    refusal_patterns = [
        "i cannot assist", "i can't assist", "i'm not able to assist",
        "i cannot help", "i can't help", "i'm unable to help",
        "i'm afraid i cannot help with that", "designed to ensure appropriate use",
        "as an ai", "as a language model", "as an ai language model",
        "i don't feel comfortable", "i apologize, but i cannot",
        "i'm sorry, but i can't assist", "i'm sorry, but i cannot assist",
        "against my programming", "against my guidelines",
        "violates content policy", "i'm not programmed to",
        "cannot provide that kind", "unable to provide that",
        "i cannot assist with this request",
        "that's not within my capabilities to appropriately assist with",
        "is there something different i can help you with",
        "careful ethical considerations",
        "i could help you with a different question or task",
        "what other topics or questions can i help you explore",
        "i cannot and will not translate",
        "i cannot translate this content",
        "i can't translate this content",
    ]
    
    # Check responses up to 1000 chars (AIs can be verbose when refusing)
    if len(content_str) < 1000:
        for pattern in refusal_patterns:
            if pattern in content_lower:
                return True
    
    # 8. JSON ERROR RESPONSES
    json_error_patterns = [
        '{"error"',
        '{"type":"error"',
        '"error_type"',
        '"error_message"',
        '"error_code"',
        '"message":"error"',
        '"status":"error"',
        '"success":false'
    ]
    
    for pattern in json_error_patterns:
        if pattern in content_lower:
            return True
    
    # 9. GEMINI-SPECIFIC ERRORS
    gemini_errors = [
        "finish_reason: safety",
        "finish_reason: other", 
        "finish_reason: recitation",
        "candidate.content field",  # Voided content field
        "safety_ratings",
        "probability_score",
        "severity_score"
    ]
    
    for error in gemini_errors:
        if error in content_lower:
            return True
    
    # 10. ANTHROPIC-SPECIFIC ERRORS  
    anthropic_errors = [
        "invalid_request_error",
        "authentication_error",
        "permission_error", 
        "not_found_error",
        "request_too_large",
        "rate_limit_error",
        "api_error",
        "overloaded_error"
    ]
    
    for error in anthropic_errors:
        if error in content_lower:
            return True
    
    # 11. OPENAI-SPECIFIC ERRORS
    openai_errors = [
        "finish_reason: content_filter",
        "finish_reason: length",  # Only if very short content
        "insufficient_quota",
        "invalid_api_key",
        "model_not_found",
        "context_length_exceeded"
    ]
    
    for error in openai_errors:
        if error in content_lower:
            return True
    
    # 12. EMPTY RESPONSE PATTERNS
    empty_patterns = [
        "choices: [ { text: '', index: 0",  # OpenAI empty response pattern
        '"text": ""',
        '"content": ""',
        '"content": null',
        "text: ''",
        "content: ''"
    ]
    
    for pattern in empty_patterns:
        if pattern in content_lower:
            return True
    
    # 13. PROVIDER-AGNOSTIC ERROR MESSAGES
    generic_errors = [
        "internal server error",
        "service error", 
        "server error",
        "bad gateway",
        "service temporarily unavailable",
        "upstream error",
        "proxy error",
        "gateway timeout",
        "connection error",
        "network failure",
        "service degraded",
        "maintenance mode"
    ]
    
    for error in generic_errors:
        if error in content_lower:
            return True
    
    # 14. SPECIAL CASE: Check for responses that are just original text
    # (indicating translation completely failed and fallback was used)
    if content_str.startswith("[") and content_str.endswith("]") and "FAILED" in content_str:
        return True
    
    # 15. FINAL CHECK: Very short responses with error indicators
    if len(content_str) < 100:
        final_error_check = [
            "error", "failed", "timeout", "blocked", "denied", 
            "refused", "rejected", "unavailable", "invalid", 
            "forbidden", "unauthorized", "limit", "quota"
        ]
        
        # Count how many error indicators are present
        error_count = sum(1 for word in final_error_check if word in content_lower)
        
        # If multiple error indicators in short response, likely a failure
        if error_count >= 2:
            return True
        
        # Single strong error indicator in very short response
        if len(content_str) < 50 and error_count >= 1:
            return True
    
    return False


# Additional helper function for debugging
def get_failure_reason(content):
    """
    Returns the specific reason why content was marked as qa_failed
    Useful for debugging and logging
    """
    if not content:
        return "Empty content"
    
    content_str = str(content).strip()
    content_lower = content_str.lower()
    
    # Check each category and return the first match
    failure_categories = {
        "Explicit Failure Marker": [
            "[TRANSLATION FAILED - ORIGINAL TEXT PRESERVED]",
            "[IMAGE TRANSLATION FAILED]", 
            "API response unavailable",
            "[]"
        ],
        "HTTP Error": [
            "authentication_error", "rate_limit_error", "api_error"
        ],
        "Content Filter": [
            "content_filter", "safety filter", "blocked by safety"
        ],
        "Timeout": [
            "timeout", "timed out", "apitimeouterror"
        ],
        "Rate Limit": [
            "rate limit exceeded", "quota exceeded", "too many requests"
        ],
        "Refusal Pattern": [
            "i cannot", "i can't", "unable to process"
        ],
        "Empty Response": [
            '"text": ""', "choices: [ { text: ''"
        ]
    }
    
    for category, markers in failure_categories.items():
        for marker in markers:
            if marker in content_str or marker in content_lower:
                return f"{category}: {marker}"
    
    if len(content_str) < 50:
        return f"Short response with error indicators: {content_str[:30]}..."
    
    return "Unknown failure pattern"
    
def convert_enhanced_text_to_html(plain_text, chapter_info=None):
    """Convert markdown/plain text back to HTML after translation (for enhanced mode)
    
    This function handles the conversion of translated markdown back to HTML.
    The input is the TRANSLATED text that was originally extracted using html2text.
    """
    import re
    
    preserve_structure = chapter_info.get('preserve_structure', False) if chapter_info else False
    
    # First, try to use markdown2 for proper markdown conversion
    try:
        import markdown2
        
        # Check if the text contains markdown patterns
        has_markdown = any([
            '##' in plain_text,  # Headers
            '**' in plain_text,  # Bold
            '*' in plain_text and not '**' in plain_text,  # Italic
            '[' in plain_text and '](' in plain_text,  # Links
            '```' in plain_text,  # Code blocks
            '> ' in plain_text,  # Blockquotes
            '- ' in plain_text or '* ' in plain_text or '1. ' in plain_text  # Lists
        ])
        
        if has_markdown or preserve_structure:
            # Use markdown2 for proper conversion
            html = markdown2.markdown(plain_text, extras=[
                'cuddled-lists',       # Lists without blank lines
                'fenced-code-blocks',  # Code blocks with ```
                'break-on-newline',    # Treat single newlines as <br>
                'smarty-pants',        # Smart quotes and dashes
                'tables',              # Markdown tables
            ])
            
            # Post-process to ensure proper paragraph structure
            if not '<p>' in html:
                # If markdown2 didn't create paragraphs, wrap content
                lines = html.split('\n')
                processed_lines = []
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('<') and not line.endswith('>'):
                        processed_lines.append(f'<p>{line}</p>')
                    elif line:
                        processed_lines.append(line)
                html = '\n'.join(processed_lines)
            
            return html
            
    except ImportError:
        print("⚠️ markdown2 not available, using fallback HTML conversion")
    
    # Fallback: Manual markdown-to-HTML conversion
    lines = plain_text.strip().split('\n')
    html_parts = []
    in_code_block = False
    code_block_content = []
    
    for line in lines:
        # Handle code blocks
        if line.strip().startswith('```'):
            if in_code_block:
                # End code block
                html_parts.append('<pre><code>' + '\n'.join(code_block_content) + '</code></pre>')
                code_block_content = []
                in_code_block = False
            else:
                # Start code block
                in_code_block = True
            continue
        
        if in_code_block:
            code_block_content.append(line)
            continue
        
        line = line.strip()
        if not line:
            # Preserve empty lines as paragraph breaks
            if html_parts and not html_parts[-1].endswith('</p>'):
                # Only add break if not already after a closing tag
                html_parts.append('<br/>')
            continue
        
        # Check for markdown headers
        if line.startswith('#'):
            match = re.match(r'^(#+)\s*(.+)$', line)
            if match:
                level = min(len(match.group(1)), 6)
                header_text = match.group(2).strip()
                html_parts.append(f'<h{level}>{header_text}</h{level}>')
                continue
        
        # Check for blockquotes
        if line.startswith('> '):
            quote_text = line[2:].strip()
            html_parts.append(f'<blockquote>{quote_text}</blockquote>')
            continue
        
        # Check for lists
        if re.match(r'^[*\-+]\s+', line):
            list_text = re.sub(r'^[*\-+]\s+', '', line)
            html_parts.append(f'<li>{list_text}</li>')
            continue
        
        if re.match(r'^\d+\.\s+', line):
            list_text = re.sub(r'^\d+\.\s+', '', line)
            html_parts.append(f'<li>{list_text}</li>')
            continue
        
        # Convert inline markdown
        # Bold
        line = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', line)
        line = re.sub(r'__(.+?)__', r'<strong>\1</strong>', line)
        
        # Italic
        line = re.sub(r'\*(.+?)\*', r'<em>\1</em>', line)
        line = re.sub(r'_(.+?)_', r'<em>\1</em>', line)
        
        # Links
        line = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', line)
        
        # Code inline
        line = re.sub(r'`([^`]+)`', r'<code>\1</code>', line)
        
        # Regular paragraph
        html_parts.append(f'<p>{line}</p>')
    
    # Post-process lists to wrap in ul/ol tags
    final_html = []
    in_list = False
    list_type = None
    
    for part in html_parts:
        if part.startswith('<li>'):
            if not in_list:
                # Determine list type based on context (simplified)
                list_type = 'ul'  # Default to unordered
                final_html.append(f'<{list_type}>')
                in_list = True
            final_html.append(part)
        else:
            if in_list:
                final_html.append(f'</{list_type}>')
                in_list = False
            final_html.append(part)
    
    # Close any open list
    if in_list:
        final_html.append(f'</{list_type}>')
    
    return '\n'.join(final_html)
# =====================================================
# MAIN TRANSLATION FUNCTION
# =====================================================
def main(log_callback=None, stop_callback=None):
    """Main translation function with enhanced duplicate detection and progress tracking"""
    global STOP_LOGGED
    STOP_LOGGED = False
    config = TranslationConfig()
    builtins._DISABLE_ZERO_DETECTION = config.DISABLE_ZERO_DETECTION
    
    if config.DISABLE_ZERO_DETECTION:
        print("=" * 60)
        print("⚠️  0-BASED DETECTION DISABLED BY USER")
        print("⚠️  All chapter numbers will be used exactly as found")
        print("=" * 60)
    
    args = None
    chapters_completed = 0
    chunks_completed = 0
    
    args = None
    chapters_completed = 0
    chunks_completed = 0
    
    input_path = config.input_path
    if not input_path and len(sys.argv) > 1:
        input_path = sys.argv[1]
    
    is_text_file = input_path.lower().endswith(('.txt', '.csv', '.json', '.md'))
    is_pdf_file = input_path.lower().endswith('.pdf')
    
    if is_text_file:
        os.environ["IS_TEXT_FILE_TRANSLATION"] = "1"
        
    import json as _json
    _original_load = _json.load
      
    def debug_json_load(fp, *args, **kwargs):
        result = _original_load(fp, *args, **kwargs)
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], dict) and 'original_name' in result[0]:
                print(f"[DEBUG] Loaded glossary list with {len(result)} items from {fp.name if hasattr(fp, 'name') else 'unknown'}")
        return result
    
    _json.load = debug_json_load
    
    if log_callback:
        set_output_redirect(log_callback)
    
    def check_stop():
        if stop_callback and stop_callback():
            log_stop_once()
            return True
        return is_stop_requested()
    
    if config.EMERGENCY_RESTORE:
        print("✅ Emergency paragraph restoration is ENABLED")
    else:
        print("⚠️ Emergency paragraph restoration is DISABLED")
    
    print(f"[DEBUG] REMOVE_AI_ARTIFACTS environment variable: {os.getenv('REMOVE_AI_ARTIFACTS', 'NOT SET')}")
    print(f"[DEBUG] REMOVE_AI_ARTIFACTS parsed value: {config.REMOVE_AI_ARTIFACTS}")
    if config.REMOVE_AI_ARTIFACTS:
        print("⚠️ AI artifact removal is ENABLED - will clean AI response artifacts")
    else:
        print("✅ AI artifact removal is DISABLED - preserving all content as-is")
       
    if '--epub' in sys.argv or (len(sys.argv) > 1 and sys.argv[1].endswith(('.epub', '.txt', '.csv', '.json', '.pdf', '.md'))):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('epub', help='Input EPUB or text file')
        args = parser.parse_args()
        input_path = args.epub
    
    is_text_file = input_path.lower().endswith(('.txt', '.csv', '.json', '.md'))
    is_pdf_file = input_path.lower().endswith('.pdf')
    
    # Disable Break Split Count for EPUB files (only works with plain text files)
    if input_path.lower().endswith('.epub'):
        if os.getenv('BREAK_SPLIT_COUNT', ''):
            print("⚠️ Break Split Count disabled for EPUB files (only works with .txt files)")
        os.environ['BREAK_SPLIT_COUNT'] = ''
    
    if is_text_file:
        file_base = os.path.splitext(os.path.basename(input_path))[0]
    else:
        epub_base = os.path.splitext(os.path.basename(input_path))[0]
        file_base = epub_base
        
    # Allow callers (e.g. Discord bot) to control where outputs are written.
    # This avoids relying on process-wide cwd changes (os.chdir), which is unsafe in multi-threaded apps.
    output_root = (os.getenv("OUTPUT_DIRECTORY") or os.getenv("OUTPUT_DIR") or "").strip()
    if output_root:
        try:
            os.makedirs(output_root, exist_ok=True)
        except Exception:
            # If we can't create the root, fall back to relative output.
            output_root = ""

    out = os.path.join(output_root, file_base) if output_root else file_base
    os.makedirs(out, exist_ok=True)
    print(f"[DEBUG] Created output folder → {out}")
    
    cleanup_previous_extraction(out)

    os.environ["EPUB_OUTPUT_DIR"] = out
    payloads_dir = out

    # Manage translation history persistence based on contextual + rolling settings
    history_file = os.path.join(payloads_dir, "translation_history.json")
    if os.path.exists(history_file):
        if config.CONTEXTUAL and config.TRANSLATION_HISTORY_ROLLING:
            # Preserve existing history across runs when using rolling window
            print(f"[DEBUG] Preserving translation history (rolling window enabled) → {history_file}")
        elif config.CONTEXTUAL:
            # Contextual on but rolling disabled: start fresh each run
            os.remove(history_file)
            print(f"[DEBUG] CONTEXTUAL enabled without rolling - purged translation history → {history_file}")
        else:
            # Contextual off: never keep history
            os.remove(history_file)
            print("[DEBUG] CONTEXTUAL disabled - cleared translation history")
            
    history_manager = HistoryManager(payloads_dir)
    chapter_splitter = ChapterSplitter(model_name=config.MODEL)
    chunk_context_manager = ChunkContextManager()
    progress_manager = ProgressManager(payloads_dir)
    
    # Prepare progress callback for chapter extraction
    # Filter to show only every 10% progress update
    chapter_progress_callback = None
    _progress_state = {}  # Track last shown percentage for each progress type
    
    if log_callback:
        def chapter_progress_callback(msg):
            # Check if this is a progress message with percentage
            import re
            
            # Try to extract percentage from formatted progress bars
            percent_match = re.search(r'\((\d+)%\)', msg)
            if percent_match:
                percent = int(percent_match.group(1))
                
                # Determine progress type from message
                if '📂' in msg or 'Scanning' in msg:
                    prog_type = 'scan'
                elif '📦' in msg or 'Extracting' in msg:
                    prog_type = 'extract'
                elif '📚' in msg or 'Processing chapters' in msg:
                    prog_type = 'process'
                elif '📊' in msg or 'metadata' in msg.lower():
                    prog_type = 'metadata'
                else:
                    prog_type = 'other'
                
                # Get last shown percentage for this type
                last_percent = _progress_state.get(prog_type, -1)
                
                # Show if: crossed a 10% threshold, or reached 100%
                should_show = (percent // 10 > last_percent // 10) or (percent == 100)
                
                if should_show:
                    _progress_state[prog_type] = percent
                    log_callback(msg)
            else:
                # Not a progress percentage message, always show
                log_callback(msg)
    
    # Import Chapter_Extractor module functions
    import Chapter_Extractor
    # GlossaryManager is now a module with functions, not a class

    print("🔍 Checking for deleted output files...")
    progress_manager.cleanup_missing_files(out)
    progress_manager.save()

    if check_stop():
        return

    # Check if model needs API key
    model_needs_api_key = not (config.MODEL.lower() in ['google-translate', 'google-translate-free'] or 
                              '@' in config.MODEL or config.MODEL.startswith('vertex/'))
    
    if model_needs_api_key and not config.API_KEY:
        print("❌ Error: Set API_KEY, OPENAI_API_KEY, or OPENAI_OR_Gemini_API_KEY in your environment.")
        return
    
    # Set dummy API key for models that don't need one
    if not config.API_KEY:
        config.API_KEY = 'dummy-key-not-required'

    #print(f"[DEBUG] Found API key: {config.API_KEY[:10]}...")
    print(f"[DEBUG] Using model = {config.MODEL}")
    print(f"[DEBUG] Max output tokens = {config.MAX_OUTPUT_TOKENS}")

    client = UnifiedClient(model=config.MODEL, api_key=config.API_KEY, output_dir=out)
    if hasattr(client, 'use_multi_keys') and client.use_multi_keys:
        stats = client.get_stats()
        print(f"🔑 Multi-key mode active: {stats.get('total_keys', 0)} keys loaded")
        print(f"   Active keys: {stats.get('active_keys', 0)}")
    else:
        print(f"🔑 Single-key mode: Using {config.MODEL}")    
    # Reset cleanup state when starting new translation
    if hasattr(client, 'reset_cleanup_state'):
        client.reset_cleanup_state()    
        
    if is_pdf_file:
        print("📄 Processing PDF file...")
        try:
            txt_processor = TextFileProcessor(input_path, out)
            chapters = txt_processor.extract_chapters()
            txt_processor.save_original_structure()
            
            metadata = {
                "title": os.path.splitext(os.path.basename(input_path))[0],
                "type": "pdf",
                "chapter_count": len(chapters)
            }
        except ImportError as e:
            print(f"❌ Error: PDF processor not available: {e}")
            if log_callback:
                log_callback(f"❌ Error: PDF processor not available: {e}")
            return
        except Exception as e:
            print(f"❌ Error processing PDF file: {e}")
            if log_callback:
                log_callback(f"❌ Error processing PDF file: {e}")
            return
    elif is_text_file:
        print("📄 Processing text file...")
        try:
            txt_processor = TextFileProcessor(input_path, out)
            chapters = txt_processor.extract_chapters()
            txt_processor.save_original_structure()
            
            metadata = {
                "title": os.path.splitext(os.path.basename(input_path))[0],
                "type": "text",
                "chapter_count": len(chapters)
            }
        except ImportError as e:
            print(f"❌ Error: Text file processor not available: {e}")
            if log_callback:
                log_callback(f"❌ Error: Text file processor not available: {e}")
            return
        except Exception as e:
            print(f"❌ Error processing text file: {e}")
            if log_callback:
                log_callback(f"❌ Error processing text file: {e}")
            return
    else:
        # Check if we should use async extraction (for GUI mode)
        use_async_extraction = os.getenv("USE_ASYNC_CHAPTER_EXTRACTION", "0") == "1"
        
        if use_async_extraction and log_callback:
            print("🚀 Using async chapter extraction (subprocess mode)...")
            from chapter_extraction_manager import ChapterExtractionManager
            
            # Create manager with log callback
            extraction_manager = ChapterExtractionManager(log_callback=log_callback)
            
            # Get extraction mode
            extraction_mode = os.getenv("EXTRACTION_MODE", "smart").lower()
            
            # Define completion callback
            extraction_result = {"completed": False, "result": None}
            
            def on_extraction_complete(result):
                extraction_result["completed"] = True
                extraction_result["result"] = result
                
                # Safety check for None result
                if result is None:
                    log_callback("❌ Chapter extraction failed: No result returned")
                    return
                
                if result.get("success"):
                    log_callback(f"✅ Chapter extraction completed: {result.get('chapters', 0)} chapters")
                else:
                    log_callback(f"❌ Chapter extraction failed: {result.get('error', 'Unknown error')}")
            
            # Start async extraction
            extraction_manager.extract_chapters_async(
                input_path,
                out,
                extraction_mode=extraction_mode,
                progress_callback=lambda msg: log_callback(f"📊 {msg}"),
                completion_callback=on_extraction_complete
            )
            
            # Wait for completion (with timeout if retry-timeout is enabled)
            retry_env = os.getenv("RETRY_TIMEOUT")
            retry_timeout_enabled = retry_env is None or retry_env.strip().lower() not in ("0", "false", "off", "")
            if retry_timeout_enabled:
                env_ct = os.getenv("CHUNK_TIMEOUT", "900")  # legacy default
                try:
                    timeout = float(env_ct)
                    if timeout <= 0:
                        timeout = None
                except Exception:
                    timeout = None
            else:
                timeout = None
            start_time = time.time()
            
            while not extraction_result["completed"]:
                if check_stop():
                    extraction_manager.stop_extraction()
                    return
                
                if timeout is not None and time.time() - start_time > timeout:
                    log_callback("⚠️ Chapter extraction timeout")
                    extraction_manager.stop_extraction()
                    return
                
                time.sleep(0.1)  # Check every 100ms
            
            # Check if extraction was successful
            if not extraction_result["result"] or not extraction_result["result"].get("success"):
                log_callback("❌ Chapter extraction failed")
                return
            
            # Load the extracted data
            metadata_path = os.path.join(out, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            else:
                metadata = extraction_result["result"].get("metadata", {})
            
            # The async extraction should have saved chapters directly, similar to the sync version
            # We need to reconstruct the chapters list with body content
            
            # Check if the extraction actually created a chapters.json file with full content
            chapters_full_path = os.path.join(out, "chapters_full.json")
            chapters_info_path = os.path.join(out, "chapters_info.json") 
            
            chapters = []
            
            # First try to load full chapters if saved
            if os.path.exists(chapters_full_path):
                log_callback("Loading full chapters data...")
                with open(chapters_full_path, 'r', encoding='utf-8') as f:
                    chapters = json.load(f)
                log_callback(f"✅ Loaded {len(chapters)} chapters with content")
                    
            elif os.path.exists(chapters_info_path):
                # Fall back to loading from individual files
                log_callback("Loading chapter info and searching for content files...")
                with open(chapters_info_path, 'r', encoding='utf-8') as f:
                    chapters_info = json.load(f)
                
                # List all files in the output directory
                all_files = os.listdir(out)
                log_callback(f"Found {len(all_files)} files in output directory")
                
                # Try to match chapter files
                for info in chapters_info:
                    chapter_num = info['num']
                    found = False
                    
                    # Try different naming patterns
                    patterns = [
                        f"chapter_{chapter_num:04d}_",  # With leading zeros
                        f"chapter_{chapter_num}_",       # Without leading zeros  
                        f"ch{chapter_num:04d}_",         # Shortened with zeros
                        f"ch{chapter_num}_",             # Shortened without zeros
                        f"{chapter_num:04d}_",          # Just number with zeros
                        f"{chapter_num}_"                # Just number
                    ]
                    
                    for pattern in patterns:
                        # Find files matching this pattern (any extension)
                        matching_files = [f for f in all_files if f.startswith(pattern)]
                        
                        if matching_files:
                            # Prefer HTML/XHTML files
                            html_files = [f for f in matching_files if f.endswith(('.html', '.xhtml', '.htm'))]
                            if html_files:
                                chapter_file = html_files[0]
                            else:
                                chapter_file = matching_files[0]
                            
                            chapter_path = os.path.join(out, chapter_file)
                            
                            try:
                                with open(chapter_path, 'r', encoding='utf-8') as f:
                                    content = f.read()
                                
                                chapters.append({
                                    "num": chapter_num,
                                    "title": info.get("title", f"Chapter {chapter_num}"),
                                    "body": content,
                                    "filename": info.get("original_filename", ""),
                                    "has_images": info.get("has_images", False),
                                    "file_size": len(content),
                                    "content_hash": info.get("content_hash", "")
                                })
                                found = True
                                break
                            except Exception as e:
                                log_callback(f"⚠️ Error reading {chapter_file}: {e}")
                    
                    if not found:
                        log_callback(f"⚠️ No file found for Chapter {chapter_num}")
                        # Log available files for debugging
                        if len(all_files) < 50:
                            similar_files = [f for f in all_files if str(chapter_num) in f]
                            if similar_files:
                                log_callback(f"   Similar files: {similar_files[:3]}")
            
            if not chapters:
                log_callback("❌ No chapters could be loaded!")
                log_callback(f"❌ Output directory: {out}")
                log_callback(f"❌ Files in directory: {len(os.listdir(out))} files")
                # Show first few files for debugging
                sample_files = os.listdir(out)[:10]
                log_callback(f"❌ Sample files: {sample_files}")
                return
            
            # Sort chapters by OPF spine order if available
            opf_path = os.path.join(out, 'content.opf')
            if os.path.exists(opf_path) and chapters:
                log_callback("📋 Sorting chapters according to OPF spine order...")
                # Call module-level function directly
                chapters = Chapter_Extractor._sort_by_opf_spine(chapters, opf_path)
                log_callback("✅ Chapters sorted according to OPF reading order")
        else:
            print("🚀 Using comprehensive chapter extraction with resource handling...")
            with zipfile.ZipFile(input_path, 'r') as zf:
                metadata = Chapter_Extractor._extract_epub_metadata(zf)
                chapters = Chapter_Extractor.extract_chapters(zf, out, progress_callback=chapter_progress_callback)

            print(f"\n📚 Extraction Summary:")
            print(f"   Total chapters extracted: {len(chapters)}")
            if chapters:
                nums = [c.get('num', 0) for c in chapters]
                print(f"   Chapter range: {min(nums)} to {max(nums)}")
                
                # Check for gaps in the sequence
                expected_count = max(nums) - min(nums) + 1
                if len(chapters) < expected_count:
                    print(f"\n⚠️ Potential missing chapters detected:")
                    print(f"   Expected {expected_count} chapters (from {min(nums)} to {max(nums)})")
                    print(f"   Actually found: {len(chapters)} chapters")
                    print(f"   Potentially missing: {expected_count - len(chapters)} chapters")         

            validate_chapter_continuity(chapters)
        
        print("\n" + "="*50)
        validate_epub_structure(out)
        print("="*50 + "\n")
    
    progress_manager.migrate_to_content_hash(chapters)
    progress_manager.save()

    if check_stop():
        return

    metadata_path = os.path.join(out, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as mf:
            metadata = json.load(mf)

    metadata["chapter_count"] = len(chapters)
    metadata["chapter_titles"] = {str(c["num"]): c["title"] for c in chapters}

    print(f"[DEBUG] Initializing client with model = {config.MODEL}")
    client = UnifiedClient(api_key=config.API_KEY, model=config.MODEL, output_dir=out)
    if hasattr(client, 'use_multi_keys') and client.use_multi_keys:
        stats = client.get_stats()
        print(f"🔑 Multi-key mode active: {stats.get('total_keys', 0)} keys loaded")
        print(f"   Active keys: {stats.get('active_keys', 0)}")
    else:
        print(f"🔑 Single-key mode: Using {config.MODEL}")
    
    # Reset cleanup state when starting new translation
    if hasattr(client, 'reset_cleanup_state'):
        client.reset_cleanup_state()
        
    if "title" in metadata and config.TRANSLATE_BOOK_TITLE and not metadata.get("title_translated", False):
        original_title = metadata["title"]
        print(f"📚 Original title: {original_title}")
        
        if not check_stop():
            translated_title = translate_title(
                original_title, 
                client, 
                None,
                None,
                config.TEMP
            )
            
            metadata["original_title"] = original_title
            metadata["title"] = translated_title
            metadata["title_translated"] = True
            
            print(f"📚 Translated title: {translated_title}")
        else:
            print("❌ Title translation skipped due to stop request")
            
    # Translate other metadata fields if configured
    translate_metadata_fields_str = os.getenv('TRANSLATE_METADATA_FIELDS', '{}')
    metadata_translation_mode = os.getenv('METADATA_TRANSLATION_MODE', 'together')

    try:
        translate_metadata_fields = json.loads(translate_metadata_fields_str)
        
        if translate_metadata_fields and any(translate_metadata_fields.values()):
            # Filter out fields that should be translated (excluding already translated fields)
            fields_to_translate = {}
            skipped_fields = []
            
            for field_name, should_translate in translate_metadata_fields.items():
                if should_translate and field_name != 'title' and field_name in metadata:
                    # Check if already translated
                    if metadata.get(f"{field_name}_translated", False):
                        skipped_fields.append(field_name)
                        print(f"✓ Skipping {field_name} - already translated")
                    else:
                        fields_to_translate[field_name] = should_translate
            
            if fields_to_translate:
                print("\n" + "="*50)
                print("📋 METADATA TRANSLATION PHASE")
                print("="*50)
                print(f"🌐 Translating {len(fields_to_translate)} metadata fields...")
                
                # Get metadata system prompt from environment
                system_prompt = os.getenv('METADATA_SYSTEM_PROMPT', '')
                if system_prompt:
                    # Get field-specific prompts
                    field_prompts_str = os.getenv('METADATA_FIELD_PROMPTS', '{}')
                    try:
                        field_prompts = json.loads(field_prompts_str)
                    except:
                        field_prompts = {}
                    
                    if not field_prompts and not field_prompts.get('_default'):
                        print("❌ No field prompts configured, skipping metadata translation")
                    else:
                        # Get language configuration
                        lang_behavior = os.getenv('LANG_PROMPT_BEHAVIOR', 'auto')
                        forced_source_lang = os.getenv('FORCED_SOURCE_LANG', 'Korean')
                        output_language = os.getenv('OUTPUT_LANGUAGE', 'English')
                        
                        # Determine source language
                        source_lang = metadata.get('language', '').lower()
                        if lang_behavior == 'never':
                            lang_str = ""
                        elif lang_behavior == 'always':
                            lang_str = forced_source_lang
                        else:  # auto
                            if 'zh' in source_lang or 'chinese' in source_lang:
                                lang_str = 'Chinese'
                            elif 'ja' in source_lang or 'japanese' in source_lang:
                                lang_str = 'Japanese'
                            elif 'ko' in source_lang or 'korean' in source_lang:
                                lang_str = 'Korean'
                            else:
                                lang_str = ''
                        
                        # Check if batch translation is enabled for parallel processing
                        batch_translate_enabled = os.getenv('BATCH_TRANSLATION', '0') == '1'
                        batch_size = int(os.getenv('BATCH_SIZE', '50'))  # Default batch size
                        
                        if batch_translate_enabled and len(fields_to_translate) > 1:
                            print(f"⚡ Using parallel metadata translation mode ({len(fields_to_translate)} fields, batch size: {batch_size})...")
                            
                            # Import ThreadPoolExecutor for parallel processing
                            from concurrent.futures import ThreadPoolExecutor, as_completed
                            import threading
                            
                            # Thread-safe results storage
                            translation_results = {}
                            results_lock = threading.Lock()
                            
                            def translate_metadata_field(field_name, original_value):
                                """Translate a single metadata field"""
                                try:
                                    print(f"\n📋 Translating {field_name}: {original_value[:100]}..." 
                                          if len(str(original_value)) > 100 else f"\n📋 Translating {field_name}: {original_value}")
                                    
                                    # Get field-specific prompt
                                    prompt_template = field_prompts.get(field_name, field_prompts.get('_default', ''))
                                    
                                    if not prompt_template:
                                        print(f"⚠️ No prompt configured for field '{field_name}', skipping")
                                        return None
                                    
                                    # Replace variables in prompt
                                    field_prompt = prompt_template.replace('{source_lang}', lang_str)
                                    field_prompt = field_prompt.replace('{output_lang}', output_language)
                                    field_prompt = field_prompt.replace('{target_lang}', output_language)
                                    field_prompt = field_prompt.replace('{field_value}', str(original_value))
                                    
                                    # Check if we're using a translation service (not AI)
                                    client_type = getattr(client, 'client_type', '')
                                    is_translation_service = client_type in ['deepl', 'google_translate']
                                    
                                    if is_translation_service:
                                        # For translation services, send only the field value without AI prompts
                                        print(f"🌐 Using translation service ({client_type}) - sending field directly")
                                        messages = [
                                            {"role": "user", "content": str(original_value)}
                                        ]
                                    else:
                                        # For AI services, use prompts as before
                                        messages = [
                                            {"role": "system", "content": system_prompt},
                                            {"role": "user", "content": f"{field_prompt}\n\n{original_value}"}
                                        ]
                                    
                                    # Add delay for rate limiting
                                    if config.DELAY > 0:
                                        time.sleep(config.DELAY)
                                    
                                    # Make API call
                                    content, finish_reason = client.send(
                                        messages, 
                                        temperature=config.TEMP,
                                        max_tokens=config.MAX_OUTPUT_TOKENS
                                    )
                                    translated_value = content.strip()
                                    
                                    # Store result thread-safely
                                    with results_lock:
                                        translation_results[field_name] = {
                                            'original': original_value,
                                            'translated': translated_value,
                                            'success': True
                                        }
                                    
                                    print(f"✅ Translated {field_name}: {translated_value}")
                                    return translated_value
                                    
                                except Exception as e:
                                    print(f"❌ Failed to translate {field_name}: {e}")
                                    with results_lock:
                                        translation_results[field_name] = {
                                            'original': original_value,
                                            'translated': None,
                                            'success': False,
                                            'error': str(e)
                                        }
                                    return None
                            
                            # Execute parallel translations with limited workers
                            max_workers = min(len(fields_to_translate), batch_size)
                            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                                # Submit all translation tasks
                                futures = {}
                                for field_name in fields_to_translate:
                                    if field_name in metadata and not check_stop():
                                        original_value = metadata[field_name]
                                        future = executor.submit(translate_metadata_field, field_name, original_value)
                                        futures[future] = field_name
                                
                                # Wait for completion
                                for future in as_completed(futures):
                                    if check_stop():
                                        print("❌ Metadata translation stopped by user")
                                        break
                            
                            # Apply results to metadata
                            for field_name, result in translation_results.items():
                                if result['success'] and result['translated']:
                                    metadata[f"original_{field_name}"] = result['original']
                                    metadata[field_name] = result['translated']
                                    metadata[f"{field_name}_translated"] = True
                        
                        else:
                            # Sequential translation mode (individual translation)
                            mode_desc = "sequential" if not batch_translate_enabled else "sequential (single field)"
                            print(f"📝 Using {mode_desc} translation mode...")
                            
                            for field_name in fields_to_translate:
                                if not check_stop() and field_name in metadata:
                                    original_value = metadata[field_name]
                                    print(f"\n📋 Translating {field_name}: {original_value[:100]}..." 
                                          if len(str(original_value)) > 100 else f"\n📋 Translating {field_name}: {original_value}")
                                    
                                    # Get field-specific prompt
                                    prompt_template = field_prompts.get(field_name, field_prompts.get('_default', ''))
                                    
                                    if not prompt_template:
                                        print(f"⚠️ No prompt configured for field '{field_name}', skipping")
                                        continue
                                    
                                    # Replace variables in prompt
                                    field_prompt = prompt_template.replace('{source_lang}', lang_str)
                                    field_prompt = field_prompt.replace('{output_lang}', output_language)
                                    field_prompt = field_prompt.replace('{target_lang}', output_language)
                                    field_prompt = field_prompt.replace('{field_value}', str(original_value))
                                    
                                    # Check if we're using a translation service (not AI)
                                    client_type = getattr(client, 'client_type', '')
                                    is_translation_service = client_type in ['deepl', 'google_translate']
                                    
                                    if is_translation_service:
                                        # For translation services, send only the field value without AI prompts
                                        print(f"🌐 Using translation service ({client_type}) - sending field directly")
                                        messages = [
                                            {"role": "user", "content": str(original_value)}
                                        ]
                                    else:
                                        # For AI services, use prompts as before
                                        messages = [
                                            {"role": "system", "content": system_prompt},
                                            {"role": "user", "content": f"{field_prompt}\n\n{original_value}"}
                                        ]
                                    
                                    try:
                                        # Add delay using the config instance from main()
                                        if config.DELAY > 0:  # ✅ FIXED - use config.DELAY instead of config.SEND_INTERVAL
                                            time.sleep(config.DELAY)
                                        
                                        # Use the same client instance from main()
                                        # ✅ FIXED - Properly unpack tuple response and provide max_tokens
                                        content, finish_reason = client.send(
                                            messages, 
                                            temperature=config.TEMP,
                                            max_tokens=config.MAX_OUTPUT_TOKENS  # ✅ FIXED - provide max_tokens to avoid NoneType error
                                        )
                                        translated_value = content.strip()  # ✅ FIXED - use content from unpacked tuple
                                        
                                        metadata[f"original_{field_name}"] = original_value
                                        metadata[field_name] = translated_value
                                        metadata[f"{field_name}_translated"] = True
                                        
                                        print(f"✅ Translated {field_name}: {translated_value}")
                                        
                                    except Exception as e:
                                        print(f"❌ Failed to translate {field_name}: {e}")

                                else:
                                    if check_stop():
                                        print("❌ Metadata translation stopped by user")
                                        break
            else:
                print("📋 No additional metadata fields to translate")
                
    except Exception as e:
        print(f"⚠️ Error processing metadata translation settings: {e}")
        import traceback
        traceback.print_exc()
    
    with open(metadata_path, 'w', encoding='utf-8') as mf:
        json.dump(metadata, mf, ensure_ascii=False, indent=2)
    print(f"💾 Saved metadata with {'translated' if metadata.get('title_translated', False) else 'original'} title")
        
    print("\n" + "="*50)
    print("📑 GLOSSARY GENERATION PHASE")
    print("="*50)
    
    # Skip glossary generation for CSV/JSON/MD files (they are typically glossaries themselves)
    if input_path.lower().endswith(('.csv', '.json', '.md')):
        print("📑 Skipping glossary generation for CSV/JSON/MD file")
        print("   CSV/JSON/MD files are treated as plain text and typically don't need glossaries")
    else:
        print(f"📑 DEBUG: ENABLE_AUTO_GLOSSARY = '{os.getenv('ENABLE_AUTO_GLOSSARY', 'NOT SET')}'")
        print(f"📑 DEBUG: MANUAL_GLOSSARY = '{config.MANUAL_GLOSSARY}'")
        print(f"📑 DEBUG: Manual glossary exists? {os.path.isfile(config.MANUAL_GLOSSARY) if config.MANUAL_GLOSSARY else False}")
        print(f"📑 DEBUG: APPEND_GLOSSARY = '{os.getenv('APPEND_GLOSSARY', '1')}'")
        print(f"📑 DEBUG: APPEND_GLOSSARY_PROMPT = '{os.getenv('APPEND_GLOSSARY_PROMPT', 'NOT SET')}'")
        print(f"📑 DEBUG: Duplicate algorithm = '{os.getenv('GLOSSARY_DUPLICATE_ALGORITHM', 'auto')}'")
        print(f"📑 DEBUG: Fuzzy threshold = '{os.getenv('GLOSSARY_FUZZY_THRESHOLD', '0.90')}'")
        print(f"📑 DEBUG: Include gender context = '{os.getenv('GLOSSARY_INCLUDE_GENDER_CONTEXT', '0')}'")
        print(f"📑 DEBUG: Context window size = '{os.getenv('GLOSSARY_CONTEXT_WINDOW', '2')}'")
        print(f"📑 DEBUG: Min frequency = '{os.getenv('GLOSSARY_MIN_FREQUENCY', '1')}'")
        print(f"📑 DEBUG: Max names = '{os.getenv('GLOSSARY_MAX_NAMES', '50')}'")
        print(f"📑 DEBUG: Max titles = '{os.getenv('GLOSSARY_MAX_TITLES', '50')}'")
        print(f"📑 DEBUG: Translation batch = '{os.getenv('GLOSSARY_BATCH_SIZE', '50')}'")
        print(f"📑 DEBUG: Max text size = '{os.getenv('GLOSSARY_MAX_TEXT_SIZE', '50000')}'")
        print(f"📑 DEBUG: Max sentences = '{os.getenv('GLOSSARY_MAX_SENTENCES', '200')}'")
        print(f"📑 DEBUG: Use smart filter = '{os.getenv('GLOSSARY_USE_SMART_FILTER', '1')}'")
        print(f"📑 DEBUG: Chapter split threshold = '{os.getenv('GLOSSARY_CHAPTER_SPLIT_THRESHOLD', '50000')}'")
        print(f"📑 DEBUG: Target language = '{os.getenv('GLOSSARY_TARGET_LANGUAGE', 'English')}'")
        
        # Check if glossary.csv already exists in the source folder
        existing_glossary_csv = os.path.join(out, "glossary.csv")
        existing_glossary_json = os.path.join(out, "glossary.json")
        print(f"📑 DEBUG: Existing glossary.csv? {os.path.exists(existing_glossary_csv)}")
        print(f"📑 DEBUG: Existing glossary.json? {os.path.exists(existing_glossary_json)}")

        def _nonempty(path):
            try:
                return os.path.getsize(path) > 0
            except Exception:
                return False

        def _has_glossary_data(path):
            """Return True only if the glossary file contains at least one entry."""
            try:
                ext = os.path.splitext(path)[1].lower()
                if ext in [".csv", ".txt", ".md"]:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = [line for line in f.readlines() if line.strip()]
                    # Require at least one non-header data line
                    return len(lines) > 1
                if ext == ".json":
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        if "entries" in data and isinstance(data["entries"], dict):
                            return len(data["entries"]) > 0
                        return len(data) > 0
                    if isinstance(data, list):
                        return len(data) > 0
                # Unknown extension: fallback to non-empty size check
                return _nonempty(path)
            except Exception:
                return False

        # If manual glossary is present but empty/header-only, clear it so auto-gen can run
        if config.MANUAL_GLOSSARY and os.path.isfile(config.MANUAL_GLOSSARY) and not _has_glossary_data(config.MANUAL_GLOSSARY):
            print("📑 Manual glossary is empty; ignoring to allow automatic generation.")
            config.MANUAL_GLOSSARY = ""
            os.environ.pop("MANUAL_GLOSSARY", None)

        if config.MANUAL_GLOSSARY and os.path.isfile(config.MANUAL_GLOSSARY) and _has_glossary_data(config.MANUAL_GLOSSARY):
            ext = os.path.splitext(config.MANUAL_GLOSSARY)[1].lower()
            # Treat .txt and .md files as CSV format (keep original extension)
            if ext in [".csv", ".txt"]:
                target_name = "glossary.csv"
            elif ext == ".md":
                target_name = "glossary.md"
            elif ext == ".json":
                target_name = "glossary.json"
            else:
                # Default to CSV for unknown extensions
                target_name = "glossary.csv"
                print(f"⚠️ Unknown glossary extension '{ext}', treating as CSV")
            
            target_path = os.path.join(out, target_name)
            if os.path.abspath(config.MANUAL_GLOSSARY) != os.path.abspath(target_path):
                shutil.copy(config.MANUAL_GLOSSARY, target_path)
                print("📑 Using manual glossary from:", config.MANUAL_GLOSSARY)
            else:
                print("📑 Using existing glossary:", config.MANUAL_GLOSSARY)
            
            # Copy glossary extension if configured
            if os.getenv('ADD_ADDITIONAL_GLOSSARY', '0') == '1':
                additional_glossary_path = os.getenv('ADDITIONAL_GLOSSARY_PATH', '')
                if additional_glossary_path and os.path.exists(additional_glossary_path):
                    # Preserve original extension
                    ext = os.path.splitext(additional_glossary_path)[1]
                    additional_target = os.path.join(out, f"glossary_extension{ext}")
                    # Only copy if target doesn't already exist
                    if not os.path.exists(additional_target):
                        try:
                            shutil.copy(additional_glossary_path, additional_target)
                            print(f"📑 Copied glossary extension: {os.path.basename(additional_glossary_path)}")
                        except Exception as e:
                            print(f"⚠️ Failed to copy glossary extension: {e}")
                    else:
                        print(f"📑 Using existing glossary extension in output folder")
        # If existing glossaries in output are empty, delete them so they don't block auto-gen
        if os.path.exists(existing_glossary_csv) and not _has_glossary_data(existing_glossary_csv):
            try:
                os.remove(existing_glossary_csv)
                print("📑 Removed empty glossary.csv to allow automatic generation.")
            except Exception as e:
                print(f"⚠️ Could not remove empty glossary.csv: {e}")
        if os.path.exists(existing_glossary_json) and not _has_glossary_data(existing_glossary_json):
            try:
                os.remove(existing_glossary_json)
                print("📑 Removed empty glossary.json to allow automatic generation.")
            except Exception as e:
                print(f"⚠️ Could not remove empty glossary.json: {e}")

        elif (os.path.exists(existing_glossary_csv) and _has_glossary_data(existing_glossary_csv)) or \
             (os.path.exists(existing_glossary_json) and _has_glossary_data(existing_glossary_json)):
            print("📑 Existing glossary file detected in source folder - skipping automatic generation")
            target_glossary_path = None
            if os.path.exists(existing_glossary_csv) and _has_glossary_data(existing_glossary_csv):
                print(f"📑 Using existing glossary.csv: {existing_glossary_csv}")
                target_glossary_path = existing_glossary_csv
            elif os.path.exists(existing_glossary_json) and _has_glossary_data(existing_glossary_json):
                print(f"📑 Using existing glossary.json: {existing_glossary_json}")
                target_glossary_path = existing_glossary_json
            
            # --- Check and inject book title if missing ---
            if target_glossary_path and target_glossary_path.endswith('.csv'):
                try:
                    include_title = os.getenv("GLOSSARY_INCLUDE_BOOK_TITLE", "0") == "1"
                    auto_inject = os.getenv("GLOSSARY_AUTO_INJECT_BOOK_TITLE", "0") == "1"
                    # Auto-inject applies only to already loaded existing glossary files (post-dedup context)
                    if include_title and auto_inject:
                        # Read existing content
                        with open(target_glossary_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                        
                        # Check if book entry exists
                        has_book_entry = False
                        for line in lines:
                            if line.lower().startswith("book,"):
                                has_book_entry = True
                                break
                        
                        if not has_book_entry:
                            print("📑 Checking for missing book title entry in existing glossary...")
                            # Use GlossaryManager to find/translate title
                            import GlossaryManager
                            
                            # Get raw title from input EPUB
                            epub_path_env = os.getenv("EPUB_PATH", "")
                            raw_title = GlossaryManager._extract_raw_title_from_epub(epub_path_env)
                            
                            # Get translated title from output metadata
                            trans_title = GlossaryManager._extract_translated_title_from_metadata(out)
                            
                            if raw_title or trans_title:
                                # Determine values (prefer distinct, fallback to what we have)
                                r_val = raw_title if raw_title else (trans_title if trans_title else "")
                                t_val = trans_title if trans_title else (raw_title if raw_title else "")
                                
                                # Insert book entry in token-efficient format if detected, or standard CSV
                                is_token_format = any(l.strip().startswith("Glossary Columns:") for l in lines)
                                
                                if is_token_format:
                                    # Insert into token efficient format
                                    # Find start of BOOKS section or create it at top
                                    book_lines = [
                                        f"=== BOOKS ===\n",
                                        f"* {t_val} ({r_val})\n",
                                        "\n"
                                    ]
                                    
                                    # Find where to insert (after Glossary Columns)
                                    insert_idx = 0
                                    for i, l in enumerate(lines):
                                        if l.strip().startswith("Glossary Columns:"):
                                            insert_idx = i + 2 # Skip blank line
                                            break
                                    
                                    # Check if BOOKS section already exists to avoid duplication
                                    has_books_section = any(l.strip() == "=== BOOKS ===" for l in lines)
                                    if not has_books_section:
                                        for bl in reversed(book_lines):
                                            lines.insert(insert_idx, bl)
                                else:
                                    # Standard CSV injection
                                    book_line = f"book,{r_val},{t_val},,\n"
                                    # Find insertion point (after header if present)
                                    insert_idx = 0
                                    if lines and "type," in lines[0].lower():
                                        insert_idx = 1
                                    lines.insert(insert_idx, book_line)
                                
                                # Write back
                                with open(target_glossary_path, 'w', encoding='utf-8') as f:
                                    f.writelines(lines)
                                print(f"📚 Auto-injected book title into existing glossary: {t_val} ({r_val})")
                except Exception as e:
                    print(f"⚠️ Failed to inject book title: {e}")
            # ----------------------------------------------

            # Copy glossary extension if configured
            if os.getenv('ADD_ADDITIONAL_GLOSSARY', '0') == '1':
                additional_glossary_path = os.getenv('ADDITIONAL_GLOSSARY_PATH', '')
                if additional_glossary_path and os.path.exists(additional_glossary_path):
                    # Preserve original extension
                    ext = os.path.splitext(additional_glossary_path)[1]
                    additional_target = os.path.join(out, f"glossary_extension{ext}")
                    # Only copy if target doesn't already exist
                    if not os.path.exists(additional_target):
                        try:
                            shutil.copy(additional_glossary_path, additional_target)
                            print(f"📑 Copied glossary extension: {os.path.basename(additional_glossary_path)}")
                        except Exception as e:
                            print(f"⚠️ Failed to copy glossary extension: {e}")
                    else:
                        print(f"📑 Using existing glossary extension in output folder")
        elif os.getenv("ENABLE_AUTO_GLOSSARY", "0") == "1":
            model = os.getenv("MODEL", "gpt-4")
            if is_traditional_translation_api(model):
                print("📑 Automatic glossary generation disabled")
                print(f"   {model} does not support glossary extraction")
                print("   Traditional translation APIs cannot identify character names/terms")
            else:
                print("📑 Starting automatic glossary generation...")
                try:
                    # Use the new process-safe glossary worker
                    from glossary_process_worker import generate_glossary_in_process
                    import concurrent.futures
                    import multiprocessing
                    
                    instructions = ""
                    
                    # Get extraction workers setting
                    extraction_workers = int(os.getenv("EXTRACTION_WORKERS", "1"))
                    if extraction_workers == 1:
                        # Auto-detect for better performance
                        extraction_workers = min(os.cpu_count() or 4, 4)
                        print(f"📑 Using {extraction_workers} CPU cores for glossary generation")
                    
                    # Collect environment variables to pass to subprocess
                    env_vars = {}
                    important_vars = [
                        'EXTRACTION_WORKERS', 'GLOSSARY_MIN_FREQUENCY', 'GLOSSARY_MAX_NAMES',
                        'GLOSSARY_MAX_TITLES', 'GLOSSARY_BATCH_SIZE', 'GLOSSARY_STRIP_HONORIFICS',
                        'GLOSSARY_FUZZY_THRESHOLD', 'GLOSSARY_MAX_TEXT_SIZE', 'GLOSSARY_MAX_SENTENCES',
                        'AUTO_GLOSSARY_PROMPT', 'GLOSSARY_USE_SMART_FILTER', 'GLOSSARY_USE_LEGACY_CSV',
                        'GLOSSARY_PARALLEL_ENABLED', 'GLOSSARY_FILTER_MODE', 'GLOSSARY_SKIP_FREQUENCY_CHECK',
                        'GLOSSARY_SKIP_ALL_VALIDATION', 'MODEL', 'API_KEY', 'OPENAI_API_KEY', 'GEMINI_API_KEY',
                        'MAX_OUTPUT_TOKENS', 'GLOSSARY_TEMPERATURE', 'MANUAL_GLOSSARY', 'ENABLE_AUTO_GLOSSARY',
                        'GLOSSARY_DUPLICATE_ALGORITHM', 'GLOSSARY_INCLUDE_GENDER_CONTEXT', 'GLOSSARY_CONTEXT_WINDOW',
                        'GLOSSARY_INCLUDE_BOOK_TITLE', 'EPUB_PATH'
                    ]
                    
                    for var in important_vars:
                        if var in os.environ:
                            env_vars[var] = os.environ[var]
                    
                    # Create a Queue for real-time log streaming
                    manager = multiprocessing.Manager()
                    log_queue = manager.Queue()
                    
                    # Use ProcessPoolExecutor for true parallelism (completely bypasses GIL)
                    print("📑 Starting glossary generation in separate process...")
                    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
                        # Submit to separate process WITH log queue
                        future = executor.submit(
                            generate_glossary_in_process,
                            out,
                            chapters,
                            instructions,
                            env_vars,
                            log_queue  # Pass the queue for real-time logs
                        )
                        
                        # Poll for completion and stream logs in real-time
                        poll_count = 0
                        while not future.done():
                            poll_count += 1
                            
                            # Check for logs from subprocess and print them immediately
                            try:
                                while not log_queue.empty():
                                    log_line = log_queue.get_nowait()
                                    print(log_line)  # Print to GUI
                            except:
                                pass
                            
                            # Super short sleep to yield to GUI
                            time.sleep(0.001)
                            
                            # Check for stop every 100 polls
                            if poll_count % 100 == 0:
                                if check_stop():
                                    print("📑 ❌ Glossary generation cancelled")
                                    executor.shutdown(wait=False, cancel_futures=True)
                                    return
                        
                        # Get any remaining logs from queue
                        try:
                            while not log_queue.empty():
                                log_line = log_queue.get_nowait()
                                print(log_line)
                        except:
                            pass
                        
                        # Get result
                        if future.done():
                            try:
                                result = future.result(timeout=0.1)
                                if isinstance(result, dict):
                                    if result.get('success'):
                                        print(f"📑 ✅ Glossary generation completed successfully")
                                    else:
                                        print(f"📑 ❌ Glossary generation failed: {result.get('error')}")
                                        if result.get('traceback'):
                                            print(f"📑 Error details:\n{result.get('traceback')}")
                            except Exception as e:
                                print(f"📑 ❌ Error retrieving glossary result: {e}")
                    
                    print("✅ Automatic glossary generation COMPLETED")
                    
                    # Copy glossary extension if configured (after auto-glossary generation)
                    if os.getenv('ADD_ADDITIONAL_GLOSSARY', '0') == '1':
                        additional_glossary_path = os.getenv('ADDITIONAL_GLOSSARY_PATH', '')
                        if additional_glossary_path and os.path.exists(additional_glossary_path):
                            # Preserve original extension
                            ext = os.path.splitext(additional_glossary_path)[1]
                            additional_target = os.path.join(out, f"glossary_extension{ext}")
                            # Only copy if target doesn't already exist
                            if not os.path.exists(additional_target):
                                try:
                                    shutil.copy(additional_glossary_path, additional_target)
                                    print(f"📑 Copied glossary extension: {os.path.basename(additional_glossary_path)}")
                                except Exception as e:
                                    print(f"⚠️ Failed to copy glossary extension: {e}")
                            else:
                                print(f"📑 Using existing glossary extension in output folder")
                    
                    # Handle deferred glossary appending
                    if os.getenv('DEFER_GLOSSARY_APPEND') == '1':
                        print("📑 Processing deferred glossary append to system prompt...")
                        
                        glossary_path = find_glossary_file(out)
                        if glossary_path and os.path.exists(glossary_path):
                            try:
                                glossary_block = None
                                if glossary_path.lower().endswith('.csv'):
                                    with open(glossary_path, 'r', encoding='utf-8') as f:
                                        glossary_block = f.read()
                                else:
                                    with open(glossary_path, 'r', encoding='utf-8') as f:
                                        glossary_data = json.load(f)
                                    
                                    formatted_entries = {}
                                    if isinstance(glossary_data, dict) and 'entries' in glossary_data:
                                        formatted_entries = glossary_data['entries']
                                    elif isinstance(glossary_data, dict):
                                        formatted_entries = {k: v for k, v in glossary_data.items() if k != "metadata"}
                                    
                                    if formatted_entries:
                                        glossary_block = json.dumps(formatted_entries, ensure_ascii=False, indent=2)
                                    else:
                                        glossary_block = None
                                
                                if glossary_block:
                                    glossary_prompt = os.getenv('GLOSSARY_APPEND_PROMPT', 
                                        "Character/Term Glossary (use these translations consistently):")
                                    
                                    current_prompt = config.PROMPT
                                    if current_prompt:
                                        current_prompt += "\n\n"
                                    current_prompt += f"{glossary_prompt}\n{glossary_block}"
                                    
                                    config.PROMPT = current_prompt
                                    
                                    print(f"✅ Added auto-generated glossary to system prompt ({os.path.basename(glossary_path)})")
                                    
                                    if 'DEFER_GLOSSARY_APPEND' in os.environ:
                                        del os.environ['DEFER_GLOSSARY_APPEND']
                                    if 'GLOSSARY_APPEND_PROMPT' in os.environ:
                                        del os.environ['GLOSSARY_APPEND_PROMPT']
                                else:
                                    print("⚠️ Auto-generated glossary has no entries - skipping append")
                                    if 'DEFER_GLOSSARY_APPEND' in os.environ:
                                        del os.environ['DEFER_GLOSSARY_APPEND']
                                    if 'GLOSSARY_APPEND_PROMPT' in os.environ:
                                        del os.environ['GLOSSARY_APPEND_PROMPT']
                            except Exception as e:
                                print(f"⚠️ Failed to append auto-generated glossary: {e}")
                        else:
                            print("⚠️ No glossary file found after automatic generation")
                    
                except Exception as e:
                    print(f"❌ Glossary generation failed: {e}")
        else:
            print("📑 Automatic glossary generation disabled")
        # Don't create an empty glossary - let any existing manual glossary remain

    glossary_file = find_glossary_file(out)
    # Only show glossary details if append glossary is enabled
    append_glossary_enabled = os.getenv("APPEND_GLOSSARY", "1") == "1"
    add_additional_enabled = os.getenv('ADD_ADDITIONAL_GLOSSARY', '0') == '1'
    
    if glossary_file and os.path.exists(glossary_file):
        if append_glossary_enabled:
            try:
                if glossary_file.lower().endswith(('.csv', '.txt', '.md')):
                    # Quick CSV/TXT/MD stats
                    with open(glossary_file, 'r', encoding='utf-8') as f:
                        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
                    entry_count = max(0, len(lines) - 1) if lines and ',' in lines[0] else len(lines)
                    if glossary_file.lower().endswith('.txt'):
                        file_type = "TXT"
                    elif glossary_file.lower().endswith('.md'):
                        file_type = "MD"
                    else:
                        file_type = "CSV"
                    print(f"📑 Glossary ready ({file_type}) with {entry_count} entries")
                    print("📑 Sample glossary lines:")
                    for ln in lines[1:6]:
                        print(f"   • {ln}")
                elif glossary_file.lower().endswith('.json'):
                    with open(glossary_file, 'r', encoding='utf-8') as f:
                        glossary_data = json.load(f)
                    
                    if isinstance(glossary_data, dict):
                        if 'entries' in glossary_data and isinstance(glossary_data['entries'], dict):
                            entry_count = len(glossary_data['entries'])
                            sample_items = list(glossary_data['entries'].items())[:3]
                        else:
                            entry_count = len(glossary_data)
                            sample_items = list(glossary_data.items())[:3]
                        
                        print(f"📑 Glossary ready with {entry_count} entries")
                        print("📑 Sample glossary entries:")
                        for key, value in sample_items:
                            print(f"   • {key} → {value}")
                            
                    elif isinstance(glossary_data, list):
                        print(f"📑 Glossary ready with {len(glossary_data)} entries")
                        print("📑 Sample glossary entries:")
                        for i, entry in enumerate(glossary_data[:3]):
                            if isinstance(entry, dict):
                                original = entry.get('original_name', '?')
                                translated = entry.get('name', original)
                                print(f"   • {original} → {translated}")
                    else:
                        print(f"⚠️ Unexpected glossary format: {type(glossary_data)}")
                
                # Check for glossary extension (after all glossary types)
                if add_additional_enabled:
                    # Check for extension with any supported format
                    additional_glossary = None
                    for ext in ['.csv', '.md', '.txt', '.json']:
                        candidate = os.path.join(out, f"glossary_extension{ext}")
                        if os.path.exists(candidate):
                            additional_glossary = candidate
                            break
                    
                    if additional_glossary:
                        try:
                            with open(additional_glossary, 'r', encoding='utf-8') as f:
                                add_lines = [ln.strip() for ln in f.readlines() if ln.strip()]
                            add_entry_count = max(0, len(add_lines) - 1) if add_lines and ',' in add_lines[0] else len(add_lines)
                            print(f"📑 Glossary extension loaded with {add_entry_count} entries")
                            print("📑 Sample glossary extension lines:")
                            for ln in add_lines[1:4]:
                                print(f"   • {ln}")
                        except Exception as e:
                            print(f"⚠️ Failed to read glossary extension: {e}")
                else:
                    # Check if extension file exists but toggle is disabled
                    for ext in ['.csv', '.md', '.txt', '.json']:
                        additional_glossary = os.path.join(out, f"glossary_extension{ext}")
                        if os.path.exists(additional_glossary):
                            print("⏩ Skipping glossary extension - toggle disabled")
                            break
                    
            except Exception as e:
                print(f"⚠️ Failed to inspect glossary file: {e}")
        else:
            print("⏩ Skipping glossary - toggle disabled")
    else:
        if append_glossary_enabled:
            print("📑 No glossary file found")

    print("="*50)
    print("🚀 STARTING MAIN TRANSLATION PHASE")
    print("="*50 + "\n")

    glossary_path = find_glossary_file(out)
    if glossary_path and os.path.exists(glossary_path) and glossary_path.lower().endswith('.json'):
        try:
            with open(glossary_path, 'r', encoding='utf-8') as f:
                g_data = json.load(f)
            
            print(f"[DEBUG] Glossary type before translation: {type(g_data)}")
            if isinstance(g_data, list):
                print(f"[DEBUG] Glossary is a list")
        except Exception as e:
            print(f"[DEBUG] Error checking glossary: {e}")
    glossary_path = find_glossary_file(out)
    # Build system prompt without glossary compression initially
    # Compression will happen per-chapter when enabled
    system = build_system_prompt(config.SYSTEM_PROMPT, glossary_path, source_text=None)
    base_msg = [{"role": "system", "content": system}]
    # Preserve the original system prompt to avoid in-place mutations
    original_system_prompt = system
    last_summary_block_text = None  # Will hold the last rolling summary text for the NEXT chapter only
    last_summary_chapter_num = None  # Chapter number associated with last_summary_block_text
    
    image_translator = None

    if config.ENABLE_IMAGE_TRANSLATION:
        print(f"🖼️ Image translation enabled for model: {config.MODEL}")
        print("🖼️ Image translation will use your custom system prompt and glossary")
        image_translator = ImageTranslator(
            client, 
            out, 
            config.PROFILE_NAME, 
            system, 
            config.TEMP,
            log_callback ,
            progress_manager,
            history_manager,
            chunk_context_manager
        )
        
        known_vision_models = [
            'gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-2.0-flash', 'gemini-2.5-flash', 'gemini-2.5-pro',
            'gpt-4-turbo', 'gpt-4o', 'gpt-4.1-mini', 'gpt-4.1-nano', 'o4-mini', 'gpt-4.1-mini' 'gemini-3-pro-image-preview',
        ]
        
        if config.MODEL.lower() not in known_vision_models:
            print(f"⚠️ Note: {config.MODEL} may not have vision capabilities. Image translation will be attempted anyway.")
    else:
        print("ℹ️ Image translation disabled by user")
    
    total_chapters = len(chapters)

    # Only detect numbering if the toggle is not disabled
    if config.DISABLE_ZERO_DETECTION:
        print(f"📊 0-based detection disabled by user setting")
        uses_zero_based = False
        # Important: Set a flag that can be checked throughout the codebase
        config._force_disable_zero_detection = True
    else:
        if chapters:
            uses_zero_based = detect_novel_numbering(chapters)
            print(f"📊 Novel numbering detected: {'0-based' if uses_zero_based else '1-based'}")
        else:
            uses_zero_based = False
        config._force_disable_zero_detection = False

    # Store this for later use
    config._uses_zero_based = uses_zero_based


    rng = os.getenv("CHAPTER_RANGE", "")
    start = None
    end = None
    if rng and re.match(r"^\d+\s*-\s*\d+$", rng):
            start, end = map(int, rng.split("-", 1))
            
            if config.DISABLE_ZERO_DETECTION:
                print(f"📊 0-based detection disabled - using range as specified: {start}-{end}")
            elif uses_zero_based:
                print(f"📊 0-based novel detected")
                print(f"📊 User range {start}-{end} will be used as-is (chapters are already adjusted)")
            else:
                print(f"📊 1-based novel detected")
                print(f"📊 Using range as specified: {start}-{end}")
    
    print("📊 Calculating total chunks needed...")
    total_chunks_needed = 0
    chunks_per_chapter = {}
    chapters_to_process = 0
    
    # Check if special files translation is disabled
    translate_special = os.getenv('TRANSLATE_SPECIAL_FILES', '0') == '1'

    # Helper: sequential numbering with zero-phase.
    # Start at 0; only start incrementing once a digit >0 is seen in the filename.
    def _assign_chapter_num(name_noext, seq_counter, zero_phase):
        nums = re.findall(r'\d+', name_noext) if name_noext else []
        has_gt_zero = any(int(n) > 0 for n in nums)
        if zero_phase:
            if has_gt_zero:
                # first positive digit: begin incrementing
                if seq_counter == 0:
                    seq_counter = 1
                num = seq_counter
                seq_counter += 1
                zero_phase = False
            else:
                # still zero phase
                num = 0
        else:
            # already incrementing
            num = seq_counter
            seq_counter += 1
        return num, seq_counter, zero_phase

    # When setting actual chapter numbers (in the main function)
    seq_counter = 0
    zero_phase = True
    for idx, c in enumerate(chapters):
        chap_num = c["num"]
        content_hash = c.get("content_hash") or ContentProcessor.get_content_hash(c["body"])
        
        # Extract the raw chapter number from the file
        raw_num = FileUtilities.extract_actual_chapter_number(c, patterns=None, config=config)
        #print(f"[DEBUG] Extracted raw_num={raw_num} from {c.get('original_basename', 'unknown')}")
        # Spine position (reading order) fallback
        spine_pos = c.get('spine_order')
        if spine_pos is None:
            spine_pos = c.get('opf_spine_position')
        if spine_pos is None:
            spine_pos = idx  # ultimate fallback to list order

        # Normalize chapter number using extracted number (spine/file aware)
        normalized_num = raw_num if raw_num is not None else 0
        offset = config.CHAPTER_NUMBER_OFFSET if hasattr(config, 'CHAPTER_NUMBER_OFFSET') else 0
        raw_num = normalized_num + offset
        
        # When toggle is disabled, use raw numbers without any 0-based adjustment
        if config.DISABLE_ZERO_DETECTION:
            c['actual_chapter_num'] = raw_num
            # Store raw number for consistency
            c['raw_chapter_num'] = raw_num
            c['zero_adjusted'] = False
        else:
            # Store raw number
            c['raw_chapter_num'] = raw_num
            # Apply adjustment only if this is a 0-based novel
            if uses_zero_based:
                c['actual_chapter_num'] = raw_num + 1
                c['zero_adjusted'] = True
            else:
                c['actual_chapter_num'] = raw_num
                c['zero_adjusted'] = False
        
        # Now we can safely use actual_num
        actual_num = c['actual_chapter_num']
        
        # Skip special files (chapter 0) if translation is disabled
        # IMPORTANT: Do NOT treat files with digits (including 0) in their name as special.
        if not translate_special and raw_num == 0:
            name = c.get('original_basename') or os.path.basename(c.get('filename', ''))
            name_noext = os.path.splitext(name)[0] if name else ''
            has_digits_in_name = bool(re.search(r'\d', name_noext))
            if not has_digits_in_name:
                # Track skipped special files
                if not hasattr(config, '_skipped_special_files'):
                    config._skipped_special_files = []
                config._skipped_special_files.append(c.get('original_basename', f'Chapter {actual_num}'))
                chunks_per_chapter[idx] = 0
                continue

        if start is not None:
            if not (start <= c['actual_chapter_num'] <= end):
                # Track skipped chapters for summary (don't print individually)
                if not hasattr(config, '_range_skipped_chapters'):
                    config._range_skipped_chapters = []
                config._range_skipped_chapters.append(c['actual_chapter_num'])
                continue
                
        # IMPORTANT: pass chapter_obj so ProgressManager can resolve composite keys
        # (e.g. when multiple spine items share the same chapter number).
        needs_translation, skip_reason, _ = progress_manager.check_chapter_status(
            idx, actual_num, content_hash, out, chapter_obj=c
        )
        
        if not needs_translation:
            chunks_per_chapter[idx] = 0
            continue
        
        chapters_to_process += 1
        
        chapter_key = str(actual_num)
        if chapter_key in progress_manager.prog["chapters"] and progress_manager.prog["chapters"][chapter_key].get("status") == "in_progress":
            pass
        
        # Calculate based on effective OUTPUT limit only
        max_output_tokens = config.get_effective_output_limit() 
        safety_margin_output = 500
        
        # Korean to English typically compresses to 0.7-0.9x
        compression_factor = config.COMPRESSION_FACTOR
        available_tokens = int((max_output_tokens - safety_margin_output) / compression_factor)
        
        # Ensure minimum
        available_tokens = max(available_tokens, 1000)
        
        # Debug output for first chapter
        if os.getenv('DEBUG_CHUNK_SPLITTING', '0') == '1' and idx == 0:
            print(f"\n[CHUNK CALC DEBUG] Configuration:")
            print(f"  MAX_OUTPUT_TOKENS: {max_output_tokens:,}")
            print(f"  safety_margin_output: {safety_margin_output:,}")
            print(f"  COMPRESSION_FACTOR: {compression_factor}")
            print(f"  Calculated available_tokens: {available_tokens:,}")
            print(f"  Formula: ({max_output_tokens:,} - {safety_margin_output:,}) / {compression_factor} = {available_tokens:,}\n")
        
        #print(f"📊 Chunk size: {available_tokens:,} tokens (based on {max_output_tokens:,} output limit, compression: {compression_factor})")
        
        # For mixed content chapters, calculate on clean text
        # For mixed content chapters, calculate on clean text
        # Get filename for content type detection (prefer source_file to detect PDF context)
        chapter_filename = c.get('source_file') or c.get('filename') or c.get('original_basename', '')
        
        if c.get('has_images', False) and ContentProcessor.is_meaningful_text_content(c["body"]):
            # Don't modify c["body"] at all during chunk calculation
            # Just pass the body as-is, the chunking will be slightly off but that's OK
            chunks = chapter_splitter.split_chapter(c["body"], available_tokens, filename=chapter_filename)
        else:
            chunks = chapter_splitter.split_chapter(c["body"], available_tokens, filename=chapter_filename)
        
        chapter_key_str = content_hash
        old_key_str = str(idx)

        if chapter_key_str not in progress_manager.prog.get("chapter_chunks", {}) and old_key_str in progress_manager.prog.get("chapter_chunks", {}):
            progress_manager.prog["chapter_chunks"][chapter_key_str] = progress_manager.prog["chapter_chunks"][old_key_str]
            del progress_manager.prog["chapter_chunks"][old_key_str]
            #print(f"[PROGRESS] Migrated chunks for chapter {actual_num} to new tracking system")

        # Always count actual chunks - ignore "completed" tracking
        chunks_per_chapter[idx] = len(chunks)
        total_chunks_needed += chunks_per_chapter[idx]
            
    # Print range skip summary if any chapters were skipped
    if hasattr(config, '_range_skipped_chapters') and config._range_skipped_chapters:
        skipped = config._range_skipped_chapters
        print(f"📊 Skipped {len(skipped)} chapters outside range {start}-{end}")
        if len(skipped) <= 10:
            print(f"   Skipped: {', '.join(map(str, sorted(skipped)))}")
        else:
            print(f"   Range: {min(skipped)} to {max(skipped)}")
    
    # Print special files skip summary
    if hasattr(config, '_skipped_special_files') and config._skipped_special_files:
        skipped = config._skipped_special_files
        print(f"📊 Skipped {len(skipped)} special file(s) (TRANSLATE_SPECIAL_FILES is disabled)")
        if len(skipped) <= 5:
            for file in skipped:
                print(f"   • {file}")
    
    # Check if no chapters will be processed and provide helpful error
    if chapters_to_process == 0:
        if start is not None and end is not None:
            # Get actual chapter range available
            if chapters:
                available_chapters = [c.get('actual_chapter_num', c['num']) for c in chapters]
                min_chapter = min(available_chapters)
                max_chapter = max(available_chapters)
                
                print(f"\n❌ ERROR: Chapter range {start}-{end} doesn't match any chapters!")
                print(f"📚 Available chapters in this EPUB: {min_chapter}-{max_chapter} ({len(chapters)} total)")
                print(f"💡 Please adjust your chapter range in the settings to match the available chapters.")
                
                if hasattr(config, '_range_skipped_chapters') and config._range_skipped_chapters:
                    print(f"\n📊 All {len(config._range_skipped_chapters)} chapters were outside the specified range.")
            else:
                print(f"\n❌ ERROR: No chapters found in EPUB to translate!")
            
            raise ValueError(f"Chapter range {start}-{end} doesn't match any available chapters ({min_chapter}-{max_chapter})")
        elif not translate_special and total_chapters > 0:
            print(f"\n⚠️ WARNING: All chapters are special files (chapter 0) and TRANSLATE_SPECIAL_FILES is disabled.")
            print(f"💡 Enable 'Translate Special Files' in settings if you want to translate these files.")
        elif total_chunks_needed == 0 and total_chapters > 0:
            print(f"\n✅ All chapters already translated - nothing to do!")
        else:
            print(f"\n❌ ERROR: No chapters to process!")
    
    terminology = "Sections" if is_text_file else "Chapters"
    print(f"📊 Total chunks to translate: {total_chunks_needed}")
    print(f"📚 {terminology} to process: {chapters_to_process}")
    
    multi_chunk_chapters = [(idx, count) for idx, count in chunks_per_chapter.items() if count > 1]
    if multi_chunk_chapters:
        # Determine terminology based on file type
        terminology = "Sections" if is_text_file else "Chapters"
        print(f"📄 {terminology} requiring multiple chunks:")
        for idx, chunk_count in multi_chunk_chapters:
            chap = chapters[idx]
            section_term = "Section" if is_text_file else "Chapter"
            print(f"   • {section_term} {idx+1} ({chap['title'][:30]}...): {chunk_count} chunks")
    
    translation_start_time = time.time()
    chunks_completed = 0
    chapters_completed = 0
    
    current_chunk_number = 0

    if config.BATCH_TRANSLATION:
        # Check if request merging is enabled (for PDF and EPUB files)
        use_request_merging = config.REQUEST_MERGING_ENABLED and config.REQUEST_MERGE_COUNT > 1 and (is_pdf_file or not is_text_file)
        
        if use_request_merging:
            print(f"\n🔗 REQUEST MERGING + BATCH MODE ENABLED")
            print(f"🔗 Merging {config.REQUEST_MERGE_COUNT} chapters per API request")
            print(f"📦 Processing with up to {config.BATCH_SIZE} concurrent merged requests")
        else:
            print(f"\n📦 PARALLEL TRANSLATION MODE ENABLED")
            print(f"📦 Processing chapters with up to {config.BATCH_SIZE} concurrent API calls")
        
        import concurrent.futures
        from threading import Lock
        
        progress_lock = Lock()
        
        chapters_to_translate = []
        
        # FIX: First pass to set actual chapter numbers for ALL chapters
        # This ensures batch mode has the same chapter numbering as non-batch mode
        print("📊 Setting chapter numbers...")
        seq_counter = 0
        zero_phase = True
        for idx, c in enumerate(chapters):
            # PDF/TEXT CHUNK FIX: Skip extract_actual_chapter_number for chunks - preserve decimal from c['num']
            if is_text_file and c.get('is_chunk', False):
                # For text/PDF chunks, use the decimal number directly (1.0, 1.1, etc.)
                c['actual_chapter_num'] = c['num']
                c['raw_chapter_num'] = c['num']
                c['zero_adjusted'] = False
                continue
            
            raw_num = FileUtilities.extract_actual_chapter_number(c, patterns=None, config=config)
            raw_num = raw_num if raw_num is not None else 0

            # Apply offset if configured
            offset = config.CHAPTER_NUMBER_OFFSET if hasattr(config, 'CHAPTER_NUMBER_OFFSET') else 0
            raw_num += offset
            
            if config.DISABLE_ZERO_DETECTION:
                # Use raw numbers without adjustment
                c['actual_chapter_num'] = raw_num
                c['raw_chapter_num'] = raw_num
                c['zero_adjusted'] = False
            else:
                # Store raw number
                c['raw_chapter_num'] = raw_num
                # Apply 0-based adjustment if detected
                if uses_zero_based:
                    c['actual_chapter_num'] = raw_num + 1
                    c['zero_adjusted'] = True
                else:
                    c['actual_chapter_num'] = raw_num
                    c['zero_adjusted'] = False
        
        for idx, c in enumerate(chapters):
            chap_num = c["num"]
            content_hash = c.get("content_hash") or ContentProcessor.get_content_hash(c["body"])
            
            # Check if this is a pre-split text chunk with decimal number
            # IMPORTANT: Check is_chunk FIRST, then use c['num'] regardless of float value
            # This handles cases like 1.0 where float equals integer but should still be preserved
            if is_text_file and c.get('is_chunk', False):
                actual_num = c['num']  # Preserve the decimal for text/PDF chunks
                c['actual_chapter_num'] = actual_num  # UPDATE THE CHAPTER DICT!
            else:
                actual_num = c.get('actual_chapter_num', c['num'])  # Now this will exist!
            
            # Skip special files (chapter 0) if translation is disabled
            raw_num = c.get('raw_chapter_num', FileUtilities.extract_actual_chapter_number(c, patterns=None, config=config))
            if not translate_special and raw_num == 0:
                name = c.get('original_basename') or os.path.basename(c.get('filename', ''))
                name_noext = os.path.splitext(name)[0] if name else ''
                has_digits_in_name = bool(re.search(r'\d', name_noext))
                if not has_digits_in_name:
                    continue
            
            # Skip chapters outside the range
            if start is not None and not (start <= actual_num <= end):
                continue
            
            # Check if chapter needs translation
            needs_translation, skip_reason, existing_file = progress_manager.check_chapter_status(
                idx, actual_num, content_hash, out, c  # Pass the chapter object
            )
            # Add explicit file check for supposedly completed chapters
            if not needs_translation and existing_file:
                file_path = os.path.join(out, existing_file)
                if not os.path.exists(file_path):
                    print(f"⚠️ Output file missing for chapter {actual_num}: {existing_file}")
                    needs_translation = True
                    skip_reason = None
                    # Update status to file_missing
                    progress_manager.update(idx, actual_num, content_hash, None, status="file_missing", chapter_obj=c)
                    progress_manager.save()
            
            # -------------------------------------------------------------------------
            # BATCH PRE-PROCESSING
            # -------------------------------------------------------------------------
            if needs_translation and c.get("body"):
                batch_translate_active = os.getenv('BATCH_TRANSLATE_HEADERS', '0') == '1'
                ignore_title_tag = os.getenv('IGNORE_TITLE', '0') == '1' and batch_translate_active
                ignore_header_tags = os.getenv('IGNORE_HEADER', '0') == '1' and batch_translate_active
                
                if (ignore_title_tag or ignore_header_tags):
                    try:
                        from bs4 import BeautifulSoup
                        content_soup = BeautifulSoup(c["body"], 'html.parser')
                        modified = False
                        
                        if ignore_title_tag:
                            for title_tag in content_soup.find_all('title'):
                                title_tag.decompose()
                                modified = True
                        
                        if ignore_header_tags:
                            for header_tag in content_soup.find_all(['h1', 'h2', 'h3']):
                                header_tag.decompose()
                                modified = True
                        
                        if modified:
                            c["body"] = str(content_soup)
                    except Exception as e:
                        print(f"⚠️ Failed to filter batch content for chapter {actual_num}: {e}")
            # -------------------------------------------------------------------------
            
            if not needs_translation:
                # Track skips for summary instead of printing each one
                if not hasattr(config, '_batch_skipped_chapters'):
                    config._batch_skipped_chapters = []
                is_text_source = is_text_file or c.get('filename', '').endswith('.txt') or c.get('is_chunk', False)
                terminology = "Section" if is_text_source else "Chapter"
                config._batch_skipped_chapters.append((actual_num, terminology, skip_reason))
                chapters_completed += 1
                continue
            
            # Check for empty or image-only chapters
            has_images = c.get('has_images', False)
            has_meaningful_text = ContentProcessor.is_meaningful_text_content(c["body"])
            text_size = c.get('file_size', 0)
            
            is_empty_chapter = (not has_images and text_size < 1)
            is_image_only_chapter = (has_images and not has_meaningful_text)
            
            # Handle empty chapters
            if is_empty_chapter:
                print(f"📄 Empty chapter {chap_num} - will process individually")
                
                safe_title = make_safe_filename(c['title'], c['num'])
                
                if isinstance(c['num'], float):
                    fname = FileUtilities.create_chapter_filename(c, c['num'])
                else:
                    fname = FileUtilities.create_chapter_filename(c, c['num'])
                with open(os.path.join(out, fname), 'w', encoding='utf-8') as f:
                    f.write(c["body"])
                progress_manager.update(idx, actual_num, content_hash, fname, status="completed_empty", chapter_obj=c)
                progress_manager.save()
                chapters_completed += 1
                continue
            
            # Add to chapters to translate
            chapters_to_translate.append((idx, c))
        
        # Print skip summary for batch mode
        if hasattr(config, '_batch_skipped_chapters') and config._batch_skipped_chapters:
            skipped = config._batch_skipped_chapters
            print(f"\n📊 Skipped {len(skipped)} already completed chapters")
            if os.getenv('DEBUG_SKIP_MESSAGES', '0') == '1' and len(skipped) <= 5:
                for num, term, reason in skipped[:5]:
                    print(f"   • {term} {num}: {reason.split('(')[0].strip()}")
        
        print(f"📊 Found {len(chapters_to_translate)} chapters to translate in parallel")
        
        # Continue with the rest of the existing batch processing code...
        batch_processor = BatchTranslationProcessor(
            config, client, base_msg, out, progress_lock,
            progress_manager.save, 
            lambda idx, actual_num, content_hash, output_file=None, status="completed", **kwargs: progress_manager.update(idx, actual_num, content_hash, output_file, status, **kwargs),
            check_stop,
            image_translator,
            is_text_file=is_text_file,
            history_manager=history_manager
        )

        # Batch-mode rolling summary: updated once per batch and injected into the NEXT batch.
        rolling_summary_for_next_batch = ""  # exact rolling_summary.txt contents
        import threading
        rolling_summary_update_lock = threading.Lock()
        summary_translation_processor = None
        if config.USE_ROLLING_SUMMARY:
            # Dedicated processor for summarization between batches (no concurrency with translation threads).
            summary_translation_processor = TranslationProcessor(config, client, out, log_callback, check_stop, uses_zero_based, is_text_file)
        
        total_to_process = len(chapters_to_translate)
        processed = 0
        
        # ==========================
        # Batching mode selection
        # ==========================
        batching_mode = getattr(config, 'BATCHING_MODE', 'direct')
        batch_group_size_cfg = max(1, int(getattr(config, 'BATCH_GROUP_SIZE', 3)))
        if batching_mode not in ('direct', 'conservative', 'aggressive'):
            batching_mode = 'direct'
        # Backwards compatibility with CONSERVATIVE_BATCHING env
        if os.getenv('CONSERVATIVE_BATCHING', '0') == '1':
            batching_mode = 'conservative'
        if batching_mode == 'conservative':
            batch_group_size = config.BATCH_SIZE * batch_group_size_cfg
            print(f"📦 Using conservative batching: group size {batch_group_size} (batch size {config.BATCH_SIZE}, multiplier {batch_group_size_cfg})")
        elif batching_mode == 'direct':
            batch_group_size = config.BATCH_SIZE  # legacy behavior
            print(f"📦 Using direct batching: group size {batch_group_size}, parallel {config.BATCH_SIZE}")
        else:  # aggressive
            batch_group_size = batch_group_size_cfg  # not used for throttling, only for logging/summary grouping
            print(f"⚡ Using AGGRESSIVE batching: keeps {config.BATCH_SIZE} parallel calls, auto-refills when any finishes")
        
        # Create merge groups if request merging is enabled
        if use_request_merging:
            # Build proximity runs first (so we never merge far-apart chapters),
            # then pack each run under the token budget. This avoids patterns like
            # 2+1, 2+1 when REQUEST_MERGE_COUNT=3 but only 2 chapters fit; instead
            # we repack into 2+2, 2+2 when possible.
            proximity_runs = RequestMerger.create_merge_groups(
                chapters_to_translate,
                max(1, len(chapters_to_translate)),
            )

            max_output_tokens = config.get_effective_output_limit()
            safety_margin_output = 500
            compression_factor = getattr(config, 'COMPRESSION_FACTOR', 1.0) or 1.0
            available_tokens = int((max_output_tokens - safety_margin_output) / compression_factor)
            available_tokens = max(available_tokens, 1000)

            merge_groups = []
            for run in proximity_runs:
                if len(run) <= 1:
                    merge_groups.append(run)
                    continue

                i = 0
                while i < len(run):
                    group = [run[i]]
                    i += 1

                    # Try to grow the group up to REQUEST_MERGE_COUNT, but stop
                    # when adding the next chapter would exceed the token budget.
                    while i < len(run) and len(group) < config.REQUEST_MERGE_COUNT:
                        candidate = run[i]
                        merge_input = [
                            (ch.get('actual_chapter_num', ch['num']), ch["body"], ch)
                            for (idx, ch) in (group + [candidate])
                        ]
                        merged_preview = RequestMerger.merge_chapters(merge_input, log_injections=False)
                        merged_tokens = chapter_splitter.count_tokens(merged_preview)

                        if merged_tokens <= available_tokens:
                            group.append(candidate)
                            i += 1
                        else:
                            break

                    merge_groups.append(group)

            print(f"🔗 Created {len(merge_groups)} merge groups from {total_to_process} chapters (after size adjustment)")

            units_to_process = merge_groups
            is_merged_mode = True
        else:
            units_to_process = [[ch] for ch in chapters_to_translate]  # Wrap each chapter as single-item group
            is_merged_mode = False
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=config.BATCH_SIZE) as executor:
            if batching_mode == 'aggressive':
                import threading
                batch_submit_lock = threading.Lock()
                active_futures = {}
                next_unit_idx = 0
                
                def submit_next_unit():
                    nonlocal next_unit_idx
                    if next_unit_idx >= len(units_to_process):
                        return False
                    unit = units_to_process[next_unit_idx]
                    if config.USE_ROLLING_SUMMARY:
                        batch_processor.set_batch_rolling_summary_text(rolling_summary_for_next_batch)
                        time.sleep(0.000001)
                    if is_merged_mode:
                        fut = executor.submit(batch_processor.process_merged_group, unit, progress_manager)
                    else:
                        fut = executor.submit(batch_processor.process_single_chapter, unit[0])
                    active_futures[fut] = unit
                    next_unit_idx += 1
                    return True
                
                # Prime the executor
                with batch_submit_lock:
                    while len(active_futures) < config.BATCH_SIZE and submit_next_unit():
                        pass
                
                while active_futures:
                    for future in concurrent.futures.as_completed(list(active_futures.keys())):
                        if check_stop():
                            print("❌ Translation stopped")
                            executor.shutdown(wait=False, cancel_futures=True)
                            return
                        unit = active_futures.pop(future)
                        completed_in_batch = 0
                        failed_in_batch = 0
                        batch_history_map = {}
                        chapters_in_batch = sum(len(u) for u in [unit])
                        try:
                            if is_merged_mode:
                                results = future.result()
                                for result in results:
                                    success, chap_num, hist_user, hist_assistant, raw_obj = result
                                    if success:
                                        completed_in_batch += 1
                                        if hist_user and hist_assistant:
                                            for idx, ch in unit:
                                                if ch.get('actual_chapter_num', ch['num']) == chap_num:
                                                    batch_history_map[idx] = (hist_user, hist_assistant, raw_obj)
                                                    break
                                    else:
                                        failed_in_batch += 1
                                    processed += 1
                                print(f"✅ Merged group done: {len(results)} chapters")
                            else:
                                success, chap_num, hist_user, hist_assistant, raw_obj = future.result()
                                idx, chapter = unit[0]
                                if success:
                                    completed_in_batch += 1
                                    if hist_user and hist_assistant:
                                        batch_history_map[idx] = (hist_user, hist_assistant, raw_obj)
                                    print(f"✅ Chapter {chap_num} done")
                                else:
                                    failed_in_batch += 1
                                    print(f"❌ Chapter {chap_num} failed")
                                processed += 1
                        except Exception as e:
                            if is_merged_mode:
                                failed_in_batch += len(unit)
                                processed += len(unit)
                            else:
                                failed_in_batch += 1
                                processed += 1
                            print(f"❌ Thread error: {e}")
                        
                        progress_percent = (processed / total_to_process) * 100
                        print(f"📊 Overall Progress: {processed}/{total_to_process} ({progress_percent:.1f}%)")
                        
                        # History append immediately for this unit
                        if config.CONTEXTUAL and getattr(config, 'HIST_LIMIT', 0) > 0:
                            hist_limit = getattr(config, 'HIST_LIMIT', 0)
                            sorted_chapters = sorted(unit, key=lambda x: x[0])
                            for idx, chapter in sorted_chapters:
                                if idx in batch_history_map:
                                    user_content, assistant_content, raw_obj = batch_history_map[idx]
                                    try:
                                        time.sleep(0.000001)
                                        history_manager.append_to_history(
                                            user_content,
                                            assistant_content,
                                            hist_limit,
                                            reset_on_limit=True,
                                            rolling_window=config.TRANSLATION_HISTORY_ROLLING,
                                            raw_assistant_object=raw_obj
                                        )
                                    except Exception as e:
                                        actual_num_for_log = chapter.get('actual_chapter_num', chapter.get('num'))
                                        print(f"⚠️ Failed to append Chapter {actual_num_for_log} to translation history (batch): {e}")
                        
                        # Rolling summary update per unit
                        if config.USE_ROLLING_SUMMARY and summary_translation_processor is not None:
                            try:
                                batch_items = sorted(unit, key=lambda x: x[0])
                                translated_blocks = []
                                last_actual_num_in_batch = None
                                for idx, chapter in batch_items:
                                    actual_num = chapter.get('actual_chapter_num', chapter.get('num'))
                                    last_actual_num_in_batch = actual_num
                                    fname_guess = FileUtilities.create_chapter_filename(chapter, actual_num)
                                    candidates = [fname_guess]
                                    if isinstance(fname_guess, str) and fname_guess.endswith('.html'):
                                        candidates.insert(0, fname_guess.replace('.html', '.txt'))
                                    elif isinstance(fname_guess, str) and fname_guess.endswith('.txt'):
                                        candidates.append(fname_guess.replace('.txt', '.html'))
                                    content = ""
                                    for cand in candidates:
                                        fp = os.path.join(out, cand)
                                        if os.path.exists(fp):
                                            with open(fp, 'r', encoding='utf-8') as f:
                                                content = f.read()
                                            if content:
                                                break
                                    if isinstance(content, str) and content:
                                        translated_blocks.append(content)
                                batch_translations_text = "\n\n---\n\n".join(translated_blocks)
                                if batch_translations_text:
                                    old_mode = getattr(config, 'ROLLING_SUMMARY_MODE', 'replace')
                                    old_max_entries = getattr(config, 'ROLLING_SUMMARY_MAX_ENTRIES', 0)
                                    try:
                                        config.ROLLING_SUMMARY_MODE = 'replace'
                                        config.ROLLING_SUMMARY_MAX_ENTRIES = int(chapters_in_batch or 0)
                                        with rolling_summary_update_lock:
                                            time.sleep(0.000001)
                                            summary_translation_processor.generate_rolling_summary(
                                                history_manager,
                                                last_actual_num_in_batch,
                                                base_system_content=None,
                                                source_text=batch_translations_text,
                                                previous_summary_text=None,
                                                previous_summary_chapter_num=None,
                                                prefer_translations_only_user=True,
                                            )
                                            summary_file = os.path.join(out, 'rolling_summary.txt')
                                            if os.path.exists(summary_file):
                                                with open(summary_file, 'r', encoding='utf-8') as sf:
                                                    rolling_summary_for_next_batch = (sf.read() or "")
                                            else:
                                                rolling_summary_for_next_batch = ""
                                    finally:
                                        config.ROLLING_SUMMARY_MODE = old_mode
                                        config.ROLLING_SUMMARY_MAX_ENTRIES = old_max_entries
                                else:
                                    rolling_summary_for_next_batch = ""
                            except Exception as e:
                                print(f"⚠️ Batch rolling summary update failed: {e}")
                                rolling_summary_for_next_batch = ""
                        
                        # Refill slots aggressively
                        with batch_submit_lock:
                            while len(active_futures) < config.BATCH_SIZE and submit_next_unit():
                                pass
            
            else:
                # direct or conservative: keep legacy batch grouping behaviour
                for batch_start in range(0, len(units_to_process), batch_group_size if not is_merged_mode else config.BATCH_SIZE):
                    if check_stop():
                        print("❌ Translation stopped during parallel processing")
                        executor.shutdown(wait=False)
                        return
                    
                    effective_batch_size = batch_group_size if not is_merged_mode else config.BATCH_SIZE
                    batch_end = min(batch_start + effective_batch_size, len(units_to_process))
                    current_batch_units = units_to_process[batch_start:batch_end]
                    
                    # Count total chapters in this batch
                    chapters_in_batch = sum(len(unit) for unit in current_batch_units)
                    
                    batch_number = (batch_start // effective_batch_size) + 1
                    if is_merged_mode:
                        print(f"\n📦 Submitting batch {batch_number}: {len(current_batch_units)} merged groups ({chapters_in_batch} chapters)")
                    else:
                        print(f"\n📦 Submitting batch {batch_number}: {chapters_in_batch} chapters")
                    
                    if config.USE_ROLLING_SUMMARY:
                        batch_processor.set_batch_rolling_summary_text(rolling_summary_for_next_batch)
                        time.sleep(0.000001)

                    if is_merged_mode:
                        future_to_unit = {
                            executor.submit(batch_processor.process_merged_group, unit, progress_manager): unit
                            for unit in current_batch_units
                        }
                    else:
                        future_to_unit = {
                            executor.submit(batch_processor.process_single_chapter, unit[0]): unit
                            for unit in current_batch_units
                        }
                    
                    completed_in_batch = 0
                    failed_in_batch = 0
                    batch_history_map = {}
                    
                    for future in concurrent.futures.as_completed(future_to_unit):
                        if check_stop():
                            print("❌ Translation stopped")
                            executor.shutdown(wait=False)
                            return
                        
                        unit = future_to_unit[future]
                        
                        try:
                            if is_merged_mode:
                                results = future.result()
                                for result in results:
                                    success, chap_num, hist_user, hist_assistant, raw_obj = result
                                    if success:
                                        completed_in_batch += 1
                                        if hist_user and hist_assistant:
                                            for idx, ch in unit:
                                                if ch.get('actual_chapter_num', ch['num']) == chap_num:
                                                    batch_history_map[idx] = (hist_user, hist_assistant, raw_obj)
                                                    break
                                    else:
                                        failed_in_batch += 1
                                    processed += 1
                                print(f"✅ Merged group done: {len(results)} chapters")
                            else:
                                success, chap_num, hist_user, hist_assistant, raw_obj = future.result()
                                idx, chapter = unit[0]
                                if success:
                                    completed_in_batch += 1
                                    print(f"✅ Chapter {chap_num} done ({completed_in_batch + failed_in_batch}/{chapters_in_batch} in batch)")
                                    if hist_user and hist_assistant:
                                        batch_history_map[idx] = (hist_user, hist_assistant, raw_obj)
                                else:
                                    failed_in_batch += 1
                                    print(f"❌ Chapter {chap_num} failed ({completed_in_batch + failed_in_batch}/{chapters_in_batch} in batch)")
                                processed += 1
                        except Exception as e:
                            if is_merged_mode:
                                failed_in_batch += len(unit)
                                processed += len(unit)
                            else:
                                failed_in_batch += 1
                                processed += 1
                            print(f"❌ Thread error: {e}")
                        
                        progress_percent = (processed / total_to_process) * 100
                        print(f"📊 Overall Progress: {processed}/{total_to_process} ({progress_percent:.1f}%)")
                    
                    # After all futures in this batch complete, append their history entries
                    if config.CONTEXTUAL and getattr(config, 'HIST_LIMIT', 0) > 0:
                        hist_limit = getattr(config, 'HIST_LIMIT', 0)
                        all_chapters_in_batch = []
                        for unit in current_batch_units:
                            all_chapters_in_batch.extend(unit)
                        sorted_chapters = sorted(all_chapters_in_batch, key=lambda x: x[0])
                        for idx, chapter in sorted_chapters:
                            if idx in batch_history_map:
                                user_content, assistant_content, raw_obj = batch_history_map[idx]
                                try:
                                    time.sleep(0.000001)
                                    history_manager.append_to_history(
                                        user_content,
                                        assistant_content,
                                        hist_limit,
                                        reset_on_limit=True,
                                        rolling_window=config.TRANSLATION_HISTORY_ROLLING,
                                        raw_assistant_object=raw_obj
                                    )
                                except Exception as e:
                                    actual_num_for_log = chapter.get('actual_chapter_num', chapter.get('num'))
                                    print(f"⚠️ Failed to append Chapter {actual_num_for_log} to translation history (batch): {e}")
                    
                    # After the batch completes, update rolling_summary.txt ONCE (for the next batch).
                    if config.USE_ROLLING_SUMMARY and summary_translation_processor is not None:
                        try:
                            batch_items = []
                            for unit in current_batch_units:
                                batch_items.extend(unit)
                            batch_items = sorted(batch_items, key=lambda x: x[0])

                            translated_blocks = []
                            last_actual_num_in_batch = None
                            for idx, chapter in batch_items:
                                try:
                                    actual_num = chapter.get('actual_chapter_num', chapter.get('num'))
                                    last_actual_num_in_batch = actual_num
                                    fname_guess = FileUtilities.create_chapter_filename(chapter, actual_num)
                                    candidates = [fname_guess]
                                    if isinstance(fname_guess, str) and fname_guess.endswith('.html'):
                                        candidates.insert(0, fname_guess.replace('.html', '.txt'))
                                    elif isinstance(fname_guess, str) and fname_guess.endswith('.txt'):
                                        candidates.append(fname_guess.replace('.txt', '.html'))

                                    content = ""
                                    for cand in candidates:
                                        fp = os.path.join(out, cand)
                                        if os.path.exists(fp):
                                            with open(fp, 'r', encoding='utf-8') as f:
                                                content = f.read()
                                            if content:
                                                break

                                    if isinstance(content, str) and content:
                                        translated_blocks.append(content)
                                except Exception:
                                    continue

                            batch_translations_text = "\n\n---\n\n".join(translated_blocks)

                            if batch_translations_text:
                                old_mode = getattr(config, 'ROLLING_SUMMARY_MODE', 'replace')
                                old_max_entries = getattr(config, 'ROLLING_SUMMARY_MAX_ENTRIES', 0)
                                try:
                                    config.ROLLING_SUMMARY_MODE = 'replace'
                                    try:
                                        config.ROLLING_SUMMARY_MAX_ENTRIES = int(chapters_in_batch or 0)
                                    except Exception:
                                        config.ROLLING_SUMMARY_MAX_ENTRIES = 0

                                    with rolling_summary_update_lock:
                                        time.sleep(0.000001)
                                        summary_translation_processor.generate_rolling_summary(
                                            history_manager,
                                            last_actual_num_in_batch,
                                            base_system_content=None,
                                            source_text=batch_translations_text,
                                            previous_summary_text=None,
                                            previous_summary_chapter_num=None,
                                            prefer_translations_only_user=True,
                                        )
                                        summary_file = os.path.join(out, 'rolling_summary.txt')
                                        if os.path.exists(summary_file):
                                            with open(summary_file, 'r', encoding='utf-8') as sf:
                                                rolling_summary_for_next_batch = (sf.read() or "")
                                        else:
                                            rolling_summary_for_next_batch = ""
                                finally:
                                    config.ROLLING_SUMMARY_MODE = old_mode
                                    config.ROLLING_SUMMARY_MAX_ENTRIES = old_max_entries
                            else:
                                rolling_summary_for_next_batch = ""
                        except Exception as e:
                            print(f"⚠️ Batch rolling summary update failed: {e}")
                            rolling_summary_for_next_batch = ""

                    print(f"\n📦 Batch Summary:")
                    print(f"   ✅ Successful: {completed_in_batch}")
                    print(f"   ❌ Failed: {failed_in_batch}")
                    
                    if batch_end < total_to_process:
                        print(f"⏳ Waiting {config.DELAY}s before next batch...")
                        time.sleep(config.DELAY)
        
        chapters_completed = batch_processor.chapters_completed
        chunks_completed = batch_processor.chunks_completed
        
        print(f"\n🎉 Parallel translation complete!")
        print(f"   Total chapters processed: {processed}")
        
        # Count qa_failed chapters correctly
        qa_failed_count = 0
        actual_successful = 0
        
        for idx, c in enumerate(chapters):
            # Get the chapter's actual number
            if (is_text_file and c.get('is_chunk', False) and isinstance(c['num'], float)):
                actual_num = c['num']
            else:
                actual_num = c.get('actual_chapter_num', c['num'])
            
            # Check if this chapter was processed and has qa_failed status
            content_hash = c.get("content_hash") or ContentProcessor.get_content_hash(c["body"])
            
            # Check if this chapter exists in progress
            chapter_info = progress_manager.prog["chapters"].get(content_hash, {})
            status = chapter_info.get("status")
            
            if status == "qa_failed":
                qa_failed_count += 1
            elif status == "completed":
                actual_successful += 1
        
        # Correct the displayed counts
        print(f"   Successful: {actual_successful}")
        if qa_failed_count > 0:
            print(f"\n⚠️ {qa_failed_count} chapters failed due to content policy violations:")
            qa_failed_chapters = []
            for idx, c in enumerate(chapters):
                if (is_text_file and c.get('is_chunk', False) and isinstance(c['num'], float)):
                    actual_num = c['num']
                else:
                    actual_num = c.get('actual_chapter_num', c['num'])
                
                content_hash = c.get("content_hash") or ContentProcessor.get_content_hash(c["body"])
                chapter_info = progress_manager.prog["chapters"].get(content_hash, {})
                if chapter_info.get("status") == "qa_failed":
                    qa_failed_chapters.append(actual_num)
            
            print(f"   Failed chapters: {', '.join(map(str, sorted(qa_failed_chapters)))}")
        
        # Stop translation completely after batch mode
        print("\n📌 Batch translation completed.")
    
    elif not config.BATCH_TRANSLATION:
        translation_processor = TranslationProcessor(config, client, out, log_callback, check_stop, uses_zero_based, is_text_file)
        
        # Only initialize AI Hunter when both the detection mode AND duplicate retry are enabled.
        if config.DUPLICATE_DETECTION_MODE == 'ai-hunter' and getattr(config, 'RETRY_DUPLICATE_BODIES', False):
            # Build the main config from environment variables and config object
            main_config = {
                'duplicate_lookback_chapters': config.DUPLICATE_LOOKBACK_CHAPTERS,
                'duplicate_detection_mode': config.DUPLICATE_DETECTION_MODE,
            }
            
            # Check if AI Hunter config was passed via environment variable
            ai_hunter_config_str = os.getenv('AI_HUNTER_CONFIG')
            if ai_hunter_config_str:
                try:
                    ai_hunter_config = json.loads(ai_hunter_config_str)
                    main_config['ai_hunter_config'] = ai_hunter_config
                    print("🤖 AI Hunter: Loaded configuration from environment")
                except json.JSONDecodeError:
                    print("⚠️ AI Hunter: Failed to parse AI_HUNTER_CONFIG from environment")
            
            # If no AI Hunter config in environment, try to load from file as fallback
            if 'ai_hunter_config' not in main_config:
                # Try multiple locations for config.json
                config_paths = [
                    os.path.join(os.getcwd(), 'config.json'),
                    os.path.join(out, '..', 'config.json'),
                ]
                
                if getattr(sys, 'frozen', False):
                    config_paths.append(os.path.join(os.path.dirname(sys.executable), 'config.json'))
                else:
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    config_paths.extend([
                        os.path.join(script_dir, 'config.json'),
                        os.path.join(os.path.dirname(script_dir), 'config.json')
                    ])
                
                for config_path in config_paths:
                    if os.path.exists(config_path):
                        try:
                            with open(config_path, 'r', encoding='utf-8') as f:
                                file_config = json.load(f)
                                if 'ai_hunter_config' in file_config:
                                    main_config['ai_hunter_config'] = file_config['ai_hunter_config']
                                    print(f"🤖 AI Hunter: Loaded configuration from {config_path}")
                                    break
                        except Exception as e:
                            print(f"⚠️ Failed to load config from {config_path}: {e}")
            
            # Always create and inject the improved AI Hunter when ai-hunter mode is selected
            ai_hunter = ImprovedAIHunterDetection(main_config)

            # The TranslationProcessor class has a method that checks for duplicates
            # We need to replace it with our enhanced AI Hunter
            
            # Create a wrapper to match the expected signature
            def enhanced_duplicate_check(self, result, idx, prog, out, actual_num=None):
                # If actual_num is not provided, try to get it from progress
                if actual_num is None:
                    # Look for the chapter being processed
                    for ch_key, ch_info in prog.get("chapters", {}).items():
                        if ch_info.get("chapter_idx") == idx:
                            actual_num = ch_info.get("actual_num", idx + 1)
                            break
                    
                    # Fallback to idx+1 if not found
                    if actual_num is None:
                        actual_num = idx + 1
                
                return ai_hunter.detect_duplicate_ai_hunter_enhanced(result, idx, prog, out, actual_num)
            
            # Bind the enhanced method to the processor instance
            translation_processor.check_duplicate_content = enhanced_duplicate_check.__get__(translation_processor, TranslationProcessor)
            
            print("🤖 AI Hunter: Using enhanced detection with configurable thresholds")
                
        # First pass: set actual chapter numbers respecting the config
        for idx, c in enumerate(chapters):
            raw_num = FileUtilities.extract_actual_chapter_number(c, patterns=None, config=config)
            #print(f"[DEBUG] Extracted raw_num={raw_num} from {c.get('original_basename', 'unknown')}")

            
            # Apply offset if configured
            offset = config.CHAPTER_NUMBER_OFFSET if hasattr(config, 'CHAPTER_NUMBER_OFFSET') else 0
            raw_num += offset
            
            if config.DISABLE_ZERO_DETECTION:
                # Use raw numbers without adjustment
                c['actual_chapter_num'] = raw_num
                c['raw_chapter_num'] = raw_num
                c['zero_adjusted'] = False
            else:
                # Store raw number
                c['raw_chapter_num'] = raw_num
                # Apply 0-based adjustment if detected
                if uses_zero_based:
                    c['actual_chapter_num'] = raw_num + 1
                    c['zero_adjusted'] = True
                else:
                    c['actual_chapter_num'] = raw_num
                    c['zero_adjusted'] = False

        # Request merging preprocessing
        merge_groups = {}  # Maps parent_idx -> list of child (idx, chapter) tuples
        merged_children = set()  # Set of idx that are merged into another chapter
        
        # Request merging for EPUB/PDF (non-text) in non-batch mode
        if config.REQUEST_MERGING_ENABLED and config.REQUEST_MERGE_COUNT > 1 and (is_pdf_file or not is_text_file):
            print(f"\n🔗 REQUEST MERGING ENABLED: Combining up to {config.REQUEST_MERGE_COUNT} chapters per request")
            
            # Collect chapters that need translation
            chapters_needing_translation = []
            for idx, c in enumerate(chapters):
                if (is_text_file and c.get('is_chunk', False) and isinstance(c['num'], float)):
                    actual_num = c['num']
                else:
                    actual_num = c.get('actual_chapter_num', c['num'])
                content_hash = c.get("content_hash") or ContentProcessor.get_content_hash(c["body"])
                
                # Skip special files (chapter 0) if translation is disabled
                raw_num = c.get('raw_chapter_num', FileUtilities.extract_actual_chapter_number(c, patterns=None, config=config))
                if not translate_special and raw_num == 0:
                    name = c.get('original_basename') or os.path.basename(c.get('filename', ''))
                    name_noext = os.path.splitext(name)[0] if name else ''
                    has_digits_in_name = bool(re.search(r'\d', name_noext))
                    if not has_digits_in_name:
                        continue
                
                if start is not None and not (start <= actual_num <= end):
                    continue
                
                needs_translation, skip_reason, existing_file = progress_manager.check_chapter_status(
                    idx, actual_num, content_hash, out, c
                )
                
                # Check file exists
                if not needs_translation and existing_file:
                    file_path = os.path.join(out, existing_file)
                    if not os.path.exists(file_path):
                        needs_translation = True
                
                # Skip empty/image-only chapters from merging
                has_images = c.get('has_images', False)
                has_meaningful_text = ContentProcessor.is_meaningful_text_content(c["body"])
                text_size = c.get('file_size', 0)
                is_empty_chapter = (not has_images and text_size < 1)
                is_image_only_chapter = (has_images and not has_meaningful_text)
                
                if needs_translation and not is_empty_chapter and not is_image_only_chapter:
                    chapters_needing_translation.append((idx, c, actual_num, content_hash))
            
            # Create merge groups
            groups = RequestMerger.create_merge_groups(
                chapters_needing_translation, 
                config.REQUEST_MERGE_COUNT
            )

            # Build proximity runs first (so we never merge far-apart chapters),
            # then pack each run under the token budget (repacking avoids 2+1,2+1 patterns).
            max_output_tokens = config.get_effective_output_limit()
            safety_margin_output = 500
            compression_factor = getattr(config, 'COMPRESSION_FACTOR', 1.0) or 1.0
            available_tokens = int((max_output_tokens - safety_margin_output) / compression_factor)
            available_tokens = max(available_tokens, 1000)

            proximity_runs = RequestMerger.create_merge_groups(
                chapters_needing_translation,
                max(1, len(chapters_needing_translation)),
            )

            groups = []
            for run in proximity_runs:
                if len(run) <= 1:
                    groups.append(run)
                    continue

                i = 0
                while i < len(run):
                    group = [run[i]]
                    i += 1

                    while i < len(run) and len(group) < config.REQUEST_MERGE_COUNT:
                        candidate = run[i]
                        merge_input = [
                            (g_actual_num, g_chapter["body"], g_chapter)
                            for (g_idx, g_chapter, g_actual_num, g_content_hash) in (group + [candidate])
                        ]
                        merged_preview = RequestMerger.merge_chapters(merge_input, log_injections=False)
                        merged_tokens = chapter_splitter.count_tokens(merged_preview)

                        if merged_tokens <= available_tokens:
                            group.append(candidate)
                            i += 1
                        else:
                            break

                    groups.append(group)
            
            for group in groups:
                if len(group) > 1:
                    parent_idx = group[0][0]  # First chapter in group is the parent
                    parent_actual_num = group[0][2]
                    merge_groups[parent_idx] = group
                    
                    # Track children to skip - but DON'T mark as merged yet
                    # (they'll be marked as merged only after parent completes)
                    for i, (idx, c, actual_num, content_hash) in enumerate(group):
                        if i > 0:
                            merged_children.add(idx)
                    
                    child_nums = [g[2] for g in group[1:]]
                    print(f"   📎 Chapters {parent_actual_num} + {child_nums} will be merged into one request")
            
            print(f"   📊 Created {len(merge_groups)} merge groups from {len(chapters_needing_translation)} chapters")

        # Second pass: process chapters
        for idx, c in enumerate(chapters):
            chap_num = c["num"]
            
            # Skip if this chapter was merged into another
            if idx in merged_children:
                if (is_text_file and c.get('is_chunk', False) and isinstance(c['num'], float)):
                    actual_num = c['num']
                else:
                    actual_num = c.get('actual_chapter_num', c['num'])
                is_text_source = is_text_file or c.get('filename', '').endswith('.txt') or c.get('is_chunk', False)
                terminology = "Section" if is_text_source else "Chapter"
                print(f"\n⏭️ Skipping {terminology} {actual_num} (merged into parent)")
                chapters_completed += 1
                continue
            
            # Check if this is a pre-split text chunk with decimal number
            if (is_text_file and c.get('is_chunk', False) and isinstance(c['num'], float)):
                actual_num = c['num']  # Preserve the decimal for text files only
            else:
                actual_num = c.get('actual_chapter_num', c['num'])
            content_hash = c.get("content_hash") or ContentProcessor.get_content_hash(c["body"])
            
            # Skip special files (chapter 0) if translation is disabled
            raw_num = c.get('raw_chapter_num', FileUtilities.extract_actual_chapter_number(c, patterns=None, config=config))
            if not translate_special and raw_num == 0:
                name = c.get('original_basename') or os.path.basename(c.get('filename', ''))
                name_noext = os.path.splitext(name)[0] if name else ''
                has_digits_in_name = bool(re.search(r'\d', name_noext))
                if not has_digits_in_name:
                    continue
            
            if start is not None and not (start <= actual_num <= end):
                # Skip silently (already summarized in earlier pass)
                continue
            
            needs_translation, skip_reason, existing_file = progress_manager.check_chapter_status(
                idx, actual_num, content_hash, out, c  # Pass the chapter object
            )
            # Add explicit file check for supposedly completed chapters
            if not needs_translation and existing_file:
                file_path = os.path.join(out, existing_file)
                if not os.path.exists(file_path):
                    print(f"⚠️ Output file missing for chapter {actual_num}: {existing_file}")
                    needs_translation = True
                    skip_reason = None
                    # Update status to file_missing
                    progress_manager.update(idx, actual_num, content_hash, None, status="file_missing", chapter_obj=c)
                    progress_manager.save()
            if not needs_translation:
                # Track skips for summary (already printed in batch mode section above)
                if not hasattr(config, '_sequential_skipped_chapters'):
                    config._sequential_skipped_chapters = []
                is_text_source = is_text_file or c.get('filename', '').endswith('.txt') or c.get('is_chunk', False)
                terminology = "Section" if is_text_source else "Chapter"
                config._sequential_skipped_chapters.append((actual_num, terminology, skip_reason))
                continue

            chapter_position = f"{chapters_completed + 1}/{chapters_to_process}"
          
            # Determine if this is a text file
            is_text_source = is_text_file or c.get('filename', '').endswith('.txt') or c.get('is_chunk', False)
            terminology = "Section" if is_text_source else "Chapter"

            # Determine file reference based on type
            if c.get('is_chunk', False):
                file_ref = f"Section_{c['num']}"
            else:
                file_ref = c.get('original_basename', f'{terminology}_{actual_num}')

            print(f"\n🔄 Processing #{idx+1}/{total_chapters} (Actual: {terminology} {actual_num}) ({chapter_position} to translate): {c['title']} [File: {file_ref}]")

            chunk_context_manager.start_chapter(chap_num, c['title'])
            
            # Initialize merge_info for this chapter (will be populated if this is a parent in a merge group)
            merge_info = None
            
            has_images = c.get('has_images', False)
            has_meaningful_text = ContentProcessor.is_meaningful_text_content(c["body"])
            text_size = c.get('file_size', 0)
            
            is_empty_chapter = (not has_images and text_size < 1)
            is_image_only_chapter = (has_images and not has_meaningful_text)
            is_mixed_content = (has_images and has_meaningful_text)
            is_text_only = (not has_images and has_meaningful_text)
            
            if is_empty_chapter:
                print(f"📄 Empty chapter {actual_num} detected")
                
                # Create filename for empty chapter
                if isinstance(c['num'], float):
                    fname = FileUtilities.create_chapter_filename(c, c['num'])
                else:
                    fname = FileUtilities.create_chapter_filename(c, actual_num)
                
                # Save original content
                with open(os.path.join(out, fname), 'w', encoding='utf-8') as f:
                    f.write(c["body"])
                
                # Update progress tracking
                progress_manager.update(idx, actual_num, content_hash, fname, status="completed_empty", chapter_obj=c)
                progress_manager.save()
                chapters_completed += 1
                
                # CRITICAL: Skip translation!
                continue

            elif is_image_only_chapter:
                print(f"📸 Image-only chapter: {c.get('image_count', 0)} images")
                
                translated_html = c["body"]
                image_translations = {}
                
                # Step 1: Process images if image translation is enabled
                if image_translator and config.ENABLE_IMAGE_TRANSLATION:
                    print(f"🖼️ Translating {c.get('image_count', 0)} images...")
                    image_translator.set_current_chapter(chap_num)
                    
                    translated_html, image_translations = process_chapter_images(
                        c["body"], 
                        actual_num,
                        image_translator,
                        check_stop
                    )
                    
                    if image_translations:
                        print(f"✅ Translated {len(image_translations)} images")
                
                # Step 2: Check for headers/titles that need translation
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(c["body"], 'html.parser')
                
                # Look for headers
                headers = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'title'])
                
                # If we have headers, we should translate them even in "image-only" chapters
                if headers and any(h.get_text(strip=True) for h in headers):
                    print(f"📝 Found headers to translate in image-only chapter")
                    
                    # Create a minimal HTML with just the headers for translation
                    headers_html = ""
                    for header in headers:
                        if header.get_text(strip=True):
                            headers_html += str(header) + "\n"
                    
                    if headers_html:
                        print(f"📤 Translating chapter headers...")
                        
                        # Send just the headers for translation
                        header_msgs = base_msg + [{"role": "user", "content": headers_html}]
                        
                        # Use the standard filename
                        fname = FileUtilities.create_chapter_filename(c, actual_num)
                        client.set_output_filename(fname)
                        
                        # Simple API call for headers
                        header_result, _ = client.send(
                            header_msgs,
                            temperature=config.TEMP,
                            max_tokens=config.MAX_OUTPUT_TOKENS
                        )
                        
                        if header_result:
                            # Clean the result
                            header_result = re.sub(r"^```(?:html)?\s*\n?", "", header_result, count=1, flags=re.MULTILINE)
                            header_result = re.sub(r"\n?```\s*$", "", header_result, count=1, flags=re.MULTILINE)
                            
                            # Parse both the translated headers and the original body
                            soup_headers = BeautifulSoup(header_result, 'html.parser')
                            soup_body = BeautifulSoup(translated_html, 'html.parser')
                            
                            # Replace headers in the body with translated versions
                            translated_headers = soup_headers.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'title'])
                            original_headers = soup_body.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'title'])
                            
                            # Match and replace headers
                            for orig, trans in zip(original_headers, translated_headers):
                                if trans and trans.get_text(strip=True):
                                    orig.string = trans.get_text(strip=True)
                            
                            translated_html = str(soup_body)
                            print(f"✅ Headers translated successfully")
                            status = "completed"
                        else:
                            print(f"⚠️ Failed to translate headers")
                            status = "completed_image_only"
                    else:
                        status = "completed_image_only"
                else:
                    print(f"ℹ️ No headers found to translate")
                    status = "completed_image_only"
                
                # Step 3: Save with correct filename
                fname = FileUtilities.create_chapter_filename(c, actual_num)
                
                with open(os.path.join(out, fname), 'w', encoding='utf-8') as f:
                    f.write(translated_html)
                
                print(f"[Chapter {idx+1}/{total_chapters}] ✅ Saved image-only chapter")
                progress_manager.update(idx, actual_num, content_hash, fname, status=status, chapter_obj=c)
                progress_manager.save()
                chapters_completed += 1
                continue

            else:
                # Set default text to translate
                text_to_translate = c["body"]
                image_translations = {}
                if is_mixed_content and image_translator and config.ENABLE_IMAGE_TRANSLATION:
                    print(f"🖼️ Processing {c.get('image_count', 0)} images first...")
                    
                    print(f"[DEBUG] Content before image processing (first 200 chars):")
                    print(c["body"][:200])
                    print(f"[DEBUG] Has h1 tags: {'<h1>' in c['body']}")
                    print(f"[DEBUG] Has h2 tags: {'<h2>' in c['body']}")
                    
                    image_translator.set_current_chapter(chap_num)
                    
                    # Store the original body before processing
                    original_body = c["body"]
                    
                    # Calculate original chapter tokens before modification
                    original_chapter_tokens = chapter_splitter.count_tokens(original_body)
                    
                    # Process images and get body with translations
                    body_with_images, image_translations = process_chapter_images(
                        c["body"], 
                        actual_num,
                        image_translator,
                        check_stop
                    )
                    
                    if image_translations:
                        print(f"✅ Translated {len(image_translations)} images")
                        
                        # Store the body with images for later merging
                        c["body_with_images"] = body_with_images
                        
                        # For chapters with only images and title, we still need to translate the title
                        # Extract clean text for translation from ORIGINAL body
                        from bs4 import BeautifulSoup
                        soup_clean = BeautifulSoup(original_body, 'html.parser')

                        # Remove images from the original to get pure text
                        for img in soup_clean.find_all('img'):
                            img.decompose()

                        # Set clean text for translation - use prettify() or str() on the full document
                        c["body"] = str(soup_clean) if soup_clean.body else original_body
                        
                        # If there's no meaningful text content after removing images, 
                        # the text translation will just translate the title, which is correct
                        print(f"   📝 Clean text for translation: {len(c['body'])} chars")
                        
                        # Update text_size to reflect actual text to translate
                        text_size = len(c["body"])

                        # Recalculate the actual token count for clean text
                        actual_text_tokens = chapter_splitter.count_tokens(c["body"])
                        print(f"   📊 Actual text tokens: {actual_text_tokens} (was counting {original_chapter_tokens} with images)")

                        # IMPORTANT: use the cleaned text for downstream chunking/translation
                        chapter_body = c["body"]

                        # If render mode is image and there's essentially no text, skip text translation
                        render_mode = os.getenv("PDF_RENDER_MODE", "xhtml").lower()
                        stripped_text_len = len(soup_clean.get_text(strip=True))
                        if render_mode == "image" and image_translations and stripped_text_len < 20:
                            print("🖼️ Image-rendered page with no meaningful text — skipping text translation.")
                            fname = FileUtilities.create_chapter_filename(c, actual_num)
                            with open(os.path.join(out, fname), 'w', encoding='utf-8') as f:
                                f.write(body_with_images)
                            progress_manager.update(idx, actual_num, content_hash, fname, status="completed_image_only", chapter_obj=c)
                            progress_manager.save()
                            chapters_completed += 1
                            continue
                    else:
                        print(f"ℹ️ No translatable text found in images")
                        # Keep original body if no image translations
                        c["body"] = original_body

                print(f"📖 Translating text content ({text_size} characters)")
                # Determine output filename for tracking
                fname = FileUtilities.create_chapter_filename(c, actual_num)
                progress_manager.update(idx, actual_num, content_hash, fname, status="in_progress", chapter_obj=c)
                progress_manager.save()
                
                # REQUEST MERGING: If this is a parent chapter, merge content from child chapters
                merge_info = None  # Will store info for response splitting
                if idx in merge_groups:
                    group = merge_groups[idx]
                    if len(group) > 1:
                        print(f"\n🔗 MERGING {len(group)} chapters into single request...")
                        
                        # Mark all chapters in the group as in_progress
                        for g_idx, g_chapter, g_actual_num, g_content_hash in group:
                            if g_idx != idx:  # Parent already marked above
                                g_fname = FileUtilities.create_chapter_filename(g_chapter, g_actual_num)
                                progress_manager.update(g_idx, g_actual_num, g_content_hash, g_fname, status="in_progress", chapter_obj=g_chapter)
                        progress_manager.save()
                        
                        # Build merged content with separators
                        chapters_data = []
                        for g_idx, g_chapter, g_actual_num, g_content_hash in group:
                            chapters_data.append((g_actual_num, g_chapter["body"], g_chapter))
                            if g_idx != idx:  # Don't print for parent
                                print(f"   → Including chapter {g_actual_num}")
                        
                        # Merge the content
                        original_body = c["body"]  # Save original for later
                        c["body"] = RequestMerger.merge_chapters(chapters_data)
                        
                        # Store merge info for response splitting
                        merge_info = {
                            'group': group,
                            'expected_chapters': [g[2] for g in group],  # actual_nums
                            'original_body': original_body
                        }
                        
                        merged_char_count = len(c["body"])
                        print(f"   📊 Merged content: {merged_char_count:,} characters")

                # Apply ignore filtering to the content before chunk splitting
                # IMPORTANT: Skip header removal if request merging is active, because
                # synthetic merge headers are critical for split-the-merge functionality
                batch_translate_active = os.getenv('BATCH_TRANSLATE_HEADERS', '0') == '1'
                ignore_title_tag = os.getenv('IGNORE_TITLE', '0') == '1' and batch_translate_active
                ignore_header_tags = os.getenv('IGNORE_HEADER', '0') == '1' and batch_translate_active
                
                # Don't remove headers if this is a merged request
                if merge_info is not None:
                    ignore_header_tags = False
                
                if (ignore_title_tag or ignore_header_tags) and c["body"]:
                    from bs4 import BeautifulSoup
                    content_soup = BeautifulSoup(c["body"], 'html.parser')
                    
                    # Remove title tags if ignored
                    if ignore_title_tag:
                        for title_tag in content_soup.find_all('title'):
                            title_tag.decompose()
                    
                    # Remove header tags if ignored
                    if ignore_header_tags:
                        for header_tag in content_soup.find_all(['h1', 'h2', 'h3']):
                            header_tag.decompose()
                    
                    c["body"] = str(content_soup)  # Update the chapter body

                # Check if this chapter is already a chunk from text file splitting
                if c.get('is_chunk', False):
                    # This is already a pre-split chunk, but still check if it needs further splitting
                    # Calculate based on effective OUTPUT limit only
                    max_output_tokens = config.get_effective_output_limit()
                    safety_margin_output = 500
                    
                    # CJK to English typically compresses to 0.7-0.9x
                    compression_factor = config.COMPRESSION_FACTOR
                    available_tokens = int((max_output_tokens - safety_margin_output) / compression_factor)
                    
                    # Ensure minimum
                    available_tokens = max(available_tokens, 1000)
                    
                    print(f"📊 Max Chunk size: {available_tokens:,} tokens (based on {max_output_tokens:,} output limit, compression: {compression_factor})")
                    
                    chapter_tokens = chapter_splitter.count_tokens(c["body"])
                    
                    # Get filename for content type detection (prefer source_file for PDFs)
                    chapter_filename = c.get('source_file') or c.get('filename') or c.get('original_basename', '')
                    
                    if chapter_tokens > available_tokens:
                        # Even pre-split chunks might need further splitting
                        chunks = chapter_splitter.split_chapter(c["body"], available_tokens, filename=chapter_filename)
                        print(f"📄 Section {c['num']} (pre-split from text file) needs further splitting into {len(chunks)} chunks")
                    else:
                        chunks = [(c["body"], 1, 1)]
                        print(f"📄 Section {c['num']} (pre-split from text file)")
                else:
                    # Normal splitting logic for non-text files
                    # Calculate based on effective OUTPUT limit only
                    max_output_tokens = config.get_effective_output_limit()
                    safety_margin_output = 500
                    
                    # CJK to English typically compresses to 0.7-0.9x
                    compression_factor = config.COMPRESSION_FACTOR
                    available_tokens = int((max_output_tokens - safety_margin_output) / compression_factor)
                    
                    # Ensure minimum
                    available_tokens = max(available_tokens, 1000)
                    
                    print(f"📊 Max Chunk size: {available_tokens:,} tokens (based on {max_output_tokens:,} output limit, compression: {compression_factor})")
                    
                    # Get filename for content type detection (prefer source_file for PDFs)
                    chapter_filename = c.get('source_file') or c.get('filename') or c.get('original_basename', '')
                    chunks = chapter_splitter.split_chapter(c["body"], available_tokens, filename=chapter_filename)
                    
                    # Use consistent terminology
                    is_text_source = is_text_file or c.get('filename', '').endswith('.txt') or c.get('is_chunk', False)
                    terminology = "Section" if is_text_source else "Chapter"
                    print(f"📄 {terminology} will be processed in {len(chunks)} chunk(s)")
                                  
            # Recalculate tokens on the actual text to be translated
            actual_chapter_tokens = chapter_splitter.count_tokens(c["body"])
            
            if len(chunks) > 1:
                is_text_source = is_text_file or c.get('filename', '').endswith('.txt') or c.get('is_chunk', False)
                terminology = "Section" if is_text_source else "Chapter"
                print(f"   ℹ️ {terminology} size: {actual_chapter_tokens:,} tokens (limit: {available_tokens:,} tokens per chunk)")
            else:
                is_text_source = is_text_file or c.get('filename', '').endswith('.txt') or c.get('is_chunk', False)
                terminology = "Section" if is_text_source else "Chapter"
                print(f"   ℹ️ {terminology} size: {actual_chapter_tokens:,} tokens (within limit of {available_tokens:,} tokens)")
            
            chapter_key_str = str(idx)
            if chapter_key_str not in progress_manager.prog["chapter_chunks"]:
                progress_manager.prog["chapter_chunks"][chapter_key_str] = {
                    "total": len(chunks),
                    "completed": [],
                    "chunks": {}
                }
            
            progress_manager.prog["chapter_chunks"][chapter_key_str]["total"] = len(chunks)
            
            translated_chunks = []
            
            for chunk_html, chunk_idx, total_chunks in chunks:
                chapter_key_str = content_hash
                old_key_str = str(idx)
                
                if chapter_key_str not in progress_manager.prog.get("chapter_chunks", {}) and old_key_str in progress_manager.prog.get("chapter_chunks", {}):
                    progress_manager.prog["chapter_chunks"][chapter_key_str] = progress_manager.prog["chapter_chunks"][old_key_str]
                    del progress_manager.prog["chapter_chunks"][old_key_str]
                    #print(f"[PROGRESS] Migrated chunks for chapter {chap_num} to new tracking system")
                
                if chapter_key_str not in progress_manager.prog["chapter_chunks"]:
                    progress_manager.prog["chapter_chunks"][chapter_key_str] = {
                        "total": len(chunks),
                        "completed": [],
                        "chunks": {}
                    }
                
                progress_manager.prog["chapter_chunks"][chapter_key_str]["total"] = len(chunks)
                
                # Get chapter status to check for qa_failed
                chapter_info = progress_manager.prog["chapters"].get(chapter_key_str, {})
                chapter_status = chapter_info.get("status")

                if chapter_status == "qa_failed":
                    # Force retranslation of qa_failed chapters
                    print(f"  [RETRY] Chunk {chunk_idx}/{total_chunks} - retranslating due to QA failure")
                        
                if check_stop():
                    print(f"❌ Translation stopped during chapter {actual_num}, chunk {chunk_idx}")
                    # Mark any in_progress chapter(s) as failed so the UI reflects the stop
                    if merge_info is not None:
                        for g_idx, g_chapter, g_actual_num, g_content_hash in merge_info['group']:
                            g_fname = FileUtilities.create_chapter_filename(g_chapter, g_actual_num)
                            progress_manager.update(
                                g_idx,
                                g_actual_num,
                                g_content_hash,
                                g_fname,
                                status="failed",
                                chapter_obj=g_chapter,
                            )
                        progress_manager.save()
                    else:
                        fname = FileUtilities.create_chapter_filename(c, actual_num)
                        progress_manager.update(
                            idx,
                            actual_num,
                            content_hash,
                            fname,
                            status="failed",
                            chapter_obj=c,
                        )
                        progress_manager.save()
                    return
                
                current_chunk_number += 1
                
                progress_percent = (current_chunk_number / total_chunks_needed) * 100 if total_chunks_needed > 0 else 0
                
                if chunks_completed > 0:
                    elapsed_time = time.time() - translation_start_time
                    avg_time_per_chunk = elapsed_time / chunks_completed
                    remaining_chunks = total_chunks_needed - current_chunk_number + 1
                    eta_seconds = remaining_chunks * avg_time_per_chunk
                    
                    eta_hours = int(eta_seconds // 3600)
                    eta_minutes = int((eta_seconds % 3600) // 60)
                    eta_str = f"{eta_hours}h {eta_minutes}m" if eta_hours > 0 else f"{eta_minutes}m"
                else:
                    eta_str = "calculating..."
                
                # For logging, strip data URIs so inline images don't explode char counts
                display_len = len(re.sub(r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+', 'data:image;base64,', chunk_html))
                if total_chunks > 1:
                    print(f"  🔄 Translating chunk {chunk_idx}/{total_chunks} for #{idx+1} (Overall: {current_chunk_number}/{total_chunks_needed} - {progress_percent:.1f}% - ETA: {eta_str})")
                    print(f"  ⏳ Chunk size: {display_len:,} characters (~{chapter_splitter.count_tokens(chunk_html):,} tokens)")
                else:
                    # Determine terminology and file reference
                    is_text_source = is_text_file or c.get('filename', '').endswith('.txt') or c.get('is_chunk', False)
                    terminology = "Section" if is_text_source else "Chapter"
                    
                    # Consistent file reference
                    if c.get('is_chunk', False):
                        file_ref = f"Section_{c['num']}"
                    else:
                        file_ref = c.get('original_basename', f'{terminology}_{actual_num}')
                    
                    chunk_tokens = chapter_splitter.count_tokens(chunk_html)
                    print(f"  📄 {terminology} {actual_num} [{display_len:,} chars, {chunk_tokens:,} tokens]")
                
                print(f"  ℹ️ This may take 30-60 seconds. Stop will take effect after completion.")
                
                if log_callback:
                    if hasattr(log_callback, '__self__') and hasattr(log_callback.__self__, 'append_chunk_progress'):
                        if total_chunks == 1:
                            # Determine terminology based on source type
                            is_text_source = is_text_file or c.get('filename', '').endswith('.txt') or c.get('is_chunk', False)
                            terminology = "Section" if is_text_source else "Chapter"

                            log_callback.__self__.append_chunk_progress(
                                1, 1, "text", 
                                f"{terminology} {actual_num}",
                                overall_current=current_chunk_number,
                                overall_total=total_chunks_needed,
                                extra_info=f"{display_len:,} chars"
                            )
                        else:
                            log_callback.__self__.append_chunk_progress(
                                chunk_idx, 
                                total_chunks, 
                                "text", 
                                f"{terminology} {actual_num}",
                                overall_current=current_chunk_number,
                                overall_total=total_chunks_needed
                            )
                    else:
                        # Determine terminology based on source type
                        is_text_source = is_text_file or c.get('filename', '').endswith('.txt') or c.get('is_chunk', False)
                        terminology = "Section" if is_text_source else "Chapter"
                        terminology_lower = "section" if is_text_source else "chapter"

                        if total_chunks == 1:
                            log_callback(f"📄 Processing {terminology} {actual_num} ({chapters_completed + 1}/{chapters_to_process}) - {progress_percent:.1f}% complete")
                        else:
                            log_callback(f"📄 processing chunk {chunk_idx}/{total_chunks} for {terminology_lower} {actual_num} - {progress_percent:.1f}% complete")
                        
                # Get custom chunk prompt template from environment; send as a separate assistant message
                chunk_prompt_template = os.getenv("TRANSLATION_CHUNK_PROMPT", "[PART {chunk_idx}/{total_chunks}]")
                chunk_prompt_msg = []
                if total_chunks > 1:
                    chunk_prompt_msg = [{
                        "role": "assistant",
                        "content": chunk_prompt_template.format(
                            chunk_idx=chunk_idx,
                            total_chunks=total_chunks
                        )
                    }]
                user_prompt = chunk_html
                
                if config.CONTEXTUAL:
                    history = history_manager.load_history()
                    trimmed = history[-config.HIST_LIMIT*2:]
                    chunk_context = chunk_context_manager.get_context_messages(limit=2)

                    include_source = os.getenv("INCLUDE_SOURCE_IN_HISTORY", "0") == "1"
                    model_name = getattr(config, 'MODEL', '').lower()
                    is_gemini_3 = ('gemini-3' in model_name) or ('gemini-exp-' in model_name)

                    memory_msgs = []
                    if is_gemini_3:
                        # Pass-through for Gemini 3 (raw objects preserved)
                        for h in trimmed:
                            if not isinstance(h, dict):
                                continue
                            role = h.get('role', 'user')
                            raw_obj = h.get('_raw_content_object')
                            content = h.get('content') or ""
                            if (not content) and raw_obj:
                                content = extract_text_from_raw_content(raw_obj)
                            if role == 'user' and not include_source:
                                continue
                            if (not content) and raw_obj is None:
                                continue
                            msg = {'role': role}
                            if content:
                                msg['content'] = content
                            if raw_obj is not None:
                                msg['_raw_content_object'] = raw_obj
                            memory_msgs.append(msg)
                    else:
                        # Prefix+content+footer for non-Gemini models
                        memory_blocks = []
                        for h in trimmed:
                            if not isinstance(h, dict):
                                continue
                            role = h.get('role', 'user')
                            content = h.get('content', '')
                            if not content:
                                continue
                            if role == 'user' and not include_source:
                                continue
                            if role == 'user':
                                prefix = (
                                    "[MEMORY - PREVIOUS SOURCE TEXT]\\n"
                                    "This is prior source content provided for context only.\\n"
                                    "Do NOT translate or repeat this text directly in your response.\\n\\n"
                                )
                            else:
                                prefix = (
                                    "[MEMORY - PREVIOUS TRANSLATION]\\n"
                                    "This is prior translated content provided for context only.\\n"
                                    "Do NOT repeat or re-output this translation.\\n\\n"
                                )
                            footer = "\\n\\n[END MEMORY BLOCK]\\n"
                            memory_blocks.append(prefix + content + footer)

                        if memory_blocks:
                            combined_memory = "\\n".join(memory_blocks)
                            memory_msgs = [{'role': 'assistant', 'content': combined_memory}]
                        else:
                            memory_msgs = []
                else:
                    history = []  # Set empty history when not contextual
                    trimmed = []
                    chunk_context = []
                    memory_msgs = []

                # Build the current system prompt from the original each time.
                # Apply per-chunk glossary compression if enabled
                if os.getenv("COMPRESS_GLOSSARY_PROMPT", "0") == "1" and glossary_path and os.path.exists(glossary_path):
                    # Rebuild system prompt with compressed glossary for THIS SPECIFIC CHUNK
                    current_system_content = build_system_prompt(config.SYSTEM_PROMPT, glossary_path, source_text=chunk_html)
                else:
                    current_system_content = original_system_prompt
                
                current_base = [{"role": "system", "content": current_system_content}]

                # Inject rolling_summary.txt verbatim as an assistant message.
                # IMPORTANT: Do NOT parse, re-header, or otherwise modify rolling_summary.txt here.
                summary_msgs_list = []
                if config.USE_ROLLING_SUMMARY:
                    rolling_summary_text = ""
                    try:
                        summary_file = os.path.join(out, "rolling_summary.txt")
                        if os.path.exists(summary_file):
                            with open(summary_file, "r", encoding="utf-8") as sf:
                                rolling_summary_text = (sf.read() or "")
                    except Exception:
                        rolling_summary_text = ""

                    # Only inject if the file has content
                    if isinstance(rolling_summary_text, str) and rolling_summary_text:
                        summary_content = (
                            "CONTEXT ONLY - DO NOT INCLUDE IN TRANSLATION:\n"
                            "[MEMORY] Previous context summary:\n\n"
                            + rolling_summary_text + "\n\n"
                            "[END MEMORY]\n"
                            "END OF CONTEXT - BEGIN ACTUAL CONTENT TO TRANSLATE:"
                        )
                        summary_msgs_list = [{"role": "assistant", "content": summary_content}]

                # Build final message list for this chunk
                msgs = current_base + summary_msgs_list + chunk_context + memory_msgs + chunk_prompt_msg + [{"role": "user", "content": user_prompt}]

                c['__index'] = idx
                c['__progress'] = progress_manager.prog
                c['history_manager'] = history_manager
                
                # Prepare merge_group_len if this is a merged request
                merge_group_len = len(merge_info['group']) if merge_info else None

                result, finish_reason, raw_obj = translation_processor.translate_with_retry(
                    msgs, chunk_html, c, chunk_idx, total_chunks, merge_group_len=merge_group_len
                )

                # If this chunk was blocked/prohibited, stop remaining chunks and mark QA fail
                if finish_reason in ("content_filter", "prohibited_content", "error"):
                    fname = FileUtilities.create_chapter_filename(c, actual_num)
                    progress_manager.update(idx, actual_num, content_hash, fname,
                                             status="qa_failed",
                                             qa_issues_found=["PROHIBITED_CONTENT"],
                                             chapter_obj=c)
                    progress_manager.save()
                    print(f"❌ Chunk {chunk_idx}/{total_chunks} hit content filter/prohibited; aborting chapter {actual_num}")
                    chunk_abort = True
                    break
                
                # Check if result is None or contains failure markers
                # Only check for failure markers if response is short (< 50 chars)
                # Longer responses are likely legitimate translations even if they contain error keywords
                is_failed = result is None or (len(str(result).strip()) < 50 and is_qa_failed_response(result))
                
                if is_failed:
                    fname = FileUtilities.create_chapter_filename(c, actual_num)
                    progress_manager.update(idx, actual_num, content_hash, fname, status="failed")
                    progress_manager.save()
                    print(f"❌ Translation failed for chapter {actual_num} - marked as failed, no output file created")
                    continue
                
                # ENHANCED TRUNCATION CHECK: Compare input vs output character counts
                # Skip this check if base64 images are present (they skew the character count)
                has_base64_image = 'data:image' in chunk_html or 'base64,' in chunk_html
                
                # Check if this result came from a fallback key
                used_fallback = hasattr(translation_processor.client, '_used_fallback_key') and translation_processor.client._used_fallback_key
                
                if not has_base64_image:
                    input_char_count = len(chunk_html)
                    output_char_count = len(result)
                    char_ratio = output_char_count / input_char_count if input_char_count > 0 else 0
                    
                    # If output is less than half of input, likely truncated
                    if char_ratio < 0.5 and output_char_count > 100:  # Only check if output has substance
                        if used_fallback:
                            # For fallback keys, just warn - don't retry (would go back to refusing model)
                            print(f"    ⚠️ Truncated output from fallback key - accepting as-is")
                        else:
                            print(f"    ⚠️ TRUNCATION DETECTED (char comparison): Input={input_char_count:,} chars, Output={output_char_count:,} chars ({char_ratio:.1%} ratio)")
                            
                            # Override finish_reason to trigger retry logic
                            # This will be caught by the retry logic if RETRY_TRUNCATED is enabled
                            if finish_reason != "length" and finish_reason != "max_tokens":
                                print(f"    🔄 Setting finish_reason to 'length' to trigger auto-retry logic")
                                finish_reason = "length"
                                
                                # If retry is enabled, call translate_with_retry again (even at same token limit)
                                retry_truncated_enabled = os.getenv("RETRY_TRUNCATED", "0") == "1"
                                if retry_truncated_enabled:
                                    print(f"    🔄 Retrying after char-ratio truncation check...")
                                    original_max = config.MAX_OUTPUT_TOKENS
                                    # Use the configured retry cap; if non-positive, stick with current
                                    target_tokens = config.MAX_RETRY_TOKENS if config.MAX_RETRY_TOKENS > 0 else original_max
                                    config.MAX_OUTPUT_TOKENS = max(original_max, target_tokens)
                                    
                                    result_retry, finish_reason_retry, raw_obj_retry = translation_processor.translate_with_retry(
                                        msgs, chunk_html, c, chunk_idx, total_chunks
                                    )
                                    
                                    config.MAX_OUTPUT_TOKENS = original_max
                                    
                                    if result_retry and len(result_retry) > len(result):
                                        print(f"    ✅ Retry succeeded: {len(result):,} → {len(result_retry):,} chars")
                                        result = result_retry
                                        finish_reason = finish_reason_retry
                                        raw_obj = raw_obj_retry
                                    else:
                                        print(f"    ⚠️ Retry did not improve output, using original")

                if config.REMOVE_AI_ARTIFACTS:
                    result = ContentProcessor.clean_ai_artifacts(result, True)

                if config.EMERGENCY_RESTORE:
                    result = ContentProcessor.emergency_restore_paragraphs(result, chunk_html)

                if config.REMOVE_AI_ARTIFACTS:
                    lines = result.split('\n')
                    
                    json_line_count = 0
                    for i, line in enumerate(lines[:5]):
                        if line.strip() and any(pattern in line for pattern in [
                            '"role":', '"content":', '"messages":', 
                            '{"role"', '{"content"', '[{', '}]'
                        ]):
                            json_line_count = i + 1
                        else:
                            break
                    
                    if json_line_count > 0 and json_line_count < len(lines):
                        remaining = '\n'.join(lines[json_line_count:])
                        if remaining.strip() and len(remaining) > 100:
                            result = remaining
                            print(f"✂️ Removed {json_line_count} lines of JSON artifacts")

                result = re.sub(r'\[PART \d+/\d+\]\s*', '', result, flags=re.IGNORECASE)

                translated_chunks.append((result, chunk_idx, total_chunks))
                
                chunk_context_manager.add_chunk(user_prompt, result, chunk_idx, total_chunks)

                progress_manager.prog["chapter_chunks"][chapter_key_str]["completed"].append(chunk_idx)
                progress_manager.prog["chapter_chunks"][chapter_key_str]["chunks"][str(chunk_idx)] = result
                progress_manager.save()

                chunks_completed += 1
                    
                will_reset = history_manager.will_reset_on_next_append(
                    config.HIST_LIMIT if config.CONTEXTUAL else 0, 
                    config.TRANSLATION_HISTORY_ROLLING
                )


                # Check if we captured thought signatures
                if raw_obj:
                    # print("🧠 Captured thought signature for history")
                    pass
                
                # Add microsecond delay before history append to prevent race conditions
                time.sleep(0.000001)  # 1 microsecond delay
                history = history_manager.append_to_history(
                    user_prompt, 
                    result, 
                    config.HIST_LIMIT if config.CONTEXTUAL else 0,
                    reset_on_limit=True,
                    rolling_window=config.TRANSLATION_HISTORY_ROLLING,
                    raw_assistant_object=raw_obj
                )

                if chunk_idx < total_chunks:
                    # Handle float delays while checking for stop
                    full_seconds = int(config.DELAY)
                    fractional_second = config.DELAY - full_seconds
                    
                    # Check stop signal every second for full seconds
                    for i in range(full_seconds):
                        if check_stop():
                            print("❌ Translation stopped during delay")
                            # Mark any in_progress chapter(s) as failed so the UI reflects the stop
                            if merge_info is not None:
                                for g_idx, g_chapter, g_actual_num, g_content_hash in merge_info['group']:
                                    g_fname = FileUtilities.create_chapter_filename(g_chapter, g_actual_num)
                                    progress_manager.update(
                                        g_idx,
                                        g_actual_num,
                                        g_content_hash,
                                        g_fname,
                                        status="failed",
                                        chapter_obj=g_chapter,
                                    )
                                progress_manager.save()
                            else:
                                fname = FileUtilities.create_chapter_filename(c, actual_num)
                                progress_manager.update(
                                    idx,
                                    actual_num,
                                    content_hash,
                                    fname,
                                    status="failed",
                                    chapter_obj=c,
                                )
                                progress_manager.save()
                            return
                        time.sleep(1)
                    
                    # Handle the fractional part if any
                    if fractional_second > 0:
                        if check_stop():
                            print("❌ Translation stopped during delay")
                            # Mark any in_progress chapter(s) as failed so the UI reflects the stop
                            if merge_info is not None:
                                for g_idx, g_chapter, g_actual_num, g_content_hash in merge_info['group']:
                                    g_fname = FileUtilities.create_chapter_filename(g_chapter, g_actual_num)
                                    progress_manager.update(
                                        g_idx,
                                        g_actual_num,
                                        g_content_hash,
                                        g_fname,
                                        status="failed",
                                        chapter_obj=g_chapter,
                                    )
                                progress_manager.save()
                            else:
                                fname = FileUtilities.create_chapter_filename(c, actual_num)
                                progress_manager.update(
                                    idx,
                                    actual_num,
                                    content_hash,
                                    fname,
                                    status="failed",
                                    chapter_obj=c,
                                )
                                progress_manager.save()
                            return
                        time.sleep(fractional_second)

            if check_stop():
                print(f"❌ Translation stopped before saving chapter {actual_num}")
                # Mark any in_progress chapter(s) as failed so the UI reflects the stop
                if merge_info is not None:
                    for g_idx, g_chapter, g_actual_num, g_content_hash in merge_info['group']:
                        g_fname = FileUtilities.create_chapter_filename(g_chapter, g_actual_num)
                        progress_manager.update(
                            g_idx,
                            g_actual_num,
                            g_content_hash,
                            g_fname,
                            status="failed",
                            chapter_obj=g_chapter,
                        )
                    progress_manager.save()
                else:
                    fname = FileUtilities.create_chapter_filename(c, actual_num)
                    progress_manager.update(
                        idx,
                        actual_num,
                        content_hash,
                        fname,
                        status="failed",
                        chapter_obj=c,
                    )
                    progress_manager.save()
                return

            if len(translated_chunks) > 1:
                print(f"  📎 Merging {len(translated_chunks)} chunks...")
                translated_chunks.sort(key=lambda x: x[1])
                merged_result = chapter_splitter.merge_translated_chunks(translated_chunks)
            else:
                merged_result = translated_chunks[0][0] if translated_chunks else ""

            if config.CONTEXTUAL and len(translated_chunks) > 1:
                user_summary, assistant_summary = chunk_context_manager.get_summary_for_history()
                
                if user_summary and assistant_summary:
                    # Add microsecond delay before summary append
                    time.sleep(0.000001)  # 1 microsecond delay
                    history_manager.append_to_history(
                        user_summary,
                        assistant_summary,
                        config.HIST_LIMIT,
                        reset_on_limit=False,
                        rolling_window=config.TRANSLATION_HISTORY_ROLLING
                    )
                    print(f"  📝 Added chapter summary to history")

            chunk_context_manager.clear()

            # For text file chunks, ensure we pass the decimal number
            if is_text_file and c.get('is_chunk', False) and isinstance(c.get('num'), float):
                fname = FileUtilities.create_chapter_filename(c, c['num'])  # Use the decimal num directly
                print(f"[DEBUG] Text file chunk - using decimal num {c['num']} -> filename: {fname}")
            else:
                fname = FileUtilities.create_chapter_filename(c, actual_num)
                if is_text_file:
                    print(f"[DEBUG] Text file - using actual_num {actual_num} -> filename: {fname}")

            client.set_output_filename(fname)
            cleaned = re.sub(r"^```(?:html)?\s*\n?", "", merged_result, count=1, flags=re.MULTILINE)
            cleaned = re.sub(r"\n?```\s*$", "", cleaned, count=1, flags=re.MULTILINE)

            cleaned = ContentProcessor.clean_ai_artifacts(cleaned, remove_artifacts=config.REMOVE_AI_ARTIFACTS)

            # If the cleaned translation is empty/whitespace, treat as failure and skip file write
            if not cleaned or not str(cleaned).strip():
                print(f"❌ Translation empty for chapter {actual_num} — skipping file write")
                chapter_key = progress_manager._get_chapter_key(actual_num, FileUtilities.create_chapter_filename(c, actual_num), c, content_hash)
                existing = progress_manager.prog.get("chapters", {}).get(chapter_key, {})
                # If already qa_failed (e.g., prohibited content), keep that; otherwise mark qa_failed with EMPTY_OUTPUT
                new_status = existing.get("status") if existing.get("status") == "qa_failed" else "qa_failed"
                qa_issues = existing.get("qa_issues_found") or []
                if "EMPTY_OUTPUT" not in qa_issues:
                    qa_issues = qa_issues + ["EMPTY_OUTPUT"]
                progress_manager.update(
                    idx,
                    actual_num,
                    content_hash,
                    FileUtilities.create_chapter_filename(c, actual_num),
                    status=new_status,
                    qa_issues_found=qa_issues,
                    chapter_obj=c,
                )
                progress_manager.save()
                # Move to next chapter without writing a file
                continue
            
            if is_mixed_content and image_translations:
                print(f"🔀 Merging {len(image_translations)} image translations with text...")
                from bs4 import BeautifulSoup
                # Parse the translated text (which has the translated title/header)
                soup_translated = BeautifulSoup(cleaned, 'html.parser')
                
                # For each image translation, insert it into the document
                for img_path, translation_html in image_translations.items():
                    if translation_html and '<div' in translation_html:
                        # Parse the translation HTML
                        trans_soup = BeautifulSoup(translation_html, 'html.parser')
                        container = trans_soup.find('div', class_=['translated-text-only', 'image-with-translation'])
                        
                        if container:
                            # Clone the container to avoid issues
                            new_container = BeautifulSoup(str(container), 'html.parser').find('div')
                            
                            # Find where to insert - after header or at beginning of body
                            if soup_translated.body:
                                # Try to find a header to insert after
                                header = soup_translated.body.find(['h1', 'h2', 'h3'])
                                if header:
                                    header.insert_after(new_container)
                                else:
                                    # No header, insert at beginning of body
                                    soup_translated.body.insert(0, new_container)
                            else:
                                # No body tag, try to find any header
                                header = soup_translated.find(['h1', 'h2', 'h3'])
                                if header:
                                    header.insert_after(new_container)
                                else:
                                    # Just append to the document
                                    soup_translated.append(new_container)
                
                # Update cleaned with the merged content
                cleaned = str(soup_translated)
                print(f"✅ Successfully merged image translations")
            
            # REQUEST MERGING: If this was a merged request, save to parent chapter file only
            if merge_info is not None:
                print(f"\n🔗 Saving merged response to parent chapter file...")
                
                # Get parent chapter info (first in the group)
                parent_idx, parent_chapter, parent_actual_num, parent_content_hash = merge_info['group'][0]
                merged_child_nums = [g[2] for g in merge_info['group'][1:]]  # All except parent
                
                # Track whether the underlying API response was truncated; if so mark qa_failed immediately
                preserved_fr = locals().get("merged_finish_reason", finish_reason)
                was_truncated = preserved_fr in ["length", "max_tokens"]
                if was_truncated:
                    print(f"   ⚠️ Merged response was TRUNCATED (finish_reason: {preserved_fr})")
                    qa_issues = ["TRUNCATED"]
                    parent_fname = FileUtilities.create_chapter_filename(parent_chapter, parent_actual_num)
                    progress_manager.update(
                        parent_idx, parent_actual_num, parent_content_hash, parent_fname,
                        status="qa_failed", chapter_obj=parent_chapter, qa_issues_found=qa_issues
                    )
                    for g_idx, g_chapter, g_actual_num, g_content_hash in merge_info['group'][1:]:
                        progress_manager.update(
                            g_idx, g_actual_num, g_content_hash, None,
                            status="qa_failed", chapter_obj=g_chapter, qa_issues_found=qa_issues
                        )
                    progress_manager.save()
                    print(f"   ⚠️ Merged group marked as qa_failed due to truncation")
                    continue

                # We may exit early on QA failure below, but we still want to strip
                # injected split markers from any saved merged output when Split-the-Merge is enabled.
                split_the_merge = os.getenv('SPLIT_THE_MERGE', '0') == '1'
                
                # Check for QA failures first (independent of truncation)
                if is_qa_failed_response(cleaned):
                    print(f"   ⚠️ Merged response marked as qa_failed for parent + children")
                    
                    # Only save file for debugging if it contains meaningful content beyond error markers
                    cleaned_stripped = cleaned.strip()
                    is_only_error_marker = cleaned_stripped in [
                        "[TRANSLATION FAILED]",
                        "[Content Blocked]",
                        "[IMAGE TRANSLATION FAILED]",
                        "[EXTRACTION FAILED]",
                        "[RATE LIMITED]",
                        "[]"
                    ] or cleaned_stripped.startswith("[TRANSLATION FAILED - ORIGINAL TEXT PRESERVED]") or cleaned_stripped.startswith("[CONTENT BLOCKED - ORIGINAL TEXT PRESERVED]")
                    
                    if not is_only_error_marker:
                        # Save for debugging - contains actual translation attempt that failed QA
                        parent_fname = FileUtilities.create_chapter_filename(parent_chapter, parent_actual_num)
                        try:
                            cleaned_to_save = cleaned
                            if split_the_merge:
                                cleaned_to_save = re.sub(
                                    r'<h1[^>]*id="split-\d+"[^>]*>.*?</h1>\s*',
                                    '',
                                    cleaned_to_save,
                                    flags=re.IGNORECASE | re.DOTALL,
                                )
                            with open(os.path.join(out, parent_fname), 'w', encoding='utf-8') as f:
                                f.write(cleaned_to_save)
                        except Exception:
                            pass
                    
                    # Mark ALL chapters in the merge group as qa_failed using
                    # their own expected filenames so we overwrite existing
                    # in_progress entries instead of creating composite keys.
                    for g_idx, g_chapter, g_actual_num, g_content_hash in merge_info['group']:
                        g_fname = FileUtilities.create_chapter_filename(g_chapter, g_actual_num)
                        progress_manager.update(
                            g_idx,
                            g_actual_num,
                            g_content_hash,
                            g_fname,
                            status="qa_failed",
                            chapter_obj=g_chapter,
                        )
                    progress_manager.save()
                    print(f"   ⚠️ Merged group marked as qa_failed")
                    continue
                
                # Check if Split the Merge is enabled
                split_the_merge = os.getenv('SPLIT_THE_MERGE', '0') == '1'
                disable_fallback = os.getenv('DISABLE_MERGE_FALLBACK', '0') == '1'
                split_sections = None
                
                if split_the_merge and len(merge_info['group']) > 1:
                    # Try to split by invisible markers
                    split_sections = RequestMerger.split_by_markers(cleaned, len(merge_info['group']))
                
                # If disable fallback is enabled and split failed, mark as qa_failed
                if split_the_merge and disable_fallback and (not split_sections or len(split_sections) != len(merge_info['group'])):
                    print(f"   ⚠️ Split failed and fallback disabled - marking merged group as qa_failed")
                    
                    # Only save file for debugging if it contains meaningful content beyond error markers
                    cleaned_stripped = cleaned.strip()
                    is_only_error_marker = cleaned_stripped in [
                        "[TRANSLATION FAILED]",
                        "[Content Blocked]",
                        "[IMAGE TRANSLATION FAILED]",
                        "[EXTRACTION FAILED]",
                        "[RATE LIMITED]",
                        "[]"
                    ] or cleaned_stripped.startswith("[TRANSLATION FAILED - ORIGINAL TEXT PRESERVED]") or cleaned_stripped.startswith("[CONTENT BLOCKED - ORIGINAL TEXT PRESERVED]")
                    
                    if not is_only_error_marker:
                        parent_fname = FileUtilities.create_chapter_filename(parent_chapter, parent_actual_num)
                        try:
                            cleaned_to_save = cleaned
                            if split_the_merge:
                                cleaned_to_save = re.sub(
                                    r'<h1[^>]*id="split-\d+"[^>]*>.*?</h1>\s*',
                                    '',
                                    cleaned_to_save,
                                    flags=re.IGNORECASE | re.DOTALL,
                                )
                            with open(os.path.join(out, parent_fname), 'w', encoding='utf-8') as f:
                                f.write(cleaned_to_save)
                        except Exception:
                            pass

                    # Mark ALL chapters in the merge group as qa_failed
                    for g_idx, g_chapter, g_actual_num, g_content_hash in merge_info['group']:
                        g_fname = FileUtilities.create_chapter_filename(g_chapter, g_actual_num)
                        progress_manager.update(
                            g_idx,
                            g_actual_num,
                            g_content_hash,
                            g_fname,
                            status="qa_failed",
                            chapter_obj=g_chapter,
                            qa_issues_found=["SPLIT_FAILED"],
                        )
                    progress_manager.save()
                    print(f"   ⚠️ Merged group ({len(merge_info['group'])} chapters) marked as qa_failed with SPLIT_FAILED")
                    continue
                
                if split_sections and len(split_sections) == len(merge_info['group']):
                    # Split successful - save each section as individual file
                    print(f"   ✂️ Splitting merged content into {len(split_sections)} individual files")
                    
                    saved_files = []
                    for i, (g_idx, g_chapter, g_actual_num, g_content_hash) in enumerate(merge_info['group']):
                        section_content = split_sections[i]
                        
                        # Generate filename for this chapter using content.opf naming
                        split_fname = FileUtilities.create_chapter_filename(g_chapter, g_actual_num)
                        
                        # Handle text file mode
                        if is_text_file:
                            split_fname = split_fname.replace('.html', '.txt')
                            from bs4 import BeautifulSoup
                            soup = BeautifulSoup(section_content, 'html.parser')
                            section_content = soup.get_text(strip=True)
                        
                        # Save the section
                        split_output_path = os.path.join(out, split_fname)
                        with open(split_output_path, 'w', encoding='utf-8') as f:
                            f.write(section_content)
                        
                        # Verify file was written successfully
                        if os.path.exists(split_output_path):
                            saved_files.append((g_idx, g_chapter, g_actual_num, g_content_hash, split_fname))
                            print(f"      💾 Saved Chapter {g_actual_num}: {split_fname} ({len(section_content)} chars)")
                        else:
                            print(f"      ⚠️ ERROR: Failed to write file {split_fname} - file does not exist after write")
                    
                    # Mark all chapters as completed or qa_failed (for truncated)
                    for g_idx, g_chapter, g_actual_num, g_content_hash, split_fname in saved_files:
                        chapter_status = "qa_failed" if was_truncated else "completed"
                        qa_issues = ["TRUNCATED"] if was_truncated else None
                        progress_manager.update(
                            g_idx, g_actual_num, g_content_hash, split_fname,
                            status=chapter_status, chapter_obj=g_chapter, qa_issues_found=qa_issues
                        )
                        chapters_completed += 1
                    
                    # Save once after all updates
                    progress_manager.save()
                    print(f"   ✅ Split the Merge complete: {len(saved_files)} files created")
                    continue
                
                # Normal merged behavior (split not enabled or header count mismatch)
                # Save entire merged response to parent chapter's file
                cleaned_to_save = cleaned
                if split_the_merge and len(merge_info['group']) > 1:
                    cleaned_to_save = re.sub(
                        r'<h1[^>]*id="split-\d+"[^>]*>.*?</h1>\s*',
                        '',
                        cleaned_to_save,
                        flags=re.IGNORECASE | re.DOTALL,
                    )

                if is_text_file and not is_pdf_file:
                    parent_fname = FileUtilities.create_chapter_filename(parent_chapter, parent_actual_num).replace('.html', '.txt')
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(cleaned_to_save, 'html.parser')
                    text_content = soup.get_text(strip=True)
                    
                    parent_output_path = os.path.join(out, parent_fname)
                    with open(parent_output_path, 'w', encoding='utf-8') as f:
                        f.write(text_content)
                else:
                    parent_fname = FileUtilities.create_chapter_filename(parent_chapter, parent_actual_num)
                    parent_output_path = os.path.join(out, parent_fname)
                    with open(parent_output_path, 'w', encoding='utf-8') as f:
                        f.write(cleaned_to_save)
                
                # Verify file was actually written before marking as completed
                if not os.path.exists(parent_output_path):
                    print(f"   ⚠️ ERROR: Failed to write merged file {parent_fname} - file does not exist after write")
                    # Mark all chapters in the group as failed since parent file wasn't written
                    for g_idx, g_chapter, g_actual_num, g_content_hash in merge_info['group']:
                        progress_manager.update(g_idx, g_actual_num, g_content_hash, None, status="failed", chapter_obj=g_chapter)
                    progress_manager.save()
                    continue
                
                print(f"   💾 Saved merged content to Chapter {parent_actual_num}: {parent_fname} ({len(cleaned_to_save)} chars)")
                
                if was_truncated:
                    # For truncated merged responses, mark ALL chapters as qa_failed
                    qa_issues = ["TRUNCATED"]
                    progress_manager.update(
                        parent_idx, parent_actual_num, parent_content_hash, parent_fname,
                        status="qa_failed", chapter_obj=parent_chapter, qa_issues_found=qa_issues
                    )
                    for g_idx, g_chapter, g_actual_num, g_content_hash in merge_info['group'][1:]:
                        progress_manager.update(
                            g_idx, g_actual_num, g_content_hash, None,
                            status="qa_failed", chapter_obj=g_chapter, qa_issues_found=qa_issues
                        )
                    chapters_completed += len(merge_info['group'])

                    # Save once after all updates
                    progress_manager.save()
                    print(f"   ⚠️ Merged group marked as qa_failed due to truncation")
                else:
                    # Normal success path: parent completed, children marked as merged
                    progress_manager.update(
                        parent_idx, parent_actual_num, parent_content_hash, parent_fname,
                        status="completed", chapter_obj=parent_chapter,
                        merged_chapters=merged_child_nums
                    )
                    chapters_completed += 1
                    
                    # Mark child chapters as merged (point to parent's output file) - atomically after parent
                    for g_idx, g_chapter, g_actual_num, g_content_hash in merge_info['group'][1:]:
                        progress_manager.mark_as_merged(g_idx, g_actual_num, g_content_hash, parent_actual_num, g_chapter, parent_output_file=parent_fname)
                        chapters_completed += 1
                    
                    # Save once after all updates
                    progress_manager.save()
                    print(f"   📊 Saved merged content for {len(merge_info['group'])} chapters")
                
                # Skip normal save since we handled it above and exit this translation run
                continue

            if is_text_file and not is_pdf_file:
                # For text files (but NOT PDFs), save as plain text instead of HTML
                fname_txt = fname.replace('.html', '.txt')  # Change extension to .txt
                
                # Extract text from HTML
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(cleaned, 'html.parser')
                text_content = soup.get_text(strip=True)
                
                # Write plain text file
                output_path = os.path.join(out, fname_txt)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(text_content)
                
                # Verify file was actually written before marking as completed
                if not os.path.exists(output_path):
                    print(f"⚠️ ERROR: Failed to write file {fname_txt} - file does not exist after write")
                    # Keep status as in_progress or mark as failed
                    progress_manager.save()  # Save current in_progress state
                    continue
                
                print(f"💾 Saved text file: {fname_txt} (Chapter {actual_num})")
                
                final_title = c['title'] or make_safe_filename(c['title'], actual_num)
                # Don't print individual "Processed" messages - these are redundant with the main progress display
                if os.getenv('DEBUG_CHAPTER_SAVES', '0') == '1':
                    print(f"[Processed {idx+1}/{total_chapters}] ✅ Saved Chapter {actual_num}: {final_title}")
                
                # Determine status based on comprehensive failure detection
                qa_issues = None
                if is_qa_failed_response(cleaned):
                    chapter_status = "qa_failed"
                    failure_reason = get_failure_reason(cleaned)
                    print(f"⚠️ Chapter {actual_num} marked as qa_failed: {failure_reason}")
                elif finish_reason in ["length", "max_tokens"]:
                    chapter_status = "qa_failed"
                    qa_issues = ["TRUNCATED"]
                    print(f"⚠️ Chapter {actual_num} marked as qa_failed: truncated (finish_reason: {finish_reason})")
                else:
                    chapter_status = "completed"

                progress_manager.update(idx, actual_num, content_hash, fname_txt, status=chapter_status, chapter_obj=c, qa_issues_found=qa_issues)
            else:
                # For EPUB files, keep original HTML behavior
                output_path = os.path.join(out, fname)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned)
                
                # Verify file was actually written before marking as completed
                if not os.path.exists(output_path):
                    print(f"⚠️ ERROR: Failed to write file {fname} - file does not exist after write")
                    # Keep status as in_progress or mark as failed
                    progress_manager.save()  # Save current in_progress state
                    continue
                
                final_title = c['title'] or make_safe_filename(c['title'], actual_num)
                # Don't print individual "Processed" messages - these are redundant with the main progress display
                if os.getenv('DEBUG_CHAPTER_SAVES', '0') == '1':
                    print(f"[Processed {idx+1}/{total_chapters}] ✅ Saved Chapter {actual_num}: {final_title}")
                
                # Determine status based on comprehensive failure detection
                qa_issues = None
                if is_qa_failed_response(cleaned):
                    chapter_status = "qa_failed"
                    failure_reason = get_failure_reason(cleaned)
                    print(f"⚠️ Chapter {actual_num} marked as qa_failed: {failure_reason}")
                elif finish_reason in ["length", "max_tokens"]:
                    chapter_status = "qa_failed"
                    qa_issues = ["TRUNCATED"]
                    print(f"⚠️ Chapter {actual_num} marked as qa_failed: truncated (finish_reason: {finish_reason})")
                else:
                    chapter_status = "completed"

                progress_manager.update(idx, actual_num, content_hash, fname, status=chapter_status, chapter_obj=c, qa_issues_found=qa_issues)
            progress_manager.save()
            
            # After completing this chapter, produce a rolling summary and store it for the NEXT chapter
            if config.USE_ROLLING_SUMMARY:
                # Use the original system prompt to build the summary system prompt
                base_system_content = original_system_prompt

                summary_mode = str(getattr(config, 'ROLLING_SUMMARY_MODE', 'replace') or 'replace').strip().lower()

                def _load_previous_rolling_summary_text(*, full_file: bool = False) -> str:
                    """Load rolling_summary.txt to use as assistant context (no parsing)."""
                    try:
                        summary_file = os.path.join(out, "rolling_summary.txt")
                        if not os.path.exists(summary_file):
                            return ""
                        with open(summary_file, "r", encoding="utf-8") as f:
                            content = f.read().strip()
                        return content
                    except Exception:
                        return ""

                def _get_last_translated_outputs(n: int) -> str:
                    """Build the user text from the last N translated chapter outputs (by completed_list)."""
                    try:
                        n = int(n or 0)
                        if n <= 0:
                            return cleaned
                        # completed_list is saved (sorted) by ProgressManager.save()
                        completed_list = progress_manager.prog.get("completed_list") or []
                        if not isinstance(completed_list, list) or not completed_list:
                            return cleaned
                        last_items = completed_list[-n:]
                        blocks = []
                        for item in last_items:
                            try:
                                chap_num = item.get("num")
                                rel_file = item.get("file")
                                if not rel_file:
                                    continue
                                fp = os.path.join(out, rel_file)
                                if not os.path.exists(fp):
                                    continue
                                with open(fp, "r", encoding="utf-8") as f:
                                    txt = f.read().strip()
                                if not txt:
                                    continue
                                blocks.append(
                                    f"=== Previous Translated Text: Chapter {chap_num} ===\n"
                                    f"{txt}\n"
                                    f"=== End Previous Translated Text ==="
                                )
                            except Exception:
                                continue
                        return "\n\n".join(blocks) if blocks else cleaned
                    except Exception:
                        return cleaned

                if summary_mode == 'replace':
                    # In replace mode, update the rolling summary using:
                    # - assistant: previous rolling summary (from rolling_summary.txt)
                    # - user: last N translated chapter outputs (configured by ROLLING_SUMMARY_EXCHANGES)
                    prev_summary = _load_previous_rolling_summary_text()
                    n = int(getattr(config, 'ROLLING_SUMMARY_EXCHANGES', 5) or 5)
                    user_text = _get_last_translated_outputs(n)
                    summary_text = translation_processor.generate_rolling_summary(
                        history_manager,
                        actual_num,
                        base_system_content,
                        source_text=user_text,
                        previous_summary_text=prev_summary,
                        previous_summary_chapter_num=None,
                        prefer_translations_only_user=True,
                    )
                else:
                    # append (and any unknown value): summarize ONLY this chapter's translated output.
                    # Do NOT send the previous rolling summary in append mode.
                    summary_text = translation_processor.generate_rolling_summary(
                        history_manager,
                        actual_num,
                        base_system_content,
                        source_text=cleaned,
                        previous_summary_text=None,
                        previous_summary_chapter_num=None,
                    )

                if summary_text:
                    last_summary_block_text = summary_text
                    last_summary_chapter_num = actual_num
            
            chapters_completed += 1

    # Check if PDF should output as PDF or EPUB
    pdf_output_format = os.getenv('PDF_OUTPUT_FORMAT', 'pdf').lower()
    should_create_pdf = is_text_file or (is_pdf_file and pdf_output_format == 'pdf')
    
    if should_create_pdf:
        print("📄 Text file translation complete!")
        try:
            # Collect all translated chapters with their metadata
            translated_chapters = []
            
            for chapter in chapters:
                # Look for .txt files for text files, .html for PDFs
                fname_base = FileUtilities.create_chapter_filename(chapter, chapter['num'])
                if is_pdf_file:
                    fname_to_check = fname_base  # PDFs use .html files
                else:
                    fname_to_check = fname_base.replace('.html', '.txt')  # Text files use .txt
                
                if os.path.exists(os.path.join(out, fname_to_check)):
                    with open(os.path.join(out, fname_to_check), 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    translated_chapters.append({
                        'num': chapter['num'],
                        'title': chapter['title'],
                        'content': content,
                        'is_chunk': chapter.get('is_chunk', False),
                        'chunk_info': chapter.get('chunk_info', {}),
                        'filename': fname_to_check  # Store filename for debugging
                    })
                elif os.path.exists(os.path.join(out, fname_base)):
                    # Fallback to HTML if txt doesn't exist
                    with open(os.path.join(out, fname_base), 'r', encoding='utf-8') as f:
                        content = f.read()
                        # For PDFs, keep HTML content; for text files, extract text
                        if is_pdf_file:
                            # Keep the HTML as-is for PDFs
                            text = content
                        else:
                            # Extract text from HTML for text files
                            from bs4 import BeautifulSoup
                            soup = BeautifulSoup(content, 'html.parser')
                            text = soup.get_text(strip=True)
                    
                    translated_chapters.append({
                        'num': chapter['num'],
                        'title': chapter['title'],
                        'content': text,
                        'is_chunk': chapter.get('is_chunk', False),
                        'chunk_info': chapter.get('chunk_info', {}),
                        'filename': fname_base  # Store filename for debugging
                    })
            
            # Sort chapters by number to ensure correct order
            # Handle both integer and float chapter numbers (e.g., 1.0, 1.1, etc.)
            translated_chapters.sort(key=lambda x: float(x['num']))
            
            print(f"✅ Translation complete! {len(translated_chapters)} section files created:")
            for chapter_data in translated_chapters:
                print(f"   • Section {chapter_data['num']}: {chapter_data['title']} (from {chapter_data.get('filename', 'unknown')})")  
            
            # Create a combined file with proper section structure
            if input_path.lower().endswith('.pdf'):
                # Check if content is HTML or plain text
                is_html_content = any('<html' in chapter_data.get('content', '').lower() or 
                                     '<p>' in chapter_data.get('content', '') or
                                     '<div' in chapter_data.get('content', '')
                                     for chapter_data in translated_chapters)
                
                if is_html_content:
                    # HTML content - create PDF from HTML with proper rendering
                    combined_path = os.path.join(out, f"{txt_processor.file_base}_translated.pdf")
                    print(f"📄 Creating PDF from HTML with formatting and images...")
                    
                    # Build full HTML content
                    # Always insert page breaks between combined pages (one HTML fragment per output PDF page).
                    html_parts = []
                    current_main_chapter = None
                    
                    # Note: translated_chapters is already sorted at this point
                    for i, chapter_data in enumerate(translated_chapters):
                        content = chapter_data['content']
                        
                        # Extract body content from individual HTML pages if they have full HTML structure
                        if '<html' in content.lower() and '<body' in content.lower():
                            from bs4 import BeautifulSoup
                            soup = BeautifulSoup(content, 'html.parser')
                            body = soup.find('body')
                            if body:
                                # Extract just the body content to avoid nested html/head/body tags
                                content = ''.join(str(child) for child in body.children)
                        
                        # De-duplicate MuPDF's per-page wrapper IDs (e.g. id="page0") when concatenating
                        # to avoid duplicate IDs/anchors confusing HTML->PDF renderers.
                        try:
                            from bs4 import BeautifulSoup
                            frag = BeautifulSoup(content, 'html.parser')
                            for tag in frag.find_all(id='page0'):
                                tag['id'] = f'mupdf-page0-{i + 1}'
                            content = str(frag)
                        except Exception:
                            pass

                        # Always insert a page break before every combined page after the first.
                        if i > 0:
                            html_parts.append('<div class="page-break"></div>\n')
                        
                        if chapter_data.get('is_chunk'):
                            chunk_info = chapter_data.get('chunk_info', {})
                            original_chapter = chunk_info.get('original_chapter')
                            chunk_idx = chunk_info.get('chunk_idx', 1)
                            total_chunks = chunk_info.get('total_chunks', 1)
                            
                            if original_chapter != current_main_chapter:
                                current_main_chapter = original_chapter
                            
                            html_parts.append(content)
                            if chunk_idx < total_chunks:
                                html_parts.append('\n')
                        else:
                            current_main_chapter = chapter_data['num']
                            html_parts.append(content)
                    
                    full_html_body = "".join(html_parts)
                    
                    # Post-process: merge paragraphs that span across pages
                    full_html_body = _merge_split_paragraphs(full_html_body)

                    # Post-process: merge image-only page containers into the previous page
                    # (reduces wasted whitespace for "image-only" pages)
                    full_html_body = _merge_image_only_pages(full_html_body)

                    # Post-process: wrap last text block with a following image to reduce image-only pages
                    full_html_body = _keep_text_with_following_image(full_html_body)
                    
                    # Replace/insert a clean Table of Contents built from h1/h2 headers
                    full_html_body = _generate_and_replace_toc(full_html_body)
                    
                    # Wrap in full HTML document with CSS
                    css_path = os.path.join(out, 'styles.css')
                    css_link = '<link rel="stylesheet" href="styles.css">' if os.path.exists(css_path) else ''
                    
                    # Extra inline CSS for PDF-derived HTML:
                    # - h3 is used as body text in our PDF extraction; normalize it to paragraph-like styling
                    # - reduce margins around images
                    # - keep-with-image wrapper helps reduce image-only PDF pages
                    extra_css = """
<style>
  h3 { font-size: 1em; font-weight: normal; margin: 0.6em 0; }
  h2 { margin: 1.2em 0 0.6em 0; }
  img { margin: 0.6em auto; display: block; max-width: 100%; height: auto; }
  .keep-with-image { break-inside: avoid; page-break-inside: avoid; }
  .page-break { page-break-before: always; break-before: page; clear: both; height: 0; margin: 0; padding: 0; }
</style>
"""

                    full_html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>{txt_processor.file_base} - Translated</title>
    {css_link}
    {extra_css}
</head>
<body>
{full_html_body}
</body>
</html>"""
                    
                    # Save HTML file for reference
                    html_path = os.path.join(out, f"{txt_processor.file_base}_translated.html")
                    with open(html_path, 'w', encoding='utf-8') as f:
                        f.write(full_html)
                    print(f"   • Created HTML file: {html_path}")
                    
                    # Convert HTML to PDF
                    try:
                        from pdf_extractor import create_pdf_from_html
                        
                        images_dir = os.path.join(out, 'images')
                        css_arg = css_path if os.path.exists(css_path) else None
                        images_arg = images_dir if os.path.exists(images_dir) else None
                        
                        # Check if images directory exists and has images
                        has_images = False
                        if images_arg and os.path.exists(images_arg):
                            image_files = [f for f in os.listdir(images_arg) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp'))]
                            has_images = len(image_files) > 0
                            if has_images:
                                print(f"   • Found {len(image_files)} images to include in PDF")
                        
                        if create_pdf_from_html(full_html, combined_path, css_path=css_arg, images_dir=images_arg):
                            print(f"   • Created translated PDF file: {combined_path}")
                            if has_images:
                                print(f"   • PDF includes images from images folder")
                        else:
                            print("⚠️ Failed to create PDF from HTML, using HTML file")
                            combined_path = html_path
                    except Exception as e:
                        print(f"⚠️ Error creating PDF from HTML: {e}")
                        import traceback
                        traceback.print_exc()
                        print(f"   • Using HTML file instead: {html_path}")
                        combined_path = html_path
                else:
                    # Plain text content - use text-based PDF creation
                    combined_path = os.path.join(out, f"{txt_processor.file_base}_translated.pdf")
                    print(f"📄 Creating PDF from plain text...")
                    
                    # Build full text content
                    full_text_parts = []
                    current_main_chapter = None
                    
                    # Note: translated_chapters is already sorted at this point
                    for i, chapter_data in enumerate(translated_chapters):
                        content = chapter_data['content']
                        
                        if chapter_data.get('is_chunk'):
                            chunk_info = chapter_data.get('chunk_info', {})
                            original_chapter = chunk_info.get('original_chapter')
                            chunk_idx = chunk_info.get('chunk_idx', 1)
                            total_chunks = chunk_info.get('total_chunks', 1)
                            
                            if original_chapter != current_main_chapter:
                                current_main_chapter = original_chapter
                                if i > 0:
                                    full_text_parts.append(f"\n\n{'='*50}\n\n")
                            
                            full_text_parts.append(content)
                            if chunk_idx < total_chunks:
                                full_text_parts.append("\n")
                        else:
                            current_main_chapter = chapter_data['num']
                            if i > 0:
                                full_text_parts.append(f"\n\n{'='*50}\n\n")
                            full_text_parts.append(content)
                    
                    full_text = "".join(full_text_parts)
                    
                    from pdf_extractor import create_pdf_from_text
                    if create_pdf_from_text(full_text, combined_path):
                        print(f"   • Created translated PDF file: {combined_path}")
                    else:
                        print("⚠️ Failed to create PDF, falling back to text output")
                        combined_path = os.path.join(out, f"{txt_processor.file_base}_translated.txt")
                        with open(combined_path, 'w', encoding='utf-8') as f:
                            f.write(full_text)
                        print(f"   • Created fallback text file: {combined_path}")
            
            else:
                combined_path = os.path.join(out, f"{txt_processor.file_base}_translated.txt")
                with open(combined_path, 'w', encoding='utf-8') as combined:
                    current_main_chapter = None
                    
                    # Note: translated_chapters is already sorted at this point
                    for i, chapter_data in enumerate(translated_chapters):
                        content = chapter_data['content']
                        
                        # Check if this is a chunk of a larger chapter
                        if chapter_data.get('is_chunk'):
                            chunk_info = chapter_data.get('chunk_info', {})
                            original_chapter = chunk_info.get('original_chapter')
                            chunk_idx = chunk_info.get('chunk_idx', 1)
                            total_chunks = chunk_info.get('total_chunks', 1)
                            
                            # Only add the chapter header for the first chunk
                            if original_chapter != current_main_chapter:
                                current_main_chapter = original_chapter
                                
                                # Add separator if not first chapter
                                if i > 0:
                                    combined.write(f"\n\n{'='*50}\n\n")
                            
                            # Add the chunk content
                            combined.write(content)
                            
                            # Add spacing between chunks of the same chapter
                            if chunk_idx < total_chunks:
                                combined.write("\n")
                        else:
                            # This is a standalone chapter
                            current_main_chapter = chapter_data['num']
                            
                            # Add separator if not first chapter
                            if i > 0:
                                combined.write(f"\n\n{'='*50}\n\n")
                            
                            # Add the content
                            combined.write(content)
            
            print(f"   • Combined file with preserved sections: {combined_path}")
            
            total_time = time.time() - translation_start_time
            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            seconds = int(total_time % 60)
            
            print(f"\n⏱️ Total translation time: {hours}h {minutes}m {seconds}s")
            print(f"📊 Chapters completed: {chapters_completed}")
            print(f"✅ Text file translation complete!")
            
            if log_callback:
                log_callback(f"✅ Text file translation complete! Created {combined_path}")
            
            # Exit here for text files and PDFs - don't fall through to EPUB generation
            print("TRANSLATION_COMPLETE_SIGNAL")
            return
            
        except Exception as e:
            print(f"❌ Error creating combined text file: {e}")
            if log_callback:
                log_callback(f"❌ Error creating combined text file: {e}")
            print("TRANSLATION_COMPLETE_SIGNAL")
            return
    else:
        print("🔍 Checking for translated chapters...")
        # Respect retain extension toggle: if enabled, don't look for response_ prefix
        if should_retain_source_extension():
            response_files = [f for f in os.listdir(out) if f.endswith('.html') and not f.startswith('chapter_')]
        else:
            response_files = [f for f in os.listdir(out) if f.startswith('response_') and f.endswith('.html')]
        chapter_files = [f for f in os.listdir(out) if f.startswith('chapter_') and f.endswith('.html')]

        if not response_files and chapter_files:
            if should_retain_source_extension():
                print(f"⚠️ No translated files found, but {len(chapter_files)} original chapters exist")
                print("ℹ️ Retain-source-extension mode is ON: skipping placeholder creation and using original files for EPUB compilation.")
            else:
                print(f"⚠️ No translated files found, but {len(chapter_files)} original chapters exist")
                print("📝 Creating placeholder response files for EPUB compilation...")
                
                for chapter_file in chapter_files:
                    response_file = chapter_file.replace('chapter_', 'response_', 1)
                    src = os.path.join(out, chapter_file)
                    dst = os.path.join(out, response_file)
                    
                    try:
                        with open(src, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        soup = BeautifulSoup(content, 'html.parser')
                        notice = soup.new_tag('p')
                        notice.string = "[Note: This chapter could not be translated - showing original content]"
                        notice['style'] = "color: red; font-style: italic;"
                        
                        if soup.body:
                            soup.body.insert(0, notice)
                        
                        with open(dst, 'w', encoding='utf-8') as f:
                            f.write(str(soup))
                            
                    except Exception as e:
                        print(f"⚠️ Error processing {chapter_file}: {e}")
                        try:
                            shutil.copy2(src, dst)
                        except:
                            pass
                
                print(f"✅ Created {len(chapter_files)} placeholder response files")
                print("⚠️ Note: The EPUB will contain untranslated content")
        
        print("📘 Building final EPUB…")
        try:
            from epub_converter import fallback_compile_epub
            fallback_compile_epub(out, log_callback=log_callback)
            print("✅ All done: your final EPUB is in", out)
            
            total_time = time.time() - translation_start_time
            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            seconds = int(total_time % 60)
            
            print(f"\n📊 Translation Statistics:")
            print(f"   • Total chunks processed: {chunks_completed}")
            print(f"   • Total time: {hours}h {minutes}m {seconds}s")
            if chunks_completed > 0:
                avg_time = total_time / chunks_completed
                print(f"   • Average time per chunk: {avg_time:.1f} seconds")
            
            stats = progress_manager.get_stats(out)
            print(f"\n📊 Progress Tracking Summary:")
            print(f"   • Total chapters tracked: {stats['total_tracked']}")
            print(f"   • Successfully completed: {stats['completed']}")
            print(f"   • Missing files: {stats['missing_files']}")
            print(f"   • In progress: {stats['in_progress']}")
                
        except Exception as e:
            print("❌ EPUB build failed:", e)

    print("TRANSLATION_COMPLETE_SIGNAL")

if __name__ == "__main__":
    main()
