"""
Enhanced QA Scanner for HTML Translation Files

This module provides comprehensive quality assurance scanning for translated HTML files,
including duplicate detection, foreign character detection, and translation artifact detection.

PERFORMANCE IMPROVEMENTS:
- Added detailed progress indicators for all slow operations
- Shows estimated time remaining for long operations  
- Displays current file being scanned
- Provides progress updates every 5-10%
- Added timing information for each phase
- MinHash optimization status messages
- Debug output for stop functionality

OPTIMIZATION TIPS:
- For datasets > 100 files, avoid AI Hunter mode (use aggressive instead)
- Install 'datasketch' package for 2-10x faster duplicate detection: pip install datasketch
- Use 'summary' report format for faster completion
- Disable checks you don't need in QA Scanner Settings
"""


import os
import hashlib
import json
import zipfile
import csv
from bs4 import BeautifulSoup
from langdetect import detect, LangDetectException
from difflib import SequenceMatcher
from collections import Counter, defaultdict
from tqdm import tqdm
import threading
import re
import unicodedata
import time
import html as html_lib
from typing import Dict, List, Tuple, Set, Optional
import warnings
from functools import lru_cache
import concurrent.futures
import multiprocessing
from threading import Lock

# Optional: psutil for process priority and CPU affinity control
try:
    import psutil  # type: ignore
except Exception:
    psutil = None

# Add a global lock for thread-safe operations
merge_lock = Lock()

# Global variable for text samples mapping
_global_text_samples = {}

warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    from datasketch import MinHash, MinHashLSH
    MINHASH_AVAILABLE = True
except ImportError:
    MINHASH_AVAILABLE = False
    #"Note: Install 'datasketch' package for faster duplicate detection on large datasets if running it as a script

# Global flag to allow stopping the scan externally
_stop_flag = False

def stop_scan():
    """Set the stop flag to True
    
    This function should be called by the GUI to stop a running scan.
    The GUI code needs to:
    1. Import this function: from scan_html_folder import stop_scan
    2. Call it in the stop_qa_scan method: stop_scan()
    3. Update the QA button to show "Stop Scan" when scan is running
    """
    global _stop_flag
    _stop_flag = True
    print("🛑 STOP SCAN CALLED - Global flag set to True")  # More visible debug
    return True  # Return True to confirm it was called

# Configuration class for duplicate detection
class DuplicateDetectionConfig:
    def __init__(self, mode='quick-scan', custom_settings=None):
        self.mode = mode
        self.custom_settings = custom_settings
        self.thresholds = {
            'aggressive': {
                'similarity': 0.75,
                'semantic': 0.70,
                'structural': 0.80,
                'consecutive_chapters': 3,
                'word_overlap': 0.65,
                'minhash_threshold': 0.70
            },
            'quick-scan': {  # Optimized for speed
                'similarity': 0.85,
                'semantic': 0.80,
                'structural': 0.90,
                'consecutive_chapters': 1,  # Only check adjacent chapters
                'word_overlap': 0.75,
                'minhash_threshold': 0.80,
                'skip_semantic': True,  # Skip expensive calculations
                'skip_structural': True,
                'skip_minhash': True,
                'sample_size': 1000,  # Smaller sample
                'check_all_pairs': False  # Never check all pairs
            },
            'custom': {
                'similarity': 0.85,
                'semantic': 0.80,
                'structural': 0.90,
                'consecutive_chapters': 2,
                'word_overlap': 0.75,
                'minhash_threshold': 0.80,
                'check_all_pairs': False,
                'sample_size': 3000,
                'min_text_length': 500
            },
            'ai-hunter': {
                'similarity': 0.30, 
                'semantic': 0.85,
                'structural': 0.85,
                'consecutive_chapters': 5,
                'word_overlap': 0.50,
                'minhash_threshold': 0.60,
                'check_all_pairs': True
            }
        }
        
        # Override with custom settings if mode is 'custom'
        if mode == 'custom' and custom_settings:
            self.thresholds['custom'].update(custom_settings.get('thresholds', {}))
            for key in ['consecutive_chapters', 'check_all_pairs', 'sample_size', 'min_text_length']:
                if key in custom_settings:
                    self.thresholds['custom'][key] = custom_settings[key]
    
    def get_threshold(self, key):
        return self.thresholds[self.mode].get(key, 0.8)

# Constants
DASH_CHARS = {
    '-', '–', '—', '―', '⸺', '⸻', '﹘', '﹣', '－', '⁃', '‐', '‑', '‒',
    '_', '━', '─', '═', '╌', '╍', '┄', '┅', '┈', '┉', '⎯', '⏤', '＿',
    '＊', '*', '~', '～', '∼', '〜', 'ㅡ'  # Added Korean dash character
}

COMMON_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'after',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might',
    'chapter', 'each', 'person', 'persons', 'he', 'she', 'it', 'they', 'them',
    'his', 'her', 'their', 'this', 'that', 'these', 'those', 'which', 'who',
    'what', 'where', 'when', 'why', 'how', 'all', 'some', 'any', 'no', 'not'
}

# Korean dash patterns to EXCLUDE from detection
KOREAN_DASH_PATTERNS = [
    r'[ㅡ―—–\-]+',  # Korean dashes and similar
    r'[\u2014\u2015\u2500-\u257F]+',  # Box drawing characters often used in Korean text
    r'[\u3161\u3163\u3164]+',  # Korean filler characters
]

# Extended Korean separator characters to exclude from non-English detection
KOREAN_SEPARATOR_CHARS = {
    'ㅡ',  # Korean dash/separator (U+3161)
    '―',   # Horizontal bar (U+2015)
    '—',   # Em dash (U+2014)
    '–',   # En dash (U+2013)
    '［', '］',  # Full-width brackets
    '【', '】',  # Black lenticular brackets
    '〔', '〕',  # Tortoise shell brackets
    '《', '》',  # Double angle brackets
    '「', '」',  # Corner brackets
    '『', '』',  # White corner brackets
}

# Translation artifacts patterns
TRANSLATION_ARTIFACTS = {
    'machine_translation': re.compile(r'(MTL note|TN:|Translator:|T/N:|TL note:|Translator\'s note:)', re.IGNORECASE),
    'encoding_issues': re.compile(r'[�□◇]{2,}'),
    'repeated_watermarks': re.compile(r'(\[[\w\s]+\.(?:com|net|org)\])\s*\1{2,}', re.IGNORECASE),
    'chapter_continuation': re.compile(r'(to be continued|continued from|continuation of|cont\.)', re.IGNORECASE),
    'split_indicators': re.compile(r'(part \d+|section \d+|\(\d+/\d+\))', re.IGNORECASE),
    'api_response_unavailable': re.compile(r'\[AI RESPONSE UNAVAILABLE\]|\[TRANSLATION FAILED - ORIGINAL TEXT PRESERVED\]|\[IMAGE TRANSLATION FAILED\]', re.IGNORECASE),
    
    'glossary_leakage_csv': re.compile(
        r'(?:type|raw_name|translated_name|gender|description)\s*,\s*(?:type|raw_name|translated_name|gender|description)',
        re.IGNORECASE
    ),
    'glossary_leakage_json': re.compile(
        r'"(?:type|raw_name|translated_name|gender|description)"\s*:\s*"[^"]+"\s*,?\s*"(?:type|raw_name|translated_name|gender|description)"',
        re.IGNORECASE
    )
}
# Cache configuration - will be updated by configure_qa_cache()
_cache_config = {
    "enabled": True,
    "sizes": {
        "normalize_text": 10000,
        "similarity_ratio": 20000,
        "content_hashes": 5000,
        "semantic_fingerprint": 2000,
        "structural_signature": 2000,
        "semantic_similarity": 5000,
        "structural_similarity": 5000,
        "file_extraction": 200
    }
}

def configure_qa_cache(config):
    """Update cache configuration"""
    global _cache_config
    _cache_config.update(config)
    # Clear existing caches after configuration
    clear_qa_caches()
    # Re-apply caches with new sizes
    _apply_caches()

def get_cache_size(func_name):
    """Get configured cache size for a function"""
    if not _cache_config.get("enabled", True):
        return 0  # Disable cache
    
    size = _cache_config.get("sizes", {}).get(func_name, 1000)
    return None if size == -1 else size

# Define functions WITHOUT decorators first
def extract_semantic_fingerprint_impl(text):
    """Extract semantic fingerprint and signature from text"""
    # For cache efficiency with long texts
    cache_text = text[:50000] if len(text) > 50000 else text
    
    # Extract features for semantic analysis
    words = cache_text.lower().split()
    
    # Character names (words starting with capital letters, appearing multiple times)
    potential_names = re.findall(r'\b[A-Z][a-z]+\b', cache_text)
    name_freq = Counter(potential_names)
    characters = [name for name, count in name_freq.items() 
                  if count >= 3 and name not in COMMON_WORDS]
    
    # Dialogue analysis
    dialogue_matches = re.findall(r'["\"\'""''『』「」]([^"\"\'""''『』「」]+)["\"\'""''『』「」]', cache_text)
    dialogue_count = len(dialogue_matches)
    dialogue_density = dialogue_count / max(1, len(words)) if words else 0
    dialogue_lengths = [len(d) for d in dialogue_matches[:30]]  # First 30 dialogue lengths
    
    # Character frequencies (sorted list)
    character_frequencies = [count for _, count in name_freq.most_common()]
    
    # Speaker sequence extraction
    speaker_patterns = re.findall(r'(\w+)\s+(?:said|asked|replied|shouted|whispered|spoke)', cache_text.lower())
    speaker_sequence = speaker_patterns[:50]  # First 50 speakers
    
    # Paragraph structure (lengths of each paragraph)
    paragraphs = [p for p in cache_text.split('\n\n') if p.strip()]
    paragraph_structure = [len(p) for p in paragraphs[:50]]  # First 50 paragraph lengths
    
    # Action words density
    action_words = len(re.findall(r'\b(\w+ed|spoke|says?|asks?|replies?|shouts?|screams?|whispers?)\b', cache_text))
    action_density = action_words / max(1, len(words)) if words else 0
    
    # Numbers in text
    numbers = re.findall(r'\b\d+\b', cache_text)
    
    # Create fingerprint string
    fingerprint = f"chars:{len(characters)}_dial:{dialogue_density:.2f}_act:{action_density:.2f}_nums:{len(numbers)}_words:{len(words)}"
    
    # Create signature dict
    signature = {
        'characters': characters[:20],  # Top 20 characters
        'dialogue_density': dialogue_density,
        'dialogue_count': dialogue_count,
        'dialogue_lengths': dialogue_lengths,
        'character_frequencies': character_frequencies,
        'speaker_sequence': speaker_sequence,
        'paragraph_structure': paragraph_structure,
        'total_words': len(words),
        'action_density': action_density,
        'numbers': numbers[:50],  # First 50 numbers
        'text_length': len(cache_text)
    }
    
    return fingerprint, signature

def extract_structural_signature_impl(text):
    """Extract structural patterns from text"""
    # For cache efficiency with long texts
    cache_text = text[:50000] if len(text) > 50000 else text
    
    lines = cache_text.split('\n')
    
    # Count different types of lines
    para_count = len([l for l in lines if len(l.strip()) > 50])
    short_lines = len([l for l in lines if 0 < len(l.strip()) < 20])
    empty_lines = len([l for l in lines if not l.strip()])
    
    # Dialogue patterns
    dialogue_lines = len(re.findall(r'["\"\'""''『』「」].*?["\"\'""''『』「」]', cache_text))
    
    # Create pattern string (first letter of each line type)
    pattern = ''
    for line in lines[:100]:  # First 100 lines
        if not line.strip():
            pattern += 'E'  # Empty
        elif len(line.strip()) < 20:
            pattern += 'S'  # Short
        elif re.search(r'["\"\'""''『』「」]', line):
            pattern += 'D'  # Dialogue
        else:
            pattern += 'P'  # Paragraph
    
    # Calculate average paragraph length
    paragraphs = [l for l in lines if len(l.strip()) > 50]
    avg_para_length = sum(len(p) for p in paragraphs) / max(1, len(paragraphs)) if paragraphs else 0
    
    # Dialogue ratio
    dialogue_ratio = dialogue_lines / max(1, len(lines))
    
    signature = {
        'pattern': pattern,
        'paragraph_count': para_count,
        'avg_paragraph_length': avg_para_length,
        'dialogue_ratio': dialogue_ratio,
        'short_lines': short_lines,
        'empty_lines': empty_lines
    }
    
    return signature

def extract_content_fingerprint_impl(text):
    """Extract key sentences that can identify duplicate content"""
    lines = [line.strip() for line in text.split('\n') 
             if len(line.strip()) > 50 and not is_dash_separator_line(line)]
    
    if len(lines) < 5:
        return ""
    
    # Take first, middle, and last substantial sentences
    fingerprint_lines = []
    if len(lines) >= 3:
        fingerprint_lines = [lines[0], lines[len(lines)//2], lines[-1]]
    else:
        fingerprint_lines = lines[:3]
    
    return ' '.join(fingerprint_lines).lower()

# Initialize cached versions
extract_semantic_fingerprint = None
extract_structural_signature = None
extract_content_fingerprint = None

def _apply_caches():
    """Apply LRU cache to functions with current configuration"""
    global extract_semantic_fingerprint, extract_structural_signature, extract_content_fingerprint
    
    # Apply caching with current sizes
    extract_semantic_fingerprint = lru_cache(maxsize=get_cache_size("semantic_fingerprint") or 2000)(extract_semantic_fingerprint_impl)
    extract_structural_signature = lru_cache(maxsize=get_cache_size("structural_signature") or 2000)(extract_structural_signature_impl)
    extract_content_fingerprint = lru_cache(maxsize=get_cache_size("content_fingerprint") or 2000)(extract_content_fingerprint_impl)

# Apply initial caches
_apply_caches()

def clear_qa_caches():
    """Clear all QA scanner caches"""
    # Clear directly cached functions
    if hasattr(normalize_text, 'cache_clear'):
        normalize_text.cache_clear()
    
    if hasattr(generate_content_hashes, 'cache_clear'):
        generate_content_hashes.cache_clear()
    
    if hasattr(calculate_similarity_ratio, 'cache_clear'):
        calculate_similarity_ratio.cache_clear()
    
    # Clear the actual cached implementations
    if hasattr(_calculate_semantic_similarity_cached, 'cache_clear'):
        _calculate_semantic_similarity_cached.cache_clear()
    
    if hasattr(_calculate_structural_similarity_cached, 'cache_clear'):
        _calculate_structural_similarity_cached.cache_clear()
    
    if hasattr(calculate_semantic_fingerprint_similarity, 'cache_clear'):
        calculate_semantic_fingerprint_similarity.cache_clear()
    
    if hasattr(extract_semantic_fingerprint, 'cache_clear'):
        extract_semantic_fingerprint.cache_clear()
    
    if hasattr(extract_structural_signature, 'cache_clear'):
        extract_structural_signature.cache_clear()
    
    if hasattr(extract_content_fingerprint, 'cache_clear'):
        extract_content_fingerprint.cache_clear()
    
    if hasattr(_extract_text_from_html_cached, 'cache_clear'):
        _extract_text_from_html_cached.cache_clear()
    
def get_cache_info():
    """Get cache statistics for all cached functions"""
    cache_info = {}
    
    # For functions that are directly cached
    if hasattr(normalize_text, 'cache_info'):
        cache_info['normalize_text'] = normalize_text.cache_info()
    
    if hasattr(generate_content_hashes, 'cache_info'):
        cache_info['content_hashes'] = generate_content_hashes.cache_info()
    
    if hasattr(calculate_similarity_ratio, 'cache_info'):
        cache_info['similarity_ratio'] = calculate_similarity_ratio.cache_info()
    
    # For wrapper functions, use the actual cached implementation
    if hasattr(_calculate_semantic_similarity_cached, 'cache_info'):
        cache_info['semantic_similarity'] = _calculate_semantic_similarity_cached.cache_info()
    
    if hasattr(_calculate_structural_similarity_cached, 'cache_info'):
        cache_info['structural_similarity'] = _calculate_structural_similarity_cached.cache_info()
    
    if hasattr(calculate_semantic_fingerprint_similarity, 'cache_info'):
        cache_info['semantic_fingerprint_similarity'] = calculate_semantic_fingerprint_similarity.cache_info()
    
    if hasattr(extract_semantic_fingerprint, 'cache_info'):
        cache_info['semantic_fingerprint'] = extract_semantic_fingerprint.cache_info()
    
    if hasattr(extract_structural_signature, 'cache_info'):
        cache_info['structural_signature'] = extract_structural_signature.cache_info()
    
    if hasattr(extract_content_fingerprint, 'cache_info'):
        cache_info['content_fingerprint'] = extract_content_fingerprint.cache_info()
    
    if hasattr(_extract_text_from_html_cached, 'cache_info'):
        cache_info['file_extraction'] = _extract_text_from_html_cached.cache_info()
    
    return cache_info

# For very long texts, we'll use a hash as cache key
def _get_cache_key(text, max_length=10000):
    """Generate a cache key for text, using hash for long texts"""
    if len(text) > max_length:
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    return text
    
def extract_text_from_html(file_path):
    """Extract text from HTML or TXT file
    
    Returns:
        str OR tuple: 
            - For backwards compatibility: just the text (if not checking HTML structure)
            - For new functionality: (text_content, has_html_tag) tuple
    """
    # Get file modification time as part of cache key
    try:
        mtime = os.path.getmtime(file_path)
        cache_key = f"{file_path}:{mtime}"
    except OSError:
        cache_key = file_path
    
    return _extract_text_from_html_cached(cache_key, file_path)

def _extract_text_from_html_cached(cache_key, file_path):
    """Cached implementation of extract_text_from_html"""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
        
    # Check if it's a .txt file
    if file_path.lower().endswith('.txt'):
        # For .txt files, just return the content directly
        return content
    
    # For HTML files, parse with BeautifulSoup
    soup = BeautifulSoup(content, "html.parser")
    text = soup.get_text(separator='\n', strip=True)
    
    # For backwards compatibility, we'll handle the HTML tag check separately
    # in the scan function rather than always returning a tuple
    return text

# Configure cache size dynamically
_extract_text_from_html_cached = lru_cache(maxsize=get_cache_size("file_extraction") or 200)(_extract_text_from_html_cached)

import re

def check_html_structure(file_path):
    """Check if an HTML file has proper HTML tags"""
    if not file_path.lower().endswith(('.html', '.xhtml', '.htm')):
        return True
        
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    
    html_tags = [
        '<html', '<head', '<title', '<body', '<h1', '<h2', '<h3', '<h4', '<h5', '<h6',
        '<p>', '<p ', '<br', '<div', '<span', '<a ', '<img', '<ul', '<ol', '<li',
        '<table', '<tr', '<td', '<th', '<form', '<input', '<button', '<meta',
        '<link', '<script', '<style', '<nav', '<header', '<footer', '<main',
        '<article', '<section', '<aside'
    ]
    
    content_lower = content.lower()
    has_html_tags = any(tag in content_lower for tag in html_tags)
    
    # DEBUG: Print what we found
    print(f"\nChecking file: {file_path}")
    print(f"First 100 chars: {content[:100]}")
    print(f"Has HTML tags: {has_html_tags}")
    
    return has_html_tags

def is_dash_separator_line(line):
    """Check if a line consists only of dash-like punctuation characters"""
    stripped = line.strip()
    if not stripped:
        return False
    
    # Check if it's a Korean dash pattern (should NOT be flagged)
    for pattern in KOREAN_DASH_PATTERNS:
        if re.match(f'^{pattern}$', stripped):
            return False
    
    # Check if all non-space characters are in our dash set
    non_space_chars = [c for c in stripped if not c.isspace()]
    if not non_space_chars:
        return False
    
    # Check various dash patterns
    if all(c in DASH_CHARS for c in non_space_chars):
        return True
    
    # Check for repeated patterns
    if re.match(r'^[\s\-–—―_*~ㅡ]+$', stripped):
        return True
    
    # Check for patterns like "---", "***", "___", "~~~" (3 or more)
    if re.match(r'^(\-{3,}|_{3,}|\*{3,}|~{3,}|–{2,}|—{2,}|―{2,}|ㅡ{2,})$', stripped):
        return True
    
    # Check for spaced patterns like "- - -", "* * *"
    if re.match(r'^([\-–—―_*~ㅡ]\s*){3,}$', stripped):
        return True
    
    return False

def filter_dash_lines(text):
    """Filter out dash separator lines from text"""
    lines = text.split('\n')
    return '\n'.join(line for line in lines if not is_dash_separator_line(line))

def has_no_spacing_or_linebreaks(text, space_threshold=0.01, min_text_length=100):
    filtered_text = filter_dash_lines(text)
    
    # Skip check for very short text (e.g., cover pages with just image + title)
    if len(filtered_text) < min_text_length:
        return False
    
    space_ratio = filtered_text.count(" ") / max(1, len(filtered_text))
    newline_count = filtered_text.count("\n")
    # Flag as issue only if both conditions are met:
    # - very few spaces (minified/malformed) AND
    # - no linebreaks (compacted content)
    # A single-line file with proper spacing is valid
    return space_ratio < space_threshold and newline_count == 0

def has_repeating_sentences(text, min_repeats=10):
    filtered_text = filter_dash_lines(text)
    sentences = [s.strip() for s in re.split(r'[.!?]+', filtered_text) 
                 if s.strip() and len(s.strip()) > 20]
    
    if len(sentences) < min_repeats:
        return False
    
    counter = Counter(sentences)
    
    for sent, count in counter.items():
        if count >= min_repeats and len(sent) > 50:
            if not any(pattern in sent.lower() for pattern in ['said', 'asked', 'replied', 'thought']):
                return True
    return False

def is_korean_separator_pattern(text, excluded_chars=None):
    """Check if text is a Korean separator pattern like [ㅡㅡㅡㅡㅡ]"""
    if excluded_chars is None:
        excluded_chars = KOREAN_SEPARATOR_CHARS
    
    # Remove brackets and spaces
    cleaned = text.strip().strip('[]').strip()
    
    if not cleaned:
        return False
    
    # Check if all characters are separators or excluded characters
    return all(c in excluded_chars or c.isspace() for c in cleaned)

def detect_non_english_content(text, qa_settings=None):
    """Detect content not in the target language.
    
    For Latin-based languages (English, Spanish, French, etc.): Uses language detection
    to identify if the text is in the wrong language.
    
    For non-Latin scripts (Korean, Japanese, Chinese, etc.): Checks for characters
    from scripts not matching the target language.
    
    Args:
        text: The text content to check
        qa_settings: Dictionary with 'foreign_char_threshold', 'excluded_characters', 
                    and 'target_language' keys
    
    Returns:
        tuple: (has_issues, list_of_issues)
    """
    if qa_settings is None:
        qa_settings = {'foreign_char_threshold': 10, 'excluded_characters': '', 'target_language': 'english'}
    
    # Get threshold, excluded characters, and target language
    threshold = qa_settings.get('foreign_char_threshold', 10)
    target_language = qa_settings.get('target_language', 'english').lower()
    excluded_chars = set()
    if qa_settings.get('excluded_characters'):
        excluded_chars = set(qa_settings['excluded_characters'].split())
    
    # Combine with existing separator chars
    all_excluded_chars = KOREAN_SEPARATOR_CHARS.copy()
    all_excluded_chars.update(excluded_chars)
    
    issues = []
    filtered_text = filter_dash_lines(text)
    
    # LANGUAGE DETECTION FOR LATIN-BASED LANGUAGES
    # Map language codes to full names
    lang_code_mapping = {
        'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
        'pt': 'Portuguese', 'it': 'Italian', 'ru': 'Russian', 'ja': 'Japanese',
        'ko': 'Korean', 'zh-cn': 'Chinese', 'zh-tw': 'Chinese', 'ar': 'Arabic',
        'he': 'Hebrew', 'th': 'Thai'
    }
    
    # Latin-based languages that need language detection
    latin_languages = ['english', 'spanish', 'french', 'german', 'portuguese', 'italian']
    
    if target_language in latin_languages and len(filtered_text.strip()) > 50:
        # Try to detect the actual language of the text
        try:
            # Use full text for language detection to catch issues anywhere in file
            # Set recursion limit temporarily to catch issues
            import sys
            old_limit = sys.getrecursionlimit()
            try:
                sys.setrecursionlimit(1000)  # Reasonable limit to catch recursion early
                detected_lang = detect(filtered_text)
            finally:
                sys.setrecursionlimit(old_limit)  # Restore original limit
            
            detected_name = lang_code_mapping.get(detected_lang, detected_lang.upper())
            
            # Map target language to expected code
            target_code_mapping = {
                'english': 'en', 'spanish': 'es', 'french': 'fr', 'german': 'de',
                'portuguese': 'pt', 'italian': 'it'
            }
            expected_code = target_code_mapping.get(target_language, 'en')
            
            # If detected language doesn't match target
            if detected_lang != expected_code:
                # Add language mismatch as an issue
                issues.append(f"Language_mismatch_detected_{detected_name}_expected_{target_language.capitalize()}")
                # Return early since we found a language mismatch
                return len(issues) > 0, issues
        except (LangDetectException, RecursionError, RuntimeError) as e:
            # Language detection failed (not enough text, ambiguous, or recursion error)
            # Fall through to script-based detection
            if isinstance(e, RecursionError):
                # Log recursion error but don't crash
                pass  # Silently continue to script-based detection
            pass
    
    # Define all script ranges
    all_script_ranges = [
        (0xAC00, 0xD7AF, 'Korean'), (0x1100, 0x11FF, 'Korean'),
        (0x3130, 0x318F, 'Korean'), (0xA960, 0xA97F, 'Korean'),
        (0xD7B0, 0xD7FF, 'Korean'), (0x3040, 0x309F, 'Japanese'),
        (0x30A0, 0x30FF, 'Japanese'), (0x31F0, 0x31FF, 'Japanese'),
        (0xFF65, 0xFF9F, 'Japanese'), (0x4E00, 0x9FFF, 'Chinese'),
        (0x3400, 0x4DBF, 'Chinese'), (0x20000, 0x2A6DF, 'Chinese'),
        (0x2A700, 0x2B73F, 'Chinese'), (0x0590, 0x05FF, 'Hebrew'),
        (0x0600, 0x06FF, 'Arabic'), (0x0700, 0x074F, 'Syriac'),
        (0x0750, 0x077F, 'Arabic'), (0x0E00, 0x0E7F, 'Thai'),
        (0x0400, 0x04FF, 'Cyrillic'), (0x0500, 0x052F, 'Cyrillic'),
    ]
    
    # Define which scripts are allowed for each target language
    # Latin-based languages allow basic Latin + extensions
    language_script_mapping = {
        'english': [],  # Only Latin (default - everything else is foreign)
        'spanish': [],  # Latin with Spanish diacritics
        'french': [],   # Latin with French diacritics  
        'german': [],   # Latin with German diacritics
        'portuguese': [],  # Latin with Portuguese diacritics
        'italian': [],  # Latin with Italian diacritics
        'russian': ['Cyrillic'],  # Cyrillic only
        'japanese': ['Japanese', 'Chinese'],  # Japanese uses kanji too
        'korean': ['Korean'],
        'chinese': ['Chinese'],
        'arabic': ['Arabic', 'Syriac'],
        'hebrew': ['Hebrew'],
        'thai': ['Thai'],
    }
    
    # Get allowed scripts for target language (empty means Latin-based)
    allowed_scripts = language_script_mapping.get(target_language, [])
    
    # Filter to only check scripts NOT in the target language
    non_target_ranges = [
        (start, end, script) for start, end, script in all_script_ranges
        if script not in allowed_scripts
    ]
    
    script_chars = {}
    total_non_target = 0
    
    # Split text into potential separator patterns and other content
    separator_pattern = r'\[[ㅡ\s―—–\-［］【】〔〕《》「」『』]+\]'
    parts = re.split(f'({separator_pattern})', filtered_text)
    
    for part in parts:
        # Skip if this part is a Korean separator pattern
        if is_korean_separator_pattern(part, all_excluded_chars):
            continue
        
        # Check characters in this part
        for char in part:
            # Skip characters in excluded set
            if char in all_excluded_chars:
                continue
            
            # Skip whitespace and common punctuation
            if char.isspace() or char in '[](){}.,;:!?\'"-':
                continue
                
            code_point = ord(char)
            for start, end, script_name in non_target_ranges:
                if start <= code_point <= end:
                    total_non_target += 1
                    if script_name not in script_chars:
                        script_chars[script_name] = {'count': 0, 'examples': []}
                    script_chars[script_name]['count'] += 1
                    if len(script_chars[script_name]['examples']) < 10:
                        script_chars[script_name]['examples'].append(char)
                    break
    
    # Check against threshold
    if total_non_target > threshold:
        for script, data in script_chars.items():
            examples = ''.join(data['examples'][:5])
            count = data['count']
            issues.append(f"{script}_text_found_{count}_chars_[{examples}]")
    
    return len(issues) > 0, issues

def detect_translation_artifacts(text):
    """Detect common translation/OCR artifacts"""
    artifacts_found = []
    
    for artifact_type, pattern in TRANSLATION_ARTIFACTS.items():
        matches = pattern.findall(text)
        if matches:
            artifacts_found.append({
                'type': artifact_type,
                'count': len(matches),
                'examples': list(set(matches))[:3]
            })
    
    return artifacts_found
    
def detect_glossary_leakage(text, threshold=2):
    """
    Detect if translated text contains raw glossary entries.
    
    Args:
        text: The translated text to check
        threshold: Minimum number of glossary-like patterns to flag as leakage
    
    Returns:
        tuple: (has_leakage, details)
    """
    import re
    
    issues_found = []
    
    # Check for CSV-style glossary headers
    csv_header_pattern = re.compile(
        r'type\s*,\s*raw_name\s*,\s*translated_name\s*,\s*gender\s*,\s*description',
        re.IGNORECASE
    )
    if csv_header_pattern.search(text):
        issues_found.append({
            'type': 'csv_header',
            'severity': 'critical',
            'description': 'Found CSV glossary header in translation'
        })
    
    # Check for multiple structured entries
    entry_patterns = [
        # JSON-like entries
        (r'\{\s*"type"\s*:\s*"[^"]+"\s*,\s*"raw_name"\s*:\s*"[^"]+"\s*,', 'json_entry'),
        # CSV-like entries with Korean/Chinese characters
        (r'(?:character|term)\s*,\s*[가-힣\u4e00-\u9fff]+\s*,\s*[A-Za-z\s]+\s*,', 'csv_entry'),
        # Tab-separated entries
        (r'(?:character|term)\t[가-힣\u4e00-\u9fff]+\t[A-Za-z\s]+\t', 'tsv_entry'),
    ]
    
    for pattern_str, pattern_type in entry_patterns:
        pattern = re.compile(pattern_str, re.IGNORECASE)
        matches = pattern.findall(text)
        if len(matches) >= threshold:
            issues_found.append({
                'type': pattern_type,
                'severity': 'high',
                'count': len(matches),
                'examples': matches[:3],
                'description': f'Found {len(matches)} {pattern_type} glossary entries'
            })
    
    # Check for repeated glossary field names
    field_names = ['type', 'raw_name', 'translated_name', 'gender', 'description']
    field_count = sum(1 for field in field_names if text.lower().count(field) >= 3)
    if field_count >= 3:
        issues_found.append({
            'type': 'repeated_field_names',
            'severity': 'medium',
            'description': f'Found {field_count} repeated glossary field names'
        })
    
    # Check for specific character/term patterns
    char_term_pattern = re.compile(
        r'(?:^|\n)\s*(?:character|term)\s*[,:\t]\s*[^\n]+(?:Male|Female|A\s+historical|Former\s+mayor|Character\s+from)',
        re.IGNORECASE | re.MULTILINE
    )
    char_matches = char_term_pattern.findall(text)
    if len(char_matches) >= 2:
        issues_found.append({
            'type': 'character_definitions',
            'severity': 'high',
            'count': len(char_matches),
            'examples': char_matches[:2],
            'description': f'Found {len(char_matches)} character/term definitions'
        })
    
    has_leakage = len(issues_found) > 0
    
    return has_leakage, issues_found    
    
def extract_semantic_fingerprint(text):
    """Extract semantic fingerprint and signature from text - CACHED VERSION"""
    # For cache efficiency with long texts
    cache_text = text[:50000] if len(text) > 50000 else text
    
    # Extract features for semantic analysis
    words = cache_text.lower().split()
    
    # Character names (words starting with capital letters, appearing multiple times)
    potential_names = re.findall(r'\b[A-Z][a-z]+\b', cache_text)
    name_freq = Counter(potential_names)
    characters = [name for name, count in name_freq.items() 
                  if count >= 3 and name not in COMMON_WORDS]
    
    # Dialogue analysis
    dialogue_matches = re.findall(r'["\"\'""''『』「」]([^"\"\'""''『』「」]+)["\"\'""''『』「」]', cache_text)
    dialogue_count = len(dialogue_matches)
    dialogue_density = dialogue_count / max(1, len(words)) if words else 0
    dialogue_lengths = [len(d) for d in dialogue_matches[:30]]  # First 30 dialogue lengths
    
    # Character frequencies (sorted list)
    character_frequencies = [count for _, count in name_freq.most_common()]
    
    # Speaker sequence extraction
    speaker_patterns = re.findall(r'(\w+)\s+(?:said|asked|replied|shouted|whispered|spoke)', cache_text.lower())
    speaker_sequence = speaker_patterns[:50]  # First 50 speakers
    
    # Paragraph structure (lengths of each paragraph)
    paragraphs = [p for p in cache_text.split('\n\n') if p.strip()]
    paragraph_structure = [len(p) for p in paragraphs[:50]]  # First 50 paragraph lengths
    
    # Action words density
    action_words = len(re.findall(r'\b(\w+ed|spoke|says?|asks?|replies?|shouts?|screams?|whispers?)\b', cache_text))
    action_density = action_words / max(1, len(words)) if words else 0
    
    # Numbers in text
    numbers = re.findall(r'\b\d+\b', cache_text)
    
    # Create fingerprint string
    fingerprint = f"chars:{len(characters)}_dial:{dialogue_density:.2f}_act:{action_density:.2f}_nums:{len(numbers)}_words:{len(words)}"
    
    # Create signature dict
    signature = {
        'characters': characters[:20],  # Top 20 characters
        'dialogue_density': dialogue_density,
        'dialogue_count': dialogue_count,
        'dialogue_lengths': dialogue_lengths,
        'character_frequencies': character_frequencies,
        'speaker_sequence': speaker_sequence,
        'paragraph_structure': paragraph_structure,
        'total_words': len(words),
        'action_density': action_density,
        'numbers': numbers[:50],  # First 50 numbers
        'text_length': len(cache_text)
    }
    
    return fingerprint, signature

# Apply dynamic caching
extract_semantic_fingerprint = lru_cache(maxsize=get_cache_size("semantic_fingerprint") or 2000)(extract_semantic_fingerprint)

def extract_structural_signature(text):
    """Extract structural patterns from text - CACHED VERSION"""
    # For cache efficiency with long texts
    cache_text = text[:50000] if len(text) > 50000 else text
    
    lines = cache_text.split('\n')
    
    # Count different types of lines
    para_count = len([l for l in lines if len(l.strip()) > 50])
    short_lines = len([l for l in lines if 0 < len(l.strip()) < 20])
    empty_lines = len([l for l in lines if not l.strip()])
    
    # Dialogue patterns
    dialogue_lines = len(re.findall(r'["\"\'""''『』「」].*?["\"\'""''『』「」]', cache_text))
    
    # Create pattern string (first letter of each line type)
    pattern = ''
    for line in lines[:100]:  # First 100 lines
        if not line.strip():
            pattern += 'E'  # Empty
        elif len(line.strip()) < 20:
            pattern += 'S'  # Short
        elif re.search(r'["\"\'""''『』「」]', line):
            pattern += 'D'  # Dialogue
        else:
            pattern += 'P'  # Paragraph
    
    # Calculate average paragraph length
    paragraphs = [l for l in lines if len(l.strip()) > 50]
    avg_para_length = sum(len(p) for p in paragraphs) / max(1, len(paragraphs)) if paragraphs else 0
    
    # Dialogue ratio
    dialogue_ratio = dialogue_lines / max(1, len(lines))
    
    signature = {
        'pattern': pattern,
        'paragraph_count': para_count,
        'avg_paragraph_length': avg_para_length,
        'dialogue_ratio': dialogue_ratio,
        'short_lines': short_lines,
        'empty_lines': empty_lines
    }
    
    return signature

def extract_content_fingerprint(text):
    """Extract key sentences that can identify duplicate content - CACHED VERSION"""
    # For cache efficiency with very long texts, limit to first 100KB
    cache_text = text[:100000] if len(text) > 100000 else text
    
    lines = [line.strip() for line in cache_text.split('\n') 
             if len(line.strip()) > 50 and not is_dash_separator_line(line)]
    
    if len(lines) < 5:
        return ""
    
    # Take first, middle, and last substantial sentences
    fingerprint_lines = []
    if len(lines) >= 3:
        fingerprint_lines = [lines[0], lines[len(lines)//2], lines[-1]]
    else:
        fingerprint_lines = lines[:3]
    
    return ' '.join(fingerprint_lines).lower()

# Configure cache size dynamically
extract_content_fingerprint = lru_cache(maxsize=get_cache_size("content_fingerprint"))(extract_content_fingerprint)

def roman_to_int(s):
    """Convert Roman numerals to integer"""
    try:
        values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        result = 0
        for i in range(len(s)):
            if i + 1 < len(s) and values[s[i]] < values[s[i + 1]]:
                result -= values[s[i]]
            else:
                result += values[s[i]]
        return result
    except:
        return None

def extract_chapter_info(filename, text):
    """Extract chapter number and title from filename and content - ENHANCED VERSION"""
    chapter_num = None
    chapter_title = ""
    
    # Enhanced filename patterns - try multiple approaches
    filename_patterns = [
        # Original patterns
        (r"response_(\d+)_(.+?)\.html", 1, 2),
        (r"response_chapter(\d+)\.html", 1, None),
        (r"chapter[\s_-]*(\d+)", 1, None),
        
        # New patterns to catch more cases
        (r"response_(\d{3,4})_", 1, None),  # Catches response_003_
        (r"response_chapter(\d{4})\.html", 1, None),  # Catches response_chapter0002
        (r"(\d{3,4})[_\.]", 1, None),  # General 3-4 digit pattern
        (r"No(\d+)Chapter", 1, None),
        (r"ch[\s_-]*(\d+)", 1, None),
        (r"_(\d+)_", 1, None),
        (r"第(\d+)[章话回]", 1, None),  # Chinese chapter markers
        (r"제(\d+)[장화회]", 1, None),  # Korean chapter markers
    ]
    
    # Try each pattern
    for pattern, num_group, title_group in filename_patterns:
        m = re.search(pattern, filename, re.IGNORECASE)
        if m:
            try:
                # Extract chapter number, removing leading zeros
                chapter_num = int(m.group(num_group).lstrip('0') or '0')
                if title_group and len(m.groups()) >= title_group:
                    chapter_title = m.group(title_group)
                break
            except (ValueError, IndexError):
                continue
    
    # If still no chapter number, try content-based extraction
    if chapter_num is None and text:
        content_patterns = [
            r'Chapter\s+(\d+)',
            r'第\s*(\d+)\s*章',
            r'제\s*(\d+)\s*장',
            r'Chapter\s+([IVXLCDM]+)',  # Roman numerals
            r'\bCh\.?\s*(\d+)',
            r'Episode\s+(\d+)',
            r'Part\s+(\d+)',
        ]
        
        for pattern in content_patterns:
            m = re.search(pattern, text[:1000], re.IGNORECASE)
            if m:
                if m.group(1).isdigit():
                    chapter_num = int(m.group(1))
                else:
                    # Try to convert Roman numerals
                    num = roman_to_int(m.group(1))
                    if num is not None:
                        chapter_num = num
                if chapter_num is not None:
                    break
    
    return chapter_num, chapter_title

def normalize_chapter_numbers(results):
    """Normalize chapter numbers to handle different formats"""
    for result in results:
        # If we have a chapter number, ensure it's normalized
        if result.get('chapter_num') is not None:
            # This helps match chapter 2 with 002, etc.
            result['normalized_chapter_num'] = int(result['chapter_num'])

def fuzzy_match_chapter_numbers(text1, text2, num1, num2):
    """Check if chapter numbers might be the same despite OCR errors"""
    if num1 == num2:
        return True
    
    # Check if numbers are close (OCR might misread)
    if abs(num1 - num2) <= 1:
        # Look for chapter declarations in text
        pattern = r'Chapter\s*(\d+|[IVXLCDM]+)'
        matches1 = re.findall(pattern, text1[:500], re.IGNORECASE)
        matches2 = re.findall(pattern, text2[:500], re.IGNORECASE)
        
        if matches1 and matches2:
            # Try to normalize roman numerals
            def roman_to_int(s):
                try:
                    values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
                    result = 0
                    for i in range(len(s)):
                        if i + 1 < len(s) and values[s[i]] < values[s[i + 1]]:
                            result -= values[s[i]]
                        else:
                            result += values[s[i]]
                    return result
                except:
                    return None
            
            for m1 in matches1:
                for m2 in matches2:
                    if m1.isdigit() and m2.isdigit():
                        if abs(int(m1) - int(m2)) <= 1:
                            return True
                    elif not m1.isdigit() and not m2.isdigit():
                        r1 = roman_to_int(m1.upper())
                        r2 = roman_to_int(m2.upper())
                        if r1 and r2 and abs(r1 - r2) <= 1:
                            return True
    
    return False

def detect_split_chapters(results):
    """Detect chapters that might have been split into multiple files
    Now with better detection to avoid false positives from intentional author formatting
    """
    split_candidates = []
    
    # Common scene break patterns that authors use intentionally
    scene_break_patterns = [
        r'[\*\s]{3,}',           # *** or * * *
        r'[─━－—\-]{3,}',        # Various dashes/lines
        r'[_]{3,}',              # ___
        r'[~～]{3,}',            # ~~~
        r'[=]{3,}',              # ===
        r'[\#]{3,}',             # ###
        r'[\.]{3,}',             # ...
        r'(?:Chapter|Scene|Part)\s+Break', # Explicit break text
        r'(?:Meanwhile|Later|Earlier)',    # Time transition words
        r'\d+\s*(?:hours?|days?|weeks?|months?|years?)\s+(?:later|earlier|ago)', # Time skips
    ]
    
    for i, result in enumerate(results):
        text = result.get('raw_text', '')
        filename = result.get('filename', '')
        
        # Skip if empty
        if not text.strip():
            continue
            
        # Check for continuation indicators from AI
        artifacts = detect_translation_artifacts(text)
        has_continuation = any(a['type'] in ['chapter_continuation', 'split_indicators'] 
                             for a in artifacts)
        
        # Check file naming patterns that suggest systematic splits
        is_systematic_split = False
        split_patterns = [
            r'chunk[\-_]?\d+',          # chunk1, chunk_2
            r'part[\-_]?\d+[\-_]?\d+',   # part1_2 (part 1 of chapter 2)
            r'response_\d+_\d+',         # response_42_3
            r'_\d+of\d+',                # _1of3
            r'_split\d+',                # _split1
            r'_continuation',            # _continuation
        ]
        for pattern in split_patterns:
            if re.search(pattern, filename, re.IGNORECASE):
                is_systematic_split = True
                break
        
        # Check if file is unusually short
        is_short = len(text) < 2000
        
        # Check for scene break indicators at start or end
        text_start = text[:500].strip()
        text_end = text[-500:].strip()
        
        has_scene_break_start = False
        has_scene_break_end = False
        
        for pattern in scene_break_patterns:
            if re.search(pattern, text_start[:100], re.IGNORECASE):
                has_scene_break_start = True
            if re.search(pattern, text_end[-100:], re.IGNORECASE):
                has_scene_break_end = True
        
        # Check if starts mid-sentence (but not after scene break)
        starts_mid = False
        if text.strip() and not has_scene_break_start:
            first_line = text.strip().split('\n')[0].strip()
            # Skip if line starts with dialogue quotes or chapter markers
            if first_line and not re.match(r'^["「『\(\[]', first_line):
                # Check if starts with lowercase (excluding certain words that commonly start sections)
                first_word = first_line.split()[0] if first_line.split() else ''
                transition_words = ['meanwhile', 'however', 'suddenly', 'later', 'earlier', 
                                  'elsewhere', 'afterward', 'afterwards', 'then']
                if first_word.lower() not in transition_words:
                    starts_mid = first_line[0].islower()
        
        # Check if ends mid-sentence (but not with scene break)
        ends_mid = False
        if text.strip() and not has_scene_break_end:
            last_line = text.strip().split('\n')[-1].strip()
            if last_line:
                # Check last character, ignoring quotes
                last_char = last_line.rstrip('」』"\'').rstrip()
                if last_char:
                    ends_mid = last_char[-1] not in '.!?。！？…'
        
        # Determine if this is likely a real split vs intentional formatting
        is_likely_real_split = False
        
        if is_systematic_split:
            # File naming strongly suggests a split
            is_likely_real_split = True
        elif has_continuation:
            # AI detected continuation markers
            is_likely_real_split = True
        elif is_short and starts_mid and ends_mid and not (has_scene_break_start or has_scene_break_end):
            # Short, starts and ends mid-sentence, no scene breaks
            is_likely_real_split = True
        elif is_short and ends_mid and not has_scene_break_end:
            # Might be a split if it's short and ends abruptly
            # Check if it ends with incomplete dialogue or mid-word
            if text.strip():
                # Check for incomplete quotes or mid-word breaks
                if (text.count('"') % 2 != 0 or text.count('「') != text.count('」') or
                    re.search(r'[a-zA-Z]-$', text.strip())):  # Ends with hyphen (mid-word)
                    is_likely_real_split = True
        
        if is_likely_real_split:
            split_candidates.append({
                'index': i,
                'filename': filename,
                'indicators': {
                    'has_continuation': has_continuation,
                    'is_systematic_split': is_systematic_split,
                    'is_short': is_short,
                    'starts_mid': starts_mid,
                    'ends_mid': ends_mid,
                    'has_scene_break_start': has_scene_break_start,
                    'has_scene_break_end': has_scene_break_end
                }
            })
    
    return split_candidates

def create_minhash_index(results, config):
    """Create LSH index for fast similarity lookups"""
    if not MINHASH_AVAILABLE:
        return None, None
    
    threshold = config.get_threshold('minhash_threshold')
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    minhashes = {}
    
    total = len(results)
    for idx, result in enumerate(results):
        if idx % 50 == 0 and idx > 0:
            print(f"   Building MinHash index: {idx}/{total} files processed...")
            
        text = result.get('normalized_text', '')
        if not text:
            continue
            
        # Create MinHash
        m = MinHash(num_perm=128)
        for word in text.split():
            m.update(word.encode('utf8'))
        
        minhashes[result['filename']] = m
        lsh.insert(result['filename'], m)
    
    return lsh, minhashes

def _normalize_text_cached(cache_key):
    """Cached implementation of normalize_text"""
    # This will be called with the actual text
    return cache_key

def normalize_text(text):
    """Normalize text for comparison - CACHED VERSION"""
    normalized = text.lower().strip()
    
    # Remove chapter indicators
    patterns = [
        r'chapter\s*\d+\s*:?\s*', r'第\s*\d+\s*章', r'제\s*\d+\s*장',
        r'chapter\s+[ivxlcdm]+\s*:?\s*', r'\bch\.?\s*\d+\s*:?\s*',
        r'^\s*\d+\s*\.?\s*', r'response_\d+_.*?\.html',
        r'\d{4}-\d{2}-\d{2}', r'\d{2}:\d{2}:\d{2}', r'<[^>]+>'
    ]
    
    for pattern in patterns:
        normalized = re.sub(pattern, '', normalized, flags=re.IGNORECASE | re.MULTILINE)
    
    # Normalize whitespace and punctuation
    normalized = re.sub(r'\s+', ' ', normalized)
    normalized = re.sub(r'[^\w\s]', '', normalized)
    
    return normalized

# Configure cache size dynamically
normalize_text = lru_cache(maxsize=get_cache_size("normalize_text"))(normalize_text)

@lru_cache(maxsize=5000)
def _generate_content_hashes_cached(text_hash):
    """Cached helper for generate_content_hashes"""
    # This is just a placeholder - actual implementation is in the main function
    return text_hash

@lru_cache(maxsize=5000)
def generate_content_hashes(text):
    """Generate multiple hashes for better duplicate detection - CACHED VERSION"""
    # For very long texts, use first 50KB for cache key
    cache_key = _get_cache_key(text, 50000)
    
    normalized = normalize_text(text)
    
    # 1. Raw hash
    raw_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
    
    # 2. Normalized hash
    normalized_hash = hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    # 3. Content fingerprint
    fingerprint = extract_content_fingerprint(text)
    fingerprint_hash = hashlib.md5(fingerprint.encode('utf-8')).hexdigest() if fingerprint else None
    
    # 4. Word frequency hash
    words = re.findall(r'\w+', normalized.lower())
    word_freq = Counter(words)
    significant_words = [(w, c) for w, c in word_freq.most_common(100) 
                        if w not in COMMON_WORDS][:50]
    word_sig = ' '.join([f"{w}:{c}" for w, c in significant_words])
    word_hash = hashlib.md5(word_sig.encode('utf-8')).hexdigest() if word_sig else None
    
    # 5. First chunk hash
    first_chunk = normalized[:1000] if len(normalized) > 1000 else normalized
    first_chunk_hash = hashlib.md5(first_chunk.encode('utf-8')).hexdigest()
    
    # 6. Semantic fingerprint hash - FIXED
    semantic_result = extract_semantic_fingerprint(text)
    if semantic_result and isinstance(semantic_result, tuple) and len(semantic_result) >= 2:
        semantic_str = semantic_result[0]
        semantic_hash = hashlib.md5(semantic_str.encode('utf-8')).hexdigest()
    else:
        # Fallback if function returns unexpected value
        semantic_hash = hashlib.md5(text[:1000].encode('utf-8')).hexdigest()
    
    # 7. Structural signature hash
    structural_sig = extract_structural_signature(text)
    if structural_sig:
        structural_str = json.dumps(structural_sig, sort_keys=True)
        structural_hash = hashlib.md5(structural_str.encode('utf-8')).hexdigest()
    else:
        # Fallback
        structural_hash = hashlib.md5(text[:500].encode('utf-8')).hexdigest()
    
    return {
        'raw': raw_hash,
        'normalized': normalized_hash,
        'fingerprint': fingerprint_hash,
        'word_freq': word_hash,
        'first_chunk': first_chunk_hash,
        'semantic': semantic_hash,
        'structural': structural_hash
    }

@lru_cache(maxsize=20000)
def _calculate_similarity_ratio_cached(text1_hash, text2_hash):
    """Cached helper for similarity ratio"""
    return (text1_hash, text2_hash)

@lru_cache(maxsize=20000)
def calculate_similarity_ratio(text1, text2):
    """Calculate similarity with optimizations for large texts - CACHED VERSION"""
    # Ensure consistent ordering for cache
    if text1 > text2:
        text1, text2 = text2, text1
    
    len_ratio = len(text1) / max(1, len(text2))
    if len_ratio < 0.7 or len_ratio > 1.3:
        return 0.0
    
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
        similarities = [SequenceMatcher(None, s1, s2).ratio() for s1, s2 in zip(samples1, samples2)]
        return sum(similarities) / len(similarities)
    else:
        return SequenceMatcher(None, text1, text2).ratio()

# Configure cache size dynamically
calculate_similarity_ratio = lru_cache(maxsize=get_cache_size("similarity_ratio"))(calculate_similarity_ratio)

# This function should NOT be cached directly
def calculate_semantic_similarity(sig1, sig2):
    """Calculate similarity between two semantic signatures
    This wrapper handles dict inputs and calls the cached implementation
    """
    # Convert dicts to JSON strings
    if isinstance(sig1, dict):
        sig1_json = json.dumps(sig1, sort_keys=True)
    else:
        sig1_json = sig1
        
    if isinstance(sig2, dict):
        sig2_json = json.dumps(sig2, sort_keys=True)
    else:
        sig2_json = sig2
    
    # Call the cached implementation with JSON strings
    return _calculate_semantic_similarity_cached(sig1_json, sig2_json)

# This function IS cached because it only receives JSON strings
def _calculate_semantic_similarity_cached(sig1_json, sig2_json):
    """Cached implementation that works with JSON strings"""
    sig1 = json.loads(sig1_json)
    sig2 = json.loads(sig2_json)
    
    # Character overlap
    chars1 = set(sig1.get('characters', []))
    chars2 = set(sig2.get('characters', []))
    char_overlap = len(chars1 & chars2) / max(1, len(chars1 | chars2))
    
    # Dialogue density similarity
    dial_sim = 1 - abs(sig1.get('dialogue_density', 0) - sig2.get('dialogue_density', 0))
    
    # Action density similarity
    act_sim = 1 - abs(sig1.get('action_density', 0) - sig2.get('action_density', 0))
    
    # Number overlap
    nums1 = set(sig1.get('numbers', []))
    nums2 = set(sig2.get('numbers', []))
    num_overlap = len(nums1 & nums2) / max(1, len(nums1 | nums2)) if nums1 or nums2 else 1
    
    # Length similarity
    len_ratio = min(sig1.get('text_length', 1), sig2.get('text_length', 1)) / max(1, max(sig1.get('text_length', 1), sig2.get('text_length', 1)))
    
    # Weighted average
    return (char_overlap * 0.4 + dial_sim * 0.2 + act_sim * 0.2 + num_overlap * 0.1 + len_ratio * 0.1)

# Apply caching ONLY to the implementation function, NOT the wrapper
_calculate_semantic_similarity_cached = lru_cache(maxsize=get_cache_size("semantic_similarity") or 5000)(_calculate_semantic_similarity_cached)

# Make sure calculate_semantic_similarity is NOT cached
# If there's any line like this, REMOVE IT:
# calculate_semantic_similarity = lru_cache(...)(calculate_semantic_similarity)


def calculate_semantic_fingerprint_similarity(text1, text2):
    """Calculate similarity based on semantic structure rather than exact wording - CACHED VERSION"""
    # For very long texts, truncate for cache efficiency
    cache_text1 = text1[:100000] if len(text1) > 100000 else text1
    cache_text2 = text2[:100000] if len(text2) > 100000 else text2
    
    fingerprint1, sig1 = extract_semantic_fingerprint(cache_text1)
    fingerprint2, sig2 = extract_semantic_fingerprint(cache_text2)
    
    similarities = []
    
    # Compare dialogue structure (very reliable indicator)
    if sig1['dialogue_count'] > 0 and sig2['dialogue_count'] > 0:
        dialogue_ratio = min(sig1['dialogue_count'], sig2['dialogue_count']) / max(sig1['dialogue_count'], sig2['dialogue_count'])
        similarities.append(dialogue_ratio)
        
        # Compare dialogue length patterns
        if sig1['dialogue_lengths'] and sig2['dialogue_lengths']:
            len_similarity = SequenceMatcher(None, sig1['dialogue_lengths'][:30], sig2['dialogue_lengths'][:30]).ratio()
            similarities.append(len_similarity)
    
    # Compare character lists (names should mostly match)
    if sig1['characters'] and sig2['characters']:
        char_set1 = set(sig1['characters'])
        char_set2 = set(sig2['characters'])
        char_overlap = len(char_set1 & char_set2) / max(len(char_set1), len(char_set2))
        similarities.append(char_overlap)
        
        # Compare character frequency patterns
        freq_similarity = SequenceMatcher(None, sig1['character_frequencies'], sig2['character_frequencies']).ratio()
        similarities.append(freq_similarity * 0.8)  # Slightly less weight
    
    # Compare numbers (very reliable - numbers rarely change)
    if sig1['numbers'] and sig2['numbers']:
        num_set1 = set(sig1['numbers'])
        num_set2 = set(sig2['numbers'])
        num_overlap = len(num_set1 & num_set2) / max(len(num_set1), len(num_set2))
        similarities.append(num_overlap)
    
    # Compare speaker sequences
    if len(sig1['speaker_sequence']) >= 5 and len(sig2['speaker_sequence']) >= 5:
        seq_similarity = SequenceMatcher(None, sig1['speaker_sequence'], sig2['speaker_sequence']).ratio()
        similarities.append(seq_similarity)
    
    # Compare paragraph structure
    if len(sig1['paragraph_structure']) >= 10 and len(sig2['paragraph_structure']) >= 10:
        # Allow for some variation in lengths (±20%)
        para_similarities = []
        for i in range(min(len(sig1['paragraph_structure']), len(sig2['paragraph_structure']))):
            len1 = sig1['paragraph_structure'][i]
            len2 = sig2['paragraph_structure'][i]
            if len1 > 0 and len2 > 0:
                ratio = min(len1, len2) / max(len1, len2)
                para_similarities.append(1.0 if ratio > 0.8 else ratio)
        
        if para_similarities:
            similarities.append(sum(para_similarities) / len(para_similarities))
    
    # Word count ratio (should be similar)
    word_ratio = min(sig1['total_words'], sig2['total_words']) / max(sig1['total_words'], sig2['total_words'])
    similarities.append(word_ratio * 0.5)  # Less weight
    
    # Calculate weighted average
    if similarities:
        return sum(similarities) / len(similarities)
    else:
        return 0.0

# Configure cache size dynamically
calculate_semantic_fingerprint_similarity = lru_cache(maxsize=get_cache_size("semantic_fingerprint"))(calculate_semantic_fingerprint_similarity)

# This function should NOT be cached directly - it's the wrapper
def calculate_structural_similarity(struct1, struct2):
    """Calculate similarity between two structural signatures
    This wrapper handles dict inputs and calls the cached implementation
    """
    # Convert dicts to JSON strings
    if isinstance(struct1, dict):
        struct1_json = json.dumps(struct1, sort_keys=True)
    else:
        struct1_json = struct1
        
    if isinstance(struct2, dict):
        struct2_json = json.dumps(struct2, sort_keys=True)
    else:
        struct2_json = struct2
    
    # Call the cached implementation with JSON strings
    return _calculate_structural_similarity_cached(struct1_json, struct2_json)

# This function IS cached because it only receives JSON strings
def _calculate_structural_similarity_cached(struct1_json, struct2_json):
    """Cached implementation that works with JSON strings"""
    # Convert JSON strings back to dictionaries
    struct1 = json.loads(struct1_json)
    struct2 = json.loads(struct2_json)
    
    # Pattern similarity
    pattern_sim = SequenceMatcher(None, struct1.get('pattern', ''), struct2.get('pattern', '')).ratio()
    
    # Paragraph count similarity
    para_ratio = min(struct1.get('paragraph_count', 1), struct2.get('paragraph_count', 1)) / \
                 max(1, max(struct1.get('paragraph_count', 1), struct2.get('paragraph_count', 1)))
    
    # Average paragraph length similarity
    len_ratio = min(struct1.get('avg_paragraph_length', 1), struct2.get('avg_paragraph_length', 1)) / \
                max(1, max(struct1.get('avg_paragraph_length', 1), struct2.get('avg_paragraph_length', 1)))
    
    # Dialogue ratio similarity
    dial_sim = 1 - abs(struct1.get('dialogue_ratio', 0) - struct2.get('dialogue_ratio', 0))
    
    # Weighted average
    return (pattern_sim * 0.5 + para_ratio * 0.2 + len_ratio * 0.15 + dial_sim * 0.15)

# Apply caching ONLY to the implementation function, NOT the wrapper
_calculate_structural_similarity_cached = lru_cache(maxsize=get_cache_size("structural_similarity") or 5000)(_calculate_structural_similarity_cached)

# Note: cache configurations are already applied earlier in the file

def extract_chapter_title(text):
    """Extract chapter title from text"""
    patterns = [
        r'Chapter\s+\d+\s*:\s*([^\n\r]+)',
        r'Chapter\s+\d+\s+([^\n\r]+)',
        r'第\s*\d+\s*章\s*[:：]?\s*([^\n\r]+)',
        r'제\s*\d+\s*장\s*[:：]?\s*([^\n\r]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text[:500], re.IGNORECASE)
        if match:
            title = match.group(1).strip()
            title = re.sub(r'\s+', ' ', title)
            title = title.split('.')[0].split('The')[0].strip()
            return title[:100] if len(title) > 100 else title
    
    return None

def merge_duplicate_groups(duplicate_groups, filename1, filename2):
    """Intelligently merge duplicate groups when new connections are found
    
    Note: When called from parallel processing, should be wrapped with a lock
    """
    group1 = duplicate_groups.get(filename1)
    group2 = duplicate_groups.get(filename2)
    
    if group1 is None and group2 is None:
        # Create new group
        new_group = max(duplicate_groups.values(), default=-1) + 1
        duplicate_groups[filename1] = new_group
        duplicate_groups[filename2] = new_group
    elif group1 is not None and group2 is None:
        # Add to existing group
        duplicate_groups[filename2] = group1
    elif group1 is None and group2 is not None:
        # Add to existing group
        duplicate_groups[filename1] = group2
    elif group1 != group2:
        # Merge two groups
        min_group = min(group1, group2)
        max_group = max(group1, group2)
        for filename, group in duplicate_groups.items():
            if group == max_group:
                duplicate_groups[filename] = min_group


def process_enhance_duplicate_batch(args):
    """Process a batch of enhanced duplicate detection - MUST BE AT MODULE LEVEL"""
    batch_type, batch_data, worker_data = args
    batch_results = []
    
    # Import what we need
    from difflib import SequenceMatcher
    import hashlib
    
    # Local caches for this worker
    similarity_cache = {}
    preview_cache = {}
    
    if batch_type == 'chapter_comparison':
        # Process chapter number group comparisons
        comparisons = batch_data
        text_data = worker_data['text_data']
        threshold = worker_data['similarity_threshold']
        
        for idx1, idx2, file1, file2, chapter_num in comparisons:
            # Get text data
            data1 = text_data[idx1]
            data2 = text_data[idx2]
            
            # Create cache key (handle None hashes)
            if data1['hash'] is None or data2['hash'] is None:
                continue  # Skip if either file is empty
            
            cache_key = (min(data1['hash'], data2['hash']), max(data1['hash'], data2['hash']))
            
            if cache_key in similarity_cache:
                similarity = similarity_cache[cache_key]
            else:
                # Check if hashes are identical
                if data1['hash'] == data2['hash']:
                    similarity = 1.0
                else:
                    # Calculate similarity
                    similarity = calculate_similarity_ratio(data1['text'], data2['text'])
                
                similarity_cache[cache_key] = similarity
            
            if similarity >= threshold:
                batch_results.append({
                    'type': 'chapter_duplicate',
                    'file1': file1,
                    'file2': file2,
                    'chapter': chapter_num,
                    'similarity': similarity,
                    'preview1': data1['text'][:100],
                    'preview2': data2['text'][:100]
                })
    
    elif batch_type == 'preview_comparison':
        # Process preview-based comparisons
        comparisons = batch_data
        text_data = worker_data['text_data']
        preview_data = worker_data['preview_data']
        threshold = worker_data['similarity_threshold']
        preview_threshold = worker_data['preview_threshold']
        
        for idx1, idx2, file1, file2 in comparisons:
            # First check preview similarity
            preview1 = preview_data[idx1]
            preview2 = preview_data[idx2]
            
            # Normalize previews (first 50 words)
            norm_preview1 = ' '.join(preview1['text'].split()[:50])
            norm_preview2 = ' '.join(preview2['text'].split()[:50])
            
            # Check preview similarity (handle None hashes)
            if preview1['hash'] is None or preview2['hash'] is None:
                continue  # Skip if either preview is empty
            
            preview_cache_key = (min(preview1['hash'], preview2['hash']), 
                               max(preview1['hash'], preview2['hash']))
            
            if preview_cache_key in preview_cache:
                preview_sim = preview_cache[preview_cache_key]
            else:
                preview_sim = calculate_similarity_ratio(norm_preview1[:500], norm_preview2[:500])
                preview_cache[preview_cache_key] = preview_sim
            
            # If previews are similar enough, check full text
            if preview_sim >= preview_threshold:
                # Get full text data
                data1 = text_data[idx1]
                data2 = text_data[idx2]
                
                # Check full text similarity (handle None hashes)
                if data1['hash'] is None or data2['hash'] is None:
                    continue  # Skip if either file is empty
                
                cache_key = (min(data1['hash'], data2['hash']), max(data1['hash'], data2['hash']))
                
                if cache_key in similarity_cache:
                    similarity = similarity_cache[cache_key]
                else:
                    if data1['hash'] == data2['hash']:
                        similarity = 1.0
                    else:
                        similarity = calculate_similarity_ratio(data1['text'], data2['text'])
                    
                    similarity_cache[cache_key] = similarity
                
                if similarity >= threshold:
                    batch_results.append({
                        'type': 'misnamed_duplicate',
                        'file1': file1,
                        'file2': file2,
                        'chapter': f"misnamed_{data1.get('chapter_num', '?')}_vs_{data2.get('chapter_num', '?')}",
                        'similarity': similarity,
                        'preview_similarity': preview_sim
                    })
    
    return batch_results


def enhance_duplicate_detection(results, duplicate_groups, duplicate_confidence, config, log, should_stop=None):
    """Additional duplicate detection - PROCESSPOOLEXECUTOR VERSION"""
    
    log("🔍 Enhanced duplicate detection (different naming formats)...")
    log("⚡ PROCESSPOOLEXECUTOR ENABLED - MAXIMUM PERFORMANCE!")
    
    # Determine number of workers
    cpu_count = multiprocessing.cpu_count()
    max_workers_config = 0
    
    try:
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                full_config = json.load(f)
                # Check multiple possible config locations
                qa_config = full_config.get('qa_scanner_config', {})
                ai_hunter_config = full_config.get('ai_hunter_config', {})
                
                # Priority: qa_scanner_config > ai_hunter_config
                max_workers_config = qa_config.get('max_workers',
                                    ai_hunter_config.get('ai_hunter_max_workers', 1))
    except:
        max_workers_config = 0
    
    if max_workers_config > 0:
        max_workers = min(max_workers_config, cpu_count)
        log(f"   🖥️ Using {max_workers} parallel processes (configured limit)")
    else:
        max_workers = cpu_count
        log(f"   🚀 Using ALL {max_workers} CPU cores for enhanced detection")
        if cpu_count > 8:
            log(f"   💡 Tip: You can limit CPU cores in QA scanner settings")
    
    # Pre-compute all data
    log("   📊 Pre-computing text and preview data...")
    
    text_data = {}
    preview_data = {}
    
    for i, result in enumerate(results):
        # Text data (first 5000 chars)
        text = result.get('raw_text', '')[:5000]
        text_data[i] = {
            'text': text,
            'hash': hashlib.md5(text.encode()).hexdigest() if text else None,
            'length': len(text),
            'chapter_num': result.get('chapter_num')
        }
        
        # Preview data (first 1000 chars)
        preview = result.get('raw_text', '')[:1000].strip()
        preview_data[i] = {
            'text': preview,
            'hash': hashlib.md5(preview.encode()).hexdigest() if preview else None
        }
    
    # First, normalize all chapter numbers
    normalize_chapter_numbers(results)
    
    # PART 1: Group by normalized chapter number
    log("   📚 Checking files with same chapter numbers...")
    
    chapter_groups = {}
    for i, result in enumerate(results):
        if result.get('normalized_chapter_num') is not None:
            num = result['normalized_chapter_num']
            if num not in chapter_groups:
                chapter_groups[num] = []
            chapter_groups[num].append((i, result))
    
    # Create comparison tasks for chapter groups
    chapter_comparisons = []
    for chapter_num, group in chapter_groups.items():
        if len(group) > 1:
            log(f"   └─ Found {len(group)} files for chapter {chapter_num}")
            
            # Create all pair comparisons for this group
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    idx1, result1 = group[i]
                    idx2, result2 = group[j]
                    chapter_comparisons.append((
                        idx1, idx2, 
                        result1['filename'], result2['filename'],
                        chapter_num
                    ))
    
    # Process chapter comparisons in batches
    duplicates_found = []
    
    if chapter_comparisons:
        log(f"   📋 Processing {len(chapter_comparisons)} chapter comparisons...")
        
        # Prepare worker data
        worker_data = {
            'text_data': text_data,
            'similarity_threshold': config.get_threshold('similarity')
        }
        
        # Create batches
        batch_size = max(100, len(chapter_comparisons) // max_workers)
        batches = []
        
        for i in range(0, len(chapter_comparisons), batch_size):
            batch = chapter_comparisons[i:i + batch_size]
            batches.append(('chapter_comparison', batch, worker_data))
        
        # Process with ProcessPoolExecutor
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker_process) as executor:
            futures = []
            
            for batch_args in batches:
                if should_stop and should_stop():
                    log("⛔ Enhanced detection interrupted by user.")
                    executor.shutdown(wait=True)
                    return duplicates_found
                
                future = executor.submit(process_enhance_duplicate_batch, batch_args)
                futures.append(future)
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                batch_results = future.result()
                
                # Process results
                for result in batch_results:
                    if result['type'] == 'chapter_duplicate':
                        # Update duplicate groups
                        with merge_lock:
                            merge_duplicate_groups(duplicate_groups, 
                                                 result['file1'], 
                                                 result['file2'])
                            pair = tuple(sorted([result['file1'], result['file2']]))
                            duplicate_confidence[pair] = max(
                                duplicate_confidence.get(pair, 0), 
                                result['similarity']
                            )
                        
                        duplicates_found.append(result)
                        
                        log(f"      ✓ DUPLICATE: {result['file1']} ≈ {result['file2']} "
                            f"({int(result['similarity']*100)}%)")
                        log(f"      Preview 1: {result['preview1']}...")
                        log(f"      Preview 2: {result['preview2']}...")
    
    # PART 2: Check for misnamed files
    log("🔍 Checking for misnamed chapters (content vs filename mismatch)...")
    
    # Create preview-based comparison tasks
    preview_comparisons = []
    total_files = len(results)
    
    # We need to check all pairs, but we can filter some obvious non-matches
    for i in range(total_files):
        if i % 100 == 0 and i > 0:
            log(f"   📊 Creating preview comparisons: {i}/{total_files} files...")
        
        for j in range(i + 1, total_files):
            # Skip if:
            # 1. Already in same duplicate group
            if (results[i]['filename'] in duplicate_groups and 
                results[j]['filename'] in duplicate_groups and
                duplicate_groups[results[i]['filename']] == duplicate_groups[results[j]['filename']]):
                continue
            
            # 2. Both have same chapter number (already checked above)
            if (results[i].get('normalized_chapter_num') is not None and 
                results[j].get('normalized_chapter_num') is not None and
                results[i]['normalized_chapter_num'] == results[j]['normalized_chapter_num']):
                continue
            
            # 3. Text lengths are very different (handle None/empty texts)
            len1 = text_data[i]['length']
            len2 = text_data[j]['length']
            if len1 == 0 or len2 == 0:
                continue  # Skip empty files
            
            len_ratio = min(len1, len2) / max(len1, len2)
            if len_ratio < 0.7:  # Skip if lengths differ by more than 30%
                continue
            
            preview_comparisons.append((i, j, results[i]['filename'], results[j]['filename']))
    
    if preview_comparisons:
        log(f"   📋 Processing {len(preview_comparisons)} preview comparisons...")
        
        # Prepare worker data
        worker_data = {
            'text_data': text_data,
            'preview_data': preview_data,
            'similarity_threshold': config.get_threshold('similarity'),
            'preview_threshold': 0.9  # High threshold for preview matching
        }
        
        # Create batches
        batch_size = max(500, len(preview_comparisons) // (max_workers * 10))
        batches = []
        
        for i in range(0, len(preview_comparisons), batch_size):
            batch = preview_comparisons[i:i + batch_size]
            batches.append(('preview_comparison', batch, worker_data))
        
        # Process with ProcessPoolExecutor
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker_process) as executor:
            futures = []
            
            for batch_args in batches:
                if should_stop and should_stop():
                    log("⛔ Enhanced detection interrupted by user.")
                    executor.shutdown(wait=True)
                    return duplicates_found
                
                future = executor.submit(process_enhance_duplicate_batch, batch_args)
                futures.append(future)
            
            # Collect results with progress
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                completed += 1
                if completed % 10 == 0:
                    log(f"   📊 Preview comparison progress: {completed}/{len(futures)} batches")
                
                batch_results = future.result()
                
                # Process results
                for result in batch_results:
                    if result['type'] == 'misnamed_duplicate':
                        # Update duplicate groups
                        with merge_lock:
                            merge_duplicate_groups(duplicate_groups, 
                                                 result['file1'], 
                                                 result['file2'])
                            pair = tuple(sorted([result['file1'], result['file2']]))
                            duplicate_confidence[pair] = max(
                                duplicate_confidence.get(pair, 0), 
                                result['similarity']
                            )
                        
                        duplicates_found.append(result)
                        
                        log(f"      ✓ Found misnamed duplicate: {result['file1']} ≈ {result['file2']} "
                            f"({int(result['similarity']*100)}%)")
    
    log(f"✅ Enhanced detection complete! Found {len(duplicates_found)} duplicates")
    
    return duplicates_found
    
def detect_duplicates(results, log, should_stop, config):
    """Detect duplicates using multiple strategies with enhanced methods - PERFORMANCE OPTIMIZED"""
    duplicate_groups = {}
    near_duplicate_groups = {}
    duplicate_confidence = defaultdict(float)
    
    total_files = len(results)
    dup_start_time = time.time()  # Track timing for progress estimates
    # Initialize comparisons_done at the function level
    comparisons_done = 0
    
    # Get minimum word count threshold for duplicate detection (default 500)
    # This prevents small files (sections, notices) from being flagged as duplicates
    min_dup_words = config.thresholds.get(config.mode, {}).get('min_duplicate_word_count', 500)
    
    # Filter out files that are too small for duplicate detection
    # These are likely section headers, notices, or metadata files
    results_for_dup_check = []
    skipped_small_files = []
    
    for result in results:
        text = result.get('raw_text', '')
        word_count = len(text.split())
        
        if word_count < min_dup_words:
            skipped_small_files.append(result['filename'])
        else:
            results_for_dup_check.append(result)
    
    if skipped_small_files:
        log(f"⏭️  Skipping {len(skipped_small_files)} files with <{min_dup_words} words (likely sections/notices)")
    
    # Use filtered results for duplicate detection
    total_files = len(results_for_dup_check)
    if total_files == 0:
        log("⚠️ No files with sufficient word count for duplicate detection")
        return duplicate_groups, near_duplicate_groups, duplicate_confidence
    
    # Create local cached functions for this detection run
    @lru_cache(maxsize=10000)
    def compare_texts_cached(text1_hash, text2_hash, max_length=2000):
        """Cached text comparison"""
        # Find texts by hash
        text1, text2 = None, None
        for result in results_for_dup_check:
            text = result.get('raw_text', '')[:max_length]
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if text_hash == text1_hash:
                text1 = text
            if text_hash == text2_hash:
                text2 = text
        
        if text1 and text2:
            return calculate_similarity_ratio(text1, text2)
        return 0.0
    
    # Pre-compute text hashes for caching
    text_hashes = {}
    for idx, result in enumerate(results_for_dup_check):
        text = result.get('raw_text', '')
        text_hashes[idx] = {
            'hash_2k': hashlib.md5(text[:2000].encode()).hexdigest() if len(text) >= 2000 else None,
            'hash_5k': hashlib.md5(text[:5000].encode()).hexdigest() if len(text) >= 5000 else None,
            'full_text': text
        }
    
    # Extract additional signatures for filtered results
    log("🔍 Extracting semantic and structural signatures...")
    for idx, result in enumerate(results_for_dup_check):
        if should_stop():
            log("⛔ Signature extraction interrupted by user.")
            return duplicate_groups, near_duplicate_groups, duplicate_confidence
            
        if idx % 10 == 0:
            progress = int((idx / total_files) * 100)
            log(f"   📊 Progress: {idx}/{total_files} files ({progress}%)")
            
        text = result.get('raw_text', '')
        _, semantic_sig = extract_semantic_fingerprint(text)
        structural_sig = extract_structural_signature(text)
        result['semantic_sig'] = semantic_sig
        result['structural_sig'] = structural_sig
        result['normalized_text'] = normalize_text(text)
    
    # Create MinHash index if available
    lsh, minhashes = None, None
    if MINHASH_AVAILABLE and len(results_for_dup_check) > 50:  # Use MinHash for larger datasets
        log("🔍 Building MinHash index for fast similarity detection...")
        lsh, minhashes = create_minhash_index(results_for_dup_check, config)
    
    # 1. Hash-based detection (exact and near-exact matches)
    content_hashes = defaultdict(lambda: defaultdict(list))
    
    for idx, result in enumerate(results_for_dup_check):
        hashes = result['hashes']
        file_info = {
            'filename': result['filename'],
            'idx': idx,
            'chapter_num': result['chapter_num'],
            'result': result
        }
        
        for hash_type, hash_value in hashes.items():
            if hash_value:
                content_hashes[hash_type][hash_value].append(file_info)
    
    # Multiple levels of duplicate detection
    duplicate_detection_levels = [
        ("exact content", 'raw', 1.0),
        ("normalized content", 'normalized', 0.95),
        ("semantic fingerprint", 'semantic', 0.85),
        ("structural pattern", 'structural', 0.80),
        ("first 1000 characters", 'first_chunk', 0.90),
        ("content fingerprints", 'fingerprint', 0.85),
        ("word frequency patterns", 'word_freq', 0.75)
    ]
    
    for level_name, hash_type, confidence in duplicate_detection_levels:
        log(f"🔍 Checking {level_name}...")
        for hash_value, files in content_hashes[hash_type].items():
            if len(files) > 1:
                for i in range(len(files)):
                    for j in range(i + 1, len(files)):
                        merge_duplicate_groups(duplicate_groups, 
                                             files[i]['filename'], 
                                             files[j]['filename'])
                        duplicate_confidence[(files[i]['filename'], files[j]['filename'])] = max(
                            duplicate_confidence[(files[i]['filename'], files[j]['filename'])],
                            confidence
                        )
                log(f"   └─ Found {len(files)} files with identical {level_name}")
    
    # 2. Enhanced duplicate detection for different naming formats
    log("🔍 Checking for same chapters with different naming...")
    enhance_duplicate_detection(results_for_dup_check, duplicate_groups, duplicate_confidence, config, log, should_stop)
    
    # 3. MinHash-based detection (if available)
    if lsh:
        log("🔍 Performing MinHash similarity detection...")
        for result in results_for_dup_check:
            if result['filename'] in minhashes:
                candidates = lsh.query(minhashes[result['filename']])
                for candidate in candidates:
                    if candidate != result['filename']:
                        # Calculate exact Jaccard similarity
                        jaccard = minhashes[result['filename']].jaccard(minhashes[candidate])
                        if jaccard >= config.get_threshold('minhash_threshold'):
                            merge_duplicate_groups(duplicate_groups, result['filename'], candidate)
                            duplicate_confidence[(result['filename'], candidate)] = jaccard
    
    # 4. Semantic similarity check - OPTIMIZED
    log("🔍 Checking semantic similarity...")
    semantic_threshold = config.get_threshold('semantic')

    # Use MinHash candidates for semantic checking if available
    if lsh and config.mode != 'ai-hunter':
        log("🚀 Using MinHash optimization for faster semantic checking...")
        checked_count = 0
        
        # For non-AI Hunter modes, use MinHash to limit comparisons
        for result in results_for_dup_check:
            if should_stop():
                log("⛔ Semantic check interrupted by user.")
                break
            
            checked_count += 1
            if checked_count % 10 == 0:
                log(f"   📊 MinHash semantic check: {checked_count}/{len(results_for_dup_check)} files processed...")
                
            if result['filename'] in minhashes:
                candidates = lsh.query(minhashes[result['filename']])
                for candidate_filename in candidates:
                    if candidate_filename == result['filename']:
                        continue
                    
                    # Find the candidate result
                    candidate_result = next((r for r in results_for_dup_check if r['filename'] == candidate_filename), None)
                    if not candidate_result:
                        continue
                    
                    # Skip if already in same group
                    if (result['filename'] in duplicate_groups and 
                        candidate_filename in duplicate_groups and
                        duplicate_groups[result['filename']] == duplicate_groups[candidate_filename]):
                        continue
                    
                    sem_sim = calculate_semantic_similarity(result['semantic_sig'], 
                                                          candidate_result['semantic_sig'])
                    if sem_sim >= semantic_threshold:
                        struct_sim = calculate_structural_similarity(result['structural_sig'],
                                                                    candidate_result['structural_sig'])
                        
                        if struct_sim >= config.get_threshold('structural'):
                            merge_duplicate_groups(duplicate_groups, 
                                                 result['filename'], 
                                                 candidate_filename)
                            confidence = (sem_sim + struct_sim) / 2
                            duplicate_confidence[(result['filename'], candidate_filename)] = confidence
                            log(f"   └─ Semantic match: {result['filename']} ≈ {candidate_filename} "
                                f"(sem: {int(sem_sim*100)}%, struct: {int(struct_sim*100)}%)")
    
    # AI Hunter mode or fallback: check all pairs
    # Skip AI Hunter in quick scan mode
    if config.mode == 'quick-scan':
        log("   ⚡ Skipping AI Hunter checks for quick scan mode")
    else:
        # AI Hunter mode or fallback: check all pairs
        if config.mode == 'ai-hunter' or not lsh:
            if config.mode == 'ai-hunter':
                log("🤖 AI Hunter mode: Enhanced semantic and structural checking active")
                log("   ⚠️ This will check ALL file pairs - may take several minutes for large datasets")
                
                total_comparisons = (len(results) * (len(results) - 1)) // 2
                log(f"   [DEBUG] Total comparisons to perform: {total_comparisons:,}")
                
                ai_start_time = time.time()  # Use local timer for AI Hunter
                
                # Initialize last_progress HERE for AI Hunter mode
                last_progress = 0  # ADD THIS LINE
                
                # Use parallel processing for AI Hunter
                comparisons_done = parallel_ai_hunter_check(results, duplicate_groups, duplicate_confidence, 
                                                          config, log, should_stop)
                
                # Log AI Hunter completion stats
                ai_time = time.time() - ai_start_time
                log(f"   [DEBUG] AI Hunter took {ai_time:.2f} seconds")
                if comparisons_done and comparisons_done > 0:
                    log(f"   [DEBUG] Comparisons/second: {int(comparisons_done/max(ai_time, 1)):,}")
                    
                # AI HUNTER IS DONE - DO NOT CONTINUE TO SEQUENTIAL CODE
                
            else:
                # Keep the original sequential code for when there's no LSH and not in AI Hunter mode
                log("⚠️ No MinHash index available - checking all pairs (slower)")
                
                total_comparisons = (len(results) * (len(results) - 1)) // 2
                comparisons_done = 0
                last_progress = 0  # This is already here for sequential mode
                ai_start_time = time.time()  # Use local timer
                
                # MOVE ALL THE SEQUENTIAL CODE HERE - INDENTED UNDER THIS ELSE BLOCK
                
                # Create cached AI Hunter comparison
                @lru_cache(maxsize=10000)
                def ai_hunter_check_cached(idx1, idx2):
                    """Cached AI Hunter check"""
                    sem_sim = calculate_semantic_similarity(results[idx1]['semantic_sig'], 
                                                          results[idx2]['semantic_sig'])
                    struct_sim = calculate_structural_similarity(results[idx1]['structural_sig'],
                                                               results[idx2]['structural_sig'])
                    
                    # Quick text check
                    hash1 = text_hashes[idx1]['hash_2k']
                    hash2 = text_hashes[idx2]['hash_2k']
                    if hash1 and hash2:
                        if hash1 > hash2:
                            hash1, hash2 = hash2, hash1
                        text_sim = compare_texts_cached(hash1, hash2, 2000)
                    else:
                        text_sim = 0.0
                    
                    return sem_sim, struct_sim, text_sim
                
                # Check EVERY pair of files
                for i in range(len(results)):
                    if should_stop():
                        log("⛔ Semantic check interrupted by user.")
                        break
                    
                    for j in range(i + 1, len(results)):
                        comparisons_done += 1
                        
                        # Show progress every 5%
                        progress = int((comparisons_done / total_comparisons) * 100)
                        if progress >= last_progress + 5:
                            elapsed = time.time() - ai_start_time
                            if elapsed > 0 and comparisons_done > 0:
                                rate = comparisons_done / elapsed
                                remaining = (total_comparisons - comparisons_done) / rate
                                log(f"   📊 AI Hunter progress: {comparisons_done}/{total_comparisons} ({progress}%) - ~{int(remaining)}s remaining")
                            else:
                                log(f"   📊 AI Hunter progress: {comparisons_done}/{total_comparisons} ({progress}%)")
                            last_progress = progress
                        
                        # Skip if already in same group
                        if (results[i]['filename'] in duplicate_groups and 
                            results[j]['filename'] in duplicate_groups and
                            duplicate_groups[results[i]['filename']] == duplicate_groups[results[j]['filename']]):
                            continue
                        
                        # Get cached comparison results
                        sem_sim, struct_sim, text_sim = ai_hunter_check_cached(i, j)
                        
                        # For AI Hunter, use a combination approach
                        if config.mode == 'ai-hunter':
                            # High semantic + high structural = likely same content
                            if sem_sim >= semantic_threshold and struct_sim >= config.get_threshold('structural'):
                                # If text similarity is low but semantic/structural is high, it's likely a retranslation
                                if text_sim < 0.6:  # Different enough text
                                    log(f"   🎯 AI Hunter: Found potential retranslation")
                                    log(f"      Files: {results[i]['filename']} ≈ {results[j]['filename']}")
                                    log(f"      Text similarity: {int(text_sim*100)}% (low)")
                                    log(f"      Semantic similarity: {int(sem_sim*100)}% (high)")
                                    log(f"      Structural similarity: {int(struct_sim*100)}% (high)")
                                    
                                    merge_duplicate_groups(duplicate_groups, 
                                                         results[i]['filename'], 
                                                         results[j]['filename'])
                                    confidence = (sem_sim + struct_sim) / 2
                                    duplicate_confidence[(results[i]['filename'], results[j]['filename'])] = confidence
                                    log(f"   └─ 🤖 Flagged as AI retranslation variant (confidence: {int(confidence*100)}%)")
                        else:
                            # Normal semantic checking
                            if sem_sim >= semantic_threshold and struct_sim >= config.get_threshold('structural'):
                                merge_duplicate_groups(duplicate_groups, 
                                                     results[i]['filename'], 
                                                     results[j]['filename'])
                                confidence = (sem_sim + struct_sim) / 2
                                duplicate_confidence[(results[i]['filename'], results[j]['filename'])] = confidence
                                log(f"   └─ Semantic match: {results[i]['filename']} ≈ {results[j]['filename']} "
                                    f"(sem: {int(sem_sim*100)}%, struct: {int(struct_sim*100)}%)")
                
                # Clear local cache
                ai_hunter_check_cached.cache_clear()

    # THIS CODE SHOULD BE OUTSIDE ALL THE IF/ELSE BLOCKS - IT RUNS AFTER DUPLICATE DETECTION
    # 5. Deep similarity check (content-based) - Now uses cached function
    if config.mode != 'quick-scan':
        perform_deep_similarity_check(results, duplicate_groups, duplicate_confidence, 
                                    config.get_threshold('similarity'), log, should_stop)
    else:
        log("   ⚡ Skipping deep similarity check for quick scan mode")

    # 6. Consecutive chapter check with fuzzy matching - SKIP IN QUICK SCAN
    if config.mode != 'quick-scan':
        check_consecutive_chapters(results, duplicate_groups, duplicate_confidence, config, log, should_stop)
        
        # 7. Split chapter detection
        split_candidates = detect_split_chapters(results)
        if split_candidates:
            log(f"🔍 Found {len(split_candidates)} potential split chapters")
            check_split_chapters(split_candidates, results, duplicate_groups, duplicate_confidence, log, should_stop)
        
        # 8. Specific pattern detection
        check_specific_patterns(results, duplicate_groups, duplicate_confidence, log, should_stop)
    
    # Clear local caches
    compare_texts_cached.cache_clear()
    
    # Summary of findings
    unique_groups = len(set(duplicate_groups.values())) if duplicate_groups else 0
    files_with_duplicates = len(duplicate_groups)
    
    if files_with_duplicates > 0:
        log(f"\n📊 Duplicate Detection Summary:")
        log(f"   Found {files_with_duplicates} files with duplicates")
        log(f"   Grouped into {unique_groups} duplicate groups")
    else:
        log(f"\n✅ No duplicates found among {len(results)} files")
    
    return duplicate_groups, near_duplicate_groups, duplicate_confidence

def process_deep_similarity_batch(args):
    """Process a batch of deep similarity comparisons with enhanced error handling"""
    try:
        batch, data = args
        batch_results = []
        
        text_samples = data['text_samples']
        threshold = data['threshold']
        
        # Import what we need inside the worker with error handling
        try:
            from difflib import SequenceMatcher
        except ImportError as e:
            return [{'error': f'Import error in worker: {e}'}]
        
        # Local cache for this worker process
        similarity_cache = {}
        semantic_cache = {}
        
        for i, j, filename_i, filename_j in batch:
            try:
                # Get text samples
                sample_i = text_samples.get(i)
                sample_j = text_samples.get(j)
                
                if not sample_i or not sample_j:
                    continue
                
                # Use hashes for similarity check with caching
                hash1 = sample_i['hash_5k']
                hash2 = sample_j['hash_5k']
                
                # Create cache key (ensure consistent ordering)
                cache_key = (min(hash1, hash2), max(hash1, hash2))
                
                # Check cache first
                if cache_key in similarity_cache:
                    similarity = similarity_cache[cache_key]
                else:
                    # Check if hashes are identical
                    if hash1 == hash2:
                        similarity = 1.0
                    else:
                        # Calculate text similarity
                        text1 = sample_i['sample_5k']
                        text2 = sample_j['sample_5k']
                        similarity = calculate_similarity_ratio(text1, text2)
                    
                    # Cache the result
                    similarity_cache[cache_key] = similarity
                
                if similarity >= threshold:
                    batch_results.append({
                        'filename1': filename_i,
                        'filename2': filename_j,
                        'similarity': similarity,
                        'is_variant': False,
                        'semantic_sim': None
                    })
                # Check for translation variants if similarity is moderate
                elif 0.5 <= similarity < threshold:
                    # Check semantic similarity with caching
                    hash1_10k = sample_i['hash_10k']
                    hash2_10k = sample_j['hash_10k']
                    
                    # Create semantic cache key
                    sem_cache_key = (min(hash1_10k, hash2_10k), max(hash1_10k, hash2_10k))
                    
                    if sem_cache_key in semantic_cache:
                        semantic_sim = semantic_cache[sem_cache_key]
                    else:
                        if hash1_10k == hash2_10k:
                            semantic_sim = 1.0
                        else:
                            text1_10k = sample_i['sample_10k']
                            text2_10k = sample_j['sample_10k']
                            semantic_sim = calculate_semantic_fingerprint_similarity(text1_10k, text2_10k)
                        
                        # Cache the result
                        semantic_cache[sem_cache_key] = semantic_sim
                    
                    if semantic_sim >= 0.75:  # High semantic similarity threshold
                        combined_score = (similarity * 0.4 + semantic_sim * 0.6)
                        
                        if combined_score >= threshold:
                            batch_results.append({
                                'filename1': filename_i,
                                'filename2': filename_j,
                                'similarity': combined_score,
                                'is_variant': True,
                                'semantic_sim': semantic_sim,
                                'base_sim': similarity
                            })
                            
            except Exception as e:
                # Log individual comparison error but continue processing
                import traceback
                batch_results.append({
                    'error': f'Error comparing {filename_i} vs {filename_j}: {str(e)}\n{traceback.format_exc()[:500]}'
                })
                continue
        
        return batch_results
        
    except Exception as e:
        # Return error information for debugging
        import traceback
        return [{'error': f'{type(e).__name__}: {str(e)}\nTraceback:\n{traceback.format_exc()}'}]


def perform_deep_similarity_check(results, duplicate_groups, duplicate_confidence, 
                                threshold, log, should_stop):
    """Perform deep similarity analysis - PROCESSPOOLEXECUTOR VERSION with fallback"""
    
    log(f"🔍 Deep content similarity analysis (threshold: {int(threshold*100)}%)...")
    
    # Pre-cache text samples for all results
    text_samples = {}
    for idx, result in enumerate(results):
        text = result.get('raw_text', '')
        if len(text) >= 500:
            text_samples[idx] = {
                'sample_5k': text[:5000],
                'sample_10k': text[:10000],
                'hash_5k': hashlib.md5(text[:5000].encode()).hexdigest(),
                'hash_10k': hashlib.md5(text[:10000].encode()).hexdigest()
            }
    
    # Determine number of workers
    cpu_count = multiprocessing.cpu_count()
    max_workers_config = 0
    
    try:
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                full_config = json.load(f)
                # Check multiple possible config locations
                qa_config = full_config.get('qa_scanner_config', {})
                deep_check_config = full_config.get('deep_check_config', {})
                ai_hunter_config = full_config.get('ai_hunter_config', {})
                
                # Priority: deep_check_config > qa_scanner_config > ai_hunter_config
                max_workers_config = deep_check_config.get('max_workers', 
                                    qa_config.get('max_workers',
                                    ai_hunter_config.get('ai_hunter_max_workers', 1)))
    except:
        max_workers_config = 0
    
    # Determine if we should use parallel processing
    use_parallel = True
    parallel_error = None
    
    if max_workers_config == 1:
        use_parallel = False
        log("   📝 Using sequential processing (configured for 1 worker)")
    elif max_workers_config > 0:
        max_workers = min(max_workers_config, cpu_count)
    else:
        max_workers = cpu_count
    
    # Create comparison tasks with smart filtering
    comparison_tasks = []
    checked_pairs = set()
    
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            # Skip if not in text_samples (too short)
            if i not in text_samples or j not in text_samples:
                continue
                
            pair = tuple(sorted([results[i]['filename'], results[j]['filename']]))
            if pair in checked_pairs:
                continue
            checked_pairs.add(pair)
            
            # Skip if already in same group
            if (results[i]['filename'] in duplicate_groups and 
                results[j]['filename'] in duplicate_groups and
                duplicate_groups[results[i]['filename']] == duplicate_groups[results[j]['filename']]):
                continue
            
            comparison_tasks.append((i, j, results[i]['filename'], results[j]['filename']))
    
    total_comparisons = len(comparison_tasks)
    log(f"   📋 Created {total_comparisons:,} comparison tasks")
    
    if total_comparisons == 0:
        log("   ✅ No comparisons needed!")
        return
    
    # Try parallel processing first
    if use_parallel:
        log("⚡ PROCESSPOOLEXECUTOR ENABLED - MAXIMUM PERFORMANCE!")
        if max_workers_config > 0:
            log(f"   🖥️ Using {max_workers} parallel processes (configured limit)")
        else:
            log(f"   🚀 Using ALL {max_workers} CPU cores - MAXIMUM PERFORMANCE!")
            if cpu_count > 8:
                log(f"   💡 Tip: You can limit CPU cores in QA scanner settings")
        
        # Progress tracking
        comparisons_done = 0
        last_progress = 0
        start_time = time.time()
        found_duplicates = []
        
        # Prepare data for workers
        worker_data = {
            'text_samples': text_samples,
            'threshold': threshold
        }
        
        # Optimal batch size for ProcessPoolExecutor
        optimal_batch_size = max(1000, total_comparisons // (max_workers * 5))
        optimal_batch_size = min(optimal_batch_size, 10000)
        
        batches = []
        for i in range(0, len(comparison_tasks), optimal_batch_size):
            batch = comparison_tasks[i:i + optimal_batch_size]
            batches.append(batch)
        
        log(f"   📦 Split into {len(batches)} batches of ~{optimal_batch_size} comparisons each")
        
        # Prepare batch arguments
        batch_args = [(batch, worker_data) for batch in batches]
        
        try:
            # Process with ProcessPoolExecutor
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all batches
                futures = []
                for args in batch_args:
                    if should_stop():
                        log("⛔ Deep similarity check interrupted by user.")
                        executor.shutdown(wait=True)
                        return
                    
                    future = executor.submit(process_deep_similarity_batch, args)
                    futures.append(future)
                
                # Process results as they complete
                for completed_future in concurrent.futures.as_completed(futures):
                    if should_stop():
                        log("⛔ Deep similarity check interrupted by user.")
                        executor.shutdown(wait=True)
                        return
                    
                    try:
                        # NO TIMEOUT - let it run as long as needed
                        batch_results = completed_future.result()
                        
                        # Check for worker errors in results
                        if batch_results and isinstance(batch_results, list):
                            # Check if first result contains an error
                            if batch_results and isinstance(batch_results[0], dict) and 'error' in batch_results[0]:
                                error_msg = batch_results[0]['error']
                                log(f"   ⚠️ Worker error detected: {error_msg}")
                                raise Exception(f"Worker error: {error_msg}")
                        
                        # Batch all updates
                        updates = []
                        for result in batch_results:
                            if 'error' not in result:  # Skip error entries
                                updates.append((
                                    result['filename1'],
                                    result['filename2'],
                                    result
                                ))
                        
                        # Apply all updates in one lock
                        if updates:
                            with merge_lock:
                                for file1, file2, result in updates:
                                    pair = tuple(sorted([file1, file2]))
                                    
                                    merge_duplicate_groups(duplicate_groups, file1, file2)
                                    duplicate_confidence[pair] = max(
                                        duplicate_confidence.get(pair, 0), 
                                        result['similarity']
                                    )
                                    
                                    # Store messages for logging
                                    if result.get('is_variant', False):
                                        msg = (f"   └─ Translation variant detected: {file1} ≈ {file2} "
                                              f"(base: {int(result.get('base_sim', 0)*100)}%, "
                                              f"semantic: {int(result['semantic_sim']*100)}%, "
                                              f"combined: {int(result['similarity']*100)}%)")
                                    else:
                                        msg = (f"   └─ Content similarity: {file1} ≈ {file2} "
                                              f"({int(result['similarity']*100)}%)")
                                    
                                    found_duplicates.append(msg)
                        
                        # Update progress
                        comparisons_done += optimal_batch_size
                        if comparisons_done > total_comparisons:
                            comparisons_done = total_comparisons
                        
                        progress = int((comparisons_done / total_comparisons) * 100)
                        
                        # Update every 10% for less overhead
                        if progress >= last_progress + 10 or progress == 100:
                            elapsed = time.time() - start_time
                            rate = comparisons_done / elapsed if elapsed > 0 else 0
                            remaining = (total_comparisons - comparisons_done) / rate if rate > 0 else 0
                            
                            log(f"   📊 Deep check progress: {comparisons_done:,}/{total_comparisons:,} "
                                f"({progress}%) - ~{int(remaining)}s remaining - "
                                f"Speed: {int(rate):,} comparisons/sec")
                            
                            # Log some found duplicates
                            for dup_msg in found_duplicates[:5]:
                                log(dup_msg)
                            found_duplicates = found_duplicates[5:]
                            
                            last_progress = progress
                        
                    except Exception as e:
                        log(f"   ⚠️ Error processing batch: {type(e).__name__}: {str(e)[:200]}")
                        import traceback
                        log(f"   Debug trace: {traceback.format_exc()[:500]}")
                        parallel_error = f"{type(e).__name__}: {str(e)[:100]}"
                        use_parallel = False
                        executor.shutdown(wait=False)
                        break
                
                # If we completed successfully
                if use_parallel:
                    # Final summary
                    elapsed = time.time() - start_time
                    log(f"✅ Deep similarity check complete! Processed {total_comparisons:,} comparisons in {elapsed:.1f}s")
                    log(f"   ⚡ Speed: {int(total_comparisons/elapsed):,} comparisons/sec")
                    log(f"   🚀 ProcessPoolExecutor: ENABLED")
                    
                    # Log remaining duplicates
                    for dup_msg in found_duplicates[-10:]:
                        log(dup_msg)
                    return  # Success - exit function
                    
        except Exception as e:
            log(f"   ⚠️ Parallel processing failed: {type(e).__name__}: {str(e)[:200]}")
            parallel_error = f"{type(e).__name__}: {str(e)[:100]}"
            use_parallel = False
    
    # Fallback to sequential processing
    if not use_parallel:
        log(f"\n   📝 FALLBACK: Using sequential processing")
        if parallel_error:
            log(f"      Reason: {parallel_error}")
        log(f"      This will be slower but more reliable")
        
        # Reset progress tracking for sequential mode
        comparisons_done = 0
        last_progress = 0
        start_time = time.time()
        found_duplicates = []
        
        # Import what we need for sequential processing
        from difflib import SequenceMatcher
        
        for idx, task in enumerate(comparison_tasks):
            if should_stop():
                log("⛔ Deep similarity check interrupted by user.")
                return
            
            i, j, filename_i, filename_j = task
            comparisons_done += 1
            
            # Show progress every 5% or every 100 comparisons (whichever is less frequent)
            progress = int((comparisons_done / total_comparisons) * 100)
            if (comparisons_done % max(100, total_comparisons // 20) == 0 or 
                comparisons_done == total_comparisons):
                if progress >= last_progress + 5 or progress == 100:
                    elapsed = time.time() - start_time
                    rate = comparisons_done / elapsed if elapsed > 0 else 0
                    remaining = (total_comparisons - comparisons_done) / rate if rate > 0 else 0
                    
                    log(f"   📊 Sequential progress: {comparisons_done:,}/{total_comparisons:,} "
                        f"({progress}%) - ~{int(remaining)}s remaining - "
                        f"Speed: {int(rate):,} comparisons/sec")
                    
                    # Log found duplicates
                    for dup_msg in found_duplicates[:3]:
                        log(dup_msg)
                    found_duplicates = found_duplicates[3:]
                    
                    last_progress = progress
            
            # Get text samples
            sample_i = text_samples.get(i)
            sample_j = text_samples.get(j)
            
            if not sample_i or not sample_j:
                continue
            
            # Calculate similarity
            if sample_i['hash_5k'] == sample_j['hash_5k']:
                similarity = 1.0
            else:
                text1 = sample_i['sample_5k']
                text2 = sample_j['sample_5k']
                similarity = calculate_similarity_ratio(text1, text2)
            
            if similarity >= threshold:
                merge_duplicate_groups(duplicate_groups, filename_i, filename_j)
                pair = tuple(sorted([filename_i, filename_j]))
                duplicate_confidence[pair] = max(
                    duplicate_confidence.get(pair, 0), 
                    similarity
                )
                msg = f"   └─ Content similarity: {filename_i} ≈ {filename_j} ({int(similarity*100)}%)"
                found_duplicates.append(msg)
                
            elif 0.5 <= similarity < threshold:
                # Check semantic similarity for translation variants
                text1_10k = sample_i['sample_10k']
                text2_10k = sample_j['sample_10k']
                
                if sample_i['hash_10k'] == sample_j['hash_10k']:
                    semantic_sim = 1.0
                else:
                    semantic_sim = calculate_semantic_fingerprint_similarity(text1_10k, text2_10k)
                
                if semantic_sim >= 0.75:
                    combined_score = (similarity * 0.4 + semantic_sim * 0.6)
                    
                    if combined_score >= threshold:
                        merge_duplicate_groups(duplicate_groups, filename_i, filename_j)
                        pair = tuple(sorted([filename_i, filename_j]))
                        duplicate_confidence[pair] = max(
                            duplicate_confidence.get(pair, 0), 
                            combined_score
                        )
                        msg = (f"   └─ Translation variant detected: {filename_i} ≈ {filename_j} "
                              f"(base: {int(similarity*100)}%, semantic: {int(semantic_sim*100)}%, "
                              f"combined: {int(combined_score*100)}%)")
                        found_duplicates.append(msg)
        
        # Final summary for sequential mode
        elapsed = time.time() - start_time
        log(f"✅ Deep similarity check complete! Processed {total_comparisons:,} comparisons in {elapsed:.1f}s")
        if elapsed > 0:
            log(f"   Speed: {int(total_comparisons/elapsed):,} comparisons/sec")
        log(f"   Mode: Sequential (fallback)")
        
        # Log remaining duplicates
        for dup_msg in found_duplicates[-10:]:
            log(dup_msg)
        
def check_consecutive_chapters(results, duplicate_groups, duplicate_confidence, config, log, should_stop=None):
    """Check for consecutive chapters with same title using fuzzy matching"""
    log("🔍 Checking consecutive same-titled chapters...")
    
    # Check for stop early
    if should_stop and should_stop():
        log("⛔ Consecutive chapter check interrupted by user.")
        return
    
    # Extract chapter titles
    for result in results:
        result['chapter_title'] = extract_chapter_title(result['raw_text'])
    
    # Sort by chapter number
    chapter_sorted = [r for r in results if r['chapter_num'] is not None]
    chapter_sorted.sort(key=lambda x: x['chapter_num'])
    
    consecutive_threshold = config.get_threshold('consecutive_chapters')
    
    for i in range(len(chapter_sorted) - 1):
        if should_stop and should_stop():
            log("⛔ Consecutive chapter check interrupted by user.")
            return
            
        current = chapter_sorted[i]
        
        for j in range(i + 1, min(i + consecutive_threshold + 1, len(chapter_sorted))):
            next_chapter = chapter_sorted[j]
            
            # Check if chapter numbers might be the same (fuzzy match)
            if fuzzy_match_chapter_numbers(current['raw_text'], next_chapter['raw_text'],
                                         current['chapter_num'], next_chapter['chapter_num']):
                # Compare content
                similarity = calculate_similarity_ratio(current['raw_text'], next_chapter['raw_text'])
                if similarity >= config.get_threshold('similarity'):
                    merge_duplicate_groups(duplicate_groups, current['filename'], next_chapter['filename'])
                    pair = tuple(sorted([current['filename'], next_chapter['filename']]))
                    duplicate_confidence[pair] = similarity
                    log(f"   └─ Fuzzy chapter match: {current['filename']} ≈ {next_chapter['filename']} ({int(similarity*100)}%)")
                    continue
            
            # Check same title
            if (current.get('chapter_title') and current['chapter_title'] == next_chapter.get('chapter_title') and
                abs(current['chapter_num'] - next_chapter['chapter_num']) <= consecutive_threshold):
                
                # Compare content without chapter headers
                text1 = re.sub(r'Chapter\s+\d+\s*:?\s*', '', current['raw_text'][:2000], flags=re.IGNORECASE)
                text2 = re.sub(r'Chapter\s+\d+\s*:?\s*', '', next_chapter['raw_text'][:2000], flags=re.IGNORECASE)
                
                similarity = calculate_similarity_ratio(text1, text2)
                
                if similarity >= config.get_threshold('similarity') * 0.9:  # Slightly lower threshold for same title
                    merge_duplicate_groups(duplicate_groups, current['filename'], next_chapter['filename'])
                    pair = tuple(sorted([current['filename'], next_chapter['filename']]))
                    duplicate_confidence[pair] = similarity
                    log(f"   └─ Same-titled chapters {current['chapter_num']} & {next_chapter['chapter_num']} "
                        f"({int(similarity*100)}% similar)")


def check_split_chapters(split_candidates, results, duplicate_groups, duplicate_confidence, log, should_stop=None):
    """Check if split chapters are parts of the same content
    Enhanced to reduce false positives from intentional author formatting
    """
    for i, candidate in enumerate(split_candidates):
        if should_stop and should_stop():
            log("⛔ Split chapter check interrupted by user.")
            return
        
        idx = candidate['index']
        indicators = candidate['indicators']
        
        # Check next few files
        for j in range(1, 4):  # Check up to 3 files ahead
            if idx + j < len(results):
                next_result = results[idx + j]
                next_text = next_result.get('raw_text', '')
                
                # Skip if next file is empty
                if not next_text.strip():
                    continue
                
                # Extract chapter numbers if present
                current_chapter_num = results[idx].get('chapter_num')
                next_chapter_num = next_result.get('chapter_num')
                
                # Strong indicator: same chapter number
                same_chapter_number = (current_chapter_num is not None and 
                                      next_chapter_num is not None and 
                                      current_chapter_num == next_chapter_num)
                
                # Check file naming pattern similarity
                current_filename = results[idx]['filename']
                next_filename = next_result['filename']
                
                # Look for systematic naming (e.g., file_1.html, file_2.html)
                naming_pattern_match = False
                if re.sub(r'\d+', 'X', current_filename) == re.sub(r'\d+', 'X', next_filename):
                    # Files have same pattern with different numbers
                    naming_pattern_match = True
                
                # Check if content flows naturally
                should_check_flow = False
                confidence_score = 0.0
                
                if indicators['is_systematic_split'] or naming_pattern_match:
                    # Strong file naming evidence
                    should_check_flow = True
                    confidence_score = 0.85
                elif same_chapter_number:
                    # Same chapter number is strong evidence
                    should_check_flow = True
                    confidence_score = 0.9
                elif indicators['ends_mid']:
                    # Only check flow if current ends mid-sentence
                    next_text_stripped = next_text.strip()
                    if next_text_stripped:
                        # Check if next starts without capital (excluding common transition words)
                        first_line = next_text_stripped.split('\n')[0].strip()
                        if first_line and not re.match(r'^["「『\(\[]', first_line):
                            first_word = first_line.split()[0] if first_line.split() else ''
                            transition_words = ['meanwhile', 'however', 'suddenly', 'later', 
                                              'earlier', 'elsewhere', 'afterward', 'afterwards', 'then']
                            if (first_word.lower() not in transition_words and 
                                first_line[0].islower()):
                                should_check_flow = True
                                confidence_score = 0.75
                
                if should_check_flow:
                    # Get text samples for flow checking
                    text1_end = results[idx].get('raw_text', '')[-500:]
                    text2_start = next_text[:500]
                    
                    # Remove any scene break markers for flow check
                    scene_breaks = [r'[\*\s]{3,}', r'[─━－—\-]{3,}', r'[_]{3,}', 
                                   r'[~～]{3,}', r'[=]{3,}', r'[\#]{3,}']
                    for pattern in scene_breaks:
                        text1_end = re.sub(pattern, '', text1_end)
                        text2_start = re.sub(pattern, '', text2_start)
                    
                    # Check if content flows
                    combined = text1_end.strip() + " " + text2_start.strip()
                    
                    # Count sentence endings in combined text
                    sentence_endings = len(re.findall(r'[.!?。！？]', combined))
                    
                    # Check for incomplete dialogue
                    incomplete_dialogue = (text1_end.count('"') + text2_start.count('"')) % 2 != 0
                    incomplete_dialogue_jp = (text1_end.count('「') + text2_start.count('「') != 
                                             text1_end.count('」') + text2_start.count('」'))
                    
                    # Determine if this is a real split
                    is_real_split = False
                    
                    if sentence_endings < 2:  # Very few sentence endings suggests continuous text
                        is_real_split = True
                        confidence_score = max(confidence_score, 0.85)
                    elif incomplete_dialogue or incomplete_dialogue_jp:
                        is_real_split = True
                        confidence_score = max(confidence_score, 0.8)
                    elif same_chapter_number or indicators['is_systematic_split']:
                        # With strong other evidence, be more lenient
                        is_real_split = True
                    
                    if is_real_split:
                        merge_duplicate_groups(duplicate_groups, current_filename, next_filename)
                        pair = tuple(sorted([current_filename, next_filename]))
                        duplicate_confidence[pair] = confidence_score
                        
                        reason = []
                        if same_chapter_number:
                            reason.append(f"same chapter #{current_chapter_num}")
                        if indicators['is_systematic_split']:
                            reason.append("systematic file naming")
                        if naming_pattern_match:
                            reason.append("matching name pattern")
                        if sentence_endings < 2:
                            reason.append("continuous text flow")
                        if incomplete_dialogue or incomplete_dialogue_jp:
                            reason.append("incomplete dialogue")
                        
                        reason_str = ", ".join(reason) if reason else "content flow analysis"
                        log(f"   └─ Split chapter detected ({reason_str}): {current_filename} → {next_filename} "
                            f"(confidence: {int(confidence_score*100)}%)")

def check_specific_patterns(results, duplicate_groups, duplicate_confidence, log, should_stop=None):
    """Check for specific known duplicate patterns"""
    log("🔍 Checking for known duplicate patterns...")
    
    if should_stop and should_stop():
        log("⛔ Pattern check interrupted by user.")
        return
    
    # Known patterns that indicate duplicates
    patterns = {
        'chapel_scene': r"under the pretense of offering a prayer.*?visited the chapel.*?hiding while holding.*?breath.*?watching the scene",
        'battle_scene': r"sword.*?clash.*?sparks.*?flew.*?metal.*?rang",
        'magic_spell': r"mana.*?gathered.*?spell.*?formation.*?glowed",
    }
    
    pattern_matches = defaultdict(list)
    
    for i, result in enumerate(results):
        text_sample = result.get('preview', '') + result.get('raw_text', '')[:2000]
        
        for pattern_name, pattern in patterns.items():
            if re.search(pattern, text_sample, re.IGNORECASE | re.DOTALL):
                pattern_matches[pattern_name].append(i)
    
    # Group files with same patterns
    for pattern_name, indices in pattern_matches.items():
        if should_stop and should_stop():
            log("⛔ Pattern check interrupted by user.")
            return
            
        if len(indices) > 1:
            log(f"   └─ Found {len(indices)} files with '{pattern_name}' pattern")
            
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    idx1, idx2 = indices[i], indices[j]
                    
                    # Verify with content similarity
                    similarity = calculate_similarity_ratio(
                        results[idx1].get('raw_text', '')[:3000],
                        results[idx2].get('raw_text', '')[:3000]
                    )
                    
                    if similarity > 0.7:  # Lower threshold for known patterns
                        merge_duplicate_groups(duplicate_groups, 
                                             results[idx1]['filename'], 
                                             results[idx2]['filename'])
                        pair = tuple(sorted([results[idx1]['filename'], results[idx2]['filename']]))
                        duplicate_confidence[pair] = similarity
                        log(f"      Pattern match confirmed: {results[idx1]['filename']} ≈ {results[idx2]['filename']}")

def generate_reports(results, folder_path, duplicate_confidence, log=print, qa_settings=None):
    """Generate output reports with enhanced duplicate information based on settings"""
    if qa_settings is None:
        qa_settings = {'report_format': 'detailed', 'auto_save_report': True}
    
    report_format = qa_settings.get('report_format', 'detailed')
    auto_save = qa_settings.get('auto_save_report', True)
    
    # Create output directory
    output_dir = os.path.basename(folder_path.rstrip('/\\')) + "_Scan Report"
    output_path = os.path.join(folder_path, output_dir)
    os.makedirs(output_path, exist_ok=True)
    
    # Prepare confidence scores for report
    for result in results:
        result['duplicate_confidence'] = 0
        for pair, confidence in duplicate_confidence.items():
            if result['filename'] in pair:
                result['duplicate_confidence'] = max(result['duplicate_confidence'], confidence)
    
    # Common function to save all reports
    def save_all_reports():
        # Save JSON report
        with open(os.path.join(output_path, "validation_results.json"), "w", encoding="utf-8") as jf:
            json.dump(results, jf, indent=2, ensure_ascii=False)
        
        # Save CSV report
        with open(os.path.join(output_path, "validation_results.csv"), "w", encoding="utf-8", newline="") as cf:
            writer = csv.DictWriter(cf, fieldnames=["file_index", "filename", "score", "issues", "duplicate_confidence"])
            writer.writeheader()
            for row in results:
                writer.writerow({
                    "file_index": row["file_index"],
                    "filename": row["filename"],
                    "score": row["score"],
                    "issues": "; ".join(row["issues"]),
                    "duplicate_confidence": f"{row.get('duplicate_confidence', 0):.2f}"
                })
        
        # Generate HTML report
        generate_html_report(results, output_path, duplicate_confidence)
        
        # Generate duplicate groups summary
        generate_duplicate_summary(results, output_path, duplicate_confidence)
    
    # Generate reports based on format setting
    if report_format == 'summary':
        # Summary format - only key statistics
        log(f"\n📊 QA Scan Summary:")
        log(f"   Total files scanned: {len(results)}")
        
        issue_count = sum(1 for r in results if r['issues'])
        log(f"   Files with issues: {issue_count}")
        
        # Count by issue type
        issue_types = {}
        for result in results:
            for issue in result['issues']:
                issue_type = issue.split('_')[0]
                issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
        
        log(f"\n   Issues by type:")
        for issue_type, count in sorted(issue_types.items(), key=lambda x: x[1], reverse=True):
            log(f"      - {issue_type}: {count}")
        
        # Save minimal summary file if auto-save enabled
        if auto_save:
            summary_file = os.path.join(output_path, "scan_summary.txt")
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"QA Scan Summary\n")
                f.write(f"===============\n\n")
                f.write(f"Total files scanned: {len(results)}\n")
                f.write(f"Files with issues: {issue_count}\n\n")
                f.write(f"Issues by type:\n")
                for issue_type, count in sorted(issue_types.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  - {issue_type}: {count}\n")
            log(f"\n📁 Summary saved to: {output_path}")
    
    elif report_format == 'verbose':
        # Verbose format - include everything including raw text samples
        if auto_save:
            # Save detailed JSON with all data
            verbose_results = []
            for result in results.copy():
                verbose_result = result.copy()
                # Include first 1000 chars of raw text in verbose mode
                if 'raw_text' in result:
                    verbose_result['text_sample'] = result['raw_text'][:1000]
                verbose_results.append(verbose_result)
            
            with open(os.path.join(output_path, "validation_results_verbose.json"), "w", encoding="utf-8") as jf:
                json.dump(verbose_results, jf, indent=2, ensure_ascii=False)
            
            # Generate detailed text report
            with open(os.path.join(output_path, "detailed_report.txt"), "w", encoding="utf-8") as tf:
                tf.write("DETAILED QA SCAN REPORT\n")
                tf.write("=" * 80 + "\n\n")
                
                for result in results:
                    tf.write(f"File: {result['filename']}\n")
                    tf.write(f"Chapter: {result.get('chapter_num', 'Unknown')}\n")
                    tf.write(f"Issues: {len(result['issues'])}\n")
                    if result['issues']:
                        for issue in result['issues']:
                            tf.write(f"  - {issue}\n")
                    tf.write(f"Duplicate Confidence: {result.get('duplicate_confidence', 0):.2f}\n")
                    tf.write(f"Preview: {result.get('preview', '')[:200]}...\n")
                    tf.write("-" * 80 + "\n\n")
        
        # All existing reports (JSON, CSV, HTML)
        save_all_reports()
    
    else:  # detailed (default)
        # Current behavior - standard reports
        if auto_save:
            save_all_reports()
        else:
            log(f"\n✅ Scan complete! Reports not saved (auto-save disabled)")
    
    log(f"\n✅ Scan complete!")
    if auto_save:
        log(f"📁 Reports saved to: {output_path}")

def generate_duplicate_summary(results, output_path, duplicate_confidence):
    """Generate a summary of duplicate groups"""
    # Collect duplicate groups
    groups = defaultdict(list)
    for result in results:
        for issue in result.get('issues', []):
            if issue.startswith('DUPLICATE:'):
                # Extract group info
                if 'part_of_' in issue:
                    group_id = issue.split('part_of_')[1].split('_')[0]
                    groups[f"group_{group_id}"].append(result['filename'])
                elif 'exact_or_near_copy_of_' in issue:
                    other = issue.split('exact_or_near_copy_of_')[1]
                    groups[f"pair_{result['filename']}_{other}"].append(result['filename'])
                    groups[f"pair_{result['filename']}_{other}"].append(other)
    
    # Create summary
    summary = {
        'total_files': len(results),
        'files_with_duplicates': sum(1 for r in results if any('DUPLICATE' in i for i in r.get('issues', []))),
        'duplicate_groups': len(groups),
        'groups': {}
    }
    
    for group_name, files in groups.items():
        unique_files = list(set(files))
        confidences = []
        for i in range(len(unique_files)):
            for j in range(i + 1, len(unique_files)):
                pair = tuple(sorted([unique_files[i], unique_files[j]]))
                if pair in duplicate_confidence:
                    confidences.append(duplicate_confidence[pair])
        
        summary['groups'][group_name] = {
            'files': unique_files,
            'count': len(unique_files),
            'avg_confidence': sum(confidences) / len(confidences) if confidences else 0
        }
    
    with open(os.path.join(output_path, "duplicate_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

def generate_html_report(results, output_path, duplicate_confidence):
    """Generate enhanced HTML report with duplicate confidence scores"""
    issue_counts = {}
    for r in results:
        for issue in r['issues']:
            issue_type = issue.split(':')[0] if ':' in issue else issue.split('_')[0]
            issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
    
    html = f"""<html>
<head>
    <meta charset='utf-8'>
    <title>Translation QA Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .error {{ background-color: #ffcccc; }}
        .warning {{ background-color: #fff3cd; }}
        .preview {{ font-size: 0.9em; color: #666; max-width: 400px; }}
        .issues {{ font-size: 0.9em; }}
        .non-english {{ color: red; font-weight: bold; }}
        .duplicate-group {{ background-color: #ffe6e6; }}
        .confidence {{ font-size: 0.8em; color: #666; }}
        .high-confidence {{ color: red; font-weight: bold; }}
        .medium-confidence {{ color: orange; }}
        .low-confidence {{ color: #666; }}
    </style>
</head>
<body>
    <h1>Translation QA Report</h1>
    <p><strong>Total Files Scanned:</strong> {len(results)}</p>
    <p><strong>Files with Issues:</strong> {sum(1 for r in results if r['issues'])}</p>
    <p><strong>Clean Files:</strong> {sum(1 for r in results if not r['issues'])}</p>
"""
    
    if issue_counts:
        html += "<h2>Issues Summary</h2><ul>"
        for issue_type, count in sorted(issue_counts.items()):
            style = ' class="non-english"' if any(x in issue_type.lower() for x in ['korean', 'chinese', 'japanese']) else ''
            html += f"<li{style}><strong>{issue_type}</strong>: {count} files</li>"
        
        # Count duplicate groups
        duplicate_groups = set()
        for result in results:
            for issue in result.get('issues', []):
                if issue.startswith('DUPLICATE:'):
                    if 'part_of_' in issue:
                        group_id = issue.split('part_of_')[1].split('_')[0]
                        duplicate_groups.add(f"group_{group_id}")
                    elif 'exact_or_near_copy_of_' in issue:
                        other = issue.split('exact_or_near_copy_of_')[1]
                        duplicate_groups.add(f"pair_{min(result['filename'], other)}_{max(result['filename'], other)}")
        
        if duplicate_groups:
            html += f"<li><strong>Duplicate Groups Found</strong>: {len(duplicate_groups)}</li>"
        
        html += "</ul>"
    
    html += "<h2>Detailed Results</h2>"
    html += "<table><tr><th>Index</th><th>Filename</th><th>Issues</th><th>Confidence</th><th>Preview</th></tr>"
    
    for row in results:
        link = f"<a href='../{row['filename']}' target='_blank'>{row['filename']}</a>"
        
        formatted_issues = []
        for issue in row["issues"]:
            if issue.startswith("DUPLICATE:"):
                formatted_issues.append(f'<span style="color: red; font-weight: bold;">{issue}</span>')
            elif issue.startswith("NEAR_DUPLICATE:"):
                formatted_issues.append(f'<span style="color: darkorange; font-weight: bold;">{issue}</span>')
            elif '_text_found_' in issue:
                formatted_issues.append(f'<span class="non-english">{issue}</span>')
            else:
                formatted_issues.append(issue)
        
        issues_str = "<br>".join(formatted_issues) if formatted_issues else "None"
        
        # Add confidence score
        confidence = row.get('duplicate_confidence', 0)
        if confidence > 0:
            conf_class = 'high-confidence' if confidence >= 0.9 else 'medium-confidence' if confidence >= 0.8 else 'low-confidence'
            confidence_str = f'<span class="confidence {conf_class}">{int(confidence * 100)}%</span>'
        else:
            confidence_str = '-'
        
        row_class = 'duplicate-group' if any('DUPLICATE:' in issue for issue in row['issues']) else ''
        if not row_class and any('NEAR_DUPLICATE:' in issue for issue in row['issues']):
            row_class = 'warning'
        if not row_class:
            row_class = 'error' if row["score"] > 1 else 'warning' if row["score"] == 1 else ''
        
        preview_escaped = html_lib.escape(row['preview'][:300])
        
        html += f"""<tr class='{row_class}'>
            <td>{row['file_index']}</td>
            <td>{link}</td>
            <td class='issues'>{issues_str}</td>
            <td>{confidence_str}</td>
            <td class='preview'>{preview_escaped}</td>
        </tr>"""
    
    html += "</table></body></html>"
    
    with open(os.path.join(output_path, "validation_results.html"), "w", encoding="utf-8") as html_file:
        html_file.write(html)

def update_progress_file(folder_path, results, log):
    """Update translation progress file"""
    prog_path = os.path.join(folder_path, "translation_progress.json")
    
    try:
        with open(prog_path, "r", encoding="utf-8") as pf:
            prog = json.load(pf)
    except FileNotFoundError:
        log("[INFO] No progress file found - nothing to update")
        return
    
    faulty_chapters = [row for row in results if row["issues"]]
    
    if not faulty_chapters:
        log("✅ No faulty chapters found - progress unchanged")
        return
    
    # Detect progress format version
    is_new_format = "chapters" in prog and isinstance(prog.get("chapters"), dict)
    
    if is_new_format:
        update_new_format_progress(prog, faulty_chapters, log, folder_path)
    else:
        update_legacy_format_progress(prog, faulty_chapters, log)
    
    # Write back updated progress
    with open(prog_path, "w", encoding="utf-8") as pf:
        json.dump(prog, pf, indent=2, ensure_ascii=False)
    
    # Log affected chapters - use the already extracted chapter numbers
    affected_chapters_for_log = []
    for faulty_row in faulty_chapters:
        # Use the chapter_num that was already extracted during scan
        chapter_num = faulty_row.get("chapter_num")
        if chapter_num is not None:
            affected_chapters_for_log.append(chapter_num)
        else:
            # Fallback if somehow chapter_num wasn't extracted
            fallback_num = faulty_row.get("file_index", 0) + 1
            if faulty_row.get("filename"):
                match = re.search(r'response_(\d+)', faulty_row["filename"])
                if match:
                    fallback_num = int(match.group(1))
            affected_chapters_for_log.append(fallback_num)

    if affected_chapters_for_log:
        log(f"📝 Chapters marked for re-translation: {', '.join(str(c) for c in sorted(affected_chapters_for_log))}")

def update_new_format_progress(prog, faulty_chapters, log, folder_path):
    """Update new format progress file with content hash support"""
    log("[INFO] Detected new progress format")
    
    # Build multiple mappings to find chapters
    output_file_to_chapter_key = {}
    actual_num_to_chapter_key = {}
    basename_to_chapter_key = {}
    merged_child_to_parent = {}  # Maps merged child actual_num -> parent actual_num
    
    for chapter_key, chapter_info in prog["chapters"].items():
        output_file = chapter_info.get("output_file")
        status = chapter_info.get("status", "")
        
        if output_file:
            # IMPORTANT: Only map non-merged chapters to output_file
            # Merged children share parent's output_file, so we don't want them to overwrite
            if status != "merged":
                # Only set if not already mapped, OR if this is a completed/parent chapter
                # This ensures parents take priority over any lingering mappings
                existing_key = output_file_to_chapter_key.get(output_file)
                if not existing_key or status == "completed":
                    output_file_to_chapter_key[output_file] = chapter_key
                
                # Also map without response_ prefix for matching
                if output_file.startswith("response_"):
                    alt_name = output_file[9:]  # Remove "response_" prefix
                    if alt_name not in output_file_to_chapter_key or status == "completed":
                        output_file_to_chapter_key[alt_name] = chapter_key
        
        # Map by actual chapter number
        actual_num = chapter_info.get("actual_num")
        if actual_num is not None:
            if actual_num not in actual_num_to_chapter_key:
                actual_num_to_chapter_key[actual_num] = []
            actual_num_to_chapter_key[actual_num].append(chapter_key)
        
        # Map by original basename
        original_basename = chapter_info.get("original_basename")
        if original_basename:
            basename_to_chapter_key[original_basename] = chapter_key
            # Also map response_ version
            basename_to_chapter_key[f"response_{original_basename}"] = chapter_key
        
        # Track merged children -> parent mapping (Method 1: from child's merged status)
        if chapter_info.get("status") == "merged":
            child_num = actual_num
            parent_num = chapter_info.get("merged_parent_chapter")
            if child_num is not None and parent_num is not None:
                merged_child_to_parent[child_num] = parent_num
        
        # Track merged children -> parent mapping (Method 2: from parent's merged_chapters list)
        # This catches cases where child's status was corrupted but parent still has the list
        merged_chapters_list = chapter_info.get("merged_chapters")
        if merged_chapters_list and actual_num is not None:
            for child_num in merged_chapters_list:
                if child_num not in merged_child_to_parent:
                    merged_child_to_parent[child_num] = actual_num
    
    updated_count = 0
    for faulty_row in faulty_chapters:
        faulty_filename = faulty_row["filename"]
        chapter_key = None
        
        # MERGED CHILDREN HANDLING: Check if this file is for a merged child chapter
        # If so, redirect QA issues to the parent chapter
        import re
        file_chapter_num = None
        num_matches = re.findall(r'(\d+)', faulty_filename)
        if num_matches:
            file_chapter_num = int(num_matches[-1])
        
        is_merged_child = False
        if file_chapter_num is not None and file_chapter_num in merged_child_to_parent:
            parent_num = merged_child_to_parent[file_chapter_num]
            is_merged_child = True
            log(f"   🔗 File {faulty_filename} (chapter {file_chapter_num}) is merged child of chapter {parent_num} - redirecting QA issues to parent")
            
            # Find the parent chapter key - this is the only key we want to update
            if parent_num in actual_num_to_chapter_key:
                parent_keys = actual_num_to_chapter_key[parent_num]
                log(f"      DEBUG: actual_num_to_chapter_key[{parent_num}] = {parent_keys}")
                chapter_key = parent_keys[0]
                log(f"      DEBUG: Using chapter_key = '{chapter_key}' for parent")
                # Skip all other matching methods - we ONLY want to update the parent
            else:
                log(f"      DEBUG: parent_num {parent_num} NOT in actual_num_to_chapter_key!")
                log(f"      DEBUG: actual_num_to_chapter_key keys = {list(actual_num_to_chapter_key.keys())}")
        
        # Method 1: Direct output file match (if not already found via merge redirect)
        if not chapter_key and not is_merged_child:
            chapter_key = output_file_to_chapter_key.get(faulty_filename)
        
        # Method 2: Try without response_ prefix
        if not chapter_key and not is_merged_child and faulty_filename.startswith("response_"):
            base_name = faulty_filename[9:]
            chapter_key = basename_to_chapter_key.get(base_name)
        
        # Method 3: Extract chapter number and match
        if not chapter_key and not is_merged_child:
            # Extract chapter number from filename
            import re
            matches = re.findall(r'(\d+)', faulty_filename)
            if matches:
                chapter_num = int(matches[-1])  # Use last number found
                
                # Look for matching chapter by number
                if chapter_num in actual_num_to_chapter_key:
                    # If multiple entries, find the one with matching output file
                    candidates = actual_num_to_chapter_key[chapter_num]
                    for candidate_key in candidates:
                        candidate_info = prog["chapters"][candidate_key]
                        candidate_output = candidate_info.get("output_file", "")
                        if candidate_output and (candidate_output == faulty_filename or candidate_output.endswith(faulty_filename)):
                            chapter_key = candidate_key
                            break
                    
                    # If still not found, use first candidate
                    if not chapter_key and candidates:
                        chapter_key = candidates[0]
        
        # Method 4: If still not found, try to calculate content hash from file
        if not chapter_key and not is_merged_child and os.path.exists(os.path.join(folder_path, faulty_filename)):
            try:
                # Read the file and calculate its content hash
                # This is a fallback for when the mapping isn't found
                with open(os.path.join(folder_path, faulty_filename), 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Try to find by scanning all chapters for matching output file
                for ch_key, ch_info in prog["chapters"].items():
                    if ch_info.get("output_file") == faulty_filename:
                        chapter_key = ch_key
                        break
            except:
                pass
        
        if chapter_key and chapter_key in prog["chapters"]:
            chapter_info = prog["chapters"][chapter_key]
            old_status = chapter_info.get("status", "unknown")
            actual_num_being_updated = chapter_info.get("actual_num")
            log(f"      DEBUG: Updating chapter_key='{chapter_key}', actual_num={actual_num_being_updated}, old_status={old_status}")
            
            # Update status to qa_failed
            chapter_info["status"] = "qa_failed"
            chapter_info["qa_issues"] = True
            chapter_info["qa_timestamp"] = time.time()
            chapter_info["qa_issues_found"] = faulty_row.get("issues", [])
            chapter_info["duplicate_confidence"] = faulty_row.get("duplicate_confidence", 0)
            
            # Ensure output_file is set (use faulty_filename if null)
            if not chapter_info.get("output_file"):
                chapter_info["output_file"] = faulty_filename
            
            updated_count += 1
            
            # Use chapter_num from faulty_row if available, otherwise fall back to actual_num
            chapter_num = faulty_row.get("chapter_num")
            if chapter_num is None:
                chapter_num = chapter_info.get('actual_num', faulty_row.get("file_index", 0) + 1)
            log(f"   └─ Marked chapter {chapter_num} as qa_failed (was: {old_status})")
            
            # IMPORTANT: Don't remove from content_hashes or chapter_chunks
            # Just mark as qa_failed so it will be retranslated
            # The translation process will handle cleanup when retranslating
            
            # Optional: Log what we're NOT removing for clarity
            content_hash = chapter_info.get("content_hash")
            if content_hash:
                log(f"   └─ Keeping content hash {content_hash[:8]}... for retranslation")
        else:
            # For merged children where we couldn't find parent, skip creating new entries
            if is_merged_child:
                log(f"   ⚠️ Merged child {faulty_filename} - could not find parent chapter {merged_child_to_parent.get(file_chapter_num, '?')} to update")
                continue
            
            # Log failure to find chapter
            log(f"   ⚠️ Could not find chapter entry for {faulty_filename}")
            
            # Try to create a new entry if we can determine the chapter number
            import re
            matches = re.findall(r'(\d+)', faulty_filename)
            if matches:
                chapter_num = int(matches[-1])
                
                # Use actual_num as key
                chapter_key = str(chapter_num)
                
                # Calculate content hash from the file if possible
                content_hash = None
                if os.path.exists(os.path.join(folder_path, faulty_filename)):
                    try:
                        with open(os.path.join(folder_path, faulty_filename), 'r', encoding='utf-8') as f:
                            content = f.read()
                        import hashlib
                        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
                    except:
                        pass
                
                # Create entry with proper field order matching regular entries
                prog["chapters"][chapter_key] = {
                    "actual_num": chapter_num,
                    "content_hash": content_hash,  # Include if we could calculate it
                    "output_file": faulty_filename,
                    "status": "qa_failed",
                    "last_updated": time.time(),  # Use same field name as regular entries
                    "zero_adjusted": False,  # Default to False since we don't know
                    # QA-specific fields come after the standard fields
                    "qa_issues": True,
                    "qa_timestamp": time.time(),
                    "qa_issues_found": faulty_row.get("issues", []),
                    "duplicate_confidence": faulty_row.get("duplicate_confidence", 0)
                }
                log(f"   └─ Created qa_failed entry for chapter {chapter_num}")
                updated_count += 1
    
    log(f"🔧 Updated {updated_count} chapters in new format")
    
def update_legacy_format_progress(prog, faulty_chapters, log):
    """Update legacy format progress file"""
    log("[INFO] Detected legacy progress format")
    
    existing = prog.get("completed", [])
    faulty_indices = [row["file_index"] for row in faulty_chapters]
    updated = [idx for idx in existing if idx not in faulty_indices]
    removed_count = len(existing) - len(updated)
    
    prog["completed"] = updated
    
    # Remove chunk data
    if "chapter_chunks" in prog:
        for faulty_idx in faulty_indices:
            chapter_key = str(faulty_idx)
            if chapter_key in prog["chapter_chunks"]:
                del prog["chapter_chunks"][chapter_key]
                log(f"   └─ Removed chunk data for chapter {faulty_idx + 1}")
    
    # Remove from content_hashes
    if "content_hashes" in prog:
        hashes_to_remove = []
        for hash_val, hash_info in prog["content_hashes"].items():
            if hash_info.get("completed_idx") in faulty_indices:
                hashes_to_remove.append(hash_val)
        
        for hash_val in hashes_to_remove:
            del prog["content_hashes"][hash_val]
            log(f"   └─ Removed content hash entry")
    
    log(f"🔧 Removed {removed_count} chapters from legacy completed list")

def extract_epub_word_counts(epub_path, log=print, min_file_length=0):
    """Extract word counts for each chapter from the original EPUB using spine order.
    
    Key: Uses content.opf SPINE ORDER (reading order) as the authoritative chapter sequence.
    This ensures that files like 0001_chapter.xhtml and 0001_section.xhtml both get
    correctly indexed by their actual position in the book, not just their filenames.
    
    Args:
        epub_path: Path to the EPUB file
        log: Logging function
        min_file_length: Minimum character length to include a file (from qa_settings)
    """
    
    def count_cjk_words(text):
        """Count actual words in CJK text with better segmentation"""
        word_count = 0
        
        # Chinese word counting (considering multi-character words)
        # Most Chinese words are 2-4 characters
        chinese_chars = re.findall(r'[\u4e00-\u9fff]+', text)
        for segment in chinese_chars:
            # Estimate words based on character count
            # Average Chinese word length is ~1.7 characters
            word_count += max(1, len(segment) / 1.7)
        
        # Japanese word counting
        # Hiragana particles/endings (usually 1-3 chars each)
        hiragana_segments = re.findall(r'[\u3040-\u309f]+', text)
        word_count += len(hiragana_segments)
        
        # Katakana words (foreign words, usually one word per segment)
        katakana_segments = re.findall(r'[\u30a0-\u30ff]+', text)
        word_count += len(katakana_segments)
        
        # Korean word counting (words are typically space-separated)
        korean_words = re.findall(r'[\uac00-\ud7af]+', text)
        word_count += len(korean_words)
        
        # Also count non-CJK words (English mixed in)
        non_cjk = re.sub(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]+', ' ', text)
        word_count += len(non_cjk.split())
        
        return int(word_count)
    
    try:
        word_counts = {}
        spine_files = {}  # Maps spine index to file info
        
        with zipfile.ZipFile(epub_path, 'r') as zf:
            # Step 1: Read and parse content.opf to get SPINE ORDER
            content_opf_data = None
            content_opf_path = None
            manifest_map = {}  # Maps item id to href
            
            try:
                for fname in zf.namelist():
                    if 'content.opf' in fname.lower():
                        content_opf_data = zf.read(fname).decode('utf-8', errors='ignore')
                        content_opf_path = fname
                        break
            except Exception as e:
                log(f"⚠️ Could not read content.opf: {e}")
            
            # Parse manifest and spine from content.opf
            if content_opf_data:
                try:
                    soup_opf = BeautifulSoup(content_opf_data, 'xml')
                    
                    # Build manifest map (id -> href)
                    manifest = soup_opf.find('manifest')
                    if manifest:
                        for item in manifest.find_all('item'):
                            item_id = item.get('id')
                            href = item.get('href')
                            if item_id and href:
                                manifest_map[item_id] = href
                    
                    # Process spine in order
                    spine = soup_opf.find('spine')
                    if spine:
                        spine_index = 1  # Start from 1 for chapter numbering
                        for itemref in spine.find_all('itemref'):
                            idref = itemref.get('idref')
                            if idref and idref in manifest_map:
                                href = manifest_map[idref]
                                spine_files[spine_index] = {
                                    'href': href,
                                    'idref': idref
                                }
                                spine_index += 1
                        
                        if spine_files:
                            log(f"📚 Found {len(spine_files)} chapters in EPUB spine order.")
                except Exception as e:
                    log(f"⚠️ Could not parse content.opf: {e}")
            
            # Step 2: Process files in spine order
            if spine_files:
                # Determine the base directory from content.opf location
                base_dir = ''
                if content_opf_path:
                    # content.opf is often in OEBPS/ or similar
                    base_dir = os.path.dirname(content_opf_path)
                    if base_dir:
                        base_dir = base_dir + '/'
                
                extracted_count = 0
                for spine_index, file_info in sorted(spine_files.items()):
                    try:
                        file_path = file_info['href']
                        
                        # Try multiple path resolutions
                        possible_paths = [
                            file_path,                          # As-is from manifest
                            base_dir + file_path,               # Relative to content.opf
                            'OEBPS/' + file_path,               # Common EPUB structure
                            'OPS/' + file_path,                 # Alternative structure
                        ]
                        
                        content = None
                        successful_path = None
                        for try_path in possible_paths:
                            try:
                                content = zf.read(try_path).decode('utf-8', errors='ignore')
                                successful_path = try_path
                                break
                            except KeyError:
                                continue
                        
                        if content is None:
                            log(f"⚠️ Could not find spine item {spine_index}: tried {possible_paths}")
                            continue
                        
                        basename = os.path.basename(file_path)
                        soup = BeautifulSoup(content, 'html.parser')
                        text = soup.get_text(strip=True)
                        
                        # Skip files shorter than minimum length setting
                        if len(text) < min_file_length:
                            continue
                        
                        # Check if text contains CJK characters
                        has_cjk = any('\u4e00' <= char <= '\u9fff' or  # Chinese
                                      '\u3040' <= char <= '\u309f' or  # Hiragana
                                      '\u30a0' <= char <= '\u30ff' or  # Katakana
                                      '\uac00' <= char <= '\ud7af'     # Korean
                                      for char in text)
                        
                        if has_cjk:
                            word_count = count_cjk_words(text)
                        else:
                            word_count = len(text.split())
                        
                        # Store using spine index as the authoritative chapter number
                        word_counts[spine_index] = {
                            'word_count': word_count,
                            'filename': basename,
                            'full_path': file_path,
                            'is_cjk': has_cjk,
                            'spine_index': spine_index
                        }
                        extracted_count += 1
                        
                    except Exception as e:
                        log(f"⚠️ Error processing spine item {spine_index} ({file_info.get('href', 'unknown')}): {e}")
                        import traceback
                        log(f"   Traceback: {traceback.format_exc()}")
                        continue
            else:
                log("⚠️ Could not read spine order, falling back to file extraction")
            
            if spine_files and extracted_count == 0:
                log(f"⚠️ Failed to extract any word counts from {len(spine_files)} spine items")
                log(f"   First spine item path: {spine_files.get(1, {}).get('href', 'unknown')}")
        
        return word_counts
        
    except Exception as e:
        log(f"❌ Error reading EPUB file: {e}")
        return {}

def detect_multiple_headers(html_content):
    """Detect if HTML content has 2 or more header tags"""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find all header tags (h1 through h6)
    headers = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    
    if len(headers) >= 2:
        header_info = []
        for header in headers[:5]:  # Show first 5 headers
            header_info.append({
                'tag': header.name,
                'text': header.get_text(strip=True)[:50]  # First 50 chars
            })
        return True, len(headers), header_info
    
    return False, len(headers), []

def cross_reference_word_counts(original_counts, translated_file, translated_text, log=print, merge_info=None):
    """Cross-reference word counts between original and translated files.
    
    Matches translated files to original EPUB chapters by filename.
    Handles response_ prefix and ignores file extensions.
    
    Args:
        original_counts: Dict of chapter word counts from original EPUB
        translated_file: Filename of the translated file
        translated_text: Text content of the translated file
        log: Logging function
        merge_info: Dict with merged_chapters info from translation_progress.json
                   Format: {parent_chapter_num: [list of merged child chapter nums]}
    """
    basename = os.path.basename(translated_file)
    # Remove extension for matching (don't consider .html, .xhtml, .htm, etc.)
    basename_no_ext = os.path.splitext(basename)[0]
    
    # Remove response_ prefix if present
    search_name = basename_no_ext
    if search_name.lower().startswith('response_'):
        search_name = search_name[9:]  # Remove 'response_' prefix
    
    # Extract chapter number from filename for merge checking
    file_chapter_num = None
    num_match = re.search(r'(\d+)', search_name)
    if num_match:
        file_chapter_num = int(num_match.group(1))
    
    # CHECK: Is this file a merged CHILD chapter? If so, skip it - the parent handles word count
    if merge_info and file_chapter_num is not None:
        for parent_num, children in merge_info.items():
            if file_chapter_num in children:
                # This is a merged child - return a special result indicating it's merged
                return {
                    'found_match': True,
                    'is_merged_child': True,
                    'parent_chapter': parent_num,
                    'chapter_num': file_chapter_num,
                    'original_wc': 0,
                    'translated_wc': 0,
                    'ratio': 1.0,
                    'percentage': 100,
                    'is_reasonable': True,
                    'is_typical': True,
                    'skip_reason': f'Merged into chapter {parent_num}'
                }
    
    # Build a filename-to-spine-idx map for merge lookups
    # This allows us to find child chapter word counts by their filenames
    filename_to_spine_idx = {}
    for sidx, cinfo in original_counts.items():
        fname = os.path.splitext(cinfo['filename'])[0].lower()
        filename_to_spine_idx[fname] = sidx
        # Also extract chapter number from filename for fallback matching
        fname_num_match = re.search(r'(\d+)', fname)
        if fname_num_match:
            fname_num = int(fname_num_match.group(1))
            # Map chapter number to spine idx (for merge_info lookup)
            if fname_num not in filename_to_spine_idx:
                filename_to_spine_idx[f"chapnum_{fname_num}"] = sidx
    
    # Try to find matching filename in original_counts
    for spine_idx, count_info in original_counts.items():
        epub_filename = os.path.splitext(count_info['filename'])[0]  # Remove extension from EPUB filename
        
        # Direct filename match (case-insensitive)
        if search_name.lower() == epub_filename.lower():
            original_wc = count_info['word_count']
            is_cjk = count_info.get('is_cjk', True)
            
            # REQUEST MERGING: If this chapter has merged children, combine word counts
            # merge_info uses actual_num (chapter numbers), need to map to spine indices
            if merge_info and file_chapter_num is not None and file_chapter_num in merge_info:
                merged_children = merge_info[file_chapter_num]
                original_wc_base = original_wc
                children_found = 0
                for child_num in merged_children:
                    # Find child's spine_idx by its chapter number
                    child_spine_key = f"chapnum_{child_num}"
                    if child_spine_key in filename_to_spine_idx:
                        child_spine_idx = filename_to_spine_idx[child_spine_key]
                        if child_spine_idx in original_counts:
                            original_wc += original_counts[child_spine_idx]['word_count']
                            children_found += 1
                    # Also try direct spine index lookup as fallback
                    elif child_num in original_counts:
                        original_wc += original_counts[child_num]['word_count']
                        children_found += 1
                if children_found > 0:
                    log(f"   🔗 Merged chapter {file_chapter_num}: combining word counts from {children_found} child chapters")
                    log(f"      Base: {original_wc_base}, Combined: {original_wc}")
            
            # Count words in translated text
            translated_wc = len(translated_text.split())
            
            # Calculate ratio
            ratio = translated_wc / max(1, original_wc)
            
            # Define VERY PERMISSIVE ratio ranges for novel translation
            if is_cjk:
                # CJK to English novel translation - reasonable bounds
                min_ratio = 0.6   # 60% - catches significant omissions
                max_ratio = 2.5   # 250% - catches excessive padding
                
                # Typical healthy range
                typical_min = 0.8   # 80%
                typical_max = 1.8   # 180%
            else:
                # Non-CJK source
                min_ratio = 0.7
                max_ratio = 1.5
                typical_min = 0.8
                typical_max = 1.2
            
            is_reasonable = min_ratio <= ratio <= max_ratio
            is_typical = typical_min <= ratio <= typical_max
            percentage = ratio * 100
            
            result = {
                'found_match': True,
                'chapter_num': spine_idx,
                'original_wc': original_wc,
                'translated_wc': translated_wc,
                'ratio': ratio,
                'percentage': percentage,
                'is_reasonable': is_reasonable,
                'is_typical': is_typical,
                'original_file': count_info['filename']
            }
            
            # Add descriptive warnings for extreme but acceptable ratios
            if ratio < 0.5:
                result['warning'] = 'very_concise_translation'
                result['warning_desc'] = 'Translation is less than 50% of original - possible summary style'
            elif ratio < typical_min:
                result['warning'] = 'concise_translation'
                result['warning_desc'] = f'Translation is {percentage:.0f}% of original - somewhat concise'
            elif ratio > 4.0:
                result['warning'] = 'very_expansive_translation'
                result['warning_desc'] = 'Translation is over 400% of original - extensive additions'
            elif ratio > typical_max:
                result['warning'] = 'expansive_translation'
                result['warning_desc'] = f'Translation is {percentage:.0f}% of original - somewhat expansive'
            
            # Only flag as error if REALLY extreme
            if not is_reasonable:
                if ratio < min_ratio:
                    result['error'] = 'possibly_missing_content'
                    result['error_desc'] = f'Translation is only {percentage:.0f}% of original'
                else:
                    result['error'] = 'possibly_excessive_content'
                    result['error_desc'] = f'Translation is {percentage:.0f}% of original'
            
            return result
    
    # Fallback: old chapter number extraction logic (kept for backwards compatibility)
    chapter_num = None
    
    # PRIORITY 1: Extract from filename patterns (most specific first)
    # Important: Check "response_No" patterns BEFORE generic "response_" patterns
    # to avoid extracting just the first number from "response_No00001Chapter"
    patterns = [
        r'response_[Nn]o(\d{4,5})(?:[_-]|$)',   # response_No00001 or response_No00001Chapter
        r'response_(\d{4,5})(?:[_-]|$)',         # response_00001 or response_00001Chapter
        r'response_chapter(\d+)',                 # response_chapter1
        r'chapter[\s_-]*(\d+)',                  # chapter1, chapter_2
        r'ch[\s_-]*(\d+)',                       # ch1, ch_2
    ]
    
    for pattern in patterns:
        match = re.search(pattern, basename_no_ext, re.IGNORECASE)
        if match:
            chapter_num = int(match.group(1))
            # Remove leading zeros for matching
            if chapter_num > 1000:
                chapter_num = chapter_num % 1000  # e.g., 00001 -> 1
            break
    
    if chapter_num is None:
        # PRIORITY 2: Try content-based matching as fallback
        content_patterns = [
            r'Chapter\s+(\d+)',
            r'第\s*(\d+)\s*章',
            r'제\s*(\d+)\s*장'
        ]
        
        for pattern in content_patterns:
            match = re.search(pattern, translated_text[:500], re.IGNORECASE)
            if match:
                chapter_num = int(match.group(1))
                break
    
    if chapter_num is not None and chapter_num in original_counts:
        original_wc = original_counts[chapter_num]['word_count']
        is_cjk = original_counts[chapter_num].get('is_cjk', True)  # Get CJK flag if available
        
        # REQUEST MERGING: If this chapter has merged children, combine word counts
        # Need to map child chapter numbers to spine indices
        if merge_info and chapter_num in merge_info:
            merged_children = merge_info[chapter_num]
            for child_num in merged_children:
                # Try to find child by chapter number mapping
                child_spine_key = f"chapnum_{child_num}"
                if child_spine_key in filename_to_spine_idx:
                    child_spine_idx = filename_to_spine_idx[child_spine_key]
                    if child_spine_idx in original_counts:
                        original_wc += original_counts[child_spine_idx]['word_count']
                # Fallback: try direct spine index
                elif child_num in original_counts:
                    original_wc += original_counts[child_num]['word_count']
        
        # Count words in translated text
        translated_wc = len(translated_text.split())
        
        # Calculate ratio (translated words / original words)
        ratio = translated_wc / max(1, original_wc)
        
        # Define VERY PERMISSIVE ratio ranges for novel translation
        # These are much looser to accommodate extreme translation cases
        if is_cjk:
            # CJK to English novel translation - reasonable bounds
            min_ratio = 0.6   # 60% - catches significant omissions
            max_ratio = 2.5   # 250% - catches excessive padding
            
            # Typical healthy range
            typical_min = 0.8   # 80%
            typical_max = 1.8   # 180%
        else:
            # Non-CJK source
            min_ratio = 0.7
            max_ratio = 1.5
            typical_min = 0.8
            typical_max = 1.2
        
        is_reasonable = min_ratio <= ratio <= max_ratio
        is_typical = typical_min <= ratio <= typical_max
        
        # Calculate percentage difference for logging
        percentage = (ratio * 100)
        
        result = {
            'found_match': True,
            'chapter_num': chapter_num,
            'original_wc': original_wc,
            'translated_wc': translated_wc,
            'ratio': ratio,
            'percentage': percentage,  # e.g., 150 = 150% of original
            'is_reasonable': is_reasonable,
            'is_typical': is_typical,
            'original_file': original_counts[chapter_num]['filename']
        }
        
        # Add descriptive warnings for extreme but acceptable ratios
        if ratio < 0.5:
            result['warning'] = 'very_concise_translation'
            result['warning_desc'] = 'Translation is less than 50% of original - possible summary style'
        elif ratio < typical_min:
            result['warning'] = 'concise_translation'
            result['warning_desc'] = f'Translation is {percentage:.0f}% of original - somewhat concise'
        elif ratio > 4.0:
            result['warning'] = 'very_expansive_translation'
            result['warning_desc'] = 'Translation is over 400% of original - extensive additions'
        elif ratio > typical_max:
            result['warning'] = 'expansive_translation'
            result['warning_desc'] = f'Translation is {percentage:.0f}% of original - somewhat expansive'
        
        # Only flag as unreasonable if REALLY extreme
        if not is_reasonable:
            if ratio < min_ratio:
                result['error'] = 'possibly_missing_content'
                result['error_desc'] = f'Translation is only {percentage:.0f}% of original'
            else:
                result['error'] = 'possibly_excessive_content'
                result['error_desc'] = f'Translation is {percentage:.0f}% of original'
        
        return result
    
    return {
        'found_match': False,
        'chapter_num': chapter_num,
        'reason': 'No matching chapter found in original'
    }

def process_html_file_batch(args):
    """Process a batch of HTML or text files - MUST BE AT MODULE LEVEL"""
    file_batch, folder_path, qa_settings, mode, original_word_counts, merge_info, text_file_mode = args
    batch_results = []
    
    # Import what we need inside the worker
    import os
    import hashlib
    
    is_quick_scan = (mode == 'quick-scan')
    
    for idx, filename in file_batch:
        full_path = os.path.join(folder_path, filename)
        
        try:
            if text_file_mode:
                # For text files, read directly
                with open(full_path, 'r', encoding='utf-8') as f:
                    raw_text = f.read()
            else:
                raw_text = extract_text_from_html(full_path)
        except Exception as e:
            # Skip files that can't be read
            continue
        
        # Check minimum file length
        min_length = qa_settings.get('min_file_length', 0)
        if len(raw_text.strip()) < min_length:
            continue
        
        chapter_num, chapter_title = extract_chapter_info(filename, raw_text)
        
        # Quick scan optimizations
        if is_quick_scan:
            hashes = {}  # Empty dict for quick scan
            preview_size = min(300, len(raw_text))
        else:
            hashes = generate_content_hashes(raw_text)
            preview_size = 500
        
        preview = raw_text[:preview_size].replace('\n', ' ')
        if len(preview) > preview_size:
            preview = preview[:preview_size-3] + '...'
        
        # Normalize preview
        preview_normalized = normalize_text(preview)[:300]
        
        # Detect translation artifacts
        artifacts = []
        if not is_quick_scan and qa_settings.get('check_translation_artifacts', False):
            artifacts = detect_translation_artifacts(raw_text)
            
        # Filter out encoding_issues if disabled
        if not qa_settings.get('check_encoding_issues', True):
            artifacts = [a for a in artifacts if a['type'] != 'encoding_issues']
        
        # Initialize issues list
        issues = []

         # Check for glossary leakage
        check_glossary = qa_settings.get('check_glossary_leakage', True)
        if check_glossary and not is_quick_scan:
            has_glossary_leak, glossary_issues = detect_glossary_leakage(raw_text)
            
            if has_glossary_leak:
                # Add to translation artifacts
                for glossary_issue in glossary_issues:
                    artifacts.append({
                        'type': f"glossary_{glossary_issue['type']}",
                        'count': glossary_issue.get('count', 1),
                        'examples': glossary_issue.get('examples', []),
                        'severity': glossary_issue.get('severity', 'medium')
                    })
                
                # Add to issues list for reporting
                critical_glossary = any(g['severity'] == 'critical' for g in glossary_issues)
                if critical_glossary:
                    issues.append(f"CRITICAL_glossary_leakage_detected")
                else:
                    total_glossary_items = sum(g.get('count', 1) for g in glossary_issues)
                    issues.append(f"glossary_leakage_{total_glossary_items}_entries_found")
                    
        # HTML tag check (skip for text files)
        check_missing_html_tag = qa_settings.get('check_missing_html_tag', True)
        check_body_tag = qa_settings.get('check_body_tag', False)
        if not text_file_mode and check_missing_html_tag and filename.lower().endswith(('.html', '.xhtml', '.htm')):
            # Create a dummy log function for the worker
            def dummy_log(msg):
                pass
            
            has_issues, html_issues = check_html_structure_issues(full_path, dummy_log, check_body_tag=check_body_tag)
            
            if has_issues:
                for issue in html_issues:
                    if issue == 'missing_html_structure':
                        issues.append("missing_html_tag")
                    elif issue == 'missing_header_tags':
                        # Check if this check is enabled
                        check_missing_header = qa_settings.get('check_missing_header_tags', True)
                        if check_missing_header:
                            issues.append("missing_header_tags")
                    elif issue == 'insufficient_paragraph_tags':
                        issues.append("insufficient_paragraph_tags")
                    elif issue == 'unwrapped_text_content':
                        issues.append("unwrapped_text_content")
                    elif issue == 'unclosed_html_tags':
                        issues.append("unclosed_html_tags")
                    elif issue == 'incomplete_html_structure':
                        issues.append("incomplete_html_structure")
                    elif issue == 'invalid_nesting':
                        if qa_settings.get('check_invalid_nesting', False):
                            issues.append("invalid_nesting")
                    elif issue == 'malformed_html':
                        issues.append("malformed_html")
                    else:
                        issues.append(issue)
        
        # Check for multiple headers
        check_multiple_headers = qa_settings.get('check_multiple_headers', True)
        has_multiple = False
        header_count = 0
        header_info = None
        
        if check_multiple_headers:
            has_multiple, header_count, header_info = detect_multiple_headers(raw_text)
            if has_multiple:
                issues.append(f"multiple_headers_{header_count}_found")
        
        # Check word count ratio
        word_count_check = None
        check_word_count = qa_settings.get('check_word_count_ratio', False)
        
        if check_word_count and original_word_counts:
            # Create dummy log for worker
            def dummy_log(msg):
                pass
            
            # For text files, skip word count analysis on individual sections
            # (sections are arbitrary splits and can't be meaningfully compared to the whole source)
            if text_file_mode and 1 in original_word_counts:
                # Skip word count check for text file sections
                # Word count analysis only makes sense for the combined file
                pass
            else:
                # Normal EPUB mode
                wc_result = cross_reference_word_counts(
                    original_word_counts, 
                    filename, 
                    raw_text,
                    dummy_log,
                    merge_info
                )
                
                if wc_result['found_match']:
                    word_count_check = wc_result
                    # Only mark as issue if ratio is unreasonable (outside safe bounds)
                    if not wc_result['is_reasonable']:
                        issues.append(f"word_count_mismatch_ratio_{wc_result['ratio']:.2f}")
                else:
                    word_count_check = wc_result
                    issues.append("word_count_no_match_found")
        
        # Create result dictionary
        result = {
            "file_index": idx,
            "filename": filename,
            "filepath": full_path,
            "issues": issues,
            "preview": preview,
            "preview_normalized": preview_normalized,
            "score": 0,
            "chapter_num": chapter_num,
            "hashes": hashes,
            "raw_text": raw_text,
            "translation_artifacts": artifacts
        }
        
        # Add optional fields
        if check_multiple_headers and has_multiple:
            result['header_count'] = header_count
            result['header_info'] = header_info
        
        if word_count_check:
            result['word_count_check'] = word_count_check
        
        batch_results.append(result)
    
    return batch_results


def _init_worker_process():
    """Initializer for worker processes to keep GUI responsive.
    - On Windows: lower priority to BELOW_NORMAL and reserve some cores via affinity
    - On POSIX: increase niceness
    Uses environment var QA_RESERVE_CORES to decide how many logical cores to leave free (default 1).
    """
    try:
        # Reserve some CPUs for the GUI/main process
        reserve = 1
        try:
            reserve_env = int(os.environ.get('QA_RESERVE_CORES', '1'))
            if reserve_env >= 0:
                reserve = reserve_env
        except Exception:
            pass

        if psutil is not None:
            p = psutil.Process()
            # Lower process priority
            if os.name == 'nt':
                try:
                    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                except Exception:
                    pass
            else:
                try:
                    p.nice(10)
                except Exception:
                    pass

            # Set CPU affinity to leave some cores free if possible
            try:
                all_cpus = list(range(psutil.cpu_count(logical=True) or 1))
                if len(all_cpus) > reserve:
                    allowed = all_cpus[:max(1, len(all_cpus) - reserve)]
                    if allowed:
                        p.cpu_affinity(allowed)
            except Exception:
                pass
        else:
            # Fallback without psutil: best-effort niceness on POSIX
            if os.name != 'nt':
                try:
                    os.nice(10)
                except Exception:
                    pass
    except Exception:
        # Silent best-effort
        pass


def scan_html_folder(folder_path, log=print, stop_flag=None, mode='quick-scan', qa_settings=None, epub_path=None, selected_files=None, text_file_mode=None):
    """
    Scan HTML folder for QA issues - PROCESSPOOLEXECUTOR VERSION
    Also supports text file scanning (auto-detected from epub_path extension)
    """
    global _stop_flag
    _stop_flag = False
    
    # Auto-detect text file mode from epub_path extension if not explicitly specified
    if text_file_mode is None and epub_path:
        text_file_mode = epub_path.lower().endswith('.txt')
        if text_file_mode:
            log(f"📄 Text file mode auto-detected from source file extension")
    elif text_file_mode is None:
        text_file_mode = False
    
    # Create a combined stop check function
    def should_stop():
        if stop_flag and stop_flag():
            log("⛔ Stop requested via GUI stop button")
            return True
        if _stop_flag:
            log("⛔ Stop requested via global stop_scan() function")
            return True
        return False
    
    start_time = time.time()
    
    # Debug info
    log(f"🔍 Starting scan with ProcessPoolExecutor")
    log(f"⚡ MAXIMUM PERFORMANCE MODE ENABLED")
    
    # Load default settings if not provided
    if qa_settings is None:
        qa_settings = {
            'foreign_char_threshold': 10,
            'excluded_characters': '',
            'target_language': 'english',
            'check_encoding_issues': False,
            'check_repetition': True,
            'check_translation_artifacts': False,
            'check_glossary_leakage': True,
            'min_file_length': 0,
            'report_format': 'detailed',
            'auto_save_report': True,
            'check_missing_html_tag': True,
            'check_body_tag': False,
            'check_missing_header_tags': True,
            'check_paragraph_structure': True,
            'check_invalid_nesting': False,
            'paragraph_threshold': 0.3,
            'check_word_count_ratio': False,
            'check_multiple_headers': True,
            'warn_name_mismatch': True
        }
    
    check_word_count = qa_settings.get('check_word_count_ratio', False)
    check_multiple_headers = qa_settings.get('check_multiple_headers', True)
    
    # Extract word counts from original EPUB/text file if needed
    original_word_counts = {}
    merge_info = {}  # For request merging support
    combined_text_file = None  # For text file mode word count analysis
    
    if check_word_count:
        if text_file_mode:
            # For text files, extract word count from the original source text file
            # The source is the epub_path parameter (which is actually a .txt file in this mode)
            if epub_path and os.path.exists(epub_path) and epub_path.lower().endswith('.txt'):
                log(f"📝 Extracting word count from original text file: {os.path.basename(epub_path)}")
                try:
                    with open(epub_path, 'r', encoding='utf-8') as f:
                        source_text = f.read()
                    source_word_count = len(source_text.split())
                    # For text files, store word count under key 1 for all sections to reference
                    original_word_counts = {1: source_word_count}
                    log(f"   Source text word count: {source_word_count} words")
                    log(f"   Will compare each section file against this total")
                    
                    # Find the combined translated file for additional reporting
                    # Look for *_translated.txt in the folder
                    try:
                        txt_files = [f for f in os.listdir(folder_path) if f.lower().endswith('_translated.txt')]
                        if txt_files:
                            combined_text_file = txt_files[0]  # Use first match
                            log(f"   Combined file found: {combined_text_file}")
                    except Exception:
                        pass
                except Exception as e:
                    log(f"   ⚠️ Could not extract word count from text file: {e}")
                    check_word_count = False
            else:
                log("⚠️ Word count cross-reference enabled but no valid text file provided - skipping this check")
                check_word_count = False
        elif epub_path and os.path.exists(epub_path):
            log(f"📚 Extracting word counts from original EPUB: {os.path.basename(epub_path)}")
            min_length = qa_settings.get('min_file_length', 0)
            original_word_counts = extract_epub_word_counts(epub_path, log, min_file_length=min_length)
            log(f"   Found word counts for {len(original_word_counts)} chapters (min length: {min_length} chars)")
            
            # Load merge info from translation_progress.json for request merging support
            progress_file = os.path.join(folder_path, 'translation_progress.json')
            if os.path.exists(progress_file):
                try:
                    with open(progress_file, 'r', encoding='utf-8') as f:
                        progress_data = json.load(f)
                    
                    # Build merge_info: {parent_chapter_num: [list of merged child chapter nums]}
                    # Method 1: From parent chapters with merged_chapters list
                    for chapter_key, chapter_data in progress_data.get('chapters', {}).items():
                        if isinstance(chapter_data, dict) and chapter_data.get('merged_chapters'):
                            try:
                                parent_num = int(chapter_key)
                                merged_children = [int(c) for c in chapter_data['merged_chapters']]
                                merge_info[parent_num] = merged_children
                            except (ValueError, TypeError):
                                pass
                    
                    # Method 2: From child chapters with status: "merged" and merged_parent_chapter
                    # This ensures we capture merging even if we only have the child info
                    for chapter_key, chapter_data in progress_data.get('chapters', {}).items():
                        if isinstance(chapter_data, dict) and chapter_data.get('status') == 'merged':
                            try:
                                child_num = chapter_data.get('actual_num')
                                if child_num is None:
                                    child_num = int(chapter_key)
                                parent_num = chapter_data.get('merged_parent_chapter')
                                if parent_num is not None:
                                    if parent_num not in merge_info:
                                        merge_info[parent_num] = []
                                    if child_num not in merge_info[parent_num]:
                                        merge_info[parent_num].append(child_num)
                            except (ValueError, TypeError):
                                pass
                    
                    if merge_info:
                        total_children = sum(len(v) for v in merge_info.values())
                        log(f"   🔗 Found {len(merge_info)} parent chapters with {total_children} merged children")
                except Exception as e:
                    log(f"   ⚠️ Could not load merge info: {e}")
        else:
            log("⚠️ Word count cross-reference enabled but no valid EPUB provided - skipping this check")
            check_word_count = False
    
    # Log settings
    log(f"\n📋 QA Settings Status:")
    log(f"   ✓ Target language: {qa_settings.get('target_language', 'english').upper()}")
    log(f"   ✓ Foreign char threshold: {qa_settings.get('foreign_char_threshold', 10)}")
    log(f"   ✓ Encoding issues check: {'ENABLED' if qa_settings.get('check_encoding_issues', True) else 'DISABLED'}")
    log(f"   ✓ Repetition check: {'ENABLED' if qa_settings.get('check_repetition', True) else 'DISABLED'}")
    log(f"   ✓ Translation artifacts check: {'ENABLED' if qa_settings.get('check_translation_artifacts', False) else 'DISABLED'}")
    log(f"   ✓ Missing HTML tag check: {'ENABLED' if qa_settings.get('check_missing_html_tag', False) else 'DISABLED'}")
    log(f"   ✓ Missing header tags check: {'ENABLED' if qa_settings.get('check_missing_header_tags', True) else 'DISABLED'}")
    log(f"   ✓ Paragraph structure check: {'ENABLED' if qa_settings.get('check_paragraph_structure', True) else 'DISABLED'}")    
    log(f"   ✓ Invalid nesting check: {'ENABLED' if qa_settings.get('check_invalid_nesting', False) else 'DISABLED'}") 
    log(f"   ✓ Word count ratio check: {'ENABLED' if qa_settings.get('check_word_count_ratio', False) else 'DISABLED'}") 
    log(f"   ✓ Multiple headers check: {'ENABLED' if qa_settings.get('check_multiple_headers', False) else 'DISABLED'}")
    
    # Initialize configuration
    custom_settings = None
    if mode == 'custom' and qa_settings and 'custom_mode_settings' in qa_settings:
        custom_settings = qa_settings['custom_mode_settings']
    config = DuplicateDetectionConfig(mode, custom_settings)
    
    # Log mode info
    mode_messages = {
        'aggressive': '🚨 AGGRESSIVE',
        'quick-scan': '⚡ Quick Scan',
        'custom': '⚙️ Custom',
        'ai-hunter': '🤖 AI HUNTER'
    }
    
    log(f"{mode_messages.get(mode, '📋 Standard')} duplicate detection mode")
    log(f"   Thresholds: {config.thresholds[mode]}")
    
    if mode == 'ai-hunter':
        log("   ⚠️ WARNING: This mode will flag almost everything as potential duplicates!")
        log("   🎯 Designed specifically for catching AI retranslations of the same content")
        log("   ⏱️ NOTE: AI Hunter mode checks EVERY file pair - but now with PARALLEL PROCESSING!")
    
    # Get files to scan (HTML or text based on mode)
    if text_file_mode:
        # For text files, scan section files (including response_ prefix versions)
        all_txt_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".txt")])
        # Filter out only combined files (_translated.txt)
        html_files = [f for f in all_txt_files if not f.endswith('_translated.txt')]
        log(f"📄 Text file mode enabled - scanning section files (response_ prefix ignored for comparison)")
    else:
        html_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith((".html", ".xhtml", ".htm"))])
    
    # If specific files were selected, filter to those (by basename)
    if selected_files:
        try:
            selected_basenames = {os.path.basename(p) for p in selected_files}
            html_files = [f for f in html_files if f in selected_basenames]
            log(f"📄 Limited scan to {len(html_files)} selected file(s)")
        except Exception:
            pass
    
    file_type = "text" if text_file_mode else "HTML"
    log(f"🔍 Found {len(html_files)} {file_type} files. Starting parallel scan...")
    
    # Determine number of workers
    cpu_count = multiprocessing.cpu_count()
    max_workers_config = 0
    
    try:
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                full_config = json.load(f)
                # Check multiple possible config locations
                qa_config = full_config.get('qa_scanner_config', {})
                ai_hunter_config = full_config.get('ai_hunter_config', {})
                
                # Priority: qa_scanner_config > ai_hunter_config
                max_workers_config = qa_config.get('max_workers',
                                    ai_hunter_config.get('ai_hunter_max_workers', 1))
    except:
        max_workers_config = 0
    
    if max_workers_config > 0:
        max_workers = min(max_workers_config, cpu_count)
        log(f"   🖥️ Using {max_workers} CPU cores for file processing (configured limit)")
    else:
        max_workers = cpu_count
        log(f"   🚀 Using ALL {max_workers} CPU cores for file processing")
        if cpu_count > 8:
            log(f"   💡 Tip: You can limit CPU cores in QA scanner settings")
    
    # Create file batches with indices
    file_list = [(idx, filename) for idx, filename in enumerate(html_files)]
    batch_size = max(10, len(html_files) // (max_workers * 5))
    batches = []
    
    for i in range(0, len(file_list), batch_size):
        batch = file_list[i:i + batch_size]
        batches.append(batch)
    
    log(f"   📦 Split into {len(batches)} batches of ~{batch_size} files each")
    
    # Prepare worker data
    worker_args = []
    for batch in batches:
        args = (batch, folder_path, qa_settings, mode, original_word_counts, merge_info, text_file_mode)
        worker_args.append(args)
    
    # Process files in parallel
    results = []
    processed_count = 0
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker_process) as executor:
        # Submit all batches
        futures = []
        
        for args in worker_args:
            if should_stop():
                log("⛔ QA scan interrupted before processing.")
                executor.shutdown(wait=True)
                return
            
            future = executor.submit(process_html_file_batch, args)
            futures.append(future)
        
        # Collect results as they complete
        for completed_idx, future in enumerate(concurrent.futures.as_completed(futures)):
            if should_stop():
                log("⛔ QA scan interrupted during processing.")
                executor.shutdown(wait=True)
                return
            
            try:
                batch_results = future.result()
                
                # Log individual file progress like original
                for result in batch_results:
                    processed_count += 1
                    idx = result['file_index']
                    filename = result['filename']
                    
                    # Progress update every 10 files (like original)
                    if processed_count % 10 == 0:
                        progress = int((processed_count / len(html_files)) * 100)
                        log(f"📄 [{processed_count}/{len(html_files)}] Scanning {filename}... ({progress}% complete)")
                        
                        # Debug: Check stop flag states periodically (like original)
                        if processed_count % 50 == 0 and processed_count > 0:
                            log(f"   [DEBUG] Global stop flag: {_stop_flag}, Stop function: {stop_flag() if stop_flag else 'N/A'}")
                    elif processed_count % 5 == 0:
                        # Show progress every 5 files for GUI visibility
                        progress = int((processed_count / len(html_files)) * 100)
                        log(f"📄 [{processed_count}/{len(html_files)}] {filename}... ({progress}%)")
                    
                    # Log issues found (like original)
                    if result.get('issues'):
                        # Check if HTML structure issues were found
                        html_issues = [i for i in result['issues'] if 'html' in i.lower() or 'paragraph' in i.lower()]
                        if html_issues:
                            log(f"   → Found HTML structure issues in {filename}: {', '.join(html_issues)}")
                        
                        # Log word count issues
                        wc_issues = [i for i in result['issues'] if 'word_count' in i]
                        if wc_issues and result.get('word_count_check'):
                            wc = result['word_count_check']
                            if wc.get('ratio'):
                                log(f"   {filename}: Word count ratio {wc['ratio']:.2f} " +
                                    f"(Original: {wc.get('original_wc', '?')}, Translated: {wc.get('translated_wc', '?')})")
                        
                        # Log encoding artifacts (if enabled)
                        if qa_settings.get('check_encoding_issues', True):
                            encoding_issues = [i for i in result['issues'] if 'encoding' in i]
                            if encoding_issues and processed_count <= 5:  # Only log first 5
                                count = next((int(i.split('_')[2]) for i in encoding_issues if '_found' in i), 0)
                                if count > 0:
                                    log(f"      → Found encoding artifacts in {filename}: {count} instances")
                        
                        # Log spacing issues
                        if 'no_spacing_or_linebreaks' in result['issues'] and processed_count <= 5:
                            log(f"      → Found spacing/linebreak issue in {filename}")
                        
                        # Log API response unavailable markers
                        api_issues = [i for i in result['issues'] if 'api_response_unavailable' in i]
                        if api_issues and processed_count <= 5:
                            count = next((int(i.split('_')[3]) for i in api_issues if '_found' in i), 0)
                            if count > 0:
                                log(f"      → Found AI response unavailable markers in {filename}: {count} instances")
                
                results.extend(batch_results)
                
            except Exception as e:
                log(f"   ❌ Error processing batch: {e}")
                import traceback
                log(f"   Traceback: {traceback.format_exc()}")
    
    # Clear the progress line (like original)
    print()  # New line after progress indicator
    
    # Sort results by file index to maintain order
    results.sort(key=lambda x: x['file_index'])
    
    log("\n✅ Initial scan complete.")
    
    # Time the duplicate detection phase
    dup_start_time = time.time()
    
    # Detect duplicates (already optimized)
    duplicate_groups, near_duplicate_groups, duplicate_confidence = detect_duplicates(
        results, log, should_stop, config
    )
    
    dup_time = time.time() - dup_start_time
    log(f"✅ Duplicate detection completed in {dup_time:.1f} seconds")
    
    # For text file mode with word count enabled, check the combined file separately
    if text_file_mode and check_word_count and combined_text_file and original_word_counts:
        log("\n📊 Analyzing word count for combined text file...")
        try:
            combined_path = os.path.join(folder_path, combined_text_file)
            if os.path.exists(combined_path):
                with open(combined_path, 'r', encoding='utf-8') as f:
                    combined_text = f.read()
                translated_word_count = len(combined_text.split())
                original_wc = original_word_counts[1]  # We stored it under key 1
                
                if original_wc > 0:
                    ratio = translated_word_count / original_wc
                    log(f"   Original: {original_wc} words")
                    log(f"   Translated: {translated_word_count} words")
                    log(f"   Ratio: {ratio:.2f}")
                    
                    # Determine if ratio is reasonable (between 0.7 and 2.0)
                    is_reasonable = 0.7 <= ratio <= 2.0
                    if is_reasonable:
                        log(f"   ✅ Word count ratio is reasonable")
                    else:
                        log(f"   ⚠️ Word count ratio seems unusual (expected 0.7-2.0)")
        except Exception as e:
            log(f"   ⚠️ Could not analyze combined file word count: {e}")
    
    # Process results and check for additional issues
    log("\n📊 Checking for other issues...")
    
    # Group files by duplicate group
    groups = {}
    for filename, group_id in duplicate_groups.items():
        if group_id not in groups:
            groups[group_id] = []
        groups[group_id].append(filename)
    
    # Check each file for all issues (this part is fast, no need to parallelize)
    for idx, result in enumerate(results):
        issues = result.get('issues', [])
        
        # Check duplicates
        if result['filename'] in duplicate_groups:
            group_id = duplicate_groups[result['filename']]
            group_files = groups[group_id]
            if len(group_files) > 1:
                others = [f for f in group_files if f != result['filename']]
                
                # Get confidence score
                confidence = 0
                for other in others:
                    pair = tuple(sorted([result['filename'], other]))
                    if pair in duplicate_confidence:
                        confidence = max(confidence, duplicate_confidence[pair])
                
                result['duplicate_confidence'] = confidence
                
                if len(others) == 1:
                    issues.append(f"DUPLICATE: exact_or_near_copy_of_{others[0]}")
                else:
                    issues.append(f"DUPLICATE: part_of_{len(group_files)}_file_group")
        
        # Check near-duplicates
        elif result['filename'] in near_duplicate_groups:
            near_group_id = near_duplicate_groups[result['filename']]
            near_group_files = [f for f, gid in near_duplicate_groups.items() if gid == near_group_id]
            if len(near_group_files) > 1:
                others = [f for f in near_group_files if f != result['filename']]
                if len(others) == 1:
                    issues.append(f"NEAR_DUPLICATE: highly_similar_to_{others[0]}")
                else:
                    issues.append(f"NEAR_DUPLICATE: similar_to_{len(near_group_files)-1}_other_files")
        
        # Check other issues
        raw_text = result['raw_text']
        
        # Non-English content
        has_non_english, lang_issues = detect_non_english_content(raw_text, qa_settings)
        if has_non_english:
            issues.extend(lang_issues)
        
        # Spacing/formatting issues
        if qa_settings.get('check_encoding_issues', True):
            min_text_len = qa_settings.get('min_text_length_for_spacing', 100)
            if has_no_spacing_or_linebreaks(raw_text, min_text_length=min_text_len):
                issues.append("no_spacing_or_linebreaks")
        
        # Repetitive content
        if qa_settings.get('check_repetition', True):
            if has_repeating_sentences(raw_text):
                issues.append("excessive_repetition")
        
        # Translation artifacts
        if result.get('translation_artifacts'):
            for artifact in result['translation_artifacts']:
                if artifact['type'] == 'machine_translation':
                    issues.append(f"machine_translation_markers_{artifact['count']}_found")
                elif artifact['type'] == 'encoding_issues':
                    if qa_settings.get('check_encoding_issues', True):
                        issues.append(f"encoding_issues_{artifact['count']}_found")
                elif artifact['type'] == 'repeated_watermarks':
                    issues.append(f"repeated_watermarks_{artifact['count']}_found")
                elif artifact['type'] == 'api_response_unavailable':
                    issues.append(f"api_response_unavailable_{artifact['count']}_found")
                elif artifact['type'] == 'chapter_continuation':
                    issues.append(f"chapter_continuation_{artifact['count']}_found")
                elif artifact['type'] == 'split_indicators':
                    issues.append(f"split_indicators_{artifact['count']}_found")
                elif 'glossary_' in artifact['type']:
                    severity = artifact.get('severity', 'medium')
                    if severity == 'critical':
                        issues.append(f"CRITICAL_{artifact['type']}_{artifact['count']}_found")
                    else:
                        issues.append(f"{artifact['type']}_{artifact['count']}_found")
                
        
        result['issues'] = issues
        result['score'] = len(issues)
        
        if issues:
            log(f"   {result['filename']}: {', '.join(issues[:2])}" + (" ..." if len(issues) > 2 else ""))
    
    # Clean up to save memory
    for result in results:
        result.pop('raw_text', None)
        result.pop('hashes', None)
        result.pop('semantic_sig', None)
        result.pop('structural_sig', None)
        result.pop('normalized_text', None)
    
    # Generate reports
    generate_reports(results, folder_path, duplicate_confidence, log, qa_settings)
    
    # Update progress file
    update_progress_file(folder_path, results, log)
    
    # Final timing
    total_time = time.time() - start_time
    log(f"\n⏱️ Total scan time: {total_time:.1f} seconds")
    if total_time > 60:
        log(f"   ({int(total_time // 60)} minutes {int(total_time % 60)} seconds)")
    
    log("⚡ ProcessPoolExecutor: ENABLED - Maximum performance achieved!")


def check_html_structure_issues(file_path, log, check_body_tag=False):
    """
    Check for HTML structure problems including unwrapped text and unclosed tags.
    
    Returns:
        tuple: (has_issues, issue_types) where issue_types is a list of specific issues found
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        issues = []
        
        # Check 1: Empty file
        if not content.strip():
            issues.append('missing_html_structure')
            return True, issues
            
        # Check 2: No HTML tags at all
        if '<' not in content or '>' not in content:
            issues.append('missing_html_structure')
            return True, issues
        
        # Check 3: Large blocks of unwrapped text
        from bs4 import BeautifulSoup, NavigableString
        try:
            soup = BeautifulSoup(content, 'html.parser')
            
            # Look for text that's sitting directly in body (not in any tag)
            body = soup.find('body')
            if body:
                unwrapped_text_total = 0
                
                # Check all direct children of body
                for element in body.children:
                    if isinstance(element, NavigableString):
                        text = str(element).strip()
                        # Count any non-whitespace text
                        if text and not text.isspace():
                            unwrapped_text_total += len(text)
                
                # If we found significant unwrapped text, that's a problem
                if unwrapped_text_total > 100:  # More than 100 chars of unwrapped text
                    issues.append('unwrapped_text_content')
                    log(f"   Found {unwrapped_text_total} characters of unwrapped text")
        
        except Exception as e:
            log(f"   Warning: Could not parse HTML structure: {e}")
        
        # Check 4: Malformed and incomplete HTML tags
        import re
        
        content_lower = content.lower()
        
        # First check for incomplete/malformed opening tags (like <p without closing >)
        # Look for < followed by tag name but missing the closing >
        malformed_patterns = [
            # Match <tag followed by anything that's not > and then encountering text/quote
            (r'<([a-zA-Z]+)(?![^>]*>)[^>]*["\w]', 'incomplete_opening_tag'),
            # Match tags like <p" or <div' where quote comes right after tag name
            (r'<([a-zA-Z]+)["\']', 'malformed_tag_with_quote'),
            # Match < followed by tag name and then immediately text without >
            (r'<(p|div|span|a|img|h[1-6])\s*[^>\s]+[^>]*$', 'incomplete_tag_at_line_end'),
        ]
        
        # Check for orphaned closing brackets (e.g., p> without <p)
        # Only check for common HTML tags to avoid false positives
        common_html_tags = ['p', 'div', 'span', 'a', 'img', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                           'ul', 'ol', 'li', 'table', 'tr', 'td', 'th', 'form', 'input', 'button',
                           'nav', 'header', 'footer', 'section', 'article', 'aside', 'main']
        orphaned_pattern = r'(?:^|\s)(' + '|'.join(common_html_tags) + r')>(?!\w)'
        orphaned_matches = re.findall(orphaned_pattern, content, re.IGNORECASE | re.MULTILINE)
        if orphaned_matches:
            malformed_patterns.append((orphaned_pattern, 'orphaned_closing_bracket'))
        
        malformed_tags_found = []
        for pattern, issue_type in malformed_patterns:
            matches = re.findall(pattern, content, re.MULTILINE | re.IGNORECASE)
            if matches:
                for match in matches[:3]:  # Show first 3 examples
                    # Find the actual malformed tag in context
                    tag_name = match if isinstance(match, str) else match[0] if isinstance(match, tuple) else str(match)
                    if issue_type == 'orphaned_closing_bracket':
                        malformed_tags_found.append(f"{tag_name}> ({issue_type})")
                    else:
                        malformed_tags_found.append(f"<{tag_name} ({issue_type})")
        
        if malformed_tags_found:
            issues.append('malformed_html')
            log(f"   Found malformed HTML tags: {', '.join(malformed_tags_found[:3])}" + 
                (" ..." if len(malformed_tags_found) > 3 else ""))
        
        # Check for unclosed HTML tags - Check common tags with simple logic
        tags_to_check = ['html', 'head', 'p', 'div', 'span']
        if check_body_tag:
            tags_to_check.insert(1, 'body')  # Add body after html if enabled
        problematic_tags = []
        
        for tag in tags_to_check:
            # Count: <tag (with space, attributes, or direct close)
            open_count = len(re.findall(rf'<{tag}(?:\s[^>]*)?>', content_lower))
            # Count: </tag>
            close_count = len(re.findall(rf'</{tag}>', content_lower))
            
            # Flag only if there's a real imbalance
            # Allow 1-2 difference for edge cases, but flag significant mismatches
            diff = abs(open_count - close_count)
            
            if open_count > 0 or close_count > 0:  # Tag exists in file
                if diff > 2:  # Significant mismatch
                    problematic_tags.append(f"{tag} (open: {open_count}, close: {close_count})")
        
        if problematic_tags:
            issues.append('unclosed_html_tags')
            log(f"   Found tag mismatches: {', '.join(problematic_tags[:3])}" + 
                (" ..." if len(problematic_tags) > 3 else ""))
        
        # Check 5: Basic HTML structure validation - only check for consistency, not completeness
        # Define all structure check variables
        html_open_exists = bool(re.search(r'<html[^>]*>', content_lower))
        html_close_exists = bool(re.search(r'</html>', content_lower))
        head_open_exists = bool(re.search(r'<head[^>]*>', content_lower))
        head_close_exists = bool(re.search(r'</head>', content_lower))
        body_open_exists = bool(re.search(r'<body[^>]*>', content_lower))
        body_close_exists = bool(re.search(r'</body>', content_lower))
        
        # Check for missing heading tags (h1-h6)
        has_heading_tag = False
        for heading_level in range(1, 7):  # h1 through h6
            if re.search(rf'<h{heading_level}[^>]*>', content_lower):
                has_heading_tag = True
                break
        
        if not has_heading_tag:
            issues.append('missing_header_tags')
            log(f"   HTML file is missing heading tags (h1-h6)")
        
        missing_structure = []
        
        # Only flag if tags are opened but not closed (or vice versa)
        if html_open_exists and not html_close_exists:
            missing_structure.append('closing </html>')
        elif html_close_exists and not html_open_exists:
            missing_structure.append('opening <html>')
            
        if head_open_exists and not head_close_exists:
            missing_structure.append('closing </head>')
        elif head_close_exists and not head_open_exists:
            missing_structure.append('opening <head>')
            
        # Only check body tags if enabled
        if check_body_tag:
            if body_open_exists and not body_close_exists:
                missing_structure.append('closing </body>')
            elif body_close_exists and not body_open_exists:
                missing_structure.append('opening <body>')
        
        # Only flag as incomplete if there are actual mismatches
        if missing_structure:
            issues.append('incomplete_html_structure')
            log(f"   Mismatched HTML structure tags: {', '.join(missing_structure)}")
        
        # Check 6: Nested tag validation using BeautifulSoup's parser errors
        try:
            # Parse with html.parser which is more strict
            soup_strict = BeautifulSoup(content, 'html.parser')
            
            # Check for common nesting issues
            # For example, p tags shouldn't contain div tags
            invalid_nesting = []
            
            # Check for p tags containing block elements
            for p_tag in soup_strict.find_all('p'):
                block_elements = p_tag.find_all(['div', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 
                                                'ul', 'ol', 'li', 'blockquote', 'pre', 'table'])
                if block_elements:
                    invalid_nesting.append(f"<p> contains block elements: {[el.name for el in block_elements[:3]]}")
            
            # Check for list items outside of lists
            all_li = soup_strict.find_all('li')
            for li in all_li:
                parent = li.parent
                if parent and parent.name not in ['ul', 'ol']:
                    invalid_nesting.append(f"<li> not inside <ul> or <ol>")
                    break  # Only report once
            
            if invalid_nesting:
                issues.append('invalid_nesting')
                log(f"   Found invalid tag nesting: {'; '.join(invalid_nesting[:3])}" + 
                    (" ..." if len(invalid_nesting) > 3 else ""))
                    
        except Exception as e:
            # BeautifulSoup might throw exceptions for severely malformed HTML
            log(f"   Warning: HTML parsing error (possible malformed structure): {str(e)[:100]}")
            issues.append('malformed_html')
        
        # Check 7: Final validation for critical mismatched tags
        # Only flag if we have opening tags without closing tags (not missing both)
        if html_open_exists and not html_close_exists:
            if 'incomplete_html_structure' not in issues:
                issues.append('incomplete_html_structure')
            if 'unclosed_html_tags' not in issues:
                issues.append('unclosed_html_tags')
                log(f"   Critical: Found opening <html> tag but missing closing </html> tag")
        
        # Only flag unclosed body tags if check is enabled
        if check_body_tag and body_open_exists and not body_close_exists:
            if 'unclosed_html_tags' not in issues:
                issues.append('unclosed_html_tags')
                log(f"   Critical: Found opening <body> tag but missing closing </body> tag")
        
        return len(issues) > 0, issues
        
    except Exception as e:
        log(f"Error checking HTML structure for {file_path}: {e}")
        return False, []

def check_insufficient_paragraph_tags(html_content, threshold=0.3):
    """
    Check if HTML content has insufficient paragraph tags.
    
    Args:
        html_content: The raw HTML content from the file
        threshold: Minimum ratio of text that should be in paragraph tags (default 0.3 = 30%)
    
    Returns:
        bool: True if file has insufficient paragraph tags
    """
    from bs4 import BeautifulSoup, NavigableString
    
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Get total text length
        total_text = soup.get_text(strip=True)
        total_length = len(total_text)
        
        # Skip short files
        if total_length < 200:
            return False
        
        # Count text in paragraph tags
        p_text_length = 0
        for p in soup.find_all('p'):
            p_text_length += len(p.get_text(strip=True))
        
        # Also check for unwrapped text in body
        body = soup.find('body')
        if body:
            for element in body.children:
                if isinstance(element, NavigableString):
                    text = str(element).strip()
                    if len(text) > 50:  # Significant unwrapped text block
                        # If we find big chunks of unwrapped text, flag it
                        return True
        
        # Calculate ratio
        if total_length == 0:
            return False
            
        ratio = p_text_length / total_length
        
        # Flag if not enough text is in paragraphs
        return ratio < threshold
        
    except Exception as e:
        print(f"Error checking paragraph tags: {e}")
        return False

        
def launch_gui():
    """Launch GUI interface with mode selection"""
    def run_scan():
        folder_path = filedialog.askdirectory(title="Select Folder with HTML Files")
        if folder_path:
            mode = mode_var.get()
            
            def scan_thread():
                scan_html_folder(folder_path, print, None, mode)
            
            threading.Thread(target=scan_thread, daemon=True).start()
            
            # Show status
            status_label.config(text=f"Scanning in {mode} mode...")
            root.update()
    
    root = tk.Tk()
    root.title("Translation QA Scanner - Enhanced Edition")
    root.geometry("690x200")
    
    # Mode selection
    mode_frame = tk.Frame(root)
    mode_frame.pack(pady=10)
    
    tk.Label(mode_frame, text="Detection Mode:").pack(side=tk.LEFT, padx=5)
    
    mode_var = tk.StringVar(value="quick-scan")
    modes = [
        ("Aggressive (75% threshold)", "aggressive"),
        ("Quick Scan (85% threshold)", "quick-scan"),
        ("Custom (Configurable)", "custom"),
        ("AI Hunter (30% text, 85% semantic)", "ai-hunter")
    ]
    
    for text, mode in modes:
        tk.Radiobutton(mode_frame, text=text, variable=mode_var, value=mode).pack(side=tk.LEFT, padx=5)
    
    # Scan button
    scan_button = tk.Button(root, text="Scan Folder for QA Issues", 
                           command=run_scan, height=2, width=30)
    scan_button.pack(pady=20)
    
    # Status label
    status_label = tk.Label(root, text="")
    status_label.pack(pady=5)
    
    # Info label
    info_text = "Enhanced scanner with semantic analysis, structural patterns, and fuzzy matching"
    if not MINHASH_AVAILABLE:
        info_text += "\n(Install 'datasketch' for faster processing of large datasets)"
    
    info_label = tk.Label(root, text=info_text, fg="gray")
    info_label.pack(pady=5)
    
    root.mainloop()

if __name__ == "__main__":
    import sys
    import argparse
    
    if len(sys.argv) < 2:
        launch_gui()
    else:
        # Parse command-line arguments
        parser = argparse.ArgumentParser(description='QA Scanner for translated HTML files')
        parser.add_argument('folder', help='Folder containing HTML files to scan')
        parser.add_argument('--mode', choices=['quick-scan', 'aggressive', 'ai-hunter', 'custom', 'standard'], 
                           default='quick-scan', help='Detection mode to use')
        parser.add_argument('--interactive', action='store_true', 
                           help='Show custom settings dialog for custom mode')
        parser.add_argument('--aggressive', action='store_true', help='Use aggressive mode (deprecated, use --mode aggressive)')
        parser.add_argument('--custom', action='store_true', help='Use custom mode (deprecated, use --mode custom)')
        parser.add_argument('--quick-scan', action='store_true', help='Use quick-scan mode (deprecated, use --mode quick-scan)')
        parser.add_argument('--ai-hunter', action='store_true', help='Use ai-hunter mode (deprecated, use --mode ai-hunter)')
        
        args = parser.parse_args()
        
        # Handle deprecated flags
        if args.aggressive:
            args.mode = 'aggressive'
        elif args.custom:
            args.mode = 'custom'
        elif args.quick_scan:
            args.mode = 'quick-scan'
        elif args.ai_hunter:
            args.mode = 'ai-hunter'
        
        # For custom mode with --interactive, show the custom detection settings dialog
        if args.mode == 'custom' and args.interactive:
            print("\ud83c\udfdb️ Opening Custom Detection Settings dialog...")
            from PySide6.QtWidgets import QApplication
            from QA_Scanner_GUI import show_custom_detection_dialog
            
            app = QApplication(sys.argv)
            
            # Show the dialog and get the settings
            settings_dict = show_custom_detection_dialog(None)
            
            if settings_dict:
                # User confirmed settings, now run the scan
                print("\u2705 Custom settings configured, starting scan...")
                
                # Create a config object from the settings
                config = DuplicateDetectionConfig()
                config.similarity_threshold = settings_dict['text_similarity'] / 100.0
                config.semantic_threshold = settings_dict['semantic_analysis'] / 100.0
                config.structural_threshold = settings_dict['structural_patterns'] / 100.0
                config.word_overlap_threshold = settings_dict['word_overlap'] / 100.0
                config.minhash_threshold = settings_dict['minhash_similarity'] / 100.0
                config.consecutive_chapters = settings_dict['consecutive_chapters']
                config.sample_size = settings_dict['sample_size']
                config.min_length = settings_dict['min_text_length']
                config.check_all_pairs = settings_dict.get('check_all_pairs', False)
                
                # Run the scan with custom settings
                scan_html_folder(args.folder, mode=args.mode, config=config)
            else:
                print("\u274c Custom settings dialog cancelled, scan aborted")
        else:
            # Run scan normally without dialog
            scan_html_folder(args.folder, mode=args.mode)



def reset_stop_flag():
    """Reset the stop flag - useful for starting a new scan"""
    global _stop_flag
    _stop_flag = False
    print("🔄 Stop flag reset to False")

def is_stop_requested():
    """Check if stop has been requested"""
    global _stop_flag
    return _stop_flag

# Export the stop_scan function so GUI can call it
__all__ = ['scan_html_folder', 'stop_scan', 'reset_stop_flag', 'is_stop_requested', 
          'DuplicateDetectionConfig', 'test_stop_functionality']

def test_stop_functionality():
    """Test function to verify stop_scan works"""
    global _stop_flag
    print(f"Before stop_scan: _stop_flag = {_stop_flag}")
    stop_scan()
    print(f"After stop_scan: _stop_flag = {_stop_flag}")
    _stop_flag = False  # Reset
    return True


# ADD THIS AT MODULE LEVEL (outside any function/class)

def process_comparison_batch_fast(args):
    """Process a batch of comparisons - MUST BE AT MODULE LEVEL FOR PICKLING"""
    batch, data = args
    batch_results = []
    
    all_data = data['all_data']
    thresholds = data['thresholds']
    
    # Import what we need inside the worker
    from difflib import SequenceMatcher
    
    # Import the similarity functions - they must also be at module level
    # If they're in the same module, you might need to import them explicitly
    # from scan_html_folder import calculate_semantic_similarity, calculate_structural_similarity
    
    for i, j in batch:
        data_i = all_data[i]
        data_j = all_data[j]
        
        # Calculate ALL similarities - NO SHORTCUTS
        
        # 1. Semantic similarity
        sem_sim = calculate_semantic_similarity(
            data_i['semantic_sig'], 
            data_j['semantic_sig']
        )
        
        # 2. Structural similarity
        struct_sim = calculate_structural_similarity(
            data_i['structural_sig'],
            data_j['structural_sig']
        )
        
        # 3. Text similarity - ALWAYS calculate
        text_sim = 0.0
        if data_i['text_hash'] and data_j['text_hash']:
            if data_i['text_hash'] == data_j['text_hash']:
                text_sim = 1.0
            else:
                # Always calculate full similarity
                text_sim = SequenceMatcher(
                    None, 
                    data_i['text'], 
                    data_j['text']
                ).ratio()
        
        # Check ALL duplicate conditions
        is_duplicate = False
        is_retranslation = False
        confidence = 0.0
        
        # AI Hunter logic: High semantic + high structural = likely duplicate
        if sem_sim >= thresholds['semantic'] and struct_sim >= thresholds['structural']:
            is_duplicate = True
            is_retranslation = text_sim < 0.6
            confidence = (sem_sim + struct_sim) / 2
        # Traditional similarity check
        elif text_sim >= thresholds['similarity']:
            is_duplicate = True
            is_retranslation = False
            confidence = text_sim
        
        # Store result if duplicate found
        if is_duplicate:
            batch_results.append({
                'i': i,
                'j': j,
                'sem_sim': sem_sim,
                'struct_sim': struct_sim,
                'text_sim': text_sim,
                'is_duplicate': True,
                'is_retranslation': is_retranslation,
                'confidence': confidence
            })
    
    return batch_results


def parallel_ai_hunter_check(results, duplicate_groups, duplicate_confidence, config, log, should_stop):
    """Parallel AI Hunter checking - FIXED FOR PROCESSPOOLEXECUTOR"""
    
    log("🤖 AI Hunter mode: Enhanced semantic and structural checking active")
    log("⚡ PARALLEL PROCESSING ENABLED - MAXIMUM PERFORMANCE!")
    
    total_comparisons = (len(results) * (len(results) - 1)) // 2
    log(f"   ⚠️ Will check ALL {total_comparisons:,} file pairs - NO COMPROMISES!")
    
    # Determine number of workers
    cpu_count = multiprocessing.cpu_count()
    max_workers_config = 0
    
    try:
        import json
        import os
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                full_config = json.load(f)
                ai_hunter_config = full_config.get('ai_hunter_config', {})
                max_workers_config = ai_hunter_config.get('ai_hunter_max_workers', 1)
    except:
        max_workers_config = 0
    
    if max_workers_config > 0:
        max_workers = min(max_workers_config, cpu_count)
        log(f"   🖥️ Using {max_workers} parallel workers (configured limit of {max_workers_config})")
    else:
        max_workers = cpu_count
        log(f"   🚀 Using ALL {max_workers} CPU cores - MAXIMUM PERFORMANCE!")
    
    # Pre-compute everything once
    log("   📊 Pre-computing all data structures...")
    
    # Build a single data structure with everything we need
    all_data = []
    text_hash_lookup = {}
    
    for idx, result in enumerate(results):
        text = result.get('normalized_text', '')[:2000]
        text_hash = hashlib.md5(text.encode()).hexdigest() if text else None
        
        data_entry = {
            'idx': idx,
            'filename': result['filename'],
            'text': text,
            'text_hash': text_hash,
            'semantic_sig': result.get('semantic_sig', {}),
            'structural_sig': result.get('structural_sig', {})
        }
        all_data.append(data_entry)
        
        if text_hash:
            text_hash_lookup[text_hash] = text_hash_lookup.get(text_hash, 0) + 1
    
    # Create ALL comparison tasks
    comparison_tasks = []
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            comparison_tasks.append((i, j))
    
    log(f"   📋 Created {len(comparison_tasks):,} comparison tasks")
    
    # Optimal batch size
    optimal_batch_size = max(1000, total_comparisons // (max_workers * 5))
    optimal_batch_size = min(optimal_batch_size, 10000)
    
    batches = []
    for i in range(0, len(comparison_tasks), optimal_batch_size):
        batch = comparison_tasks[i:i + optimal_batch_size]
        batches.append(batch)
    
    log(f"   📦 Split into {len(batches)} batches of ~{optimal_batch_size} comparisons each")
    
    # Progress tracking
    comparisons_done = 0
    last_progress = 0
    start_time = time.time()
    found_duplicates = []
    
    # Prepare data for multiprocessing
    worker_data = {
        'all_data': all_data,
        'thresholds': {
            'semantic': config.get_threshold('semantic'),
            'structural': config.get_threshold('structural'),
            'similarity': config.get_threshold('similarity')
        }
    }
    
    # Prepare batch arguments
    batch_args = [(batch, worker_data) for batch in batches]
    
    # Process with ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker_process) as executor:
        # Submit all batches
        futures = []
        for args in batch_args:
            if should_stop():
                log("⛔ AI Hunter interrupted by user.")
                executor.shutdown(wait=True)
                return comparisons_done
            
            future = executor.submit(process_comparison_batch_fast, args)
            futures.append(future)
        
        # Process results as they complete
        for completed_future in concurrent.futures.as_completed(futures):
            if should_stop():
                log("⛔ AI Hunter interrupted by user.")
                executor.shutdown(wait=True)
                return comparisons_done
            
            # Get results
            batch_results = completed_future.result()
            
            # Batch all updates
            updates = []
            for result in batch_results:
                if result['is_duplicate']:
                    file1 = all_data[result['i']]['filename']
                    file2 = all_data[result['j']]['filename']
                    updates.append((file1, file2, result))
            
            # Apply all updates in one lock
            if updates:
                with merge_lock:
                    for file1, file2, result in updates:
                        merge_duplicate_groups(duplicate_groups, file1, file2)
                        duplicate_confidence[(file1, file2)] = result['confidence']
                        
                        # Log findings
                        if result['is_retranslation']:
                            msg = (f"🎯 AI Hunter: Found potential retranslation\n"
                                  f"      Files: {file1} ≈ {file2}\n"
                                  f"      Text similarity: {int(result['text_sim']*100)}% (low)\n"
                                  f"      Semantic similarity: {int(result['sem_sim']*100)}% (high)\n"
                                  f"      Structural similarity: {int(result['struct_sim']*100)}% (high)")
                            found_duplicates.append(msg)
                            
                            if len(found_duplicates) <= 3:
                                log(f"\n   [DEBUG] AI Hunter Retranslation Detection:")
                                log(f"   [DEBUG] File 1: {file1}")
                                log(f"   [DEBUG] File 2: {file2}")
                                log(f"   [DEBUG] Text Similarity: {result['text_sim']:.4f}")
                                log(f"   [DEBUG] Semantic Similarity: {result['sem_sim']:.4f}")
                                log(f"   [DEBUG] Structural Similarity: {result['struct_sim']:.4f}")
                                log(f"   [DEBUG] Confidence: {result['confidence']:.4f}")
                        else:
                            msg = (f"   📄 Found duplicate: {file1} ≈ {file2} "
                                  f"(confidence: {int(result['confidence']*100)}%)")
                            found_duplicates.append(msg)
            
            # Update progress
            comparisons_done += optimal_batch_size
            if comparisons_done > total_comparisons:
                comparisons_done = total_comparisons
            
            progress = int((comparisons_done / total_comparisons) * 100)
            
            if progress >= last_progress + 10 or progress == 100:
                elapsed = time.time() - start_time
                rate = comparisons_done / elapsed if elapsed > 0 else 0
                remaining = (total_comparisons - comparisons_done) / rate if rate > 0 else 0
                
                log(f"   📊 AI Hunter progress: {comparisons_done:,}/{total_comparisons:,} "
                    f"({progress}%) - ~{int(remaining)}s remaining - "
                    f"Speed: {int(rate):,} comparisons/sec")
                
                for msg in found_duplicates[:5]:
                    log(msg)
                found_duplicates = found_duplicates[5:]
                
                last_progress = progress
    
    # Final summary
    elapsed = time.time() - start_time
    log(f"✅ AI Hunter complete! Processed {total_comparisons:,} comparisons in {int(elapsed)}s")
    log(f"   ⚡ Speed: {int(total_comparisons/elapsed):,} comparisons/sec")
    
    log(f"\n   [DEBUG] === AI HUNTER FINAL STATISTICS ===")
    log(f"   [DEBUG] Total comparisons: {total_comparisons:,}")
    log(f"   [DEBUG] Time taken: {elapsed:.2f} seconds")
    log(f"   [DEBUG] Comparisons per second: {int(total_comparisons/elapsed):,}")
    log(f"   [DEBUG] Duplicate groups found: {len(set(duplicate_groups.values()))}")
    log(f"   [DEBUG] Total duplicate pairs: {len(duplicate_confidence)}")
    log(f"   [DEBUG] Parallel workers used: {max_workers}")
    log(f"   [DEBUG] ProcessPoolExecutor: ENABLED")
    log(f"   [DEBUG] =====================================\n")
    
    for msg in found_duplicates[-10:]:
        log(msg)
    
    return comparisons_done
