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
import tkinter as tk
from tkinter import filedialog, messagebox
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
    'encoding_issues': re.compile(r'[�□◇]{2,}'),  # Replacement characters
    'repeated_watermarks': re.compile(r'(\[[\w\s]+\.(?:com|net|org)\])\s*\1{2,}', re.IGNORECASE),
    'chapter_continuation': re.compile(r'(to be continued|continued from|continuation of|cont\.)', re.IGNORECASE),
    'split_indicators': re.compile(r'(part \d+|section \d+|\(\d+/\d+\))', re.IGNORECASE),
    'api_response_unavailable': re.compile(r'\[AI RESPONSE UNAVAILABLE\]|\[TRANSLATION FAILED - ORIGINAL TEXT PRESERVED\]|\[IMAGE TRANSLATION FAILED\]', re.IGNORECASE)
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
    
    # Dialogue density
    dialogue_count = len(re.findall(r'["\"\'""''『』「」]([^"\"\'""''『』「」]+)["\"\'""''『』「」]', cache_text))
    dialogue_density = dialogue_count / max(1, len(words)) if words else 0
    
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
    if not file_path.lower().endswith('.html'):
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

def has_no_spacing_or_linebreaks(text, space_threshold=0.01):
    filtered_text = filter_dash_lines(text)
    space_ratio = filtered_text.count(" ") / max(1, len(filtered_text))
    newline_count = filtered_text.count("\n")
    return space_ratio < space_threshold or newline_count == 0

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
    """Detect ONLY non-Latin script characters (not romanized text), excluding Korean separators"""
    if qa_settings is None:
        qa_settings = {'foreign_char_threshold': 10, 'excluded_characters': ''}
    
    # Get threshold and excluded characters
    threshold = qa_settings.get('foreign_char_threshold', 10)
    excluded_chars = set()
    if qa_settings.get('excluded_characters'):
        excluded_chars = set(qa_settings['excluded_characters'].split())
    
    # Combine with existing separator chars
    all_excluded_chars = KOREAN_SEPARATOR_CHARS.copy()
    all_excluded_chars.update(excluded_chars)
    
    issues = []
    filtered_text = filter_dash_lines(text)
    
    # Define non-Latin script ranges
    non_latin_ranges = [
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
    
    script_chars = {}
    total_non_latin = 0
    
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
            for start, end, script_name in non_latin_ranges:
                if start <= code_point <= end:
                    total_non_latin += 1
                    if script_name not in script_chars:
                        script_chars[script_name] = {'count': 0, 'examples': []}
                    script_chars[script_name]['count'] += 1
                    if len(script_chars[script_name]['examples']) < 10:
                        script_chars[script_name]['examples'].append(char)
                    break
    
    # Check against threshold
    if total_non_latin > threshold:
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
    
    # Dialogue density
    dialogue_count = len(re.findall(r'["\"\'""''『』「」]([^"\"\'""''『』「」]+)["\"\'""''『』「」]', cache_text))
    dialogue_density = dialogue_count / max(1, len(words)) if words else 0
    
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
    """Detect chapters that might have been split into multiple files"""
    split_candidates = []
    
    for i, result in enumerate(results):
        # Check for continuation indicators
        text = result.get('raw_text', '')
        artifacts = detect_translation_artifacts(text)
        
        has_continuation = any(a['type'] in ['chapter_continuation', 'split_indicators'] 
                             for a in artifacts)
        
        # Check if file is unusually short
        is_short = len(text) < 2000
        
        # Check if starts mid-sentence (no capital letter at beginning)
        starts_mid = text.strip() and not text.strip()[0].isupper()
        
        # Check if ends mid-sentence (no punctuation at end)
        ends_mid = text.strip() and text.strip()[-1] not in '.!?"」』'
        
        if has_continuation or (is_short and (starts_mid or ends_mid)):
            split_candidates.append({
                'index': i,
                'filename': result['filename'],
                'indicators': {
                    'has_continuation': has_continuation,
                    'is_short': is_short,
                    'starts_mid': starts_mid,
                    'ends_mid': ends_mid
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
    
    sig1 = _extract_semantic_fingerprint_for_similarity(cache_text1)
    sig2 = _extract_semantic_fingerprint_for_similarity(cache_text2)
    
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

# Configure cache sizes for helper functions
extract_semantic_fingerprint = lru_cache(maxsize=get_cache_size("semantic_fingerprint"))(extract_semantic_fingerprint)
extract_structural_signature = lru_cache(maxsize=get_cache_size("structural_signature"))(extract_structural_signature)
extract_content_fingerprint = lru_cache(maxsize=get_cache_size("content_fingerprint"))(extract_content_fingerprint)

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


def enhance_duplicate_detection(results, duplicate_groups, duplicate_confidence, config, log, should_stop=None):
    """Additional duplicate detection specifically for different naming formats - CACHED VERSION"""
    
    # Create cached comparison functions for this detection run
    @lru_cache(maxsize=5000)
    def compare_texts_cached(text1_hash, text2_hash, text1_len, text2_len):
        """Cached text comparison - returns similarity ratio"""
        # Find actual texts by hash in our results
        text1, text2 = None, None
        for result in results:
            text = result.get('raw_text', '')[:5000]
            if hashlib.md5(text.encode()).hexdigest() == text1_hash:
                text1 = text
            if hashlib.md5(text.encode()).hexdigest() == text2_hash:
                text2 = text
        
        if text1 and text2:
            return calculate_similarity_ratio(text1, text2)
        return 0.0
    
    @lru_cache(maxsize=2000)
    def normalize_preview_cached(preview_hash):
        """Cached preview normalization"""
        # Find actual preview by hash
        for result in results:
            preview = result.get('raw_text', '')[:1000].strip()
            if hashlib.md5(preview.encode()).hexdigest() == preview_hash:
                return ' '.join(preview.split()[:50])  # First 50 words
        return ""
    
    # Pre-compute hashes for all texts to enable caching
    text_hashes = {}
    preview_hashes = {}
    for i, result in enumerate(results):
        text = result.get('raw_text', '')[:5000]
        preview = result.get('raw_text', '')[:1000].strip()
        text_hashes[i] = {
            'hash': hashlib.md5(text.encode()).hexdigest(),
            'length': len(text)
        }
        preview_hashes[i] = {
            'hash': hashlib.md5(preview.encode()).hexdigest(),
            'text': preview
        }
    
    # First, normalize all chapter numbers
    normalize_chapter_numbers(results)
    
    # Group by normalized chapter number
    chapter_groups = {}
    for i, result in enumerate(results):
        if result.get('normalized_chapter_num') is not None:
            num = result['normalized_chapter_num']
            if num not in chapter_groups:
                chapter_groups[num] = []
            chapter_groups[num].append((i, result))
    
    # Check each group for duplicates
    duplicates_found = []
    for chapter_num, group in chapter_groups.items():
        if should_stop and should_stop():
            log("⛔ Duplicate check interrupted by user.")
            return duplicates_found
            
        if len(group) > 1:
            log(f"   └─ Found {len(group)} files for chapter {chapter_num}")
            
            # Multiple files with same chapter number - check if they're duplicates
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    idx1, result1 = group[i]
                    idx2, result2 = group[j]
                    
                    # Use cached comparison
                    hash1 = text_hashes[idx1]['hash']
                    hash2 = text_hashes[idx2]['hash']
                    len1 = text_hashes[idx1]['length']
                    len2 = text_hashes[idx2]['length']
                    
                    # Ensure consistent ordering for cache
                    if hash1 > hash2:
                        hash1, hash2 = hash2, hash1
                        len1, len2 = len2, len1
                    
                    similarity = compare_texts_cached(hash1, hash2, len1, len2)
                    
                    # Log what we're comparing
                    log(f"      Comparing: {result1['filename']} vs {result2['filename']}")
                    log(f"      Preview 1: {result1.get('raw_text', '')[:100]}...")
                    log(f"      Preview 2: {result2.get('raw_text', '')[:100]}...")
                    log(f"      Similarity: {int(similarity*100)}%")
                    
                    if similarity >= config.get_threshold('similarity'):
                        merge_duplicate_groups(duplicate_groups, 
                                             result1['filename'], 
                                             result2['filename'])
                        pair = tuple(sorted([result1['filename'], result2['filename']]))
                        duplicate_confidence[pair] = max(duplicate_confidence.get(pair, 0), similarity)
                        
                        duplicates_found.append({
                            'file1': result1['filename'],
                            'file2': result2['filename'],
                            'chapter': chapter_num,
                            'similarity': similarity
                        })
                        
                        log(f"      ✓ DUPLICATE: {result1['filename']} ≈ {result2['filename']} ({int(similarity*100)}%)")
                    else:
                        log(f"      ✗ NOT SIMILAR ENOUGH (threshold: {int(config.get_threshold('similarity')*100)}%)")
    
    # ALSO check for misnamed files - compare all files with different chapter numbers
    log("🔍 Checking for misnamed chapters (content vs filename mismatch)...")
    
    # Group files by their content preview for faster checking
    preview_groups = {}
    total_files = len(results)
    
    # Create a cache for preview similarity checks
    @lru_cache(maxsize=10000)
    def compare_preview_similarity(preview_hash1, preview_hash2):
        """Cached preview similarity comparison"""
        preview1 = normalize_preview_cached(preview_hash1)
        preview2 = normalize_preview_cached(preview_hash2)
        return calculate_similarity_ratio(preview1[:500], preview2[:500])
    
    for i, result in enumerate(results):
        if i % 20 == 0 and i > 0:
            log(f"   📊 Grouping previews: {i}/{total_files} files processed...")
        
        preview_hash = preview_hashes[i]['hash']
        if not preview_hashes[i]['text']:
            continue
        
        # Get normalized preview using cache
        normalized_preview = normalize_preview_cached(preview_hash)
        
        # Check against existing groups
        found_group = False
        for group_key, group_indices in preview_groups.items():
            # Compare with first item in group
            if group_indices:
                first_idx = group_indices[0][0]
                first_hash = preview_hashes[first_idx]['hash']
                
                # Ensure consistent ordering for cache
                hash1, hash2 = preview_hash, first_hash
                if hash1 > hash2:
                    hash1, hash2 = hash2, hash1
                
                similarity = compare_preview_similarity(hash1, hash2)
                
                if similarity >= 0.9:  # High threshold for preview matching
                    group_indices.append((i, result))
                    found_group = True
                    break
        
        if not found_group:
            # Use preview hash as group key
            preview_groups[preview_hash] = [(i, result)]
    
    # Check groups with multiple files
    for preview_key, group in preview_groups.items():
        if should_stop and should_stop():
            log("⛔ Duplicate check interrupted by user.")
            return duplicates_found
            
        if len(group) > 1:
            log(f"   └─ Found {len(group)} files with similar content")
            
            # Check all pairs in this group
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    idx1, result1 = group[i]
                    idx2, result2 = group[j]
                    
                    # Do a more thorough check using cached comparison
                    hash1 = text_hashes[idx1]['hash']
                    hash2 = text_hashes[idx2]['hash']
                    len1 = text_hashes[idx1]['length']
                    len2 = text_hashes[idx2]['length']
                    
                    # Ensure consistent ordering for cache
                    if hash1 > hash2:
                        hash1, hash2 = hash2, hash1
                        len1, len2 = len2, len1
                    
                    similarity = compare_texts_cached(hash1, hash2, len1, len2)
                    
                    if similarity >= config.get_threshold('similarity'):
                        log(f"      ✓ Found duplicate content: {result1['filename']} ≈ {result2['filename']} ({int(similarity*100)}%)")
                        
                        merge_duplicate_groups(duplicate_groups, 
                                             result1['filename'], 
                                             result2['filename'])
                        pair = tuple(sorted([result1['filename'], result2['filename']]))
                        duplicate_confidence[pair] = max(duplicate_confidence.get(pair, 0), similarity)
                        
                        duplicates_found.append({
                            'file1': result1['filename'],
                            'file2': result2['filename'],
                            'chapter': f"misnamed_{result1.get('chapter_num', '?')}_vs_{result2.get('chapter_num', '?')}",
                            'similarity': similarity
                        })
    
    # Clear local caches when done
    compare_texts_cached.cache_clear()
    normalize_preview_cached.cache_clear()
    compare_preview_similarity.cache_clear()
    
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
    
    # Create local cached functions for this detection run
    @lru_cache(maxsize=10000)
    def compare_texts_cached(text1_hash, text2_hash, max_length=2000):
        """Cached text comparison"""
        # Find texts by hash
        text1, text2 = None, None
        for result in results:
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
    for idx, result in enumerate(results):
        text = result.get('raw_text', '')
        text_hashes[idx] = {
            'hash_2k': hashlib.md5(text[:2000].encode()).hexdigest() if len(text) >= 2000 else None,
            'hash_5k': hashlib.md5(text[:5000].encode()).hexdigest() if len(text) >= 5000 else None,
            'full_text': text
        }
    
    # Extract additional signatures for all results
    log("🔍 Extracting semantic and structural signatures...")
    for idx, result in enumerate(results):
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
    if MINHASH_AVAILABLE and len(results) > 50:  # Use MinHash for larger datasets
        log("🔍 Building MinHash index for fast similarity detection...")
        lsh, minhashes = create_minhash_index(results, config)
    
    # 1. Hash-based detection (exact and near-exact matches)
    content_hashes = defaultdict(lambda: defaultdict(list))
    
    for idx, result in enumerate(results):
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
    enhance_duplicate_detection(results, duplicate_groups, duplicate_confidence, config, log, should_stop)
    
    # 3. MinHash-based detection (if available)
    if lsh:
        log("🔍 Performing MinHash similarity detection...")
        for result in results:
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
        for result in results:
            if should_stop():
                log("⛔ Semantic check interrupted by user.")
                break
            
            checked_count += 1
            if checked_count % 10 == 0:
                log(f"   📊 MinHash semantic check: {checked_count}/{len(results)} files processed...")
                
            if result['filename'] in minhashes:
                candidates = lsh.query(minhashes[result['filename']])
                for candidate_filename in candidates:
                    if candidate_filename == result['filename']:
                        continue
                    
                    # Find the candidate result
                    candidate_result = next((r for r in results if r['filename'] == candidate_filename), None)
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
    perform_deep_similarity_check(results, duplicate_groups, duplicate_confidence, 
                                config.get_threshold('similarity'), log, should_stop)

    # 6. Consecutive chapter check with fuzzy matching
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

def perform_deep_similarity_check(results, duplicate_groups, duplicate_confidence, 
                                threshold, log, should_stop):
    """Perform deep similarity analysis between files - PARALLEL OPTIMIZED VERSION
    
    This function must be defined at module level, before detect_duplicates
    """
    global _global_text_samples
    
    log(f"🔍 Deep content similarity analysis (threshold: {int(threshold*100)}%)...")
    log("⚡ PARALLEL PROCESSING ENABLED - Using all CPU cores for maximum speed!")
    
    # Pre-cache text samples for all results to avoid repeated slicing
    text_samples = {}
    for idx, result in enumerate(results):
        text = result.get('raw_text', '')
        if len(text) >= 500:
            text_samples[idx] = {
                'full': text,
                'sample_5k': text[:5000],
                'sample_10k': text[:10000],
                'hash_5k': hashlib.md5(text[:5000].encode()).hexdigest(),
                'hash_10k': hashlib.md5(text[:10000].encode()).hexdigest()
            }
    
    # Set global mapping for the cached functions to use
    _global_text_samples = text_samples
    
    # Create local cached functions that use the global mapping
    # Using similarity_ratio size for text comparison cache
    # Using semantic_fingerprint size for semantic comparison cache
    similarity_cache_size = get_cache_size("similarity_ratio") or 20000
    semantic_cache_size = get_cache_size("semantic_fingerprint") or 2000
    
    @lru_cache(maxsize=similarity_cache_size)
    def check_similarity_cached(hash1, hash2):
        """Check similarity between two text hashes"""
        # Find the actual texts by their hashes
        text1, text2 = None, None
        for samples in _global_text_samples.values():
            if samples['hash_5k'] == hash1:
                text1 = samples['sample_5k']
            if samples['hash_5k'] == hash2:
                text2 = samples['sample_5k']
        
        if text1 and text2:
            return calculate_similarity_ratio(text1, text2)
        return 0.0
    
    @lru_cache(maxsize=semantic_cache_size)
    def check_semantic_similarity_cached(hash1, hash2):
        """Check semantic similarity between two text hashes"""
        # Find the actual texts by their hashes
        text1, text2 = None, None
        for samples in _global_text_samples.values():
            if samples['hash_10k'] == hash1:
                text1 = samples['sample_10k']
            if samples['hash_10k'] == hash2:
                text2 = samples['sample_10k']
        
        if text1 and text2:
            return calculate_semantic_fingerprint_similarity(text1, text2)
        return 0.0
    
    # Determine number of workers
    cpu_count = multiprocessing.cpu_count()
    
    # Check if there's a configured limit
    max_workers_config = 0
    try:
        # Try to read from config.json if it exists
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                full_config = json.load(f)
                # Check for qa_scanner_config first, then deep_check_config
                qa_config = full_config.get('qa_scanner_config', {})
                deep_check_config = full_config.get('deep_check_config', {})
                # Priority: deep_check_config > qa_scanner_config.max_workers > 0
                max_workers_config = deep_check_config.get('max_workers', 
                                                          qa_config.get('max_workers', 0))
    except:
        max_workers_config = 0
    
    if max_workers_config > 0:
        max_workers = min(max_workers_config, cpu_count)
        log(f"   🖥️ Using {max_workers} parallel workers (configured limit of {max_workers_config})")
    else:
        max_workers = cpu_count
        log(f"   🚀 Using ALL {max_workers} CPU cores - MAXIMUM PERFORMANCE!")
        if cpu_count > 8:
            log(f"   💡 Tip: You can limit CPU cores in QA scanner settings or config.json")
    
    # Create all comparison tasks
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
        log("   ✅ No comparisons needed - all files already grouped or too short")
        return
    
    # Progress tracking
    comparisons_done = 0
    last_progress = 0
    start_time = time.time()
    found_duplicates = []
    
    def process_comparison_batch(batch):
        """Process a batch of comparisons"""
        batch_results = []
        
        for i, j, filename_i, filename_j in batch:
            if should_stop():
                return batch_results
            
            # Use cached similarity check
            hash1 = text_samples[i]['hash_5k']
            hash2 = text_samples[j]['hash_5k']
            
            # Ensure consistent ordering for cache
            if hash1 > hash2:
                hash1, hash2 = hash2, hash1
            
            similarity = check_similarity_cached(hash1, hash2)
            
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
                # Check semantic fingerprint using cached version
                hash1_10k = text_samples[i]['hash_10k']
                hash2_10k = text_samples[j]['hash_10k']
                
                # Ensure consistent ordering for cache
                if hash1_10k > hash2_10k:
                    hash1_10k, hash2_10k = hash2_10k, hash1_10k
                
                semantic_sim = check_semantic_similarity_cached(hash1_10k, hash2_10k)
                
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
        
        return batch_results
    
    # Split tasks into batches
    batch_size = max(10, total_comparisons // (max_workers * 100))  # Dynamic batch size
    batch_size = min(batch_size, 1000)  # Cap batch size for memory efficiency
    batches = [comparison_tasks[i:i + batch_size] for i in range(0, len(comparison_tasks), batch_size)]
    
    log(f"   📦 Split into {len(batches)} batches of ~{batch_size} comparisons each")
    
    # Process batches in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all batches
        future_to_batch = {executor.submit(process_comparison_batch, batch): batch for batch in batches}
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_batch):
            if should_stop():
                log("⛔ Deep similarity check interrupted by user.")
                executor.shutdown(wait=False)
                # Clear caches before returning
                check_similarity_cached.cache_clear()
                check_semantic_similarity_cached.cache_clear()
                return
            
            try:
                batch_results = future.result()
                
                # Thread-safe merge of results
                with merge_lock:
                    for result in batch_results:
                        file1 = result['filename1']
                        file2 = result['filename2']
                        pair = tuple(sorted([file1, file2]))
                        
                        # Use the original merge_duplicate_groups function
                        merge_duplicate_groups(duplicate_groups, file1, file2)
                        
                        # Update confidence
                        duplicate_confidence[pair] = max(
                            duplicate_confidence.get(pair, 0), 
                            result['similarity']
                        )
                        
                        # Log findings
                        if result['is_variant']:
                            msg = (f"   └─ Translation variant detected: {file1} ≈ {file2} "
                                  f"(base: {int(result.get('base_sim', 0)*100)}%, "
                                  f"semantic: {int(result['semantic_sim']*100)}%, "
                                  f"combined: {int(result['similarity']*100)}%)")
                        else:
                            msg = (f"   └─ Content similarity: {file1} ≈ {file2} "
                                  f"({int(result['similarity']*100)}%)")
                        
                        found_duplicates.append(msg)
                
                # Update progress
                comparisons_done += len(future_to_batch[future])
                progress = int((comparisons_done / total_comparisons) * 100)
                
                if progress >= last_progress + 5:  # Update every 5%
                    elapsed = time.time() - start_time
                    rate = comparisons_done / elapsed if elapsed > 0 else 0
                    remaining = (total_comparisons - comparisons_done) / rate if rate > 0 else 0
                    
                    log(f"   📊 Deep check progress: {comparisons_done:,}/{total_comparisons:,} "
                        f"({progress}%) - ~{int(remaining)}s remaining - "
                        f"Speed: {int(rate):,} comparisons/sec")
                    
                    # Log some found duplicates (limit output to avoid spam)
                    for dup_msg in found_duplicates[:5]:
                        log(dup_msg)
                    found_duplicates = found_duplicates[5:]
                    
                    last_progress = progress
                    
            except Exception as e:
                log(f"   ❌ Error in parallel processing: {e}")
                import traceback
                log(f"   Traceback: {traceback.format_exc()}")
    
    # Final summary
    elapsed = time.time() - start_time
    log(f"✅ Deep similarity check complete! Processed {total_comparisons:,} comparisons in {int(elapsed)}s")
    log(f"   ⚡ Speed: {int(total_comparisons/elapsed):,} comparisons/sec")
    
    # Log remaining duplicates (last 10)
    for dup_msg in found_duplicates[-10:]:
        log(dup_msg)
    
    # Clear local caches when done
    check_similarity_cached.cache_clear()
    check_semantic_similarity_cached.cache_clear()
    
    # Clear global mapping
    _global_text_samples.clear()

        
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
    """Check if split chapters are parts of the same content"""
    for i, candidate in enumerate(split_candidates):
        if should_stop and should_stop():
            log("⛔ Split chapter check interrupted by user.")
            return
        idx = candidate['index']
        
        # Check next few files
        for j in range(1, 4):  # Check up to 3 files ahead
            if idx + j < len(results):
                next_result = results[idx + j]
                
                # Check if they might be connected
                if candidate['indicators']['ends_mid'] and not next_result['raw_text'].strip()[0].isupper():
                    # Likely continuation
                    text1_end = results[idx]['raw_text'][-500:]
                    text2_start = next_result['raw_text'][:500]
                    
                    # Check if content flows
                    combined = text1_end + " " + text2_start
                    if len(re.findall(r'[.!?]', combined)) < 2:  # Few sentence endings
                        merge_duplicate_groups(duplicate_groups, results[idx]['filename'], next_result['filename'])
                        pair = tuple(sorted([results[idx]['filename'], next_result['filename']]))
                        duplicate_confidence[pair] = 0.9  # High confidence for split chapters
                        log(f"   └─ Split chapter detected: {results[idx]['filename']} continues in {next_result['filename']}")

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
        update_new_format_progress(prog, faulty_chapters, log)
    else:
        update_legacy_format_progress(prog, faulty_chapters, log)
    
    # Write back updated progress
    with open(prog_path, "w", encoding="utf-8") as pf:
        json.dump(prog, pf, indent=2, ensure_ascii=False)
    
    # Log affected chapters
    affected_chapters = []
    affected_chapters_for_log = []
    for faulty_row in faulty_chapters:
        # For internal use (progress file updates, etc.)
        chapter_num = faulty_row.get("file_index", 0) + 1
        if faulty_row.get("filename"):
            match = re.search(r'response_(\d+)', faulty_row["filename"])
            if match:
                chapter_num = int(match.group(1))
        affected_chapters.append(chapter_num)
        
        # For the log display (to match HTML report)
        affected_chapters_for_log.append(faulty_row.get("file_index", 0))

    if affected_chapters_for_log:
        log(f"📝 Chapters marked for re-translation: {', '.join(str(c) for c in sorted(affected_chapters_for_log))}")

def update_new_format_progress(prog, faulty_chapters, log):
    """Update new format progress file"""
    log("[INFO] Detected new progress format")
    
    # Build reverse mapping
    output_file_to_chapter_key = {}
    for chapter_key, chapter_info in prog["chapters"].items():
        output_file = chapter_info.get("output_file")
        if output_file:
            output_file_to_chapter_key[output_file] = chapter_key
    
    updated_count = 0
    for faulty_row in faulty_chapters:
        faulty_filename = faulty_row["filename"]
        chapter_key = output_file_to_chapter_key.get(faulty_filename)
        
        if chapter_key and chapter_key in prog["chapters"]:
            chapter_info = prog["chapters"][chapter_key]
            old_status = chapter_info.get("status", "unknown")
            
            chapter_info["status"] = "qa_failed"
            chapter_info["qa_issues"] = True
            chapter_info["qa_timestamp"] = time.time()
            chapter_info["qa_issues_found"] = faulty_row.get("issues", [])
            chapter_info["duplicate_confidence"] = faulty_row.get("duplicate_confidence", 0)
            
            updated_count += 1
            
            # Use chapter_num from faulty_row if available, otherwise fall back to actual_num
            chapter_num = faulty_row.get("chapter_num")
            if chapter_num is None:
                chapter_num = chapter_info.get('actual_num', faulty_row.get("file_index", 0) + 1)
            log(f"   └─ Marked chapter {chapter_num} as qa_failed (was: {old_status})")
            
            # Remove from content_hashes
            content_hash = chapter_info.get("content_hash")
            if content_hash and content_hash in prog.get("content_hashes", {}):
                del prog["content_hashes"][content_hash]
            
            # Remove chunk data
            if "chapter_chunks" in prog and chapter_key in prog["chapter_chunks"]:
                del prog["chapter_chunks"][chapter_key]
                log(f"   └─ Removed chunk data for chapter {chapter_num}")
    
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

def extract_epub_word_counts(epub_path, log=print):
    """Extract word counts for each chapter from the original EPUB"""
    
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
        
        with zipfile.ZipFile(epub_path, 'r') as zf:
            # Get all HTML/XHTML files from inside the EPUB (no .txt files in EPUBs)
            html_files = [f for f in zf.namelist() 
                         if f.lower().endswith(('.html', '.xhtml', '.htm'))]
            
            log(f"📚 Found {len(html_files)} HTML files in EPUB.")
            
            for file_path in html_files:
                try:
                    # Extract chapter number from filename
                    basename = os.path.basename(file_path)
                    chapter_num = None
                    
                    # Try various patterns to extract chapter number
                    patterns = [
                        r'(\d{3,4})',  # 3-4 digit numbers
                        r'chapter[\s_-]*(\d+)',
                        r'ch[\s_-]*(\d+)',
                        r'c(\d+)',
                        r'第(\d+)[章话回]',
                        r'제(\d+)[장화회]'
                    ]
                    
                    for pattern in patterns:
                        match = re.search(pattern, basename, re.IGNORECASE)
                        if match:
                            chapter_num = int(match.group(1))
                            break
                    
                    # Read and parse the file
                    content = zf.read(file_path).decode('utf-8', errors='ignore')
                    soup = BeautifulSoup(content, 'html.parser')
                    
                    # Get text and count words
                    text = soup.get_text(strip=True)
                    
                    # Check if text contains CJK characters
                    has_cjk = any('\u4e00' <= char <= '\u9fff' or  # Chinese
                                  '\u3040' <= char <= '\u309f' or  # Hiragana
                                  '\u30a0' <= char <= '\u30ff' or  # Katakana
                                  '\uac00' <= char <= '\ud7af'     # Korean
                                  for char in text)
                    
                    if has_cjk:
                        # Use proper CJK word counting
                        word_count = count_cjk_words(text)
                    else:
                        # For other languages, count space-separated words
                        word_count = len(text.split())
                    
                    if chapter_num is not None:
                        word_counts[chapter_num] = {
                            'word_count': word_count,
                            'filename': basename,
                            'full_path': file_path,
                            'is_cjk': has_cjk  # Track if source was CJK
                        }
                    
                except Exception as e:
                    log(f"⚠️ Error processing {file_path}: {e}")
                    continue
        
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

def cross_reference_word_counts(original_counts, translated_file, translated_text, log=print):
    """Cross-reference word counts between original and translated files"""
    # Extract chapter number from translated filename
    basename = os.path.basename(translated_file)
    chapter_num = None
    
    # Try to extract chapter number
    patterns = [
        r'response_(\d+)',
        r'response_chapter(\d+)',
        r'chapter[\s_-]*(\d+)',
        r'(\d{3,4})',
        r'ch[\s_-]*(\d+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, basename, re.IGNORECASE)
        if match:
            chapter_num = int(match.group(1))
            break
    
    if chapter_num is None:
        # Try content-based matching as fallback
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

def scan_html_folder(folder_path, log=print, stop_flag=None, mode='quick-scan', qa_settings=None, epub_path=None):
    """
    Scan HTML folder for QA issues with configurable settings
    
    Args:
        folder_path: Path to folder containing HTML files
        log: Logging function
        stop_flag: Function that returns True to stop scanning
        mode: Detection mode ('ai-hunter', 'aggressive', 'standard', 'strict')
        qa_settings: Dictionary of QA scanner settings
    """
    global _stop_flag
    _stop_flag = False
    
    # Create a combined stop check function
    def should_stop():
        # Check both the passed stop_flag and global flag
        if stop_flag and stop_flag():
            log("⛔ Stop requested via GUI stop button")
            return True
        if _stop_flag:
            log("⛔ Stop requested via global stop_scan() function")
            return True
        return False
    
    start_time = time.time()
    
    # Debug info
    log(f"🔍 Starting scan with stop_flag={'provided' if stop_flag else 'not provided'}")
    if stop_flag:
        log(f"   Stop flag callable: {callable(stop_flag)}")
        try:
            current_state = stop_flag()
            log(f"   Stop flag current state: {current_state}")
        except:
            log("   Could not check stop flag state")
    
    # Load default settings if not provided
    if qa_settings is None:
        qa_settings = {
            'foreign_char_threshold': 10,
            'excluded_characters': '',
            'check_encoding_issues': False,
            'check_repetition': True,
            'check_translation_artifacts': True,
            'min_file_length': 0,
            'report_format': 'detailed',
            'auto_save_report': True,
            'check_missing_html_tag': True, 
            'check_paragraph_structure': True,       # NEW
            'paragraph_threshold': 0.3,              # NEW - 30% minimum            
            'check_word_count_ratio': False,     
            'check_multiple_headers': True,   
            'warn_name_mismatch': True           
        }
        # Get settings for new features (OUTSIDE the if block!)
    check_word_count = qa_settings.get('check_word_count_ratio', False)
    check_multiple_headers = qa_settings.get('check_multiple_headers', True)
    
    # Extract word counts from original EPUB if needed
    original_word_counts = {}
    if check_word_count:
        if epub_path and os.path.exists(epub_path):
            log(f"📚 Extracting word counts from original EPUB: {os.path.basename(epub_path)}")
            original_word_counts = extract_epub_word_counts(epub_path, log)
            log(f"   Found word counts for {len(original_word_counts)} chapters")
        else:
            log("⚠️ Word count cross-reference enabled but no valid EPUB provided - skipping this check")
            check_word_count = False
            
    log(f"\n📋 QA Settings Status:")
    log(f"   ✓ Encoding issues check: {'ENABLED' if qa_settings.get('check_encoding_issues', True) else 'DISABLED'}")
    log(f"   ✓ Repetition check: {'ENABLED' if qa_settings.get('check_repetition', True) else 'DISABLED'}")
    log(f"   ✓ Translation artifacts check: {'ENABLED' if qa_settings.get('check_translation_artifacts', True) else 'DISABLED'}")
    log(f"   ✓ Foreign char threshold: {qa_settings.get('foreign_char_threshold', 10)}")
    log(f"   ✓ Missing HTML tag check: {'ENABLED' if qa_settings.get('check_missing_html_tag', False) else 'DISABLED'}")
    log(f"   ✓ Paragraph structure check: {'ENABLED' if qa_settings.get('check_paragraph_structure', True) else 'DISABLED'}")    
    log(f"   ✓ Word count ratio check: {'ENABLED' if qa_settings.get('check_word_count_ratio', False) else 'DISABLED'}") 
    log(f"   ✓ Multiple headers check: {'ENABLED' if qa_settings.get('check_multiple_headers', False) else 'DISABLED'}")  
    
    # Initialize configuration
    custom_settings = None
    if mode == 'custom' and qa_settings and 'custom_mode_settings' in qa_settings:
        custom_settings = qa_settings['custom_mode_settings']
    config = DuplicateDetectionConfig(mode, custom_settings)
        
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
        log("   ⏱️ NOTE: AI Hunter mode checks EVERY file pair - this can take several minutes!")
    elif mode == 'aggressive':
        log("   ⚡ Aggressive mode: Lower thresholds for catching more potential duplicates")
    elif mode == 'quick-scan':
        log("   ⚡ Quick Scan mode: Optimized for speed with balanced accuracy")
    elif mode == 'custom':
        log("   ⚙️ Custom mode: Using user-defined thresholds and settings")
        if custom_settings:
            log(f"   Sample size: {custom_settings.get('sample_size', 3000)} characters")
            log(f"   Check all pairs: {custom_settings.get('check_all_pairs', False)}")
    
    html_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".html")])
    log(f"🔍 Found {len(html_files)} HTML files. Starting scan...")
    
    # Warn about AI Hunter mode with large datasets
    if mode == 'ai-hunter' and len(html_files) > 100:
        total_comparisons = (len(html_files) * (len(html_files) - 1)) // 2
        estimated_time = total_comparisons * 0.001  # Rough estimate: 1ms per comparison
        log(f"   ⚠️ AI Hunter mode with {len(html_files)} files = {total_comparisons:,} comparisons")
        log(f"   ⏱️ Estimated time: {int(estimated_time)} seconds ({int(estimated_time/60)} minutes)")
        log(f"   💡 Consider using 'aggressive' mode for faster scanning of large datasets")
    
    results = []
    
    # First pass: collect all data
    # Determine if we're in quick scan mode
    is_quick_scan = (mode == 'quick-scan')
    
    # Quick scan optimizations
    if is_quick_scan:
        log("   ⚡ Quick Scan optimizations enabled:")
        log("      • Reduced sample size (1000 chars)")
        log("      • Skipping AI Hunter checks")
        log("      • Simplified similarity calculations")
        log("      • Checking only consecutive chapters")
    
    results = []
    
    # First pass: collect all data
    for idx, filename in enumerate(html_files):
        if should_stop():
            log("⛔ QA scan interrupted by user.")
            return
        
        # Progress update every 10 files
        if idx % 10 == 0:
            progress = int((idx / len(html_files)) * 100)
            log(f"📄 [{idx+1}/{len(html_files)}] Scanning {filename}... ({progress}% complete)")
            
            # Debug: Check stop flag states periodically
            if idx % 50 == 0 and idx > 0:
                log(f"   [DEBUG] Global stop flag: {_stop_flag}, Stop function: {stop_flag() if stop_flag else 'N/A'}")
        else:
            # Less verbose for other files - show every file but compact
            print(f"\r📄 Scanning: {filename} [{idx+1}/{len(html_files)}]", end='', flush=True)
        
        full_path = os.path.join(folder_path, filename)
        try:
            raw_text = extract_text_from_html(full_path)
        except Exception as e:
            log(f"⚠️ Failed to read {filename}: {e}")
            continue
        
        # Check for stop after each file read
        if should_stop():
            log("⛔ QA scan interrupted during file reading.")
            return
        
        # Check minimum file length from settings
        min_length = qa_settings.get('min_file_length', 0)
        if len(raw_text.strip()) < min_length:
            log(f"⚠️ Skipped {filename}: Too short (< {min_length} chars)")
            continue
        
        chapter_num, chapter_title = extract_chapter_info(filename, raw_text)
        
        # Quick scan: Skip expensive hash calculations
        if is_quick_scan:
            hashes = {}  # Empty dict for quick scan
            preview_size = min(300, len(raw_text))  # Smaller preview
        else:
            hashes = generate_content_hashes(raw_text)
            preview_size = 500
        
        preview = raw_text[:preview_size].replace('\n', ' ')
        if len(preview) > preview_size:
            preview = preview[:preview_size-3] + '...'
        
        # Normalize preview
        preview_normalized = normalize_text(preview)[:300]
        
        # Detect translation artifacts only if enabled and not quick scan
        artifacts = []
        if not is_quick_scan and qa_settings.get('check_translation_artifacts', True):
            artifacts = detect_translation_artifacts(raw_text)
            
        # Filter out encoding_issues if check_encoding_issues is disabled
        if not qa_settings.get('check_encoding_issues', True):
            original_count = len(artifacts)
            artifacts = [a for a in artifacts if a['type'] != 'encoding_issues']
            if original_count != len(artifacts):
                log(f"      → Filtered out encoding artifacts (check disabled)")
        
        # Initialize issues list
        issues = []


        # HTML tag check:
        check_missing_html_tag = qa_settings.get('check_missing_html_tag', True)

        if check_missing_html_tag and filename.lower().endswith('.html'):
            # Use the new comprehensive check
            has_issues, html_issues = check_html_structure_issues(full_path, log)
            
            if has_issues:
                for issue in html_issues:
                    if issue == 'missing_html_structure':
                        issues.append("missing_html_tag")
                    elif issue == 'insufficient_paragraph_tags':
                        issues.append("insufficient_paragraph_tags")
                    elif issue == 'unwrapped_text_content':
                        issues.append("unwrapped_text_content")
                    elif issue == 'unclosed_html_tags':
                        issues.append("unclosed_html_tags")  # ADD THIS
                    elif issue == 'incomplete_html_structure':
                        issues.append("incomplete_html_structure")  # ADD THIS
                    elif issue == 'invalid_nesting':
                        issues.append("invalid_nesting")  # ADD THIS
                    elif issue == 'malformed_html':
                        issues.append("malformed_html")  # ADD THIS
                    else:
                        # Fallback for any new issue types
                        issues.append(issue)  # ADD THIS AS SAFETY NET
                
                # Log the issues found
                log(f"   → Found HTML structure issues in {filename}: {', '.join(html_issues)}")
        
        
        # Check for multiple headers
        if check_multiple_headers:
            has_multiple, header_count, header_info = detect_multiple_headers(raw_text)
            if has_multiple:
                issues.append(f"multiple_headers_{header_count}_found")
        
        # Check word count ratio
        word_count_check = None
        if check_word_count and original_word_counts:
            wc_result = cross_reference_word_counts(
                original_word_counts, 
                filename, 
                raw_text,  # Use the preview text
                log
            )
            
            if wc_result['found_match']:
                word_count_check = wc_result
                if not wc_result['is_reasonable']:
                    issues.append(f"word_count_mismatch_ratio_{wc_result['ratio']:.2f}")
                    log(f"   {filename}: Word count ratio {wc_result['ratio']:.2f} " +
                        f"(Original: {wc_result['original_wc']}, Translated: {wc_result['translated_wc']})")
            else:
                word_count_check = wc_result
                issues.append("word_count_no_match_found")
        
        # Create result dictionary
        result = {
            "file_index": idx,
            "filename": filename,
            "filepath": full_path,
            "issues": issues,  # Use the issues list we created
            "preview": preview,
            "preview_normalized": preview_normalized,
            "score": 0,
            "chapter_num": chapter_num,
            "hashes": hashes,
            "raw_text": raw_text,
            "translation_artifacts": artifacts
        }
        
        # Add optional fields if they exist
        if check_multiple_headers and has_multiple:
            result['header_count'] = header_count
            result['header_info'] = header_info
        
        if word_count_check:
            result['word_count_check'] = word_count_check
        
        results.append(result)
    
    # Clear the progress line
    print()  # New line after progress indicator
    
    log("\n✅ Initial scan complete.")
    
    # Time the duplicate detection phase
    dup_start_time = time.time()
    
    # Detect duplicates with enhanced methods
    duplicate_groups, near_duplicate_groups, duplicate_confidence = detect_duplicates(
        results, log, should_stop, config
    )
    
    dup_time = time.time() - dup_start_time
    log(f"✅ Duplicate detection completed in {dup_time:.1f} seconds")
    
    # Process results and check for issues
    log("\n📊 Checking for other issues...")
    
    # Group files by duplicate group
    groups = {}
    for filename, group_id in duplicate_groups.items():
        if group_id not in groups:
            groups[group_id] = []
        groups[group_id].append(filename)
    
    # Check each file for all issues
    for result in results:
        issues = result.get('issues', []) 
        
        # Check duplicates
        if result['filename'] in duplicate_groups:
            group_id = duplicate_groups[result['filename']]
            group_files = groups[group_id]
            if len(group_files) > 1:
                others = [f for f in group_files if f != result['filename']]
                
                # Get the highest confidence score for this file
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
        
        # Non-English content (excluding Korean separators) - pass settings
        has_non_english, lang_issues = detect_non_english_content(raw_text, qa_settings)
        if has_non_english:
            issues.extend(lang_issues)
        
        # Spacing/formatting issues - only if encoding check is enabled
        if qa_settings.get('check_encoding_issues', True):
            if has_no_spacing_or_linebreaks(raw_text):
                issues.append("no_spacing_or_linebreaks")
                # Debug log when issue is found
                if idx < 5:  # Only log for first 5 files to avoid spam
                    log(f"      → Found spacing/linebreak issue in {result['filename']}")
        
        # Repetitive content - only if repetition check is enabled
        if qa_settings.get('check_repetition', True):
            if has_repeating_sentences(raw_text):
                issues.append("excessive_repetition")
        
        # Translation artifacts - already handled above
        if result.get('translation_artifacts'):
            for artifact in result['translation_artifacts']:
                if artifact['type'] == 'machine_translation':
                    issues.append(f"machine_translation_markers_{artifact['count']}_found")
                elif artifact['type'] == 'encoding_issues':
                    # Only add encoding issues if the check is enabled
                    if qa_settings.get('check_encoding_issues', True):
                        issues.append(f"encoding_issues_{artifact['count']}_found")
                        # Debug log
                        if idx < 5:  # Only log for first 5 files
                            log(f"      → Found encoding artifacts in {result['filename']}: {artifact['count']} instances")
                elif artifact['type'] == 'repeated_watermarks':
                    issues.append(f"repeated_watermarks_{artifact['count']}_found")
                elif artifact['type'] == 'api_response_unavailable':
                    issues.append(f"api_response_unavailable_{artifact['count']}_found")
                    if idx < 5:  # Log for debugging
                        log(f"      → Found AI response unavailable markers in {result['filename']}: {artifact['count']} instances")
                elif artifact['type'] == 'chapter_continuation':
                    issues.append(f"chapter_continuation_{artifact['count']}_found")
                elif artifact['type'] == 'split_indicators':
                    issues.append(f"split_indicators_{artifact['count']}_found")

        
        result['issues'] = issues
        result['score'] = len(issues)
        
        if issues:
            log(f"   {result['filename']}: {', '.join(issues[:2])}" + (" ..." if len(issues) > 2 else ""))
    
    # Clean up raw_text to save memory
    for result in results:
        result.pop('raw_text', None)
        result.pop('hashes', None)
        result.pop('semantic_sig', None)
        result.pop('structural_sig', None)
        result.pop('normalized_text', None)
    
    # Generate reports with enhanced information and settings
    generate_reports(results, folder_path, duplicate_confidence, log, qa_settings)
    
    # Update progress file
    update_progress_file(folder_path, results, log)
    
    # Final timing
    total_time = time.time() - start_time
    log(f"\n⏱️ Total scan time: {total_time:.1f} seconds")
    if total_time > 60:
        log(f"   ({int(total_time // 60)} minutes {int(total_time % 60)} seconds)")


def check_html_structure_issues(file_path, log=print):
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
        
        # Check 4: Unclosed HTML tags
        import re
        
        # Track key structural tags for later validation
        content_lower = content.lower()
        html_open_exists = bool(re.search(r'<html[^>]*>', content_lower))
        html_close_exists = bool(re.search(r'</html>', content_lower))
        body_open_exists = bool(re.search(r'<body[^>]*>', content_lower))
        body_close_exists = bool(re.search(r'</body>', content_lower))
        
        # Tags that require closing tags (not self-closing)
        # Include html and body explicitly in this check
        paired_tags = [
            'html', 'body', 'head', 'title', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
            'p', 'div', 'span', 'a', 'ul', 'ol', 'li', 'table', 'tr', 'td', 'th',
            'form', 'button', 'script', 'style', 'nav', 'header', 'footer', 'main',
            'article', 'section', 'aside', 'strong', 'em', 'b', 'i', 'u', 'small',
            'blockquote', 'pre', 'code', 'kbd', 'var', 'samp', 'cite', 'q', 'mark',
            'time', 'address', 'figcaption', 'figure', 'label', 'select', 'option',
            'textarea', 'fieldset', 'legend', 'details', 'summary', 'dialog'
        ]
        
        unclosed_tags = []
        
        for tag in paired_tags:
            # Count opening tags (including those with attributes)
            open_pattern = rf'<{tag}(?:\s+[^>]*)?>'
            close_pattern = rf'</{tag}>'
            
            # Also check for self-closing tags like <tag />
            self_closing_pattern = rf'<{tag}(?:\s+[^>]*)?/>'
            
            open_count = len(re.findall(open_pattern, content_lower, re.IGNORECASE))
            close_count = len(re.findall(close_pattern, content_lower, re.IGNORECASE))
            self_closing_count = len(re.findall(self_closing_pattern, content_lower, re.IGNORECASE))
            
            # Adjust open count by removing self-closing tags
            effective_open_count = open_count - self_closing_count
            
            if effective_open_count > close_count:
                unclosed_tags.append(f"{tag} ({effective_open_count - close_count} unclosed)")
            elif close_count > effective_open_count:
                unclosed_tags.append(f"{tag} ({close_count - effective_open_count} extra closing tags)")
        
        if unclosed_tags:
            issues.append('unclosed_html_tags')
            log(f"   Found unclosed/mismatched tags: {', '.join(unclosed_tags[:5])}" + 
                (" ..." if len(unclosed_tags) > 5 else ""))
        
        # Check 5: Basic HTML structure validation - only check for consistency, not completeness
        # Note: Variables like html_open_exists are already defined in Check 4
        head_open_exists = bool(re.search(r'<head[^>]*>', content_lower))
        head_close_exists = bool(re.search(r'</head>', content_lower))
        
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
        
        if body_open_exists and not body_close_exists:
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
    if len(sys.argv) < 2:
        launch_gui()
    else:
        mode = 'standard'
        if len(sys.argv) > 2:
            if sys.argv[2] == "--aggressive":
                mode = 'aggressive'
            elif sys.argv[2] == "--custom":
                mode = 'custom'
            elif sys.argv[2] == "--quick-scan":
                mode = 'quick-scan'
            elif sys.argv[2] == "--ai-hunter":
                mode = 'ai-hunter'
        scan_html_folder(sys.argv[1], mode=mode)



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

# Add this to scan_html_folder.py - parallel AI Hunter implementation

import concurrent.futures
import multiprocessing
from threading import Lock

# Add a global lock for thread-safe operations
merge_lock = Lock()

def parallel_ai_hunter_check(results, duplicate_groups, duplicate_confidence, config, log, should_stop):
    """Parallel AI Hunter checking - no quality compromises, just pure speed"""
    
    log("🤖 AI Hunter mode: Enhanced semantic and structural checking active")
    log("⚡ PARALLEL PROCESSING ENABLED - Using all CPU cores for maximum speed!")
    
    total_comparisons = (len(results) * (len(results) - 1)) // 2
    log(f"   ⚠️ Will check ALL {total_comparisons:,} file pairs - but in parallel!")
    
    # Add debugging output
    log(f"   [DEBUG] AI Hunter parallel processing started")
    log(f"   [DEBUG] Total files to compare: {len(results)}")
    log(f"   [DEBUG] Total comparisons needed: {total_comparisons:,}")
    
    # Determine number of workers
    cpu_count = multiprocessing.cpu_count()
    
    # Check if there's a configured limit
    # Try to read from a global config file or environment variable
    max_workers_config = 0
    try:
        # Try to read from config.json if it exists
        import json
        import os
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                full_config = json.load(f)
                ai_hunter_config = full_config.get('ai_hunter_config', {})
                max_workers_config = ai_hunter_config.get('ai_hunter_max_workers', 0)
    except:
        # If config reading fails, default to 0 (use all cores)
        max_workers_config = 0
    
    if max_workers_config > 0:
        max_workers = min(max_workers_config, cpu_count)
        log(f"   🖥️ Using {max_workers} parallel workers (configured limit of {max_workers_config})")
    else:
        max_workers = cpu_count  # No limit - use all available cores!
        log(f"   🚀 Using ALL {max_workers} CPU cores - MAXIMUM PERFORMANCE!")
        if cpu_count > 8:
            log(f"   💡 Tip: You can limit CPU cores under the QA scanner settings")
    
    # Pre-compute text hashes for all results
    text_hashes = {}
    for idx, result in enumerate(results):
        text = result.get('normalized_text', '')[:2000]
        text_hashes[idx] = {
            'hash_2k': hashlib.md5(text.encode()).hexdigest() if text else None,
            'text_2k': text
        }
    
    # Create all comparison tasks
    comparison_tasks = []
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            comparison_tasks.append((i, j))
    
    log(f"   📋 Created {len(comparison_tasks):,} comparison tasks")
    
    # Progress tracking
    comparisons_done = 0
    last_progress = 0
    start_time = time.time()
    found_duplicates = []
    
    def process_comparison_batch(batch):
        """Process a batch of comparisons"""
        batch_results = []
        
        for i, j in batch:
            if should_stop():
                return batch_results
            
            # Calculate all similarities
            sem_sim = calculate_semantic_similarity(
                results[i]['semantic_sig'], 
                results[j]['semantic_sig']
            )
            
            struct_sim = calculate_structural_similarity(
                results[i]['structural_sig'],
                results[j]['structural_sig']
            )
            
            # Text similarity
            text_sim = 0.0
            if text_hashes[i]['hash_2k'] and text_hashes[j]['hash_2k']:
                if text_hashes[i]['hash_2k'] == text_hashes[j]['hash_2k']:
                    text_sim = 1.0
                else:
                    text_sim = SequenceMatcher(
                        None, 
                        text_hashes[i]['text_2k'], 
                        text_hashes[j]['text_2k']
                    ).ratio()
            
            # AI Hunter logic: High semantic + high structural = likely duplicate
            if sem_sim >= config.get_threshold('semantic') and struct_sim >= config.get_threshold('structural'):
                # If text similarity is low but semantic/structural is high, it's likely a retranslation
                is_retranslation = text_sim < 0.6
                
                batch_results.append({
                    'i': i,
                    'j': j,
                    'sem_sim': sem_sim,
                    'struct_sim': struct_sim,
                    'text_sim': text_sim,
                    'is_duplicate': True,
                    'is_retranslation': is_retranslation,
                    'confidence': (sem_sim + struct_sim) / 2
                })
            
            # Also check traditional similarity
            elif text_sim >= config.get_threshold('similarity'):
                batch_results.append({
                    'i': i,
                    'j': j,
                    'sem_sim': sem_sim,
                    'struct_sim': struct_sim,
                    'text_sim': text_sim,
                    'is_duplicate': True,
                    'is_retranslation': False,
                    'confidence': text_sim
                })
        
        return batch_results
    
    # Split tasks into batches
    batch_size = max(10, total_comparisons // (max_workers * 100))  # Dynamic batch size
    batches = [comparison_tasks[i:i + batch_size] for i in range(0, len(comparison_tasks), batch_size)]
    
    log(f"   📦 Split into {len(batches)} batches of ~{batch_size} comparisons each")
    
    # Process batches in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all batches
        future_to_batch = {executor.submit(process_comparison_batch, batch): batch for batch in batches}
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_batch):
            if should_stop():
                log("⛔ AI Hunter interrupted by user.")
                executor.shutdown(wait=False)
                return
            
            try:
                batch_results = future.result()
                
                # Thread-safe merge of results
                with merge_lock:
                    for result in batch_results:
                        if result['is_duplicate']:
                            file1 = results[result['i']]['filename']
                            file2 = results[result['j']]['filename']
                            
                            merge_duplicate_groups(duplicate_groups, file1, file2)
                            duplicate_confidence[(file1, file2)] = result['confidence']
                            
                            if result['is_retranslation']:
                                found_duplicates.append(
                                    f"🎯 AI Hunter: Found potential retranslation\n"
                                    f"      Files: {file1} ≈ {file2}\n"
                                    f"      Text similarity: {int(result['text_sim']*100)}% (low)\n"
                                    f"      Semantic similarity: {int(result['sem_sim']*100)}% (high)\n"
                                    f"      Structural similarity: {int(result['struct_sim']*100)}% (high)"
                                )
                                
                                # Debug output for retranslations
                                log(f"\n   [DEBUG] AI Hunter Retranslation Detection:")
                                log(f"   [DEBUG] File 1: {file1}")
                                log(f"   [DEBUG] File 2: {file2}")
                                log(f"   [DEBUG] Text Similarity: {result['text_sim']:.4f}")
                                log(f"   [DEBUG] Semantic Similarity: {result['sem_sim']:.4f}")
                                log(f"   [DEBUG] Structural Similarity: {result['struct_sim']:.4f}")
                                log(f"   [DEBUG] Confidence: {result['confidence']:.4f}")
                            else:
                                found_duplicates.append(
                                    f"   📄 Found duplicate: {file1} ≈ {file2} "
                                    f"(confidence: {int(result['confidence']*100)}%)"
                                )
                                
                                # Debug output for regular duplicates
                                if len(found_duplicates) <= 5:  # Only debug first few to avoid spam
                                    log(f"   [DEBUG] Regular duplicate: {file1} ≈ {file2} (confidence: {result['confidence']:.4f})")
                
                # Update progress
                comparisons_done += len(future_to_batch[future])
                progress = int((comparisons_done / len(comparison_tasks)) * 100)
                
                if progress >= last_progress + 5:  # Update every 5%
                    elapsed = time.time() - start_time
                    rate = comparisons_done / elapsed if elapsed > 0 else 0
                    remaining = (len(comparison_tasks) - comparisons_done) / rate if rate > 0 else 0
                    
                    log(f"   📊 AI Hunter progress: {comparisons_done:,}/{len(comparison_tasks):,} "
                        f"({progress}%) - ~{int(remaining)}s remaining - "
                        f"Speed: {int(rate):,} comparisons/sec")
                    
                    # Debug output
                    log(f"   [DEBUG] Completed batches: {comparisons_done // batch_size}/{len(batches)}")
                    log(f"   [DEBUG] Active threads: {len([f for f in future_to_batch if not f.done()])}")
                    log(f"   [DEBUG] Duplicates found so far: {len(duplicate_groups)}")
                    
                    # Log some found duplicates
                    if found_duplicates and len(found_duplicates) <= 5:
                        for dup_msg in found_duplicates:
                            log(dup_msg)
                        found_duplicates.clear()
                    
                    last_progress = progress
                    
            except Exception as e:
                log(f"   ❌ Error in parallel processing: {e}")
    
    # Final summary
    elapsed = time.time() - start_time
    log(f"✅ AI Hunter complete! Processed {total_comparisons:,} comparisons in {int(elapsed)}s")
    log(f"   ⚡ Speed improvement: {int(total_comparisons/elapsed):,} comparisons/sec")
    
    # Debug final statistics
    log(f"\n   [DEBUG] === AI HUNTER FINAL STATISTICS ===")
    log(f"   [DEBUG] Total comparisons: {total_comparisons:,}")
    log(f"   [DEBUG] Time taken: {elapsed:.2f} seconds")
    log(f"   [DEBUG] Comparisons per second: {int(total_comparisons/elapsed):,}")
    log(f"   [DEBUG] Duplicate groups found: {len(set(duplicate_groups.values()))}")
    log(f"   [DEBUG] Total duplicate pairs: {len(duplicate_confidence)}")
    log(f"   [DEBUG] Parallel workers used: {max_workers}")
    log(f"   [DEBUG] =====================================\n")
    
    # Log remaining duplicates
    for dup_msg in found_duplicates[-10:]:  # Show last 10
        log(dup_msg)
    
    # Return the comparisons count
    return comparisons_done
