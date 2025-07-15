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

def extract_text_from_html(file_path):
    """Extract text from HTML or TXT file
    
    Returns:
        str OR tuple: 
            - For backwards compatibility: just the text (if not checking HTML structure)
            - For new functionality: (text_content, has_html_tag) tuple
    """
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

def check_html_structure(file_path):
    """Check if an HTML file has proper <html> tag
    
    Returns:
        bool: True if file has <html> tag or is not an HTML file
    """
    if not file_path.lower().endswith('.html'):
        return True  # Not an HTML file, so no check needed
        
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    
    soup = BeautifulSoup(content, "html.parser")
    return soup.find('html') is not None

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

def extract_content_fingerprint(text):
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

def extract_semantic_fingerprint(text):
    """Extract key narrative elements for semantic comparison"""
    # Extract potential character names (capitalized words appearing multiple times)
    words = re.findall(r'\b[A-Z][a-z]+\b', text)
    name_candidates = Counter(words)
    likely_names = [name for name, count in name_candidates.items() 
                   if count >= 3 and name not in COMMON_WORDS]
    
    # Extract quoted dialogue count
    dialogue_count = len(re.findall(r'[""]([^""]+)[""]', text))
    
    # Extract action verbs (past tense)
    action_verbs = len(re.findall(r'\b\w+ed\b', text))
    
    # Extract numbers and quantities
    numbers = re.findall(r'\b\d+\b', text)
    
    # Create semantic signature
    semantic_sig = {
        'characters': sorted(likely_names)[:10],  # Top 10 character names
        'dialogue_density': dialogue_count / max(1, len(text.split('\n'))),
        'action_density': action_verbs / max(1, len(text.split())),
        'numbers': sorted(set(numbers))[:20],
        'text_length': len(text)
    }
    
    # Convert to string for hashing
    semantic_str = f"chars:{','.join(semantic_sig['characters'])}" \
                  f"_dial:{semantic_sig['dialogue_density']:.2f}" \
                  f"_act:{semantic_sig['action_density']:.2f}" \
                  f"_nums:{','.join(semantic_sig['numbers'])}"
    
    return semantic_str, semantic_sig

def extract_structural_signature(text):
    """Create a signature based on paragraph lengths and dialogue patterns"""
    lines = text.split('\n')
    structure = []
    paragraph_lengths = []
    current_para_length = 0
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_para_length > 0:
                paragraph_lengths.append(current_para_length)
                current_para_length = 0
            continue
            
        current_para_length += len(stripped)
        
        # Classify line type
        if any(quote in stripped for quote in ['"', '"', '「', '『', "'", '"']):
            structure.append('D')  # Dialogue
        elif len(stripped) < 50:
            structure.append('S')  # Short
        else:
            structure.append('N')  # Narrative
    
    if current_para_length > 0:
        paragraph_lengths.append(current_para_length)
    
    # Create structural pattern
    structural_pattern = ''.join(structure)
    
    # Compress pattern by counting consecutive similar elements
    compressed = []
    if structural_pattern:
        current = structural_pattern[0]
        count = 1
        for char in structural_pattern[1:]:
            if char == current:
                count += 1
            else:
                compressed.append(f"{current}{count}")
                current = char
                count = 1
        compressed.append(f"{current}{count}")
    
    return {
        'pattern': ''.join(compressed),
        'paragraph_count': len(paragraph_lengths),
        'avg_paragraph_length': sum(paragraph_lengths) / max(1, len(paragraph_lengths)),
        'dialogue_ratio': structure.count('D') / max(1, len(structure))
    }

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

def normalize_text(text):
    """Normalize text for comparison"""
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

def generate_content_hashes(text):
    """Generate multiple hashes for better duplicate detection"""
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
    
    # 6. Semantic fingerprint hash
    semantic_str, _ = extract_semantic_fingerprint(text)
    semantic_hash = hashlib.md5(semantic_str.encode('utf-8')).hexdigest()
    
    # 7. Structural signature hash
    structural_sig = extract_structural_signature(text)
    structural_str = json.dumps(structural_sig, sort_keys=True)
    structural_hash = hashlib.md5(structural_str.encode('utf-8')).hexdigest()
    
    return {
        'raw': raw_hash,
        'normalized': normalized_hash,
        'fingerprint': fingerprint_hash,
        'word_freq': word_hash,
        'first_chunk': first_chunk_hash,
        'semantic': semantic_hash,
        'structural': structural_hash
    }

def calculate_similarity_ratio(text1, text2):
    """Calculate similarity with optimizations for large texts"""
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

def calculate_semantic_similarity(sig1, sig2):
    """Calculate similarity between two semantic signatures"""
    # Character overlap
    chars1 = set(sig1['characters'])
    chars2 = set(sig2['characters'])
    char_overlap = len(chars1 & chars2) / max(1, len(chars1 | chars2))
    
    # Dialogue density similarity
    dial_sim = 1 - abs(sig1['dialogue_density'] - sig2['dialogue_density'])
    
    # Action density similarity
    act_sim = 1 - abs(sig1['action_density'] - sig2['action_density'])
    
    # Number overlap
    nums1 = set(sig1['numbers'])
    nums2 = set(sig2['numbers'])
    num_overlap = len(nums1 & nums2) / max(1, len(nums1 | nums2)) if nums1 or nums2 else 1
    
    # Length similarity
    len_ratio = min(sig1['text_length'], sig2['text_length']) / max(1, max(sig1['text_length'], sig2['text_length']))
    
    # Weighted average
    return (char_overlap * 0.4 + dial_sim * 0.2 + act_sim * 0.2 + num_overlap * 0.1 + len_ratio * 0.1)
    
def calculate_semantic_fingerprint_similarity(text1, text2):
    """Calculate similarity based on semantic structure rather than exact wording"""
    
    # Step 1: Extract structural elements that persist across translations
    def extract_semantic_fingerprint(text):
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Extract all quoted dialogue (preserving order)
        dialogue_pattern = r'["\"\'""''『』「」]([^"\"\'""''『』「」]+)["\"\'""''『』「」]'
        dialogues = re.findall(dialogue_pattern, text)
        
        # Extract character names that appear multiple times
        potential_names = re.findall(r'\b[A-Z][a-z]+\b', text)
        name_freq = {}
        for name in potential_names:
            if name not in ['The', 'A', 'An', 'In', 'On', 'At', 'To', 'From', 'With', 'By', 'For', 'Of', 'As', 'But', 'And', 'Or']:
                name_freq[name] = name_freq.get(name, 0) + 1
        
        # Characters mentioned 3+ times are likely actual characters
        character_names = [name for name, count in name_freq.items() if count >= 3]
        
        # Extract numbers (these rarely change in translation)
        numbers = re.findall(r'\b\d+\b', text)
        
        # Extract the sequence of speaker+action patterns
        # This captures "X said", "Y asked", etc.
        speaker_actions = re.findall(r'([A-Z][a-z]+)\s+(\w+ed|spoke|says?|asks?|replies?|shouts?|screams?|whispers?)', text)
        
        # Extract paragraph structure (length of each paragraph)
        paragraphs = text.split('\n')
        para_lengths = [len(p.strip()) for p in paragraphs if len(p.strip()) > 20]
        
        # Create a structural signature
        signature = {
            'dialogue_count': len(dialogues),
            'dialogue_lengths': [len(d) for d in dialogues[:50]],  # First 50 dialogue lengths
            'characters': sorted(character_names),
            'character_frequencies': sorted([name_freq[name] for name in character_names]),
            'numbers': sorted(numbers),
            'speaker_sequence': [f"{speaker}_{action}" for speaker, action in speaker_actions[:30]],
            'paragraph_structure': para_lengths[:50],  # First 50 paragraph lengths
            'unique_words': len(set(text.lower().split())),
            'total_words': len(text.split())
        }
        
        return signature
    
    sig1 = extract_semantic_fingerprint(text1)
    sig2 = extract_semantic_fingerprint(text2)
    
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

def calculate_structural_similarity(struct1, struct2):
    """Calculate similarity between two structural signatures"""
    # Pattern similarity
    pattern_sim = SequenceMatcher(None, struct1['pattern'], struct2['pattern']).ratio()
    
    # Paragraph count similarity
    para_ratio = min(struct1['paragraph_count'], struct2['paragraph_count']) / \
                 max(1, max(struct1['paragraph_count'], struct2['paragraph_count']))
    
    # Average paragraph length similarity
    len_ratio = min(struct1['avg_paragraph_length'], struct2['avg_paragraph_length']) / \
                max(1, max(struct1['avg_paragraph_length'], struct2['avg_paragraph_length']))
    
    # Dialogue ratio similarity
    dial_sim = 1 - abs(struct1['dialogue_ratio'] - struct2['dialogue_ratio'])
    
    # Weighted average
    return (pattern_sim * 0.5 + para_ratio * 0.2 + len_ratio * 0.15 + dial_sim * 0.15)

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
    """Intelligently merge duplicate groups when new connections are found"""
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
    """Additional duplicate detection specifically for different naming formats"""
    
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
                    
                    # Check content similarity
                    text1 = result1.get('raw_text', '')[:5000]
                    text2 = result2.get('raw_text', '')[:5000]
                    
                    similarity = calculate_similarity_ratio(text1, text2)
                    
                    # Log what we're comparing
                    log(f"      Comparing: {result1['filename']} vs {result2['filename']}")
                    log(f"      Preview 1: {text1[:100]}...")
                    log(f"      Preview 2: {text2[:100]}...")
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
    
    for i, result in enumerate(results):
        if i % 20 == 0 and i > 0:
            log(f"   📊 Grouping previews: {i}/{total_files} files processed...")
            
        preview = result.get('raw_text', '')[:1000].strip()
        if not preview:
            continue
            
        # Normalize the preview for comparison
        normalized_preview = ' '.join(preview.split()[:50])  # First 50 words
        
        # Check against existing groups
        found_group = False
        for group_preview, group_indices in preview_groups.items():
            similarity = calculate_similarity_ratio(normalized_preview[:500], group_preview[:500])
            if similarity >= 0.9:  # High threshold for preview matching
                group_indices.append((i, result))
                found_group = True
                break
        
        if not found_group:
            preview_groups[normalized_preview] = [(i, result)]
    
    # Check groups with multiple files
    for preview, group in preview_groups.items():
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
                    
                    # Do a more thorough check
                    text1 = result1.get('raw_text', '')[:5000]
                    text2 = result2.get('raw_text', '')[:5000]
                    similarity = calculate_similarity_ratio(text1, text2)
                    
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
    
    return duplicates_found


def detect_duplicates(results, log, should_stop, config):
    """Detect duplicates using multiple strategies with enhanced methods - PERFORMANCE OPTIMIZED"""
    duplicate_groups = {}
    near_duplicate_groups = {}
    duplicate_confidence = defaultdict(float)
    
    total_files = len(results)
    dup_start_time = time.time()  # Track timing for progress estimates
    
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
        comparisons_done = 0
        last_progress = 0
        ai_start_time = time.time()  # Use local timer for AI Hunter
        
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
                    last_progress = progress
                # Skip if already in same group
                if (results[i]['filename'] in duplicate_groups and 
                    results[j]['filename'] in duplicate_groups and
                    duplicate_groups[results[i]['filename']] == duplicate_groups[results[j]['filename']]):
                    continue
                
                # Get both semantic and structural signatures
                sem_sim = calculate_semantic_similarity(results[i]['semantic_sig'], 
                                                       results[j]['semantic_sig'])
                struct_sim = calculate_structural_similarity(results[i]['structural_sig'],
                                                           results[j]['structural_sig'])
                
                # For AI Hunter, use a combination approach
                if config.mode == 'ai-hunter':
                    # High semantic + high structural = likely same content
                    if sem_sim >= semantic_threshold and struct_sim >= config.get_threshold('structural'):
                        # Do a quick text check to see if they're actually different
                        text_sim = calculate_similarity_ratio(
                            results[i].get('raw_text', '')[:2000],
                            results[j].get('raw_text', '')[:2000]
                        )
                        
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
    
    # 5. Deep similarity check (content-based) - OPTIMIZED
    similarity_threshold = config.get_threshold('similarity')
    log(f"🔍 Deep content similarity analysis (threshold: {int(similarity_threshold*100)}%)...")

    # Use MinHash candidates for deep checking if available
    if lsh and config.mode != 'ai-hunter':
        for result in results:
            if should_stop():
                log("⛔ Similarity check interrupted by user.")
                break
            
            if result['filename'] in minhashes:
                candidates = lsh.query(minhashes[result['filename']])
                for candidate_filename in candidates:
                    if candidate_filename == result['filename']:
                        continue
                    
                    # Skip if already in same group
                    if (result['filename'] in duplicate_groups and 
                        candidate_filename in duplicate_groups and
                        duplicate_groups[result['filename']] == duplicate_groups[candidate_filename]):
                        continue
                    
                    # Find the candidate result
                    candidate_result = next((r for r in results if r['filename'] == candidate_filename), None)
                    if not candidate_result:
                        continue
                    
                    # Check similarity
                    text1_preview = result.get('raw_text', '')[:2000]
                    text2_preview = candidate_result.get('raw_text', '')[:2000]
                    
                    similarity = calculate_similarity_ratio(text1_preview, text2_preview)
                    
                    if similarity >= similarity_threshold:
                        merge_duplicate_groups(duplicate_groups, result['filename'], candidate_filename)
                        pair = tuple(sorted([result['filename'], candidate_filename]))
                        duplicate_confidence[pair] = max(duplicate_confidence.get(pair, 0), similarity)
                        log(f"   └─ Content match: {result['filename']} ≈ {candidate_filename} ({int(similarity*100)}%)")
    else:
        # Fallback: check all pairs (slower but thorough)
        total_comparisons = (len(results) * (len(results) - 1)) // 2
        comparisons_done = 0
        last_progress = 0
        
        log(f"   📊 Checking {total_comparisons} file pairs for content similarity...")
        
        for i in range(len(results)):
            if should_stop():
                log("⛔ Similarity check interrupted by user.")
                break
            
            for j in range(i + 1, len(results)):
                comparisons_done += 1
                
                # Show progress every 10% or every 100 comparisons
                if comparisons_done % 100 == 0 or (total_comparisons < 1000 and comparisons_done % 10 == 0):
                    progress = int((comparisons_done / total_comparisons) * 100)
                    if progress >= last_progress + 10:
                        log(f"   📊 Content similarity progress: {comparisons_done}/{total_comparisons} ({progress}%)")
                        last_progress = progress
                # Check if already in same group
                if (results[i]['filename'] in duplicate_groups and 
                    results[j]['filename'] in duplicate_groups and
                    duplicate_groups[results[i]['filename']] == duplicate_groups[results[j]['filename']]):
                    continue
                
                # Always check first 2000 chars for similarity
                text1_preview = results[i].get('raw_text', '')[:2000]
                text2_preview = results[j].get('raw_text', '')[:2000]
                
                # Quick preview check
                if text1_preview == text2_preview and len(text1_preview) > 100:
                    # Exact match - definitely duplicates
                    merge_duplicate_groups(duplicate_groups, results[i]['filename'], results[j]['filename'])
                    pair = tuple(sorted([results[i]['filename'], results[j]['filename']]))
                    duplicate_confidence[pair] = 1.0
                    log(f"   └─ Exact match: {results[i]['filename']} ≡ {results[j]['filename']} (100%)")
                    continue
                
                # Calculate similarity
                similarity = calculate_similarity_ratio(text1_preview, text2_preview)
                
                if similarity >= similarity_threshold:
                    merge_duplicate_groups(duplicate_groups, results[i]['filename'], results[j]['filename'])
                    pair = tuple(sorted([results[i]['filename'], results[j]['filename']]))
                    duplicate_confidence[pair] = max(duplicate_confidence.get(pair, 0), similarity)
                    log(f"   └─ Content match: {results[i]['filename']} ≈ {results[j]['filename']} ({int(similarity*100)}%)")
    
    # 6. Consecutive chapter check with fuzzy matching
    check_consecutive_chapters(results, duplicate_groups, duplicate_confidence, config, log, should_stop)
    
    # 7. Split chapter detection
    split_candidates = detect_split_chapters(results)
    if split_candidates:
        log(f"🔍 Found {len(split_candidates)} potential split chapters")
        check_split_chapters(split_candidates, results, duplicate_groups, duplicate_confidence, log, should_stop)
    
    # 8. Specific pattern detection
    check_specific_patterns(results, duplicate_groups, duplicate_confidence, log, should_stop)
    
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
    """Perform deep similarity analysis between files"""
    log(f"🔍 Deep content similarity analysis (threshold: {int(threshold*100)}%)...")
    
    checked_pairs = set()
    
    for i in range(len(results)):
        if should_stop():
            log("⛔ Similarity check interrupted by user.")
            break
        
        if i % 10 == 0 and i > 0:
            log(f"   Progress: {i}/{len(results)} files analyzed...")
        
        for j in range(i + 1, len(results)):
            pair = tuple(sorted([results[i]['filename'], results[j]['filename']]))
            if pair in checked_pairs:
                continue
            checked_pairs.add(pair)
            
            # Skip if already in same group
            if (results[i]['filename'] in duplicate_groups and 
                results[j]['filename'] in duplicate_groups and
                duplicate_groups[results[i]['filename']] == duplicate_groups[results[j]['filename']]):
                continue
            
            # Get text samples
            text1 = results[i].get('raw_text', '')
            text2 = results[j].get('raw_text', '')
            
            if len(text1) < 500 or len(text2) < 500:
                continue
            
            # Calculate standard similarity
            similarity = calculate_similarity_ratio(text1[:5000], text2[:5000])
            
            if similarity >= threshold:
                merge_duplicate_groups(duplicate_groups, results[i]['filename'], results[j]['filename'])
                duplicate_confidence[pair] = max(duplicate_confidence[pair], similarity)
                log(f"   └─ Content similarity: {results[i]['filename']} ≈ {results[j]['filename']} ({int(similarity*100)}%)")
            
            # Check for translation variants if similarity is moderate
            elif 0.5 <= similarity < threshold:
                log(f"   Checking potential translation variant: {results[i]['filename']} vs {results[j]['filename']} (base: {int(similarity*100)}%)")
                
                # Check semantic fingerprint
                semantic_sim = calculate_semantic_fingerprint_similarity(text1[:10000], text2[:10000])
                
                if semantic_sim >= 0.75:  # High semantic similarity threshold
                    combined_score = (similarity * 0.4 + semantic_sim * 0.6)
                    
                    if combined_score >= threshold:
                        log(f"   └─ Translation variant detected (semantic: {int(semantic_sim*100)}%, combined: {int(combined_score*100)}%)")
                        merge_duplicate_groups(duplicate_groups, results[i]['filename'], results[j]['filename'])
                        duplicate_confidence[pair] = combined_score
                    else:
                        log(f"   └─ Not similar enough (semantic: {int(semantic_sim*100)}%, combined: {int(combined_score*100)}%)")

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
                    # Count words for CJK languages differently
                    if any('\u4e00' <= char <= '\u9fff' or  # Chinese
                          '\u3040' <= char <= '\u309f' or  # Hiragana
                          '\u30a0' <= char <= '\u30ff' or  # Katakana
                          '\uac00' <= char <= '\ud7af'     # Korean
                          for char in text):
                        # For CJK, count characters as words
                        word_count = len(re.findall(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]', text))
                    else:
                        # For other languages, count space-separated words
                        word_count = len(text.split())
                    
                    if chapter_num is not None:
                        word_counts[chapter_num] = {
                            'word_count': word_count,
                            'filename': basename,
                            'full_path': file_path
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
    
    if chapter_num and chapter_num in original_counts:
        original_wc = original_counts[chapter_num]['word_count']
        
        # Count words in translated text
        translated_wc = len(translated_text.split())
        
        # Calculate ratio (accounting for language differences)
        # Korean/Japanese/Chinese typically expand 1.2-2.5x when translated to English
        ratio = translated_wc / max(1, original_wc)
        
        # Define reasonable ratio ranges
        min_ratio = 0.8  # Some compression is possible
        max_ratio = 3.0  # Maximum reasonable expansion
        
        is_reasonable = min_ratio <= ratio <= max_ratio
        
        return {
            'found_match': True,
            'chapter_num': chapter_num,
            'original_wc': original_wc,
            'translated_wc': translated_wc,
            'ratio': ratio,
            'is_reasonable': is_reasonable,
            'original_file': original_counts[chapter_num]['filename']
        }
    
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
            'min_file_length': 100,
            'report_format': 'detailed',
            'auto_save_report': True
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
    log(f"   ✓ Missing HTML tag check: {'ENABLED' if qa_settings.get('check_missing_html_tag', False) else 'DISABLED'}")  # ADD THIS LINE
    log(f"   ✓ Word count ratio check: {'ENABLED' if qa_settings.get('check_word_count_ratio', False) else 'DISABLED'}")  # OPTIONAL
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
        min_length = qa_settings.get('min_file_length', 100)
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
            if not check_html_structure(full_path):
                issues.append("missing_html_tag")
                if idx < 5:  # Log only for first few files to avoid spam
                    log(f"   → Found missing <html> tag in {filename}")
        
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
                preview,  # Use the preview text
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
        issues = []
        
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
