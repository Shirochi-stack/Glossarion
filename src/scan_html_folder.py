"""
Enhanced QA Scanner for HTML Translation Files - Fully Refactored

This module provides comprehensive quality assurance scanning for translated HTML files,
with configurable settings and support for Full Scan (with AI Hunter) and Quick Scan modes.

PERFORMANCE IMPROVEMENTS:
- Added detailed progress indicators for all slow operations
- Shows estimated time remaining for long operations  
- Displays current file being scanned
- Provides progress updates every 5-10%
- Added timing information for each phase
- MinHash optimization status messages
- Debug output for stop functionality

OPTIMIZATION TIPS:
- For datasets > 100 files, use Quick Scan instead of Full Scan
- Install 'datasketch' package for 2-10x faster duplicate detection: pip install datasketch
- Use 'summary' report format for faster completion
- Disable checks you don't need in QA Scanner Settings
"""

import os
import hashlib
import json
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
    # Note: Install 'datasketch' package for faster duplicate detection on large datasets

# Global flag to allow stopping the scan externally
_stop_flag = False

def stop_scan():
    """Set the stop flag to True
    
    This function should be called by the GUI to stop a running scan.
    The GUI code needs to:
    1. Import this function: from scan_html_folder import stop_scan
    2. Call it in the stop_qa_scan method: stop_scan()
    3. The scan will check this flag periodically and stop gracefully
    """
    global _stop_flag
    _stop_flag = True
    print(f"[stop_scan] Stop flag set to: {_stop_flag}")

def reset_stop_flag():
    """Reset the stop flag to False for the next scan"""
    global _stop_flag
    _stop_flag = False
    print(f"[reset_stop_flag] Stop flag reset to: {_stop_flag}")

def is_stop_requested():
    """Check if stop has been requested"""
    global _stop_flag
    return _stop_flag

# Constants
ENCODING_ERRORS = ['�', '□', '◇', '※', '⬜', '▯']
DASH_CHARS = ['―', '—', '–', '-', 'ー', '一', '﹘', '﹣', '－']
COMMON_ENGLISH_WORDS = {
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
    'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
    'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
    'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their',
    'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go',
    'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know',
    'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them',
    'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over',
    'think', 'also', 'back', 'after', 'use', 'two', 'how', 'our', 'work',
    'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these',
    'give', 'day', 'most', 'us', 'is', 'was', 'are', 'been', 'being', 'have', 'has', 'had',
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
}

class DuplicateDetectionConfig:
    """Configuration for duplicate detection from settings"""
    def __init__(self, qa_settings):
        self.qa_settings = qa_settings or {}
        
        # Check if AI Hunter is enabled
        self.use_ai_hunter = qa_settings.get('ai_hunter_enabled', False)
        
        # Get the appropriate thresholds
        if self.use_ai_hunter:
            # Use AI Hunter thresholds from ai_hunter_config if available
            ai_config = qa_settings.get('ai_hunter_config', {})
            ai_thresholds = ai_config.get('thresholds', {})
            
            self.thresholds = {
                'similarity': ai_thresholds.get('text', 30) / 100.0,  # Convert from percentage
                'semantic': ai_thresholds.get('semantic', 85) / 100.0,
                'structural': ai_thresholds.get('structural', 85) / 100.0,
                'word_overlap': 0.50,  # Default values for these
                'minhash_threshold': 0.60,
                'exact': ai_thresholds.get('exact', 70) / 100.0,
                'text': ai_thresholds.get('text', 30) / 100.0,
                'character': ai_thresholds.get('character', 40) / 100.0,
                'pattern': ai_thresholds.get('pattern', 50) / 100.0,
                'consecutive_chapters': 5,
                'check_all_pairs': True
            }
        else:
            # Use standard duplicate thresholds from qa_settings
            dup_thresholds = qa_settings.get('duplicate_thresholds', {})
            
            self.thresholds = {
                'similarity': dup_thresholds.get('similarity', 0.85),
                'semantic': dup_thresholds.get('semantic', 0.80),
                'structural': dup_thresholds.get('structural', 0.90),
                'word_overlap': dup_thresholds.get('word_overlap', 0.75),
                'minhash_threshold': dup_thresholds.get('minhash_threshold', 0.80),
                'consecutive_chapters': 2,
                'check_all_pairs': False
            }
    
    def get_threshold(self, key):
        """Get threshold value for given key"""
        return self.thresholds.get(key, 0.85)

# Helper functions
def filter_dash_lines(text):
    """Filter out lines that are purely dashes (Korean separator patterns)"""
    lines = text.split('\n')
    filtered_lines = []
    
    for line in lines:
        stripped = line.strip()
        # Skip if line is only dashes/separators
        if stripped and not all(c in DASH_CHARS or c.isspace() for c in stripped):
            # Also skip Korean separator patterns
            is_separator = False
            for pattern in KOREAN_DASH_PATTERNS:
                if re.fullmatch(pattern, stripped):
                    is_separator = True
                    break
            
            if not is_separator:
                filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)

def extract_text_from_html(filepath):
    """Extract plain text from HTML file"""
    encodings = ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr', 'utf-16']
    
    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                html_content = f.read()
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    else:
        # If all encodings fail, try with errors='ignore'
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            html_content = f.read()
    
    # Parse with BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    
    # Get text
    text = soup.get_text()
    
    # Break into lines and remove leading/trailing space
    lines = (line.strip() for line in text.splitlines())
    # Break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # Drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)
    
    return text

def detect_foreign_characters(text, threshold=10, excluded_chars=''):
    """Detect foreign characters in text"""
    excluded_set = set(excluded_chars.split())
    total_chars = 0
    foreign_chars = 0
    char_counts = Counter()
    
    for char in text:
        if char.isspace() or char in excluded_set:
            continue
        
        total_chars += 1
        
        # Check if character is in common ranges
        if not (ord(char) < 128 or  # ASCII
                0xAC00 <= ord(char) <= 0xD7AF or  # Korean
                0x1100 <= ord(char) <= 0x11FF or  # Korean Jamo
                0x3130 <= ord(char) <= 0x318F):   # Korean compatibility
            foreign_chars += 1
            char_counts[char] += 1
    
    if total_chars == 0:
        return 0, {}
    
    percentage = (foreign_chars / total_chars) * 100
    return percentage, dict(char_counts.most_common(10))

def has_excessive_dashes(text, threshold=30):
    """Check if text has excessive dashes or similar characters"""
    if not text:
        return False
    
    dash_count = sum(1 for char in text if char in DASH_CHARS)
    total_chars = len([c for c in text if not c.isspace()])
    
    if total_chars == 0:
        return False
    
    dash_percentage = (dash_count / total_chars) * 100
    return dash_percentage > threshold

def has_repeating_sentences(text, min_length=20, min_repeats=3):
    """Detect if text has repeating sentences"""
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > min_length]
    
    if len(sentences) < min_repeats:
        return False
    
    sentence_counts = Counter(sentences)
    
    for sentence, count in sentence_counts.items():
        if count >= min_repeats:
            return True
    
    return False

def has_no_spacing_or_linebreaks(text, min_length=1000):
    """Check if text lacks proper spacing or line breaks"""
    if len(text) < min_length:
        return False
    
    # Check for very long lines
    lines = text.split('\n')
    long_lines = [line for line in lines if len(line) > 500]
    
    if len(long_lines) > len(lines) * 0.5:
        return True
    
    # Check for lack of spaces
    words = text.split()
    if len(words) < len(text) / 20:  # Average word length > 20 chars indicates issues
        return True
    
    return False

def detect_translation_artifacts(text):
    """Detect common translation artifacts and watermarks"""
    artifacts = []
    
    # Machine translation markers
    mtl_patterns = [
        r'Translated by.*?(?:AI|MT|Machine)',
        r'This translation was.*?generated',
        r'MTL\s*(?:Translation|Note)',
        r'Papago|Google\s*Translate|DeepL',
        r'기계\s*번역|자동\s*번역',
        r'Note:.*?machine translated',
        r'\[T/?N:.*?\]',
        r'Translator\'s? Note:',
    ]
    
    mtl_count = 0
    for pattern in mtl_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        mtl_count += len(matches)
    
    if mtl_count > 0:
        artifacts.append({
            'type': 'machine_translation',
            'count': mtl_count
        })
    
    # Encoding issues
    encoding_chars = ['�', '□', '◇', '※', '⬜', '▯']
    encoding_count = sum(text.count(char) for char in encoding_chars)
    
    if encoding_count > 5:
        artifacts.append({
            'type': 'encoding_issues',
            'count': encoding_count
        })
    
    # Repeated watermarks
    watermark_patterns = [
        r'(https?://[^\s]+)\s*\1{2,}',
        r'(@[^\s]+)\s*\1{2,}',
        r'(Chapter \d+.*?translated by.*?)\s*\1{2,}',
    ]
    
    for pattern in watermark_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            artifacts.append({
                'type': 'repeated_watermarks',
                'count': len(matches)
            })
            break
    
    return artifacts

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
    
    # Get threshold (now in character count, not percentage)
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
            # Skip excluded characters
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
    
    # Apply threshold - now it's a character count, not percentage
    if total_non_latin > threshold:
        for script, data in script_chars.items():
            if data['count'] > 0:  # Only include scripts that were found
                examples = ''.join(data['examples'][:5])
                count = data['count']
                issues.append(f"{script}_text_found_{count}_chars_[{examples}]")
    
    return len(issues) > 0, issues

# Advanced detection functions
def generate_content_hashes(text):
    """Generate multiple hashes for duplicate detection"""
    # Normalize text
    normalized = normalize_text(text)
    
    # Full hash
    full_hash = hashlib.md5(normalized.encode()).hexdigest()
    
    # Partial hashes (beginning, middle, end)
    text_len = len(normalized)
    chunk_size = min(1000, text_len // 3)
    
    beginning_hash = hashlib.md5(normalized[:chunk_size].encode()).hexdigest()
    
    if text_len > chunk_size * 2:
        middle_start = (text_len - chunk_size) // 2
        middle_hash = hashlib.md5(normalized[middle_start:middle_start + chunk_size].encode()).hexdigest()
        end_hash = hashlib.md5(normalized[-chunk_size:].encode()).hexdigest()
    else:
        middle_hash = full_hash
        end_hash = full_hash
    
    return {
        'full': full_hash,
        'beginning': beginning_hash,
        'middle': middle_hash,
        'end': end_hash
    }

def normalize_text(text):
    """Normalize text for comparison"""
    # Remove HTML tags if any remain
    text = re.sub(r'<[^>]+>', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Normalize unicode
    text = unicodedata.normalize('NFKD', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove punctuation for hash comparison
    text = re.sub(r'[^\w\s]', '', text)
    
    return text

def calculate_similarity(text1, text2):
    """Calculate similarity between two texts"""
    return SequenceMatcher(None, text1, text2).ratio()

def calculate_word_overlap(text1, text2):
    """Calculate word overlap ratio between texts"""
    words1 = set(normalize_text(text1).split())
    words2 = set(normalize_text(text2).split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1 & words2
    union = words1 | words2
    
    return len(intersection) / len(union)

def extract_character_names(text):
    """Extract potential character names from text"""
    # Simple heuristic: capitalized words that appear multiple times
    words = re.findall(r'\b[A-Z][a-z]+\b', text)
    
    # Count occurrences
    name_counts = Counter(words)
    
    # Filter likely names (appear at least 3 times, not common words)
    potential_names = {
        name for name, count in name_counts.items()
        if count >= 3 and name.lower() not in COMMON_ENGLISH_WORDS
    }
    
    return potential_names

def calculate_character_overlap(text1, text2):
    """Calculate character name overlap between texts"""
    names1 = extract_character_names(text1)
    names2 = extract_character_names(text2)
    
    if not names1 or not names2:
        return 0.0
    
    overlap = len(names1 & names2)
    total = len(names1 | names2)
    
    return overlap / total if total > 0 else 0.0

def semantic_similarity(text1, text2, sample_size=1000):
    """Calculate semantic similarity based on content markers"""
    # Extract samples
    sample1 = text1[:sample_size] if len(text1) > sample_size else text1
    sample2 = text2[:sample_size] if len(text2) > sample_size else text2
    
    # Extract semantic markers
    markers = {
        'numbers': r'\b\d+\b',
        'dialogue': r'"[^"]*"',
        'names': r'\b[A-Z][a-z]+\b',
        'questions': r'[^.!?]*\?',
        'exclamations': r'[^.!?]*!',
    }
    
    scores = []
    for marker_type, pattern in markers.items():
        matches1 = set(re.findall(pattern, sample1))
        matches2 = set(re.findall(pattern, sample2))
        
        if matches1 or matches2:
            overlap = len(matches1 & matches2)
            total = len(matches1 | matches2)
            score = overlap / total if total > 0 else 0.0
            scores.append(score)
    
    return sum(scores) / len(scores) if scores else 0.0

def structural_similarity(text1, text2):
    """Calculate structural similarity based on paragraph patterns"""
    # Extract paragraph lengths
    paragraphs1 = [len(p) for p in text1.split('\n\n') if p.strip()]
    paragraphs2 = [len(p) for p in text2.split('\n\n') if p.strip()]
    
    if not paragraphs1 or not paragraphs2:
        return 0.0
    
    # Compare patterns
    pattern1 = [1 if p > 100 else 0 for p in paragraphs1[:20]]  # Long vs short
    pattern2 = [1 if p > 100 else 0 for p in paragraphs2[:20]]
    
    # Calculate similarity
    min_len = min(len(pattern1), len(pattern2))
    if min_len == 0:
        return 0.0
    
    matches = sum(1 for i in range(min_len) if pattern1[i] == pattern2[i])
    return matches / min_len

def create_minhash(text, num_perm=128):
    """Create MinHash signature for text"""
    if not MINHASH_AVAILABLE:
        return None
    
    minhash = MinHash(num_perm=num_perm)
    
    # Create shingles (3-word sequences)
    words = normalize_text(text).split()
    for i in range(len(words) - 2):
        shingle = ' '.join(words[i:i+3])
        minhash.update(shingle.encode('utf8'))
    
    return minhash

def calculate_minhash_similarity(minhash1, minhash2):
    """Calculate Jaccard similarity using MinHash"""
    if not MINHASH_AVAILABLE or minhash1 is None or minhash2 is None:
        return 0.0
    
    return minhash1.jaccard(minhash2)

def calculate_semantic_similarity(sig1, sig2):
    """Calculate similarity between semantic signatures"""
    if not sig1 or not sig2:
        return 0.0
    
    # If signatures are dictionaries with scores
    if isinstance(sig1, dict) and isinstance(sig2, dict):
        # Compare each component
        scores = []
        for key in set(sig1.keys()) | set(sig2.keys()):
            val1 = sig1.get(key, 0)
            val2 = sig2.get(key, 0)
            if val1 + val2 > 0:
                similarity = 1 - abs(val1 - val2) / (val1 + val2)
                scores.append(similarity)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    # Fallback to simple ratio
    return calculate_similarity(str(sig1), str(sig2))

def calculate_structural_similarity(sig1, sig2):
    """Calculate similarity between structural signatures"""
    if not sig1 or not sig2:
        return 0.0
    
    # If signatures are dictionaries
    if isinstance(sig1, dict) and isinstance(sig2, dict):
        # Compare patterns
        pattern1 = sig1.get('pattern', '')
        pattern2 = sig2.get('pattern', '')
        
        if not pattern1 or not pattern2:
            return 0.0
        
        # Use sequence matching on patterns
        return calculate_similarity(pattern1, pattern2)
    
    # Fallback
    return calculate_similarity(str(sig1), str(sig2))

def calculate_hash(text):
    """Calculate hash for duplicate detection"""
    normalized = normalize_text(text)
    return hashlib.md5(normalized.encode()).hexdigest()

def is_likely_duplicate_ai_hunter(result1, result2, config):
    """Enhanced duplicate detection using AI Hunter methods"""
    try:
        # Import AI Hunter module if available
        from ai_hunter_enhanced import ImprovedAIHunterDetection
        
        detector = ImprovedAIHunterDetection(config.config if hasattr(config, 'config') else {})
        
        # Get AI Hunter similarity scores
        similarity_results = detector.calculate_all_similarities(
            result1['raw_text'], 
            result2['raw_text']
        )
        
        # Check if enough methods triggered
        methods_triggered = []
        for method, score in similarity_results.items():
            threshold = config.get_threshold(method)
            if score >= threshold:
                methods_triggered.append((method, score))
        
        # Require multiple methods for AI Hunter mode
        if len(methods_triggered) >= 3:
            # Calculate weighted confidence
            total_weight = 0
            weighted_sum = 0
            
            weights = {
                'exact': 1.5,
                'text': 1.2,
                'semantic': 1.0,
                'structural': 1.0,
                'character': 0.8,
                'pattern': 0.8
            }
            
            for method, score in methods_triggered:
                weight = weights.get(method, 1.0)
                weighted_sum += score * weight
                total_weight += weight
            
            confidence = weighted_sum / total_weight if total_weight > 0 else 0
            
            return True, confidence, methods_triggered
        
        return False, 0, []
        
    except ImportError:
        # Fallback to basic detection if AI Hunter not available
        return is_likely_duplicate_basic(result1, result2, config)

def is_likely_duplicate_basic(result1, result2, config):
    """Basic duplicate detection"""
    # Check exact hash match first
    if result1['hashes']['full'] == result2['hashes']['full']:
        return True, 1.0, [('exact', 1.0)]
    
    # Check partial hashes
    partial_matches = 0
    for key in ['beginning', 'middle', 'end']:
        if result1['hashes'][key] == result2['hashes'][key]:
            partial_matches += 1
    
    # If 2+ partial matches, likely duplicate
    if partial_matches >= 2:
        return True, 0.9, [('partial_hash', partial_matches/3)]
    
    # Text similarity
    text_sim = calculate_similarity(
        normalize_text(result1['raw_text'][:5000]),
        normalize_text(result2['raw_text'][:5000])
    )
    
    if text_sim >= config.get_threshold('similarity'):
        return True, text_sim, [('text', text_sim)]
    
    return False, 0, []

def structural_signature(text):
    """Create a structural signature of the text"""
    # Split into paragraphs
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    
    if not paragraphs:
        return {}
    
    # Analyze paragraph structure
    paragraph_lengths = [len(p) for p in paragraphs]
    
    # Create a pattern based on paragraph types
    structure = []
    for p in paragraphs[:50]:  # Analyze first 50 paragraphs
        if '"' in p or '"' in p or '「' in p:
            structure.append('D')  # Dialogue
        elif len(p) < 50:
            structure.append('S')  # Short
        elif len(p) < 200:
            structure.append('M')  # Medium
        else:
            structure.append('L')  # Long
    
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
        (r"(\d{3,4})[_\.]html", 1, None),  # Catches 0001.html or 0001_whatever.html
        (r"ch(?:apter)?[\s_-]*(\d+)", 1, None),  # Catches ch1, ch_1, chapter-1
        (r"ep(?:isode)?[\s_-]*(\d+)", 1, None),  # Catches ep1, episode_1
        (r"part[\s_-]*(\d+)", 1, None),  # Catches part1, part_1
    ]
    
    # Try filename patterns
    for pattern, num_group, title_group in filename_patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            try:
                chapter_num = int(match.group(num_group))
                if title_group and len(match.groups()) >= title_group:
                    chapter_title = match.group(title_group).replace('_', ' ').strip()
                break
            except:
                continue
    
    # If no chapter number found in filename, try content
    if chapter_num is None:
        # Enhanced content patterns
        content_patterns = [
            # Standard patterns
            r"Chapter\s+(\d+)(?:\s*[:\-]\s*(.+?))?(?:\n|$)",
            r"Chapter\s+([IVXLCDM]+)(?:\s*[:\-]\s*(.+?))?(?:\n|$)",  # Roman numerals
            r"第\s*(\d+)\s*[章话]",  # Chinese/Japanese chapter markers
            r"Episode\s+(\d+)(?:\s*[:\-]\s*(.+?))?(?:\n|$)",
            r"Part\s+(\d+)(?:\s*[:\-]\s*(.+?))?(?:\n|$)",
            
            # More flexible patterns
            r"^\s*(\d+)\s*[\.:\-]\s*(.+?)$",  # Lines starting with number
            r"#\s*(\d+)(?:\s*[:\-]\s*(.+?))?",  # Markdown style headers
            r"\[Chapter\s+(\d+)\]",  # Bracketed chapters
            r"Ch\.?\s*(\d+)",  # Abbreviated chapter
        ]
        
        # Search in first 1000 characters
        preview = text[:1000]
        
        for pattern in content_patterns:
            match = re.search(pattern, preview, re.IGNORECASE | re.MULTILINE)
            if match:
                try:
                    # Check if it's roman numerals
                    if match.group(1).upper() in 'IVXLCDM' and all(c in 'IVXLCDM' for c in match.group(1).upper()):
                        chapter_num = roman_to_int(match.group(1).upper())
                    else:
                        chapter_num = int(match.group(1))
                    
                    if len(match.groups()) > 1 and match.group(2):
                        chapter_title = match.group(2).strip()
                    break
                except:
                    continue
    
    # Final fallback - extract from response_XXX pattern
    if chapter_num is None:
        simple_match = re.search(r"response[_\s]*(\d+)", filename, re.IGNORECASE)
        if simple_match:
            try:
                chapter_num = int(simple_match.group(1))
            except:
                pass
    
    # If still no number, try to extract any number from filename
    if chapter_num is None:
        numbers = re.findall(r'\d+', filename)
        if numbers:
            # Take the first reasonable number (not too large)
            for num_str in numbers:
                num = int(num_str)
                if 0 < num < 10000:  # Reasonable chapter range
                    chapter_num = num
                    break
    
    return chapter_num, chapter_title

# Report generation functions
def generate_summary_stats(results, duplicate_groups):
    """Generate summary statistics"""
    total_files = len(results)
    files_with_issues = sum(1 for r in results if r['issues'])
    
    issue_counts = Counter()
    for result in results:
        for issue in result['issues']:
            # Simplify issue names for counting
            issue_type = issue.split('_')[0]
            issue_counts[issue_type] += 1
    
    stats = {
        'total_files': total_files,
        'files_with_issues': files_with_issues,
        'duplicate_groups': len(duplicate_groups),
        'issue_breakdown': dict(issue_counts)
    }
    
    return stats

def generate_html_report(results, duplicate_groups, stats):
    """Generate HTML report"""
    html = f"""
    <html>
    <head>
        <title>QA Scan Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .summary {{ background: #f0f0f0; padding: 15px; border-radius: 5px; }}
            .issue {{ background: #ffe0e0; padding: 10px; margin: 5px 0; border-radius: 3px; }}
            .duplicate {{ background: #fff0e0; padding: 10px; margin: 5px 0; border-radius: 3px; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
            .high-priority {{ color: #d32f2f; font-weight: bold; }}
            .medium-priority {{ color: #f57c00; }}
            .low-priority {{ color: #388e3c; }}
        </style>
    </head>
    <body>
        <h1>QA Scan Report</h1>
        <div class="summary">
            <h2>Summary</h2>
            <p>Total files scanned: {stats['total_files']}</p>
            <p>Files with issues: {stats['files_with_issues']}</p>
            <p>Duplicate groups found: {stats['duplicate_groups']}</p>
        </div>
    """
    
    # Add issue breakdown
    if stats['issue_breakdown']:
        html += """
        <h2>Issue Breakdown</h2>
        <ul>
        """
        for issue_type, count in sorted(stats['issue_breakdown'].items(), key=lambda x: x[1], reverse=True):
            html += f"<li>{issue_type}: {count} files</li>"
        html += "</ul>"
    
    # Add duplicate groups
    if duplicate_groups:
        html += """
        <h2>Duplicate Groups</h2>
        """
        for group_id, group in duplicate_groups.items():
            html += f"""
            <div class="duplicate">
                <h3>Group {group_id} (Confidence: {group['confidence']:.2%})</h3>
                <ul>
            """
            for filename in sorted(group['files']):
                html += f"<li>{filename}</li>"
            html += """
                </ul>
            </div>
            """
    
    # Add detailed results table
    html += """
    <h2>Detailed Results</h2>
    <table>
        <tr>
            <th>Filename</th>
            <th>Chapter</th>
            <th>Issues</th>
            <th>Foreign %</th>
            <th>Duplicate Group</th>
        </tr>
    """
    
    for result in sorted(results, key=lambda x: x['score'], reverse=True):
        if result['issues']:
            # Find duplicate group
            dup_group = ""
            for group_id, group in duplicate_groups.items():
                if result['filename'] in group['files']:
                    dup_group = f"Group {group_id}"
                    break
            
            # Determine priority
            priority_class = ""
            if result['score'] >= 5:
                priority_class = "high-priority"
            elif result['score'] >= 3:
                priority_class = "medium-priority"
            else:
                priority_class = "low-priority"
            
            html += f"""
            <tr class="{priority_class}">
                <td>{result['filename']}</td>
                <td>{result.get('chapter_num', 'N/A')}</td>
                <td>{', '.join(result['issues'][:3])}{'...' if len(result['issues']) > 3 else ''}</td>
                <td>{result['foreign_percentage']:.1f}%</td>
                <td>{dup_group}</td>
            </tr>
            """
    
    html += """
        </table>
    </body>
    </html>
    """
    
    return html

def generate_reports(results, folder_path, duplicate_groups, log, qa_settings):
    """Generate reports based on configured format"""
    report_format = qa_settings.get('report_format', 'detailed')
    auto_save = qa_settings.get('auto_save_report', True)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    # Calculate statistics
    stats = generate_summary_stats(results, duplicate_groups)
    
    # Log summary
    log(f"\n📊 QA Scan Summary:")
    log(f"   Total files: {stats['total_files']}")
    log(f"   Files with issues: {stats['files_with_issues']}")
    log(f"   Duplicate groups: {stats['duplicate_groups']}")
    
    if stats['issue_breakdown']:
        log(f"\n📋 Issue breakdown:")
        for issue_type, count in sorted(stats['issue_breakdown'].items(), key=lambda x: x[1], reverse=True):
            log(f"   - {issue_type}: {count} files")
    
    if not auto_save:
        log(f"\n✅ Scan complete! Use 'Generate Report' button to save results.")
        return
    
    # Generate report based on format
    if report_format == 'simple':
        # Simple text summary
        summary_file = os.path.join(folder_path, f"qa_summary_{timestamp}.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("QA SCAN SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Scan Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Files: {stats['total_files']}\n")
            f.write(f"Files with Issues: {stats['files_with_issues']}\n")
            f.write(f"Duplicate Groups: {stats['duplicate_groups']}\n\n")
            
            if duplicate_groups:
                f.write("DUPLICATE GROUPS:\n")
                for group_id, group in duplicate_groups.items():
                    f.write(f"\nGroup {group_id} ({len(group['files'])} files):\n")
                    for filename in sorted(group['files']):
                        f.write(f"  - {filename}\n")
        
        log(f"📄 Summary report saved: {summary_file}")
    
    elif report_format == 'detailed' or report_format == 'verbose':
        # Detailed CSV report
        detailed_file = os.path.join(folder_path, f"qa_report_{timestamp}.csv")
        
        with open(detailed_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'filename', 'file_size', 'foreign_char_percentage', 
                'foreign_chars_found', 'issues', 'duplicate_of'
            ])
            writer.writeheader()
            
            for result in results:
                # Find duplicate group
                duplicate_of = None
                for group_id, group in duplicate_groups.items():
                    if result['filename'] in group['files']:
                        duplicate_of = f"Group_{group_id}"
                        break
                
                writer.writerow({
                    'filename': result['filename'],
                    'file_size': result['file_size'],
                    'foreign_char_percentage': f"{result['foreign_percentage']:.2f}%",
                    'foreign_chars_found': ', '.join(f"{char}({count})" 
                        for char, count in result.get('top_foreign_chars', {}).items()),
                    'issues': ', '.join(result['issues']),
                    'duplicate_of': duplicate_of or ''
                })
        
        log(f"📄 Detailed report saved: {detailed_file}")
    
    # Verbose report (if selected)
    if report_format == 'verbose':
        verbose_file = os.path.join(folder_path, f"qa_report_verbose_{timestamp}.json")
        
        # Prepare data for JSON export
        export_data = {
            'scan_info': {
                'date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_files': len(results),
                'files_with_issues': sum(1 for r in results if r['issues']),
                'duplicate_groups': len(duplicate_groups)
            },
            'results': results,
            'duplicate_groups': [
                {
                    'group_id': group_id,
                    'files': list(group['files']),
                    'confidence': group['confidence']
                }
                for group_id, group in duplicate_groups.items()
            ]
        }
        
        with open(verbose_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        log(f"📄 Verbose report saved: {verbose_file}")
    
    # Always generate HTML report for visual inspection
    html_file = os.path.join(folder_path, f"qa_report_{timestamp}.html")
    html_content = generate_html_report(results, duplicate_groups, stats)
    
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    log(f"📄 HTML report saved: {html_file}")
    
    log(f"\n✅ All reports generated successfully!")

def update_progress_file(folder_path, results, log):
    """Update progress.json with QA scan results"""
    progress_file = os.path.join(folder_path, "progress.json")
    
    if not os.path.exists(progress_file):
        return
    
    try:
        with open(progress_file, 'r', encoding='utf-8') as f:
            progress_data = json.load(f)
        
        # Add QA scan results
        progress_data['qa_scan'] = {
            'last_scan': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_files': len(results),
            'files_with_issues': sum(1 for r in results if r['issues']),
            'issues_found': sum(len(r['issues']) for r in results)
        }
        
        # Update per-chapter info if available
        if 'chapters' in progress_data:
            for result in results:
                if result.get('chapter_num') is not None:
                    chapter_key = str(result['chapter_num'] - 1)  # Adjust for 0-indexing
                    if chapter_key in progress_data['chapters']:
                        progress_data['chapters'][chapter_key]['qa_issues'] = result['issues']
                        progress_data['chapters'][chapter_key]['qa_score'] = result['score']
        
        # Save updated progress
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, indent=2, ensure_ascii=False)
        
        log("✅ Updated progress.json with QA results")
    
    except Exception as e:
        log(f"⚠️ Could not update progress file: {e}")

def remove_faulty_chapters_from_progress(prog, faulty_indices, log):
    """Remove faulty chapters from the legacy completed list in progress data"""
    # Only process if there's an old-style completed list
    if "completed" not in prog or not isinstance(prog["completed"], list):
        return
    
    log(f"\n🔧 Cleaning up legacy completed list...")
    log(f"   Found {len(prog['completed'])} entries in old completed list")
    
    # Remove faulty indices from completed list
    original_count = len(prog["completed"])
    prog["completed"] = [idx for idx in prog["completed"] if idx not in faulty_indices]
    removed_count = original_count - len(prog["completed"])
    
    # Also remove from chapter_chunks if present
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

def scan_html_folder(folder_path, log=print, stop_flag=None, qa_settings=None):
    """
    Scan HTML folder for QA issues with configurable settings
    
    Args:
        folder_path: Path to folder containing HTML files
        log: Logging function
        stop_flag: Function that returns True to stop scanning
        mode: Detection mode ('ai-hunter', 'aggressive', 'standard', 'strict')
        qa_settings: Dictionary of QA scanner settings
    """
    # Load default settings if not provided
    if qa_settings is None:
        qa_settings = {
            'foreign_char_threshold': 10,
            'excluded_characters': '',
            'check_encoding_issues': True,
            'check_repetition': True,
            'check_translation_artifacts': True,
            'min_file_length': 100,
            'report_format': 'detailed',
            'auto_save_report': True
        }
    
    # Initialize configuration
    config = DuplicateDetectionConfig(qa_settings)
    
    # Display detection mode
    if config.use_ai_hunter:
        log(f"🤖 AI HUNTER duplicate detection active")
        log(f"   Thresholds: similarity={config.get_threshold('similarity'):.0%}, "
            f"semantic={config.get_threshold('semantic'):.0%}, "
            f"structural={config.get_threshold('structural'):.0%}")
        log("   ⚠️ WARNING: AI Hunter checks ALL file pairs - this can take several minutes!")
        log("   🎯 Designed to catch AI retranslations of the same content")
    else:
        log(f"📋 Standard duplicate detection")
        log(f"   Thresholds: similarity={config.get_threshold('similarity'):.0%}, "
            f"semantic={config.get_threshold('semantic'):.0%}, "
        f"structural={config.get_threshold('structural'):.0%}")
    
    html_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".html")])
    log(f"🔍 Found {len(html_files)} HTML files. Starting scan...")
    
    results = []
    
    # First pass: collect all data
    for idx, filename in enumerate(html_files):
        if stop_flag and stop_flag():
            log("⛔ QA scan interrupted by user.")
            return
        
        log(f"📄 [{idx+1}/{len(html_files)}] Scanning {filename}...")
        
        full_path = os.path.join(folder_path, filename)
        try:
            raw_text = extract_text_from_html(full_path)
        except Exception as e:
            log(f"⚠️ Failed to read {filename}: {e}")
            continue
        
        # Check minimum file length from settings
        min_length = qa_settings.get('min_file_length', 100)
        if len(raw_text.strip()) < min_length:
            log(f"⚠️ Skipped {filename}: Too short (< {min_length} chars)")
            continue
        
        chapter_num, chapter_title = extract_chapter_info(filename, raw_text)
        hashes = generate_content_hashes(raw_text)
        
        preview = raw_text[:500].replace('\n', ' ')
        if len(preview) > 500:
            preview = preview[:497] + '...'
        
        # Normalize preview
        preview_normalized = normalize_text(preview)[:300]
        
        # Detect translation artifacts only if enabled
        artifacts = []
        if qa_settings.get('check_translation_artifacts', True):
            artifacts = detect_translation_artifacts(raw_text)
        
        results.append({
            "file_index": idx,
            "filename": filename,
            "filepath": full_path,
            "issues": [],
            "preview": preview,
            "preview_normalized": preview_normalized,
            "score": 0,
            "chapter_num": chapter_num,
            "hashes": hashes,
            "raw_text": raw_text,
            "translation_artifacts": artifacts,
            "file_size": len(raw_text)
        })
    
    log("\n✅ Initial scan complete.")
    
    # Second pass: detect duplicates
    log("\n🔍 Checking for duplicates...")
    duplicate_groups = {}
    duplicate_confidence = {}
    
    # MinHash optimization for large datasets
    minhashes = {}
    lsh = None
    
    if MINHASH_AVAILABLE and len(results) > 50:
        log("   Using MinHash optimization for faster duplicate detection...")
        threshold = config.get_threshold('minhash_threshold')
        lsh = MinHashLSH(threshold=threshold, num_perm=128)
        
        for result in results:
            minhash = create_minhash(result['raw_text'])
            if minhash:
                minhashes[result['filename']] = minhash
                lsh.insert(result['filename'], minhash)
    
    # Check each pair for duplicates
    for i in range(len(results)):
        if stop_flag and stop_flag():
            log("⛔ Duplicate detection interrupted.")
            break
        
        # If MinHash is available, only check similar candidates
        if lsh and results[i]['filename'] in minhashes:
            candidates = lsh.query(minhashes[results[i]['filename']])
            # Convert filenames to indices
            candidate_indices = []
            for candidate_file in candidates:
                for idx, r in enumerate(results):
                    if r['filename'] == candidate_file and idx > i:
                        candidate_indices.append(idx)
        else:
            # No MinHash, check all pairs (slow)
            candidate_indices = range(i + 1, len(results))
        
        for j in candidate_indices:
            pair = (results[i]['filename'], results[j]['filename'])
            
            # Skip if already marked as duplicates
            if pair in duplicate_confidence and duplicate_confidence[pair] >= 0.95:
                continue
            
            # Check for duplicates based on mode
            if config.use_ai_hunter:
                is_dup, confidence, methods = is_likely_duplicate_ai_hunter(results[i], results[j], config)
            else:
                is_dup, confidence, methods = is_likely_duplicate_basic(results[i], results[j], config)
            
            if is_dup:
                duplicate_confidence[pair] = confidence
                
                # Add to duplicate groups
                group_found = False
                for group_id, group in duplicate_groups.items():
                    if results[i]['filename'] in group['files'] or results[j]['filename'] in group['files']:
                        group['files'].add(results[i]['filename'])
                        group['files'].add(results[j]['filename'])
                        group['confidence'] = max(group['confidence'], confidence)
                        group_found = True
                        break
                
                if not group_found:
                    duplicate_groups[f"dup_{len(duplicate_groups)}"] = {
                        'files': {results[i]['filename'], results[j]['filename']},
                        'confidence': confidence
                    }
    
    log(f"✅ Found {len(duplicate_groups)} duplicate groups")
    
    # Third pass: compile issues
    log("\n📋 Compiling issues...")
    for result in results:
        issues = []
        
        # Check exact duplicates
        if result['filename'] in duplicate_groups:
            group_id = duplicate_groups[result['filename']]
            group_files = [f for f, gid in duplicate_groups.items() if gid == group_id]
            if len(group_files) > 1:
                others = [f for f in group_files if f != result['filename']]
                issues.append(f"EXACT_DUPLICATE: {len(group_files)}_file_group")
        
        # Check near-duplicates
        near_duplicates = []
        for other_file, confidence in duplicate_confidence.items():
            if result['filename'] in other_file and confidence < 0.95:
                near_duplicates.append((other_file, confidence))
        
        if near_duplicates:
            issues.append(f"NEAR_DUPLICATE: {len(near_duplicates)}_similar_files")
        
        # Check other issues
        raw_text = result['raw_text']
        
        # Non-English content (excluding Korean separators) - pass settings
        has_non_english, lang_issues = detect_non_english_content(raw_text, qa_settings)
        if has_non_english:
            issues.extend(lang_issues)
        
        # Foreign characters - separate check (keeping for compatibility)
        foreign_percentage, top_foreign = detect_foreign_characters(
            raw_text, 
            qa_settings.get('foreign_char_threshold', 10),
            qa_settings.get('excluded_characters', '')
        )
        result['foreign_percentage'] = foreign_percentage
        result['top_foreign_chars'] = top_foreign
        
        # Spacing/formatting issues - only if encoding check is enabled
        if qa_settings.get('check_encoding_issues', True):
            if has_no_spacing_or_linebreaks(raw_text):
                issues.append("no_spacing_or_linebreaks")
        
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
                    issues.append(f"encoding_issues_{artifact['count']}_found")
                elif artifact['type'] == 'repeated_watermarks':
                    issues.append(f"repeated_watermarks_found")
        
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
    generate_reports(results, folder_path, duplicate_groups, log, qa_settings)
    
    # Update progress file
    update_progress_file(folder_path, results, log)

def launch_gui():
    """Launch GUI interface"""
    def run_scan():
        folder_path = filedialog.askdirectory(title="Select Folder with HTML Files")
        if folder_path:
            def scan_thread():
                # Run scan with default settings
                scan_html_folder(folder_path, print, None)
                messagebox.showinfo("Complete", "QA scan completed!")
            
            # Run in thread to prevent UI freeze
            thread = threading.Thread(target=scan_thread)
            thread.start()
    
    # Create GUI
    root = tk.Tk()
    root.title("QA Scanner")
    root.geometry("400x200")
    
    # Info label
    tk.Label(root, text="QA Scanner\nConfigure settings in the main application", 
             font=("Arial", 12)).pack(pady=20)
    
    # Scan button
    tk.Button(root, text="Select Folder and Scan", command=run_scan, 
              bg="#4CAF50", fg="white", font=("Arial", 12)).pack(pady=20)
    
    # Info label
    info = tk.Label(root, text="Tip: For large datasets (100+ files),\ndisable AI Hunter for faster scanning", 
                    font=("Arial", 9), fg="gray")
    info.pack()
    
    root.mainloop()

# Main execution
if __name__ == "__main__":
    if len(sys.argv) < 2:
        launch_gui()
    else:
        # Command line mode
        folder_path = sys.argv[1]
        
        # Simple default settings
        config = {
            'foreign_char_threshold': 10,
            'excluded_characters': '',
            'check_encoding_issues': True,
            'check_repetition': True,
            'check_translation_artifacts': True,
            'min_file_length': 100,
            'report_format': 'detailed',
            'auto_save_report': True,
            'ai_hunter_enabled': False,
            'duplicate_thresholds': {
                'similarity': 0.85,
                'semantic': 0.80,
                'structural': 0.90,
                'word_overlap': 0.75,
                'minhash_threshold': 0.80
            }
        }
        
        print(f"Starting scan of: {folder_path}")
        scan_html_folder(folder_path, qa_settings=config)

# Test function for debugging
def test_stop_functionality():
    """Test function to verify stop_scan works"""
    global _stop_flag
    print(f"Before stop_scan: _stop_flag = {_stop_flag}")
    stop_scan()
    print(f"After stop_scan: _stop_flag = {_stop_flag}")
    reset_stop_flag()
    print(f"After reset: _stop_flag = {_stop_flag}")
    return True

# Export all public functions
__all__ = ['scan_html_folder', 'stop_scan', 'reset_stop_flag', 'is_stop_requested', 
           'test_stop_functionality']
