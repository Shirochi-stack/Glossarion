# -*- coding: utf-8 -*-
import json
import logging
import shutil
import threading
import queue
import os, sys, io, zipfile, time, re, mimetypes, subprocess, tiktoken
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from collections import Counter
from unified_api_client import UnifiedClient, UnifiedClientError
import hashlib
import unicodedata
from difflib import SequenceMatcher
import unicodedata
import re

# Import the new modules
from history_manager import HistoryManager
from chapter_splitter import ChapterSplitter

def make_safe_filename(title, chapter_num):
    """Create a safe filename that works across different filesystems"""
    
    # First, try to clean the title
    if not title:
        return f"chapter_{chapter_num:03d}"
    
    # Normalize unicode
    title = unicodedata.normalize('NFC', str(title))
    
    # Replace path separators and other dangerous characters
    dangerous_chars = {
        '/': '_', '\\': '_', ':': '_', '*': '_', '?': '_',
        '"': '_', '<': '_', '>': '_', '|': '_', '\0': '',
        '\n': ' ', '\r': ' ', '\t': ' '
    }
    
    for old, new in dangerous_chars.items():
        title = title.replace(old, new)
    
    # Remove control characters
    title = ''.join(char for char in title if ord(char) >= 32)
    
    # Replace multiple spaces with single underscore
    title = re.sub(r'\s+', '_', title)
    
    # Remove leading/trailing underscores and dots
    title = title.strip('_.• \t')
    
    # Limit length (leave room for response_XXX_ prefix)
    if len(title) > 40:
        title = title[:40].rstrip('_.')
    
    # If title is empty after cleaning, use chapter number
    if not title or title == '_' * len(title):
        title = f"chapter_{chapter_num:03d}"
    
    return title

# optional: turn on HTTP‐level debugging in the OpenAI client
logging.basicConfig(level=logging.DEBUG)

# Fix for PyInstaller - handle stdout reconfigure more carefully
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

# Global stop flag for GUI integration
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
        class CallbackWriter:
            def __init__(self, callback):
                self.callback = callback
                
            def write(self, text):
                if text.strip():
                    self.callback(text.strip())
                    
            def flush(self):
                pass
                
        sys.stdout = CallbackWriter(log_callback)

def get_instructions(lang):
    """Get minimal technical instructions only"""
    # Only return the technical requirement that applies to all languages
    return "Preserve ALL HTML tags exactly as they appear in the source, including <head>, <title> ,<h1>, <h2>, <p>, <br>, <div>, etc."

# =============================================================================
# PROGRESS TRACKING AND MANAGEMENT
# =============================================================================

def init_progress_tracking(payloads_dir):
    """Initialize or load progress tracking with improved structure"""
    PROGRESS_FILE = os.path.join(payloads_dir, "translation_progress.json")
    
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as pf:
                prog = json.load(pf)
        except json.JSONDecodeError as e:
            print(f"⚠️ Warning: Progress file is corrupted: {e}")
            print("🔧 Attempting to fix JSON syntax...")
            
            # Try to fix common JSON errors
            try:
                with open(PROGRESS_FILE, "r", encoding="utf-8") as pf:
                    content = pf.read()
                
                # Fix trailing commas in arrays
                import re
                # Fix trailing commas before closing brackets
                content = re.sub(r',\s*\]', ']', content)
                # Fix trailing commas before closing braces
                content = re.sub(r',\s*\}', '}', content)
                
                # Try to parse the fixed content
                prog = json.loads(content)
                
                # Save the fixed content back
                with open(PROGRESS_FILE, "w", encoding="utf-8") as pf:
                    json.dump(prog, pf, ensure_ascii=False, indent=2)
                print("✅ Successfully fixed and saved progress file")
                
            except Exception as fix_error:
                print(f"❌ Could not fix progress file: {fix_error}")
                print("🔄 Creating backup and starting fresh...")
                
                # Create a backup of the corrupted file
                import time
                backup_name = f"translation_progress_backup_{int(time.time())}.json"
                backup_path = os.path.join(payloads_dir, backup_name)
                try:
                    shutil.copy(PROGRESS_FILE, backup_path)
                    print(f"📁 Backup saved to: {backup_name}")
                except:
                    pass
                
                # Start with a fresh progress file
                prog = {
                    "chapters": {},
                    "content_hashes": {},
                    "chapter_chunks": {},
                    "version": "2.0"
                }
        
        # Migrate old format to new format if needed
        if "chapters" not in prog:
            prog["chapters"] = {}
            
            # Migrate from old completed list
            for idx in prog.get("completed", []):
                prog["chapters"][str(idx)] = {
                    "status": "completed",
                    "timestamp": None
                }
        
        # Ensure all required fields exist
        if "content_hashes" not in prog:
            prog["content_hashes"] = {}
        if "chapter_chunks" not in prog:
            prog["chapter_chunks"] = {}
            
    else:
        prog = {
            "chapters": {},          # Main tracking: idx -> chapter info
            "content_hashes": {},    # Content deduplication
            "chapter_chunks": {},    # Chunk tracking for large chapters
            "version": "2.0"         # Track format version
        }
    
    return prog, PROGRESS_FILE

def check_chapter_status(prog, chapter_idx, chapter_num, content_hash, output_dir):
    """Check if a chapter needs translation"""
    chapter_key = str(chapter_idx)
    
    # FIRST: Always check if the actual output file exists
    if chapter_key in prog["chapters"]:
        chapter_info = prog["chapters"][chapter_key]
        output_file = chapter_info.get("output_file")
        
        if output_file:
            output_path = os.path.join(output_dir, output_file)
            if not os.path.exists(output_path):
                # File was deleted! Mark chapter as needing retranslation
                print(f"⚠️ Output file missing for chapter {chapter_num}: {output_file}")
                print(f"🔄 Chapter {chapter_num} will be retranslated")
                
                # Clean up progress tracking for this chapter
                if content_hash in prog["content_hashes"]:
                    stored_info = prog["content_hashes"][content_hash]
                    if stored_info.get("chapter_idx") == chapter_idx:
                        del prog["content_hashes"][content_hash]
                
                # Mark chapter as needing retranslation
                chapter_info["status"] = "file_deleted"
                chapter_info["output_file"] = None
                
                # Also clear any chunk data
                if str(chapter_idx) in prog.get("chapter_chunks", {}):
                    del prog["chapter_chunks"][str(chapter_idx)]
                
                return True, None, None
            else:
                # File exists, check if it's a valid translation
                try:
                    with open(output_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if len(content.strip()) < 100:
                            print(f"⚠️ Output file for chapter {chapter_num} seems corrupted/empty")
                            return True, None, None
                except Exception as e:
                    print(f"⚠️ Error reading output file for chapter {chapter_num}: {e}")
                    return True, None, None
                
                return False, f"Chapter {chapter_num} already translated (file exists: {output_file})", output_file
        
        elif chapter_info.get("status") in ["in_progress", "file_deleted"]:
            return True, None, None
    
    # Check for duplicate content ONLY if we haven't already determined we need to translate
    if content_hash in prog["content_hashes"]:
        duplicate_info = prog["content_hashes"][content_hash]
        duplicate_idx = duplicate_info.get("chapter_idx")
        
        # Only skip if the duplicate's file ACTUALLY exists
        if str(duplicate_idx) in prog["chapters"]:
            dup_chapter = prog["chapters"][str(duplicate_idx)]
            dup_output_file = dup_chapter.get("output_file")
            if dup_output_file:
                dup_path = os.path.join(output_dir, dup_output_file)
                if os.path.exists(dup_path):
                    try:
                        with open(dup_path, 'r', encoding='utf-8') as f:
                            dup_content = f.read()
                            if len(dup_content.strip()) < 100:
                                return True, None, None
                    except:
                        return True, None, None
                    
                    return False, f"Chapter {chapter_num} has same content as chapter {duplicate_info.get('chapter_num')} (already translated)", None
                else:
                    return True, None, None
    
    return True, None, None

def cleanup_missing_files(prog, output_dir):
    """Scan progress tracking and clean up any references to missing files"""
    cleaned_count = 0
    
    for chapter_key, chapter_info in list(prog["chapters"].items()):
        output_file = chapter_info.get("output_file")
        
        if output_file:
            output_path = os.path.join(output_dir, output_file)
            if not os.path.exists(output_path):
                print(f"🧹 Found missing file for chapter {chapter_info.get('chapter_num', chapter_key)}: {output_file}")
                
                chapter_info["status"] = "file_deleted"
                chapter_info["output_file"] = None
                
                content_hash = chapter_info.get("content_hash")
                if content_hash and content_hash in prog["content_hashes"]:
                    stored_info = prog["content_hashes"][content_hash]
                    if stored_info.get("chapter_idx") == int(chapter_key):
                        del prog["content_hashes"][content_hash]
                
                if chapter_key in prog.get("chapter_chunks", {}):
                    del prog["chapter_chunks"][chapter_key]
                
                cleaned_count += 1
    
    if cleaned_count > 0:
        print(f"🔄 Marked {cleaned_count} chapters for retranslation due to missing files")
    
    return prog

def update_progress(prog, chapter_idx, chapter_num, content_hash, output_filename=None, status="completed"):
    """Update progress tracking after successful translation"""
    chapter_key = str(chapter_idx)
    
    prog["chapters"][chapter_key] = {
        "chapter_num": chapter_num,
        "content_hash": content_hash,
        "output_file": output_filename,
        "status": status,
        "timestamp": time.time()
    }
    
    if output_filename and status == "completed":
        prog["content_hashes"][content_hash] = {
            "chapter_idx": chapter_idx,
            "chapter_num": chapter_num,
            "output_file": output_filename
        }
    
    return prog

def cleanup_progress_tracking(prog, output_dir):
    """Remove entries for files that no longer exist"""
    cleaned_count = 0
    
    for chapter_key, chapter_info in list(prog["chapters"].items()):
        if chapter_info.get("output_file"):
            output_path = os.path.join(output_dir, chapter_info["output_file"])
            if not os.path.exists(output_path):
                chapter_info["status"] = "file_missing"
                cleaned_count += 1
                print(f"🧹 Marked chapter {chapter_info.get('chapter_num', chapter_key)} as missing (file not found: {chapter_info['output_file']})")
    
    if cleaned_count > 0:
        print(f"🧹 Found {cleaned_count} chapters with missing files")
    
    return prog

def get_translation_stats(prog, output_dir):
    """Get statistics about translation progress"""
    stats = {
        "total_tracked": len(prog["chapters"]),
        "completed": 0,
        "missing_files": 0,
        "in_progress": 0
    }
    
    for chapter_info in prog["chapters"].values():
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

# =============================================================================
# CONTENT PROCESSING AND ENHANCEMENT
# =============================================================================

def emergency_restore_paragraphs(text, original_html=None, verbose=True):
    """Emergency restoration when AI returns wall of text without proper paragraph tags"""
    def log(message):
        if verbose:
            print(message)
    
    # Check if we already have proper paragraph structure
    if text.count('</p>') >= 3:
        return text
    
    # If we have the original HTML, try to match its structure
    if original_html:
        original_para_count = original_html.count('<p>')
        current_para_count = text.count('<p>')
        
        if current_para_count < original_para_count / 2:
            log(f"⚠️ Paragraph mismatch! Original: {original_para_count}, Current: {current_para_count}")
            log("🔧 Attempting emergency paragraph restoration...")
    
    # If no paragraph tags found and text is long, we have a problem
    if '</p>' not in text and len(text) > 300:
        log("❌ No paragraph tags found - applying emergency restoration")
        
        # Strategy 1: Look for double line breaks
        if '\n\n' in text:
            parts = text.split('\n\n')
            paragraphs = ['<p>' + part.strip() + '</p>' for part in parts if part.strip()]
            return '\n'.join(paragraphs)
        
        # Strategy 2: Look for dialogue patterns
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
        
        # Strategy 3: Split by sentence patterns
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
        
        # Strategy 4: Last resort - fixed size chunks
        words = text.split()
        if len(words) > 100:
            paragraphs = []
            words_per_para = max(100, len(words) // 10)
            
            for i in range(0, len(words), words_per_para):
                chunk = ' '.join(words[i:i + words_per_para])
                if chunk.strip():
                    paragraphs.append('<p>' + chunk.strip() + '</p>')
            
            return '\n'.join(paragraphs)
    
    # Handle incomplete structure
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

def clean_ai_artifacts(text, remove_artifacts=True):
    """Remove AI response artifacts from text - but ONLY when enabled"""
    if not remove_artifacts:
        return text
    
    lines = text.split('\n', 2)
    
    if len(lines) < 2:
        return text
    
    first_line = lines[0].strip()
    
    if not first_line:
        if len(lines) > 1:
            first_line = lines[1].strip()
            if not first_line:
                return text
            lines = lines[1:]
        else:
            return text
    
    # Common AI artifact patterns - be very specific
    ai_patterns = [
        r'^(?:Sure|Okay|Understood|Of course|Got it|Alright|Certainly|Here\'s|Here is)',
        r'^(?:I\'ll|I will|Let me) (?:translate|help|assist)',
        r'^(?:System|Assistant|AI|User|Human|Model)\s*:',
        r'^\[PART\s+\d+/\d+\]',
        r'^(?:Translation note|Note|Here\'s the translation|I\'ve translated)',
        r'^```(?:html)?',
        r'^<!DOCTYPE',
    ]
    
    for pattern in ai_patterns:
        if re.search(pattern, first_line, re.IGNORECASE):
            remaining_text = '\n'.join(lines[1:]) if len(lines) > 1 else ''
            
            if remaining_text.strip():
                if (re.search(r'<h[1-6]', remaining_text, re.IGNORECASE) or 
                    re.search(r'Chapter\s+\d+', remaining_text, re.IGNORECASE) or
                    re.search(r'第\s*\d+\s*[章節話话回]', remaining_text) or
                    re.search(r'제\s*\d+\s*[장화]', remaining_text) or
                    len(remaining_text.strip()) > 100):
                    
                    print(f"✂️ Removed AI artifact: {first_line[:50]}...")
                    return remaining_text.lstrip()
    
    # Additional check: single word artifacts
    if first_line.lower() in ['html', 'text', 'content', 'translation', 'output']:
        remaining_text = '\n'.join(lines[1:]) if len(lines) > 1 else ''
        if remaining_text.strip():
            print(f"✂️ Removed single word artifact: {first_line}")
            return remaining_text.lstrip()
    
    return text

# =============================================================================
# EPUB METADATA AND STRUCTURE EXTRACTION
# =============================================================================

def extract_epub_metadata(zf):
    """Extract comprehensive metadata from EPUB file"""
    meta = {}
    try:
        for name in zf.namelist():
            if name.lower().endswith('.opf'):
                opf_content = zf.read(name)
                soup = BeautifulSoup(opf_content, 'xml')
                
                for tag in ['title', 'creator', 'language', 'publisher', 'date', 'subject']:
                    element = soup.find(tag)
                    if element:
                        meta[tag] = element.get_text(strip=True)
                
                description = soup.find('description')
                if description:
                    meta['description'] = description.get_text(strip=True)
                
                meta_tags = soup.find_all('meta')
                for meta_tag in meta_tags:
                    name = meta_tag.get('name', '').lower()
                    content = meta_tag.get('content', '')
                    if 'series' in name and content:
                        meta['series'] = content
                    elif 'calibre:series' in name and content:
                        meta['series'] = content
                
                break
    except Exception as e:
        print(f"[WARNING] Failed to extract metadata: {e}")
    
    return meta

def detect_content_language(text_sample):
    """Detect the primary language of content"""
    scripts = {
        'korean': 0,
        'japanese_hiragana': 0,
        'japanese_katakana': 0,
        'chinese': 0,
        'latin': 0
    }
    
    for char in text_sample:
        code = ord(char)
        if 0xAC00 <= code <= 0xD7AF:  # Hangul syllables
            scripts['korean'] += 1
        elif 0x3040 <= code <= 0x309F:  # Hiragana
            scripts['japanese_hiragana'] += 1
        elif 0x30A0 <= code <= 0x30FF:  # Katakana
            scripts['japanese_katakana'] += 1
        elif 0x4E00 <= code <= 0x9FFF:  # CJK Unified Ideographs
            scripts['chinese'] += 1
        elif 0x0020 <= code <= 0x007F:  # Basic Latin
            scripts['latin'] += 1
    
    total_cjk = scripts['korean'] + scripts['japanese_hiragana'] + scripts['japanese_katakana'] + scripts['chinese']
    
    if scripts['korean'] > total_cjk * 0.3:
        return 'korean'
    elif scripts['japanese_hiragana'] + scripts['japanese_katakana'] > total_cjk * 0.2:
        return 'japanese'
    elif scripts['chinese'] > total_cjk * 0.3:
        return 'chinese'
    elif scripts['latin'] > len(text_sample) * 0.7:
        return 'english'
    else:
        return 'unknown'

def extract_comprehensive_content_hash(html_content):
    """Create a more comprehensive hash that captures content structure and meaning"""
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        content_parts = []
        
        # Text content (normalized)
        text_content = soup.get_text(strip=True).lower()
        text_content = re.sub(r'chapter\s+\d+[:\-\s]*', '', text_content, flags=re.IGNORECASE)
        text_content = re.sub(r'第\s*\d+\s*[章節话回][:\-\s]*', '', text_content)
        text_content = re.sub(r'제\s*\d+\s*[장화][:\-\s]*', '', text_content)
        text_content = re.sub(r'\s+', ' ', text_content).strip()
        content_parts.append(text_content[:2000])
        
        # Structural elements
        structure_info = []
        for tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            headers = soup.find_all(tag)
            for header in headers:
                header_text = header.get_text(strip=True).lower()
                if header_text and len(header_text) < 200:
                    structure_info.append(f"{tag}:{header_text}")
        
        # Paragraph count and average length
        paragraphs = soup.find_all('p')
        if paragraphs:
            total_p_length = sum(len(p.get_text()) for p in paragraphs)
            avg_p_length = total_p_length // len(paragraphs)
            structure_info.append(f"paragraphs:{len(paragraphs)}:{avg_p_length}")
        
        # Image and link information
        for img in soup.find_all('img'):
            src = img.get('src', '')
            alt = img.get('alt', '')
            if src:
                structure_info.append(f"img:{os.path.basename(src)}")
            if alt:
                structure_info.append(f"alt:{alt.lower()}")
        
        for link in soup.find_all('a'):
            href = link.get('href', '')
            link_text = link.get_text(strip=True).lower()
            if href and not href.startswith('#'):
                structure_info.append(f"link:{href}")
            if link_text:
                structure_info.append(f"linktext:{link_text}")
        
        # Combine all parts
        content_parts.extend(structure_info)
        combined_content = '|||'.join(content_parts)
        
        # Create multiple hash signatures
        main_hash = hashlib.md5(combined_content.encode('utf-8')).hexdigest()
        structure_hash = hashlib.md5(''.join(structure_info).encode('utf-8')).hexdigest()
        text_hash = hashlib.md5(text_content.encode('utf-8')).hexdigest()
        
        return f"{main_hash}_{structure_hash[:8]}_{text_hash[:8]}"
        
    except Exception as e:
        print(f"[WARNING] Failed to create comprehensive hash: {e}")
        simple_text = BeautifulSoup(html_content, 'html.parser').get_text()
        return hashlib.md5(simple_text.encode('utf-8')).hexdigest()

def get_content_hash(html_content):
    """Create a comprehensive hash of content to detect duplicates"""
    return extract_comprehensive_content_hash(html_content)

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
    if len(name) > 50:
        name = name[:50]
    
    if not name:
        name = 'resource'
    
    return name + ext

# =============================================================================
# CONSOLIDATED CHAPTER AND RESOURCE EXTRACTION
# =============================================================================

def extract_chapters(zf, output_dir):
    """
    Extract chapters and all resources from EPUB with comprehensive handling
    
    This function provides:
    - Chapter extraction with advanced detection (multiple languages)
    - Resource extraction (CSS, fonts, images, EPUB structure files)
    - Duplicate prevention on re-runs
    - Comprehensive metadata extraction
    - Validation and reporting
    - Smart cleanup of previous extractions
    
    Args:
        zf: ZipFile object of the EPUB
        output_dir: Directory to extract content to
        
    Returns:
        List of chapter dictionaries with enhanced metadata
    """
    
    print("🚀 Starting comprehensive EPUB extraction...")
    print("✅ Using enhanced extraction with full resource handling")
    
    # Step 1: Extract all resources (with duplicate prevention)
    extracted_resources = _extract_all_resources(zf, output_dir)
    
    # Step 2: Extract comprehensive metadata (only if not exists)
    metadata_path = os.path.join(output_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        print("📋 Loading existing metadata...")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    else:
        print("📋 Extracting fresh metadata...")
        metadata = extract_epub_metadata(zf)
        print(f"📋 Extracted metadata: {list(metadata.keys())}")
    
    # Step 3: Extract chapters with advanced detection
    chapters, detected_language = _extract_advanced_chapter_info(zf)
    
    if not chapters:
        print("❌ No chapters could be extracted!")
        return []
    
    # Step 4: Enhance metadata with extracted information
    metadata.update({
        'chapter_count': len(chapters),
        'detected_language': detected_language,
        'extracted_resources': extracted_resources,
        'extraction_summary': {
            'total_chapters': len(chapters),
            'chapter_range': f"{chapters[0]['num']}-{chapters[-1]['num']}",
            'resources_extracted': sum(len(files) for files in extracted_resources.values())
        }
    })
    
    # Add chapter titles to metadata
    metadata['chapter_titles'] = {
        str(c['num']): c['title'] for c in chapters
    }
    
    # Step 5: Save enhanced metadata
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Saved comprehensive metadata to: {metadata_path}")
    
    # Step 6: Create/update extraction report
    _create_extraction_report(output_dir, metadata, chapters, extracted_resources)
    
    # Step 7: Final validation and summary
    _log_extraction_summary(chapters, extracted_resources, detected_language)
    
    print("🔍 VERIFICATION: Comprehensive chapter extraction completed successfully")
    
    return chapters

def _extract_all_resources(zf, output_dir):
    """Extract all resources (CSS, fonts, images, EPUB structure files) with duplicate prevention"""
    extracted_resources = {
        'css': [],
        'fonts': [],
        'images': [],
        'epub_structure': [],
        'other': []
    }
    
    # Check if resources were already extracted
    extraction_marker = os.path.join(output_dir, '.resources_extracted')
    if os.path.exists(extraction_marker):
        print("📦 Resources already extracted, skipping resource extraction...")
        return _count_existing_resources(output_dir, extracted_resources)
    
    # Clean up any partial extractions from previous runs
    _cleanup_old_resources(output_dir)
    
    # Create fresh resource directories
    for resource_type in ['css', 'fonts', 'images']:
        os.makedirs(os.path.join(output_dir, resource_type), exist_ok=True)
    
    print(f"📦 Extracting all resources from EPUB...")
    
    # Extract each file
    for file_path in zf.namelist():
        if file_path.endswith('/') or not os.path.basename(file_path):
            continue
            
        try:
            file_data = zf.read(file_path)
            resource_info = _categorize_resource(file_path, os.path.basename(file_path))
            
            if resource_info:
                resource_type, target_dir, safe_filename = resource_info
                target_path = os.path.join(output_dir, target_dir, safe_filename) if target_dir else os.path.join(output_dir, safe_filename)
                
                # Write file
                with open(target_path, 'wb') as f:
                    f.write(file_data)
                
                extracted_resources[resource_type].append(safe_filename)
                
                # Log appropriately
                if resource_type == 'epub_structure':
                    print(f"   📋 Extracted EPUB structure: {safe_filename}")
                else:
                    print(f"   📄 Extracted {resource_type}: {safe_filename}")
                
        except Exception as e:
            print(f"[WARNING] Failed to extract {file_path}: {e}")
    
    # Create extraction marker
    with open(extraction_marker, 'w') as f:
        f.write(f"Resources extracted at {time.time()}")
    
    # Summary and validation
    _validate_critical_files(output_dir, extracted_resources)
    
    return extracted_resources

def _categorize_resource(file_path, file_name):
    """Categorize a file and return (resource_type, target_dir, safe_filename)"""
    file_path_lower = file_path.lower()
    file_name_lower = file_name.lower()
    
    if file_path_lower.endswith('.css'):
        return 'css', 'css', sanitize_resource_filename(file_name)
    elif file_path_lower.endswith(('.ttf', '.otf', '.woff', '.woff2', '.eot')):
        return 'fonts', 'fonts', sanitize_resource_filename(file_name)
    elif file_path_lower.endswith(('.jpg', '.jpeg', '.png', '.gif', '.svg', '.bmp', '.webp')):
        return 'images', 'images', sanitize_resource_filename(file_name)
    elif (file_path_lower.endswith(('.opf', '.ncx')) or 
          file_name_lower == 'container.xml' or
          'container.xml' in file_path_lower):
        # EPUB structure files go to root
        if 'container.xml' in file_path_lower:
            safe_filename = 'container.xml'
        else:
            safe_filename = file_name
        return 'epub_structure', None, safe_filename
    elif file_path_lower.endswith(('.js', '.xml', '.txt')):
        return 'other', None, sanitize_resource_filename(file_name)
    
    return None

def _cleanup_old_resources(output_dir):
    """Clean up old resource directories and EPUB structure files - with error handling"""
    print("🧹 Cleaning up any existing resource directories...")
    
    cleanup_success = True
    
    # Remove resource directories
    for resource_type in ['css', 'fonts', 'images']:
        resource_dir = os.path.join(output_dir, resource_type)
        if os.path.exists(resource_dir):
            try:
                shutil.rmtree(resource_dir)
                print(f"   🗑️ Removed old {resource_type} directory")
            except PermissionError as e:
                print(f"   ⚠️ Cannot remove {resource_type} directory (permission denied) - will merge with existing files")
                cleanup_success = False
            except Exception as e:
                print(f"   ⚠️ Error removing {resource_type} directory: {e} - will merge with existing files")
                cleanup_success = False
    
    # Remove EPUB structure files
    epub_structure_files = ['container.xml', 'content.opf', 'toc.ncx']
    for epub_file in epub_structure_files:
        epub_path = os.path.join(output_dir, epub_file)
        if os.path.exists(epub_path):
            try:
                os.remove(epub_path)
                print(f"   🗑️ Removed old {epub_file}")
            except PermissionError:
                print(f"   ⚠️ Cannot remove {epub_file} (permission denied) - will use existing file")
            except Exception as e:
                print(f"   ⚠️ Error removing {epub_file}: {e}")
    
    # Clean up any other .opf or .ncx files
    try:
        for file in os.listdir(output_dir):
            if file.lower().endswith(('.opf', '.ncx')):
                file_path = os.path.join(output_dir, file)
                try:
                    os.remove(file_path)
                    print(f"   🗑️ Removed old EPUB file: {file}")
                except PermissionError:
                    print(f"   ⚠️ Cannot remove {file} (permission denied)")
                except Exception as e:
                    print(f"   ⚠️ Error removing {file}: {e}")
    except Exception as e:
        print(f"⚠️ Error scanning for EPUB files: {e}")
    
    if not cleanup_success:
        print("⚠️ Some cleanup operations failed due to file permissions")
        print("   The program will continue and merge with existing files")
    
    return cleanup_success

def _count_existing_resources(output_dir, extracted_resources):
    """Count existing resources when skipping extraction"""
    for resource_type in ['css', 'fonts', 'images', 'epub_structure']:
        if resource_type == 'epub_structure':
            # EPUB structure files are in root directory
            epub_files = []
            for file in ['container.xml', 'content.opf', 'toc.ncx']:
                if os.path.exists(os.path.join(output_dir, file)):
                    epub_files.append(file)
            # Also check for any .opf files with different names
            try:
                for file in os.listdir(output_dir):
                    if file.lower().endswith(('.opf', '.ncx')) and file not in epub_files:
                        epub_files.append(file)
            except:
                pass
            extracted_resources[resource_type] = epub_files
        else:
            resource_dir = os.path.join(output_dir, resource_type)
            if os.path.exists(resource_dir):
                try:
                    files = [f for f in os.listdir(resource_dir) if os.path.isfile(os.path.join(resource_dir, f))]
                    extracted_resources[resource_type] = files
                except:
                    extracted_resources[resource_type] = []
    
    total_existing = sum(len(files) for files in extracted_resources.values())
    print(f"✅ Found {total_existing} existing resource files")
    return extracted_resources

def _validate_critical_files(output_dir, extracted_resources):
    """Validate that critical EPUB files were extracted"""
    total_extracted = sum(len(files) for files in extracted_resources.values())
    print(f"✅ Extracted {total_extracted} resource files:")
    
    for resource_type, files in extracted_resources.items():
        if files:
            if resource_type == 'epub_structure':
                print(f"   • EPUB Structure: {len(files)} files")
                for file in files:
                    print(f"     - {file}")
            else:
                print(f"   • {resource_type.title()}: {len(files)} files")
    
    # Validate critical files
    critical_files = ['container.xml']
    missing_critical = [f for f in critical_files if not os.path.exists(os.path.join(output_dir, f))]
    
    if missing_critical:
        print(f"⚠️ WARNING: Missing critical EPUB files: {missing_critical}")
        print("   This may prevent proper EPUB reconstruction!")
    else:
        print("✅ All critical EPUB structure files extracted successfully")
    
    # Check for OPF file
    opf_files = [f for f in extracted_resources['epub_structure'] if f.lower().endswith('.opf')]
    if not opf_files:
        print("⚠️ WARNING: No OPF file found! This will prevent EPUB reconstruction.")
    else:
        print(f"✅ Found OPF file(s): {opf_files}")

def _extract_advanced_chapter_info(zf):
    """Extract comprehensive chapter information with improved detection"""
    chapters = []
    
    # Get all potential document files
    html_files = []
    for name in zf.namelist():
        if name.lower().endswith(('.xhtml', '.html', '.htm')):
            # Skip obvious non-content files
            lower_name = name.lower()
            if any(skip in lower_name for skip in [
                'nav', 'toc', 'contents', 'cover', 'title', 'index',
                'copyright', 'acknowledgment', 'dedication'
            ]):
                continue
            html_files.append(name)
    
    print(f"📚 Found {len(html_files)} potential content files")
    
    # Enhanced chapter detection patterns
    chapter_patterns = [
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
    
    # Chinese number conversion table
    chinese_nums = {
        '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
        '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
        '十一': 11, '十二': 12, '十三': 13, '十四': 14, '十五': 15,
        '十六': 16, '十七': 17, '十八': 18, '十九': 19, '二十': 20,
        '二十一': 21, '二十二': 22, '二十三': 23, '二十四': 24, '二十五': 25,
        '三十': 30, '四十': 40, '五十': 50, '六十': 60,
        '七十': 70, '八十': 80, '九十': 90, '百': 100,
    }
    
    def convert_chinese_number(cn_num):
        """Convert Chinese number to integer"""
        if cn_num in chinese_nums:
            return chinese_nums[cn_num]
        
        if '十' in cn_num:
            parts = cn_num.split('十')
            if len(parts) == 2:
                tens = chinese_nums.get(parts[0], 1) if parts[0] else 1
                ones = chinese_nums.get(parts[1], 0) if parts[1] else 0
                return tens * 10 + ones
        
        return None
    
    # Content analysis for language detection
    sample_texts = []
    
    # Track duplicates more comprehensively
    content_hashes = {}
    seen_chapters = {}
    file_size_groups = {}
    
    print("🔍 Analyzing content files for chapters...")
    
    for idx, file_path in enumerate(sorted(html_files)):
        try:
            file_data = zf.read(file_path)
            
            # Try multiple encodings
            html_content = None
            for encoding in ['utf-8', 'utf-16', 'gb18030', 'shift_jis', 'euc-kr']:
                try:
                    html_content = file_data.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if not html_content:
                print(f"[WARNING] Could not decode {file_path}")
                continue
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract content for analysis
            if soup.body:
                content_html = soup.body.decode_contents()
                content_text = soup.body.get_text(strip=True)
            else:
                content_html = str(soup)
                content_text = soup.get_text(strip=True)
            
            # Skip very short files
            if len(content_text.strip()) < 200:
                print(f"[DEBUG] Skipping short file: {file_path} ({len(content_text)} chars)")
                continue
            
            # Create comprehensive content hash
            content_hash = extract_comprehensive_content_hash(content_html)
            
            # Group files by size
            file_size = len(content_text)
            if file_size not in file_size_groups:
                file_size_groups[file_size] = []
            file_size_groups[file_size].append(file_path)
            
            # Check for exact duplicates
            if content_hash in content_hashes:
                duplicate_info = content_hashes[content_hash]
                print(f"[DEBUG] Skipping duplicate content: {file_path} (matches {duplicate_info['filename']})")
                continue
            
            # Collect sample text for language detection
            if len(sample_texts) < 5:
                sample_texts.append(content_text[:1000])
            
            # Try to extract chapter number from various sources
            chapter_num = None
            chapter_title = None
            detection_method = None
            
            # Method 1: Check filename
            for pattern, flags, method in chapter_patterns:
                if method.endswith('_number'):
                    match = re.search(pattern, file_path, flags)
                else:
                    continue
                
                if match:
                    try:
                        num_str = match.group(1)
                        if num_str.isdigit():
                            chapter_num = int(num_str)
                            detection_method = f"filename_{method}"
                            break
                        elif method == 'chinese_chapter_cn':
                            converted = convert_chinese_number(num_str)
                            if converted:
                                chapter_num = converted
                                detection_method = f"filename_{method}"
                                break
                    except (ValueError, IndexError):
                        continue
            
            # Method 2: Check content headers and title
            if not chapter_num:
                # Look in title tag first
                if soup.title and soup.title.string:
                    title_text = soup.title.string.strip()
                    for pattern, flags, method in chapter_patterns:
                        if method.endswith('_number'):
                            continue
                        match = re.search(pattern, title_text, flags)
                        if match:
                            try:
                                num_str = match.group(1)
                                if num_str.isdigit():
                                    chapter_num = int(num_str)
                                    chapter_title = title_text
                                    detection_method = f"title_{method}"
                                    break
                                elif method == 'chinese_chapter_cn':
                                    converted = convert_chinese_number(num_str)
                                    if converted:
                                        chapter_num = converted
                                        chapter_title = title_text
                                        detection_method = f"title_{method}"
                                        break
                            except (ValueError, IndexError):
                                continue
                        if chapter_num:
                            break
                
                # Look in headers if not found in title
                if not chapter_num:
                    for header_tag in ['h1', 'h2', 'h3']:
                        headers = soup.find_all(header_tag)
                        for header in headers:
                            header_text = header.get_text(strip=True)
                            if not header_text:
                                continue
                            
                            for pattern, flags, method in chapter_patterns:
                                if method.endswith('_number'):
                                    continue
                                match = re.search(pattern, header_text, flags)
                                if match:
                                    try:
                                        num_str = match.group(1)
                                        if num_str.isdigit():
                                            chapter_num = int(num_str)
                                            chapter_title = header_text
                                            detection_method = f"header_{method}"
                                            break
                                        elif method == 'chinese_chapter_cn':
                                            converted = convert_chinese_number(num_str)
                                            if converted:
                                                chapter_num = converted
                                                chapter_title = header_text
                                                detection_method = f"header_{method}"
                                                break
                                    except (ValueError, IndexError):
                                        continue
                            
                            if chapter_num:
                                break
                        if chapter_num:
                            break
            
            # Method 3: Check first few paragraphs
            if not chapter_num:
                first_elements = soup.find_all(['p', 'div'])[:5]
                for elem in first_elements:
                    elem_text = elem.get_text(strip=True)
                    if not elem_text:
                        continue
                    
                    for pattern, flags, method in chapter_patterns:
                        if method.endswith('_number'):
                            continue
                        match = re.search(pattern, elem_text, flags)
                        if match:
                            try:
                                num_str = match.group(1)
                                if num_str.isdigit():
                                    chapter_num = int(num_str)
                                    detection_method = f"content_{method}"
                                    break
                                elif method == 'chinese_chapter_cn':
                                    converted = convert_chinese_number(num_str)
                                    if converted:
                                        chapter_num = converted
                                        detection_method = f"content_{method}"
                                        break
                            except (ValueError, IndexError):
                                continue
                    
                    if chapter_num:
                        break
            
            # Fallback: Assign sequential number
            if not chapter_num:
                chapter_num = len(chapters) + 1
                while chapter_num in seen_chapters:
                    chapter_num += 1
                detection_method = "sequential_fallback"
                print(f"[DEBUG] No chapter number found in {file_path}, assigning: {chapter_num}")
            
            # Handle duplicate chapter numbers
            if chapter_num in seen_chapters:
                existing_info = seen_chapters[chapter_num]
                existing_hash = existing_info['content_hash']
                
                if existing_hash != content_hash:
                    # Different content with same chapter number
                    original_num = chapter_num
                    while chapter_num in seen_chapters:
                        chapter_num += 1
                    print(f"[WARNING] Chapter {original_num} already exists with different content")
                    print(f"[INFO] Reassigning {file_path} to chapter {chapter_num}")
                    detection_method += "_reassigned"
                else:
                    # Same content, skip
                    print(f"[DEBUG] Skipping duplicate chapter {chapter_num}: {file_path}")
                    continue
            
            # Extract or generate title
            if not chapter_title:
                if soup.title and soup.title.string:
                    chapter_title = soup.title.string.strip()
                else:
                    # Look for first header
                    for header_tag in ['h1', 'h2', 'h3']:
                        header = soup.find(header_tag)
                        if header:
                            chapter_title = header.get_text(strip=True)
                            break
                
                # Fallback title
                if not chapter_title:
                    chapter_title = f"Chapter {chapter_num}"
            
            # Clean and validate title
            chapter_title = re.sub(r'\s+', ' ', chapter_title).strip()
            if len(chapter_title) > 150:
                chapter_title = chapter_title[:147] + "..."
            
            # Store chapter information
            chapter_info = {
                "num": chapter_num,
                "title": chapter_title,
                "body": content_html,
                "filename": file_path,
                "content_hash": content_hash,
                "detection_method": detection_method,
                "file_size": file_size,
                "language_sample": content_text[:500]
            }
            
            chapters.append(chapter_info)
            
            # Update tracking
            content_hashes[content_hash] = {
                'filename': file_path,
                'chapter_num': chapter_num
            }
            seen_chapters[chapter_num] = {
                'content_hash': content_hash,
                'filename': file_path
            }
            
            print(f"[DEBUG] ✅ Chapter {chapter_num}: {chapter_title[:50]}... ({detection_method})")
            
        except Exception as e:
            print(f"[ERROR] Failed to process {file_path}: {e}")
            continue
    
    # Sort chapters by number
    chapters.sort(key=lambda x: x["num"])
    
    # Detect primary language
    combined_sample = ' '.join(sample_texts)
    detected_language = detect_content_language(combined_sample)
    
    # Final validation and reporting
    if chapters:
        print(f"\n📊 Chapter Extraction Summary:")
        print(f"   • Total chapters extracted: {len(chapters)}")
        print(f"   • Chapter range: {chapters[0]['num']} to {chapters[-1]['num']}")
        print(f"   • Detected language: {detected_language}")
        
        # Check for missing chapters
        expected_chapters = set(range(chapters[0]['num'], chapters[-1]['num'] + 1))
        actual_chapters = set(c['num'] for c in chapters)
        missing = expected_chapters - actual_chapters
        if missing:
            print(f"   ⚠️ Missing chapter numbers: {sorted(missing)}")
        
        # Show detection method statistics
        method_stats = Counter(c['detection_method'] for c in chapters)
        print(f"   📈 Detection methods used:")
        for method, count in method_stats.most_common():
            print(f"      • {method}: {count} chapters")
        
        # Show duplicate file size groups
        large_groups = [size for size, files in file_size_groups.items() if len(files) > 1]
        if large_groups:
            print(f"   ⚠️ Found {len(large_groups)} file size groups with potential duplicates")
    
    return chapters, detected_language

def _create_extraction_report(output_dir, metadata, chapters, extracted_resources):
    """Create comprehensive extraction report"""
    report_path = os.path.join(output_dir, 'extraction_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("EPUB Extraction Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("METADATA:\n")
        for key, value in metadata.items():
            if key not in ['chapter_titles', 'extracted_resources']:
                f.write(f"  {key}: {value}\n")
        
        f.write(f"\nCHAPTERS ({len(chapters)}):\n")
        for chapter in chapters:
            f.write(f"  {chapter['num']:3d}. {chapter['title']} ({chapter['detection_method']})\n")
        
        f.write(f"\nRESOURCES EXTRACTED:\n")
        for resource_type, files in extracted_resources.items():
            if files:
                if resource_type == 'epub_structure':
                    f.write(f"  EPUB Structure: {len(files)} files\n")
                    for file in files:
                        f.write(f"    - {file}\n")
                else:
                    f.write(f"  {resource_type.title()}: {len(files)} files\n")
                    for file in files[:5]:  # Show first 5
                        f.write(f"    - {file}\n")
                    if len(files) > 5:
                        f.write(f"    ... and {len(files) - 5} more\n")
    
    print(f"📄 Saved extraction report to: {report_path}")

def _log_extraction_summary(chapters, extracted_resources, detected_language):
    """Log final extraction summary"""
    print(f"\n✅ Comprehensive extraction complete!")
    print(f"   📚 Chapters: {len(chapters)}")
    print(f"   🎨 Resources: {sum(len(files) for files in extracted_resources.values())}")
    print(f"   🌍 Language: {detected_language}")
    
    # Show EPUB structure file status
    epub_files = extracted_resources.get('epub_structure', [])
    if epub_files:
        print(f"   📋 EPUB Structure: {len(epub_files)} files ({', '.join(epub_files)})")
    else:
        print(f"   ⚠️ No EPUB structure files extracted!")

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_chapter_continuity(chapters):
    """Validate chapter continuity and warn about issues"""
    if not chapters:
        return
    
    issues = []
    
    # Check for duplicate chapter numbers
    chapter_nums = [c['num'] for c in chapters]
    duplicates = [num for num in chapter_nums if chapter_nums.count(num) > 1]
    if duplicates:
        issues.append(f"Duplicate chapter numbers found: {set(duplicates)}")
    
    # Check for missing chapters
    min_num = min(chapter_nums)
    max_num = max(chapter_nums)
    expected = set(range(min_num, max_num + 1))
    actual = set(chapter_nums)
    missing = expected - actual
    if missing:
        issues.append(f"Missing chapter numbers: {sorted(missing)}")
    
    # Check for suspiciously similar titles
    for i in range(len(chapters) - 1):
        for j in range(i + 1, len(chapters)):
            title1 = chapters[i]['title'].lower()
            title2 = chapters[j]['title'].lower()
            if title1 == title2 and chapters[i]['num'] != chapters[j]['num']:
                issues.append(f"Chapters {chapters[i]['num']} and {chapters[j]['num']} have identical titles")
    
    if issues:
        print("\n⚠️  Chapter Validation Issues:")
        for issue in issues:
            print(f"  - {issue}")
        print()

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
    
    # Check for container.xml
    container_path = os.path.join(output_dir, 'container.xml')
    if os.path.exists(container_path):
        found_files['container.xml'] = 'Found'
        print("   ✅ container.xml - Found")
    else:
        missing_files.append('container.xml')
        print("   ❌ container.xml - Missing (CRITICAL)")
    
    # Check for OPF files
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
    
    # Check for translated HTML files
    html_files = [f for f in os.listdir(output_dir) if f.lower().endswith('.html') and f.startswith('response_')]
    if html_files:
        print(f"   ✅ Translated chapters - Found: {len(html_files)} files")
    else:
        print("   ⚠️ No translated chapter files found")
    
    # Overall status
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
    
    # Check structure files
    if not validate_epub_structure(output_dir):
        issues.append("Missing critical EPUB structure files")
    
    # Check for translated content
    html_files = [f for f in os.listdir(output_dir) if f.lower().endswith('.html') and f.startswith('response_')]
    if not html_files:
        issues.append("No translated chapter files found")
    else:
        print(f"   ✅ Found {len(html_files)} translated chapters")
    
    # Check for metadata
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
    
    # Check for resources
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
    
    # Final assessment
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
    """Clean up any files from previous extraction runs"""
    cleanup_items = [
        # Resource directories
        'css', 'fonts', 'images',
        # Extraction marker
        '.resources_extracted'
    ]
    
    # Also clean up EPUB structure files
    epub_structure_files = [
        'container.xml', 'content.opf', 'toc.ncx'
    ]
    
    cleaned_count = 0
    
    # Clean up directories
    for item in cleanup_items:
        if item.startswith('.'):  # Skip marker files for now
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
    
    # Clean up any other .opf or .ncx files with different names
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
    
    # Clean up extraction marker
    marker_path = os.path.join(output_dir, '.resources_extracted')
    try:
        if os.path.isfile(marker_path):
            os.remove(marker_path)
            print(f"🧹 Removed extraction marker")
            cleaned_count += 1
    except Exception as e:
        print(f"⚠️ Could not remove extraction marker: {e}")
    
    if cleaned_count > 0:
        print(f"🧹 Cleaned up {cleaned_count} items from previous runs")
    
    return cleaned_count

# =============================================================================
# GLOSSARY MANAGEMENT
# =============================================================================
def save_glossary(output_dir, chapters, instructions, language="korean"):
    """
    automatic glossary generator - works for ANY language
    Finds character names by looking for words that appear with honorifics
    PRESERVED ALL ORIGINAL FUNCTIONALITY
    """
    
    print("📑 Automatic Glossary Generator v2.0")
    print("📑 Language-agnostic character name detection")
    
    # Load settings
    min_frequency = int(os.getenv("GLOSSARY_MIN_FREQUENCY", "2"))
    max_names = int(os.getenv("GLOSSARY_MAX_NAMES", "50"))
    max_suffixes = int(os.getenv("GLOSSARY_MAX_SUFFIXES", "50"))
    batch_size = int(os.getenv("GLOSSARY_BATCH_SIZE", "50"))
    
    print(f"📑 Min frequency: {min_frequency}, Max names: {max_names}, Max suffixes: {max_suffixes}")
    
    def clean_html(html_text):
        """Remove HTML tags to get clean text"""
        soup = BeautifulSoup(html_text, 'html.parser')
        return soup.get_text()
    
    # Extract and combine all text from chapters
    all_text = ' '.join(clean_html(chapter["body"]) for chapter in chapters)
    print(f"📑 Processing {len(all_text):,} characters of text")
    
    # automatic honorifics from all languages
    UNIVERSAL_HONORIFICS = [
        # Korean honorifics and titles
        '님', '씨', '선배', '형', '누나', '언니', '오빠', '선생님', '교수님', 
        '공주', '왕자', '폐하', '전하', '각하', '원님', '대감', '영감',
        
        # Japanese honorifics and titles
        'さん', 'ちゃん', '君', 'くん', '様', 'さま', '先生', 'せんせい',
        '殿', 'どの', '姫', 'ひめ', '王', '皇', '陛下', '閣下', '大人',
        
        # Chinese honorifics and titles
        '公子', '小姐', '夫人', '先生', '大人', '师父', '师傅', '老师',
        '陛下', '殿下', '王爷', '公主', '皇上', '将军', '大师', '长老',
        
        # English honorifics (for mixed content)
        '-san', '-chan', '-kun', '-sama', '-sensei', '-senpai'
    ]
    
    # Universal exclusion list
    UNIVERSAL_EXCLUSIONS = {
        # Korean common words and particles
        '나', '너', '그', '이', '저', '것', '수', '때', '곳', '중', '안', '밖', '위', '아래', '앞', '뒤',
        '나는', '나를', '나의', '나도', '나만', '나조차', '나부터', '나까지',
        '너는', '너를', '너의', '너도', '너만', '그는', '그를', '그의', '그도',
        '그것', '이것', '저것', '무엇', '어떤', '모든', '각각', '서로', '함께', '혼자',
        '같은', '다른', '새로운', '오래된', '많은', '적은', '좋은', '나쁜', '큰', '작은',
        '지금', '오늘', '내일', '어제', '이제', '그때', '언제', '항상', '가끔', '자주',
        '여기', '거기', '저기', '어디', '어디서', '어디로', '어디까지', '어디든',
        '하는', '되는', '있는', '없는', '하고', '되고', '있고', '없고', '해서', '돼서',
        '때문', '위해', '대해', '통해', '의해', '로서', '로써', '부터', '까지', '마다',
        
        # Japanese common words and particles  
        'それ', 'これ', 'あれ', 'どれ', 'その', 'この', 'あの', 'どの',
        'そう', 'こう', 'ああ', 'どう', 'そこ', 'ここ', 'あそこ', 'どこ',
        'みんな', 'だれ', 'なに', 'なん', 'いつ', 'なぜ', 'どんな', 'いろいろ',
        'とても', 'すごく', 'ちょっと', 'すこし', 'たくさん', 'いっぱい', 'もっと',
        'もう', 'まだ', 'きっと', 'たぶん', 'やっぱり', 'やはり', 'ほんとう',
        'ひと', 'もの', 'こと', 'とき', 'ところ', 'ほう', 'よう', 'ふう', 'やつ', 'かた',
        'です', 'ます', 'した', 'する', 'いる', 'ある', 'ない', 'から', 'まで', 'など',
        
        # Chinese common words
        '那个', '这个', '什么', '哪里', '怎么', '为什么', '什么时候', '多少', '怎样', '如何',
        '所有', '每个', '任何', '没有', '一些', '很多', '一点', '非常', '特别', '真的',
        '可能', '应该', '必须', '需要', '想要', '喜欢', '觉得', '认为', '知道', '明白',
        
        # English common words (for mixed content)
        'that', 'this', 'what', 'where', 'when', 'how', 'why', 'who', 'which',
        'some', 'many', 'very', 'good', 'bad', 'big', 'small', 'new', 'old',
        'here', 'there', 'now', 'then', 'yes', 'no', 'and', 'but', 'or',
        
        # Single characters and particles (usually not names)
        'a', 'i', 'u', 'e', 'o', 'の', 'は', 'が', 'を', 'に', 'で', 'と', 'も', 'か', 'よ', 'ね',
        '은', '는', '이', '가', '을', '를', '에', '의', '와', '과', '도', '만', '부터', '까지',
        '了', '的', '在', '是', '有', '不', '我', '你', '他', '她', '它', '们', '也', '都'
    }
    
    print(f"📑 Using {len(UNIVERSAL_HONORIFICS)} honorifics from all languages")
    print(f"📑 Excluding {len(UNIVERSAL_EXCLUSIONS)} common words")
    
    JAPANESE_EXCLUSIONS = {
        # Common Japanese particles and words
        'これ', 'それ', 'あれ', 'この', 'その', 'あの', 'こんな', 'そんな', 'あんな',
        'ここ', 'そこ', 'あそこ', 'こちら', 'そちら', 'あちら', 'こっち', 'そっち', 'あっち',
        'いつ', 'いま', 'きょう', 'あした', 'きのう', 'まえ', 'あと', 'つぎ',
        'みんな', 'だれ', 'なに', 'どこ', 'どう', 'なぜ', 'いくつ', 'いくら',
        'ひと', 'もの', 'こと', 'とき', 'ところ', 'かた', 'ほう',
        'わたし', 'あなた', 'かれ', 'かのじょ', 'われわれ', 'かれら',
        'じぶん', 'みずから', 'おれ', 'ぼく', 'きみ', 'おまえ',
        # Common kanji words
        '今日', '明日', '昨日', '今', '後', '前', '時', '人', '事', '物',
        '所', '方', '何', '誰', '私', '僕', '君', '彼', '彼女', '自分',
        '皆', '全部', '少し', '沢山', '大体', '普通', '特別', '本当',
        # Common non-name kanji combinations
        '時間', '場所', '毎日', '最近', '大切', '必要', '可能', '簡単', 
        '困難', '重要', '部屋', '学校', '会社', '仕事', '友達', '家族'
    }
    
    # Combine exclusions - FIXED: union() works with both sets and lists
    ALL_EXCLUSIONS = set(UNIVERSAL_EXCLUSIONS).union(set(JAPANESE_EXCLUSIONS))

    # Then replace the entire find_names_with_honorifics function:
    def find_names_with_honorifics(text, honorifics, exclusions, min_freq):
        """Core algorithm: Find names that appear with honorifics - ENHANCED for Japanese"""
        names_with_honorifics = []
        standalone_names = set()
        
        print("📑 Scanning for name + honorific combinations...")
        
        # Detect if text contains Japanese
        japanese_chars = sum(1 for char in text if 
                           (12352 <= ord(char) <= 12447) or  # Hiragana
                           (12448 <= ord(char) <= 12543) or  # Katakana
                           (19968 <= ord(char) <= 40959))    # Kanji
        
        total_chars = len(text)
        is_japanese = (japanese_chars / total_chars > 0.1) if total_chars > 0 else False
        
        if is_japanese:
            print("📑 Detected Japanese text - using enhanced detection")
        
        # Process each honorific
        for honorific in honorifics:
            honorific_count = 0
            found_names = set()  # Avoid duplicates per honorific
            
            # Japanese honorifics need special handling
            if honorific in ['さん', 'ちゃん', '君', 'くん', '様', 'さま', '先生', 'せんせい', 
                           '殿', 'どの', '姫', 'ひめ', '陛下', '閣下', '-san', '-chan', 
                           '-kun', '-sama', '-sensei', '-senpai']:
                
                # Search for honorific in text
                search_pos = 0
                while True:
                    pos = text.find(honorific, search_pos)
                    if pos == -1:
                        break
                    
                    # Extract name before honorific
                    name_chars = []
                    char_pos = pos - 1
                    
                    # Scan backwards to find name
                    while char_pos >= 0 and len(name_chars) < 6:  # Max 6 chars for safety
                        char = text[char_pos]
                        char_code = ord(char)
                        
                        # Check if character could be part of a name
                        if (12352 <= char_code <= 12447 or    # Hiragana
                            12448 <= char_code <= 12543 or    # Katakana  
                            19968 <= char_code <= 40959 or    # Kanji
                            (char.isalpha() and char_code < 128)):  # ASCII letters
                            name_chars.insert(0, char)
                            char_pos -= 1
                        # Check for Japanese punctuation and spaces
                        elif char in ' ' or char == '　' or char in '、。「」『』・':
                            break
                        else:
                            break
                    
                    # Process found name
                    if name_chars:
                        potential_name = ''.join(name_chars)
                        
                        if is_valid_name(potential_name, exclusions) and potential_name not in found_names:
                            full_combination = potential_name + honorific
                            count = text.count(full_combination)
                            
                            if count >= min_freq:
                                names_with_honorifics.append(full_combination)
                                standalone_names.add(potential_name)
                                found_names.add(potential_name)
                                honorific_count += 1
                                
                                if honorific_count <= 10:  # Show first 10
                                    print("   ✅ {} + {} ({}x)".format(potential_name, honorific, count))
                    
                    search_pos = pos + len(honorific)
            
            else:
                # Non-Japanese honorifics - use word splitting
                words = text.split()
                for word in words:
                    if word.endswith(honorific) and len(word) > len(honorific):
                        potential_name = word[:-len(honorific)]
                        
                        if is_valid_name(potential_name, exclusions):
                            full_combination = potential_name + honorific
                            count = text.count(full_combination)
                            
                            if count >= min_freq:
                                names_with_honorifics.append(full_combination)
                                standalone_names.add(potential_name)
                                honorific_count += 1
                                print("   ✅ {} + {} ({}x)".format(potential_name, honorific, count))
            
            if honorific_count > 0:
                print("   📊 Found {} unique names with '{}'".format(honorific_count, honorific))
        
        # Additional pattern detection for Japanese names
        if is_japanese and len(standalone_names) < 10:  # If we found few names
            print("📑 Running additional Japanese name pattern detection...")
            
            # Common name kanji patterns
            name_patterns = [
                # Family name endings
                ('田', 1, 3), ('山', 1, 3), ('川', 1, 3), ('村', 1, 3), 
                ('木', 1, 3), ('野', 1, 3), ('原', 1, 3), ('藤', 1, 3),
                ('井', 1, 3), ('本', 1, 3), ('谷', 1, 3), ('口', 1, 3),
                # Given name patterns  
                ('子', 1, 2), ('美', 1, 2), ('郎', 1, 2), ('太', 1, 2)
            ]
            
            additional_names = set()
            for pattern_char, min_prefix, max_prefix in name_patterns:
                search_pos = 0
                while True:
                    pos = text.find(pattern_char, search_pos)
                    if pos == -1:
                        break
                    
                    # Try different name lengths
                    for prefix_len in range(min_prefix, max_prefix + 1):
                        if pos >= prefix_len:
                            candidate = text[pos - prefix_len:pos + 1]
                            
                            # Validate it's all kanji
                            if all(19968 <= ord(c) <= 40959 for c in candidate):
                                if is_valid_name(candidate, exclusions):
                                    count = text.count(candidate)
                                    if count >= min_freq:
                                        additional_names.add(candidate)
                    
                    search_pos = pos + 1
            
            # Add additional names found
            for name in additional_names:
                if name not in standalone_names:
                    standalone_names.add(name)
                    print("   ✅ {} (pattern match, {}x)".format(name, text.count(name)))
        
        return list(set(names_with_honorifics)), standalone_names

    # And update is_valid_name function:
    def is_valid_name(name, exclusions):
        """Check if a potential name meets our criteria - ENHANCED for Japanese"""
        if not name or not name.strip():
            return False
        
        name = name.strip()
        
        # Check against all exclusions
        if name in exclusions:
            return False
        
        # Length validation based on script
        name_length = len(name)
        contains_kanji = any(19968 <= ord(char) <= 40959 for char in name)
        contains_kana = any((12352 <= ord(char) <= 12447) or 
                           (12448 <= ord(char) <= 12543) for char in name)
        
        if contains_kanji or contains_kana:
            # Japanese names: 1-4 characters typical
            if not (1 <= name_length <= 5):
                return False
            
            # Single character validation for Japanese
            if name_length == 1:
                # Common single kanji that aren't names
                single_exclusions = {'人', '日', '月', '年', '時', '分', '今', '中', 
                                   '上', '下', '左', '右', '前', '後', '内', '外'}
                if name in single_exclusions:
                    return False
        else:
            # Non-Japanese names: 2-12 characters
            if not (2 <= name_length <= 12):
                return False
        
        # Reject pure numbers
        if name.isdigit():
            return False
        
        # Reject if too many non-letter characters
        letter_count = sum(1 for c in name if c.isalpha() or 19968 <= ord(c) <= 40959)
        if letter_count < name_length * 0.8:
            return False
        
        return True
    
    # Make sure to use ALL_EXCLUSIONS when calling the function:
    names_with_honorifics, standalone_names = find_names_with_honorifics(
        all_text, UNIVERSAL_HONORIFICS, ALL_EXCLUSIONS, min_frequency
    )
    
    # Language detection for translation
    def detect_primary_language(text):
        """Simple language detection for translation purposes"""
        korean_chars = sum(1 for char in text if 44032 <= ord(char) <= 55215)
        japanese_chars = sum(1 for char in text if 12352 <= ord(char) <= 12543)
        chinese_chars = sum(1 for char in text if 19968 <= ord(char) <= 40959)
        
        if korean_chars > japanese_chars and korean_chars > chinese_chars:
            return "korean"
        elif japanese_chars > chinese_chars:
            return "japanese"
        elif chinese_chars > 0:
            return "chinese"
        else:
            return "unknown"
    
    detected_language = detect_primary_language(all_text)
    print(f"\n📑 Auto-detected language for translation: {detected_language}")
    all_terms = final_standalone_names + final_names_with_honorifics
    
    print(f"📑 Translating {len(all_terms)} character names...")
    
    # Translate all terms
    translations = translate_terms_batch(all_terms, detected_language, batch_size)
    
    # Build final glossary
    glossary_entries = {}
    for term in all_terms:
        if term in translations and translations[term] != term:
            glossary_entries[term] = translations[term]
        else:
            glossary_entries[term] = term  # Keep original if translation failed
    
    # Save glossary
    glossary_path = os.path.join(output_dir, "glossary.json")
    with open(glossary_path, 'w', encoding='utf-8') as f:
        json.dump(glossary_entries, f, ensure_ascii=False, indent=2)
    
    print(f"\n📑 ✅ AUTOMATIC GLOSSARY SAVED!")
    print(f"📑 File: {glossary_path}")
    print(f"📑 Entries: {len(glossary_entries)}")
    
    # Final display
    if glossary_entries:
        print(f"\n📑 Final glossary preview:")
        sample_count = min(15, len(glossary_entries))
        for orig, trans in list(glossary_entries.items())[:sample_count]:
            print(f"   • {orig} → {trans}")
        
        if len(glossary_entries) > sample_count:
            print(f"   ... and {len(glossary_entries) - sample_count} more entries")
    
    print(f"\n📑 🌍 AUTOMATIC detection complete!")
    print(f"📑 Works for Korean, Japanese, Chinese, and mixed content!")

def translate_terms_batch(term_list, source_lang, batch_size=50):
    """IMPROVED: Use GUI-controlled batch size for translation"""
    if not term_list or os.getenv("DISABLE_GLOSSARY_TRANSLATION", "0") == "1":
        print(f"📑 Glossary translation disabled or no terms to translate")
        return {term: term for term in term_list}
    
    try:
        # Get API settings
        MODEL = os.getenv("MODEL", "gemini-1.5-flash")
        API_KEY = (os.getenv("API_KEY") or 
                   os.getenv("OPENAI_API_KEY") or 
                   os.getenv("OPENAI_OR_Gemini_API_KEY") or
                   os.getenv("GEMINI_API_KEY"))
        
        if not API_KEY:
            print(f"📑 No API key found, skipping translation")
            return {term: term for term in term_list}
        
        print(f"📑 Translating {len(term_list)} {source_lang} terms to English using batch size {batch_size}...")
        
        client = UnifiedClient(model=MODEL, api_key=API_KEY)
        
        all_translations = {}
        
        # Process all terms in batches using GUI-controlled batch size
        for i in range(0, len(term_list), batch_size):
            batch = term_list[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(term_list) + batch_size - 1) // batch_size
            
            print(f"📑 Processing batch {batch_num}/{total_batches} ({len(batch)} terms)...")
            
            # Create a simple numbered list
            terms_text = ""
            for idx, term in enumerate(batch, 1):
                terms_text += f"{idx}. {term}\n"
            
            # More focused prompt for names/terms only
            system_prompt = (
                f"You are translating {source_lang} character names and important terms to English. "
                "For character names, provide English transliterations or keep as romanized. "
                "Retain honorifics/suffixes in romaji. "
                "Keep the same number format in your response."
            )
            
            user_prompt = (
                f"Translate these {source_lang} character names and terms to English:\n\n"
                f"{terms_text}\n"
                "Respond with the same numbered format. For names, provide transliteration or keep romanized."
            )
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            try:
                # Single attempt per batch - no retries for glossary
                response, _ = client.send(messages, temperature=0.1, max_tokens=4096)
                
                # Parse the response
                batch_translations = parse_translation_response(response, batch)
                all_translations.update(batch_translations)
                
                print(f"📑 Batch {batch_num} completed: {len(batch_translations)} translations")
                
                # Short delay between batches
                if i + batch_size < len(term_list):
                    time.sleep(0.5)
                    
            except Exception as e:
                print(f"⚠️ Translation failed for batch {batch_num}: {e}")
                # Add untranslated terms
                for term in batch:
                    all_translations[term] = term
        
        # Ensure all terms are included
        for term in term_list:
            if term not in all_translations:
                all_translations[term] = term
        
        # Count successful translations
        translated_count = sum(1 for term, translation in all_translations.items() 
                             if translation != term and translation.strip())
        
        print(f"📑 Successfully translated {translated_count}/{len(term_list)} terms")
        return all_translations
        
    except Exception as e:
        print(f"⚠️ Glossary translation failed: {e}")
        return {term: term for term in term_list}

def parse_translation_response(response, original_terms):
    """Parse translation response - handles numbered format"""
    translations = {}
    lines = response.strip().split('\n')
    
    # Look for numbered format (1. term -> translation)
    for line in lines:
        line = line.strip()
        if not line or not line[0].isdigit():
            continue
            
        try:
            # Extract number and content
            number_match = re.match(r'^(\d+)\.?\s*(.+)', line)
            if number_match:
                num = int(number_match.group(1)) - 1  # Convert to 0-based index
                content = number_match.group(2).strip()
                
                if 0 <= num < len(original_terms):
                    original_term = original_terms[num]
                    
                    # Try to extract translation from various formats
                    for separator in ['->', '→', ':', '-', '—', '=']:
                        if separator in content:
                            parts = content.split(separator, 1)
                            if len(parts) == 2:
                                translation = parts[1].strip()
                                # Clean up the translation
                                translation = translation.strip('"\'()[]')
                                if translation and translation != original_term:
                                    translations[original_term] = translation
                                    break
                    else:
                        # No separator found, treat whole content as translation
                        if content != original_term:
                            translations[original_term] = content
                            
        except (ValueError, IndexError):
            continue
    
    return translations

      
# =============================================================================
# API AND TRANSLATION UTILITIES
# =============================================================================

def send_with_interrupt(messages, client, temperature, max_tokens, stop_check_fn):
    """Send API request with interrupt capability"""
    result_queue = queue.Queue()
    
    def api_call():
        try:
            result = client.send(messages, temperature=temperature, max_tokens=max_tokens)
            result_queue.put(result)
        except Exception as e:
            result_queue.put(e)
    
    api_thread = threading.Thread(target=api_call)
    api_thread.daemon = True
    api_thread.start()
    
    # Check for stop every 0.5 seconds while waiting for API
    timeout = 300  # 5 minute total timeout
    check_interval = 0.5
    elapsed = 0
    
    while elapsed < timeout:
        try:
            result = result_queue.get(timeout=check_interval)
            if isinstance(result, Exception):
                raise result
            return result
        except queue.Empty:
            if stop_check_fn():
                raise UnifiedClientError("Translation stopped by user")
            elapsed += check_interval
    
    raise UnifiedClientError(f"API call timed out after {timeout} seconds")

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
    
    # Default fallback
    return 1000000, "1000000 (default)"

def build_system_prompt(user_prompt, glossary_path, instructions):
    """Build the system prompt with glossary"""
    # Check if we should append glossary (default is True if not set)
    append_glossary = os.getenv("APPEND_GLOSSARY", "1") == "1"
    
    # Check if system prompt is disabled
    if os.getenv("DISABLE_SYSTEM_PROMPT", "0") == "1":
        # Use only user prompt, but still append glossary if enabled
        system = user_prompt if user_prompt else ""
        
        # Append glossary if the toggle is on
        if append_glossary and os.path.exists(glossary_path):
            with open(glossary_path, "r", encoding="utf-8") as gf:
                entries = json.load(gf)
            glossary_block = json.dumps(entries, ensure_ascii=False, indent=2)
            if system:
                system += "\n\n"
            system += (
                "Use the following glossary entries exactly as given:\n"
                f"{glossary_block}"
            )
        
        return system
    
    # Normal flow when hardcoded prompts are enabled
    if user_prompt:
        system = user_prompt
        # Append glossary if the toggle is on
        if append_glossary and os.path.exists(glossary_path):
            with open(glossary_path, "r", encoding="utf-8") as gf:
                entries = json.load(gf)
            glossary_block = json.dumps(entries, ensure_ascii=False, indent=2)
            system += (
                "\n\nUse the following glossary entries exactly as given:\n"
                f"{glossary_block}"
            )
            
    elif os.path.exists(glossary_path) and append_glossary:
        with open(glossary_path, "r", encoding="utf-8") as gf:
            entries = json.load(gf)
        glossary_block = json.dumps(entries, ensure_ascii=False, indent=2)
        system = (
            instructions + "\n\n"
            "Use the following glossary entries exactly as given:\n"
            f"{glossary_block}"
        )
    else:
        system = instructions

    return system

# =============================================================================
# MAIN TRANSLATION FUNCTION
# =============================================================================

def main(log_callback=None, stop_callback=None):
    """Main translation function with enhanced duplicate detection"""
    if log_callback:
        set_output_redirect(log_callback)
    
    # Set up stop checking
    def check_stop():
        if stop_callback and stop_callback():
            print("❌ Translation stopped by user request.")
            return True
        return is_stop_requested()
    
    # Parse all environment variables
    MODEL = os.getenv("MODEL", "gemini-1.5-flash")
    EPUB_PATH = os.getenv("EPUB_PATH", "default.epub")
    TRANSLATION_LANG = os.getenv("TRANSLATION_LANG", "korean").lower()
    CONTEXTUAL = os.getenv("CONTEXTUAL", "1") == "1"
    DELAY = int(os.getenv("SEND_INTERVAL_SECONDS", "2"))
    SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "").strip()
    REMOVE_AI_ARTIFACTS = os.getenv("REMOVE_AI_ARTIFACTS", "0") == "1"
    TEMP = float(os.getenv("TRANSLATION_TEMPERATURE", "0.3"))
    HIST_LIMIT = int(os.getenv("TRANSLATION_HISTORY_LIMIT", "20"))
    MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "8192"))
    EMERGENCY_RESTORE = os.getenv("EMERGENCY_PARAGRAPH_RESTORE", "1") == "1"

    # Log the setting
    if EMERGENCY_RESTORE:
        print("✅ Emergency paragraph restoration is ENABLED")
    else:
        print("⚠️ Emergency paragraph restoration is DISABLED")
    
    # Add debug logging
    print(f"[DEBUG] REMOVE_AI_ARTIFACTS environment variable: {os.getenv('REMOVE_AI_ARTIFACTS', 'NOT SET')}")
    print(f"[DEBUG] REMOVE_AI_ARTIFACTS parsed value: {REMOVE_AI_ARTIFACTS}")
    if REMOVE_AI_ARTIFACTS:
        print("⚠️ AI artifact removal is ENABLED - will clean AI response artifacts")
    else:
        print("✅ AI artifact removal is DISABLED - preserving all content as-is")
        
    # Parse chapter range
    rng = os.getenv("CHAPTER_RANGE", "")
    if rng and re.match(r"^\d+\s*-\s*\d+$", rng):
        start, end = map(int, rng.split("-", 1))
    else:
        start, end = None, None
    
    # Get instructions for the language
    instructions = get_instructions(TRANSLATION_LANG)
    
    # Set up tokenizer
    try:
        enc = tiktoken.encoding_for_model(MODEL)
    except:
        enc = tiktoken.get_encoding("cl100k_base")
    
    # Get API key
    API_KEY = (os.getenv("API_KEY") or 
               os.getenv("OPENAI_API_KEY") or 
               os.getenv("OPENAI_OR_Gemini_API_KEY") or
               os.getenv("GEMINI_API_KEY"))

    if not API_KEY:
        print("❌ Error: Set API_KEY, OPENAI_API_KEY, or OPENAI_OR_Gemini_API_KEY in your environment.")
        return

    print(f"[DEBUG] Found API key: {API_KEY[:10]}...")
    print(f"[DEBUG] Using model = {MODEL}")
    print(f"[DEBUG] Max output tokens = {MAX_OUTPUT_TOKENS}")

    # Initialize client
    client = UnifiedClient(model=MODEL, api_key=API_KEY)
        
    # Set up paths
    epub_path = sys.argv[1] if len(sys.argv) > 1 else EPUB_PATH
    epub_base = os.path.splitext(os.path.basename(epub_path))[0]
    out = epub_base
    os.makedirs(out, exist_ok=True)
    print(f"[DEBUG] Created output folder → {out}")

    # FIXED: Clean up previous extraction files to prevent duplicates
    cleanup_previous_extraction(out)

    # Set output directory in environment
    os.environ["EPUB_OUTPUT_DIR"] = out
    payloads_dir = out
    
    # Initialize HistoryManager and ChapterSplitter
    history_manager = HistoryManager(payloads_dir)
    chapter_splitter = ChapterSplitter(model_name=MODEL)
    
    # Purge old translation history on startup
    history_file = os.path.join(payloads_dir, "translation_history.json")
    if os.path.exists(history_file):
        os.remove(history_file)
        print(f"[DEBUG] Purged translation history → {history_file}")
        
    # Initialize improved progress tracking
    prog, PROGRESS_FILE = init_progress_tracking(payloads_dir)

    def save_progress():
        try:
            # Write to a temporary file first
            temp_file = PROGRESS_FILE + '.tmp'
            with open(temp_file, "w", encoding="utf-8") as pf:
                json.dump(prog, pf, ensure_ascii=False, indent=2)
            
            # If successful, replace the original file
            if os.path.exists(PROGRESS_FILE):
                os.remove(PROGRESS_FILE)
            os.rename(temp_file, PROGRESS_FILE)
        except Exception as e:
            print(f"⚠️ Warning: Failed to save progress: {e}")
            # Clean up temp file if it exists
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass

    # Always scan for missing files at startup
    print("🔍 Checking for deleted output files...")
    prog = cleanup_missing_files(prog, out)

    # Save the cleaned progress immediately
    save_progress()

    # Reset failed chapters if toggle is enabled
    if os.getenv("RESET_FAILED_CHAPTERS", "1") == "1":
        reset_count = 0
        for chapter_key, chapter_info in list(prog["chapters"].items()):
            status = chapter_info.get("status")
            
            # Reset chapters that failed, have QA issues, or had files deleted
            if status in ["failed", "qa_failed", "file_missing", "error", "file_deleted"]:
                # Remove the chapter from progress to force re-translation
                del prog["chapters"][chapter_key]
                
                # Also remove from content hashes
                content_hash = chapter_info.get("content_hash")
                if content_hash and content_hash in prog["content_hashes"]:
                    del prog["content_hashes"][content_hash]
                
                # Remove chunk data if exists
                if chapter_key in prog.get("chapter_chunks", {}):
                    del prog["chapter_chunks"][chapter_key]
                    
                reset_count += 1
        
        if reset_count > 0:
            print(f"🔄 Reset {reset_count} failed/deleted chapters for re-translation")
            save_progress()

    # Clean up any orphaned entries at startup
    prog = cleanup_progress_tracking(prog, out)
    save_progress()

    # Check for stop before starting
    if check_stop():
        return

    # Extract EPUB contents WITH CONSOLIDATED EXTRACTION
    print("🚀 Using comprehensive chapter extraction with resource handling...")
    with zipfile.ZipFile(epub_path, 'r') as zf:
        metadata = extract_epub_metadata(zf)
        chapters = extract_chapters(zf, out)  # ONE FUNCTION, ALL FUNCTIONALITY!
        
        # Validate chapters
        validate_chapter_continuity(chapters)

    # FIXED: Add validation after extraction
    print("\n" + "="*50)
    validate_epub_structure(out)
    print("="*50 + "\n")

    # Check for stop after file processing
    if check_stop():
        return

    # Write metadata with chapter info (enhanced metadata is already saved by extract_chapters)
    # Just ensure we have the basic metadata.json for backward compatibility
    if not os.path.exists(os.path.join(out, "metadata.json")):
        metadata["chapter_count"] = len(chapters)
        metadata["chapter_titles"] = {str(c["num"]): c["title"] for c in chapters}
        with open(os.path.join(out, "metadata.json"), 'w', encoding='utf-8') as mf:
            json.dump(metadata, mf, ensure_ascii=False, indent=2)
        
# Replace the glossary handling section in main() function (around line 1400)

    # Handle glossary - ENSURE IT COMPLETES BEFORE TRANSLATION
    manual_gloss = os.getenv("MANUAL_GLOSSARY")
    disable_auto_glossary = os.getenv("DISABLE_AUTO_GLOSSARY", "0") == "1"

    print("\n" + "="*50)
    print("📑 GLOSSARY GENERATION PHASE")
    print("="*50)

    if manual_gloss and os.path.isfile(manual_gloss):
        shutil.copy(manual_gloss, os.path.join(out, "glossary.json"))
        print("📑 Using manual glossary from:", manual_gloss)
    elif not disable_auto_glossary:
        print("📑 Starting automatic glossary generation...")
        

        
        # Generate glossary and WAIT for completion
        try:
            save_glossary(out, chapters, instructions)  # No language parameter needed
            print("✅ Automatic glossary generation COMPLETED")
        except Exception as e:
            print(f"❌ Glossary generation failed: {e}")
    else:
        print("📑 Automatic glossary disabled - no glossary will be used")
        # Create empty glossary file for consistency
        with open(os.path.join(out, "glossary.json"), 'w', encoding='utf-8') as f:
            json.dump({}, f, ensure_ascii=False, indent=2)

    # Verify glossary exists before proceeding
    glossary_path = os.path.join(out, "glossary.json")
    if os.path.exists(glossary_path):
        try:
            with open(glossary_path, 'r', encoding='utf-8') as f:
                glossary_data = json.load(f)
            print(f"📑 Glossary ready with {len(glossary_data)} entries")
            
            # Show sample entries
            if glossary_data:
                sample_items = list(glossary_data.items())[:3]
                print("📑 Sample glossary entries:")
                for key, value in sample_items:
                    print(f"   • {key} → {value}")
        except Exception as e:
            print(f"⚠️ Glossary file exists but is corrupted: {e}")
            # Create empty glossary as fallback
            with open(glossary_path, 'w', encoding='utf-8') as f:
                json.dump({}, f, ensure_ascii=False, indent=2)
    else:
        print("⚠️ No glossary file found, creating empty one")
        with open(glossary_path, 'w', encoding='utf-8') as f:
            json.dump({}, f, ensure_ascii=False, indent=2)

    print("="*50)
    print("🚀 STARTING MAIN TRANSLATION PHASE")
    print("="*50 + "\n")

    # Build system prompt (this will now use the completed glossary)
    system = build_system_prompt(SYSTEM_PROMPT, glossary_path, instructions)
    base_msg = [{"role": "system", "content": system}]
    
    total_chapters = len(chapters)
    
    # First pass: Count total chunks needed
    print("📊 Calculating total chunks needed...")
    total_chunks_needed = 0
    chunks_per_chapter = {}
    
    for idx, c in enumerate(chapters):
        chap_num = c["num"]
        content_hash = c.get("content_hash") or get_content_hash(c["body"])
        
        # Apply chapter range filter
        if start is not None and not (start <= chap_num <= end):
            continue
        
        # Check chapter status
        needs_translation, skip_reason, _ = check_chapter_status(
            prog, idx, chap_num, content_hash, out
        )
        
        if not needs_translation:
            chunks_per_chapter[idx] = 0
            continue
        
        # For in-progress chapters, check if they have partial chunks completed
        chapter_key = str(idx)
        if chapter_key in prog["chapters"] and prog["chapters"][chapter_key].get("status") == "in_progress":
            # Chapter is in progress, calculate remaining chunks
            pass  # Will be handled below
        
        # Parse token limit for counting
        _tok_env = os.getenv("MAX_INPUT_TOKENS", "1000000").strip()
        max_tokens_limit, _ = parse_token_limit(_tok_env)
        
        # Calculate available tokens
        system_tokens = chapter_splitter.count_tokens(system)
        history_tokens = HIST_LIMIT * 2 * 1000
        safety_margin = 1000
        
        if max_tokens_limit is not None:
            available_tokens = max_tokens_limit - system_tokens - history_tokens - safety_margin
            chunks = chapter_splitter.split_chapter(c["body"], available_tokens)
        else:
            chunks = [(c["body"], 1, 1)]
        
        # Count chunks needed for this chapter
        chapter_key_str = str(idx)
        if chapter_key_str in prog.get("chapter_chunks", {}):
            completed_chunks = len(prog["chapter_chunks"][chapter_key_str].get("completed", []))
            chunks_needed = len(chunks) - completed_chunks
            chunks_per_chapter[idx] = max(0, chunks_needed)
        else:
            chunks_per_chapter[idx] = len(chunks)
        
        total_chunks_needed += chunks_per_chapter[idx]
    
    print(f"📊 Total chunks to translate: {total_chunks_needed}")
    
    # Print chapter breakdown if there are multi-chunk chapters
    multi_chunk_chapters = [(idx, count) for idx, count in chunks_per_chapter.items() if count > 1]
    if multi_chunk_chapters:
        print("📄 Chapters requiring multiple chunks:")
        for idx, chunk_count in multi_chunk_chapters:
            chap = chapters[idx]
            print(f"   • Chapter {idx+1} ({chap['title'][:30]}...): {chunk_count} chunks")
    
    # Track timing for ETA calculation
    translation_start_time = time.time()
    chunks_completed = 0
    
    # Process each chapter with chunk counting
    current_chunk_number = 0
    
    for idx, c in enumerate(chapters):
        # Check for stop at the beginning of each chapter
        if check_stop():
            print(f"❌ Translation stopped at chapter {idx+1}")
            return
            
        chap_num = c["num"]
        content_hash = c.get("content_hash") or get_content_hash(c["body"])

        # Apply chapter range filter
        if start is not None and not (start <= chap_num <= end):
            continue

        # Check chapter status with improved logic
        needs_translation, skip_reason, existing_file = check_chapter_status(
            prog, idx, chap_num, content_hash, out
        )
        
        if not needs_translation:
            print(f"[SKIP] {skip_reason}")
            continue

        print(f"\n🔄 Processing Chapter {idx+1}/{total_chapters}: {c['title']}")
        
        # Mark as in-progress (output_filename is None until completed)
        update_progress(prog, idx, chap_num, content_hash, output_filename=None, status="in_progress")
        save_progress()

        # Parse token limit
        _tok_env = os.getenv("MAX_INPUT_TOKENS", "1000000").strip()
        max_tokens_limit, budget_str = parse_token_limit(_tok_env)
        
        # Calculate available tokens for content
        system_tokens = chapter_splitter.count_tokens(system)
        history_tokens = HIST_LIMIT * 2 * 1000
        safety_margin = 1000
        
        # Determine if we need to split the chapter
        if max_tokens_limit is not None:
            available_tokens = max_tokens_limit - system_tokens - history_tokens - safety_margin
            chunks = chapter_splitter.split_chapter(c["body"], available_tokens)
        else:
            chunks = [(c["body"], 1, 1)]
        
        print(f"📄 Chapter will be processed in {len(chunks)} chunk(s)")
        
        # Show token information if split was needed
        if len(chunks) > 1:
            chapter_tokens = chapter_splitter.count_tokens(c["body"])
            print(f"   ℹ️ Chapter size: {chapter_tokens:,} tokens (limit: {available_tokens:,} tokens per chunk)")
        else:
            chapter_tokens = chapter_splitter.count_tokens(c["body"])
            if max_tokens_limit is not None:
                print(f"   ℹ️ Chapter size: {chapter_tokens:,} tokens (within limit of {available_tokens:,} tokens)")
        
        # Track translated chunks for this chapter
        chapter_key_str = str(idx)
        if chapter_key_str not in prog["chapter_chunks"]:
            prog["chapter_chunks"][chapter_key_str] = {
                "total": len(chunks),
                "completed": [],
                "chunks": {}
            }
        
        prog["chapter_chunks"][chapter_key_str]["total"] = len(chunks)
        
        translated_chunks = []
        
        # Process each chunk
        for chunk_html, chunk_idx, total_chunks in chunks:
            # Check if this chunk was already translated
            if chunk_idx in prog["chapter_chunks"][chapter_key_str]["completed"]:
                saved_chunk = prog["chapter_chunks"][chapter_key_str]["chunks"].get(str(chunk_idx))
                if saved_chunk:
                    translated_chunks.append((saved_chunk, chunk_idx, total_chunks))
                    print(f"  [SKIP] Chunk {chunk_idx}/{total_chunks} already translated")
                    continue
                    
            # Check if history will reset on this chapter
            if CONTEXTUAL and history_manager.will_reset_on_next_append(HIST_LIMIT):
                print(f"  📌 History will reset after this chunk (current: {len(history_manager.load_history())//2}/{HIST_LIMIT} exchanges)")
                
            if check_stop():
                print(f"❌ Translation stopped during chapter {idx+1}, chunk {chunk_idx}")
                return
            
            current_chunk_number += 1
            
            # Calculate progress and ETA
            progress_percent = (current_chunk_number / total_chunks_needed) * 100
            
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
            
            if total_chunks > 1:
                print(f"  🔄 Translating chunk {chunk_idx}/{total_chunks} (Overall: {current_chunk_number}/{total_chunks_needed} - {progress_percent:.1f}% - ETA: {eta_str})")
            else:
                print(f"  🔄 Translating chapter (Overall: {current_chunk_number}/{total_chunks_needed} - {progress_percent:.1f}% - ETA: {eta_str})")
            
            # Add chunk context to prompt if multi-chunk
            if total_chunks > 1:
                user_prompt = f"[PART {chunk_idx}/{total_chunks}]\n{chunk_html}"
            else:
                user_prompt = chunk_html
            
            # Load history using thread-safe manager
            history = history_manager.load_history()
            
            # Build messages with context
            if CONTEXTUAL:
                trimmed = history[-HIST_LIMIT*2:]
            else:
                trimmed = []

            # Build messages - FILTER OUT OLD SUMMARIES
            if base_msg:
                # Remove any existing summary messages if rolling summary is disabled
                if os.getenv("USE_ROLLING_SUMMARY", "0") == "0":
                    filtered_base = [msg for msg in base_msg if "summary of the previous" not in msg.get("content", "")]
                    msgs = filtered_base + trimmed + [{"role": "user", "content": user_prompt}]
                else:
                    msgs = base_msg + trimmed + [{"role": "user", "content": user_prompt}]
            else:
                if trimmed:
                    msgs = trimmed + [{"role": "user", "content": user_prompt}]
                else:
                    msgs = [{"role": "user", "content": user_prompt}]

            while True:
                # Check for stop before API call
                if check_stop():
                    print(f"❌ Translation stopped during chapter {idx+1}")
                    return
                    
                try:
                    # Calculate actual token usage
                    total_tokens = sum(chapter_splitter.count_tokens(m["content"]) for m in msgs)
                    print(f"    [DEBUG] Chunk {chunk_idx}/{total_chunks} tokens = {total_tokens:,} / {budget_str}")
                    
                    client.context = 'translation'
                    
                    # BASIC RETRY LOGIC WITH GRADUAL TEMPERATURE INCREASE
                    retry_count = 0
                    max_retries = 3
                    duplicate_retry_count = 0
                    max_duplicate_retries = 6
                    history_purged = False
                    
                    # Store original values for retry
                    original_max_tokens = MAX_OUTPUT_TOKENS
                    original_temp = TEMP
                    original_user_prompt = user_prompt
                    
                    while retry_count <= max_retries or (duplicate_retry_count < max_duplicate_retries):
                        # Use current values (may be modified by retry logic)
                        current_max_tokens = MAX_OUTPUT_TOKENS
                        current_temp = TEMP
                        current_user_prompt = user_prompt
                        
                        # Make API call
                        result, finish_reason = send_with_interrupt(
                            msgs, client, current_temp, current_max_tokens, check_stop
                        )
                        
                        # Check if retry is needed
                        retry_needed = False
                        retry_reason = ""
                        is_duplicate_retry = False
                        
                        # Check for truncation (existing toggle)
                        if finish_reason == "length" and os.getenv("RETRY_TRUNCATED", "0") == "1":
                            if retry_count < max_retries:
                                retry_needed = True
                                retry_reason = "truncated output"
                                retry_max_tokens = int(os.getenv("MAX_RETRY_TOKENS", "16384"))
                                MAX_OUTPUT_TOKENS = min(MAX_OUTPUT_TOKENS * 2, retry_max_tokens)
                        
                        # SIMPLIFIED duplicate detection (safer)
                        if not retry_needed and os.getenv("RETRY_DUPLICATE_BODIES", "1") == "1":
                            if duplicate_retry_count < max_duplicate_retries:
                                try:
                                    # Simple text comparison
                                    result_clean = re.sub(r'<[^>]+>', '', result).strip().lower()
                                    result_sample = result_clean[:1000]  # First 1000 chars
                                    
                                    # Check against last few chapters only
                                    lookback_chapters = int(os.getenv("DUPLICATE_LOOKBACK_CHAPTERS", "3"))
                                    
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
                                                    
                                                    # Simple similarity check
                                                    if len(result_sample) > 100 and len(prev_sample) > 100:
                                                        # Count common words
                                                        result_words = set(result_sample.split())
                                                        prev_words = set(prev_sample.split())
                                                        
                                                        if len(result_words) > 0 and len(prev_words) > 0:
                                                            common = len(result_words & prev_words)
                                                            total = len(result_words | prev_words)
                                                            similarity = common / total if total > 0 else 0
                                                            
                                                            # More conservative 85% threshold
                                                            if similarity > 0.85:
                                                                retry_needed = True
                                                                is_duplicate_retry = True
                                                                retry_reason = f"duplicate content (similarity: {int(similarity*100)}%)"
                                                                
                                                                # Temperature management
                                                                if duplicate_retry_count >= 3 and not history_purged:
                                                                    print(f"    🧹 Clearing history after 3 attempts...")
                                                                    history_manager.save_history([])
                                                                    history = []
                                                                    trimmed = []
                                                                    history_purged = True
                                                                    TEMP = original_temp
                                                                    
                                                                    # Rebuild messages
                                                                    if base_msg:
                                                                        msgs = base_msg + [{"role": "user", "content": user_prompt}]
                                                                    else:
                                                                        msgs = [{"role": "user", "content": user_prompt}]
                                                                
                                                                elif duplicate_retry_count == 0:
                                                                    # First retry: same temperature
                                                                    print(f"    🔄 First duplicate retry - same temperature")
                                                                
                                                                elif history_purged:
                                                                    # Post-purge: gradual increase
                                                                    attempts_since_purge = duplicate_retry_count - 3
                                                                    TEMP = min(original_temp + (0.1 * attempts_since_purge), 1.0)
                                                                    print(f"    🌡️ Post-purge temp: {TEMP}")
                                                                
                                                                else:
                                                                    # Pre-purge: gradual increase
                                                                    TEMP = min(original_temp + (0.1 * duplicate_retry_count), 1.0)
                                                                    print(f"    🌡️ Gradual temp increase: {TEMP}")
                                                                
                                                                # Simple prompt variation
                                                                if duplicate_retry_count == 0:
                                                                    user_prompt = f"[RETRY] Chapter {c['num']}: Ensure unique translation.\n{chunk_html}"
                                                                elif duplicate_retry_count <= 2:
                                                                    user_prompt = f"[ATTEMPT {duplicate_retry_count + 1}] Translate uniquely:\n{chunk_html}"
                                                                else:
                                                                    user_prompt = f"Chapter {c['num']}:\n{chunk_html}"
                                                                
                                                                msgs[-1] = {"role": "user", "content": user_prompt}
                                                                break
                                                
                                                except Exception as e:
                                                    print(f"    [WARN] Error checking file: {e}")
                                                    continue
                                
                                except Exception as e:
                                    print(f"    [WARN] Duplicate check error: {e}")
                                    # Continue without duplicate detection
                        
                        # Break if no retry needed
                        if not retry_needed:
                            break
                            
                        # Update counters
                        if is_duplicate_retry:
                            duplicate_retry_count += 1
                            if duplicate_retry_count > max_duplicate_retries:
                                print(f"    ❌ Max duplicate retries reached, proceeding")
                                break
                        else:
                            retry_count += 1
                            if retry_count > max_retries:
                                print(f"    ❌ Max retries reached, proceeding")
                                break
                        
                        # Simple logging
                        if is_duplicate_retry:
                            print(f"    🔄 Duplicate retry {duplicate_retry_count}/{max_duplicate_retries}")
                        else:
                            print(f"    🔄 Retry {retry_count}/{max_retries}: {retry_reason}")
                        
                        time.sleep(2)
                    
                    # Restore original values
                    # Restore original values
                    MAX_OUTPUT_TOKENS = original_max_tokens
                    TEMP = original_temp
                    user_prompt = original_user_prompt
                    
                    # Only print restoration message if values were actually changed
                    if retry_count > 0 or duplicate_retry_count > 0:
                        if duplicate_retry_count > 0:
                            print(f"    🔄 Restored original temperature: {TEMP} (after {duplicate_retry_count} duplicate retries)")
                        elif retry_count > 0:
                            print(f"    🔄 Restored original settings after {retry_count} retries")
                    
                    # If duplicate was detected but not resolved, add a warning
                    if duplicate_retry_count >= max_duplicate_retries:
                        print(f"    ⚠️ WARNING: Duplicate content issue persists after {max_duplicate_retries} attempts")
                    
                    # Clean AI artifacts ONLY if the toggle is enabled
                    if REMOVE_AI_ARTIFACTS:
                        result = clean_ai_artifacts(result)
                    
                    if EMERGENCY_RESTORE:
                        result = emergency_restore_paragraphs(result, chunk_html)
                    
                    # Additional cleaning if remove artifacts is enabled
                    if REMOVE_AI_ARTIFACTS:
                        # Remove any JSON artifacts at the very beginning
                        lines = result.split('\n')
                        
                        # Only check the first few lines for JSON artifacts
                        json_line_count = 0
                        for i, line in enumerate(lines[:5]):  # Only check first 5 lines
                            if line.strip() and any(pattern in line for pattern in [
                                '"role":', '"content":', '"messages":', 
                                '{"role"', '{"content"', '[{', '}]'
                            ]):
                                json_line_count = i + 1
                            else:
                                # Found a non-JSON line, stop here
                                break
                        
                        if json_line_count > 0 and json_line_count < len(lines):
                            # Only remove if we found JSON and there's content after it
                            remaining = '\n'.join(lines[json_line_count:])
                            if remaining.strip() and len(remaining) > 100:
                                result = remaining
                                print(f"✂️ Removed {json_line_count} lines of JSON artifacts")
                    
                    # Remove chunk markers if present
                    result = re.sub(r'\[PART \d+/\d+\]\s*', '', result, flags=re.IGNORECASE)
                    
                    # Save chunk result
                    translated_chunks.append((result, chunk_idx, total_chunks))
                    
                    # Update progress for this chunk
                    prog["chapter_chunks"][chapter_key_str]["completed"].append(chunk_idx)
                    prog["chapter_chunks"][chapter_key_str]["chunks"][str(chunk_idx)] = result
                    save_progress()
                    
                    # Increment completed chunks counter
                    chunks_completed += 1
                        
                    # Check if we're about to reset history BEFORE appending
                    will_reset = history_manager.will_reset_on_next_append(HIST_LIMIT if CONTEXTUAL else 0)

                    # Generate rolling summary BEFORE history reset
                    if will_reset and os.getenv("USE_ROLLING_SUMMARY", "0") == "1" and CONTEXTUAL:
                        if check_stop():
                            print(f"❌ Translation stopped during summary generation for chapter {idx+1}")
                            return
                        
                        # Get current history before it's cleared
                        current_history = history_manager.load_history()
                        if len(current_history) >= 4:  # At least 2 exchanges
                            # Extract recent assistant responses
                            assistant_responses = []
                            for h in current_history[-8:]:  # Last 4 exchanges
                                if h.get("role") == "assistant":
                                    assistant_responses.append(h["content"])
                            
                            if assistant_responses:
                                # Generate summary
                                summary_prompt = (
                                    "Summarize the key events, characters, tone, and important details from these translations. "
                                    "Focus on: character names/relationships, plot developments, and any special terminology used.\n\n"
                                    + "\n---\n".join(assistant_responses[-3:])  # Last 3 responses
                                )
                                
                                summary_msgs = [
                                    {"role": "system", "content": "Create a concise summary for context continuity."},
                                    {"role": "user", "content": summary_prompt}
                                ]
                                
                                try:
                                    summary_resp, _ = send_with_interrupt(
                                        summary_msgs, client, TEMP, min(2000, MAX_OUTPUT_TOKENS), check_stop
                                    )
                                    
                                    # Save summary to file
                                    summary_file = os.path.join(out, "rolling_summary.txt")
                                    with open(summary_file, "a", encoding="utf-8") as sf:  # Append mode
                                        sf.write(f"\n\n=== Summary before chapter {idx+1}, chunk {chunk_idx} ===\n")
                                        sf.write(summary_resp.strip())
                                    
                                    # Update base_msg to include summary
                                    # First, remove any existing summary message
                                    base_msg[:] = [msg for msg in base_msg if "summary of the previous" not in msg.get("content", "")]
                                    
                                    # Add new summary
                                    summary_msg = {
                                        "role": os.getenv("SUMMARY_ROLE", "user"),
                                        "content": (
                                            "Here is a summary of the previous context to maintain continuity:\n\n"
                                            f"{summary_resp.strip()}"
                                        )
                                    }
                                    
                                    # Insert after system message
                                    if base_msg and base_msg[0].get("role") == "system":
                                        base_msg.insert(1, summary_msg)
                                    else:
                                        base_msg.insert(0, summary_msg)
                                    
                                    print(f"📝 Generated rolling summary before history reset")
                                    
                                except Exception as e:
                                    print(f"⚠️ Failed to generate rolling summary: {e}")

                    # NOW append to history (which may reset it)
                    history = history_manager.append_to_history(
                        user_prompt, 
                        result, 
                        HIST_LIMIT if CONTEXTUAL else 0,
                        reset_on_limit=True
                    )

                    # Delay between chunks/API calls
                    if chunk_idx < total_chunks:
                        for i in range(DELAY):
                            if check_stop():
                                print("❌ Translation stopped during delay")
                                return
                            time.sleep(1)
                    break

                except UnifiedClientError as e:
                    error_msg = str(e)
                    if "stopped by user" in error_msg:
                        print("❌ Translation stopped by user during API call")
                        return
                    elif "timed out" in error_msg:
                        print(f"⚠️ {error_msg}, retrying...")
                        continue
                    elif getattr(e, "http_status", None) == 429:
                        print("⚠️ Rate limited, sleeping 60s…")
                        for i in range(60):
                            if check_stop():
                                print("❌ Translation stopped during rate limit wait")
                                return
                            time.sleep(1)
                    else:
                        raise

        # Check for stop before merging and saving
        if check_stop():
            print(f"❌ Translation stopped before saving chapter {idx+1}")
            return

        # Merge all chunks back together
        if len(translated_chunks) > 1:
            print(f"  📎 Merging {len(translated_chunks)} chunks...")
            translated_chunks.sort(key=lambda x: x[1])
            merged_result = chapter_splitter.merge_translated_chunks(translated_chunks)
        else:
            merged_result = translated_chunks[0][0] if translated_chunks else ""

        # Save translated chapter
        safe_title = make_safe_filename(c['title'], c['num'])   
        fname = f"response_{c['num']:03d}_{safe_title}.html"

        # Clean up code fences only
        cleaned = re.sub(r"^```(?:html)?\s*\n?", "", merged_result, count=1, flags=re.MULTILINE)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned, count=1, flags=re.MULTILINE)

        # Final artifact cleanup - NOW RESPECTS THE TOGGLE and is conservative
        cleaned = clean_ai_artifacts(cleaned, remove_artifacts=REMOVE_AI_ARTIFACTS)

        # Write HTML file
        with open(os.path.join(out, fname), 'w', encoding='utf-8') as f:
            f.write(cleaned)
        
        final_title = c['title'] or safe_title
        print(f"[Chapter {idx+1}/{total_chapters}] ✅ Saved Chapter {c['num']}: {final_title}")
        
        # Update progress with completed status
        update_progress(prog, idx, chap_num, content_hash, fname, status="completed")
        save_progress()

    # Check for stop before building EPUB
    if check_stop():
        print("❌ Translation stopped before building EPUB")
        return

    # Build final EPUB
    print("📘 Building final EPUB…")
    try:
        from epub_converter import fallback_compile_epub
        fallback_compile_epub(out, log_callback=log_callback)
        print("✅ All done: your final EPUB is in", out)
        
        # Print final statistics
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
        
        # Print progress tracking statistics
        stats = get_translation_stats(prog, out)
        print(f"\n📊 Progress Tracking Summary:")
        print(f"   • Total chapters tracked: {stats['total_tracked']}")
        print(f"   • Successfully completed: {stats['completed']}")
        print(f"   • Missing files: {stats['missing_files']}")
        print(f"   • In progress: {stats['in_progress']}")
            
    except Exception as e:
        print("❌ EPUB build failed:", e)

    # Signal completion to GUI
    print("TRANSLATION_COMPLETE_SIGNAL")

if __name__ == "__main__":
    main()
