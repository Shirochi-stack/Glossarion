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
    """
    Check if a chapter needs translation
    Returns: (needs_translation, skip_reason, existing_file)
    """
    chapter_key = str(chapter_idx)
    
    # FIRST: Always check if the actual output file exists
    # This is the most important check - if file is deleted, we must retranslate
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
                # Remove from content hashes to prevent duplicate detection issues
                if content_hash in prog["content_hashes"]:
                    stored_info = prog["content_hashes"][content_hash]
                    if stored_info.get("chapter_idx") == chapter_idx:
                        del prog["content_hashes"][content_hash]
                
                # Mark chapter as needing retranslation
                chapter_info["status"] = "file_deleted"
                chapter_info["output_file"] = None  # Clear the reference
                
                # Also clear any chunk data
                if str(chapter_idx) in prog.get("chapter_chunks", {}):
                    del prog["chapter_chunks"][str(chapter_idx)]
                
                return True, None, None
            else:
                # File exists, check if it's a valid translation
                try:
                    # Verify file is not empty or corrupted
                    with open(output_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if len(content.strip()) < 100:  # Suspiciously short
                            print(f"⚠️ Output file for chapter {chapter_num} seems corrupted/empty")
                            return True, None, None
                except Exception as e:
                    print(f"⚠️ Error reading output file for chapter {chapter_num}: {e}")
                    return True, None, None
                
                # File exists and is valid
                return False, f"Chapter {chapter_num} already translated (file exists: {output_file})", output_file
        
        elif chapter_info.get("status") == "in_progress":
            # Chapter is currently being processed
            return True, None, None
        elif chapter_info.get("status") == "file_deleted":
            # Chapter was marked for retranslation due to deleted file
            return True, None, None
    
    # Check for duplicate content ONLY if we haven't already determined we need to translate
    # This prevents skipping chapters when their duplicate's file also doesn't exist
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
                    # Verify the duplicate file is also valid
                    try:
                        with open(dup_path, 'r', encoding='utf-8') as f:
                            dup_content = f.read()
                            if len(dup_content.strip()) < 100:
                                # Duplicate file is corrupted, don't rely on it
                                return True, None, None
                    except:
                        # Can't read duplicate file, need to translate this one
                        return True, None, None
                    
                    return False, f"Chapter {chapter_num} has same content as chapter {duplicate_info.get('chapter_num')} (already translated)", None
                else:
                    # Duplicate's file doesn't exist either, need to translate
                    return True, None, None
    
    # Chapter needs translation
    return True, None, None


def cleanup_missing_files(prog, output_dir):
    """
    Scan progress tracking and clean up any references to missing files
    This ensures deleted files will trigger retranslation
    """
    cleaned_count = 0
    
    # Check each chapter entry
    for chapter_key, chapter_info in list(prog["chapters"].items()):
        output_file = chapter_info.get("output_file")
        
        if output_file:
            output_path = os.path.join(output_dir, output_file)
            if not os.path.exists(output_path):
                # File is missing!
                print(f"🧹 Found missing file for chapter {chapter_info.get('chapter_num', chapter_key)}: {output_file}")
                
                # Mark chapter for retranslation
                chapter_info["status"] = "file_deleted"
                chapter_info["output_file"] = None
                
                # Remove from content hashes
                content_hash = chapter_info.get("content_hash")
                if content_hash and content_hash in prog["content_hashes"]:
                    stored_info = prog["content_hashes"][content_hash]
                    if stored_info.get("chapter_idx") == int(chapter_key):
                        del prog["content_hashes"][content_hash]
                
                # Remove chunk data
                if chapter_key in prog.get("chapter_chunks", {}):
                    del prog["chapter_chunks"][chapter_key]
                
                cleaned_count += 1
    
    if cleaned_count > 0:
        print(f"🔄 Marked {cleaned_count} chapters for retranslation due to missing files")
    
    return prog

def update_progress(prog, chapter_idx, chapter_num, content_hash, output_filename=None, status="completed"):
    """Update progress tracking after successful translation"""
    chapter_key = str(chapter_idx)
    
    # Update main chapter tracking
    prog["chapters"][chapter_key] = {
        "chapter_num": chapter_num,
        "content_hash": content_hash,
        "output_file": output_filename,  # Can be None for in_progress
        "status": status,
        "timestamp": time.time()
    }
    
    # Only update content hash mapping if we have a completed file
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
    
    # Check each chapter entry
    for chapter_key, chapter_info in list(prog["chapters"].items()):
        # Only check if output_file exists and is not None
        if chapter_info.get("output_file"):
            output_path = os.path.join(output_dir, chapter_info["output_file"])
            if not os.path.exists(output_path):
                # File is missing, mark as incomplete
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
            # Verify file exists
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

def emergency_restore_paragraphs(text, original_html=None, verbose=True):
    """
    Emergency restoration when AI returns wall of text without proper paragraph tags.
    This function attempts to restore paragraph structure using various heuristics.
    """
    # Helper function for logging
    def log(message):
        if verbose:
            print(message)
    
    # Check if we already have proper paragraph structure
    if text.count('</p>') >= 3:  # Assume 3+ paragraphs means structure is OK
        return text
    
    # If we have the original HTML, try to match its structure
    if original_html:
        original_para_count = original_html.count('<p>')
        current_para_count = text.count('<p>')
        
        if current_para_count < original_para_count / 2:  # Less than half the expected paragraphs
            log(f"⚠️ Paragraph mismatch! Original: {original_para_count}, Current: {current_para_count}")
            log("🔧 Attempting emergency paragraph restoration...")
    
    # If no paragraph tags found and text is long, we have a problem
    if '</p>' not in text and len(text) > 300:
        log("❌ No paragraph tags found - applying emergency restoration")
        
        # First, try to preserve any existing HTML tags
        has_html = '<' in text and '>' in text
        
        # Clean up any broken tags
        text = text.replace('</p><p>', '</p>\n<p>')  # Ensure line breaks between paragraphs
        
        # Strategy 1: Look for double line breaks (often indicates paragraph break)
        if '\n\n' in text:
            parts = text.split('\n\n')
            paragraphs = ['<p>' + part.strip() + '</p>' for part in parts if part.strip()]
            return '\n'.join(paragraphs)
        
        # Strategy 2: Look for dialogue patterns (quotes often start new paragraphs)
        dialogue_pattern = r'(?<=[.!?])\s+(?=[""\u201c\u201d])'
        if re.search(dialogue_pattern, text):
            parts = re.split(dialogue_pattern, text)
            paragraphs = []
            for part in parts:
                part = part.strip()
                if part:
                    # Check if it already has tags
                    if not part.startswith('<p>'):
                        part = '<p>' + part
                    if not part.endswith('</p>'):
                        part = part + '</p>'
                    paragraphs.append(part)
            return '\n'.join(paragraphs)
        
        # Strategy 3: Split by sentence patterns
        # Look for: period/exclamation/question mark + space + capital letter
        sentence_boundary = r'(?<=[.!?])\s+(?=[A-Z\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af])'
        sentences = re.split(sentence_boundary, text)
        
        if len(sentences) > 1:
            # Group sentences into paragraphs
            # Aim for 3-5 sentences per paragraph, or natural breaks
            paragraphs = []
            current_para = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                current_para.append(sentence)
                
                # Create new paragraph if:
                # - We have 3-4 sentences
                # - Current sentence ends with closing quote
                # - Next sentence would start with quote
                # - Current sentence seems like scene break
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
            
            # Don't forget the last paragraph
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
        # Split into chunks of ~150-200 words
        words = text.split()
        if len(words) > 100:
            paragraphs = []
            words_per_para = max(100, len(words) // 10)  # Aim for ~10 paragraphs
            
            for i in range(0, len(words), words_per_para):
                chunk = ' '.join(words[i:i + words_per_para])
                if chunk.strip():
                    paragraphs.append('<p>' + chunk.strip() + '</p>')
            
            return '\n'.join(paragraphs)
    
    # If text has some structure but seems incomplete
    elif '<p>' in text and text.count('<p>') < 3 and len(text) > 1000:
        log("⚠️ Very few paragraphs for long text - checking if more breaks needed")
        
        # Extract existing paragraphs
        soup = BeautifulSoup(text, 'html.parser')
        existing_paras = soup.find_all('p')
        
        # Check if any paragraph is too long
        new_paragraphs = []
        for para in existing_paras:
            para_text = para.get_text()
            if len(para_text) > 500:  # Paragraph seems too long
                # Split this paragraph
                sentences = re.split(r'(?<=[.!?])\s+', para_text)
                if len(sentences) > 5:
                    # Re-group into smaller paragraphs
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
    
    # Return original text if no restoration needed
    return text

def extract_epub_metadata(zf):
    """Extract comprehensive metadata from EPUB file"""
    meta = {}
    try:
        # Find OPF file
        for name in zf.namelist():
            if name.lower().endswith('.opf'):
                opf_content = zf.read(name)
                soup = BeautifulSoup(opf_content, 'xml')
                
                # Extract Dublin Core metadata
                for tag in ['title', 'creator', 'language', 'publisher', 'date', 'subject']:
                    element = soup.find(tag)
                    if element:
                        meta[tag] = element.get_text(strip=True)
                
                # Extract additional metadata
                description = soup.find('description')
                if description:
                    meta['description'] = description.get_text(strip=True)
                
                # Extract series information if available
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
    # Count characters by script
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
    
    # Determine primary language
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


def extract_all_resources(zf, output_dir):
    """Extract all resources (CSS, fonts, images) from EPUB"""
    extracted_resources = {
        'css': [],
        'fonts': [],
        'images': [],
        'other': []
    }
    
    # Create resource directories
    for resource_type in ['css', 'fonts', 'images']:
        resource_dir = os.path.join(output_dir, resource_type)
        os.makedirs(resource_dir, exist_ok=True)
    
    print(f"📦 Extracting all resources from EPUB...")
    
    for file_path in zf.namelist():
        # Skip directories
        if file_path.endswith('/'):
            continue
            
        file_name = os.path.basename(file_path)
        if not file_name:
            continue
            
        try:
            file_data = zf.read(file_path)
            resource_type = None
            target_dir = None
            
            # Determine resource type
            if file_path.lower().endswith('.css'):
                resource_type = 'css'
                target_dir = os.path.join(output_dir, 'css')
            elif file_path.lower().endswith(('.ttf', '.otf', '.woff', '.woff2', '.eot')):
                resource_type = 'fonts'
                target_dir = os.path.join(output_dir, 'fonts')
            elif file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.svg', '.bmp', '.webp')):
                resource_type = 'images'
                target_dir = os.path.join(output_dir, 'images')
            elif file_path.lower().endswith(('.js', '.xml', '.txt')):
                resource_type = 'other'
                target_dir = output_dir
            
            if resource_type and target_dir:
                # Sanitize filename
                safe_filename = sanitize_resource_filename(file_name)
                target_path = os.path.join(target_dir, safe_filename)
                
                # Avoid overwriting files with same name
                counter = 1
                original_path = target_path
                while os.path.exists(target_path):
                    name, ext = os.path.splitext(safe_filename)
                    target_path = os.path.join(target_dir, f"{name}_{counter}{ext}")
                    counter += 1
                
                # Write file
                with open(target_path, 'wb') as f:
                    f.write(file_data)
                
                extracted_resources[resource_type].append(safe_filename)
                print(f"   📄 Extracted {resource_type}: {safe_filename}")
                
        except Exception as e:
            print(f"[WARNING] Failed to extract {file_path}: {e}")
    
    # Summary
    total_extracted = sum(len(files) for files in extracted_resources.values())
    print(f"✅ Extracted {total_extracted} resource files:")
    for resource_type, files in extracted_resources.items():
        if files:
            print(f"   • {resource_type.title()}: {len(files)} files")
    
    return extracted_resources


def sanitize_resource_filename(filename):
    """Sanitize resource filenames for filesystem compatibility"""
    # Normalize unicode
    filename = unicodedata.normalize('NFC', filename)
    
    # Replace problematic characters
    replacements = {
        '/': '_', '\\': '_', ':': '_', '*': '_',
        '?': '_', '"': '_', '<': '_', '>': '_',
        '|': '_', '\0': '', '\n': '_', '\r': '_'
    }
    
    for old, new in replacements.items():
        filename = filename.replace(old, new)
    
    # Remove control characters
    filename = ''.join(char for char in filename if ord(char) >= 32)
    
    # Limit length
    name, ext = os.path.splitext(filename)
    if len(name) > 50:
        name = name[:50]
    
    if not name:
        name = 'resource'
    
    return name + ext


def extract_comprehensive_content_hash(html_content):
    """Create a more comprehensive hash that captures content structure and meaning"""
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract different types of content separately
        content_parts = []
        
        # 1. Text content (normalized)
        text_content = soup.get_text(strip=True).lower()
        # Remove chapter numbers and common markers
        text_content = re.sub(r'chapter\s+\d+[:\-\s]*', '', text_content, flags=re.IGNORECASE)
        text_content = re.sub(r'第\s*\d+\s*[章節话回][:\-\s]*', '', text_content)
        text_content = re.sub(r'제\s*\d+\s*[장화][:\-\s]*', '', text_content)
        text_content = re.sub(r'\s+', ' ', text_content).strip()
        content_parts.append(text_content[:2000])  # First 2000 chars
        
        # 2. Structural elements
        structure_info = []
        for tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            headers = soup.find_all(tag)
            for header in headers:
                header_text = header.get_text(strip=True).lower()
                if header_text and len(header_text) < 200:
                    structure_info.append(f"{tag}:{header_text}")
        
        # 3. Paragraph count and average length
        paragraphs = soup.find_all('p')
        if paragraphs:
            total_p_length = sum(len(p.get_text()) for p in paragraphs)
            avg_p_length = total_p_length // len(paragraphs)
            structure_info.append(f"paragraphs:{len(paragraphs)}:{avg_p_length}")
        
        # 4. Image information
        images = soup.find_all('img')
        for img in images:
            src = img.get('src', '')
            alt = img.get('alt', '')
            if src:
                structure_info.append(f"img:{os.path.basename(src)}")
            if alt:
                structure_info.append(f"alt:{alt.lower()}")
        
        # 5. Link information  
        links = soup.find_all('a')
        for link in links:
            href = link.get('href', '')
            link_text = link.get_text(strip=True).lower()
            if href and not href.startswith('#'):
                structure_info.append(f"link:{href}")
            if link_text:
                structure_info.append(f"linktext:{link_text}")
        
        # Combine all parts
        content_parts.extend(structure_info)
        
        # Create multiple hash signatures
        combined_content = '|||'.join(content_parts)
        
        # Main content hash
        main_hash = hashlib.md5(combined_content.encode('utf-8')).hexdigest()
        
        # Structural hash (for catching reordered content)
        structure_hash = hashlib.md5(''.join(structure_info).encode('utf-8')).hexdigest()
        
        # Text-only hash (for catching text with different formatting)
        text_hash = hashlib.md5(text_content.encode('utf-8')).hexdigest()
        
        # Combined signature
        return f"{main_hash}_{structure_hash[:8]}_{text_hash[:8]}"
        
    except Exception as e:
        print(f"[WARNING] Failed to create comprehensive hash: {e}")
        # Fallback to simple text hash
        simple_text = BeautifulSoup(html_content, 'html.parser').get_text()
        return hashlib.md5(simple_text.encode('utf-8')).hexdigest()


def extract_advanced_chapter_info(zf):
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
        (r'(\d+)', 0, 'any_number'),  # Last resort
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
        
        # Handle compound numbers
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
            
            # Skip very short files (likely not actual chapters)
            if len(content_text.strip()) < 200:
                print(f"[DEBUG] Skipping short file: {file_path} ({len(content_text)} chars)")
                continue
            
            # Create comprehensive content hash
            content_hash = extract_comprehensive_content_hash(content_html)
            
            # Group files by size (helps detect duplicates)
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
                # Try to find a meaningful title
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
                "language_sample": content_text[:500]  # For language detection
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
        
        # Show duplicate file size groups (potential duplicates)
        large_groups = [size for size, files in file_size_groups.items() if len(files) > 1]
        if large_groups:
            print(f"   ⚠️ Found {len(large_groups)} file size groups with potential duplicates")
    
    return chapters, detected_language


def enhanced_extract_chapters(zf, output_dir):
    """Enhanced chapter extraction with comprehensive resource handling"""
    
    print("🚀 Starting enhanced chapter extraction...")
    
    # Step 1: Extract all resources
    extracted_resources = extract_all_resources(zf, output_dir)
    
    # Step 2: Extract comprehensive metadata
    metadata = extract_epub_metadata(zf)
    print(f"📋 Extracted metadata: {list(metadata.keys())}")
    
    # Step 3: Extract chapters with advanced detection
    chapters, detected_language = extract_advanced_chapter_info(zf)
    
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
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Saved enhanced metadata to: {metadata_path}")
    
    # Step 6: Create extraction report
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
                f.write(f"  {resource_type.title()}: {len(files)} files\n")
                for file in files[:5]:  # Show first 5
                    f.write(f"    - {file}\n")
                if len(files) > 5:
                    f.write(f"    ... and {len(files) - 5} more\n")
    
    print(f"📄 Saved extraction report to: {report_path}")
    
    print(f"\n✅ Enhanced extraction complete!")
    print(f"   📚 Chapters: {len(chapters)}")
    print(f"   🎨 Resources: {sum(len(files) for files in extracted_resources.values())}")
    print(f"   🌍 Language: {detected_language}")
    
    return chapters


# Make this the new default extract_chapters function
extract_chapters = enhanced_extract_chapters

def get_content_hash(html_content):
    """Create a comprehensive hash of content to detect duplicates"""
    return extract_comprehensive_content_hash(html_content)

def clean_ai_artifacts(text, remove_artifacts=True):
    """Remove AI response artifacts from text - but ONLY when enabled"""
    if not remove_artifacts:
        return text
    
    # Only remove the first sentence/line if it looks like an AI artifact
    lines = text.split('\n', 2)  # Split into max 3 parts (first line, second line, rest)
    
    if len(lines) < 2:
        return text  # Nothing to remove if there's only one line
    
    first_line = lines[0].strip()
    
    # Skip if first line is empty
    if not first_line:
        # Check second line if first is empty
        if len(lines) > 1:
            first_line = lines[1].strip()
            if not first_line:
                return text
            lines = lines[1:]  # Adjust for empty first line
        else:
            return text
    
    # Common AI artifact patterns - be very specific
    ai_patterns = [
        # Direct AI responses
        r'^(?:Sure|Okay|Understood|Of course|Got it|Alright|Certainly|Here\'s|Here is)',
        r'^(?:I\'ll|I will|Let me) (?:translate|help|assist)',
        # Role markers
        r'^(?:System|Assistant|AI|User|Human|Model)\s*:',
        # Part markers
        r'^\[PART\s+\d+/\d+\]',
        # Common AI explanations
        r'^(?:Translation note|Note|Here\'s the translation|I\'ve translated)',
        # HTML/Code markers at the very start
        r'^```(?:html)?',
        r'^<!DOCTYPE',  # Sometimes AI starts with this unnecessarily
    ]
    
    # Check if the first line matches ANY of these patterns
    for pattern in ai_patterns:
        if re.search(pattern, first_line, re.IGNORECASE):
            # Only remove the first line/sentence
            remaining_text = '\n'.join(lines[1:]) if len(lines) > 1 else ''
            
            # Make sure we're not removing actual content
            # Check if what remains starts with a chapter header or has substantial content
            if remaining_text.strip():
                # Verify remaining text has actual chapter content
                if (re.search(r'<h[1-6]', remaining_text, re.IGNORECASE) or 
                    re.search(r'Chapter\s+\d+', remaining_text, re.IGNORECASE) or
                    re.search(r'第\s*\d+\s*[章节話话回]', remaining_text) or
                    re.search(r'제\s*\d+\s*[장화]', remaining_text) or
                    len(remaining_text.strip()) > 100):  # Has substantial content
                    
                    print(f"✂️ Removed AI artifact: {first_line[:50]}...")
                    return remaining_text.lstrip()
    
    # Additional check: if first line is just "html" or similar single words
    if first_line.lower() in ['html', 'text', 'content', 'translation', 'output']:
        remaining_text = '\n'.join(lines[1:]) if len(lines) > 1 else ''
        if remaining_text.strip():
            print(f"✂️ Removed single word artifact: {first_line}")
            return remaining_text.lstrip()
    
    # No artifacts detected, return original
    return text

def save_glossary(output_dir, chapters, instructions, language="korean"):
    """Generate and save glossary from chapters with proper CJK support"""
    samples = []
    for c in chapters:
        samples.append(c["body"])
    
    names = set()
    suffixes = set()
    terms = set()
    
    # Remove HTML tags for better text processing
    def clean_html(html_text):
        soup = BeautifulSoup(html_text, 'html.parser')
        return soup.get_text()
    
    for txt in samples:
        clean_text = clean_html(txt)
        
        if language == "korean":
            # Korean names (2-4 character Korean names)
            korean_names = re.findall(r'[가-힣]{2,4}(?:님|씨|야|아|이|군|양)?', clean_text)
            names.update(korean_names)
            
            # Korean suffixes
            korean_suffixes = re.findall(r'[가-힣]+(?:님|씨|야|아|이|형|누나|언니|오빠|선배|후배|군|양)', clean_text)
            suffixes.update(korean_suffixes)
            
            # Also catch romanized versions
            for s in re.findall(r"\b\w+[-~]?(?:nim|ssi|ah|ya|ie|hyung|noona|unnie|oppa|sunbae|hoobae|gun|yang)\b", clean_text, re.I):
                suffixes.add(s)
        
        elif language == "japanese":
            # Japanese names (kanji names, usually 2-4 characters)
            japanese_names = re.findall(r'[\u4e00-\u9fff]{2,4}(?:さん|様|ちゃん|君|先生|殿)?', clean_text)
            names.update(japanese_names)
            
            # Hiragana/Katakana names
            kana_names = re.findall(r'[\u3040-\u309f\u30a0-\u30ff]{2,8}(?:さん|様|ちゃん|君)?', clean_text)
            names.update(kana_names)
            
            # Japanese honorifics (in Japanese script)
            jp_honorifics = re.findall(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]+(?:さん|様|ちゃん|君|先輩|後輩|先生|殿)', clean_text)
            suffixes.update(jp_honorifics)
            
            # Family terms
            jp_family = re.findall(r'(?:お兄|お姉|おじ|おば|兄|姉)(?:さん|様|ちゃん)?', clean_text)
            terms.update(jp_family)
            
            # Also catch romanized versions
            for s in re.findall(r"\b\w+[-~]?(?:san|sama|chan|kun|senpai|kouhai|sensei|dono)\b", clean_text, re.I):
                suffixes.add(s)
        
        elif language == "chinese":
            # Chinese names (2-4 character names, avoiding common words)
            chinese_names = []
            
            # Common Chinese surnames (top 100)
            surnames = '王李张刘陈杨赵黄周吴徐孙胡朱高林何郭马罗梁宋郑谢韩唐冯于董萧程曹袁邓许傅沈曾彭吕苏卢蒋蔡贾丁魏薛叶阎余潘杜戴夏钟汪田任姜范方石姚谭廖邹熊金陆郝孔白崔康毛邱秦江史顾侯邵孟龙万段章钱汤尹黎易常武乔贺赖龚文'
            
            # Find names starting with common surnames
            for match in re.finditer(f'[{surnames}][\u4e00-\u9fff]{{1,3}}', clean_text):
                name = match.group()
                # Filter out common words that might match pattern
                if len(name) <= 4:
                    chinese_names.append(name)
            
            names.update(chinese_names)
            
            # Chinese titles and honorifics
            chinese_titles = re.findall(r'[\u4e00-\u9fff]{2,4}(?:公子|小姐|夫人|先生|大人|少爷|姑娘|老爷)', clean_text)
            terms.update(chinese_titles)
            
            # Cultivation/xianxia terms if present
            cultivation_terms = re.findall(r'(?:师尊|师父|师傅|道长|真人|上人|尊者|圣人|仙人|掌门|宗主|长老)', clean_text)
            terms.update(cultivation_terms)
            
            # Family terms
            family_terms = re.findall(r'(?:阿|啊)?(?:爹|娘|爷|奶|公|婆|哥|姐|弟|妹|叔|姨|舅)', clean_text)
            terms.update(family_terms)
            
            # Also check for pinyin names
            pinyin_names = re.findall(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', clean_text)
            names.update(pinyin_names)
        
        # Also extract any romanized names for all languages
        for nm in re.findall(r"\b[A-Z][a-z]{2,20}\b", clean_text):
            names.add(nm)
    
    # Filter and clean up results
    names = [n for n in names if len(n) > 1 and not n.isdigit()]
    suffixes = [s for s in suffixes if len(s) > 1]
    terms = [t for t in terms if len(t) > 1]
    
    # Sort for consistency
    names = sorted(list(set(names)))[:100]
    suffixes = sorted(list(set(suffixes)))[:50]
    terms = sorted(list(set(terms)))[:50]
    
    # Build glossary based on language
    gloss = {}
    
    if language == "korean":
        gloss["Korean_Names"] = names
        gloss["Korean_Honorifics"] = suffixes
    elif language == "japanese":
        gloss["Japanese_Names"] = names
        gloss["Japanese_Honorifics"] = suffixes
        if terms:
            gloss["Japanese_Family_Terms"] = terms
    elif language == "chinese":
        gloss["Chinese_Names"] = names
        if terms:
            gloss["Chinese_Titles"] = terms
        if suffixes:
            gloss["Chinese_Terms"] = suffixes
    
    # Add a note about the glossary
    gloss["_note"] = f"Auto-generated glossary for {language} text"
    
    with open(os.path.join(output_dir, "glossary.json"), 'w', encoding='utf-8') as f:
        json.dump(gloss, f, ensure_ascii=False, indent=2)

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
            # Simple similarity check
            if title1 == title2 and chapters[i]['num'] != chapters[j]['num']:
                issues.append(f"Chapters {chapters[i]['num']} and {chapters[j]['num']} have identical titles")
    
    if issues:
        print("\n⚠️  Chapter Validation Issues:")
        for issue in issues:
            print(f"  - {issue}")
        print()


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
    EMERGENCY_RESTORE = os.getenv("EMERGENCY_PARAGRAPH_RESTORE", "1") == "1"  # Default to enabled

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

    # NEW: Always scan for missing files at startup
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
            save_progress()  # Now this will work because save_progress is defined above

    # Clean up any orphaned entries at startup
    prog = cleanup_progress_tracking(prog, out)
    save_progress()

    # Check for stop before starting
    if check_stop():
        return

    # Extract EPUB contents WITH ENHANCED EXTRACTION
    with zipfile.ZipFile(epub_path, 'r') as zf:
        metadata = extract_epub_metadata(zf)
        chapters = extract_chapters(zf, out)  # This now uses the enhanced extraction!
        
        # The enhanced extraction already handles duplicates more comprehensively
        # Validate chapters
        validate_chapter_continuity(chapters)

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
        
    # Handle glossary
    manual_gloss = os.getenv("MANUAL_GLOSSARY")
    disable_auto_glossary = os.getenv("DISABLE_AUTO_GLOSSARY", "0") == "1"

    if manual_gloss and os.path.isfile(manual_gloss):
        shutil.copy(manual_gloss, os.path.join(out, "glossary.json"))
        print("📑 Using manual glossary")
    elif not disable_auto_glossary:
        # Use detected language from enhanced extraction
        detected_lang = metadata.get('detected_language', TRANSLATION_LANG)
        save_glossary(out, chapters, instructions, detected_lang)
        print("📑 Generated automatic glossary")
    else:
        print("📑 Automatic glossary disabled - no glossary will be used")

    # Build system prompt
    glossary_path = os.path.join(out, "glossary.json")
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
                    
                    # ENHANCED RETRY LOGIC WITH HISTORY PURGING
                    retry_count = 0
                    max_retries = 3
                    duplicate_retry_count = 0  # Track duplicate-specific retries
                    max_duplicate_retries = 6  # Total allowed duplicate retries (3 + 3 after history purge)
                    history_purged = False  # Track if we've purged history
                    
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
                        
                        # Check for duplicate body content (enhanced logic)
                        if not retry_needed and os.getenv("RETRY_DUPLICATE_BODIES", "1") == "1":
                            if duplicate_retry_count < max_duplicate_retries:
                                # Extract body from the result (remove headers)
                                result_soup = BeautifulSoup(result, 'html.parser')
                                
                                # Remove headers to get just body content
                                for header in result_soup.find_all(['h1', 'h2', 'h3', 'title']):
                                    header.decompose()
                                
                                # Get more content for comparison (not just 1000 chars)
                                result_body = result_soup.get_text(strip=True)
                                
                                # Normalize the text for better comparison
                                def normalize_text(text):
                                    # Remove extra whitespace
                                    text = re.sub(r'\s+', ' ', text).strip()
                                    # Remove numbers that might be chapter numbers
                                    text = re.sub(r'\b\d+\b', '', text)
                                    # Take first 2000 chars for comparison
                                    return text[:2000].lower()
                                
                                normalized_result = normalize_text(result_body)
                                
                                # Check against previously translated chapters
                                lookback_chapters = int(os.getenv("DUPLICATE_LOOKBACK_CHAPTERS", "5"))
                                
                                for prev_idx in range(max(0, idx - lookback_chapters), idx):
                                    prev_key = str(prev_idx)
                                    if prev_key in prog["chapters"] and prog["chapters"][prev_key].get("output_file"):
                                        prev_file = prog["chapters"][prev_key]["output_file"]
                                        prev_path = os.path.join(out, prev_file)
                                        
                                        if os.path.exists(prev_path):
                                            try:
                                                with open(prev_path, 'r', encoding='utf-8') as f:
                                                    prev_content = f.read()
                                                
                                                # Extract body from previous chapter
                                                prev_soup = BeautifulSoup(prev_content, 'html.parser')
                                                for header in prev_soup.find_all(['h1', 'h2', 'h3', 'title']):
                                                    header.decompose()
                                                prev_body = prev_soup.get_text(strip=True)
                                                
                                                normalized_prev = normalize_text(prev_body)
                                                
                                                # Calculate similarity instead of exact match
                                                from difflib import SequenceMatcher
                                                similarity = SequenceMatcher(None, normalized_result, normalized_prev).ratio()
                                                
                                                # If similarity is very high (>85%), it's likely a duplicate
                                                if similarity > 0.85:
                                                    retry_needed = True
                                                    is_duplicate_retry = True
                                                    retry_reason = f"duplicate body content (matches chapter {chapters[prev_idx]['num']} with {int(similarity*100)}% similarity)"
                                                    
                                                    # ENHANCED: After 3 duplicate retries, purge history and try again
                                                    if duplicate_retry_count >= 3 and not history_purged:
                                                        print(f"    🧹 Purging translation history after 3 duplicate attempts...")
                                                        
                                                        # Clear the history
                                                        history_manager.save_history([])
                                                        history = []
                                                        trimmed = []
                                                        history_purged = True
                                                        
                                                        # Rebuild messages without history
                                                        if base_msg:
                                                            msgs = base_msg + [{"role": "user", "content": user_prompt}]
                                                        else:
                                                            msgs = [{"role": "user", "content": user_prompt}]
                                                        
                                                        print(f"    🔄 Retrying with fresh history (attempt {duplicate_retry_count + 1}/{max_duplicate_retries})")
                                                    else:
                                                        # Regular duplicate retry - increase temperature more aggressively
                                                        TEMP = min(TEMP + 0.3, 1.0)
                                                    
                                                    # Create a more specific prompt
                                                    user_prompt = f"""[CRITICAL: This is Chapter {c['num']} titled "{c['title']}". 
                        The previous response was {int(similarity*100)}% similar to Chapter {chapters[prev_idx]['num']}.

                        You MUST translate the UNIQUE content below for Chapter {c['num']}. This chapter contains DIFFERENT events than Chapter {chapters[prev_idx]['num']}.

                        Key differences to focus on:
                        - This is a DIFFERENT chapter with its own unique story progression
                        - Characters may be in different situations
                        - Events should progress from where Chapter {c['num']-1} ended
                        - Do NOT repeat any content from Chapter {chapters[prev_idx]['num']}

                        IMPORTANT: Each chapter in this novel has its own unique content. Even if chapter titles seem similar, the actual events and story progression are different.]

                        {chunk_html}"""
                                                    
                                                    # Update messages with new prompt
                                                    msgs[-1] = {"role": "user", "content": user_prompt}
                                                    
                                                    break
                                            except Exception as e:
                                                print(f"    [WARN] Error checking previous chapter: {e}")
                        
                        # If no retry needed or max retries reached, break
                        if not retry_needed:
                            break
                            
                        # Check retry limits
                        if is_duplicate_retry:
                            duplicate_retry_count += 1
                            if duplicate_retry_count > max_duplicate_retries:
                                print(f"    ❌ Still getting {retry_reason} after {max_duplicate_retries} total attempts, proceeding anyway")
                                break
                        else:
                            retry_count += 1
                            if retry_count > max_retries:
                                print(f"    ❌ Still getting {retry_reason} after {max_retries} retries, proceeding anyway")
                                break
                        
                        # Retry needed
                        print(f"    ⚠️ Detected {retry_reason}!")
                        
                        if is_duplicate_retry:
                            if history_purged:
                                print(f"    🔄 Retrying with purged history (duplicate attempt {duplicate_retry_count}/{max_duplicate_retries})")
                            else:
                                print(f"    🔄 Retrying with increased temperature (duplicate attempt {duplicate_retry_count}/{max_duplicate_retries})")
                                print(f"    📊 Increased temperature to {TEMP}")
                        else:
                            print(f"    🔄 Retrying translation (attempt {retry_count}/{max_retries})")
                            if "truncated" in retry_reason:
                                print(f"    📊 Increased max tokens to {MAX_OUTPUT_TOKENS}")
                        
                        # Brief delay before retry
                        time.sleep(2)
                    
                    # Restore original values after retry loop
                    MAX_OUTPUT_TOKENS = original_max_tokens
                    TEMP = original_temp
                    user_prompt = original_user_prompt
                    
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
