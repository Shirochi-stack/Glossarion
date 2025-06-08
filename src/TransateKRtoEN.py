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
import time

# Import the new modules
from history_manager import HistoryManager
from chapter_splitter import ChapterSplitter
from image_translator import ImageTranslator
from typing import Dict, List, Tuple 


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
    
def process_chapter_images(chapter_html: str, chapter_num: int, image_translator: ImageTranslator, 
                         check_stop_fn=None) -> Tuple[str, Dict[str, str]]:
    """
    Process and translate images in a chapter
    
    Args:
        chapter_html: HTML content of the chapter
        chapter_num: Chapter number for context
        image_translator: ImageTranslator instance
        check_stop_fn: Function to check if translation should stop
        
    Returns:
        (updated_html, image_translations_map)
    """
    # Extract images from chapter
    images = image_translator.extract_images_from_chapter(chapter_html)
    
    if not images:
        return chapter_html, {}
        
    print(f"🖼️ Found {len(images)} images in chapter {chapter_num}")
    
    # Parse the HTML to modify it
    soup = BeautifulSoup(chapter_html, 'html.parser')
    
    image_translations = {}
    translated_count = 0
    
    # Check for limits
    max_images_per_chapter = int(os.getenv('MAX_IMAGES_PER_CHAPTER', '10'))
    if len(images) > max_images_per_chapter:
        print(f"   ⚠️ Chapter has {len(images)} images - processing first {max_images_per_chapter} only")
        images = images[:max_images_per_chapter]
    
    for idx, img_info in enumerate(images, 1):
        if check_stop_fn and check_stop_fn():
            print("❌ Image translation stopped by user")
            break
            
        img_src = img_info['src']
        
        # Build full image path
        if img_src.startswith('../'):
            img_path = os.path.join(image_translator.output_dir, img_src[3:])
        elif img_src.startswith('./'):
            img_path = os.path.join(image_translator.output_dir, img_src[2:])
        elif img_src.startswith('/'):
            img_path = os.path.join(image_translator.output_dir, img_src[1:])
        else:
            # Check multiple possible locations
            possible_paths = [
                os.path.join(image_translator.images_dir, os.path.basename(img_src)),
                os.path.join(image_translator.output_dir, img_src),
                os.path.join(image_translator.output_dir, 'images', os.path.basename(img_src)),
                os.path.join(image_translator.output_dir, os.path.basename(img_src)),
                # Also check without 'images' subdirectory
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
        
        # Normalize path
        img_path = os.path.normpath(img_path)
        
        # Check if file exists
        if not os.path.exists(img_path):
            print(f"   ⚠️ Image not found: {img_path}")
            print(f"   📁 Images directory: {image_translator.images_dir}")
            print(f"   📁 Output directory: {image_translator.output_dir}")
            print(f"   📁 Working directory: {os.getcwd()}")
            
            # List files in images directory to help debug
            if os.path.exists(image_translator.images_dir):
                files = os.listdir(image_translator.images_dir)
                print(f"   📁 Files in images dir: {files[:5]}...")  # Show first 5 files
            continue
        
        
        print(f"   🔍 Processing image {idx}/{len(images)}: {os.path.basename(img_path)}")
        
        # Build context for translation
        context = f"Chapter {chapter_num}"
        if img_info.get('alt'):
            context += f", Alt text: {img_info['alt']}"
            
        # Add delay between API calls
        if translated_count > 0:
            delay = float(os.getenv('IMAGE_API_DELAY', '1.0'))
            time.sleep(delay)
            
        # Translate the image
        translation_result = image_translator.translate_image(img_path, context, check_stop_fn)
        
        if translation_result:
            # CRITICAL FIX: Update the HTML to include translation
            # Find the image in the soup
            img_tag = None
            for img in soup.find_all('img'):
                if img.get('src') == img_src:
                    img_tag = img
                    break
            
            if img_tag:
                # Create a container div for the image and translation
                container = soup.new_tag('div', **{'class': 'image-with-translation'})
                
                # Add a note about translation
                note = soup.new_tag('p', **{'class': 'translation-note'})
                note.string = f"[Image {idx} - Translated text below]"
                container.append(note)
                
                # Keep original image
                img_tag.replace_with(container)
                container.append(img_tag)
                
                # Add translation below image
                translation_div = soup.new_tag('div', **{'class': 'image-translation'})
                
                # Parse the translation result to extract just the text
                if '<div class="image-translation">' in translation_result:
                    # Extract the translated text from the result
                    trans_soup = BeautifulSoup(translation_result, 'html.parser')
                    trans_content = trans_soup.find('div', class_='image-translation')
                    if trans_content:
                        # Add the content to our div
                        for element in trans_content.children:
                            if element.name:
                                translation_div.append(element)
                else:
                    # Just add the text
                    trans_p = soup.new_tag('p')
                    trans_p.string = translation_result
                    translation_div.append(trans_p)
                
                container.append(translation_div)
                
                translated_count += 1
                
                # Save individual translation file
                trans_filename = f"ch{chapter_num:03d}_img{idx:02d}_translation.html"
                trans_filepath = os.path.join(image_translator.translated_images_dir, trans_filename)
                
                with open(trans_filepath, 'w', encoding='utf-8') as f:
                    f.write(f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8"/>
    <title>Chapter {chapter_num} - Image {idx} Translation</title>
</head>
<body>
    <h2>Chapter {chapter_num} - Image {idx}</h2>
    <p>Original: {os.path.basename(img_path)}</p>
    <hr/>
    {translation_div}
</body>
</html>""")
                
                print(f"   ✅ Saved translation to: {trans_filename}")
            else:
                print(f"   ⚠️ Could not find image tag in HTML for: {img_src}")
    
    # Summary
    if translated_count > 0:
        print(f"   🖼️ Successfully translated {translated_count} images")
        
        # Save translation log
        image_translator.save_translation_log(chapter_num, image_translations)
        
        # Return updated HTML
        return str(soup), image_translations
    else:
        print(f"   ℹ️ No images were successfully translated")
        
    return chapter_html, {}

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
    Extract chapters and all resources from EPUB
    This only extracts resources and loads chapters into memory - no HTML writing
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
    
    # Step 4: REMOVED - Don't write HTML files during extraction
    # Original HTML content is already stored in chapter['body']
    
    # Step 5: Create detailed chapter info file for debugging
    chapters_info_path = os.path.join(output_dir, 'chapters_info.json')
    chapters_info = []
    
    for c in chapters:
        info = {
            'num': c['num'],
            'title': c['title'],
            'original_filename': c.get('filename', ''),
            'has_images': c.get('has_images', False),
            'image_count': c.get('image_count', 0),
            'text_length': c.get('file_size', len(c.get('body', ''))),
            'detection_method': c.get('detection_method', 'unknown'),
            'content_hash': c.get('content_hash', '')
        }
        
        # Add image details if present
        if c.get('has_images'):
            try:
                soup = BeautifulSoup(c.get('body', ''), 'html.parser')
                images = soup.find_all('img')
                info['images'] = [img.get('src', '') for img in images]
            except:
                info['images'] = []
        
        chapters_info.append(info)
    
    with open(chapters_info_path, 'w', encoding='utf-8') as f:
        json.dump(chapters_info, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Saved detailed chapter info to: chapters_info.json")
    
    # Step 6: Enhance metadata with extracted information
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
    
    # Step 7: Save enhanced metadata
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Saved comprehensive metadata to: {metadata_path}")
    
    # Step 8: Create/update extraction report
    _create_extraction_report(output_dir, metadata, chapters, extracted_resources)
    
    # Step 9: Final validation and summary
    _log_extraction_summary(chapters, extracted_resources, detected_language)
    
    print("🔍 VERIFICATION: Comprehensive chapter extraction completed successfully")
    
    return chapters


def _log_extraction_summary(chapters, extracted_resources, detected_language, html_files_written=0):
    """Log final extraction summary with HTML file information"""
    print(f"\n✅ Comprehensive extraction complete!")
    print(f"   📚 Chapters: {len(chapters)}")
    print(f"   📄 HTML files written: {html_files_written}")
    print(f"   🎨 Resources: {sum(len(files) for files in extracted_resources.values())}")
    print(f"   🌍 Language: {detected_language}")
    
    # Count image-only chapters
    image_only_count = sum(1 for c in chapters if c.get('has_images') and c.get('file_size', 0) < 500)
    if image_only_count > 0:
        print(f"   📸 Image-only chapters: {image_only_count}")
    
    # Show EPUB structure file status
    epub_files = extracted_resources.get('epub_structure', [])
    if epub_files:
        print(f"   📋 EPUB Structure: {len(epub_files)} files ({', '.join(epub_files)})")
    else:
        print(f"   ⚠️ No EPUB structure files extracted!")
    
    # Verify pre-flight check will pass
    print(f"\n🔍 Pre-flight check readiness:")
    print(f"   ✅ HTML files: {'READY' if html_files_written > 0 else 'NOT READY'}")
    print(f"   ✅ Metadata: READY")
    print(f"   ✅ Resources: READY")

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
    """Extract chapters - either comprehensively or with smart filtering based on toggle"""
    
    # Check if comprehensive extraction is enabled
    comprehensive_mode = os.getenv("COMPREHENSIVE_EXTRACTION", "0") == "1"
    
    if comprehensive_mode:
        print("📚 Using COMPREHENSIVE extraction mode (all files)")
        return _extract_all_chapters_comprehensive(zf)
    else:
        print("📚 Using SMART extraction mode (filtered)")
        return _extract_chapters_smart(zf)


def _extract_all_chapters_comprehensive(zf):
    """Comprehensive extraction - includes ALL HTML files including image-only chapters"""
    chapters = []
    
    # Get ALL HTML-type files
    all_html_files = []
    for name in zf.namelist():
        if name.lower().endswith(('.xhtml', '.html', '.htm')):
            all_html_files.append(name)
    
    print(f"📚 Found {len(all_html_files)} HTML files in EPUB")
    
    # Sort files to maintain order
    all_html_files.sort()
    
    # Minimal skip list
    skip_keywords = ['nav.', 'toc.', 'contents.', 'copyright.', 'cover.']
    
    # Process ALL files
    chapter_num = 0
    for idx, file_path in enumerate(all_html_files):
        try:
            # Check if we should skip
            lower_name = file_path.lower()
            basename = os.path.basename(lower_name)
            
            # Only skip if filename exactly matches skip patterns
            should_skip = False
            for skip in skip_keywords:
                if basename == skip + 'xhtml' or basename == skip + 'html' or basename == skip + 'htm':
                    should_skip = True
                    break
            
            if should_skip:
                print(f"[SKIP] Navigation/TOC file: {file_path}")
                continue
            
            # Read the file
            file_data = zf.read(file_path)
            
            # Try to decode
            html_content = None
            for encoding in ['utf-8', 'utf-16', 'gb18030', 'shift_jis', 'euc-kr', 'gbk', 'big5']:
                try:
                    html_content = file_data.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if not html_content:
                print(f"[WARNING] Could not decode {file_path}")
                continue
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract content
            if soup.body:
                content_html = str(soup.body)
                content_text = soup.body.get_text(strip=True)
            else:
                content_html = html_content
                content_text = soup.get_text(strip=True)
            
            # Increment chapter number
            chapter_num += 1
            
            # Try to extract title
            chapter_title = None
            
            # From title tag
            if soup.title and soup.title.string:
                chapter_title = soup.title.string.strip()
            
            # From first header (including h2)
            if not chapter_title:
                for header_tag in ['h1', 'h2', 'h3']:
                    header = soup.find(header_tag)
                    if header:
                        chapter_title = header.get_text(strip=True)
                        break
            
            # From filename
            if not chapter_title:
                chapter_title = os.path.splitext(os.path.basename(file_path))[0]
            
            # Count images
            images = soup.find_all('img')
            
            # Determine if it's image-only
            is_image_only = len(images) > 0 and len(content_text.strip()) < 500
            
            # Create content hash
            content_hash = hashlib.md5(content_html.encode('utf-8')).hexdigest()
            
            # Store chapter
            chapter_info = {
                "num": chapter_num,
                "title": chapter_title or f"Chapter {chapter_num}",
                "body": content_html,
                "filename": file_path,
                "content_hash": content_hash,
                "detection_method": "comprehensive_sequential",
                "file_size": len(content_text),
                "has_images": len(images) > 0,
                "image_count": len(images),
                "is_empty": len(content_text.strip()) == 0,
                "is_image_only": is_image_only  # Add this flag
            }
            
            chapters.append(chapter_info)
            
            # Log what we found
            if is_image_only:
                print(f"[{chapter_num:04d}] 📸 Image-only chapter: {chapter_title} ({len(images)} images)")
            elif len(images) > 0 and len(content_text) >= 500:
                print(f"[{chapter_num:04d}] 📖📸 Mixed chapter: {chapter_title} ({len(content_text)} chars, {len(images)} images)")
            elif len(content_text) < 50:
                print(f"[{chapter_num:04d}] 📄 Empty/placeholder: {chapter_title}")
            else:
                print(f"[{chapter_num:04d}] 📖 Text chapter: {chapter_title} ({len(content_text)} chars)")
                
        except Exception as e:
            print(f"[ERROR] Failed to process {file_path}: {e}")
            continue
    
    print(f"\n📊 Comprehensive extraction complete: {len(chapters)} chapters found")
    
    # Quick stats
    image_only = sum(1 for c in chapters if c.get('is_image_only', False))
    text_chapters = sum(1 for c in chapters if c.get('file_size', 0) >= 500 and not c.get('is_image_only', False))
    mixed_chapters = sum(1 for c in chapters if c.get('has_images') and c.get('file_size', 0) >= 500)
    empty_chapters = sum(1 for c in chapters if c.get('is_empty'))
    
    print(f"   • Text chapters: {text_chapters}")
    print(f"   • Image-only chapters: {image_only}")
    print(f"   • Mixed content chapters: {mixed_chapters}")
    print(f"   • Empty/placeholder: {empty_chapters}")
    
    return chapters, 'unknown'
    
def _extract_chapters_smart(zf):
    """Extract comprehensive chapter information with improved detection for image-only chapters"""
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
    
    # Track h1 vs h2 usage for better detection
    h1_count = 0
    h2_count = 0
    
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
            
            # Check for h1 and h2 usage
            h1_tags = soup.find_all('h1')
            h2_tags = soup.find_all('h2')
            if h1_tags:
                h1_count += 1
            if h2_tags:
                h2_count += 1
            
            # Check for images FIRST - this is critical for image-only chapters
            images = soup.find_all('img')
            has_images = len(images) > 0
            
            # CRITICAL FIX: Check if this is an image-only chapter
            is_image_only_chapter = has_images and len(content_text.strip()) < 500
            
            # Skip only if it has NO content at all (no text AND no images)
            if len(content_text.strip()) < 10 and not has_images:
                print(f"[DEBUG] Skipping empty file: {file_path} (no text, no images)")
                continue
            
            # For image-only chapters, always process them
            if is_image_only_chapter:
                print(f"[DEBUG] Image-only chapter detected: {file_path} ({len(images)} images, {len(content_text)} chars)")
            
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
            
            # Determine header priority based on usage
            header_priority = ['h2', 'h1', 'h3'] if h2_count > h1_count else ['h1', 'h2', 'h3']
            
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
                    for header_tag in header_priority:
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
                                            detection_method = f"header_{header_tag}_{method}"
                                            break
                                        elif method == 'chinese_chapter_cn':
                                            converted = convert_chinese_number(num_str)
                                            if converted:
                                                chapter_num = converted
                                                chapter_title = header_text
                                                detection_method = f"header_{header_tag}_{method}"
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
                    for header_tag in header_priority:
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
            
            # Store chapter information with image-only flag
            chapter_info = {
                "num": chapter_num,
                "title": chapter_title,
                "body": content_html,
                "filename": file_path,
                "content_hash": content_hash,
                "detection_method": detection_method,
                "file_size": file_size,
                "has_images": has_images,
                "image_count": len(images),
                "is_empty": len(content_text.strip()) == 0,
                "is_image_only": is_image_only_chapter,  # Add this flag
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
            
            # Enhanced logging for different chapter types
            if is_image_only_chapter:
                print(f"[DEBUG] ✅ Chapter {chapter_num}: {chapter_title[:50]}... (IMAGE-ONLY, {detection_method})")
            else:
                print(f"[DEBUG] ✅ Chapter {chapter_num}: {chapter_title[:50]}... ({detection_method})")
            
        except Exception as e:
            print(f"[ERROR] Failed to process {file_path}: {e}")
            continue
    
    # Sort chapters by number
    chapters.sort(key=lambda x: x["num"])
    
    # Report h1 vs h2 usage
    if h2_count > h1_count:
        print(f"📊 Note: This EPUB uses primarily <h2> tags ({h2_count} files) vs <h1> tags ({h1_count} files)")
    
    # Detect primary language
    combined_sample = ' '.join(sample_texts)
    detected_language = detect_content_language(combined_sample)
    
    # Final validation and reporting
    if chapters:
        print(f"\n📊 Chapter Extraction Summary:")
        print(f"   • Total chapters extracted: {len(chapters)}")
        print(f"   • Chapter range: {chapters[0]['num']} to {chapters[-1]['num']}")
        print(f"   • Detected language: {detected_language}")
        print(f"   • Primary header type: {'<h2>' if h2_count > h1_count else '<h1>'}")
        
        # Count different chapter types
        image_only_count = sum(1 for c in chapters if c.get('is_image_only', False))
        text_only_count = sum(1 for c in chapters if not c.get('has_images', False) and c.get('file_size', 0) >= 500)
        mixed_count = sum(1 for c in chapters if c.get('has_images', False) and c.get('file_size', 0) >= 500)
        
        print(f"   • Text-only chapters: {text_only_count}")
        print(f"   • Image-only chapters: {image_only_count}")
        print(f"   • Mixed content chapters: {mixed_count}")
        
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
    """Create comprehensive extraction report with HTML file tracking"""
    report_path = os.path.join(output_dir, 'extraction_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("EPUB Extraction Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("METADATA:\n")
        for key, value in metadata.items():
            if key not in ['chapter_titles', 'extracted_resources']:
                f.write(f"  {key}: {value}\n")
        
        f.write(f"\nCHAPTERS ({len(chapters)}):\n")
        
        # Group chapters by type
        text_chapters = []
        image_only_chapters = []
        mixed_chapters = []
        
        for chapter in chapters:
            if chapter.get('has_images') and chapter.get('file_size', 0) < 500:
                image_only_chapters.append(chapter)
            elif chapter.get('has_images') and chapter.get('file_size', 0) >= 500:
                mixed_chapters.append(chapter)
            else:
                text_chapters.append(chapter)
        
        # Write chapter listings by type
        if text_chapters:
            f.write(f"\n  TEXT CHAPTERS ({len(text_chapters)}):\n")
            for c in text_chapters:
                f.write(f"    {c['num']:3d}. {c['title']} ({c['detection_method']})\n")
                if c.get('original_html_file'):
                    f.write(f"         → {c['original_html_file']}\n")
        
        if image_only_chapters:
            f.write(f"\n  IMAGE-ONLY CHAPTERS ({len(image_only_chapters)}):\n")
            for c in image_only_chapters:
                f.write(f"    {c['num']:3d}. {c['title']} (images: {c.get('image_count', 0)})\n")
                if c.get('original_html_file'):
                    f.write(f"         → {c['original_html_file']}\n")
                # List image sources if available
                if 'body' in c:
                    try:
                        soup = BeautifulSoup(c['body'], 'html.parser')
                        images = soup.find_all('img')
                        for img in images[:3]:  # Show first 3 images
                            src = img.get('src', 'unknown')
                            f.write(f"         • Image: {src}\n")
                        if len(images) > 3:
                            f.write(f"         • ... and {len(images) - 3} more images\n")
                    except:
                        pass
        
        if mixed_chapters:
            f.write(f"\n  MIXED CONTENT CHAPTERS ({len(mixed_chapters)}):\n")
            for c in mixed_chapters:
                f.write(f"    {c['num']:3d}. {c['title']} (text: {c.get('file_size', 0)} chars, images: {c.get('image_count', 0)})\n")
                if c.get('original_html_file'):
                    f.write(f"         → {c['original_html_file']}\n")
        
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
        
        # Add HTML files section
        f.write(f"\nHTML FILES WRITTEN:\n")
        html_files_written = metadata.get('html_files_written', 0)
        f.write(f"  Total: {html_files_written} files\n")
        f.write(f"  Location: Main directory and 'originals' subdirectory\n")
        
        # Check for potential issues
        f.write(f"\nPOTENTIAL ISSUES:\n")
        issues = []
        
        if image_only_chapters:
            issues.append(f"  • {len(image_only_chapters)} chapters contain only images (may need OCR)")
        
        missing_html = sum(1 for c in chapters if not c.get('original_html_file'))
        if missing_html > 0:
            issues.append(f"  • {missing_html} chapters failed to write HTML files")
        
        if not extracted_resources.get('epub_structure'):
            issues.append("  • No EPUB structure files found (may affect reconstruction)")
        
        if not issues:
            f.write("  None detected - extraction appears successful!\n")
        else:
            for issue in issues:
                f.write(issue + "\n")
    
    print(f"📄 Saved extraction report to: {report_path}")

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
# =============================================================================
# GLOSSARY MANAGEMENT - Complete Implementation
# =============================================================================

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

def translate_terms_batch(term_list, source_lang, batch_size=50):
    """Use GUI-controlled batch size for translation"""
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
        
        # Import UnifiedClient
        from unified_api_client import UnifiedClient
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

def save_glossary(output_dir, chapters, instructions, language="korean"):
    """
    Targeted glossary generator - Focuses on titles and CJK names with honorifics
    """
    
    print("📑 Targeted Glossary Generator v3.0")
    print("📑 Extracting titles and names with honorifics only")
    
    # Check for existing manual glossary first
    manual_glossary_path = os.getenv("MANUAL_GLOSSARY")
    if manual_glossary_path and os.path.exists(manual_glossary_path):
        print(f"📑 Manual glossary detected: {os.path.basename(manual_glossary_path)}")
        
        # Copy manual glossary to output directory
        target_path = os.path.join(output_dir, "glossary.json")
        try:
            shutil.copy2(manual_glossary_path, target_path)
            print(f"📑 ✅ Manual glossary copied to: {target_path}")
            print(f"📑 Skipping automatic glossary generation")
            
            # Also create a simple format version for compatibility
            with open(manual_glossary_path, 'r', encoding='utf-8') as f:
                manual_data = json.load(f)
            
            # Convert manual format to simple format
            simple_entries = {}
            if isinstance(manual_data, list):
                for char in manual_data:
                    original = char.get('original_name', '')
                    translated = char.get('name', original)
                    if original and translated:
                        simple_entries[original] = translated
            
            simple_path = os.path.join(output_dir, "glossary_simple.json")
            with open(simple_path, 'w', encoding='utf-8') as f:
                json.dump(simple_entries, f, ensure_ascii=False, indent=2)
            
            return simple_entries
            
        except Exception as e:
            print(f"⚠️ Could not copy manual glossary: {e}")
            print(f"📑 Proceeding with automatic generation...")
    
    # Check if there's an existing glossary from the manual extractor
    glossary_folder_path = os.path.join(output_dir, "Glossary")
    existing_glossary = None
    
    if os.path.exists(glossary_folder_path):
        # Look for JSON files in the Glossary folder
        for file in os.listdir(glossary_folder_path):
            if file.endswith("_glossary.json"):
                existing_path = os.path.join(glossary_folder_path, file)
                try:
                    with open(existing_path, 'r', encoding='utf-8') as f:
                        existing_glossary = json.load(f)
                    print(f"📑 Found existing glossary from manual extraction: {file}")
                    break
                except Exception as e:
                    print(f"⚠️ Could not load existing glossary: {e}")
    
    # Load settings
    min_frequency = int(os.getenv("GLOSSARY_MIN_FREQUENCY", "2"))
    max_names = int(os.getenv("GLOSSARY_MAX_NAMES", "50"))
    max_titles = int(os.getenv("GLOSSARY_MAX_TITLES", "30"))
    batch_size = int(os.getenv("GLOSSARY_BATCH_SIZE", "50"))
    
    print(f"📑 Settings: Min frequency: {min_frequency}, Max names: {max_names}, Max titles: {max_titles}")
    
    def clean_html(html_text):
        """Remove HTML tags to get clean text"""
        soup = BeautifulSoup(html_text, 'html.parser')
        return soup.get_text()
    
    # Extract and combine all text from chapters
    all_text = ' '.join(clean_html(chapter["body"]) for chapter in chapters)
    print(f"📑 Processing {len(all_text):,} characters of text")
    
    # Focused honorifics for CJK languages only
    CJK_HONORIFICS = {
        'korean': ['님', '씨', '선배', '형', '누나', '언니', '오빠', '선생님', '교수님', '사장님', '회장님'],
        'japanese': ['さん', 'ちゃん', '君', 'くん', '様', 'さま', '先生', 'せんせい', '殿', 'どの', '先輩', 'せんぱい'],
        'chinese': ['先生', '小姐', '夫人', '公子', '大人', '老师', '师父', '师傅', '同志', '同学'],
        'english': ['-san', '-chan', '-kun', '-sama', '-sensei', '-senpai', '-dono']  # For romanized content
    }
    
    # Title patterns for various languages
    TITLE_PATTERNS = {
        'korean': [
            r'\b(왕|여왕|왕자|공주|황제|황후|대왕|대공|공작|백작|자작|남작|기사|장군|대장|원수|제독|함장|대신|재상|총리|대통령|시장|지사|검사|판사|변호사|의사|박사|교수|신부|목사|스님|도사)\b',
            r'\b(폐하|전하|각하|예하|님|대감|영감|나리|도련님|아가씨|부인|선생)\b'
        ],
        'japanese': [
            r'\b(王|女王|王子|姫|皇帝|皇后|天皇|皇太子|大王|大公|公爵|伯爵|子爵|男爵|騎士|将軍|大将|元帥|提督|艦長|大臣|宰相|総理|大統領|市長|知事|検事|裁判官|弁護士|医者|博士|教授|神父|牧師|僧侶|道士)\b',
            r'\b(陛下|殿下|閣下|猊下|様|大人|殿|卿|君|氏)\b'
        ],
        'chinese': [
            r'\b(王|女王|王子|公主|皇帝|皇后|大王|大公|公爵|伯爵|子爵|男爵|骑士|将军|大将|元帅|提督|舰长|大臣|宰相|总理|大总统|市长|知事|检察官|法官|律师|医生|博士|教授|神父|牧师|和尚|道士)\b',
            r'\b(陛下|殿下|阁下|大人|老爷|夫人|小姐|公子|少爷|姑娘|先生)\b'
        ],
        'english': [
            r'\b(King|Queen|Prince|Princess|Emperor|Empress|Duke|Duchess|Marquis|Marquess|Earl|Count|Countess|Viscount|Viscountess|Baron|Baroness|Knight|Lord|Lady|Sir|Dame|General|Admiral|Captain|Major|Colonel|Commander|Lieutenant|Sergeant|Minister|Chancellor|President|Mayor|Governor|Judge|Doctor|Professor|Father|Reverend|Master|Mistress)\b',
            r'\b(His|Her|Your|Their)\s+(Majesty|Highness|Grace|Excellency|Honor|Worship|Lordship|Ladyship)\b'
        ]
    }
    
    # Enhanced exclusions - more strict
    COMMON_WORDS = {
        # Korean particles and common words that might appear with honorifics
        '이', '그', '저', '우리', '너희', '자기', '당신', '여기', '거기', '저기',
        '오늘', '내일', '어제', '지금', '아까', '나중', '먼저', '다음', '마지막',
        '모든', '어떤', '무슨', '이런', '그런', '저런', '같은', '다른', '새로운',
        
        # Japanese particles and common words
        'この', 'その', 'あの', 'どの', 'これ', 'それ', 'あれ', 'どれ',
        'わたし', 'あなた', 'かれ', 'かのじょ', 'わたしたち', 'あなたたち',
        'きょう', 'あした', 'きのう', 'いま', 'あとで', 'まえ', 'つぎ',
        
        # Chinese common words
        '这', '那', '哪', '这个', '那个', '哪个', '这里', '那里', '哪里',
        '我', '你', '他', '她', '它', '我们', '你们', '他们', '她们',
        '今天', '明天', '昨天', '现在', '刚才', '以后', '以前', '后来',
        
        # Single characters that are definitely not names
        '一', '二', '三', '四', '五', '六', '七', '八', '九', '十',
        'a', 'i', 'u', 'e', 'o', 'の', 'は', 'が', 'を', 'に', 'で', 'と', 'も',
        '은', '는', '이', '가', '을', '를', '에', '의', '와', '과', '도', '만'
    }
    
    def is_valid_name(name, language_hint='unknown'):
        """Strict validation for proper names only"""
        if not name or len(name.strip()) < 1:
            return False
            
        name = name.strip()
        
        # Exclude common words
        if name.lower() in COMMON_WORDS or name in COMMON_WORDS:
            return False
        
        # Language-specific validation
        if language_hint == 'korean':
            # Korean names are usually 2-4 characters (family name + given name)
            if not (2 <= len(name) <= 4):
                return False
            # Should be all Hangul
            if not all(0xAC00 <= ord(char) <= 0xD7AF for char in name):
                return False
                
        elif language_hint == 'japanese':
            # Japanese names can be 2-6 characters
            if not (2 <= len(name) <= 6):
                return False
            # Should contain kanji or kana
            has_kanji = any(0x4E00 <= ord(char) <= 0x9FFF for char in name)
            has_kana = any((0x3040 <= ord(char) <= 0x309F) or (0x30A0 <= ord(char) <= 0x30FF) for char in name)
            if not (has_kanji or has_kana):
                return False
                
        elif language_hint == 'chinese':
            # Chinese names are usually 2-4 characters
            if not (2 <= len(name) <= 4):
                return False
            # Should be all Chinese characters
            if not all(0x4E00 <= ord(char) <= 0x9FFF for char in name):
                return False
                
        elif language_hint == 'english':
            # English names should start with capital letter
            if not name[0].isupper():
                return False
            # Should be mostly letters
            if sum(1 for c in name if c.isalpha()) < len(name) * 0.8:
                return False
            # Reasonable length
            if not (2 <= len(name) <= 20):
                return False
        
        return True
    
    def detect_language_hint(text_sample):
        """Quick language detection for validation purposes"""
        # Count script usage
        korean_chars = sum(1 for char in text_sample[:1000] if 0xAC00 <= ord(char) <= 0xD7AF)
        japanese_kana = sum(1 for char in text_sample[:1000] if (0x3040 <= ord(char) <= 0x309F) or (0x30A0 <= ord(char) <= 0x30FF))
        chinese_chars = sum(1 for char in text_sample[:1000] if 0x4E00 <= ord(char) <= 0x9FFF)
        latin_chars = sum(1 for char in text_sample[:1000] if 0x0041 <= ord(char) <= 0x007A)
        
        if korean_chars > 50:
            return 'korean'
        elif japanese_kana > 20:
            return 'japanese'
        elif chinese_chars > 50 and japanese_kana < 10:
            return 'chinese'
        elif latin_chars > 100:
            return 'english'
        else:
            return 'unknown'
    
    # Detect primary language
    language_hint = detect_language_hint(all_text)
    print(f"📑 Detected primary language: {language_hint}")
    
    # Select appropriate honorifics
    honorifics_to_use = []
    if language_hint in CJK_HONORIFICS:
        honorifics_to_use.extend(CJK_HONORIFICS[language_hint])
    # Always include English romanized honorifics
    honorifics_to_use.extend(CJK_HONORIFICS['english'])
    
    print(f"📑 Using {len(honorifics_to_use)} honorifics for {language_hint}")
    
    # Find names with honorifics
    names_with_honorifics = {}
    standalone_names = {}
    
    print("📑 Scanning for names with honorifics...")
    
    # Process each honorific
    for honorific in honorifics_to_use:
        # Use regex to find name + honorific patterns
        if honorific.startswith('-'):
            # English style suffix
            pattern = r'(\b[A-Z][a-zA-Z]+)' + re.escape(honorific) + r'\b'
        else:
            # CJK style suffix - need to handle no word boundaries
            if language_hint == 'japanese':
                # Japanese names can be 1-6 chars before honorific
                pattern = r'([\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]{1,6})' + re.escape(honorific)
            elif language_hint == 'korean':
                # Korean names are typically 2-4 chars
                pattern = r'([\uac00-\ud7af]{2,4})' + re.escape(honorific)
            elif language_hint == 'chinese':
                # Chinese names are typically 2-4 chars
                pattern = r'([\u4e00-\u9fff]{2,4})' + re.escape(honorific)
            else:
                # Generic pattern
                pattern = r'(\S+)' + re.escape(honorific) + r'(?:\s|$|[。、.,!?])'
        
        matches = re.finditer(pattern, all_text)
        
        for match in matches:
            potential_name = match.group(1)
            
            if is_valid_name(potential_name, language_hint):
                full_form = potential_name + honorific
                
                # Count occurrences
                count = len(re.findall(re.escape(full_form), all_text))
                
                if count >= min_frequency:
                    if full_form not in names_with_honorifics:
                        names_with_honorifics[full_form] = count
                    if potential_name not in standalone_names:
                        standalone_names[potential_name] = count
    
    print(f"📑 Found {len(standalone_names)} unique names with honorifics")
    
    # Find titles
    print("📑 Scanning for titles...")
    found_titles = {}
    
    # Use appropriate title patterns
    title_patterns_to_use = []
    if language_hint in TITLE_PATTERNS:
        title_patterns_to_use.extend(TITLE_PATTERNS[language_hint])
    # Always check for English titles too
    title_patterns_to_use.extend(TITLE_PATTERNS['english'])
    
    for pattern in title_patterns_to_use:
        matches = re.finditer(pattern, all_text, re.IGNORECASE if 'english' in pattern else 0)
        
        for match in matches:
            title = match.group(0)
            count = len(re.findall(re.escape(title), all_text, re.IGNORECASE if 'english' in pattern else 0))
            
            if count >= min_frequency:
                # Normalize case for English titles
                if re.match(r'[A-Za-z]', title):
                    title = title.title()
                
                if title not in found_titles:
                    found_titles[title] = count
    
    print(f"📑 Found {len(found_titles)} unique titles")
    
    # Sort by frequency and apply limits
    sorted_names = sorted(names_with_honorifics.items(), key=lambda x: x[1], reverse=True)[:max_names]
    sorted_titles = sorted(found_titles.items(), key=lambda x: x[1], reverse=True)[:max_titles]
    
    # Combine for translation
    all_terms = []
    
    # Add names with count info
    for name, count in sorted_names:
        all_terms.append(name)
        
    # Add titles with count info  
    for title, count in sorted_titles:
        all_terms.append(title)
    
    print(f"📑 Total terms to translate: {len(all_terms)}")
    
    # Show sample of what was found
    if sorted_names:
        print("\n📑 Sample names found:")
        for name, count in sorted_names[:5]:
            print(f"   • {name} ({count}x)")
    
    if sorted_titles:
        print("\n📑 Sample titles found:")
        for title, count in sorted_titles[:5]:
            print(f"   • {title} ({count}x)")
    
    # Translate if enabled
    if os.getenv("DISABLE_GLOSSARY_TRANSLATION", "0") == "1":
        print("📑 Translation disabled - keeping original terms")
        translations = {term: term for term in all_terms}
    else:
        print(f"📑 Translating {len(all_terms)} terms...")
        translations = translate_terms_batch(all_terms, language_hint, batch_size)
    
    # Build final glossary with categories
    glossary_entries = {}
    
    # Add names
    for name, _ in sorted_names:
        if name in translations:
            glossary_entries[name] = translations[name]
    
    # Add titles
    for title, _ in sorted_titles:
        if title in translations:
            glossary_entries[title] = translations[title]
    
    # Merge with existing glossary if found
    if existing_glossary and isinstance(existing_glossary, list):
        print("📑 Merging with existing manual glossary...")
        merged_count = 0
        
        for char in existing_glossary:
            original = char.get('original_name', '')
            translated = char.get('name', original)
            
            # Add to our glossary if not already present
            if original and translated and original not in glossary_entries:
                glossary_entries[original] = translated
                merged_count += 1
        
        print(f"📑 Added {merged_count} entries from existing glossary")
    
    # Save glossary with metadata
    glossary_data = {
        "metadata": {
            "language": language_hint,
            "names_count": len(sorted_names),
            "titles_count": len(sorted_titles),
            "min_frequency": min_frequency,
            "generation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "merged_with_manual": existing_glossary is not None
        },
        "entries": glossary_entries
    }
    
    glossary_path = os.path.join(output_dir, "glossary.json")
    with open(glossary_path, 'w', encoding='utf-8') as f:
        json.dump(glossary_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n📑 ✅ TARGETED GLOSSARY SAVED!")
    print(f"📑 File: {glossary_path}")
    print(f"📑 Names: {len(sorted_names)}, Titles: {len(sorted_titles)}")
    if existing_glossary:
        print(f"📑 Total entries (including manual): {len(glossary_entries)}")
    
    # For compatibility, also save simple format
    simple_glossary_path = os.path.join(output_dir, "glossary_simple.json")
    with open(simple_glossary_path, 'w', encoding='utf-8') as f:
        json.dump(glossary_entries, f, ensure_ascii=False, indent=2)
    
    return glossary_entries
      
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
    """Build the system prompt with glossary - handles multiple formats"""
    # Check if we should append glossary (default is True if not set)
    append_glossary = os.getenv("APPEND_GLOSSARY", "1") == "1"
    
    # Helper function to format glossary for prompt
    def format_glossary_for_prompt(glossary_data):
        """Convert various glossary formats into a unified prompt format"""
        formatted_entries = {}
        
        # Handle manual glossary format (array of character objects)
        if isinstance(glossary_data, list):
            for char in glossary_data:
                # Extract original and translated names
                original = char.get('original_name', '')
                translated = char.get('name', original)
                if original and translated:
                    formatted_entries[original] = translated
                
                # Also add title with original name if present
                title = char.get('title')
                if title and original:
                    formatted_entries[f"{original} ({title})"] = f"{translated} ({title})"
                
                # Add any how_they_refer_to_others entries
                refer_map = char.get('how_they_refer_to_others', {})
                if isinstance(refer_map, dict):
                    for other_name, reference in refer_map.items():
                        if other_name and reference:
                            # This helps with consistency in dialogue
                            formatted_entries[f"{original} → {other_name}"] = f"{translated} → {reference}"
        
        # Handle automatic glossary format (dict with entries)
        elif isinstance(glossary_data, dict):
            if "entries" in glossary_data:
                # New automatic format with metadata
                formatted_entries = glossary_data["entries"]
            else:
                # Simple dict format
                formatted_entries = glossary_data
        
        return formatted_entries
    
    # Check if system prompt is disabled
    if os.getenv("DISABLE_SYSTEM_PROMPT", "0") == "1":
        # Use only user prompt, but still append glossary if enabled
        system = user_prompt if user_prompt else ""
        
        # Append glossary if the toggle is on
        if append_glossary and os.path.exists(glossary_path):
            try:
                with open(glossary_path, "r", encoding="utf-8") as gf:
                    glossary_data = json.load(gf)
                
                formatted_entries = format_glossary_for_prompt(glossary_data)
                
                if formatted_entries:
                    glossary_block = json.dumps(formatted_entries, ensure_ascii=False, indent=2)
                    if system:
                        system += "\n\n"
                    system += (
                        "Character/Term Glossary (use these translations consistently):\n"
                        f"{glossary_block}"
                    )
            except Exception as e:
                print(f"Warning: Could not load glossary: {e}")
        
        return system
    
    # Normal flow when hardcoded prompts are enabled
    if user_prompt:
        system = user_prompt
        # Append glossary if the toggle is on
        if append_glossary and os.path.exists(glossary_path):
            try:
                with open(glossary_path, "r", encoding="utf-8") as gf:
                    glossary_data = json.load(gf)
                
                formatted_entries = format_glossary_for_prompt(glossary_data)
                
                if formatted_entries:
                    glossary_block = json.dumps(formatted_entries, ensure_ascii=False, indent=2)
                    system += (
                        "\n\nCharacter/Term Glossary (use these translations consistently):\n"
                        f"{glossary_block}"
                    )
            except Exception as e:
                print(f"Warning: Could not load glossary: {e}")
            
    elif os.path.exists(glossary_path) and append_glossary:
        try:
            with open(glossary_path, "r", encoding="utf-8") as gf:
                glossary_data = json.load(gf)
            
            formatted_entries = format_glossary_for_prompt(glossary_data)
            
            if formatted_entries:
                glossary_block = json.dumps(formatted_entries, ensure_ascii=False, indent=2)
                system = (
                    instructions + "\n\n"
                    "Character/Term Glossary (use these translations consistently):\n"
                    f"{glossary_block}"
                )
            else:
                system = instructions
        except Exception as e:
            print(f"Warning: Could not load glossary: {e}")
            system = instructions
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
    
    # Initialize image translator based on toggle only
    image_translator = None
    ENABLE_IMAGE_TRANSLATION = os.getenv("ENABLE_IMAGE_TRANSLATION", "1") == "1"

    if ENABLE_IMAGE_TRANSLATION:
        print(f"🖼️ Image translation enabled for model: {MODEL}")
        print("🖼️ Image translation will use your custom system prompt and glossary")
        # Pass the complete system prompt (including GUI prompt + glossary + instructions)
        image_translator = ImageTranslator(client, out, TRANSLATION_LANG, system, TEMP)
        
        # Optional: Warn about potentially unsupported models
        known_vision_models = [
            'gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-2.0-flash', 'gemini-2.0-flash-exp', 
            'gpt-4-turbo', 'gpt-4o', 'gpt-4.1-mini', 'gpt-4.1-nano', 'o4-mini'
        ]
        
        if MODEL.lower() not in known_vision_models:
            print(f"⚠️ Note: {MODEL} may not have vision capabilities. Image translation will be attempted anyway.")
    else:
        print("ℹ️ Image translation disabled by user")
    
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
        
        # Determine chapter type
        is_image_chapter = c.get('is_image_only', False) or (c.get('has_images', False) and c.get('file_size', 0) < 500)
        is_empty_chapter = c.get('is_empty', False) or c.get('file_size', 0) < 50

        # Handle different chapter types
        if is_empty_chapter and not c.get('has_images'):
            # Empty chapter with no images - just copy it
            print(f"📄 Empty chapter detected - copying as-is")
            # ... existing empty chapter handling ...
            
        elif is_image_chapter:
            # Image-only chapter
            print(f"📸 Image-only chapter detected with {c.get('image_count', 0)} images")
            
            if image_translator and ENABLE_IMAGE_TRANSLATION:
                # Process with image translator
                translated_html, image_translations = process_chapter_images(
                    c["body"], 
                    chap_num, 
                    image_translator,
                    check_stop
                )
            else:
                # Just copy the chapter as-is
                translated_html = c["body"]
                image_translations = {}
            
            # Save the result
            safe_title = make_safe_filename(c['title'], c['num'])   
            fname = f"response_{c['num']:03d}_{safe_title}.html"
            
            with open(os.path.join(out, fname), 'w', encoding='utf-8') as f:
                f.write(translated_html)
            
            if '<div class="image-translation">' in translated_html:
                print(f"[Chapter {idx+1}/{total_chapters}] ✅ Saved image-translated chapter")
                status = "completed"
            else:
                print(f"[Chapter {idx+1}/{total_chapters}] ✅ Saved image-only chapter (no translations)")
                status = "completed_image_only"
            
            # Update progress
            update_progress(prog, idx, chap_num, content_hash, fname, status=status)
            save_progress()
            continue
            
        elif c.get('has_images') and c.get('file_size', 0) >= 500:
            # Mixed content - has both text and images
            print(f"📖 Mixed content chapter with {c.get('image_count', 0)} images and {c.get('file_size', 0)} chars of text")
            
            # First, process any images if image translation is enabled
            chapter_content = c["body"]
            if image_translator and ENABLE_IMAGE_TRANSLATION:
                print(f"   🖼️ Processing images first...")
                chapter_content, image_translations = process_chapter_images(
                    chapter_content, 
                    chap_num, 
                    image_translator,
                    check_stop
                )
                if image_translations:
                    print(f"   ✅ Translated {len(image_translations)} images")
            
            # Now process the text content normally
            # Update c["body"] with the image-processed content
            c["body"] = chapter_content
            
        print(f"📖 Processing as text chapter ({c.get('file_size', 0)} characters)")
        
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
    # FIX: Ensure we have response_ files for EPUB builder
    print("🔍 Checking for translated chapters...")
    response_files = [f for f in os.listdir(out) if f.startswith('response_') and f.endswith('.html')]
    chapter_files = [f for f in os.listdir(out) if f.startswith('chapter_') and f.endswith('.html')]

    if not response_files and chapter_files:
        print(f"⚠️ No translated files found, but {len(chapter_files)} original chapters exist")
        print("📝 Creating placeholder response files for EPUB compilation...")
        
        # Create response files from chapter files
        # These are the ORIGINAL untranslated files, but at least EPUB can build
        for chapter_file in chapter_files:
            response_file = chapter_file.replace('chapter_', 'response_', 1)
            src = os.path.join(out, chapter_file)
            dst = os.path.join(out, response_file)
            
            try:
                # Read the original content
                with open(src, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Add a notice that this is untranslated
                soup = BeautifulSoup(content, 'html.parser')
                notice = soup.new_tag('p')
                notice.string = "[Note: This chapter could not be translated - showing original content]"
                notice['style'] = "color: red; font-style: italic;"
                
                # Insert notice at the beginning of body
                if soup.body:
                    soup.body.insert(0, notice)
                
                # Write as response file
                with open(dst, 'w', encoding='utf-8') as f:
                    f.write(str(soup))
                    
            except Exception as e:
                print(f"⚠️ Error processing {chapter_file}: {e}")
                # Fallback - just copy the file
                try:
                    shutil.copy2(src, dst)
                except:
                    pass
        
        print(f"✅ Created {len(chapter_files)} placeholder response files")
        print("⚠️ Note: The EPUB will contain untranslated content")
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
