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

# Import the new modules
from history_manager import HistoryManager
from chapter_splitter import ChapterSplitter

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

# Modifications for TransateKRtoEN.py

def emergency_restore_paragraphs(text, original_html=None, verbose=True):
    """
    Emergency restoration when AI returns wall of text without proper paragraph tags.
    This function attempts to restore paragraph structure using various heuristics.
    
    Args:
        text: The translated text that may have lost formatting
        original_html: The original HTML to compare structure (optional)
        verbose: Whether to print debug messages (default True)
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
    """Extract metadata from EPUB file"""
    meta = {}
    for n in zf.namelist():
        if n.lower().endswith('.opf'):
            soup = BeautifulSoup(zf.read(n), 'xml')
            for t in ['title','creator','language']:
                e = soup.find(t)
                if e: meta[t] = e.get_text(strip=True)
            break
    return meta

def get_content_hash(html_content):
    """Create a hash of content to detect duplicates"""
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text(strip=True).lower()
    
    # Remove all types of chapter markers for better duplicate detection
    text = re.sub(r'chapter\s*\d+\s*:?\s*', '', text)
    text = re.sub(r'第\s*\d+\s*[章节話话回]', '', text)
    text = re.sub(r'제\s*\d+\s*[장화권부]', '', text)
    text = re.sub(r'第\s*\d+\s*話', '', text)
    text = re.sub(r'\bch\.?\s*\d+\b', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Use first 1000 chars for fingerprint
    fingerprint = text[:1000]
    return hashlib.md5(fingerprint.encode('utf-8')).hexdigest()

def clean_ai_artifacts(text, remove_artifacts=True):
    """Remove AI response artifacts from text - but ONLY when enabled"""
    if not remove_artifacts:
        return text
        
    original_text = text
    
    # 1. Remove [PART X/Y] markers
    text = re.sub(r'^\[PART\s+\d+/\d+\]\s*\n?', '', text, flags=re.IGNORECASE)
    
    # 2. Remove clear AI response patterns
    clear_ai_prefixes = [
        r'^(?:Okay|Sure|Understood|Of course|Got it|Alright|Certainly),?\s+(?:I\'ll|I will|let me|here\'s|here is)\s+(?:translate|help|assist)',
        r'^(?:I\'ll translate|I will translate|Let me translate|Here\'s the translation|Here is the translation)',
        r'^(?:System|Assistant|AI|User|Human|Model)\s*:\s*(?:Okay|Sure|I\'ll|Let me)',
        r'^(?:Note|Translation note|Translator\'s note)\s*:\s*(?:I\'ve|I have|I will|The following)',
    ]
    
    for pattern in clear_ai_prefixes:
        match = re.match(pattern, text.strip(), re.IGNORECASE)
        if match:
            text = text[len(match.group(0)):].strip()
            break
    
    # 3. Remove JSON artifacts
    if text.strip().startswith(('{', '[')) and '"role"' in text[:200]:
        bracket_stack = []
        json_end = -1
        
        for i, char in enumerate(text):
            if char in '{[':
                bracket_stack.append(char)
            elif char == '}' and bracket_stack and bracket_stack[-1] == '{':
                bracket_stack.pop()
                if not bracket_stack:
                    json_end = i
                    break
            elif char == ']' and bracket_stack and bracket_stack[-1] == '[':
                bracket_stack.pop()
                if not bracket_stack:
                    json_end = i
                    break
        
        if json_end > 0 and json_end < len(text) - 1:
            text = text[json_end + 1:].strip()
    
    # 4. Remove markdown code fences
    if text.strip().startswith('```'):
        code_fence_match = re.match(r'^```(?:json|html|xml)?\s*\n(.*?)\n```', text.strip(), re.DOTALL)
        if code_fence_match:
            inner_content = code_fence_match.group(1)
            if '"role"' in inner_content or text.strip().endswith('```'):
                text = inner_content
    
    # 5. Remove glossary JSON arrays
    if text.strip().startswith('[') and '"original_name"' in text[:500] and '"traits"' in text[:500]:
        bracket_count = 0
        for i, char in enumerate(text):
            if char == '[':
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    if i < len(text) - 1:
                        text = text[i + 1:].strip()
                    break
    
    # 6. Check for AI artifacts before HTML headers
    header_patterns = [
        r'<h[1-6][^>]*>',  # HTML headers
        r'Chapter\s+\d+',   # Plain text chapter markers
        r'第\s*\d+\s*[章节話话回]',  # Chinese chapter markers
        r'제\s*\d+\s*[장화]',  # Korean chapter markers
    ]
    
    # Find the first header in the text
    first_header_pos = float('inf')
    for pattern in header_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match and match.start() < first_header_pos:
            first_header_pos = match.start()
    
    # If we found a header and there's text before it
    if first_header_pos < float('inf') and first_header_pos > 0:
        pre_header_text = text[:first_header_pos].strip()
        
        # Check if the pre-header text looks like AI commentary
        ai_commentary_patterns = [
            r'(?:Here\'s|Here is|This is|I\'ve translated)',
            r'(?:The following|Below is|Find below)',
            r'(?:Translation of|Translated from)',
            r'(?:Chapter \d+ of|From the)',
            r'(?:Continuing with|Moving on to)',
        ]
        
        found_ai_commentary = False
        for pattern in ai_commentary_patterns:
            if re.search(pattern, pre_header_text, re.IGNORECASE):
                # Remove everything before the header
                text = text[first_header_pos:]
                print(f"✂️ Removed AI commentary before header: '{pre_header_text[:50]}...'")
                found_ai_commentary = True
                break
        
        # Also check if pre-header text is suspiciously short (likely AI artifact)
        if not found_ai_commentary and len(pre_header_text) < 100 and not re.search(r'<[^>]+>', pre_header_text):
            # Short non-HTML text before header is likely an artifact
            text = text[first_header_pos:]
            print(f"✂️ Removed short text before header: '{pre_header_text}'")
    
    # 7. Remove empty lines at the start
    text = re.sub(r'^\s*\n+', '', text, count=1)
    
    # 8. Remove any remaining "Here's the translation:" type phrases
    simple_ai_phrases = [
        r'^Here\'s the (?:translated |translation[:\s])',
        r'^The translated (?:chapter|text)[:\s]',
        r'^Translation[:\s]',
        r'^Translated[:\s]',
    ]
    
    for phrase in simple_ai_phrases:
        text = re.sub(phrase, '', text.strip(), flags=re.IGNORECASE)
    
    # 9. Final safety check
    if len(text.strip()) < 10 and len(original_text.strip()) > 50:
        text = original_text
        text = re.sub(
            r'^(?:Okay|Sure|Understood),\s+(?:I\'ll|I will)\s+translate[^.!?\n]*?[\n\r]+',
            '', text, flags=re.IGNORECASE, count=1
        )
    
    return text.strip()


def detect_and_wrap_headers(text):
    """
    Detect plain text headers and wrap them in proper HTML header tags.
    This is especially useful for CJK novels where headers might be plain text.
    """
    # Don't process if we already have HTML headers
    if re.search(r'<h[1-6][^>]*>', text[:500]):  # Only check the beginning
        return text
    
    # Check if the first line/element looks like a header
    lines = text.strip().split('\n')
    if not lines:
        return text
    
    first_line = lines[0].strip()
    
    # Remove any HTML tags for analysis
    first_line_text = re.sub(r'<[^>]+>', '', first_line).strip()
    
    # Heuristics to determine if this is likely a header:
    # 1. It's relatively short (headers are usually < 100 chars)
    # 2. It doesn't end with typical sentence punctuation
    # 3. It's followed by a break or significant whitespace
    # 4. It contains chapter/episode markers OR looks like a title
    
    is_likely_header = (
        len(first_line_text) < 100 and
        first_line_text and
        not first_line_text.endswith(('.', '。', '？', '！', '?', '!')) and
        (
            # Has chapter/episode markers
            re.search(r'(?:Chapter|第|제|その|Episode|エピソード|에피소드|Part|Ch\.|C\d+|E\d+|Ep\d+)', first_line_text, re.IGNORECASE) or
            # Has number markers
            re.search(r'(?:^\d+|[-:：]\s*\d+$|\d+\s*[-:：])', first_line_text) or
            # Is followed by HTML breaks
            '<br' in first_line or '<hr' in first_line or
            # Is short and starts with capital/CJK
            (len(first_line_text) < 50 and (first_line_text[0].isupper() or ord(first_line_text[0]) > 127))
        )
    )
    
    if is_likely_header and not first_line.startswith('<h'):
        # Extract just the text content for the header
        header_text = first_line_text
        
        # Reconstruct the text with the header wrapped
        remaining_lines = lines[1:] if len(lines) > 1 else []
        
        # Check if the first line had HTML breaks that should go after the header
        breaks = ''
        if '<br' in first_line or '<hr' in first_line:
            # Extract the breaks
            breaks = re.findall(r'(?:<br\s*/?>|<hr\s*/?>)+', first_line)
            breaks = ''.join(breaks)
        
        new_text = f'<h2>{header_text}</h2>{breaks}'
        if remaining_lines:
            new_text += '\n' + '\n'.join(remaining_lines)
        
        print(f"✅ Wrapped header in HTML: <h2>{header_text[:50]}{'...' if len(header_text) > 50 else ''}</h2>")
        return new_text
    
    return text


def restore_missing_headers(translated_text, original_html):
    """
    Enhanced version that better preserves headers including custom ones.
    """
    # Parse both texts
    original_soup = BeautifulSoup(original_html, 'html.parser')
    trans_soup = BeautifulSoup(translated_text, 'html.parser')
    
    # Find all headers in original
    original_headers = original_soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    if not original_headers:
        return translated_text  # No headers to restore
    
    # Check if translation has any headers
    trans_headers = trans_soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    
    # If original has headers but translation doesn't, we need to restore them
    if original_headers and not trans_headers:
        print("⚠️ Headers missing in translation - attempting restoration")
        
        # Get the first header from original
        first_header = original_headers[0]
        header_tag = first_header.name
        
        # Get the translation text
        trans_text = translated_text.strip()
        lines = trans_text.split('\n')
        
        # Check if the first line looks like it could be a header
        # (short, no HTML tags, not starting with lowercase)
        if lines and len(lines[0]) < 200 and not lines[0].strip().startswith('<') and lines[0].strip() and lines[0].strip()[0].isupper():
            # This first line is probably the header
            header_text = lines[0].strip()
            remaining_lines = lines[1:] if len(lines) > 1 else []
            
            # Reconstruct with proper header tag
            restored = f'<{header_tag}>{header_text}</{header_tag}>\n'
            if remaining_lines:
                restored += '\n'.join(remaining_lines)
            
            print(f"✅ Restored header: <{header_tag}>{header_text[:50]}...</{header_tag}>")
            return restored
        
        # Alternative: Check if first paragraph might actually be the header
        first_p = trans_soup.find('p')
        if first_p:
            p_text = first_p.get_text(strip=True)
            # If it's short and looks like a title (no period at end, titlecase, etc)
            if (len(p_text) < 100 and 
                not p_text.endswith('.') and 
                not p_text.endswith('。') and
                p_text[0].isupper()):
                
                # Convert this paragraph to a header
                first_p.name = header_tag
                print(f"✅ Converted paragraph to header: <{header_tag}>{p_text[:50]}...</{header_tag}>")
                return str(trans_soup)
    
    return translated_text
    
def debug_headers(text, label=""):
    """Debug function to check header presence"""
    soup = BeautifulSoup(text, 'html.parser')
    headers = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    if headers:
        print(f"[DEBUG-HEADERS] {label} - Found {len(headers)} headers:")
        for h in headers[:3]:  # Show first 3
            print(f"  - <{h.name}>{h.get_text(strip=True)[:50]}...</{h.name}>")
    else:
        print(f"[DEBUG-HEADERS] {label} - No headers found!")
        # Check if header text exists but without tags
        text_start = soup.get_text(strip=True)[:200]
        print(f"  - Text start: {text_start[:100]}...")
        
def extract_chapters(zf):
    """Extract chapters from EPUB file with robust duplicate detection"""
    chaps = []
    
    # Get all HTML files and sort them to maintain order
    html_files = sorted([name for name in zf.namelist() 
                        if name.lower().endswith(('.xhtml', '.html'))])
    
    print(f"[DEBUG] Processing {len(html_files)} HTML files from EPUB")
    
    # Track content to avoid duplicates
    content_hashes = {}
    seen_chapters = {}
    chapter_patterns = [
        # English patterns
        (r'chapter[\s_-]*(\d+)', re.IGNORECASE),
        (r'\bch\.?\s*(\d+)\b', re.IGNORECASE),
        (r'part[\s_-]*(\d+)', re.IGNORECASE),
        
        # Chinese patterns
        (r'第\s*(\d+)\s*[章节話话回]', 0),
        (r'第\s*([一二三四五六七八九十百千]+)\s*[章节話话回]', 0),
        (r'(\d+)[章节話话回]', 0),
        
        # Japanese patterns
        (r'第\s*(\d+)\s*話', 0),
        (r'第\s*(\d+)\s*章', 0),
        (r'その\s*(\d+)', 0),
        
        # Korean patterns
        (r'제\s*(\d+)\s*[장화권부]', 0),
        (r'(\d+)\s*[장화권부]', 0),
        
        # Generic patterns
        (r'^\s*(\d+)\s*[-–—.]', re.MULTILINE),
        (r'_(\d+)\.x?html?$', re.IGNORECASE),
        (r'(\d+)', 0),  # Last resort - any number
    ]
    
    # Chinese number conversion
    chinese_nums = {
        '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
        '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
        '十一': 11, '十二': 12, '十三': 13, '十四': 14, '十五': 15,
        '十六': 16, '十七': 17, '十八': 18, '十九': 19, '二十': 20,
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
    
    for idx, name in enumerate(html_files):
        try:
            raw = zf.read(name)
            soup = BeautifulSoup(raw, 'html.parser')
            
            # Get body content
            if soup.body:
                full_body_html = soup.body.decode_contents()
                body_text = soup.body.get_text(strip=True)
            else:
                full_body_html = str(soup)
                body_text = soup.get_text(strip=True)
            
            # Skip empty or very short files
            if len(body_text.strip()) < 100:
                print(f"[DEBUG] Skipping short file: {name} ({len(body_text)} chars)")
                continue
            
            # Create content hash to detect duplicates
            content_hash = get_content_hash(full_body_html)
            
            # Check if we've seen this content before
            if content_hash in content_hashes:
                print(f"[DEBUG] Skipping duplicate content in {name} (matches {content_hashes[content_hash]['filename']})")
                continue
            
            # Try to extract chapter number from various sources
            chapter_num = None
            chapter_title = None
            
            # Method 1: Check filename
            for pattern, flags in chapter_patterns:
                m = re.search(pattern, name, flags)
                if m:
                    try:
                        num_str = m.group(1)
                        if num_str.isdigit():
                            chapter_num = int(num_str)
                        else:
                            # Try Chinese number conversion
                            chapter_num = convert_chinese_number(num_str)
                        
                        if chapter_num:
                            break
                    except:
                        continue
            
            # Method 2: Check content headers
            if not chapter_num:
                for header in soup.find_all(['h1', 'h2', 'h3', 'title']):
                    header_text = header.get_text(strip=True)
                    if not header_text:
                        continue
                    
                    for pattern, flags in chapter_patterns:
                        m = re.search(pattern, header_text, flags)
                        if m:
                            try:
                                num_str = m.group(1)
                                if num_str.isdigit():
                                    chapter_num = int(num_str)
                                else:
                                    chapter_num = convert_chinese_number(num_str)
                                
                                if chapter_num:
                                    chapter_title = header_text
                                    break
                            except:
                                continue
                    
                    if chapter_num:
                        break
            
            # Method 3: Check first few paragraphs
            if not chapter_num:
                first_elements = soup.find_all(['p', 'div'])[:5]
                for elem in first_elements:
                    elem_text = elem.get_text(strip=True)
                    if not elem_text:
                        continue
                    
                    for pattern, flags in chapter_patterns:
                        m = re.search(pattern, elem_text, flags)
                        if m:
                            try:
                                num_str = m.group(1)
                                if num_str.isdigit():
                                    chapter_num = int(num_str)
                                else:
                                    chapter_num = convert_chinese_number(num_str)
                                
                                if chapter_num:
                                    break
                            except:
                                continue
                    
                    if chapter_num:
                        break
            
            # If still no chapter number, assign next available
            if not chapter_num:
                chapter_num = len(chaps) + 1
                while chapter_num in seen_chapters:
                    chapter_num += 1
                print(f"[DEBUG] No chapter number found in {name}, assigning: {chapter_num}")
            
            # Handle duplicate chapter numbers
            if chapter_num in seen_chapters:
                existing_hash = seen_chapters[chapter_num]['hash']
                if existing_hash != content_hash:
                    # Different content with same chapter number
                    original_num = chapter_num
                    while chapter_num in seen_chapters:
                        chapter_num += 1
                    print(f"[WARNING] Chapter {original_num} already exists with different content, reassigning to {chapter_num}")
            
            # Get title
            if not chapter_title:
                # Try to find a title from headers
                for header_tag in ['h1', 'h2', 'h3', 'title']:
                    title_elem = soup.find(header_tag)
                    if title_elem:
                        chapter_title = title_elem.get_text(strip=True)
                        break
                
                if not chapter_title:
                    chapter_title = f"Chapter {chapter_num}"
            
            # Clean and limit title length
            chapter_title = re.sub(r'\s+', ' ', chapter_title).strip()
            if len(chapter_title) > 100:
                chapter_title = chapter_title[:97] + "..."
            
            # Store chapter
            chapter_info = {
                "num": chapter_num,
                "title": chapter_title,
                "body": full_body_html,
                "filename": name,
                "content_hash": content_hash
            }
            
            chaps.append(chapter_info)
            content_hashes[content_hash] = {
                'filename': name,
                'chapter_num': chapter_num
            }
            seen_chapters[chapter_num] = {
                'hash': content_hash,
                'filename': name
            }
            
        except Exception as e:
            print(f"[ERROR] Failed to process {name}: {e}")
            # Add placeholder to maintain chapter sequence
            chapter_info = {
                "num": len(chaps) + 1,
                "title": f"Error: {name}",
                "body": f"<p>Error loading chapter from {name}: {str(e)}</p>",
                "filename": name,
                "content_hash": f"error_{idx}"
            }
            chaps.append(chapter_info)
    
    # Sort by chapter number
    chaps.sort(key=lambda x: x["num"])
    
    # Final validation - check for gaps
    if chaps:
        print(f"[DEBUG] Extracted {len(chaps)} unique chapters")
        print(f"[DEBUG] Chapter range: {chaps[0]['num']} to {chaps[-1]['num']}")
        
        # Check for missing chapters
        expected_chapters = set(range(chaps[0]['num'], chaps[-1]['num'] + 1))
        actual_chapters = set(c['num'] for c in chaps)
        missing = expected_chapters - actual_chapters
        if missing:
            print(f"[WARNING] Missing chapter numbers: {sorted(missing)}")
    
    return chaps

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
        
    # Load or init progress file with enhanced tracking
    PROGRESS_FILE = os.path.join(payloads_dir, "translation_progress.json")
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as pf:
            prog = json.load(pf)
        # Ensure all required fields exist
        if "chapter_chunks" not in prog:
            prog["chapter_chunks"] = {}
        if "content_hashes" not in prog:
            prog["content_hashes"] = {}
        if "chapter_metadata" not in prog:
            prog["chapter_metadata"] = {}
    else:
        prog = {
            "completed": [],
            "chapter_chunks": {},
            "content_hashes": {},
            "chapter_metadata": {}
        }

    def save_progress():
        with open(PROGRESS_FILE, "w", encoding="utf-8") as pf:
            json.dump(prog, pf, ensure_ascii=False, indent=2)

    # Check for stop before starting
    if check_stop():
        return

    # Extract EPUB contents
    with zipfile.ZipFile(epub_path, 'r') as zf:
        metadata = extract_epub_metadata(zf)
        chapters = extract_chapters(zf)

        # Validate chapters
        validate_chapter_continuity(chapters)


        # Extract images
        imgdir = os.path.join(out, "images")
        os.makedirs(imgdir, exist_ok=True)
        for n in zf.namelist():
            if n.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.svg')):
                with open(os.path.join(imgdir, os.path.basename(n)), 'wb') as f:
                    f.write(zf.read(n))

    # Check for stop after file processing
    if check_stop():
        return

    # Write metadata with chapter info
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
        save_glossary(out, chapters, instructions, TRANSLATION_LANG)
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
        
        # Apply chapter range filter
        if start is not None and not (start <= chap_num <= end):
            continue
        
        # Check if content was already translated (duplicate detection)
        content_hash = c.get("content_hash") or get_content_hash(c["body"])
        if content_hash in prog["content_hashes"]:
            existing = prog["content_hashes"][content_hash]
            if existing.get("completed_idx") in prog["completed"]:
                print(f"[SKIP] Chapter {chap_num} has same content as already translated chapter {existing.get('chapter_num')}")
                chunks_per_chapter[idx] = 0
                continue
            
        # Skip already completed chapters
        if idx in prog["completed"]:
            chunks_per_chapter[idx] = 0
            continue
        
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
        chapter_key = str(idx)
        if chapter_key in prog.get("chapter_chunks", {}):
            completed_chunks = len(prog["chapter_chunks"][chapter_key].get("completed", []))
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

        # Check for duplicate content
        if content_hash in prog["content_hashes"]:
            existing = prog["content_hashes"][content_hash]
            if existing.get("completed_idx") in prog["completed"]:
                print(f"[SKIP] Chapter {chap_num} is duplicate of already translated chapter {existing.get('chapter_num')}")
                continue

        # Skip already completed chapters
        if idx in prog["completed"]:
            print(f"[SKIP] Chapter #{idx+1} (EPUB-num {chap_num}) already done, skipping.")
            continue

        print(f"\n🔄 Processing Chapter {idx+1}/{total_chapters}: {c['title']}")

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
        chapter_key = str(idx)
        if chapter_key not in prog["chapter_chunks"]:
            prog["chapter_chunks"][chapter_key] = {
                "total": len(chunks),
                "completed": [],
                "chunks": {}
            }
        
        prog["chapter_chunks"][chapter_key]["total"] = len(chunks)
        
        translated_chunks = []
        
        # Process each chunk
        for chunk_html, chunk_idx, total_chunks in chunks:
            # Check if this chunk was already translated
            if chunk_idx in prog["chapter_chunks"][chapter_key]["completed"]:
                saved_chunk = prog["chapter_chunks"][chapter_key]["chunks"].get(str(chunk_idx))
                if saved_chunk:
                    translated_chunks.append((saved_chunk, chunk_idx, total_chunks))
                    print(f"  [SKIP] Chunk {chunk_idx}/{total_chunks} already translated")
                    continue
            # NEW: Check if history will reset on this chapter
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
                
            # Build messages
            if base_msg:
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
                    result, finish_reason = send_with_interrupt(
                        msgs, client, TEMP, MAX_OUTPUT_TOKENS, check_stop
                    )
                    
                    if finish_reason == "length":
                        print(f"    [WARN] Output was truncated!")
                    
                    # Clean AI artifacts ONLY if the toggle is enabled
                    if REMOVE_AI_ARTIFACTS:
                        result = clean_ai_artifacts(result)
                    
                    if EMERGENCY_RESTORE:
                        result = emergency_restore_paragraphs(result, chunk_html)
                    
                    # Additional cleaning if remove header is enabled
                    if REMOVE_AI_ARTIFACTS:
                        # Remove any remaining JSON or artifacts
                        if result.strip().startswith('{') or result.strip().startswith('['):
                            # Find first non-JSON line
                            lines = result.split('\n')
                            for i, line in enumerate(lines):
                                if line.strip() and not any(char in line for char in ['{', '}', '[', ']', '"role"', '"content"']):
                                    result = '\n'.join(lines[i:])
                                    break
                    
                    # Remove chunk markers if present
                    result = re.sub(r'\[PART \d+/\d+\]\s*', '', result, flags=re.IGNORECASE)
                    
                    # Save chunk result
                    translated_chunks.append((result, chunk_idx, total_chunks))
                    
                    # Update progress for this chunk
                    prog["chapter_chunks"][chapter_key]["completed"].append(chunk_idx)
                    prog["chapter_chunks"][chapter_key]["chunks"][str(chunk_idx)] = result
                    save_progress()
                    
                    # Increment completed chunks counter
                    chunks_completed += 1
                        
                    # Update history using thread-safe manager with reset functionality
                    history = history_manager.append_to_history(
                        user_prompt, 
                        result, 
                        HIST_LIMIT if CONTEXTUAL else 0,  # 0 means no history
                        reset_on_limit=True
                    )

                    # Check if history was reset (will be only 2 entries if just reset)
                    history_trimmed = len(history) == 2 and HIST_LIMIT > 0

                    if history_trimmed:
                        print(f"    [DBG] History was reset to maintain prompt adherence")

                    # Handle rolling summary if enabled and history was reset
                    if history_trimmed and os.getenv("USE_ROLLING_SUMMARY", "0") == "1":
                        if check_stop():
                            print(f"❌ Translation stopped during summary generation for chapter {idx+1}")
                            return
                            
                        # Generate summary
                        recent_entries = [h for h in history[-2:] if h["role"] == "assistant"]
                        summary_prompt = (
                            "Summarize the key events, tone, and terminology used in the following translation.\n"
                            "The summary will be used to maintain consistency in the next chapter.\n\n"
                            + "\n\n".join(e["content"] for e in recent_entries)
                        )

                        summary_msgs = [
                            {"role": "system", "content": "You are a summarizer."},
                            {"role": "user", "content": summary_prompt}
                        ]
                        
                        summary_resp, _ = send_with_interrupt(
                            summary_msgs, client, TEMP, MAX_OUTPUT_TOKENS, check_stop
                        )

                        # Save summary
                        summary_file = os.path.join(out, "rolling_summary.txt")
                        with open(summary_file, "w", encoding="utf-8") as sf:
                            sf.write(summary_resp.strip())

                        # Inject summary into base prompt
                        base_msg.insert(1, {
                            "role": os.getenv("SUMMARY_ROLE", "user"),
                            "content": (
                                "Here is a concise summary of the previous chapter. "
                                "Use this to ensure accurate tone, terminology, and character continuity:\n\n"
                                f"{summary_resp.strip()}"
                            )
                        })

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
        safe_title = re.sub(r'\W+', '_', c['title'])[:40]
        fname = f"response_{c['num']:03d}_{safe_title}.html"

        # Clean up code fences
        cleaned = re.sub(r"^```(?:html)?\s*", "", merged_result, flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE)

        # Final artifact cleanup - NOW RESPECTS THE TOGGLE
        cleaned = clean_ai_artifacts(cleaned, remove_artifacts=REMOVE_AI_ARTIFACTS)

        # Debug the final result
        if idx < 3:  # Debug first 3 chapters
            debug_headers(cleaned, f"Chapter {idx+1} - Final before save")

        # Write HTML file
        with open(os.path.join(out, fname), 'w', encoding='utf-8') as f:
            f.write(cleaned)
        
        final_title = c['title'] or safe_title
        print(f"[Chapter {idx+1}/{total_chapters}] ✅ Saved Chapter {c['num']}: {final_title}")
        
        # Record completion with content tracking
        prog["completed"].append(idx)
        prog["content_hashes"][content_hash] = {
            "chapter_num": c["num"],
            "completed_idx": idx,
            "filename": fname
        }
        
        # Save chapter metadata
        if "chapter_metadata" not in prog:
            prog["chapter_metadata"] = {}
        
        prog["chapter_metadata"][str(c["num"])] = {
            "title": c["title"],
            "filename": fname,
            "content_hash": content_hash,
            "processed_idx": idx
        }
        
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
            
    except Exception as e:
        print("❌ EPUB build failed:", e)

    # Signal completion to GUI
    print("TRANSLATION_COMPLETE_SIGNAL")

if __name__ == "__main__":
    main()
