#!/usr/bin/env python3
"""
EPUB Converter - Compiles translated HTML files into EPUB format
Supports extraction of translated titles from chapter content
"""

import os
import sys
import io
import json
import mimetypes
import re
import zipfile
import unicodedata
import html as html_module
from xml.etree import ElementTree as ET
from typing import Dict, List, Tuple, Optional, Callable

from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup
from metadata_batch_translator import enhance_epub_compiler
try:
    from unified_api_client import UnifiedClient
except ImportError:
    UnifiedClient = None

# Configure stdout for UTF-8
def configure_utf8_output():
    """Configure stdout for UTF-8 encoding"""
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


# Global configuration
configure_utf8_output()
_global_log_callback = None


def set_global_log_callback(callback: Optional[Callable]):
    """Set the global log callback for module-level functions"""
    global _global_log_callback
    _global_log_callback = callback


def log(message: str):
    """Module-level logging that works with or without callback"""
    if _global_log_callback:
        _global_log_callback(message)
    else:
        print(message)


class HTMLEntityDecoder:
    """Handles comprehensive HTML entity decoding with full Unicode support"""
    
    # Comprehensive entity replacement dictionary
    ENTITY_MAP = {
        # Quotation marks and apostrophes
        '&quot;': '"', '&QUOT;': '"',
        '&apos;': "'", '&APOS;': "'",
        '&lsquo;': '\u2018', '&rsquo;': '\u2019',
        '&ldquo;': '\u201c', '&rdquo;': '\u201d',
        '&sbquo;': '‚', '&bdquo;': '„',
        '&lsaquo;': '‹', '&rsaquo;': '›',
        '&laquo;': '«', '&raquo;': '»',
        
        # Spaces and dashes
        '&nbsp;': ' ', '&NBSP;': ' ',
        '&ensp;': ' ', '&emsp;': ' ',
        '&thinsp;': ' ', '&zwnj;': '\u200c',
        '&zwj;': '\u200d', '&lrm;': '\u200e',
        '&rlm;': '\u200f',
        '&ndash;': '–', '&mdash;': '—',
        '&minus;': '−', '&hyphen;': '‐',
        
        # Common symbols
        '&hellip;': '…', '&mldr;': '…',
        '&bull;': '•', '&bullet;': '•',
        '&middot;': '·', '&centerdot;': '·',
        '&sect;': '§', '&para;': '¶',
        '&dagger;': '†', '&Dagger;': '‡',
        '&loz;': '◊', '&diams;': '♦',
        '&clubs;': '♣', '&hearts;': '♥',
        '&spades;': '♠',
        
        # Currency symbols
        '&cent;': '¢', '&pound;': '£',
        '&yen;': '¥', '&euro;': '€',
        '&curren;': '¤',
        
        # Mathematical symbols
        '&plusmn;': '±', '&times;': '×',
        '&divide;': '÷', '&frasl;': '⁄',
        '&permil;': '‰', '&pertenk;': '‱',
        '&prime;': '\u2032', '&Prime;': '\u2033',
        '&infin;': '∞', '&empty;': '∅',
        '&nabla;': '∇', '&partial;': '∂',
        '&sum;': '∑', '&prod;': '∏',
        '&int;': '∫', '&radic;': '√',
        '&asymp;': '≈', '&ne;': '≠',
        '&equiv;': '≡', '&le;': '≤',
        '&ge;': '≥', '&sub;': '⊂',
        '&sup;': '⊃', '&nsub;': '⊄',
        '&sube;': '⊆', '&supe;': '⊇',
        
        # Intellectual property
        '&copy;': '©', '&COPY;': '©',
        '&reg;': '®', '&REG;': '®',
        '&trade;': '™', '&TRADE;': '™',
    }
    
    # Common encoding fixes
    ENCODING_FIXES = {
        # UTF-8 decoded as Latin-1
        'Ã¢â‚¬â„¢': "'", 'Ã¢â‚¬Å"': '"', 'Ã¢â‚¬ï¿½': '"',
        'Ã¢â‚¬â€œ': '–', 'Ã¢â‚¬â€': '—',
        'Ã‚Â ': ' ', 'Ã‚Â': '', 
        'ÃƒÂ¢': 'â', 'ÃƒÂ©': 'é', 'ÃƒÂ¨': 'è',
        'ÃƒÂ¤': 'ä', 'ÃƒÂ¶': 'ö', 'ÃƒÂ¼': 'ü',
        'ÃƒÂ±': 'ñ', 'ÃƒÂ§': 'ç',
        # Common mojibake patterns
        'â€™': "'", 'â€œ': '"', 'â€': '"',
        'â€"': '—', 'â€"': '–',
        'â€¦': '…', 'â€¢': '•',
        'â„¢': '™', 'Â©': '©', 'Â®': '®',
        # Windows-1252 interpreted as UTF-8
        'â€˜': '\u2018', 'â€™': '\u2019', 
        'â€œ': '\u201c', 'â€': '\u201d',
        'â€¢': '•', 'â€"': '–', 'â€"': '—',
    }
    
    @classmethod
    def decode(cls, text: str) -> str:
        """Comprehensive HTML entity decoding - PRESERVES UNICODE"""
        if not text:
            return text
        
        # Fix common encoding issues first
        for bad, good in cls.ENCODING_FIXES.items():
            text = text.replace(bad, good)
        
        # Multiple passes to handle nested/double-encoded entities
        max_passes = 3
        for _ in range(max_passes):
            prev_text = text
            
            # Use html module for standard decoding
            text = html_module.unescape(text)
            
            # Handle entities without semicolons
            text = re.sub(r'&(nbsp|quot|amp|lt|gt|apos)(?!;)', r'&\1;', text)
            text = re.sub(r'&#(\d{1,7})(?!;)', r'&#\1;', text)
            text = re.sub(r'&#x([0-9a-fA-F]{1,6})(?!;)', r'&#x\1;', text)
            
            # Second pass
            text = html_module.unescape(text)
            
            # Handle numeric entities manually
            text = re.sub(r'&#(\d+);?', cls._decode_decimal, text)
            text = re.sub(r'&#[xX]([0-9a-fA-F]+);?', cls._decode_hex, text)
            
            if text == prev_text:
                break
        
        # Apply entity replacements
        for entity, char in cls.ENTITY_MAP.items():
            text = text.replace(entity, char)
            text = text.replace(entity.lower(), char)
            text = text.replace(entity.upper(), char)
        
        # Final cleanup
        text = re.sub(r'&(?!(?:[a-zA-Z][a-zA-Z0-9]*|#[0-9]+|#x[0-9a-fA-F]+);)', '&amp;', text)
        
        return text
    
    @staticmethod
    def _decode_decimal(match):
        """Decode decimal HTML entity"""
        try:
            code = int(match.group(1))
            if XMLValidator.is_valid_char_code(code):
                return chr(code)
        except:
            pass
        return match.group(0)
    
    @staticmethod
    def _decode_hex(match):
        """Decode hexadecimal HTML entity"""
        try:
            code = int(match.group(1), 16)
            if XMLValidator.is_valid_char_code(code):
                return chr(code)
        except:
            pass
        return match.group(0)


class XMLValidator:
    """Handles XML validation and character checking"""
    
    @staticmethod
    def is_valid_char_code(codepoint: int) -> bool:
        """Check if a codepoint is valid for XML"""
        return (
            codepoint == 0x9 or 
            codepoint == 0xA or 
            codepoint == 0xD or 
            (0x20 <= codepoint <= 0xD7FF) or 
            (0xE000 <= codepoint <= 0xFFFD) or 
            (0x10000 <= codepoint <= 0x10FFFF)
        )
    
    @staticmethod
    def is_valid_char(c: str) -> bool:
        """Check if a character is valid for XML"""
        return XMLValidator.is_valid_char_code(ord(c))
    
    @staticmethod
    def clean_for_xml(text: str) -> str:
        """Remove invalid XML characters"""
        return ''.join(c for c in text if XMLValidator.is_valid_char(c))


class ContentProcessor:
    """Handles content cleaning and processing - UPDATED WITH UNICODE PRESERVATION"""
    
    @staticmethod
    def safe_escape(text: str) -> str:
        """Escape XML special characters"""
        if not text:
            return ''
        
        # Only escape the 5 XML special characters
        text = str(text)
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        text = text.replace('"', '&quot;')
        text = text.replace("'", '&apos;')
        
        return text
    
    @staticmethod
    def fix_malformed_tags(html_content: str) -> str:
        """Fix various types of malformed tags - ENHANCED WITH BEAUTIFULSOUP"""
        # Fix common entity issues
        html_content = html_content.replace('&&', '&amp;&amp;')
        html_content = html_content.replace('& ', '&amp; ')
        html_content = html_content.replace(' & ', ' &amp; ')
        
        # Fix unescaped ampersands more thoroughly
        html_content = re.sub(
            r'&(?!(?:'
            r'amp|lt|gt|quot|apos|'
            r'[a-zA-Z][a-zA-Z0-9]{0,30}|'
            r'#[0-9]{1,7}|'
            r'#x[0-9a-fA-F]{1,6}'
            r');)',
            '&amp;',
            html_content
        )
        
        # Remove completely broken tags with ="" patterns
        html_content = re.sub(r'<[^>]*?=\s*""\s*[^>]*?=\s*""[^>]*?>', '', html_content)
        html_content = re.sub(r'<[^>]*?="""[^>]*?>', '', html_content)
        html_content = re.sub(r'<[^>]*?="{3,}[^>]*?>', '', html_content)
        
        # Fix skill/ability/spell tags - just remove them
        game_tags = r'skill|ability|spell|detect|status|class|level|stat|buff|debuff|item|quest'
        html_content = re.sub(rf'<({game_tags})\b[^>]*?>', '', html_content, flags=re.IGNORECASE)
        html_content = re.sub(rf'</({game_tags})>', '', html_content, flags=re.IGNORECASE)
        
        # Don't convert title tags to [title] format - that's causing the display issue
        # Just remove broken title tags
        html_content = re.sub(r'<title\s*=\s*""\s*[^>]*?>', '', html_content)
        
        # Fix attributes without quotes
        html_content = re.sub(r'<(\w+)\s+(\w+)=([^\s"\'>]+)(\s|>)', r'<\1 \2="\3"\4', html_content)
        
        # Fix unclosed tags at end
        if html_content.rstrip().endswith('<'):
            html_content = html_content.rstrip()[:-1]
        
        # NEW: Fix mismatched tags using BeautifulSoup
        try:
            soup = BeautifulSoup(html_content, 'html.parser', from_encoding='utf-8')
            
            # Fix nested paragraphs (not allowed in XHTML)
            for p in soup.find_all('p'):
                nested_ps = p.find_all('p')
                for nested_p in nested_ps:
                    nested_p.unwrap()
            
            # Remove empty tags that might cause issues
            for tag in soup.find_all():
                if not tag.get_text(strip=True) and not tag.find_all() and tag.name not in ['br', 'hr', 'img', 'meta', 'link']:
                    tag.decompose()
            
            html_content = str(soup)
        except:
            pass
        
        return html_content
    
    @staticmethod
    def fix_self_closing_tags(content: str) -> str:
        """Fix self-closing tags for XHTML compliance"""
        void_elements = ['area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input',
                        'link', 'meta', 'param', 'source', 'track', 'wbr']
        
        for tag in void_elements:
            # Fix simple tags
            content = re.sub(f'<{tag}>', f'<{tag} />', content)
            
            # Fix tags with attributes - IMPROVED REGEX
            content = re.sub(f'<{tag}(\\s+[^/>]*?)(?<!/)>', f'<{tag}\\1 />', content)
            
            # Remove any closing tags for void elements
            content = re.sub(f'</{tag}>', '', content)
        
        return content
    
    @staticmethod
    def clean_chapter_content(html_content: str) -> str:
        """Clean and prepare chapter content for XHTML conversion - PRESERVES UNICODE"""
        # First, remove any [tag] patterns that might have been created
        html_content = re.sub(r'\[(title|skill|ability|spell|detect|status|class|level|stat|buff|debuff|item|quest)[^\]]*?\]', '', html_content)
        
        # Fix any smart quotes that might be in the content
        # Using Unicode escape sequences to avoid parsing issues
        html_content = re.sub(r'[\u201c\u201d\u2018\u2019\u201a\u201e]', '"', html_content)
        html_content = re.sub(r'[\u2018\u2019\u0027]', "'", html_content)
        
        # Decode entities first - NOW PRESERVES UNICODE
        html_content = HTMLEntityDecoder.decode(html_content)
        
        # Remove XML declarations and DOCTYPE (we'll add clean ones later)
        html_content = re.sub(r'<\?xml[^>]*\?>', '', html_content, flags=re.IGNORECASE)
        html_content = re.sub(r'<!DOCTYPE[^>]*>', '', html_content, flags=re.IGNORECASE)
        html_content = re.sub(r'<html[^>]*>', '<html>', html_content, flags=re.IGNORECASE)
        
        # Remove control characters
        html_content = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', html_content)
        
        # Remove Unicode control characters
        control_chars = ''.join(
            chr(i) for i in range(0x00, 0x20) if i not in [0x09, 0x0A, 0x0D]
        ) + ''.join(chr(i) for i in range(0x7F, 0xA0))
        
        translator = str.maketrans('', '', control_chars)
        html_content = html_content.translate(translator)
        
        # Fix malformed tags
        html_content = ContentProcessor.fix_malformed_tags(html_content)
        
        # Remove BOM
        if html_content.startswith('\ufeff'):
            html_content = html_content[1:]
        
        # Clean for XML - PRESERVES VALID UNICODE
        html_content = XMLValidator.clean_for_xml(html_content)
        
        # Normalize line endings
        html_content = html_content.replace('\r\n', '\n').replace('\r', '\n')
        
        # DO NOT convert Unicode to XML character references
        # Keep characters like "同期 (dōki)" as-is
        
        return html_content


class TitleExtractor:
    """Handles extraction of titles from HTML content - UPDATED WITH UNICODE PRESERVATION"""
    
    @staticmethod
    def extract_from_html(html_content: str, chapter_num: Optional[int] = None, 
                         filename: Optional[str] = None) -> Tuple[str, float]:
        """Extract title from HTML content with confidence score"""
        try:
            # Decode entities first - PRESERVES UNICODE
            html_content = HTMLEntityDecoder.decode(html_content)
            
            soup = BeautifulSoup(html_content, 'html.parser', from_encoding='utf-8')
            candidates = []
            
            # Strategy 1: <title> tag
            title_tag = soup.find('title')
            if title_tag and title_tag.string:
                title_text = HTMLEntityDecoder.decode(title_tag.string.strip())
                if title_text and len(title_text) > 1 and title_text.lower() not in ['untitled', 'chapter']:
                    candidates.append((title_text, 0.95, "title_tag"))
            
            # Strategy 2-5: Various heading tags
            strategies = [
                ('h1', 0.9, 300),
                ('h2', 0.8, 250),
                ('h3', 0.7, 200),
            ]
            
            for tag_name, confidence, max_len in strategies:
                for tag in soup.find_all(tag_name):
                    text = HTMLEntityDecoder.decode(tag.get_text(strip=True))
                    if text and len(text) < max_len:
                        candidates.append((text, confidence, f"{tag_name}_tag"))
            
            # Strategy 6: Bold text in first elements
            first_elements = soup.find_all(['p', 'div'])[:5]
            for elem in first_elements:
                for bold in elem.find_all(['b', 'strong']):
                    bold_text = HTMLEntityDecoder.decode(bold.get_text(strip=True))
                    if bold_text and 3 <= len(bold_text) <= 150:
                        candidates.append((bold_text, 0.6, "bold_text"))
            
            # Strategy 7: Patterns in first paragraph
            first_p = soup.find('p')
            if first_p:
                p_text = HTMLEntityDecoder.decode(first_p.get_text(strip=True))
                # Using Unicode escape sequences for various dash types
                chapter_pattern = re.match(
                    r'^(Chapter\s+\d+\s*[:\-\u2013\u2014\u2015\u2012\u2e3a\u2e3b\u301c\u3030\ufe58\ufe63\uff0d\u2010\u2011\u2043\s]*)\s*(.{3,100})(?:\.|$|Chapter)',
                    p_text, re.IGNORECASE
                )
                if chapter_pattern:
                    title_part = chapter_pattern.group(2).strip()
                    if title_part and len(title_part) >= 3:
                        candidates.append((title_part, 0.8, "paragraph_pattern"))
                elif len(p_text) <= 100:
                    candidates.append((p_text, 0.4, "paragraph_standalone"))
            
            # Strategy 8: Filename
            if filename:
                filename_match = re.search(r'response_\d+_(.+?)\.html', filename)
                if filename_match:
                    filename_title = filename_match.group(1).replace('_', ' ').title()
                    if len(filename_title) > 2:
                        candidates.append((filename_title, 0.3, "filename"))
            
            # Filter and rank candidates
            if candidates:
                unique_candidates = {}
                for title, confidence, source in candidates:
                    title = TitleExtractor.clean_title(title)
                    if title and len(title) > 2 and TitleExtractor.is_valid_title(title):
                        if title not in unique_candidates or unique_candidates[title][1] < confidence:
                            unique_candidates[title] = (title, confidence, source)
                
                if unique_candidates:
                    sorted_candidates = sorted(unique_candidates.values(), key=lambda x: x[1], reverse=True)
                    return sorted_candidates[0][0], sorted_candidates[0][1]
            
            # Fallback
            if chapter_num:
                return f"Chapter {chapter_num}", 0.1
            return "Untitled Chapter", 0.0
            
        except Exception as e:
            log(f"[WARNING] Error extracting title: {e}")
            if chapter_num:
                return f"Chapter {chapter_num}", 0.1
            return "Untitled Chapter", 0.0
    
    @staticmethod
    def clean_title(title: str) -> str:
        """Clean and normalize extracted title - PRESERVES UNICODE"""
        if not title:
            return ""
        
        # Remove any [tag] patterns first
        title = re.sub(r'\[(title|skill|ability|spell|detect|status|class|level|stat|buff|debuff|item|quest)[^\]]*?\]', '', title)
        
        # Decode entities - PRESERVES UNICODE
        title = HTMLEntityDecoder.decode(title)
        
        # Remove HTML tags
        title = re.sub(r'<[^>]+>', '', title)
        
        # Normalize spaces
        title = re.sub(r'[\xa0\u2000-\u200a\u202f\u205f\u3000]+', ' ', title)
        title = re.sub(r'\s+', ' ', title).strip()
        
        # Remove leading/trailing punctuation
        # Using Unicode escape sequences for special characters
        # Brackets placed differently to avoid escaping issues
        title = re.sub(r'^[][(){}\s\-\u2013\u2014\u2015\u2012\u2e3a\u2e3b\u301c\u3030\ufe58\ufe63\uff0d\u2010\u2011\u2043:;,.|/\\]+', '', title).strip()
        
        # Remove quotes if they wrap the entire title
        quote_pairs = [
            ('"', '"'), ("'", "'"),
            ('\u201c', '\u201d'), ('\u2018', '\u2019'),  # Smart quotes
            ('«', '»'), ('‹', '›'),  # Guillemets
            ('「', '」'), ('『', '』'),  # Japanese quotes
            ('《', '》'), ('〈', '〉'),  # Chinese quotes
            ('【', '】'), ('〖', '〗'),  # Asian brackets
        ]
        
        for open_q, close_q in quote_pairs:
            if title.startswith(open_q) and title.endswith(close_q):
                title = title[len(open_q):-len(close_q)].strip()
                break
        
        # Normalize Unicode - PRESERVES READABILITY
        title = unicodedata.normalize('NFC', title)
        
        # Remove zero-width characters
        title = re.sub(r'[\u200b\u200c\u200d\u200e\u200f\ufeff]', '', title)
        
        # Final cleanup
        title = ' '.join(title.split())
        
        # Truncate if too long
        if len(title) > 150:
            truncated = title[:147]
            last_space = truncated.rfind(' ')
            if last_space > 100:
                truncated = truncated[:last_space]
            title = truncated + "..."
        
        return title
    
    @staticmethod
    def is_valid_title(title: str) -> bool:
        """Check if extracted title is valid"""
        if not title or len(title) < 2:
            return False
        
        # Check invalid patterns
        invalid_patterns = [
            r'^\d+$',  # Just numbers
            r'^[^\w\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]+$',  # Just punctuation
            r'^(untitled)$',  # Just "untitled"
        ]
        
        for pattern in invalid_patterns:
            if re.match(pattern, title.lower().strip()):
                return False
        
        # Skip filler phrases
        filler_phrases = [
            'click here', 'read more', 'continue reading', 'next chapter',
            'previous chapter', 'table of contents', 'back to top'
        ]
        
        title_lower = title.lower().strip()
        if any(phrase in title_lower for phrase in filler_phrases):
            return False
        
        return True


class XHTMLConverter:
    """Handles XHTML conversion and compliance - UPDATED WITH UNICODE PRESERVATION"""
    
    @staticmethod
    def ensure_compliance(html_content: str, title: str = "Chapter", 
                         css_links: Optional[List[str]] = None) -> str:
        """Ensure HTML content is XHTML-compliant"""
        try:
            # Clean content first
            html_content = ContentProcessor.clean_chapter_content(html_content)
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser', from_encoding='utf-8')
            
            # Extract title if available
            if soup.title and soup.title.string:
                title = str(soup.title.string).strip()
            elif soup.h1:
                title = soup.h1.get_text(strip=True)
            elif soup.h2:
                title = soup.h2.get_text(strip=True)
            
            # Extract CSS links if needed
            if css_links is None and soup.head:
                css_links = []
                for link in soup.head.find_all('link', rel='stylesheet'):
                    href = link.get('href', '')
                    if href:
                        css_links.append(href)
            
            # Extract body content
            body_content = XHTMLConverter._extract_body_content(soup)
            
            # Build proper XHTML
            return XHTMLConverter._build_xhtml(title, body_content, css_links)
            
        except Exception as e:
            log(f"[WARNING] Failed to ensure XHTML compliance: {e}")
            # Return a minimal valid XHTML
            return XHTMLConverter._build_fallback_xhtml(title)
    
    @staticmethod
    def _extract_body_content(soup) -> str:
        """Extract body content from BeautifulSoup object"""
        if soup.body:
            # Remove any [tag] text that might have been created by malformed tag fixing
            for text in soup.body.find_all(text=True):
                if re.match(r'^\[(title|skill|ability|spell|detect|status|class|level|stat|buff|debuff|item|quest)\]', str(text)):
                    text.replace_with('')
            
            # Fix img tags in body
            for img in soup.body.find_all('img'):
                if not img.get('alt'):
                    img['alt'] = ''
                if not img.get('src'):
                    img.decompose()
            
            # Get body HTML
            body_parts = []
            for child in soup.body.children:
                child_str = str(child)
                # Remove any remaining [tag] patterns
                child_str = re.sub(r'\[(title|skill|ability|spell|detect|status|class|level|stat|buff|debuff|item|quest)[^\]]*?\]', '', child_str)
                # Fix self-closing tags
                child_str = ContentProcessor.fix_self_closing_tags(child_str)
                body_parts.append(child_str)
            
            return '\n'.join(body_parts)
        else:
            # No body tag - get all content
            for tag in soup(['script', 'style']):
                tag.decompose()
            
            # Fix img tags
            for img in soup.find_all('img'):
                if not img.get('alt'):
                    img['alt'] = ''
                if not img.get('src'):
                    img.decompose()
            
            content = str(soup)
            # Remove any [tag] patterns
            content = re.sub(r'\[(title|skill|ability|spell|detect|status|class|level|stat|buff|debuff|item|quest)[^\]]*?\]', '', content)
            return ContentProcessor.fix_self_closing_tags(content)
    
    @staticmethod
    def _build_xhtml(title: str, body_content: str, css_links: Optional[List[str]] = None) -> str:
        """Build XHTML document - PRESERVES UNICODE"""
        if not body_content.strip():
            body_content = '<p>Empty chapter</p>'
        
        # Ensure title is XML-safe (but preserves Unicode)
        title = ContentProcessor.safe_escape(title)
        
        # Ensure body content is XML-safe - PRESERVES UNICODE
        body_content = XHTMLConverter._ensure_xml_safe_readable(body_content)
        
        # Ensure we use standard ASCII quotes and no hidden characters
        xml_declaration = '<?xml version="1.0" encoding="utf-8"?>'
        doctype = '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">'
        
        xhtml_parts = [
            xml_declaration,
            doctype,
            '<html xmlns="http://www.w3.org/1999/xhtml">',
            '<head>',
            '<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />',
            f'<title>{title}</title>'
        ]
        
        # Add CSS links
        if css_links:
            for css_link in css_links:
                if css_link.startswith('<link'):
                    href_match = re.search(r'href="([^"]+)"', css_link)
                    if href_match:
                        css_link = href_match.group(1)
                    else:
                        continue
                xhtml_parts.append(f'<link rel="stylesheet" type="text/css" href="{ContentProcessor.safe_escape(css_link)}" />')
        
        xhtml_parts.extend([
            '</head>',
            '<body>',
            body_content,
            '</body>',
            '</html>'
        ])
        
        return '\n'.join(xhtml_parts)
    
    @staticmethod
    def _ensure_xml_safe_readable(content: str) -> str:
        """Ensure content is XML-safe while keeping full readability"""
        try:
            soup = BeautifulSoup(content, 'html.parser')
            
            # BeautifulSoup automatically handles XML escaping in text nodes
            # while preserving Unicode characters
            result = str(soup)
            
            # Ensure self-closing tags are properly formatted
            result = ContentProcessor.fix_self_closing_tags(result)
            
            return result
        except:
            # Fallback: basic XML escaping
            return XHTMLConverter._basic_xml_escape(content)
    
    @staticmethod
    def _basic_xml_escape(content: str) -> str:
        """Basic XML escaping as fallback - PRESERVES UNICODE"""
        # Protect existing entities
        entities = []
        placeholder = "###ENTITY###"
        
        def protect_entity(match):
            entities.append(match.group(0))
            return f"{placeholder}{len(entities)-1}###"
        
        # Protect valid entities
        content = re.sub(
            r'&(?:[a-zA-Z][a-zA-Z0-9]*|#[0-9]+|#x[0-9a-fA-F]+);',
            protect_entity,
            content
        )
        
        # Escape unescaped special characters
        content = content.replace('&', '&amp;')
        
        # Restore entities
        for i, entity in enumerate(entities):
            content = content.replace(f"{placeholder}{i}###", entity)
        
        return content
    
    @staticmethod
    def _build_fallback_xhtml(title: str) -> str:
        """Build minimal fallback XHTML"""
        safe_title = re.sub(r'[<>&"\']+', '', str(title))
        if not safe_title:
            safe_title = "Chapter"
        
        parts = []
        parts.append('<?xml version="1.0" encoding="utf-8"?>')
        parts.append('<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">')
        parts.append('<html xmlns="http://www.w3.org/1999/xhtml">')
        parts.append('<head>')
        parts.append('<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />')
        parts.append(f'<title>{ContentProcessor.safe_escape(safe_title)}</title>')
        parts.append('</head>')
        parts.append('<body>')
        parts.append('<p>Error processing content. Please check the source file.</p>')
        parts.append('</body>')
        parts.append('</html>')
        
        return '\n'.join(parts)
    
    @staticmethod
    def validate(content: str) -> str:
        """Validate and fix XHTML content - PRESERVES UNICODE"""
        # Ensure XML declaration
        if not content.strip().startswith('<?xml'):
            content = '<?xml version="1.0" encoding="utf-8"?>\n' + content
        
        # Remove control characters
        content = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', content)
        
        # Fix malformed tags
        content = ContentProcessor.fix_malformed_tags(content)
        
        # Fix unescaped ampersands
        content = re.sub(
            r'&(?!(?:'
            r'amp|lt|gt|quot|apos|'
            r'[a-zA-Z][a-zA-Z0-9]{1,31}|'
            r'#[0-9]{1,7}|'
            r'#x[0-9a-fA-F]{1,6}'
            r');)',
            '&amp;',
            content
        )
        
        # Fix self-closing tags
        content = ContentProcessor.fix_self_closing_tags(content)
        
        # Fix unquoted attributes
        content = re.sub(r'<([^>]+)\s+(\w+)=([^\s"\'>]+)([>\s])', r'<\1 \2="\3"\4', content)
        
        # Clean for XML
        content = XMLValidator.clean_for_xml(content)
        
        # DO NOT convert Unicode to character references
        # Keep "同期 (dōki)" as readable text
        
        # Try to parse for validation
        try:
            ET.fromstring(content.encode('utf-8'))
        except ET.ParseError as e:
            log(f"[WARNING] XHTML validation failed: {e}")
            # Try to recover
            content = XHTMLConverter._attempt_recovery(content, e)
        
        return content
    
    @staticmethod
    def _attempt_recovery(content: str, error: ET.ParseError) -> str:
        """Attempt to recover from XML parse errors - ENHANCED"""
        try:
            # Use BeautifulSoup to fix structure
            soup = BeautifulSoup(content, 'html.parser')
            
            # Ensure we have proper XHTML structure
            if not soup.find('html'):
                new_soup = BeautifulSoup('<html xmlns="http://www.w3.org/1999/xhtml"></html>', 'html.parser')
                html_tag = new_soup.html
                for child in list(soup.children):
                    html_tag.append(child)
                soup = new_soup
            
            # Ensure we have head and body
            if not soup.find('head'):
                head = soup.new_tag('head')
                meta = soup.new_tag('meta')
                meta['http-equiv'] = 'Content-Type'
                meta['content'] = 'text/html; charset=utf-8'
                head.append(meta)
                
                title_tag = soup.new_tag('title')
                title_tag.string = 'Chapter'
                head.append(title_tag)
                
                if soup.html:
                    soup.html.insert(0, head)
            
            if not soup.find('body'):
                body = soup.new_tag('body')
                if soup.html:
                    for child in list(soup.html.children):
                        if child.name not in ['head', 'body']:
                            body.append(child.extract())
                    soup.html.append(body)
            
            # Convert back to string
            recovered = str(soup)
            
            # Ensure proper XML declaration
            if not recovered.strip().startswith('<?xml'):
                recovered = '<?xml version="1.0" encoding="utf-8"?>\n' + recovered
            
            # Add DOCTYPE if missing
            if '<!DOCTYPE' not in recovered:
                lines = recovered.split('\n')
                lines.insert(1, '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">')
                recovered = '\n'.join(lines)
            
            # Final validation
            ET.fromstring(recovered.encode('utf-8'))
            log(f"[INFO] Successfully recovered XHTML")
            return recovered
            
        except Exception as recovery_error:
            log(f"[WARNING] Recovery attempt failed: {recovery_error}")
            # Last resort: use fallback
            return XHTMLConverter._build_fallback_xhtml("Chapter")


class FileUtils:
    """File handling utilities"""
    
    @staticmethod
    def sanitize_filename(filename: str, allow_unicode: bool = False) -> str:
        """Sanitize filename for safety"""
        if allow_unicode:
            filename = unicodedata.normalize('NFC', filename)
            replacements = {
                '/': '_', '\\': '_', ':': '_', '*': '_',
                '?': '_', '"': '_', '<': '_', '>': '_',
                '|': '_', '\0': '_',
            }
            for old, new in replacements.items():
                filename = filename.replace(old, new)
            filename = ''.join(char for char in filename if ord(char) >= 32 or ord(char) == 9)
        else:
            filename = unicodedata.normalize('NFKD', filename)
            try:
                filename = filename.encode('ascii', 'ignore').decode('ascii')
            except:
                filename = ''.join(c if ord(c) < 128 else '_' for c in filename)
            
            replacements = {
                '/': '_', '\\': '_', ':': '_', '*': '_',
                '?': '_', '"': '_', '<': '_', '>': '_',
                '|': '_', '\n': '_', '\r': '_', '\t': '_',
                '&': '_and_', '#': '_num_', ' ': '_',
            }
            for old, new in replacements.items():
                filename = filename.replace(old, new)
            
            filename = ''.join(char for char in filename if ord(char) >= 32)
            filename = re.sub(r'_+', '_', filename)
            filename = filename.strip('_')
        
        # Limit length
        name, ext = os.path.splitext(filename)
        if len(name) > 100:
            name = name[:100]
        
        if not name or name == '_':
            name = 'file'
        
        return name + ext
    
    @staticmethod
    def ensure_bytes(content) -> bytes:
        """Ensure content is bytes"""
        if content is None:
            return b''
        if isinstance(content, bytes):
            return content
        if not isinstance(content, str):
            content = str(content)
        return content.encode('utf-8')


class EPUBCompiler:
    """Main EPUB compilation class"""
    
    def __init__(self, base_dir: str, log_callback: Optional[Callable] = None):
        self.base_dir = os.path.abspath(base_dir)
        self.log_callback = log_callback
        self.output_dir = self.base_dir
        self.images_dir = os.path.join(self.output_dir, "images")
        self.css_dir = os.path.join(self.output_dir, "css")
        self.fonts_dir = os.path.join(self.output_dir, "fonts")
        self.metadata_path = os.path.join(self.output_dir, "metadata.json")
        
        # Set global log callback
        set_global_log_callback(log_callback)
        
        # translation features
        self.html_dir = self.output_dir  # For compatibility
        self.translate_titles = os.getenv('TRANSLATE_BOOK_TITLE', '1') == '1'
        
        # Initialize API client if needed
        self.api_client = None
        if self.translate_titles or os.getenv('BATCH_TRANSLATE_HEADERS', '0') == '1':
            model = os.getenv('MODEL')
            api_key = os.getenv('API_KEY')
            if model and api_key and UnifiedClient:
                self.api_client = UnifiedClient(api_key=api_key, model=model, output_dir=self.output_dir)
            elif model and api_key and not UnifiedClient:
                self.log("Warning: UnifiedClient module not available, translation features disabled")
        
        # Enhance with translation features
        enhance_epub_compiler(self)
    
    def log(self, message: str):
        """Log a message"""
        if self.log_callback:
            self.log_callback(message)
        else:
            print(message)
            
    def compile(self):
        """Main compilation method"""
        try:    
            # Debug: Check what metadata enhancement was done
            self.log("[DEBUG] Checking metadata translation setup...")
            self.log(f"[DEBUG] Has api_client: {hasattr(self, 'api_client') and self.api_client is not None}")
            self.log(f"[DEBUG] Has metadata_translator: {hasattr(self, 'metadata_translator')}")
            self.log(f"[DEBUG] Has translate_metadata_fields: {hasattr(self, 'translate_metadata_fields')}")
            
            if hasattr(self, 'translate_metadata_fields'):
                self.log(f"[DEBUG] translate_metadata_fields content: {self.translate_metadata_fields}")
                enabled_fields = [k for k, v in self.translate_metadata_fields.items() if v]
                self.log(f"[DEBUG] Enabled metadata fields: {enabled_fields}")
                
            # Pre-flight check
            if not self._preflight_check():
                return
            
            # Analyze chapters FIRST to get the structure
            chapter_titles_info = self._analyze_chapters()
            
            # Debug: Check if batch translation is enabled
            self.log(f"[DEBUG] Batch translation enabled: {getattr(self, 'batch_translate_headers', False)}")
            self.log(f"[DEBUG] Has header translator: {hasattr(self, 'header_translator')}")
            self.log(f"[DEBUG] EPUB_PATH env: {os.getenv('EPUB_PATH', 'NOT SET')}")
            self.log(f"[DEBUG] HTML dir: {self.html_dir}")
            
            # Extract source headers AND current titles if batch translation is enabled
            source_headers = {}
            current_titles = {}
            if (hasattr(self, 'batch_translate_headers') and self.batch_translate_headers and 
                hasattr(self, 'header_translator') and self.header_translator):
                
                # Check if the extraction method exists
                if hasattr(self, '_extract_source_headers_and_current_titles'):
                    # Use the new extraction method
                    source_headers, current_titles = self._extract_source_headers_and_current_titles()
                    self.log(f"[DEBUG] Extraction complete: {len(source_headers)} source, {len(current_titles)} current")
                else:
                    self.log("⚠️ Missing _extract_source_headers_and_current_titles method!")
            
            # Batch translate headers if we have source headers
            translated_headers = {}
            if source_headers and hasattr(self, 'header_translator') and self.header_translator:
                # Check if translated_headers.txt already exists
                translations_file = os.path.join(self.output_dir, "translated_headers.txt")
                
                if os.path.exists(translations_file):
                    # File exists - skip translation entirely
                    self.log("📁 Found existing translated_headers.txt - skipping header translation")
                    # No need to parse or do anything else
                else:
                    # No existing file - proceed with translation
                    self.log("🌐 Batch translating chapter headers...")
                    
                    try:
                        # Check if the translator has been initialized properly
                        if not hasattr(self.header_translator, 'client') or not self.header_translator.client:
                            self.log("⚠️ Header translator not properly initialized, skipping batch translation")
                        else:
                            self.log(f"📚 Found {len(source_headers)} headers to translate")
                            self.log(f"📚 Found {len(current_titles)} current titles in HTML files")
                            
                            # Debug: Show a few examples
                            for num in list(source_headers.keys())[:3]:
                                self.log(f"  Example - Chapter {num}: {source_headers[num]}")
                            
                            # Translate headers with current titles info
                            translated_headers = self.header_translator.translate_and_save_headers(
                                html_dir=self.html_dir,
                                headers_dict=source_headers,
                                batch_size=getattr(self, 'headers_per_batch', 400),
                                output_dir=self.output_dir,
                                update_html=getattr(self, 'update_html_headers', True),
                                save_to_file=getattr(self, 'save_header_translations', True),
                                current_titles=current_titles  # Pass current titles for exact replacement
                            )
                            
                            # Update chapter_titles_info with translations
                            if translated_headers:
                                self.log("\n📝 Updating chapter titles in EPUB structure...")
                                for chapter_num, translated_title in translated_headers.items():
                                    if chapter_num in chapter_titles_info:
                                        # Keep the original confidence and method, just update the title
                                        orig_title, confidence, method = chapter_titles_info[chapter_num]
                                        chapter_titles_info[chapter_num] = (translated_title, confidence, method)
                                        self.log(f"✓ Chapter {chapter_num}: {source_headers.get(chapter_num, 'Unknown')} → {translated_title}")
                                    else:
                                        # Add new entry if not in chapter_titles_info
                                        chapter_titles_info[chapter_num] = (translated_title, 1.0, 'batch_translation')
                                        self.log(f"✓ Added Chapter {chapter_num}: {translated_title}")
                            
                    except Exception as e:
                        self.log(f"⚠️ Batch translation failed: {e}")
                        import traceback
                        self.log(traceback.format_exc())
                        # Continue with compilation even if translation fails
            else:
                if not source_headers:
                    self.log("⚠️ No source headers found, skipping batch translation")
                elif not hasattr(self, 'header_translator'):
                    self.log("⚠️ No header translator available")
            
            # Rest of the compile method continues as before...
            # Find HTML files
            html_files = self._find_html_files()
            if not html_files:
                raise Exception("No translated chapters found to compile into EPUB")
            
            # Load metadata
            metadata = self._load_metadata()

            # Translate metadata if configured
            if hasattr(self, 'metadata_translator') and self.metadata_translator:
                if hasattr(self, 'translate_metadata_fields') and any(self.translate_metadata_fields.values()):
                    self.log("🌐 Translating metadata fields...")
                    
                    try:
                        translated_metadata = self.metadata_translator.translate_metadata(
                            metadata,
                            self.translate_metadata_fields,
                            mode=getattr(self, 'metadata_translation_mode', 'together')
                        )
                        
                        # Preserve original values
                        for field in self.translate_metadata_fields:
                            if field in metadata and field in translated_metadata:
                                if metadata[field] != translated_metadata[field]:
                                    translated_metadata[f'original_{field}'] = metadata[field]
                        
                        metadata = translated_metadata
                    except Exception as e:
                        self.log(f"⚠️ Metadata translation failed: {e}")
                        # Continue with original metadata
            
            # Create EPUB book
            book = self._create_book(metadata)
            
            # Process all components
            spine = []
            toc = []
            
            # Add CSS
            css_items = self._add_css_files(book)
            
            # Add fonts
            self._add_fonts(book)
            
            # Process images and cover
            processed_images, cover_file = self._process_images()
            
            # Add images to book
            self._add_images_to_book(book, processed_images, cover_file)
            
            # Add cover page if exists
            if cover_file:
                cover_page = self._create_cover_page(book, cover_file, processed_images, css_items, metadata)
                if cover_page:
                    spine.insert(0, cover_page)
            
            # Process chapters with updated titles
            chapters_added = self._process_chapters(
                book, html_files, chapter_titles_info, 
                css_items, processed_images, spine, toc, metadata
            )
            
            if chapters_added == 0:
                raise Exception("No chapters could be added to the EPUB")
            
            # Add optional gallery (unless disabled)
            disable_gallery = os.environ.get('DISABLE_EPUB_GALLERY', '0') == '1'
            if disable_gallery:
                self.log("📷 Image gallery disabled by user preference")
            else:
                gallery_images = [img for img in processed_images.values() if img != cover_file]
                if gallery_images:
                    self.log(f"📷 Creating image gallery with {len(gallery_images)} images...")
                    gallery_page = self._create_gallery_page(book, gallery_images, css_items, metadata)
                    spine.append(gallery_page)
                    toc.append(gallery_page)
                else:
                    self.log("📷 No images found for gallery")
            
            # Finalize book
            self._finalize_book(book, spine, toc, cover_file)
            
            # Write EPUB
            self._write_epub(book, metadata)
            
            # Show summary
            self._show_summary(chapter_titles_info, css_items)
            
        except Exception as e:
            self.log(f"❌ EPUB compilation failed: {e}")
            raise

    def _fix_encoding_issues(self, content: str) -> str:
        """Convert smart quotes and other Unicode punctuation to ASCII."""
        # Convert smart quotes to regular quotes and other punctuation
        fixes = {
            '’': "'",   # Right single quotation mark
            '‘': "'",   # Left single quotation mark
            '“': '"',   # Left double quotation mark
            '”': '"',   # Right double quotation mark
            '—': '-',   # Em dash to hyphen
            '–': '-',   # En dash to hyphen
            '…': '...', # Ellipsis to three dots
        }

        for bad, good in fixes.items():
            if bad in content:
                content = content.replace(bad, good)
                #self.log(f"[DEBUG] Replaced {bad!r} with {good!r}")

        return content


    def _preflight_check(self) -> bool:
        """Pre-flight check before compilation with progressive fallback"""
        # Check if we have standard files
        if self._has_standard_files():
            # Use original strict check
            return self._preflight_check_strict()
        else:
            # Use progressive check for non-standard files
            result = self._preflight_check_progressive()
            return result is not None

    def _has_standard_files(self) -> bool:
        """Check if directory contains standard response_ files"""
        if not os.path.exists(self.base_dir):
            return False
        
        html_files = [f for f in os.listdir(self.base_dir) if f.endswith('.html')]
        response_files = [f for f in html_files if f.startswith('response_')]
        
        return len(response_files) > 0

    def _preflight_check_strict(self) -> bool:
        """Original strict pre-flight check - for standard files"""
        self.log("\n📋 Pre-flight Check")
        self.log("=" * 50)
        
        issues = []
        
        if not os.path.exists(self.base_dir):
            issues.append(f"Directory does not exist: {self.base_dir}")
            return False
        
        html_files = [f for f in os.listdir(self.base_dir) if f.endswith('.html')]
        response_files = [f for f in html_files if f.startswith('response_')]
        
        if not html_files:
            issues.append("No HTML files found in directory")
        elif not response_files:
            issues.append(f"Found {len(html_files)} HTML files but none start with 'response_'")
        else:
            self.log(f"✅ Found {len(response_files)} chapter files")
        
        if not os.path.exists(self.metadata_path):
            self.log("⚠️  No metadata.json found (will use defaults)")
        else:
            self.log("✅ Found metadata.json")
        
        for subdir in ['css', 'images', 'fonts']:
            path = os.path.join(self.base_dir, subdir)
            if os.path.exists(path):
                count = len(os.listdir(path))
                self.log(f"✅ Found {subdir}/ with {count} files")
        
        if issues:
            self.log("\n❌ Pre-flight check FAILED:")
            for issue in issues:
                self.log(f"  • {issue}")
            return False
        
        self.log("\n✅ Pre-flight check PASSED")
        return True

    def _preflight_check_progressive(self) -> dict:
        """Progressive pre-flight check for non-standard files"""
        self.log("\n📋 Starting Progressive Pre-flight Check")
        self.log("=" * 50)
        
        # Critical check - always required
        if not os.path.exists(self.base_dir):
            self.log(f"❌ CRITICAL: Directory does not exist: {self.base_dir}")
            return None
        
        # Phase 1: Try strict mode (response_ files) - already checked in caller
        
        # Phase 2: Try relaxed mode (any HTML files)
        self.log("\n[Phase 2] Checking for any HTML files...")
        
        html_files = [f for f in os.listdir(self.base_dir) if f.endswith('.html')]
        
        if html_files:
            self.log(f"✅ Found {len(html_files)} HTML files:")
            # Show first 5 files as examples
            for i, f in enumerate(html_files[:5]):
                self.log(f"    • {f}")
            if len(html_files) > 5:
                self.log(f"    ... and {len(html_files) - 5} more")
            
            self._check_optional_resources()
            self.log("\n⚠️  Pre-flight check PASSED with warnings (relaxed mode)")
            return {'success': True, 'mode': 'relaxed'}
        
        # Phase 3: No HTML files at all
        self.log("❌ No HTML files found in directory")
        self.log("\n[Phase 3] Checking directory contents...")
        
        all_files = os.listdir(self.base_dir)
        self.log(f"📁 Directory contains {len(all_files)} total files")
        
        # Look for any potential content
        potential_content = [f for f in all_files if not f.startswith('.')]
        if potential_content:
            self.log("⚠️  Found non-HTML files:")
            for i, f in enumerate(potential_content[:5]):
                self.log(f"    • {f}")
            if len(potential_content) > 5:
                self.log(f"    ... and {len(potential_content) - 5} more")
            
            self.log("\n⚠️  BYPASSING standard checks - compilation may fail!")
            return {'success': True, 'mode': 'bypass'}
        
        self.log("\n❌ Directory appears to be empty")
        return None

    def _check_optional_resources(self):
        """Check for optional resources (metadata, CSS, images, fonts)"""
        self.log("\n📁 Checking optional resources:")
        
        if os.path.exists(self.metadata_path):
            self.log("✅ Found metadata.json")
        else:
            self.log("⚠️  No metadata.json found (will use defaults)")
        
        resources_found = False
        for subdir in ['css', 'images', 'fonts']:
            path = os.path.join(self.base_dir, subdir)
            if os.path.exists(path):
                items = os.listdir(path)
                if items:
                    self.log(f"✅ Found {subdir}/ with {len(items)} files")
                    resources_found = True
                else:
                    self.log(f"📁 Found {subdir}/ (empty)")
        
        if not resources_found:
            self.log("⚠️  No resource directories found (CSS/images/fonts)")

    def _analyze_chapters(self) -> Dict[int, Tuple[str, float, str]]:
            """Analyze chapter files and extract titles"""
            self.log("\n📖 Extracting translated titles from chapter files...")
            
            chapter_info = {}
            
            # Check if we have standard files
            if self._has_standard_files():
                # Original logic for standard files
                html_files = [f for f in os.listdir(self.output_dir) 
                             if f.startswith("response_") and f.endswith(".html")]
            else:
                # Progressive logic for non-standard files
                html_extensions = ('.html', '.htm', '.xhtml')
                all_html = [f for f in os.listdir(self.output_dir) 
                           if f.lower().endswith(html_extensions)]
                
                # Filter out obvious non-chapter files
                exclude_patterns = [
                    'index', 'toc', 'contents', 'cover', 'title',
                    'copyright', 'about', 'nav', 'style', 'template',
                    'metadata', 'acknowledgments', 'dedication'
                ]
                
                html_files = [f for f in all_html 
                             if not any(exclude in f.lower() for exclude in exclude_patterns)]
            
            if not html_files:
                self.log("⚠️ No translated chapter files found!")
                return chapter_info
            
            self.log(f"📖 Analyzing {len(html_files)} translated chapter files for titles...")
            
            # Sort files using the same logic as _find_html_files
            if self._has_standard_files():
                # Check if files have the -h- pattern
                if any('-h-' in f for f in html_files):
                    # Use the robust sort key for -h- pattern files
                    sorted_files = sorted(html_files, key=self.get_robust_sort_key)
                else:
                    # Use numeric sorting for standard response_ files without -h-
                    def extract_number(filename):
                        match = re.match(r'response_(\d+)_', filename)
                        if match:
                            return int(match.group(1))
                        return 0
                    
                    sorted_files = sorted(html_files, key=extract_number)
            else:
                # Progressive sorting for non-standard files
                sorted_files = sorted(html_files, key=self.get_robust_sort_key)
            
            for idx, filename in enumerate(sorted_files):
                file_path = os.path.join(self.output_dir, filename)
                
                try:
                    # FIXED: Extract chapter number using the unified extraction method
                    # This now properly checks for -h- pattern FIRST
                    chapter_num = self._extract_chapter_number(filename, idx)
                    
                    # Read content
                    with open(file_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()

                    # Extract title
                    title, confidence = TitleExtractor.extract_from_html(
                        html_content, chapter_num, filename
                    )
                    
                    chapter_info[chapter_num] = (title, confidence, filename)
                    
                    # Log with confidence indicator
                    indicator = "✅" if confidence > 0.7 else "🟡" if confidence > 0.4 else "🔴"
                    self.log(f"  {indicator} Chapter {chapter_num}: '{title}' (confidence: {confidence:.2f})")
                    
                except Exception as e:
                    self.log(f"❌ Error processing {filename}: {e}")
                    # Use fallback chapter number
                    fallback_num = self._extract_chapter_number(filename, idx)
                    chapter_info[fallback_num] = (f"Chapter {fallback_num}", 0.0, filename)
            
            if chapter_info:
                confident = sum(1 for _, (_, conf, _) in chapter_info.items() if conf > 0.5)
                self.log(f"📊 Title extraction summary: {confident}/{len(chapter_info)} with high confidence")
            
            return chapter_info

    def _find_html_files(self) -> List[str]:
            """Find HTML files with ROBUST sorting for ANY pattern"""
            self.log(f"\n[DEBUG] Scanning directory: {self.output_dir}")
            
            all_files = os.listdir(self.output_dir)
            
            # Get ALL HTML files, not just response_ ones
            html_extensions = ('.html', '.htm', '.xhtml')
            html_files = [f for f in all_files if f.lower().endswith(html_extensions)]
            
            # Filter out common non-chapter files
            exclude_patterns = [
                'index', 'toc', 'contents', 'cover', 'title',
                'copyright', 'about', 'nav', 'style', 'template'
            ]
            
            # First, try to find response_ files (your current pattern)
            response_files = [f for f in html_files if f.startswith('response_')]
            
            # If we have response_ files, use those
            if response_files:
                html_files = response_files
                self.log(f"[DEBUG] Found {len(response_files)} response_ files")
            else:
                # Otherwise, filter out obvious non-chapter files
                html_files = [f for f in html_files 
                             if not any(exclude in f.lower() for exclude in exclude_patterns)]
                self.log(f"[DEBUG] Found {len(html_files)} HTML files (no response_ prefix)")
            
            if not html_files:
                self.log("[ERROR] No HTML files found!")
                return []
            
            # Sort files
            if response_files:
                # FIXED: Check if files have -h- pattern
                if any('-h-' in f for f in response_files):
                    # Use special sorting for -h- pattern
                    def extract_h_number(filename):
                        match = re.search(r'-h-(\d+)', filename)
                        if match:
                            return int(match.group(1))
                        return 999999  # Put non-matching files at end
                    
                    html_files.sort(key=extract_h_number)
                else:
                    # Use numeric sorting for standard response_ files
                    def extract_number(filename):
                        match = re.match(r'response_(\d+)_', filename)
                        if match:
                            return int(match.group(1))
                        return 0
                    
                    html_files.sort(key=extract_number)
            else:
                # Progressive sorting for non-standard files
                html_files.sort(key=self.get_robust_sort_key)
            
            # Debug output
            self.log(f"\n[DEBUG] Sorted {len(html_files)} files:")
            for i, f in enumerate(html_files[:10]):
                self.log(f"  {i+1}. {f}")
            if len(html_files) > 10:
                self.log(f"  ... and {len(html_files)-10} more")
            
            return html_files

    def _process_chapters(self, book: epub.EpubBook, html_files: List[str],
                         chapter_titles_info: Dict[int, Tuple[str, float, str]],
                         css_items: List[epub.EpubItem], processed_images: Dict[str, str],
                         spine: List, toc: List, metadata: dict) -> int:
        """Process chapters - NO FANCY PARSING"""
        chapters_added = 0
        
        self.log(f"\n📚 Processing {len(html_files)} chapters...")
        
        # Check if we're dealing with standard files
        is_standard = any(f.startswith('response_') for f in html_files)
        
        # Process in EXACT order - no parsing, no reordering
        for idx, filename in enumerate(html_files):
            # FIXED: Always extract chapter number from filename, not from position
            chapter_num = self._extract_chapter_number(filename, idx)
            
            try:
                if self._process_single_chapter(
                    book, chapter_num, filename, chapter_titles_info, css_items, 
                    processed_images, spine, toc, metadata
                ):
                    chapters_added += 1
            except Exception as e:
                self.log(f"[ERROR] Chapter {chapter_num} failed: {e}")
                
                # Add placeholder
                try:
                    chapter = epub.EpubHtml(
                        title=f"Chapter {chapter_num}",
                        file_name=f"chapter_{chapter_num:03d}.xhtml",
                        lang=metadata.get("language", "en")
                    )
                    chapter.content = f"""<?xml version="1.0" encoding="utf-8"?>
    <!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">
    <html xmlns="http://www.w3.org/1999/xhtml">
    <head><title>Chapter {chapter_num}</title></head>
    <body>
    <h1>Chapter {chapter_num}</h1>
    <p>Error loading: {filename}</p>
    </body>
    </html>""".encode('utf-8')
                    
                    book.add_item(chapter)
                    spine.append(chapter)
                    toc.append(chapter)
                    chapters_added += 1
                except:
                    pass
        
        self.log(f"Added {chapters_added} chapters")
        return chapters_added

    def _process_single_chapter(self, book: epub.EpubBook, num: int, filename: str,
                               chapter_titles_info: Dict[int, Tuple[str, float, str]],
                               css_items: List[epub.EpubItem], processed_images: Dict[str, str],
                               spine: List, toc: List, metadata: dict) -> bool:
        """Process a single chapter"""
        path = os.path.join(self.output_dir, filename)
        
        try:
            # Read content
            with open(path, 'r', encoding='utf-8') as f:
                raw_content = f.read()
            
            # Force fix the encoding issues FIRST
            raw_content = self._fix_encoding_issues(raw_content)

            # Then do the normal decode (for other entities)
            raw_content = HTMLEntityDecoder.decode(raw_content)
            
            if not raw_content.strip():
                self.log(f"[WARNING] Chapter {num} is empty")
                return False
            
            # Clean content - with progressive handling for web-scraped pages
            if not filename.startswith('response_'):
                # Try to extract main content for non-standard files
                raw_content = self._extract_main_content(raw_content, filename)
            
            raw_content = ContentProcessor.clean_chapter_content(raw_content)
            
            # Get title
            title = self._get_chapter_title(num, filename, raw_content, chapter_titles_info)
            
            # Prepare CSS links
            css_links = [f"css/{item.file_name.split('/')[-1]}" for item in css_items]
            
            # Convert to XHTML
            try:
                xhtml_content = XHTMLConverter.ensure_compliance(raw_content, title, css_links)
            except Exception as e:
                self.log(f"[WARNING] XHTML conversion failed for chapter {num}: {e}")
                # Create minimal valid XHTML
                xhtml_content = XHTMLConverter._build_xhtml(
                    title,
                    f'<h1>{ContentProcessor.safe_escape(title)}</h1><p>Chapter content could not be processed.</p>',
                    css_links
                )
            
            # Process images in content
            xhtml_content = self._process_chapter_images(xhtml_content, processed_images)
            
            # Validate
            final_content = XHTMLConverter.validate(xhtml_content)
            
            # Final validation check
            try:
                ET.fromstring(final_content.encode('utf-8'))
            except ET.ParseError:
                # Use a simple fallback that we know is valid
                final_content = '<?xml version="1.0" encoding="utf-8"?>\n' + \
                    '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" ' + \
                    '"http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">\n' + \
                    '<html xmlns="http://www.w3.org/1999/xhtml">\n' + \
                    '<head>\n' + \
                    '<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />\n' + \
                    f'<title>{ContentProcessor.safe_escape(title)}</title>\n' + \
                    '</head>\n' + \
                    '<body>\n' + \
                    f'<h1>{ContentProcessor.safe_escape(title)}</h1>\n' + \
                    '<p>Chapter content could not be processed properly.</p>\n' + \
                    '</body>\n' + \
                    '</html>'
            
            # Create chapter
            safe_fn = f"chapter_{num:03d}.xhtml"
            chapter = epub.EpubHtml(
                title=title,
                file_name=safe_fn,
                lang=metadata.get("language", "en")
            )
            chapter.content = FileUtils.ensure_bytes(final_content)
            
            # Add to book
            book.add_item(chapter)
            spine.append(chapter)
            toc.append(chapter)
            
            self.log(f"✅ Added chapter {num}: '{title}'")
            return True
            
        except Exception as e:
            self.log(f"[ERROR] Failed to process chapter {num}: {e}")
            # Add a placeholder chapter
            try:
                title = f"Chapter {num}"
                if num in chapter_titles_info:
                    title = chapter_titles_info[num][0]
                
                placeholder_content = '<?xml version="1.0" encoding="utf-8"?>\n' + \
                    '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" ' + \
                    '"http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">\n' + \
                    '<html xmlns="http://www.w3.org/1999/xhtml">\n' + \
                    '<head>\n' + \
                    '<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />\n' + \
                    f'<title>{ContentProcessor.safe_escape(title)}</title>\n' + \
                    '</head>\n' + \
                    '<body>\n' + \
                    f'<h1>{ContentProcessor.safe_escape(title)}</h1>\n' + \
                    '<p>Error loading chapter content.</p>\n' + \
                    '</body>\n' + \
                    '</html>'
                
                safe_fn = f"chapter_{num:03d}.xhtml"
                chapter = epub.EpubHtml(
                    title=title,
                    file_name=safe_fn,
                    lang=metadata.get("language", "en")
                )
                chapter.content = FileUtils.ensure_bytes(placeholder_content)
                
                book.add_item(chapter)
                spine.append(chapter)
                toc.append(chapter)
                
                self.log(f"⚠️ Added placeholder for chapter {num}")
                return True
            except:
                return False

    def _get_chapter_title(self, num: int, filename: str, content: str,
                          chapter_titles_info: Dict[int, Tuple[str, float, str]]) -> str:
        """Get chapter title with fallbacks"""
        title = None
        confidence = 0.0
        
        # Try pre-analyzed title
        if num in chapter_titles_info:
            title, confidence, _ = chapter_titles_info[num]
        
        # Re-extract if low confidence
        if not title or confidence < 0.5:
            backup_title, backup_confidence = TitleExtractor.extract_from_html(content, num, filename)
            if backup_confidence > confidence:
                title = backup_title
                confidence = backup_confidence
        
        # Clean and validate
        if title:
            title = TitleExtractor.clean_title(title)
            if not TitleExtractor.is_valid_title(title):
                title = None
        
        # NEW FALLBACK LOGIC - only if TitleExtractor failed AND not standard file
        if not title and not filename.startswith('response_'):
            # Try enhanced extraction methods for web-scraped content
            title = self._fallback_title_extraction(content, filename, num)
        
        # Final fallback
        if not title:
            title = f"Chapter {num}"
        
        return title

    # Progressive helper methods - only used for non-standard files

    def get_robust_sort_key(self, filename):
        """Extract chapter/sequence number using multiple patterns"""
        
        # Pattern 1: -h-NUMBER (your current pattern)
        match = re.search(r'-h-(\d+)', filename)
        if match:
            return (1, int(match.group(1)))
        
        # Pattern 2: chapter-NUMBER or chapter_NUMBER or chapterNUMBER
        match = re.search(r'chapter[-_\s]?(\d+)', filename, re.IGNORECASE)
        if match:
            return (2, int(match.group(1)))
        
        # Pattern 3: ch-NUMBER or ch_NUMBER or chNUMBER  
        match = re.search(r'\bch[-_\s]?(\d+)\b', filename, re.IGNORECASE)
        if match:
            return (3, int(match.group(1)))
        
        # Pattern 4: response_NUMBER_ (if response_ prefix exists)
        if filename.startswith('response_'):
            match = re.match(r'response_(\d+)[-_]', filename)
            if match:
                return (4, int(match.group(1)))
        
        # Pattern 5: book_NUMBER, story_NUMBER, part_NUMBER, section_NUMBER
        match = re.search(r'(?:book|story|part|section)[-_\s]?(\d+)', filename, re.IGNORECASE)
        if match:
            return (5, int(match.group(1)))
        
        # Pattern 6: split_NUMBER (Calibre pattern)
        match = re.search(r'split_(\d+)', filename)
        if match:
            return (6, int(match.group(1)))
        
        # Pattern 7: Just NUMBER.html (like 1.html, 2.html)
        match = re.match(r'^(\d+)\.(?:html?|xhtml)$', filename)
        if match:
            return (7, int(match.group(1)))
        
        # Pattern 8: -NUMBER at end before extension
        match = re.search(r'-(\d+)\.(?:html?|xhtml)$', filename)
        if match:
            return (8, int(match.group(1)))
        
        # Pattern 9: _NUMBER at end before extension
        match = re.search(r'_(\d+)\.(?:html?|xhtml)$', filename)
        if match:
            return (9, int(match.group(1)))
        
        # Pattern 10: (NUMBER) in parentheses anywhere
        match = re.search(r'\((\d+)\)', filename)
        if match:
            return (10, int(match.group(1)))
        
        # Pattern 11: [NUMBER] in brackets anywhere
        match = re.search(r'\[(\d+)\]', filename)
        if match:
            return (11, int(match.group(1)))
        
        # Pattern 12: page-NUMBER or p-NUMBER or pg-NUMBER
        match = re.search(r'(?:page|pg?)[-_\s]?(\d+)', filename, re.IGNORECASE)
        if match:
            return (12, int(match.group(1)))
        
        # Pattern 13: Any file ending with NUMBER before extension
        match = re.search(r'(\d+)\.(?:html?|xhtml)$', filename)
        if match:
            return (13, int(match.group(1)))
        
        # Pattern 14: Roman numerals (I, II, III, IV, etc.)
        roman_pattern = r'\b(M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3}))\b'
        match = re.search(roman_pattern, filename)
        if match:
            roman = match.group(1)
            # Convert roman to number
            roman_dict = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}
            val = 0
            for i in range(len(roman)):
                if i > 0 and roman_dict[roman[i]] > roman_dict[roman[i-1]]:
                    val += roman_dict[roman[i]] - 2 * roman_dict[roman[i-1]]
                else:
                    val += roman_dict[roman[i]]
            return (14, val)
        
        # Pattern 15: First significant number found
        numbers = re.findall(r'\d+', filename)
        if numbers:
            # Skip common year numbers (1900-2099) unless it's the only number
            significant_numbers = [int(n) for n in numbers if not (1900 <= int(n) <= 2099)]
            if significant_numbers:
                return (15, significant_numbers[0])
            elif numbers:
                return (15, int(numbers[0]))
        
        # Final fallback: alphabetical
        return (99, filename)

    def _extract_chapter_number(self, filename: str, default_idx: int) -> int:
            """Extract chapter number using multiple patterns"""
            
            # FIXED: Pattern 1 - Check -h-NUMBER FIRST (YOUR FILES USE THIS!)
            match = re.search(r'-h-(\d+)', filename)
            if match:
                return int(match.group(1))
            
            # Pattern 2: response_NUMBER_ (standard pattern)
            match = re.match(r"response_(\d+)_", filename)
            if match:
                return int(match.group(1))
            
            # Pattern 3: chapter-NUMBER, chapter_NUMBER, chapterNUMBER
            match = re.search(r'chapter[-_\s]?(\d+)', filename, re.IGNORECASE)
            if match:
                return int(match.group(1))
            
            # Pattern 4: ch-NUMBER, ch_NUMBER, chNUMBER
            match = re.search(r'\bch[-_\s]?(\d+)\b', filename, re.IGNORECASE)
            if match:
                return int(match.group(1))
            
            # Pattern 5: Just NUMBER.html (like 127.html)
            match = re.match(r'^(\d+)\.(?:html?|xhtml)$', filename)
            if match:
                return int(match.group(1))
            
            # Pattern 6: _NUMBER at end before extension
            match = re.search(r'_(\d+)\.(?:html?|xhtml)$', filename)
            if match:
                return int(match.group(1))
            
            # Pattern 7: -NUMBER at end before extension
            match = re.search(r'-(\d+)\.(?:html?|xhtml)$', filename)
            if match:
                return int(match.group(1))
            
            # Pattern 8: (NUMBER) in parentheses
            match = re.search(r'\((\d+)\)', filename)
            if match:
                return int(match.group(1))
            
            # Pattern 9: [NUMBER] in brackets
            match = re.search(r'\[(\d+)\]', filename)
            if match:
                return int(match.group(1))
            
            # Pattern 10: Use the sort key logic
            sort_key = self.get_robust_sort_key(filename)
            if isinstance(sort_key[1], int) and sort_key[1] > 0:
                return sort_key[1]
            
            # Final fallback: use position + 1
            return default_idx + 1

    def _extract_main_content(self, html_content: str, filename: str) -> str:
        """Extract main content from web-scraped HTML pages
        
        This method tries to find the actual chapter content within a full webpage
        """
        try:
            # For web-scraped content, try to extract just the chapter part
            # Common patterns for chapter content containers
            content_patterns = [
                # Look for specific class names commonly used for content
                (r'<div[^>]*class="[^"]*(?:chapter-content|entry-content|epcontent|post-content|content-area|main-content)[^"]*"[^>]*>(.*?)</div>', re.DOTALL | re.IGNORECASE),
                # Look for article tags with content
                (r'<article[^>]*>(.*?)</article>', re.DOTALL | re.IGNORECASE),
                # Look for main tags
                (r'<main[^>]*>(.*?)</main>', re.DOTALL | re.IGNORECASE),
                # Look for specific id patterns
                (r'<div[^>]*id="[^"]*(?:content|chapter|post)[^"]*"[^>]*>(.*?)</div>', re.DOTALL | re.IGNORECASE),
            ]
            
            for pattern, flags in content_patterns:
                match = re.search(pattern, html_content, flags)
                if match:
                    extracted = match.group(1)
                    # Make sure we got something substantial
                    if len(extracted.strip()) > 100:
                        self.log(f"📄 Extracted main content using pattern for {filename}")
                        return extracted
            
            # If no patterns matched, check if this looks like a full webpage
            if '<html' in html_content.lower() and '<body' in html_content.lower():
                # Try to extract body content
                body_match = re.search(r'<body[^>]*>(.*?)</body>', html_content, re.DOTALL | re.IGNORECASE)
                if body_match:
                    self.log(f"📄 Extracted body content for {filename}")
                    return body_match.group(1)
            
            # If all else fails, return original content
            self.log(f"📄 Using original content for {filename}")
            return html_content
            
        except Exception as e:
            self.log(f"⚠️  Content extraction failed for {filename}: {e}")
            return html_content

    def _fallback_title_extraction(self, content: str, filename: str, num: int) -> Optional[str]:
        """Fallback title extraction for when TitleExtractor fails
        
        This handles web-scraped pages and other non-standard formats
        """
        # Try filename-based extraction first (often more reliable for web scrapes)
        filename_title = self._extract_title_from_filename_fallback(filename, num)
        if filename_title:
            return filename_title
        
        # Try HTML content extraction with patterns TitleExtractor might miss
        html_title = self._extract_title_from_html_fallback(content, num)
        if html_title:
            return html_title
        
        return None

    def _extract_title_from_html_fallback(self, content: str, num: int) -> Optional[str]:
        """Fallback HTML title extraction for web-scraped content"""
        
        # Look for title patterns that TitleExtractor might miss
        # Specifically for web-scraped novel sites
        patterns = [
            # Title tags with site separators
            r'<title[^>]*>([^|–\-]+?)(?:\s*[|–\-]\s*[^<]+)?</title>',
            # Specific class patterns from novel sites
            r'<div[^>]*class="[^"]*cat-series[^"]*"[^>]*>([^<]+)</div>',
            r'<h1[^>]*class="[^"]*entry-title[^"]*"[^>]*>([^<]+)</h1>',
            r'<span[^>]*class="[^"]*chapter-title[^"]*"[^>]*>([^<]+)</span>',
            # Meta property patterns
            r'<meta[^>]*property="og:title"[^>]*content="([^"]+)"',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                title = match.group(1).strip()
                # Decode HTML entities
                title = HTMLEntityDecoder.decode(title)
                
                # Additional cleanup for web-scraped content
                title = re.sub(r'\s+', ' ', title)  # Normalize whitespace
                title = title.strip()
                
                # Validate it's reasonable
                if 3 < len(title) < 200 and title.lower() != 'untitled':
                    self.log(f"📝 Fallback extracted title from HTML: '{title}'")
                    return title
        
        return None

    def _extract_title_from_filename_fallback(self, filename: str, num: int) -> Optional[str]:
        """Fallback filename title extraction"""
        
        # Remove extension
        base_name = re.sub(r'\.(html?|xhtml)$', '', filename, flags=re.IGNORECASE)
        
        # Web-scraped filename patterns
        patterns = [
            # "theend-chapter-127-apocalypse-7" -> "Chapter 127 - Apocalypse 7"
            r'(?:theend|story|novel)[-_]chapter[-_](\d+)[-_](.+)',
            # "chapter-127-apocalypse-7" -> "Chapter 127 - Apocalypse 7"  
            r'chapter[-_](\d+)[-_](.+)',
            # "ch127-title" -> "Chapter 127 - Title"
            r'ch[-_]?(\d+)[-_](.+)',
            # Just the title part after number
            r'^\d+[-_](.+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, base_name, re.IGNORECASE)
            if match:
                if match.lastindex == 2:  # Pattern with chapter number and title
                    chapter_num = match.group(1)
                    title_part = match.group(2)
                else:  # Pattern with just title
                    chapter_num = str(num)
                    title_part = match.group(1)
                
                # Clean up the title part
                title_part = title_part.replace('-', ' ').replace('_', ' ')
                # Capitalize properly
                words = title_part.split()
                title_part = ' '.join(word.capitalize() if len(word) > 2 else word for word in words)
                
                title = f"Chapter {chapter_num} - {title_part}"
                self.log(f"📝 Fallback extracted title from filename: '{title}'")
                return title
        
        return None
    
    def _load_metadata(self) -> dict:
        """Load metadata from JSON file"""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                self.log("[DEBUG] Metadata loaded successfully")
                return metadata
            except Exception as e:
                self.log(f"[WARNING] Failed to load metadata.json: {e}")
        else:
            self.log("[WARNING] metadata.json not found, using defaults")
        
        return {}
    
    def _create_book(self, metadata: dict) -> epub.EpubBook:
        """Create and configure EPUB book"""
        book = epub.EpubBook()
        
        # Set identifier
        book.set_identifier(metadata.get("identifier", f"translated-{os.path.basename(self.base_dir)}"))
        
        # Determine title
        book_title = self._determine_book_title(metadata)
        book.set_title(book_title)
        
        # Set language
        book.set_language(metadata.get("language", "en"))
        
        # Add original title if different
        #if metadata.get('original_title') and metadata.get('original_title') != book_title:
        #    book.add_metadata('DC', 'title', metadata['original_title'], {'type': 'original'})
        
        # Set author
        if metadata.get("creator"):
            book.add_author(metadata["creator"])
            self.log(f"[INFO] Set author: {metadata['creator']}")
        
        return book
    
    def _determine_book_title(self, metadata: dict) -> str:
        """Determine the book title from metadata"""
        # Try translated title
        if metadata.get('title') and str(metadata['title']).strip():
            title = str(metadata['title']).strip()
            self.log(f"✅ Using translated title: '{title}'")
            return title
        
        # Try original title
        if metadata.get('original_title') and str(metadata['original_title']).strip():
            title = str(metadata['original_title']).strip()
            self.log(f"⚠️ Using original title: '{title}'")
            return title
        
        # Fallback to directory name
        title = os.path.basename(self.base_dir)
        self.log(f"📁 Using directory name: '{title}'")
        return title
    
    def _create_default_css(self) -> str:
        """Create default CSS for proper chapter formatting"""
        return """
/* Default EPUB CSS */
body {
    margin: 1em;
    padding: 0;
    font-family: serif;
    line-height: 1.6;
}

h1, h2, h3, h4, h5, h6 {
    font-weight: bold;
    margin-top: 1em;
    margin-bottom: 0.5em;
    page-break-after: avoid;
}

h1 {
    font-size: 1.5em;
    text-align: center;
    margin-top: 2em;
    margin-bottom: 2em;
}

p {
    margin: 1em 0;
    text-indent: 0;
}

img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 1em auto;
}

/* Prevent any overlay issues */
* {
    position: static !important;
    z-index: auto !important;
}

/* Remove any floating elements */
.title, [class*="title"] {
    position: static !important;
    float: none !important;
    background: transparent !important;
}
"""

    def _add_css_files(self, book: epub.EpubBook) -> List[epub.EpubItem]:
        """Add CSS files to book"""
        css_items = []
        
        # First, add a default CSS to ensure proper formatting
        default_css = epub.EpubItem(
            uid="css_default",
            file_name="css/default.css",
            media_type="text/css",
            content=FileUtils.ensure_bytes(self._create_default_css())
        )
        book.add_item(default_css)
        css_items.append(default_css)
        self.log("✅ Added default CSS")
        
        # Then add user CSS files
        if not os.path.isdir(self.css_dir):
            return css_items
        
        css_files = [f for f in sorted(os.listdir(self.css_dir)) if f.endswith('.css')]
        self.log(f"[DEBUG] Found {len(css_files)} CSS files")
        
        for css_file in css_files:
            css_path = os.path.join(self.css_dir, css_file)
            try:
                with open(css_path, 'r', encoding='utf-8') as f:
                    css_content = f.read()
                
                css_item = epub.EpubItem(
                    uid=f"css_{css_file}",
                    file_name=f"css/{css_file}",
                    media_type="text/css",
                    content=FileUtils.ensure_bytes(css_content)
                )
                book.add_item(css_item)
                css_items.append(css_item)
                self.log(f"✅ Added CSS: {css_file}")
                
            except Exception as e:
                self.log(f"[WARNING] Failed to add CSS {css_file}: {e}")
        
        return css_items
    
    def _add_fonts(self, book: epub.EpubBook):
        """Add font files to book"""
        if not os.path.isdir(self.fonts_dir):
            return
        
        for font_file in os.listdir(self.fonts_dir):
            font_path = os.path.join(self.fonts_dir, font_file)
            if not os.path.isfile(font_path):
                continue
            
            try:
                mime_type = 'application/font-woff'
                if font_file.endswith('.ttf'):
                    mime_type = 'font/ttf'
                elif font_file.endswith('.otf'):
                    mime_type = 'font/otf'
                elif font_file.endswith('.woff2'):
                    mime_type = 'font/woff2'
                
                with open(font_path, 'rb') as f:
                    book.add_item(epub.EpubItem(
                        uid=f"font_{font_file}",
                        file_name=f"fonts/{font_file}",
                        media_type=mime_type,
                        content=f.read()
                    ))
                self.log(f"✅ Added font: {font_file}")
                
            except Exception as e:
                self.log(f"[WARNING] Failed to add font {font_file}: {e}")
    
    def _process_images(self) -> Tuple[Dict[str, str], Optional[str]]:
        """Process images and find cover"""
        processed_images = {}
        cover_file = None
        
        try:
            if not os.path.isdir(self.images_dir):
                return processed_images, cover_file
            
            # Process all images
            for img in sorted(os.listdir(self.images_dir)):
                path = os.path.join(self.images_dir, img)
                if not os.path.isfile(path):
                    continue
                
                ctype, _ = mimetypes.guess_type(path)
                if ctype and ctype.startswith("image"):
                    safe_name = FileUtils.sanitize_filename(img, allow_unicode=False)
                    
                    # Ensure extension
                    if not os.path.splitext(safe_name)[1]:
                        ext = os.path.splitext(img)[1]
                        if ext:
                            safe_name += ext
                        elif ctype == 'image/jpeg':
                            safe_name += '.jpg'
                        elif ctype == 'image/png':
                            safe_name += '.png'
                    
                    processed_images[img] = safe_name
                    self.log(f"[DEBUG] Found image: {img} -> {safe_name}")
            
            # Find cover (case-insensitive search)
            if processed_images:
                # Search for images starting with "cover" or "front" (case-insensitive)
                cover_prefixes = ['cover', 'front']
                for original_name, safe_name in processed_images.items():
                    name_lower = original_name.lower()
                    if any(name_lower.startswith(prefix) for prefix in cover_prefixes):
                        cover_file = safe_name
                        self.log(f"[DEBUG] Found cover image: {original_name} -> {cover_file}")
                        break
                
                # If no cover-like image found, use first image
                if not cover_file:
                    cover_file = next(iter(processed_images.values()))
                    self.log(f"[DEBUG] Using first image as cover: {cover_file}")
            
        except Exception as e:
            self.log(f"[WARNING] Error processing images: {e}")
            # Return empty results instead of None
        
        return processed_images, cover_file
    
    def _add_images_to_book(self, book: epub.EpubBook, processed_images: Dict[str, str], 
                           cover_file: Optional[str]):
        """Add images to book (except cover)"""
        for original_name, safe_name in processed_images.items():
            if safe_name == cover_file:
                continue
            
            img_path = os.path.join(self.images_dir, original_name)
            try:
                ctype, _ = mimetypes.guess_type(img_path)
                with open(img_path, 'rb') as f:
                    book.add_item(epub.EpubItem(
                        uid=safe_name,
                        file_name=f"images/{safe_name}",
                        media_type=ctype or "image/jpeg",
                        content=f.read()
                    ))
            except Exception as e:
                self.log(f"[WARNING] Failed to embed image {original_name}: {e}")
    
    def _create_cover_page(self, book: epub.EpubBook, cover_file: str, 
                          processed_images: Dict[str, str], css_items: List[epub.EpubItem],
                          metadata: dict) -> Optional[epub.EpubHtml]:
        """Create cover page"""
        # Find original filename
        original_cover = None
        for orig, safe in processed_images.items():
            if safe == cover_file:
                original_cover = orig
                break
        
        if not original_cover:
            return None
        
        cover_path = os.path.join(self.images_dir, original_cover)
        try:
            with open(cover_path, 'rb') as f:
                cover_data = f.read()
            
            # Add cover image
            cover_img = epub.EpubItem(
                uid="cover-image",
                file_name=f"images/{cover_file}",
                media_type=mimetypes.guess_type(cover_path)[0] or "image/jpeg",
                content=cover_data
            )
            book.add_item(cover_img)
            
            # Set cover metadata
            cover_img.properties = ["cover-image"]
            book.add_metadata('http://purl.org/dc/elements/1.1/', 'cover', 'cover-image')
            
            # Create cover page
            cover_page = epub.EpubHtml(
                title="Cover",
                file_name="cover.xhtml",
                lang=metadata.get("language", "en")
            )
            
            # Build cover HTML
            cover_css_links = [f"css/{item.file_name.split('/')[-1]}" for item in css_items]
            cover_body = f'''<div style="text-align: center;">
<img src="images/{cover_file}" alt="Cover" style="max-width: 100%; height: auto;" />
</div>'''
            
            cover_page.content = FileUtils.ensure_bytes(
                XHTMLConverter.validate(
                    XHTMLConverter.ensure_compliance(cover_body, "Cover", cover_css_links)
                )
            )
            
            book.add_item(cover_page)
            self.log(f"✅ Set cover image: {cover_file}")
            return cover_page
            
        except Exception as e:
            self.log(f"[WARNING] Failed to add cover: {e}")
            return None
    
    def _process_chapter_images(self, xhtml_content: str, processed_images: Dict[str, str]) -> str:
        """Process image paths in chapter content"""
        try:
            soup = BeautifulSoup(xhtml_content, 'html.parser')
            changed = False
            
            for img in soup.find_all('img'):
                src = img.get('src', '')
                if not src:
                    continue
                
                # Get the base filename
                basename = os.path.basename(src.split('?')[0])
                
                # Look up the safe name
                if basename in processed_images:
                    safe_name = processed_images[basename]
                    new_src = f"images/{safe_name}"
                    
                    if src != new_src:
                        img['src'] = new_src
                        changed = True
                
                # Ensure alt attribute exists (required for XHTML)
                if not img.get('alt'):
                    img['alt'] = ''
                    changed = True
            
            if changed:
                # Get just the body content
                if soup.body:
                    body_content = ''.join(str(child) for child in soup.body.children)
                else:
                    body_content = str(soup)
                
                # Get title
                title = "Chapter"
                if soup.title:
                    title = soup.title.get_text()
                
                # Get CSS links
                css_links = []
                if soup.head:
                    for link in soup.head.find_all('link', rel='stylesheet'):
                        href = link.get('href', '')
                        if href:
                            css_links.append(href)
                
                # Rebuild with clean structure
                return XHTMLConverter._build_xhtml(title, body_content, css_links)
            
            return xhtml_content
            
        except Exception as e:
            self.log(f"[WARNING] Failed to process images in chapter: {e}")
            return xhtml_content
    
    def _create_gallery_page(self, book: epub.EpubBook, images: List[str],
                            css_items: List[epub.EpubItem], metadata: dict) -> epub.EpubHtml:
        """Create image gallery page"""
        gallery_page = epub.EpubHtml(
            title="Gallery",
            file_name="gallery.xhtml",
            lang=metadata.get("language", "en")
        )
        
        gallery_body_parts = ['<h1>Image Gallery</h1>']
        for img in images:
            gallery_body_parts.append(
                f'<div style="text-align: center; margin: 20px;">'
                f'<img src="images/{img}" alt="{img}" />'
                f'</div>'
            )
        
        css_links = [f"css/{item.file_name.split('/')[-1]}" for item in css_items]
        
        gallery_page.content = FileUtils.ensure_bytes(
            XHTMLConverter.validate(
                XHTMLConverter.ensure_compliance(
                    '\n'.join(gallery_body_parts), "Gallery", css_links
                )
            )
        )
        
        book.add_item(gallery_page)
        return gallery_page
        
    def _create_nav_content(self, toc_items, book_title="Book"):
        """Create navigation content manually"""
        nav_content = '''<?xml version="1.0" encoding="utf-8"?>
    <!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">
    <html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
    <head>
    <title>Table of Contents</title>
    </head>
    <body>
    <nav epub:type="toc" id="toc">
    <h1>Table of Contents</h1>
    <ol>'''
        
        for item in toc_items:
            if hasattr(item, 'title') and hasattr(item, 'file_name'):
                nav_content += f'\n<li><a href="{item.file_name}">{ContentProcessor.safe_escape(item.title)}</a></li>'
        
        nav_content += '''
    </ol>
    </nav>
    </body>
    </html>'''
        
        return nav_content

    def _finalize_book(self, book: epub.EpubBook, spine: List, toc: List, 
                      cover_file: Optional[str]):
        """Finalize book structure"""
        # Check if we should use NCX-only
        use_ncx_only = os.environ.get('FORCE_NCX_ONLY', '0') == '1'
        
        # Check if first item in spine is a cover
        has_cover = False
        cover_item = None
        if spine and len(spine) > 0:
            first_item = spine[0]
            if hasattr(first_item, 'title') and first_item.title == "Cover":
                has_cover = True
                cover_item = first_item
                spine = spine[1:]  # Remove cover from spine temporarily
        
        # Set TOC
        book.toc = toc
        
        # Add NCX
        ncx = epub.EpubNcx()
        book.add_item(ncx)
        
        if use_ncx_only:
            self.log(f"[INFO] NCX-only navigation forced - {len(toc)} chapters")
            
            # Build final spine: Cover (if exists) → Chapters
            final_spine = []
            if has_cover:
                final_spine.append(cover_item)
            final_spine.extend(spine)
            
            book.spine = final_spine
            
            self.log("📖 Using EPUB 3.3 with NCX navigation only")
            if has_cover:
                self.log("📖 Reading order: Cover → Chapters")
            else:
                self.log("📖 Reading order: Chapters")
                
        else:
            # Normal EPUB3 processing with Nav
            self.log(f"[INFO] EPUB3 format - {len(toc)} chapters")
            
            # Create Nav with manual content
            nav = epub.EpubNav()
            nav.content = self._create_nav_content(toc, book.title).encode('utf-8')
            nav.uid = 'nav'
            nav.file_name = 'nav.xhtml'
            book.add_item(nav)
            
            # Build final spine: Cover (if exists) → Nav → Chapters
            final_spine = []
            if has_cover:
                final_spine.append(cover_item)
            final_spine.append(nav)
            final_spine.extend(spine)
            
            book.spine = final_spine
            
            self.log("📖 Using EPUB3 format with full navigation")
            if has_cover:
                self.log("📖 Reading order: Cover → Table of Contents → Chapters")
            else:
                self.log("📖 Reading order: Table of Contents → Chapters")

    def _write_epub(self, book: epub.EpubBook, metadata: dict):
        """Write EPUB file with automatic format selection"""
        # Determine output filename
        book_title = book.title
        if book_title and book_title != os.path.basename(self.output_dir):
            safe_filename = FileUtils.sanitize_filename(book_title, allow_unicode=True)
            out_path = os.path.join(self.output_dir, f"{safe_filename}.epub")
        else:
            base_name = os.path.basename(self.output_dir)
            out_path = os.path.join(self.output_dir, f"{base_name}.epub")
        
        self.log(f"\n[DEBUG] Writing EPUB to: {out_path}")
        
        # Always write as EPUB3
        try:
            opts = {'epub3': True}
            epub.write_epub(out_path, book, opts)
            self.log("[SUCCESS] Written as EPUB 3.3")
            
        except Exception as e:
            self.log(f"[ERROR] Write failed: {e}")
            raise
        
        # Verify the file
        if os.path.exists(out_path):
            file_size = os.path.getsize(out_path)
            if file_size > 0:
                self.log(f"✅ EPUB created: {out_path}")
                self.log(f"📊 File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
                self.log("📝 Format: EPUB 3.3")
            else:
                raise Exception("EPUB file is empty")
        else:
            raise Exception("EPUB file was not created")
    
    def _show_summary(self, chapter_titles_info: Dict[int, Tuple[str, float, str]],
                     css_items: List[epub.EpubItem]):
        """Show compilation summary"""
        if chapter_titles_info:
            high = sum(1 for _, (_, conf, _) in chapter_titles_info.items() if conf > 0.7)
            medium = sum(1 for _, (_, conf, _) in chapter_titles_info.items() if 0.4 < conf <= 0.7)
            low = sum(1 for _, (_, conf, _) in chapter_titles_info.items() if conf <= 0.4)
            
            self.log(f"\n📊 Title Extraction Summary:")
            self.log(f"   • High confidence: {high} chapters")
            self.log(f"   • Medium confidence: {medium} chapters")
            self.log(f"   • Low confidence: {low} chapters")
        
        if css_items:
            self.log(f"\n✅ Successfully embedded {len(css_items)} CSS files")
        # Gallery status
        if os.environ.get('DISABLE_EPUB_GALLERY', '0') == '1':
            self.log("\n📷 Image Gallery: Disabled by user preference")
        
        self.log("\n📱 Compatibility Notes:")
        self.log("   • XHTML 1.1 compliant")
        self.log("   • All tags properly closed")
        self.log("   • Special characters escaped")
        self.log("   • Extracted translated titles")
        self.log("   • Enhanced entity decoding")


# Main entry point
def compile_epub(base_dir: str, log_callback: Optional[Callable] = None):
    """Compile translated HTML files into EPUB"""
    compiler = EPUBCompiler(base_dir, log_callback)
    compiler.compile()


# Legacy alias
fallback_compile_epub = compile_epub


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python epub_converter.py <directory_path>")
        sys.exit(1)
    
    directory_path = sys.argv[1]
    
    try:
        compile_epub(directory_path)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
