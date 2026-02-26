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
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from unified_api_client import UnifiedClient
except ImportError:
    UnifiedClient = None

try:
    from translate_headers_standalone import run_translation as run_standalone_header_translation
except ImportError:
    run_standalone_header_translation = None

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
_stop_flag = False


def set_stop_flag(value: bool):
    """Set the stop flag for EPUB converter"""
    global _stop_flag
    _stop_flag = value


def is_stop_requested() -> bool:
    """Check if stop has been requested"""
    global _stop_flag
    return _stop_flag


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
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)
        if not text:
            return text
        
        # Fix common encoding issues first
        for bad, good in cls.ENCODING_FIXES.items():
            text = text.replace(bad, good)
        
        # Multiple passes to handle nested/double-encoded entities
        max_passes = 3
        for _ in range(max_passes):
            prev_text = text
            
            # Use html module for standard decoding (this handles &lt;, &gt;, etc.)
            text = html_module.unescape(text)
            
            if text == prev_text:
                break
        
        # Apply any remaining entity replacements
        for entity, char in cls.ENTITY_MAP.items():
            text = text.replace(entity, char)
        
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
        """Escape XML special characters for use in XHTML titles/attributes"""
        if text is None:
            return ""
        if not isinstance(text, str):
            try:
                text = str(text)
            except Exception:
                return ""
        # Use html.escape to handle &, <, > and quotes; then escape single quotes
        escaped = html_module.escape(text, quote=True)
        escaped = escaped.replace("'", "&apos;")
        return escaped


class TitleExtractor:
    """Handles extraction of titles from HTML content - UPDATED WITH UNICODE PRESERVATION"""
    
    @staticmethod
    def extract_from_html(html_content: str, chapter_num: Optional[int] = None, 
                         filename: Optional[str] = None) -> Tuple[str, float]:
        """Extract title from HTML content with confidence score - KEEP ALL HEADERS INCLUDING NUMBERS"""
        try:
            # Decode entities first - PRESERVES UNICODE
            html_content = HTMLEntityDecoder.decode(html_content)
            
            soup = BeautifulSoup(html_content, 'lxml', from_encoding='utf-8')
            candidates = []
            
            # Strategy 1: <title> tag (highest confidence)
            title_tag = soup.find('title')
            if title_tag and title_tag.string:
                title_text = HTMLEntityDecoder.decode(title_tag.string.strip())
                if title_text and len(title_text) > 0 and title_text.lower() not in ['untitled', 'chapter', 'document']:
                    candidates.append((title_text, 0.95, "title_tag"))
            
            # Strategy 2: h1 tags (very high confidence)
            h1_tags = soup.find_all('h1')
            for i, h1 in enumerate(h1_tags[:3]):  # Check first 3 h1 tags
                text = HTMLEntityDecoder.decode(h1.get_text(strip=True))
                if text and len(text) < 300:
                    # First h1 gets highest confidence
                    confidence = 0.9 if i == 0 else 0.85
                    candidates.append((text, confidence, f"h1_tag_{i+1}"))
            
            # Strategy 3: h2 tags (high confidence)
            h2_tags = soup.find_all('h2')
            for i, h2 in enumerate(h2_tags[:3]):  # Check first 3 h2 tags
                text = HTMLEntityDecoder.decode(h2.get_text(strip=True))
                if text and len(text) < 250:
                    # First h2 gets highest confidence among h2s
                    confidence = 0.8 if i == 0 else 0.75
                    candidates.append((text, confidence, f"h2_tag_{i+1}"))
            
            # Strategy 4: h3 tags (moderate confidence)
            h3_tags = soup.find_all('h3')
            for i, h3 in enumerate(h3_tags[:3]):  # Check first 3 h3 tags
                text = HTMLEntityDecoder.decode(h3.get_text(strip=True))
                if text and len(text) < 200:
                    confidence = 0.7 if i == 0 else 0.65
                    candidates.append((text, confidence, f"h3_tag_{i+1}"))
            
            # Strategy 5: Bold text in first elements (lower confidence)
            first_elements = soup.find_all(['p', 'div'])[:5]
            for elem in first_elements:
                for bold in elem.find_all(['b', 'strong'])[:2]:  # Limit to first 2 bold items
                    bold_text = HTMLEntityDecoder.decode(bold.get_text(strip=True))
                    if bold_text and 2 <= len(bold_text) <= 150:
                        candidates.append((bold_text, 0.6, "bold_text"))
            
            # Strategy 6: Center-aligned text (common for chapter titles)
            center_elements = soup.find_all(['center', 'div', 'p'], 
                                           attrs={'align': 'center'}) or \
                             soup.find_all(['div', 'p'], 
                                         style=lambda x: x and 'text-align' in x and 'center' in x)
            
            for center in center_elements[:3]:  # Check first 3 centered elements
                text = HTMLEntityDecoder.decode(center.get_text(strip=True))
                if text and 2 <= len(text) <= 200:
                    candidates.append((text, 0.65, "centered_text"))
            
            # Strategy 7: All-caps text (common for titles in older books)
            for elem in soup.find_all(['h1', 'h2', 'h3', 'p', 'div'])[:10]:
                text = elem.get_text(strip=True)
                # Check if text is mostly uppercase
                if text and len(text) > 2 and text.isupper():
                    decoded_text = HTMLEntityDecoder.decode(text)
                    # Keep it as-is (don't convert to title case automatically)
                    candidates.append((decoded_text, 0.55, "all_caps_text"))
            
            # Strategy 8: Patterns in first paragraph
            first_p = soup.find('p')
            if first_p:
                p_text = HTMLEntityDecoder.decode(first_p.get_text(strip=True))
                
                # Look for "Chapter X: Title" patterns
                chapter_pattern = re.match(
                    r'^(Chapter\s+[\dIVXLCDM]+\s*[:\-\u2013\u2014]\s*)(.{2,100})(?:\.|$)',
                    p_text, re.IGNORECASE
                )
                if chapter_pattern:
                    # Extract just the title part after "Chapter X:"
                    title_part = chapter_pattern.group(2).strip()
                    if title_part:
                        candidates.append((title_part, 0.8, "paragraph_pattern_title"))
                    # Also add the full "Chapter X: Title" as a lower confidence option
                    full_title = chapter_pattern.group(0).strip().rstrip('.')
                    candidates.append((full_title, 0.75, "paragraph_pattern_full"))
                elif len(p_text) <= 100 and len(p_text) > 2:
                    # Short first paragraph might be the title
                    candidates.append((p_text, 0.4, "paragraph_standalone"))
            
            # Strategy 9: Filename
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
                    # Clean the title but keep roman numerals and short titles
                    title = TitleExtractor.clean_title(title)
                    
                    # Don't reject short titles (like "III", "IX") - they're valid!
                    if title and len(title) > 0:
                        # Don't apply is_valid_title check too strictly
                        # Roman numerals and chapter numbers are valid titles
                        if title not in unique_candidates or unique_candidates[title][1] < confidence:
                            unique_candidates[title] = (title, confidence, source)
                
                if unique_candidates:
                    sorted_candidates = sorted(unique_candidates.values(), key=lambda x: x[1], reverse=True)
                    best_title, best_confidence, best_source = sorted_candidates[0]
                    
                    # Log what we found for debugging (only if debug mode is enabled)
                    import os
                    debug_mode_enabled = os.environ.get('DEBUG_MODE', '0') == '1'
                    if debug_mode_enabled:
                        log(f"[DEBUG] Best title candidate: '{best_title}' (confidence: {best_confidence:.2f}, source: {best_source})")
                    
                    return best_title, best_confidence
            
            # Fallback - only use generic chapter number if we really found nothing
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
        """Clean and normalize extracted title - trust HTML headers, minimal cleanup only"""
        if not title:
            return ""
        
        # Decode HTML entities - PRESERVES UNICODE
        title = HTMLEntityDecoder.decode(title)
        
        # Remove HTML tags only
        title = re.sub(r'<[^>]+>', '', title)
        
        # Normalize whitespace (convert non-breaking spaces, multiple spaces, etc.)
        title = re.sub(r'[\xa0\u2000-\u200a\u202f\u205f\u3000]+', ' ', title)
        title = re.sub(r'\s+', ' ', title).strip()
        
        # Normalize Unicode to NFC form
        title = unicodedata.normalize('NFC', title)
        
        # Remove invisible zero-width characters only
        title = re.sub(r'[\u200b\u200c\u200d\u200e\u200f\ufeff]', '', title)
        
        # Final whitespace cleanup
        title = ' '.join(title.split())
        
        # Truncate if excessively long (safety check only)
        if len(title) > 200:
            truncated = title[:197]
            last_space = truncated.rfind(' ')
            if last_space > 150:
                truncated = truncated[:last_space]
            title = truncated + "..."
        
        return title
    
    @staticmethod
    def is_valid_title(title: str) -> bool:
        """Check if extracted title is valid - ACCEPT SHORT TITLES LIKE ROMAN NUMERALS"""
        if not title:
            return False
        
        # Accept any non-empty title after cleaning
        # Don't reject roman numerals or short titles
        
        # Only reject truly invalid patterns
        invalid_patterns = [
            r'^untitled$',  # Just "untitled"
            r'^chapter$',   # Just "chapter" without a number
            r'^document$',  # Just "document"
        ]
        
        for pattern in invalid_patterns:
            if re.match(pattern, title.lower().strip()):
                return False
        
        # Skip obvious filler phrases
        filler_phrases = [
            'click here', 'read more', 'continue reading', 'next chapter',
            'previous chapter', 'table of contents', 'back to top'
        ]
        
        title_lower = title.lower().strip()
        if any(phrase in title_lower for phrase in filler_phrases):
            return False
        
        # Accept everything else, including roman numerals and short titles
        return True


class XHTMLConverter:
    """Handles XHTML conversion and compliance"""
    
    # Default language for generated XHTML (used for html[@lang] and html[@xml:lang])
    # This will be synchronized with the EPUB book language by EPUBCompiler.
    DEFAULT_LANG = "en"

    @classmethod
    def set_default_language(cls, lang_code: str):
        """Set default language code used for html lang/xml:lang attributes"""
        if not lang_code:
            return
        try:
            cls.DEFAULT_LANG = str(lang_code).strip() or "en"
        except Exception:
            cls.DEFAULT_LANG = "en"
    
    @staticmethod
    def ensure_compliance(html_content: str, title: str = "Chapter", 
                         css_links: Optional[List[str]] = None) -> str:
        """Ensure HTML content is XHTML-compliant while PRESERVING story tags"""
        try:
            import html
            import re
            
            # Unescape HTML entities but PRESERVE &lt; and &gt; so fake angle brackets in narrative
            # text don't become real tags (which breaks parsing across paragraphs like the sample).
            if any(ent in html_content for ent in ['&amp;', '&quot;', '&#', '&lt;', '&gt;']):
                # Temporarily protect &lt; and &gt; (both cases) from unescaping
                placeholder_lt = '\ue000'
                placeholder_gt = '\ue001'
                html_content = html_content.replace('&lt;', placeholder_lt).replace('&LT;', placeholder_lt)
                html_content = html_content.replace('&gt;', placeholder_gt).replace('&GT;', placeholder_gt)
                # Unescape remaining entities
                html_content = html.unescape(html_content)
                # Restore protected angle bracket entities
                html_content = html_content.replace(placeholder_lt, '&lt;').replace(placeholder_gt, '&gt;')
            
            # Strip out ANY existing DOCTYPE, XML declaration, or html wrapper
            # We only want the body content
            
            # Try to extract just body content
            body_match = re.search(r'<body[^>]*>(.*?)</body>', html_content, re.DOTALL | re.IGNORECASE)
            if body_match:
                html_content = body_match.group(1)
            else:
                # No body tags, strip any DOCTYPE/html tags if present
                html_content = re.sub(r'<\?xml[^>]*\?>', '', html_content)
                html_content = re.sub(r'<!DOCTYPE[^>]*>', '', html_content)
                html_content = re.sub(r'</?html[^>]*>', '', html_content)
                html_content = re.sub(r'<head[^>]*>.*?</head>', '', html_content, flags=re.DOTALL)
            
            # Now process the content normally
            # Fix broken attributes with ="" pattern
            def fix_broken_attributes_only(match):
                tag_content = match.group(0)
                
                if '=""' in tag_content and tag_content.count('=""') > 2:
                    tag_match = re.match(r'<(\w+)', tag_content)
                    if tag_match:
                        tag_name = tag_match.group(1)
                        words = re.findall(r'(\w+)=""', tag_content)
                        if words:
                            content = ' '.join(words)
                            return f'<{tag_name}>{content}</{tag_name}>'
                    return ''
                
                return tag_content
            
            # Fix <p"Text... -> <p>"Text... and orphaned < inside <p>
            def _fix_malformed_p_tags(text: str) -> str:
                # Part 1: Fix <p"Text... -> <p>"Text...
                # This handles the specific case where the opening p tag is malformed as <p"
                text = re.sub(r'<p"([^>]*?</p>)', r'<p>"\1', text, flags=re.IGNORECASE | re.DOTALL)
                
                # Part 1.5: Fix p>Text... -> <p>Text...
                # Handles cases where the opening bracket is missing for the p tag
                # Only matches if it starts a line or follows a closing tag, to be safe
                text = re.sub(r'(^|>)\s*p>([^<]*?</p>)', r'\1<p>\2', text, flags=re.IGNORECASE | re.DOTALL)

                # Part 2: Fix orphaned < inside <p> tags (e.g. <p><Yeah...)
                def _process_p_content(m):
                    open_tag = m.group(1)
                    content = m.group(2)
                    close_tag = m.group(3)
                    
                    new_content = []
                    last_pos = 0
                    
                    # Iterate through all < characters in the content
                    # We use a manual loop to handle logic correctly
                    i = 0
                    while i < len(content):
                        if content[i] == '<':
                            # Found a <. Check if it's a valid tag or orphaned/text
                            # Append everything before this <
                            new_content.append(content[last_pos:i])
                            
                            # Check for valid tag structure
                            # 1. Must have a closing >
                            next_gt = content.find('>', i)
                            
                            should_escape = False
                            if next_gt == -1:
                                # No closing >, definitely orphaned
                                should_escape = True
                            else:
                                # Has closing >. Check interior.
                                inner_text = content[i+1:next_gt]
                                
                                # If inner text contains <, then the first < is likely text 
                                # (e.g. "x < y and y < z") unless it's nested tags which is rare inside simple P
                                # But simpler check: valid tag names start with alpha or /alpha
                                # and generally don't contain spaces immediately unless attributes
                                
                                # Check if it looks like a tag: <tagname...> or </tagname...>
                                # Strip whitespace to handle <br >
                                inner_stripped = inner_text.strip()
                                tag_match = re.match(r'^/?([a-zA-Z0-9]+)', inner_stripped)
                                
                                if not tag_match:
                                    # < 3 or < . or < ? (though <? is processing instruction)
                                    should_escape = True
                                else:
                                    # It has a tag-like name.
                                    # Check against known HTML tags if we want to be strict, 
                                    # but user said "if it's anything else, then it's a whole sentence"
                                    # The case <Yeah... matches [a-zA-Z]+ but had NO >. We handled that above.
                                    
                                    # What if we have <Yeah >? 
                                    # User's example was <Yeah... (no >). 
                                    # If we have <Yeah >, it is technically a tag "Yeah".
                                    # But let's assume if it's not a standard HTML tag, it might be text?
                                    # The user didn't explicitly ask to fix <Text>, only orphaned ones.
                                    # So we stick to the "orphaned" logic (missing >) OR "obviously not a tag" logic.
                                    
                                    # For the purpose of this fix, the "missing >" check handles the user's <Yeah example.
                                    # The <I haven't example also has "missing >".
                                    
                                    # One edge case: <p>Text <br> Text</p>
                                    # <br> has > and matches [a-zA-Z]. Kept.
                                    pass

                            if should_escape:
                                new_content.append('&lt;')
                                last_pos = i + 1
                            else:
                                # It's a tag (or looks enough like one), keep the <
                                new_content.append('<')
                                last_pos = i + 1
                        
                        i += 1
                        
                    new_content.append(content[last_pos:])
                    return f"{open_tag}{''.join(new_content)}{close_tag}"

                return re.sub(r'(<p[^>]*>)(.*?)(</p>)', _process_p_content, text, flags=re.IGNORECASE | re.DOTALL)

            html_content = _fix_malformed_p_tags(html_content)

            html_content = re.sub(r'<[^>]*?=\"\"[^>]*?>', fix_broken_attributes_only, html_content)

            # Sanitize attributes that contain a colon (:) but are NOT valid namespaces.
            # Example: <status effects:="" high="" temperature="" unconscious=""></status>
            # becomes: <status data-effects="" high="" temperature="" unconscious=""></status>
            def _sanitize_colon_attributes_in_tags(text: str) -> str:
                # Process only inside start tags; skip closing tags, comments, doctypes, processing instructions
                def _process_tag(tag_match):
                    tag = tag_match.group(0)
                    if tag.startswith('</') or tag.startswith('<!') or tag.startswith('<?'):
                        return tag
                    
                    def _attr_repl(m):
                        before, name, eqval = m.group(1), m.group(2), m.group(3)
                        lname = name.lower()
                        # Preserve known namespace attributes
                        if (
                            lname.startswith('xml:') or lname.startswith('xlink:') or lname.startswith('epub:') or
                            lname == 'xmlns' or lname.startswith('xmlns:')
                        ):
                            return m.group(0)
                        if ':' not in name:
                            return m.group(0)
                        # Replace colon(s) with dashes and prefix with data-
                        safe = re.sub(r'[:]+', '-', name).strip('-')
                        safe = re.sub(r'[^A-Za-z0-9_.-]', '-', safe) or 'attr'
                        if not safe.startswith('data-'):
                            safe = 'data-' + safe
                        return f'{before}{safe}{eqval}'
                    
                    # Replace attributes with colon in the name (handles both single and double quoted values)
                    tag = re.sub(r'(\s)([A-Za-z_:][A-Za-z0-9_.:-]*:[A-Za-z0-9_.:-]*)(\s*=\s*(?:"[^"]*"|\'[^\']*\'))', _attr_repl, tag)
                    return tag
                
                return re.sub(r'<[^>]+>', _process_tag, text)
            
            html_content = _sanitize_colon_attributes_in_tags(html_content)
            
            # Convert only "story tags" whose TAG NAME contains a colon (e.g., <System:Message>),
            # but DO NOT touch valid HTML/SVG tags where colons appear in attributes (e.g., style="color:red" or xlink:href)
            # and DO NOT touch namespaced tags like <svg:rect>.
            allowed_ns_prefixes = {"svg", "math", "xlink", "xml", "xmlns", "epub"}

            # NEW: Fix for "Empty Attribute Tags" (e.g. <unique ability=""></unique>)
            # Transforms hallucinated patterns like <unique ability=""></unique> into <unique ability>
            if os.getenv('FIX_EMPTY_ATTR_TAGS_EPUB', '0') == '1':
                def _escape_empty_attr_tags(text: str) -> str:
                    # Known HTML tags to preserve
                    known_tags = {
                        'html','head','body','title','meta','link','style','script','noscript',
                        'p','div','span','br','hr','img','a','h1','h2','h3','h4','h5','h6',
                        'ul','ol','li','dl','dt','dd',
                        'pre','code','em','strong','b','i','u','s','strike','del','ins','mark','small','sub','sup',
                        'table','thead','tbody','tr','td','th','caption','col','colgroup',
                        'blockquote','q','cite',
                        'section','article','header','footer','nav','main','aside','details','summary',
                        'figure','figcaption',
                        'form','input','button','select','option','textarea','label','fieldset','legend',
                        'iframe','canvas','svg','math',
                        'video','audio','source','track','embed','object','param',
                        'map','area',
                        'center', 'font', 'base'
                    }
                    
                    # Transform: <Tag Attr="">Content</Tag>  -->  &lt;Tag Attr&gt;Content
                    # This removes the empty attribute value and the closing tag, creating a visual "token" style representation
                    def _repl_pair(m):
                        tagname = m.group(1)
                        if tagname.lower() in known_tags:
                            return m.group(0)
                        attrname = m.group(2)
                        content = m.group(3)
                        return f"&lt;{tagname} {attrname}&gt;{content}"
                    
                    # Match <Tag Attr=""></Tag> (allow whitespace and content)
                    text = re.sub(r'<([a-zA-Z0-9_\-]+)\s+([a-zA-Z0-9_\-]+)=""\s*>(.*?)</\1>', _repl_pair, text, flags=re.DOTALL)
                    
                    return text
                
                html_content = _escape_empty_attr_tags(html_content)

            def _escape_story_tag(match):
                full_tag = match.group(0)   # Entire <...> or </...>
                tag_name = match.group(1)   # The tag name possibly containing ':'
                prefix = tag_name.split(':', 1)[0].lower()
                # If this is a known namespace prefix (e.g., svg:rect), leave it alone
                if prefix in allowed_ns_prefixes:
                    return full_tag
                # Otherwise, treat as a story/fake tag and replace angle brackets with Chinese brackets
                return full_tag.replace('<', '《').replace('>', '》')

            # Escape invalid story tags (tag names containing ':') so they render literally with angle brackets.
            allowed_ns_prefixes = {"svg", "math", "xlink", "xml", "xmlns", "epub"}
            def _escape_story_tag_entities(m):
                tagname = m.group(1)
                prefix = tagname.split(':', 1)[0].lower()
                if prefix in allowed_ns_prefixes:
                    return m.group(0)
                tag_text = m.group(0)
                return tag_text.replace('<', '&lt;').replace('>', '&gt;')
            # Apply in order: self-closing, opening, closing
            html_content = re.sub(r'<([A-Za-z][\w.-]*:[\w.-]*)\s*([^>]*)/>', _escape_story_tag_entities, html_content)
            html_content = re.sub(r'<([A-Za-z][\w.-]*:[\w.-]*)\s*([^>]*)>', _escape_story_tag_entities, html_content)
            html_content = re.sub(r'</([A-Za-z][\w.-]*:[\w.-]*)\s*>', _escape_story_tag_entities, html_content)

            # PREVENT malformed "fake tags" like <You are a farmer.> from being parsed as tags
            # We only target angle-bracketed text that has spaces and NO '=' (so it's not real attributes)
            # and ends with either '>' or the entity '&gt;'.
            def _escape_plaintext_angle_brackets(txt: str) -> str:
                def repl(m):
                    inner = m.group(1)
                    # If looks like a real tag (has '=' or '/') keep it
                    # Check for start chars /!? or if it has attributes (=)
                    if '=' in inner or inner.strip().startswith(('/', '!', '?')):
                        return m.group(0)
                    
                    # If the first token is a known HTML tag name, keep it
                    tokens = inner.strip().split()
                    if not tokens:
                        return m.group(0)
                        
                    first = tokens[0].lower()
                    
                    # Handle self-closing tags like <br/> by removing trailing slash
                    if first.endswith('/'):
                        first = first[:-1]
                    
                    known = {
                        'html','head','body','title','meta','link','style','script','noscript',
                        'p','div','span','br','hr','img','a','h1','h2','h3','h4','h5','h6',
                        'ul','ol','li','dl','dt','dd',
                        'pre','code','em','strong','b','i','u','s','strike','del','ins','mark','small','sub','sup',
                        'table','thead','tbody','tr','td','th','caption','col','colgroup',
                        'blockquote','q','cite',
                        'section','article','header','footer','nav','main','aside','details','summary',
                        'figure','figcaption',
                        'form','input','button','select','option','textarea','label','fieldset','legend',
                        'iframe','canvas','svg','math',
                        'video','audio','source','track','embed','object','param',
                        'map','area',
                        'center', 'font', 'base'
                    }
                    if first in known:
                        return m.group(0)
                        
                    # Otherwise, treat as narrative text in angle brackets and escape
                    return f'&lt;{inner}&gt;'
                
                # Match <...> where content matches non-brackets.
                # Allow single words without spaces (e.g. <luck>)
                pattern = r'<([^<>]+)>'
                txt = re.sub(pattern, repl, txt)
                # Also handle cases where closing bracket is already an entity
                pattern_gt = r'<([^<>]+)&gt;'
                txt = re.sub(pattern_gt, lambda m: f'&lt;{m.group(1)}&gt;', txt)
                return txt

            html_content = _escape_plaintext_angle_brackets(html_content)
            
            # Parse with lxml
            from lxml import html as lxml_html, etree
            
            parser = lxml_html.HTMLParser(recover=True)
            doc = lxml_html.document_fromstring(f"<div>{html_content}</div>", parser=parser)
            
            # Get the content back
            # Use HTML method if enabled (better whitespace preservation for buggy readers like Freda)
            # but may reduce XHTML compliance. Default: xml (strict XHTML)
            serialize_method = 'html' if os.getenv('EPUB_USE_HTML_METHOD', '0') == '1' else 'xml'
            body_xhtml = etree.tostring(doc, method=serialize_method, encoding='unicode')
            # Remove the wrapper div we added
            body_xhtml = re.sub(r'^<div[^>]*>|</div>$', '', body_xhtml)
            
            # Optionally replace angle-bracket entities with Chinese brackets
            # Default behavior: keep them as entities (&lt; &gt;) so the output preserves the original text
            bracket_style = os.getenv('ANGLE_BRACKET_OUTPUT', 'entity').lower()
            if '&lt;' in body_xhtml or '&gt;' in body_xhtml:
                if bracket_style in ('cjk', 'chinese', 'cjk_brackets'):
                    body_xhtml = body_xhtml.replace('&lt;', '《').replace('&gt;', '》')
                # else: keep as entities
            
            # Build our own clean XHTML document
            return XHTMLConverter._build_xhtml(title, body_xhtml, css_links)
            
        except Exception as e:
            log(f"[WARNING] Failed to ensure XHTML compliance: {e}")
            import traceback
            log(f"[DEBUG] Full traceback:\n{traceback.format_exc()}")
            log(f"[DEBUG] Failed chapter title: {title}")
            log(f"[DEBUG] First 500 chars of input: {html_content[:500] if html_content else 'EMPTY'}")
            
            return XHTMLConverter._build_fallback_xhtml(title)
        
    @staticmethod
    def _build_xhtml(title: str, body_content: str, css_links: Optional[List[str]] = None) -> str:
        """Build XHTML document"""
        if not body_content.strip():
            body_content = '<p>Empty chapter</p>'
        
        title = ContentProcessor.safe_escape(title)
        body_content = XHTMLConverter._ensure_xml_safe_readable(body_content)
        
        xml_declaration = '<?xml version="1.0" encoding="utf-8"?>'
        doctype = '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">'
        
        # Use class-level default language for html element language attributes
        lang = getattr(XHTMLConverter, "DEFAULT_LANG", "en")
        
        xhtml_parts = [
            xml_declaration,
            doctype,
            f'<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" xml:lang="{lang}" lang="{lang}">',
            '<head>',
            '<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />',
            f'<title>{title}</title>'
        ]
        
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
        """Ensure content is XML-safe"""
        content = re.sub(
            r'&(?!(?:'
            r'[a-zA-Z][a-zA-Z0-9]{0,30};|'
            r'#[0-9]{1,7};|'
            r'#x[0-9a-fA-F]{1,6};'
            r'))',
            '&amp;',
            content
        )
        return content
    
    @staticmethod
    def _build_fallback_xhtml(title: str) -> str:
        """Build minimal fallback XHTML"""
        safe_title = re.sub(r'[<>&"\']+', '', str(title))
        if not safe_title:
            safe_title = "Chapter"
        
        lang = getattr(XHTMLConverter, "DEFAULT_LANG", "en")
        
        return f'''<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="{lang}" lang="{lang}">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
<title>{ContentProcessor.safe_escape(safe_title)}</title>
</head>
<body>
<p>Error processing content. Please check the source file.</p>
</body>
</html>'''
    
    
    @staticmethod
    def validate(content: str) -> str:
        """Validate and fix XHTML content - WITH DEBUGGING"""
        import re
        # Ensure XML declaration
        if not content.strip().startswith('<?xml'):
            content = '<?xml version="1.0" encoding="utf-8"?>\n' + content
        
        # Remove control characters
        content = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', content)
        
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
        

        # Fix unquoted attributes
        try:
            content = re.sub(r'<([^>]+)\s+(\w+)=([^\s"\'>]+)([>\s])', r'<\1 \2="\3"\4', content)
        except re.error:
            pass  # Skip if regex fails      

        # Sanitize invalid colon-containing attribute names (preserve XML/xlink/epub/xmlns)
        def _sanitize_colon_attrs_in_content(text: str) -> str:
            def _process_tag(m):
                tag = m.group(0)
                if tag.startswith('</') or tag.startswith('<!') or tag.startswith('<?'):
                    return tag
                def _attr_repl(am):
                    before, name, eqval = am.group(1), am.group(2), am.group(3)
                    lname = name.lower()
                    if (
                        lname.startswith('xml:') or lname.startswith('xlink:') or lname.startswith('epub:') or
                        lname == 'xmlns' or lname.startswith('xmlns:')
                    ):
                        return am.group(0)
                    if ':' not in name:
                        return am.group(0)
                    safe = re.sub(r'[:]+', '-', name).strip('-')
                    safe = re.sub(r'[^A-Za-z0-9_.-]', '-', safe) or 'attr'
                    if not safe.startswith('data-'):
                        safe = 'data-' + safe
                    return f'{before}{safe}{eqval}'
                return re.sub(r'(\s)([A-Za-z_:][A-Za-z0-9_.:-]*:[A-Za-z0-9_.:-]*)(\s*=\s*(?:"[^"]*"|\'[^\']*\'))', _attr_repl, tag)
            return re.sub(r'<[^>]+>', _process_tag, text)

        content = _sanitize_colon_attrs_in_content(content)
            
        # Escape invalid story tags so they render literally with angle brackets in output
        allowed_ns_prefixes = {"svg", "math", "xlink", "xml", "xmlns", "epub"}
        def _escape_story_tag_entities(m):
            tagname = m.group(1)
            prefix = tagname.split(':', 1)[0].lower()
            if prefix in allowed_ns_prefixes:
                return m.group(0)
            tag_text = m.group(0)
            return tag_text.replace('<', '&lt;').replace('>', '&gt;')
        # Apply in order: self-closing, opening, closing
        content = re.sub(r'<([A-Za-z][\w.-]*:[\w.-]*)\s*([^>]*)/>', _escape_story_tag_entities, content)
        content = re.sub(r'<([A-Za-z][\w.-]*:[\w.-]*)\s*([^>]*)>', _escape_story_tag_entities, content)
        content = re.sub(r'</([A-Za-z][\w.-]*:[\w.-]*)\s*>', _escape_story_tag_entities, content)
            
        # Clean for XML
        content = XMLValidator.clean_for_xml(content)
        
        # Try to parse for validation
        try:
            ET.fromstring(content.encode('utf-8'))
        except ET.ParseError as e:
            log(f"[WARNING] XHTML validation failed: {e}")
            
            # DEBUG: Show what's at the error location
            import re
            match = re.search(r'line (\d+), column (\d+)', str(e))
            if match:
                line_num = int(match.group(1))
                col_num = int(match.group(2))
                
                lines = content.split('\n')
                log(f"[DEBUG] Error at line {line_num}, column {col_num}")
                log(f"[DEBUG] Total lines in content: {len(lines)}")
                
                if line_num <= len(lines):
                    problem_line = lines[line_num - 1]
                    log(f"[DEBUG] Full problem line: {problem_line!r}")
                    
                    # Show the problem area
                    if col_num <= len(problem_line):
                        # Show 40 characters before and after
                        start = max(0, col_num - 40)
                        end = min(len(problem_line), col_num + 40)
                        
                        log(f"[DEBUG] Context around error: {problem_line[start:end]!r}")
                        log(f"[DEBUG] Character at column {col_num}: {problem_line[col_num-1]!r} (U+{ord(problem_line[col_num-1]):04X})")
                        
                        # Show 5 characters before and after with hex
                        for i in range(max(0, col_num-5), min(len(problem_line), col_num+5)):
                            char = problem_line[i]
                            marker = " <-- ERROR" if i == col_num-1 else ""
                            log(f"[DEBUG] Col {i+1}: {char!r} (U+{ord(char):04X}){marker}")
                    else:
                        log(f"[DEBUG] Column {col_num} is beyond line length {len(problem_line)}")
                else:
                    log(f"[DEBUG] Line {line_num} doesn't exist (only {len(lines)} lines)")
                    # Show last few lines
                    for i in range(max(0, len(lines)-3), len(lines)):
                        log(f"[DEBUG] Line {i+1}: {lines[i][:100]!r}...")
            
            # Try to recover
            content = XHTMLConverter._attempt_recovery(content, e)
        
        return content
    
    @staticmethod
    def _attempt_recovery(content: str, error: ET.ParseError) -> str:
        """Attempt to recover from XML parse errors - ENHANCED"""
        try:
            # Use BeautifulSoup to fix structure
            soup = BeautifulSoup(content, 'lxml')
            
            # Ensure we have proper XHTML structure
            if not soup.find('html'):
                # Use default XHTML namespace and language attributes
                lang = getattr(XHTMLConverter, "DEFAULT_LANG", "en")
                new_soup = BeautifulSoup(f'<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="{lang}" lang="{lang}"></html>', 'lxml')
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
    
    def __init__(self, base_dir: str, log_callback: Optional[Callable] = None, stop_callback: Optional[Callable] = None):
        self.base_dir = os.path.abspath(base_dir)
        self.log_callback = log_callback
        self.stop_callback = stop_callback
        self.output_dir = self.base_dir
        self.images_dir = os.path.join(self.output_dir, "images")
        self.css_dir = os.path.join(self.output_dir, "css")
        self.fonts_dir = os.path.join(self.output_dir, "fonts")
        self.metadata_path = os.path.join(self.output_dir, "metadata.json")
        self.attach_css_to_chapters = os.getenv('ATTACH_CSS_TO_CHAPTERS', '0') == '1'  # Default to '0' (disabled)
        self.max_workers = int(os.environ.get("EXTRACTION_WORKERS", "4"))
        self.log(f"[INFO] Using {self.max_workers} workers for parallel processing")
        
        # Track auxiliary (non-chapter) HTML files to include in spine but omit from TOC
        self.auxiliary_html_files: set[str] = set()
        
        # SVG rasterization settings
        self.rasterize_svg = os.getenv('RASTERIZE_SVG_FALLBACK', '1') == '1'
        try:
            import cairosvg  # noqa: F401
            self._cairosvg_available = True
        except Exception:
            self._cairosvg_available = False
        
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
    
    def is_stopped(self) -> bool:
        """Check if stop has been requested"""
        # Check both the global flag and the callback
        if is_stop_requested():
            return True
        if self.stop_callback and self.stop_callback():
            return True
        return False
            
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
            
            # Check stop flag
            if self.is_stopped():
                self.log("🛑 EPUB converter stopped by user")
                return
            
            # Analyze chapters FIRST to get the structure
            chapter_titles_info = self._analyze_chapters()
            
            # Check stop flag after chapter analysis
            if self.is_stopped():
                self.log("🛑 EPUB converter stopped by user")
                return
            
            # Debug: Check if batch translation is enabled
            self.log(f"[DEBUG] Batch translation enabled: {getattr(self, 'batch_translate_headers', False)}")
            self.log(f"[DEBUG] Has header translator: {hasattr(self, 'header_translator')}")
            self.log(f"[DEBUG] EPUB_PATH env: {os.getenv('EPUB_PATH', 'NOT SET')}")
            self.log(f"[DEBUG] HTML dir: {self.html_dir}")
            
            # PRIORITY: Try standalone header translation first
            standalone_success = False
            if (hasattr(self, 'batch_translate_headers') and self.batch_translate_headers and 
                run_standalone_header_translation is not None):
                
                self.log("\n🔄 Attempting standalone header translation (content.opf based)...")
                try:
                    # Check if translated_headers.txt already exists
                    translations_file = os.path.join(self.output_dir, "translated_headers.txt")
                    
                    if os.path.exists(translations_file):
                        self.log("📁 Found existing translated_headers.txt - applying existing translations...")
                        self.log(f"  Translation file: {translations_file}")
                        self.log(f"  HTML directory: {self.html_dir}")
                        
                        try:
                            # Import the functions from standalone module
                            from translate_headers_standalone import (
                                load_translations_from_file, 
                                apply_existing_translations,
                                extract_source_chapters_with_opf_mapping,
                                match_output_to_source_chapters
                            )
                            self.log("✅ Successfully imported standalone module functions")
                            
                            # Load existing translations
                            self.log("🔍 Loading translations from file...")
                            chapters_info, translated_headers, _ = load_translations_from_file(translations_file, self.log)
                            
                            if translated_headers:
                                self.log(f"📋 Loaded {len(translated_headers)} existing translations:")
                                # Show first 3 translations for debugging
                                for num in list(translated_headers.keys())[:3]:
                                    self.log(f"    Chapter {num}: {translated_headers[num]}")
                                
                                # Get the source EPUB path
                                source_epub_path = os.getenv('EPUB_PATH')
                                self.log(f"  Source EPUB path: {source_epub_path}")
                                
                                if source_epub_path and os.path.exists(source_epub_path):
                                    self.log(f"🔄 Applying translations to HTML files in: {self.html_dir}")
                                    
                                    # Apply translations using the standalone module's function
                                    result = apply_existing_translations(
                                        epub_path=source_epub_path,
                                        output_dir=self.html_dir,
                                        translations_file=translations_file,
                                        update_html=True,  # Make sure to update the HTML files
                                        log_callback=self.log
                                    )
                                    
                                    if result:
                                        self.log(f"✅ Successfully applied translations to {len(result)} files:")
                                        # Show first 3 updated files
                                        for filename in list(result.keys())[:3]:
                                            self.log(f"    {filename}: {result[filename]}")
                                        
                                        # CRITICAL: Update chapter_titles_info so TOC uses translated titles now
                                        updated_for_toc = 0
                                        for chap_num, (orig_title, conf, fname) in list(chapter_titles_info.items()):
                                            base = os.path.basename(fname)
                                            if base in result:
                                                chapter_titles_info[chap_num] = (result[base], max(conf, 0.95), 'existing_translation')
                                                updated_for_toc += 1
                                        self.log(f"📝 Updated TOC titles from existing translations: {updated_for_toc} entries")
                                    else:
                                        self.log("⚠️ No files were updated with translations (result was empty)")
                                else:
                                    self.log(f"⚠️ Source EPUB not found or doesn't exist: {source_epub_path}")
                                    self.log("  Cannot apply translations without source EPUB for mapping")
                            else:
                                self.log("⚠️ No translations were loaded from file (empty result)")
                            
                            standalone_success = True
                        except ImportError as e:
                            self.log(f"⚠️ Failed to import standalone module: {e}")
                            self.log("  Make sure translate_headers_standalone.py is in the same directory")
                            import traceback
                            self.log(traceback.format_exc())
                            standalone_success = False
                        except Exception as e:
                            self.log(f"⚠️ Failed to apply existing translations: {e}")
                            self.log(f"  Exception type: {type(e).__name__}")
                            import traceback
                            self.log(traceback.format_exc())
                            standalone_success = False
                    else:
                        # Get the source EPUB path from environment
                        source_epub_path = os.getenv('EPUB_PATH')
                        
                        if source_epub_path and os.path.exists(source_epub_path):
                            self.log(f"📚 Source EPUB: {os.path.basename(source_epub_path)}")
                            self.log(f"📂 Output HTML dir: {self.html_dir}")
                            
                            # Run standalone header translation
                            result = run_standalone_header_translation(
                                source_epub_path=source_epub_path,
                                output_html_dir=self.html_dir,
                                log_callback=self.log
                            )
                            
                            if result:
                                self.log("✅ Standalone header translation completed successfully")
                                standalone_success = True
                                
                                # CRITICAL: Update chapter_titles_info so TOC uses translated titles
                                # result is a dict mapping filename -> translated_title
                                updated_for_toc = 0
                                for chap_num, (orig_title, conf, fname) in list(chapter_titles_info.items()):
                                    base = os.path.basename(fname)
                                    if base in result:
                                        chapter_titles_info[chap_num] = (result[base], max(conf, 0.95), 'standalone_translation')
                                        updated_for_toc += 1
                                self.log(f"📝 Updated TOC titles from standalone translation: {updated_for_toc} entries")
                            else:
                                self.log("⚠️ Standalone header translation returned no result")
                        else:
                            self.log(f"⚠️ Source EPUB not found: {source_epub_path}")
                            
                except Exception as e:
                    self.log(f"⚠️ Standalone header translation failed: {e}")
                    import traceback
                    self.log(traceback.format_exc())
            
            # FALLBACK: Extract source headers AND current titles if batch translation is enabled
            # Only run if standalone translation was not successful AND fallback is enabled
            source_headers = {}
            current_titles = {}
            use_fallback = os.getenv('USE_SORTED_FALLBACK', '0') == '1'
            
            if (not standalone_success and use_fallback and
                hasattr(self, 'batch_translate_headers') and self.batch_translate_headers and 
                hasattr(self, 'header_translator') and self.header_translator):
                
                self.log("\n🔄 Using fallback header translation method...")
                
                # Check if the extraction method exists
                if hasattr(self, '_extract_source_headers_and_current_titles'):
                    # Use the new extraction method
                    source_headers, current_titles = self._extract_source_headers_and_current_titles()
                    self.log(f"[DEBUG] Extraction complete: {len(source_headers)} source, {len(current_titles)} current")
                else:
                    self.log("⚠️ Missing _extract_source_headers_and_current_titles method!")
            
            # Batch translate headers if we have source headers (fallback only)
            translated_headers = {}
            if (not standalone_success and source_headers and 
                hasattr(self, 'header_translator') and self.header_translator):
                # Check if translated_headers.txt already exists
                translations_file = os.path.join(self.output_dir, "translated_headers.txt")
                
                if os.path.exists(translations_file):
                    # File exists - load and apply existing translations
                    self.log("📁 Found existing translated_headers.txt - applying existing translations...")
                    
                    try:
                        # Import the functions from standalone module
                        from translate_headers_standalone import load_translations_from_file
                        
                        # Load existing translations
                        _, translated_headers, _ = load_translations_from_file(translations_file, self.log)
                        
                        if translated_headers:
                            self.log(f"📋 Loaded {len(translated_headers)} existing translations")
                            
                            # Apply translations to HTML files using the same method
                            if hasattr(self, 'update_html_headers') and self.update_html_headers:
                                self.header_translator._update_html_headers_exact(
                                    self.html_dir, 
                                    translated_headers, 
                                    current_titles
                                )
                                self.log("✅ Applied existing translations to HTML files")
                                
                                # CRITICAL: Update chapter_titles_info so TOC gets translated titles on this run
                                updated_for_toc = 0
                                # Build filename -> translated mapping from current_titles' indices
                                file_to_trans = {}
                                for num, info in current_titles.items():
                                    if num in translated_headers:
                                        base = os.path.basename(info.get('filename', ''))
                                        if base:
                                            file_to_trans[base] = translated_headers[num]
                                for chap_num, (orig_title, conf, fname) in list(chapter_titles_info.items()):
                                    base = os.path.basename(fname)
                                    if base in file_to_trans:
                                        chapter_titles_info[chap_num] = (file_to_trans[base], max(conf, 0.95), 'existing_translation')
                                        updated_for_toc += 1
                                self.log(f"📝 Updated TOC titles from existing translations (fallback path): {updated_for_toc} entries")
                            
                            # Update toc.ncx if it exists
                            toc_path = os.path.join(self.output_dir, 'toc.ncx')
                            if os.path.exists(toc_path):
                                from translate_headers_standalone import update_toc_ncx
                                update_toc_ncx(toc_path, translated_headers, current_titles, self.log)
                        else:
                            self.log("⚠️ No translations found in existing file")
                    
                    except Exception as e:
                        self.log(f"⚠️ Failed to apply existing translations: {e}")
                        import traceback
                        self.log(traceback.format_exc())
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
            elif not standalone_success and not use_fallback and hasattr(self, 'batch_translate_headers') and self.batch_translate_headers:
                # Standalone failed but fallback is disabled (only warn if batch translation is enabled)
                self.log("⚠️ Standalone header translation failed and sorted fallback is disabled")
                self.log("   Enable 'Use Sorted Fallback' in Other Settings if needed")
            else:
                if not standalone_success and not source_headers:
                    self.log("⚠️ No source headers found, skipping batch translation")
                elif not hasattr(self, 'header_translator'):
                    self.log("⚠️ No header translator available")

            # Find HTML files
            html_files = self._find_html_files()
            if not html_files:
                raise Exception("No translated chapters found to compile into EPUB")
            
            # Check stop flag
            if self.is_stopped():
                self.log("🛑 EPUB converter stopped by user")
                return
            
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
                        
                        # Preserve original values and mark fields as translated so we don't
                        # keep re-translating metadata on every EPUB rebuild.
                        for field, should_translate in self.translate_metadata_fields.items():
                            if not should_translate:
                                continue
                            if field in metadata and field in translated_metadata:
                                if metadata[field] != translated_metadata[field]:
                                    # Store original value (if not already present)
                                    original_key = f'original_{field}'
                                    if original_key not in translated_metadata:
                                        translated_metadata[original_key] = metadata[field]
                                    # Mark as translated so future runs can detect it
                                    translated_metadata[f'{field}_translated'] = True
                        
                        metadata = translated_metadata
                    except Exception as e:
                        self.log(f"⚠️ Metadata translation failed: {e}")
                        # Continue with original metadata

            # Decide language for EPUB based primarily on GUI's OUTPUT_LANGUAGE
            try:
                detected_lang = self._detect_primary_language_from_html(html_files)
                if detected_lang:
                    current_lang = metadata.get("language")
                    if not current_lang or current_lang.lower() != detected_lang.lower():
                        self.log(f"[INFO] Setting metadata language from OUTPUT_LANGUAGE: {current_lang!r} -> '{detected_lang}'")
                        metadata["language"] = detected_lang
            except Exception as e:
                self.log(f"[WARNING] Failed to determine EPUB language from OUTPUT_LANGUAGE: {e}")
            
            # Check stop flag
            if self.is_stopped():
                self.log("🛑 EPUB converter stopped by user")
                return
            
            # Create EPUB book
            book = self._create_book(metadata)
            
            # Process all components
            spine = []
            toc = []
            
            # Add CSS
            css_items = self._add_css_files(book)
            
            # Check stop flag
            if self.is_stopped():
                self.log("🛑 EPUB converter stopped by user")
                return
            
            # Add fonts
            self._add_fonts(book)
            
            # Check stop flag
            if self.is_stopped():
                self.log("🛑 EPUB converter stopped by user")
                return
            
            # Process images and cover
            processed_images, cover_file = self._process_images()
            
            # Compress images if enabled (before adding to EPUB)
            if os.environ.get('ENABLE_IMAGE_COMPRESSION', '0') == '1':
                processed_images, cover_file = self._compress_images(processed_images, cover_file)
            
            # Check stop flag
            if self.is_stopped():
                self.log("🛑 EPUB converter stopped by user")
                return
            
            # Add images to book
            self._add_images_to_book(book, processed_images, cover_file)
            
            # Check stop flag
            if self.is_stopped():
                self.log("🛑 EPUB converter stopped by user")
                return
            
            # Add cover page if exists
            if cover_file:
                cover_page = self._create_cover_page(book, cover_file, processed_images, css_items, metadata)
                if cover_page:
                    spine.insert(0, cover_page)
            
            # Check stop flag
            if self.is_stopped():
                self.log("🛑 EPUB converter stopped by user")
                return
            
            # Process chapters with updated titles
            chapters_added = self._process_chapters(
                book, html_files, chapter_titles_info, 
                css_items, processed_images, spine, toc, metadata
            )
            
            if chapters_added == 0:
                raise Exception("No chapters could be added to the EPUB")
            
            # Check stop flag
            if self.is_stopped():
                self.log("🛑 EPUB converter stopped by user")
                return
            
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
            
            # Check stop flag
            if self.is_stopped():
                self.log("🛑 EPUB converter stopped by user")
                return
            
            # Finalize book
            self._finalize_book(book, spine, toc, cover_file)
            
            # Check stop flag
            if self.is_stopped():
                self.log("🛑 EPUB converter stopped by user")
                return
            
            # Write EPUB
            self._write_epub(book, metadata)
            
            # Generate PDF if enabled
            if os.environ.get('ENABLE_PDF_OUTPUT', '0') == '1':
                if self.is_stopped():
                    self.log("🛑 PDF generation skipped - stop requested")
                else:
                    try:
                        self._generate_pdf(html_files, chapter_titles_info, processed_images, cover_file, metadata)
                    except Exception as e:
                        self.log(f"⚠️ PDF generation failed: {e}")
                        import traceback
                        self.log(f"[DEBUG] {traceback.format_exc()}")
            
            # Persist updated metadata (including translated fields/language)
            try:
                self._save_metadata(metadata)
            except Exception as e:
                self.log(f"[WARNING] Failed to save updated metadata.json: {e}")
            
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
        
        html_exts = ('.html', '.xhtml', '.htm')
        html_files = [f for f in os.listdir(self.base_dir) if f.lower().endswith(html_exts)]
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
        
        html_exts = ('.html', '.xhtml', '.htm')
        html_files = [f for f in os.listdir(self.base_dir) if f.lower().endswith(html_exts)]
        
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

    def _detect_primary_language_from_html(self, html_files: List[str]) -> Optional[str]:
        """Determine language for EPUB/output based on GUI target language.

        Priority:
          1) OUTPUT_LANGUAGE env var set by TranslatorGUI ("English", "Spanish", ...)
          2) metadata["language"] if already present (2-letter/ISO-ish code)
          3) Fallback: "en".

        Returns a BCP‑47 style language code like "en", "es", "ru", "tr", "ko", ...
        """
        # 1) First, try OUTPUT_LANGUAGE name from GUI (English, Spanish, ...)
        output_lang_name = os.getenv('OUTPUT_LANGUAGE', '').strip().lower()

        gui_name_to_code = {
            # Matches translator_gui.py target_lang_combo entries
            'english': 'en',
            'spanish': 'es',
            'french': 'fr',
            'german': 'de',
            'italian': 'it',
            'portuguese': 'pt',
            'russian': 'ru',
            'arabic': 'ar',
            'hindi': 'hi',
            'chinese (simplified)': 'zh-CN',
            'chinese (traditional)': 'zh-TW',
            'chinese': 'zh-CN',
            'japanese': 'ja',
            'korean': 'ko',
            'turkish': 'tr',
        }

        if output_lang_name:
            code = gui_name_to_code.get(output_lang_name)
            if code:
                self.log(f"[DEBUG] Using OUTPUT_LANGUAGE from GUI for EPUB language: '{output_lang_name}' -> '{code}'")
                return code
            else:
                # If user typed a raw code like "es" or "pt-BR", just pass it through
                self.log(f"[DEBUG] OUTPUT_LANGUAGE not in predefined map, using raw value: '{output_lang_name}'")
                return output_lang_name

        # 2) Nothing from GUI – leave decision to caller by returning None
        return None

    def _analyze_chapters(self) -> Dict[int, Tuple[str, float, str]]:
        """Analyze chapter files and extract titles using parallel processing"""
        self.log("\n📖 Extracting translated titles from chapter files...")
        
        chapter_info = {}
        sorted_files = self._find_html_files()
        
        if not sorted_files:
            self.log("⚠️ No translated chapter files found!")
            return chapter_info
        
        self.log(f"📖 Analyzing {len(sorted_files)} translated chapter files for titles...")
        self.log(f"🔧 Using {self.max_workers} parallel workers")
        
        def analyze_single_file(idx_filename):
            """Worker function to analyze a single file"""
            idx, filename = idx_filename
            file_path = os.path.join(self.output_dir, filename)
            
            try:
                # Read and process file
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw_html_content = f.read()
                
                # Decode HTML entities
                import html
                html_content = html.unescape(raw_html_content)
                html_content = self._fix_encoding_issues(html_content)
                html_content = HTMLEntityDecoder.decode(html_content)
                
                # Extract title
                title, confidence = TitleExtractor.extract_from_html(
                    html_content, idx, filename
                )
                
                return idx, (title, confidence, filename)
                
            except Exception as e:
                return idx, (f"Chapter {idx}", 0.0, filename), str(e)
        
        # Process files in parallel using environment variable worker count
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(analyze_single_file, (idx, filename)): idx 
                for idx, filename in enumerate(sorted_files)
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(futures):
                # Check stop flag
                if self.is_stopped():
                    self.log("🛑 Chapter analysis stopped by user")
                    break
                
                try:
                    result = future.result()
                    completed += 1
                    
                    if len(result) == 2:  # Success
                        idx, info = result
                        chapter_info[idx] = info
                        
                        # Log progress - only show issues (low confidence) unless debug mode is on
                        title, confidence, filename = info
                        debug_mode_enabled = os.environ.get('DEBUG_MODE', '0') == '1'
                        
                        # Always log low confidence (issues) or errors
                        if confidence <= 0.4:
                            indicator = "🔴"
                            self.log(f"  [{completed}/{len(sorted_files)}] {indicator} Chapter {idx}: '{title}' (confidence: {confidence:.2f})")
                        elif debug_mode_enabled:
                            # In debug mode, log all chapters
                            indicator = "✅" if confidence > 0.7 else "🟡"
                            self.log(f"  [{completed}/{len(sorted_files)}] {indicator} Chapter {idx}: '{title}' (confidence: {confidence:.2f})")
                    else:  # Error
                        idx, info, error = result
                        chapter_info[idx] = info
                        # Always log errors
                        self.log(f"❌ [{completed}/{len(sorted_files)}] Error processing chapter {idx}: {error}")
                        
                except Exception as e:
                    idx = futures[future]
                    self.log(f"❌ Failed to process chapter {idx}: {e}")
                    chapter_info[idx] = (f"Chapter {idx}", 0.0, sorted_files[idx])
        
        return chapter_info
    
    def _process_chapters(self, book: epub.EpubBook, html_files: List[str],
                         chapter_titles_info: Dict[int, Tuple[str, float, str]],
                         css_items: List[epub.EpubItem], processed_images: Dict[str, str],
                         spine: List, toc: List, metadata: dict) -> int:
        """Process chapters using parallel processing with AGGRESSIVE DEBUGGING"""
        chapters_added = 0
        self.log(f"\n{'='*80}")
        self.log(f"📚 STARTING CHAPTER PROCESSING")
        self.log(f"📚 Total files to process: {len(html_files)}")
        self.log(f"🔧 Using {self.max_workers} parallel workers")
        self.log(f"📂 Output directory: {self.output_dir}")
        self.log(f"{'='*80}")
        
        # Debug chapter titles info
        self.log(f"\n[DEBUG] Chapter titles info has {len(chapter_titles_info)} entries")
        for num in list(chapter_titles_info.keys())[:5]:
            title, conf, method = chapter_titles_info[num]
            self.log(f"  Chapter {num}: {title[:50]}... (conf: {conf}, method: {method})")
        
        # Prepare chapter data
        chapter_data = []
        for idx, filename in enumerate(html_files):
            chapter_num = idx
            if chapter_num not in chapter_titles_info and (chapter_num + 1) in chapter_titles_info:
                chapter_num = idx + 1
            chapter_data.append((chapter_num, filename))
            
            # Debug specific problem chapters
            if 49 <= chapter_num <= 56:
                self.log(f"[DEBUG] Problem chapter found: {chapter_num} -> {filename}")
        
        def process_chapter_content(data):
            """Worker function to process chapter content with FULL DEBUGGING"""
            chapter_num, filename = data
            path = os.path.join(self.output_dir, filename)
            
            # Debug tracking for problem chapters
            is_problem_chapter = 49 <= chapter_num <= 56
            
            try:
                if is_problem_chapter:
                    self.log(f"\n[DEBUG] {'*'*60}")
                    self.log(f"[DEBUG] PROCESSING PROBLEM CHAPTER {chapter_num}: {filename}")
                    self.log(f"[DEBUG] Full path: {path}")
                
                # Check file exists
                if not os.path.exists(path):
                    error_msg = f"File does not exist: {path}"
                    self.log(f"[ERROR] {error_msg}")
                    raise FileNotFoundError(error_msg)
                
                # Get file size
                file_size = os.path.getsize(path)
                if is_problem_chapter:
                    self.log(f"[DEBUG] File size: {file_size} bytes")
                
                # Read and decode
                raw_content = self._read_and_decode_html_file(path)
                if is_problem_chapter:
                    self.log(f"[DEBUG] Raw content length after reading: {len(raw_content) if raw_content else 'NULL'}")
                    if raw_content:
                        self.log(f"[DEBUG] First 200 chars: {raw_content[:200]}")
                
                # Fix encoding
                raw_content = self._fix_encoding_issues(raw_content)
                if is_problem_chapter:
                    self.log(f"[DEBUG] Content length after encoding fix: {len(raw_content) if raw_content else 'NULL'}")
                
                if not raw_content or not raw_content.strip():
                    error_msg = f"Empty content after reading/decoding: {filename}"
                    if is_problem_chapter:
                        self.log(f"[ERROR] {error_msg}")
                    raise ValueError(error_msg)
                
                # Extract main content
                if not filename.startswith('response_'):
                    before_len = len(raw_content)
                    raw_content = self._extract_main_content(raw_content, filename)
                    if is_problem_chapter:
                        self.log(f"[DEBUG] Content extraction: {before_len} -> {len(raw_content)} chars")
                
                # Get title
                title = self._get_chapter_title(chapter_num, filename, raw_content, chapter_titles_info)
                if is_problem_chapter:
                    self.log(f"[DEBUG] Chapter title: {title}")
                
                # Prepare CSS links
                css_links = [f"css/{item.file_name.split('/')[-1]}" for item in css_items]
                if is_problem_chapter:
                    self.log(f"[DEBUG] CSS links: {css_links}")
                
                # XHTML conversion - THE CRITICAL PART
                if is_problem_chapter:
                    self.log(f"[DEBUG] Starting XHTML conversion...")
                
                xhtml_content = XHTMLConverter.ensure_compliance(raw_content, title, css_links)
                
                if is_problem_chapter:
                    self.log(f"[DEBUG] XHTML content length: {len(xhtml_content) if xhtml_content else 'NULL'}")
                    if xhtml_content:
                        self.log(f"[DEBUG] XHTML first 300 chars: {xhtml_content[:300]}")
                
                # Process images
                xhtml_content = self._process_chapter_images(xhtml_content, processed_images)
                
                # Validate
                if is_problem_chapter:
                    self.log(f"[DEBUG] Starting validation...")
                
                final_content = XHTMLConverter.validate(xhtml_content)
                
                if is_problem_chapter:
                    self.log(f"[DEBUG] Final content length: {len(final_content)}")
                
                # Final XML validation
                try:
                    ET.fromstring(final_content.encode('utf-8'))
                    if is_problem_chapter:
                        self.log(f"[DEBUG] XML validation PASSED")
                except ET.ParseError as e:
                    if is_problem_chapter:
                        self.log(f"[ERROR] XML validation FAILED: {e}")
                        # Show the exact error location
                        lines = final_content.split('\n')
                        import re
                        match = re.search(r'line (\d+), column (\d+)', str(e))
                        if match:
                            line_num = int(match.group(1))
                            if line_num <= len(lines):
                                self.log(f"[ERROR] Problem line {line_num}: {lines[line_num-1][:100]}")
                    
                    # Create fallback
                    final_content = XHTMLConverter._build_fallback_xhtml(title)
                    if is_problem_chapter:
                        self.log(f"[DEBUG] Using fallback XHTML")
                
                if is_problem_chapter:
                    self.log(f"[DEBUG] Chapter processing SUCCESSFUL")
                    self.log(f"[DEBUG] {'*'*60}\n")
                
                return {
                    'num': chapter_num,
                    'filename': filename,
                    'title': title,
                    'content': final_content,
                    'success': True
                }
                
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                
                if is_problem_chapter:
                    self.log(f"[ERROR] {'!'*60}")
                    self.log(f"[ERROR] CHAPTER {chapter_num} PROCESSING FAILED")
                    self.log(f"[ERROR] Exception type: {type(e).__name__}")
                    self.log(f"[ERROR] Exception: {e}")
                    self.log(f"[ERROR] Full traceback:\n{tb}")
                    self.log(f"[ERROR] {'!'*60}\n")
                
                return {
                    'num': chapter_num,
                    'filename': filename,
                    'title': chapter_titles_info.get(chapter_num, (f"Chapter {chapter_num}", 0, ""))[0],
                    'error': str(e),
                    'traceback': tb,
                    'success': False
                }
        
        # Process in parallel
        processed_chapters = []
        completed = 0
        total_chapters = len(chapter_data)
        
        # Use reduced logging for large EPUBs
        use_reduced_logging = total_chapters > 50
        log_interval = max(1, total_chapters // 20) if use_reduced_logging else 1
        
        self.log(f"\n[DEBUG] Starting parallel processing...")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(process_chapter_content, data): data[0] 
                for data in chapter_data
            }
            
            for future in as_completed(futures):
                # Check stop flag
                if self.is_stopped():
                    self.log("🛑 Chapter processing stopped by user")
                    break
                
                try:
                    result = future.result()
                    if result:
                        processed_chapters.append(result)
                        completed += 1
                        
                        # Always log failures
                        if not result['success']:
                            self.log(f"  [{completed}/{total_chapters}] ❌ Failed: {result['filename']} - {result['error']}")
                        # For successes: log at intervals for large EPUBs, or always for small ones
                        elif not use_reduced_logging or completed == 1 or completed % log_interval == 0 or completed == total_chapters:
                            current_percent = (completed * 100) // total_chapters
                            self.log(f"  [{completed}/{total_chapters}] ({current_percent}%) ✅")
                            
                except Exception as e:
                    completed += 1
                    chapter_num = futures[future]
                    self.log(f"  [{completed}/{total_chapters}] ❌ Exception processing chapter {chapter_num}: {e}")
                    import traceback
                    self.log(f"[ERROR] Traceback:\n{traceback.format_exc()}")
        
        # Sort by chapter number to maintain order
        processed_chapters.sort(key=lambda x: x['num'])
        
        # Debug what we have
        self.log(f"\n[DEBUG] Processed {len(processed_chapters)} chapters")
        failed_chapters = [c for c in processed_chapters if not c['success']]
        if failed_chapters:
            self.log(f"[WARNING] {len(failed_chapters)} chapters failed:")
            for fc in failed_chapters:
                self.log(f"  - Chapter {fc['num']}: {fc['filename']} - {fc.get('error', 'Unknown error')}")
        
        # Add chapters to book in order (this must be sequential)
        self.log("\n📦 Adding chapters to EPUB structure...")
        
        # Use reduced logging for large EPUBs
        total_to_add = len(processed_chapters)
        use_reduced_logging = total_to_add > 50
        log_interval = max(1, total_to_add // 20) if use_reduced_logging else 1
        
        for idx, chapter_data in enumerate(processed_chapters, 1):
            if chapter_data['success']:
                try:
                    # Create EPUB chapter
                    import html
                    chapter = epub.EpubHtml(
                        title=html.unescape(chapter_data['title']),
                        file_name=os.path.basename(chapter_data['filename']),
                        lang=metadata.get("language", "en")
                    )
                    chapter.content = FileUtils.ensure_bytes(chapter_data['content'])
                    
                    if self.attach_css_to_chapters:
                        for css_item in css_items:
                            chapter.add_item(css_item)
                    
                    # Add to book
                    book.add_item(chapter)
                    spine.append(chapter)

                    # Include auxiliary files in spine but omit from TOC
                    base_name = os.path.basename(chapter_data['filename'])
                    if hasattr(self, 'auxiliary_html_files') and base_name in self.auxiliary_html_files:
                        self.log(f"  🛈 Added auxiliary page to spine (not in TOC): {base_name}")
                    else:
                        toc.append(chapter)
                    chapters_added += 1
                    
                    # Log auxiliary files always, or at intervals for regular chapters
                    if base_name in getattr(self, 'auxiliary_html_files', set()):
                        self.log(f"  ✅ Added auxiliary page (spine only): '{base_name}'")
                    elif not use_reduced_logging or idx == 1 or idx % log_interval == 0 or idx == total_to_add:
                        current_percent = (idx * 100) // total_to_add
                        self.log(f"  [{idx}/{total_to_add}] ({current_percent}%) ✅")
                    
                except Exception as e:
                    self.log(f"  ❌ Failed to add chapter {chapter_data['num']} to book: {e}")
                    import traceback
                    self.log(f"[ERROR] Traceback:\n{traceback.format_exc()}")
                    # Add error placeholder
                    self._add_error_chapter_from_data(book, chapter_data, spine, toc, metadata)
                    chapters_added += 1
            else:
                self.log(f"  ⚠️ Adding error placeholder for chapter {chapter_data['num']}")
                # Add error placeholder
                self._add_error_chapter_from_data(book, chapter_data, spine, toc, metadata)
                chapters_added += 1
        
        self.log(f"\n{'='*80}")
        self.log(f"✅ CHAPTER PROCESSING COMPLETE")
        self.log(f"✅ Added {chapters_added} chapters to EPUB")
        self.log(f"{'='*80}\n")
        
        return chapters_added
    
    def _add_error_chapter_from_data(self, book, chapter_data, spine, toc, metadata):
        """Helper to add an error placeholder chapter"""
        try:
            title = chapter_data.get('title', f"Chapter {chapter_data['num']}")
            chapter = epub.EpubHtml(
                title=title,
                file_name=f"chapter_{chapter_data['num']:03d}.xhtml",
                lang=metadata.get("language", "en")
            )
            
            lang = metadata.get("language", "en")
            error_content = f"""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="{lang}" lang="{lang}">
<head><title>{ContentProcessor.safe_escape(title)}</title></head>
<body>
<h1>{ContentProcessor.safe_escape(title)}</h1>
<p>Error loading chapter content.</p>
<p>File: {chapter_data.get('filename', 'unknown')}</p>
<p>Error: {chapter_data.get('error', 'unknown error')}</p>
</body>
</html>"""
            
            chapter.content = error_content.encode('utf-8')
            book.add_item(chapter)
            spine.append(chapter)
            toc.append(chapter)
            
        except Exception as e:
            self.log(f"  ❌ Failed to add error placeholder: {e}")


    def _get_chapter_order_from_opf(self) -> Dict[str, int]:
        """Get chapter order from content.opf or source EPUB
        Returns dict mapping original_filename -> chapter_number
        """
        # First, try to find content.opf in the current directory
        opf_path = os.path.join(self.output_dir, "content.opf")
        
        if os.path.exists(opf_path):
            self.log("✅ Found content.opf - using for chapter ordering")
            return self._parse_opf_file(opf_path)
        
        # If not found, try to extract from source EPUB
        source_epub = os.getenv('EPUB_PATH')
        if source_epub and os.path.exists(source_epub):
            self.log(f"📚 Extracting chapter order from source EPUB: {source_epub}")
            return self._extract_order_from_epub(source_epub)
        
        # Fallback to translation_progress.json if available
        progress_file = os.path.join(self.output_dir, "translation_progress.json")
        if os.path.exists(progress_file):
            self.log("📄 Using translation_progress.json for chapter order")
            return self._get_order_from_progress_file(progress_file)
        
        return None

    def _parse_opf_file(self, opf_path: str) -> Dict[str, int]:
        """Parse content.opf to get chapter order from spine
        Returns dict mapping original_filename -> chapter_number
        """
        try:
            tree = ET.parse(opf_path)
            root = tree.getroot()
            
            # Handle namespaces
            ns = {'opf': 'http://www.idpf.org/2007/opf'}
            if root.tag.startswith('{'):
                # Extract default namespace
                default_ns = root.tag[1:root.tag.index('}')]
                ns = {'opf': default_ns}
            
            # Get manifest to map IDs to files
            manifest = {}
            for item in root.findall('.//opf:manifest/opf:item', ns):
                item_id = item.get('id')
                href = item.get('href')
                media_type = item.get('media-type', '')
                
                # Only include HTML/XHTML files
                if item_id and href and ('html' in media_type.lower() or href.endswith(('.html', '.xhtml', '.htm'))):
                    # Get just the filename without path
                    filename = os.path.basename(href)
                    manifest[item_id] = filename
            
            # Get spine order
            filename_to_order = {}
            chapter_num = 0  # Start from 0 for array indexing
            
            spine = root.find('.//opf:spine', ns)
            if spine is not None:
                # Build dynamic skip list based on TRANSLATE_SPECIAL_FILES toggle
                translate_special = os.environ.get('TRANSLATE_SPECIAL_FILES', '0') == '1'
                # Backward compatibility: also check old TRANSLATE_COVER_HTML
                translate_special = translate_special or (os.environ.get('TRANSLATE_COVER_HTML', '0') == '1')
                
                if translate_special:
                    # When override is enabled, include ALL files in chapter ordering
                    skip_list = []
                    self.log("  📝 Special files mode ENABLED - including all files in TOC")
                else:
                    # Default behavior: skip navigation/metadata files
                    skip_list = ['nav', 'toc', 'contents', 'cover']
                    self.log("  📝 Special files mode DISABLED - excluding navigation files")
                
                # Count total items first to decide on logging
                itemrefs = spine.findall('opf:itemref', ns)
                total_items = len(itemrefs)
                use_reduced_logging = total_items > 50
                log_interval = max(1, total_items // 20) if use_reduced_logging else 1
                
                for idx, itemref in enumerate(itemrefs):
                    idref = itemref.get('idref')
                    if idref and idref in manifest:
                        filename = manifest[idref]
                        
                        # CRITICAL: Files with numbers are always regular chapters, regardless of keywords!
                        name_without_ext = os.path.splitext(filename)[0].lower()
                        has_numbers = bool(re.search(r'\\d', name_without_ext))
                        
                        # If file has numbers, it's a chapter - include it
                        if has_numbers:
                            filename_to_order[filename] = chapter_num
                            # Only log periodically for large EPUBs
                            if not use_reduced_logging or idx % log_interval == 0 or idx == 0 or idx == total_items - 1:
                                if use_reduced_logging:
                                    percent = (idx * 100) // total_items
                                    self.log(f"  [{idx}/{total_items}] ({percent}%) ✅")
                                else:
                                    self.log(f"  Chapter {chapter_num}: {filename} (numbered)")
                            chapter_num += 1
                        # Otherwise, check skip list for special files
                        elif not skip_list or not any(skip in filename.lower() for skip in skip_list):
                            filename_to_order[filename] = chapter_num
                            # Only log periodically for large EPUBs
                            if not use_reduced_logging or idx % log_interval == 0 or idx == 0 or idx == total_items - 1:
                                if use_reduced_logging:
                                    percent = (idx * 100) // total_items
                                    self.log(f"  [{idx}/{total_items}] ({percent}%) ✅")
                                else:
                                    self.log(f"  Chapter {chapter_num}: {filename}")
                            chapter_num += 1
                        else:
                            # Always log skipped files (these are rare)
                            self.log(f"  Skipping special file (no numbers): {filename}")
            
            return filename_to_order
            
        except Exception as e:
            self.log(f"⚠️ Error parsing content.opf: {e}")
            import traceback
            self.log(traceback.format_exc())
            return None

    def _extract_order_from_epub(self, epub_path: str) -> List[Tuple[int, str]]:
        """Extract chapter order from source EPUB file"""
        try:
            import zipfile
            
            with zipfile.ZipFile(epub_path, 'r') as zf:
                # Find content.opf (might be in different locations)
                opf_file = None
                for name in zf.namelist():
                    if name.endswith('content.opf'):
                        opf_file = name
                        break
                
                if not opf_file:
                    # Try META-INF/container.xml to find content.opf
                    try:
                        container = zf.read('META-INF/container.xml')
                        # Parse container.xml to find content.opf location
                        container_tree = ET.fromstring(container)
                        rootfile = container_tree.find('.//{urn:oasis:names:tc:opendocument:xmlns:container}rootfile')
                        if rootfile is not None:
                            opf_file = rootfile.get('full-path')
                    except:
                        pass
                
                if opf_file:
                    opf_content = zf.read(opf_file)
                    # Save temporarily and parse
                    temp_opf = os.path.join(self.output_dir, "temp_content.opf")
                    with open(temp_opf, 'wb') as f:
                        f.write(opf_content)
                    
                    result = self._parse_opf_file(temp_opf)
                    
                    # Clean up temp file
                    if os.path.exists(temp_opf):
                        os.remove(temp_opf)
                        
                    return result
                    
        except Exception as e:
            self.log(f"⚠️ Error extracting from EPUB: {e}")
            return None

    def _find_html_files(self) -> List[str]:
        """Find HTML files using OPF-based ordering when available"""
        self.log(f"\n[DEBUG] Scanning directory: {self.output_dir}")
        
        # Get all HTML files in directory
        all_files = os.listdir(self.output_dir)
        html_extensions = ('.html', '.htm', '.xhtml')
        html_files = [f for f in all_files if f.lower().endswith(html_extensions)]
        
        if not html_files:
            self.log("[ERROR] No HTML files found!")
            return []
        
        # Try to get authoritative order from OPF/EPUB
        opf_order = self._get_chapter_order_from_opf()
        
        if opf_order:
            self.log("✅ Using authoritative chapter order from OPF/EPUB")
            
            # Check if debug mode is enabled
            debug_mode_enabled = os.environ.get('DEBUG_MODE', '0') == '1'
            if debug_mode_enabled:
                self.log(f"[DEBUG] OPF entries (first 5): {list(opf_order.items())[:5]}")
            
            # Create mapping based on core filename (strip response_ and strip ALL extensions)
            ordered_files = []
            unmapped_files = []
            
            def strip_all_ext(name: str) -> str:
                # Remove all trailing known extensions
                core = name
                while True:
                    parts = core.rsplit('.', 1)
                    if len(parts) == 2 and parts[1].lower() in ['html', 'htm', 'xhtml', 'xml']:
                        core = parts[0]
                    else:
                        break
                return core
            
            for output_file in html_files:
                core_name = output_file[9:] if output_file.startswith('response_') else output_file
                core_name = strip_all_ext(core_name)
                
                matched = False
                for opf_name, chapter_order in opf_order.items():
                    opf_file = opf_name.split('/')[-1]
                    opf_core = strip_all_ext(opf_file)
                    if core_name == opf_core:
                        ordered_files.append((chapter_order, output_file))
                        if debug_mode_enabled:
                            self.log(f"  Mapped: {output_file} -> {opf_name} (order: {chapter_order})")
                        matched = True
                        break
                if not matched:
                    unmapped_files.append(output_file)
                    # Always log unmapped files (these are warnings)
                    self.log(f"  ⚠️ Could not map: {output_file} (core: {core_name})")
            
            if ordered_files:
                # Sort by chapter order and extract just the filenames
                ordered_files.sort(key=lambda x: x[0])
                final_order = [f for _, f in ordered_files]
                
                # Append any unmapped files at the end
                if unmapped_files:
                    self.log(f"⚠️ Adding {len(unmapped_files)} unmapped files at the end")
                    final_order.extend(sorted(unmapped_files))
                    # Mark non-response unmapped files as auxiliary (omit from TOC)
                    aux = {f for f in unmapped_files if not f.startswith('response_')}
                    # If special files override is enabled, do NOT treat special files as auxiliary
                    translate_special = os.environ.get('TRANSLATE_SPECIAL_FILES', '0') == '1'
                    # Backward compatibility
                    translate_special = translate_special or (os.environ.get('TRANSLATE_COVER_HTML', '0') == '1')
                    if translate_special:
                        # Don't exclude any special files when override is enabled
                        aux = set()
                    self.auxiliary_html_files = aux
                else:
                    self.auxiliary_html_files = set()
                
                self.log(f"✅ Successfully ordered {len(final_order)} chapters using OPF")
                return final_order
            else:
                self.log("⚠️ Could not map any files using OPF order, falling back to pattern matching")
        
        # Fallback to original pattern matching logic
        self.log("⚠️ No OPF/EPUB found or mapping failed, using filename pattern matching")
        
        # First, try to find response_ files
        response_files = [f for f in html_files if f.startswith('response_')]
        
        if response_files:
            # Sort response_ files as primary chapters
            main_files = list(response_files)
            self.log(f"[DEBUG] Found {len(response_files)} response_ files")
            
            # Check if files have -h- pattern
            if any('-h-' in f for f in response_files):
                # Use special sorting for -h- pattern
                def extract_h_number(filename):
                    match = re.search(r'-h-(\d+)', filename)
                    if match:
                        return int(match.group(1))
                    return 999999
                
                main_files.sort(key=extract_h_number)
            else:
                # Use numeric sorting for standard response_ files
                def extract_number(filename):
                    match = re.match(r'response_(\d+)_', filename)
                    if match:
                        return int(match.group(1))
                    return 0
                
                main_files.sort(key=extract_number)
            
            # Append non-response files as auxiliary pages (not in TOC)
            aux_files = sorted([f for f in html_files if not f.startswith('response_')])
            if aux_files:
                aux_set = set(aux_files)
                # If special files override is enabled, don't mark special files as auxiliary
                translate_special = os.environ.get('TRANSLATE_SPECIAL_FILES', '0') == '1'
                # Backward compatibility
                translate_special = translate_special or (os.environ.get('TRANSLATE_COVER_HTML', '0') == '1')
                if translate_special:
                    # Don't exclude any files when override is enabled
                    aux_set = set()
                self.auxiliary_html_files = aux_set
                self.log(f"[DEBUG] Appending {len(aux_set)} auxiliary HTML file(s) (not in TOC): {list(aux_set)[:5]}")
            else:
                self.auxiliary_html_files = set()
            
            return main_files + aux_files
        else:
            # Progressive sorting for non-standard files
            html_files.sort(key=self.get_robust_sort_key)
            # No response_ files -> treat none as auxiliary
            self.auxiliary_html_files = set()
        
        return html_files

    def _read_and_decode_html_file(self, file_path: str) -> str:
        """Read HTML file and decode entities, preserving &lt; and &gt; as text.
        This prevents narrative angle-bracket text from becoming bogus tags."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if not content:
            return content
        
        import re
        import html
        
        # Placeholders for angle bracket entities
        LT_PLACEHOLDER = "\ue000"
        GT_PLACEHOLDER = "\ue001"
        
        # Patterns for common representations of < and >
        _lt_entity_patterns = [r'&lt;', r'&LT;', r'&#0*60;', r'&#x0*3[cC];']
        _gt_entity_patterns = [r'&gt;', r'&GT;', r'&#0*62;', r'&#x0*3[eE];']
        
        def protect_angle_entities(s: str) -> str:
            # Replace all forms of &lt; and &gt; with placeholders so unescape won't turn them into real < >
            for pat in _lt_entity_patterns:
                s = re.sub(pat, LT_PLACEHOLDER, s)
            for pat in _gt_entity_patterns:
                s = re.sub(pat, GT_PLACEHOLDER, s)
            return s
        
        max_iterations = 5
        for _ in range(max_iterations):
            prev_content = content
            # Protect before each pass in case of double-encoded entities
            content = protect_angle_entities(content)
            # html.unescape handles all standard HTML entities (except our placeholders)
            content = html.unescape(content)
            if content == prev_content:
                break
        
        # Restore placeholders back to entities so they remain literal text in XHTML
        content = content.replace(LT_PLACEHOLDER, '&lt;').replace(GT_PLACEHOLDER, '&gt;')
        
        return content

    def _process_single_chapter(self, book: epub.EpubBook, num: int, filename: str,
                               chapter_titles_info: Dict[int, Tuple[str, float, str]],
                               css_items: List[epub.EpubItem], processed_images: Dict[str, str],
                               spine: List, toc: List, metadata: dict) -> bool:
        """Process a single chapter with COMPREHENSIVE debugging"""
        path = os.path.join(self.output_dir, filename)
        
        # Flag for extra debugging on problem chapters
        is_problem_chapter = 49 <= num <= 56
        is_response_file = filename.startswith('response_')
        
        try:
            if is_problem_chapter:
                self.log(f"\n{'='*70}")
                self.log(f"[DEBUG] PROCESSING PROBLEM CHAPTER {num}")
                self.log(f"[DEBUG] Filename: {filename}")
                self.log(f"[DEBUG] Is response file: {is_response_file}")
                self.log(f"[DEBUG] Full path: {path}")
            
            # Check file exists and size
            if not os.path.exists(path):
                self.log(f"[ERROR] File does not exist: {path}")
                return False
            
            file_size = os.path.getsize(path)
            if is_problem_chapter:
                self.log(f"[DEBUG] File size: {file_size} bytes")
            
            if file_size == 0:
                self.log(f"[ERROR] File is empty (0 bytes): {filename}")
                return False
            
            # Read and decode
            if is_problem_chapter:
                self.log(f"[DEBUG] Reading and decoding file...")
            
            raw_content = self._read_and_decode_html_file(path)
            
            if is_problem_chapter:
                self.log(f"[DEBUG] Raw content length: {len(raw_content) if raw_content else 'NULL'}")
                if raw_content:
                    # Show first and last parts
                    self.log(f"[DEBUG] First 300 chars of raw content:")
                    self.log(f"  {raw_content[:300]!r}")
                    self.log(f"[DEBUG] Last 300 chars of raw content:")
                    self.log(f"  {raw_content[-300:]!r}")
                    
                    # Check for common issues
                    if '&lt;' in raw_content[:500]:
                        self.log(f"[DEBUG] Found &lt; entities in content")
                    if '&gt;' in raw_content[:500]:
                        self.log(f"[DEBUG] Found &gt; entities in content")
                    if '<Official' in raw_content[:500] or '<System' in raw_content[:500]:
                        self.log(f"[DEBUG] Found story tags in content")
            
            # Fix encoding issues
            if is_problem_chapter:
                self.log(f"[DEBUG] Fixing encoding issues...")
            
            before_fix = len(raw_content) if raw_content else 0
            raw_content = self._fix_encoding_issues(raw_content)
            after_fix = len(raw_content) if raw_content else 0
            
            if is_problem_chapter:
                self.log(f"[DEBUG] Encoding fix: {before_fix} -> {after_fix} chars")
                if before_fix != after_fix:
                    self.log(f"[DEBUG] Content changed during encoding fix")
            
            if not raw_content or not raw_content.strip():
                self.log(f"[WARNING] Chapter {num} is empty after decoding/encoding fix")
                if is_problem_chapter:
                    self.log(f"[ERROR] Problem chapter {num} has no content!")
                return False
            
            # Extract main content if needed
            if not filename.startswith('response_'):
                if is_problem_chapter:
                    self.log(f"[DEBUG] Extracting main content (not a response file)...")
                
                before_extract = len(raw_content)
                raw_content = self._extract_main_content(raw_content, filename)
                after_extract = len(raw_content)
                
                if is_problem_chapter:
                    self.log(f"[DEBUG] Content extraction: {before_extract} -> {after_extract} chars")
                    if after_extract < before_extract / 2:
                        self.log(f"[WARNING] Lost more than 50% of content during extraction!")
                        self.log(f"[DEBUG] Content after extraction (first 300 chars):")
                        self.log(f"  {raw_content[:300]!r}")
            else:
                if is_problem_chapter:
                    self.log(f"[DEBUG] Skipping content extraction for response file")
                    self.log(f"[DEBUG] Response file content structure:")
                    # Check what's in a response file
                    if '<body>' in raw_content:
                        self.log(f"  Has <body> tag")
                    if '<html>' in raw_content:
                        self.log(f"  Has <html> tag")
                    if '<!DOCTYPE' in raw_content:
                        self.log(f"  Has DOCTYPE declaration")
                    # Check for any obvious issues
                    if raw_content.strip().startswith('Error'):
                        self.log(f"[ERROR] Response file starts with 'Error'")
                    if 'failed' in raw_content.lower()[:500]:
                        self.log(f"[WARNING] Response file contains 'failed' in first 500 chars")
            
            # Get chapter title
            if is_problem_chapter:
                self.log(f"[DEBUG] Getting chapter title...")
            
            title = self._get_chapter_title(num, filename, raw_content, chapter_titles_info)
            
            if is_problem_chapter:
                self.log(f"[DEBUG] Chapter title: {title!r}")
                if title == f"Chapter {num}" or title.startswith("Chapter"):
                    self.log(f"[WARNING] Using generic title, couldn't extract proper title")
            
            # Prepare CSS links
            css_links = [f"css/{item.file_name.split('/')[-1]}" for item in css_items]
            if is_problem_chapter:
                self.log(f"[DEBUG] CSS links: {css_links}")
            
            # XHTML conversion - CRITICAL PART
            if is_problem_chapter:
                self.log(f"[DEBUG] Starting XHTML conversion...")
                self.log(f"[DEBUG] Content length before XHTML: {len(raw_content)}")
            
            xhtml_content = XHTMLConverter.ensure_compliance(raw_content, title, css_links)
            
            if is_problem_chapter:
                self.log(f"[DEBUG] XHTML conversion complete")
                self.log(f"[DEBUG] XHTML content length: {len(xhtml_content) if xhtml_content else 'NULL'}")
                if xhtml_content:
                    # Check if it's the fallback
                    if 'Error processing content' in xhtml_content:
                        self.log(f"[ERROR] Got fallback XHTML - conversion failed!")
                    else:
                        self.log(f"[DEBUG] XHTML first 400 chars:")
                        self.log(f"  {xhtml_content[:400]!r}")
            
            # Process chapter images
            if is_problem_chapter:
                self.log(f"[DEBUG] Processing chapter images...")
            
            xhtml_content = self._process_chapter_images(xhtml_content, processed_images)
            
            # Validate final content
            if is_problem_chapter:
                self.log(f"[DEBUG] Validating final XHTML...")
            
            final_content = XHTMLConverter.validate(xhtml_content)
            
            if is_problem_chapter:
                self.log(f"[DEBUG] Validation complete")
                self.log(f"[DEBUG] Final content length: {len(final_content)}")
                # Check for fallback again
                if 'Error processing content' in final_content:
                    self.log(f"[ERROR] Final content is fallback error page!")
            
            # Create chapter object
            import html
            chapter = epub.EpubHtml(
                title=html.unescape(title),
                file_name=os.path.basename(filename),
                lang=metadata.get("language", "en")
            )
            
            chapter.content = FileUtils.ensure_bytes(final_content)
            
            if is_problem_chapter:
                self.log(f"[DEBUG] Chapter object created")
                self.log(f"[DEBUG] Chapter content size: {len(chapter.content)} bytes")
            
            # Attach CSS if configured
            if self.attach_css_to_chapters:
                for css_item in css_items:
                    chapter.add_item(css_item)
                if is_problem_chapter:
                    self.log(f"[DEBUG] Attached {len(css_items)} CSS files")
            
            # Add to book
            book.add_item(chapter)
            spine.append(chapter)
            toc.append(chapter)
            
            if is_problem_chapter:
                self.log(f"[SUCCESS] Problem chapter {num} successfully added to EPUB!")
                self.log(f"{'='*70}\n")
            else:
                self.log(f"  ✓ Chapter {num}: {title}")
            
            return True
            
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            
            self.log(f"\n{'!'*70}")
            self.log(f"[ERROR] Failed to process chapter {num}: {filename}")
            self.log(f"[ERROR] Exception type: {type(e).__name__}")
            self.log(f"[ERROR] Exception message: {e}")
            
            if is_problem_chapter:
                self.log(f"[ERROR] PROBLEM CHAPTER {num} FAILED!")
                self.log(f"[ERROR] Full traceback:")
                self.log(tb)
                
                # Try to identify the exact failure point
                if 'ensure_compliance' in tb:
                    self.log(f"[ERROR] Failed during XHTML compliance")
                elif 'validate' in tb:
                    self.log(f"[ERROR] Failed during validation")
                elif '_extract_main_content' in tb:
                    self.log(f"[ERROR] Failed during content extraction")
                elif '_read_and_decode' in tb:
                    self.log(f"[ERROR] Failed during file reading/decoding")
            
            self.log(f"{'!'*70}\n")
            
            # Add error chapter as fallback
            self._add_error_chapter(book, num, title if 'title' in locals() else f"Chapter {num}", 
                                    spine, toc, metadata, str(e))
            return False

    def _get_chapter_title(self, num: int, filename: str, content: str,
                          chapter_titles_info: Dict[int, Tuple[str, float, str]]) -> str:
        """Get chapter title with fallbacks - uses position-based numbering"""
        title = None
        confidence = 0.0
        
        # Primary source: pre-analyzed title using position-based number
        if num in chapter_titles_info:
            title, confidence, stored_filename = chapter_titles_info[num]
        
        # Re-extract if low confidence or missing
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
        
        # Fallback for non-standard files
        if not title and not filename.startswith('response_'):
            # Try enhanced extraction methods for web-scraped content
            title = self._fallback_title_extraction(content, filename, num)
        
        # Final fallback - use position-based chapter number
        if not title:
            title = f"Chapter {num}"
        
        return title

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
                import html
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                self.log("[DEBUG] Metadata loaded successfully")
                return metadata
            except Exception as e:
                self.log(f"[WARNING] Failed to load metadata.json: {e}")
        else:
            self.log("[WARNING] metadata.json not found, using defaults")
        
        return {}
    
    def _save_metadata(self, metadata: dict) -> None:
        """Persist metadata.json alongside the translated HTML output.

        This ensures that once metadata fields (title, creator, subject, description, etc.)
        have been translated, future EPUB rebuilds can reuse them instead of sending the
        same fields to the API again.
        """
        if not isinstance(metadata, dict):
            return
        try:
            os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            self.log("[DEBUG] Saved updated metadata.json")
        except Exception as e:
            # Let caller decide how to log/handle, but don't crash compilation here
            raise e
    
    def _create_book(self, metadata: dict) -> epub.EpubBook:
        """Create and configure EPUB book with complete metadata"""
        book = epub.EpubBook()
        
        # Set identifier
        book.set_identifier(metadata.get("identifier", f"translated-{os.path.basename(self.base_dir)}"))
        
        # Fix encoding issues in titles before using them
        if metadata.get('title'):
            metadata['title'] = self._fix_encoding_issues(metadata['title'])
        if metadata.get('original_title'):
            metadata['original_title'] = self._fix_encoding_issues(metadata['original_title'])
        
        # Determine title
        book_title = self._determine_book_title(metadata)
        book.set_title(book_title)
        
        # Set language (dc:language)
        language_code = metadata.get("language", "en")
        book.set_language(language_code)

        # Keep XHTMLConverter in sync so generated html/xml:lang attributes match book language
        try:
            XHTMLConverter.set_default_language(language_code)
        except Exception:
            pass
        
        # Store original title as alternative metadata (not as another dc:title)
        # This prevents EPUB readers from getting confused about which title to display
        if metadata.get('original_title') and metadata.get('original_title') != book_title:
            # Use 'alternative' field instead of 'title' to avoid display issues
            book.add_metadata('DC', 'alternative', metadata['original_title'])
            # Also store in a custom field for reference
            book.add_metadata('calibre', 'original_title', metadata['original_title'])
            self.log(f"[INFO] Stored original title as alternative: {metadata['original_title']}")
        
        # Set author/creator
        if metadata.get("creator"):
            book.add_author(metadata["creator"])
            self.log(f"[INFO] Set author: {metadata['creator']}")
        
        # ADD DESCRIPTION - This is what Calibre looks for
        if metadata.get("description"):
            # Clean the description of any HTML entities
            description = HTMLEntityDecoder.decode(str(metadata["description"]))
            book.add_metadata('DC', 'description', description)
            self.log(f"[INFO] Set description: {description[:100]}..." if len(description) > 100 else f"[INFO] Set description: {description}")
        
        # Add publisher
        if metadata.get("publisher"):
            book.add_metadata('DC', 'publisher', metadata["publisher"])
            self.log(f"[INFO] Set publisher: {metadata['publisher']}")
        
        # Add publication date
        if metadata.get("date"):
            book.add_metadata('DC', 'date', metadata["date"])
            self.log(f"[INFO] Set date: {metadata['date']}")
        
        # Add rights/copyright
        if metadata.get("rights"):
            book.add_metadata('DC', 'rights', metadata["rights"])
            self.log(f"[INFO] Set rights: {metadata['rights']}")
        
        # Add subject/genre/tags
        if metadata.get("subject"):
            if isinstance(metadata["subject"], list):
                for subject in metadata["subject"]:
                    book.add_metadata('DC', 'subject', subject)
                    self.log(f"[INFO] Added subject: {subject}")
            else:
                book.add_metadata('DC', 'subject', metadata["subject"])
                self.log(f"[INFO] Set subject: {metadata['subject']}")
        
        # Add series information if available
        if metadata.get("series"):
            # Calibre uses a custom metadata field for series
            book.add_metadata('calibre', 'series', metadata["series"])
            self.log(f"[INFO] Set series: {metadata['series']}")
            
            # Add series index if available
            if metadata.get("series_index"):
                book.add_metadata('calibre', 'series_index', str(metadata["series_index"]))
                self.log(f"[INFO] Set series index: {metadata['series_index']}")
        
        # Add custom metadata for translator info
        if metadata.get("translator"):
            book.add_metadata('DC', 'contributor', metadata["translator"], {'role': 'translator'})
            self.log(f"[INFO] Set translator: {metadata['translator']}")
        
        # Add source information
        if metadata.get("source"):
            book.add_metadata('DC', 'source', metadata["source"])
            self.log(f"[INFO] Set source: {metadata['source']}")
        
        # Add any ISBN if available
        if metadata.get("isbn"):
            book.add_metadata('DC', 'identifier', f"ISBN:{metadata['isbn']}", {'scheme': 'ISBN'})
            self.log(f"[INFO] Set ISBN: {metadata['isbn']}")
        
        # Add coverage (geographic/temporal scope) if available
        if metadata.get("coverage"):
            book.add_metadata('DC', 'coverage', metadata["coverage"])
            self.log(f"[INFO] Set coverage: {metadata['coverage']}")
        
        # Add any custom metadata that might be in the JSON
        # This handles any additional fields that might be present
        custom_metadata_fields = [
            'contributor', 'format', 'relation', 'type'
        ]
        
        for field in custom_metadata_fields:
            if metadata.get(field):
                book.add_metadata('DC', field, metadata[field])
                self.log(f"[INFO] Set {field}: {metadata[field]}")
        
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
    white-space: normal;
}

/* Ensure proper word spacing for readers like Freda */
body, p, div, span {
    word-spacing: normal;
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
        """Add CSS files to book.

        Behavior:
          * Always adds our built‑in default.css.
          * If EPUB_CSS_OVERRIDE_PATH is set, add ONLY that CSS (plus default) and
            skip all CSS files from the extracted EPUB/css directory.
          * Also overwrites PDF-generated styles.css in the output directory if override is set.
          * Otherwise, add all .css files from self.css_dir as before.
        """
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
        
        # Check for explicit CSS override from GUI
        # Only apply if attach_css_to_chapters is enabled
        override_path = os.getenv('EPUB_CSS_OVERRIDE_PATH', '').strip()
        if override_path and self.attach_css_to_chapters:
            try:
                if os.path.isfile(override_path):
                    self.log(f"[INFO] Using override CSS for EPUB: {override_path}")
                    with open(override_path, 'r', encoding='utf-8') as f:
                        css_content = f.read()
                    
                    # IMPORTANT: Overwrite PDF-generated styles.css if it exists
                    styles_css_path = os.path.join(self.output_dir, 'styles.css')
                    if os.path.exists(styles_css_path):
                        try:
                            with open(styles_css_path, 'w', encoding='utf-8') as f:
                                f.write(css_content)
                            self.log(f"✅ Overwrote PDF-generated styles.css with loaded CSS")
                        except Exception as e:
                            self.log(f"[WARNING] Failed to overwrite styles.css: {e}")
                    
                    override_item = epub.EpubItem(
                        uid="css_override",
                        file_name="css/override.css",
                        media_type="text/css",
                        content=FileUtils.ensure_bytes(css_content)
                    )
                    book.add_item(override_item)
                    css_items.append(override_item)
                    return css_items
                else:
                    self.log(f"[WARNING] EPUB_CSS_OVERRIDE_PATH does not exist: {override_path}")
            except Exception as e:
                self.log(f"[WARNING] Failed to load override CSS '{override_path}': {e}")
                # Fall back to normal behavior below
        elif override_path and not self.attach_css_to_chapters:
            self.log(f"[INFO] CSS override set but 'Attach CSS to Chapters' is disabled - ignoring override")
        
        # Then add user CSS files from css/ directory (original behavior)
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
        """Process images using parallel processing"""
        processed_images = {}
        cover_file = None
        
        try:
            # Find the images directory
            actual_images_dir = None
            possible_dirs = [
                self.images_dir,
                os.path.join(self.base_dir, "images"),
                os.path.join(self.output_dir, "images"),
            ]
            
            for test_dir in possible_dirs:
                self.log(f"[DEBUG] Checking for images in: {test_dir}")
                if os.path.isdir(test_dir):
                    files = os.listdir(test_dir)
                    if files:
                        self.log(f"[DEBUG] Found {len(files)} files in {test_dir}")
                        actual_images_dir = test_dir
                        break
            
            if not actual_images_dir:
                self.log("[WARNING] No images directory found or directory is empty")
                return processed_images, cover_file
            
            self.images_dir = actual_images_dir
            self.log(f"[INFO] Using images directory: {self.images_dir}")
            
            # Get list of files to process
            image_files = sorted(os.listdir(self.images_dir))
            self.log(f"🖼️ Processing {len(image_files)} potential images with {self.max_workers} workers")
            
            def process_single_image(img):
                """Worker function to process a single image"""
                path = os.path.join(self.images_dir, img)
                if not os.path.isfile(path):
                    return None
                
                # Check MIME type
                ctype, _ = mimetypes.guess_type(path)
                
                # If MIME type detection fails, check extension
                if not ctype:
                    ext = os.path.splitext(img)[1].lower()
                    mime_map = {
                        '.jpg': 'image/jpeg',
                        '.jpeg': 'image/jpeg',
                        '.png': 'image/png',
                        '.gif': 'image/gif',
                        '.bmp': 'image/bmp',
                        '.webp': 'image/webp',
                        '.svg': 'image/svg+xml'
                    }
                    ctype = mime_map.get(ext)
                
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
                    
                    # Special handling for SVG: rasterize to PNG fallback for reader compatibility
                    if ctype == 'image/svg+xml' and self.rasterize_svg and self._cairosvg_available:
                        try:
                            from cairosvg import svg2png
                            png_name = os.path.splitext(safe_name)[0] + '.png'
                            png_path = os.path.join(self.images_dir, png_name)
                            # Generate PNG only if not already present
                            if not os.path.exists(png_path):
                                svg2png(url=path, write_to=png_path)
                                self.log(f"  🖼️ Rasterized SVG → PNG: {img} -> {png_name}")
                            # Return the PNG as the image to include
                            return (png_name, png_name, 'image/png')
                        except Exception as e:
                            self.log(f"[WARNING] SVG rasterization failed for {img}: {e}")
                            # Fall back to adding the raw SVG
                            return (img, safe_name, ctype)
                    
                    return (img, safe_name, ctype)
                else:
                    return None
            
            # Process images in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(process_single_image, img) for img in image_files]
                
                completed = 0
                # Only log periodically for large image sets to avoid GUI lag
                use_reduced_logging = len(image_files) > 50
                # Log at 5% intervals for large sets (20 updates total)
                log_interval = max(1, len(image_files) // 20) if use_reduced_logging else 1
                last_logged_percent = -1
                
                for future in as_completed(futures):
                    # Check stop flag
                    if self.is_stopped():
                        self.log("🛑 Image processing stopped by user")
                        break
                    
                    try:
                        result = future.result()
                        completed += 1
                        
                        if result:
                            original, safe, ctype = result
                            processed_images[original] = safe
                            # Log based on image count
                            if use_reduced_logging:
                                # For large sets: log at percentage milestones or interval
                                current_percent = (completed * 100) // len(image_files)
                                should_log = (completed % log_interval == 0 or completed == 1 or completed == len(image_files))
                                # Also log when percentage changes for better feedback
                                if should_log or (current_percent != last_logged_percent and current_percent % 5 == 0):
                                    self.log(f"  [{completed}/{len(image_files)}] ({current_percent}%) ✅")
                                    last_logged_percent = current_percent
                            else:
                                # For small sets: log every image
                                self.log(f"  [{completed}/{len(image_files)}] ✅ Processed: {original} -> {safe}")
                        else:
                            # Log skipped files based on image count
                            if not use_reduced_logging or completed % log_interval == 0:
                                self.log(f"  [{completed}/{len(image_files)}] ⏭️ Skipped non-image file")
                            
                    except Exception as e:
                        completed += 1
                        # Always log errors
                        self.log(f"  [{completed}/{len(image_files)}] ❌ Failed to process image: {e}")
            
            # Find cover (sequential - quick operation)
            # Respect user preference to disable automatic cover creation
            disable_auto_cover = os.environ.get('DISABLE_AUTOMATIC_COVER_CREATION', '0') == '1'
            if processed_images and not disable_auto_cover:
                cover_prefixes = ['cover', 'front']
                for original_name, safe_name in processed_images.items():
                    name_lower = original_name.lower()
                    if any(name_lower.startswith(prefix) for prefix in cover_prefixes):
                        cover_file = safe_name
                        self.log(f"📔 Found cover image: {original_name} -> {cover_file}")
                        break
                
                if not cover_file:
                    cover_file = next(iter(processed_images.values()))
                    self.log(f"📔 Using first image as cover: {cover_file}")
            
            self.log(f"✅ Processed {len(processed_images)} images successfully")
            
        except Exception as e:
            self.log(f"[ERROR] Error processing images: {e}")
            import traceback
            self.log(f"[DEBUG] Traceback: {traceback.format_exc()}")
        
        return processed_images, cover_file

    def _add_images_to_book(self, book: epub.EpubBook, processed_images: Dict[str, str], 
                           cover_file: Optional[str]):
        """Add images to book using parallel processing for reading files"""
        
        # Filter out cover image
        images_to_add = [(orig, safe) for orig, safe in processed_images.items() 
                         if safe != cover_file]
        
        if not images_to_add:
            self.log("No images to add (besides cover)")
            return
        
        self.log(f"📚 Adding {len(images_to_add)} images to EPUB with {self.max_workers} workers")
        
        def read_image_file(image_data):
            """Worker function to read image file"""
            original_name, safe_name = image_data
            img_path = os.path.join(self.images_dir, original_name)
            
            # If original was compressed to a different format (e.g. png→webp),
            # the original file may have been deleted. Fall back to the
            # compressed file whose extension matches safe_name.
            if not os.path.isfile(img_path):
                safe_ext = os.path.splitext(safe_name)[1]
                if safe_ext:
                    alt_path = os.path.join(
                        self.images_dir,
                        os.path.splitext(original_name)[0] + safe_ext
                    )
                    if os.path.isfile(alt_path):
                        img_path = alt_path
            
            try:
                ctype, _ = mimetypes.guess_type(img_path)
                if not ctype:
                    ctype = "image/jpeg"  # Default fallback
                
                with open(img_path, 'rb') as f:
                    content = f.read()
                
                return {
                    'original': original_name,
                    'safe': safe_name,
                    'ctype': ctype,
                    'content': content,
                    'success': True
                }
            except Exception as e:
                return {
                    'original': original_name,
                    'safe': safe_name,
                    'error': str(e),
                    'success': False
                }
        
        # Read all images in parallel
        image_data_list = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(read_image_file, img_data) for img_data in images_to_add]
            
            completed = 0
            # Only log periodically for large image sets to avoid GUI lag
            use_reduced_logging = len(images_to_add) > 50
            # Log at 5% intervals for large sets (20 updates total)
            log_interval = max(1, len(images_to_add) // 20) if use_reduced_logging else 1
            last_logged_percent = -1
            
            for future in as_completed(futures):
                # Check stop flag
                if self.is_stopped():
                    self.log("🛑 Image reading stopped by user")
                    break
                
                try:
                    result = future.result()
                    completed += 1
                    
                    if result['success']:
                        image_data_list.append(result)
                        # Log based on image count
                        if use_reduced_logging:
                            # For large sets: log at percentage milestones or interval
                            current_percent = (completed * 100) // len(images_to_add)
                            should_log = (completed % log_interval == 0 or completed == 1 or completed == len(images_to_add))
                            # Also log when percentage changes for better feedback
                            if should_log or (current_percent != last_logged_percent and current_percent % 5 == 0):
                                self.log(f"  [{completed}/{len(images_to_add)}] ({current_percent}%) ✅")
                                last_logged_percent = current_percent
                        else:
                            # For small sets: log every image
                            self.log(f"  [{completed}/{len(images_to_add)}] ✅ Read: {result['original']}")
                    else:
                        # Always log failures
                        self.log(f"  [{completed}/{len(images_to_add)}] ❌ Failed: {result['original']} - {result['error']}")
                        
                except Exception as e:
                    completed += 1
                    # Always log exceptions
                    self.log(f"  [{completed}/{len(images_to_add)}] ❌ Exception reading image: {e}")
        
        # Add images to book sequentially (required by ebooklib)
        self.log("\n📦 Adding images to EPUB structure...")
        added = 0
        use_reduced_logging = len(image_data_list) > 50
        log_interval = max(1, len(image_data_list) // 20) if use_reduced_logging else 1
        
        for idx, img_data in enumerate(image_data_list, 1):
            # Check stop flag
            if self.is_stopped():
                self.log(f"🛑 Image addition stopped by user ({added}/{len(image_data_list)} images added)")
                break
            
            try:
                book.add_item(epub.EpubItem(
                    uid=img_data['safe'],
                    file_name=f"images/{img_data['safe']}",
                    media_type=img_data['ctype'],
                    content=img_data['content']
                ))
                added += 1
                # Only log periodically for large sets
                if use_reduced_logging:
                    if idx % log_interval == 0 or idx == 1 or idx == len(image_data_list):
                        percent = (idx * 100) // len(image_data_list)
                        self.log(f"  [{idx}/{len(image_data_list)}] ({percent}%) ✅")
                else:
                    self.log(f"  ✅ Added: {img_data['original']}")
            except Exception as e:
                self.log(f"  ❌ Failed to add {img_data['original']} to EPUB: {e}")
        
        if self.is_stopped():
            self.log(f"⚠️ Image addition incomplete: {added}/{len(images_to_add)} images were added before stopping")
        else:
            self.log(f"✅ Successfully added {added}/{len(images_to_add)} images to EPUB")
    
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
        
        # If original was compressed to a different format (e.g. jpg→webp),
        # the original file may have been deleted. Fall back to the
        # compressed file whose extension matches cover_file.
        if not os.path.isfile(cover_path):
            cover_ext = os.path.splitext(cover_file)[1]
            if cover_ext:
                alt_path = os.path.join(
                    self.images_dir,
                    os.path.splitext(original_cover)[0] + cover_ext
                )
                if os.path.isfile(alt_path):
                    cover_path = alt_path
        
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
            
            # Build cover HTML directly without going through ensure_compliance
            # Since it's simple and controlled, we can build it directly
            lang = metadata.get("language", "en")
            cover_content = f'''<?xml version="1.0" encoding="utf-8"?>
    <!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">
    <html xmlns="http://www.w3.org/1999/xhtml" xml:lang="{lang}" lang="{lang}">
    <head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
    <title>Cover</title>
    </head>
    <body>
    <div style="text-align: center;">
    <img src="images/{cover_file}" alt="Cover" style="max-width: 100%; height: auto;" />
    </div>
    </body>
    </html>'''
            
            cover_page.content = cover_content.encode('utf-8')
            
            # Associate CSS with cover page if needed
            if self.attach_css_to_chapters:
                for css_item in css_items:
                    cover_page.add_item(css_item)

            book.add_item(cover_page)
            self.log(f"✅ Set cover image: {cover_file}")
            return cover_page
            
        except Exception as e:
            self.log(f"[WARNING] Failed to add cover: {e}")
            return None
    
    def _process_chapter_images(self, xhtml_content: str, processed_images: Dict[str, str]) -> str:
        """Process image paths and inline SVG in chapter content.
        - Rewrites <img src> to use images/ paths and prefers PNG fallback for SVGs.
        - Converts inline <svg> elements to <img src="data:image/png;base64,..."> when CairoSVG is available.
        """
        try:
            soup = BeautifulSoup(xhtml_content, 'lxml')
            changed = False
            
            # Track statistics for summary
            total_images = 0
            found_images = 0
            missing_images = []
            
            # 1) Handle <img> tags that reference files
            for img in soup.find_all('img'):
                src = img.get('src', '')
                if not src:
                    self.log(f"[WARNING] Image tag with no src attribute found")
                    continue
                
                total_images += 1
                
                # Get the base filename - handle various path formats
                # Remove query parameters first
                clean_src = src.split('?')[0]
                basename = os.path.basename(clean_src)
                
                # Look up the safe name
                if basename in processed_images:
                    safe_name = processed_images[basename]
                    new_src = f"images/{safe_name}"
                    
                    if src != new_src:
                        img['src'] = new_src
                        changed = True
                    found_images += 1
                else:
                    # Try without extension variations
                    name_without_ext = os.path.splitext(basename)[0]
                    found = False
                    for original_name, safe_name in processed_images.items():
                        if os.path.splitext(original_name)[0] == name_without_ext:
                            new_src = f"images/{safe_name}"
                            img['src'] = new_src
                            changed = True
                            found = True
                            found_images += 1
                            break
                    
                    if not found:
                        missing_images.append(basename)
                        # Still update the path to use images/ prefix if it doesn't have it
                        if not src.startswith('images/'):
                            img['src'] = f"images/{basename}"
                            changed = True
                
                # Ensure alt attribute exists (required for XHTML)
                if not img.get('alt'):
                    img['alt'] = ''
                    changed = True
            
            # 2) Convert inline SVG wrappers that point to raster images into plain <img>
            #    Example: <svg ...><image xlink:href="../images/00002.jpeg"/></svg>
            for svg_tag in soup.find_all('svg'):
                try:
                    image_child = svg_tag.find('image')
                    if image_child:
                        href = (
                            image_child.get('xlink:href') or
                            image_child.get('href') or
                            image_child.get('{http://www.w3.org/1999/xlink}href')
                        )
                        if href:
                            clean_href = href.split('?')[0]
                            basename = os.path.basename(clean_href)
                            # Map to processed image name
                            if basename in processed_images:
                                safe_name = processed_images[basename]
                            else:
                                name_wo = os.path.splitext(basename)[0]
                                safe_name = None
                                for orig, safe in processed_images.items():
                                    if os.path.splitext(orig)[0] == name_wo:
                                        safe_name = safe
                                        break
                            new_src = f"images/{safe_name}" if safe_name else f"images/{basename}"
                            new_img = soup.new_tag('img')
                            new_img['src'] = new_src
                            new_img['alt'] = svg_tag.get('aria-label') or svg_tag.get('title') or ''
                            new_img['style'] = 'width:100%; height:auto; display:block;'
                            svg_tag.replace_with(new_img)
                            changed = True
                            self.log(f"[DEBUG] Rewrote inline SVG<image> to <img src='{new_src}'>")
                except Exception as e:
                    self.log(f"[WARNING] Failed to rewrite inline SVG wrapper: {e}")
            
            # 3) Convert remaining inline <svg> (complex vector art) to PNG data URIs if possible
            if self.rasterize_svg and self._cairosvg_available:
                try:
                    from cairosvg import svg2png
                    import base64
                    for svg_tag in soup.find_all('svg'):
                        try:
                            svg_markup = str(svg_tag)
                            png_bytes = svg2png(bytestring=svg_markup.encode('utf-8'))
                            b64 = base64.b64encode(png_bytes).decode('ascii')
                            alt_text = svg_tag.get('aria-label') or svg_tag.get('title') or ''
                            new_img = soup.new_tag('img')
                            new_img['src'] = f'data:image/png;base64,{b64}'
                            new_img['alt'] = alt_text
                            new_img['style'] = 'width:100%; height:auto; display:block;'
                            svg_tag.replace_with(new_img)
                            changed = True
                            self.log("[DEBUG] Converted inline <svg> to PNG data URI")
                        except Exception as e:
                            self.log(f"[WARNING] Failed to rasterize inline SVG: {e}")
                except Exception:
                    pass
            
            # Log summary only if there are issues
            if total_images > 0 and missing_images:
                self.log(f"[WARNING] Chapter images: {found_images}/{total_images} found. Missing: {missing_images[:5]}{'...' if len(missing_images) > 5 else ''}")
            
            if changed:
                # Return the modified content
                return str(soup)
            
            return xhtml_content
            
        except Exception as e:
            self.log(f"[WARNING] Failed to process images in chapter: {e}")
            return xhtml_content
    
    def _create_gallery_page(self, book: epub.EpubBook, images: List[str],
                            css_items: List[epub.EpubItem], metadata: dict) -> epub.EpubHtml:
        """Create image gallery page - FIXED to avoid escaping HTML tags"""
        gallery_page = epub.EpubHtml(
            title="Gallery",
            file_name="gallery.xhtml",
            lang=metadata.get("language", "en")
        )
        
        # Build the gallery body content
        gallery_body_parts = ['<h1>Image Gallery</h1>']
        for img in images:
            gallery_body_parts.append(
                f'<div style="text-align: center; margin: 20px;">'
                f'<img src="images/{img}" alt="{img}" />'
                f'</div>'
            )
        
        gallery_body_content = '\n'.join(gallery_body_parts)
        
        # Build XHTML directly without going through ensure_compliance
        # which might escape our HTML tags
        css_links = [f"css/{item.file_name.split('/')[-1]}" for item in css_items]
        
        # Build the complete XHTML document manually
        lang = metadata.get("language", "en")
        xhtml_content = f'''<?xml version="1.0" encoding="utf-8"?>
    <!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">
    <html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" xml:lang="{lang}" lang="{lang}">
    <head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
    <title>Gallery</title>'''
        
        # Add CSS links
        for css_link in css_links:
            xhtml_content += f'\n<link rel="stylesheet" type="text/css" href="{css_link}" />'
        
        xhtml_content += f'''
    </head>
    <body>
    {gallery_body_content}
    </body>
    </html>'''
        
        # Validate the XHTML
        validated_content = XHTMLConverter.validate(xhtml_content)
        
        # Set the content
        gallery_page.content = FileUtils.ensure_bytes(validated_content)
        
        # Associate CSS with gallery page
        if self.attach_css_to_chapters:
            for css_item in css_items:
                gallery_page.add_item(css_item)
        
        book.add_item(gallery_page)
        return gallery_page
            
    def _create_nav_content(self, toc_items, book_title="Book"):
        """Create navigation content manually"""
        # Use the same primary language as the rest of the book for nav.xhtml
        # We read from XHTMLConverter.DEFAULT_LANG, which is kept in sync with book language.
        lang = getattr(XHTMLConverter, "DEFAULT_LANG", "en")
        nav_content = f'''<?xml version="1.0" encoding="utf-8"?>
    <!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">
    <html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" xml:lang="{lang}" lang="{lang}">
    <head>
    <title>Table of Contents</title>
    </head>
    <body>
    <nav epub:type="toc" id="toc">
    <h1>Table of Contents</h1>
    <ol>'''
        
        # The toc_items are already sorted properly by _finalize_book
        # Don't re-sort them here - just use them as-is
        for item in toc_items:
            if hasattr(item, 'title') and hasattr(item, 'file_name'):
                nav_content += f'\n<li><a href="{item.file_name}">{ContentProcessor.safe_escape(item.title)}</a></li>'
        
        nav_content += '''
    </ol>
    </nav>
    </body>
    </html>'''
        
        return nav_content


    def _get_order_from_progress_file(self, progress_file: str) -> Dict[str, int]:
        """Get chapter order from translation_progress.json
        Returns dict mapping original_filename -> chapter_number
        """
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)
            
            filename_to_order = {}
            
            # Extract chapter order from progress data
            chapters = progress_data.get('chapters', {})
            
            for chapter_key, chapter_info in chapters.items():
                # Get the original basename from progress data
                original_basename = chapter_info.get('original_basename', '')
                if original_basename:
                    # Map to chapter position (key is usually the chapter number)
                    try:
                        chapter_num = int(chapter_key)
                        filename_to_order[original_basename] = chapter_num - 1  # Convert to 0-based
                        self.log(f"  Progress mapping: {original_basename} -> Chapter {chapter_num}")
                    except (ValueError, TypeError):
                        pass
            
            return filename_to_order if filename_to_order else None
            
        except Exception as e:
            self.log(f"⚠️ Error reading translation_progress.json: {e}")
            return None

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
        
        # DEBUG: Log what we have before sorting (only if debug mode is enabled)
        debug_mode_enabled = os.environ.get('DEBUG_MODE', '0') == '1'
        if debug_mode_enabled:
            self.log("\n[DEBUG] Before sorting TOC:")
            self.log("Spine order:")
            for idx, item in enumerate(spine):
                if hasattr(item, 'file_name') and hasattr(item, 'title'):
                    self.log(f"  Spine[{idx}]: {item.file_name} -> {item.title}")
            
            for idx, item in enumerate(toc):
                if hasattr(item, 'file_name') and hasattr(item, 'title'):
                    self.log(f"  TOC[{idx}]: {item.file_name} -> {item.title}")
        
        # CRITICAL FIX: Sort TOC to match spine order
        # Create a mapping of file_name to spine position
        spine_order = {}
        for idx, item in enumerate(spine):
            if hasattr(item, 'file_name'):
                spine_order[item.file_name] = idx
        
        # Sort the TOC based on spine order
        sorted_toc = []
        unsorted_items = []
        
        for toc_item in toc:
            if hasattr(toc_item, 'file_name'):
                if toc_item.file_name in spine_order:
                    sorted_toc.append((spine_order[toc_item.file_name], toc_item))
                else:
                    # Items not in spine (like gallery) go at the end
                    unsorted_items.append(toc_item)
            else:
                unsorted_items.append(toc_item)
        
        # Sort by spine position
        sorted_toc.sort(key=lambda x: x[0])
        
        # Extract just the items (remove the sort key)
        final_toc = [item for _, item in sorted_toc]
        
        # Add any unsorted items at the end (like gallery)
        final_toc.extend(unsorted_items)
        
        # DEBUG: Log after sorting (only if debug mode is enabled)
        if debug_mode_enabled:
            self.log("\nTOC order (after sorting to match spine):")
            for idx, item in enumerate(final_toc):
                if hasattr(item, 'file_name') and hasattr(item, 'title'):
                    self.log(f"  TOC[{idx}]: {item.file_name} -> {item.title}")
        
        # Set the sorted TOC
        book.toc = final_toc
        
        # Add NCX
        ncx = epub.EpubNcx()
        book.add_item(ncx)
        
        if use_ncx_only:
            self.log(f"[INFO] NCX-only navigation forced - {len(final_toc)} chapters")
            
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
            self.log(f"[INFO] EPUB3 format - {len(final_toc)} chapters")
            
            # Create Nav with manual content using SORTED TOC
            nav = epub.EpubNav()
            nav.content = self._create_nav_content(final_toc, book.title).encode('utf-8')
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
        import time
        import threading
        
        # Determine output filename
        book_title = book.title
        if book_title and book_title != os.path.basename(self.output_dir):
            safe_filename = FileUtils.sanitize_filename(book_title, allow_unicode=True)
            out_path = os.path.join(self.output_dir, f"{safe_filename}.epub")
        else:
            base_name = os.path.basename(self.output_dir)
            out_path = os.path.join(self.output_dir, f"{base_name}.epub")
        
        # Check stop flag before starting the write operation
        if self.is_stopped():
            self.log("🛑 EPUB write cancelled - stop requested before write started")
            return
        
        self.log(f"\n[DEBUG] Writing EPUB to: {out_path}")
        
        # Test if we can write to the target file BEFORE generating EPUB
        try:
            # Try to open the file in write mode to detect if it's locked
            with open(out_path, 'ab') as test_file:
                pass  # Just checking if we can open it
        except PermissionError as e:
            self.log(f"[ERROR] Cannot write to file - it may be opened in another program")
            self.log(f"[ERROR] File: {out_path}")
            self.log(f"[ERROR] Details: {e}")
            raise Exception(f"File is locked or inaccessible: {out_path}") from e
        except Exception as e:
            self.log(f"[ERROR] Cannot access file for writing: {e}")
            raise
        
        self.log("⏳ Writing EPUB file... (this may take a while for large files)")
        
        # Track elapsed time with periodic updates
        start_time = time.time()
        write_completed = threading.Event()
        
        def progress_logger():
            """Log progress every 5 seconds during write"""
            while not write_completed.is_set():
                if write_completed.wait(5):  # Wait 5 seconds or until completed
                    break
                # Check stop flag during write
                if self.is_stopped():
                    elapsed = time.time() - start_time
                    self.log(f"⏳ Still writing... ({elapsed:.0f}s elapsed) - Stop requested, will finish current write operation")
                else:
                    elapsed = time.time() - start_time
                    self.log(f"⏳ Still writing... ({elapsed:.0f}s elapsed)")
        
        # Start progress logger thread
        logger_thread = threading.Thread(target=progress_logger, daemon=True)
        logger_thread.start()
        
        # Always write as EPUB3
        try:
            opts = {'epub3': True}
            epub.write_epub(out_path, book, opts)
            write_completed.set()  # Signal completion
            logger_thread.join(timeout=1)  # Wait for logger to finish
            
            # Check if stop was requested during write
            if self.is_stopped():
                elapsed = time.time() - start_time
                self.log(f"[SUCCESS] Written as EPUB 3.3 (took {elapsed:.1f}s) - Write completed before stop")
                self.log("🛑 Note: Stop was requested but write operation finished normally")
            else:
                elapsed = time.time() - start_time
                self.log(f"[SUCCESS] Written as EPUB 3.3 (took {elapsed:.1f}s)")
            
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


    def _compress_images(self, processed_images: Dict[str, str], cover_file: Optional[str]) -> Tuple[Dict[str, str], Optional[str]]:
        """Compress images: convert to .webp, with configurable cover/GIF exclusion and quality"""
        try:
            from PIL import Image
        except ImportError:
            self.log("⚠️ Pillow not installed - image compression disabled. Install with: pip install Pillow")
            return processed_images, cover_file
        
        # Read compression settings
        quality = int(os.environ.get('IMAGE_COMPRESSION_QUALITY', '80'))
        exclude_cover = os.environ.get('EXCLUDE_COVER_COMPRESSION', '1') == '1'
        exclude_gif = os.environ.get('EXCLUDE_GIF_COMPRESSION', '1') == '1'
        
        self.log(f"\n🗜️ Compressing images (quality: {quality}%, exclude cover: {exclude_cover}, exclude GIF: {exclude_gif})...")
        new_processed = {}
        new_cover = cover_file
        compressed_count = 0
        skipped_count = 0
        
        for original_name, safe_name in processed_images.items():
            if self.is_stopped():
                self.log("🛑 Image compression stopped by user")
                break
            
            img_path = os.path.join(self.images_dir, original_name)
            if not os.path.isfile(img_path):
                new_processed[original_name] = safe_name
                continue
            
            ext = os.path.splitext(original_name)[1].lower()
            is_cover = (safe_name == cover_file)
            is_gif = (ext == '.gif')
            
            # Skip cover page if excluded
            if is_cover and exclude_cover:
                self.log(f"  ⏭️ Skipping cover: {original_name}")
                new_processed[original_name] = safe_name
                skipped_count += 1
                continue
            
            # Skip GIF if excluded
            if is_gif and exclude_gif:
                self.log(f"  ⏭️ Skipping GIF: {original_name}")
                new_processed[original_name] = safe_name
                skipped_count += 1
                continue
            
            try:
                if is_gif:
                    # Compress GIF in place - optimize without changing format
                    im = Image.open(img_path)
                    if hasattr(im, 'n_frames') and im.n_frames > 1:
                        # Animated GIF: save with optimization
                        frames = []
                        try:
                            while True:
                                frames.append(im.copy())
                                im.seek(im.tell() + 1)
                        except EOFError:
                            pass
                        if frames:
                            frames[0].save(
                                img_path, save_all=True, append_images=frames[1:],
                                optimize=True, loop=im.info.get('loop', 0)
                            )
                    else:
                        # Static GIF: optimize
                        im.save(img_path, optimize=True)
                    im.close()
                    new_processed[original_name] = safe_name
                    compressed_count += 1
                else:
                    # Convert to .webp
                    webp_name = os.path.splitext(safe_name)[0] + '.webp'
                    webp_path = os.path.join(self.images_dir, os.path.splitext(original_name)[0] + '.webp')
                    
                    im = Image.open(img_path)
                    # Convert RGBA/P modes for webp compatibility
                    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
                        im = im.convert('RGBA')
                    elif im.mode != 'RGB':
                        im = im.convert('RGB')
                    
                    im.save(webp_path, 'WEBP', quality=quality, method=4)
                    im.close()
                    
                    # Remove original if webp was created successfully and is different file
                    if os.path.exists(webp_path) and webp_path != img_path:
                        try:
                            os.remove(img_path)
                        except Exception:
                            pass
                    
                    # Keep original key so HTML image references still resolve correctly.
                    # _add_images_to_book and _create_cover_page have fallback logic to
                    # find the compressed .webp file on disk when the original is gone.
                    new_processed[original_name] = webp_name
                    compressed_count += 1
                    
                    # Update cover reference if this image was the cover
                    if is_cover:
                        new_cover = webp_name
                    
            except Exception as e:
                self.log(f"  ⚠️ Failed to compress {original_name}: {e}")
                new_processed[original_name] = safe_name
                skipped_count += 1
        
        self.log(f"✅ Image compression complete: {compressed_count} compressed, {skipped_count} skipped")
        return new_processed, new_cover

    def _generate_pdf(self, html_files: List[str], chapter_titles_info: Dict[int, Tuple[str, float, str]],
                      processed_images: Dict[str, str], cover_file: Optional[str], metadata: dict):
        """Generate PDF from the output folder using WeasyPrint"""
        import time as _time
        
        # Ensure fontconfig is available on Windows for WeasyPrint
        import tempfile as _tempfile
        if not os.environ.get("FONTCONFIG_FILE"):
            _fc_dir = _tempfile.mkdtemp(prefix="fontconfig_")
            _fc_path = os.path.join(_fc_dir, "fonts.conf")
            if not os.path.exists(_fc_path):
                with open(_fc_path, "w", encoding="utf-8") as _f:
                    _f.write('<?xml version="1.0"?>\n<!DOCTYPE fontconfig SYSTEM "fonts.dtd">\n'
                             '<fontconfig><dir>WINDOWSFONTDIR</dir>'
                             '<cachedir>~/.cache/fontconfig</cachedir></fontconfig>\n')
            os.environ["FONTCONFIG_FILE"] = _fc_path
            os.environ["FONTCONFIG_PATH"] = _fc_dir
            os.environ["FC_CONFIG_FILE"] = _fc_path
        
        try:
            from weasyprint import HTML as WeasyHTML
        except ImportError:
            self.log("⚠️ WeasyPrint not installed - PDF generation disabled. Install with: pip install weasyprint")
            self.log("   Also requires GTK libraries. See: https://doc.courtbouillon.org/weasyprint/stable/first_steps.html")
            return
        
        self.log("\n📄 Generating PDF...")
        start_time = _time.time()
        
        settings = {
            'page_numbers': os.environ.get('PDF_PAGE_NUMBERS', '1') == '1',
            'page_number_alignment': os.environ.get('PDF_PAGE_NUMBER_ALIGNMENT', 'center'),
            'toc': os.environ.get('PDF_GENERATE_TOC', '1') == '1',
            'toc_numbers': os.environ.get('PDF_TOC_PAGE_NUMBERS', '1') == '1',
        }
        
        # Determine PDF output path
        book_title = metadata.get('title', os.path.basename(self.output_dir))
        safe_title = FileUtils.sanitize_filename(book_title, allow_unicode=True)
        pdf_path = os.path.join(self.output_dir, f"{safe_title}.pdf")
        
        self.log(f"  PDF output: {pdf_path}")
        self.log(f"  Settings: page_numbers={settings['page_numbers']}, toc={settings['toc']}, "
                 f"toc_numbers={settings['toc_numbers']}, alignment={settings['page_number_alignment']}")
        compression_enabled = os.environ.get('ENABLE_IMAGE_COMPRESSION', '0') == '1'
        if compression_enabled:
            pdf_img_fmt = os.environ.get('PDF_IMAGE_FORMAT', 'jpeg').upper()
            if pdf_img_fmt == 'PNG':
                img_detail = f"format=PNG"
            else:
                img_quality = os.environ.get('IMAGE_COMPRESSION_QUALITY', '80')
                img_detail = f"format=JPEG, quality={img_quality}"
            self.log(f"  Image: compression=enabled, {img_detail}")
        else:
            self.log(f"  Image: compression=disabled")
        
        import base64
        import re
        
        # Build image lookup from images directory
        _mime_fallback = {
            '.webp': 'image/webp', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
            '.png': 'image/png', '.gif': 'image/gif', '.bmp': 'image/bmp',
            '.svg': 'image/svg+xml',
        }
        images_by_name = {}
        if os.path.isdir(self.images_dir):
            for fname in os.listdir(self.images_dir):
                fpath = os.path.join(self.images_dir, fname)
                if os.path.isfile(fpath):
                    ctype, _ = mimetypes.guess_type(fpath)
                    if not ctype:
                        ctype = _mime_fallback.get(os.path.splitext(fname)[1].lower())
                    if ctype and ctype.startswith('image/'):
                        images_by_name[fname] = (fpath, ctype)
                        # Also index without extension for flexible matching
                        base = os.path.splitext(fname)[0]
                        images_by_name[base] = (fpath, ctype)
        
        def _read_image_as_pdf_compatible(fpath, ctype):
            """Read image bytes, converting webp to JPEG/PNG since WeasyPrint doesn't support webp.
            Respects PDF_IMAGE_FORMAT, IMAGE_COMPRESSION_QUALITY, and PDF_PNG_OPTIMIZE settings."""
            if ctype == 'image/webp':
                try:
                    from PIL import Image
                    from io import BytesIO
                    pdf_fmt = os.environ.get('PDF_IMAGE_FORMAT', 'jpeg').lower()
                    with Image.open(fpath) as img:
                        buf = BytesIO()
                        if pdf_fmt == 'png':
                            # PNG: preserves transparency
                            optimize = os.environ.get('PDF_PNG_OPTIMIZE', '1') == '1'
                            compress_level = int(os.environ.get('PDF_PNG_COMPRESS_LEVEL', '6'))
                            if img.mode not in ('RGB', 'RGBA', 'L', 'LA'):
                                img = img.convert('RGBA')
                            img.save(buf, format='PNG', optimize=optimize, compress_level=compress_level)
                            return buf.getvalue(), 'image/png'
                        else:
                            # JPEG: lossy with quality control, flatten transparency
                            quality = int(os.environ.get('IMAGE_COMPRESSION_QUALITY', '80'))
                            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                                background = Image.new('RGB', img.size, (255, 255, 255))
                                if img.mode != 'RGBA':
                                    img = img.convert('RGBA')
                                background.paste(img, mask=img.split()[3])
                                img = background
                            elif img.mode != 'RGB':
                                img = img.convert('RGB')
                            img.save(buf, format='JPEG', quality=quality, optimize=True)
                            return buf.getvalue(), 'image/jpeg'
                except Exception:
                    pass
            with open(fpath, 'rb') as f:
                return f.read(), ctype
        
        def _data_uri_for_src(src_value):
            """Convert image src to data URI"""
            if not src_value or src_value.startswith('data:'):
                return None
            from urllib.parse import unquote
            raw = unquote(src_value).replace('\\', '/')
            filename = os.path.basename(raw)
            
            # Try exact match, then basename
            for key in [raw, filename, os.path.splitext(filename)[0]]:
                if key in images_by_name:
                    fpath, ctype = images_by_name[key]
                    try:
                        img_bytes, final_ctype = _read_image_as_pdf_compatible(fpath, ctype)
                        b64 = base64.b64encode(img_bytes).decode('utf-8')
                        return f"data:{final_ctype};base64,{b64}"
                    except Exception:
                        pass
            return None
        
        def _embed_images_in_html(content):
            """Replace image src attributes with data URIs"""
            def replace_attr(match):
                attr = match.group(1)
                quote = match.group(2)
                src_value = match.group(3)
                data_uri = _data_uri_for_src(src_value)
                if data_uri:
                    return f'{attr}={quote}{data_uri}{quote}'
                return match.group(0)
            return re.sub(
                r'(\b(?:src|href|xlink:href)\b)\s*=\s*([\'"])([^\'"]+)\2',
                replace_attr, content, flags=re.IGNORECASE
            )
        
        def _extract_body_content(html_content):
            """Extract just the <body> inner content from a full HTML document to avoid nested HTML."""
            body_match = re.search(r'<body[^>]*>(.*)</body>', html_content, re.DOTALL | re.IGNORECASE)
            if body_match:
                return body_match.group(1)
            return html_content
        
        # Collect CSS
        styles = ""
        if os.path.isdir(self.css_dir):
            for css_file in sorted(os.listdir(self.css_dir)):
                if css_file.endswith('.css'):
                    try:
                        with open(os.path.join(self.css_dir, css_file), 'r', encoding='utf-8') as f:
                            styles += f.read() + "\n"
                    except Exception:
                        pass
        
        # Add image styles for PDF rendering
        styles += " img { max-width: 100%; height: auto; display: block; margin: 10px auto; } "
        
        # Page number CSS
        alignment = settings['page_number_alignment']
        page_position = f'@bottom-{alignment}' if alignment != 'center' else '@bottom-center'
        if settings['page_numbers']:
            styles += f" @page {{ {page_position} {{ content: counter(page); color: rgba(0,0,0,0.4); font-size: 10pt; }} }} "
        
        # Process chapters
        documents = []
        chapter_page_map = {}
        current_page = 0
        
        # Render cover page first (TOC goes after cover)
        if cover_file:
            try:
                # Find the actual cover file in the images directory
                cover_path = None
                for fname in os.listdir(self.images_dir):
                    fpath = os.path.join(self.images_dir, fname)
                    if os.path.isfile(fpath):
                        # Match by safe name or by base name
                        if fname == cover_file or os.path.splitext(fname)[0] == os.path.splitext(cover_file)[0]:
                            cover_path = fpath
                            break
                
                if cover_path:
                    ctype, _ = mimetypes.guess_type(cover_path)
                    if not ctype:
                        ctype = _mime_fallback.get(os.path.splitext(cover_path)[1].lower())
                    if ctype and ctype.startswith('image/'):
                        cover_bytes, cover_ctype = _read_image_as_pdf_compatible(cover_path, ctype)
                        b64 = base64.b64encode(cover_bytes).decode('utf-8')
                        cover_data_uri = f"data:{cover_ctype};base64,{b64}"
                        cover_html = (
                            f'<html><head><style>'
                            f'@page {{ margin: 0; }}'
                            f'@page {{ @bottom-left {{ content: none; }} @bottom-center {{ content: none; }} @bottom-right {{ content: none; }} }}'
                            f'body {{ margin: 0; padding: 0; display: flex; justify-content: center; align-items: center; height: 100vh; }}'
                            f'img {{ max-width: 100%; max-height: 100%; object-fit: contain; }}'
                            f'</style></head><body>'
                            f'<img src="{cover_data_uri}" alt="Cover" />'
                            f'</body></html>'
                        )
                        cover_doc = WeasyHTML(string=cover_html, base_url=self.output_dir).render()
                        documents.append(cover_doc)
                        current_page += len(cover_doc.pages)
                        self.log(f"  ✅ Added cover page to PDF")
            except Exception as e:
                self.log(f"  ⚠️ Failed to add cover to PDF: {e}")
        
        # TOC dry run to estimate page count (after cover, before chapters)
        toc_page_count = 0
        toc_insert_index = len(documents)  # Insert TOC after whatever is already in documents (cover)
        if settings['toc']:
            self.log("  Calculating TOC size...")
            dummy_toc_html = self._build_pdf_toc_html(chapter_titles_info, settings, None)
            # TOC page should not have page numbers
            toc_css = "<style>@page { @bottom-left { content: none; } @bottom-center { content: none; } @bottom-right { content: none; } }</style>"
            dummy_toc_html = dummy_toc_html.replace("</head>", f"{toc_css}</head>")
            try:
                toc_doc = WeasyHTML(string=dummy_toc_html, base_url=self.output_dir).render()
                toc_page_count = len(toc_doc.pages)
            except Exception as e:
                self.log(f"  ⚠️ TOC size estimation failed: {e}")
                toc_page_count = 1
            current_page += toc_page_count
            self.log(f"  Estimated TOC: {toc_page_count} pages")
        
        # Create mapping from source filename to chapter number for TOC
        source_to_chapter = {}
        for chap_num, (title, conf, source) in chapter_titles_info.items():
            if source:
                source_to_chapter[source] = chap_num
        
        # Build all chapters as a single HTML document for continuous page numbering
        self.log(f"  Building combined chapter document ({len(html_files)} chapters)...")
        all_chapters_html = ""
        chapters_order = []  # Track chapter order for page mapping
        
        for i, html_file in enumerate(html_files):
            if self.is_stopped():
                self.log("🛑 PDF generation stopped by user")
                return
            
            file_path = os.path.join(self.output_dir, html_file)
            if not os.path.exists(file_path):
                continue
            
            # Determine chapter number
            chap_num = source_to_chapter.get(html_file, i + 1)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                content = _embed_images_in_html(content)
                
                # Extract just the body content to avoid nested <html><body> structures
                body_content = _extract_body_content(content)
                
                # Add anchor ID for TOC linking and page break before each chapter (except first)
                if i > 0:
                    body_content = f'<div style="page-break-before: always;" id="chapter-{chap_num}">{body_content}</div>'
                else:
                    body_content = f'<div id="chapter-{chap_num}">{body_content}</div>'
                
                # Add to combined HTML
                all_chapters_html += body_content
                chapters_order.append((html_file, chap_num))
                
                if (i + 1) % 10 == 0 or (i + 1) == len(html_files):
                    self.log(f"  [{i+1}/{len(html_files)}] Added to document")
                    
            except Exception as e:
                self.log(f"  ⚠️ Failed to process {html_file}: {e}")
        
        # Render all chapters as a single document
        if all_chapters_html:
            self.log("  Rendering combined chapter document...")
            combined_html = f"<html><head><style>{styles}</style></head><body>{all_chapters_html}</body></html>"
            try:
                chapters_doc = WeasyHTML(string=combined_html, base_url=self.output_dir).render()
                documents.append(chapters_doc)
                
                # Calculate page numbers for TOC based on page breaks
                # Each chapter starts on a new page after the first one
                chapter_start_page = current_page + 1  # First chapter starts here
                for idx, (html_file, chap_num) in enumerate(chapters_order):
                    chapter_page_map[html_file] = chapter_start_page
                    chapter_page_map[chap_num] = chapter_start_page
                    # Increment page for next chapter (assuming each chapter starts on new page)
                    if idx < len(chapters_order) - 1:  # Not the last chapter
                        chapter_start_page += 1  # Minimum increment, actual pages may vary
                
                current_page += len(chapters_doc.pages)
                self.log(f"  Combined document: {len(chapters_doc.pages)} pages")
            except Exception as e:
                self.log(f"  ⚠️ Failed to render combined document: {e}")
        
        if not documents:
            self.log("⚠️ No chapters rendered for PDF")
            return
        
        # Generate final TOC with real page numbers
        if settings['toc'] and chapter_titles_info:
            self.log("  Generating TOC with page numbers...")
            real_toc_html = self._build_pdf_toc_html(chapter_titles_info, settings, chapter_page_map)
            # TOC page should not have page numbers
            toc_css = "<style>@page { @bottom-left { content: none; } @bottom-center { content: none; } @bottom-right { content: none; } }</style>"
            real_toc_html = real_toc_html.replace("</head>", f"{toc_css}</head>")
            
            try:
                toc_doc = WeasyHTML(string=real_toc_html, base_url=self.output_dir).render()
                documents.insert(toc_insert_index, toc_doc)
            except Exception as e:
                self.log(f"  ⚠️ TOC generation failed: {e}")
        
        # Merge and write PDF
        self.log("  Merging all pages...")
        all_pages = [page for doc in documents for page in doc.pages]
        self.log(f"  Total pages: {len(all_pages)}")
        self.log("  Writing PDF to disk...")
        
        documents[0].copy(all_pages).write_pdf(pdf_path)
        
        elapsed = _time.time() - start_time
        if os.path.exists(pdf_path):
            file_size = os.path.getsize(pdf_path)
            self.log(f"✅ PDF created: {pdf_path}")
            self.log(f"📊 PDF size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
            self.log(f"⏱️ PDF generation took {elapsed:.1f}s")
        else:
            self.log("❌ PDF file was not created")

    def _build_pdf_toc_html(self, chapter_titles_info: Dict[int, Tuple[str, float, str]],
                            settings: dict, chapter_page_map: Optional[Dict[str, int]]) -> str:
        """Build HTML for PDF table of contents with clickable links and page numbers"""
        import html as _html
        
        toc_html = (
            '<html><head><style>'
            'body { font-family: serif; margin: 40px; }'
            'h1 { text-align: center; margin-bottom: 30px; }'
            'ul { list-style-type: none; padding: 0; margin: 0; }'
            'li { margin-bottom: 6px; padding: 4px 0; border-bottom: 1px dotted #ccc; }'
            'a { text-decoration: none; color: #333; display: flex; justify-content: space-between; }'
            'a:hover { color: #0066cc; text-decoration: underline; }'
            '.title { flex: 1; }'
            '.page-num { font-weight: bold; min-width: 40px; text-align: right; margin-left: 10px; }'
            '</style></head><body>'
            '<h1>Table of Contents</h1><ul>'
        )
        
        for chap_num in sorted(chapter_titles_info.keys()):
            title, confidence, source = chapter_titles_info[chap_num]
            safe_title = _html.escape(title)
            
            # Get page number - try chapter number key directly first
            page_num = ""
            if settings.get('toc_numbers') and chapter_page_map:
                # Try chapter number directly (stored as integer key)
                page_num = chapter_page_map.get(chap_num, "")
                
                # Fallback to source filename
                if not page_num and source:
                    page_num = chapter_page_map.get(source, "")
                
                page_num = str(page_num) if page_num else ""
            
            page_span = f'<span class="page-num">{page_num}</span>' if page_num else ''
            
            # Create clickable link
            toc_html += f'<li><a href="#chapter-{chap_num}"><span class="title">{safe_title}</span>{page_span}</a></li>'
        
        toc_html += '</ul></body></html>'
        return toc_html


# Main entry point
def compile_epub(base_dir: str, log_callback: Optional[Callable] = None):
    """Compile translated HTML files into EPUB"""
    # Reset stop flag for new compilation
    set_stop_flag(False)
    
    compiler = EPUBCompiler(base_dir, log_callback)
    compiler.compile()


# Legacy alias
fallback_compile_epub = compile_epub


if __name__ == "__main__":
    from shutdown_utils import run_cli_main
    def _main():
        if len(sys.argv) < 2:
            print("Usage: python epub_converter.py <directory_path>")
            return 1
        
        directory_path = sys.argv[1]
        
        try:
            compile_epub(directory_path)
        except Exception as e:
            print(f"Error: {e}")
            return 1
        return 0
    run_cli_main(_main)
