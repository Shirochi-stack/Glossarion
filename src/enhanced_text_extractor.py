#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Text Extractor Module with CJK Support
Provides superior text extraction from HTML with proper Unicode handling
Optimized for Korean, Japanese, and Chinese content extraction
"""

import os
import re
import html
import unicodedata
from typing import Tuple, Optional
import chardet

# BEAUTIFUL SOUP IMPORT MONKEY FIX - Import BeautifulSoup BEFORE html2text
# This prevents certain parser initialization issues
try:
    from bs4 import BeautifulSoup
    # Force BeautifulSoup to initialize its parsers
    _ = BeautifulSoup("", 'html.parser')
except ImportError:
    BeautifulSoup = None
    raise ImportError("BeautifulSoup is required. Install with: pip install beautifulsoup4")

# Now import html2text AFTER BeautifulSoup
try:
    import html2text
except ImportError:
    html2text = None
    raise ImportError("html2text is required. Install with: pip install html2text")


class EnhancedTextExtractor:
    """Enhanced text extraction with proper Unicode and CJK handling"""
    
    # Unicode preservation mappings
    UNICODE_QUOTES = {
        # Western quotes
        '&ldquo;': '\u201c',  # Left double quotation mark
        '&rdquo;': '\u201d',  # Right double quotation mark
        '&lsquo;': '\u2018',  # Left single quotation mark
        '&rsquo;': '\u2019',  # Right single quotation mark
        '&quot;': '"',        # Standard double quote
        '&apos;': "'",        # Standard apostrophe
        
        # CJK quotes and punctuation
        '&#12300;': '「',  # Japanese left corner bracket
        '&#12301;': '」',  # Japanese right corner bracket
        '&#12302;': '『',  # Japanese left white corner bracket
        '&#12303;': '』',  # Japanese right white corner bracket
        '&#65288;': '（',  # Fullwidth left parenthesis
        '&#65289;': '）',  # Fullwidth right parenthesis
        '&#12304;': '【',  # Left black lenticular bracket
        '&#12305;': '】',  # Right black lenticular bracket
        '&#12298;': '《',  # Left double angle bracket
        '&#12299;': '》',  # Right double angle bracket
        '&#65307;': '；',  # Fullwidth semicolon
        '&#65306;': '：',  # Fullwidth colon
        '&#12290;': '。',  # Ideographic full stop
        '&#65311;': '？',  # Fullwidth question mark
        '&#65281;': '！',  # Fullwidth exclamation mark
        '&#12289;': '、',  # Ideographic comma
        
        # Numeric entities
        '&#8220;': '\u201c',  # Left double quote (numeric)
        '&#8221;': '\u201d',  # Right double quote (numeric)
        '&#8216;': '\u2018',  # Left single quote (numeric)
        '&#8217;': '\u2019',  # Right single quote (numeric)
        
        # Common CJK entities
        '&hellip;': '…',     # Horizontal ellipsis
        '&mdash;': '—',      # Em dash
        '&ndash;': '–',      # En dash
        '&nbsp;': '\u00A0',  # Non-breaking space
    }
    
    # CJK-specific punctuation to preserve
    CJK_PUNCTUATION = {
        '。', '、', '！', '？', '…', '—', '～', '・',
        '「', '」', '『', '』', '（', '）', '【', '】',
        '《', '》', '〈', '〉', '〔', '〕', '［', '］',
        '：', '；', '"', '"', ''', ''',
        '，', '．', '？', '！', '：', '；',
        '"', '"', '‚', '„', '«', '»',
    }
    
    # Quote protection markers
    QUOTE_MARKERS = {
        '"': '␥',   # Opening double quote marker
        '"': '␦',   # Closing double quote marker  
        '"': '␦',   # Alternative closing quote
        "'": '␣',   # Opening single quote marker
        "'": '␤',   # Closing single quote marker
        "'": '␤',   # Alternative closing quote
    }
    
    # Angle bracket protection markers for invalid tags
    ANGLE_BRACKET_MARKERS = {
        '<': '‹',   # Single left-pointing angle quotation mark (U+2039)
        '>': '›',   # Single right-pointing angle quotation mark (U+203A)
    }
    
    
    def __init__(self, filtering_mode: str = "smart", preserve_structure: bool = True):
        """Initialize the enhanced text extractor"""
        if not html2text:
            raise ImportError("html2text is required for enhanced extraction")
        
        if not BeautifulSoup:
            raise ImportError("BeautifulSoup is required for enhanced extraction")
            
        self.filtering_mode = filtering_mode
        self.preserve_structure = preserve_structure
        self.h2t = None
        self.detected_language = None
        
        self._configure_html2text()
    
    def _detect_encoding(self, content: bytes) -> str:
        """Detect the encoding of the content"""
        try:
            # Try chardet detection
            detected = chardet.detect(content)
            if detected['confidence'] > 0.7:
                return detected['encoding']
        except Exception:
            pass
        
        # Try common CJK encodings in order
        for encoding in ['utf-8', 'gb2312', 'gbk', 'gb18030', 'big5', 'shift_jis', 'euc-kr', 'euc-jp']:
            try:
                content.decode(encoding)
                return encoding
            except Exception:
                continue
        
        return 'utf-8'  # Default fallback
    
    def _detect_content_language(self, text: str) -> str:
        """Detect the primary language of content"""
        if not text:
            return 'unknown'
        
        # Take a sample of the text
        sample = text[:5000]
        
        # Count characters by script
        korean_chars = sum(1 for char in sample if 0xAC00 <= ord(char) <= 0xD7AF)
        japanese_kana = sum(1 for char in sample if (0x3040 <= ord(char) <= 0x309F) or (0x30A0 <= ord(char) <= 0x30FF))
        chinese_chars = sum(1 for char in sample if 0x4E00 <= ord(char) <= 0x9FFF)
        latin_chars = sum(1 for char in sample if 0x0041 <= ord(char) <= 0x007A)
        
        # Determine primary language
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
    
    def _configure_html2text(self):
        """Configure html2text with optimal Unicode and CJK settings"""
        self.h2t = html2text.HTML2Text()
        
        # Core settings for Unicode preservation
        self.h2t.unicode_snob = True
        self.h2t.escape_snob = os.getenv('HTML2TEXT_ESCAPE_SNOB', '0') == '1'
        self.h2t.use_automatic_links = False
        
        # Layout settings
        self.h2t.body_width = 0
        # Check environment variable for single line break setting
        single_line_break = os.getenv('ENHANCED_SINGLE_LINE_BREAK', '0') == '1'
        self.h2t.single_line_break = single_line_break
        
        # Content filtering
        self.h2t.ignore_links = False
        self.h2t.ignore_images = False
        self.h2t.ignore_anchors = False
        self.h2t.skip_internal_links = False
        self.h2t.ignore_tables = False
        
        # Image handling - CRITICAL: Force html2text to preserve img tags as HTML
        self.h2t.images_as_html = True  # Keep images as <img> tags instead of ![]()
        self.h2t.images_to_alt = False  # Don't convert to alt text only
        self.h2t.images_with_size = True  # Include width/height attributes
        
        # Additional settings
        self.h2t.wrap_links = False
        self.h2t.wrap_list_items = False
        self.h2t.protect_links = True
        
        # Structure preservation settings
        if self.preserve_structure:
            self.h2t.bypass_tables = False
            self.h2t.ignore_emphasis = False
            self.h2t.mark_code = True
            self.h2t.ul_item_mark = '•'
        else:
            self.h2t.bypass_tables = True
            self.h2t.ignore_emphasis = True
            self.h2t.mark_code = False
    
    def _decode_entities(self, text: str) -> str:
        """Decode HTML entities to Unicode characters with CJK support"""
        if not text:
            return text
        
        # First pass: Apply known CJK-aware replacements
        for entity, unicode_char in self.UNICODE_QUOTES.items():
            text = text.replace(entity, unicode_char)
        
        # Second pass: standard HTML unescape
        text = html.unescape(text)
        
        # Third pass: handle numeric entities
        def decode_decimal(match):
            try:
                code = int(match.group(1))
                if code < 0x110000:
                    return chr(code)
            except Exception:
                pass
            return match.group(0)
        
        def decode_hex(match):
            try:
                code = int(match.group(1), 16)
                if code < 0x110000:
                    return chr(code)
            except Exception:
                pass
            return match.group(0)
        
        text = re.sub(r'&#(\d+);?', decode_decimal, text)
        text = re.sub(r'&#x([0-9a-fA-F]+);?', decode_hex, text)
        
        # Fourth pass: handle special CJK entities
        cjk_special_entities = {
            '&lang;': '〈', '&rang;': '〉',
            '&lceil;': '⌈', '&rceil;': '⌉',
            '&lfloor;': '⌊', '&rfloor;': '⌋',
        }
        
        for entity, char in cjk_special_entities.items():
            text = text.replace(entity, char)
        
        return text
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode with CJK awareness"""
        if self.detected_language in ['korean', 'japanese', 'chinese']:
            return text
        else:
            return unicodedata.normalize('NFC', text)
    
    def _protect_quotes(self, text: str) -> str:
        """Protect quotes by replacing with special markers"""
        for original, marker in self.QUOTE_MARKERS.items():
            text = text.replace(original, marker)
        return text
    
    def _restore_quotes(self, text: str) -> str:
        """Restore quotes from special markers"""
        for original, marker in self.QUOTE_MARKERS.items():
            text = text.replace(marker, original)
        return text
    
    def _protect_cjk_angle_brackets(self, text: str) -> str:
        """Protect angle brackets containing CJK text from being treated as HTML tags."""
        import re
        # Pattern to match angle brackets containing CJK characters
        cjk_pattern = r'[\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff\uac00-\ud7af]'
        bracket_pattern = rf'<([^<>]*{cjk_pattern}[^<>]*)>'
        
        def replace_brackets(match):
            content = match.group(1)
            # Use special Unicode markers instead of HTML entities
            return f"‹{content}›"
        
        return re.sub(bracket_pattern, replace_brackets, text)
    
    def _restore_cjk_angle_brackets(self, text: str) -> str:
        """Restore angle brackets from special markers."""
        # Convert markers back to HTML entities for proper display
        text = text.replace('‹', '&lt;')
        text = text.replace('›', '&gt;')
        return text
    
    def _protect_empty_attr_tags(self, text: str) -> str:
        """Protect tags with empty attributes from being stripped by html2text"""
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
        def _repl_pair(m):
            tagname = m.group(1)
            if tagname.lower() in known_tags:
                return m.group(0)
            attrname = m.group(2)
            content = m.group(3)
            return f"&lt;{tagname} {attrname}&gt;{content}"
        
        # Match <Tag Attr=""></Tag> (allow whitespace and content)
        text = re.sub(r'<([a-zA-Z0-9_\-]+)\s+([a-zA-Z0-9_\-]+)=\"\"\s*>(.*?)</\1>', _repl_pair, text, flags=re.DOTALL)
        
        return text
    
    def _preprocess_html_for_quotes(self, html_content: str) -> str:
        """Pre-process HTML to protect quotes from conversion"""
        def protect_quotes_in_text(match):
            text = match.group(1)
            return f'>{self._protect_quotes(text)}<'
        
        # Apply to all text between tags
        html_content = re.sub(r'>([^<]+)<', protect_quotes_in_text, html_content)
        return html_content
    
    def _protect_quotes_in_soup(self, soup: BeautifulSoup) -> None:
        """Protect quotes in BeautifulSoup object before processing"""
        for element in soup.find_all(string=True):
            if element.parent.name not in ['script', 'style', 'noscript']:
                original_text = str(element)
                protected_text = self._protect_quotes(original_text)
                element.replace_with(protected_text)
    
    def _minimal_parser_fix(self, html_content: str) -> str:
        """Apply minimal fixes only for parser errors"""
        # Fix tags with ="" pattern
        html_content = re.sub(r'<[^>]*?=\s*""\s*[^>]*?>', '', html_content)
        
        # Fix malformed closing tags
        html_content = re.sub(r'</\s+(\w+)>', r'</\1>', html_content)
        html_content = re.sub(r'</\s*>', '', html_content)
        html_content = re.sub(r'<//+(\w+)>', r'</\1>', html_content)
        
        # Fix orphaned brackets
        html_content = re.sub(r'<(?![a-zA-Z/!?])', '&lt;', html_content)
        html_content = re.sub(r'(?<![a-zA-Z0-9"/])>', '&gt;', html_content)
        
        # Fix unclosed tags at the end
        if html_content.rstrip().endswith('<'):
            html_content = html_content.rstrip()[:-1]
        
        # Remove nested opening brackets
        html_content = re.sub(r'<[^>]*?<[^>]*?>', '', html_content)
        
        return html_content
    
    def _clean_text_cjk_aware(self, text: str, preserve_structure: bool) -> str:
        """Clean extracted text with CJK awareness"""
        if not preserve_structure and self.detected_language not in ['korean', 'japanese', 'chinese']:
            # Only do aggressive cleanup for non-CJK text
            text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
            text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
            text = re.sub(r'\*(.*?)\*', r'\1', text)
            text = re.sub(r'__(.*?)__', r'\1', text)
            text = re.sub(r'_(.*?)_', r'\1', text)
            text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
            text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', '', text)
            text = re.sub(r'`([^`]+)`', r'\1', text)
            text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)
            text = re.sub(r'^[-*+]\s+', '', text, flags=re.MULTILINE)
            text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
            text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)
            text = re.sub(r'^[-_*]{3,}$', '', text, flags=re.MULTILINE)
        
        # Clean whitespace
        if self.detected_language in ['korean', 'japanese', 'chinese']:
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = re.sub(r'[ ]{3,}', '  ', text)
        else:
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = re.sub(r' {2,}', ' ', text)
        
        # Remove invisible characters
        invisible_chars = ['\u200b', '\u200c', '\u200d', '\ufeff', '\u2060']
        for char in invisible_chars:
            text = text.replace(char, '')
        
        return text.strip()
    
    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract chapter title from various sources"""
        # Try title tag first
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
            title = self._decode_entities(title)
            return title
        
        # Try headers in order
        for header_tag in ['h1', 'h2', 'h3', 'h4']:
            headers = soup.find_all(header_tag)
            for header in headers:
                title = header.get_text(strip=True)
                if title:
                    title = self._decode_entities(title)
                    if self._is_chapter_title(title):
                        return title
        
        return None
    
    def _is_chapter_title(self, text: str) -> bool:
        """Check if text looks like a chapter title"""
        if not text or len(text) > 200:
            return False
        
        # Common chapter patterns
        patterns = [
            r'第.{1,10}[章回話话]',
            r'Chapter\s+\d+',
            r'제\s*\d+\s*화',
            r'第\d+話',
            r'\d+\s*화',
            r'EP\.?\s*\d+',
            r'Part\s+\d+',
        ]
        
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        # Check if it's short and doesn't contain too much punctuation
        if len(text) < 100:
            punct_count = sum(1 for c in text if c in '.,;:!?。、！？')
            if punct_count < len(text) * 0.2:
                return True
        
        return False
    
    def _extract_body_content(self, soup: BeautifulSoup, full_html: str) -> str:
        """Extract body content while preserving Unicode"""
        # Remove script and style elements first
        for element in soup(['script', 'style', 'noscript']):
            element.decompose()
        
        if soup.body:
            return str(soup.body)
        else:
            return str(soup)
    
    def extract_chapter_content(self, html_content: str, extraction_mode: str = None) -> Tuple[str, str, Optional[str]]:
        """Extract chapter content with proper Unicode and CJK handling"""
        try:
            # Use instance filtering_mode if not overridden
            if extraction_mode is None:
                extraction_mode = self.filtering_mode
            
            # Handle encoding if content is bytes
            if isinstance(html_content, bytes):
                encoding = self._detect_encoding(html_content)
                html_content = html_content.decode(encoding, errors='replace')
            
            # Pre-process HTML to protect quotes
            html_content = self._preprocess_html_for_quotes(html_content)
            
            # Pre-process HTML to decode all entities
            html_content = self._decode_entities(html_content)
            
            # Detect language early (don't log - already logged in extraction summary)
            self.detected_language = self._detect_content_language(html_content)
            
            # Parse with BeautifulSoup
            parser = 'html.parser'
            if self.detected_language in ['korean', 'japanese', 'chinese']:
                # For CJK content, lxml might handle encoding better if available
                try:
                    import lxml
                    parser = 'lxml'
                except ImportError:
                    pass
            
            soup = BeautifulSoup(html_content, parser)
            
            # Protect quotes before any processing
            self._protect_quotes_in_soup(soup)
            
            # Extract title
            chapter_title = self._extract_title(soup)
            
            # Respect GUI toggles to exclude headers/titles BEFORE conversion
            try:
                batch_translate_active = os.getenv('BATCH_TRANSLATE_HEADERS', '0') == '1'
                ignore_title_tag = os.getenv('IGNORE_TITLE', '0') == '1' and batch_translate_active
                ignore_header_tags = os.getenv('IGNORE_HEADER', '0') == '1' and batch_translate_active
                remove_duplicate_h1_p = os.getenv('REMOVE_DUPLICATE_H1_P', '0') == '1'
                
                if ignore_title_tag and soup.title:
                    # Remove <title> so it isn't included when using full extraction
                    soup.title.decompose()
                
                if ignore_header_tags:
                    # Remove visible headers from body prior to conversion
                    for tag_name in ['h1', 'h2', 'h3']:
                        for hdr in soup.find_all(tag_name):
                            hdr.decompose()
                
                # Remove duplicate H1+P pairs (where P immediately follows or precedes H1 with same text)
                if remove_duplicate_h1_p:
                    for h1_tag in soup.find_all('h1'):
                        # Skip split marker H1 tags
                        h1_id = h1_tag.get('id', '')
                        if h1_id and h1_id.startswith('split-'):
                            continue
                        h1_text = h1_tag.get_text(strip=True)
                        if 'SPLIT MARKER' in h1_text:
                            continue
                        
                        # Check next sibling (P after H1)
                        next_sibling = h1_tag.find_next_sibling()
                        if next_sibling and next_sibling.name == 'p':
                            p_text = next_sibling.get_text(strip=True)
                            if h1_text == p_text:
                                next_sibling.decompose()
                                continue
                        
                        # Check previous sibling (P before H1)
                        prev_sibling = h1_tag.find_previous_sibling()
                        if prev_sibling and prev_sibling.name == 'p':
                            p_text = prev_sibling.get_text(strip=True)
                            if h1_text == p_text:
                                prev_sibling.decompose()
            except Exception:
                # Non-fatal – proceed with original soup if anything goes wrong
                pass
            
            # Determine content to convert (after removals)
            if extraction_mode == "full":
                content_to_convert = str(soup)
            else:
                content_to_convert = self._extract_body_content(soup, html_content)
            
            # Convert using html2text
            content_to_convert = self._decode_entities(content_to_convert)
            
            # Protect CJK text in angle brackets using special markers
            # that html2text won't interpret as HTML
            content_to_convert = self._protect_cjk_angle_brackets(content_to_convert)
            
            # Apply Empty Attribute Tag Fix if enabled
            if os.getenv('FIX_EMPTY_ATTR_TAGS_EXTRACT', '0') == '1':
                content_to_convert = self._protect_empty_attr_tags(content_to_convert)
            
            # Convert to text with error handling
            try:
                clean_text = self.h2t.handle(content_to_convert)
            except (AssertionError, UnboundLocalError) as e:
                error_msg = str(e)
                if "cannot access local variable" in error_msg or "we should not get here!" in error_msg or "unexpected call to parse_endtag" in error_msg or "unexpected call to parse_starttag" in error_msg:
                    print(f"⚠️ html2text encountered malformed HTML: {error_msg}")
                    print(f"⚠️ Applying minimal fixes...")
                    # Apply minimal fixes
                    content_to_convert = self._minimal_parser_fix(content_to_convert)
                    try:
                        clean_text = self.h2t.handle(content_to_convert)
                        print(f"✅ Successfully processed after minimal fixes")
                    except Exception as e2:
                        print(f"⚠️ html2text still failing: {e2}")
                        # Last resort fallback
                        clean_text = soup.get_text(separator='\n', strip=True)
                        print(f"✅ Used BeautifulSoup fallback")
                else:
                    # Re-raise if it's a different error
                    raise
            except Exception as e:
                print(f"⚠️ Unexpected error in html2text: {e}")
                # Fallback to BeautifulSoup
                clean_text = soup.get_text(separator='\n', strip=True)
            
            # Normalize only if appropriate
            clean_text = self._normalize_unicode(clean_text)
            
            # Clean based on settings and language
            clean_text = self._clean_text_cjk_aware(clean_text, self.preserve_structure)
            
            # Restore protected quotes
            clean_text = self._restore_quotes(clean_text)
            
            # Restore CJK angle brackets as HTML entities
            clean_text = self._restore_cjk_angle_brackets(clean_text)
            
            # For enhanced mode, both display and translation content are the same
            return clean_text, clean_text, chapter_title
                
        except Exception as e:
            print(f"❌ Enhanced extraction failed: {e}")
            raise


# Test function
def test_cjk_preservation():
    """Test that CJK characters and quotes are properly preserved"""
    test_cases = [
        # Korean test with quotes
        '''<html>
        <head><title>제국의 붉은 사신</title></head>
        <body>
            <p>"왜 이러는 겁니까? 우리가 무슨 잘못을 했다고!"</p>
            <p>"......"</p>
            <p>"한 번만 살려주시오! 가족을 지키려면 어쩔 수 없었소!"</p>
            <p>"응애! 응애! 응애!"</p>
            <p>"미안하구나. 모든 죄는 내가 짊어지고 사마."</p>
        </body>
        </html>'''
        
        # Japanese test with quotes
        '''<html>
        <head><title>第1話：始まり</title></head>
        <body>
            <h1>第1話：始まり</h1>
            <p>「こんにちは！これは日本語のテストです。」</p>
            <p>彼は言った。「これで全部ですか？」</p>
            <p>「はい、そうです」と答えた。</p>
        </body>
        </html>''',
        
        # Chinese test with quotes
        '''<html>
        <head><title>第一章：开始</title></head>
        <body>
            <h1>第一章：开始</h1>
            <p>"你好！这是中文测试。"</p>
            <p>他说："这就是全部吗？"</p>
            <p>"是的，"她回答道。</p>
        </body>
        </html>''',
    ]
    
    extractor = EnhancedTextExtractor()
    
    print("=== CJK and Quote Preservation Test ===\n")
    
    for i, test_html in enumerate(test_cases, 1):
        print(f"--- Test Case {i} ---")
        try:
            content, _, title = extractor.extract_chapter_content(test_html)
            
            print(f"Title: {title}")
            print(f"Content:\n{content}\n")
            
            # Check for quotes preservation
            quote_checks = [
                ('"', 'Western double quotes'),
                ('「', 'Japanese left bracket'),
                ('」', 'Japanese right bracket'),
                ('“', 'Chinese double quote'),
            ]
            
            print("Quote preservation check:")
            quote_found = False
            
            for quote_char, desc in quote_checks:
                if quote_char in content:
                    print(f"  ✓ Found {desc}: {quote_char}")
                    quote_found = True
            
            if not quote_found:
                print("  ❌ No quotes found!")
            else:
                print("  ✅ Quotes preserved successfully!")
            
            # Check for image tag preservation (html2text now preserves them natively)
            img_count = content.count('<img')
            if img_count > 0:
                print(f"  ✓ Found {img_count} HTML img tags (preserved natively by html2text)")
                print("  ✅ Image tags preserved successfully!")
            else:
                print("  ℹ️ No images in this test case")
                
        except Exception as e:
            print(f"Error processing test case {i}: {e}")
        
        print("-" * 50 + "\n")


if __name__ == "__main__":
    from shutdown_utils import run_cli_main
    def _main():
        test_cjk_preservation()
        return 0
    run_cli_main(_main)
