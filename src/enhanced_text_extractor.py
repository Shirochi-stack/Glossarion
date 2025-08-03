#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Text Extractor Module with CJK Support
Provides superior text extraction from HTML with proper Unicode handling
Optimized for Korean, Japanese, and Chinese content extraction
Refactored to focus on html2text without BeautifulSoup fallback
"""

import re
import html
import unicodedata
from typing import Tuple, Optional
import chardet

try:
    import html2text
except ImportError:
    html2text = None
    raise ImportError("⚠️ html2text is required. Install with: pip install html2text")

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None
    raise ImportError("⚠️ BeautifulSoup is required. Install with: pip install beautifulsoup4")


class EnhancedTextExtractor:
    """Enhanced text extraction with proper Unicode and CJK handling"""
    
    # Unicode preservation mappings - Extended for CJK
    UNICODE_QUOTES = {
        # Western quotes
        '&ldquo;': '\u201c',  # " Left double quotation mark
        '&rdquo;': '\u201d',  # " Right double quotation mark
        '&lsquo;': '\u2018',  # ' Left single quotation mark
        '&rsquo;': '\u2019',  # ' Right single quotation mark
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
        '&#8220;': '\u201c',  # " Left double quote (numeric)
        '&#8221;': '\u201d',  # " Right double quote (numeric)
        '&#8216;': '\u2018',  # ' Left single quote (numeric)
        '&#8217;': '\u2019',  # ' Right single quote (numeric)
        
        # Common CJK entities
        '&hellip;': '…',     # Horizontal ellipsis
        '&mdash;': '—',      # Em dash
        '&ndash;': '–',      # En dash
        '&nbsp;': '\u00A0',  # Non-breaking space
    }
    
    # CJK-specific punctuation to preserve
    CJK_PUNCTUATION = {
        '。', '、', '！', '？', '…', '—', '～', '・',  # Japanese
        '「', '」', '『', '』', '（', '）', '【', '】',  # Japanese brackets
        '《', '》', '〈', '〉', '〔', '〕', '［', '］',  # CJK brackets
        '：', '；', '"', '"', ''', ''',  # Chinese punctuation
        '，', '．', '？', '！', '：', '；',  # Fullwidth punctuation
        '"', '"', '‚', '„', '«', '»',  # Additional quote marks
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
    
    def __init__(self, filtering_mode: str = "smart", preserve_structure: bool = True):
        """Initialize the enhanced text extractor
        
        Args:
            filtering_mode: Extraction filtering mode ('smart', 'comprehensive', 'full')
            preserve_structure: Whether to preserve markdown formatting
        """
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
        except:
            pass
        
        # Try common CJK encodings in order
        for encoding in ['utf-8', 'gb2312', 'gbk', 'gb18030', 'big5', 'shift_jis', 'euc-kr', 'euc-jp']:
            try:
                content.decode(encoding)
                return encoding
            except:
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
        self.h2t.unicode_snob = False      # CRITICAL: Use Unicode characters
        self.h2t.escape_snob = False       # CRITICAL: Don't escape special chars
        self.h2t.use_automatic_links = False
        
        # Layout settings - adjusted for CJK
        self.h2t.body_width = 0          # Don't wrap lines (important for CJK)
        self.h2t.single_line_break = True
        
        # Content filtering
        self.h2t.ignore_links = False     # Preserve links for context
        self.h2t.ignore_images = False
        self.h2t.ignore_anchors = True
        self.h2t.skip_internal_links = True
        
        # Additional Unicode-friendly settings
        self.h2t.wrap_links = False
        self.h2t.wrap_list_items = False
        self.h2t.protect_links = True
        
        # Settings to prevent quote issues
        self.h2t.pad_tables = False
        self.h2t.dash_unordered_list = False
        
        # Try to prevent quote-to-comma conversion
        self.h2t.emphasis_mark = '_'
        self.h2t.strong_mark = '__'
        
        # Disable any quote style conversion
        self.h2t.open_quote = '"'
        self.h2t.close_quote = '"'
        
        # Disable quote markers entirely
        self.h2t.open_quote = ''
        self.h2t.close_quote = ''
        # Structure preservation settings
        if self.preserve_structure:
            self.h2t.bypass_tables = False
            self.h2t.ignore_emphasis = False
            self.h2t.mark_code = True
            self.h2t.ul_item_mark = '•'  # Use bullet for better CJK compatibility
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
        
        # Third pass: handle numeric entities (including CJK ranges)
        text = re.sub(r'&#(\d+);?', lambda m: chr(int(m.group(1))) if int(m.group(1)) < 0x110000 else m.group(0), text)
        text = re.sub(r'&#x([0-9a-fA-F]+);?', lambda m: chr(int(m.group(1), 16)) if int(m.group(1), 16) < 0x110000 else m.group(0), text)
        
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
        # For CJK text, avoid aggressive normalization
        if self.detected_language in ['korean', 'japanese', 'chinese']:
            # Skip normalization for CJK to preserve character distinctions
            return text
        else:
            # For non-CJK text, use NFC normalization
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
        for element in soup.find_all(text=True):
            if element.parent.name not in ['script', 'style', 'noscript']:
                original_text = str(element)
                protected_text = self._protect_quotes(original_text)
                element.replace_with(protected_text)

    
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
        
        # Clean whitespace - but be careful with CJK
        if self.detected_language in ['korean', 'japanese', 'chinese']:
            text = re.sub(r'\n{3,}', '\n\n', text)  # Only reduce 3+ newlines
            text = re.sub(r'[ ]{3,}', '  ', text)  # Only reduce 3+ spaces to 2
        else:
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = re.sub(r' {2,}', ' ', text)
        
        # Remove zero-width spaces and other invisible characters
        invisible_chars = ['\u200b', '\u200c', '\u200d', '\ufeff', '\u2060']
        for char in invisible_chars:
            text = text.replace(char, '')
        
        return text.strip()
    
    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract chapter title from various sources with CJK support"""
        # Try <title> tag first
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
        
        # Common chapter patterns in different languages
        patterns = [
            r'第.{1,10}[章回話话]',  # Chinese/Japanese chapter markers
            r'Chapter\s+\d+',        # English
            r'제\s*\d+\s*화',        # Korean
            r'第\d+話',              # Japanese
            r'\d+\s*화',             # Korean simple
            r'EP\.?\s*\d+',          # Episode markers
            r'Part\s+\d+',           # Part markers
        ]
        
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        # Check if it's short and doesn't contain too much punctuation
        punct_ratio = sum(1 for c in text if c in '.,;:!?。、！？') / len(text)
        if len(text) < 100 and punct_ratio < 0.2:
            return True
        
        return False
    
    def _extract_body_content(self, soup: BeautifulSoup, full_html: str) -> str:
        """Extract body content while preserving Unicode and CJK text"""
        # Remove script and style elements first
        for element in soup(['script', 'style', 'noscript']):
            element.decompose()
        
        if soup.body:
            return str(soup.body)
        else:
            return str(soup)
    
    def extract_chapter_content(self, html_content: str, extraction_mode: str = None) -> Tuple[str, str, Optional[str]]:
        """Extract chapter content with proper Unicode and CJK handling
        
        Args:
            html_content: Raw HTML content
            extraction_mode: Extraction mode override (defaults to instance filtering_mode)
            
        Returns:
            Tuple of (content_for_display, content_for_translation, chapter_title)
        """
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
            
            # Detect language early
            self.detected_language = self._detect_content_language(html_content)
            print(f"🌐 Detected language: {self.detected_language}")
            
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
            
            # Determine content to convert
            if extraction_mode == "full":
                content_to_convert = html_content
            else:
                content_to_convert = self._extract_body_content(soup, html_content)
            
            # Convert using html2text
            # Ensure content is properly decoded before conversion
            content_to_convert = self._decode_entities(content_to_convert)
            
            # Convert to text
            clean_text = self.h2t.handle(content_to_convert)
            
            # Fix quote conversion issues
            #clean_text = self._fix_quote_conversion_issues(clean_text)
            
            # Normalize only if appropriate
            clean_text = self._normalize_unicode(clean_text)
            
            # Clean based on settings and language
            clean_text = self._clean_text_cjk_aware(clean_text, self.preserve_structure)
            
            # Restore protected quotes
            clean_text = self._restore_quotes(clean_text)
            
            # For enhanced mode, both display and translation content are the same clean text
            return clean_text, clean_text, chapter_title
                
        except Exception as e:
            print(f"❌ Enhanced extraction failed: {e}")
            raise


# Test function to verify Unicode and CJK handling
def test_cjk_preservation():
    """Test that CJK characters and quotes are properly preserved"""
    test_cases = [
        # Korean test with quotes (from user's example)
        '''<html>
        <head><title>제국의 붉은 사신</title></head>
        <body>
            <p>"왜 이러는 겁니까? 우리가 무슨 잘못을 했다고!"</p>
            <p>"......"</p>
            <p>"한 번만 살려주시오! 가족을 지키려면 어쩔 수 없었소!"</p>
            <p>"응애! 응애! 응애!"</p>
            <p>"미안하구나. 모든 죄는 내가 짊어지고 사마."</p>
        </body>
        </html>''',
        
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
        
        # Mixed content with various quote styles
        '''<html>
        <body>
            <h1>Mixed Quotes Test</h1>
            <p>"English quotes here"</p>
            <p>「日本語の引用」</p>
            <p>"中文引号"</p>
            <p>"한국어 인용문"</p>
        </body>
        </html>'''
    ]
    
    extractor = EnhancedTextExtractor()
    
    print("=== CJK and Quote Preservation Test ===\n")
    
    for i, test_html in enumerate(test_cases, 1):
        print(f"--- Test Case {i} ---")
        content, _, title = extractor.extract_chapter_content(test_html)
        
        print(f"Title: {title}")
        print(f"Content:\n{content}\n")
        
        # Check for quotes preservation
        quote_checks = [
            ('"', 'Western double quotes'),
            ('「', 'Japanese left bracket'),
            ('」', 'Japanese right bracket'),
            ('"', 'Chinese double quote'),
        ]
        
        print("Quote preservation check:")
        quote_found = False
        comma_issue = False
        
        for quote_char, desc in quote_checks:
            if quote_char in content:
                print(f"  ✓ Found {desc}: {quote_char}")
                quote_found = True
        
        # Check if any quotes were converted to commas
        lines = content.split('\n')
        for line in lines:
            if line.startswith(',') and any(0xAC00 <= ord(c) <= 0xD7AF for c in line if c != ','):
                print(f"  ⚠️ WARNING: Line starts with comma (likely converted quote): {line[:50]}...")
                comma_issue = True
        
        if not quote_found and not comma_issue:
            print("  ❌ No quotes found - they may have been lost in conversion!")
        elif comma_issue:
            print("  ❌ Some quotes were converted to commas!")
        else:
            print("  ✅ Quotes preserved successfully!")
        
        print("-" * 50 + "\n")


if __name__ == "__main__":
    # Run test if executed directly
    test_cjk_preservation()
