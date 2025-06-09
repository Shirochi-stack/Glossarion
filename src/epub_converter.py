import os
import sys
import io
import json
import mimetypes
import re
import zipfile
from ebooklib import epub, ITEM_DOCUMENT, ITEM_IMAGE, ITEM_STYLE
from bs4 import BeautifulSoup
import unicodedata
from xml.etree import ElementTree as ET
import html as html_module  # Use the standard html module for better entity handling

# Configure stdout for UTF-8
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


def decode_html_entities(text):
    """
    Comprehensive HTML entity decoding that handles all types of entities
    """
    if not text:
        return text
    
    # First pass: decode using html module (handles named and numeric entities)
    text = html_module.unescape(text)
    
    # Second pass: handle any remaining numeric entities manually
    # Decimal entities
    def decode_decimal(match):
        try:
            code = int(match.group(1))
            if is_valid_xml_char_code(code):
                return chr(code)
        except:
            pass
        return match.group(0)  # Return original if can't decode
    
    # Hexadecimal entities
    def decode_hex(match):
        try:
            code = int(match.group(1), 16)
            if is_valid_xml_char_code(code):
                return chr(code)
        except:
            pass
        return match.group(0)  # Return original if can't decode
    
    text = re.sub(r'&#(\d+);', decode_decimal, text)
    text = re.sub(r'&#x([0-9a-fA-F]+);', decode_hex, text)
    
    # Third pass: handle common entities that might have been missed
    entity_replacements = {
        '&nbsp;': ' ',
        '&ldquo;': '"',
        '&rdquo;': '"',
        '&lsquo;': ''',
        '&rsquo;': ''',
        '&mdash;': '—',
        '&ndash;': '–',
        '&hellip;': '…',
        '&bull;': '•',
        '&trade;': '™',
        '&copy;': '©',
        '&reg;': '®',
        '&uuml;': 'ü',
        '&auml;': 'ä',
        '&ouml;': 'ö',
        '&Uuml;': 'Ü',
        '&Auml;': 'Ä',
        '&Ouml;': 'Ö',
        '&szlig;': 'ß',
        '&agrave;': 'à',
        '&egrave;': 'è',
        '&eacute;': 'é',
        '&iuml;': 'ï',
        '&ccedil;': 'ç',
        '&ntilde;': 'ñ',
    }
    
    for entity, char in entity_replacements.items():
        text = text.replace(entity, char)
    
    return text


def safe_escape(text):
    """Enhanced escape function that preserves Unicode characters"""
    if not text:
        return ''
    
    # First decode any HTML entities to prevent double-escaping
    text = decode_html_entities(str(text))
    
    # Only escape XML special characters, not entities
    # Be careful not to re-escape already valid entities
    text = re.sub(r'&(?!amp;|lt;|gt;|quot;|apos;|#\d+;|#x[0-9a-fA-F]+;)', '&amp;', text)
    text = text.replace('<', '&lt;')
    text = text.replace('>', '&gt;')
    text = text.replace('"', '&quot;')
    text = text.replace("'", '&apos;')
    
    return text


def extract_translated_title_from_html(html_content, chapter_num=None, filename=None):
    """
    Extract the translated title from HTML content using multiple strategies
    Returns: (title, confidence_score)
    """
    try:
        # First decode all HTML entities in the content
        html_content = decode_html_entities(html_content)
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        candidates = []
        
        # Strategy 1: Look for <title> tag (highest confidence)
        title_tag = soup.find('title')
        if title_tag and title_tag.string:
            title_text = decode_html_entities(title_tag.string.strip())
            if title_text and len(title_text) > 1 and title_text.lower() not in ['untitled', 'chapter']:
                candidates.append((title_text, 0.95, "title_tag"))
        
        # Strategy 2: Look for h1 tags (high confidence)
        h1_tags = soup.find_all('h1')
        for h1 in h1_tags:
            h1_text = decode_html_entities(h1.get_text(strip=True))
            if h1_text and len(h1_text) < 300:
                candidates.append((h1_text, 0.9, "h1_tag"))
        
        # Strategy 3: Look for h2 tags (medium-high confidence)
        h2_tags = soup.find_all('h2')
        for h2 in h2_tags:
            h2_text = decode_html_entities(h2.get_text(strip=True))
            if h2_text and len(h2_text) < 250:
                candidates.append((h2_text, 0.8, "h2_tag"))
        
        # Strategy 4: Look for h3 tags (medium confidence) 
        h3_tags = soup.find_all('h3')
        for h3 in h3_tags:
            h3_text = decode_html_entities(h3.get_text(strip=True))
            if h3_text and len(h3_text) < 200:
                candidates.append((h3_text, 0.7, "h3_tag"))
        
        # Strategy 5: Look for bold/strong text in first few elements
        first_elements = soup.find_all(['p', 'div'])[:5]
        for elem in first_elements:
            bold_tags = elem.find_all(['b', 'strong'])
            for bold in bold_tags:
                bold_text = decode_html_entities(bold.get_text(strip=True))
                if bold_text and 3 <= len(bold_text) <= 150:
                    candidates.append((bold_text, 0.6, "bold_text"))
        
        # Strategy 6: Look for patterns in first paragraph
        first_p = soup.find('p')
        if first_p:
            p_text = decode_html_entities(first_p.get_text(strip=True))
            # Look for patterns like "Chapter X: Title"
            chapter_pattern = re.match(r'^(Chapter\s+\d+\s*[:\-\u2013\u2014]\s*)(.{3,100})(?:\.|$|Chapter)', p_text, re.IGNORECASE)
            if chapter_pattern:
                title_part = chapter_pattern.group(2).strip()
                if title_part and len(title_part) >= 3:
                    candidates.append((title_part, 0.8, "paragraph_pattern"))
            elif len(p_text) <= 100:
                # Might be a standalone title
                candidates.append((p_text, 0.4, "paragraph_standalone"))
        
        # Strategy 7: Extract from filename if available (fallback)
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
                # Clean title
                title = clean_extracted_title(title)
                if title and len(title) > 2 and is_valid_title(title):
                    if title not in unique_candidates or unique_candidates[title][1] < confidence:
                        unique_candidates[title] = (title, confidence, source)
            
            if unique_candidates:
                # Sort by confidence
                sorted_candidates = sorted(unique_candidates.values(), key=lambda x: x[1], reverse=True)
                best_title, best_confidence, best_source = sorted_candidates[0]
                return best_title, best_confidence
        
        # Fallback: generate from chapter number
        if chapter_num:
            return f"Chapter {chapter_num}", 0.1
        else:
            return "Untitled Chapter", 0.0
            
    except Exception as e:
        print(f"[WARNING] Error extracting title from HTML: {e}")
        if chapter_num:
            return f"Chapter {chapter_num}", 0.1
        else:
            return "Untitled Chapter", 0.0


def clean_extracted_title(title):
    """Clean and normalize extracted title"""
    if not title:
        return ""
    
    # Decode any remaining HTML entities
    title = decode_html_entities(title)
    
    # Remove HTML tags if any
    title = re.sub(r'<[^>]+>', '', title)
    
    # Normalize whitespace
    title = re.sub(r'\s+', ' ', title).strip()
    
    # Remove leading punctuation only
    title = re.sub(r'^[:\-\u2013\u2014\s]+', '', title).strip()
    
    # Remove quotes if they wrap the entire title
    if (title.startswith('"') and title.endswith('"')) or (title.startswith("'") and title.endswith("'")):
        title = title[1:-1].strip()
    
    # Truncate if too long
    if len(title) > 150:
        title = title[:147] + "..."
    
    return title


def is_valid_title(title):
    """Check if extracted title is valid and meaningful"""
    if not title or len(title) < 2:
        return False
    
    # Check for common invalid patterns
    invalid_patterns = [
        r'^\d+$',  # Just numbers
        r'^[^\w\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]+$',  # Just punctuation (allow Unicode)
        r'^(untitled)$',  # Just "untitled"
    ]
    
    for pattern in invalid_patterns:
        if re.match(pattern, title.lower().strip()):
            return False
    
    # Skip common filler phrases
    title_lower = title.lower().strip()
    filler_phrases = [
        'click here', 'read more', 'continue reading', 'next chapter',
        'previous chapter', 'table of contents', 'back to top'
    ]
    
    if any(phrase in title_lower for phrase in filler_phrases):
        return False
    
    return True


def clean_chapter_content(html_content):
    """Clean and prepare chapter content for XHTML conversion"""
    # Decode all HTML entities first
    html_content = decode_html_entities(html_content)
    
    # Remove any existing XML declarations or DOCTYPE
    html_content = re.sub(r'<\?xml[^>]*\?>', '', html_content)
    html_content = re.sub(r'<!DOCTYPE[^>]*>', '', html_content)
    
    # Remove any namespace declarations from html tag
    html_content = re.sub(r'<html[^>]*>', '<html>', html_content)
    
    # Remove NULL bytes and control characters
    html_content = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', html_content)
    
    # Fix malformed tags
    html_content = fix_malformed_tags(html_content)
    
    # Remove any remaining invalid characters
    html_content = ''.join(c for c in html_content if is_valid_xml_char(c))
    
    return html_content


def ensure_xhtml_compliance(html_content, title="Chapter", css_links=None):
    """
    Ensure HTML content is XHTML-compliant for strict EPUB readers.
    """
    try:
        # First, check if the content is already properly formatted XHTML
        if html_content.strip().startswith('<?xml') and '<html xmlns=' in html_content:
            return html_content
        
        # Clean the content
        html_content = clean_chapter_content(html_content)
        
        # Parse with BeautifulSoup - use 'html.parser' to avoid entity issues
        if isinstance(html_content, bytes):
            soup = BeautifulSoup(html_content, 'html.parser', from_encoding='utf-8')
        else:
            soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract title if available (this is where we get the translated title)
        extracted_title = None
        if soup.title and soup.title.string:
            extracted_title = str(soup.title.string).strip()
        elif soup.h1:
            extracted_title = soup.h1.get_text(strip=True)
        elif soup.h2:
            extracted_title = soup.h2.get_text(strip=True)
        
        # Use extracted title if found, otherwise use provided title
        if extracted_title:
            title = extracted_title
        
        # Clean title
        title = re.sub(r'[<>&"\']+', '', title)
        if not title:
            title = "Chapter"
        
        # Extract existing CSS links if not provided
        if css_links is None and soup.head:
            css_links = []
            for link in soup.head.find_all('link', rel='stylesheet'):
                href = link.get('href', '')
                if href:
                    css_links.append(href)
        
        # Extract body content
        body_content = ""
        if soup.body:
            body_parts = []
            for child in soup.body.children:
                if hasattr(child, 'name'):  # It's a tag
                    try:
                        # Use minimal formatting to avoid re-escaping entities
                        if hasattr(child, 'decode_contents'):
                            child_str = str(child)
                        else:
                            child_str = str(child)
                    except:
                        child_str = str(child)
                    
                    # Fix self-closing tags
                    child_str = fix_self_closing_tags(child_str)
                    # Don't re-escape ampersands that are part of valid entities
                    child_str = re.sub(r'&(?!amp;|lt;|gt;|quot;|apos;|#\d+;|#x[0-9a-fA-F]+;)', '&amp;', child_str)
                    body_parts.append(child_str)
                else:  # It's text
                    text = str(child)
                    if text.strip():
                        # Don't escape if it looks like it contains entities
                        if '&' in text and ';' in text:
                            # Check if it contains valid entities
                            if re.search(r'&[a-zA-Z]+;|&#\d+;|&#x[0-9a-fA-F]+;', text):
                                # Decode entities first, then escape
                                text = decode_html_entities(text)
                        # Now escape XML special characters
                        text = safe_escape(text)
                        body_parts.append(text)
            body_content = '\n'.join(body_parts)
        else:
            # If no body tag, try to extract meaningful content
            for tag in soup(['script', 'style']):
                tag.decompose()
            
            body_content = str(soup)
            body_content = fix_self_closing_tags(body_content)
        
        # If body_content is still empty, add some default content
        if not body_content.strip():
            body_content = '<p>Empty chapter</p>'
        
        # Build proper XHTML document
        xhtml_parts = []
        
        # XML declaration and DOCTYPE
        xhtml_parts.append('<?xml version="1.0" encoding="utf-8"?>')
        xhtml_parts.append('<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">')
        xhtml_parts.append('<html xmlns="http://www.w3.org/1999/xhtml">')
        xhtml_parts.append('<head>')
        xhtml_parts.append('<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />')
        xhtml_parts.append(f'<title>{safe_escape(title)}</title>')
        
        # Add CSS links
        if css_links:
            for css_link in css_links:
                if css_link.startswith('<link'):
                    # Extract href from existing link tag
                    href_match = re.search(r'href="([^"]+)"', css_link)
                    if href_match:
                        css_link = href_match.group(1)
                    else:
                        continue
                
                xhtml_parts.append(f'<link rel="stylesheet" type="text/css" href="{safe_escape(css_link)}" />')
        
        xhtml_parts.append('</head>')
        xhtml_parts.append('<body>')
        xhtml_parts.append(body_content)
        xhtml_parts.append('</body>')
        xhtml_parts.append('</html>')
        
        return '\n'.join(xhtml_parts)
        
    except Exception as e:
        # If anything fails, return a minimal but valid XHTML document
        print(f"[WARNING] Failed to ensure XHTML compliance: {e}")
        safe_title = re.sub(r'[<>&"\']+', '', str(title))
        if not safe_title:
            safe_title = "Chapter"
            
        return f'''<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
<title>{safe_escape(safe_title)}</title>
</head>
<body>
<p>Error processing content. Please check the source file.</p>
</body>
</html>'''


# Keep all the other functions from the original file unchanged
def extract_chapter_info_from_filename(filename):
    """Extract chapter number and basic info from filename"""
    # Pattern: response_001_title.html
    match = re.match(r"response_(\d+)_(.+?)\.html", filename)
    if match:
        chapter_num = int(match.group(1))
        title_part = match.group(2).replace('_', ' ').title()
        return chapter_num, title_part
    
    # Fallback patterns
    num_match = re.search(r'(\d+)', filename)
    if num_match:
        return int(num_match.group(1)), None
    
    return None, None


def analyze_chapter_files(output_dir, log_callback=None):
    """
    Analyze all chapter files and extract titles with confidence scores
    Returns: dict of {chapter_num: (title, confidence, filename)}
    """
    def log(message):
        if log_callback:
            log_callback(message)
        else:
            print(message)
    
    chapter_info = {}
    
    # Find all response HTML files
    html_files = [f for f in os.listdir(output_dir) if f.startswith("response_") and f.endswith(".html")]
    
    if not html_files:
        log("⚠️ No translated chapter files found!")
        return chapter_info
    
    log(f"📖 Analyzing {len(html_files)} translated chapter files for titles...")
    
    for filename in sorted(html_files):
        file_path = os.path.join(output_dir, filename)
        
        try:
            # Extract chapter number from filename
            chapter_num, filename_title = extract_chapter_info_from_filename(filename)
            
            if chapter_num is None:
                log(f"⚠️ Could not extract chapter number from: {filename}")
                continue
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Extract title from content
            extracted_title, confidence = extract_translated_title_from_html(
                html_content, chapter_num, filename
            )
            
            # Store result
            chapter_info[chapter_num] = (extracted_title, confidence, filename)
            
            # Log result with confidence indicator
            if confidence > 0.7:
                indicator = "✅"
            elif confidence > 0.4:
                indicator = "🟡"
            else:
                indicator = "🔴"
            
            log(f"  {indicator} Chapter {chapter_num}: '{extracted_title}' (confidence: {confidence:.2f})")
            
        except Exception as e:
            log(f"❌ Error processing {filename}: {e}")
            # Add fallback entry
            if chapter_num:
                chapter_info[chapter_num] = (f"Chapter {chapter_num}", 0.0, filename)
    
    return chapter_info


def sanitize_filename(filename, allow_unicode=False):
    """
    Sanitize filename to be safe for EPUB and filesystem.
    More aggressive sanitization for better compatibility.
    """
    if allow_unicode:
        # Keep Unicode characters but remove filesystem-unsafe ones
        filename = unicodedata.normalize('NFC', filename)
        
        # Replace only the most problematic characters
        replacements = {
            '/': '_', '\\': '_', ':': '_', '*': '_',
            '?': '_', '"': '_', '<': '_', '>': '_',
            '|': '_', '\0': '_',
        }
        
        for old, new in replacements.items():
            filename = filename.replace(old, new)
            
        # Remove control characters but keep Unicode
        filename = ''.join(char for char in filename if ord(char) >= 32 or ord(char) == 9)
        
    else:
        # For internal EPUB files, convert to ASCII
        filename = unicodedata.normalize('NFKD', filename)
        
        # Try to get ASCII representation
        try:
            filename = filename.encode('ascii', 'ignore').decode('ascii')
        except:
            # If that fails, replace non-ASCII with underscores
            filename = ''.join(c if ord(c) < 128 else '_' for c in filename)
        
        # Replace problematic characters with safe alternatives
        replacements = {
            '/': '_', '\\': '_', ':': '_', '*': '_',
            '?': '_', '"': '_', '<': '_', '>': '_',
            '|': '_', '\n': '_', '\r': '_', '\t': '_',
            '&': '_and_', '#': '_num_', ' ': '_',
        }
        
        for old, new in replacements.items():
            filename = filename.replace(old, new)
        
        # Remove any remaining control characters
        filename = ''.join(char for char in filename if ord(char) >= 32)
        
        # Clean up multiple underscores
        filename = re.sub(r'_+', '_', filename)
        filename = filename.strip('_')
    
    # Limit length to avoid filesystem issues
    name, ext = os.path.splitext(filename)
    if len(name) > 100:
        name = name[:100]
    
    # Ensure we have a valid filename
    if not name or name == '_':
        name = 'file'
    
    return name + ext


def fix_malformed_tags(html_content):
    """
    Fix various types of malformed tags that break XML parsing.
    Consolidates all malformed tag fixes in one place.
    """
    # Pattern 1: <tag =="" word="">  ->  [tag: word]
    html_content = re.sub(r'<(\w+)\s*=\s*""\s*(\w+)\s*=\s*"">', r'[\1: \2]', html_content)
    html_content = re.sub(r'<(\w+)\s*=\s*""\s*([^">]+)\s*=?\s*"">', r'[\1: \2]', html_content)
    
    # Pattern 2: <tag =""" word="">  ->  remove
    html_content = re.sub(r'<[^>]*?="""[^>]*?>', '', html_content)
    
    # Pattern 3: Multiple quotes in attributes
    html_content = re.sub(r'<([^>]+?)=""[^"]*""([^>]*?)>', r'<\1\2>', html_content)
    html_content = re.sub(r'="""\s*\w+\s*=""', '=""', html_content)
    
    # Pattern 4: Skill/ability/spell/detect tags with broken attributes
    def fix_skill_tag(match):
        tag_name = match.group(1)
        attrs = match.group(2)
        
        # Try to extract skill name from various patterns
        skill_match = re.search(r'(?:name\s*=\s*"([^"]+)"|"([^"]+)"|(\w+))', attrs)
        if skill_match:
            skill_name = skill_match.group(1) or skill_match.group(2) or skill_match.group(3)
            return f'[{tag_name}: {skill_name}]'
        return f'[{tag_name}]'
    
    html_content = re.sub(r'<(skill|ability|spell|detect)([^>]*?)/?>', fix_skill_tag, html_content)
    html_content = re.sub(r'</(skill|ability|spell|detect)>', '', html_content)
    
    # Pattern 5: Clean up any remaining problematic patterns
    html_content = re.sub(r'<(\w+)\s+=\s*""\s*([^">]+)\s*="">', r'[\1: \2]', html_content)
    
    return html_content


def fix_self_closing_tags(content):
    """Fix self-closing tags for XHTML compliance"""
    # List of void elements that should be self-closed in XHTML
    void_elements = ['area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 
                     'link', 'meta', 'param', 'source', 'track', 'wbr']
    
    for tag in void_elements:
        # Fix simple tags without attributes
        content = re.sub(f'<{tag}>', f'<{tag} />', content)
        
        # Fix tags with attributes
        def replacer(match):
            full_match = match.group(0)
            # Check if it already ends with />
            if full_match.rstrip().endswith('/>'):
                return full_match
            # Otherwise, add the self-closing /
            return full_match[:-1] + ' />'
        
        pattern = f'<{tag}(\\s+[^>]*)?>'
        content = re.sub(pattern, replacer, content)
    
    return content


def is_valid_xml_char_code(codepoint):
    """Check if a codepoint is valid for XML"""
    return (
        codepoint == 0x9 or 
        codepoint == 0xA or 
        codepoint == 0xD or 
        (0x20 <= codepoint <= 0xD7FF) or 
        (0xE000 <= codepoint <= 0xFFFD) or 
        (0x10000 <= codepoint <= 0x10FFFF)
    )


def is_valid_xml_char(c):
    """Check if a character is valid for XML"""
    return is_valid_xml_char_code(ord(c))


def ensure_bytes(content):
    """Ensure content is bytes for ebooklib"""
    if content is None:
        return b''
    if isinstance(content, bytes):
        return content
    # Convert to string first if it's not already
    if not isinstance(content, str):
        content = str(content)
    return content.encode('utf-8')


def validate_xhtml(content):
    """Validate XHTML content and fix common issues"""
    # Ensure proper XML declaration
    if not content.strip().startswith('<?xml'):
        content = '<?xml version="1.0" encoding="utf-8"?>\n' + content
    
    # Remove NULL bytes and other control characters
    content = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', content)
    
    # Fix malformed tags one more time (safety net)
    content = fix_malformed_tags(content)
    
    # Fix unescaped ampersands
    content = re.sub(r'&(?!amp;|lt;|gt;|quot;|apos;|#\d+;|#x[0-9a-fA-F]+;)', '&amp;', content)
    
    # Fix self-closing tags
    content = fix_self_closing_tags(content)
    
    # Fix unquoted attributes
    content = re.sub(r'<([^>]+)\s+(\w+)=([^\s"\'>]+)([>\s])', r'<\1 \2="\3"\4', content)
    
    # Remove any invalid XML characters
    content = ''.join(c for c in content if is_valid_xml_char(c))
    
    # Try to parse it to ensure it's valid
    try:
        ET.fromstring(content.encode('utf-8'))
    except ET.ParseError as e:
        print(f"[WARNING] XHTML validation failed: {e}")
        
        # Try to extract error context
        try:
            lines = content.split('\n')
            error_line = getattr(e, 'position', (0, 0))[0] - 1
            error_col = getattr(e, 'position', (0, 0))[1] - 1
            
            if 0 <= error_line < len(lines):
                print(f"[DEBUG] Error context (line {error_line + 1}):")
                print(f"[DEBUG] {lines[error_line]}")
                if error_col >= 0:
                    print(f"[DEBUG] {' ' * error_col}^")
                    
                # Check for specific patterns and try to fix
                error_line_content = lines[error_line]
                if '==""' in error_line_content:
                    print("[DEBUG] Found malformed attribute pattern")
                    lines[error_line] = fix_malformed_tags(error_line_content)
                    content = '\n'.join(lines)
                    
                    # Try validation again
                    try:
                        ET.fromstring(content.encode('utf-8'))
                        print("[INFO] XHTML auto-fixed by removing malformed attributes")
                        return content
                    except:
                        pass
        except:
            pass
        
        # Try to fix with BeautifulSoup
        try:
            soup = BeautifulSoup(content, 'html.parser')
            
            if soup.body:
                body_content = str(soup.body)
                # Re-wrap in proper XHTML
                content = f'''<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
<title>Chapter</title>
</head>
{body_content}
</html>'''
            
            # Fix self-closing tags again
            content = fix_self_closing_tags(content)
            
            # Try to validate again
            ET.fromstring(content.encode('utf-8'))
            print("[INFO] XHTML auto-fixed successfully")
            
        except Exception as e2:
            print(f"[WARNING] Failed to auto-fix XHTML: {e2}")
    
    return content


def verify_css_in_chapter(chapter_html, css_files):
    """Verify that CSS links are present in chapter HTML"""
    soup = BeautifulSoup(chapter_html, 'html.parser')
    found_css = []
    
    for link in soup.find_all('link', rel='stylesheet'):
        href = link.get('href', '')
        for css_file in css_files:
            if css_file in href:
                found_css.append(css_file)
    
    return found_css


def preflight_check(base_dir, log_callback=None):
    """Pre-flight check before EPUB conversion"""
    def log(message):
        if log_callback:
            log_callback(message)
        else:
            print(message)
    
    log("\n📋 Pre-flight Check")
    log("=" * 50)
    
    issues = []
    
    # Check directory exists
    if not os.path.exists(base_dir):
        issues.append(f"Directory does not exist: {base_dir}")
        return False, issues
    
    # Check for HTML files
    html_files = [f for f in os.listdir(base_dir) if f.endswith('.html')]
    response_files = [f for f in html_files if f.startswith('response_')]
    
    if not html_files:
        issues.append("No HTML files found in directory")
    elif not response_files:
        issues.append(f"Found {len(html_files)} HTML files but none start with 'response_'")
        log(f"  Example files: {html_files[:3]}")
    else:
        log(f"✅ Found {len(response_files)} chapter files")
        
        # Check if files are empty
        empty_files = []
        for f in response_files[:5]:  # Check first 5
            path = os.path.join(base_dir, f)
            if os.path.getsize(path) < 100:
                empty_files.append(f)
        
        if empty_files:
            issues.append(f"Found empty chapter files: {empty_files}")
    
    # Check for metadata.json
    metadata_path = os.path.join(base_dir, 'metadata.json')
    if not os.path.exists(metadata_path):
        log("⚠️  No metadata.json found (will use defaults)")
    else:
        log("✅ Found metadata.json")
    
    # Check for subdirectories
    for subdir in ['css', 'images', 'fonts']:
        path = os.path.join(base_dir, subdir)
        if os.path.exists(path):
            count = len(os.listdir(path))
            log(f"✅ Found {subdir}/ with {count} files")
    
    # Report results
    if issues:
        log("\n❌ Pre-flight check FAILED:")
        for issue in issues:
            log(f"  • {issue}")
        return False, issues
    else:
        log("\n✅ Pre-flight check PASSED")
        return True, []


def fallback_compile_epub(base_dir, log_callback=None):
    """Compile translated HTML files into an EPUB with extracted translated titles"""
    def log(message):
        if log_callback:
            log_callback(message)
        else:
            print(message)
    
    try:
        # Run pre-flight check
        passed, issues = preflight_check(base_dir, log_callback)
        if not passed:
            raise Exception(f"Pre-flight check failed: {'; '.join(issues)}")
        
        # Set up paths
        OUTPUT_DIR = os.path.abspath(base_dir)
        IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
        CSS_DIR = os.path.join(OUTPUT_DIR, "css")
        FONTS_DIR = os.path.join(OUTPUT_DIR, "fonts")
        METADATA = os.path.join(OUTPUT_DIR, "metadata.json")
        
        log(f"[DEBUG] Working with output directory: {OUTPUT_DIR}")

        # MAJOR IMPROVEMENT: Analyze chapter files to extract translated titles
        log("\n📖 Extracting translated titles from chapter files...")
        chapter_titles_info = analyze_chapter_files(OUTPUT_DIR, log_callback)
        
        if chapter_titles_info:
            log(f"✅ Successfully extracted titles for {len(chapter_titles_info)} chapters")
            
            # Show summary of extracted titles
            confident_titles = sum(1 for _, (_, confidence, _) in chapter_titles_info.items() if confidence > 0.5)
            log(f"📊 Title extraction summary: {confident_titles}/{len(chapter_titles_info)} with high confidence")
        else:
            log("⚠️ No chapter titles could be extracted - will use fallback titles")

        # Scan directory for files to process
        log(f"[DEBUG] Scanning directory: {OUTPUT_DIR}")
        try:
            all_files = os.listdir(OUTPUT_DIR)
            log(f"[DEBUG] Total files in directory: {len(all_files)}")
            
            # Find HTML files
            all_html_files = [f for f in all_files if f.endswith('.html')]
            if all_html_files:
                log(f"[DEBUG] All HTML files found: {all_html_files[:10]}...")
            else:
                log("[DEBUG] No HTML files found at all!")
            
        except Exception as e:
            log(f"[ERROR] Cannot list directory contents: {e}")
            raise Exception(f"Cannot access output directory: {OUTPUT_DIR}")
            
        html_files = [f for f in all_files if f.startswith("response_") and f.endswith(".html")]
        log(f"[DEBUG] Found {len(html_files)} translated HTML files")
        
        if not html_files:
            log("❌ No translated HTML files found")
            
            # Check for alternate patterns
            alternate_patterns = [
                ("chapter_", ".html"),
                ("Chapter", ".html"),
                ("", ".html"),
                ("response", ".html"),
                ("translated_", ".html"),
            ]
            
            for prefix, suffix in alternate_patterns:
                alt_files = [f for f in all_files if f.startswith(prefix) and f.endswith(suffix)]
                if alt_files:
                    log(f"[INFO] Found {len(alt_files)} files with pattern '{prefix}*{suffix}'")
                    log(f"[INFO] Example files: {alt_files[:3]}")
            
            raise Exception("No translated chapters found to compile into EPUB")

        # Load metadata
        if os.path.exists(METADATA):
            try:
                with open(METADATA, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                log("[DEBUG] Metadata loaded successfully")
                
                # Debug: Show ALL metadata keys
                log(f"[DEBUG] Metadata keys: {list(meta.keys())}")
                
                # Debug: Show title-related fields
                if 'title' in meta:
                    log(f"[DEBUG] Found 'title' in metadata: '{meta['title']}'")
                else:
                    log("[DEBUG] No 'title' field in metadata!")
                    
                if 'original_title' in meta:
                    log(f"[DEBUG] Found 'original_title' in metadata: '{meta['original_title']}'")
                else:
                    log("[DEBUG] No 'original_title' field in metadata")
                    
                # Check for any other title variations
                for key in meta.keys():
                    if 'title' in key.lower():
                        log(f"[DEBUG] Found title-related key: {key} = '{meta[key]}'")
                    
            except (json.JSONDecodeError, IOError) as e:
                log(f"[WARNING] Failed to load metadata.json: {e}")
                meta = {}
        else:
            log("[WARNING] metadata.json not found, using defaults")
            meta = {}

        # Initialize book
        book = epub.EpubBook()
        
        # Set book metadata
        book.set_identifier(meta.get("identifier", f"translated-{os.path.basename(base_dir)}"))
        
        # ENHANCED: Use translated title with better fallback logic
        book_title = None
        
        # First try: Look for 'title' field (should be the translated title)
        if 'title' in meta and meta['title'] and str(meta['title']).strip():
            book_title = str(meta['title']).strip()
            log(f"✅ [INFO] Using translated title from metadata: '{book_title}'")
            
        # Second try: Look for 'original_title' field
        elif 'original_title' in meta and meta['original_title'] and str(meta['original_title']).strip():
            book_title = str(meta['original_title']).strip()
            log(f"⚠️ [INFO] Using original title (no translation found): '{book_title}'")
            
        # Third try: Check if there's a title_translated flag and the title is actually translated
        elif 'title_translated' in meta and meta.get('title_translated') == True:
            # Title was supposedly translated but might be empty
            log("⚠️ [WARNING] Title was marked as translated but is empty or missing")
            book_title = os.path.basename(base_dir)
            log(f"📁 [INFO] Falling back to directory name: '{book_title}'")
            
        # Final fallback: Use directory name
        else:
            book_title = os.path.basename(base_dir)
            log(f"📁 [INFO] No title found in metadata, using directory name: '{book_title}'")
        
        # Ensure we have a valid title
        if not book_title or not book_title.strip():
            book_title = os.path.basename(base_dir)
            log(f"⚠️ [WARNING] Title was empty, using directory name: '{book_title}'")
        
        # Set the title
        print(f"DEBUG: meta.get('title') = {meta.get('title')}, basename = {os.path.basename(base_dir)}")
        book.set_title(book_title)
        log(f"📚 [FINAL] Book title set to: '{book_title}'")
        
        book.set_language(meta.get("language", "en"))
        
        # Also add original title as metadata if it exists and is different
        if meta.get('original_title') and meta.get('original_title') != book_title:
            book.add_metadata('DC', 'title', meta['original_title'], {'type': 'original'})
            log(f"[INFO] Added original title as metadata: '{meta['original_title']}'")
        
        if meta.get("creator"):
            book.add_author(meta["creator"])
            log(f"[INFO] Set author: {meta['creator']}")
        spine = []
        toc = []
        processed_images = {}
        chapters_added = 0
        empty_chapters = 0

        # Add CSS files
        css_items = []
        if os.path.isdir(CSS_DIR):
            css_files = [f for f in sorted(os.listdir(CSS_DIR)) if f.endswith('.css')]
            log(f"[DEBUG] Found {len(css_files)} CSS files to add")
            
            for css_file in css_files:
                css_path = os.path.join(CSS_DIR, css_file)
                try:
                    with open(css_path, 'r', encoding='utf-8') as f:
                        css_content = f.read()
                    
                    log(f"[DEBUG] Reading CSS: {css_file} ({len(css_content)} bytes)")
                    
                    css_item = epub.EpubItem(
                        uid=f"css_{css_file}",
                        file_name=f"css/{css_file}",
                        media_type="text/css",
                        content=ensure_bytes(css_content)
                    )
                    book.add_item(css_item)
                    css_items.append(css_item)
                    log(f"✅ Added CSS: {css_file}")
                except Exception as e:
                    log(f"[WARNING] Failed to add CSS {css_file}: {e}")
        
        # Add fonts
        if os.path.isdir(FONTS_DIR):
            for font_file in os.listdir(FONTS_DIR):
                font_path = os.path.join(FONTS_DIR, font_file)
                if os.path.isfile(font_path):
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
                        log(f"✅ Added font: {font_file}")
                    except Exception as e:
                        log(f"[WARNING] Failed to add font {font_file}: {e}")

        # Process images
        image_files = []
        if os.path.isdir(IMAGES_DIR):
            try:
                for img in sorted(os.listdir(IMAGES_DIR)):
                    path = os.path.join(IMAGES_DIR, img)
                    if os.path.isfile(path):
                        ctype, _ = mimetypes.guess_type(path)
                        if ctype and ctype.startswith("image"):
                            safe_name = sanitize_filename(img, allow_unicode=False)
                            
                            # Ensure proper extension
                            if not os.path.splitext(safe_name)[1]:
                                ext = os.path.splitext(img)[1]
                                if ext:
                                    safe_name += ext
                                else:
                                    if ctype == 'image/jpeg':
                                        safe_name += '.jpg'
                                    elif ctype == 'image/png':
                                        safe_name += '.png'
                                    elif ctype == 'image/gif':
                                        safe_name += '.gif'
                            
                            image_files.append(safe_name)
                            processed_images[img] = safe_name
                            log(f"[DEBUG] Found image: {img} -> {safe_name}")
            except OSError as e:
                log(f"[WARNING] Error reading images directory: {e}")

        # Find cover image
        cover_file = None
        cover_patterns = ['cover', 'Cover', 'COVER', 'front', 'Front']
        
        for pattern in cover_patterns:
            for ext in ["jpg", "jpeg", "png"]:
                candidate = f"{pattern}.{ext}"
                safe_candidate = sanitize_filename(candidate, allow_unicode=False)
                
                if safe_candidate in image_files or candidate in processed_images:
                    cover_file = processed_images.get(candidate, safe_candidate)
                    log(f"[DEBUG] Found cover by name match: {cover_file}")
                    break
            if cover_file:
                break
        
        if not cover_file and image_files:
            cover_file = image_files[0]
            log(f"[DEBUG] Using first image as cover: {cover_file}")

        # Embed images (except cover)
        for original_name, safe_name in processed_images.items():
            if safe_name == cover_file:
                continue
            
            img_path = os.path.join(IMAGES_DIR, original_name)
            try:
                ctype, _ = mimetypes.guess_type(img_path)
                with open(img_path, 'rb') as fp:
                    data = fp.read()
                
                book.add_item(epub.EpubItem(
                    uid=safe_name,
                    file_name=f"images/{safe_name}",
                    media_type=ctype or "image/jpeg",
                    content=data
                ))
            except (IOError, OSError) as e:
                log(f"[WARNING] Failed to embed image {original_name}: {e}")

        # Add cover if found
        if cover_file:
            original_cover = None
            for orig, safe in processed_images.items():
                if safe == cover_file:
                    original_cover = orig
                    break
            
            if original_cover:
                cover_path = os.path.join(IMAGES_DIR, original_cover)
                try:
                    with open(cover_path, 'rb') as fp:
                        cover_data = fp.read()
                    
                    # Create cover image item
                    cover_img = epub.EpubItem(
                        uid="cover-image",
                        file_name=f"images/{cover_file}",
                        media_type=mimetypes.guess_type(cover_path)[0] or "image/jpeg",
                        content=cover_data
                    )
                    book.add_item(cover_img)
                    
                    # Set cover metadata
                    book.add_metadata('http://purl.org/dc/elements/1.1/', 'cover', 'cover-image')
                    
                    # Create cover page
                    cover_page = epub.EpubHtml(
                        title="Cover",
                        file_name="cover.xhtml",
                        lang=meta.get("language", "en")
                    )
                    
                    # Build cover page with CSS links
                    cover_css_links = []
                    if css_items:
                        for css_item in css_items:
                            css_filename = css_item.file_name.split('/')[-1]
                            cover_css_links.append(f"css/{css_filename}")
                    
                    cover_body = f'''<div style="text-align: center;">
<img src="images/{cover_file}" alt="Cover" style="max-width: 100%; height: auto;" />
</div>'''
                    
                    # Ensure content is bytes
                    cover_page.content = ensure_bytes(validate_xhtml(ensure_xhtml_compliance(
                        cover_body, 
                        title="Cover",
                        css_links=cover_css_links
                    )))
                    
                    book.add_item(cover_page)
                    spine.insert(0, cover_page)
                    chapters_added += 1
                    
                    log(f"✅ Set cover image: {cover_file}")
                    
                except (IOError, OSError) as e:
                    log(f"[WARNING] Failed to add cover image: {e}")

        # Process chapters WITH IMPROVED TITLE EXTRACTION
        chapter_tuples = []
        chapter_seen = set()
        
        try:
            for fn in sorted(html_files):
                log(f"[DEBUG] Processing file: {fn}")
                m = re.match(r"response_(\d+)_", fn)
                if m:
                    num = int(m.group(1))
                    if num not in chapter_seen:
                        chapter_tuples.append((num, fn))
                        chapter_seen.add(num)
                    else:
                        log(f"[WARNING] Skipping duplicate chapter {num}: {fn}")
                else:
                    log(f"[WARNING] File doesn't match expected pattern: {fn}")
                    # Try alternative patterns
                    alt_match = re.search(r'(\d+)', fn)
                    if alt_match:
                        num = int(alt_match.group(1))
                        if num not in chapter_seen:
                            chapter_tuples.append((num, fn))
                            chapter_seen.add(num)
                            log(f"[INFO] Using alternative numbering for {fn}: Chapter {num}")
        except OSError as e:
            log(f"[ERROR] Failed to read output directory: {e}")
            raise

        # Sort chapters
        chapter_tuples.sort(key=lambda x: x[0])
        log(f"[DEBUG] Found {len(chapter_tuples)} unique chapters to process")

        # Add chapters WITH EXTRACTED TITLES - IMPROVED VERSION
        log(f"\n📚 Processing {len(chapter_tuples)} chapters with extracted titles...")
        for num, fn in chapter_tuples:
            path = os.path.join(OUTPUT_DIR, fn)
            log(f"\n[DEBUG] Processing chapter {num} from file: {fn}")
            
            try:
                log(f"[DEBUG] Reading file: {path}")
                with open(path, 'r', encoding='utf-8') as f:
                    raw = f.read()
                
                log(f"[DEBUG] File size: {len(raw)} characters")
                
                # IMPORTANT: Decode HTML entities immediately after reading
                raw = decode_html_entities(raw)
                log(f"[DEBUG] Decoded HTML entities in content")
                
                if not raw.strip():
                    log(f"[WARNING] Chapter {num} is empty, skipping")
                    empty_chapters += 1
                    continue
                
                # Clean the content (this will not re-encode entities)
                raw = clean_chapter_content(raw)
                
                # IMPROVED: Get title with multiple fallback attempts
                title = None
                confidence = 0.0
                title_source = "fallback"
                
                # First: Try pre-analyzed titles
                if num in chapter_titles_info:
                    title, confidence, source_filename = chapter_titles_info[num]
                    title_source = "pre-analyzed"
                    log(f"[DEBUG] Using pre-analyzed title: '{title}' (confidence: {confidence:.2f})")
                
                # Second: If low confidence or no title, re-extract from current content
                if not title or confidence < 0.5:
                    log(f"[DEBUG] Re-extracting title from content (current confidence: {confidence:.2f})")
                    backup_title, backup_confidence = extract_translated_title_from_html(raw, num, fn)
                    
                    if backup_confidence > confidence:
                        title = backup_title
                        confidence = backup_confidence
                        title_source = "re-extracted"
                        log(f"[DEBUG] Using re-extracted title: '{title}' (confidence: {confidence:.2f})")
                
                # Third: Clean and validate the title
                if title:
                    original_title = title
                    title = clean_extracted_title(title)
                    if not is_valid_title(title):
                        log(f"[DEBUG] Title '{original_title}' failed validation")
                        title = None
                
                # Final fallback
                if not title:
                    title = f"Chapter {num}"
                    title_source = "final_fallback"
                    log(f"[DEBUG] Using final fallback title: '{title}'")
                
                log(f"[INFO] ✅ Final title for chapter {num}: '{title}' (source: {title_source})")
                
                # Prepare CSS links
                chapter_css_links = []
                if css_items:
                    for css_item in css_items:
                        css_filename = css_item.file_name.split('/')[-1]
                        chapter_css_links.append(f"css/{css_filename}")
                
                # Convert to XHTML (this will also extract title from content as backup)
                try:
                    xhtml_content = ensure_xhtml_compliance(raw, title=title, css_links=chapter_css_links)
                except Exception as e:
                    log(f"[ERROR] Failed to convert chapter {num} to XHTML: {e}")
                    safe_title = re.sub(r'[<>&"\']+', '', title)
                    xhtml_content = f'''<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
<title>{safe_escape(safe_title)}</title>
</head>
<body>
{raw}
</body>
</html>'''
                
                # Process images in content
                try:
                    # Use html.parser to avoid XML parsing issues
                    soup = BeautifulSoup(xhtml_content, 'html.parser')
                except Exception as e:
                    log(f"[ERROR] Failed to parse chapter {num}: {e}")
                    continue
                
                # Fix image paths
                changed = False
                for img in soup.find_all('img'):
                    src = img.get('src', '')
                    basename = os.path.basename(src.split('?')[0])
                    safe_name = processed_images.get(basename, sanitize_filename(basename))
                    new_src = f"images/{safe_name}"
                    if src != new_src:
                        img['src'] = new_src
                        changed = True
                
                # Re-serialize if changed
                if changed:
                    try:
                        final_content = soup.prettify(formatter='html')
                    except:
                        final_content = str(soup)
                    
                    # Ensure it's a regular string, not NavigableString
                    final_content = str(final_content)
                    
                    if not final_content.strip().startswith('<?xml'):
                        final_content = '<?xml version="1.0" encoding="utf-8"?>\n' + final_content
                else:
                    # Ensure xhtml_content is a regular string
                    final_content = str(xhtml_content)
                
                # Create chapter
                safe_fn = f"chapter_{num:03d}.xhtml"
                
                chap = epub.EpubHtml(
                    title=title,  # This now uses the improved extracted translated title!
                    file_name=safe_fn,
                    lang=meta.get("language", "en")
                )
                
                # Validate and set content
                final_content = validate_xhtml(final_content)
                
                # Final safety check
                try:
                    ET.fromstring(final_content.encode('utf-8'))
                except Exception as e:
                    log(f"[ERROR] Chapter {num} XHTML is invalid: {e}")
                    final_content = f'''<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
<title>{safe_escape(title)}</title>
</head>
<body>
<h1>{safe_escape(title)}</h1>
<p>Chapter content could not be properly formatted. Please check the source file.</p>
</body>
</html>'''
                
                # CRITICAL FIX: Ensure content is bytes!
                chap.content = ensure_bytes(final_content)
                
                # Debug: Verify content is set
                log(f"   └─ Content size: {len(chap.content):,} bytes")
                
                book.add_item(chap)
                spine.append(chap)
                toc.append(chap)  # The TOC will now show the extracted translated title!
                chapters_added += 1
                
                log(f"✅ Added chapter {num}: '{title}' (File: {safe_fn}, Source: {title_source})")
                
                # Verify CSS links
                if css_items:
                    css_files = [item.file_name.split('/')[-1] for item in css_items]
                    found_css = verify_css_in_chapter(str(soup), css_files)
                    if found_css:
                        log(f"   └─ CSS files linked: {', '.join(found_css)}")
                
            except (IOError, OSError) as e:
                log(f"[WARNING] Failed to add chapter {num} from {fn}: {e}")
                continue

        # Check if we have content
        if chapters_added == 0:
            log("❌ No chapters were successfully added to the EPUB")
            if empty_chapters == len(chapter_tuples):
                raise Exception("All chapter files were empty. Please check the translation output.")
            else:
                raise Exception("No chapters could be added to the EPUB")

        # Debug: Verify chapters have content
        log("\n[DEBUG] Verifying chapter content:")
        for item in book.get_items():
            if item.get_type() == ITEM_DOCUMENT:
                content_size = len(item.content) if item.content else 0
                log(f"  • {item.file_name}: {content_size:,} bytes (Title: '{item.title}')")
                if content_size == 0:
                    log(f"    ⚠️ WARNING: Empty content!")

        log(f"\n[DEBUG] Summary before writing EPUB:")
        log(f"[DEBUG] Total chapters added: {chapters_added}")
        log(f"[DEBUG] Total items in spine: {len(spine)}")
        log(f"[DEBUG] Total items in TOC: {len(toc)}")
        log(f"[DEBUG] Total CSS files: {len(css_items)}")
        log(f"[DEBUG] Cover image: {'Yes' if cover_file else 'No'}")

        # Optional gallery
        gallery_images = [img for img in image_files if img != cover_file]
        if gallery_images:
            gallery_page = epub.EpubHtml(
                title="Gallery",
                file_name="gallery.xhtml",
                lang=meta.get("language", "en")
            )
            
            gallery_body_parts = ['<h1>Image Gallery</h1>']
            for img in gallery_images:
                gallery_body_parts.append(
                    f'<div style="text-align: center; margin: 20px;">'
                    f'<img src="images/{img}" alt="{img}" />'
                    f'</div>'
                )
            
            gallery_body = '\n'.join(gallery_body_parts)
            
            gallery_css_links = []
            if css_items:
                for css_item in css_items:
                    css_filename = css_item.file_name.split('/')[-1]
                    gallery_css_links.append(f"css/{css_filename}")
            
            # Ensure content is bytes
            gallery_page.content = ensure_bytes(validate_xhtml(ensure_xhtml_compliance(
                gallery_body,
                title="Gallery",
                css_links=gallery_css_links
            )))
            
            book.add_item(gallery_page)
            spine.append(gallery_page)
            toc.append(gallery_page)

        # Add navigation
        log("\n[DEBUG] Adding navigation files...")
        book.add_item(epub.EpubNav())
        book.add_item(epub.EpubNcx())
        log("[DEBUG] Navigation files added")
        
        # Set TOC and spine
        book.toc = toc  # This will now contain chapters with extracted translated titles!
        
        log(f"\n[DEBUG] Setting up spine...")
        log(f"[DEBUG] Spine items before setup: {len(spine)}")
        
        if spine and spine[0].title == "Cover":
            book.spine = [spine[0], 'nav'] + spine[1:]
            log("📖 Reading order: Cover → Table of Contents → Chapters")
        else:
            book.spine = ['nav'] + spine
            log("📖 Reading order: Table of Contents → Chapters")
        
        log(f"[DEBUG] Final spine length: {len(book.spine)}")

        # Add guide for cover
        if spine and spine[0].title == "Cover":
            book.guide = [
                {"type": "cover", "title": "Cover", "href": spine[0].file_name}
            ]

        # Write EPUB
        # Write EPUB - Use translated title for filename if available
        if book_title and book_title != os.path.basename(OUTPUT_DIR):
            # Sanitize the translated title for use as filename
            safe_filename = sanitize_filename(book_title, allow_unicode=True)
            out_path = os.path.join(OUTPUT_DIR, f"{safe_filename}.epub")
            log(f"[INFO] Using translated title for filename: {safe_filename}.epub")
        else:
            # Fallback to directory name
            base_name = os.path.basename(OUTPUT_DIR)
            out_path = os.path.join(OUTPUT_DIR, f"{base_name}.epub")
            log(f"[INFO] Using directory name for filename: {base_name}.epub")
        try:
            log(f"[DEBUG] Writing EPUB to: {out_path}")
            log(f"[DEBUG] Total CSS files included: {len(css_items)}")
            
            # Add debugging info
            log(f"[DEBUG] Book metadata:")
            log(f"  • Title: {book.title}")
            log(f"  • Language: {book.language}")
            log(f"  • Identifier: {book.uid}")
            log(f"[DEBUG] Book contents:")
            log(f"  • Items: {len(list(book.get_items()))}")
            log(f"  • Spine length: {len(book.spine)}")
            log(f"  • TOC entries: {len(book.toc)}")
            
            # Show final TOC with extracted titles
            log(f"\n📚 Final Table of Contents with Extracted Titles:")
            for i, item in enumerate(toc):
                confidence_info = ""
                if hasattr(item, 'file_name'):
                    # Try to find confidence info
                    chapter_match = re.search(r'chapter_(\d+)', item.file_name)
                    if chapter_match:
                        chapter_num = int(chapter_match.group(1))
                        if chapter_num in chapter_titles_info:
                            confidence = chapter_titles_info[chapter_num][1]
                            confidence_info = f" (confidence: {confidence:.2f})"
                
                log(f"  {i+1:2d}. {item.title}{confidence_info}")
            
            # Write the EPUB
            log("\n[DEBUG] Writing EPUB file...")
            epub.write_epub(out_path, book, {})
            
            # Verify the file was created
            if os.path.exists(out_path):
                file_size = os.path.getsize(out_path)
                log(f"✅ EPUB created at: {out_path}")
                log(f"📊 File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
                
                # Quick validation
                try:
                    with zipfile.ZipFile(out_path, 'r') as test_zip:
                        if 'mimetype' in test_zip.namelist():
                            log("✅ EPUB structure verified (mimetype present)")
                        else:
                            log("⚠️ WARNING: EPUB might be malformed (missing mimetype)")
                except Exception as e:
                    log(f"⚠️ WARNING: EPUB validation error: {e}")
                    # Don't raise an error if the file exists and has content
                    if file_size > 0:
                        log("ℹ️ Note: EPUB file exists with content, ignoring validation warning")
            else:
                log("❌ ERROR: EPUB file was not created!")
                raise Exception("EPUB file was not created")
            
            # Final notes
            if css_items:
                log(f"✅ Successfully embedded {len(css_items)} CSS files")
                for css_item in css_items:
                    log(f"   • {css_item.file_name}")
            
            # Show title extraction summary
            if chapter_titles_info:
                high_confidence = sum(1 for _, (_, conf, _) in chapter_titles_info.items() if conf > 0.7)
                medium_confidence = sum(1 for _, (_, conf, _) in chapter_titles_info.items() if 0.4 < conf <= 0.7)
                low_confidence = sum(1 for _, (_, conf, _) in chapter_titles_info.items() if conf <= 0.4)
                
                log(f"\n📊 Title Extraction Summary:")
                log(f"   • High confidence (>70%): {high_confidence} chapters")
                log(f"   • Medium confidence (40-70%): {medium_confidence} chapters")
                log(f"   • Low confidence (≤40%): {low_confidence} chapters")
                log(f"   • Total extracted: {len(chapter_titles_info)} chapters")
            
            log("\n📱 Compatibility Notes:")
            log("   • XHTML 1.1 compliant for strict readers")
            log("   • All tags properly self-closed")
            log("   • Special characters properly escaped")
            log("   • Malformed tags automatically fixed")
            log("   • Table of Contents uses extracted translated titles")
            log("   • Enhanced HTML entity decoding for umlauts and special characters")
            
        except Exception as e:
            log(f"❌ Failed to write EPUB: {e}")
            # Check if file was actually created despite the error
            if 'out_path' in locals() and os.path.exists(out_path):
                file_size = os.path.getsize(out_path)
                if file_size > 0:
                    log(f"ℹ️ NOTE: EPUB file exists at {out_path} ({file_size:,} bytes)")
                    log("ℹ️ The error may be a false positive - check if the EPUB opens correctly")
                    return  # Don't raise if file exists
            raise

    except Exception as e:
        log(f"❌ EPUB compilation failed with error: {e}")
        raise


# Legacy function alias
compile_epub = fallback_compile_epub


# Main execution
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
