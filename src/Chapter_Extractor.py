# Chapter_Extractor.py - Module-level chapter extraction functions
import os
import re
import sys
import json
import threading
import time
import shutil
import hashlib
import warnings

# Lazy import for PatternManager to speed up ProcessPoolExecutor worker startup on Windows
# The heavy TransateKRtoEN import is deferred until actually needed
_PatternManager = None
_PM = None

def _get_pattern_manager():
    """Lazy initialization of PatternManager to avoid slow imports in worker processes"""
    global _PatternManager, _PM
    if _PatternManager is None:
        pass  # TransateKRtoEN import removed (patterns inlined)
        _PatternManager = PM_Class
        _PM = PM_Class()
    return _PM

# For backward compatibility - property-like access
class _LazyPM:
    def __getattr__(self, name):
        return getattr(_get_pattern_manager(), name)

PM = _LazyPM()

from bs4 import BeautifulSoup
try:
    from bs4 import XMLParsedAsHTMLWarning
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
except ImportError:
    pass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from collections import Counter

# ---------------------------------------------------------------------------
# Inlined pattern constants for chapter extraction.
# These MUST NOT trigger TransateKRtoEN import (which takes ~18s).
# PPE worker functions use these instead of PM.* to avoid that import.
# ---------------------------------------------------------------------------
_CHAPTER_PATTERNS = [
    (r'chapter[\s_-]*(\d+)', re.IGNORECASE, 'english_chapter'),
    (r'\bch\.?\s*(\d+)\b', re.IGNORECASE, 'english_ch'),
    (r'part[\s_-]*(\d+)', re.IGNORECASE, 'english_part'),
    (r'episode[\s_-]*(\d+)', re.IGNORECASE, 'english_episode'),
    (r'第\s*(\d+)\s*[章节話话回]', 0, 'chinese_chapter'),
    (r'第\s*([一二三四五六七八九十百千万]+)\s*[章节話话回]', 0, 'chinese_chapter_cn'),
    (r'(\d+)[章节話话回]', 0, 'chinese_short'),
    (r'第\s*(\d+)\s*話', 0, 'japanese_wa'),
    (r'第\s*(\d+)\s*章', 0, 'japanese_chapter'),
    (r'その\s*(\d+)', 0, 'japanese_sono'),
    (r'(\d+)話目', 0, 'japanese_wame'),
    (r'제\s*(\d+)\s*[장화권부편]', 0, 'korean_chapter'),
    (r'(\d+)\s*[장화권부편]', 0, 'korean_short'),
    (r'에피소드\s*(\d+)', 0, 'korean_episode'),
    (r'^\s*(\d+)\s*[-–—.\:]', re.MULTILINE, 'generic_numbered'),
    (r'_(\d+)\.x?html?$', re.IGNORECASE, 'filename_number'),
    (r'/(\d+)\.x?html?$', re.IGNORECASE, 'path_number'),
    (r'(\d+)', 0, 'any_number'),
]

_CHINESE_NUMS = {
    '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
    '十一': 11, '十二': 12, '十三': 13, '十四': 14, '十五': 15,
    '十六': 16, '十七': 17, '十八': 18, '十九': 19, '二十': 20,
    '二十一': 21, '二十二': 22, '二十三': 23, '二十四': 24, '二十五': 25,
    '三十': 30, '四十': 40, '五十': 50, '六十': 60, '七十': 70, '八十': 80, '九十': 90, '百': 100,
    '壹': 1, '贰': 2, '叁': 3, '肆': 4, '伍': 5, '陆': 6, '柒': 7, '捌': 8, '玖': 9, '拾': 10,
    '佰': 100, '仟': 1000, '萬': 10000, '万': 10000,
    '第一': 1, '第二': 2, '第三': 3, '第四': 4, '第五': 5,
    '首': 1, '次': 2, '初': 1, '末': -1,
}

_FILENAME_EXTRACT_PATTERNS = [
    r'^\d{3}(\d)_(\d{2})_\.x?html?$',
    r'^\d{4}_(\d+)\.x?html?$',
    r'^\d+_(\d+)[_\.]',
    r'^(\d+)[_\.]',
    r'response_(\d+)_',
    r'response_(\d+)\.',
    r'(\d{3,5})[_\.]',
    r'[Cc]hapter[_\s]*(\d+)',
    r'[Cc]h[_\s]*(\d+)',
    r'No(\d+)Chapter',
    r'No(\d+)Section',
    r'No(\d+)(?=\.|_|$)',
    r'第(\d+)[章话回]',
    r'_(\d+)(?:_|\.|$)',
    r'^(\d+)(?:_|\.|$)',
    r'(\d+)',
]

# Stop request function (can be overridden)
def is_stop_requested():
    """Check if stop has been requested - default implementation"""
    return False

# Progress bar for terminal output
class ProgressBar:
    """Simple in-place progress bar for terminal output"""
    _last_line_length = 0
    
    @classmethod
    def update(cls, current, total, prefix="Progress", bar_length=30):
        if total == 0:
            return
        percent = min(100, int(100 * current / total))
        filled = int(bar_length * current / total)
        bar = '█' * filled + '░' * (bar_length - filled)
        line = f"\r{prefix}: [{bar}] {current}/{total} ({percent}%)"
        if len(line) < cls._last_line_length:
            line += ' ' * (cls._last_line_length - len(line))
        cls._last_line_length = len(line)
        print(line, end='', flush=True)
    
    @classmethod
    def finish(cls):
        print()
        cls._last_line_length = 0

# Helper for resource filename sanitization
def sanitize_resource_filename(filename):
    """Sanitize resource filenames to be filesystem-safe"""
    import unicodedata
    # Normalize unicode - use NFC to preserve Korean/CJK characters
    # NFKD decomposes Korean Hangul into jamo components, corrupting them
    filename = unicodedata.normalize('NFC', filename)
    # Remove or replace problematic characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    return filename

def _collect_image_srcs(soup):
    """Collect all image source URLs from a BeautifulSoup document.
    
    Supports: <img src>, SVG <image href>, <object data>, <video poster>,
    and CSS background-image: url(...) in inline styles.
    """
    image_srcs = []
    
    # <img src>
    for tag in soup.find_all('img'):
        src = tag.get('src', '')
        if src:
            image_srcs.append(src)
    
    # SVG <image> (xlink:href / href) — one per tag
    for tag in soup.find_all('image'):
        href = (tag.get('xlink:href') or 
                tag.get('href') or 
                tag.get('{http://www.w3.org/1999/xlink}href') or '')
        if href:
            image_srcs.append(href)
    
    # <object data> (embedded images/SVGs)
    for tag in soup.find_all('object'):
        data = tag.get('data', '')
        if data:
            image_srcs.append(data)
    
    # <video poster>
    for tag in soup.find_all('video'):
        poster = tag.get('poster', '')
        if poster:
            image_srcs.append(poster)
    
    # CSS background-image: url(...) in inline styles
    for tag in soup.find_all(style=True):
        style = tag.get('style', '')
        if 'url(' in style:
            for m in re.finditer(r'url\(["\']?([^"\')\s]+)["\']?\)', style):
                url = m.group(1)
                if url:
                    image_srcs.append(url)
    
    return image_srcs

def _update_image_refs_in_soup(soup, rename_map):
    """Update all image references in a BeautifulSoup document using the rename map.
    
    Returns True if any references were modified.
    Supports: <img src>, SVG <image href>, <object data>, <video poster>,
    and CSS background-image: url(...) in inline styles.
    """
    modified = False
    
    def _update_attr(tag, attr):
        nonlocal modified
        val = tag.get(attr, '')
        if not val or val.startswith('data:'):
            return
        clean = val.split('?')[0]
        basename = os.path.basename(clean)
        if basename in rename_map:
            new_name = rename_map[basename]
            dir_part = os.path.dirname(clean)
            tag[attr] = f"{dir_part}/{new_name}" if dir_part else new_name
            modified = True
    
    # <img src>
    for tag in soup.find_all('img'):
        _update_attr(tag, 'src')
    
    # SVG <image> (all href variants)
    for tag in soup.find_all('image'):
        for attr in ['xlink:href', 'href', '{http://www.w3.org/1999/xlink}href']:
            _update_attr(tag, attr)
    
    # <object data>
    for tag in soup.find_all('object'):
        _update_attr(tag, 'data')
    
    # <video poster>
    for tag in soup.find_all('video'):
        _update_attr(tag, 'poster')
    
    # CSS background-image: url(...) in inline styles
    for tag in soup.find_all(style=True):
        style = tag.get('style', '')
        if 'url(' not in style:
            continue
        new_style = style
        for m in re.finditer(r'url\(["\']?([^"\')\s]+)["\']?\)', style):
            url = m.group(1)
            if not url or url.startswith('data:'):
                continue
            clean = url.split('?')[0]
            basename = os.path.basename(clean)
            if basename in rename_map:
                new_name = rename_map[basename]
                dir_part = os.path.dirname(clean)
                new_url = f"{dir_part}/{new_name}" if dir_part else new_name
                new_style = new_style.replace(url, new_url)
                modified = True
        if new_style != style:
            tag['style'] = new_style
    
    return modified

def _rename_images_to_chapter_format(chapters, output_dir, progress_callback=None):
    """Rename image files to chapter-based format and update all references.
    
    Format: chapter{NNN}_img_{M}.{ext}
    - NNN: 3-digit zero-padded chapter number
    - M: sequential image number within that chapter (1-based)
    - ext: original file extension preserved
    
    Also saves image_rename_map.json for use by TransateKRtoEN.py and epub_converter.py.
    This operation is idempotent — re-running produces the same result.
    """
    images_dir = os.path.join(output_dir, 'images')
    rename_map_path = os.path.join(output_dir, 'image_rename_map.json')
    
    if not os.path.isdir(images_dir):
        print("📸 No images directory found — skipping image rename")
        return chapters
    
    # Collect existing image files
    existing_images = set()
    try:
        existing_images = {f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))}
    except Exception as e:
        print(f"⚠️ Could not list images directory: {e}")
        return chapters
    
    if not existing_images:
        print("📸 No image files found — skipping image rename")
        return chapters
    
    print(f"\n🖼️ Renaming {len(existing_images)} images to chapter-based format...")
    
    # Build mapping: scan each chapter body for <img> references
    # Track which images belong to which chapter (first reference wins)
    rename_map = {}  # original_name -> new_name
    claimed_images = set()  # images that have been assigned to a chapter
    
    for chapter in chapters:
        body = chapter.get('body', '')
        if not body:
            continue
        
        # Get the actual chapter filename stem for naming
        chapter_basename = chapter.get('original_basename', '')
        if not chapter_basename:
            # Fallback: derive from filename
            chapter_filename = chapter.get('filename', '')
            chapter_basename = os.path.splitext(os.path.basename(chapter_filename))[0] if chapter_filename else ''
        if not chapter_basename:
            # Last resort: use chapter number
            chapter_basename = f"chapter{int(chapter.get('num', 0)):03d}"
        
        # Remove extension from basename if it still has one
        chapter_stem = os.path.splitext(chapter_basename)[0]
        
        try:
            soup = BeautifulSoup(body, 'html.parser')
        except Exception:
            continue
        
        # Collect all image references from all supported formats
        image_srcs = _collect_image_srcs(soup)
        
        img_counter = 1
        for src in image_srcs:
            # Skip data URIs
            if src.startswith('data:'):
                continue
            
            # Extract basename from various path formats
            # Handle: ../images/foo.jpg, images/foo.jpg, foo.jpg, etc.
            clean_src = src.split('?')[0]  # Remove query params
            basename = os.path.basename(clean_src)
            
            if not basename or basename in claimed_images:
                continue
            
            # Check if file actually exists
            if basename not in existing_images:
                # Try case-insensitive match
                matched = None
                for existing in existing_images:
                    if existing.lower() == basename.lower():
                        matched = existing
                        break
                if matched:
                    basename = matched
                else:
                    continue
            
            # Generate new name using actual chapter filename
            ext = os.path.splitext(basename)[1]  # Preserve original extension
            new_name = f"{chapter_stem}_img_{img_counter}{ext}"
            
            # Handle collision
            while new_name in rename_map.values():
                img_counter += 1
                new_name = f"{chapter_stem}_img_{img_counter}{ext}"
            
            rename_map[basename] = new_name
            claimed_images.add(basename)
            img_counter += 1
    
    # Handle unclaimed images (not referenced by any chapter) — name as Cover
    unclaimed = existing_images - claimed_images
    unclaimed_to_rename = sorted(unclaimed)
    if unclaimed_to_rename:
        for idx, img_name in enumerate(unclaimed_to_rename):
            ext = os.path.splitext(img_name)[1]
            new_name = f"{idx}_Cover{ext}"
            while new_name in rename_map.values():
                idx += 1
                new_name = f"{idx}_Cover{ext}"
            rename_map[img_name] = new_name
        print(f"   📎 {len(unclaimed_to_rename)} unclaimed images named as Cover")
    
    if not rename_map:
        print("📸 No image references found in chapters — skipping rename")
        return chapters
    
    # Phase 1: Physically rename image files (use temp names to avoid collisions)
    print(f"   📁 Renaming {len(rename_map)} image files...")
    temp_renames = {}  # temp_name -> final_name
    successful_renames = {}  # original_name -> new_name (only successful ones)
    
    # First pass: rename to temporary names
    for original, new_name in rename_map.items():
        original_path = os.path.join(images_dir, original)
        if not os.path.exists(original_path):
            continue
        temp_name = f"_temp_rename_{original}"
        temp_path = os.path.join(images_dir, temp_name)
        try:
            os.rename(original_path, temp_path)
            temp_renames[temp_name] = new_name
            successful_renames[original] = new_name
        except Exception as e:
            print(f"   ⚠️ Could not rename {original}: {e}")
    
    # Second pass: rename from temp to final names
    for temp_name, final_name in temp_renames.items():
        temp_path = os.path.join(images_dir, temp_name)
        final_path = os.path.join(images_dir, final_name)
        try:
            os.rename(temp_path, final_path)
        except Exception as e:
            print(f"   ⚠️ Could not finalize rename {temp_name} -> {final_name}: {e}")
            # Try to restore original name
            original_name = temp_name.replace('_temp_rename_', '', 1)
            try:
                os.rename(temp_path, os.path.join(images_dir, original_name))
                if original_name in successful_renames:
                    del successful_renames[original_name]
            except Exception:
                pass
    
    print(f"   ✅ Successfully renamed {len(successful_renames)} image files")
    
    # Phase 2: Update all image references in chapter bodies
    print(f"   📝 Updating image references in {len(chapters)} chapters...")
    updated_chapters = 0
    for chapter in chapters:
        body = chapter.get('body', '')
        if not body:
            continue
        
        try:
            soup = BeautifulSoup(body, 'html.parser')
            modified = _update_image_refs_in_soup(soup, successful_renames)
            
            if modified:
                chapter['body'] = str(soup)
                updated_chapters += 1
        except Exception as e:
            print(f"   ⚠️ Error updating chapter {chapter.get('num', '?')}: {e}")
    
    print(f"   ✅ Updated image references in {updated_chapters} chapters")
    
    # Phase 3: Save rename map for other scripts
    try:
        with open(rename_map_path, 'w', encoding='utf-8') as f:
            json.dump(successful_renames, f, ensure_ascii=False, indent=2)
        print(f"   💾 Saved image rename map to: image_rename_map.json")
    except Exception as e:
        print(f"   ⚠️ Could not save rename map: {e}")
    
    # Phase 4: Update on-disk HTML files that were already saved during extraction
    # These files were written before the rename, so they still reference old image names
    print(f"   📄 Updating on-disk HTML files in output directory...")
    disk_updated = 0
    try:
        for fname in os.listdir(output_dir):
            fpath = os.path.join(output_dir, fname)
            if not os.path.isfile(fpath):
                continue
            if not fname.lower().endswith(('.html', '.xhtml', '.htm')):
                continue
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                soup = BeautifulSoup(content, 'html.parser')
                file_modified = _update_image_refs_in_soup(soup, successful_renames)
                
                if file_modified:
                    with open(fpath, 'w', encoding='utf-8') as f:
                        f.write(str(soup))
                    disk_updated += 1
            except Exception as e:
                print(f"   ⚠️ Error updating {fname}: {e}")
    except Exception as e:
        print(f"   ⚠️ Error scanning output directory: {e}")
    
    if disk_updated > 0:
        print(f"   ✅ Updated {disk_updated} on-disk HTML files")
    
    print(f"🖼️ Image renaming complete: {len(successful_renames)} images renamed")
    return chapters

def _get_best_parser():
    """Determine the best parser available, preferring lxml for CJK text"""
    try:
        import lxml
        return 'lxml'
    except ImportError:
        return 'html.parser'

def _sort_by_opf_spine(chapters, opf_path):
    """Sort chapters according to OPF spine order"""
    try:
        import xml.etree.ElementTree as ET
        
        # Read OPF file
        with open(opf_path, 'r', encoding='utf-8') as f:
            opf_content = f.read()
        
        # Parse OPF
        root = ET.fromstring(opf_content)
        
        # Find namespaces
        ns = {'opf': 'http://www.idpf.org/2007/opf'}
        if root.tag.startswith('{'):
            default_ns = root.tag[1:root.tag.index('}')]
            ns = {'opf': default_ns}
        
        # Build manifest map (id -> href)
        manifest = {}
        for item in root.findall('.//opf:manifest/opf:item', ns):
            item_id = item.get('id')
            href = item.get('href')
            if item_id and href:
                manifest[item_id] = href
        
        # Get spine order
        spine_order = []
        spine = root.find('.//opf:spine', ns)
        if spine is not None:
            for itemref in spine.findall('opf:itemref', ns):
                idref = itemref.get('idref')
                if idref and idref in manifest:
                    href = manifest[idref]
                    spine_order.append(href)
        
        if not spine_order:
            print("⚠️ No spine order found in OPF, keeping original order")
            return chapters
        
        # Create a mapping of filenames to spine position
        spine_map = {}
        for idx, href in enumerate(spine_order):
            # Try different matching strategies
            basename = os.path.basename(href)
            spine_map[basename] = idx
            spine_map[href] = idx
            # Also store without extension for flexible matching
            name_no_ext = os.path.splitext(basename)[0]
            spine_map[name_no_ext] = idx
        
        print(f"📋 OPF spine contains {len(spine_order)} items")
        
        # Sort chapters based on spine order
        def get_spine_position(chapter):
            # Try to match chapter to spine
            filename = chapter.get('filename', '')
            basename = chapter.get('original_basename', '')
            
            # Try exact filename match
            if filename in spine_map:
                return spine_map[filename]
            
            # Try basename match
            if basename in spine_map:
                return spine_map[basename]
            
            # Try basename of filename
            if filename:
                fname_base = os.path.basename(filename)
                if fname_base in spine_map:
                    return spine_map[fname_base]
            
            # Try without extension
            if basename:
                if basename + '.html' in spine_map:
                    return spine_map[basename + '.html']
                if basename + '.xhtml' in spine_map:
                    return spine_map[basename + '.xhtml']
            
            # Fallback to chapter number * 1000 (to sort after spine items)
            return 1000000 + chapter.get('num', 0)
        
        # Sort chapters
        sorted_chapters = sorted(chapters, key=get_spine_position)
        
        # Renumber chapters based on new order
        for idx, chapter in enumerate(sorted_chapters, 1):
            chapter['spine_order'] = idx
            # Optionally update chapter numbers to match spine order
            # chapter['num'] = idx  # Uncomment if you want to renumber
        
        # Log reordering info
        reordered_count = 0
        for idx, chapter in enumerate(sorted_chapters):
            original_idx = chapters.index(chapter)
            if original_idx != idx:
                reordered_count += 1
        
        if reordered_count > 0:
            print(f"🔄 Reordered {reordered_count} chapters to match OPF spine")
        else:
            print(f"✅ Chapter order already matches OPF spine")
        
        return sorted_chapters
        
    except Exception as e:
        print(f"⚠️ Could not sort by OPF spine: {e}")
        import traceback
        traceback.print_exc()
        return chapters


def protect_angle_brackets_with_korean(text: str) -> str:
    """Protect CJK text in angle brackets from HTML parsing"""
    if text is None:
        return ""
    
    import re
    # Extended pattern to include Korean, Chinese, and Japanese characters
    cjk_pattern = r'[가-힣ㄱ-ㅎㅏ-ㅣ一-龿ぁ-ゟァ-ヿ]'
    bracket_pattern = rf'<([^<>]*{cjk_pattern}[^<>]*)>'
    
    def replace_brackets(match):
        content = match.group(1)
        return f'&#60;{content}&#62;'
    
    return re.sub(bracket_pattern, replace_brackets, text)

def ensure_all_opf_chapters_extracted(zf, chapters, out):
    """Ensure ALL chapters from OPF spine are extracted, not just what ChapterExtractor found"""
    
    # Parse OPF to get ALL chapters in spine
    opf_chapters = []
    
    try:
        # Find content.opf
        opf_content = None
        for name in zf.namelist():
            if name.endswith('content.opf'):
                opf_content = zf.read(name)
                break
        
        if not opf_content:
            return chapters  # No OPF, return original
        
        import xml.etree.ElementTree as ET
        root = ET.fromstring(opf_content)
        
        # Handle namespaces
        ns = {'opf': 'http://www.idpf.org/2007/opf'}
        if root.tag.startswith('{'):
            default_ns = root.tag[1:root.tag.index('}')]
            ns = {'opf': default_ns}
        
        # Get manifest
        manifest = {}
        for item in root.findall('.//opf:manifest/opf:item', ns):
            item_id = item.get('id')
            href = item.get('href')
            media_type = item.get('media-type', '')
            
            if item_id and href and ('html' in media_type.lower() or href.endswith(('.html', '.xhtml', '.htm'))):
                manifest[item_id] = href
        
        # Get spine order
        spine = root.find('.//opf:spine', ns)
        if spine:
            for itemref in spine.findall('opf:itemref', ns):
                idref = itemref.get('idref')
                if idref and idref in manifest:
                    href = manifest[idref]
                    filename = os.path.basename(href)
                    
                    # Skip nav, toc, cover - BUT only if filename has NO numbers
                    # Files with numbers like 'nav01', 'toc05' are real chapters
                    import re
                    has_numbers = bool(re.search(r'\d', filename))
                    if not has_numbers and any(skip in filename.lower() for skip in ['nav', 'toc', 'cover']):
                        continue
                    
                    opf_chapters.append(href)
        
        print(f"📚 OPF spine contains {len(opf_chapters)} chapters")
        
        # Check which OPF chapters are missing from extraction
        extracted_files = set()
        for c in chapters:
            if 'filename' in c:
                extracted_files.add(c['filename'])
            if 'original_basename' in c:
                extracted_files.add(c['original_basename'])
        
        missing_chapters = []
        for opf_chapter in opf_chapters:
            basename = os.path.basename(opf_chapter)
            if basename not in extracted_files and opf_chapter not in extracted_files:
                missing_chapters.append(opf_chapter)
        
        if missing_chapters:
            print(f"⚠️ {len(missing_chapters)} chapters in OPF but not extracted!")
            print(f"   Missing: {missing_chapters[:5]}{'...' if len(missing_chapters) > 5 else ''}")
            
            # Extract the missing chapters
            for href in missing_chapters:
                try:
                    # Read the chapter content
                    content = zf.read(href).decode('utf-8')
                    
                    # Extract chapter number
                    import re
                    basename = os.path.basename(href)
                    matches = re.findall(r'(\d+)', basename)
                    if matches:
                        chapter_num = int(matches[-1])
                    else:
                        chapter_num = len(chapters) + 1
                    
                    # Create chapter entry
                    from bs4 import BeautifulSoup
                    parser = 'lxml' if 'lxml' in sys.modules else 'html.parser'
                    soup = BeautifulSoup(content, parser)
                    
                    # Get title
                    title = "Chapter " + str(chapter_num)
                    title_tag = soup.find('title')
                    if title_tag:
                        title = title_tag.get_text().strip() or title
                    else:
                        for tag in ['h1', 'h2', 'h3']:
                            header = soup.find(tag)
                            if header:
                                title = header.get_text().strip() or title
                                break
                    
                    # Save the chapter file
                    output_filename = f"chapter_{chapter_num:04d}_{basename}"
                    output_path = os.path.join(out, output_filename)
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    # Add to chapters list
                    new_chapter = {
                        'num': chapter_num,
                        'title': title,
                        'body': content,
                        'filename': href,
                        'original_basename': basename,
                        'file_size': len(content),
                        'has_images': bool(soup.find_all('img')),
                        'detection_method': 'opf_recovery',
                        'content_hash': None  # Will be calculated later
                    }
                    
                    chapters.append(new_chapter)
                    print(f"   ✅ Recovered chapter {chapter_num}: {basename}")
                    
                except Exception as e:
                    print(f"   ❌ Failed to extract {href}: {e}")
            
            # Re-sort chapters by number
            chapters.sort(key=lambda x: x['num'])
            print(f"✅ Total chapters after OPF recovery: {len(chapters)}")
        
    except Exception as e:
        print(f"⚠️ Error checking OPF chapters: {e}")
        import traceback
        traceback.print_exc()
    
    return chapters
    
def extract_chapters(zf, output_dir, parser=None, progress_callback=None, pattern_manager=None):
    """Extract chapters and all resources from EPUB using ThreadPoolExecutor
    
    Args:
        zf: ZipFile object of the EPUB
        output_dir: Output directory for extracted files
        parser: BeautifulSoup parser to use ('lxml' or 'html.parser')
        progress_callback: Optional callback for progress updates
        pattern_manager: Optional PatternManager instance for chapter detection
    """
    import time
    
    # Initialize defaults if not provided
    if parser is None:
        parser = _get_best_parser()
    # pattern_manager is no longer used - kept for API compatibility
    
    # Check stop at the very beginning
    if is_stop_requested():
        print("❌ Extraction stopped by user")
        return []
        
    print("🚀 Starting EPUB extraction with ThreadPoolExecutor...")
    print(f"📄 Using parser: {parser} {'(optimized for CJK)' if parser == 'lxml' else '(standard)'}")
    
    # Initial progress
    if progress_callback:
        progress_callback("Starting EPUB extraction...")
    
    # First, extract and save content.opf for reference
    for name in zf.namelist():
        if name.endswith('.opf'):
            try:
                opf_content = zf.read(name).decode('utf-8', errors='ignore')
                opf_output_path = os.path.join(output_dir, 'content.opf')
                with open(opf_output_path, 'w', encoding='utf-8') as f:
                    f.write(opf_content)
                print(f"📋 Saved OPF file: {name} → content.opf")
                break
            except Exception as e:
                print(f"⚠️ Could not save OPF file: {e}")
    
    # Get extraction mode from environment
    extraction_mode = os.getenv("EXTRACTION_MODE", "smart").lower()
    print(f"✅ Using {extraction_mode.capitalize()} extraction mode")
    
    # Get number of workers from environment or use default
    max_workers = int(os.getenv("EXTRACTION_WORKERS", "2"))
    print(f"🔧 Using {max_workers} workers for parallel processing")
    
    extracted_resources = _extract_all_resources(zf, output_dir, progress_callback)
    
    # Check stop after resource extraction
    if is_stop_requested():
        print("❌ Extraction stopped by user")
        return []
    
    metadata_path = os.path.join(output_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        print("📋 Loading existing metadata...")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    else:
        print("📋 Extracting fresh metadata...")
        metadata = _extract_epub_metadata(zf)
        print(f"📋 Extracted metadata: {list(metadata.keys())}")
    
    chapters, detected_language = _extract_chapters_universal(zf, extraction_mode, parser, progress_callback, pattern_manager)
    
    # Sort chapters according to OPF spine order if available
    opf_path = os.path.join(output_dir, 'content.opf')
    if os.path.exists(opf_path) and chapters:
        print("📋 Sorting chapters according to OPF spine order...")
        chapters = _sort_by_opf_spine(chapters, opf_path)
        print(f"✅ Chapters sorted according to OPF reading order")
    
    # Check stop after chapter extraction
    if is_stop_requested():
        print("❌ Extraction stopped by user")
        return []
    
    if not chapters:
        print("❌ No chapters could be extracted!")
        return []
    
    # Rename images to chapter-based format (chapter001_img_1.jpg, etc.)
    chapters = _rename_images_to_chapter_format(chapters, output_dir, progress_callback)
    
    chapters_info_path = os.path.join(output_dir, 'chapters_info.json')
    chapters_info = []
    chapters_info_lock = threading.Lock()
    
    def process_chapter(chapter):
        """Process a single chapter"""
        # Check stop in worker
        if is_stop_requested():
            return None
            
        info = {
            'num': chapter['num'],
            'title': chapter['title'],
            'original_filename': chapter.get('filename', ''),
            'has_images': chapter.get('has_images', False),
            'image_count': chapter.get('image_count', 0),
            'text_length': chapter.get('file_size', len(chapter.get('body', ''))),
            'detection_method': chapter.get('detection_method', 'unknown'),
            'content_hash': chapter.get('content_hash', '')
        }
        
        if chapter.get('has_images'):
            try:
                soup = BeautifulSoup(chapter.get('body', ''), parser)
                images = soup.find_all('img')
                info['images'] = [img.get('src', '') for img in images]
            except:
                info['images'] = []
        
        return info
    
    # Process chapters in parallel
    print(f"🔄 Processing {len(chapters)} chapters in parallel...")
    
    if progress_callback:
        progress_callback(f"Processing {len(chapters)} chapters...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_chapter = {
            executor.submit(process_chapter, chapter): chapter 
            for chapter in chapters
        }
        
        # Process completed tasks
        completed = 0
        for future in as_completed(future_to_chapter):
            if is_stop_requested():
                print("❌ Extraction stopped by user")
                # Cancel remaining futures
                for f in future_to_chapter:
                    f.cancel()
                return []
            
            try:
                result = future.result()
                if result:
                    with chapters_info_lock:
                        chapters_info.append(result)
                    completed += 1
                    
                    # Yield to GUI periodically (can be disabled for max speed)
                    if completed % 5 == 0 and os.getenv("ENABLE_GUI_YIELD", "1") == "1":
                        time.sleep(0.001)
                    
                    # Progress updates
                    if completed % 10 == 0 or completed == len(chapters):
                        if progress_callback:
                            progress_msg = f"Processed {completed}/{len(chapters)} chapters"
                            progress_callback(progress_msg)
                        else:
                            # Show progress bar in terminal
                            ProgressBar.update(completed, len(chapters), prefix="📊 Processing metadata")
            except Exception as e:
                chapter = future_to_chapter[future]
                print(f"   ❌ Error processing chapter {chapter['num']}: {e}")
    
    # Finish progress bar
    if not progress_callback:
        ProgressBar.finish()
    
    # Sort chapters_info by chapter number to maintain order
    chapters_info.sort(key=lambda x: x['num'])
    
    print(f"✅ Successfully processed {len(chapters_info)} chapters")
    
    with open(chapters_info_path, 'w', encoding='utf-8') as f:
        json.dump(chapters_info, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Saved detailed chapter info to: chapters_info.json")
    
    metadata.update({
        'chapter_count': len(chapters),
        'detected_language': detected_language,
        'extracted_resources': extracted_resources,
        'extraction_mode': extraction_mode,
        'extraction_summary': {
            'total_chapters': len(chapters),
            'chapter_range': f"{chapters[0]['num']}-{chapters[-1]['num']}",
            'resources_extracted': sum(len(files) for files in extracted_resources.values())
        }
    })
    
    metadata['chapter_titles'] = {
        str(c['num']): c['title'] for c in chapters
    }
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Saved comprehensive metadata to: {metadata_path}")
    
    _create_extraction_report(output_dir, metadata, chapters, extracted_resources)
    _log_extraction_summary(chapters, extracted_resources, detected_language)
    
    print(f"🔍 VERIFICATION: {extraction_mode.capitalize()} chapter extraction completed successfully")
    print(f"⚡ Used {max_workers} workers for parallel processing")
    
    return chapters

def _extract_all_resources(zf, output_dir, progress_callback=None):
    """Extract all resources with parallel processing"""
    import time
    
    extracted_resources = {
        'css': [],
        'fonts': [],
        'images': [],
        'epub_structure': [],
        'other': []
    }
    
    # Check if already extracted
    extraction_marker = os.path.join(output_dir, '.resources_extracted')
    if os.path.exists(extraction_marker):
        print("📦 Resources already extracted, skipping...")
        return _count_existing_resources(output_dir, extracted_resources)
    
    _cleanup_old_resources(output_dir)
    
    # Create directories
    for resource_type in ['css', 'fonts', 'images']:
        os.makedirs(os.path.join(output_dir, resource_type), exist_ok=True)
    
    # Only print if no callback (avoid duplicates in subprocess)
    if not progress_callback:
        print(f"📦 Extracting resources in parallel...")
    
    # Get list of files to process
    file_list = [f for f in zf.namelist() if not f.endswith('/') and os.path.basename(f)]
    
    # Thread-safe lock for extracted_resources
    resource_lock = threading.Lock()
    
    def extract_single_resource(file_path):
        if is_stop_requested():
            return None
            
        try:
            file_data = zf.read(file_path)
            resource_info = _categorize_resource(file_path, os.path.basename(file_path))
            
            if resource_info:
                resource_type, target_dir, safe_filename = resource_info
                target_path = os.path.join(output_dir, target_dir, safe_filename) if target_dir else os.path.join(output_dir, safe_filename)
                
                with open(target_path, 'wb') as f:
                    f.write(file_data)
                
                # Thread-safe update
                with resource_lock:
                    extracted_resources[resource_type].append(safe_filename)
                
                return (resource_type, safe_filename)
        except Exception as e:
            print(f"[WARNING] Failed to extract {file_path}: {e}")
            return None
    
    # Process files in parallel
    total_resources = len(file_list)
    extracted_count = 0
    
    # Use same worker count as chapter processing
    resource_workers = int(os.getenv("EXTRACTION_WORKERS", "2"))
    
    with ThreadPoolExecutor(max_workers=resource_workers) as executor:
        futures = {executor.submit(extract_single_resource, file_path): file_path 
                  for file_path in file_list}
        
        for future in as_completed(futures):
            if is_stop_requested():
                executor.shutdown(wait=False)
                break
            
            extracted_count += 1
            
            # Progress update every 20 files
            if extracted_count % 20 == 0:
                if progress_callback:
                    progress_callback(f"Extracting resources: {extracted_count}/{total_resources}")
                else:
                    # Print progress bar in terminal
                    ProgressBar.update(extracted_count, total_resources, prefix="📦 Extracting resources")
            
            # Yield to GUI periodically (can be disabled for max speed)
            if extracted_count % 10 == 0 and os.getenv("ENABLE_GUI_YIELD", "1") == "1":
                time.sleep(0.001)
                
            result = future.result()
            if result:
                resource_type, filename = result
                # Only print for important resources
                if extracted_count < 10 or resource_type in ['css', 'fonts']:
                    print(f"   📄 Extracted {resource_type}: {filename}")
    
    # Show 100% completion
    if progress_callback:
        progress_callback(f"Extracting resources: {total_resources}/{total_resources}")
    else:
        ProgressBar.update(total_resources, total_resources, prefix="📦 Extracting resources")
        ProgressBar.finish()
    
    # Mark as complete
    with open(extraction_marker, 'w') as f:
        f.write(f"Resources extracted at {time.time()}")
    
    _validate_critical_files(output_dir, extracted_resources)
    return extracted_resources

def _extract_chapters_universal(zf, extraction_mode="smart", parser=None, progress_callback=None, pattern_manager=None):
    """Universal chapter extraction with four modes: smart, comprehensive, full, enhanced
    
    All modes now properly merge Section/Chapter pairs
    Enhanced mode uses html2text for superior text processing
    Now with parallel processing for improved performance
    """
    # Initialize defaults if not provided
    if parser is None:
        parser = _get_best_parser()
    # pattern_manager is no longer used - kept for API compatibility
    
    # Check stop at the beginning
    if is_stop_requested():
        print("❌ Chapter extraction stopped by user")
        return [], 'unknown'
    
    # Import time for yielding
    import time
    
    # Initialize enhanced extractor if using enhanced mode
    enhanced_extractor = None
    enhanced_filtering = extraction_mode  # Default fallback
    preserve_structure = True
    
    # Check if user wants to translate special files (info.xhtml, message.xhtml, etc.)
    # By default, skip them as they're typically metadata/navigation
    translate_special = os.getenv('TRANSLATE_SPECIAL_FILES', '0') == '1'
    
    if translate_special:
        print("📝 Special files translation is ENABLED (info.xhtml, message.xhtml, etc.)")
    else:
        print("📝 Special files translation is DISABLED - skipping navigation/metadata files")
    
    if extraction_mode == "enhanced":
        print("🚀 Initializing Enhanced extraction mode with html2text...")
        
        # Get enhanced mode configuration from environment
        enhanced_filtering = os.getenv("ENHANCED_FILTERING", "smart")
        # Avoid 'full' with html2text to prevent XML declaration artifacts; use 'comprehensive' instead
        if str(enhanced_filtering).lower() == 'full':
            enhanced_filtering = 'comprehensive'
        preserve_structure = os.getenv("ENHANCED_PRESERVE_STRUCTURE", "1") == "1"
        
        print(f"  • Enhanced filtering level: {enhanced_filtering}")
        print(f"  • Preserve structure: {preserve_structure}")
        
        # Try to initialize enhanced extractor
        try:
            # Import our enhanced extractor (assume it's in the same directory or importable)
            from enhanced_text_extractor import EnhancedTextExtractor
            enhanced_extractor = EnhancedTextExtractor(
                filtering_mode=enhanced_filtering,
                preserve_structure=preserve_structure
            )
            print("✅ Enhanced text extractor initialized successfully")
                
        except ImportError as e:
            print(f"❌ Enhanced text extractor module not found: {e}")
            print(f"❌ Cannot use enhanced extraction mode. Please install enhanced_text_extractor or select a different extraction mode.")
            raise e
        except Exception as e:
            print(f"❌ Enhanced extractor initialization failed: {e}")
            print(f"❌ Cannot use enhanced extraction mode. Please select a different extraction mode.")
            raise e
    
    chapters = []
    sample_texts = []
    
    # First phase: Collect HTML files
    html_files = []
    file_list = zf.namelist()
    total_files = len(file_list)
    
    # Update progress for file collection
    if progress_callback and total_files > 100:
        progress_callback(f"Scanning {total_files} files in EPUB...")
    elif total_files > 100 and not progress_callback:
        # Print initial message for progress bar (only if no callback)
        print(f"📂 Scanning {total_files} files in EPUB...")
    
    for idx, name in enumerate(file_list):
        # Check stop while collecting files
        if is_stop_requested():
            print("❌ Chapter extraction stopped by user")
            return [], 'unknown'
        
        # Yield to GUI every 50 files (can be disabled for max speed)
        if idx % 50 == 0 and idx > 0:
            if os.getenv("ENABLE_GUI_YIELD", "1") == "1":
                time.sleep(0.001)  # Brief yield to GUI
            if total_files > 100:
                if progress_callback:
                    progress_callback(f"Scanning files: {idx}/{total_files}")
                else:
                    # Print progress bar in terminal
                    ProgressBar.update(idx, total_files, prefix="📂 Scanning files")
            
        if name.lower().endswith(('.xhtml', '.html', '.htm')):
            basename = os.path.basename(name).lower()
            
            # Skip cover files unless special file translation is enabled
            if basename in ['cover.html', 'cover.xhtml', 'cover.htm']:
                if not translate_special:
                    print(f"[SKIP] Cover file excluded: {name}")
                    continue
                else:
                    print(f"[INCLUDE] Cover file included (special files enabled): {name}")
            
            # All filtering is now controlled by TRANSLATE_SPECIAL_FILES toggle and extraction mode
            # No hardcoded special file patterns
            html_files.append(name)
    
    # Print final 100% progress update before finishing
    if total_files > 100:
        if progress_callback:
            progress_callback(f"Scanning files: {total_files}/{total_files}")
        else:
            # Show 100% completion
            ProgressBar.update(total_files, total_files, prefix="📂 Scanning files")
    
    # Finish progress bar if we were using it
    if total_files > 100 and not progress_callback:
        ProgressBar.finish()
    
    # Update mode description to include enhanced mode
    mode_description = {
        "smart": "potential content files",
        "comprehensive": "HTML files", 
        "full": "ALL HTML/XHTML files (no filtering)",
        "enhanced": f"files (enhanced with {enhanced_filtering} filtering)"
    }
    print(f"📚 Found {len(html_files)} {mode_description.get(extraction_mode, 'files')} in EPUB")
    
    # Sort files to ensure proper order
    html_files.sort()
    
    # Check if merging is disabled via environment variable
    disable_merging = os.getenv("DISABLE_CHAPTER_MERGING", "0") == "1"
    
    processed_files = set()
    merge_candidates = {}  # Store potential merges without reading files yet
    
    if disable_merging:
        print("📌 Chapter merging is DISABLED - processing all files independently")
    else:
        print("📌 Chapter merging is ENABLED")
        
        # Only do merging logic if not disabled
        file_groups = {}
        
        # Group files by their base number to detect Section/Chapter pairs
        for file_path in html_files:
            filename = os.path.basename(file_path)
            
            # Try different patterns to extract base number
            base_num = None
            
            # Pattern 1: "No00014" from "No00014Section.xhtml"
            match = re.match(r'(No\d+)', filename)
            if match:
                base_num = match.group(1)
            else:
                # Pattern 2: "0014" from "0014_section.html" or "0014_chapter.html"
                match = re.match(r'^(\d+)[_\-]', filename)
                if match:
                    base_num = match.group(1)
                else:
                    # Pattern 3: Just numbers at the start
                    match = re.match(r'^(\d+)', filename)
                    if match:
                        base_num = match.group(1)
            
            if base_num:
                if base_num not in file_groups:
                    file_groups[base_num] = []
                file_groups[base_num].append(file_path)
        
        # Identify merge candidates WITHOUT reading files yet
        for base_num, group_files in sorted(file_groups.items()):
            if len(group_files) == 2:
                # Check if we have a Section/Chapter pair based on filenames only
                section_file = None
                chapter_file = None
                
                for file_path in group_files:
                    basename = os.path.basename(file_path)
                    # More strict detection - must have 'section' or 'chapter' in the filename
                    if 'section' in basename.lower() and 'chapter' not in basename.lower():
                        section_file = file_path
                    elif 'chapter' in basename.lower() and 'section' not in basename.lower():
                        chapter_file = file_path
                
                if section_file and chapter_file:
                    # Store as potential merge candidate
                    merge_candidates[chapter_file] = section_file
                    processed_files.add(section_file)
                    print(f"[DEBUG] Potential merge candidate: {base_num}")
                    print(f"  Section: {os.path.basename(section_file)}")
                    print(f"  Chapter: {os.path.basename(chapter_file)}")
    
    # Filter out section files that were marked for merging
    files_to_process = []
    for file_path in html_files:
        if not disable_merging and file_path in processed_files:
            print(f"[DEBUG] Skipping section file: {file_path}")
            continue
        files_to_process.append(file_path)
    
    print(f"📚 Processing {len(files_to_process)} files after merge analysis")
    if progress_callback:
        progress_callback(f"Preparing to process {len(files_to_process)} chapters...")
    
    # Initialize collections for aggregating results
    file_size_groups = {}
    h1_count = 0
    h2_count = 0
    skipped_files = []
    
    # Progress tracking
    total_files = len(files_to_process)
    
    # Prepare arguments for parallel processing
    zip_file_path = zf.filename
    
    # Process files in parallel or sequentially based on file count
    # Only print if no callback (avoid duplicates)
    if not progress_callback:
        print(f"🚀 Processing {len(files_to_process)} HTML files...")
    
    # Initial progress - no message needed, progress bar will show
    
    candidate_chapters = []  # For smart mode
    chapters_direct = []      # For other modes
    
    # Decide whether to use parallel processing
    use_parallel = len(files_to_process) > 10
    
    if use_parallel:
        # Get worker count from environment variable
        max_workers = int(os.getenv("EXTRACTION_WORKERS", "2"))
        print(f"📦 Using parallel processing with {max_workers} workers...")
        if progress_callback:
            progress_callback(f"Starting {max_workers} extraction workers...")
        
        # --- Heartbeat: show elapsed time while workers start up ----
        _heartbeat_stop = threading.Event()
        _startup_start = time.time()

        def _heartbeat():
            elapsed = 0
            while not _heartbeat_stop.is_set():
                _heartbeat_stop.wait(3.0)       # tick every 3 s
                if _heartbeat_stop.is_set():
                    break
                elapsed = time.time() - _startup_start
                msg = f"⏱️ Spawning workers... elapsed {elapsed:.0f}s"
                if progress_callback:
                    progress_callback(msg)
                else:
                    print(msg, flush=True)

        _hb_thread = threading.Thread(target=_heartbeat, daemon=True)
        _hb_thread.start()
        # -----------------------------------------------------------

        # Use ProcessPoolExecutor for true multi-process parallelism
        # Now that all functions are at module level and picklable, we can use processes
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all files for processing
            future_to_file = {
                executor.submit(
                    _process_single_html_file,
                    file_path=file_path,
                    file_index=idx,
                    zip_file_path=zip_file_path,
                    parser=parser,
                    merge_candidates=merge_candidates,
                    disable_merging=disable_merging,
                    enhanced_extractor=enhanced_extractor,
                    extraction_mode=extraction_mode,
                    enhanced_filtering=enhanced_filtering,
                    preserve_structure=preserve_structure,
                    protect_angle_brackets_func=protect_angle_brackets_with_korean,
                    pattern_manager=pattern_manager,
                    files_to_process=files_to_process,
                    is_stop_requested=is_stop_requested
                ): (file_path, idx)
                for idx, file_path in enumerate(files_to_process)
            }
            # Collect results as they complete with progress tracking
            processed_count = 0
            for future in as_completed(future_to_file):
                # Stop heartbeat on first result (workers are now active)
                if not _heartbeat_stop.is_set():
                    _heartbeat_stop.set()
                    _startup_elapsed = time.time() - _startup_start
                    if _startup_elapsed >= 2.0:
                        msg = f"✅ Workers ready ({_startup_elapsed:.1f}s)"
                        if progress_callback:
                            progress_callback(msg)
                        else:
                            print(msg, flush=True)
                if is_stop_requested():
                    print("❌ Chapter processing stopped by user")
                    executor.shutdown(wait=False)
                    return [], 'unknown'
                
                try:
                    # Unpack result from _process_single_html_file
                    result = future.result()
                    chapter_info, h1_found, h2_found, file_size, sample_text, skipped_info = result
                    
                    # Update progress
                    processed_count += 1
                    if processed_count % 5 == 0:
                        if progress_callback:
                            progress_msg = f"Processing chapters: {processed_count}/{total_files} ({processed_count*100//total_files}%)"
                            progress_callback(progress_msg)
                        else:
                            # Print progress bar in terminal
                            ProgressBar.update(processed_count, total_files, prefix="📚 Processing chapters")
                    
                    # Aggregate header counts
                    if h1_found:
                        h1_count += 1
                    if h2_found:
                        h2_count += 1
                    
                    # Collect file size groups and sample texts
                    if chapter_info:
                        effective_mode = enhanced_filtering if extraction_mode == "enhanced" else extraction_mode
                        if effective_mode == "smart" and file_size > 0:
                            if file_size not in file_size_groups:
                                file_size_groups[file_size] = []
                            file_path, _ = future_to_file[future]
                            file_size_groups[file_size].append(file_path)
                            
                            # Collect sample texts
                            if sample_text and len(sample_texts) < 5:
                                sample_texts.append(sample_text)
                        
                        # For smart mode when merging is enabled, collect candidates
                        # Otherwise, add directly to chapters
                        if effective_mode == "smart" and not disable_merging:
                            candidate_chapters.append(chapter_info)
                        else:
                            chapters_direct.append(chapter_info)
                    
                    # Collect skipped info
                    if skipped_info:
                        skipped_files.append(skipped_info)
                        
                except Exception as e:
                    file_path, idx = future_to_file[future]
                    print(f"[ERROR] Process error processing {file_path}: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Show 100% completion
        if progress_callback:
            progress_callback(f"Processing chapters: {total_files}/{total_files} (100%)")
        else:
            ProgressBar.update(total_files, total_files, prefix="📚 Processing chapters")
    else:
        print("📦 Using sequential processing (small file count)...")
        
        # Process files sequentially for small EPUBs
        for idx, file_path in enumerate(files_to_process):
            if is_stop_requested():
                print("❌ Chapter processing stopped by user")
                return [], 'unknown'
            
            # Call the module-level function directly
            result = _process_single_html_file(
                file_path=file_path,
                file_index=idx,
                zip_file_path=zip_file_path,
                parser=parser,
                merge_candidates=merge_candidates,
                disable_merging=disable_merging,
                enhanced_extractor=enhanced_extractor,
                extraction_mode=extraction_mode,
                enhanced_filtering=enhanced_filtering,
                preserve_structure=preserve_structure,
                protect_angle_brackets_func=protect_angle_brackets_with_korean,
                pattern_manager=pattern_manager,
                files_to_process=files_to_process,
                is_stop_requested=is_stop_requested
            )
            
            # Unpack result
            chapter_info, h1_found, h2_found, file_size, sample_text, skipped_info = result
            
            # Update progress
            if (idx + 1) % 5 == 0:
                if progress_callback:
                    progress_msg = f"Processing chapters: {idx+1}/{total_files} ({(idx+1)*100//total_files}%)"
                    progress_callback(progress_msg)
                else:
                    # Print progress bar in terminal
                    ProgressBar.update(idx+1, total_files, prefix="📚 Processing chapters")
            
            # Aggregate header counts
            if h1_found:
                h1_count += 1
            if h2_found:
                h2_count += 1
            
            # Collect file size groups and sample texts
            if chapter_info:
                effective_mode = enhanced_filtering if extraction_mode == "enhanced" else extraction_mode
                if effective_mode == "smart" and file_size > 0:
                    if file_size not in file_size_groups:
                        file_size_groups[file_size] = []
                    file_size_groups[file_size].append(file_path)
                    
                    # Collect sample texts
                    if sample_text and len(sample_texts) < 5:
                        sample_texts.append(sample_text)
                
                # For smart mode when merging is enabled, collect candidates
                # Otherwise, add directly to chapters
                if effective_mode == "smart" and not disable_merging:
                    candidate_chapters.append(chapter_info)
                else:
                    chapters_direct.append(chapter_info)
            
            # Collect skipped info
            if skipped_info:
                skipped_files.append(skipped_info)
        
        # Show 100% completion for sequential mode
        if progress_callback:
            progress_callback(f"Processing chapters: {total_files}/{total_files} (100%)")
        else:
            ProgressBar.update(total_files, total_files, prefix="📚 Processing chapters")
    
    # Final progress update and cleanup progress bar
    if not progress_callback:
        ProgressBar.finish()
    else:
        progress_callback(f"Chapter processing complete: {len(candidate_chapters) + len(chapters_direct)} chapters")
    
    import time as _post_time
    _post_start = _post_time.time()
    
    # Print skip summary if any files were skipped
    if skipped_files:
        print(f"\n📊 Skipped {len(skipped_files)} files during processing:")
        empty_count = sum(1 for _, reason, _ in skipped_files if reason == 'empty')
        if empty_count > 0:
            print(f"   • {empty_count} nearly empty files")
        # Show first 3 examples if debug enabled
        if os.getenv('DEBUG_SKIP_MESSAGES', '0') == '1' and skipped_files:
            print("   Examples:")
            for path, reason, size in skipped_files[:3]:
                print(f"     - {os.path.basename(path)} ({size} chars)")
    
    # Sort direct chapters by file index to maintain order
    chapters_direct.sort(key=lambda x: x["file_index"])
    
    _sort_elapsed = _post_time.time() - _post_start
    if _sort_elapsed > 1.0:
        msg = f"  ⏱️ Post-processing: sorting ({_sort_elapsed:.1f}s)"
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)
    
    # Post-process smart mode candidates (only when merging is enabled)
    effective_mode = enhanced_filtering if extraction_mode == "enhanced" else extraction_mode
    if effective_mode == "smart" and candidate_chapters and not disable_merging:
        # Check stop before post-processing
        if is_stop_requested():
            print("❌ Chapter post-processing stopped by user")
            return chapters, 'unknown'
            
        print(f"\n[SMART MODE] Processing {len(candidate_chapters)} candidate files...")
        
        # Sort candidates by file index to maintain order
        candidate_chapters.sort(key=lambda x: x["file_index"])
        
        # Debug: Show what files we have
        section_files = [c for c in candidate_chapters if 'section' in c['original_basename'].lower()]
        chapter_files = [c for c in candidate_chapters if 'chapter' in c['original_basename'].lower() and 'section' not in c['original_basename'].lower()]
        other_files = [c for c in candidate_chapters if c not in section_files and c not in chapter_files]
        
        print(f"  📊 File breakdown:")
        print(f"    • Section files: {len(section_files)}")
        print(f"    • Chapter files: {len(chapter_files)}")
        print(f"    • Other files: {len(other_files)}")
        
        # Original smart mode logic when merging is enabled
        # First, separate files with detected chapter numbers from those without
        numbered_chapters = []
        unnumbered_chapters = []
        
        for idx, chapter in enumerate(candidate_chapters):
            # Yield periodically during categorization (can be disabled for max speed)
            if idx % 10 == 0 and idx > 0 and os.getenv("ENABLE_GUI_YIELD", "1") == "1":
                time.sleep(0.001)
                
            if chapter["num"] is not None:
                numbered_chapters.append(chapter)
            else:
                unnumbered_chapters.append(chapter)
        
        print(f"  • Files with chapter numbers: {len(numbered_chapters)}")
        print(f"  • Files without chapter numbers: {len(unnumbered_chapters)}")
        
        # Check if we have hash-based filenames (no numbered chapters found)
        if not numbered_chapters and unnumbered_chapters:
            print("  ⚠️ No chapter numbers found - likely hash-based filenames")
            print("  → Using file order as chapter sequence")
            
            # Sort by file index to maintain order
            unnumbered_chapters.sort(key=lambda x: x["file_index"])
            
            # Assign sequential numbers
            for i, chapter in enumerate(unnumbered_chapters, 1):
                chapter["num"] = i
                chapter["detection_method"] = f"{extraction_mode}_hash_filename_sequential" if extraction_mode == "enhanced" else "hash_filename_sequential"
                if not chapter["title"] or chapter["title"] == chapter["original_basename"]:
                    chapter["title"] = f"Chapter {i}"
            
            chapters = unnumbered_chapters
        else:
            # We have some numbered chapters
            chapters = numbered_chapters
            
            # For unnumbered files, check if they might be duplicates or appendices
            if unnumbered_chapters:
                print(f"  → Analyzing {len(unnumbered_chapters)} unnumbered files...")
                
                # Get the max chapter number
                max_num = max(c["num"] for c in numbered_chapters)
                
                # Check each unnumbered file
                for chapter in unnumbered_chapters:
                    # Check stop in post-processing loop
                    if is_stop_requested():
                        print("❌ Chapter post-processing stopped by user")
                        return chapters, 'unknown'
                        
                    # Check if it's very small (might be a separator or note)
                    if chapter["file_size"] < 200:
                        # Collect for summary instead of printing
                        # Note: _smart_mode_skips defined in outer scope
                        _smart_mode_skips.append(('small', chapter['filename'], chapter['file_size']))
                        continue
                    
                    # Check if it has similar size to existing chapters (might be duplicate)
                    size = chapter["file_size"]
                    similar_chapters = [c for c in numbered_chapters 
                                      if abs(c["file_size"] - size) < 50]
                    
                    if similar_chapters:
                        # Might be a duplicate, skip it (collect for summary)
                        _smart_mode_skips.append(('duplicate', chapter['filename'], len(similar_chapters)))
                        continue
                    
                    # Otherwise, add as appendix
                    max_num += 1
                    chapter["num"] = max_num
                    chapter["detection_method"] = f"{extraction_mode}_appendix_sequential" if extraction_mode == "enhanced" else "appendix_sequential"
                    if not chapter["title"] or chapter["title"] == chapter["original_basename"]:
                        chapter["title"] = f"Appendix {max_num}"
                    chapters.append(chapter)
                    print(f"    [ADD] Added as chapter {max_num}: {chapter['filename']}")
    else:
        # For other modes or smart mode with merging disabled
        chapters = chapters_direct
    
    # Print smart mode skip summary if any
    if '_smart_mode_skips' in locals() and _smart_mode_skips:
        print(f"\n📊 Smart mode filtering summary:")
        small_count = sum(1 for reason, _, _ in _smart_mode_skips if reason == 'small')
        dup_count = sum(1 for reason, _, _ in _smart_mode_skips if reason == 'duplicate')
        if small_count > 0:
            print(f"   • Skipped {small_count} very small files")
        if dup_count > 0:
            print(f"   • Skipped {dup_count} possible duplicates")
        # Show examples if debug enabled
        if os.getenv('DEBUG_SKIP_MESSAGES', '0') == '1':
            print("   Examples:")
            for reason, filename, detail in _smart_mode_skips[:3]:
                if reason == 'small':
                    print(f"     - {filename} ({detail} chars)")
                else:
                    print(f"     - {filename} (similar to {detail} chapters)")
        # Clear the list
        _smart_mode_skips = []
    
    # Sort chapters by number
    chapters.sort(key=lambda x: x["num"])
    
    # Ensure chapter numbers are integers
    # When merging is disabled, all chapters should have integer numbers anyway
    for chapter in chapters:
        if isinstance(chapter["num"], float):
            chapter["num"] = int(chapter["num"])
    
    # Final validation
    if chapters:
        print(f"\n✅ Final chapter count: {len(chapters)}")
        print(f"   • Chapter range: {chapters[0]['num']} - {chapters[-1]['num']}")
        
        # Enhanced mode summary
        if extraction_mode == "enhanced":
            enhanced_count = sum(1 for c in chapters if c.get('enhanced_extraction', False))
            total_chars = sum(len(c.get('body', '')) for c in chapters if c.get('enhanced_extraction', False))
            avg_chars = total_chars // enhanced_count if enhanced_count > 0 else 0
            print(f"   🚀 Enhanced extraction: {enhanced_count}/{len(chapters)} chapters, {total_chars:,} total chars (avg: {avg_chars:,})")
        
        # Check for gaps
        chapter_nums = [c["num"] for c in chapters]
        expected_nums = list(range(min(chapter_nums), max(chapter_nums) + 1))
        missing = set(expected_nums) - set(chapter_nums)
        if missing:
            print(f"   ⚠️ Missing chapter numbers: {sorted(missing)}")
    
    # Language detection
    combined_sample = ' '.join(sample_texts) if effective_mode == "smart" else ''
    detected_language = _detect_content_language(combined_sample) if combined_sample else 'unknown'
    
    _total_post = _post_time.time() - _post_start
    if _total_post > 1.0:
        msg = f"  ⏱️ Post-processing complete ({_total_post:.1f}s)"
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)
    
    if chapters:
        _print_extraction_summary(chapters, detected_language, extraction_mode, 
                                     h1_count if effective_mode == "smart" else 0, 
                                     h2_count if effective_mode == "smart" else 0,
                                     file_size_groups if effective_mode == "smart" else {})
    
    return chapters, detected_language

def _extract_chapter_info(soup, file_path, content_text, html_content, pattern_manager):
    """Extract chapter number and title from various sources with parallel pattern matching"""
    chapter_num = None
    chapter_title = None
    detection_method = None
    
    # SPECIAL HANDLING: When we have Section/Chapter pairs, differentiate them
    filename = os.path.basename(file_path)
    
    # Handle different naming patterns for Section/Chapter files
    if ('section' in filename.lower() or '_section' in filename.lower()) and 'chapter' not in filename.lower():
        # For Section files, add 0.1 to the base number
        # Try different patterns
        match = re.search(r'No(\d+)', filename)
        if not match:
            match = re.search(r'^(\d+)[_\-]', filename)
        if not match:
            match = re.search(r'^(\d+)', filename)
            
        if match:
            base_num = int(match.group(1))
            chapter_num = base_num + 0.1  # Section gets .1
            detection_method = "filename_section_special"
            
    elif ('chapter' in filename.lower() or '_chapter' in filename.lower()) and 'section' not in filename.lower():
        # For Chapter files, use the base number
        # Try different patterns
        match = re.search(r'No(\d+)', filename)
        if not match:
            match = re.search(r'^(\d+)[_\-]', filename)
        if not match:
            match = re.search(r'^(\d+)', filename)
            
        if match:
            chapter_num = int(match.group(1))
            detection_method = "filename_chapter_special"
    
    # If not handled by special logic, continue with normal extraction
    if not chapter_num:
        # Try filename first - use parallel pattern matching for better performance
        chapter_patterns = [(pattern, flags, method) for pattern, flags, method in _CHAPTER_PATTERNS 
                          if method.endswith('_number')]
        
        if len(chapter_patterns) > 3:  # Only parallelize if we have enough patterns
            # Parallel pattern matching for filename
            with ThreadPoolExecutor(max_workers=min(4, len(chapter_patterns))) as executor:
                def try_pattern(pattern_info):
                    pattern, flags, method = pattern_info
                    match = re.search(pattern, file_path, flags)
                    if match:
                        try:
                            num_str = match.group(1)
                            if num_str.isdigit():
                                return int(num_str), f"filename_{method}"
                            elif method == 'chinese_chapter_cn':
                                pass  # TransateKRtoEN import removed (patterns inlined)
                                pm = None  # No longer needed
                                converted = _convert_chinese_number(num_str, pm)
                                if converted:
                                    return converted, f"filename_{method}"
                        except (ValueError, IndexError):
                            pass
                    return None, None
                
                # Submit all patterns
                futures = [executor.submit(try_pattern, pattern_info) for pattern_info in chapter_patterns]
                
                # Check results as they complete
                for future in as_completed(futures):
                    try:
                        num, method = future.result()
                        if num:
                            chapter_num = num
                            detection_method = method
                            # Cancel remaining futures
                            for f in futures:
                                f.cancel()
                            break
                    except Exception:
                        continue
        else:
            # Sequential processing for small pattern sets
            for pattern, flags, method in chapter_patterns:
                match = re.search(pattern, file_path, flags)
                if match:
                    try:
                        num_str = match.group(1)
                        if num_str.isdigit():
                            chapter_num = int(num_str)
                            detection_method = f"filename_{method}"
                            break
                        elif method == 'chinese_chapter_cn':
                            pass  # TransateKRtoEN import removed (patterns inlined)
                            pm = None  # No longer needed
                            converted = _convert_chinese_number(num_str, pm)
                            if converted:
                                chapter_num = converted
                                detection_method = f"filename_{method}"
                                break
                    except (ValueError, IndexError):
                        continue
    
    # Try content if not found in filename
    if not chapter_num:
        # Check ignore settings for batch translation
        batch_translate_active = os.getenv('BATCH_TRANSLATE_HEADERS', '0') == '1'
        use_title_tag = os.getenv('USE_TITLE', '0') == '1' or not batch_translate_active
        ignore_header_tags = os.getenv('IGNORE_HEADER', '0') == '1' and batch_translate_active
        
        # Prepare all text sources to check in parallel
        text_sources = []
        
        # Add title tag if using titles
        if use_title_tag and soup.title and soup.title.string:
            title_text = soup.title.string.strip()
            text_sources.append(("title", title_text, True))  # True means this can be chapter_title
        
        # Add headers if not ignored
        if not ignore_header_tags:
            for header_tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                headers = soup.find_all(header_tag)
                for header in headers[:3]:  # Limit to first 3 of each type
                    header_text = header.get_text(strip=True)
                    if header_text:
                        text_sources.append((f"header_{header_tag}", header_text, True))
        
        # Add first paragraphs
        first_elements = soup.find_all(['p', 'div'])[:5]
        for elem in first_elements:
            elem_text = elem.get_text(strip=True)
            if elem_text:
                text_sources.append(("content", elem_text, False))  # False means don't use as chapter_title
        
        # Process text sources in parallel if we have many
        if len(text_sources) > 5:
            with ThreadPoolExecutor(max_workers=min(6, len(text_sources))) as executor:
                def extract_from_source(source_info):
                    source_type, text, can_be_title = source_info
                    num, method = _extract_from_text(text, source_type, pattern_manager)
                    return num, method, text if (num and can_be_title) else None
                
                # Submit all text sources
                future_to_source = {executor.submit(extract_from_source, source): source 
                                  for source in text_sources}
                
                # Process results as they complete
                for future in as_completed(future_to_source):
                    try:
                        num, method, title = future.result()
                        if num:
                            chapter_num = num
                            detection_method = method
                            if title and not chapter_title:
                                chapter_title = title
                            # Cancel remaining futures
                            for f in future_to_source:
                                f.cancel()
                            break
                    except Exception:
                        continue
        else:
            # Sequential processing for small text sets
            for source_type, text, can_be_title in text_sources:
                num, method = _extract_from_text(text, source_type, pattern_manager)
                if num:
                    chapter_num = num
                    detection_method = method
                    if can_be_title and not chapter_title:
                        chapter_title = text
                    break
        
        # Final fallback to filename patterns
        if not chapter_num:
            filename_base = os.path.basename(file_path)
            # Parallel pattern matching for filename extraction
            if len(_FILENAME_EXTRACT_PATTERNS) > 3:
                with ThreadPoolExecutor(max_workers=min(4, len(_FILENAME_EXTRACT_PATTERNS))) as executor:
                    def try_filename_pattern(pattern):
                        match = re.search(pattern, filename_base, re.IGNORECASE)
                        if match:
                            try:
                                return int(match.group(1))
                            except (ValueError, IndexError):
                                pass
                        return None
                    
                    futures = [executor.submit(try_filename_pattern, pattern) 
                             for pattern in _FILENAME_EXTRACT_PATTERNS]
                    
                    for future in as_completed(futures):
                        try:
                            num = future.result()
                            if num:
                                chapter_num = num
                                detection_method = "filename_number"
                                for f in futures:
                                    f.cancel()
                                break
                        except Exception:
                            continue
            else:
                # Sequential for small pattern sets
                for pattern in _FILENAME_EXTRACT_PATTERNS:
                    match = re.search(pattern, filename_base, re.IGNORECASE)
                    if match:
                        chapter_num = int(match.group(1))
                        detection_method = "filename_number"
                        break
    
    # Extract title if not already found (with ignore settings support)
    if not chapter_title:
        # Check settings for batch translation
        batch_translate_active = os.getenv('BATCH_TRANSLATE_HEADERS', '0') == '1'
        use_title_tag = os.getenv('USE_TITLE', '0') == '1' or not batch_translate_active
        ignore_header_tags = os.getenv('IGNORE_HEADER', '0') == '1' and batch_translate_active
        
        # Try title tag if using titles
        if use_title_tag and soup.title and soup.title.string:
            chapter_title = soup.title.string.strip()
        
        # Try header tags if not ignored and no title found
        if not chapter_title and not ignore_header_tags:
            for header_tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                header = soup.find(header_tag)
                if header:
                    chapter_title = header.get_text(strip=True)
                    break
        
        # Final fallback
        if not chapter_title:
            chapter_title = f"Chapter {chapter_num}" if chapter_num else None
    
    chapter_title = re.sub(r'\s+', ' ', chapter_title).strip() if chapter_title else None
    
    return chapter_num, chapter_title, detection_method


def _extract_from_text(text, source_type, pattern_manager):
    """Extract chapter number from text using patterns with parallel matching for large pattern sets"""
    # Get patterns that don't end with '_number'
    text_patterns = [(pattern, flags, method) for pattern, flags, method in _CHAPTER_PATTERNS
                    if not method.endswith('_number')]
    
    # Only use parallel processing if we have many patterns
    if len(text_patterns) > 5:
        with ThreadPoolExecutor(max_workers=min(4, len(text_patterns))) as executor:
            def try_text_pattern(pattern_info):
                pattern, flags, method = pattern_info
                match = re.search(pattern, text, flags)
                if match:
                    try:
                        num_str = match.group(1)
                        if num_str.isdigit():
                            return int(num_str), f"{source_type}_{method}"
                        elif method == 'chinese_chapter_cn':
                            pass  # TransateKRtoEN import removed (patterns inlined)
                            pm = None  # No longer needed
                            converted = _convert_chinese_number(num_str, pm)
                            if converted:
                                return converted, f"{source_type}_{method}"
                    except (ValueError, IndexError):
                        pass
                return None, None
            
            # Submit all patterns
            futures = [executor.submit(try_text_pattern, pattern_info) for pattern_info in text_patterns]
            
            # Check results as they complete
            for future in as_completed(futures):
                try:
                    num, method = future.result()
                    if num:
                        # Cancel remaining futures
                        for f in futures:
                            f.cancel()
                        return num, method
                except Exception:
                    continue
    else:
        # Sequential processing for small pattern sets
        for pattern, flags, method in text_patterns:
            match = re.search(pattern, text, flags)
            if match:
                try:
                    num_str = match.group(1)
                    if num_str.isdigit():
                        return int(num_str), f"{source_type}_{method}"
                    elif method == 'chinese_chapter_cn':
                        pass  # TransateKRtoEN import removed (patterns inlined)
                        pm = None  # No longer needed
                        converted = _convert_chinese_number(num_str, pm)
                        if converted:
                            return converted, f"{source_type}_{method}"
                except (ValueError, IndexError):
                    continue
    
    return None, None

def _convert_chinese_number(cn_num, pattern_manager):
    """Convert Chinese number to integer"""
    if cn_num in _CHINESE_NUMS:
        return _CHINESE_NUMS[cn_num]
    
    if '十' in cn_num:
        parts = cn_num.split('十')
        if len(parts) == 2:
            tens = _CHINESE_NUMS.get(parts[0], 1) if parts[0] else 1
            ones = _CHINESE_NUMS.get(parts[1], 0) if parts[1] else 0
            return tens * 10 + ones
    
    return None

def _detect_content_language( text_sample):
    """Detect the primary language of content with parallel processing for large texts"""
    
    # For very short texts, use sequential processing
    if len(text_sample) < 1000:
        scripts = {
            'korean': 0,
            'japanese_hiragana': 0,
            'japanese_katakana': 0,
            'chinese': 0,
            'latin': 0
        }
        
        for char in text_sample:
            code = ord(char)
            if 0xAC00 <= code <= 0xD7AF:
                scripts['korean'] += 1
            elif 0x3040 <= code <= 0x309F:
                scripts['japanese_hiragana'] += 1
            elif 0x30A0 <= code <= 0x30FF:
                scripts['japanese_katakana'] += 1
            elif 0x4E00 <= code <= 0x9FFF:
                scripts['chinese'] += 1
            elif 0x0020 <= code <= 0x007F:
                scripts['latin'] += 1
    else:
        # For longer texts, use parallel processing
        # Split text into chunks for parallel processing
        chunk_size = max(500, len(text_sample) // (os.cpu_count() or 4))
        chunks = [text_sample[i:i + chunk_size] for i in range(0, len(text_sample), chunk_size)]
        
        # Thread-safe accumulator
        scripts_lock = threading.Lock()
        scripts = {
            'korean': 0,
            'japanese_hiragana': 0,
            'japanese_katakana': 0,
            'chinese': 0,
            'latin': 0
        }
        
        def process_chunk(text_chunk):
            """Process a chunk of text and return script counts"""
            local_scripts = {
                'korean': 0,
                'japanese_hiragana': 0,
                'japanese_katakana': 0,
                'chinese': 0,
                'latin': 0
            }
            
            for char in text_chunk:
                code = ord(char)
                if 0xAC00 <= code <= 0xD7AF:
                    local_scripts['korean'] += 1
                elif 0x3040 <= code <= 0x309F:
                    local_scripts['japanese_hiragana'] += 1
                elif 0x30A0 <= code <= 0x30FF:
                    local_scripts['japanese_katakana'] += 1
                elif 0x4E00 <= code <= 0x9FFF:
                    local_scripts['chinese'] += 1
                elif 0x0020 <= code <= 0x007F:
                    local_scripts['latin'] += 1
            
            return local_scripts
        
        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=min(os.cpu_count() or 4, len(chunks))) as executor:
            # Submit all chunks
            futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
            
            # Collect results
            for future in as_completed(futures):
                try:
                    chunk_scripts = future.result()
                    # Thread-safe accumulation
                    with scripts_lock:
                        for script, count in chunk_scripts.items():
                            scripts[script] += count
                except Exception as e:
                    print(f"[WARNING] Error processing chunk in language detection: {e}")
    
    # Language determination logic (same as original)
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

# Global flag to track if language has been printed
_language_printed = False

def _print_extraction_summary( chapters, detected_language, extraction_mode, h1_count, h2_count, file_size_groups):
    """Print extraction summary"""
    global _language_printed
    
    print(f"\n📊 Chapter Extraction Summary ({extraction_mode.capitalize()} Mode):")
    print(f"   • Total chapters extracted: {len(chapters)}")
    
    # Format chapter range handling both int and float
    first_num = chapters[0]['num']
    last_num = chapters[-1]['num']
    
    print(f"   • Chapter range: {first_num} to {last_num}")
    
    # Only print detected language once per session
    if not _language_printed and detected_language and detected_language != 'unknown':
        print(f"   🌐 Detected language: {detected_language}")
        _language_printed = True
    
    if extraction_mode == "smart":
        print(f"   • Primary header type: {'<h2>' if h2_count > h1_count else '<h1>'}")
    
    image_only_count = sum(1 for c in chapters if c.get('is_image_only', False))
    text_only_count = sum(1 for c in chapters if not c.get('has_images', False) and c.get('file_size', 0) >= 500)
    mixed_count = sum(1 for c in chapters if c.get('has_images', False) and c.get('file_size', 0) >= 500)
    empty_count = sum(1 for c in chapters if c.get('file_size', 0) < 50)
    
    print(f"   • Text-only chapters: {text_only_count}")
    print(f"   • Image-only chapters: {image_only_count}")
    print(f"   • Mixed content chapters: {mixed_count}")
    print(f"   • Empty/minimal content: {empty_count}")
    
    # Check for merged chapters
    merged_count = sum(1 for c in chapters if c.get('was_merged', False))
    if merged_count > 0:
        print(f"   • Merged chapters: {merged_count}")
    
    # Check for missing chapters (only for integer sequences)
    expected_chapters = set(range(chapters[0]['num'], chapters[-1]['num'] + 1))
    actual_chapters = set(c['num'] for c in chapters)
    missing = expected_chapters - actual_chapters
    if missing:
        print(f"   ⚠️ Missing chapter numbers: {sorted(missing)}")
    
    if extraction_mode == "smart":
        method_stats = Counter(c['detection_method'] for c in chapters)
        print(f"   📈 Detection methods used:")
        for method, count in method_stats.most_common():
            print(f"      • {method}: {count} chapters")
        
        large_groups = [size for size, files in file_size_groups.items() if len(files) > 1]
        if large_groups:
            print(f"   ⚠️ Found {len(large_groups)} file size groups with potential duplicates")
    else:
        print(f"   • Empty/placeholder: {empty_count}")
        
    if extraction_mode == "full":
        print(f"   🔍 Full extraction preserved all HTML structure and tags")

def _extract_epub_metadata(zf):
    """Extract comprehensive metadata from EPUB file including all custom fields"""
    meta = {}
    # Use lxml for XML if available
    try:
        import lxml
        xml_parser = 'lxml-xml'
    except ImportError:
        xml_parser = 'xml'
    try:
        for name in zf.namelist():
            if name.lower().endswith('.opf'):
                opf_content = zf.read(name)
                soup = BeautifulSoup(opf_content, xml_parser)
                
                # Extract ALL Dublin Core elements (expanded list)
                dc_elements = ['title', 'creator', 'subject', 'description', 
                              'publisher', 'contributor', 'date', 'type', 
                              'format', 'identifier', 'source', 'language', 
                              'relation', 'coverage', 'rights']
                
                for element in dc_elements:
                    tag = soup.find(element)
                    if tag and tag.get_text(strip=True):
                        meta[element] = tag.get_text(strip=True)
                
                # Extract ALL meta tags (not just series)
                meta_tags = soup.find_all('meta')
                for meta_tag in meta_tags:
                    # Try different attribute names for the metadata name
                    name = meta_tag.get('name') or meta_tag.get('property', '')
                    content = meta_tag.get('content', '')
                    
                    if name and content:
                        # Store original name for debugging
                        original_name = name
                        
                        # Clean up common prefixes
                        if name.startswith('calibre:'):
                            name = name[8:]  # Remove 'calibre:' prefix
                        elif name.startswith('dc:'):
                            name = name[3:]  # Remove 'dc:' prefix
                        elif name.startswith('opf:'):
                            name = name[4:]  # Remove 'opf:' prefix
                        
                        # Normalize the field name - replace hyphens with underscores
                        name = name.replace('-', '_')
                        
                        # Don't overwrite if already exists (prefer direct tags over meta tags)
                        if name not in meta:
                            meta[name] = content
                            
                            # Debug output for custom fields
                            if original_name != name:
                                print(f"   • Found custom field: {original_name} → {name}")
                
                # Special handling for series information (maintain compatibility)
                if 'series' not in meta:
                    series_tags = soup.find_all('meta', attrs={'name': lambda x: x and 'series' in x.lower()})
                    for series_tag in series_tags:
                        series_name = series_tag.get('content', '')
                        if series_name:
                            meta['series'] = series_name
                            break
                
                # Extract refines metadata (used by some EPUB creators)
                refines_metas = soup.find_all('meta', attrs={'refines': True})
                for refine in refines_metas:
                    property_name = refine.get('property', '')
                    content = refine.get_text(strip=True) or refine.get('content', '')
                    
                    if property_name and content:
                        # Clean property name
                        if ':' in property_name:
                            property_name = property_name.split(':')[-1]
                        property_name = property_name.replace('-', '_')
                        
                        if property_name not in meta:
                            meta[property_name] = content
                
                # Log extraction summary
                print(f"📋 Extracted {len(meta)} metadata fields")
                
                # Show standard vs custom fields
                standard_keys = {'title', 'creator', 'language', 'subject', 'description', 
                               'publisher', 'date', 'identifier', 'source', 'rights', 
                               'contributor', 'type', 'format', 'relation', 'coverage'}
                custom_keys = set(meta.keys()) - standard_keys
                
                if custom_keys:
                    print(f"📋 Standard fields: {len(standard_keys & set(meta.keys()))}")
                    print(f"📋 Custom fields found: {sorted(custom_keys)}")
                    
                    # Show sample values for custom fields (truncated)
                    for key in sorted(custom_keys)[:5]:  # Show first 5 custom fields
                        value = str(meta[key])
                        if len(value) > 50:
                            value = value[:47] + "..."
                        print(f"   • {key}: {value}")
                    
                    if len(custom_keys) > 5:
                        print(f"   • ... and {len(custom_keys) - 5} more custom fields")
                
                break
                
    except Exception as e:
        print(f"[WARNING] Failed to extract metadata: {e}")
        import traceback
        traceback.print_exc()
    
    return meta

def _categorize_resource( file_path, file_name):
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
        if 'container.xml' in file_path_lower:
            safe_filename = 'container.xml'
        else:
            safe_filename = file_name
        return 'epub_structure', None, safe_filename
    elif file_path_lower.endswith(('.js', '.xml', '.txt')):
        return 'other', None, sanitize_resource_filename(file_name)
    
    return None

def _cleanup_old_resources( output_dir):
    """Clean up old resource directories and EPUB structure files"""
    print("🧹 Cleaning up any existing resource directories...")
    
    cleanup_success = True
    
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
    
    epub_structure_files = ['container.xml', 'content.opf', 'toc.ncx']
    for epub_file in epub_structure_files:
        input_path = os.path.join(output_dir, epub_file)
        if os.path.exists(input_path):
            try:
                os.remove(input_path)
                print(f"   🗑️ Removed old {epub_file}")
            except PermissionError:
                print(f"   ⚠️ Cannot remove {epub_file} (permission denied) - will use existing file")
            except Exception as e:
                print(f"   ⚠️ Error removing {epub_file}: {e}")
    
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

def _count_existing_resources( output_dir, extracted_resources):
    """Count existing resources when skipping extraction"""
    for resource_type in ['css', 'fonts', 'images', 'epub_structure']:
        if resource_type == 'epub_structure':
            epub_files = []
            for file in ['container.xml', 'content.opf', 'toc.ncx']:
                if os.path.exists(os.path.join(output_dir, file)):
                    epub_files.append(file)
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

def _validate_critical_files( output_dir, extracted_resources):
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
    
    critical_files = ['container.xml']
    missing_critical = [f for f in critical_files if not os.path.exists(os.path.join(output_dir, f))]
    
    if missing_critical:
        print(f"⚠️ WARNING: Missing critical EPUB files: {missing_critical}")
        print("   This may prevent proper EPUB reconstruction!")
    else:
        print("✅ All critical EPUB structure files extracted successfully")
    
    opf_files = [f for f in extracted_resources['epub_structure'] if f.lower().endswith('.opf')]
    if not opf_files:
        print("⚠️ WARNING: No OPF file found! This will prevent EPUB reconstruction.")
    else:
        print(f"✅ Found OPF file(s): {opf_files}")

def _create_extraction_report( output_dir, metadata, chapters, extracted_resources):
    """Create comprehensive extraction report with HTML file tracking"""
    report_path = os.path.join(output_dir, 'extraction_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("EPUB Extraction Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"EXTRACTION MODE: {metadata.get('extraction_mode', 'unknown').upper()}\n\n")
        
        f.write("METADATA:\n")
        for key, value in metadata.items():
            if key not in ['chapter_titles', 'extracted_resources', 'extraction_mode']:
                f.write(f"  {key}: {value}\n")
        
        f.write(f"\nCHAPTERS ({len(chapters)}):\n")
        
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
                if 'body' in c:
                    try:
                        soup = BeautifulSoup(c['body'], 'html.parser')
                        images = soup.find_all('img')
                        for img in images[:3]:
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
                    for file in files[:5]:
                        f.write(f"    - {file}\n")
                    if len(files) > 5:
                        f.write(f"    ... and {len(files) - 5} more\n")
        
        f.write(f"\nHTML FILES WRITTEN:\n")
        html_files_written = metadata.get('html_files_written', 0)
        f.write(f"  Total: {html_files_written} files\n")
        f.write(f"  Location: Main directory and 'originals' subdirectory\n")
        
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

def _log_extraction_summary( chapters, extracted_resources, detected_language, html_files_written=0):
    """Log final extraction summary with HTML file information"""
    extraction_mode = chapters[0].get('extraction_mode', 'unknown') if chapters else 'unknown'
    
    print(f"\n✅ {extraction_mode.capitalize()} extraction complete!")
    print(f"   📚 Chapters: {len(chapters)}")
    print(f"   📄 HTML files written: {html_files_written}")
    print(f"   🎨 Resources: {sum(len(files) for files in extracted_resources.values())}")
    print(f"   🌍 Language: {detected_language}")
    
    image_only_count = sum(1 for c in chapters if c.get('has_images') and c.get('file_size', 0) < 500)
    if image_only_count > 0:
        print(f"   📸 Image-only chapters: {image_only_count}")
    
    epub_files = extracted_resources.get('epub_structure', [])
    if epub_files:
        print(f"   📋 EPUB Structure: {len(epub_files)} files ({', '.join(epub_files)})")
    else:
        print(f"   ⚠️ No EPUB structure files extracted!")
    
    print(f"\n🔍 Pre-flight check readiness:")
    print(f"   ✅ HTML files: {'READY' if html_files_written > 0 else 'NOT READY'}")
    print(f"   ✅ Metadata: READY")
    print(f"   ✅ Resources: READY")
    
def _process_single_html_file(
    file_path,
    file_index,
    zip_file_path,
    parser,
    merge_candidates,
    disable_merging,
    enhanced_extractor,
    extraction_mode,
    enhanced_filtering,
    preserve_structure,
    protect_angle_brackets_func,
    pattern_manager,
    files_to_process,
    is_stop_requested
):
    """Process a single HTML file from an EPUB - standalone function for multiprocessing.
    
    This function is at module level to be picklable for ProcessPoolExecutor.
    All needed data must be passed as parameters.
    
    Returns:
        tuple: (chapter_info, h1_found, h2_found, file_size, sample_text, skipped_info)
        - chapter_info: dict with chapter data, or None if skipped/error
        - h1_found: bool indicating if h1 tags were found
        - h2_found: bool indicating if h2 tags were found  
        - file_size: int size of content text
        - sample_text: str text sample for language detection
        - skipped_info: tuple (file_path, reason, detail) if skipped, else None
    """
    from bs4 import BeautifulSoup
    import os
    import zipfile
    
    # Check stop
    if is_stop_requested():
        return None, False, False, 0, '', None
    
    try:
        # Open our own ZipFile instance for thread safety
        with zipfile.ZipFile(zip_file_path, 'r') as zf:
            # Read file data
            file_data = zf.read(file_path)
        
        # Decode the file data
        html_content = None
        detected_encoding = None
        for encoding in ['utf-8', 'utf-16', 'gb18030', 'shift_jis', 'euc-kr', 'gbk', 'big5']:
            try:
                html_content = file_data.decode(encoding)
                detected_encoding = encoding
                break
            except UnicodeDecodeError:
                continue
        
        if not html_content:
            print(f"[WARNING] Could not decode {file_path}")
            return None, False, False, 0, '', None
        
        # Check if this file needs merging
        if not disable_merging and file_path in merge_candidates:
            section_file = merge_candidates[file_path]
            print(f"[DEBUG] Processing merge for: {file_path}")
            
            try:
                # Read section file with our own ZipFile
                with zipfile.ZipFile(zip_file_path, 'r') as zf:
                    section_data = zf.read(section_file)
                section_html = None
                for encoding in ['utf-8', 'utf-16', 'gb18030', 'shift_jis', 'euc-kr', 'gbk', 'big5']:
                    try:
                        section_html = section_data.decode(encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                
                if section_html:
                    # Quick check if section is small enough to merge
                    section_soup = BeautifulSoup(section_html, parser)
                    section_text = section_soup.get_text(strip=True)
                    
                    if len(section_text) < 200:  # Merge if section is small
                        # Extract body content
                        chapter_soup = BeautifulSoup(html_content, parser)
                        
                        if section_soup.body:
                            section_body_content = ''.join(str(child) for child in section_soup.body.children)
                        else:
                            section_body_content = section_html
                        
                        if chapter_soup.body:
                            chapter_body_content = ''.join(str(child) for child in chapter_soup.body.children)
                        else:
                            chapter_body_content = html_content
                        
                        # Merge content
                        html_content = section_body_content + "\n<hr/>\n" + chapter_body_content
                        print(f"  → MERGED: Section ({len(section_text)} chars) + Chapter")
                    else:
                        print(f"  → NOT MERGED: Section too large ({len(section_text)} chars)")
                
            except Exception as e:
                print(f"[WARNING] Failed to merge {file_path}: {e}")
        
        # === ENHANCED EXTRACTION POINT ===
        content_html = None
        content_text = None
        chapter_title = None
        enhanced_extraction_used = False
        
        # Determine whether to use enhanced extractor
        use_enhanced = enhanced_extractor and extraction_mode == "enhanced"
        force_bs_traditional = False
        try:
            force_bs = os.getenv('FORCE_BS_FOR_TRADITIONAL', '0') == '1'
            model_env = os.getenv('MODEL', '')
            # Check for traditional translation API (inline to avoid circular imports)
            is_traditional_api = model_env in ['deepl', 'google-translate', 'google-translate-free'] or model_env.startswith('deepl/') or model_env.startswith('google-translate/')
            if force_bs and is_traditional_api:
                use_enhanced = False
                force_bs_traditional = True
        except Exception:
            pass
        
        # Use enhanced extractor if available and allowed
        if use_enhanced:
            clean_content, _, chapter_title = enhanced_extractor.extract_chapter_content(
                html_content, enhanced_filtering
            )
            enhanced_extraction_used = True
            
            content_html = clean_content
            content_text = clean_content
        
        # BeautifulSoup method (only for non-enhanced modes)
        if not enhanced_extraction_used:
            if extraction_mode == "enhanced" and not force_bs_traditional:
                print(f"❌ Skipping {file_path} - enhanced extraction required but not available")
                return None, False, False, 0, '', None
            
            # Parse the (possibly merged) content
            protected_html = protect_angle_brackets_func(html_content)
            soup = BeautifulSoup(protected_html, parser)
            
            # Get effective mode for filtering
            effective_filtering = enhanced_filtering if extraction_mode == "enhanced" else extraction_mode
            
            # In full mode, keep the entire HTML structure
            if effective_filtering == "full":
                content_html = html_content
                content_text = soup.get_text(strip=True)
            else:
                # Smart and comprehensive modes extract body content
                if soup.body:
                    content_html = str(soup.body)
                    content_text = soup.body.get_text(strip=True)
                else:
                    content_html = html_content
                    content_text = soup.get_text(strip=True)
            
            # Extract title (with ignore settings support)
            chapter_title = None
            
            # Check settings for batch translation
            batch_translate_active = os.getenv('BATCH_TRANSLATE_HEADERS', '0') == '1'
            use_title_tag = os.getenv('USE_TITLE', '0') == '1' or not batch_translate_active
            ignore_header_tags = os.getenv('IGNORE_HEADER', '0') == '1' and batch_translate_active
            
            # Extract from title tag if using titles
            if use_title_tag and soup.title and soup.title.string:
                chapter_title = soup.title.string.strip()
            
            # Extract from header tags if not ignored and no title found
            if not chapter_title and not ignore_header_tags:
                for header_tag in ['h1', 'h2', 'h3']:
                    header = soup.find(header_tag)
                    if header:
                        chapter_title = header.get_text(strip=True)
                        break
            
            # Fallback to filename if nothing found
            if not chapter_title:
                chapter_title = os.path.splitext(os.path.basename(file_path))[0]
        
        # Get the effective extraction mode for processing logic
        effective_mode = enhanced_filtering if extraction_mode == "enhanced" else extraction_mode
        
        # Skip truly empty files in smart mode
        if effective_mode == "smart" and not disable_merging and len(content_text.strip()) < 10:
            skipped_info = (file_path, 'empty', len(content_text))
            return None, False, False, 0, '', skipped_info
        
        # Get actual chapter number based on original position
        actual_chapter_num = files_to_process.index(file_path) + 1
        
        # Mode-specific logic
        detection_method = None
        h1_found = False
        h2_found = False
        
        if effective_mode == "comprehensive" or effective_mode == "full":
            # For comprehensive/full mode, use sequential numbering
            chapter_num = actual_chapter_num
            
            if not chapter_title:
                chapter_title = os.path.splitext(os.path.basename(file_path))[0]
            
            detection_method = f"{extraction_mode}_sequential" if extraction_mode == "enhanced" else f"{effective_mode}_sequential"
            
        elif effective_mode == "smart":
            # For smart mode, when merging is disabled, use sequential numbering
            if disable_merging:
                chapter_num = actual_chapter_num
                
                if not chapter_title:
                    chapter_title = os.path.splitext(os.path.basename(file_path))[0]
                
                detection_method = f"{extraction_mode}_sequential_no_merge" if extraction_mode == "enhanced" else "sequential_no_merge"
            else:
                # When merging is enabled, try to extract chapter info
                protected_html = protect_angle_brackets_func(html_content)
                soup = BeautifulSoup(protected_html, parser)
                
                # Count headers
                h1_tags = soup.find_all('h1')
                h2_tags = soup.find_all('h2')
                h1_found = len(h1_tags) > 0
                h2_found = len(h2_tags) > 0
                
                # Extract chapter number and title
                chapter_num, extracted_title, detection_method = _extract_chapter_info(
                    soup, file_path, content_text, html_content, pattern_manager
                )
                
                # Use extracted title if we don't have one
                if extracted_title and not chapter_title:
                    chapter_title = extracted_title
                
                # For hash-based filenames, chapter_num might be None
                if chapter_num is None:
                    chapter_num = actual_chapter_num
                    detection_method = f"{extraction_mode}_sequential_fallback" if extraction_mode == "enhanced" else "sequential_fallback"
                    print(f"[DEBUG] No chapter number found in {file_path}, assigning: {chapter_num}")
        
        # Filter content_html for title/header settings (before processing)
        batch_translate_active = os.getenv('BATCH_TRANSLATE_HEADERS', '0') == '1'
        use_title_tag = os.getenv('USE_TITLE', '0') == '1' or not batch_translate_active
        ignore_header_tags = os.getenv('IGNORE_HEADER', '0') == '1' and batch_translate_active
        remove_duplicate_h1_p = os.getenv('REMOVE_DUPLICATE_H1_P', '0') == '1'
        
        if (not use_title_tag or ignore_header_tags or remove_duplicate_h1_p) and content_html and not enhanced_extraction_used:
            # Parse the content HTML to remove unused tags
            content_soup = BeautifulSoup(content_html, parser)
            
            # Remove title tags if not using titles
            if not use_title_tag:
                for title_tag in content_soup.find_all('title'):
                    title_tag.decompose()
            
            # Remove header tags if ignored
            if ignore_header_tags:
                for header_tag in content_soup.find_all(['h1', 'h2', 'h3']):
                    header_tag.decompose()
            
            # Remove duplicate H1+P pairs (where P immediately follows H1 with same text)
            if remove_duplicate_h1_p:
                for h1_tag in content_soup.find_all('h1'):
                    # Skip split marker H1 tags
                    h1_id = h1_tag.get('id', '')
                    if h1_id and h1_id.startswith('split-'):
                        continue
                    h1_text = h1_tag.get_text(strip=True)
                    if 'SPLIT MARKER' in h1_text:
                        continue
                    
                    # Get the next sibling (skipping whitespace/text nodes)
                    next_sibling = h1_tag.find_next_sibling()
                    if next_sibling and next_sibling.name == 'p':
                        # Compare text content (stripped)
                        p_text = next_sibling.get_text(strip=True)
                        if h1_text == p_text:
                            # Remove the duplicate paragraph
                            next_sibling.decompose()
            
            # Update content_html with filtered version
            content_html = str(content_soup)
        
        # Process images and metadata
        protected_html = protect_angle_brackets_func(html_content)
        soup = BeautifulSoup(protected_html, parser)
        images = soup.find_all('img')
        has_images = len(images) > 0
        is_image_only_chapter = has_images and len(content_text.strip()) < 500
        
        if is_image_only_chapter:
            print(f"[DEBUG] Image-only chapter detected: {file_path} ({len(images)} images, {len(content_text)} chars)")
        
        # Calculate content hash (inline to avoid circular imports)
        import hashlib
        content_hash = hashlib.sha256(content_html.encode('utf-8', errors='ignore')).hexdigest()
        
        file_size = len(content_text)
        sample_text = content_text[:500] if effective_mode == "smart" else ''
        
        # Ensure chapter_num is always an integer
        if isinstance(chapter_num, float):
            chapter_num = int(chapter_num)
        
        # Create chapter info
        chapter_info = {
            "num": chapter_num,
            "title": chapter_title or f"Chapter {chapter_num}",
            "body": content_html,
            "filename": file_path,
            # IMPORTANT: For PDFs, we must preserve the original filename including extension
            # so that chapter_splitter.py can detect it as PDF content.
            # But we also want to preserve the basename for display/logging.
            "source_file": os.path.basename(zip_file_path) if zip_file_path else file_path,
            "original_filename": os.path.basename(file_path),
            "original_basename": os.path.splitext(os.path.basename(file_path))[0],
            "content_hash": content_hash,
            "detection_method": detection_method if detection_method else "pending",
            "file_size": file_size,
            "has_images": has_images,
            "image_count": len(images),
            "is_empty": len(content_text.strip()) == 0,
            "is_image_only": is_image_only_chapter,
            "extraction_mode": extraction_mode,
            "file_index": file_index
        }
        
        # Add enhanced extraction info if used
        if enhanced_extraction_used:
            chapter_info["enhanced_extraction"] = True
            chapter_info["enhanced_filtering"] = enhanced_filtering
            chapter_info["preserve_structure"] = preserve_structure
            # Store original HTML for image restoration
            chapter_info["original_html"] = html_content
        
        # Add merge info if applicable
        if not disable_merging and file_path in merge_candidates:
            chapter_info["was_merged"] = True
            chapter_info["merged_with"] = merge_candidates[file_path]
        
        if effective_mode == "smart":
            chapter_info["language_sample"] = content_text[:500]
            # Debug for section files
            if 'section' in chapter_info['original_basename'].lower():
                print(f"[DEBUG] Added section file to candidates: {chapter_info['original_basename']} (size: {chapter_info['file_size']})")
        
        return chapter_info, h1_found, h2_found, file_size, sample_text, None
                    
    except Exception as e:
        print(f"[ERROR] Failed to process {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None, False, False, 0, '', None
