"""
Standalone Chapter Header Translation Module
Translates chapter headers using strict content.opf-based mapping

This module can be used independently to translate chapter headers in HTML files
by matching them to source EPUB chapters using content.opf spine order.
"""

import os
import re
import json
import zipfile
import xml.etree.ElementTree as ET
import logging
from typing import Dict, Tuple, Optional, List
from bs4 import BeautifulSoup


def get_basename_without_ext(filename: str) -> str:
    """
    Get filename without extension(s) - handles double extensions like .htm.xhtml
    
    Args:
        filename: The filename
        
    Returns:
        Filename without extension(s)
    """
    # Strip all extensions (handles .htm.xhtml, .html, etc.)
    name = filename
    while True:
        name_without_ext, ext = os.path.splitext(name)
        if ext and ext.lower() in ['.html', '.xhtml', '.htm', '.xml']:
            name = name_without_ext
        else:
            break
    return name


def extract_source_chapters_with_opf_mapping(
    epub_path: str, 
    log_callback=None
) -> Tuple[Dict[str, str], List[str]]:
    """
    Extract source chapter titles from EPUB using strict OPF spine ordering
    
    Args:
        epub_path: Path to the source EPUB file
        log_callback: Optional callback for logging
        
    Returns:
        Tuple of (chapter_mapping, spine_order) where:
        - chapter_mapping: Maps normalized source filename to title
        - spine_order: List of source filenames in spine order
    """
    def log(message):
        if log_callback:
            log_callback(message)
        else:
            print(message)
    
    chapter_mapping = {}
    spine_order = []
    
    if not os.path.exists(epub_path):
        log(f"⚠️ Source EPUB not found: {epub_path}")
        return chapter_mapping, spine_order
    
    try:
        with zipfile.ZipFile(epub_path, 'r') as zf:
            # Find and parse OPF file
            opf_content = None
            opf_path = None
            
            for name in zf.namelist():
                if name.endswith('.opf'):
                    opf_path = name
                    opf_content = zf.read(name)
                    log(f"📋 Found OPF file: {name}")
                    break
            
            if not opf_content:
                try:
                    container = zf.read('META-INF/container.xml')
                    tree = ET.fromstring(container)
                    rootfile = tree.find('.//{urn:oasis:names:tc:opendocument:xmlns:container}rootfile')
                    if rootfile is not None:
                        opf_path = rootfile.get('full-path')
                        if opf_path:
                            opf_content = zf.read(opf_path)
                            log(f"📋 Found OPF via container.xml: {opf_path}")
                except:
                    pass
            
            # Parse OPF to get spine order
            if opf_content:
                try:
                    root = ET.fromstring(opf_content)
                    
                    ns = {'opf': 'http://www.idpf.org/2007/opf'}
                    if root.tag.startswith('{'):
                        default_ns = root.tag[1:root.tag.index('}')]
                        ns = {'opf': default_ns}
                    
                    # Get manifest to map IDs to files
                    manifest = {}
                    opf_dir = os.path.dirname(opf_path) if opf_path else ''
                    
                    for item in root.findall('.//opf:manifest/opf:item', ns):
                        item_id = item.get('id')
                        href = item.get('href')
                        media_type = item.get('media-type', '')
                        
                        if item_id and href and ('html' in media_type.lower() or href.endswith(('.html', '.xhtml', '.htm'))):
                            if opf_dir:
                                full_path = os.path.join(opf_dir, href).replace('\\', '/')
                            else:
                                full_path = href
                            manifest[item_id] = full_path
                    
                    # Get spine order - filter out nav/toc/cover
                    spine = root.find('.//opf:spine', ns)
                    if spine is not None:
                        skip_keywords = ['nav', 'toc', 'contents', 'cover']
                        for itemref in spine.findall('opf:itemref', ns):
                            idref = itemref.get('idref')
                            if idref and idref in manifest:
                                file_path = manifest[idref]
                                basename = os.path.basename(file_path).lower()
                                
                                # Skip navigation/toc files - BUT only if basename has NO numbers
                                # Files with numbers like 'nav01', 'toc05' are real chapters
                                import re
                                has_numbers = bool(re.search(r'\d', basename))
                                if not has_numbers and any(skip in basename for skip in skip_keywords):
                                    continue
                                spine_order.append(file_path)
                    
                    log(f"📋 Found {len(spine_order)} content chapters in OPF spine order")
                    
                except Exception as e:
                    log(f"⚠️ Error parsing OPF: {e}")
                    spine_order = []
            
            # Use spine order if available, otherwise alphabetical
            if spine_order:
                epub_html_files = spine_order
                log("✅ Using STRICT OPF spine order for source headers")
            else:
                # Fallback: alphabetical order
                skip_keywords = ['nav', 'toc', 'contents', 'cover']
                import re
                epub_html_files = sorted([
                    f for f in zf.namelist() 
                    if f.endswith(('.html', '.xhtml', '.htm')) 
                    and not f.startswith('__MACOSX')
                    # Skip only if basename has NO numbers (files with numbers like 'nav01' are real chapters)
                    and not (not bool(re.search(r'\d', os.path.basename(f))) and any(skip in os.path.basename(f).lower() for skip in skip_keywords))
                ])
                log("⚠️ No OPF spine found, using alphabetical order")
            
            log(f"📚 Processing {len(epub_html_files)} content files from source EPUB")
            
            # Extract titles from source EPUB files (in order)
            for idx, content_file in enumerate(epub_html_files):
                try:
                    html_content = zf.read(content_file).decode('utf-8', errors='ignore')
                    
                    if not html_content:
                        continue
                    
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    title = None
                    for tag_name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                        tag = soup.find(tag_name)
                        if tag:
                            text = tag.get_text().strip()
                            if text:
                                title = text
                                break
                    
                    if title:
                        # Store by basename without extension
                        basename_no_ext = get_basename_without_ext(os.path.basename(content_file))
                        chapter_mapping[basename_no_ext] = title
                        if idx < 5:
                            log(f"  Source[{idx}] ({os.path.basename(content_file)}): {title}")
                    
                except Exception as e:
                    log(f"  ⚠️ Error reading source chapter {idx}: {e}")
                    continue
            
            log(f"📚 Extracted {len(chapter_mapping)} titles from source EPUB")
    
    except Exception as e:
        log(f"❌ Error extracting source chapters: {e}")
        import traceback
        log(traceback.format_exc())
    
    return chapter_mapping, spine_order


def match_output_to_source_chapters(
    output_dir: str,
    source_mapping: Dict[str, str],
    spine_order: List[str],
    log_callback=None,
    explicit_mapping: Dict[str, str] = None
) -> Dict[str, Tuple[str, str, str]]:
    """
    Match output HTML files to source chapters by checking if source basename appears in output filename
    
    Args:
        output_dir: Directory containing translated HTML files
        source_mapping: Mapping of source basename (no ext) to title
        spine_order: List of source filenames in spine order
        log_callback: Optional callback for logging
        explicit_mapping: Optional mapping of source basename to expected output clean name
        
    Returns:
        Dict mapping output_filename to (source_title, current_title, output_filename)
        Only includes chapters where source basename is found in output filename
    """
    def log(message):
        if log_callback:
            log_callback(message)
        else:
            print(message)
    
    matches = {}
    
    # Get all HTML files from output directory
    html_extensions = ('.html', '.xhtml', '.htm')
    output_files_set = set([
        f for f in os.listdir(output_dir) 
        if f.lower().endswith(html_extensions)
    ])
    
    log(f"📁 Found {len(output_files_set)} HTML files in output directory")
    log(f"📚 Have {len(source_mapping)} source chapters to match")
    
    if not output_files_set:
        log("⚠️ No HTML files found in output directory!")
        return matches
    
    matched_count = 0
    skipped_count = 0
    
    # Iterate in spine order instead of alphabetical
    for source_file in spine_order:
        source_basename = get_basename_without_ext(os.path.basename(source_file))
        
        # Skip if not in source_mapping
        if source_basename not in source_mapping:
            continue
        
        source_title = source_mapping[source_basename]
        
        # Determine target match name (explicit or basename)
        target_match_name = source_basename
        if explicit_mapping and source_basename in explicit_mapping:
            target_match_name = explicit_mapping[source_basename]
        
        # Find matching output file
        output_file = None
        for candidate in output_files_set:
            candidate_no_ext = get_basename_without_ext(candidate)
            # Strip response_ prefix if present
            if candidate_no_ext.startswith('response_'):
                candidate_no_ext = candidate_no_ext[9:]
            
            if candidate_no_ext == target_match_name:
                output_file = candidate
                break
        
        # Always include this chapter for translation, even if output file doesn't exist
        current_title = source_title  # Use source title as default
        
        if output_file:
            # Read current title from output file if it exists
            try:
                output_path = os.path.join(output_dir, output_file)
                if not os.path.exists(output_path):
                    log(f"  ⚠️ Skipping {output_file} (file not found)")
                    continue
                with open(output_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                soup = BeautifulSoup(content, 'html.parser')
                
                for tag_name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    tag = soup.find(tag_name)
                    if tag:
                        text = tag.get_text().strip()
                        if text:
                            current_title = text
                            break
                
            except Exception as e:
                log(f"  ⚠️ Error reading {output_file}: {e}")
        
        # Add to matches regardless of whether output file exists
        matches[output_file or f"{source_basename}.html"] = (source_title, current_title, output_file or f"{source_basename}.html")
        matched_count += 1
        
        if matched_count <= 5:
            if output_file:
                log(f"  ✓ Matched: {output_file}")
            else:
                log(f"  ⊕ Added (no output file): {source_basename}")
            log(f"    Source title: '{source_title}'")
    
    log(f"\n📊 Matching results:")
    log(f"  ✓ Matched: {matched_count} chapters")
    log(f"  ⊝ Skipped: {skipped_count} chapters (no match)")
    
    return matches


def load_translations_from_file(translations_file: str, log_callback=None) -> Tuple[Dict[int, str], Dict[int, str], Dict[int, str]]:
    """
    Load translations from the translated_headers.txt file
    
    Args:
        translations_file: Path to translated_headers.txt
        log_callback: Optional callback for logging
        
    Returns:
        Tuple of (source_headers, translated_headers, output_files) where:
        - source_headers: Maps chapter number to original title
        - translated_headers: Maps chapter number to translated title
        - output_files: Maps chapter number to output filename (basename/clean name)
    """
    def log(message):
        if log_callback:
            log_callback(message)
        else:
            print(message)
    
    source_headers = {}
    translated_headers = {}
    output_files = {}
    
    try:
        with open(translations_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the actual file format used by BatchHeaderTranslator._save_translations_to_file
        # Format:
        # Chapter X:
        #   Original:   [title]
        #   Translated: [title]
        # ----------------------------------------
        
        import re
        
        # Split by chapter blocks
        chapter_blocks = re.split(r'-{3,}', content)
        
        for block in chapter_blocks:
            if not block.strip():
                continue
            
            # Parse chapter number
            chapter_match = re.search(r'Chapter (\d+):', block)
            if not chapter_match:
                continue
            
            chapter_num = int(chapter_match.group(1))
            
            # Parse original title
            original_match = re.search(r'Original:\s*(.+?)(?:\n|$)', block)
            if original_match:
                source_headers[chapter_num] = original_match.group(1).strip()
            
            # Parse translated title
            translated_match = re.search(r'Translated:\s*(.+?)(?:\n|$)', block)
            if translated_match:
                translated_headers[chapter_num] = translated_match.group(1).strip()
            
            # Parse output file or target URI (TOC uses Target URI, Headers use Output File)
            output_match = re.search(r'(?:Output File|Target URI):\s*(.+?)(?:\n|$)', block)
            if output_match:
                output_files[chapter_num] = output_match.group(1).strip()
        
        log(f"📋 Loaded {len(translated_headers)} translations from file")
        
        # Log first few for debugging
        if translated_headers:
            for num in list(sorted(translated_headers.keys()))[:3]:
                log(f"  Chapter {num}: {translated_headers[num]}")
            if len(translated_headers) > 3:
                log(f"  ... and {len(translated_headers) - 3} more")
        
    except Exception as e:
        log(f"⚠️ Error loading translations: {e}")
        import traceback
        log(traceback.format_exc())
    
    return source_headers, translated_headers, output_files


def repair_translation_file(
    translations_file: str,
    epub_path: str,
    output_dir: str,
    log_callback=None
) -> bool:
    """Repair a translated_headers.txt or TOC.txt file:
    1. Sort all entries by chapter number
    2. Discover and add missing Output File fields retroactively
    
    Args:
        translations_file: Path to the translations file to repair
        epub_path: Path to the source EPUB (for spine order)
        output_dir: Output directory containing translated HTML files
        log_callback: Optional logging callback
        
    Returns:
        True if the file was modified, False otherwise
    """
    log = log_callback or (lambda msg: None)
    
    if not os.path.exists(translations_file):
        return False
    
    # Load existing data
    source_headers, translated_headers, output_files = load_translations_from_file(
        translations_file, log_callback=lambda msg: None  # suppress load logs during repair
    )
    
    if not source_headers:
        return False
    
    # Get spine order from EPUB for output file discovery
    try:
        source_mapping, spine_order = extract_source_chapters_with_opf_mapping(
            epub_path, log_callback=lambda msg: None  # suppress logs
        )
    except Exception:
        spine_order = []
    
    # Check if repair is needed: unsorted entries or missing Output File fields
    sorted_nums = sorted(source_headers.keys())
    current_order = list(source_headers.keys())
    needs_sort = current_order != sorted_nums
    
    # Detect file type from header line FIRST
    try:
        with open(translations_file, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
    except Exception:
        first_line = ""
    
    is_toc = 'TOC' in first_line
    file_label = "TOC.txt" if is_toc else "translated_headers.txt"
    header_title = "TOC Translations" if is_toc else "Chapter Header Translations"

    # Discover missing Output File entries (ONLY for headers, TOC.txt mappings are not 1:1 with spine)
    missing_out = set()
    if not is_toc:
        try:
            out_files_list = os.listdir(output_dir)
        except OSError:
            out_files_list = []
        
        for num in source_headers:
            if num not in output_files and spine_order:
                # Try to find output file from spine order
                if 0 < num <= len(spine_order):
                    src_file = spine_order[num - 1]
                    src_bn = get_basename_without_ext(os.path.basename(src_file))
                    for candidate in out_files_list:
                        cand_bn = get_basename_without_ext(candidate)
                        if cand_bn.startswith('response_'):
                            cand_bn = cand_bn[9:]
                        if cand_bn == src_bn:
                            clean = get_basename_without_ext(candidate)
                            if clean.startswith('response_'):
                                clean = clean[9:]
                            output_files[num] = clean
                            missing_out.add(num)
                            break
    
    if not needs_sort and not missing_out:
        return False  # Nothing to repair
    
    # Rebuild the file sorted with Output File data
    try:
        with open(translations_file, 'w', encoding='utf-8') as f:
            f.write(f"{header_title}\n")
            f.write("=" * 50 + "\n\n")
            
            has_chapter_zero = 0 in sorted_nums
            if has_chapter_zero and not is_toc:
                f.write("Note: This novel uses 0-based chapter numbering (starts with Chapter 0)\n")
                f.write("-" * 50 + "\n\n")
            
            for num in sorted_nums:
                orig = source_headers.get(num, "Unknown")
                trans = translated_headers.get(num, orig)
                f.write(f"Chapter {num}:\n")
                f.write(f"  Original:   {orig}\n")
                f.write(f"  Translated: {trans}\n")
                if num in output_files and output_files[num]:
                    label = "Target URI" if is_toc else "Output File"
                    f.write(f"  {label}: {output_files[num]}\n")
                if num not in translated_headers:
                    f.write("  Status:     ⚠️ Using original (translation failed)\n")
                f.write("-" * 40 + "\n")
            
            f.write(f"\nSummary:\n")
            count_label = "Total entries" if is_toc else "Total chapters"
            f.write(f"{count_label}: {len(sorted_nums)}\n")
            if sorted_nums:
                f.write(f"{'Entry' if is_toc else 'Chapter'} range: {min(sorted_nums)} to {max(sorted_nums)}\n")
            f.write(f"Successfully translated: {len(translated_headers)}\n")
        
        repairs = []
        if needs_sort:
            repairs.append("sorted")
        if missing_out:
            repairs.append(f"{len(missing_out)} Output File(s) discovered")
        log(f"🔧 Repaired {file_label}: {', '.join(repairs)}")
        return True
        
    except Exception as e:
        log(f"⚠️ Failed to repair {file_label}: {e}")
        return False


def apply_existing_translations(
    epub_path: str,
    output_dir: str,
    translations_file: str,
    update_html: bool = True,
    log_callback=None
) -> Dict[str, str]:
    """
    Apply existing translations from translated_headers.txt to HTML files and toc.ncx
    
    This uses the same OPF-based matching as the full translation process,
    but reads translations from the existing file instead of calling the API.
    
    Args:
        epub_path: Path to source EPUB file
        output_dir: Directory containing HTML files and toc.ncx
        translations_file: Path to translated_headers.txt
        update_html: Whether to update HTML files and toc.ncx
        log_callback: Optional callback for logging
        
    Returns:
        Dict mapping output filename to translated title
    """
    def log(message):
        if log_callback:
            log_callback(message)
        else:
            print(message)
    
    # Step 1: Load existing translations from file
    log("📖 Loading existing translations...")
    source_headers, translated_headers, output_files = load_translations_from_file(translations_file, log_callback)
    
    if not translated_headers:
        log("⚠️ No translations found in file")
        return {}
    
    # Step 2: Extract source chapters with OPF mapping to match against loaded translations
    log("📚 Extracting source chapter information from EPUB...")
    source_mapping, spine_order = extract_source_chapters_with_opf_mapping(epub_path, log_callback)
    
    # Build explicit mapping if output files are available
    explicit_mapping = {}
    if output_files:
        # Build a reverse map: output_clean_name -> chapter_num
        # Then match source basenames by using the source_headers to find the right source file
        # This avoids the fragile spine-position = chapter-number assumption
        for chapter_num, output_clean_name in output_files.items():
            # Find the matching source file by checking if the output_clean_name
            # appears as a source basename in source_mapping
            # The output_clean_name was originally derived from the source file during translation
            if output_clean_name in source_mapping:
                # Direct match: output_clean_name IS the source basename
                explicit_mapping[output_clean_name] = output_clean_name
            else:
                # Try matching via the source title from translated_headers.txt
                if chapter_num in source_headers:
                    orig_title = source_headers[chapter_num]
                    for src_basename, src_title in source_mapping.items():
                        if src_title == orig_title:
                            explicit_mapping[src_basename] = output_clean_name
                            break
    
    # Step 3: Match output files to source chapters
    log("🔗 Matching output files to source chapters...")
    matches = match_output_to_source_chapters(output_dir, source_mapping, spine_order, log_callback, explicit_mapping)
    
    if not matches:
        log("⚠️ No matching chapters found")
        return {}
    
    # Step 4: Build current titles map for exact replacement
    current_titles_map = {}
    chapter_to_output = {}
    
    for idx, (output_file, (source_title, current_title, _)) in enumerate(matches.items(), 1):
        current_titles_map[idx] = {
            'title': current_title,
            'filename': output_file
        }
        chapter_to_output[idx] = output_file
    
    # Step 5: Apply translations if update_html is enabled
    result = {}
    
    if update_html:
        log("\n📝 Updating HTML files and toc.ncx with existing translations...")
        
        # Import BatchHeaderTranslator for its update methods
        from metadata_batch_translator import BatchHeaderTranslator
        
        # Create a minimal translator instance just for the update methods
        # We don't need the API client since we're not translating
        class DummyClient:
            pass
        
        translator = BatchHeaderTranslator(DummyClient(), {})
        
        # Use the exact replacement method to update HTML files
        translator._update_html_headers_exact(output_dir, translated_headers, current_titles_map)
        
        # Update toc.ncx if it exists
        toc_path = os.path.join(output_dir, 'toc.ncx')
        if os.path.exists(toc_path):
            log("📖 Updating toc.ncx...")
            update_toc_ncx(toc_path, translated_headers, current_titles_map, log_callback)
        
        # Build result mapping
        for idx, translated_title in translated_headers.items():
            if idx in chapter_to_output:
                result[chapter_to_output[idx]] = translated_title
    
    log(f"✅ Applied translations to {len(result)} files")
    return result


def update_toc_ncx(toc_path: str, translated_headers: Dict[int, str], 
                   current_titles_map: Dict[int, Dict[str, str]], log_callback=None):
    """
    Update toc.ncx file with translated chapter titles
    
    Args:
        toc_path: Path to toc.ncx file
        translated_headers: Dict mapping chapter numbers to translated titles
        current_titles_map: Dict mapping chapter numbers to current title info
        log_callback: Optional callback for logging
    """
    def log(message):
        if log_callback:
            log_callback(message)
        else:
            print(message)
    
    try:
        import xml.etree.ElementTree as ET
        
        # Parse the toc.ncx file
        tree = ET.parse(toc_path)
        root = tree.getroot()
        
        # Define namespace
        ns = {'ncx': 'http://www.daisy.org/z3986/2005/ncx/'}
        
        updated_count = 0
        
        # Find all navPoint elements
        for navPoint in root.findall('.//ncx:navPoint', ns):
            # Get the content src to identify which chapter this is
            content = navPoint.find('ncx:content', ns)
            if content is not None:
                src = content.get('src', '')
                # Remove any fragment identifier (#...)
                if '#' in src:
                    src = src.split('#')[0]
                
                # Try to match this with our chapter mappings
                src_basename = os.path.basename(src)
                
                # Find which chapter this corresponds to
                for chapter_num, info in current_titles_map.items():
                    if info['filename'] == src_basename:
                        # Found the matching chapter
                        if chapter_num in translated_headers:
                            # Update the navLabel text
                            navLabel = navPoint.find('ncx:navLabel', ns)
                            if navLabel is not None:
                                text_elem = navLabel.find('ncx:text', ns)
                                if text_elem is not None:
                                    old_text = text_elem.text
                                    text_elem.text = translated_headers[chapter_num]
                                    updated_count += 1
                                    log(f"  ✓ Updated navPoint: '{old_text}' → '{translated_headers[chapter_num]}'")
                        break
        
        if updated_count > 0:
            # Save the updated toc.ncx
            tree.write(toc_path, encoding='utf-8', xml_declaration=True)
            log(f"✅ Updated {updated_count} entries in toc.ncx")
        else:
            log("ℹ️ No updates needed for toc.ncx")
    
    except Exception as e:
        log(f"⚠️ Error updating toc.ncx: {e}")


def _cross_reference_from_file(
    entries_to_translate: Dict[int, str],
    other_file_path: str,
    log_callback=None
) -> tuple:
    """Cross-reference entries against an existing translation file to reuse translations.

    Before sending entries to the API, check if the other translation file
    (TOC.txt or translated_headers.txt) already contains an identical raw entry.
    If so, reuse the existing translation to avoid redundant API calls and
    ensure consistency.

    Args:
        entries_to_translate: Dict[int, str] mapping index -> raw text
        other_file_path: Path to the other translation file
        log_callback: Optional callback for logging

    Returns:
        Tuple of (reused, remaining) dicts
    """
    def log(message):
        if log_callback:
            log_callback(message)
        else:
            print(message)

    reused: Dict[int, str] = {}
    remaining: Dict[int, str] = dict(entries_to_translate)

    if not other_file_path or not os.path.exists(other_file_path):
        return reused, remaining

    try:
        other_originals, other_translated, _ = load_translations_from_file(
            other_file_path, log_callback
        )

        if not other_originals or not other_translated:
            return reused, remaining

        # Build a map: stripped raw text -> translated text from the other file
        raw_to_translated: Dict[str, str] = {}
        for idx, raw in other_originals.items():
            key = (raw or '').strip()
            if key and idx in other_translated:
                trans = (other_translated[idx] or '').strip()
                if trans and trans != key:
                    raw_to_translated[key] = trans

        if not raw_to_translated:
            return reused, remaining

        other_label = os.path.basename(other_file_path)

        # Match entries against the other file's originals
        for idx, raw in list(remaining.items()):
            key = (raw or '').strip()
            if key in raw_to_translated:
                reused[idx] = raw_to_translated[key]
                del remaining[idx]

        if reused:
            log(f"♻️ Reused {len(reused)}/{len(entries_to_translate)} "
                f"header translation(s) from {other_label}")
            for idx in list(reused.keys())[:3]:
                log(f"  ♻️ [{idx}] {entries_to_translate[idx]} → {reused[idx]}")
            if len(reused) > 3:
                log(f"  ... and {len(reused) - 3} more")

    except Exception as e:
        log(f"⚠️ Cross-reference check failed ({e}) — will translate all entries")
        remaining = dict(entries_to_translate)
        reused = {}

    return reused, remaining


def translate_headers_standalone(
    epub_path: str,
    output_dir: str,
    api_client,
    config: dict = None,
    update_html: bool = True,
    save_to_file: bool = True,
    log_callback=None,
    gui_instance=None
) -> Dict[str, str]:
    """
    Standalone function to translate chapter headers with exact OPF-based matching
    
    Uses IDENTICAL translation and HTML update method as the pipeline's BatchHeaderTranslator.
    The ONLY difference is the content.opf-based chapter matching instead of fuzzy matching.
    
    Args:
        epub_path: Path to source EPUB file
        output_dir: Directory containing translated HTML files
        api_client: UnifiedClient instance for translation
        config: Optional config dict with translation settings
        update_html: Whether to update HTML files with translations
        save_to_file: Whether to save translations to file
        log_callback: Optional callback for logging
        gui_instance: Optional GUI instance to store translator reference for stop button
        
    Returns:
        Dict mapping output filename to translated title
    """
    def log(message):
        if log_callback:
            log_callback(message)
        else:
            print(message)
    
    log("=" * 80)
    log("Starting Standalone Header Translation (Content.OPF Based)")
    log("=" * 80)
    
    # Step 1: Extract source chapters with OPF mapping (UNIQUE TO STANDALONE)
    log("\nStep 1: Extracting source chapter titles from EPUB (strict OPF spine order)...")
    source_mapping, spine_order = extract_source_chapters_with_opf_mapping(epub_path, log_callback)
    
    if not source_mapping:
        log("ERROR: No source chapters found!")
        return {}
    
    # Step 2: Match output files to source chapters - EXACT matching only (UNIQUE TO STANDALONE)
    log("\nStep 2: Matching output files to source chapters (exact name match)...")
    matches = match_output_to_source_chapters(output_dir, source_mapping, spine_order, log_callback)
    
    if not matches:
        log("ERROR: No matching chapters found!")
        return {}
    
    # Step 3: Prepare headers for translation (SAME AS PIPELINE)
    log("\nStep 3: Preparing headers for translation...")
    headers_to_translate = {}
    current_titles_map = {}
    
    for idx, (output_file, (source_title, current_title, _)) in enumerate(matches.items(), 1):
        headers_to_translate[idx] = source_title
        current_titles_map[idx] = {
            'title': current_title,
            'filename': output_file
        }
    
    log(f"Prepared {len(headers_to_translate)} headers for translation")
    
    # Step 4: Translate using BatchHeaderTranslator - IDENTICAL TO PIPELINE
    log("\nStep 4: Translating headers using pipeline's BatchHeaderTranslator...")
    log("(This ensures 1:1 compatibility with pipeline translation)")
    
    from metadata_batch_translator import BatchHeaderTranslator
    
    # Create translator with same config as pipeline
    translator = BatchHeaderTranslator(api_client, config or {})
    
    # Store reference in GUI instance so stop button can access it
    if gui_instance is not None:
        gui_instance._batch_header_translator = translator
        # Reset stop flag at the start of translation
        if hasattr(gui_instance, '_headers_stop_requested'):
            gui_instance._headers_stop_requested = False
        
        # Hook up stop callback to propagate GUI stop to API client and translator
        def check_gui_stop():
            if hasattr(gui_instance, '_headers_stop_requested') and gui_instance._headers_stop_requested:
                # Also set translator stop flag
                translator.set_stop_flag(True)
                return True
            return False
        
        # Set stop callback on API client to check GUI flag
        if hasattr(api_client, '_stop_callback'):
            # Store original callback if exists
            original_stop_callback = api_client._stop_callback
            api_client._stop_callback = lambda: original_stop_callback() or check_gui_stop()
        else:
            # Just set our callback
            api_client._stop_callback = check_gui_stop
    
    # ── Cross-reference: reuse translations from TOC.txt if it exists ──
    toc_file = os.path.join(output_dir, 'TOC.txt')
    reused_from_toc, hdr_remaining = _cross_reference_from_file(
        headers_to_translate, toc_file, log_callback
    )
    
    # Call translate_and_save_headers - IDENTICAL TO PIPELINE
    # This method uses the EXACT same translation prompts, HTML update logic, and file saving
    try:
        if hdr_remaining:
            # Only translate entries not reused from TOC.txt
            translated_headers = translator.translate_and_save_headers(
                html_dir=output_dir,
                headers_dict=hdr_remaining,
                batch_size=config.get('headers_per_batch', -1) if config else None,
                output_dir=output_dir,
                update_html=update_html,  # Uses _update_html_headers_exact - same as pipeline
                save_to_file=save_to_file,  # Saves to translated_headers.txt - same as pipeline
                current_titles=current_titles_map  # Enables exact title replacement
            )
        else:
            translated_headers = {}
        
        # Merge reused translations
        if reused_from_toc:
            translated_headers.update(reused_from_toc)
            # Re-save the full file with merged entries
            if save_to_file:
                translator._save_translations_to_file(
                    headers_to_translate, translated_headers,
                    os.path.join(output_dir, 'translated_headers.txt'),
                    current_titles_map
                )
                log("📝 Re-saved translated_headers.txt with cross-referenced entries")
    except KeyboardInterrupt:
        log("\n⛔ Translation interrupted by user")
        translator.set_stop_flag(True)
        return {}
    except Exception as e:
        # Check if this was a stop request
        if translator.stop_flag or (gui_instance and hasattr(gui_instance, '_headers_stop_requested') and gui_instance._headers_stop_requested):
            log("\n⛔ Translation stopped by user")
            return {}
        raise
    
    # Step 5: Map back to output filenames
    log("\nStep 5: Mapping translations to output files...")
    result = {}
    for idx, translated_title in translated_headers.items():
        if idx in current_titles_map:
            output_file = current_titles_map[idx]['filename']
            result[output_file] = translated_title
            output_path = os.path.join(output_dir, output_file)
            if os.path.exists(output_path):
                log(f"  {output_file}: {translated_title}")
            # Skip log if the file is missing to avoid misleading output
    
    log("\n" + "=" * 80)
    log(f"Translation complete! Translated {len(result)} chapter headers")
    log("=" * 80)
    
    return result


def run_translation(
    source_epub_path: str,
    output_html_dir: str,
    log_callback=None
) -> Dict[str, str]:
    """
    Pipeline wrapper for standalone header translation
    
    This function is called by the EPUB compilation pipeline.
    It initializes the API client from environment variables and runs the translation.
    
    Args:
        source_epub_path: Path to the source EPUB file
        output_html_dir: Directory containing translated HTML files
        log_callback: Optional callback for logging
        
    Returns:
        Dict mapping output filename to translated title
    """
    def log(message):
        if log_callback:
            log_callback(message)
        else:
            print(message)
    
    try:
        # Initialize API client from environment variables
        from unified_api_client import UnifiedClient
        
        model = os.getenv('MODEL')
        api_key = os.getenv('API_KEY')
        
        if not model or not api_key:
            log("⚠️ Missing MODEL or API_KEY environment variables")
            return {}
        
        log(f"🔧 Initializing API client with model: {model}")
        api_client = UnifiedClient(api_key=api_key, model=model, output_dir=output_html_dir)
        
        # Get configuration from environment variables
        config = {
            'headers_per_batch': int(os.getenv('HEADERS_PER_BATCH', '-1')),
            'temperature': float(os.getenv('TRANSLATION_TEMPERATURE', '0.3')),
            'max_tokens': int(os.getenv('MAX_OUTPUT_TOKENS', '12000')),
            # Add Chapter Headers prompts from environment variables - use None if not set so fallback works
            'batch_header_system_prompt': os.getenv('BATCH_HEADER_SYSTEM_PROMPT') or None,
            'batch_header_prompt': os.getenv('BATCH_HEADER_PROMPT') or None,
        }
        
        # Get options from environment
        update_html = os.getenv('UPDATE_HTML_HEADERS', '1') == '1'
        save_to_file = os.getenv('SAVE_HEADER_TRANSLATIONS', '1') == '1'
        
        log(f"📋 Config: batch_size={config['headers_per_batch']}, update_html={update_html}, save_to_file={save_to_file}")
        
        # Run the translation
        result = translate_headers_standalone(
            epub_path=source_epub_path,
            output_dir=output_html_dir,
            api_client=api_client,
            config=config,
            update_html=update_html,
            save_to_file=save_to_file,
            log_callback=log_callback
        )
        
        return result
        
    except ImportError as e:
        log(f"⚠️ Failed to import UnifiedClient: {e}")
        return {}
    except Exception as e:
        log(f"❌ Error in run_translation: {e}")
        import traceback
        log(traceback.format_exc())
        return {}


def _attach_logging_handlers(gui_instance):
    """Attach logging handlers to reclaim HTTP logs for translator GUI.
    Matches the implementation in translator_gui.py's _attach_gui_logging_handlers.
    """
    try:
        # Define a simple handler that forwards logs to the GUI
        class GuiLogHandler(logging.Handler):
            def __init__(self, outer, level=logging.INFO):
                super().__init__(level)
                self.outer = outer
                self.outer_id = id(outer)  # Store ID to identify handlers from same instance
            
            def emit(self, record):
                try:
                    # Use the raw message without logger name/level prefixes
                    msg = record.getMessage()
                    if hasattr(self.outer, 'append_log'):
                        self.outer.append_log(msg)
                except Exception:
                    # Never raise from logging path
                    pass
        
        # Build handler
        handler = GuiLogHandler(gui_instance, level=logging.INFO)
        fmt = logging.Formatter('%(message)s')
        handler.setFormatter(fmt)
        
        # Target relevant loggers - includes httpx for HTTP request logs
        target_loggers = [
            'unified_api_client',
            'httpx',
            'requests.packages.urllib3',
            'openai'
        ]
        
        gui_id = id(gui_instance)
        
        for name in target_loggers:
            try:
                lg = logging.getLogger(name)
                # Remove any existing handlers from THIS SAME gui_instance
                # Don't use isinstance since GuiLogHandler is redefined each call
                lg.handlers = [h for h in lg.handlers 
                              if not (hasattr(h, 'outer_id') and h.outer_id == gui_id)]
                # Now add the new handler
                lg.addHandler(handler)
                # Ensure at least INFO level to see HTTP requests and retry/backoff notices
                if lg.level > logging.INFO or lg.level == logging.NOTSET:
                    lg.setLevel(logging.INFO)
            except Exception:
                pass
    except Exception as e:
        try:
            if hasattr(gui_instance, 'append_log'):
                gui_instance.append_log(f"⚠️ Failed to attach GUI log handlers: {e}")
        except Exception:
            pass


def run_translate_headers_gui(gui_instance):
    """
    GUI wrapper for standalone header translation
    
    Args:
        gui_instance: The GUI instance (translator_gui or other_settings)
    """
    from PySide6.QtWidgets import QMessageBox
    
    try:
        # Re-attach GUI logging handlers to reclaim logs from manga integration
        _attach_logging_handlers(gui_instance)
        
        # Get EPUB files - check if multiple files are selected first
        epub_files = []
        
        # Try to get multiple selected files from the GUI (if available)
        if hasattr(gui_instance, 'selected_files') and gui_instance.selected_files:
            # Filter for only EPUB files
            epub_files = [f for f in gui_instance.selected_files if f.lower().endswith('.epub')]
            if epub_files:
                gui_instance.append_log(f"📚 Found {len(epub_files)} EPUB file(s) in selection")
        
        # Fallback to single EPUB path
        if not epub_files:
            epub_path = gui_instance.get_current_epub_path()
            if not epub_path or not os.path.exists(epub_path):
                QMessageBox.critical(
                    None, 
                    "Error", 
                    "No EPUB file selected or file does not exist."
                )
                return
            
            # Check if epub_path is a directory containing multiple EPUBs
            if os.path.isdir(epub_path):
                # Process all EPUBs in the directory
                gui_instance.append_log(f"📁 Scanning directory for EPUB files: {epub_path}")
                for file in os.listdir(epub_path):
                    if file.lower().endswith('.epub'):
                        full_path = os.path.join(epub_path, file)
                        epub_files.append(full_path)
                
                if not epub_files:
                    QMessageBox.critical(
                        None,
                        "Error",
                        f"No EPUB files found in directory: {epub_path}"
                    )
                    return
                
                gui_instance.append_log(f"📚 Found {len(epub_files)} EPUB file(s) to process")
            else:
                # Single EPUB file
                epub_files = [epub_path]
        
        # Check API client once before processing any files
        if not hasattr(gui_instance, 'api_client') or not gui_instance.api_client:
            QMessageBox.critical(
                None, 
                "Error", 
                "API client not initialized. Please check your API settings."
            )
            return
        
        # Get config from GUI once
        config = {
            'headers_per_batch': int(getattr(gui_instance, 'headers_per_batch_var', -1)),
            'temperature': float(os.getenv('TRANSLATION_TEMPERATURE', '0.3')),
            'max_tokens': int(os.getenv('MAX_OUTPUT_TOKENS', '12000')),
            # Add Chapter Headers prompts from active profile/config
            'batch_header_system_prompt': gui_instance.config.get('batch_header_system_prompt', ''),
            'batch_header_prompt': gui_instance.config.get('batch_header_prompt', ''),
        }
        
        # Get options once
        update_html = getattr(gui_instance, 'update_html_headers_var', True)
        save_to_file = getattr(gui_instance, 'save_header_translations_var', True)
        
        # Process each EPUB file
        total_files = len(epub_files)
        successful = 0
        failed = 0
        
        gui_instance.append_log(f"📊 Will process {total_files} EPUB file(s)")
        
        for idx, current_epub in enumerate(epub_files, 1):
            # Check if stop was requested before starting a new EPUB
            if hasattr(gui_instance, '_headers_stop_requested') and gui_instance._headers_stop_requested:
                gui_instance.append_log("\n⛔ Translation stopped by user")
                gui_instance.append_log(f"📊 Stopped after processing {successful + failed}/{total_files} file(s)")
                break
            
            gui_instance.append_log(f"\n{'='*60}")
            gui_instance.append_log(f"📄 Processing EPUB {idx}/{total_files}: {os.path.basename(current_epub)}")
            gui_instance.append_log(f"{'='*60}")
            
            # Get output directory for this EPUB
            # Using EXACT same logic as QA Scanner auto-search (lines 1189-1193)
            epub_base = os.path.splitext(os.path.basename(current_epub))[0]
            current_dir = os.getcwd()
            script_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Check the most common locations in order of priority (same as QA Scanner)
            candidates = [
                os.path.join(current_dir, epub_base),        # current working directory
                os.path.join(script_dir, epub_base),         # src directory (where output typically goes)
                os.path.join(current_dir, 'src', epub_base), # src subdirectory from current dir
            ]
            
            # Add output directory override if configured (matches QA scanner behavior)
            override_dir = os.environ.get('OUTPUT_DIRECTORY') or gui_instance.config.get('output_directory')
            if override_dir:
                candidates.insert(0, os.path.join(override_dir, epub_base))
                gui_instance.append_log(f"🔍 Checking override directory: {override_dir}")
            
            output_dir = None
            checked_locations = []
            
            for candidate in candidates:
                checked_locations.append(candidate)
                if os.path.isdir(candidate):
                    # Verify it actually has HTML files
                    try:
                        files = os.listdir(candidate)
                        html_files = [f for f in files if f.lower().endswith(('.html', '.xhtml', '.htm'))]
                        if html_files:
                            output_dir = candidate
                            gui_instance.append_log(f"✓ Found output directory: {candidate}")
                            break
                    except Exception:
                        continue
            
            if not output_dir or not os.path.exists(output_dir):
                gui_instance.append_log(f"⚠️ Output directory not found for: {epub_base}")
                gui_instance.append_log(f"   Checked all {len(checked_locations)} locations:")
                for loc in checked_locations:
                    gui_instance.append_log(f"     - {loc}")
                failed += 1
                gui_instance.append_log(f"⏭️ Skipping to next EPUB... ({successful + failed}/{total_files} processed)\n")
                # Force GUI event processing
                try:
                    from PySide6.QtWidgets import QApplication
                    QApplication.processEvents()
                except Exception:
                    pass
                continue
            
            # Check if translated_headers.txt already exists
            translations_file = os.path.join(output_dir, "translated_headers.txt")
            if os.path.exists(translations_file):
                gui_instance.append_log(f"📁 Found existing translated_headers.txt for: {epub_base}")
                gui_instance.append_log(f"   File: {translations_file}")
                
                # --- Reconcile: check if the source EPUB has new chapters ---
                try:
                    existing_source, existing_trans, existing_out = load_translations_from_file(
                        translations_file, gui_instance.append_log
                    )
                    source_mapping, spine_order = extract_source_chapters_with_opf_mapping(
                        current_epub, gui_instance.append_log
                    )
                    
                    # Build numbered source headers from spine for comparison
                    current_source_count = len(source_mapping)
                    existing_count = len(existing_source)
                    
                    if current_source_count > existing_count:
                        new_count = current_source_count - existing_count
                        gui_instance.append_log(f"📦 Source EPUB has {new_count} NEW chapter(s) ({existing_count} → {current_source_count})")
                        
                        # Build the full numbered headers from spine order
                        all_headers = {}
                        for idx, src_file in enumerate(spine_order, 1):
                            src_basename = get_basename_without_ext(os.path.basename(src_file))
                            if src_basename in source_mapping:
                                all_headers[idx] = source_mapping[src_basename]
                        
                        # Find which chapter numbers are new
                        new_nums = set(all_headers.keys()) - set(existing_source.keys())
                        
                        if new_nums:
                            new_headers_to_translate = {n: all_headers[n] for n in sorted(new_nums)}

                            # ── Cross-reference new entries from TOC.txt (two-way reuse) ──
                            _toc_file = os.path.join(output_dir, 'TOC.txt')
                            _reused_from_toc, _hdr_api_remaining = _cross_reference_from_file(
                                new_headers_to_translate, _toc_file, gui_instance.append_log
                            )
                            new_translations: Dict[int, str] = dict(_reused_from_toc)

                            _has_api = hasattr(gui_instance, 'api_client') and gui_instance.api_client

                            try:
                                if _hdr_api_remaining and _has_api:
                                    gui_instance.append_log(
                                        f"🌐 Translating {len(_hdr_api_remaining)} new chapter header(s)..."
                                    )
                                    from metadata_batch_translator import BatchHeaderTranslator
                                    tr = BatchHeaderTranslator(gui_instance.api_client, config or {})

                                    if hasattr(gui_instance, '_batch_header_translator'):
                                        gui_instance._batch_header_translator = tr

                                    _api_trans = tr.translate_headers_batch(
                                        _hdr_api_remaining,
                                        batch_size=config.get('headers_per_batch', -1) if config else None,
                                        translation_type='header'
                                    ) or {}
                                    new_translations.update(_api_trans)
                                elif _hdr_api_remaining and not _has_api:
                                    gui_instance.append_log(
                                        f"⚠️ {len(_hdr_api_remaining)} new chapter(s) remaining and no API client — will use originals"
                                    )
                                elif not _hdr_api_remaining:
                                    gui_instance.append_log(
                                        "♻️ All new header entries were reused from cached file; skipping API call."
                                    )
                                
                                if new_translations:
                                    _api_count = len(new_translations) - len(_reused_from_toc)
                                    if _reused_from_toc and _api_count > 0:
                                        gui_instance.append_log(
                                            f"✅ Resolved {len(new_translations)} new chapter header(s) "
                                            f"({len(_reused_from_toc)} reused + {_api_count} translated)"
                                        )
                                    elif _reused_from_toc:
                                        gui_instance.append_log(
                                            f"✅ Reused {len(new_translations)} new chapter header(s) from cached file"
                                        )
                                    else:
                                        gui_instance.append_log(
                                            f"✅ Translated {len(new_translations)} new chapter header(s)"
                                        )

                                    # ── Rebuild the entire file sorted, with Output File for all entries ──
                                    # Merge originals: existing + new
                                    merged_source = dict(existing_source)
                                    merged_source.update(all_headers)
                                    
                                    # Merge translations: existing + new
                                    merged_trans = dict(existing_trans)
                                    merged_trans.update(new_translations)
                                    
                                    # Merge output files: existing + discover for ALL missing entries
                                    merged_out = dict(existing_out)
                                    # Build output dir listing once for efficiency
                                    try:
                                        _out_files_list = os.listdir(output_dir)
                                    except OSError:
                                        _out_files_list = []
                                    for num in merged_source:
                                        if num not in merged_out:
                                            # Try to find output file from spine order
                                            if 0 < num <= len(spine_order):
                                                src_file = spine_order[num - 1]
                                                src_bn = get_basename_without_ext(os.path.basename(src_file))
                                                # Check if an output file with this basename exists
                                                for candidate in _out_files_list:
                                                    cand_bn = get_basename_without_ext(candidate)
                                                    if cand_bn.startswith('response_'):
                                                        cand_bn = cand_bn[9:]
                                                    if cand_bn == src_bn:
                                                        # Strip response_ prefix and extensions
                                                        clean = get_basename_without_ext(candidate)
                                                        if clean.startswith('response_'):
                                                            clean = clean[9:]
                                                        merged_out[num] = clean
                                                        break
                                    
                                    # Write the entire file sorted by chapter number
                                    all_nums = sorted(merged_source.keys())
                                    with open(translations_file, 'w', encoding='utf-8') as f:
                                        f.write("Chapter Header Translations\n")
                                        f.write("=" * 50 + "\n\n")
                                        
                                        has_chapter_zero = 0 in all_nums
                                        if has_chapter_zero:
                                            f.write("Note: This novel uses 0-based chapter numbering (starts with Chapter 0)\n")
                                            f.write("-" * 50 + "\n\n")
                                        
                                        for num in all_nums:
                                            orig = merged_source.get(num, "Unknown")
                                            trans = merged_trans.get(num, orig)
                                            f.write(f"Chapter {num}:\n")
                                            f.write(f"  Original:   {orig}\n")
                                            f.write(f"  Translated: {trans}\n")
                                            if num in merged_out:
                                                f.write(f"  Output File: {merged_out[num]}\n")
                                            if num not in merged_trans:
                                                f.write("  Status:     ⚠️ Using original (translation failed)\n")
                                            f.write("-" * 40 + "\n")
                                        
                                        f.write(f"\nSummary:\n")
                                        f.write(f"Total chapters: {len(all_nums)}\n")
                                        if all_nums:
                                            f.write(f"Chapter range: {min(all_nums)} to {max(all_nums)}\n")
                                        total_translated = len(merged_trans)
                                        f.write(f"Successfully translated: {total_translated}\n")
                                    
                                    gui_instance.append_log(f"📝 Rebuilt translated_headers.txt with {len(new_translations)} new entries (sorted)")
                            except Exception as new_err:
                                gui_instance.append_log(f"⚠️ Failed to translate new headers: {new_err}")
                except Exception as recon_err:
                    gui_instance.append_log(f"⚠️ Reconciliation check failed: {recon_err} — proceeding with existing file")
                
                # ── Repair: sort entries and discover missing Output File fields ──
                try:
                    repair_translation_file(
                        translations_file, current_epub, output_dir,
                        log_callback=gui_instance.append_log
                    )
                    # Also repair TOC.txt if it exists
                    toc_txt_file = os.path.join(output_dir, "TOC.txt")
                    if os.path.exists(toc_txt_file):
                        repair_translation_file(
                            toc_txt_file, current_epub, output_dir,
                            log_callback=gui_instance.append_log
                        )
                except Exception as repair_err:
                    gui_instance.append_log(f"⚠️ Repair check failed: {repair_err}")
                
                # Use existing translations to update HTML files and toc.ncx
                gui_instance.append_log(f"   🔄 Will update HTML files using existing translations...")
                try:
                    result = apply_existing_translations(
                        epub_path=current_epub,
                        output_dir=output_dir,
                        translations_file=translations_file,
                        update_html=update_html,
                        log_callback=gui_instance.append_log
                    )
                    
                    if result:
                        gui_instance.append_log(f"✅ Successfully updated {len(result)} files using existing translations!")
                        if update_html:
                            gui_instance.append_log(f"🗂️ HTML files and toc.ncx updated in: {output_dir}")
                        successful += 1
                    else:
                        gui_instance.append_log(f"⚠️ No files were updated for: {epub_base}")
                        failed += 1
                    
                except Exception as e:
                    gui_instance.append_log(f"❌ Error applying existing translations: {e}")
                    failed += 1
                
                # Force GUI event processing
                try:
                    from PySide6.QtWidgets import QApplication
                    QApplication.processEvents()
                except Exception:
                    pass
                continue
            
            # Log starting message
            if total_files == 1:
                gui_instance.append_log("🌐 Starting standalone header translation...")
            
            # Run translation
            try:
                result = translate_headers_standalone(
                    epub_path=current_epub,
                    output_dir=output_dir,
                    api_client=gui_instance.api_client,
                    config=config,
                    update_html=update_html,
                    save_to_file=save_to_file,
                    log_callback=gui_instance.append_log,
                    gui_instance=gui_instance  # Pass GUI instance for stop button support
                )
            except KeyboardInterrupt:
                gui_instance.append_log("\n⛔ Translation interrupted by user")
                failed += 1
                break
            
            # Check if translation was stopped
            if hasattr(gui_instance, '_headers_stop_requested') and gui_instance._headers_stop_requested:
                gui_instance.append_log(f"⛔ Translation stopped for: {epub_base}")
                failed += 1
            # Log results
            elif result:
                gui_instance.append_log(f"✅ Successfully translated {len(result)} chapter headers!")
                # Show the translated_headers.txt file path only if saving was enabled
                if save_to_file:
                    translations_file = os.path.join(output_dir, "translated_headers.txt")
                    gui_instance.append_log(f"📄 Translations saved to: {translations_file}")
                if update_html:
                    gui_instance.append_log(f"🗂️ HTML files updated in: {output_dir}")
                successful += 1
            else:
                gui_instance.append_log(f"⚠️ No chapters were translated for: {epub_base}")
                failed += 1
            
            # Force GUI event processing after each file
            try:
                from PySide6.QtWidgets import QApplication
                QApplication.processEvents()
            except Exception:
                pass
        
        # Show summary if multiple files were processed
        if total_files > 1:
            gui_instance.append_log(f"\n{'='*60}")
            gui_instance.append_log(f"📊 Translation Summary:")
            gui_instance.append_log(f"  ✅ Successful: {successful}/{total_files}")
            if failed > 0:
                gui_instance.append_log(f"  ❌ Failed: {failed}/{total_files}")
            gui_instance.append_log(f"{'='*60}")
    
    except Exception as e:
        import traceback
        error_msg = f"Error during header translation: {e}\n\n{traceback.format_exc()}"
        gui_instance.append_log(f"❌ {error_msg}")
        QMessageBox.critical(None, "Error", error_msg)
