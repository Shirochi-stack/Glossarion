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
    Get filename without extension
    
    Args:
        filename: The filename
        
    Returns:
        Filename without extension
    """
    return os.path.splitext(filename)[0]


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
                    for tag_name in ['h1', 'h2', 'h3', 'title']:
                        tag = soup.find(tag_name)
                        if tag:
                            text = tag.get_text().strip()
                            if text:
                                title = text
                                break
                    
                    if not title:
                        p = soup.find('p')
                        if p:
                            text = p.get_text().strip()
                            if text and len(text) < 100:
                                title = text
                    
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
    log_callback=None
) -> Dict[str, Tuple[str, str, str]]:
    """
    Match output HTML files to source chapters by checking if source basename appears in output filename
    
    Args:
        output_dir: Directory containing translated HTML files
        source_mapping: Mapping of source basename (no ext) to title
        spine_order: List of source filenames in spine order
        log_callback: Optional callback for logging
        
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
    output_files = sorted([
        f for f in os.listdir(output_dir) 
        if f.lower().endswith(html_extensions)
    ])
    
    log(f"📁 Found {len(output_files)} HTML files in output directory")
    log(f"📚 Have {len(source_mapping)} source chapters to match")
    
    if not output_files:
        log("⚠️ No HTML files found in output directory!")
        return matches
    
    matched_count = 0
    skipped_count = 0
    
    for output_file in output_files:
        # Get output filename without extension
        output_no_ext = get_basename_without_ext(output_file)
        
        # Try to match with each source chapter
        matched = False
        for source_basename, source_title in source_mapping.items():
            # Check if source basename appears in output filename (handles response_ prefix)
            if source_basename in output_no_ext:
                # Read current title from output file
                try:
                    output_path = os.path.join(output_dir, output_file)
                    with open(output_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    soup = BeautifulSoup(content, 'html.parser')
                    
                    current_title = None
                    for tag_name in ['h1', 'h2', 'h3', 'title']:
                        tag = soup.find(tag_name)
                        if tag:
                            text = tag.get_text().strip()
                            if text:
                                current_title = text
                                break
                    
                    if not current_title:
                        current_title = f"Chapter {source_basename}"
                    
                    matches[output_file] = (source_title, current_title, output_file)
                    matched_count += 1
                    matched = True
                    
                    if matched_count <= 5:
                        log(f"  ✓ Matched: {output_file}")
                        log(f"    Contains source: '{source_basename}'")
                        log(f"    Source title: '{source_title}'")
                        log(f"    Current title: '{current_title}'")
                    
                    break  # Found a match, stop checking other sources
                    
                except Exception as e:
                    log(f"  ⚠️ Error reading {output_file}: {e}")
                    break
        
        if not matched:
            skipped_count += 1
            if skipped_count <= 3:
                log(f"  ⊝ Skipped (no match): {output_file}")
    
    log(f"\n📊 Matching results:")
    log(f"  ✓ Matched: {matched_count} chapters")
    log(f"  ⊝ Skipped: {skipped_count} chapters (no match)")
    
    return matches


def load_translations_from_file(translations_file: str, log_callback=None) -> Tuple[Dict[int, str], Dict[int, str]]:
    """
    Load translations from the translated_headers.txt file
    
    Args:
        translations_file: Path to translated_headers.txt
        log_callback: Optional callback for logging
        
    Returns:
        Tuple of (source_headers, translated_headers) where both are dicts mapping chapter numbers to titles
    """
    def log(message):
        if log_callback:
            log_callback(message)
        else:
            print(message)
    
    source_headers = {}
    translated_headers = {}
    
    try:
        with open(translations_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Parse the file format
        # Format: Chapter X: "Source Title" -> "Translated Title"
        for line in lines:
            line = line.strip()
            if not line or line.startswith('Translation Summary') or line.startswith('Total chapters') or line.startswith('Successfully') or line.startswith('Failed'):
                continue
            
            # Parse chapter number and titles
            import re
            match = re.match(r'Chapter (\d+): "([^"]+)" -> "([^"]+)"', line)
            if match:
                chapter_num = int(match.group(1))
                source_title = match.group(2)
                translated_title = match.group(3)
                source_headers[chapter_num] = source_title
                translated_headers[chapter_num] = translated_title
        
        log(f"📋 Loaded {len(translated_headers)} translations from file")
        
    except Exception as e:
        log(f"⚠️ Error loading translations: {e}")
    
    return source_headers, translated_headers


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
    source_headers, translated_headers = load_translations_from_file(translations_file, log_callback)
    
    if not translated_headers:
        log("⚠️ No translations found in file")
        return {}
    
    # Step 2: Extract source chapters with OPF mapping to match against loaded translations
    log("📚 Extracting source chapter information from EPUB...")
    source_mapping, spine_order = extract_source_chapters_with_opf_mapping(epub_path, log_callback)
    
    # Step 3: Match output files to source chapters
    log("🔗 Matching output files to source chapters...")
    matches = match_output_to_source_chapters(output_dir, source_mapping, spine_order, log_callback)
    
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
    
    # Call translate_and_save_headers - IDENTICAL TO PIPELINE
    # This method uses the EXACT same translation prompts, HTML update logic, and file saving
    translated_headers = translator.translate_and_save_headers(
        html_dir=output_dir,
        headers_dict=headers_to_translate,
        batch_size=config.get('headers_per_batch', 350) if config else 350,
        output_dir=output_dir,
        update_html=update_html,  # Uses _update_html_headers_exact - same as pipeline
        save_to_file=save_to_file,  # Saves to translated_headers.txt - same as pipeline
        current_titles=current_titles_map  # Enables exact title replacement
    )
    
    # Step 5: Map back to output filenames
    log("\nStep 5: Mapping translations to output files...")
    result = {}
    for idx, translated_title in translated_headers.items():
        if idx in current_titles_map:
            output_file = current_titles_map[idx]['filename']
            result[output_file] = translated_title
            log(f"  {output_file}: {translated_title}")
    
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
            'headers_per_batch': int(os.getenv('HEADERS_PER_BATCH', '350')),
            'temperature': float(os.getenv('TRANSLATION_TEMPERATURE', '0.3')),
            'max_tokens': int(os.getenv('MAX_OUTPUT_TOKENS', '12000')),
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
            'headers_per_batch': int(getattr(gui_instance, 'headers_per_batch_var', 350)),
            'temperature': float(os.getenv('TRANSLATION_TEMPERATURE', '0.3')),
            'max_tokens': int(os.getenv('MAX_OUTPUT_TOKENS', '12000')),
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
            # Check if stop was requested
            if hasattr(gui_instance, '_headers_stop_requested') and gui_instance._headers_stop_requested:
                gui_instance.append_log("\n⛔ Translation stopped by user")
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
                gui_instance.append_log(f"   🔄 Will update HTML files using existing translations...")
                
                # Use existing translations to update HTML files and toc.ncx
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
            
            # Log results
            if result:
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
