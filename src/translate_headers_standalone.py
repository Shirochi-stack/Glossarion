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
                                
                                # Skip navigation/toc files
                                if not any(skip in basename for skip in skip_keywords):
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
                epub_html_files = sorted([
                    f for f in zf.namelist() 
                    if f.endswith(('.html', '.xhtml', '.htm')) 
                    and not f.startswith('__MACOSX')
                    and not any(skip in os.path.basename(f).lower() for skip in skip_keywords)
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


def translate_headers_standalone(
    epub_path: str,
    output_dir: str,
    api_client,
    config: dict = None,
    update_html: bool = True,
    save_to_file: bool = True,
    log_callback=None
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


def run_translate_headers_gui(gui_instance):
    """
    GUI wrapper for standalone header translation
    
    Args:
        gui_instance: The GUI instance (translator_gui or other_settings)
    """
    from PySide6.QtWidgets import QMessageBox
    
    try:
        # Get EPUB path
        epub_path = gui_instance.get_current_epub_path()
        if not epub_path or not os.path.exists(epub_path):
            QMessageBox.critical(
                None, 
                "Error", 
                "No EPUB file selected or file does not exist."
            )
            return
        
        # Get output directory
        epub_base = os.path.splitext(os.path.basename(epub_path))[0]
        current_dir = os.getcwd()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Try multiple locations
        candidates = [
            os.path.join(os.path.dirname(epub_path), epub_base),  # Same dir as EPUB
            os.path.join(current_dir, epub_base),                 # Current working dir
            os.path.join(script_dir, epub_base),                  # Script dir
            os.path.join(script_dir, '..', epub_base),            # Parent of script dir
        ]
        
        output_dir = None
        
        for candidate in candidates:
            if os.path.isdir(candidate):
                # Verify it actually has HTML files
                try:
                    files = os.listdir(candidate)
                    html_files = [f for f in files if f.lower().endswith(('.html', '.xhtml', '.htm'))]
                    if html_files:
                        output_dir = candidate
                        break
                except Exception:
                    continue
        
        if not output_dir or not os.path.exists(output_dir):
            from PySide6.QtGui import QIcon
            icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Halgakos.ico")
            icon = QIcon(icon_path) if os.path.exists(icon_path) else QIcon()
            
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Critical)
            msg_box.setWindowTitle("Error")
            msg_box.setText(f"Output directory not found for: {epub_base}")
            msg_box.setInformativeText(
                "Please extract the EPUB first.\n\n"
                "The tool checked these locations:\n" + "\n".join([f"  - {c}" for c in candidates[:3]])
            )
            msg_box.setWindowIcon(icon)
            msg_box.exec()
            return
        
        # Get API client
        if not hasattr(gui_instance, 'api_client') or not gui_instance.api_client:
            QMessageBox.critical(
                None, 
                "Error", 
                "API client not initialized. Please check your API settings."
            )
            return
        
        # Get config from GUI
        config = {
            'headers_per_batch': int(getattr(gui_instance, 'headers_per_batch_var', 350)),
            'temperature': float(os.getenv('TRANSLATION_TEMPERATURE', '0.3')),
            'max_tokens': int(os.getenv('MAX_OUTPUT_TOKENS', '12000')),
        }
        
        # Get options
        update_html = getattr(gui_instance, 'update_html_headers_var', True)
        save_to_file = getattr(gui_instance, 'save_header_translations_var', True)
        
        gui_instance.append_log("🌐 Starting standalone header translation...")
        
        # Run translation
        result = translate_headers_standalone(
            epub_path=epub_path,
            output_dir=output_dir,
            api_client=gui_instance.api_client,
            config=config,
            update_html=update_html,
            save_to_file=save_to_file,
            log_callback=gui_instance.append_log
        )
        
        # Log results instead of showing message box
        if result:
            gui_instance.append_log(f"✅ Successfully translated {len(result)} chapter headers!")
            # Show the translated_headers.txt file path only if saving was enabled
            if save_to_file:
                translations_file = os.path.join(output_dir, "translated_headers.txt")
                gui_instance.append_log(f"📄 Translations saved to: {translations_file}")
            if update_html:
                gui_instance.append_log(f"🗂️ HTML files updated in: {output_dir}")
        else:
            gui_instance.append_log("⚠️ No chapters were translated. Please check the logs above.")
    
    except Exception as e:
        import traceback
        error_msg = f"Error during header translation: {e}\n\n{traceback.format_exc()}"
        gui_instance.append_log(f"❌ {error_msg}")
        QMessageBox.critical(None, "Error", error_msg)
