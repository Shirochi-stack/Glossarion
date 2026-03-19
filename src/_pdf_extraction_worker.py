#!/usr/bin/env python3
"""
PDF Extraction Worker - Runs PDF extraction in a separate process to prevent GUI freezing.
Follows the same pattern as _pdf_worker.py.

Usage:
    python _pdf_extraction_worker.py <config_path>
    OR (frozen builds):
    glossarion.exe --run-pdf-extraction <config_path>

Config JSON:
    {
        "pdf_path": "/path/to/file.pdf",
        "output_dir": "/path/to/output",
        "render_mode": "xhtml",
        "extract_images": true,
        "generate_css": true,
        "html2text": false,
        "css_override_path": "",
        "attach_css_enabled": false
    }

Output:
    Writes result JSON to <output_dir>/_pdf_extraction_result.json
    Streams [PROGRESS] messages to stdout for real-time log forwarding
"""

import json
import os
import sys
import time
import traceback

# Module-level log queue, set by run_pdf_extraction when called from ProcessPoolExecutor
_log_queue = None


def log(msg):
    """Send a progress message to the parent process (via queue) or print to stdout."""
    try:
        if _log_queue is not None:
            _log_queue.put(msg)
        else:
            print(f"[PROGRESS] {msg}", flush=True)
    except Exception:
        pass


def run_pdf_extraction(config_path, log_queue=None):
    """Main PDF extraction logic, running in subprocess.
    
    Args:
        config_path: Path to the JSON config file with extraction parameters.
        log_queue: Optional multiprocessing.Queue for forwarding log messages
                   to the parent process (used by ProcessPoolExecutor).
    """
    global _log_queue
    _log_queue = log_queue
    
    # Redirect stdout so that ALL print() calls (including from pdf_extractor.py)
    # are forwarded through the log queue to the GUI
    _original_stdout = sys.stdout
    if log_queue is not None:
        class _QueueWriter:
            """Wraps a multiprocessing.Queue to look like sys.stdout."""
            def write(self, msg):
                if msg and msg.strip():
                    try:
                        # Strip [PROGRESS] prefix if log() already added it
                        clean = msg.strip()
                        if clean.startswith("[PROGRESS] "):
                            clean = clean[len("[PROGRESS] "):]
                        log_queue.put(clean)
                    except Exception:
                        pass
            def flush(self):
                pass
        sys.stdout = _QueueWriter()
    
    try:
        return _run_pdf_extraction_inner(config_path)
    finally:
        sys.stdout = _original_stdout


def _run_pdf_extraction_inner(config_path):
    """Inner extraction logic (separated so stdout redirect works in a try/finally)."""
    start_time = time.time()

    # Load config
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        log(f"❌ Failed to load config: {e}")
        return {"success": False, "error": f"Config load failed: {e}"}

    pdf_path = config.get("pdf_path", "")
    output_dir = config.get("output_dir", "")
    render_mode = config.get("render_mode", "xhtml").lower()
    extract_images = config.get("extract_images", True)
    generate_css = config.get("generate_css", True)
    html2text_enabled = config.get("html2text", False)
    css_override_path = config.get("css_override_path", "").strip()
    attach_css_enabled = config.get("attach_css_enabled", False)
    stop_file = config.get("stop_file", "")

    # Make stop file path available to pdf_extractor via env var
    if stop_file:
        os.environ['PDF_EXTRACTION_STOP_FILE'] = stop_file

    def _is_stopped():
        return stop_file and os.path.exists(stop_file)

    if not pdf_path or not os.path.isfile(pdf_path):
        log(f"❌ PDF file not found: {pdf_path}")
        return {"success": False, "error": f"PDF file not found: {pdf_path}"}

    os.makedirs(output_dir, exist_ok=True)

    # Set environment variables that pdf_extractor reads
    os.environ["PDF_RENDER_MODE"] = render_mode
    os.environ["PDF_EXTRACT_IMAGES"] = "1" if extract_images else "0"
    os.environ["PDF_GENERATE_CSS"] = "1" if generate_css else "0"

    # Import pdf_extractor
    try:
        from pdf_extractor import extract_pdf_with_formatting, generate_css_from_pdf
    except ImportError as e:
        log(f"❌ Could not import pdf_extractor: {e}")
        return {"success": False, "error": f"Import failed: {e}"}

    # Phase 1: Extract PDF content with formatting
    log(f"📄 Extracting PDF: {os.path.basename(pdf_path)} (render mode: {render_mode})")

    if _is_stopped():
        log("🛑 PDF extraction cancelled before start")
        return {"success": False, "error": "Cancelled by user"}

    try:
        content, images_info = extract_pdf_with_formatting(
            pdf_path,
            output_dir,
            extract_images=extract_images,
            page_by_page=True
        )
    except Exception as e:
        log(f"❌ PDF extraction failed: {e}")
        traceback.print_exc()
        return {"success": False, "error": f"Extraction failed: {e}"}

    # Phase 2: Generate CSS if enabled
    if _is_stopped():
        log("🛑 PDF extraction cancelled after content extraction")
        return {"success": False, "error": "Cancelled by user"}

    css_generated = False
    if generate_css:
        css_path = os.path.join(output_dir, 'styles.css')

        if css_override_path and os.path.exists(css_override_path) and attach_css_enabled:
            try:
                import shutil
                shutil.copy2(css_override_path, css_path)
                log(f"✅ Using loaded CSS (overrides PDF-generated CSS): {os.path.basename(css_override_path)}")
                css_generated = True
            except Exception as e:
                log(f"⚠️ Failed to copy loaded CSS, generating from PDF instead: {e}")

        if not css_generated:
            if css_override_path and not attach_css_enabled:
                log("ℹ️ CSS override set but 'Attach CSS to Chapters' is disabled - generating from PDF")

            try:
                css_content = generate_css_from_pdf(pdf_path)
                with open(css_path, 'w', encoding='utf-8') as f:
                    f.write(css_content)
                log("✅ Generated styles.css from PDF")
                css_generated = True
            except Exception as e:
                log(f"⚠️ CSS generation failed: {e}")

    # Phase 3: Convert to markdown if html2text is enabled
    if html2text_enabled and isinstance(content, list):
        try:
            import html2text
            h = html2text.HTML2Text()
            h.body_width = 0
            h.unicode_snob = True
            h.images_as_html = True
            h.images_to_alt = False
            h.protect_links = True

            converted = []
            for page_num, page_html in content:
                converted.append((page_num, h.handle(page_html)))
            content = converted
            log("✅ Converted HTML to Markdown")
        except ImportError:
            log("⚠️ html2text not available, keeping HTML format")
        except Exception as e:
            log(f"⚠️ Failed to convert HTML to markdown: {e}")

    # Write results to output JSON
    result_path = os.path.join(output_dir, '_pdf_extraction_result.json')

    # Serialize content: list of [page_num, html] pairs
    serialized_content = []
    if isinstance(content, list):
        for item in content:
            if isinstance(item, tuple) and len(item) == 2:
                serialized_content.append([item[0], item[1]])
            else:
                serialized_content.append(item)
    else:
        # Single string content
        serialized_content = content

    # Serialize images_info (convert int keys to strings for JSON, Rect objects to lists)
    serialized_images = {}
    if isinstance(images_info, dict):
        for k, v in images_info.items():
            page_images = []
            if isinstance(v, list):
                for img in v:
                    if isinstance(img, dict):
                        img_copy = dict(img)
                        # Convert fitz.Rect to plain list for JSON serialization
                        if 'bbox' in img_copy:
                            bbox = img_copy['bbox']
                            try:
                                img_copy['bbox'] = [float(bbox.x0), float(bbox.y0), float(bbox.x1), float(bbox.y1)]
                            except (AttributeError, TypeError):
                                img_copy['bbox'] = list(bbox) if bbox else [0, 0, 0, 0]
                        page_images.append(img_copy)
                    else:
                        page_images.append(img)
            serialized_images[str(k)] = page_images

    elapsed = time.time() - start_time

    result = {
        "success": True,
        "content": serialized_content,
        "images_info": serialized_images,
        "css_generated": css_generated,
        "is_page_list": isinstance(content, list),
        "page_count": len(serialized_content) if isinstance(serialized_content, list) else 1,
        "elapsed": round(elapsed, 1)
    }

    try:
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False)
        log(f"✅ PDF extraction complete: {result['page_count']} pages in {elapsed:.1f}s")
    except Exception as e:
        log(f"❌ Failed to write result: {e}")
        return {"success": False, "error": f"Failed to write result: {e}"}

    # Also print the result path so the manager knows where to find it
    print(f"[RESULT] {json.dumps({'success': True, 'result_path': result_path, 'page_count': result['page_count'], 'elapsed': result['elapsed']})}", flush=True)

    return result


def main():
    """Entry point for standalone execution."""
    if len(sys.argv) < 2:
        print("Usage: python _pdf_extraction_worker.py <config_path>", file=sys.stderr)
        sys.exit(1)

    config_path = sys.argv[1]
    result = run_pdf_extraction(config_path)

    if not result.get("success"):
        sys.exit(1)


if __name__ == "__main__":
    main()
