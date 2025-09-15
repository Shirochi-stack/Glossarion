#!/usr/bin/env python3
"""
Chapter Extraction Worker - Runs chapter extraction in a separate process to prevent GUI freezing
"""

import sys
import os
import io

# Force UTF-8 encoding for stdout/stderr on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import json
import zipfile
import time
import traceback
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def run_chapter_extraction(epub_path, output_dir, extraction_mode="smart", progress_callback=None):
    """
    Run chapter extraction in this worker process
    
    Args:
        epub_path: Path to EPUB file
        output_dir: Output directory for extracted content
        extraction_mode: Extraction mode (smart, comprehensive, full, enhanced)
        progress_callback: Callback function for progress updates (uses print for IPC)
    
    Returns:
        dict: Extraction results including chapters and metadata
    """
    try:
        # Import here to avoid loading heavy modules until needed
        from TransateKRtoEN import ChapterExtractor
        
        # Create progress callback that prints to stdout for IPC
        def worker_progress_callback(message):
            # Use special prefix for progress messages
            print(f"[PROGRESS] {message}", flush=True)
        
        # Create extractor with progress callback
        extractor = ChapterExtractor(progress_callback=worker_progress_callback)
        
        # Set extraction mode
        os.environ["EXTRACTION_MODE"] = extraction_mode
        
        # Open EPUB and extract chapters
        print(f"[INFO] Starting extraction of: {epub_path}", flush=True)
        print(f"[INFO] Output directory: {output_dir}", flush=True)
        print(f"[INFO] Extraction mode: {extraction_mode}", flush=True)
        
        with zipfile.ZipFile(epub_path, 'r') as zf:
            # Extract metadata first
            metadata = extractor._extract_epub_metadata(zf)
            print(f"[INFO] Extracted metadata: {list(metadata.keys())}", flush=True)
            
            # Extract chapters
            chapters = extractor.extract_chapters(zf, output_dir)
            
            print(f"[INFO] Extracted {len(chapters)} chapters", flush=True)
            
            # The extract_chapters method already handles OPF sorting internally
            # Just log if OPF was used
            opf_path = os.path.join(output_dir, 'content.opf')
            if os.path.exists(opf_path):
                print(f"[INFO] OPF file available for chapter ordering", flush=True)
            
            # CRITICAL: Save the full chapters with body content!
            # This is what the main process needs to load
            chapters_full_path = os.path.join(output_dir, "chapters_full.json")
            try:
                with open(chapters_full_path, 'w', encoding='utf-8') as f:
                    json.dump(chapters, f, ensure_ascii=False)
                print(f"[INFO] Saved full chapters data to: {chapters_full_path}", flush=True)
            except Exception as e:
                print(f"[WARNING] Could not save full chapters: {e}", flush=True)
                # Fall back to saving individual files
                for chapter in chapters:
                    try:
                        chapter_file = f"chapter_{chapter['num']:04d}_{chapter.get('filename', 'content').replace('/', '_')}.html"
                        chapter_path = os.path.join(output_dir, chapter_file)
                        with open(chapter_path, 'w', encoding='utf-8') as f:
                            f.write(chapter.get('body', ''))
                        print(f"[INFO] Saved chapter {chapter['num']} to {chapter_file}", flush=True)
                    except Exception as ce:
                        print(f"[WARNING] Could not save chapter {chapter.get('num')}: {ce}", flush=True)
            
            # Return results as JSON for IPC
            result = {
                "success": True,
                "chapters": len(chapters),
                "metadata": metadata,
                "chapter_info": [
                    {
                        "num": ch.get("num"),
                        "title": ch.get("title"),
                        "has_images": ch.get("has_images", False),
                        "file_size": ch.get("file_size", 0),
                        "content_hash": ch.get("content_hash", "")
                    }
                    for ch in chapters
                ]
            }
            
            # Output result as JSON
            print(f"[RESULT] {json.dumps(result)}", flush=True)
            return result
            
    except Exception as e:
        # Send error information
        error_info = {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        print(f"[ERROR] {str(e)}", flush=True)
        print(f"[RESULT] {json.dumps(error_info)}", flush=True)
        return error_info


def main():
    """Main entry point for worker process"""
    
    # Parse command line arguments
    if len(sys.argv) < 3:
        print("[ERROR] Usage: chapter_extraction_worker.py <epub_path> <output_dir> [extraction_mode]", flush=True)
        sys.exit(1)
    
    epub_path = sys.argv[1]
    output_dir = sys.argv[2]
    extraction_mode = sys.argv[3] if len(sys.argv) > 3 else "smart"
    
    # Validate inputs
    if not os.path.exists(epub_path):
        print(f"[ERROR] EPUB file not found: {epub_path}", flush=True)
        sys.exit(1)
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Run extraction
    result = run_chapter_extraction(epub_path, output_dir, extraction_mode)
    
    # Exit with appropriate code
    sys.exit(0 if result.get("success", False) else 1)


if __name__ == "__main__":
    main()