#!/usr/bin/env python3
"""Worker entry point for SDLXLIFF extraction."""

import io
import json
import os
import sys
import traceback

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def run_sdlxliff_extraction(sdlxliff_path, output_dir):
    try:
        from sdlxliff_extractor import extract_sdlxliff_to_chapters

        print(f"[INFO] Starting SDLXLIFF extraction: {sdlxliff_path}", flush=True)
        print(f"[INFO] Output directory: {output_dir}", flush=True)
        result = extract_sdlxliff_to_chapters(sdlxliff_path, output_dir)
        print(f"[PROGRESS] Extracted {result.get('chapters', 0)}/{result.get('segments', 0)} eligible segment(s)", flush=True)
        print(f"[RESULT] {json.dumps(result, ensure_ascii=False)}", flush=True)
        return result
    except Exception as exc:
        error = {
            "success": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        print(f"[ERROR] {exc}", flush=True)
        print(f"[RESULT] {json.dumps(error, ensure_ascii=False)}", flush=True)
        return error


def main():
    if len(sys.argv) < 3:
        print("[ERROR] Usage: sdlxliff_extraction_worker.py <sdlxliff_path> <output_dir>", flush=True)
        sys.exit(1)
    sdlxliff_path = sys.argv[1]
    output_dir = sys.argv[2]
    if not os.path.exists(sdlxliff_path):
        print(f"[ERROR] SDLXLIFF file not found: {sdlxliff_path}", flush=True)
        sys.exit(1)
    os.makedirs(output_dir, exist_ok=True)
    result = run_sdlxliff_extraction(sdlxliff_path, output_dir)
    sys.exit(0 if result.get("success") else 1)


if __name__ == "__main__":
    from shutdown_utils import run_cli_main

    def _main():
        try:
            import multiprocessing

            multiprocessing.freeze_support()
        except Exception:
            pass
        main()
        return 0

    run_cli_main(_main)
