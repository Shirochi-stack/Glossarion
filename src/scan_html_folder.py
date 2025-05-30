import os
import hashlib
import json
import csv
from bs4 import BeautifulSoup
from langdetect import detect, LangDetectException
from difflib import SequenceMatcher
from collections import Counter
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import re

# Global flag to allow stopping the scan externally
_stop_flag = False

def stop_scan():
    """Set the stop flag to True"""
    global _stop_flag
    _stop_flag = True

def extract_text_from_html(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, "html.parser")
        return soup.get_text(separator='\n', strip=True)

def is_similar(text1, text2, threshold=0.95):
    return SequenceMatcher(None, text1, text2).ratio() >= threshold

def has_no_spacing_or_linebreaks(text, space_threshold=0.01):
    space_ratio = text.count(" ") / max(1, len(text))
    newline_count = text.count("\n")
    return space_ratio < space_threshold or newline_count == 0

def has_repeating_sentences(text, min_repeats=5):
    sentences = [s.strip() for s in text.replace("\n", " ").split('.') if s.strip()]
    if len(sentences) < min_repeats:
        return False
    counter = Counter(sentences)
    for sent, count in counter.items():
        if count >= min_repeats and len(sent) > 10:
            return True
    return False

def scan_html_folder(folder_path, log=print, stop_flag=None):
    global _stop_flag
    _stop_flag = False
    hashes = []
    texts = []
    results = []
    chapter_contents = {}  # Store chapter content for duplicate detection
    html_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".html")]
    html_files.sort()

    log(f"🔍 Found {len(html_files)} HTML files. Starting scan...")

    # First pass: collect all chapter data and detect issues
    for idx, filename in enumerate(html_files):
        if stop_flag and stop_flag():
            log("⛔ QA scan interrupted by user.")
            return
        
        full_path = os.path.join(folder_path, filename)
        try:
            raw_text = extract_text_from_html(full_path)
        except Exception as e:
            log(f"⚠️ Failed to read {filename}: {e}")
            continue

        norm_text = raw_text.strip().lower()
        if len(norm_text) < 100:
            log(f"⚠️ Skipped {filename}: Too short")
            continue

        # Extract chapter number from filename
        chapter_num = None
        m = re.match(r"response_(\d+)_", filename)
        if m:
            chapter_num = int(m.group(1))
            # Store the normalized text for this chapter number
            if chapter_num not in chapter_contents:
                chapter_contents[chapter_num] = []
            chapter_contents[chapter_num].append({
                'filename': filename,
                'norm_text': norm_text,
                'idx': idx
            })

        issues = []
        preview = raw_text[:500].replace('\n', ' ') + '...'

        # Check for exact duplicates using hash
        hash_digest = hashlib.md5(norm_text.encode("utf-8")).hexdigest()
        if hash_digest in hashes:
            issues.append("duplicate")
        else:
            # Check for similar content (not exact match)
            for prev_text in texts:
                if is_similar(norm_text, prev_text):
                    issues.append("duplicate")
                    break
            else:
                hashes.append(hash_digest)
                texts.append(norm_text)

        # Language detection
        try:
            lang = detect(norm_text)
            if lang != 'en':
                issues.append(f"non_english ({lang})")
        except LangDetectException:
            issues.append("non_english (unknown)")

        # Spacing/formatting issues
        if has_no_spacing_or_linebreaks(raw_text):
            issues.append("no_spacing_or_linebreaks")

        # Repetitive content
        if has_repeating_sentences(raw_text):
            issues.append("repetitive_sentences")

        results.append({
            "file_index": idx,
            "filename": filename,
            "filepath": full_path,
            "issues": issues,
            "preview": preview,
            "score": len(issues),
            "chapter_num": chapter_num
        })

        log(f"📄 {filename}: {', '.join(issues) if issues else '✅ OK'}")

    # Second pass: detect chapters with the same number that have similar content
    duplicate_chapter_numbers = []
    for chapter_num, entries in chapter_contents.items():
        if len(entries) > 1:
            # Multiple files for the same chapter number
            log(f"⚠️ Found {len(entries)} files for Chapter {chapter_num}")
            # Mark all but the first as duplicates
            for i in range(1, len(entries)):
                for result in results:
                    if result['filename'] == entries[i]['filename']:
                        if 'duplicate_chapter' not in result['issues']:
                            result['issues'].append(f'duplicate_chapter_{chapter_num}')
                            result['score'] = len(result['issues'])
                duplicate_chapter_numbers.append(chapter_num)

    # Generate reports
    output_dir = os.path.basename(folder_path.rstrip('/\\')) + "_Scan Report"
    output_path = os.path.join(folder_path, output_dir)
    os.makedirs(output_path, exist_ok=True)

    with open(os.path.join(output_path, "validation_results.json"), "w", encoding="utf-8") as jf:
        json.dump(results, jf, indent=2, ensure_ascii=False)

    with open(os.path.join(output_path, "validation_results.csv"), "w", encoding="utf-8", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=["file_index", "filename", "score", "issues"])
        writer.writeheader()
        for row in results:
            writer.writerow({
                "file_index": row["file_index"],
                "filename": row["filename"],
                "score": row["score"],
                "issues": "; ".join(row["issues"])
            })

    html_report = "<html><head><meta charset='utf-8'><title>Translation QA Report</title></head><body>"
    html_report += "<h1>Translation QA Report</h1>"
    html_report += f"<p>Total Files Scanned: {len(results)}</p>"
    html_report += f"<p>Files with Issues: {sum(1 for r in results if r['issues'])}</p>"
    html_report += f"<p>Clean Files: {sum(1 for r in results if not r['issues'])}</p>"
    
    if duplicate_chapter_numbers:
        html_report += f"<p style='color:red;'>⚠️ Found duplicate chapter numbers: {sorted(set(duplicate_chapter_numbers))}</p>"
    
    html_report += "<table border='1'><tr><th>#</th><th>Filename</th><th>Issues</th><th>Preview</th></tr>"

    for row in results:
        # Use relative path for HTML links to work properly
        link = f"<a href='../{row['filename']}' target='_blank'>{row['filename']}</a>"
        issues_str = "; ".join(row["issues"])
        # Highlight rows with issues
        row_style = ' style="background-color: #ffeeee;"' if row["issues"] else ''
        html_report += f"<tr{row_style}><td>{row['file_index']}</td><td>{link}</td><td>{issues_str}</td><td>{row['preview']}</td></tr>"

    html_report += "</table></body></html>"

    with open(os.path.join(output_path, "validation_results.html"), "w", encoding="utf-8") as html_file:
        html_file.write(html_report)

    # Fix: Update progress file correctly using file indices, not chapter numbers
    prog_path = os.path.join(folder_path, "translation_progress.json")
    try:
        with open(prog_path, "r", encoding="utf-8") as pf:
            prog = json.load(pf)
            existing = prog.get("completed", [])
    except FileNotFoundError:
        existing = []

    # Build list of FILE INDICES that have issues (not chapter numbers)
    faulty_indices = []
    for row in results:
        if row["issues"]:
            # Get the file index from the results
            faulty_indices.append(row["file_index"])

    # Remove faulty indices from the completed list
    updated = [idx for idx in existing if idx not in faulty_indices]

    # Write back the pruned history
    with open(prog_path, "w", encoding="utf-8") as pf:
        json.dump({"completed": updated}, pf, indent=2)

    log(f"[QA Scan] Removed file indices with issues: {sorted(faulty_indices)}")
    log(f"[QA Scan] Chapter numbers affected: {sorted(set(r['chapter_num'] for r in results if r['issues'] and r['chapter_num']))}")

    log("\n✅ Validation complete.")
    log(f" - {output_path}/validation_results.json")
    log(f" - {output_path}/validation_results.csv")
    log(f" - {output_path}/validation_results.html")
    log(f" - Pruned {len(faulty_indices)} indices from {prog_path}; {len(updated)} chapters remain")

def launch_gui():
    def run_scan():
        folder_path = filedialog.askdirectory(title="Select Folder with HTML Files")
        if folder_path:
            threading.Thread(target=scan_html_folder, args=(folder_path,), daemon=True).start()

    root = tk.Tk()
    root.title("Translation QA Scanner")
    root.geometry("400x100")
    scan_button = tk.Button(root, text="Scan Folder for QA Issues", command=run_scan)
    scan_button.pack(pady=20)
    root.mainloop()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        launch_gui()
    else:
        scan_html_folder(sys.argv[1])
