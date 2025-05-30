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
import unicodedata

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

def is_similar(text1, text2, threshold=0.85):
    return SequenceMatcher(None, text1, text2).ratio() >= threshold

def has_no_spacing_or_linebreaks(text, space_threshold=0.01):
    space_ratio = text.count(" ") / max(1, len(text))
    newline_count = text.count("\n")
    return space_ratio < space_threshold or newline_count == 0

def has_repeating_sentences(text, min_repeats=10):
    # More sophisticated repetition detection
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip() and len(s.strip()) > 20]
    if len(sentences) < min_repeats:
        return False
    
    # Count exact repetitions
    counter = Counter(sentences)
    
    # Check for truly repetitive content (not just common phrases)
    for sent, count in counter.items():
        # Ignore common dialogue patterns and short sentences
        if count >= min_repeats and len(sent) > 50:
            # Check if it's not just dialogue attribution
            if not any(pattern in sent.lower() for pattern in ['said', 'asked', 'replied', 'thought']):
                return True
    return False

def detect_non_english_content(text):
    """Detect ONLY non-Latin script characters (not romanized text)"""
    issues = []
    
    # Define ALL non-Latin script Unicode ranges with friendly names
    non_latin_ranges = [
        # Korean
        (0xAC00, 0xD7AF, 'Korean'),
        (0x1100, 0x11FF, 'Korean'),
        (0x3130, 0x318F, 'Korean'),
        (0xA960, 0xA97F, 'Korean'),
        (0xD7B0, 0xD7FF, 'Korean'),
        
        # Japanese
        (0x3040, 0x309F, 'Japanese'),
        (0x30A0, 0x30FF, 'Japanese'),
        (0x31F0, 0x31FF, 'Japanese'),
        (0xFF65, 0xFF9F, 'Japanese'),
        
        # Chinese
        (0x4E00, 0x9FFF, 'Chinese'),
        (0x3400, 0x4DBF, 'Chinese'),
        (0x20000, 0x2A6DF, 'Chinese'),
        (0x2A700, 0x2B73F, 'Chinese'),
        
        # Other scripts
        (0x0590, 0x05FF, 'Hebrew'),
        (0x0600, 0x06FF, 'Arabic'),
        (0x0700, 0x074F, 'Syriac'),
        (0x0750, 0x077F, 'Arabic'),
        (0x0E00, 0x0E7F, 'Thai'),
        (0x0400, 0x04FF, 'Cyrillic'),
        (0x0500, 0x052F, 'Cyrillic'),
    ]
    
    # Check each character in the text
    script_chars = {}
    total_non_latin = 0
    
    for char in text:
        code_point = ord(char)
        for start, end, script_name in non_latin_ranges:
            if start <= code_point <= end:
                total_non_latin += 1
                if script_name not in script_chars:
                    script_chars[script_name] = {'count': 0, 'examples': []}
                script_chars[script_name]['count'] += 1
                if len(script_chars[script_name]['examples']) < 10:
                    script_chars[script_name]['examples'].append(char)
                break
    
    # If ANY non-Latin characters found, report them
    if total_non_latin > 0:
        for script, data in script_chars.items():
            examples = ''.join(data['examples'][:5])
            count = data['count']
            # Create user-friendly issue description
            issues.append(f"{script}_text_found_{count}_chars_[{examples}]")
    
    return len(issues) > 0, issues

def generate_content_hash(text):
    """Generate hash from normalized content - less aggressive cleaning"""
    # First, create a raw hash of the entire content for exact matching
    raw_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
    
    # For content-based hash, do minimal cleaning
    cleaned = text.lower().strip()
    
    # Only remove obvious file artifacts, not all numbers
    cleaned = re.sub(r'response_\d+_.*?\.html', '', cleaned, flags=re.IGNORECASE)
    
    # Normalize whitespace
    cleaned = ' '.join(cleaned.split())
    
    # Create content hash from cleaned version
    content_hash = hashlib.md5(cleaned.encode('utf-8')).hexdigest()
    
    return content_hash, raw_hash


def extract_chapter_info(filename, text):
    """Extract chapter number and title from filename and content"""
    chapter_num = None
    chapter_title = ""
    
    # Try to extract from filename
    m = re.match(r"response_(\d+)_(.+?)\.html", filename)
    if m:
        chapter_num = int(m.group(1))
        chapter_title = m.group(2) if len(m.groups()) > 1 else ""
    
    return chapter_num, chapter_title

def scan_html_folder(folder_path, log=print, stop_flag=None):
    global _stop_flag
    _stop_flag = False
    content_hashes = {}  # Map content hash to list of files
    results = []
    chapter_contents = {}
    html_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".html")]
    html_files.sort()

    log(f"🔍 Found {len(html_files)} HTML files. Starting aggressive scan...")

    # First pass: collect all data
    for idx, filename in enumerate(html_files):
        if stop_flag and stop_flag():
            log("⛔ QA scan interrupted by user.")
            return
        
        # Progress indicator
        log(f"📄 [{idx+1}/{len(html_files)}] Scanning {filename}...")
        
        full_path = os.path.join(folder_path, filename)
        try:
            raw_text = extract_text_from_html(full_path)
        except Exception as e:
            log(f"⚠️ Failed to read {filename}: {e}")
            continue

        if len(raw_text.strip()) < 100:
            log(f"⚠️ Skipped {filename}: Too short")
            continue

        # Extract chapter info
        chapter_num, chapter_title = extract_chapter_info(filename, raw_text)
        
        # Generate content hash
        content_hash = generate_content_hash(raw_text)
        
        # Track content duplicates
        if content_hash not in content_hashes:
            content_hashes[content_hash] = []
        content_hashes[content_hash].append({
            'filename': filename,
            'idx': idx,
            'chapter_num': chapter_num,
            'raw_text': raw_text  # Store raw text for similarity check
        })
        
        # Store chapter data
        if chapter_num is not None:
            if chapter_num not in chapter_contents:
                chapter_contents[chapter_num] = []
            chapter_contents[chapter_num].append({
                'filename': filename,
                'text': raw_text,
                'idx': idx,
                'hash': content_hash
            })

        issues = []
        preview = raw_text[:500].replace('\n', ' ')
        if len(preview) > 500:
            preview = preview[:497] + '...'

        # Aggressive non-English detection
        has_non_english, lang_issues = detect_non_english_content(raw_text)
        if has_non_english:
            issues.extend(lang_issues)
            
        # Log issues found immediately
        if issues:
            log(f"   └─ Found: {', '.join(issues[:2])}" + (" ..." if len(issues) > 2 else ""))

        # Spacing/formatting issues
        if has_no_spacing_or_linebreaks(raw_text):
            issues.append("no_spacing_or_linebreaks")

        # Repetitive content
        if has_repeating_sentences(raw_text):
            issues.append("excessive_repetition")

        results.append({
            "file_index": idx,
            "filename": filename,
            "filepath": full_path,
            "issues": issues,
            "preview": preview,
            "preview_normalized": ' '.join(preview.lower().split())[:300],  # Normalized preview for comparison
            "score": len(issues),
            "chapter_num": chapter_num,
            "content_hash": content_hash,
            "raw_text": raw_text  # Keep for similarity checking
        })

    log("\n✅ Initial scan complete. Checking for duplicates...")

    # Create preview-based duplicate detection
    preview_map = {}
    for result in results:
        preview_key = result['preview_normalized']
        if preview_key not in preview_map:
            preview_map[preview_key] = []
        preview_map[preview_key].append(result)

    # Check for duplicates based on preview
    duplicate_count = 0
    for preview_key, matching_results in preview_map.items():
        if len(matching_results) > 1:
            # Sort by file index
            matching_results.sort(key=lambda x: x['file_index'])
            first = matching_results[0]
            
            for dup in matching_results[1:]:
                if not any('DUPLICATE:' in issue for issue in dup['issues']):
                    dup['issues'].insert(0, f"DUPLICATE: same_preview_as_{first['filename']}")
                    dup['score'] = len(dup['issues'])
                    duplicate_count += 1
            
            log(f"⚠️ DUPLICATE GROUP (by preview): {[r['filename'] for r in matching_results]}")

    # Second pass: find ALL duplicates (exact hash matches)
    for hash_value, files in content_hashes.items():
        if len(files) > 1:
            # Sort files by index to identify the first occurrence
            files_sorted = sorted(files, key=lambda x: x['idx'])
            first_file = files_sorted[0]['filename']
            
            # Mark all duplicates with clear labeling
            for file_info in files_sorted[1:]:
                for result in results:
                    if result['filename'] == file_info['filename']:
                        # Add DUPLICATE prefix for clarity
                        if not any('DUPLICATE:' in issue for issue in result['issues']):
                            result['issues'].insert(0, f"DUPLICATE: exact_copy_of_{first_file}")
                            result['score'] = len(result['issues'])
                            duplicate_count += 1
            
            # Log the duplicate group
            duplicate_files = [f['filename'] for f in files_sorted]
            log(f"⚠️ DUPLICATE GROUP (by hash): {duplicate_files}")

    log(f"   └─ Found {duplicate_count} exact duplicates")

    # Third pass: Check for high similarity (only between adjacent files or same chapter numbers)
    # This is much faster than checking all pairs
    log("\n🔍 Checking for similar content (optimized)...")
    
    similar_count = 0
    total_checks = 0
    
    # Only check files that might be related
    for i in range(len(results)):
        if stop_flag and stop_flag():
            log("⛔ Similarity check interrupted by user.")
            break
            
        # Progress for similarity check
        if i % 10 == 0:
            log(f"   Progress: {i}/{len(results)} files checked...")
        
        # Skip if already marked as duplicate
        if any('DUPLICATE:' in issue for issue in results[i]['issues']):
            continue
        
        # Only check against:
        # 1. The next few files (likely to be similar)
        # 2. Files with the same chapter number
        candidates = []
        
        # Check next 3 files
        for j in range(i + 1, min(i + 4, len(results))):
            candidates.append(j)
        
        # Check files with same chapter number
        if results[i]['chapter_num'] is not None:
            for j in range(len(results)):
                if j != i and results[j]['chapter_num'] == results[i]['chapter_num']:
                    if j not in candidates:
                        candidates.append(j)
        
        for j in candidates:
            if j <= i:
                continue
                
            # Skip if already marked as duplicate
            if any('DUPLICATE:' in issue for issue in results[j]['issues']):
                continue
            
            # Compare actual text content
            text1 = results[i]['raw_text']
            text2 = results[j]['raw_text']
            
            # Quick length check first
            len_ratio = len(text1) / max(1, len(text2))
            if len_ratio < 0.8 or len_ratio > 1.2:
                continue  # Too different in length
            
            total_checks += 1
            similarity = SequenceMatcher(None, text1, text2).ratio()
            
            if similarity > 0.95:  # 95% similar = essentially duplicate
                results[j]['issues'].insert(0, f"DUPLICATE: {int(similarity*100)}%_match_with_{results[i]['filename']}")
                results[j]['score'] = len(results[j]['issues'])
                similar_count += 1
                log(f"   └─ Found near-duplicate: {results[j]['filename']} ≈ {results[i]['filename']} ({int(similarity*100)}%)")
            elif similarity > 0.85:  # 85-95% = very similar
                results[j]['issues'].insert(0, f"SIMILAR: {int(similarity*100)}%_match_with_{results[i]['filename']}")
                results[j]['score'] = len(results[j]['issues'])
                similar_count += 1
    
    log(f"   └─ Checked {total_checks} file pairs, found {similar_count} similar files")
    
    # Clean up raw_text from results to save memory
    for result in results:
        result.pop('raw_text', None)

    # Log summary of issues found
    log(f"\n📊 Issues Summary:")
    issue_counts = {}
    for r in results:
        for issue in r['issues']:
            # Simplify issue type for counting
            if 'DUPLICATE:' in issue:
                issue_type = 'DUPLICATE'
            elif 'SIMILAR:' in issue:
                issue_type = 'SIMILAR'
            elif '_text_found_' in issue:
                issue_type = issue.split('_text_found_')[0]
            else:
                issue_type = issue.split('_')[0]
            issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
    
    for issue_type, count in sorted(issue_counts.items()):
        log(f"  - {issue_type}: {count} files")

    # Generate reports
    output_dir = os.path.basename(folder_path.rstrip('/\\')) + "_Scan Report"
    output_path = os.path.join(folder_path, output_dir)
    os.makedirs(output_path, exist_ok=True)

    # Save detailed results
    with open(os.path.join(output_path, "validation_results.json"), "w", encoding="utf-8") as jf:
        json.dump(results, jf, indent=2, ensure_ascii=False)

    # Create CSV report (without chapter column)
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

    # Generate HTML report
    html_report = """<html>
<head>
    <meta charset='utf-8'>
    <title>Translation QA Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        table { border-collapse: collapse; width: 100%; margin-top: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #4CAF50; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        .error { background-color: #ffcccc; }
        .warning { background-color: #fff3cd; }
        .preview { font-size: 0.9em; color: #666; max-width: 400px; }
        .issues { font-size: 0.9em; }
        .non-english { color: red; font-weight: bold; }
    </style>
</head>
<body>"""
    
    html_report += "<h1>Translation QA Report</h1>"
    html_report += f"<p><strong>Total Files Scanned:</strong> {len(results)}</p>"
    html_report += f"<p><strong>Files with Issues:</strong> {sum(1 for r in results if r['issues'])}</p>"
    html_report += f"<p><strong>Clean Files:</strong> {sum(1 for r in results if not r['issues'])}</p>"
    
    if issue_counts:
        html_report += "<h2>Issues Summary</h2><ul>"
        for issue_type, count in sorted(issue_counts.items()):
            style = ' class="non-english"' if 'korean' in issue_type.lower() or 'chinese' in issue_type.lower() or 'japanese' in issue_type.lower() else ''
            html_report += f"<li{style}><strong>{issue_type}</strong>: {count} files</li>"
        html_report += "</ul>"
    
    html_report += "<h2>Detailed Results</h2>"
    html_report += "<table><tr><th>Index</th><th>Filename</th><th>Issues</th><th>Preview</th></tr>"

    for row in results:
        link = f"<a href='../{row['filename']}' target='_blank'>{row['filename']}</a>"
        
        # Format issues with highlighting
        formatted_issues = []
        for issue in row["issues"]:
            if issue.startswith("DUPLICATE:"):
                formatted_issues.append(f'<span style="color: red; font-weight: bold;">{issue}</span>')
            elif issue.startswith("SIMILAR:"):
                formatted_issues.append(f'<span style="color: orange; font-weight: bold;">{issue}</span>')
            elif '_text_found_' in issue:
                # Make non-Latin text issues more readable
                formatted_issues.append(f'<span class="non-english">{issue}</span>')
            else:
                formatted_issues.append(issue)
        
        issues_str = "<br>".join(formatted_issues) if formatted_issues else "None"
        row_class = 'error' if row["score"] > 1 else 'warning' if row["score"] == 1 else ''
        
        # Escape preview text for HTML
        import html
        preview_escaped = html.escape(row['preview'][:300])
        
        html_report += f"""<tr class='{row_class}'>
            <td>{row['file_index']}</td>
            <td>{link}</td>
            <td class='issues'>{issues_str}</td>
            <td class='preview'>{preview_escaped}</td>
        </tr>"""

    html_report += "</table></body></html>"

    with open(os.path.join(output_path, "validation_results.html"), "w", encoding="utf-8") as html_file:
        html_file.write(html_report)

    # Update progress file
    prog_path = os.path.join(folder_path, "translation_progress.json")
    try:
        with open(prog_path, "r", encoding="utf-8") as pf:
            prog = json.load(pf)
            existing = prog.get("completed", [])
    except FileNotFoundError:
        existing = []

    # Build list of FILE INDICES that have issues
    faulty_indices = [row["file_index"] for row in results if row["issues"]]

    # Remove faulty indices from the completed list
    updated = [idx for idx in existing if idx not in faulty_indices]

    # Write back the pruned history
    with open(prog_path, "w", encoding="utf-8") as pf:
        json.dump({"completed": updated}, pf, indent=2)

    log(f"\n✅ Scan complete!")
    log(f"📁 Reports saved to: {output_path}")
    log(f"🔧 Removed {len(faulty_indices)} problematic files from progress tracking")

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
