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
import time
import html as html_lib

# Global flag to allow stopping the scan externally
_stop_flag = False

def stop_scan():
    """Set the stop flag to True"""
    global _stop_flag
    _stop_flag = True

# Constants
DASH_CHARS = {
    '-', '–', '—', '―', '⸺', '⸻', '﹘', '﹣', '－', '⁃', '‐', '‑', '‒',
    '_', '━', '─', '═', '╌', '╍', '┄', '┅', '┈', '┉', '⎯', '⏤', '＿',
    '＊', '*', '~', '～', '∼', '〜', 'ㅡ'  # Added Korean dash character
}

COMMON_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'after',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might',
    'chapter', 'each', 'person', 'persons'
}

# Korean dash patterns to EXCLUDE from detection
KOREAN_DASH_PATTERNS = [
    r'[ㅡ―—–\-]+',  # Korean dashes and similar
    r'[\u2014\u2015\u2500-\u257F]+',  # Box drawing characters often used in Korean text
    r'[\u3161\u3163\u3164]+',  # Korean filler characters
]

# Extended Korean separator characters to exclude from non-English detection
KOREAN_SEPARATOR_CHARS = {
    'ㅡ',  # Korean dash/separator (U+3161)
    '―',   # Horizontal bar (U+2015)
    '—',   # Em dash (U+2014)
    '–',   # En dash (U+2013)
    '［', '］',  # Full-width brackets
    '【', '】',  # Black lenticular brackets
    '〔', '〕',  # Tortoise shell brackets
    '《', '》',  # Double angle brackets
    '「', '」',  # Corner brackets
    '『', '』',  # White corner brackets
}

def extract_text_from_html(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, "html.parser")
        return soup.get_text(separator='\n', strip=True)

def is_dash_separator_line(line):
    """Check if a line consists only of dash-like punctuation characters"""
    stripped = line.strip()
    if not stripped:
        return False
    
    # Check if it's a Korean dash pattern (should NOT be flagged)
    for pattern in KOREAN_DASH_PATTERNS:
        if re.match(f'^{pattern}$', stripped):
            return False
    
    # Check if all non-space characters are in our dash set
    non_space_chars = [c for c in stripped if not c.isspace()]
    if not non_space_chars:
        return False
    
    # Check various dash patterns
    if all(c in DASH_CHARS for c in non_space_chars):
        return True
    
    # Check for repeated patterns
    if re.match(r'^[\s\-–—―_*~ㅡ]+$', stripped):
        return True
    
    # Check for patterns like "---", "***", "___", "~~~" (3 or more)
    if re.match(r'^(\-{3,}|_{3,}|\*{3,}|~{3,}|–{2,}|—{2,}|―{2,}|ㅡ{2,})$', stripped):
        return True
    
    # Check for spaced patterns like "- - -", "* * *"
    if re.match(r'^([\-–—―_*~ㅡ]\s*){3,}$', stripped):
        return True
    
    return False

def filter_dash_lines(text):
    """Filter out dash separator lines from text"""
    lines = text.split('\n')
    return '\n'.join(line for line in lines if not is_dash_separator_line(line))

def has_no_spacing_or_linebreaks(text, space_threshold=0.01):
    filtered_text = filter_dash_lines(text)
    space_ratio = filtered_text.count(" ") / max(1, len(filtered_text))
    newline_count = filtered_text.count("\n")
    return space_ratio < space_threshold or newline_count == 0

def has_repeating_sentences(text, min_repeats=10):
    filtered_text = filter_dash_lines(text)
    sentences = [s.strip() for s in re.split(r'[.!?]+', filtered_text) 
                 if s.strip() and len(s.strip()) > 20]
    
    if len(sentences) < min_repeats:
        return False
    
    counter = Counter(sentences)
    
    for sent, count in counter.items():
        if count >= min_repeats and len(sent) > 50:
            if not any(pattern in sent.lower() for pattern in ['said', 'asked', 'replied', 'thought']):
                return True
    return False

def is_korean_separator_pattern(text):
    """Check if text is a Korean separator pattern like [ㅡㅡㅡㅡㅡ]"""
    # Remove brackets and spaces
    cleaned = text.strip().strip('[]').strip()
    
    # Check if it's only Korean separator characters
    if not cleaned:
        return False
    
    # Check if all characters are Korean separators
    return all(c in KOREAN_SEPARATOR_CHARS or c.isspace() for c in cleaned)

def detect_non_english_content(text):
    """Detect ONLY non-Latin script characters (not romanized text), excluding Korean separators"""
    issues = []
    filtered_text = filter_dash_lines(text)
    
    # Define non-Latin script ranges
    non_latin_ranges = [
        (0xAC00, 0xD7AF, 'Korean'), (0x1100, 0x11FF, 'Korean'),
        (0x3130, 0x318F, 'Korean'), (0xA960, 0xA97F, 'Korean'),
        (0xD7B0, 0xD7FF, 'Korean'), (0x3040, 0x309F, 'Japanese'),
        (0x30A0, 0x30FF, 'Japanese'), (0x31F0, 0x31FF, 'Japanese'),
        (0xFF65, 0xFF9F, 'Japanese'), (0x4E00, 0x9FFF, 'Chinese'),
        (0x3400, 0x4DBF, 'Chinese'), (0x20000, 0x2A6DF, 'Chinese'),
        (0x2A700, 0x2B73F, 'Chinese'), (0x0590, 0x05FF, 'Hebrew'),
        (0x0600, 0x06FF, 'Arabic'), (0x0700, 0x074F, 'Syriac'),
        (0x0750, 0x077F, 'Arabic'), (0x0E00, 0x0E7F, 'Thai'),
        (0x0400, 0x04FF, 'Cyrillic'), (0x0500, 0x052F, 'Cyrillic'),
    ]
    
    script_chars = {}
    total_non_latin = 0
    
    # Split text into potential separator patterns and other content
    # This regex finds patterns like [ㅡㅡㅡ] or similar
    separator_pattern = r'\[[ㅡ\s―—–\-［］【】〔〕《》「」『』]+\]'
    parts = re.split(f'({separator_pattern})', filtered_text)
    
    for part in parts:
        # Skip if this part is a Korean separator pattern
        if is_korean_separator_pattern(part):
            continue
        
        # Check characters in this part
        for char in part:
            # Skip Korean separator characters
            if char in KOREAN_SEPARATOR_CHARS:
                continue
            
            # Skip whitespace and common punctuation
            if char.isspace() or char in '[](){}.,;:!?\'"-':
                continue
                
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
    
    if total_non_latin > 0:
        for script, data in script_chars.items():
            examples = ''.join(data['examples'][:5])
            count = data['count']
            issues.append(f"{script}_text_found_{count}_chars_[{examples}]")
    
    return len(issues) > 0, issues

def extract_content_fingerprint(text):
    """Extract key sentences that can identify duplicate content"""
    lines = [line.strip() for line in text.split('\n') 
             if len(line.strip()) > 50 and not is_dash_separator_line(line)]
    
    if len(lines) < 5:
        return ""
    
    # Take first, middle, and last substantial sentences
    fingerprint_lines = []
    if len(lines) >= 3:
        fingerprint_lines = [lines[0], lines[len(lines)//2], lines[-1]]
    else:
        fingerprint_lines = lines[:3]
    
    return ' '.join(fingerprint_lines).lower()

def normalize_text(text):
    """Normalize text for comparison"""
    normalized = text.lower().strip()
    
    # Remove chapter indicators
    patterns = [
        r'chapter\s*\d+\s*:?\s*', r'第\s*\d+\s*章', r'제\s*\d+\s*장',
        r'chapter\s+[ivxlcdm]+\s*:?\s*', r'\bch\.?\s*\d+\s*:?\s*',
        r'^\s*\d+\s*\.?\s*', r'response_\d+_.*?\.html',
        r'\d{4}-\d{2}-\d{2}', r'\d{2}:\d{2}:\d{2}', r'<[^>]+>'
    ]
    
    for pattern in patterns:
        normalized = re.sub(pattern, '', normalized, flags=re.IGNORECASE | re.MULTILINE)
    
    # Normalize whitespace and punctuation
    normalized = re.sub(r'\s+', ' ', normalized)
    normalized = re.sub(r'[^\w\s]', '', normalized)
    
    return normalized

def generate_content_hashes(text):
    """Generate multiple hashes for better duplicate detection"""
    normalized = normalize_text(text)
    
    # 1. Raw hash
    raw_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
    
    # 2. Normalized hash
    normalized_hash = hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    # 3. Content fingerprint
    fingerprint = extract_content_fingerprint(text)
    fingerprint_hash = hashlib.md5(fingerprint.encode('utf-8')).hexdigest() if fingerprint else None
    
    # 4. Word frequency hash
    words = re.findall(r'\w+', normalized.lower())
    word_freq = Counter(words)
    significant_words = [(w, c) for w, c in word_freq.most_common(100) 
                        if w not in COMMON_WORDS][:50]
    word_sig = ' '.join([f"{w}:{c}" for w, c in significant_words])
    word_hash = hashlib.md5(word_sig.encode('utf-8')).hexdigest() if word_sig else None
    
    # 5. First chunk hash
    first_chunk = normalized[:1000] if len(normalized) > 1000 else normalized
    first_chunk_hash = hashlib.md5(first_chunk.encode('utf-8')).hexdigest()
    
    return {
        'raw': raw_hash,
        'normalized': normalized_hash,
        'fingerprint': fingerprint_hash,
        'word_freq': word_hash,
        'first_chunk': first_chunk_hash
    }

def extract_chapter_info(filename, text):
    """Extract chapter number and title from filename and content"""
    chapter_num = None
    chapter_title = ""
    
    m = re.match(r"response_(\d+)_(.+?)\.html", filename)
    if m:
        chapter_num = int(m.group(1))
        chapter_title = m.group(2) if len(m.groups()) > 1 else ""
    
    return chapter_num, chapter_title

def calculate_similarity_ratio(text1, text2):
    """Calculate similarity with optimizations for large texts"""
    len_ratio = len(text1) / max(1, len(text2))
    if len_ratio < 0.7 or len_ratio > 1.3:
        return 0.0
    
    if len(text1) > 10000:
        sample_size = 3000
        samples1 = [
            text1[:sample_size],
            text1[len(text1)//2 - sample_size//2:len(text1)//2 + sample_size//2],
            text1[-sample_size:]
        ]
        samples2 = [
            text2[:sample_size],
            text2[len(text2)//2 - sample_size//2:len(text2)//2 + sample_size//2],
            text2[-sample_size:]
        ]
        similarities = [SequenceMatcher(None, s1, s2).ratio() for s1, s2 in zip(samples1, samples2)]
        return sum(similarities) / len(similarities)
    else:
        return SequenceMatcher(None, text1, text2).ratio()

def extract_chapter_title(text):
    """Extract chapter title from text"""
    patterns = [
        r'Chapter\s+\d+\s*:\s*([^\n\r]+)',
        r'Chapter\s+\d+\s+([^\n\r]+)',
        r'第\s*\d+\s*章\s*[:：]?\s*([^\n\r]+)',
        r'제\s*\d+\s*장\s*[:：]?\s*([^\n\r]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text[:500], re.IGNORECASE)
        if match:
            title = match.group(1).strip()
            title = re.sub(r'\s+', ' ', title)
            title = title.split('.')[0].split('The')[0].strip()
            return title[:100] if len(title) > 100 else title
    
    return None

def detect_duplicates(results, log, stop_flag, aggressive_mode):
    """Detect duplicates using multiple strategies"""
    content_hashes = {'raw': {}, 'normalized': {}, 'fingerprint': {}, 
                     'word_freq': {}, 'first_chunk': {}}
    duplicate_groups = {}
    near_duplicate_groups = {}
    next_group_id = 0
    near_duplicate_next_id = 1000
    
    # Build hash dictionaries
    for idx, result in enumerate(results):
        hashes = result['hashes']
        file_info = {
            'filename': result['filename'],
            'idx': idx,
            'chapter_num': result['chapter_num'],
            'raw_text': result['raw_text'],
            'hashes': hashes
        }
        
        for hash_type, hash_value in hashes.items():
            if hash_value:
                if hash_value not in content_hashes[hash_type]:
                    content_hashes[hash_type][hash_value] = []
                content_hashes[hash_type][hash_value].append(file_info)
    
    # Multiple levels of duplicate detection
    duplicate_detection_levels = [
        ("exact content", 'raw', lambda files: len(files) > 1),
        ("normalized content", 'normalized', lambda files: len(files) > 1),
        ("first 1000 characters", 'first_chunk', lambda files: len(files) > 1),
        ("content fingerprints", 'fingerprint', lambda files: len(files) > 1),
        ("word frequency patterns", 'word_freq', lambda files: len(files) > 1)
    ]
    
    for level_name, hash_type, condition in duplicate_detection_levels:
        log(f"🔍 Checking {level_name}...")
        for hash_value, files in content_hashes[hash_type].items():
            if hash_value and condition(files):
                process_duplicate_group(files, duplicate_groups, next_group_id, log)
                next_group_id = max(duplicate_groups.values(), default=-1) + 1
    
    # Deep similarity check
    similarity_threshold = 0.75 if aggressive_mode else 0.85
    perform_deep_similarity_check(results, duplicate_groups, similarity_threshold, 
                                 next_group_id, log, stop_flag)
    
    # Additional checks for specific patterns
    check_consecutive_chapters(results, duplicate_groups, aggressive_mode, log)
    check_specific_patterns(results, duplicate_groups, log)
    
    return duplicate_groups, near_duplicate_groups

def process_duplicate_group(files, duplicate_groups, next_group_id, log):
    """Process a group of duplicate files"""
    existing_group = None
    for file_info in files:
        if file_info['filename'] in duplicate_groups:
            existing_group = duplicate_groups[file_info['filename']]
            break
    
    group_id = existing_group if existing_group is not None else next_group_id
    
    for file_info in files:
        duplicate_groups[file_info['filename']] = group_id
    
    if existing_group is None:
        log(f"   └─ Found duplicate group: {[f['filename'] for f in files]}")

def perform_deep_similarity_check(results, duplicate_groups, threshold, next_group_id, log, stop_flag):
    """Perform deep similarity analysis between files"""
    log(f"🔍 Deep similarity analysis (threshold: {int(threshold*100)}%)...")
    
    for i in range(len(results)):
        if stop_flag and stop_flag():
            log("⛔ Similarity check interrupted by user.")
            break
        
        if i % 10 == 0:
            log(f"   Progress: {i}/{len(results)} files analyzed...")
        
        if results[i]['filename'] in duplicate_groups:
            continue
        
        for j in range(i + 1, len(results)):
            if results[j]['filename'] in duplicate_groups:
                continue
            
            similarity = calculate_similarity_ratio(results[i]['raw_text'], results[j]['raw_text'])
            
            if similarity > threshold:
                if results[i]['filename'] not in duplicate_groups and results[j]['filename'] not in duplicate_groups:
                    duplicate_groups[results[i]['filename']] = next_group_id
                    duplicate_groups[results[j]['filename']] = next_group_id
                    next_group_id += 1
                
                log(f"   └─ Found similarity match: {results[i]['filename']} ≈ {results[j]['filename']} ({int(similarity*100)}%)")

def check_consecutive_chapters(results, duplicate_groups, aggressive_mode, log):
    """Check for consecutive chapters with same title"""
    log("🔍 Checking consecutive same-titled chapters...")
    
    # Extract chapter titles
    for result in results:
        result['chapter_title'] = extract_chapter_title(result['raw_text'])
    
    # Sort by chapter number
    chapter_sorted = [r for r in results if r['chapter_num'] is not None and r['chapter_title']]
    chapter_sorted.sort(key=lambda x: x['chapter_num'])
    
    for i in range(len(chapter_sorted) - 1):
        current = chapter_sorted[i]
        
        for j in range(i + 1, min(i + 4, len(chapter_sorted))):
            next_chapter = chapter_sorted[j]
            
            if (current['chapter_title'] == next_chapter['chapter_title'] and
                abs(current['chapter_num'] - next_chapter['chapter_num']) <= 3 and
                current['filename'] not in duplicate_groups and 
                next_chapter['filename'] not in duplicate_groups):
                
                # Compare content
                text1 = re.sub(r'Chapter\s+\d+\s*:?\s*', '', current['raw_text'][:2000], flags=re.IGNORECASE)
                text2 = re.sub(r'Chapter\s+\d+\s*:?\s*', '', next_chapter['raw_text'][:2000], flags=re.IGNORECASE)
                
                similarity = calculate_similarity_ratio(text1, text2)
                similarity_threshold = 0.70 if aggressive_mode else 0.80
                
                if similarity > similarity_threshold:
                    group_id = max(duplicate_groups.values(), default=-1) + 1
                    duplicate_groups[current['filename']] = group_id
                    duplicate_groups[next_chapter['filename']] = group_id
                    log(f"   └─ Chapters {current['chapter_num']} & {next_chapter['chapter_num']} marked as duplicates (same title, {int(similarity*100)}% similar)")

def check_specific_patterns(results, duplicate_groups, log):
    """Check for specific known duplicate patterns"""
    log("🔍 Checking for known duplicate patterns...")
    
    # Check for specific content patterns
    chapel_pattern = r"under the pretense of offering a prayer.*?visited the chapel.*?hiding while holding.*?breath.*?watching the scene"
    
    for i in range(len(results)):
        if results[i]['filename'] in duplicate_groups:
            continue
        
        if re.search(chapel_pattern, results[i]['preview'], re.IGNORECASE | re.DOTALL):
            for j in range(i + 1, len(results)):
                if results[j]['filename'] in duplicate_groups:
                    continue
                
                if re.search(chapel_pattern, results[j]['preview'], re.IGNORECASE | re.DOTALL):
                    group_id = max(duplicate_groups.values(), default=-1) + 1
                    duplicate_groups[results[i]['filename']] = group_id
                    duplicate_groups[results[j]['filename']] = group_id
                    log(f"   └─ Pattern match found: {results[i]['filename']} ≈ {results[j]['filename']}")

def generate_reports(results, folder_path, log):
    """Generate output reports"""
    output_dir = os.path.basename(folder_path.rstrip('/\\')) + "_Scan Report"
    output_path = os.path.join(folder_path, output_dir)
    os.makedirs(output_path, exist_ok=True)
    
    # Save JSON report
    with open(os.path.join(output_path, "validation_results.json"), "w", encoding="utf-8") as jf:
        json.dump(results, jf, indent=2, ensure_ascii=False)
    
    # Save CSV report
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
    generate_html_report(results, output_path)
    
    log(f"\n✅ Scan complete!")
    log(f"📁 Reports saved to: {output_path}")

def generate_html_report(results, output_path):
    """Generate HTML report"""
    issue_counts = {}
    for r in results:
        for issue in r['issues']:
            issue_type = issue.split(':')[0] if ':' in issue else issue.split('_')[0]
            issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
    
    html = f"""<html>
<head>
    <meta charset='utf-8'>
    <title>Translation QA Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .error {{ background-color: #ffcccc; }}
        .warning {{ background-color: #fff3cd; }}
        .preview {{ font-size: 0.9em; color: #666; max-width: 400px; }}
        .issues {{ font-size: 0.9em; }}
        .non-english {{ color: red; font-weight: bold; }}
        .duplicate-group {{ background-color: #ffe6e6; }}
    </style>
</head>
<body>
    <h1>Translation QA Report</h1>
    <p><strong>Total Files Scanned:</strong> {len(results)}</p>
    <p><strong>Files with Issues:</strong> {sum(1 for r in results if r['issues'])}</p>
    <p><strong>Clean Files:</strong> {sum(1 for r in results if not r['issues'])}</p>
"""
    
    if issue_counts:
        html += "<h2>Issues Summary</h2><ul>"
        for issue_type, count in sorted(issue_counts.items()):
            style = ' class="non-english"' if any(x in issue_type.lower() for x in ['korean', 'chinese', 'japanese']) else ''
            html += f"<li{style}><strong>{issue_type}</strong>: {count} files</li>"
        html += "</ul>"
    
    html += "<h2>Detailed Results</h2>"
    html += "<table><tr><th>Index</th><th>Filename</th><th>Issues</th><th>Preview</th></tr>"
    
    for row in results:
        link = f"<a href='../{row['filename']}' target='_blank'>{row['filename']}</a>"
        
        formatted_issues = []
        for issue in row["issues"]:
            if issue.startswith("DUPLICATE:"):
                formatted_issues.append(f'<span style="color: red; font-weight: bold;">{issue}</span>')
            elif issue.startswith("NEAR_DUPLICATE:"):
                formatted_issues.append(f'<span style="color: darkorange; font-weight: bold;">{issue}</span>')
            elif '_text_found_' in issue:
                formatted_issues.append(f'<span class="non-english">{issue}</span>')
            else:
                formatted_issues.append(issue)
        
        issues_str = "<br>".join(formatted_issues) if formatted_issues else "None"
        
        row_class = 'duplicate-group' if any('DUPLICATE:' in issue for issue in row['issues']) else ''
        if not row_class and any('NEAR_DUPLICATE:' in issue for issue in row['issues']):
            row_class = 'warning'
        if not row_class:
            row_class = 'error' if row["score"] > 1 else 'warning' if row["score"] == 1 else ''
        
        preview_escaped = html_lib.escape(row['preview'][:300])
        
        html += f"""<tr class='{row_class}'>
            <td>{row['file_index']}</td>
            <td>{link}</td>
            <td class='issues'>{issues_str}</td>
            <td class='preview'>{preview_escaped}</td>
        </tr>"""
    
    html += "</table></body></html>"
    
    with open(os.path.join(output_path, "validation_results.html"), "w", encoding="utf-8") as html_file:
        html_file.write(html)

def update_progress_file(folder_path, results, log):
    """Update translation progress file"""
    prog_path = os.path.join(folder_path, "translation_progress.json")
    
    try:
        with open(prog_path, "r", encoding="utf-8") as pf:
            prog = json.load(pf)
    except FileNotFoundError:
        log("[INFO] No progress file found - nothing to update")
        return
    
    faulty_chapters = [row for row in results if row["issues"]]
    
    if not faulty_chapters:
        log("✅ No faulty chapters found - progress unchanged")
        return
    
    # Detect progress format version
    is_new_format = "chapters" in prog and isinstance(prog.get("chapters"), dict)
    
    if is_new_format:
        update_new_format_progress(prog, faulty_chapters, log)
    else:
        update_legacy_format_progress(prog, faulty_chapters, log)
    
    # Write back updated progress
    with open(prog_path, "w", encoding="utf-8") as pf:
        json.dump(prog, pf, indent=2, ensure_ascii=False)
    
    # Log affected chapters
    affected_chapters = []
    for faulty_row in faulty_chapters:
        chapter_num = faulty_row.get("file_index", 0) + 1
        if faulty_row.get("filename"):
            match = re.search(r'response_(\d+)', faulty_row["filename"])
            if match:
                chapter_num = int(match.group(1))
        affected_chapters.append(chapter_num)
    
    if affected_chapters:
        log(f"📝 Chapters marked for re-translation: {', '.join(str(c) for c in sorted(affected_chapters))}")

def update_new_format_progress(prog, faulty_chapters, log):
    """Update new format progress file"""
    log("[INFO] Detected new progress format")
    
    # Build reverse mapping
    output_file_to_chapter_key = {}
    for chapter_key, chapter_info in prog["chapters"].items():
        output_file = chapter_info.get("output_file")
        if output_file:
            output_file_to_chapter_key[output_file] = chapter_key
    
    updated_count = 0
    for faulty_row in faulty_chapters:
        faulty_filename = faulty_row["filename"]
        chapter_key = output_file_to_chapter_key.get(faulty_filename)
        
        if chapter_key and chapter_key in prog["chapters"]:
            chapter_info = prog["chapters"][chapter_key]
            old_status = chapter_info.get("status", "unknown")
            
            chapter_info["status"] = "qa_failed"
            chapter_info["qa_issues"] = True
            chapter_info["qa_timestamp"] = time.time()
            chapter_info["qa_issues_found"] = faulty_row.get("issues", [])
            
            updated_count += 1
            
            chapter_num = chapter_info.get('actual_num', faulty_row.get("file_index", 0) + 1)
            log(f"   └─ Marked chapter {chapter_num} as qa_failed (was: {old_status})")
            
            # Remove from content_hashes
            content_hash = chapter_info.get("content_hash")
            if content_hash and content_hash in prog.get("content_hashes", {}):
                del prog["content_hashes"][content_hash]
            
            # Remove chunk data
            if "chapter_chunks" in prog and chapter_key in prog["chapter_chunks"]:
                del prog["chapter_chunks"][chapter_key]
                log(f"   └─ Removed chunk data for chapter {chapter_num}")
    
    log(f"🔧 Updated {updated_count} chapters in new format")

def update_legacy_format_progress(prog, faulty_chapters, log):
    """Update legacy format progress file"""
    log("[INFO] Detected legacy progress format")
    
    existing = prog.get("completed", [])
    faulty_indices = [row["file_index"] for row in faulty_chapters]
    updated = [idx for idx in existing if idx not in faulty_indices]
    removed_count = len(existing) - len(updated)
    
    prog["completed"] = updated
    
    # Remove chunk data
    if "chapter_chunks" in prog:
        for faulty_idx in faulty_indices:
            chapter_key = str(faulty_idx)
            if chapter_key in prog["chapter_chunks"]:
                del prog["chapter_chunks"][chapter_key]
                log(f"   └─ Removed chunk data for chapter {faulty_idx + 1}")
    
    # Remove from content_hashes
    if "content_hashes" in prog:
        hashes_to_remove = []
        for hash_val, hash_info in prog["content_hashes"].items():
            if hash_info.get("completed_idx") in faulty_indices:
                hashes_to_remove.append(hash_val)
        
        for hash_val in hashes_to_remove:
            del prog["content_hashes"][hash_val]
            log(f"   └─ Removed content hash entry")
    
    log(f"🔧 Removed {removed_count} chapters from legacy completed list")

def scan_html_folder(folder_path, log=print, stop_flag=None, aggressive_mode=True):
    """Main scanning function"""
    global _stop_flag
    _stop_flag = False
    
    log(f"{'🚨 AGGRESSIVE' if aggressive_mode else '📋 Standard'} duplicate detection mode")
    
    html_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".html")])
    log(f"🔍 Found {len(html_files)} HTML files. Starting scan...")
    
    results = []
    
    # First pass: collect all data
    for idx, filename in enumerate(html_files):
        if stop_flag and stop_flag():
            log("⛔ QA scan interrupted by user.")
            return
        
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
        
        chapter_num, chapter_title = extract_chapter_info(filename, raw_text)
        hashes = generate_content_hashes(raw_text)
        
        preview = raw_text[:500].replace('\n', ' ')
        if len(preview) > 500:
            preview = preview[:497] + '...'
        
        # Normalize preview
        preview_normalized = normalize_text(preview)[:300]
        
        results.append({
            "file_index": idx,
            "filename": filename,
            "filepath": full_path,
            "issues": [],
            "preview": preview,
            "preview_normalized": preview_normalized,
            "score": 0,
            "chapter_num": chapter_num,
            "hashes": hashes,
            "raw_text": raw_text
        })
    
    log("\n✅ Initial scan complete. Performing duplicate detection...")
    
    # Detect duplicates
    duplicate_groups, near_duplicate_groups = detect_duplicates(results, log, stop_flag, aggressive_mode)
    
    # Process results and check for issues
    log("\n📊 Checking for other issues...")
    
    # Group files by duplicate group
    groups = {}
    for filename, group_id in duplicate_groups.items():
        if group_id not in groups:
            groups[group_id] = []
        groups[group_id].append(filename)
    
    # Check each file for all issues
    for result in results:
        issues = []
        
        # Check duplicates
        if result['filename'] in duplicate_groups:
            group_id = duplicate_groups[result['filename']]
            group_files = groups[group_id]
            if len(group_files) > 1:
                others = [f for f in group_files if f != result['filename']]
                if len(others) == 1:
                    issues.append(f"DUPLICATE: exact_or_near_copy_of_{others[0]}")
                else:
                    issues.append(f"DUPLICATE: part_of_{len(group_files)}_file_group")
        
        # Check near-duplicates
        elif result['filename'] in near_duplicate_groups:
            near_group_id = near_duplicate_groups[result['filename']]
            near_group_files = [f for f, gid in near_duplicate_groups.items() if gid == near_group_id]
            if len(near_group_files) > 1:
                others = [f for f in near_group_files if f != result['filename']]
                if len(others) == 1:
                    issues.append(f"NEAR_DUPLICATE: highly_similar_to_{others[0]}")
                else:
                    issues.append(f"NEAR_DUPLICATE: similar_to_{len(near_group_files)-1}_other_files")
        
        # Check other issues
        raw_text = result['raw_text']
        
        # Non-English content (excluding Korean separators)
        has_non_english, lang_issues = detect_non_english_content(raw_text)
        if has_non_english:
            issues.extend(lang_issues)
        
        # Spacing/formatting issues
        if has_no_spacing_or_linebreaks(raw_text):
            issues.append("no_spacing_or_linebreaks")
        
        # Repetitive content
        if has_repeating_sentences(raw_text):
            issues.append("excessive_repetition")
        
        result['issues'] = issues
        result['score'] = len(issues)
        
        if issues:
            log(f"   {result['filename']}: {', '.join(issues[:2])}" + (" ..." if len(issues) > 2 else ""))
    
    # Clean up raw_text to save memory
    for result in results:
        result.pop('raw_text', None)
        result.pop('hashes', None)
    
    # Generate reports
    generate_reports(results, folder_path, log)
    
    # Update progress file
    update_progress_file(folder_path, results, log)

def launch_gui():
    """Launch GUI interface"""
    def run_scan():
        folder_path = filedialog.askdirectory(title="Select Folder with HTML Files")
        if folder_path:
            threading.Thread(target=scan_html_folder, args=(folder_path, print, None, True), daemon=True).start()
    
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
        aggressive = True
        if len(sys.argv) > 2 and sys.argv[2] == "--standard":
            aggressive = False
        scan_html_folder(sys.argv[1], aggressive_mode=aggressive)
