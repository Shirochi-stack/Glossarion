# -*- coding: utf-8 -*-
# This is for automatic glossary generation only, unrelated to the more thorough glossary generation you get from clicking the "Extract Glossary" button

import os
import re
import os
import sys
import threading
import tempfile
import queue
import time
from bs4 import BeautifulSoup
import PatternManager as PM
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed



# Class-level shared lock for API submission timing
_api_submission_lock = threading.Lock()
_last_api_submission_time = 0
_results_lock = threading.Lock()
_file_write_lock = threading.Lock()

# Timing variables
_extraction_time = 0
_api_time = 0
_freq_check_time = 0
_dedup_time = 0
_io_time = 0


# Function to check if stop is requested (can be overridden)
def is_stop_requested():
    """Check if stop has been requested - default implementation"""
    return False

def set_output_redirect(log_callback=None):
    """Redirect print statements to a callback function for GUI integration"""
    if log_callback:
        import threading
        
        class CallbackWriter:
            def __init__(self, callback):
                self.callback = callback
                self.main_thread = threading.main_thread()
                
            def write(self, text):
                if text.strip():
                    # The callback (append_log) is already thread-safe - it handles QTimer internally
                    # So we can call it directly from any thread
                    self.callback(text.strip())
                    
            def flush(self):
                pass
                
        sys.stdout = CallbackWriter(log_callback)

def is_traditional_translation_api(model: str) -> bool:
    """Check if the model is a traditional translation API"""
    return model in ['deepl', 'google-translate', 'google-translate-free'] or model.startswith('deepl/') or model.startswith('google-translate/')

def send_with_interrupt(*args, **kwargs):
    """Lazy wrapper to avoid circular import"""
    from TransateKRtoEN import send_with_interrupt as _send_with_interrupt
    return _send_with_interrupt(*args, **kwargs)


# Class-level shared lock for API submission timing
_api_submission_lock = threading.Lock()
_last_api_submission_time = 0
_results_lock = threading.Lock()
_file_write_lock = threading.Lock()

# Timing variables
_extraction_time = 0
_api_time = 0
_freq_check_time = 0
_dedup_time = 0
_io_time = 0



def _atomic_write_file(filepath, content, encoding='utf-8'):
    """Atomically write to a file to prevent corruption from concurrent writes"""
    
    # Create temp file in same directory to ensure same filesystem
    dir_path = os.path.dirname(filepath)
    
    with _file_write_lock:
        try:
            # Write to temporary file first
            with tempfile.NamedTemporaryFile(mode='w', encoding=encoding, 
                                            dir=dir_path, delete=False) as tmp_file:
                tmp_file.write(content)
                tmp_path = tmp_file.name
            
            # Atomic rename (on same filesystem)
            if os.name == 'nt':  # Windows
                # Windows doesn't support atomic rename if target exists
                if os.path.exists(filepath):
                    os.remove(filepath)
                os.rename(tmp_path, filepath)
            else:  # Unix/Linux/Mac
                os.rename(tmp_path, filepath)
            
            return True
            
        except Exception as e:
            print(f"⚠️ Atomic write failed: {e}")
            # Cleanup temp file if it exists
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except:
                    pass
            
            # Fallback to direct write with lock
            try:
                with open(filepath, 'w', encoding=encoding) as f:
                    f.write(content)
                return True
            except Exception as e2:
                print(f"⚠️ Fallback write also failed: {e2}")
                return False

def save_glossary(output_dir, chapters, instructions, language="korean", log_callback=None):
    """Targeted glossary generator with true CSV format output and parallel processing"""
    # Note: Don't redirect stdout here if log_callback is provided by subprocess worker
    # The worker already captures stdout and sends to queue
    # Only redirect if we're NOT in a subprocess (i.e., log_callback is a real GUI callback)
    import sys
    in_subprocess = hasattr(sys.stdout, 'queue')  # Worker's LogCapture has a queue attribute
    
    if log_callback and not in_subprocess:
        set_output_redirect(log_callback)
    
    print("📱 Targeted Glossary Generator v6.0 (CSV Format + Parallel)")
    
    # CRITICAL: Reload ALL glossary settings from environment variables at the START
    # This ensures child processes spawned by ProcessPoolExecutor get the latest values
    # Force fresh read of all environment variables (they were set by save_config)
    print("🔄 Reloading glossary settings from environment variables...")
    
    # Check stop flag at start
    # Ensure output directory exists
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as _e:
        print(f"⚠️ Could not ensure output directory exists: {output_dir} ({_e})")
    if is_stop_requested():
        print("📁 ❌ Glossary generation stopped by user")
        return {}
    
    # Check if glossary already exists; if so, we'll MERGE it later (do not return early)
    glossary_path = os.path.join(output_dir, "glossary.csv")
    existing_glossary_content = None
    if os.path.exists(glossary_path):
        print(f"📁 Existing glossary detected (will merge): {glossary_path}")
        try:
            with open(glossary_path, 'r', encoding='utf-8') as f:
                existing_glossary_content = f.read()
        except Exception as e:
            print(f"⚠️ Could not read existing glossary: {e}")
    
    # Rest of the method continues as before...
    print("📁 Extracting names and terms with configurable options")
    
    # Check stop flag before processing
    if is_stop_requested():
        print("📁 ❌ Glossary generation stopped by user")
        return {}
    
    # Check for manual glossary first (CSV only)
    manual_glossary_path = os.getenv("MANUAL_GLOSSARY")
    existing_glossary = None
    if manual_glossary_path and os.path.exists(manual_glossary_path):
        print(f"📁 Manual glossary detected: {os.path.basename(manual_glossary_path)}")
        try:
            with open(manual_glossary_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # Treat as CSV text and stage it for merge; also copy to output for visibility
            target_path = os.path.join(output_dir, "glossary.csv")
            with open(target_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"📁 ✅ Manual CSV glossary copied to: {target_path}")
            existing_glossary = content
        except Exception as e:
            print(f"⚠️ Could not copy manual glossary: {e}")
            print(f"📁 Proceeding with automatic generation...")
    
    # Check for existing glossary from manual extraction
    glossary_folder_path = os.path.join(output_dir, "Glossary")
    # existing_glossary may already be set by MANUAL_GLOSSARY above
    
    if os.path.exists(glossary_folder_path):
        for file in os.listdir(glossary_folder_path):
            if file.endswith("_glossary.json"):
                existing_path = os.path.join(glossary_folder_path, file)
                try:
                    with open(existing_path, 'r', encoding='utf-8') as f:
                        existing_content = f.read()
                    existing_glossary = existing_content
                    print(f"📁 Found existing glossary from manual extraction: {file}")
                    break
                except Exception as e:
                    print(f"⚠️ Could not load existing glossary: {e}")
    
    # Get configuration from environment variables (FRESH READ)
    min_frequency = int(os.getenv("GLOSSARY_MIN_FREQUENCY", "2"))
    max_names = int(os.getenv("GLOSSARY_MAX_NAMES", "50"))
    max_titles = int(os.getenv("GLOSSARY_MAX_TITLES", "30"))
    batch_size = int(os.getenv("GLOSSARY_BATCH_SIZE", "50"))
    strip_honorifics = os.getenv("GLOSSARY_STRIP_HONORIFICS", "1") == "1"
    fuzzy_threshold = float(os.getenv("GLOSSARY_FUZZY_THRESHOLD", "0.90"))
    max_text_size = int(os.getenv("GLOSSARY_MAX_TEXT_SIZE", "50000"))
    
    # DEBUG: Show what we're reading from environment
    max_sentences_env = os.getenv("GLOSSARY_MAX_SENTENCES", "200")
    print(f"🔍 [DEBUG] Reading GLOSSARY_MAX_SENTENCES from environment: '{max_sentences_env}'")
    max_sentences = int(max_sentences_env)
    print(f"🔍 [DEBUG] Converted to integer: {max_sentences}")
    
    print(f"📑 Settings: Min frequency: {min_frequency}, Max names: {max_names}, Max titles: {max_titles}")
    print(f"📑 Strip honorifics: {'✅ Yes' if strip_honorifics else '❌ No'}")
    print(f"📑 Fuzzy matching threshold: {fuzzy_threshold}")
    print(f"📑 Max sentences for filtering: {max_sentences}")
    
    # Get custom prompt from environment
    custom_prompt = os.getenv("AUTO_GLOSSARY_PROMPT", "").strip()
    
    def clean_html(html_text):
        """Remove HTML tags to get clean text"""
        soup = BeautifulSoup(html_text, 'html.parser')
        return soup.get_text()
    
    # Check stop before processing chapters
    if is_stop_requested():
        print("📑 ❌ Glossary generation stopped by user")
        return {}
    
    # Get chapter split threshold and filter mode
    chapter_split_threshold = int(os.getenv("GLOSSARY_CHAPTER_SPLIT_THRESHOLD", "100000"))
    filter_mode = os.getenv("GLOSSARY_FILTER_MODE", "all")  # all, only_with_honorifics, only_without_honorifics
    
    # Check if parallel extraction is enabled for automatic glossary
    extraction_workers = int(os.getenv("EXTRACTION_WORKERS", "1"))
    batch_translation = os.getenv("BATCH_TRANSLATION", "0") == "1"
    api_batch_size = int(os.getenv("BATCH_SIZE", "5"))
    
    # Log the settings
    print(f"📑 Filter mode: {filter_mode}")
    if extraction_workers > 1:
        print(f"📑 Parallel extraction enabled: {extraction_workers} workers")
    if batch_translation:
        print(f"📑 Batch API calls enabled: {api_batch_size} chunks per batch")
    
    all_text = ' '.join(clean_html(chapter["body"]) for chapter in chapters)
    print(f"📑 Processing {len(all_text):,} characters of text")
    
    # Apply smart filtering FIRST to check actual size needed
    use_smart_filter = os.getenv("GLOSSARY_USE_SMART_FILTER", "1") == "1"
    effective_text_size = len(all_text)
    
    filtered_text_cache = None
    if use_smart_filter and custom_prompt:  # Only apply for AI extraction
        print(f"📁 Smart filtering enabled - checking effective text size after filtering...")
        # Perform filtering ONCE and reuse for chunking
        filtered_sample, _ = _filter_text_for_glossary(all_text, min_frequency, max_sentences)
        filtered_text_cache = filtered_sample
        effective_text_size = len(filtered_sample)
        # Calculate token count using tiktoken
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            token_count = len(enc.encode(filtered_sample))
            print(f"📁 Text reduction: {len(all_text):,} → {effective_text_size:,} chars ({100*(1-effective_text_size/len(all_text)):.1f}% reduction) | {token_count:,} tokens")
        except:
            print(f"📁 Text reduction: {len(all_text):,} → {effective_text_size:,} chars ({100*(1-effective_text_size/len(all_text)):.1f}% reduction)")
    
    # Safety check: Calculate actual token count for chunking decision
    estimated_tokens = None
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        estimated_tokens = len(enc.encode(filtered_text_cache if filtered_text_cache else all_text))
    except:
        # Fallback estimate: 1 token ≈ 3-4 characters for Asian languages
        estimated_tokens = effective_text_size // 3
    
    max_output_tokens = int(os.getenv("MAX_OUTPUT_TOKENS", "65536"))
    
    # Use compression factor to determine safe input limit (from CJK→English compression ratio)
    compression_factor = float(os.getenv("COMPRESSION_FACTOR", "1.0"))
    # Safe input limit is max_output divided by compression factor
    # (e.g., if compression is 0.7, output will be 70% of input, so we can use 1/0.7 = 1.43x for safety)
    safe_input_limit = int(max_output_tokens / max(compression_factor, 0.1)) if compression_factor > 0 else int(max_output_tokens * 0.8)
    
    if estimated_tokens > safe_input_limit:
        # Only show detailed token logs if using token-based chunking (threshold == 0)
        if chapter_split_threshold == 0:
            print(f"⚠️ Text too large for single API call!")
            print(f"   Estimated tokens: {estimated_tokens:,}")
            print(f"   Safe input limit: {safe_input_limit:,} (based on {compression_factor:.1f}x compression factor and {max_output_tokens:,} max output tokens)")
            print(f"   Will use ChapterSplitter for token-based chunking...")
        else:
            # Character-based threshold already set, just use it silently
            pass
    
    # Check if we need to split into chunks based on EFFECTIVE size after filtering
    needs_chunking = (chapter_split_threshold == 0 and estimated_tokens > safe_input_limit) or \
                     (chapter_split_threshold > 0 and effective_text_size > chapter_split_threshold)
    
    if needs_chunking:
        if chapter_split_threshold == 0:
            # Use ChapterSplitter for token-based intelligent chunking
            print(f"📑 Text exceeds safe token limit, using ChapterSplitter for token-based chunking...")
            from chapter_splitter import ChapterSplitter
            
            # Get the model name for the tokenizer
            model = os.getenv("MODEL", "gemini-2.0-flash")
            splitter = ChapterSplitter(model_name=model, target_tokens=safe_input_limit)
            
            # Get the text to split (filtered or raw)
            text_to_split = filtered_text_cache if (use_smart_filter and custom_prompt and filtered_text_cache) else all_text
            
            # Use ChapterSplitter to intelligently split based on tokens
            split_results = splitter.split_chapter(text_to_split, max_tokens=safe_input_limit)
            chunks_to_process = [(i, chunk) for i, (chunk, _, _) in enumerate(split_results, 1)]
            
            print(f"📑 ChapterSplitter created {len(chunks_to_process)} token-balanced chunks")
            all_glossary_entries = []
        else:
            # Use character-based splitting with fixed threshold
            print(f"📑 Effective text exceeds {chapter_split_threshold:,} chars, will process in chunks...")
            
            # If using smart filter, we need to split the FILTERED text, not raw text
            if use_smart_filter and custom_prompt:
                # Split the filtered text into chunks (reuse cached filtered text)
                filtered_text = filtered_text_cache if filtered_text_cache is not None else _filter_text_for_glossary(all_text, min_frequency, max_sentences)[0]
                chunks_to_process = []
                
                # Split filtered text into chunks of appropriate size
                chunk_size = chapter_split_threshold
                for i in range(0, len(filtered_text), chunk_size):
                    chunk_text = filtered_text[i:i + chunk_size]
                    chunks_to_process.append((len(chunks_to_process) + 1, chunk_text))
                
                print(f"📑 Split filtered text into {len(chunks_to_process)} chunks")
                all_glossary_entries = []
            else:
                # Original logic for unfiltered text
                all_glossary_entries = []
                chunk_size = 0
                chunk_chapters = []
                chunks_to_process = []
                
                for idx, chapter in enumerate(chapters):
                    if is_stop_requested():
                        print("📑 ❌ Glossary generation stopped by user")
                        return all_glossary_entries
                    
                    chapter_text = clean_html(chapter["body"])
                    chunk_size += len(chapter_text)
                    chunk_chapters.append(chapter)
                    
                    # Process chunk when it reaches threshold or last chapter
                    if chunk_size >= chapter_split_threshold or idx == len(chapters) - 1:
                        chunk_text = ' '.join(clean_html(ch["body"]) for ch in chunk_chapters)
                        chunks_to_process.append((len(chunks_to_process) + 1, chunk_text))
                        
                        # Reset for next chunk
                        chunk_size = 0
                        chunk_chapters = []
        
        print(f"📑 Split into {len(chunks_to_process)} chunks for processing")
        
        # Batch toggle decides concurrency: ON => parallel API calls; OFF => strict sequential
        if batch_translation and custom_prompt and len(chunks_to_process) > 1:
            print(f"📑 Processing chunks in batch mode with {api_batch_size} chunks per batch...")
            # Set fast mode for batch processing
            os.environ["GLOSSARY_SKIP_ALL_VALIDATION"] = "1"
            
            # Use batch API calls for AI extraction
            all_csv_lines = _process_chunks_batch_api(
                chunks_to_process, custom_prompt, language, 
                min_frequency, max_names, max_titles, 
                output_dir, strip_honorifics, fuzzy_threshold, 
                filter_mode, api_batch_size, extraction_workers, max_sentences
            )
            
            # Reset validation mode
            os.environ["GLOSSARY_SKIP_ALL_VALIDATION"] = "0"
            
            print(f"📑 All chunks completed. Aggregated raw lines: {len(all_csv_lines)}")
            
            # Process all collected entries at once (even if empty)
            # Add header so downstream steps can work uniformly
            include_gender_context = os.getenv("GLOSSARY_INCLUDE_GENDER_CONTEXT", "0") == "1"
            include_description = os.getenv("GLOSSARY_INCLUDE_DESCRIPTION", "0") == "1"
            if include_description:
                all_csv_lines.insert(0, "type,raw_name,translated_name,gender,description")
            elif include_gender_context:
                all_csv_lines.insert(0, "type,raw_name,translated_name,gender")
            else:
                all_csv_lines.insert(0, "type,raw_name,translated_name")
            
            # Merge with any on-disk glossary first (to avoid overwriting user edits)
            on_disk_path = os.path.join(output_dir, "glossary.csv")
            if os.path.exists(on_disk_path):
                try:
                    with open(on_disk_path, 'r', encoding='utf-8') as f:
                        on_disk_content = f.read()
                    all_csv_lines = _merge_csv_entries(all_csv_lines, on_disk_content, strip_honorifics, language)
                    print("📑 Merged with existing on-disk glossary")
                except Exception as e:
                    print(f"⚠️ Failed to merge with existing on-disk glossary: {e}")
            
            # Apply filter mode if needed
            if filter_mode == "only_with_honorifics":
                filtered = [all_csv_lines[0]]  # Keep header
                for line in all_csv_lines[1:]:
                    parts = line.split(',', 2)
                    if len(parts) >= 3 and parts[0] == "character":
                        filtered.append(line)
                all_csv_lines = filtered
                print(f"📑 Filter applied: {len(all_csv_lines)-1} character entries with honorifics kept")
            
            # Apply fuzzy deduplication (deferred until after all chunks)
            try:
                print(f"📑 Applying fuzzy deduplication (threshold: {fuzzy_threshold})...")
                all_csv_lines = _deduplicate_glossary_with_fuzzy(all_csv_lines, fuzzy_threshold)
            except Exception as e:
                print(f"⚠️ Deduplication error: {e} — continuing without dedup")
            
            # Sort by type and name
            print(f"📑 Sorting glossary by type and name...")
            header = all_csv_lines[0]
            entries = all_csv_lines[1:]
            if entries:
                entries.sort(key=lambda x: (0 if x.startswith('character,') else 1, x.split(',')[1].lower()))
            all_csv_lines = [header] + entries
            
            # Save
            # Check format preference
            use_legacy_format = os.getenv('GLOSSARY_USE_LEGACY_CSV', '0') == '1'

            if not use_legacy_format:
                # Convert to token-efficient format
                all_csv_lines = _convert_to_token_efficient_format(all_csv_lines)

            # Final sanitize to prevent stray headers
            all_csv_lines = _sanitize_final_glossary_lines(all_csv_lines, use_legacy_format)

            # Save
            csv_content = '\n'.join(all_csv_lines)
            glossary_path = os.path.join(output_dir, "glossary.csv")
            _atomic_write_file(glossary_path, csv_content)
            
            # Verify file exists; fallback direct write if needed
            if not os.path.exists(glossary_path):
                try:
                    with open(glossary_path, 'w', encoding='utf-8') as f:
                        f.write(csv_content)
                    print("📑 Fallback write succeeded for glossary.csv")
                except Exception as e:
                    print(f"❌ Failed to write glossary.csv: {e}")
            
            print(f"\n📑 ✅ GLOSSARY SAVED!")
            print(f"📑 ✅ AI GLOSSARY SAVED!")
            c_count, t_count, total = _count_glossary_entries(all_csv_lines, use_legacy_format)
            print(f"📑 Character entries: {c_count}")
            # print(f"📑 Term entries: {t_count}")
            print(f"📑 Total entries: {total}")
            
            return _parse_csv_to_dict(csv_content)
        else:
            # Strict sequential processing (one API call at a time)
            _prev_defer = os.getenv("GLOSSARY_DEFER_SAVE")
            _prev_filtered = os.getenv("_CHUNK_ALREADY_FILTERED")
            _prev_force_disable = os.getenv("GLOSSARY_FORCE_DISABLE_SMART_FILTER")
            os.environ["GLOSSARY_DEFER_SAVE"] = "1"
            # Tell the extractor each chunk is already filtered to avoid re-running smart filter per chunk
            os.environ["_CHUNK_ALREADY_FILTERED"] = "1"
            os.environ["GLOSSARY_FORCE_DISABLE_SMART_FILTER"] = "1"
            try:
                for chunk_idx, chunk_text in chunks_to_process:
                    if is_stop_requested():
                        break
                    
                    print(f"📑 Processing chunk {chunk_idx}/{len(chunks_to_process)} ({len(chunk_text):,} chars)...")
                    
                    if custom_prompt:
                        chunk_glossary = _extract_with_custom_prompt(
                            custom_prompt, chunk_text, language, 
                            min_frequency, max_names, max_titles, 
                            None, output_dir,  # Don't pass existing glossary to chunks
                            strip_honorifics, fuzzy_threshold, filter_mode, max_sentences, log_callback
                        )
                    else:
                        chunk_glossary = _extract_with_patterns(
                            chunk_text, language, min_frequency, 
                            max_names, max_titles, batch_size, 
                            None, output_dir,  # Don't pass existing glossary to chunks
                            strip_honorifics, fuzzy_threshold, filter_mode
                        )
                    
                    # Normalize to CSV lines and aggregate
                    chunk_lines = []
                    if isinstance(chunk_glossary, list):
                        for line in chunk_glossary:
                            if line and not line.startswith('type,'):
                                all_glossary_entries.append(line)
                                chunk_lines.append(line)
                    else:
                        for raw_name, translated_name in chunk_glossary.items():
                            entry_type = "character" if _has_honorific(raw_name) else "term"
                            line = f"{entry_type},{raw_name},{translated_name}"
                            all_glossary_entries.append(line)
                            chunk_lines.append(line)
                    
                    # Incremental update
                    try:
                        _incremental_update_glossary(output_dir, chunk_lines, strip_honorifics, language, filter_mode)
                        print(f"📑 Incremental write: +{len(chunk_lines)} entries")
                    except Exception as e2:
                        print(f"⚠️ Incremental write failed: {e2}")
            finally:
                if _prev_defer is None:
                    if "GLOSSARY_DEFER_SAVE" in os.environ:
                        del os.environ["GLOSSARY_DEFER_SAVE"]
                else:
                    os.environ["GLOSSARY_DEFER_SAVE"] = _prev_defer
                if _prev_filtered is None:
                    os.environ.pop("_CHUNK_ALREADY_FILTERED", None)
                else:
                    os.environ["_CHUNK_ALREADY_FILTERED"] = _prev_filtered
                if _prev_force_disable is None:
                    os.environ.pop("GLOSSARY_FORCE_DISABLE_SMART_FILTER", None)
                else:
                    os.environ["GLOSSARY_FORCE_DISABLE_SMART_FILTER"] = _prev_force_disable
        
        # Build CSV from aggregated entries
        print(f"📑 DEBUG: all_glossary_entries count before merge: {len(all_glossary_entries)}")
        
        # START WITH INCREMENTAL GLOSSARY AS BASE IF IT EXISTS AND IS LARGER
        # This ensures that if memory was lost (e.g. during a long sequential run), we rely on the disk backup
        incremental_path = os.path.join(output_dir, "glossary.incremental.csv")
        base_entries = list(all_glossary_entries)
        using_incremental_as_base = False
        
        if os.path.exists(incremental_path):
            try:
                with open(incremental_path, 'r', encoding='utf-8') as f:
                    inc_content = f.read()
                
                # Simple parse to count lines/entries
                inc_lines = [line for line in inc_content.split('\n') if line.strip() and not line.startswith('type,')]
                print(f"📑 Found incremental glossary: {len(inc_lines)} entries (Memory: {len(all_glossary_entries)} entries)")
                
                if len(inc_lines) > len(all_glossary_entries):
                    print("📑 🔄 Incremental glossary is larger than memory - using it as primary source")
                    # We need to ensure it has the header for csv_lines logic below
                    # But csv_lines construction adds header anyway.
                    # So we just REPLACE base_entries with inc_lines
                    base_entries = inc_lines
                    using_incremental_as_base = True
            except Exception as e:
                print(f"⚠️ Failed to check incremental glossary: {e}")
        
        include_gender_context = os.getenv("GLOSSARY_INCLUDE_GENDER_CONTEXT", "0") == "1"
        include_description = os.getenv("GLOSSARY_INCLUDE_DESCRIPTION", "0") == "1"
        
        if include_description:
            csv_lines = ["type,raw_name,translated_name,gender,description"] + base_entries
        elif include_gender_context:
            csv_lines = ["type,raw_name,translated_name,gender"] + base_entries
        else:
            csv_lines = ["type,raw_name,translated_name"] + base_entries
            
        # If we used incremental as base, we must merge MEMORY into it (to capture the last chunk if it wasn't in incremental yet)
        if using_incremental_as_base and all_glossary_entries:
             print("📑 Merging memory entries into incremental base...")
             # Create a mini-CSV for memory entries
             mem_csv = ["type,raw_name,translated_name"] + all_glossary_entries
             csv_lines = _merge_csv_entries(csv_lines, '\n'.join(mem_csv), strip_honorifics, language)

        # Merge with any provided existing glossary AND on-disk glossary to avoid overwriting
        on_disk_path = os.path.join(output_dir, "glossary.csv")
        
        merge_sources = []
        if existing_glossary:
            merge_sources.append(existing_glossary)
            
        # We already handled incremental above as the base, so we don't add it to merge_sources here
        
        if os.path.exists(on_disk_path):
            try:
                with open(on_disk_path, 'r', encoding='utf-8') as f:
                    merge_sources.append(f.read())
                print("📑 Found existing on-disk glossary to merge")
            except Exception as e:
                print(f"⚠️ Failed to read on-disk glossary for merging: {e}")
        # Also merge the main on-disk glossary if it was present at start
        if existing_glossary_content:
            csv_lines = _merge_csv_entries(csv_lines, existing_glossary_content, strip_honorifics, language)
        for src in merge_sources:
            before_merge_count = len(csv_lines)
            csv_lines = _merge_csv_entries(csv_lines, src, strip_honorifics, language)
            print(f"📑 DEBUG: Merged source. Count: {before_merge_count} -> {len(csv_lines)}")
        
        # Apply filter mode to final results
        csv_lines = _filter_csv_by_mode(csv_lines, filter_mode)
        
        # Apply fuzzy deduplication (deferred until after all chunks)
        print(f"📑 Applying fuzzy deduplication (threshold: {fuzzy_threshold})...")
        original_count = len(csv_lines) - 1
        csv_lines = _deduplicate_glossary_with_fuzzy(csv_lines, fuzzy_threshold)
        deduped_count = len(csv_lines) - 1
        if original_count > deduped_count:
            print(f"📑 Removed {original_count - deduped_count} duplicate entries")
        
        # Sort by type and name
        print(f"📑 Sorting glossary by type and name...")
        header = csv_lines[0]
        entries = csv_lines[1:]
        entries.sort(key=lambda x: (0 if x.startswith('character,') else 1, x.split(',')[1].lower() if ',' in x else x.lower()))
        csv_lines = [header] + entries
        
        # Token-efficient format if enabled
        use_legacy_format = os.getenv('GLOSSARY_USE_LEGACY_CSV', '0') == '1'
        if not use_legacy_format:
            csv_lines = _convert_to_token_efficient_format(csv_lines)
        
        # Final sanitize to prevent stray headers and section titles at end
        csv_lines = _sanitize_final_glossary_lines(csv_lines, use_legacy_format)
        
        try:
            # Save
            csv_content = '\n'.join(csv_lines)
            glossary_path = os.path.join(output_dir, "glossary.csv")
            _atomic_write_file(glossary_path, csv_content)
            
            # Verify file exists; fallback direct write if needed
            if not os.path.exists(glossary_path):
                try:
                    with open(glossary_path, 'w', encoding='utf-8') as f:
                        f.write(csv_content)
                    print("📑 Fallback write succeeded for glossary.csv")
                except Exception as e:
                    print(f"❌ Failed to write glossary.csv: {e}")
        finally:
            print(f"\n📑 ✅ CHUNKED GLOSSARY SAVED!")
            print(f"📑 ✅ AI GLOSSARY SAVED!")
            print(f"📑 File: {glossary_path}")
            c_count, t_count, total = _count_glossary_entries(csv_lines, use_legacy_format)
            print(f"📑 Character entries: {c_count}")
            # print(f"📑 Term entries: {t_count}")
            print(f"📑 Total entries: {total}")
        
        return _parse_csv_to_dict(csv_content)
    
    # Original single-text processing
    if custom_prompt:
        # Pass cached filtered text if available to avoid re-filtering
        text_to_process = filtered_text_cache if filtered_text_cache is not None else all_text
        already_filtered = filtered_text_cache is not None
        
        # Set environment flag to indicate text is already filtered
        if already_filtered:
            os.environ["_TEXT_ALREADY_FILTERED"] = "1"
        
        try:
            return _extract_with_custom_prompt(custom_prompt, text_to_process, language, 
                                                   min_frequency, max_names, max_titles, 
                                                   existing_glossary, output_dir, 
                                                   strip_honorifics, fuzzy_threshold, filter_mode, max_sentences, log_callback)
        finally:
            if already_filtered:
                os.environ.pop("_TEXT_ALREADY_FILTERED", None)
    else:
        return _extract_with_patterns(all_text, language, min_frequency, 
                                         max_names, max_titles, batch_size, 
                                         existing_glossary, output_dir, 
                                         strip_honorifics, fuzzy_threshold, filter_mode)

    total_time = time.time() - total_start_time
    print(f"\n📑 ========== GLOSSARY GENERATION COMPLETE ==========")
    print(f"📑 Total time: {total_time:.1f}s")
    print(f"📑 Performance breakdown:")
    print(f"📑   - Extraction: {0:.1f}s")
    print(f"📑   - API calls: {0:.1f}s")
    print(f"📑   - Frequency checking: {0:.1f}s")
    print(f"📑   - Deduplication: {0:.1f}s")
    print(f"📑   - File I/O: {0:.1f}s")
    print(f"📑 ================================================")
    
    return result  # This is the existing return statement

def _convert_to_token_efficient_format(csv_lines):
    """Convert CSV lines to token-efficient format with sections and asterisks"""
    if len(csv_lines) <= 1:
        return csv_lines
    
    header = csv_lines[0]
    entries = csv_lines[1:]
    
    # Group by type (only from valid CSV lines)
    import re as _re
    grouped = {}
    for line in entries:
        if not line.strip():
            continue
        # Only accept proper CSV rows: at least 3 fields and a sane type token
        parts_full = [p.strip() for p in line.split(',')]
        if len(parts_full) < 3:
            continue
        entry_type = parts_full[0].lower()
        if not _re.match(r'^[a-z_]+$', entry_type):
            continue
        if entry_type not in grouped:
            grouped[entry_type] = []
        grouped[entry_type].append(line)
    
    # Rebuild with token-efficient format
    result = []
    # Extract column headers from CSV to show in dynamic header
    columns = ['translated_name', 'raw_name']
    # Check for gender and description columns
    header_parts = [p.strip() for p in header.split(',')] if header else []
    if 'gender' in header_parts:
        columns.append('gender')
    if 'description' in header_parts:
        columns.append('description')
    # Add any other custom fields (exclude type, raw_name, translated_name, gender, description)
    standard_cols = {'type', 'raw_name', 'translated_name', 'gender', 'description'}
    for col in header_parts:
        if col.lower() not in standard_cols and col:
            columns.append(col)
    result.append(f"Glossary Columns: {', '.join(columns)}\n")
    
    # Process in order: character first, then term, then others
    type_order = ['character', 'term'] + [t for t in grouped.keys() if t not in ['character', 'term']]
    
    for entry_type in type_order:
        if entry_type not in grouped:
            continue
            
        entries = grouped[entry_type]
        
        # Add section header
        section_name = entry_type.upper() + 'S' if not entry_type.upper().endswith('S') else entry_type.upper()
        result.append(f"=== {section_name} ===")
        
        # Add entries in new format
        for line in entries:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 3:
                raw_name = parts[1]
                translated_name = parts[2]
                
                # Format: * TranslatedName (RawName)
                entry_line = f"* {translated_name} ({raw_name})"
                
                # Add gender if present and not Unknown - ONLY for character entries
                if entry_type == 'character' and len(parts) > 3 and parts[3] and parts[3] != 'Unknown':
                    # Validate gender field - reject if malformed
                    gender_val = parts[3].strip()
                    if not gender_val.startswith(('[', '(')):
                        entry_line += f" [{gender_val}]"
                
                # Add any additional fields as description
                if len(parts) > 4:
                    description = ', '.join(parts[4:])
                    if description.strip():
                        entry_line += f": {description}"
                
                result.append(entry_line)
        
        result.append("")  # Blank line between sections
    
    return result

def _count_glossary_entries(lines, use_legacy_format=False):
    """Return (char_count, term_count, total_count) for either format."""
    if not lines:
        return 0, 0, 0
    if use_legacy_format:
        data = lines[1:] if lines and lines[0].lower().startswith('type,raw_name') else lines
        char_count = sum(1 for ln in data if ln.startswith('character,'))
        term_count = sum(1 for ln in data if ln.startswith('term,'))
        total = sum(1 for ln in data if ln and ',' in ln)
        return char_count, term_count, total
    # token-efficient
    current = None
    char_count = term_count = total = 0
    for ln in lines:
        s = ln.strip()
        if s.startswith('=== ') and 'CHARACTER' in s.upper():
            current = 'character'
            continue
        if s.startswith('=== ') and 'TERM' in s.upper():
            current = 'term'
            continue
        if s.startswith('* '):
            total += 1
            if current == 'character':
                char_count += 1
            elif current == 'term':
                term_count += 1
    return char_count, term_count, total

def _sanitize_final_glossary_lines(lines, use_legacy_format=False):
    """Remove stray CSV headers and normalize header placement before saving.
    - In legacy CSV mode, ensure exactly one header at the very top.
    - In token-efficient mode, remove any CSV header lines entirely.
    """
    header_norm = "type,raw_name,translated_name"
    if not lines:
        return lines
    
    if use_legacy_format:
        sanitized = []
        header_seen = False
        for ln in lines:
            txt = ln.strip()
            if txt.lower().startswith("type,raw_name"):
                if not header_seen:
                    sanitized.append(header_norm)
                    header_seen = True
                # skip duplicates
            else:
                sanitized.append(ln)
        # ensure header at top
        if sanitized and not sanitized[0].strip().lower().startswith("type,raw_name"):
            sanitized.insert(0, header_norm)
        return sanitized
    else:
        # remove any CSV header lines anywhere and duplicate top headers/sections
        cleaned = []
        glossary_header_seen = False
        for i, ln in enumerate(lines):
            txt = ln.strip()
            low = txt.lower()
            # Drop CSV headers
            if low.startswith("type,raw_name"):
                continue
            # Keep only the first main glossary header
            if low.startswith("glossary:"):
                if glossary_header_seen:
                    continue
                glossary_header_seen = True
                cleaned.append(ln)
                continue
            # Remove bogus section like '=== GLOSSARY: ... ==='
            if low.startswith("=== glossary:"):
                continue
            cleaned.append(ln)
        return cleaned

def _process_chunks_batch_api(chunks_to_process, custom_prompt, language, 
                              min_frequency, max_names, max_titles, 
                              output_dir, strip_honorifics, fuzzy_threshold, 
                              filter_mode, api_batch_size, extraction_workers, max_sentences=200):
    """Process chunks using batch API calls for AI extraction with thread delay"""
    
    print(f"📑 Using batch API mode with {api_batch_size} chunks per batch")
    
    # Ensure we defer saving and heavy merging when processing chunks
    _prev_defer = os.getenv("GLOSSARY_DEFER_SAVE")
    os.environ["GLOSSARY_DEFER_SAVE"] = "1"
    
    # Get thread submission delay
    thread_delay = float(os.getenv("THREAD_SUBMISSION_DELAY_SECONDS", "0.5"))
    if thread_delay > 0:
        print(f"📑 Thread submission delay: {thread_delay}s between parallel calls")
    
    # CHANGE: Collect raw CSV lines instead of dictionary
    all_csv_lines = []  # Collect all entries as CSV lines
    total_chunks = len(chunks_to_process)
    completed_chunks = 0
    
    # Ensure per-chunk smart filtering is disabled globally during batch processing
    _prev_filtered = os.getenv("_CHUNK_ALREADY_FILTERED")
    _prev_force_disable = os.getenv("GLOSSARY_FORCE_DISABLE_SMART_FILTER")
    os.environ["_CHUNK_ALREADY_FILTERED"] = "1"
    os.environ["GLOSSARY_FORCE_DISABLE_SMART_FILTER"] = "1"

    # Process all chunks in parallel (no batching)
    # Use extraction_workers setting (already passed as parameter)
    max_workers = min(extraction_workers, len(chunks_to_process))
    print(f"📑 Processing all {len(chunks_to_process)} chunks with {max_workers} parallel workers...")
    
    # Use ThreadPoolExecutor for all chunks at once
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        last_submission_time = 0
        
        for chunk_idx, chunk_text in chunks_to_process:
            if is_stop_requested():
                break
            
            # Apply thread submission delay
            if thread_delay > 0 and last_submission_time > 0:
                time_since_last = time.time() - last_submission_time
                if time_since_last < thread_delay:
                    sleep_time = thread_delay - time_since_last
                    print(f"🧵 Thread delay: {sleep_time:.1f}s for chunk {chunk_idx}")
                    time.sleep(sleep_time)
            
            future = executor.submit(
                _extract_with_custom_prompt,
                custom_prompt, chunk_text, language,
                min_frequency, max_names, max_titles,
                None, output_dir, strip_honorifics,
                fuzzy_threshold, filter_mode, max_sentences
            )
            futures[future] = chunk_idx
            last_submission_time = time.time()
        
        # Collect results
        for future in as_completed(futures):
            if is_stop_requested():
                break
            
            try:
                chunk_glossary = future.result()
                print(f"📑 DEBUG: Chunk {futures[future]} returned type={type(chunk_glossary)}, len={len(chunk_glossary)}")

                # Normalize to CSV lines (without header)
                chunk_lines = []
                if isinstance(chunk_glossary, dict):
                    for raw_name, translated_name in chunk_glossary.items():
                        entry_type = "character" if _has_honorific(raw_name) else "term"
                        chunk_lines.append(f"{entry_type},{raw_name},{translated_name}")
                elif isinstance(chunk_glossary, list):
                    for line in chunk_glossary:
                        if line and not line.startswith('type,'):
                            chunk_lines.append(line)
                
                # Aggregate for end-of-run
                all_csv_lines.extend(chunk_lines)
                
                # DISABLED: Don't do incremental writes in batch mode
                # This prevents chunks from merging with each other's results
                # All merging will happen at the end in save_glossary()
                # try:
                #     _incremental_update_glossary(output_dir, chunk_lines, strip_honorifics, language, filter_mode)
                #     print(f"📑 Incremental write: +{len(chunk_lines)} entries")
                # except Exception as e2:
                #     print(f"⚠️ Incremental write failed: {e2}")
                
                completed_chunks += 1
                
                # Print progress for GUI
                progress_percent = (completed_chunks / total_chunks) * 100
                print(f"📑 Progress: {completed_chunks}/{total_chunks} chunks ({progress_percent:.0f}%)")
                print(f"📑 Chunk {futures[future]} completed and aggregated")
                
            except Exception as e:
                print(f"⚠️ API call for chunk {futures[future]} failed: {e}")
                completed_chunks += 1
                progress_percent = (completed_chunks / total_chunks) * 100
                print(f"📑 Progress: {completed_chunks}/{total_chunks} chunks ({progress_percent:.0f}%)")
    
    # CHANGE: Return CSV lines instead of dictionary
    
    # Restore per-chunk filter disabling envs
    if _prev_filtered is None:
        os.environ.pop("_CHUNK_ALREADY_FILTERED", None)
    else:
        os.environ["_CHUNK_ALREADY_FILTERED"] = _prev_filtered
    if _prev_force_disable is None:
        os.environ.pop("GLOSSARY_FORCE_DISABLE_SMART_FILTER", None)
    else:
        os.environ["GLOSSARY_FORCE_DISABLE_SMART_FILTER"] = _prev_force_disable

    # Restore previous defer setting
    if _prev_defer is None:
        # Default back to not deferring if it wasn't set
        if "GLOSSARY_DEFER_SAVE" in os.environ:
            del os.environ["GLOSSARY_DEFER_SAVE"]
    else:
        os.environ["GLOSSARY_DEFER_SAVE"] = _prev_defer
    
    return all_csv_lines

def _incremental_update_glossary(output_dir, chunk_lines, strip_honorifics, language, filter_mode):
    """Incrementally update glossary.csv (token-efficient) using an on-disk CSV aggregator.
    This keeps glossary.csv present and growing after each chunk while preserving
    token-efficient format for the visible file.
    """
    if not chunk_lines:
        return
    # Paths
    agg_path = os.path.join(output_dir, "glossary.incremental.csv")
    vis_path = os.path.join(output_dir, "glossary.csv")
    # Ensure output dir
    os.makedirs(output_dir, exist_ok=True)
    # Compose CSV with header for merging
    include_gender_context = os.getenv("GLOSSARY_INCLUDE_GENDER_CONTEXT", "0") == "1"
    include_description = os.getenv("GLOSSARY_INCLUDE_DESCRIPTION", "0") == "1"
    if include_description:
        new_csv_lines = ["type,raw_name,translated_name,gender,description"] + chunk_lines
    elif include_gender_context:
        new_csv_lines = ["type,raw_name,translated_name,gender"] + chunk_lines
    else:
        new_csv_lines = ["type,raw_name,translated_name"] + chunk_lines
    # Load existing aggregator content, if any
    existing_csv = None
    if os.path.exists(agg_path):
        try:
            with open(agg_path, 'r', encoding='utf-8') as f:
                existing_csv = f.read()
        except Exception as e:
            print(f"⚠️ Incremental: cannot read aggregator: {e}")
    # Merge (exact merge, no fuzzy to keep this fast)
    merged_csv_lines = _merge_csv_entries(new_csv_lines, existing_csv or "", strip_honorifics, language)
    # Optional filter mode
    merged_csv_lines = _filter_csv_by_mode(merged_csv_lines, filter_mode)
    # Save aggregator (CSV)
    _atomic_write_file(agg_path, "\n".join(merged_csv_lines))
    # Convert to token-efficient format for visible glossary.csv
    token_lines = _convert_to_token_efficient_format(merged_csv_lines)
    token_lines = _sanitize_final_glossary_lines(token_lines, use_legacy_format=False)
    _atomic_write_file(vis_path, "\n".join(token_lines))
    if not os.path.exists(vis_path):
        with open(vis_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(token_lines))

def _process_single_chunk(chunk_idx, chunk_text, custom_prompt, language,
                         min_frequency, max_names, max_titles, batch_size,
                         output_dir, strip_honorifics, fuzzy_threshold, filter_mode,
                         already_filtered=False, max_sentences=200):
    """Process a single chunk - wrapper for parallel execution"""
    print(f"📑 Worker processing chunk {chunk_idx} ({len(chunk_text):,} chars)...")
    
    if custom_prompt:
        # Pass flag to indicate if text is already filtered
        os.environ["_CHUNK_ALREADY_FILTERED"] = "1" if already_filtered else "0"
        _prev_defer = os.getenv("GLOSSARY_DEFER_SAVE")
        os.environ["GLOSSARY_DEFER_SAVE"] = "1"
        try:
            result = _extract_with_custom_prompt(
                custom_prompt, chunk_text, language, 
                min_frequency, max_names, max_titles, 
                None, output_dir,
                strip_honorifics, fuzzy_threshold, filter_mode, max_sentences, log_callback=None
            )
        finally:
            os.environ["_CHUNK_ALREADY_FILTERED"] = "0"  # Reset
            if _prev_defer is None:
                if "GLOSSARY_DEFER_SAVE" in os.environ:
                    del os.environ["GLOSSARY_DEFER_SAVE"]
            else:
                os.environ["GLOSSARY_DEFER_SAVE"] = _prev_defer
        return result
    else:
        return _extract_with_patterns(
            chunk_text, language, min_frequency, 
            max_names, max_titles, batch_size, 
            None, output_dir,
            strip_honorifics, fuzzy_threshold, filter_mode
        )

def _apply_final_filter(entries, filter_mode):
    """Apply final filtering based on mode to ensure only requested types are included"""
    if filter_mode == "only_with_honorifics":
        # Filter to keep only entries that look like they have honorifics
        filtered = {}
        for key, value in entries.items():
            # Check if the key contains known honorific patterns
            if _has_honorific(key):
                filtered[key] = value
        print(f"📑 Final filter: Kept {len(filtered)} entries with honorifics (from {len(entries)} total)")
        return filtered
    elif filter_mode == "only_without_honorifics":
        # Filter to keep only entries without honorifics
        filtered = {}
        for key, value in entries.items():
            if not _has_honorific(key):
                filtered[key] = value
        print(f"📑 Final filter: Kept {len(filtered)} entries without honorifics (from {len(entries)} total)")
        return filtered
    else:
        return entries

def _looks_like_name(text):
    """Check if text looks like a character name"""
    if not text:
        return False
    
    # Check for various name patterns
    # Korean names (2-4 hangul characters)
    if all(0xAC00 <= ord(char) <= 0xD7AF for char in text) and 2 <= len(text) <= 4:
        return True
    
    # Japanese names (mix of kanji/kana, 2-6 chars)
    has_kanji = any(0x4E00 <= ord(char) <= 0x9FFF for char in text)
    has_kana = any((0x3040 <= ord(char) <= 0x309F) or (0x30A0 <= ord(char) <= 0x30FF) for char in text)
    if (has_kanji or has_kana) and 2 <= len(text) <= 6:
        return True
    
    # Chinese names (EXPANDED: 2-6 Chinese characters for cultivation novels)
    if all(0x4E00 <= ord(char) <= 0x9FFF for char in text) and 2 <= len(text) <= 6:
        # Check if it starts with a known surname (1 or 2 chars)
        if len(text) >= 2:
            # Check single-char surname
            if text[0] in PM.CHINESE_SINGLE_SURNAMES:
                return True
            # Check two-char compound surname
            if len(text) >= 3 and text[:2] in PM.CHINESE_COMPOUND_SURNAMES:
                return True
        # Even without surname match, if it's 2-6 chars it could be a valid term
        return True
    
    # English names (starts with capital, mostly letters)
    if text[0].isupper() and sum(1 for c in text if c.isalpha()) >= len(text) * 0.8:
        return True
    
    return False

def _has_honorific(term):
    """Check if a term contains an honorific using PatternManager's comprehensive list"""
    if not term:
        return False
    
    term_lower = term.lower()
    
    # Check all language honorifics from PatternManager
    for language, honorifics_list in PM.CJK_HONORIFICS.items():
        for honorific in honorifics_list:
            # For romanized/English honorifics with spaces or dashes
            if honorific.startswith(' ') or honorific.startswith('-'):
                if term_lower.endswith(honorific.lower()):
                    return True
            # For CJK honorifics (no separator)
            else:
                if honorific in term:
                    return True
    
    return False

def _strip_all_honorifics(term, language='korean'):
    """Strip all honorifics from a term using PatternManager's lists"""
    if not term:
        return term
    
    result = term
    
    # Get honorifics for the specific language and English romanizations
    honorifics_to_strip = []
    if language in PM.CJK_HONORIFICS:
        honorifics_to_strip.extend(PM.CJK_HONORIFICS[language])
    honorifics_to_strip.extend(PM.CJK_HONORIFICS.get('english', []))
    
    # Sort by length (longest first) to avoid partial matches
    honorifics_to_strip.sort(key=len, reverse=True)
    
    # Strip honorifics
    for honorific in honorifics_to_strip:
        if honorific.startswith(' ') or honorific.startswith('-'):
            # For romanized honorifics with separators
            if result.lower().endswith(honorific.lower()):
                result = result[:-len(honorific)]
        else:
            # For CJK honorifics (no separator)
            if result.endswith(honorific):
                result = result[:-len(honorific)]
    
    return result.strip()

def _convert_to_csv_format(data):
    """Convert various glossary formats to CSV string format with enforced 3 columns"""
    csv_lines = ["type,raw_name,translated_name"]
    
    if isinstance(data, str):
        # Already CSV string
        if data.strip().startswith('type,raw_name'):
            return data
        # Try to parse as JSON
        try:
            data = json.loads(data)
        except:
            return data
    
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                if 'type' in item and 'raw_name' in item:
                    # Already in correct format
                    line = f"{item['type']},{item['raw_name']},{item.get('translated_name', item['raw_name'])}"
                    csv_lines.append(line)
                else:
                    # Old format - default to 'term' type
                    entry_type = 'term'
                    raw_name = item.get('original_name', '')
                    translated_name = item.get('name', raw_name)
                    if raw_name and translated_name:
                        csv_lines.append(f"{entry_type},{raw_name},{translated_name}")
                        
    elif isinstance(data, dict):
        if 'entries' in data:
            # Has metadata wrapper, extract entries
            for original, translated in data['entries'].items():
                csv_lines.append(f"term,{original},{translated}")
        else:
            # Plain dictionary - default to 'term' type
            for original, translated in data.items():
                csv_lines.append(f"term,{original},{translated}")
    
    return '\n'.join(csv_lines)

def _parse_csv_to_dict(csv_content):
    """Parse CSV content to dictionary for backward compatibility"""
    result = {}
    lines = csv_content.strip().split('\n')
    
    for line in lines[1:]:  # Skip header
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 3:
            result[parts[1]] = parts[2]  # raw_name -> translated_name
    
    return result

def _fuzzy_match(term1, term2, threshold=0.90):
    """Check if two terms match using fuzzy matching"""
    ratio = SequenceMatcher(None, term1.lower(), term2.lower()).ratio()
    return ratio >= threshold

def _fuzzy_match_rapidfuzz(term_lower, text_lower, threshold, term_len):
    """Use rapidfuzz library for MUCH faster fuzzy matching"""
    from rapidfuzz import fuzz
    
    print(f"📑     Using RapidFuzz (C++ speed)...")
    start_time = time.time()
    
    matches_count = 0
    threshold_percent = threshold * 100  # rapidfuzz uses 0-100 scale
    
    # Can use smaller step because rapidfuzz is so fast
    step = 1  # Check every position - rapidfuzz can handle it
    
    # Process text
    for i in range(0, len(text_lower) - term_len + 1, step):
        # Check stop flag every 10000 positions
        if i > 0 and i % 10000 == 0:
            if is_stop_requested():
                print(f"📑     RapidFuzz stopped at position {i}")
                return matches_count
        
        window = text_lower[i:i + term_len]
        
        # rapidfuzz is fast enough we can check every position
        if fuzz.ratio(term_lower, window) >= threshold_percent:
            matches_count += 1
    
    elapsed = time.time() - start_time
    print(f"📑     RapidFuzz found {matches_count} matches in {elapsed:.2f}s")
    return matches_count

def _batch_compute_frequencies(terms, all_text, fuzzy_threshold=0.90, min_frequency=2):
    """Compute frequencies for all terms at once - MUCH faster than individual checking"""
    print(f"📑 Computing frequencies for {len(terms)} terms in batch mode...")
    start_time = time.time()
    
    # Result dictionary
    term_frequencies = {}
    
    # First pass: exact matching (very fast)
    print(f"📑   Phase 1: Exact matching...")
    text_lower = all_text.lower()
    for term in terms:
        if is_stop_requested():
            return term_frequencies
        term_lower = term.lower()
        count = text_lower.count(term_lower)
        term_frequencies[term] = count
    
    exact_time = time.time() - start_time
    high_freq_terms = sum(1 for count in term_frequencies.values() if count >= min_frequency)
    print(f"📑   Exact matching complete: {high_freq_terms}/{len(terms)} terms meet threshold ({exact_time:.1f}s)")
    
    # If fuzzy matching is disabled, we're done
    if fuzzy_threshold >= 1.0:
        return term_frequencies
    
    # Second pass: fuzzy matching ONLY for low-frequency terms
    low_freq_terms = [term for term, count in term_frequencies.items() if count < min_frequency]
    
    if low_freq_terms:
        print(f"📑   Phase 2: Fuzzy matching for {len(low_freq_terms)} low-frequency terms...")
        
        # Try to use RapidFuzz batch processing
        try:
            from rapidfuzz import process, fuzz
            
            # For very large texts, sample it for fuzzy matching
            if len(text_lower) > 500000:
                print(f"📑   Text too large ({len(text_lower):,} chars), sampling for fuzzy matching...")
                # Sample every Nth character to reduce size
                sample_rate = max(1, len(text_lower) // 100000)
                sampled_text = text_lower[::sample_rate]
            else:
                sampled_text = text_lower
            
            # Create chunks of text for fuzzy matching
            chunk_size = 1000  # Process text in chunks
            text_chunks = [sampled_text[i:i+chunk_size] for i in range(0, len(sampled_text), chunk_size//2)]  # Overlapping chunks
            
            print(f"📑   Processing {len(text_chunks)} text chunks...")
            threshold_percent = fuzzy_threshold * 100
            
            # Process in batches to avoid memory issues
            batch_size = 100  # Process 100 terms at a time
            for batch_start in range(0, len(low_freq_terms), batch_size):
                if is_stop_requested():
                    break
                
                batch_end = min(batch_start + batch_size, len(low_freq_terms))
                batch_terms = low_freq_terms[batch_start:batch_end]
                
                for term in batch_terms:
                    if is_stop_requested():
                        break
                    
                    # Quick fuzzy search in chunks
                    fuzzy_count = 0
                    for chunk in text_chunks[:50]:  # Limit to first 50 chunks for speed
                        if fuzz.partial_ratio(term.lower(), chunk) >= threshold_percent:
                            fuzzy_count += 1
                    
                    if fuzzy_count > 0:
                        # Scale up based on sampling
                        if len(text_lower) > 500000:
                            fuzzy_count *= (len(text_lower) // len(sampled_text))
                        term_frequencies[term] += fuzzy_count
                
                if (batch_end % 500 == 0) or (batch_end == len(low_freq_terms)):
                    elapsed = time.time() - start_time
                    print(f"📑   Processed {batch_end}/{len(low_freq_terms)} terms ({elapsed:.1f}s)")
            
        except ImportError:
            print("📑   RapidFuzz not available, skipping fuzzy matching")
    
    total_time = time.time() - start_time
    final_high_freq = sum(1 for count in term_frequencies.values() if count >= min_frequency)
    print(f"📑 Batch frequency computation complete: {final_high_freq}/{len(terms)} terms accepted ({total_time:.1f}s)")
    
    return term_frequencies

def _find_fuzzy_matches(term, text, threshold=0.90):
    """Find fuzzy matches of a term in text using efficient method with parallel processing"""
    start_time = time.time()
    
    term_lower = term.lower()
    text_lower = text.lower()
    term_len = len(term)
    
    # Only log for debugging if explicitly enabled
    debug_search = os.getenv("GLOSSARY_DEBUG_SEARCH", "0") == "1"
    if debug_search and len(text) > 100000:
        print(f"📑     Searching for '{term}' in {len(text):,} chars (threshold: {threshold})")
    
    # Strategy 1: Use exact matching first for efficiency
    exact_start = time.time()
    matches_count = text_lower.count(term_lower)
    exact_time = time.time() - exact_start
    
    if matches_count > 0:
        if debug_search and len(text) > 100000:
            print(f"📑     Found {matches_count} exact matches in {exact_time:.3f}s")
        return matches_count
    
    # Strategy 2: Try rapidfuzz if available (much faster)
    if matches_count == 0 and threshold < 1.0:
        try:
            from rapidfuzz import fuzz
            return _fuzzy_match_rapidfuzz(term_lower, text_lower, threshold, term_len)
        except ImportError:
            pass  # Fall back to parallel/sequential
        
        # Strategy 3: Fall back to parallel/sequential if rapidfuzz not available
        # Check if parallel processing is enabled
        extraction_workers = int(os.getenv("EXTRACTION_WORKERS", "1"))
        
        if extraction_workers > 1 and len(text) > 50000:  # Use parallel for large texts
            return _parallel_fuzzy_search(term_lower, text_lower, threshold, term_len, extraction_workers)
        else:
            return _sequential_fuzzy_search(term_lower, text_lower, threshold, term_len)
        # Check if parallel processing is enabled
        extraction_workers = int(os.getenv("EXTRACTION_WORKERS", "1"))
        
        if extraction_workers > 1 and len(text) > 50000:  # Use parallel for large texts
            return _parallel_fuzzy_search(term_lower, text_lower, threshold, term_len, extraction_workers)
        else:
            return _sequential_fuzzy_search(term_lower, text_lower, threshold, term_len)
    
    return matches_count

def _parallel_fuzzy_search(term_lower, text_lower, threshold, term_len, num_workers):
    """Parallel fuzzy search using ThreadPoolExecutor"""
    print(f"📑     Starting parallel fuzzy search with {num_workers} workers...")
    
    text_len = len(text_lower)
    matches_count = 0
    
    # Split text into overlapping chunks for parallel processing
    chunk_size = max(text_len // num_workers, term_len * 100)
    chunks = []
    
    for i in range(0, text_len, chunk_size):
        # Add overlap to avoid missing matches at boundaries
        end = min(i + chunk_size + term_len - 1, text_len)
        chunks.append((i, text_lower[i:end]))
    
    print(f"📑     Split into {len(chunks)} chunks of ~{chunk_size:,} chars each")
    
    # Process chunks in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        
        for chunk_idx, (start_pos, chunk_text) in enumerate(chunks):
            if is_stop_requested():
                return matches_count
            
            future = executor.submit(
                _fuzzy_search_chunk,
                term_lower, chunk_text, threshold, term_len, chunk_idx, len(chunks)
            )
            futures.append(future)
        
        # Collect results
        for future in as_completed(futures):
            if is_stop_requested():
                executor.shutdown(wait=False)
                return matches_count
            
            try:
                chunk_matches = future.result()
                matches_count += chunk_matches
            except Exception as e:
                print(f"📑     ⚠️ Chunk processing error: {e}")
    
    print(f"📑     Parallel fuzzy search found {matches_count} matches")
    return matches_count

def _fuzzy_search_chunk(term_lower, chunk_text, threshold, term_len, chunk_idx, total_chunks):
    """Process a single chunk for fuzzy matches"""
    chunk_matches = 0
    
    # Use a more efficient step size - no need to check every position
    step = max(1, term_len // 3)  # Check every third of term length
    
    for i in range(0, len(chunk_text) - term_len + 1, step):
        # Check stop flag periodically
        if i > 0 and i % 1000 == 0:
            if is_stop_requested():
                return chunk_matches
        
        window = chunk_text[i:i + term_len]
        
        # Use SequenceMatcher for fuzzy matching
        if SequenceMatcher(None, term_lower, window).ratio() >= threshold:
            chunk_matches += 1
    
    # Log progress for this chunk
    if total_chunks > 1:
        print(f"📑     Chunk {chunk_idx + 1}/{total_chunks} completed: {chunk_matches} matches")
    
    return chunk_matches

def _sequential_fuzzy_search(term_lower, text_lower, threshold, term_len):
    """Sequential fuzzy search (fallback for small texts or single worker)"""
    print(f"📑     Starting sequential fuzzy search...")
    fuzzy_start = time.time()
    
    matches_count = 0
    
    # More efficient step size
    step = max(1, term_len // 3)
    total_windows = (len(text_lower) - term_len + 1) // step
    
    print(f"📑     Checking ~{total_windows:,} windows with step size {step}")
    
    windows_checked = 0
    for i in range(0, len(text_lower) - term_len + 1, step):
        # Check stop flag frequently
        if i > 0 and i % (step * 100) == 0:
            if is_stop_requested():
                return matches_count
            
            # Progress log for very long operations
            if windows_checked % 1000 == 0 and windows_checked > 0:
                elapsed = time.time() - fuzzy_start
                rate = windows_checked / elapsed if elapsed > 0 else 0
                eta = (total_windows - windows_checked) / rate if rate > 0 else 0
                print(f"📑     Progress: {windows_checked}/{total_windows} windows, {rate:.0f} w/s, ETA: {eta:.1f}s")
        
        window = text_lower[i:i + term_len]
        if SequenceMatcher(None, term_lower, window).ratio() >= threshold:
            matches_count += 1
        
        windows_checked += 1
    
    fuzzy_time = time.time() - fuzzy_start
    print(f"📑     Sequential fuzzy search completed in {fuzzy_time:.2f}s, found {matches_count} matches")
    
    return matches_count

def _fuzzy_match(term1, term2, threshold=0.90):
    """Check if two terms match using fuzzy matching (unchanged)"""
    ratio = SequenceMatcher(None, term1.lower(), term2.lower()).ratio()
    return ratio >= threshold

def _strip_honorific(term, language_hint='unknown'):
    """Strip honorific from a term if present"""
    if not term:
        return term
        
    # Get honorifics for the detected language
    honorifics_to_check = []
    if language_hint in PM.CJK_HONORIFICS:
        honorifics_to_check.extend(PM.CJK_HONORIFICS[language_hint])
    honorifics_to_check.extend(PM.CJK_HONORIFICS.get('english', []))
    
    # Check and remove honorifics
    for honorific in honorifics_to_check:
        if honorific.startswith('-') or honorific.startswith(' '):
            # English-style suffix
            if term.endswith(honorific):
                return term[:-len(honorific)].strip()
        else:
            # CJK-style suffix (no separator)
            if term.endswith(honorific):
                return term[:-len(honorific)]
    
    return term

def _filter_text_for_glossary(text, min_frequency=2, max_sentences=None):
    """Filter text to extract only meaningful content for glossary extraction
    
    Args:
        text: Input text to filter
        min_frequency: Minimum frequency threshold for terms
        max_sentences: Maximum number of sentences to return (reads from env if None)
    """
    import re
    from collections import Counter
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time
    
    filter_start_time = time.time()
    print(f"📑 Starting smart text filtering...")
    print(f"📑 Input text size: {len(text):,} characters")
    
    # Clean HTML if present
    print(f"📑 Step 1/7: Cleaning HTML tags...")
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(text, 'html.parser')
    clean_text = soup.get_text()
    print(f"📑 Clean text size: {len(clean_text):,} characters")
    
    # Detect primary language for better filtering
    print(f"📑 Step 2/7: Detecting primary language...")
    def detect_primary_language(text_sample):
        sample = text_sample[:1000]
        korean_chars = sum(1 for char in sample if 0xAC00 <= ord(char) <= 0xD7AF)
        japanese_kana = sum(1 for char in sample if (0x3040 <= ord(char) <= 0x309F) or (0x30A0 <= ord(char) <= 0x30FF))
        chinese_chars = sum(1 for char in sample if 0x4E00 <= ord(char) <= 0x9FFF)
        
        if korean_chars > 50:
            return 'korean'
        elif japanese_kana > 20:
            return 'japanese'
        elif chinese_chars > 50 and japanese_kana < 10:
            return 'chinese'
        else:
            return 'english'
    
    primary_lang = detect_primary_language(clean_text)
    print(f"📑 Detected primary language: {primary_lang}")
    
    # Split into sentences for better context
    print(f"📁 Step 3/7: Splitting text into sentences...")
    # Use language-specific sentence splitting for better accuracy
    if primary_lang == 'chinese':
        # Split on major punctuation, but keep 、 and ， within sentences
        # This preserves more context for Chinese cultivation/wuxia terms
        sentences = re.split(r'[。！？；：]+', clean_text)
    else:
        sentences = re.split(r'[.!?。！？]+', clean_text)
    print(f"📁 Found {len(sentences):,} sentences")
    
    # Extract potential terms (words/phrases that appear multiple times)
    print(f"📑 Step 4/7: Setting up extraction patterns and exclusion rules...")
    word_freq = Counter()
    
    # Pattern for detecting potential names/terms based on capitalization or special characters
    # Korean names: 2-4 hangul characters WITHOUT honorifics
    korean_pattern = r'[가-힣]{2,4}'
    # Japanese names: kanji/hiragana/katakana combinations
    japanese_pattern = r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]{2,6}'
    # Chinese names: EXPANDED to 2-8 characters for cultivation/wuxia novels
    # This captures longer compound names, titles, and cultivation terms
    chinese_pattern = r'[\u4e00-\u9fff]{2,8}'
    # English proper nouns: Capitalized words
    english_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
    
    # Combine patterns
    combined_pattern = f'({korean_pattern}|{japanese_pattern}|{chinese_pattern}|{english_pattern})'
    print(f"📑 Using combined regex pattern for {primary_lang} text")
    
    # Get honorifics and title patterns for the detected language
    honorifics_to_exclude = set()
    if primary_lang in PM.CJK_HONORIFICS:
        honorifics_to_exclude.update(PM.CJK_HONORIFICS[primary_lang])
    # Also add English romanizations
    honorifics_to_exclude.update(PM.CJK_HONORIFICS.get('english', []))
    
    # Compile title patterns for the language
    title_patterns = []
    if primary_lang in PM.TITLE_PATTERNS:
        for pattern in PM.TITLE_PATTERNS[primary_lang]:
            title_patterns.append(re.compile(pattern))
    
    # Function to check if a term should be excluded
    def should_exclude_term(term):
        term_lower = term.lower()
        
        # Check if it's a common word
        if term in PM.COMMON_WORDS or term_lower in PM.COMMON_WORDS:
            return True
        
        # Check if it contains honorifics
        for honorific in honorifics_to_exclude:
            if honorific in term or (honorific.startswith('-') and term.endswith(honorific[1:])):
                return True
        
        # Check if it matches title patterns
        for pattern in title_patterns:
            if pattern.search(term):
                return True
        
        # Check if it's a number (including Chinese numbers)
        if term in PM.CHINESE_NUMS:
            return True
        
        # Check if it's just digits
        if term.isdigit():
            return True
        
        # For Chinese text, INCLUDE domain-specific terms (don't exclude them)
        if primary_lang == 'chinese' and len(term) >= 2:
            # Check if it's a cultivation term - these should NOT be excluded
            for category in PM.CHINESE_CULTIVATION_TERMS.values():
                if term in category:
                    return False  # Keep cultivation terms!
            
            # Check if it's a wuxia term - these should NOT be excluded
            for category in PM.CHINESE_WUXIA_TERMS.values():
                if term in category:
                    return False  # Keep wuxia terms!
            
            # Check relationship terms (important character relationships)
            for category in PM.CHINESE_RELATIONSHIP_TERMS.values():
                if term in category:
                    return False  # Keep relationship terms!
            
            # Check mythological terms (creatures, artifacts, legendary beings)
            for category in PM.CHINESE_MYTHOLOGICAL_TERMS.values():
                if term in category:
                    return False  # Keep mythological terms!
            
            # Check elemental/natural force terms
            for category in PM.CHINESE_ELEMENTAL_TERMS.values():
                if term in category:
                    return False  # Keep elemental terms!
            
            # Check physique/spiritual root terms
            for category in PM.CHINESE_PHYSIQUE_TERMS.values():
                if term in category:
                    return False  # Keep physique terms!
            
            # Check treasure grades
            for category in PM.CHINESE_TREASURE_GRADES.values():
                if term in category:
                    return False  # Keep treasure grade terms!
            
            # Check power system terms (levels, stars, etc.)
            for category in PM.CHINESE_POWER_SYSTEMS.values():
                if term in category:
                    return False  # Keep power system terms!
            
            # Check location types
            for category in PM.CHINESE_LOCATION_TYPES.values():
                if term in category:
                    return False  # Keep location terms!
            
            # Check battle terms
            for category in PM.CHINESE_BATTLE_TERMS.values():
                if term in category:
                    return False  # Keep battle terms!
        
        return False
    
    # Extract potential terms from each sentence
    print(f"📑 Step 5/7: Extracting and filtering terms from sentences...")
    
    # Check if we should use parallel processing
    extraction_workers = int(os.getenv("EXTRACTION_WORKERS", "1"))
    # Auto-detect optimal workers if not set
    if extraction_workers == 1 and len(sentences) > 1000:
        # Use more cores for better parallelization
        cpu_count = os.cpu_count() or 4
        extraction_workers = min(cpu_count, 12)  # Use up to 12 cores
        print(f"📑 Auto-detected {cpu_count} CPU cores, using {extraction_workers} workers")
    
    use_parallel = extraction_workers > 1 and len(sentences) > 100
    
    if use_parallel:
        print(f"📑 Using parallel processing with {extraction_workers} workers")
        print(f"📑 Estimated speedup: {extraction_workers}x faster")
    
    important_sentences = []
    seen_contexts = set()
    processed_count = 0
    total_sentences = len(sentences)
    last_progress_time = time.time()
    
    def process_sentence_batch(batch_sentences, batch_idx):
        """Process a batch of sentences"""
        local_word_freq = Counter()
        local_important = []
        local_seen = set()
        
        for sentence in batch_sentences:
            sentence = sentence.strip()
            if len(sentence) < 10 or len(sentence) > 500:
                continue
                
            # Find all potential terms in this sentence
            matches = re.findall(combined_pattern, sentence)
            
            if matches:
                # Filter out excluded terms
                filtered_matches = []
                for match in matches:
                    if not should_exclude_term(match):
                        local_word_freq[match] += 1
                        filtered_matches.append(match)
                
                # Keep sentences with valid potential terms
                if filtered_matches:
                    sentence_key = ' '.join(sorted(filtered_matches))
                    if sentence_key not in local_seen:
                        local_important.append(sentence)
                        local_seen.add(sentence_key)
        
        return local_word_freq, local_important, local_seen, batch_idx
    
    if use_parallel:
        # Force SMALL batches for real parallelization
        # We want MANY small batches, not few large ones!
        
        # Calculate based on total sentences
        total_sentences = len(sentences)
        
        # CRITICAL: Batch size must balance two factors:
        # 1. Small batches = more parallelism but higher overhead
        # 2. Large batches = less overhead but limits parallelism
        # 
        # For Windows ProcessPoolExecutor, overhead is HIGH, so we prefer LARGE batches
        # Target: Each worker should get 3-10 batches (not 100+ tiny batches)
        
        # Calculate batch size based on workers to minimize overhead
        target_batches_per_worker = 5  # Sweet spot: enough work distribution, minimal overhead
        ideal_batch_size = max(500, total_sentences // (extraction_workers * target_batches_per_worker))
        
        # Apply sensible limits
        if total_sentences < 1000:
            optimal_batch_size = 100  # Small dataset: normal batching
        elif total_sentences < 10000:
            optimal_batch_size = min(500, ideal_batch_size)
        elif total_sentences < 50000:
            optimal_batch_size = min(2000, ideal_batch_size)
        elif total_sentences < 200000:
            optimal_batch_size = min(5000, ideal_batch_size)
        else:
            # For 754K sentences with 12 workers: 
            # target_batches = 12 * 5 = 60 batches
            # batch_size = 754K / 60 = ~12,500 sentences/batch
            # This is MUCH better than 1887 batches of 400!
            optimal_batch_size = min(20000, ideal_batch_size)
        
        # Ensure we have enough batches for all workers
        min_batches = extraction_workers * 3  # At least 3 batches per worker
        max_batch_size = max(50, total_sentences // min_batches)
        optimal_batch_size = min(optimal_batch_size, max_batch_size)
        
        print(f"📑 Total sentences: {total_sentences:,}")
        print(f"📑 Target batch size: {optimal_batch_size} sentences")
        
        # Calculate expected number of batches
        expected_batches = (total_sentences + optimal_batch_size - 1) // optimal_batch_size
        print(f"📑 Expected batches: {expected_batches} (for {extraction_workers} workers)")
        print(f"📑 Batches per worker: ~{expected_batches // extraction_workers} batches")
        
        batches = [sentences[i:i + optimal_batch_size] for i in range(0, len(sentences), optimal_batch_size)]
        print(f"📑 Processing {len(batches)} batches of ~{optimal_batch_size} sentences each")
        print(f"📑 Expected speedup: {min(extraction_workers, len(batches))}x (using {extraction_workers} workers)")
        
        # Decide between ThreadPoolExecutor and ProcessPoolExecutor
        import multiprocessing
        in_subprocess = multiprocessing.current_process().name != 'MainProcess'
        
        # Use ProcessPoolExecutor for better parallelism on larger datasets
        # On Windows, we CAN use ProcessPoolExecutor in subprocess with spawn context
        use_process_pool = len(sentences) > 5000  # Remove subprocess check!
        
        if use_process_pool:
            # Check if we're in a daemonic process (can't spawn children)
            is_daemon = multiprocessing.current_process().daemon if hasattr(multiprocessing.current_process(), 'daemon') else False
            
            if in_subprocess and is_daemon:
                # Daemonic processes can't spawn children - fall back to ThreadPoolExecutor
                print(f"⚠️  Running in daemonic subprocess - cannot use ProcessPoolExecutor")
                print(f"📁 Falling back to ThreadPoolExecutor (limited parallelism due to GIL)")
                use_process_pool = False
                executor_class = ThreadPoolExecutor
                executor_kwargs = {'max_workers': extraction_workers}
                use_mp_pool = False
            else:
                # We can use ProcessPoolExecutor
                if in_subprocess:
                    print(f"📁 Using ProcessPoolExecutor in non-daemonic subprocess")
                    print(f"📁 This enables TRUE parallelism even from within a subprocess!")
                else:
                    print(f"📁 Using ProcessPoolExecutor for maximum performance (true parallelism)")
                
                mp_context = multiprocessing.get_context('spawn')
                executor_class = mp_context.Pool
                
                # Capture CURRENT environment variable values from parent process
                current_env_vars = {
                    'GLOSSARY_MAX_SENTENCES': os.getenv('GLOSSARY_MAX_SENTENCES', '200'),
                    'GLOSSARY_MIN_FREQUENCY': os.getenv('GLOSSARY_MIN_FREQUENCY', '2'),
                    'GLOSSARY_MAX_NAMES': os.getenv('GLOSSARY_MAX_NAMES', '50'),
                    'GLOSSARY_MAX_TITLES': os.getenv('GLOSSARY_MAX_TITLES', '30'),
                    'GLOSSARY_BATCH_SIZE': os.getenv('GLOSSARY_BATCH_SIZE', '50'),
                    'GLOSSARY_STRIP_HONORIFICS': os.getenv('GLOSSARY_STRIP_HONORIFICS', '1'),
                    'GLOSSARY_FUZZY_THRESHOLD': os.getenv('GLOSSARY_FUZZY_THRESHOLD', '0.90'),
                }
                print(f"📁 Passing env vars to child processes: GLOSSARY_MAX_SENTENCES={current_env_vars['GLOSSARY_MAX_SENTENCES']}")
                
                # For multiprocessing.Pool, we use different kwargs
                # Use module-level init function (can't use local function due to pickling)
                executor_kwargs = {
                    'processes': extraction_workers,
                    'initializer': _init_worker_with_env,
                    'initargs': (current_env_vars,)
                }
                use_mp_pool = True  # Flag to use different API
        else:
            print(f"📁 Using ThreadPoolExecutor for sentence processing (dataset < 5000 sentences)")
            executor_class = ThreadPoolExecutor
            executor_kwargs = {'max_workers': extraction_workers}
            use_mp_pool = False
        
        # Handle multiprocessing.Pool vs concurrent.futures differently
        if use_process_pool and use_mp_pool:
            # Use multiprocessing.Pool API (map_async)
            with executor_class(**executor_kwargs) as pool:
                # Prepare data for process pool
                exclude_check_data = (
                    list(honorifics_to_exclude),
                    [p.pattern for p in title_patterns],
                    PM.COMMON_WORDS,
                    PM.CHINESE_NUMS
                )
                
                # Prepare all arguments
                all_args = [(batch, idx, combined_pattern, exclude_check_data) 
                           for idx, batch in enumerate(batches)]
                
                print(f"📁 Submitting {len(all_args)} batches to process pool...")
                
                # Use map_async with chunksize for better distribution
                # chunksize=1 means each worker gets one batch at a time
                result_async = pool.map_async(_process_sentence_batch_for_extraction, all_args, chunksize=1)
                
                # Poll for completion with progress estimates
                completed_batches = 0
                batch_start_time = time.time()
                last_estimate_time = batch_start_time
                
                print(f"📁 Processing batches with {extraction_workers} parallel workers...")
                
                while not result_async.ready():
                    time.sleep(2)  # Check every 2 seconds
                    elapsed = time.time() - batch_start_time
                    
                    # Show periodic updates even without exact progress
                    if elapsed - (last_estimate_time - batch_start_time) >= 5:  # Every 5 seconds
                        # Estimate progress based on time and worker count
                        # With 16 workers and ~0.3s per batch, we process ~53 batches/sec total
                        # So elapsed * workers / 0.3 gives rough estimate
                        batches_per_second = extraction_workers / 0.3  # Rough estimate: 53 for 16 workers
                        estimated_completed = min(int(elapsed * batches_per_second), len(all_args))
                        
                        # Cap estimate at 95% until actually complete (avoid showing 100% while still working)
                        estimated_progress = min(95, (estimated_completed / len(all_args)) * 100)
                        estimated_sentences = min(estimated_completed * optimal_batch_size, total_sentences)
                        
                        if estimated_progress < 95:  # Only show if not at cap
                            print(f"📁 Processing... ~{estimated_progress:.0f}% estimated (~{estimated_sentences:,} sentences) | {elapsed:.0f}s elapsed")
                        else:
                            print(f"📁 Processing... finalizing last batches | {elapsed:.0f}s elapsed")
                        last_estimate_time = time.time()
                
                # Get all results
                total_elapsed = time.time() - batch_start_time
                print(f"📁 All batches completed in {total_elapsed:.1f}s! Collecting results...")
                all_results = result_async.get()
                
                # Process all results
                for local_word_freq, local_important, local_seen, batch_idx in all_results:
                    # Merge results
                    word_freq.update(local_word_freq)
                    for sentence in local_important:
                        sentence_key = ' '.join(sorted(re.findall(combined_pattern, sentence)))
                        if sentence_key not in seen_contexts:
                            important_sentences.append(sentence)
                            seen_contexts.add(sentence_key)
                    
                    processed_count += len(batches[batch_idx])
                    completed_batches += 1
                    
                    # Show progress
                    progress_interval = 1 if len(batches) <= 20 else (5 if len(batches) <= 100 else 10)
                    if completed_batches % progress_interval == 0 or completed_batches == len(batches):
                        progress = (processed_count / total_sentences) * 100
                        elapsed = time.time() - batch_start_time
                        rate = (processed_count / elapsed) if elapsed > 0 else 0
                        print(f"📑 Progress: {processed_count:,}/{total_sentences:,} sentences ({progress:.1f}%) | Batch {completed_batches}/{len(batches)} | {rate:.0f} sent/sec")
        else:
            # Use concurrent.futures API (ThreadPoolExecutor or ProcessPoolExecutor)
            with executor_class(**executor_kwargs) as executor:
                futures = []
                
                # Prepare data for ProcessPoolExecutor if needed
                if use_process_pool:
                    # Serialize exclusion check data for process pool
                    exclude_check_data = (
                        list(honorifics_to_exclude),
                        [p.pattern for p in title_patterns],
                        PM.COMMON_WORDS,
                        PM.CHINESE_NUMS
                    )
                
                for idx, batch in enumerate(batches):
                    if use_process_pool:
                        # Use module-level function for ProcessPoolExecutor
                        future = executor.submit(_process_sentence_batch_for_extraction, 
                                               (batch, idx, combined_pattern, exclude_check_data))
                    else:
                        # Use local function for ThreadPoolExecutor
                        future = executor.submit(process_sentence_batch, batch, idx)
                    
                    futures.append(future)
                    # Yield to GUI when submitting futures
                    if idx % 10 == 0:
                        time.sleep(0.001)
                
                # Collect results with progress
                completed_batches = 0
                batch_start_time = time.time()
                for future in as_completed(futures):
                    # Get result without timeout - as_completed already handles waiting
                    local_word_freq, local_important, local_seen, batch_idx = future.result()
                    
                    # Merge results
                    word_freq.update(local_word_freq)
                    for sentence in local_important:
                        sentence_key = ' '.join(sorted(re.findall(combined_pattern, sentence)))
                        if sentence_key not in seen_contexts:
                            important_sentences.append(sentence)
                            seen_contexts.add(sentence_key)
                    
                    processed_count += len(batches[batch_idx])
                    completed_batches += 1
                    
                    # Show progress more frequently for better user feedback
                    progress_interval = 1 if len(batches) <= 20 else (5 if len(batches) <= 100 else 10)
                    
                    if completed_batches % progress_interval == 0 or completed_batches == len(batches):
                        progress = (processed_count / total_sentences) * 100
                        elapsed = time.time() - batch_start_time
                        rate = (processed_count / elapsed) if elapsed > 0 else 0
                        print(f"📑 Progress: {processed_count:,}/{total_sentences:,} sentences ({progress:.1f}%) | Batch {completed_batches}/{len(batches)} | {rate:.0f} sent/sec")
                    
                    # Yield to GUI after each batch completes
                    time.sleep(0.001)
    else:
        # Sequential processing with progress
        for idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) < 10 or len(sentence) > 500:
                continue
                
            # Find all potential terms in this sentence
            matches = re.findall(combined_pattern, sentence)
            
            if matches:
                # Filter out excluded terms
                filtered_matches = []
                for match in matches:
                    if not should_exclude_term(match):
                        word_freq[match] += 1
                        filtered_matches.append(match)
                
                # Keep sentences with valid potential terms
                if filtered_matches:
                    sentence_key = ' '.join(sorted(filtered_matches))
                    if sentence_key not in seen_contexts:
                        important_sentences.append(sentence)
                        seen_contexts.add(sentence_key)
            
            # Show progress every 1000 sentences or 2 seconds
            if idx % 1000 == 0 or (time.time() - last_progress_time > 2):
                progress = ((idx + 1) / total_sentences) * 100
                print(f"📑 Processing sentences: {idx + 1:,}/{total_sentences:,} ({progress:.1f}%)")
                last_progress_time = time.time()
                # Yield to GUI thread every 1000 sentences
                time.sleep(0.001)  # Tiny sleep to let GUI update
                # Yield to GUI thread every 1000 sentences
                time.sleep(0.001)  # Tiny sleep to let GUI update
    
    print(f"📑 Found {len(important_sentences):,} sentences with potential glossary terms")
    
    # Step 6/7: Deduplicate and normalize terms
    print(f"📑 Step 6/7: Normalizing and deduplicating {len(word_freq):,} unique terms...")
    
    # Since should_exclude_term already filters honorifics, we just need to deduplicate
    # based on normalized forms (lowercase, etc.)
    combined_freq = Counter()
    term_count = 0
    
    for term, count in word_freq.items():
        # Normalize term for deduplication (but keep original form)
        normalized = term.lower().strip()
        
        # Keep the version with highest count
        if normalized in combined_freq:
            # If we already have this normalized form, keep the one with higher count
            if count > combined_freq[normalized]:
                # Remove old entry and add new one
                del combined_freq[normalized]
                combined_freq[term] = count
        else:
            combined_freq[term] = count
        
        term_count += 1
        # Yield to GUI every 1000 terms
        if term_count % 1000 == 0:
            time.sleep(0.001)
    
    print(f"📑 Deduplicated to {len(combined_freq):,} unique terms")
    
    # Filter to keep only terms that appear at least min_frequency times
    frequent_terms = {term: count for term, count in combined_freq.items() if count >= min_frequency}
    
    # Build filtered text focusing on sentences containing frequent terms
    print(f"📑 Step 7/7: Building filtered text from relevant sentences...")
    
    # OPTIMIZATION: Skip sentences that already passed filtering in step 5
    # These sentences already contain glossary terms, no need to check again!
    # We just need to limit the sample size
    
    filtered_sentences = important_sentences  # Already filtered!
    print(f"📑 Using {len(filtered_sentences):,} pre-filtered sentences (already contain glossary terms)")
    
    # For extremely large datasets, we can optionally do additional filtering
    if len(filtered_sentences) > 10000 and len(frequent_terms) > 1000:
        print(f"📑 Large dataset detected - applying frequency-based filtering...")
        print(f"📑 Filtering {len(filtered_sentences):,} sentences for top frequent terms...")
        
        # Sort terms by frequency to prioritize high-frequency ones
        sorted_terms = sorted(frequent_terms.items(), key=lambda x: x[1], reverse=True)
        top_terms = dict(sorted_terms[:1000])  # Focus on top 1000 most frequent terms
        
        print(f"📑 Using top {len(top_terms):,} most frequent terms for final filtering")
        
        # Use parallel processing only if really needed
        if use_parallel and len(filtered_sentences) > 5000:
            import multiprocessing
            in_subprocess = multiprocessing.current_process().name != 'MainProcess'
            
            # Create a simple set of terms for fast lookup (no variations needed)
            term_set = set(top_terms.keys())
            
            print(f"📑 Using parallel filtering with {extraction_workers} workers...")
            
            # Optimize batch size for ProcessPoolExecutor (reduce overhead)
            # Use larger batches since this is a simpler operation than term extraction
            check_batch_size = max(1000, len(filtered_sentences) // (extraction_workers * 5))
            check_batches = [filtered_sentences[i:i + check_batch_size] 
                           for i in range(0, len(filtered_sentences), check_batch_size)]
            
            print(f"📑 Processing {len(check_batches)} batches of ~{check_batch_size} sentences")
            
            # Use ProcessPoolExecutor for true parallelism (if not already in subprocess)
            use_process_pool_filtering = (not in_subprocess and len(check_batches) > 3)
            
            if use_process_pool_filtering:
                print(f"📑 Using ProcessPoolExecutor for true parallel filtering")
                new_filtered = []
                with ProcessPoolExecutor(max_workers=extraction_workers) as executor:
                    # Use the module-level function _check_sentence_batch_for_terms
                    futures = [executor.submit(_check_sentence_batch_for_terms, (batch, term_set)) 
                              for batch in check_batches]
                    
                    for future in as_completed(futures):
                        new_filtered.extend(future.result())
            else:
                print(f"📑 Using ThreadPoolExecutor for filtering (small dataset or in subprocess)")
                # Simple function to check if sentence contains any top term
                def check_batch_simple(batch):
                    result = []
                    for sentence in batch:
                        # Simple substring check - much faster than regex
                        for term in term_set:
                            if term in sentence:
                                result.append(sentence)
                                break
                    return result
                
                new_filtered = []
                with ThreadPoolExecutor(max_workers=extraction_workers) as executor:
                    futures = [executor.submit(check_batch_simple, batch) for batch in check_batches]
                    
                    for future in as_completed(futures):
                        new_filtered.extend(future.result())
            
            filtered_sentences = new_filtered
            print(f"📑 Filtered to {len(filtered_sentences):,} sentences containing top terms")
        else:
            # For smaller datasets, simple sequential filtering
            print(f"📑 Using sequential filtering...")
            new_filtered = []
            for i, sentence in enumerate(filtered_sentences):
                for term in top_terms:
                    if term in sentence:
                        new_filtered.append(sentence)
                        break
                if i % 1000 == 0:
                    print(f"📑 Progress: {i:,}/{len(filtered_sentences):,} sentences")
                    time.sleep(0.001)
            
            filtered_sentences = new_filtered
            print(f"📑 Filtered to {len(filtered_sentences):,} sentences containing top terms")
    
    print(f"📑 Selected {len(filtered_sentences):,} sentences containing frequent terms")
    
    # Limit the number of sentences to reduce token usage
    if max_sentences is None:
        max_sentences_fallback = os.getenv("GLOSSARY_MAX_SENTENCES", "200")
        print(f"🔍 [DEBUG] max_sentences was None, reading from environment: '{max_sentences_fallback}'")
        max_sentences = int(max_sentences_fallback)
    else:
        print(f"🔍 [DEBUG] max_sentences parameter was provided: {max_sentences}")
    
    print(f"🔍 [DEBUG] Final GLOSSARY_MAX_SENTENCES value being used: {max_sentences}")
    # Handle max_sentences = 0 as "include all sentences"
    if max_sentences > 0 and len(filtered_sentences) > max_sentences:
        print(f"📁 Limiting to {max_sentences} representative sentences (from {len(filtered_sentences):,})")
        # Take a representative sample
        step = len(filtered_sentences) // max_sentences
        filtered_sentences = filtered_sentences[::step][:max_sentences]
    elif max_sentences == 0:
        print(f"📁 Including ALL {len(filtered_sentences):,} sentences (max_sentences=0)")
    
    # Check if gender context expansion is enabled
    include_gender_context = os.getenv("GLOSSARY_INCLUDE_GENDER_CONTEXT", "0") == "1"
    
    if include_gender_context:
        context_window = int(os.getenv("GLOSSARY_CONTEXT_WINDOW", "2"))
        print(f"📑 Gender context enabled: Expanding snippets with {context_window}-sentence windows...")
        
        # Split full text into sentences for context extraction
        all_sentences_list = re.split(r'[.!?。！？]+', clean_text)
        all_sentences_list = [s.strip() for s in all_sentences_list if s.strip()]
        
        # Create index map for fast lookup - OPTIMIZED to O(n) instead of O(n²)
        # Build a lookup dict: sentence -> index for fast matching
        sentence_to_index = {}
        all_sentences_normalized = {s.strip(): idx for idx, s in enumerate(all_sentences_list)}
        
        print(f"📑 Mapping {len(filtered_sentences):,} filtered sentences to context positions...")
        for filtered_sent in filtered_sentences:
            filtered_normalized = filtered_sent.strip()
            
            # Try exact match first (fastest)
            if filtered_normalized in all_sentences_normalized:
                sentence_to_index[filtered_sent] = all_sentences_normalized[filtered_normalized]
            else:
                # Try substring match (slower fallback)
                found = False
                for sentence, idx in all_sentences_normalized.items():
                    if filtered_normalized in sentence or sentence in filtered_normalized:
                        sentence_to_index[filtered_sent] = idx
                        found = True
                        break
                
                if not found:
                    # Last resort: try finding in original list
                    for idx, sentence in enumerate(all_sentences_list):
                        if filtered_normalized in sentence or sentence in filtered_normalized:
                            sentence_to_index[filtered_sent] = idx
                            break
        
        # Build context windows
        context_groups = []
        included_indices = set()
        
        for filtered_sent in filtered_sentences:
            if filtered_sent not in sentence_to_index:
                # If we can't find it, just use the sentence as-is
                context_groups.append(filtered_sent)
                continue
            
            idx = sentence_to_index[filtered_sent]
            
            # Skip if already included in a previous window
            if idx in included_indices:
                continue
            
            # Get context window: [idx-context_window ... idx ... idx+context_window]
            start_idx = max(0, idx - context_window)
            end_idx = min(len(all_sentences_list), idx + context_window + 1)
            
            # Mark all sentences in this window as included
            for i in range(start_idx, end_idx):
                included_indices.add(i)
            
            # Extract the window
            window_sentences = all_sentences_list[start_idx:end_idx]
            context_group = ' '.join(window_sentences)
            context_groups.append(context_group)
        
        print(f"📑 Created {len(context_groups):,} context windows (up to {context_window*2+1} sentences each)")
        filtered_text = '\n\n'.join(context_groups)  # Separate windows with double newline
        print(f"📑 Context-expanded text: {len(filtered_text):,} characters")
    else:
        filtered_text = ' '.join(filtered_sentences)
    
    # Calculate and display filtering statistics
    filter_end_time = time.time()
    filter_duration = filter_end_time - filter_start_time
    
    original_length = len(clean_text)
    filtered_length = len(filtered_text)
    size_change_percent = ((original_length - filtered_length) / original_length * 100) if original_length > 0 else 0
    
    print(f"\n📑 === FILTERING COMPLETE ===")
    print(f"📑 Duration: {filter_duration:.1f} seconds")
    if size_change_percent >= 0:
        print(f"📑 Text reduction: {original_length:,} → {filtered_length:,} chars ({size_change_percent:.1f}% reduction)")
    else:
        print(f"📑 Text expansion: {original_length:,} → {filtered_length:,} chars ({abs(size_change_percent):.1f}% expansion)")
    print(f"📑 Terms found: {len(frequent_terms):,} unique terms (min frequency: {min_frequency})")
    print(f"📑 Final output: {len(filtered_sentences)} sentences, {filtered_length:,} characters")
    print(f"📑 Performance: {(original_length / filter_duration / 1000):.1f}K chars/second")
    print(f"📑 ========================\n")
    
    return filtered_text, frequent_terms

def _extract_with_custom_prompt(custom_prompt, all_text, language, 
                              min_frequency, max_names, max_titles, 
                              existing_glossary, output_dir, 
                              strip_honorifics=True, fuzzy_threshold=0.90, filter_mode='all', max_sentences=200, log_callback=None):
    """Extract glossary using custom AI prompt with proper filtering"""
    # Redirect stdout to GUI log if callback provided (but not in subprocess - worker handles it)
    import sys
    in_subprocess = hasattr(sys.stdout, 'queue')
    if log_callback and not in_subprocess:
        set_output_redirect(log_callback)
    
    print("📑 Using custom automatic glossary prompt")
    extraction_start = time.time()
    
    # Check stop flag
    if is_stop_requested():
        print("📑 ❌ Glossary extraction stopped by user")
        return {}
    
    # Note: Filter mode can be controlled via the configurable prompt environment variable
    # No hardcoded filter instructions are added here
    
    try:
        MODEL = os.getenv("MODEL", "gemini-2.0-flash")
        API_KEY = (os.getenv("API_KEY") or 
                   os.getenv("OPENAI_API_KEY") or 
                   os.getenv("OPENAI_OR_Gemini_API_KEY") or
                   os.getenv("GEMINI_API_KEY"))
        
        if is_traditional_translation_api(MODEL):
            return _translate_chunk_traditional(chunk_text, chunk_index, total_chunks, chapter_title)
        
        elif not API_KEY:
            print(f"📑 No API key found, falling back to pattern-based extraction")
            return _extract_with_patterns(all_text, language, min_frequency, 
                                             max_names, max_titles, 50,
                                             existing_glossary, output_dir, 
                                             strip_honorifics, fuzzy_threshold, filter_mode)
        else:
            print(f"📑 Using AI-assisted extraction with custom prompt")
            
            from unified_api_client import UnifiedClient, UnifiedClientError
            client = UnifiedClient(model=MODEL, api_key=API_KEY, output_dir=output_dir)
            if hasattr(client, 'reset_cleanup_state'):
                client.reset_cleanup_state()
            
            # Apply thread submission delay using the client's method
            thread_delay = float(os.getenv("THREAD_SUBMISSION_DELAY_SECONDS", "0.5"))
            if thread_delay > 0:
                client._apply_thread_submission_delay()
                
                # Check if cancelled during delay
                if hasattr(client, '_cancelled') and client._cancelled:
                    print("📑 ❌ Glossary extraction stopped during delay")
                    return {}
                
            # Check if text is already filtered (from chunking or cache)
            already_filtered = (os.getenv("_CHUNK_ALREADY_FILTERED", "0") == "1" or 
                               os.getenv("_TEXT_ALREADY_FILTERED", "0") == "1")
            
            if already_filtered:
                # print("📑 Text already filtered, skipping re-filtering")
                text_sample = all_text  # Use as-is since it's already filtered
                detected_terms = {}
            else:
            # Apply smart filtering to reduce noise and focus on meaningful content
                force_disable = os.getenv("GLOSSARY_FORCE_DISABLE_SMART_FILTER", "0") == "1"
                use_smart_filter = (os.getenv("GLOSSARY_USE_SMART_FILTER", "1") == "1") and not force_disable
                
                if not use_smart_filter:
                    # Smart filter disabled - send FULL text without any filtering or truncation
                    print("📁 Smart filtering DISABLED by user - sending full text to API (this will be expensive!)")
                    text_sample = all_text
                    detected_terms = {}
                else:
                    # Smart filter enabled - apply intelligent filtering
                    print("📁 Applying smart text filtering to reduce noise...")
                    # Use max_sentences parameter (passed from parent, already read from environment)
                    print(f"🔍 [DEBUG] In _extract_with_custom_prompt: max_sentences={max_sentences}")
                    text_sample, detected_terms = _filter_text_for_glossary(all_text, min_frequency, max_sentences)
            
            # Replace placeholders in prompt
            # Get target language from environment (used in the prompt for translation output)
            target_language = os.getenv('GLOSSARY_TARGET_LANGUAGE', 'English')
            system_prompt = custom_prompt.replace('{language}', target_language)
            system_prompt = system_prompt.replace('{min_frequency}', str(min_frequency))
            system_prompt = system_prompt.replace('{max_names}', str(max_names))
            system_prompt = system_prompt.replace('{max_titles}', str(max_titles))
            
            # Send system prompt and text as separate messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{text_sample}"}
            ]
            
            # Check stop before API call
            if is_stop_requested():
                print("📑 ❌ Glossary extraction stopped before API call")
                return {}
            
            try:
                temperature = float(os.getenv("TEMPERATURE", "0.3"))
                max_tokens = int(os.getenv("MAX_OUTPUT_TOKENS", "4096"))
                
                # Use send_with_interrupt for interruptible API call
                # Respect RETRY_TIMEOUT toggle - if disabled, use None for infinite timeout
                retry_timeout_enabled = os.getenv("RETRY_TIMEOUT", "0") == "1"
                if retry_timeout_enabled:
                    chunk_timeout = int(os.getenv("CHUNK_TIMEOUT", "900"))  # 15 minute default for glossary
                    print(f"📑 Sending AI extraction request (timeout: {chunk_timeout}s, interruptible)...")
                else:
                    chunk_timeout = None
                    print(f"📑 Sending AI extraction request (timeout: disabled, interruptible)...")
                
                # Before API call
                api_start = time.time()
                print(f"📑 Preparing API request (text size: {len(text_sample):,} chars)...")
                print(f"📑 ⏳ Processing {len(text_sample):,} characters... Please wait, this may take 5-10 minutes")

                response = send_with_interrupt(
                    messages=messages,
                    client=client,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop_check_fn=is_stop_requested,
                    chunk_timeout=chunk_timeout
                )
                api_time = time.time() - api_start
                print(f"📑 API call completed in {api_time:.1f}s")

                # Get the actual text from the response
                if hasattr(response, 'content'):
                    response_text = response.content
                else:
                    response_text = str(response)

                # Before processing response
                process_start = time.time()
                # print(f"📑 Processing AI response...")              
                # Process response and build CSV
                csv_lines = _process_ai_response(response_text, all_text, min_frequency, 
                                                     strip_honorifics, fuzzy_threshold, 
                                                     language, filter_mode)
                
                print(f"📑 AI extracted {len(csv_lines) - 1} valid terms (header excluded)")

                process_time = time.time() - process_start
                # print(f"📑 Response processing took {process_time:.1f}s")
                
                # If we're running per-chunk, defer all heavy work and saving
                if os.getenv("GLOSSARY_DEFER_SAVE", "0") == "1":
                    return csv_lines
                
                # Check stop before merging
                if is_stop_requested():
                    print("📑 ❌ Glossary generation stopped before merging")
                    return {}
                
                # Merge with existing glossary if present
                if existing_glossary:
                    csv_lines = _merge_csv_entries(csv_lines, existing_glossary, strip_honorifics, language)

                # Fuzzy matching deduplication
                skip_frequency_check = os.getenv("GLOSSARY_SKIP_FREQUENCY_CHECK", "0") == "1"
                if not skip_frequency_check:  # Only dedupe if we're checking frequencies
                    # Time the deduplication
                    dedup_start = time.time()
                    original_count = len(csv_lines) - 1  # Exclude header
                    
                    csv_lines = _deduplicate_glossary_with_fuzzy(csv_lines, fuzzy_threshold)
                    
                    dedup_time = time.time() - dedup_start
                    final_count = len(csv_lines) - 1  # Exclude header
                    removed_count = original_count - final_count
                    
                    print(f"📑 Deduplication completed in {dedup_time:.1f}s")
                    print(f"📑   - Original entries: {original_count}")
                    print(f"📑   - Duplicates removed: {removed_count}")
                    print(f"📑   - Final entries: {final_count}")
                    
                    # Store for summary statistics
                    _dedup_time = 0 + dedup_time
                else:
                    print(f"📑 Skipping deduplication (frequency check disabled)")
                
                # Apply filter mode to final results
                csv_lines = _filter_csv_by_mode(csv_lines, filter_mode)
                
                # Check if we should use token-efficient format
                use_legacy_format = os.getenv('GLOSSARY_USE_LEGACY_CSV', '0') == '1'

                if not use_legacy_format:
                    # Convert to token-efficient format
                    csv_lines = _convert_to_token_efficient_format(csv_lines)
                
                # Final sanitize to prevent stray headers
                csv_lines = _sanitize_final_glossary_lines(csv_lines, use_legacy_format)
                
                # Create final CSV content
                csv_content = '\n'.join(csv_lines)
                
                # Save glossary as CSV with proper extension
                glossary_path = os.path.join(output_dir, "glossary.csv")
                _atomic_write_file(glossary_path, csv_content)
                
                print(f"\n📑 ✅ AI-ASSISTED GLOSSARY SAVED!")
                print(f"📑 File: {glossary_path}")
                c_count, t_count, total = _count_glossary_entries(csv_lines, use_legacy_format)
                print(f"📑 Character entries: {c_count}")
                # print(f"📑 Term entries: {t_count}")
                print(f"📑 Total entries: {total}")
                total_time = time.time() - extraction_start
                print(f"📑 Total extraction time: {total_time:.1f}s")
                return _parse_csv_to_dict(csv_content)
                
            except UnifiedClientError as e:
                if "stopped by user" in str(e).lower():
                    print(f"📑 ❌ AI extraction interrupted by user")
                    return {}
                else:
                    print(f"⚠️ AI extraction failed: {e}")
                    print("📑 ❌ Glossary generation failed - returning empty glossary")
                    return {}
            except Exception as e:
                print(f"⚠️ AI extraction failed: {e}")
                import traceback
                traceback.print_exc()
                print("📑 ❌ Glossary generation failed - returning empty glossary")
                return {}
                
    except Exception as e:
        print(f"⚠️ Custom prompt processing failed: {e}")
        import traceback
        traceback.print_exc()
        print("📑 ❌ Glossary generation failed - returning empty glossary")
        return {}

def _filter_csv_by_mode(csv_lines, filter_mode):
    """Filter CSV lines based on the filter mode"""
    if filter_mode == "all":
        return csv_lines
    
    filtered = [csv_lines[0]]  # Keep header
    
    for line in csv_lines[1:]:
        if not line.strip():
            continue
        
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 3:
            continue
        
        entry_type = parts[0].lower()
        raw_name = parts[1]
        
        if filter_mode == "only_with_honorifics":
            # Only keep character entries with honorifics
            if entry_type == "character" and _has_honorific(raw_name):
                filtered.append(line)
        elif filter_mode == "only_without_honorifics":
            # Keep terms and characters without honorifics
            if entry_type == "term" or (entry_type == "character" and not _has_honorific(raw_name)):
                filtered.append(line)
    
    print(f"📑 Filter '{filter_mode}': {len(filtered)-1} entries kept from {len(csv_lines)-1}")
    return filtered

def _process_ai_response(response_text, all_text, min_frequency, 
                       strip_honorifics, fuzzy_threshold, language, filter_mode):
    """Process AI response and return CSV lines"""

    # Check if gender context and description are enabled (used throughout the function)
    include_gender_context = os.getenv("GLOSSARY_INCLUDE_GENDER_CONTEXT", "0") == "1"
    include_description = os.getenv("GLOSSARY_INCLUDE_DESCRIPTION", "0") == "1"
    
    # option to completely skip frequency validation for speed
    skip_all_validation = os.getenv("GLOSSARY_SKIP_ALL_VALIDATION", "0") == "1"

    # if skip_all_validation:
    #     print("📑 ⚡ FAST MODE: Skipping all frequency validation (accepting all AI results)")

    # Clean response text
    response_text = response_text.strip()
    
    # Remove string representation artifacts if they wrap the entire response
    if response_text.startswith('("') and response_text.endswith('")'):
        response_text = response_text[2:-2]
    elif response_text.startswith('"') and response_text.endswith('"'):
        response_text = response_text[1:-1]
    elif response_text.startswith('(') and response_text.endswith(')'):
        response_text = response_text[1:-1]
    
    # Unescape the string
    response_text = response_text.replace('\\n', '\n')
    response_text = response_text.replace('\\r', '')
    response_text = response_text.replace('\\t', '\t')
    response_text = response_text.replace('\\"', '"')
    response_text = response_text.replace("\\'", "'")
    response_text = response_text.replace('\\\\', '\\')
    
    # Clean up markdown code blocks if present
    if '```' in response_text:
        parts = response_text.split('```')
        for part in parts:
            if 'csv' in part[:10].lower():
                response_text = part[part.find('\n')+1:]
                break
            elif part.strip() and ('type,raw_name' in part or 'character,' in part or 'term,' in part):
                response_text = part
                break
    
    # Normalize line endings
    response_text = response_text.replace('\r\n', '\n').replace('\r', '\n')
    lines = [line.strip() for line in response_text.strip().split('\n') if line.strip()]
    
    csv_lines = []
    header_found = False
    
    # Check if we should skip frequency check
    skip_frequency_check = os.getenv("GLOSSARY_SKIP_FREQUENCY_CHECK", "0") == "1"

    # Add option to completely skip ALL validation for maximum speed
    skip_all_validation = os.getenv("GLOSSARY_SKIP_ALL_VALIDATION", "0") == "1"
    
    if skip_all_validation:
        # print("📑 ⚡ FAST MODE: Skipping all frequency validation (accepting all AI results)")
        
        # Use appropriate header based on gender and description settings
        if include_description:
            csv_lines.append("type,raw_name,translated_name,gender,description")
        elif include_gender_context:
            csv_lines.append("type,raw_name,translated_name,gender")
            # print("📑 Fast mode: Using 4-column format with gender")
        else:
            csv_lines.append("type,raw_name,translated_name")
        
        # Process the AI response
        for line in lines:
            # Skip header lines
            if 'type' in line.lower() and 'raw_name' in line.lower():
                continue
                
            # Parse CSV line
            parts = [p.strip() for p in line.split(',')]
            
            # Replace invalid 'stop' values with empty string
            parts = ['' if p == "'stop'" or p == "stop" else p for p in parts]
            
            if include_description and len(parts) >= 5:
                # Has all 5 columns (with gender and description)
                entry_type = parts[0]
                raw_name = parts[1]
                translated_name = parts[2]
                gender = parts[3] if len(parts) > 3 else ''
                description = parts[4] if len(parts) > 4 else ''
                
                # Validate - reject malformed entries that look like tuples/lists or quoted strings
                if (raw_name and translated_name and 
                    not (raw_name.startswith(('[', '(', "'", '"')) or translated_name.startswith(('[', '(', "'", '"'))) and
                    not (raw_name.endswith(("'", '"')) or translated_name.endswith(("'", '"')))):
                    csv_lines.append(f"{entry_type},{raw_name},{translated_name},{gender},{description}")
            elif include_gender_context and len(parts) >= 4:
                # Has all 4 columns (with gender)
                entry_type = parts[0]
                raw_name = parts[1]
                translated_name = parts[2]
                gender = parts[3] if len(parts) > 3 else ''
                
                # Validate - reject malformed entries that look like tuples/lists or quoted strings
                if (raw_name and translated_name and 
                    not (raw_name.startswith(('[', '(', "'", '"')) or translated_name.startswith(('[', '(', "'", '"'))) and
                    not (raw_name.endswith(("'", '"')) or translated_name.endswith(("'", '"')))):
                    csv_lines.append(f"{entry_type},{raw_name},{translated_name},{gender}")
            elif len(parts) >= 3:
                # Has at least 3 columns
                entry_type = parts[0]
                raw_name = parts[1]
                translated_name = parts[2]
                # Validate - reject malformed entries that look like tuples/lists or quoted strings
                if (raw_name and translated_name and
                    not (raw_name.startswith(('[', '(', "'", '"')) or translated_name.startswith(('[', '(', "'", '"'))) and
                    not (raw_name.endswith(("'", '"')) or translated_name.endswith(("'", '"')))):
                    if include_description:
                        # Add empty gender and description columns when 5 columns expected
                        gender = parts[3] if len(parts) > 3 else ''
                        description = parts[4] if len(parts) > 4 else ''
                        csv_lines.append(f"{entry_type},{raw_name},{translated_name},{gender},{description}")
                    elif include_gender_context:
                        # Add empty gender column for 3-column entries when 4 columns expected
                        gender = parts[3] if len(parts) > 3 else ''
                        csv_lines.append(f"{entry_type},{raw_name},{translated_name},{gender}")
                    else:
                        csv_lines.append(f"{entry_type},{raw_name},{translated_name}")
            elif len(parts) == 2:
                # Missing type, default to 'term'
                raw_name = parts[0]
                translated_name = parts[1]
                # Validate - reject malformed entries that look like tuples/lists or quoted strings
                if (raw_name and translated_name and
                    not (raw_name.startswith(('[', '(', "'", '"')) or translated_name.startswith(('[', '(', "'", '"'))) and
                    not (raw_name.endswith(("'", '"')) or translated_name.endswith(("'", '"')))):
                    if include_description:
                        csv_lines.append(f"term,{raw_name},{translated_name},,")
                    elif include_gender_context:
                        csv_lines.append(f"term,{raw_name},{translated_name},")
                    else:
                        csv_lines.append(f"term,{raw_name},{translated_name}")
        
        # print(f"📑 Fast mode: Accepted {len(csv_lines) - 1} entries without validation")
        return csv_lines
    
    # For "only_with_honorifics" mode, ALWAYS skip frequency check
    if filter_mode == "only_with_honorifics":
        skip_frequency_check = True
        print("📑 Filter mode 'only_with_honorifics': Bypassing frequency checks")
    
    print(f"📑 Processing {len(lines)} lines from AI response...")
    print(f"📑 Text corpus size: {len(all_text):,} chars")
    print(f"📑 Frequency checking: {'DISABLED' if skip_frequency_check else f'ENABLED (min: {min_frequency})'}")  
    print(f"📑 Fuzzy threshold: {fuzzy_threshold}")
    
    # Collect all terms first for batch processing
    all_terms_to_check = []
    term_info_map = {}  # Map term to its full info
    
    if not skip_frequency_check:
        # First pass: collect all terms that need frequency checking
        for line in lines:
            if 'type' in line.lower() and 'raw_name' in line.lower():
                continue  # Skip header
            
            parts = [p.strip() for p in line.split(',')]
            
            # Replace invalid 'stop' values with empty string
            parts = ['' if p == "'stop'" or p == "stop" else p for p in parts]
            
            # Strip orphaned quotes and filter empty columns
            parts = [p.strip('"').strip("'").strip() for p in parts]
            parts = [p for p in parts if p]  # Remove empty strings
            
            if len(parts) >= 3:
                entry_type = parts[0].lower()
                raw_name = parts[1]
                translated_name = parts[2]
                gender = parts[3] if len(parts) > 3 else ''
                description = parts[4] if len(parts) > 4 else ''
            elif len(parts) == 2:
                entry_type = 'term'
                raw_name = parts[0]
                translated_name = parts[1]
                gender = ''
                description = ''
            else:
                continue
            
            # Validate - reject malformed entries that look like tuples/lists or quoted strings
            if not raw_name or not translated_name:
                continue
            if (raw_name.startswith(('[', '(', "'", '"')) or translated_name.startswith(('[', '(', "'", '"')) or
                raw_name.endswith(("'", '"')) or translated_name.endswith(("'", '"'))):
                continue
            
            if raw_name and translated_name:
                # Store for batch processing
                original_raw = raw_name
                if strip_honorifics:
                    raw_name = _strip_honorific(raw_name, language)
                
                all_terms_to_check.append(raw_name)
                term_info_map[raw_name] = {
                    'entry_type': entry_type,
                    'original_raw': original_raw,
                    'translated_name': translated_name,
                    'gender': gender,
                    'description': description,
                    'line': line
                }
        
        # Batch compute all frequencies at once
        if all_terms_to_check:
            print(f"📑 Computing frequencies for {len(all_terms_to_check)} terms...")
            term_frequencies = _batch_compute_frequencies(
                all_terms_to_check, all_text, fuzzy_threshold, min_frequency
            )
        else:
            term_frequencies = {}

    # Now process the results using pre-computed frequencies
    entries_processed = 0
    entries_accepted = 0
    # Process based on mode
    if filter_mode == "only_with_honorifics" or skip_frequency_check:
        # For these modes, accept all entries
        if include_description:
            csv_lines.append("type,raw_name,translated_name,gender,description")  # Header with description
        elif include_gender_context:
            csv_lines.append("type,raw_name,translated_name,gender")  # Header with gender
        else:
            csv_lines.append("type,raw_name,translated_name")  # Header
        
        for line in lines:
            if 'type' in line.lower() and 'raw_name' in line.lower():
                continue  # Skip header
            
            parts = [p.strip() for p in line.split(',')]
            
            # Replace invalid 'stop' values with empty string
            parts = ['' if p == "'stop'" or p == "stop" else p for p in parts]
            
            # Strip orphaned quotes and filter empty columns
            parts = [p.strip('"').strip("'").strip() for p in parts]
            parts = [p for p in parts if p]  # Remove empty strings
            
            if len(parts) >= 3:
                entry_type = parts[0].lower()
                raw_name = parts[1]
                translated_name = parts[2]
                gender = parts[3] if len(parts) > 3 else ''
                description = parts[4] if len(parts) > 4 else ''
            elif len(parts) == 2:
                entry_type = 'term'
                raw_name = parts[0]
                translated_name = parts[1]
                gender = ''
                description = ''
            else:
                continue
            
            # Validate - reject malformed entries that look like tuples/lists or quoted strings
            if not raw_name or not translated_name:
                continue
            if (raw_name.startswith(('[', '(', "'", '"')) or translated_name.startswith(('[', '(', "'", '"')) or
                raw_name.endswith(("'", '"')) or translated_name.endswith(("'", '"'))):
                continue
            
            if raw_name and translated_name:
                if include_description:
                    csv_line = f"{entry_type},{raw_name},{translated_name},{gender},{description}"
                elif include_gender_context:
                    csv_line = f"{entry_type},{raw_name},{translated_name},{gender}"
                else:
                    csv_line = f"{entry_type},{raw_name},{translated_name}"
                csv_lines.append(csv_line)
                entries_accepted += 1
        
        print(f"📑 Accepted {entries_accepted} entries (frequency check disabled)")
    
    else:
        # Use pre-computed frequencies
        if include_description:
            csv_lines.append("type,raw_name,translated_name,gender,description")  # Header with description
        elif include_gender_context:
            csv_lines.append("type,raw_name,translated_name,gender")  # Header with gender
        else:
            csv_lines.append("type,raw_name,translated_name")  # Header
        
        for term, info in term_info_map.items():
            count = term_frequencies.get(term, 0)
            
            # Also check original form if it was stripped
            if info['original_raw'] != term:
                count += term_frequencies.get(info['original_raw'], 0)
            
            if count >= min_frequency:
                if include_description:
                    csv_line = f"{info['entry_type']},{term},{info['translated_name']},{info['gender']},{info['description']}"
                elif include_gender_context:
                    csv_line = f"{info['entry_type']},{term},{info['translated_name']},{info['gender']}"
                else:
                    csv_line = f"{info['entry_type']},{term},{info['translated_name']}"
                csv_lines.append(csv_line)
                entries_accepted += 1
                
                # Log first few examples
                if entries_accepted <= 5:
                    print(f"📑   ✓ Example: {term} -> {info['translated_name']} (freq: {count})")
        
        print(f"📑 Frequency filtering complete: {entries_accepted}/{len(term_info_map)} terms accepted")
    
    # Ensure we have at least the header
    if len(csv_lines) == 0:
        if include_description:
            csv_lines.append("type,raw_name,translated_name,gender,description")
        elif include_gender_context:
            csv_lines.append("type,raw_name,translated_name,gender")
        else:
            csv_lines.append("type,raw_name,translated_name")
    
    # Print final summary
    print(f"📑 Processing complete: {entries_accepted} terms accepted")
    
    return csv_lines

def _deduplicate_glossary_with_fuzzy(csv_lines, fuzzy_threshold):
    """Apply advanced fuzzy matching to remove duplicate entries from the glossary with stop flag checks
    
    Uses a 2-pass approach:
    Pass 1: Remove entries with similar raw names (existing logic)
    Pass 2: Remove entries with identical translated names (new logic)
    """
    from difflib import SequenceMatcher
    
    # Try to import advanced libraries
    try:
        from rapidfuzz import fuzz as rfuzz
        use_rapidfuzz = True
    except ImportError:
        use_rapidfuzz = False
    
    try:
        import jellyfish
        use_jellyfish = True
    except ImportError:
        use_jellyfish = False
    
    algo_info = []
    if use_rapidfuzz:
        algo_info.append("RapidFuzz")
    if use_jellyfish:
        algo_info.append("Jaro-Winkler")
    if not algo_info:
        algo_info.append("difflib")
    
    # Check if translated name deduplication is enabled
    # GLOSSARY_DEDUPE_TRANSLATIONS: "1" = enable Pass 2 (remove entries with identical translations)
    #                              : "0" = disable Pass 2 (only remove entries with similar raw names)
    dedupe_translations = os.getenv("GLOSSARY_DEDUPE_TRANSLATIONS", "1") == "1"
    
    print(f"📋 Applying 2-pass fuzzy deduplication (threshold: {fuzzy_threshold})...")
    print(f"📋 Pass 1: Raw name deduplication (fuzzy matching)")
    if dedupe_translations:
        print(f"📋 Pass 2: Translated name deduplication (exact matching)")
    else:
        print(f"📋 Pass 2: DISABLED (GLOSSARY_DEDUPE_TRANSLATIONS=0)")
    print(f"📋 Using algorithms: {', '.join(algo_info)}")
    
    # Check stop flag at start
    if is_stop_requested():
        print(f"📑 ❌ Deduplication stopped by user")
        return csv_lines
    
    header_line = csv_lines[0]  # Keep header
    entry_lines = csv_lines[1:]  # Data lines
    original_count = len(entry_lines)
    
    print(f"📑 Starting deduplication with {original_count} entries...")
    
    # PASS 1: Raw name deduplication (existing fuzzy matching logic)
    print(f"📑 🔄 PASS 1: Raw name deduplication...")
    pass1_results = _deduplicate_pass1_raw_names(
        entry_lines, fuzzy_threshold, use_rapidfuzz, use_jellyfish
    )
    
    pass1_count = len(pass1_results)
    pass1_removed = original_count - pass1_count
    print(f"📑 ✅ PASS 1 complete: {pass1_removed} duplicates removed ({pass1_count} remaining)")
    
    # PASS 2: Translated name deduplication (if enabled)
    if dedupe_translations:
        print(f"📑 🔄 PASS 2: Translated name deduplication...")
        final_results = _deduplicate_pass2_translated_names(pass1_results)
        pass2_removed = pass1_count - len(final_results)
        print(f"📑 ✅ PASS 2 complete: {pass2_removed} duplicates removed ({len(final_results)} remaining)")
        total_removed = pass1_removed + pass2_removed
    else:
        final_results = pass1_results
        total_removed = pass1_removed
        print(f"📑 ⏭️ PASS 2 skipped (translation deduplication disabled)")
    
    # Rebuild CSV with header
    deduplicated = [header_line] + final_results
    
    print(f"📑 ✅ Total deduplication complete: {total_removed} duplicates removed")
    print(f"📑 Final glossary size: {len(final_results)} unique entries")
    
    return deduplicated


def _deduplicate_pass1_raw_names(entry_lines, fuzzy_threshold, use_rapidfuzz, use_jellyfish):
    """Pass 1: Remove entries with similar raw names using fuzzy matching"""
    from difflib import SequenceMatcher
    
    if use_rapidfuzz:
        from rapidfuzz import fuzz as rfuzz
    
    if use_jellyfish:
        import jellyfish
    
    deduplicated = []
    seen_entries = {}  # raw_name -> (entry_type, translated_name)
    seen_names_lower = set()  # Quick exact match check
    removed_count = 0
    total_entries = len(entry_lines)
    
    for idx, line in enumerate(entry_lines):
        # Check stop flag every 100 entries
        if idx > 0 and idx % 100 == 0:
            if is_stop_requested():
                print(f"📑 ❌ Pass 1 stopped at entry {idx}/{total_entries}")
                break
        
        # Show progress for large glossaries
        if total_entries > 500 and idx % 200 == 0:
            progress = (idx / total_entries) * 100
            print(f"📑 Pass 1 progress: {progress:.1f}% ({idx}/{total_entries})")
        
        if not line.strip():
            continue
            
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 3:
            continue
            
        entry_type = parts[0]
        raw_name = parts[1]
        translated_name = parts[2]
        raw_name_lower = raw_name.lower()
        
        # Fast exact duplicate check first
        if raw_name_lower in seen_names_lower:
            removed_count += 1
            if removed_count <= 10:  # Only log first few
                print(f"📋   Pass 1: Removing exact duplicate: '{raw_name}'")
            continue
        
        # For fuzzy matching, only check if threshold is less than 1.0
        is_duplicate = False
        if fuzzy_threshold < 1.0:
            # Use a more efficient approach: only check similar length strings
            name_len = len(raw_name)
            min_len = int(name_len * 0.7)
            max_len = int(name_len * 1.3)
            
            # Only compare with entries of similar length
            candidates = []
            for seen_name, (seen_type, seen_trans) in seen_entries.items():
                if min_len <= len(seen_name) <= max_len:
                    candidates.append(seen_name)
            
            # Check fuzzy similarity with candidates using multiple algorithms
            for seen_name in candidates:
                # Quick character overlap check before expensive comparison
                char_overlap = len(set(raw_name_lower) & set(seen_name.lower()))
                if char_overlap < len(raw_name_lower) * 0.5:
                    continue  # Too different, skip
                
                # Try multiple algorithms and take the best score
                scores = []
                
                if use_rapidfuzz:
                    # RapidFuzz basic ratio
                    scores.append(rfuzz.ratio(raw_name_lower, seen_name.lower()) / 100.0)
                    # Token sort (handles word order)
                    try:
                        scores.append(rfuzz.token_sort_ratio(raw_name_lower, seen_name.lower()) / 100.0)
                    except:
                        pass
                    # Partial ratio (substring)
                    try:
                        scores.append(rfuzz.partial_ratio(raw_name_lower, seen_name.lower()) / 100.0)
                    except:
                        pass
                else:
                    # Fallback to difflib
                    scores.append(SequenceMatcher(None, raw_name_lower, seen_name.lower()).ratio())
                
                # Try Jaro-Winkler (better for names)
                if use_jellyfish:
                    try:
                        jaro = jellyfish.jaro_winkler_similarity(raw_name, seen_name)
                        scores.append(jaro)
                    except:
                        pass
                
                # Take best score
                best_similarity = max(scores) if scores else 0.0
                
                if best_similarity >= fuzzy_threshold:
                    if removed_count < 10:  # Only log first few
                        print(f"📋   Pass 1: Removing fuzzy duplicate: '{raw_name}' ~= '{seen_name}' (score: {best_similarity:.2%})")
                    removed_count += 1
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            seen_entries[raw_name] = (entry_type, translated_name)
            seen_names_lower.add(raw_name_lower)
            deduplicated.append(line)
    
    return deduplicated


def _deduplicate_pass2_translated_names(entry_lines):
    """Pass 2: Remove entries with identical translated names"""
    deduplicated = []
    seen_translations = {}  # translated_name.lower() -> (raw_name, line)
    removed_count = 0
    
    for line in entry_lines:
        if not line.strip():
            continue
            
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 3:
            continue
            
        entry_type = parts[0]
        raw_name = parts[1]
        translated_name = parts[2]
        translated_lower = translated_name.lower().strip()
        
        # Skip empty translations
        if not translated_lower:
            deduplicated.append(line)
            continue
        
        # Check if we've seen this translation before
        if translated_lower in seen_translations:
            existing_raw, existing_line = seen_translations[translated_lower]
            # Get the existing translated name from the line
            existing_parts = existing_line.split(',')
            existing_translated = existing_parts[2] if len(existing_parts) >= 3 else translated_name
            
            # Count fields in both entries (more fields = higher priority)
            current_field_count = len([f.strip() for f in parts if f.strip()])
            existing_field_count = len([f.strip() for f in existing_parts if f.strip()])
            
            # If current entry has more fields, replace the existing one
            if current_field_count > existing_field_count:
                # Remove existing entry from deduplicated list
                deduplicated = [l for l in deduplicated if l != existing_line]
                # Replace with current entry
                seen_translations[translated_lower] = (raw_name, line)
                deduplicated.append(line)
                removed_count += 1
                if removed_count <= 10:  # Only log first few
                    print(f"📋   Pass 2: Replacing '{existing_raw}' -> '{existing_translated}' ({existing_field_count} fields) with '{raw_name}' -> '{translated_name}' ({current_field_count} fields) - more detailed entry")
            else:
                # Keep existing entry (has same or more fields)
                removed_count += 1
                if removed_count <= 10:  # Only log first few
                    extra_info = f" ({current_field_count} vs {existing_field_count} fields)" if current_field_count != existing_field_count else ""
                    print(f"📋   Pass 2: Removing '{raw_name}' -> '{translated_name}' (duplicate translation of '{existing_raw}' -> '{existing_translated}'){extra_info}")
        else:
            # New translation, keep it
            seen_translations[translated_lower] = (raw_name, line)
            deduplicated.append(line)
    
    return deduplicated

def _merge_csv_entries(new_csv_lines, existing_glossary, strip_honorifics, language):
    """Merge CSV entries with existing glossary with stop flag checks"""
    
    # Check stop flag at start
    if is_stop_requested():
        print(f"📑 ❌ Glossary merge stopped by user")
        return new_csv_lines
    
    # Parse existing glossary
    existing_lines = []
    existing_names = set()
    
    if isinstance(existing_glossary, str):
        # Already CSV format
        lines = existing_glossary.strip().split('\n')
        total_lines = len(lines)
        
        for idx, line in enumerate(lines):
            # Check stop flag every 50 lines
            if idx > 0 and idx % 50 == 0:
                if is_stop_requested():
                    print(f"📑 ❌ Merge stopped while processing existing glossary at line {idx}/{total_lines}")
                    return new_csv_lines
                
                if total_lines > 200:
                    progress = (idx / total_lines) * 100
                    print(f"📑 Processing existing glossary: {progress:.1f}%")
            
            if 'type,raw_name' in line.lower():
                continue  # Skip header
            
            line_stripped = line.strip()
            # Skip token-efficient lines and section/bullet markers
            if not line_stripped or line_stripped.startswith('===') or line_stripped.startswith('*') or line_stripped.lower().startswith('glossary:'):
                continue
            
            parts = [p.strip() for p in line.split(',')]
            # Require at least 3 fields (type, raw_name, translated_name)
            if len(parts) < 3:
                continue
            
            entry_type = parts[0].strip().lower()
            # Only accept reasonable type tokens (letters/underscores only)
            import re as _re
            if not _re.match(r'^[a-z_]+$', entry_type):
                continue
            
            raw_name = parts[1]
            if strip_honorifics:
                raw_name = _strip_honorific(raw_name, language)
                parts[1] = raw_name
            if raw_name not in existing_names:
                existing_lines.append(','.join(parts))
                existing_names.add(raw_name)
    
    # Check stop flag before processing new names
    if is_stop_requested():
        print(f"📑 ❌ Merge stopped before processing new entries")
        return new_csv_lines
    
    # Get new names
    new_names = set()
    final_lines = []
    
    for idx, line in enumerate(new_csv_lines):
        # Check stop flag every 50 lines
        if idx > 0 and idx % 50 == 0:
            if is_stop_requested():
                print(f"📑 ❌ Merge stopped while processing new entries at line {idx}")
                return final_lines if final_lines else new_csv_lines
        
        if 'type,raw_name' in line.lower():
            final_lines.append(line)  # Keep header
            continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 2:
            new_names.add(parts[1])
            final_lines.append(line)
    
    # Check stop flag before adding existing entries
    if is_stop_requested():
        print(f"📑 ❌ Merge stopped before combining entries")
        return final_lines
    
    # Add non-duplicate existing entries
    added_count = 0
    for idx, line in enumerate(existing_lines):
        # Check stop flag every 50 additions
        if idx > 0 and idx % 50 == 0:
            if is_stop_requested():
                print(f"📑 ❌ Merge stopped while adding existing entries ({added_count} added)")
                return final_lines
        
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 2 and parts[1] not in new_names:
            final_lines.append(line)
            added_count += 1
    
    print(f"📑 Merged {added_count} entries from existing glossary")
    return final_lines

def _extract_with_patterns(all_text, language, min_frequency, 
                          max_names, max_titles, batch_size, 
                          existing_glossary, output_dir, 
                          strip_honorifics=True, fuzzy_threshold=0.90, filter_mode='all'):
    """Extract glossary using pattern matching with true CSV format output and stop flag checks"""
    print("📑 Using pattern-based extraction")
    
    # Check stop flag at start
    if is_stop_requested():
        print("📑 ❌ Pattern-based extraction stopped by user")
        return {}
    
    def is_valid_name(name, language_hint='unknown'):
        """Strict validation for proper names only"""
        if not name or len(name.strip()) < 1:
            return False
            
        name = name.strip()
        
        if name.lower() in PM.COMMON_WORDS or name in PM.COMMON_WORDS:
            return False
        
        if language_hint == 'korean':
            if not (2 <= len(name) <= 4):
                return False
            if not all(0xAC00 <= ord(char) <= 0xD7AF for char in name):
                return False
            if len(set(name)) == 1:
                return False
                
        elif language_hint == 'japanese':
            if not (2 <= len(name) <= 6):
                return False
            has_kanji = any(0x4E00 <= ord(char) <= 0x9FFF for char in name)
            has_kana = any((0x3040 <= ord(char) <= 0x309F) or (0x30A0 <= ord(char) <= 0x30FF) for char in name)
            if not (has_kanji or has_kana):
                return False
                
        elif language_hint == 'chinese':
            if not (2 <= len(name) <= 4):
                return False
            if not all(0x4E00 <= ord(char) <= 0x9FFF for char in name):
                return False
                
        elif language_hint == 'english':
            if not name[0].isupper():
                return False
            if sum(1 for c in name if c.isalpha()) < len(name) * 0.8:
                return False
            if not (2 <= len(name) <= 20):
                return False
        
        return True
    
    def detect_language_hint(text_sample):
        """Quick language detection for validation purposes"""
        sample = text_sample[:1000]
        
        korean_chars = sum(1 for char in sample if 0xAC00 <= ord(char) <= 0xD7AF)
        japanese_kana = sum(1 for char in sample if (0x3040 <= ord(char) <= 0x309F) or (0x30A0 <= ord(char) <= 0x30FF))
        chinese_chars = sum(1 for char in sample if 0x4E00 <= ord(char) <= 0x9FFF)
        latin_chars = sum(1 for char in sample if 0x0041 <= ord(char) <= 0x007A)
        
        if korean_chars > 50:
            return 'korean'
        elif japanese_kana > 20:
            return 'japanese'
        elif chinese_chars > 50 and japanese_kana < 10:
            return 'chinese'
        elif latin_chars > 100:
            return 'english'
        else:
            return 'unknown'
    
    language_hint = detect_language_hint(all_text)
    print(f"📑 Detected primary language: {language_hint}")
    
    # Check stop flag after language detection
    if is_stop_requested():
        print("📑 ❌ Extraction stopped after language detection")
        return {}
    
    honorifics_to_use = []
    if language_hint in PM.CJK_HONORIFICS:
        honorifics_to_use.extend(PM.CJK_HONORIFICS[language_hint])
    honorifics_to_use.extend(PM.CJK_HONORIFICS.get('english', []))
    
    print(f"📑 Using {len(honorifics_to_use)} honorifics for {language_hint}")
    
    names_with_honorifics = {}
    standalone_names = {}
    
    # Check if parallel processing is enabled
    extraction_workers = int(os.getenv("EXTRACTION_WORKERS", "1"))
    
    # PARALLEL HONORIFIC PROCESSING
    if extraction_workers > 1 and len(honorifics_to_use) > 3:
        print(f"📑 Scanning for names with honorifics (parallel with {extraction_workers} workers)...")
        
        # Create a wrapper function that can be called in parallel
        def process_honorific(args):
            """Process a single honorific in a worker thread"""
            honorific, idx, total = args
            
            # Check stop flag
            if is_stop_requested():
                return None, None
            
            print(f"📑 Worker processing honorific {idx}/{total}: '{honorific}'")
            
            # Local dictionaries for this worker
            local_names_with = {}
            local_standalone = {}
            
            # Call the extraction method
            _extract_names_for_honorific(
                honorific, all_text, language_hint,
                min_frequency, local_names_with,
                local_standalone, is_valid_name, fuzzy_threshold
            )
            
            return local_names_with, local_standalone
        
        # Prepare arguments for parallel processing
        honorific_args = [
            (honorific, idx + 1, len(honorifics_to_use))
            for idx, honorific in enumerate(honorifics_to_use)
        ]
        
        # Process honorifics in parallel
        with ThreadPoolExecutor(max_workers=min(extraction_workers, len(honorifics_to_use))) as executor:
            futures = []
            
            for args in honorific_args:
                if is_stop_requested():
                    executor.shutdown(wait=False)
                    return {}
                
                future = executor.submit(process_honorific, args)
                futures.append(future)
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(futures):
                if is_stop_requested():
                    executor.shutdown(wait=False)
                    return {}
                
                try:
                    result = future.result()
                    if result and result[0] is not None:
                        local_names_with, local_standalone = result
                        
                        # Merge results (thread-safe since we're in main thread)
                        for name, count in local_names_with.items():
                            if name not in names_with_honorifics:
                                names_with_honorifics[name] = count
                            else:
                                names_with_honorifics[name] = max(names_with_honorifics[name], count)
                        
                        for name, count in local_standalone.items():
                            if name not in standalone_names:
                                standalone_names[name] = count
                            else:
                                standalone_names[name] = max(standalone_names[name], count)
                    
                    completed += 1
                    if completed % 5 == 0 or completed == len(honorifics_to_use):
                        print(f"📑 Honorific processing: {completed}/{len(honorifics_to_use)} completed")
                        
                except Exception as e:
                    print(f"⚠️ Failed to process honorific: {e}")
                    completed += 1
        
        print(f"📑 Parallel honorific processing completed: found {len(names_with_honorifics)} names")
        
    else:
        # SEQUENTIAL PROCESSING (fallback)
        print("📑 Scanning for names with honorifics...")
        
        # Extract names with honorifics
        total_honorifics = len(honorifics_to_use)
        for idx, honorific in enumerate(honorifics_to_use):
            # Check stop flag before each honorific
            if is_stop_requested():
                print(f"📑 ❌ Extraction stopped at honorific {idx}/{total_honorifics}")
                return {}
            
            print(f"📑 Processing honorific {idx + 1}/{total_honorifics}: '{honorific}'")
            
            _extract_names_for_honorific(honorific, all_text, language_hint, 
                                            min_frequency, names_with_honorifics, 
                                            standalone_names, is_valid_name, fuzzy_threshold)
    
    # Check stop flag before processing terms
    if is_stop_requested():
        print("📑 ❌ Extraction stopped before processing terms")
        return {}
    
    # Apply filter mode
    filtered_names = {}
    if filter_mode == 'only_with_honorifics':
        # Only keep names that have honorifics (no standalone names)
        filtered_names = names_with_honorifics.copy()
        print(f"📑 Filter: Keeping only names with honorifics ({len(filtered_names)} names)")
    elif filter_mode == 'only_without_honorifics':
        # Keep standalone names that were NOT found with honorifics
        for name, count in standalone_names.items():
            # Check if this name also appears with honorifics
            appears_with_honorific = False
            for honorific_name in names_with_honorifics.keys():
                if _strip_honorific(honorific_name, language_hint) == name:
                    appears_with_honorific = True
                    break
            
            # Only add if it doesn't appear with honorifics
            if not appears_with_honorific:
                filtered_names[name] = count
        
        print(f"📑 Filter: Keeping only names without honorifics ({len(filtered_names)} names)")
    else:  # 'all' mode
        # Keep all names (both with and without honorifics)
        filtered_names = names_with_honorifics.copy()
        # Also add standalone names
        for name, count in standalone_names.items():
            if name not in filtered_names and not any(
                _strip_honorific(n, language_hint) == name for n in filtered_names.keys()
            ):
                filtered_names[name] = count
        print(f"📑 Filter: Keeping all names ({len(filtered_names)} names)")
    
    # Process extracted terms
    final_terms = {}
    
    term_count = 0
    total_terms = len(filtered_names)
    for term, count in filtered_names.items():
        term_count += 1
        
        # Check stop flag every 20 terms
        if term_count % 20 == 0:
            if is_stop_requested():
                print(f"📑 ❌ Term processing stopped at {term_count}/{total_terms}")
                return {}
        
        if strip_honorifics:
            clean_term = _strip_honorific(term, language_hint)
            if clean_term in final_terms:
                final_terms[clean_term] = final_terms[clean_term] + count
            else:
                final_terms[clean_term] = count
        else:
            final_terms[term] = count
    
    # Check stop flag before finding titles
    if is_stop_requested():
        print("📑 ❌ Extraction stopped before finding titles")
        return {}
    
    # Find titles (but respect filter mode)
    print("📑 Scanning for titles...")
    found_titles = {}
    
    # Extract titles for all modes EXCEPT "only_with_honorifics"
    # (titles are included in "only_without_honorifics" since titles typically don't have honorifics)
    if filter_mode != 'only_with_honorifics':
        title_patterns_to_use = []
        if language_hint in PM.TITLE_PATTERNS:
            title_patterns_to_use.extend(PM.TITLE_PATTERNS[language_hint])
        title_patterns_to_use.extend(PM.TITLE_PATTERNS.get('english', []))
        
        total_patterns = len(title_patterns_to_use)
        for pattern_idx, pattern in enumerate(title_patterns_to_use):
            # Check stop flag before each pattern
            if is_stop_requested():
                print(f"📑 ❌ Title extraction stopped at pattern {pattern_idx}/{total_patterns}")
                return {}
            
            print(f"📑 Processing title pattern {pattern_idx + 1}/{total_patterns}")
            
            matches = list(re.finditer(pattern, all_text, re.IGNORECASE if 'english' in pattern else 0))
            
            for match_idx, match in enumerate(matches):
                # Check stop flag every 50 matches
                if match_idx > 0 and match_idx % 50 == 0:
                    if is_stop_requested():
                        print(f"📑 ❌ Title extraction stopped at match {match_idx}")
                        return {}
                
                title = match.group(0)
                
                # Skip if this title is already in names
                if title in filtered_names or title in names_with_honorifics:
                    continue
                    
                count = _find_fuzzy_matches(title, all_text, fuzzy_threshold)
                
                # Check if stopped during fuzzy matching
                if is_stop_requested():
                    print(f"📑 ❌ Title extraction stopped during fuzzy matching")
                    return {}
                
                if count >= min_frequency:
                    if re.match(r'[A-Za-z]', title):
                        title = title.title()
                    
                    if strip_honorifics:
                        title = _strip_honorific(title, language_hint)
                    
                    if title not in found_titles:
                        found_titles[title] = count
        
        if filter_mode == 'only_without_honorifics':
            print(f"📑 Found {len(found_titles)} titles (included in 'without honorifics' mode)")
        else:
            print(f"📑 Found {len(found_titles)} unique titles")
    else:
        print(f"📑 Skipping title extraction (filter mode: only_with_honorifics)")
    
    # Check stop flag before sorting and translation
    if is_stop_requested():
        print("📑 ❌ Extraction stopped before sorting terms")
        return {}
    
    # Combine and sort
    sorted_names = sorted(final_terms.items(), key=lambda x: x[1], reverse=True)[:max_names]
    sorted_titles = sorted(found_titles.items(), key=lambda x: x[1], reverse=True)[:max_titles]
    
    all_terms = []
    for name, count in sorted_names:
        all_terms.append(name)
    for title, count in sorted_titles:
        all_terms.append(title)
    
    print(f"📑 Total terms to translate: {len(all_terms)}")
    
    # Check stop flag before translation
    if is_stop_requested():
        print("📑 ❌ Extraction stopped before translation")
        return {}
    
    # Translate terms
    if os.getenv("DISABLE_GLOSSARY_TRANSLATION", "0") == "1":
        print("📑 Translation disabled - keeping original terms")
        translations = {term: term for term in all_terms}
    else:
        print(f"📑 Translating {len(all_terms)} terms...")
        translations = _translate_terms_batch(all_terms, language_hint, batch_size, output_dir)
    
    # Check if translation was stopped
    if is_stop_requested():
        print("📑 ❌ Extraction stopped after translation")
        return translations  # Return partial results
    
    # Build CSV lines
    csv_lines = ["type,raw_name,translated_name"]
    
    for name, _ in sorted_names:
        if name in translations:
            csv_lines.append(f"character,{name},{translations[name]}")
    
    for title, _ in sorted_titles:
        if title in translations:
            csv_lines.append(f"term,{title},{translations[title]}")
    
    # Check stop flag before merging
    if is_stop_requested():
        print("📑 ❌ Extraction stopped before merging with existing glossary")
        # Still save what we have
        csv_content = '\n'.join(csv_lines)
        glossary_path = os.path.join(output_dir, "glossary.json")
        _atomic_write_file(glossary_path, csv_content)
        return _parse_csv_to_dict(csv_content)
    
    # Merge with existing glossary
    if existing_glossary:
        csv_lines = _merge_csv_entries(csv_lines, existing_glossary, strip_honorifics, language_hint)
    
    # Check stop flag before deduplication
    if is_stop_requested():
        print("📑 ❌ Extraction stopped before deduplication")
        csv_content = '\n'.join(csv_lines)
        glossary_path = os.path.join(output_dir, "glossary.json")
        _atomic_write_file(glossary_path, csv_content)
        return _parse_csv_to_dict(csv_content)
    
    # Fuzzy matching deduplication
    csv_lines = _deduplicate_glossary_with_fuzzy(csv_lines, fuzzy_threshold)
    
    # Create CSV content
    csv_content = '\n'.join(csv_lines)
    # Save glossary as CSV
    glossary_path = os.path.join(output_dir, "glossary.csv")
    _atomic_write_file(glossary_path, csv_content)
    
    print(f"\n📑 ✅ TARGETED GLOSSARY SAVED!")
    print(f"📑 File: {glossary_path}")
    print(f"📑 Total entries: {len(csv_lines) - 1}")  # Exclude header
    
    return _parse_csv_to_dict(csv_content)

def _translate_terms_batch(term_list, profile_name, batch_size=50, output_dir=None, log_callback=None):
    """Use fully configurable prompts for translation with interrupt support"""
    # Redirect stdout to GUI log if callback provided
    if log_callback:
        set_output_redirect(log_callback)
    
    if not term_list or os.getenv("DISABLE_GLOSSARY_TRANSLATION", "0") == "1":
        print(f"📑 Glossary translation disabled or no terms to translate")
        return {term: term for term in term_list}
    
    # Check stop flag
    if is_stop_requested():
        print("📑 ❌ Glossary translation stopped by user")
        return {term: term for term in term_list}
    
    try:
        MODEL = os.getenv("MODEL", "gemini-1.5-flash")
        API_KEY = (os.getenv("API_KEY") or 
                   os.getenv("OPENAI_API_KEY") or 
                   os.getenv("OPENAI_OR_Gemini_API_KEY") or
                   os.getenv("GEMINI_API_KEY"))

        if is_traditional_translation_api(MODEL):
            return
        
        if not API_KEY:
            print(f"📑 No API key found, skipping translation")
            return {term: term for term in term_list}
        
        print(f"📑 Translating {len(term_list)} {profile_name} terms to English using batch size {batch_size}...")
        
        from unified_api_client import UnifiedClient, UnifiedClientError
        client = UnifiedClient(model=MODEL, api_key=API_KEY, output_dir=output_dir)
        if hasattr(client, 'reset_cleanup_state'):
            client.reset_cleanup_state()
        
        # Get custom translation prompt from environment
        translation_prompt_template = os.getenv("GLOSSARY_TRANSLATION_PROMPT", "")
        
        if not translation_prompt_template:
            translation_prompt_template = """You are translating {language} character names and important terms to English.
For character names, provide English transliterations or keep as romanized.
Keep honorifics/suffixes only if they are integral to the name.
Respond with the same numbered format.

Terms to translate:
{terms_list}

Provide translations in the same numbered format."""
        
        all_translations = {}
        all_responses = []  # Collect raw responses
        chunk_timeout = int(os.getenv("CHUNK_TIMEOUT", "300"))  # 5 minute default
        
        for i in range(0, len(term_list), batch_size):
            # Check stop flag before each batch
            if is_stop_requested():
                print(f"📑 ❌ Translation stopped at batch {(i // batch_size) + 1}")
                # Return partial translations
                for term in term_list:
                    if term not in all_translations:
                        all_translations[term] = term
                return all_translations
            
            batch = term_list[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(term_list) + batch_size - 1) // batch_size
            
            print(f"📑 Processing batch {batch_num}/{total_batches} ({len(batch)} terms)...")
            
            # Format terms list
            terms_text = ""
            for idx, term in enumerate(batch, 1):
                terms_text += f"{idx}. {term}\n"
            
            # Replace placeholders in prompt
            prompt = translation_prompt_template.replace('{language}', profile_name)
            prompt = prompt.replace('{terms_list}', terms_text.strip())
            prompt = prompt.replace('{batch_size}', str(len(batch)))
            
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            try:
                temperature = float(os.getenv("TEMPERATURE", "0.3"))
                max_tokens = int(os.getenv("MAX_OUTPUT_TOKENS", "4096"))
                
                # Use send_with_interrupt for interruptible API call
                print(f"📑 Sending translation request for batch {batch_num} (interruptible)...")
                
                response = send_with_interrupt(
                    messages=messages,
                    client=client,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop_check_fn=is_stop_requested,
                    chunk_timeout=chunk_timeout
                )
                
                # Handle response properly
                if hasattr(response, 'content'):
                    response_text = response.content
                else:
                    response_text = str(response)
                
                # Store raw response with batch info
                all_responses.append((batch, response_text))
                
                print(f"📑 Batch {batch_num} completed - response received")
                
                # Small delay between batches to avoid rate limiting (configurable)
                if i + batch_size < len(term_list):
                    # Check stop before sleep
                    if is_stop_requested():
                        print(f"📑 ❌ Translation stopped after batch {batch_num}")
                        # Fill in missing translations
                        for term in term_list:
                            if term not in all_translations:
                                all_translations[term] = term
                        return all_translations
                    # Use configurable batch delay or default to 0.1s (much faster than 0.5s)
                    batch_delay = float(os.getenv("GLOSSARY_BATCH_DELAY", "0.001"))
                    if batch_delay > 0:
                        time.sleep(batch_delay)
                    
            except UnifiedClientError as e:
                if "stopped by user" in str(e).lower():
                    print(f"📑 ❌ Translation interrupted by user at batch {batch_num}")
                    # Fill in remaining terms with originals
                    for term in term_list:
                        if term not in all_translations:
                            all_translations[term] = term
                    return all_translations
                else:
                    print(f"⚠️ Translation failed for batch {batch_num}: {e}")
                    for term in batch:
                        all_translations[term] = term
            except Exception as e:
                print(f"⚠️ Translation failed for batch {batch_num}: {e}")
                for term in batch:
                    all_translations[term] = term
        
        # Parse all responses at the end
        print(f"📑 Parsing {len(all_responses)} batch responses...")
        for batch, response_text in all_responses:
            batch_translations = _parse_translation_response(response_text, batch)
            all_translations.update(batch_translations)
        
        # Ensure all terms have translations
        for term in term_list:
            if term not in all_translations:
                all_translations[term] = term
        
        translated_count = sum(1 for term, translation in all_translations.items() 
                             if translation != term and translation.strip())
        
        print(f"📑 Successfully translated {translated_count}/{len(term_list)} terms")
        return all_translations
        
    except Exception as e:
        print(f"⚠️ Glossary translation failed: {e}")
        return {term: term for term in term_list}


def _extract_names_for_honorific(honorific, all_text, language_hint, 
                                min_frequency, names_with_honorifics, 
                                standalone_names, is_valid_name, fuzzy_threshold=0.90):
    """Extract names for a specific honorific with fuzzy matching and stop flag checks"""
    
    # Check stop flag at start
    if is_stop_requested():
        print(f"📑 ❌ Name extraction for '{honorific}' stopped by user")
        return
    
    if language_hint == 'korean' and not honorific.startswith('-'):
        pattern = r'([\uac00-\ud7af]{2,4})(?=' + re.escape(honorific) + r'(?:\s|[,.\!?]|$))'
        
        matches = list(re.finditer(pattern, all_text))
        total_matches = len(matches)
        
        for idx, match in enumerate(matches):
            # Check stop flag every 50 matches
            if idx > 0 and idx % 50 == 0:
                if is_stop_requested():
                    print(f"📑 ❌ Korean name extraction stopped at {idx}/{total_matches}")
                    return
                
                # Show progress for large sets
                if total_matches > 500:
                    progress = (idx / total_matches) * 100
                    print(f"📑 Processing Korean names: {progress:.1f}% ({idx}/{total_matches})")
            
            potential_name = match.group(1)
            
            if is_valid_name(potential_name, 'korean'):
                full_form = potential_name + honorific
                
                # Use fuzzy matching for counting with stop check
                count = _find_fuzzy_matches(full_form, all_text, fuzzy_threshold)
                
                # Check if stopped during fuzzy matching
                if is_stop_requested():
                    print(f"📑 ❌ Name extraction stopped during fuzzy matching")
                    return
                
                if count >= min_frequency:
                    context_patterns = [
                        full_form + r'[은는이가]',
                        full_form + r'[을를]',
                        full_form + r'[에게한테]',
                        r'["]' + full_form,
                        full_form + r'[,]',
                    ]
                    
                    context_count = 0
                    for ctx_pattern in context_patterns:
                        context_count += len(re.findall(ctx_pattern, all_text))
                    
                    if context_count > 0:
                        names_with_honorifics[full_form] = count
                        standalone_names[potential_name] = count
                        
    elif language_hint == 'japanese' and not honorific.startswith('-'):
        pattern = r'([\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]{2,5})(?=' + re.escape(honorific) + r'(?:\s|[、。！？]|$))'
        
        matches = list(re.finditer(pattern, all_text))
        total_matches = len(matches)
        
        for idx, match in enumerate(matches):
            # Check stop flag every 50 matches
            if idx > 0 and idx % 50 == 0:
                if is_stop_requested():
                    print(f"📑 ❌ Japanese name extraction stopped at {idx}/{total_matches}")
                    return
                
                if total_matches > 500:
                    progress = (idx / total_matches) * 100
                    print(f"📑 Processing Japanese names: {progress:.1f}% ({idx}/{total_matches})")
            
            potential_name = match.group(1)
            
            if is_valid_name(potential_name, 'japanese'):
                full_form = potential_name + honorific
                count = _find_fuzzy_matches(full_form, all_text, fuzzy_threshold)
                
                if is_stop_requested():
                    print(f"📑 ❌ Name extraction stopped during fuzzy matching")
                    return
                
                if count >= min_frequency:
                    names_with_honorifics[full_form] = count
                    standalone_names[potential_name] = count
                        
    elif language_hint == 'chinese' and not honorific.startswith('-'):
        pattern = r'([\u4e00-\u9fff]{2,4})(?=' + re.escape(honorific) + r'(?:\s|[，。！？]|$))'
        
        matches = list(re.finditer(pattern, all_text))
        total_matches = len(matches)
        
        for idx, match in enumerate(matches):
            # Check stop flag every 50 matches
            if idx > 0 and idx % 50 == 0:
                if is_stop_requested():
                    print(f"📑 ❌ Chinese name extraction stopped at {idx}/{total_matches}")
                    return
                
                if total_matches > 500:
                    progress = (idx / total_matches) * 100
                    print(f"📑 Processing Chinese names: {progress:.1f}% ({idx}/{total_matches})")
            
            potential_name = match.group(1)
            
            if is_valid_name(potential_name, 'chinese'):
                full_form = potential_name + honorific
                count = _find_fuzzy_matches(full_form, all_text, fuzzy_threshold)
                
                if is_stop_requested():
                    print(f"📑 ❌ Name extraction stopped during fuzzy matching")
                    return
                
                if count >= min_frequency:
                    names_with_honorifics[full_form] = count
                    standalone_names[potential_name] = count
                        
    elif honorific.startswith('-') or honorific.startswith(' '):
        is_space_separated = honorific.startswith(' ')
        
        if is_space_separated:
            pattern_english = r'\b([A-Z][a-zA-Z]+)' + re.escape(honorific) + r'(?=\s|[,.\!?]|$)'
        else:
            pattern_english = r'\b([A-Z][a-zA-Z]+)' + re.escape(honorific) + r'\b'
        
        matches = list(re.finditer(pattern_english, all_text))
        total_matches = len(matches)
        
        for idx, match in enumerate(matches):
            # Check stop flag every 50 matches
            if idx > 0 and idx % 50 == 0:
                if is_stop_requested():
                    print(f"📑 ❌ English name extraction stopped at {idx}/{total_matches}")
                    return
                
                if total_matches > 500:
                    progress = (idx / total_matches) * 100
                    print(f"📑 Processing English names: {progress:.1f}% ({idx}/{total_matches})")
            
            potential_name = match.group(1)
            
            if is_valid_name(potential_name, 'english'):
                full_form = potential_name + honorific
                count = _find_fuzzy_matches(full_form, all_text, fuzzy_threshold)
                
                if is_stop_requested():
                    print(f"📑 ❌ Name extraction stopped during fuzzy matching")
                    return
                
                if count >= min_frequency:
                    names_with_honorifics[full_form] = count
                    standalone_names[potential_name] = count

def _parse_translation_response(response, original_terms):
    """Extract translations from AI response by matching numbered lines to original terms"""
    translations = {}
    
    # Handle UnifiedResponse object
    if hasattr(response, 'content'):
        response_text = response.content
    else:
        response_text = str(response)
    
    # Split into lines
    lines = response_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Match numbered format: "1. Translation" or "1) Translation" etc
        number_match = re.match(r'^(\d+)[\.):\-\s]+(.+)', line)
        if number_match:
            idx = int(number_match.group(1)) - 1  # Convert to 0-based
            translation = number_match.group(2).strip()
            
            # Remove trailing explanations in parentheses
            translation = re.sub(r'\s*\([^)]+\)\s*$', '', translation)
            
            if 0 <= idx < len(original_terms):
                translations[original_terms[idx]] = translation
    
    print(f"📑 Extracted {len(translations)}/{len(original_terms)} translations")
    return translations
    
    
def _init_worker_with_env(env_vars_dict):
    """Initialize worker process with environment variables from parent.
    
    MUST be at module level for pickling by multiprocessing.Pool.
    """
    import os
    for k, v in env_vars_dict.items():
        os.environ[k] = str(v)

def _check_sentence_batch_for_terms(args):
    """Check a batch of sentences for term matches - used by ProcessPoolExecutor"""
    batch_sentences, terms = args
    filtered = []
    
    # Use pre-compiled term list for fast checking
    for sentence in batch_sentences:
        # Quick check using any() - stops at first match
        if any(term in sentence for term in terms):
            filtered.append(sentence)
    
    return filtered

def _process_sentence_batch_for_extraction(args):
    """Process sentences to extract terms - used by ProcessPoolExecutor"""
    batch_sentences, batch_idx, combined_pattern, exclude_check_data = args
    from collections import Counter
    import re
    
    local_word_freq = Counter()
    local_important = []
    local_seen = set()
    
    # Rebuild the exclusion check function from data
    honorifics_to_exclude, title_patterns_str, common_words, chinese_nums = exclude_check_data
    title_patterns = [re.compile(p) for p in title_patterns_str]
    
    def should_exclude_term(term):
        term_lower = term.lower()
        
        # Check if it's a common word
        if term in common_words or term_lower in common_words:
            return True
        
        # Check if it contains honorifics
        for honorific in honorifics_to_exclude:
            if honorific in term or (honorific.startswith('-') and term.endswith(honorific[1:])):
                return True
        
        # Check if it matches title patterns
        for pattern in title_patterns:
            if pattern.search(term):
                return True
        
        # Check if it's a number
        if term in chinese_nums or term.isdigit():
            return True
        
        return False
    
    for sentence in batch_sentences:
        sentence = sentence.strip()
        if len(sentence) < 10 or len(sentence) > 500:
            continue
            
        # Find all potential terms in this sentence
        matches = re.findall(combined_pattern, sentence)
        
        if matches:
            # Filter out excluded terms
            filtered_matches = []
            for match in matches:
                if not should_exclude_term(match):
                    local_word_freq[match] += 1
                    filtered_matches.append(match)
            
            # Keep sentences with valid potential terms
            if filtered_matches:
                sentence_key = ' '.join(sorted(filtered_matches))
                if sentence_key not in local_seen:
                    local_important.append(sentence)
                    local_seen.add(sentence_key)
    
    return local_word_freq, local_important, local_seen, batch_idx