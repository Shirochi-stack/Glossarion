import re
import os
from bs4 import BeautifulSoup
import tiktoken

class ChapterSplitter:
    """Split large chapters into smaller chunks while preserving structure"""
    
    def __init__(self, model_name="gpt-3.5-turbo", target_tokens=80000, compression_factor=1.0):
        """
        Initialize splitter with token counter
        target_tokens: Target size for each chunk (leaving room for system prompt & history)
        compression_factor: Expected compression ratio from source to target language (0.7-1.0)
        """
        try:
            self.enc = tiktoken.encoding_for_model(model_name)
        except:
            self.enc = tiktoken.get_encoding("cl100k_base")
        self.target_tokens = target_tokens
        self.compression_factor = compression_factor
    
    def count_tokens(self, text):
        """Count tokens in text"""
        try:
            return len(self.enc.encode(text))
        except:
            # Fallback estimation
            return len(text) // 4
    
    def split_chapter(self, chapter_html, max_tokens=None, filename=None):
        """
        Split a chapter into smaller chunks.
        Splits based on EITHER token limit OR line break count (whichever comes first).
        Args:
            chapter_html: The chapter content (HTML or plain text)
            max_tokens: Maximum tokens per chunk
            filename: Optional filename to help determine content type
        Returns: List of (chunk_html, chunk_index, total_chunks)
        """
        if max_tokens is None:
            max_tokens = self.target_tokens
        
        effective_max_tokens = max_tokens
        
        # Check for break split configuration (skip for PDF files)
        # Check both the filename parameter and if it looks like a path ending in .pdf
        # Also check if '.pdf' is anywhere in the filename (case-insensitive) to catch PDF chapters
        is_pdf_file = filename and (filename.lower().endswith('.pdf') or '.pdf' in filename.lower())
        break_split = os.getenv('BREAK_SPLIT_COUNT', '')
        max_elements = None
        if break_split and break_split.isdigit() and not is_pdf_file:
            max_elements = int(break_split)
            print(f"✅ Break split enabled: {max_elements} per chunk")
        
        # First check if splitting is needed
        total_tokens = self.count_tokens(chapter_html)
        
        if os.getenv('DEBUG_CHUNK_SPLITTING', '0') == '1':
            print(f"[CHUNK DEBUG] Total tokens: {total_tokens:,}")
            print(f"[CHUNK DEBUG] Effective max tokens: {effective_max_tokens:,}")
            print(f"[CHUNK DEBUG] Max elements per chunk: {max_elements if max_elements else 'None (token-only)'}")
            print(f"[CHUNK DEBUG] Needs split: {total_tokens > effective_max_tokens}")
        
        if total_tokens <= effective_max_tokens and max_elements is None:
            return [(chapter_html, 1, 1)]  # No split needed
        
        # Determine if content is plain text based on filename extension
        # Check this FIRST before any HTML parsing
        is_plain_text_file = False
        if filename:
            # Check if it's a known plain text extension
            is_plain_text_file = any(filename.lower().endswith(suffix) for suffix in ['.csv', '.json', '.txt'])
            if is_plain_text_file and max_elements and not is_pdf_file:
                print(f"📄 Detected plain text file format (forcing line-based splitting)")
        
        # If it's a plain text file, skip HTML parsing and go directly to line-based splitting
        if not is_plain_text_file:
            soup = BeautifulSoup(chapter_html, 'html.parser')
            
            if soup.body:
                elements = list(soup.body.children)
            else:
                elements = list(soup.children)
            
            # Check if we have actual HTML tags (not just text)
            # Count non-empty elements
            non_empty_elements = [elem for elem in elements if not (isinstance(elem, str) and elem.strip() == '')]
            has_html_tags = any(hasattr(elem, 'name') for elem in non_empty_elements)
        else:
            # For plain text files, set these to trigger line-based mode
            has_html_tags = False
            non_empty_elements = []
        
        # Force plain text mode for .csv, .json, .txt files OR if no HTML tags OR only 1 element
        if is_plain_text_file or not has_html_tags or len(non_empty_elements) <= 1:
            # Plain text mode - split by line count OR token limit
            lines = chapter_html.split('\n')
            if max_elements and not is_pdf_file:
                print(f"📝 Total lines in file: {len(lines):,}")
            
            # Calculate tokens for all lines first for balanced splitting
            line_tokens = [self.count_tokens(line) for line in lines]
            
            # Calculate how many chunks we need for balanced distribution
            if max_elements:
                # Element-based splitting
                num_chunks = (len(lines) + max_elements - 1) // max_elements
            else:
                # Token-based splitting - calculate optimal number of chunks
                num_chunks = (total_tokens + effective_max_tokens - 1) // effective_max_tokens
            
            if num_chunks == 1:
                return [(chapter_html, 1, 1)]
            
            # Balanced splitting: distribute lines evenly across chunks
            chunks = []
            target_tokens_per_chunk = total_tokens / num_chunks
            current_lines = []
            current_tokens = 0
            # Pre-compute marker indices for gender-context boundaries
            gender_footer_indices = {
                idx for idx, line in enumerate(lines)
                if line.strip().startswith("=== CONTEXT ") and "END ===" in line
            }
            # Build prefix sums for fast token range queries
            prefix_tokens = [0]
            for tok in line_tokens:
                prefix_tokens.append(prefix_tokens[-1] + tok)
            
            def tokens_between(start_idx, end_exclusive):
                return prefix_tokens[end_exclusive] - prefix_tokens[start_idx]
            
            chunk_start = 0
            
            for i, line in enumerate(lines):
                line_tok = line_tokens[i]
                
                # Add line to current chunk
                current_lines.append(line)
                current_tokens += line_tok
                
                # Check if we should end this chunk
                is_last_line = (i == len(lines) - 1)
                chunks_remaining = num_chunks - len(chunks)
                lines_remaining = len(lines) - i - 1
                
                # Split if:
                # 1. We've reached target tokens AND there are enough lines left for remaining chunks
                # 2. OR we've exceeded max tokens
                # 3. OR this is the last line
                should_split = False
                if current_tokens >= target_tokens_per_chunk and chunks_remaining > 1:
                    # Make sure there are enough lines remaining for the remaining chunks
                    if lines_remaining >= chunks_remaining - 1:
                        should_split = True
                elif current_tokens > effective_max_tokens:
                    should_split = True
                elif is_last_line:
                    should_split = True
                
                if should_split:
                    # Prefer to split near a gender footer that is closest to the token target
                    prev_footer = max((idx for idx in gender_footer_indices if chunk_start <= idx <= i), default=None)
                    next_footer = min((idx for idx in gender_footer_indices if idx > i), default=None)
                    
                    # Token counts if we snap to footer
                    def tokens_to(idx_inclusive):
                        return tokens_between(chunk_start, idx_inclusive + 1)
                    
                    # Candidate selection: pick footer with tokens closest to target, within 15% of hard cap if forward
                    best_split_idx = None
                    best_diff = None
                    
                    target = target_tokens_per_chunk
                    
                    if prev_footer is not None:
                        t_prev = tokens_to(prev_footer)
                        diff = abs(t_prev - target)
                        best_split_idx, best_diff = prev_footer + 1, diff
                    
                    if next_footer is not None:
                        t_next = tokens_to(next_footer)
                        # Allow slight overflow beyond effective_max_tokens
                        if t_next <= effective_max_tokens * 1.15:
                            diff = abs(t_next - target)
                            if best_diff is None or diff < best_diff:
                                best_split_idx, best_diff = next_footer + 1, diff
                    
                    if best_split_idx is None:
                        best_split_idx = i + 1  # fallback to current boundary
                    
                    chunk_lines = lines[chunk_start:best_split_idx]
                    chunks.append('\n'.join(chunk_lines))
                    
                    # Prepare next chunk state
                    chunk_start = best_split_idx
                    current_lines = lines[chunk_start:i + 1]
                    current_tokens = tokens_between(chunk_start, i + 1)
            
            # Safety: add any remaining lines
            if current_lines:
                chunks.append('\n'.join(current_lines))
            
            if not chunks:
                chunks = [chapter_html]
            
            total_chunks = len(chunks)
            return [(chunk, i+1, total_chunks) for i, chunk in enumerate(chunks)]
        
        # HTML mode - balanced split by element tokens (with optional element cap)
        # Count total elements first if Break Split is enabled (skip for PDFs)
        if max_elements and not is_pdf_file:
            total_elements = sum(1 for elem in elements if not (isinstance(elem, str) and elem.strip() == ''))
            print(f"🏷️ Total HTML elements in file: {total_elements:,}")
        
        # Pre-compute token counts for non-empty elements
        elem_html_list = []
        elem_tokens_list = []
        for element in elements:
            if isinstance(element, str) and element.strip() == '':
                continue
            element_html = str(element)
            tokens = self.count_tokens(element_html)
            elem_html_list.append(element_html)
            elem_tokens_list.append(tokens)
        
        if not elem_html_list:
            return [(chapter_html, 1, 1)]
        
        total_elem_tokens = sum(elem_tokens_list)
        # Determine number of chunks
        if max_elements:
            num_chunks = (len(elem_html_list) + max_elements - 1) // max_elements
        else:
            num_chunks = (total_elem_tokens + effective_max_tokens - 1) // effective_max_tokens
        num_chunks = max(1, num_chunks)
        if num_chunks == 1:
            return [(chapter_html, 1, 1)]
        
        target_tokens_per_chunk = total_elem_tokens / num_chunks
        chunks = []
        current_chunk_elements = []
        current_chunk_tokens = 0
        
        for i, element_html in enumerate(elem_html_list):
            element_tokens = elem_tokens_list[i]
            
            # Oversized single element: make it a standalone chunk
            if element_tokens > effective_max_tokens:
                if current_chunk_elements:
                    chunks.append(self._create_chunk_html(current_chunk_elements))
                    current_chunk_elements = []
                    current_chunk_tokens = 0
                chunks.append(element_html)
                continue
            
            current_chunk_elements.append(element_html)
            current_chunk_tokens += element_tokens
            
            is_last = (i == len(elem_html_list) - 1)
            chunks_remaining = num_chunks - len(chunks)
            elems_remaining = len(elem_html_list) - i - 1
            
            should_split = False
            # Prefer splitting when we reach target and enough elements remain
            if current_chunk_tokens >= target_tokens_per_chunk and chunks_remaining > 1:
                if elems_remaining >= (chunks_remaining - 1):
                    should_split = True
            # Hard cap by tokens or element count
            if current_chunk_tokens > effective_max_tokens:
                should_split = True
            if max_elements and len(current_chunk_elements) >= max_elements:
                should_split = True
            if is_last:
                should_split = True
            
            if should_split:
                chunks.append(self._create_chunk_html(current_chunk_elements))
                current_chunk_elements = []
                current_chunk_tokens = 0
        
        if current_chunk_elements:
            chunks.append(self._create_chunk_html(current_chunk_elements))
        
        if not chunks:
            chunks = [chapter_html]
        
        if os.getenv('DEBUG_CHUNK_SPLITTING', '0') == '1':
            print(f"[CHUNK DEBUG] Created {len(chunks)} chunks")
            for idx, chunk in enumerate(chunks, 1):
                chunk_tokens = self.count_tokens(chunk)
                print(f"[CHUNK DEBUG]   Chunk {idx}: {chunk_tokens:,} tokens ({len(chunk):,} chars)")
        
        total_chunks = len(chunks)
        return [(chunk, i+1, total_chunks) for i, chunk in enumerate(chunks)]
    
    def _split_large_element(self, element, max_tokens):
        """Split a single large element (like a long paragraph)"""
        chunks = []
        
        if element.name == 'p' or not hasattr(element, 'children'):
            # For paragraphs or text elements, split by sentences
            text = element.get_text()
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            current_chunk = []
            current_tokens = 0
            
            for sentence in sentences:
                sentence_tokens = self.count_tokens(sentence)
                
                if current_tokens + sentence_tokens > max_tokens * 0.8 and current_chunk:
                    # Create paragraph with current sentences
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(f"<p>{chunk_text}</p>")
                    current_chunk = [sentence]
                    current_tokens = sentence_tokens
                else:
                    current_chunk.append(sentence)
                    current_tokens += sentence_tokens
            
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(f"<p>{chunk_text}</p>")
                
        else:
            # For other elements, try to split by children
            children = list(element.children)
            current_chunk = []
            current_tokens = 0
            
            for child in children:
                child_html = str(child)
                child_tokens = self.count_tokens(child_html)
                
                if current_tokens + child_tokens > max_tokens * 0.8 and current_chunk:
                    # Wrap in parent element type
                    wrapper = BeautifulSoup(f"<{element.name}></{element.name}>", 'html.parser')
                    wrapper_elem = wrapper.find(element.name)
                    for item in current_chunk:
                        wrapper_elem.append(BeautifulSoup(item, 'html.parser'))
                    chunks.append(str(wrapper))
                    
                    current_chunk = [child_html]
                    current_tokens = child_tokens
                else:
                    current_chunk.append(child_html)
                    current_tokens += child_tokens
            
            if current_chunk:
                wrapper = BeautifulSoup(f"<{element.name}></{element.name}>", 'html.parser')
                wrapper_elem = wrapper.find(element.name)
                for item in current_chunk:
                    wrapper_elem.append(BeautifulSoup(item, 'html.parser'))
                chunks.append(str(wrapper))
        
        return chunks
    
    def _create_chunk_html(self, elements):
        """Create a valid HTML chunk from list of elements"""
        # Join elements and wrap in basic HTML structure if needed
        content = '\n'.join(elements)
        
        # Check if it already has body tags
        if '<body' not in content.lower():
            # Just return the content, let the translation handle it
            return content
        else:
            return content
    
    def merge_translated_chunks(self, translated_chunks):
        """
        Merge translated chunks back together
        translated_chunks: List of (translated_html, chunk_index, total_chunks)
        """
        # Sort by chunk index to ensure correct order
        sorted_chunks = sorted(translated_chunks, key=lambda x: x[1])
        
        # Extract just the HTML content
        html_parts = [chunk[0] for chunk in sorted_chunks]
        
        # Simply concatenate - the chunks should maintain structure
        merged = '\n'.join(html_parts)
        
        # Clean up any duplicate body tags if they exist
        soup = BeautifulSoup(merged, 'html.parser')
        
        # If multiple body tags, merge their contents
        bodies = soup.find_all('body')
        if len(bodies) > 1:
            # Keep first body, move all content from others into it
            main_body = bodies[0]
            for extra_body in bodies[1:]:
                for child in list(extra_body.children):
                    main_body.append(child)
                extra_body.decompose()
            
            return str(soup)
        
        return merged
