import re
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
    
    def split_chapter(self, chapter_html, max_tokens=None):
        """
        Split a chapter into smaller chunks if it exceeds token limit.
        Splits by token count, not HTML structure, for even distribution.
        Returns: List of (chunk_html, chunk_index, total_chunks)
        """
        if max_tokens is None:
            max_tokens = self.target_tokens
        
        # The max_tokens passed in has already been adjusted for compression in TransateKRtoEN.py
        # (via available_tokens = (max_output_tokens - safety_margin) / compression_factor)
        # So we use it directly without any further compression adjustment
        effective_max_tokens = max_tokens
            
        # First check if splitting is needed
        total_tokens = self.count_tokens(chapter_html)
        
        # Debug output to diagnose chunking issues
        import os
        if os.getenv('DEBUG_CHUNK_SPLITTING', '0') == '1':
            print(f"[CHUNK DEBUG] Total tokens: {total_tokens:,}")
            print(f"[CHUNK DEBUG] Effective max tokens: {effective_max_tokens:,}")
            print(f"[CHUNK DEBUG] Needs split: {total_tokens > effective_max_tokens}")
        
        if total_tokens <= effective_max_tokens:
            return [(chapter_html, 1, 1)]  # No split needed
        
        # NEW APPROACH: Split by token count, treating content as a stream
        # This ensures even chunk sizes regardless of HTML structure
        soup = BeautifulSoup(chapter_html, 'html.parser')
        
        # Get all text-containing elements (paragraphs, divs, spans, etc.)
        # We'll process them as a stream and split when we hit the token limit
        if soup.body:
            elements = list(soup.body.children)
        else:
            elements = list(soup.children)
        
        chunks = []
        current_chunk_elements = []
        current_chunk_tokens = 0
        
        for element in elements:
            # Skip empty text nodes
            if isinstance(element, str) and element.strip() == '':
                continue
            
            element_html = str(element)
            element_tokens = self.count_tokens(element_html)
            
            # Special case: if a single element exceeds the limit
            if element_tokens > effective_max_tokens:
                # Save current chunk if we have one
                if current_chunk_elements:
                    chunks.append(self._create_chunk_html(current_chunk_elements))
                    current_chunk_elements = []
                    current_chunk_tokens = 0
                
                # Add the oversized element as its own chunk (unavoidable)
                # We could split it further, but for now, just include it
                chunks.append(element_html)
                continue
            
            # If adding this element would exceed the limit AND we have content
            if current_chunk_tokens + element_tokens > effective_max_tokens and current_chunk_elements:
                # Save current chunk at element boundary (elements are typically paragraphs/lines)
                chunks.append(self._create_chunk_html(current_chunk_elements))
                # Start new chunk with this element
                current_chunk_elements = [element_html]
                current_chunk_tokens = element_tokens
            else:
                # Add to current chunk
                current_chunk_elements.append(element_html)
                current_chunk_tokens += element_tokens
        
        # Don't forget the last chunk
        if current_chunk_elements:
            chunks.append(self._create_chunk_html(current_chunk_elements))
        
        # Fallback: if we somehow got 0 chunks, return the whole content
        if not chunks:
            chunks = [chapter_html]
        
        # Debug output for chunk details
        if os.getenv('DEBUG_CHUNK_SPLITTING', '0') == '1':
            print(f"[CHUNK DEBUG] Created {len(chunks)} chunks")
            for idx, chunk in enumerate(chunks, 1):
                chunk_tokens = self.count_tokens(chunk)
                print(f"[CHUNK DEBUG]   Chunk {idx}: {chunk_tokens:,} tokens ({len(chunk):,} chars)")
        
        # Return chunks with metadata
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
