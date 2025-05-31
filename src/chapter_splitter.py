import re
from bs4 import BeautifulSoup
import tiktoken

class ChapterSplitter:
    """Split large chapters into smaller chunks while preserving structure"""
    
    def __init__(self, model_name="gpt-3.5-turbo", target_tokens=80000):
        """
        Initialize splitter with token counter
        target_tokens: Target size for each chunk (leaving room for system prompt & history)
        """
        try:
            self.enc = tiktoken.encoding_for_model(model_name)
        except:
            self.enc = tiktoken.get_encoding("cl100k_base")
        self.target_tokens = target_tokens
    
    def count_tokens(self, text):
        """Count tokens in text"""
        try:
            return len(self.enc.encode(text))
        except:
            # Fallback estimation
            return len(text) // 4
    
    def split_chapter(self, chapter_html, max_tokens=None):
        """
        Split a chapter into smaller chunks if it exceeds token limit
        Returns: List of (chunk_html, chunk_index, total_chunks)
        """
        if max_tokens is None:
            max_tokens = self.target_tokens
            
        # First check if splitting is needed
        total_tokens = self.count_tokens(chapter_html)
        if total_tokens <= max_tokens:
            return [(chapter_html, 1, 1)]  # No split needed
        
        # Parse HTML
        soup = BeautifulSoup(chapter_html, 'html.parser')
        
        # Try to find natural break points
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        # Get all direct children of body, or all top-level elements
        if soup.body:
            elements = list(soup.body.children)
        else:
            elements = list(soup.children)
        
        for element in elements:
            if isinstance(element, str) and element.strip() == '':
                continue
                
            element_html = str(element)
            element_tokens = self.count_tokens(element_html)
            
            # If single element is too large, try to split it
            if element_tokens > max_tokens:
                sub_chunks = self._split_large_element(element, max_tokens)
                for sub_chunk in sub_chunks:
                    chunks.append(sub_chunk)
            else:
                # Check if adding this element would exceed limit
                if current_tokens + element_tokens > max_tokens and current_chunk:
                    # Save current chunk
                    chunks.append(self._create_chunk_html(current_chunk))
                    current_chunk = [element_html]
                    current_tokens = element_tokens
                else:
                    current_chunk.append(element_html)
                    current_tokens += element_tokens
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(self._create_chunk_html(current_chunk))
        
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