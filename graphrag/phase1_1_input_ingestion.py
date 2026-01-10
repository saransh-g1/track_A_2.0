"""
PHASE 1.1: INPUT INGESTION & CHUNKING

Implements:
- Chapter-aware chunking
- Chunk size: 600 tokens
- Overlap: 120 tokens

Output schema: Chunk {chunk_id, chapter_id, text, token_range}
"""

import re
import tiktoken
from typing import List, Tuple
from schemas import Chunk
from config import CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS


class InputIngestion:
    """Handles novel input ingestion and chapter-aware chunking."""
    
    def __init__(self):
        # Initialize tokenizer (using cl100k_base as proxy, adjust if needed)
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def parse_chapters(self, novel_text: str) -> List[Tuple[int, str]]:
        """
        Parse novel text into chapters.
        
        Assumptions:
        - Chapters are marked with patterns like "CHAPTER X", "Chapter X", "Chapter X: Title", etc.
        - If no chapter markers found, treat entire text as single chapter
        
        Returns: List of (chapter_id, chapter_text) tuples
        """
        # Common chapter patterns
        chapter_patterns = [
            r'(?i)^\s*CHAPTER\s+(\d+)(?:\s*:.*)?$',
            r'(?i)^\s*Chapter\s+(\d+)(?:\s*:.*)?$',
            r'(?i)^\s*Ch\.\s*(\d+)(?:\s*:.*)?$',
        ]
        
        # Try to find chapter breaks
        lines = novel_text.split('\n')
        chapter_breaks = [0]  # Start of first chapter
        
        for i, line in enumerate(lines):
            for pattern in chapter_patterns:
                if re.match(pattern, line.strip()):
                    chapter_breaks.append(i)
                    break
        
        # If no chapters found, treat as single chapter
        if len(chapter_breaks) == 1:
            return [(1, novel_text)]
        
        # Also add end of text
        chapter_breaks.append(len(lines))
        
        chapters = []
        for i in range(len(chapter_breaks) - 1):
            chapter_id = i + 1
            start_line = chapter_breaks[i]
            end_line = chapter_breaks[i + 1]
            
            # Get chapter text (skip chapter header line)
            if i > 0:  # Skip first break (line 0)
                start_line += 1
            
            chapter_text = '\n'.join(lines[start_line:end_line]).strip()
            if chapter_text:
                chapters.append((chapter_id, chapter_text))
        
        return chapters
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
    
    def chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[Tuple[str, int, int]]:
        """
        Chunk text into overlapping segments.
        
        Returns: List of (chunk_text, start_token, end_token) tuples
        """
        tokens = self.encoding.encode(text)
        chunks = []
        
        if len(tokens) == 0:
            return chunks
        
        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)
            
            chunks.append((chunk_text, start, end - 1))
            
            if end >= len(tokens):
                break
            
            # Move start by (chunk_size - overlap) for next chunk
            start = end - overlap
        
        return chunks
    
    def ingest_novel(self, novel_text: str) -> List[Chunk]:
        """
        Main ingestion method.
        
        Process:
        1. Parse chapters
        2. For each chapter, create chunks with specified size and overlap
        3. Assign chapter_id to each chunk
        4. Return list of Chunk objects
        
        Args:
            novel_text: Full text of the novel
            
        Returns:
            List of Chunk objects with chunk_id, chapter_id, text, token_range
        """
        chapters = self.parse_chapters(novel_text)
        all_chunks = []
        chunk_id_counter = 1
        
        for chapter_id, chapter_text in chapters:
            # Chunk this chapter
            chapter_chunks = self.chunk_text(
                chapter_text,
                CHUNK_SIZE_TOKENS,
                CHUNK_OVERLAP_TOKENS
            )
            
            for chunk_text, start_token, end_token in chapter_chunks:
                chunk = Chunk(
                    chunk_id=chunk_id_counter,
                    chapter_id=chapter_id,
                    text=chunk_text,
                    token_range=(start_token, end_token)
                )
                all_chunks.append(chunk)
                chunk_id_counter += 1
        
        return all_chunks
    
    def ingest_from_file(self, file_path: str) -> List[Chunk]:
        """
        Ingest novel from a text file.
        
        Args:
            file_path: Path to the novel text file
            
        Returns:
            List of Chunk objects
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            novel_text = f.read()
        
        return self.ingest_novel(novel_text)


if __name__ == "__main__":
    # Test example
    test_text = """CHAPTER 1

It was a dark and stormy night. The protagonist walked down the street.
He thought about the events that led him here.

CHAPTER 2

The next morning, everything had changed. The storm had passed, but new challenges awaited."""
    
    ingester = InputIngestion()
    chunks = ingester.ingest_novel(test_text)
    
    print(f"Generated {len(chunks)} chunks:")
    for chunk in chunks[:3]:  # Show first 3
        print(f"  Chunk {chunk.chunk_id} (Chapter {chunk.chapter_id}): "
              f"Tokens {chunk.token_range[0]}-{chunk.token_range[1]}")

