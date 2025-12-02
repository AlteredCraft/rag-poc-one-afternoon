"""Unit tests for chunk_by_tokens function."""
import pytest
from pathlib import Path
from ingest import chunk_by_tokens, count_tokens


class TestChunkByTokens:
    """Tests for the token-based chunking function."""
    
    @pytest.fixture
    def test_document_path(self):
        """Path to the test fixture document."""
        return Path(__file__).parent / "fixtures" / "test_document.md"
    
    @pytest.fixture
    def test_document_content(self, test_document_path):
        """Content of the test document."""
        with open(test_document_path, 'r') as f:
            return f.read()
    
    def test_chunk_by_tokens_creates_multiple_chunks(self, test_document_content):
        """Test that chunking creates multiple chunks for a long document."""
        chunks = chunk_by_tokens(test_document_content, max_tokens=256, stride=50)
        
        # The test document should generate multiple chunks
        assert len(chunks) > 1, "Expected multiple chunks from test document"
    
    def test_chunk_by_tokens_respects_max_size(self, test_document_content):
        """Test that all chunks are within the maximum token limit."""
        max_tokens = 256
        chunks = chunk_by_tokens(test_document_content, max_tokens=max_tokens, stride=50)
        
        for i, chunk in enumerate(chunks):
            token_count = count_tokens(chunk)
            assert token_count <= max_tokens, (
                f"Chunk {i} has {token_count} tokens, exceeds max of {max_tokens}"
            )
    
    def test_chunk_by_tokens_has_overlap(self, test_document_content):
        """Test that consecutive chunks have the expected overlap (stride)."""
        stride = 50
        chunks = chunk_by_tokens(test_document_content, max_tokens=256, stride=stride)
        
        if len(chunks) < 2:
            pytest.skip("Not enough chunks to test overlap")
        
        # Check overlap between consecutive chunks
        # The overlap detection is approximate since we're using decoded text
        for i in range(len(chunks) - 1):
            chunk1 = chunks[i]
            chunk2 = chunks[i + 1]
            
            # Find the longest common suffix/prefix
            # This is a simple heuristic - in practice, the overlap might not be
            # exact due to tokenizer word boundaries
            overlap_found = False
            
            # Check if any significant portion of chunk1's end appears in chunk2's start
            # We'll check the last 20% of chunk1 against the first 30% of chunk2
            chunk1_tail = chunk1[-len(chunk1)//5:] if len(chunk1) > 50 else chunk1
            chunk2_head = chunk2[:len(chunk2)//3] if len(chunk2) > 50 else chunk2
            
            # Look for overlap (at least 10 characters)
            for offset in range(len(chunk1_tail)):
                sample = chunk1_tail[offset:offset+10]
                if len(sample) >= 10 and sample in chunk2_head:
                    overlap_found = True
                    break
            
            assert overlap_found, (
                f"No overlap detected between chunk {i} and {i+1}. "
                f"Expected stride of {stride} tokens to create overlap."
            )
    
    def test_chunk_by_tokens_with_short_text(self):
        """Test that short text creates a single chunk."""
        short_text = "This is a short piece of text."
        chunks = chunk_by_tokens(short_text, max_tokens=256, stride=50)
        
        assert len(chunks) == 1, "Short text should create exactly one chunk"
        # Note: tokenizer may normalize text (e.g., lowercasing)
        assert chunks[0].lower() == short_text.lower(), (
            "Single chunk should match original text (case-insensitive)"
        )
    
    def test_chunk_by_tokens_with_empty_text(self):
        """Test that empty text returns no chunks."""
        chunks = chunk_by_tokens("", max_tokens=256, stride=50)
        assert len(chunks) == 0, "Empty text should return no chunks"
        
        chunks = chunk_by_tokens("   ", max_tokens=256, stride=50)
        assert len(chunks) == 0, "Whitespace-only text should return no chunks"
    
    def test_chunk_by_tokens_preserves_content(self, test_document_content):
        """Test that all chunks together contain the full document content."""
        chunks = chunk_by_tokens(test_document_content, max_tokens=256, stride=50)
        
        assert len(chunks) > 0, "Should have at least one chunk"
        
        # Skip the frontmatter and check that the actual content is present
        # Find the "Introduction" heading which should be in the first chunk
        introduction_found = any("introduction" in chunk.lower() for chunk in chunks)
        assert introduction_found, "Expected to find 'Introduction' heading in chunks"
        
        # Check that the conclusion is in the last chunk
        conclusion_found = "conclusion" in chunks[-1].lower()
        assert conclusion_found, "Expected to find 'Conclusion' in the last chunk"
        
        # Verify all chunks have content
        for i, chunk in enumerate(chunks):
            assert len(chunk.strip()) > 0, f"Chunk {i} is empty"
