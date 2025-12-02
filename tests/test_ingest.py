"""Unit tests for ingest.py"""

from pathlib import Path

import pytest

from ingest import chunk_by_tokens, find_markdown_files, ingest_directory, parse_markdown

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestFindMarkdownFiles:
    def test_finds_all_md_files(self):
        files = find_markdown_files(FIXTURES_DIR)
        assert len(files) == 6
        names = {f.name for f in files}
        assert "simple.md" in names
        assert "with_sections.md" in names
        assert "no_frontmatter.md" in names
        assert "empty.md" in names

    def test_returns_sorted_paths(self):
        files = find_markdown_files(FIXTURES_DIR)
        assert files == sorted(files)

    def test_empty_directory(self, tmp_path):
        files = find_markdown_files(tmp_path)
        assert files == []

    def test_nested_directory(self, tmp_path):
        # Create nested structure
        nested = tmp_path / "subdir" / "nested"
        nested.mkdir(parents=True)
        (tmp_path / "root.md").write_text("# Root")
        (nested / "nested.md").write_text("# Nested")

        files = find_markdown_files(tmp_path)
        assert len(files) == 2


class TestParseMarkdown:
    def test_parses_frontmatter(self):
        metadata, content = parse_markdown(FIXTURES_DIR / "simple.md")
        assert metadata["title"] == "Simple Post"
        assert str(metadata["date"]) == "2024-01-01"
        assert metadata["tags"] == ["test", "simple"]

    def test_parses_content(self):
        metadata, content = parse_markdown(FIXTURES_DIR / "simple.md")
        assert "# Simple Post" in content
        assert "simple post with no sections" in content

    def test_no_frontmatter(self):
        metadata, content = parse_markdown(FIXTURES_DIR / "no_frontmatter.md")
        assert metadata == {}
        assert "# Post Without Frontmatter" in content

    def test_empty_file(self):
        metadata, content = parse_markdown(FIXTURES_DIR / "empty.md")
        assert metadata == {}
        assert content == ""


class TestChunkByTokens:
    def test_single_section(self):
        content = "# Title\n\nSome content here."
        chunks = chunk_by_tokens(content)
        assert len(chunks) == 1
        assert "title" in chunks[0].lower()

    def test_multiple_h2_sections(self):
        content = """# Title

Intro text.

## Section One

Content one.

## Section Two

Content two."""
        chunks = chunk_by_tokens(content, max_tokens=256, stride=50)
        assert len(chunks) >= 1
        # Check sections are present
        all_content = "\n".join(chunks)
        assert "title" in all_content.lower()
        assert "section one" in all_content.lower()
        assert "section two" in all_content.lower()

    def test_h3_sections(self):
        content = """## Main Section

Intro.

### Sub One

Content.

### Sub Two

More content."""
        chunks = chunk_by_tokens(content)
        all_content = "\n".join(chunks)
        assert "sub one" in all_content.lower()
        assert "sub two" in all_content.lower()

    def test_empty_content(self):
        chunks = chunk_by_tokens("")
        assert chunks == []

    def test_whitespace_only(self):
        chunks = chunk_by_tokens("   \n\n   ")
        assert chunks == []

    def test_long_content_creates_multiple_chunks(self):
        # Create content that exceeds 256 tokens
        content = " ".join([f"Word{i}" for i in range(500)])
        chunks = chunk_by_tokens(content, max_tokens=256, stride=50)
        # Should create multiple chunks
        assert len(chunks) > 1

    def test_preserves_headers_in_chunks(self):
        content = """## First

Content.

## Second

More content."""
        chunks = chunk_by_tokens(content, max_tokens=256, stride=50)
        all_content = "\n".join(chunks).lower()
        assert "first" in all_content
        assert "second" in all_content


class TestIngestDirectory:
    def test_ingests_fixtures(self, tmp_path):
        # Use a temporary chromadb path
        persist_path = str(tmp_path / ".chromadb")
        result = ingest_directory(
            FIXTURES_DIR,
            collection_name="test_collection",
            persist_path=persist_path,
        )
        assert result["files_processed"] == 6
        assert result["chunks_added"] > 0
        assert result["collection_name"] == "test_collection"

    def test_nonexistent_directory(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ingest_directory(tmp_path / "nonexistent")

    def test_empty_directory(self, tmp_path):
        persist_path = str(tmp_path / ".chromadb")
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        result = ingest_directory(
            empty_dir,
            collection_name="empty_test",
            persist_path=persist_path,
        )
        assert result["files_processed"] == 0
        assert result["chunks_added"] == 0

    def test_creates_searchable_collection(self, tmp_path):
        from chroma_client import get_chroma_client

        persist_path = str(tmp_path / ".chromadb")
        ingest_directory(
            FIXTURES_DIR,
            collection_name="search_test",
            persist_path=persist_path,
        )

        # Query the collection
        client = get_chroma_client(persist_path=persist_path)
        collection = client.get_collection("search_test")

        results = collection.query(
            query_texts=["sections"],
            n_results=1,
        )
        assert len(results["documents"][0]) == 1
        assert results["metadatas"][0][0]["source_file"] is not None

    def test_metadata_stored_correctly(self, tmp_path):
        from chroma_client import get_chroma_client

        persist_path = str(tmp_path / ".chromadb")
        ingest_directory(
            FIXTURES_DIR,
            collection_name="metadata_test",
            persist_path=persist_path,
        )

        client = get_chroma_client(persist_path=persist_path)
        collection = client.get_collection("metadata_test")

        # Get all documents
        all_docs = collection.get(include=["metadatas"])

        # Find a document from simple.md
        simple_docs = [
            m for m in all_docs["metadatas"] if m["source_file"] == "simple.md"
        ]
        assert len(simple_docs) > 0
        assert simple_docs[0]["title"] == "Simple Post"
        assert "test" in simple_docs[0]["tags"]

    def test_all_frontmatter_fields_stored(self, tmp_path):
        """Test that all frontmatter fields (id, type, author, url) are stored."""
        from chroma_client import get_chroma_client

        persist_path = str(tmp_path / ".chromadb")
        ingest_directory(
            FIXTURES_DIR,
            collection_name="frontmatter_test",
            persist_path=persist_path,
        )

        client = get_chroma_client(persist_path=persist_path)
        collection = client.get_collection("frontmatter_test")

        all_docs = collection.get(include=["metadatas"])

        # Find document from full_frontmatter.md
        full_docs = [
            m for m in all_docs["metadatas"] if m["source_file"] == "full_frontmatter.md"
        ]
        assert len(full_docs) > 0
        meta = full_docs[0]
        assert meta["id"] == "2024-01-15_test-article"
        assert meta["type"] == "deep_dive"
        assert meta["author"] == "Test Author"
        assert meta["url"] == "https://example.com/test-article"

    def test_no_chunk_option(self, tmp_path):
        """Test that no_chunk=True ingests entire document as single chunk."""
        from chroma_client import get_chroma_client

        persist_path = str(tmp_path / ".chromadb")
        ingest_directory(
            FIXTURES_DIR,
            collection_name="no_chunk_test",
            persist_path=persist_path,
            no_chunk=True,
        )

        client = get_chroma_client(persist_path=persist_path)
        collection = client.get_collection("no_chunk_test")

        all_docs = collection.get(include=["metadatas"])

        # Each file with content should have exactly 1 chunk
        simple_docs = [
            m for m in all_docs["metadatas"] if m["source_file"] == "simple.md"
        ]
        assert len(simple_docs) == 1
        assert simple_docs[0]["total_chunks"] == 1


class TestChromaClient:
    def test_invalid_client_type_raises_error(self):
        from chroma_client import get_chroma_client

        with pytest.raises(ValueError, match="Unknown client type"):
            get_chroma_client(client_type="invalid")
