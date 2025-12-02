"""Tests for the chat CLI module."""

from unittest.mock import MagicMock, patch

import pytest

from chat import (
    RAGConfig,
    ChatContext,
    build_context_prompt,
    handle_set_command,
    query_chromadb,
)


class TestRAGConfig:
    """Tests for RAGConfig dataclass."""

    def test_default_values(self):
        config = RAGConfig()
        assert config.n_results == 5
        assert config.min_relevance is None
        assert config.filter_type is None
        assert config.filter_author is None
        assert config.include_sources is True

    def test_custom_values(self):
        config = RAGConfig(
            n_results=10,
            min_relevance=0.5,
            filter_type="deep_dive",
            filter_author="Sam",
            include_sources=False,
        )
        assert config.n_results == 10
        assert config.min_relevance == 0.5
        assert config.filter_type == "deep_dive"
        assert config.filter_author == "Sam"
        assert config.include_sources is False


class TestChatContext:
    """Tests for ChatContext dataclass."""

    def test_default_values(self):
        ctx = ChatContext(
            collection_name="test_collection",
            rag_config=RAGConfig(),
        )
        assert ctx.collection_name == "test_collection"
        assert ctx.history == []
        assert ctx.last_sources == []

    def test_with_history(self):
        ctx = ChatContext(
            collection_name="test",
            rag_config=RAGConfig(),
            history=[{"role": "user", "content": "hello"}],
        )
        assert len(ctx.history) == 1


class TestHandleSetCommand:
    """Tests for the /set command handler."""

    def test_set_n_results(self):
        config = RAGConfig()
        result = handle_set_command(["n_results", "10"], config)
        assert result is True
        assert config.n_results == 10

    def test_set_n_results_minimum(self):
        config = RAGConfig()
        handle_set_command(["n_results", "0"], config)
        assert config.n_results == 1  # Minimum is 1

    def test_set_min_relevance(self):
        config = RAGConfig()
        result = handle_set_command(["min_relevance", "0.7"], config)
        assert result is True
        assert config.min_relevance == 0.7

    def test_set_min_relevance_clamped(self):
        config = RAGConfig()
        handle_set_command(["min_relevance", "1.5"], config)
        assert config.min_relevance == 1.0

        handle_set_command(["min_relevance", "-0.5"], config)
        assert config.min_relevance == 0.0

    def test_set_min_relevance_none(self):
        config = RAGConfig(min_relevance=0.5)
        result = handle_set_command(["min_relevance", "none"], config)
        assert result is True
        assert config.min_relevance is None

    def test_set_filter_type(self):
        config = RAGConfig()
        result = handle_set_command(["filter_type", "deep_dive"], config)
        assert result is True
        assert config.filter_type == "deep_dive"

    def test_set_filter_type_none(self):
        config = RAGConfig(filter_type="test")
        result = handle_set_command(["filter_type", "none"], config)
        assert result is True
        assert config.filter_type is None

    def test_set_filter_author(self):
        config = RAGConfig()
        result = handle_set_command(["filter_author", "Sam"], config)
        assert result is True
        assert config.filter_author == "Sam"

    def test_set_include_sources_true(self):
        config = RAGConfig(include_sources=False)
        result = handle_set_command(["include_sources", "true"], config)
        assert result is True
        assert config.include_sources is True

    def test_set_include_sources_false(self):
        config = RAGConfig()
        result = handle_set_command(["include_sources", "false"], config)
        assert result is True
        assert config.include_sources is False

    def test_set_unknown_option(self):
        config = RAGConfig()
        result = handle_set_command(["unknown", "value"], config)
        assert result is False

    def test_set_invalid_value(self):
        config = RAGConfig()
        result = handle_set_command(["n_results", "not_a_number"], config)
        assert result is False

    def test_set_missing_args(self):
        config = RAGConfig()
        result = handle_set_command(["n_results"], config)
        assert result is False


class TestBuildContextPrompt:
    """Tests for context prompt building."""

    def test_empty_documents(self):
        result = build_context_prompt([], [])
        assert result == ""

    def test_single_document(self):
        docs = ["This is test content."]
        metas = [{"source_file": "test.md", "title": "Test Article"}]

        result = build_context_prompt(docs, metas)

        assert "relevant context" in result
        assert "Document 1: Test Article" in result
        assert "source: test.md" in result
        assert "This is test content." in result

    def test_multiple_documents(self):
        docs = ["Content one.", "Content two."]
        metas = [
            {"source_file": "one.md", "title": "Article One"},
            {"source_file": "two.md", "title": "Article Two"},
        ]

        result = build_context_prompt(docs, metas)

        assert "Document 1: Article One" in result
        assert "Document 2: Article Two" in result
        assert "Content one." in result
        assert "Content two." in result

    def test_missing_metadata_fields(self):
        docs = ["Content here."]
        metas = [{}]  # No title or source_file

        result = build_context_prompt(docs, metas)

        assert "Document 1: Untitled" in result
        assert "source: unknown" in result


class TestQueryChromaDB:
    """Tests for ChromaDB querying."""

    @patch("chat.get_chroma_client")
    def test_query_basic(self, mock_get_client):
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"title": "Test 1"}, {"title": "Test 2"}]],
            "distances": [[0.1, 0.2]],
        }

        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_get_client.return_value = mock_client

        config = RAGConfig(n_results=2)
        docs, metas = query_chromadb("test query", "test_collection", config)

        assert len(docs) == 2
        assert len(metas) == 2
        mock_collection.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=2,
            where=None,
            include=["documents", "metadatas", "distances"],
        )

    @patch("chat.get_chroma_client")
    def test_query_with_type_filter(self, mock_get_client):
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["doc1"]],
            "metadatas": [[{"type": "deep_dive"}]],
            "distances": [[0.1]],
        }

        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_get_client.return_value = mock_client

        config = RAGConfig(filter_type="deep_dive")
        query_chromadb("test", "collection", config)

        call_args = mock_collection.query.call_args
        assert call_args.kwargs["where"] == {"type": "deep_dive"}

    @patch("chat.get_chroma_client")
    def test_query_with_author_filter(self, mock_get_client):
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["doc1"]],
            "metadatas": [[{"author": "Sam"}]],
            "distances": [[0.1]],
        }

        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_get_client.return_value = mock_client

        config = RAGConfig(filter_author="Sam")
        query_chromadb("test", "collection", config)

        call_args = mock_collection.query.call_args
        assert call_args.kwargs["where"] == {"author": "Sam"}

    @patch("chat.get_chroma_client")
    def test_query_with_multiple_filters(self, mock_get_client):
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["doc1"]],
            "metadatas": [[{}]],
            "distances": [[0.1]],
        }

        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_get_client.return_value = mock_client

        config = RAGConfig(filter_type="deep_dive", filter_author="Sam")
        query_chromadb("test", "collection", config)

        call_args = mock_collection.query.call_args
        where_filter = call_args.kwargs["where"]
        assert "$and" in where_filter
        assert {"type": "deep_dive"} in where_filter["$and"]
        assert {"author": "Sam"} in where_filter["$and"]

    @patch("chat.get_chroma_client")
    def test_query_with_min_relevance(self, mock_get_client):
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["doc1", "doc2", "doc3"]],
            "metadatas": [[{"title": "A"}, {"title": "B"}, {"title": "C"}]],
            "distances": [[0.1, 0.5, 2.0]],  # 0.1 is close, 2.0 is far
        }

        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_get_client.return_value = mock_client

        # Set high min_relevance to filter out distant results
        config = RAGConfig(min_relevance=0.5)
        docs, metas = query_chromadb("test", "collection", config)

        # Only the closest document should pass the filter
        # similarity = 1/(1+0.1) = 0.909, 1/(1+0.5) = 0.667, 1/(1+2.0) = 0.333
        assert len(docs) == 2  # First two pass 0.5 threshold

    @patch("chat.get_chroma_client")
    def test_query_collection_not_found(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.get_collection.side_effect = Exception("Collection not found")
        mock_get_client.return_value = mock_client

        config = RAGConfig()
        docs, metas = query_chromadb("test", "nonexistent", config)

        assert docs == []
        assert metas == []
