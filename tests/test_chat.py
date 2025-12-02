"""Tests for the chat CLI module."""

import io
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from chat import (
    RAGConfig,
    ChatContext,
    QueryResult,
    RAGService,
    ChatService,
    build_context_prompt,
    build_where_filter,
    filter_by_relevance,
    handle_set_command,
    print_banner,
    print_help,
    print_config,
    print_sources,
    query_chromadb,
    chat_with_anthropic,
)


# --- Test Fixtures ---


@pytest.fixture
def mock_console():
    """Create a console that captures output."""
    return Console(file=io.StringIO(), force_terminal=True)


@pytest.fixture
def mock_chroma_client():
    """Create a mock ChromaDB client."""
    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "documents": [["doc1", "doc2"]],
        "metadatas": [[{"title": "Test 1"}, {"title": "Test 2"}]],
        "distances": [[0.1, 0.2]],
    }

    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_collection
    return mock_client


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client."""
    mock_stream = MagicMock()
    mock_stream.text_stream = iter(["Hello", " world", "!"])
    mock_stream.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream.__exit__ = MagicMock(return_value=False)

    mock_messages = MagicMock()
    mock_messages.stream.return_value = mock_stream

    mock_client = MagicMock()
    mock_client.messages = mock_messages
    return mock_client


# --- Data Class Tests ---


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


class TestQueryResult:
    """Tests for QueryResult dataclass."""

    def test_query_result(self):
        result = QueryResult(
            documents=["doc1", "doc2"],
            metadatas=[{"title": "A"}, {"title": "B"}],
        )
        assert len(result.documents) == 2
        assert len(result.metadatas) == 2


# --- Pure Function Tests ---


class TestBuildWhereFilter:
    """Tests for build_where_filter function."""

    def test_no_filters(self):
        config = RAGConfig()
        assert build_where_filter(config) is None

    def test_type_filter_only(self):
        config = RAGConfig(filter_type="deep_dive")
        assert build_where_filter(config) == {"type": "deep_dive"}

    def test_author_filter_only(self):
        config = RAGConfig(filter_author="Sam")
        assert build_where_filter(config) == {"author": "Sam"}

    def test_both_filters(self):
        config = RAGConfig(filter_type="deep_dive", filter_author="Sam")
        result = build_where_filter(config)
        assert "$and" in result
        assert {"type": "deep_dive"} in result["$and"]
        assert {"author": "Sam"} in result["$and"]


class TestFilterByRelevance:
    """Tests for filter_by_relevance function."""

    def test_no_filtering_when_all_pass(self):
        docs = ["doc1", "doc2"]
        metas = [{"title": "A"}, {"title": "B"}]
        distances = [0.1, 0.2]  # similarity: 0.909, 0.833

        filtered_docs, filtered_metas = filter_by_relevance(
            docs, metas, distances, min_relevance=0.5
        )

        assert len(filtered_docs) == 2
        assert len(filtered_metas) == 2

    def test_filters_low_relevance(self):
        docs = ["doc1", "doc2", "doc3"]
        metas = [{"title": "A"}, {"title": "B"}, {"title": "C"}]
        distances = [0.1, 0.5, 2.0]  # similarity: 0.909, 0.667, 0.333

        filtered_docs, filtered_metas = filter_by_relevance(
            docs, metas, distances, min_relevance=0.5
        )

        assert len(filtered_docs) == 2
        assert filtered_docs == ["doc1", "doc2"]

    def test_filters_all_when_threshold_high(self):
        docs = ["doc1", "doc2"]
        metas = [{"title": "A"}, {"title": "B"}]
        distances = [1.0, 2.0]  # similarity: 0.5, 0.333

        filtered_docs, filtered_metas = filter_by_relevance(
            docs, metas, distances, min_relevance=0.9
        )

        assert len(filtered_docs) == 0


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


# --- Handle Set Command Tests ---


class TestHandleSetCommand:
    """Tests for the /set command handler."""

    def test_set_n_results(self, mock_console):
        config = RAGConfig()
        result = handle_set_command(["n_results", "10"], config, mock_console)
        assert result is True
        assert config.n_results == 10

    def test_set_n_results_minimum(self, mock_console):
        config = RAGConfig()
        handle_set_command(["n_results", "0"], config, mock_console)
        assert config.n_results == 1  # Minimum is 1

    def test_set_min_relevance(self, mock_console):
        config = RAGConfig()
        result = handle_set_command(["min_relevance", "0.7"], config, mock_console)
        assert result is True
        assert config.min_relevance == 0.7

    def test_set_min_relevance_clamped(self, mock_console):
        config = RAGConfig()
        handle_set_command(["min_relevance", "1.5"], config, mock_console)
        assert config.min_relevance == 1.0

        handle_set_command(["min_relevance", "-0.5"], config, mock_console)
        assert config.min_relevance == 0.0

    def test_set_min_relevance_none(self, mock_console):
        config = RAGConfig(min_relevance=0.5)
        result = handle_set_command(["min_relevance", "none"], config, mock_console)
        assert result is True
        assert config.min_relevance is None

    def test_set_filter_type(self, mock_console):
        config = RAGConfig()
        result = handle_set_command(["filter_type", "deep_dive"], config, mock_console)
        assert result is True
        assert config.filter_type == "deep_dive"

    def test_set_filter_type_none(self, mock_console):
        config = RAGConfig(filter_type="test")
        result = handle_set_command(["filter_type", "none"], config, mock_console)
        assert result is True
        assert config.filter_type is None

    def test_set_filter_author(self, mock_console):
        config = RAGConfig()
        result = handle_set_command(["filter_author", "Sam"], config, mock_console)
        assert result is True
        assert config.filter_author == "Sam"

    def test_set_filter_author_none(self, mock_console):
        config = RAGConfig(filter_author="Sam")
        result = handle_set_command(["filter_author", "none"], config, mock_console)
        assert result is True
        assert config.filter_author is None

    def test_set_include_sources_true(self, mock_console):
        config = RAGConfig(include_sources=False)
        result = handle_set_command(["include_sources", "true"], config, mock_console)
        assert result is True
        assert config.include_sources is True

    def test_set_include_sources_yes(self, mock_console):
        config = RAGConfig(include_sources=False)
        result = handle_set_command(["include_sources", "yes"], config, mock_console)
        assert result is True
        assert config.include_sources is True

    def test_set_include_sources_1(self, mock_console):
        config = RAGConfig(include_sources=False)
        result = handle_set_command(["include_sources", "1"], config, mock_console)
        assert result is True
        assert config.include_sources is True

    def test_set_include_sources_false(self, mock_console):
        config = RAGConfig()
        result = handle_set_command(["include_sources", "false"], config, mock_console)
        assert result is True
        assert config.include_sources is False

    def test_set_unknown_option(self, mock_console):
        config = RAGConfig()
        result = handle_set_command(["unknown", "value"], config, mock_console)
        assert result is False

    def test_set_invalid_value(self, mock_console):
        config = RAGConfig()
        result = handle_set_command(["n_results", "not_a_number"], config, mock_console)
        assert result is False

    def test_set_missing_args(self, mock_console):
        config = RAGConfig()
        result = handle_set_command(["n_results"], config, mock_console)
        assert result is False

    def test_set_empty_args(self, mock_console):
        config = RAGConfig()
        result = handle_set_command([], config, mock_console)
        assert result is False


# --- RAGService Tests ---


class TestRAGService:
    """Tests for RAGService class with DI."""

    def test_query_basic(self, mock_chroma_client):
        service = RAGService(mock_chroma_client)
        config = RAGConfig(n_results=2)

        result = service.query("test query", "test_collection", config)

        assert len(result.documents) == 2
        assert len(result.metadatas) == 2
        mock_chroma_client.get_collection.assert_called_once_with("test_collection")

    def test_query_with_type_filter(self, mock_chroma_client):
        service = RAGService(mock_chroma_client)
        config = RAGConfig(filter_type="deep_dive")

        service.query("test", "collection", config)

        call_args = mock_chroma_client.get_collection().query.call_args
        assert call_args.kwargs["where"] == {"type": "deep_dive"}

    def test_query_with_author_filter(self, mock_chroma_client):
        service = RAGService(mock_chroma_client)
        config = RAGConfig(filter_author="Sam")

        service.query("test", "collection", config)

        call_args = mock_chroma_client.get_collection().query.call_args
        assert call_args.kwargs["where"] == {"author": "Sam"}

    def test_query_with_multiple_filters(self, mock_chroma_client):
        service = RAGService(mock_chroma_client)
        config = RAGConfig(filter_type="deep_dive", filter_author="Sam")

        service.query("test", "collection", config)

        call_args = mock_chroma_client.get_collection().query.call_args
        where_filter = call_args.kwargs["where"]
        assert "$and" in where_filter

    def test_query_with_min_relevance(self):
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["doc1", "doc2", "doc3"]],
            "metadatas": [[{"title": "A"}, {"title": "B"}, {"title": "C"}]],
            "distances": [[0.1, 0.5, 2.0]],
        }

        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection

        service = RAGService(mock_client)
        config = RAGConfig(min_relevance=0.5)

        result = service.query("test", "collection", config)

        # Only first two docs pass the 0.5 threshold
        assert len(result.documents) == 2

    def test_query_collection_not_found(self):
        mock_client = MagicMock()
        mock_client.get_collection.side_effect = Exception("Collection not found")

        service = RAGService(mock_client)
        config = RAGConfig()

        result = service.query("test", "nonexistent", config)

        assert result.documents == []
        assert result.metadatas == []


# --- ChatService Tests ---


class TestChatService:
    """Tests for ChatService class with DI."""

    def test_chat_basic(self, mock_anthropic_client, mock_console):
        service = ChatService(mock_anthropic_client)

        response = service.chat(
            query="Hello",
            context="",
            history=[],
            model="test-model",
            console=mock_console,
        )

        assert response == "Hello world!"
        mock_anthropic_client.messages.stream.assert_called_once()

    def test_chat_with_context(self, mock_anthropic_client, mock_console):
        service = ChatService(mock_anthropic_client)

        response = service.chat(
            query="What is AI?",
            context="AI stands for Artificial Intelligence.",
            history=[],
            model="test-model",
            console=mock_console,
        )

        call_args = mock_anthropic_client.messages.stream.call_args
        system_prompt = call_args.kwargs["system"]
        assert "AI stands for Artificial Intelligence" in system_prompt

    def test_chat_with_history(self, mock_anthropic_client, mock_console):
        service = ChatService(mock_anthropic_client)
        history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]

        service.chat(
            query="How are you?",
            context="",
            history=history,
            model="test-model",
            console=mock_console,
        )

        call_args = mock_anthropic_client.messages.stream.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 3  # 2 history + 1 new
        assert messages[-1]["content"] == "How are you?"

    def test_chat_system_prompt_structure(self, mock_anthropic_client, mock_console):
        service = ChatService(mock_anthropic_client)

        service.chat(
            query="Test",
            context="",
            history=[],
            console=mock_console,
        )

        call_args = mock_anthropic_client.messages.stream.call_args
        system_prompt = call_args.kwargs["system"]
        assert "helpful assistant" in system_prompt
        assert "knowledge base" in system_prompt


# --- Legacy Wrapper Tests ---


class TestQueryChromaDB:
    """Tests for query_chromadb legacy wrapper."""

    def test_with_injected_client(self, mock_chroma_client):
        config = RAGConfig(n_results=2)
        docs, metas = query_chromadb(
            "test query",
            "test_collection",
            config,
            chroma_client=mock_chroma_client,
        )

        assert len(docs) == 2
        assert len(metas) == 2

    @patch("chat.get_chroma_client")
    def test_creates_default_client(self, mock_get_client, mock_chroma_client):
        mock_get_client.return_value = mock_chroma_client
        config = RAGConfig()

        query_chromadb("test", "collection", config)

        mock_get_client.assert_called_once()


class TestChatWithAnthropic:
    """Tests for chat_with_anthropic legacy wrapper."""

    def test_with_injected_client(self, mock_anthropic_client, mock_console):
        response = chat_with_anthropic(
            query="Hello",
            context="",
            history=[],
            model="test-model",
            anthropic_client=mock_anthropic_client,
            output_console=mock_console,
        )

        assert response == "Hello world!"


# --- UI Function Tests ---


class TestUIFunctions:
    """Tests for UI printing functions."""

    def test_print_banner(self, mock_console):
        print_banner(mock_console)
        output = mock_console.file.getvalue()
        assert "RAG" in output
        assert "Chat" in output

    def test_print_help(self, mock_console):
        print_help(mock_console)
        output = mock_console.file.getvalue()
        assert "/help" in output
        assert "/quit" in output
        assert "/config" in output

    def test_print_config(self, mock_console):
        config = RAGConfig(n_results=10, filter_type="test")
        print_config(config, mock_console)
        output = mock_console.file.getvalue()
        assert "10" in output
        assert "test" in output

    def test_print_config_with_relevance(self, mock_console):
        config = RAGConfig(min_relevance=0.7)
        print_config(config, mock_console)
        output = mock_console.file.getvalue()
        assert "0.7" in output

    def test_print_sources_empty(self, mock_console):
        print_sources([], mock_console)
        output = mock_console.file.getvalue()
        assert "No sources available" in output

    def test_print_sources_with_data(self, mock_console):
        sources = [
            {"source_file": "test.md", "title": "Test Article", "type": "deep_dive"},
            {"source_file": "other.md", "title": "Other Article", "type": "review"},
        ]
        print_sources(sources, mock_console)
        output = mock_console.file.getvalue()
        assert "test.md" in output
        assert "Test Article" in output
        assert "deep_dive" in output

    def test_print_sources_truncates_long_titles(self, mock_console):
        sources = [
            {"source_file": "test.md", "title": "A" * 100, "type": "test"},
        ]
        print_sources(sources, mock_console)
        output = mock_console.file.getvalue()
        # Title should be truncated to 40 chars
        assert "A" * 40 in output
        assert "A" * 100 not in output

    def test_print_sources_handles_missing_fields(self, mock_console):
        sources = [{}]  # No fields
        print_sources(sources, mock_console)
        output = mock_console.file.getvalue()
        assert "unknown" in output
        assert "untitled" in output


# --- CLI Command Tests ---


class TestCLICommands:
    """Tests for CLI commands using Typer test runner."""

    def test_chat_missing_api_key(self, monkeypatch):
        """Test that chat command fails without API key."""
        from typer.testing import CliRunner
        from chat import app

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        runner = CliRunner()
        result = runner.invoke(app, ["chat", "--no-banner"])

        assert result.exit_code == 1
        assert "ANTHROPIC_API_KEY" in result.stdout

    def test_query_missing_api_key(self, monkeypatch):
        """Test that query command fails without API key."""
        from typer.testing import CliRunner
        from chat import app

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        runner = CliRunner()
        result = runner.invoke(app, ["query", "test question"])

        assert result.exit_code == 1
        assert "ANTHROPIC_API_KEY" in result.stdout

    def test_query_with_mocked_services(self, monkeypatch, mock_chroma_client, mock_anthropic_client):
        """Test query command with mocked services."""
        from typer.testing import CliRunner
        from chat import app

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        # Patch the service creation
        with patch("chat.get_chroma_client", return_value=mock_chroma_client):
            with patch("anthropic.Anthropic", return_value=mock_anthropic_client):
                runner = CliRunner()
                result = runner.invoke(app, ["query", "What is AI?", "-n", "3"])

                # Should find documents
                assert "relevant document" in result.stdout.lower() or result.exit_code == 0

    def test_chat_quit_command(self, monkeypatch, mock_chroma_client):
        """Test that /quit exits the chat loop."""
        from typer.testing import CliRunner
        from chat import app

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        with patch("chat.get_chroma_client", return_value=mock_chroma_client):
            runner = CliRunner()
            result = runner.invoke(app, ["chat", "--no-banner"], input="/quit\n")

            assert "Goodbye" in result.stdout

    def test_chat_exit_command(self, monkeypatch, mock_chroma_client):
        """Test that /exit exits the chat loop."""
        from typer.testing import CliRunner
        from chat import app

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        with patch("chat.get_chroma_client", return_value=mock_chroma_client):
            runner = CliRunner()
            result = runner.invoke(app, ["chat", "--no-banner"], input="/exit\n")

            assert "Goodbye" in result.stdout

    def test_chat_q_command(self, monkeypatch, mock_chroma_client):
        """Test that /q exits the chat loop."""
        from typer.testing import CliRunner
        from chat import app

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        with patch("chat.get_chroma_client", return_value=mock_chroma_client):
            runner = CliRunner()
            result = runner.invoke(app, ["chat", "--no-banner"], input="/q\n")

            assert "Goodbye" in result.stdout

    def test_chat_help_command(self, monkeypatch, mock_chroma_client):
        """Test /help command in chat."""
        from typer.testing import CliRunner
        from chat import app

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        with patch("chat.get_chroma_client", return_value=mock_chroma_client):
            runner = CliRunner()
            result = runner.invoke(app, ["chat", "--no-banner"], input="/help\n/quit\n")

            assert "/help" in result.stdout
            assert "/quit" in result.stdout

    def test_chat_config_command(self, monkeypatch, mock_chroma_client):
        """Test /config command in chat."""
        from typer.testing import CliRunner
        from chat import app

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        with patch("chat.get_chroma_client", return_value=mock_chroma_client):
            runner = CliRunner()
            result = runner.invoke(app, ["chat", "--no-banner", "-n", "10"], input="/config\n/quit\n")

            assert "10" in result.stdout

    def test_chat_sources_empty(self, monkeypatch, mock_chroma_client):
        """Test /sources command with no sources."""
        from typer.testing import CliRunner
        from chat import app

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        with patch("chat.get_chroma_client", return_value=mock_chroma_client):
            runner = CliRunner()
            result = runner.invoke(app, ["chat", "--no-banner"], input="/sources\n/quit\n")

            assert "No sources available" in result.stdout

    def test_chat_clear_command(self, monkeypatch, mock_chroma_client):
        """Test /clear command in chat."""
        from typer.testing import CliRunner
        from chat import app

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        with patch("chat.get_chroma_client", return_value=mock_chroma_client):
            runner = CliRunner()
            result = runner.invoke(app, ["chat", "--no-banner"], input="/clear\n/quit\n")

            assert "cleared" in result.stdout.lower()

    def test_chat_set_command(self, monkeypatch, mock_chroma_client):
        """Test /set command in chat."""
        from typer.testing import CliRunner
        from chat import app

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        with patch("chat.get_chroma_client", return_value=mock_chroma_client):
            runner = CliRunner()
            result = runner.invoke(app, ["chat", "--no-banner"], input="/set n_results 20\n/quit\n")

            assert "Set n_results" in result.stdout

    def test_chat_unknown_command(self, monkeypatch, mock_chroma_client):
        """Test unknown command in chat."""
        from typer.testing import CliRunner
        from chat import app

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        with patch("chat.get_chroma_client", return_value=mock_chroma_client):
            runner = CliRunner()
            result = runner.invoke(app, ["chat", "--no-banner"], input="/unknown\n/quit\n")

            assert "Unknown command" in result.stdout

    def test_chat_empty_input_ignored(self, monkeypatch, mock_chroma_client):
        """Test that empty input is ignored."""
        from typer.testing import CliRunner
        from chat import app

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        with patch("chat.get_chroma_client", return_value=mock_chroma_client):
            runner = CliRunner()
            result = runner.invoke(app, ["chat", "--no-banner"], input="\n\n/quit\n")

            assert "Goodbye" in result.stdout

    def test_chat_with_banner(self, monkeypatch, mock_chroma_client):
        """Test chat with banner displayed."""
        from typer.testing import CliRunner
        from chat import app

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        with patch("chat.get_chroma_client", return_value=mock_chroma_client):
            runner = CliRunner()
            result = runner.invoke(app, ["chat"], input="/quit\n")

            assert "RAG" in result.stdout

    def test_chat_displays_collection_info(self, monkeypatch, mock_chroma_client):
        """Test that chat displays collection and model info."""
        from typer.testing import CliRunner
        from chat import app

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        with patch("chat.get_chroma_client", return_value=mock_chroma_client):
            runner = CliRunner()
            result = runner.invoke(app, ["chat", "--no-banner", "-c", "my_collection"], input="/quit\n")

            assert "my_collection" in result.stdout
