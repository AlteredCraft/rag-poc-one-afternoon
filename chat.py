#!/usr/bin/env python3
"""Interactive CLI chat with RAG-powered responses using Anthropic and ChromaDB."""

import json
import logging
import os

# Load .env before reading any environment variables
from dotenv import load_dotenv

load_dotenv()

# Disable tokenizers parallelism to avoid fork warnings when used with Rich's Live display
# This must be set before any HuggingFace tokenizers are imported (via ChromaDB)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dataclasses import dataclass, field
from typing import Protocol

# Configure file-based logging (no stdout to avoid interfering with Rich UI)
LOG_FILE = os.environ.get("CHAT_LOG_FILE", "chat.log")
LOG_LEVEL = os.environ.get("CHAT_LOG_LEVEL", "DEBUG")

# Set up file handler with formatting
_file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
_file_handler.setFormatter(
    logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
)

# Root logger at WARNING to quiet dependencies (httpx, chromadb, anthropic, etc.)
logging.root.setLevel(logging.WARNING)
logging.root.addHandler(_file_handler)

# App loggers at configured level (default DEBUG)
_app_level = getattr(logging, LOG_LEVEL.upper(), logging.DEBUG)
for _module in ("chat", "chroma_client", "ingest", "__main__"):
    logging.getLogger(_module).setLevel(_app_level)

logger = logging.getLogger(__name__)

# Model and prompt configuration
DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
SYSTEM_PROMPT_FILE = os.environ.get("SYSTEM_PROMPT_FILE", "prompt.txt")


def load_system_prompt() -> str:
    """Load system prompt from file, with fallback to default."""
    default_prompt = """You are a helpful assistant that answers questions based on the provided context from a knowledge base.

When answering:
- Use the provided context to give accurate, relevant answers
- If the context doesn't contain enough information, say so clearly
- Be concise but thorough
- Reference specific sources when relevant
- If you're unsure, express uncertainty rather than making things up"""

    try:
        with open(SYSTEM_PROMPT_FILE, encoding="utf-8") as f:
            prompt = f.read().strip()
            logger.debug(f"Loaded system prompt from {SYSTEM_PROMPT_FILE}")
            return prompt
    except FileNotFoundError:
        logger.warning(f"System prompt file not found: {SYSTEM_PROMPT_FILE}, using default")
        return default_prompt
    except Exception as e:
        logger.error(f"Error loading system prompt: {e}, using default")
        return default_prompt

import anthropic
import typer
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.theme import Theme

from chroma_client import get_chroma_client

# Custom theme for the chat
custom_theme = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "success": "bold green",
        "user": "bold blue",
        "assistant": "bold magenta",
        "context": "dim cyan",
    }
)

console = Console(theme=custom_theme)
app = typer.Typer(
    name="rag-chat",
    help="Interactive chat with RAG-powered responses",
    add_completion=False,
)


# --- Protocols for Dependency Injection ---


class CollectionProtocol(Protocol):
    """Protocol for ChromaDB collection."""

    def query(
        self,
        query_texts: list[str],
        n_results: int,
        where: dict | None,
        include: list[str],
    ) -> dict: ...


class ChromaClientProtocol(Protocol):
    """Protocol for ChromaDB client."""

    def get_collection(self, name: str) -> CollectionProtocol: ...


class AnthropicClientProtocol(Protocol):
    """Protocol for Anthropic client."""

    class Messages:
        def stream(self, **kwargs): ...

    @property
    def messages(self) -> Messages: ...


# --- Data Classes ---


@dataclass
class RAGConfig:
    """Configuration for RAG retrieval."""

    n_results: int = 5
    min_relevance: float | None = None
    filter_type: str | None = None
    filter_author: str | None = None
    include_sources: bool = True


@dataclass
class ChatContext:
    """Maintains chat state and configuration."""

    collection_name: str
    rag_config: RAGConfig
    history: list[dict] = field(default_factory=list)
    last_sources: list[dict] = field(default_factory=list)


@dataclass
class QueryResult:
    """Result from a ChromaDB query."""

    documents: list[str]
    metadatas: list[dict]


# --- Core Service Classes with DI ---


class RAGService:
    """Service for RAG retrieval operations."""

    def __init__(self, chroma_client: ChromaClientProtocol):
        self.chroma_client = chroma_client

    def query(
        self,
        query_text: str,
        collection_name: str,
        config: RAGConfig,
    ) -> QueryResult:
        """Query ChromaDB for relevant documents."""
        logger.debug(f"RAG query: '{query_text}'")
        logger.debug(f"RAG config: n_results={config.n_results}, min_relevance={config.min_relevance}, "
                     f"filter_type={config.filter_type}, filter_author={config.filter_author}")

        try:
            collection = self.chroma_client.get_collection(collection_name)
        except Exception as e:
            logger.warning(f"Failed to get collection '{collection_name}': {e}")
            return QueryResult(documents=[], metadatas=[])

        # Build where filter
        where_filter = build_where_filter(config)
        logger.debug(f"ChromaDB where filter: {where_filter}")

        # Query the collection
        results = collection.query(
            query_texts=[query_text],
            n_results=config.n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        logger.debug(f"ChromaDB returned {len(documents)} documents")
        for i, (meta, dist) in enumerate(zip(metadatas, distances)):
            logger.debug(f"  [{i}] distance={dist:.4f} source={meta.get('source_file', 'unknown')}")

        # Filter by relevance if specified
        if config.min_relevance is not None:
            documents, metadatas = filter_by_relevance(
                documents, metadatas, distances, config.min_relevance
            )
            logger.debug(f"After relevance filtering: {len(documents)} documents")

        return QueryResult(documents=documents, metadatas=metadatas)


class ChatService:
    """Service for chat operations with Anthropic."""

    def __init__(self, anthropic_client: AnthropicClientProtocol, system_prompt: str | None = None):
        self.anthropic_client = anthropic_client
        self.system_prompt = system_prompt or load_system_prompt()

    def chat(
        self,
        query: str,
        context: str,
        history: list[dict],
        model: str = DEFAULT_MODEL,
        console: Console | None = None,
    ) -> str:
        """Send a message to Anthropic and get a response."""
        system_prompt = self.system_prompt
        if context:
            system_prompt += f"\n\n{context}"

        messages = history + [{"role": "user", "content": query}]

        # Log the full request details
        logger.info(f"Sending request to Anthropic model: {model}")
        logger.debug("=" * 80)
        logger.debug("SYSTEM PROMPT:")
        logger.debug("=" * 80)
        logger.debug(system_prompt)
        logger.debug("=" * 80)
        logger.debug(f"CONVERSATION HISTORY ({len(history)} messages):")
        logger.debug("=" * 80)
        for i, msg in enumerate(history):
            logger.debug(f"[{i}] {msg['role'].upper()}: {msg['content'][:200]}{'...' if len(msg['content']) > 200 else ''}")
        logger.debug("=" * 80)
        logger.debug(f"USER QUERY: {query}")
        logger.debug("=" * 80)

        # Stream the response
        full_response = ""
        output_console = console or Console()

        with Live(console=output_console, refresh_per_second=10) as live:
            with self.anthropic_client.messages.stream(
                model=model,
                max_tokens=2048,
                system=system_prompt,
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    full_response += text
                    live.update(
                        Panel(
                            Markdown(full_response),
                            title="[assistant]Assistant[/]",
                            border_style="magenta",
                        )
                    )

        logger.debug("=" * 80)
        logger.debug("ASSISTANT RESPONSE:")
        logger.debug("=" * 80)
        logger.debug(full_response)
        logger.debug("=" * 80)
        logger.info(f"Response received: {len(full_response)} chars")

        return full_response


# --- Pure Helper Functions ---


def build_where_filter(config: RAGConfig) -> dict | None:
    """Build ChromaDB where filter from config."""
    conditions = []

    if config.filter_type:
        conditions.append({"type": config.filter_type})
    if config.filter_author:
        conditions.append({"author": config.filter_author})

    if len(conditions) == 0:
        return None
    elif len(conditions) == 1:
        return conditions[0]
    else:
        return {"$and": conditions}


def filter_by_relevance(
    documents: list[str],
    metadatas: list[dict],
    distances: list[float],
    min_relevance: float,
) -> tuple[list[str], list[dict]]:
    """Filter documents by minimum relevance score."""
    filtered_docs = []
    filtered_meta = []

    for doc, meta, dist in zip(documents, metadatas, distances):
        # ChromaDB returns L2 distance; convert to similarity
        # Lower distance = higher similarity
        similarity = 1 / (1 + dist)
        if similarity >= min_relevance:
            filtered_docs.append(doc)
            filtered_meta.append(meta)

    return filtered_docs, filtered_meta


def build_context_prompt(documents: list[str], metadatas: list[dict]) -> str:
    """Build a context string from retrieved documents."""
    if not documents:
        return ""

    context_parts = ["Here is relevant context from the knowledge base:\n"]

    for i, (doc, meta) in enumerate(zip(documents, metadatas), 1):
        title = meta.get("title", "Untitled")
        meta_json = json.dumps(meta, indent=2, default=str)

        context_parts.append(f"--- Document {i}: {title} ---")
        context_parts.append(f"Metadata: {meta_json}")
        context_parts.append(doc)
        context_parts.append("")

    return "\n".join(context_parts)


# --- UI Functions ---


def print_banner(output_console: Console | None = None):
    """Print a stylish welcome banner."""
    c = output_console or console
    banner = r"""
[bold magenta]  ____      _    ____    ____ _           _   [/]
[bold magenta] |  _ \    / \  / ___|  / ___| |__   __ _| |_ [/]
[bold magenta] | |_) |  / _ \| |  _  | |   | '_ \ / _` | __|[/]
[bold magenta] |  _ <  / ___ \ |_| | | |___| | | | (_| | |_ [/]
[bold magenta] |_| \_\/_/   \_\____|  \____|_| |_|\__,_|\__|[/]
    """
    c.print(banner)
    c.print(
        Panel(
            "[info]Chat with your knowledge base using AI-powered retrieval[/]\n\n"
            "[dim]Commands:[/]\n"
            "  [bold]/help[/]     - Show all commands\n"
            "  [bold]/config[/]   - Show current RAG settings\n"
            "  [bold]/sources[/]  - Show sources from last response\n"
            "  [bold]/clear[/]    - Clear chat history\n"
            "  [bold]/quit[/]     - Exit the chat",
            title="[bold]Welcome to RAG Chat[/]",
            border_style="magenta",
        )
    )


def print_help(output_console: Console | None = None):
    """Print help information."""
    c = output_console or console
    table = Table(title="Available Commands", border_style="cyan")
    table.add_column("Command", style="bold")
    table.add_column("Description")

    commands = [
        ("/help", "Show this help message"),
        ("/config", "Display current RAG configuration"),
        ("/sources", "Show sources used in the last response"),
        ("/set <option> <value>", "Change a RAG setting"),
        ("/clear", "Clear conversation history"),
        ("/quit, /exit, /q", "Exit the chat"),
    ]

    for cmd, desc in commands:
        table.add_row(cmd, desc)

    c.print(table)

    # Config options
    c.print("\n[bold]RAG Configuration Options (/set):[/]")
    config_table = Table(border_style="dim")
    config_table.add_column("Option", style="bold")
    config_table.add_column("Type")
    config_table.add_column("Description")

    options = [
        ("n_results", "int", "Number of documents to retrieve (default: 5)"),
        ("min_relevance", "float", "Minimum similarity score 0-1 (default: none)"),
        ("filter_type", "str", "Filter by document type (e.g., deep_dive, weekly_review)"),
        ("filter_author", "str", "Filter by author name"),
        ("include_sources", "bool", "Show source references (default: true)"),
    ]

    for opt, typ, desc in options:
        config_table.add_row(opt, typ, desc)

    c.print(config_table)


def print_config(config: RAGConfig, output_console: Console | None = None):
    """Display current RAG configuration."""
    c = output_console or console
    table = Table(title="Current RAG Configuration", border_style="cyan")
    table.add_column("Setting", style="bold")
    table.add_column("Value")

    table.add_row("n_results", str(config.n_results))
    table.add_row("min_relevance", str(config.min_relevance) if config.min_relevance else "none")
    table.add_row("filter_type", config.filter_type or "none")
    table.add_row("filter_author", config.filter_author or "none")
    table.add_row("include_sources", str(config.include_sources).lower())

    c.print(table)


def print_sources(sources: list[dict], output_console: Console | None = None):
    """Display sources from the last response."""
    c = output_console or console
    if not sources:
        c.print("[warning]No sources available. Ask a question first![/]")
        return

    table = Table(title="Sources Used", border_style="cyan")
    table.add_column("#", style="dim", width=3)
    table.add_column("Source", style="bold")
    table.add_column("Title")
    table.add_column("Type", style="dim")

    for i, src in enumerate(sources, 1):
        table.add_row(
            str(i),
            src.get("source_file", "unknown"),
            src.get("title", "untitled")[:40],
            src.get("type", ""),
        )

    c.print(table)


def handle_set_command(
    args: list[str],
    config: RAGConfig,
    output_console: Console | None = None,
) -> bool:
    """Handle /set command to update configuration."""
    c = output_console or console

    if len(args) < 2:
        c.print("[error]Usage: /set <option> <value>[/]")
        return False

    option, value = args[0].lower(), args[1]

    try:
        if option == "n_results":
            config.n_results = max(1, int(value))
        elif option == "min_relevance":
            if value.lower() == "none":
                config.min_relevance = None
            else:
                config.min_relevance = max(0.0, min(1.0, float(value)))
        elif option == "filter_type":
            config.filter_type = None if value.lower() == "none" else value
        elif option == "filter_author":
            config.filter_author = None if value.lower() == "none" else value
        elif option == "include_sources":
            config.include_sources = value.lower() in ("true", "1", "yes")
        else:
            c.print(f"[error]Unknown option: {option}[/]")
            return False

        c.print(f"[success]Set {option} = {value}[/]")
        return True
    except ValueError as e:
        c.print(f"[error]Invalid value: {e}[/]")
        return False


# --- Legacy wrapper for backwards compatibility ---


def query_chromadb(
    query: str,
    collection_name: str,
    config: RAGConfig,
    chroma_client: ChromaClientProtocol | None = None,
) -> tuple[list[str], list[dict]]:
    """Query ChromaDB for relevant documents.

    Args:
        query: The search query
        collection_name: Name of the ChromaDB collection
        config: RAG configuration
        chroma_client: Optional ChromaDB client (for DI). If None, creates default client.

    Returns:
        Tuple of (documents, metadatas)
    """
    client = chroma_client or get_chroma_client()
    service = RAGService(client)
    result = service.query(query, collection_name, config)
    return result.documents, result.metadatas


def chat_with_anthropic(
    query: str,
    context: str,
    history: list[dict],
    model: str = DEFAULT_MODEL,
    anthropic_client: AnthropicClientProtocol | None = None,
    output_console: Console | None = None,
) -> str:
    """Send a message to Anthropic and get a response.

    Args:
        query: User's question
        context: RAG context string
        history: Conversation history
        model: Anthropic model to use
        anthropic_client: Optional Anthropic client (for DI). If None, creates default client.
        output_console: Optional console for output (for DI/testing).

    Returns:
        The assistant's response
    """
    client = anthropic_client or anthropic.Anthropic()
    service = ChatService(client)
    return service.chat(query, context, history, model, output_console)


# --- CLI Commands ---


@app.command()
def chat(
    collection: str = typer.Option(
        "ac_articles",
        "--collection",
        "-c",
        help="ChromaDB collection name to query",
    ),
    n_results: int = typer.Option(
        5,
        "--n-results",
        "-n",
        help="Number of documents to retrieve per query",
    ),
    min_relevance: float | None = typer.Option(
        None,
        "--min-relevance",
        "-r",
        help="Minimum relevance score (0-1) for retrieved documents",
    ),
    filter_type: str | None = typer.Option(
        None,
        "--type",
        "-t",
        help="Filter documents by type (e.g., deep_dive, weekly_review)",
    ),
    filter_author: str | None = typer.Option(
        None,
        "--author",
        "-a",
        help="Filter documents by author",
    ),
    model: str = typer.Option(
        DEFAULT_MODEL,
        "--model",
        "-m",
        help="Anthropic model to use",
    ),
    no_banner: bool = typer.Option(
        False,
        "--no-banner",
        help="Skip the welcome banner",
    ),
):
    """Start an interactive RAG-powered chat session."""
    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[error]ANTHROPIC_API_KEY environment variable not set![/]")
        raise typer.Exit(1)

    # Initialize configuration
    rag_config = RAGConfig(
        n_results=n_results,
        min_relevance=min_relevance,
        filter_type=filter_type,
        filter_author=filter_author,
    )

    ctx = ChatContext(
        collection_name=collection,
        rag_config=rag_config,
    )

    if not no_banner:
        print_banner()

    console.print(f"\n[info]Using collection:[/] [bold]{collection}[/]")
    console.print(f"[info]Model:[/] [bold]{model}[/]\n")

    logger.info("=" * 80)
    logger.info("CHAT SESSION STARTED")
    logger.info(f"Collection: {collection}, Model: {model}")
    logger.info(f"RAG config: n_results={n_results}, min_relevance={min_relevance}, "
                f"filter_type={filter_type}, filter_author={filter_author}")
    logger.info("=" * 80)

    # Main chat loop
    while True:
        try:
            user_input = Prompt.ask("[user]You[/]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[info]Goodbye![/]")
            logger.info("Session ended (keyboard interrupt)")
            break

        user_input = user_input.strip()

        if not user_input:
            continue

        # Handle commands
        if user_input.startswith("/"):
            parts = user_input[1:].split()
            cmd = parts[0].lower() if parts else ""
            logger.debug(f"Command: /{cmd} {parts[1:] if len(parts) > 1 else ''}")

            if cmd in ("quit", "exit", "q"):
                console.print("[info]Goodbye![/]")
                logger.info("Session ended (quit command)")
                break
            elif cmd == "help":
                print_help()
            elif cmd == "config":
                print_config(ctx.rag_config)
            elif cmd == "sources":
                print_sources(ctx.last_sources)
            elif cmd == "set":
                handle_set_command(parts[1:], ctx.rag_config)
            elif cmd == "clear":
                ctx.history = []
                ctx.last_sources = []
                console.print("[success]Chat history cleared![/]")
            else:
                console.print(f"[warning]Unknown command: /{cmd}. Type /help for available commands.[/]")
            continue

        # Query ChromaDB for context
        with console.status("[info]Searching knowledge base...[/]", spinner="dots"):
            documents, metadatas = query_chromadb(
                user_input,
                ctx.collection_name,
                ctx.rag_config,
            )

        ctx.last_sources = metadatas

        if documents and ctx.rag_config.include_sources:
            console.print(f"[context]Found {len(documents)} relevant document(s)[/]")

        # Build context and get response
        context_prompt = build_context_prompt(documents, metadatas)

        try:
            response = chat_with_anthropic(
                user_input,
                context_prompt,
                ctx.history,
                model,
            )

            # Update history
            ctx.history.append({"role": "user", "content": user_input})
            ctx.history.append({"role": "assistant", "content": response})

        except anthropic.APIError as e:
            logger.error(f"Anthropic API error: {e}")
            console.print(f"[error]API Error: {e}[/]")
        except Exception as e:
            logger.exception(f"Unexpected error during chat: {e}")
            console.print(f"[error]Error: {e}[/]")

        console.print()  # Add spacing


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask"),
    collection: str = typer.Option(
        "blog_posts",
        "--collection",
        "-c",
        help="ChromaDB collection name",
    ),
    n_results: int = typer.Option(
        5,
        "--n-results",
        "-n",
        help="Number of documents to retrieve",
    ),
    model: str = typer.Option(
        DEFAULT_MODEL,
        "--model",
        "-m",
        help="Anthropic model to use",
    ),
):
    """Ask a single question and get an answer (non-interactive)."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[error]ANTHROPIC_API_KEY environment variable not set![/]")
        raise typer.Exit(1)

    rag_config = RAGConfig(n_results=n_results)

    with console.status("[info]Searching knowledge base...[/]", spinner="dots"):
        documents, metadatas = query_chromadb(question, collection, rag_config)

    if documents:
        console.print(f"[context]Found {len(documents)} relevant document(s)[/]\n")

    context_prompt = build_context_prompt(documents, metadatas)

    try:
        response = chat_with_anthropic(question, context_prompt, [], model)
        console.print()
    except anthropic.APIError as e:
        console.print(f"[error]API Error: {e}[/]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
