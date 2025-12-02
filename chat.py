#!/usr/bin/env python3
"""Interactive CLI chat with RAG-powered responses using Anthropic and ChromaDB."""

import os
from dataclasses import dataclass, field

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


def print_banner():
    """Print a stylish welcome banner."""
    banner = r"""
[bold magenta]  ____      _    ____    ____ _           _   [/]
[bold magenta] |  _ \    / \  / ___|  / ___| |__   __ _| |_ [/]
[bold magenta] | |_) |  / _ \| |  _  | |   | '_ \ / _` | __|[/]
[bold magenta] |  _ <  / ___ \ |_| | | |___| | | | (_| | |_ [/]
[bold magenta] |_| \_\/_/   \_\____|  \____|_| |_|\__,_|\__|[/]
    """
    console.print(banner)
    console.print(
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


def print_help():
    """Print help information."""
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

    console.print(table)

    # Config options
    console.print("\n[bold]RAG Configuration Options (/set):[/]")
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

    console.print(config_table)


def print_config(config: RAGConfig):
    """Display current RAG configuration."""
    table = Table(title="Current RAG Configuration", border_style="cyan")
    table.add_column("Setting", style="bold")
    table.add_column("Value")

    table.add_row("n_results", str(config.n_results))
    table.add_row("min_relevance", str(config.min_relevance) if config.min_relevance else "none")
    table.add_row("filter_type", config.filter_type or "none")
    table.add_row("filter_author", config.filter_author or "none")
    table.add_row("include_sources", str(config.include_sources).lower())

    console.print(table)


def print_sources(sources: list[dict]):
    """Display sources from the last response."""
    if not sources:
        console.print("[warning]No sources available. Ask a question first![/]")
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

    console.print(table)


def handle_set_command(args: list[str], config: RAGConfig) -> bool:
    """Handle /set command to update configuration."""
    if len(args) < 2:
        console.print("[error]Usage: /set <option> <value>[/]")
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
            console.print(f"[error]Unknown option: {option}[/]")
            return False

        console.print(f"[success]Set {option} = {value}[/]")
        return True
    except ValueError as e:
        console.print(f"[error]Invalid value: {e}[/]")
        return False


def query_chromadb(query: str, collection_name: str, config: RAGConfig) -> tuple[list[str], list[dict]]:
    """Query ChromaDB for relevant documents."""
    client = get_chroma_client()

    try:
        collection = client.get_collection(collection_name)
    except Exception:
        console.print(f"[error]Collection '{collection_name}' not found![/]")
        console.print("[info]Run 'uv run python ingest.py <directory>' to ingest documents first.[/]")
        return [], []

    # Build where filter
    where_filter = None
    conditions = []

    if config.filter_type:
        conditions.append({"type": config.filter_type})
    if config.filter_author:
        conditions.append({"author": config.filter_author})

    if len(conditions) == 1:
        where_filter = conditions[0]
    elif len(conditions) > 1:
        where_filter = {"$and": conditions}

    # Query the collection
    results = collection.query(
        query_texts=[query],
        n_results=config.n_results,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    # Filter by relevance if specified
    if config.min_relevance is not None:
        filtered_docs = []
        filtered_meta = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            # ChromaDB returns L2 distance; convert to similarity
            # Lower distance = higher similarity
            similarity = 1 / (1 + dist)
            if similarity >= config.min_relevance:
                filtered_docs.append(doc)
                filtered_meta.append(meta)
        documents = filtered_docs
        metadatas = filtered_meta

    return documents, metadatas


def build_context_prompt(documents: list[str], metadatas: list[dict]) -> str:
    """Build a context string from retrieved documents."""
    if not documents:
        return ""

    context_parts = ["Here is relevant context from the knowledge base:\n"]

    for i, (doc, meta) in enumerate(zip(documents, metadatas), 1):
        source = meta.get("source_file", "unknown")
        title = meta.get("title", "Untitled")
        context_parts.append(f"--- Document {i}: {title} (source: {source}) ---")
        context_parts.append(doc)
        context_parts.append("")

    return "\n".join(context_parts)


def chat_with_anthropic(
    query: str,
    context: str,
    history: list[dict],
    model: str = "claude-sonnet-4-20250514",
) -> str:
    """Send a message to Anthropic and get a response."""
    client = anthropic.Anthropic()

    system_prompt = """You are a helpful assistant that answers questions based on the provided context from a knowledge base.

When answering:
- Use the provided context to give accurate, relevant answers
- If the context doesn't contain enough information, say so clearly
- Be concise but thorough
- Reference specific sources when relevant
- If you're unsure, express uncertainty rather than making things up"""

    if context:
        system_prompt += f"\n\n{context}"

    messages = history + [{"role": "user", "content": query}]

    # Stream the response
    full_response = ""
    with Live(console=console, refresh_per_second=10) as live:
        with client.messages.stream(
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

    return full_response


@app.command()
def chat(
    collection: str = typer.Option(
        "blog_posts",
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
        "claude-sonnet-4-20250514",
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

    # Main chat loop
    while True:
        try:
            user_input = Prompt.ask("[user]You[/]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[info]Goodbye![/]")
            break

        user_input = user_input.strip()

        if not user_input:
            continue

        # Handle commands
        if user_input.startswith("/"):
            parts = user_input[1:].split()
            cmd = parts[0].lower() if parts else ""

            if cmd in ("quit", "exit", "q"):
                console.print("[info]Goodbye![/]")
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
            console.print(f"[error]API Error: {e}[/]")
        except Exception as e:
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
        "claude-sonnet-4-20250514",
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
