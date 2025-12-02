# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A RAG (Retrieval-Augmented Generation) system for ingesting markdown blog posts into ChromaDB. It parses markdown files with YAML frontmatter, chunks them using a sliding window tokenizer, and stores them in a vector database for semantic search.

## Commands

```bash
# Install dependencies
uv sync

# Run tests with coverage
uv run pytest

# Ingest markdown files into ChromaDB
uv run python ingest.py ./articles [collection_name]

# Sync articles from RSS feed
uv run python scripts/sync_articles.py
```

## Architecture

**chroma_client.py** - Factory function `get_chroma_client()` that returns either a `PersistentClient` (local storage in `.chromadb/`) or `CloudClient` based on environment variables.

**ingest.py** - Main ingestion pipeline:
- Uses `python-frontmatter` to parse markdown with YAML metadata
- Tokenizes with `sentence-transformers/all-MiniLM-L6-v2` (same model as ChromaDB's default embeddings)
- `chunk_by_tokens()` creates overlapping chunks (256 tokens max, 50 token stride) using the tokenizer's truncation/overflow feature
- Each chunk is stored with metadata: `source_file`, `chunk_index`, `total_chunks`, plus frontmatter fields

**scripts/sync_articles.py** - RSS sync from alteredcraft.com. Converts HTML to markdown using BeautifulSoup + markdownify.

## Environment Variables

- `CHROMA_CLIENT_TYPE` - `persistent` (default) or `cloud`
- `CHROMA_PERSIST_PATH` - Local storage path (default: `./.chromadb`)
- `CHROMA_TENANT`, `CHROMA_DATABASE`, `CHROMA_API_KEY` - For cloud mode

## [IMPORTANT] Operating directives

- After completing a coding task, you MUST run tests using `uv run pytest` and all the tests must pass. 
- If test coverage is below 80%, ask the user if they would like you to increase test coverage to get greater than 80% 
