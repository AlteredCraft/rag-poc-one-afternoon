# Chroma RAG

A simple RAG (Retrieval-Augmented Generation) system for ingesting markdown blog posts into ChromaDB. Supports both local persistent storage and Chroma Cloud.

## Features

- Ingest markdown files with YAML frontmatter
- Section-based chunking (splits on `##` and `###` headers)
- Switchable storage backend (local or cloud) via environment variables
- ChromaDB's default embedding model (all-MiniLM-L6-v2)
- RSS sync for pulling new articles from alteredcraft.com

## Installation

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Tests

```bash
uv run pytest
```

## Usage

### Sync Articles from RSS

```bash
# Pull new articles from alteredcraft.com RSS feed
uv run scripts/sync_articles.py
```

### Ingest Markdown Files

```bash
# Ingest all .md files from articles directory
uv run ingest.py ./articles

# Specify a custom collection name
uv run ingest.py ./articles my_collection
```

### Query the Database

```python
from chroma_client import get_chroma_client

client = get_chroma_client()
collection = client.get_collection("blog_posts")

results = collection.query(
    query_texts=["How do I build an AI agent?"],
    n_results=3
)

for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"Source: {meta['source_file']}")
    print(f"Content: {doc[:200]}...")
```

## Configuration

By default, data is stored locally in `./.chromadb`. To use Chroma Cloud:

```bash
export CHROMA_CLIENT_TYPE=cloud
export CHROMA_TENANT=your-tenant-id
export CHROMA_DATABASE=your-database-name
export CHROMA_API_KEY=your-api-key

uv run python ingest.py ./articles
```

| Variable | Description | Default |
|----------|-------------|---------|
| `CHROMA_CLIENT_TYPE` | `persistent` or `cloud` | `persistent` |
| `CHROMA_PERSIST_PATH` | Local storage path | `./.chromadb` |
| `CHROMA_TENANT` | Cloud tenant ID | - |
| `CHROMA_DATABASE` | Cloud database name | - |
| `CHROMA_API_KEY` | Cloud API key | - |

## Project Structure

```
.
├── chroma_client.py    # Client factory (persistent/cloud)
├── ingest.py           # Markdown ingestion logic
├── sync_articles.py    # Sync new articles from RSS feed
├── articles/           # Markdown articles
├── tests/              # Unit tests
├── .chromadb/          # Local database (auto-created)
└── pyproject.toml
```

## Markdown Format

Articles use YAML frontmatter:

```markdown
---
author: Sam Keen
publish_date: January 15, 2024
title: My Article
type: weekly_review
url: https://alteredcraft.com/p/my-article
---

# My Article

Introduction...

## Section One

Content...
```

The ingester extracts frontmatter as metadata and chunks content by headers.
