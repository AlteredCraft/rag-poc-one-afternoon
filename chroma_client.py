"""ChromaDB client factory with support for persistent and cloud clients."""

import os

import chromadb
from chromadb import CloudClient, PersistentClient


def get_chroma_client(
    client_type: str | None = None,
    persist_path: str = "./.chromadb",
    tenant: str | None = None,
    database: str | None = None,
    api_key: str | None = None,
) -> chromadb.ClientAPI:
    """
    Create a ChromaDB client based on configuration.

    Args:
        client_type: "persistent" or "cloud". Defaults to CHROMA_CLIENT_TYPE env var or "persistent"
        persist_path: Path for local persistent storage. Defaults to CHROMA_PERSIST_PATH env var or "./.chromadb"
        tenant: Chroma Cloud tenant ID. Defaults to CHROMA_TENANT env var
        database: Chroma Cloud database name. Defaults to CHROMA_DATABASE env var
        api_key: Chroma Cloud API key. Defaults to CHROMA_API_KEY env var

    Returns:
        A ChromaDB client instance

    Raises:
        ValueError: If client_type is not "persistent" or "cloud"
    """
    client_type = client_type or os.getenv("CHROMA_CLIENT_TYPE", "persistent")
    persist_path = persist_path or os.getenv("CHROMA_PERSIST_PATH", "./.chromadb")

    if client_type == "persistent":
        return PersistentClient(path=persist_path)
    elif client_type == "cloud":
        return CloudClient(
            tenant=tenant or os.getenv("CHROMA_TENANT"),
            database=database or os.getenv("CHROMA_DATABASE"),
            api_key=api_key or os.getenv("CHROMA_API_KEY"),
        )
    else:
        raise ValueError(f"Unknown client type: {client_type}. Use 'persistent' or 'cloud'.")
