from chroma_client import get_chroma_client

client = get_chroma_client()
collection = client.get_collection("test_00")

results = collection.query(
    query_texts=["How do I install Python?"],
    n_results=3
)

for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"Source: {meta['source_file']}")
    print(f"Content: {doc[:200]}...")
    print()

