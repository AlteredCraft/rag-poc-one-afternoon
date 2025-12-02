---
title: Test Document for Chunking
author: Test Author
date: 2024-01-01
---

# Introduction

This is a test document used to validate the chunking functionality. We need enough content to generate multiple chunks with the expected overlap.

## Section One

The first section contains information about the chunking algorithm. We use a sliding window approach with a maximum of 256 tokens per chunk and a stride of 50 tokens for overlap between consecutive chunks.

This ensures that context is preserved across chunk boundaries, which is important for semantic search and retrieval-augmented generation systems.

## Section Two

The second section discusses the importance of proper chunking. When documents are too large for embedding models, we must split them into smaller pieces. However, we don't want to lose important context at the boundaries.

By overlapping chunks, we ensure that sentences or concepts that might be split between chunks are still fully represented in at least one chunk. This improves the quality of search results.

## Section Three

The third section provides additional context about token counting. We use the same tokenizer as ChromaDB's default embedding function: all-MiniLM-L6-v2. This model has an optimal input size of 256 tokens.

Chunks that exceed this size will have their embeddings truncated, meaning the content at the end won't be searchable. Our chunking strategy prevents this issue.

## Section Four

This is the fourth section with more content to ensure we get enough chunks for testing. We want to verify that the chunking function correctly splits the document and maintains the expected overlap.

The overlap is measured in tokens, not characters. With a stride of 50, we expect approximately 50 tokens to be shared between consecutive chunks.

## Conclusion

This concludes our test document. It should generate several chunks when processed with our chunking function, allowing us to validate both the token count limits and the stride overlap.
