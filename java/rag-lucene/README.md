# Lucene Hybrid RAG Demo (Java 21)

This module contains a small Java 21 application that showcases how to build a Retrieval Augmented Generation (RAG)
stack using Apache Lucene 10.3.1. The example focuses on:

* **Hybrid retrieval** – every document is indexed with BM25 friendly text fields and dense vectors stored in a `KnnFloatVectorField`.
* **Tenant filtering** – all lookups include a mandatory tenant clause so each customer only sees their own content.
* **Windowed rebuilds** – the `RagIndexer` can rebuild the entire index in configurable windows so that large datasets can
  be streamed without exhausting heap memory.

## Structure

* `RagIndexer` – handles index rebuilds and incremental upserts. Documents contain lexical text, metadata and vector
  embeddings.
* `HybridSearcher` – merges BM25 and KNN vector scores into a single hybrid ranking.
* `HashingEmbeddingFunction` – deterministic embedding generator used in the sample to avoid external model
  dependencies. Swap this implementation for your actual embedding service.
* `RagDemoApp` – wiring that indexes a few demo documents and executes a tenant constrained hybrid query.

## Running the demo

```bash
cd java/rag-lucene
mvn -q package
java -jar target/rag-lucene-1.0.0-SNAPSHOT-shaded.jar
```

The program writes a Lucene index to `build/lucene-index/` and prints the hybrid ranked results for one tenant.

## Integrating into your project

1. Replace `HashingEmbeddingFunction` with a component that calls your embedding model (OpenAI, HuggingFace, etc.).
2. Feed `RagIndexer` with batched windows from your data lake. Each window is committed independently, which makes it easy
   to retry failed batches without repeating completed ones.
3. Use `HybridSearcher.search(...)` at query time to retrieve top ranked passages per tenant. Combine the resulting text
   chunks with your LLM prompt to build the final answer.

Feel free to adjust the hybrid weight (`alpha`) to balance BM25 relevance and vector similarity for your dataset.
