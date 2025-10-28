package com.example.rag;

import org.apache.lucene.queryparser.classic.ParseException;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/**
 * Minimal runnable example that demonstrates the hybrid Lucene RAG stack.
 */
public final class RagDemoApp {

    private RagDemoApp() {
    }

    public static void main(String[] args) throws IOException, ParseException {
        Path indexPath = Path.of("build/lucene-index");
        EmbeddingFunction embeddingFunction = new HashingEmbeddingFunction(128);

        List<TenantDocument> documents = List.of(
                new TenantDocument("tenant-a", "1", "Java 21 introduced virtual threads for easier concurrency.", null,
                        Map.of("source", "blog", "language", "en")),
                new TenantDocument("tenant-a", "2", "Lucene 10 adds faster vector search capabilities.", null,
                        Map.of("source", "release-notes")),
                new TenantDocument("tenant-b", "1", "A apple a day keeps the doctor away.", null,
                        Map.of("source", "proverb")),
                new TenantDocument("tenant-b", "2", "Hybrid search combines dense and sparse retrieval for better recall.", null,
                        Map.of("source", "whitepaper"))
        );

        try (RagIndexer indexer = new RagIndexer(indexPath, embeddingFunction)) {
            indexer.rebuildIndex(documents, 2);
        }

        try (HybridSearcher searcher = new HybridSearcher(indexPath, embeddingFunction)) {
            List<SearchResult> results = searcher.search("tenant-a", "Como melhorar busca híbrida em Java?", 3, 0.5f);
            System.out.println("Resultados para tenant-a:");
            for (SearchResult result : results) {
                System.out.printf("doc=%s score=%.3f (lex=%.3f vec=%.3f) source=%s%n",
                        result.documentId(),
                        result.hybridScore(),
                        result.lexicalScore(),
                        result.vectorScore(),
                        result.metadata().getOrDefault("source", "n/a"));
            }
        }
    }
}
