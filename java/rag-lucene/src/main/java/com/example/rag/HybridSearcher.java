package com.example.rag;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;

import java.io.Closeable;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeSet;

/**
 * Performs hybrid search that combines BM25 scores and vector similarity per tenant.
 */
public final class HybridSearcher implements Closeable {

    private final Path indexPath;
    private final Analyzer analyzer;
    private final EmbeddingFunction embeddingFunction;

    public HybridSearcher(Path indexPath, EmbeddingFunction embeddingFunction) {
        this(indexPath, new StandardAnalyzer(), embeddingFunction);
    }

    public HybridSearcher(Path indexPath, Analyzer analyzer, EmbeddingFunction embeddingFunction) {
        this.indexPath = Objects.requireNonNull(indexPath, "indexPath");
        this.analyzer = Objects.requireNonNull(analyzer, "analyzer");
        this.embeddingFunction = Objects.requireNonNull(embeddingFunction, "embeddingFunction");
    }

    public List<SearchResult> search(String tenantId, String queryText, int topK, float alpha)
            throws IOException, ParseException {
        if (alpha < 0f || alpha > 1f) {
            throw new IllegalArgumentException("alpha must be between 0 and 1");
        }
        try (Directory directory = FSDirectory.open(indexPath); DirectoryReader reader = DirectoryReader.open(directory)) {
            IndexSearcher searcher = new IndexSearcher(reader);

            QueryParser parser = new QueryParser(RagIndexer.FIELD_TEXT, analyzer);
            Query lexicalQuery = parser.parse(QueryParser.escape(queryText));
            Query tenantFilter = new TermQuery(new Term(RagIndexer.FIELD_TENANT, tenantId));

            BooleanQuery hybridLexical = new BooleanQuery.Builder()
                    .add(lexicalQuery, BooleanClause.Occur.MUST)
                    .add(tenantFilter, BooleanClause.Occur.FILTER)
                    .build();
            TopDocs lexicalDocs = searcher.search(hybridLexical, topK);

            float[] queryVector = embeddingFunction.embed(queryText);
            KnnFloatVectorQuery vectorQuery = new KnnFloatVectorQuery(RagIndexer.FIELD_VECTOR, queryVector, topK, tenantFilter);
            TopDocs vectorDocs = searcher.search(vectorQuery, topK);

            Map<Integer, Float> lexicalScores = collectScores(lexicalDocs);
            Map<Integer, Float> vectorScores = collectScores(vectorDocs);

            float lexicalMax = lexicalScores.values().stream().map(Math::abs).max(Float::compare).orElse(1f);
            float vectorMax = vectorScores.values().stream().map(Math::abs).max(Float::compare).orElse(1f);

            Set<Integer> docIds = new TreeSet<>();
            docIds.addAll(lexicalScores.keySet());
            docIds.addAll(vectorScores.keySet());

            List<SearchResult> results = new ArrayList<>();
            for (int docId : docIds) {
                float lexicalScore = lexicalScores.getOrDefault(docId, 0f);
                float vectorScore = vectorScores.getOrDefault(docId, 0f);
                float lexicalNorm = lexicalMax == 0f ? 0f : lexicalScore / lexicalMax;
                float vectorNorm = vectorMax == 0f ? 0f : vectorScore / vectorMax;
                float hybridScore = alpha * lexicalNorm + (1f - alpha) * vectorNorm;

                Document doc = searcher.doc(docId);
                results.add(new SearchResult(
                        doc.get(RagIndexer.FIELD_ID),
                        doc.get(RagIndexer.FIELD_TENANT),
                        doc.get(RagIndexer.FIELD_TEXT),
                        deserializeMetadata(doc.get(RagIndexer.FIELD_METADATA)),
                        lexicalScore,
                        vectorScore,
                        hybridScore
                ));
            }

            results.sort((a, b) -> Float.compare(b.hybridScore(), a.hybridScore()));
            return results.size() > topK ? results.subList(0, topK) : results;
        }
    }

    private static Map<Integer, Float> collectScores(TopDocs topDocs) {
        Map<Integer, Float> scores = new HashMap<>();
        if (topDocs == null || topDocs.scoreDocs == null) {
            return scores;
        }
        for (ScoreDoc sd : topDocs.scoreDocs) {
            scores.put(sd.doc, sd.score);
        }
        return scores;
    }

    private static Map<String, String> deserializeMetadata(String json) {
        if (json == null) {
            return Map.of();
        }
        if (json.length() <= 2) {
            return Map.of();
        }
        Map<String, String> metadata = new HashMap<>();
        // Very small JSON parser matching the serializer in RagIndexer
        String body = json.substring(1, json.length() - 1);
        for (String entry : body.split(",")) {
            if (entry.isBlank()) {
                continue;
            }
            String[] kv = entry.split(":", 2);
            if (kv.length == 2) {
                metadata.put(unescape(stripQuotes(kv[0])), unescape(stripQuotes(kv[1])));
            }
        }
        return metadata;
    }

    private static String stripQuotes(String value) {
        String trimmed = value.trim();
        if (trimmed.startsWith("\"") && trimmed.endsWith("\"")) {
            return trimmed.substring(1, trimmed.length() - 1);
        }
        return trimmed;
    }

    private static String unescape(String value) {
        return value.replace("\\\"", "\"").replace("\\\\", "\\");
    }

    @Override
    public void close() throws IOException {
        // nothing to close, directory readers are scoped per call
    }
}
