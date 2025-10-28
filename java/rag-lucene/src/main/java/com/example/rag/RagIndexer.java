package com.example.rag;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.Term;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import java.io.Closeable;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Handles windowed rebuilds of a Lucene index that stores lexical and vector representations.
 */
public final class RagIndexer implements Closeable {

    public static final String FIELD_TENANT = "tenantId";
    public static final String FIELD_ID = "documentId";
    public static final String FIELD_TEXT = "text";
    public static final String FIELD_VECTOR = "embedding";
    public static final String FIELD_METADATA = "metadata";
    public static final String FIELD_UNIQUE_ID = "tenantDocId";

    private final Path indexPath;
    private final Analyzer analyzer;
    private final EmbeddingFunction embeddingFunction;
    private final int expectedDimension;

    public RagIndexer(Path indexPath, EmbeddingFunction embeddingFunction) {
        this(indexPath, new StandardAnalyzer(), embeddingFunction);
    }

    public RagIndexer(Path indexPath, Analyzer analyzer, EmbeddingFunction embeddingFunction) {
        this.indexPath = Objects.requireNonNull(indexPath, "indexPath");
        this.analyzer = Objects.requireNonNull(analyzer, "analyzer");
        this.embeddingFunction = Objects.requireNonNull(embeddingFunction, "embeddingFunction");
        this.expectedDimension = resolveDimension(embeddingFunction);
    }

    public void rebuildIndex(List<TenantDocument> documents, int windowSize) throws IOException {
        if (windowSize <= 0) {
            throw new IllegalArgumentException("windowSize must be positive");
        }
        Objects.requireNonNull(documents, "documents");

        Files.createDirectories(indexPath);
        try (Directory directory = FSDirectory.open(indexPath);
             IndexWriter writer = new IndexWriter(directory, indexWriterConfig(IndexWriterConfig.OpenMode.CREATE))) {
            List<List<TenantDocument>> windows = partition(documents, windowSize);
            for (List<TenantDocument> window : windows) {
                indexWindow(writer, window);
                writer.commit();
            }
        }
    }

    public void upsertDocuments(List<TenantDocument> documents) throws IOException {
        Objects.requireNonNull(documents, "documents");
        Files.createDirectories(indexPath);
        try (Directory directory = FSDirectory.open(indexPath);
             IndexWriter writer = new IndexWriter(directory, indexWriterConfig(IndexWriterConfig.OpenMode.CREATE_OR_APPEND))) {
            indexWindow(writer, documents);
            writer.commit();
        }
    }

    private IndexWriterConfig indexWriterConfig(IndexWriterConfig.OpenMode mode) {
        IndexWriterConfig config = new IndexWriterConfig(analyzer);
        config.setOpenMode(mode);
        return config;
    }

    private void indexWindow(IndexWriter writer, List<TenantDocument> window) throws IOException {
        for (TenantDocument doc : window) {
            Document luceneDoc = new Document();
            luceneDoc.add(new StringField(FIELD_TENANT, doc.tenantId(), Field.Store.YES));
            luceneDoc.add(new StringField(FIELD_ID, doc.documentId(), Field.Store.YES));
            luceneDoc.add(new StringField(FIELD_UNIQUE_ID, uniqueId(doc), Field.Store.YES));
            luceneDoc.add(new TextField(FIELD_TEXT, doc.text(), Field.Store.YES));
            luceneDoc.add(new StoredField(FIELD_METADATA, serializeMetadata(doc.metadata())));

            float[] embedding = doc.embedding();
            if (embedding == null) {
                embedding = embeddingFunction.embed(doc.text());
            }
            if (expectedDimension > 0 && embedding.length != expectedDimension) {
                throw new IllegalArgumentException("Embedding dimension mismatch for document " + doc.documentId());
            }
            luceneDoc.add(new KnnFloatVectorField(FIELD_VECTOR, embedding));

            writer.updateDocument(new Term(FIELD_UNIQUE_ID, uniqueId(doc)), luceneDoc);
        }
    }

    private static String uniqueId(TenantDocument doc) {
        return doc.tenantId() + "::" + doc.documentId();
    }

    private static int resolveDimension(EmbeddingFunction embeddingFunction) {
        try {
            return embeddingFunction.dimension();
        } catch (UnsupportedOperationException ignored) {
            return -1;
        }
    }

    private static List<List<TenantDocument>> partition(List<TenantDocument> documents, int windowSize) {
        Objects.requireNonNull(documents, "documents");
        List<List<TenantDocument>> windows = new ArrayList<>();
        if (documents.isEmpty()) {
            return windows;
        }
        for (int start = 0; start < documents.size(); start += windowSize) {
            int end = Math.min(start + windowSize, documents.size());
            windows.add(new ArrayList<>(documents.subList(start, end)));
        }
        return windows;
    }

    private static String serializeMetadata(Map<String, String> metadata) {
        if (metadata == null || metadata.isEmpty()) {
            return "{}";
        }
        StringBuilder builder = new StringBuilder("{");
        boolean first = true;
        for (Map.Entry<String, String> entry : metadata.entrySet()) {
            if (!first) {
                builder.append(',');
            }
            builder.append('"').append(escape(entry.getKey())).append('"')
                    .append(':')
                    .append('"').append(escape(entry.getValue())).append('"');
            first = false;
        }
        builder.append('}');
        return builder.toString();
    }

    private static String escape(String value) {
        return value.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    @Override
    public void close() throws IOException {
        // No resources are kept open between operations
    }
}
