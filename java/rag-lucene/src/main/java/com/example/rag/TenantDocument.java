package com.example.rag;

import java.util.Map;
import java.util.Objects;

/**
 * Immutable representation of a RAG source document belonging to a tenant.
 */
public record TenantDocument(
        String tenantId,
        String documentId,
        String text,
        float[] embedding,
        Map<String, String> metadata
) {
    public TenantDocument {
        Objects.requireNonNull(tenantId, "tenantId is required");
        Objects.requireNonNull(documentId, "documentId is required");
        Objects.requireNonNull(text, "text is required");
        Objects.requireNonNull(metadata, "metadata must not be null");
    }

    public TenantDocument(String tenantId, String documentId, String text, float[] embedding) {
        this(tenantId, documentId, text, embedding, Map.of());
    }

    public TenantDocument withEmbedding(float[] newEmbedding) {
        return new TenantDocument(tenantId, documentId, text, newEmbedding, metadata);
    }
}
