package com.example.rag;

import java.util.Map;

/**
 * Immutable search result combining lexical and dense scores.
 */
public record SearchResult(
        String documentId,
        String tenantId,
        String text,
        Map<String, String> metadata,
        float lexicalScore,
        float vectorScore,
        float hybridScore
) { }
