package com.example.rag;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Locale;
import java.util.StringTokenizer;

/**
 * A lightweight embedding function that deterministically hashes tokens into a dense vector.
 * This implementation does not aim to provide semantically rich embeddings but keeps the
 * example self-contained without additional model dependencies.
 */
public final class HashingEmbeddingFunction implements EmbeddingFunction {

    private final int dimension;

    public HashingEmbeddingFunction(int dimension) {
        if (dimension <= 0) {
            throw new IllegalArgumentException("dimension must be > 0");
        }
        this.dimension = dimension;
    }

    @Override
    public float[] embed(String text) {
        float[] vector = new float[dimension];
        if (text == null || text.isBlank()) {
            return vector;
        }

        MessageDigest digest = messageDigest();
        StringTokenizer tokenizer = new StringTokenizer(text.toLowerCase(Locale.ROOT));
        while (tokenizer.hasMoreTokens()) {
            String token = tokenizer.nextToken();
            byte[] hash = digest.digest(token.getBytes(StandardCharsets.UTF_8));
            int bucket = Math.floorMod(bytesToInt(hash), dimension);
            vector[bucket] += 1f;
        }

        // Normalize vector length to unit magnitude to better match cosine similarity.
        float magnitude = 0f;
        for (float value : vector) {
            magnitude += value * value;
        }
        magnitude = (float) Math.sqrt(magnitude);
        if (magnitude > 0f) {
            for (int i = 0; i < vector.length; i++) {
                vector[i] /= magnitude;
            }
        }

        return vector;
    }

    @Override
    public int dimension() {
        return dimension;
    }

    private static MessageDigest messageDigest() {
        try {
            return MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 must be available", e);
        }
    }

    private static int bytesToInt(byte[] bytes) {
        int value = 0;
        for (int i = 0; i < 4 && i < bytes.length; i++) {
            value = (value << 8) | (bytes[i] & 0xFF);
        }
        return value;
    }
}
