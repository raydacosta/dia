package com.example.rag;

/**
 * Functional interface that provides dense embeddings for free-form text.
 */
@FunctionalInterface
public interface EmbeddingFunction {

    /**
     * Computes a dense vector embedding for the provided text.
     *
     * @param text query or document text
     * @return embedding array; implementors should always return the same dimension
     */
    float[] embed(String text);

    /**
     * Returns the dimensionality of the produced embeddings.
     *
     * @return number of dimensions
     */
    default int dimension() {
        throw new UnsupportedOperationException("Embedding dimension must be provided by the implementation");
    }
}
