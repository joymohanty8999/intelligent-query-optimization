def experiment_latency(queries, vector_db, batch_size=10):
    """
    Measures latency for querying embeddings in a vector database.

    Parameters:
        queries (np.ndarray): Array of query embeddings.
        vector_db (VectorDatabase): A vector database object.
        batch_size (int): Number of queries per batch.

    Returns:
        float: Average latency for querying the database.
    """
    import time

    latencies = []
    for i in range(0, len(queries), batch_size):
        batch_queries = queries[i:i + batch_size]
        start_time = time.perf_counter()
        for query in batch_queries:
            vector_db.query(query, top_k=10)  # Query the vector DB
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) / len(batch_queries))  # Per query latency

    avg_latency = sum(latencies) / len(latencies)
    return avg_latency