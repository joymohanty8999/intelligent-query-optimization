def experiment_bloom_filter(queries):
    """
    Evaluates cache hit rate for different Bloom Filter configurations.
    """
    from src.bloom_filter import QueryCache

    capacities = [len(queries) * 2, len(queries) * 5, len(queries) * 10]
    error_rates = [0.01, 0.05, 0.1]
    results = []

    for capacity in capacities:
        for error_rate in error_rates:
            cache = QueryCache(capacity=capacity, error_rate=error_rate)
            hit_count = 0
            for query in queries:
                if cache.check_and_insert(query):
                    hit_count += 1

            hit_rate = hit_count / len(queries)
            results.append((capacity, error_rate, hit_rate))

    return results