import logging
import numpy as np

def experiment_query_difficulty(query_embeddings, vector_db, easy_threshold=0.8, hard_threshold=0.5):
    """
    Analyzes query difficulty based on retrieval scores.
    
    Parameters:
        query_embeddings (np.ndarray): Query embeddings.
        vector_db (VectorDatabase): Vector database for document retrieval.
        easy_threshold (float): Threshold for easy queries.
        hard_threshold (float): Threshold for hard queries.
        
    Returns:
        tuple: Lists of easy and hard queries.
    """
    easy_queries = []
    hard_queries = []

    for idx, query in enumerate(query_embeddings):
        distances, _ = vector_db.query(query, top_k=10)
        avg_distance = np.mean(distances)

        if avg_distance >= easy_threshold:
            easy_queries.append(idx)
        elif avg_distance <= hard_threshold:
            hard_queries.append(idx)

    logging.info(f"Query Difficulty Analysis: Easy={len(easy_queries)}, Hard={len(hard_queries)}")
    return easy_queries, hard_queries