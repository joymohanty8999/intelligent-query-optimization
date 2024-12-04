import numpy as np

def compute_precision_recall_f1(retrieved, relevant, k=10):
    """
    Computes precision, recall, and F1-score for the top-k retrieved documents.
    
    Parameters:
        retrieved (list): List of retrieved document IDs.
        relevant (set): Set of relevant document IDs.
        k (int): Number of top results to evaluate.
        
    Returns:
        tuple: Precision, recall, and F1-score.
    """
    retrieved_at_k = retrieved[:k]
    relevant_retrieved = set(retrieved_at_k).intersection(relevant)
    
    precision = len(relevant_retrieved) / len(retrieved_at_k) if retrieved_at_k else 0
    recall = len(relevant_retrieved) / len(relevant) if relevant else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1_score


def compute_recall_at_k(query_embeddings, vector_db, qrels, k=10):
    """
    Computes Recall@k for the given queries and relevance judgments.
    
    Parameters:
        query_embeddings (np.ndarray): Embeddings of the queries.
        vector_db (VectorDatabase): Vector database to perform nearest neighbor search.
        qrels (pd.DataFrame): DataFrame containing QueryID, PassageID, and Relevance.
        k (int): Number of top results to consider for recall computation.
    
    Returns:
        float: Average Recall@k across all queries.
    """
    # Validate columns in QRELs
    expected_columns = {"QueryID", "PassageID", "Relevance"}
    missing_columns = expected_columns - set(qrels.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in QRELs file: {missing_columns}. Found columns: {qrels.columns}")
    
    recalls = []
    for idx, query_embedding in enumerate(query_embeddings):
        # Get relevant passages for the query
        relevant_passages = set(qrels[qrels['QueryID'] == idx]['PassageID'].tolist())
        _, indices = vector_db.query(query_embedding, top_k=k)
        retrieved_passages = set(indices.flatten())
        
        # Compute recall
        recall = len(relevant_passages & retrieved_passages) / len(relevant_passages) if relevant_passages else 0
        recalls.append(recall)
    return sum(recalls) / len(recalls) if recalls else 0