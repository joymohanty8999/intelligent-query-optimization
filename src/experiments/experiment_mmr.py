import logging
import numpy as np

def experiment_mmr(query_embeddings, document_embeddings, k=10, lambda_val=0.5):
    """
    Computes Maximal Marginal Relevance (MMR) for query-document pairs.
    
    Parameters:
        query_embeddings (np.ndarray): Query embeddings.
        document_embeddings (np.ndarray): Document embeddings.
        k (int): Number of documents to retrieve.
        lambda_val (float): Trade-off parameter between relevance and diversity.
        
    Returns:
        list: Selected document indices for the first query.
    """
    selected_indices = []
    remaining_indices = list(range(len(document_embeddings)))

    query = query_embeddings[0]  # Take the first query
    while len(selected_indices) < k and remaining_indices:
        scores = []
        for doc_idx in remaining_indices:
            relevance = np.dot(query, document_embeddings[doc_idx])
            diversity = max(
                np.dot(document_embeddings[doc_idx], document_embeddings[selected])
                for selected in selected_indices
            ) if selected_indices else 0

            mmr_score = lambda_val * relevance - (1 - lambda_val) * diversity
            scores.append((doc_idx, mmr_score))

        best_doc = max(scores, key=lambda x: x[1])[0]
        selected_indices.append(best_doc)
        remaining_indices.remove(best_doc)

    return selected_indices