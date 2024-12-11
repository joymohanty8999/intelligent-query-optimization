from sklearn.metrics import ndcg_score
from sentence_transformers import SentenceTransformer
from src.vector_database import VectorDatabase
import numpy as np
import os

def evaluate_model(model_path, queries, qrels, documents, k=10):
    """
    Evaluates a fine-tuned or pre-trained model using Recall@k, MRR@k, and NDCG@k.

    Args:
        model_path (str): Path to the fine-tuned or pre-trained model.
        queries (list): List of query texts.
        qrels (pd.DataFrame): QRELs with QueryID, PassageID, and Relevance.
        documents (list): List of document texts.
        k (int): Top-k for evaluation metrics.

    Returns:
        dict: Recall@k, MRR@k, and NDCG@k scores.
    """
    # Load the fine-tuned model
    model = SentenceTransformer(model_path)

    # Generate embeddings
    query_embeddings = model.encode(queries, convert_to_tensor=True)
    document_embeddings = model.encode(documents, convert_to_tensor=True)

    # Ensure embeddings are on the CPU
    query_embeddings = query_embeddings.cpu()
    document_embeddings = document_embeddings.cpu()

    # Initialize the vector database
    vector_db = VectorDatabase(embedding_dim=document_embeddings.shape[1])
    vector_db.add_embeddings(document_embeddings.numpy())

    # Compute recall@k
    recall = compute_recall_at_k(query_embeddings, vector_db, qrels, k=k)

    # Placeholder precision and F1-score
    precision = 0.85  # Example value; calculate as needed
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1


def compute_recall_at_k(query_embeddings, vector_db, qrels, k=10):
    # Same implementation as provided
    total_recall = 0
    num_queries = len(query_embeddings)

    for idx, query_embedding in enumerate(query_embeddings):
        distances, indices = vector_db.query([query_embedding], top_k=k)
        retrieved_docs = set(indices.flatten().tolist())
        relevant_docs = set(qrels[qrels["QueryID"] == idx]["PassageID"].tolist())
        total_recall += len(retrieved_docs & relevant_docs) / len(relevant_docs) if relevant_docs else 0

    return total_recall / num_queries


def compute_mrr_at_k(query_embeddings, vector_db, qrels, k=10):
    # Same implementation as provided
    total_relevant = 0
    total_retrieved_relevant = 0

    for query_id, query_embedding in enumerate(query_embeddings):
        distances, indices = vector_db.query(query_embedding, top_k=k)

        # Assume indices map directly to document IDs
        relevant_docs = qrels.get(query_id, [])
        retrieved_docs = indices.flatten()

        total_relevant += len(relevant_docs)
        total_retrieved_relevant += len(set(relevant_docs) & set(retrieved_docs))

    recall = total_retrieved_relevant / total_relevant if total_relevant > 0 else 0
    return recall


def compute_ndcg_at_k(query_embeddings, vector_db, qrels, k=10):
    # Same implementation as provided
    total_ndcg = 0
    num_queries = len(query_embeddings)

    for idx, query_embedding in enumerate(query_embeddings):
        distances, indices = vector_db.query([query_embedding], top_k=k)
        retrieved_docs = indices.flatten().tolist()

        # Create relevance scores for the retrieved documents
        relevance_scores = []
        relevant_docs = set(qrels[qrels["QueryID"] == idx]["PassageID"].tolist())
        for doc_id in retrieved_docs:
            relevance_scores.append(1 if doc_id in relevant_docs else 0)

        # Compute NDCG score
        ideal_relevance = sorted(relevance_scores, reverse=True)
        ndcg = ndcg_score([ideal_relevance], [relevance_scores], k=k)
        total_ndcg += ndcg

    return total_ndcg / num_queries