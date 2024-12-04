import logging
from src.evaluation import compute_recall_at_k
from src.embedding_generator import EmbeddingGenerator
from src.vector_database import VectorDatabase

def experiment_recall(queries, documents, qrels, models, k=10):
    """
    Evaluates recall@k for different embedding models.
    
    Parameters:
        query_texts (list): List of query strings.
        document_texts (list): List of document strings.
        qrels (DataFrame): Query relevance judgments.
        models (list): List of embedding model names.
        k (int): Number of top results to evaluate recall on.
        
    Returns:
        list: Recall scores for each model.
    """
    results = []
    for model_name in models:
        print(f"Evaluating recall for model: {model_name}")
        generator = EmbeddingGenerator(model_name=model_name)
        query_embeddings = generator.generate_embeddings(queries)
        
        # Initialize vector database
        vector_db = VectorDatabase(embedding_dim=query_embeddings.shape[1])
        document_embeddings = generator.generate_embeddings(documents)
        vector_db.add_embeddings(document_embeddings)
        
        # Compute Recall@k
        recall_at_k = compute_recall_at_k(query_embeddings, vector_db, qrels, k)
        print(f"Model: {model_name}, Recall@{k}: {recall_at_k:.4f}")
        results.append((model_name, recall_at_k))
    return results