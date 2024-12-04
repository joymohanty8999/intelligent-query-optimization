import os
import sys
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import load_collection, load_queries_and_qrels
from src.embedding_generator import EmbeddingGenerator
from src.vector_database import VectorDatabase
from src.bloom_filter import QueryCache
from src.dynamic_bloom_filter import DynamicBloomFilter
from src.evaluation import compute_precision_recall_f1
from src.experiments.experiment_bloom_filter import experiment_bloom_filter
from src.experiments.experiment_latency import experiment_latency
from src.experiments.experiment_recall import experiment_recall
from src.experiments.experiment_query_difficulty import experiment_query_difficulty
from src.experiments.experiment_mmr import experiment_mmr

import matplotlib.pyplot as plt
import numpy as np


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    # Paths
    data_dir = "data/"
    embeddings_dir = "embeddings/"
    reports_dir = "reports/"
    os.makedirs(reports_dir, exist_ok=True)

    collection_file = os.path.join(data_dir, "collection/collection.tsv")
    queries_file = os.path.join(data_dir, "queries/queries.train.tsv")
    qrels_file = os.path.join(data_dir, "qrels.train.tsv")
    queries_embeddings_file = os.path.join(embeddings_dir, "queries.npy")
    documents_embeddings_file = os.path.join(embeddings_dir, "documents.npy")

    # Step 1: Load Data
    logging.info("Loading data...")
    collection = load_collection(collection_file)
    queries, qrels = load_queries_and_qrels(queries_file, qrels_file)

    logging.info(f"Collection size: {len(collection)}")
    logging.info(f"Queries size: {len(queries)}")
    logging.info(f"QRELs size: {len(qrels)}")

    # Step 2: Generate Embeddings
    logging.info("Generating embeddings...")
    os.makedirs(embeddings_dir, exist_ok=True)
    generator = EmbeddingGenerator()

    query_texts = queries["Query"].tolist()[:10000]  # Dynamic subset
    document_texts = collection["Passage"].tolist()[:100000]  # Dynamic subset

    queries_embeddings = generator.generate_embeddings(query_texts, batch_size=64)
    documents_embeddings = generator.generate_embeddings(document_texts, batch_size=64)

    np.save(queries_embeddings_file, queries_embeddings)
    np.save(documents_embeddings_file, documents_embeddings)
    logging.info("Embeddings saved.")

    # Step 3: Experiments
    logging.info("Running experiments...")

    # Bloom Filter Experiment
    bloom_results = experiment_bloom_filter(query_texts)
    capacities, error_rates, hit_rates = zip(*bloom_results)
    plt.figure()
    plt.plot(error_rates, hit_rates, marker="o", linestyle="--", label="Hit Rate")
    plt.title("Cache Hit Rate vs Bloom Filter Parameters")
    plt.xlabel("Error Rate")
    plt.ylabel("Cache Hit Rate")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "bloom_filter_params.png"))

    # Latency Experiment
    vector_db = VectorDatabase(embedding_dim=documents_embeddings.shape[1])
    vector_db.add_embeddings(documents_embeddings[:500])
    latency = experiment_latency(queries_embeddings[:50], vector_db)
    logging.info(f"Latency for 500 documents: {latency:.4f} seconds")

    # Recall Experiment
    models = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/paraphrase-MiniLM-L6-v2",
        "sentence-transformers/paraphrase-mpnet-base-v2",
    ]
    for k in [50, 100, 200]:
        recall_results = experiment_recall(query_texts[:50], document_texts[:500], qrels, models, k=k)
        model_names, recall_scores = zip(*recall_results)
        short_model_names = [name.split("/")[-1] for name in model_names]
        plt.figure()
        plt.bar(short_model_names, recall_scores, width=0.4)
        plt.title(f"Recall@{k} for Different Embedding Models")
        plt.xlabel("Model")
        plt.ylabel(f"Recall@{k}")
        plt.yscale("log")
        for i, value in enumerate(recall_scores):
            plt.text(i, value, f'{value:.2e}', ha='center', va='bottom', fontsize=8)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(reports_dir, f"recall_vs_models_k{k}.png"))

    # Query Difficulty Analysis
    easy_queries, hard_queries = experiment_query_difficulty(queries_embeddings[:100], vector_db)
    logging.info(f"Easy Queries: {len(easy_queries)}, Hard Queries: {len(hard_queries)}")

    # MMR Experiment
    mmr_results = experiment_mmr(queries_embeddings[:5], documents_embeddings[:500], k=10, lambda_val=0.5)
    logging.info(f"MMR-selected documents for the first query: {mmr_results}")

    logging.info("Experiments completed. Results saved in the reports directory.")


if __name__ == "__main__":
    main()