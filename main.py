import os
import sys
import logging
import torch
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import load_collection, load_queries_and_qrels, extract_tar_file
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
from fine_tuning.fine_tune import fine_tune_model
from fine_tuning.evaluation import evaluate_model

import matplotlib.pyplot as plt
import numpy as np


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_synthetic_qrels(queries, documents, num_relevant=3):
    """
    Generate synthetic relevance judgments for evaluation.
    """
    qrels = {}
    for query_id in range(len(queries)):
        relevant_docs = random.sample(range(len(documents)), min(num_relevant, len(documents)))
        qrels[query_id] = relevant_docs
    return qrels



def run_experiments(data_dir, embeddings_dir, reports_dir, collection, queries, qrels):
    # Generate or Load Embeddings
    queries_embeddings_file = os.path.join(embeddings_dir, "queries.npy")
    documents_embeddings_file = os.path.join(embeddings_dir, "documents.npy")
    
    logging.info("Checking for existing embeddings...")
    if os.path.exists(queries_embeddings_file) and os.path.exists(documents_embeddings_file):
        logging.info("Embeddings found. Loading existing embeddings...")
        queries_embeddings = np.load(queries_embeddings_file)
        documents_embeddings = np.load(documents_embeddings_file)

        query_texts = queries["Query"].tolist()[:len(queries_embeddings)]
        document_texts = collection["Passage"].tolist()[:len(documents_embeddings)]
    else:
        logging.info("Embeddings not found. Generating embeddings...")
        os.makedirs(embeddings_dir, exist_ok=True)
        generator = EmbeddingGenerator()

        query_texts = queries["Query"].tolist()[:10000]  # Subset for speed
        document_texts = collection["Passage"].tolist()[:100000]

        queries_embeddings = generator.generate_embeddings(query_texts, batch_size=64)
        documents_embeddings = generator.generate_embeddings(document_texts, batch_size=64)

        np.save(queries_embeddings_file, queries_embeddings)
        np.save(documents_embeddings_file, documents_embeddings)
        logging.info("Embeddings saved.")

    # Run Experiments
    logging.info("Running experiments...")

    # Bloom Filter Experiment
    bloom_results = experiment_bloom_filter(queries["Query"].tolist()[:1000])
    capacities, error_rates, hit_rates = zip(*bloom_results)
    plt.figure()
    plt.plot(error_rates, hit_rates, marker="o", linestyle="--", label="Static Bloom Filter")
    plt.title("Cache Hit Rate vs Bloom Filter Parameters")
    plt.xlabel("Error Rate")
    plt.ylabel("Cache Hit Rate")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "bloom_filter_params.png"))

    # Dynamic Bloom Filter Experiment
    logging.info("Running Dynamic Bloom Filter Experiment...")
    dynamic_results = []
    for capacity in [1000, 5000, 10000]:
        dynamic_bloom_filter = DynamicBloomFilter(initial_capacity=capacity, error_rate=0.05)
        dynamic_hit_count = 0
        for query in queries["Query"].tolist()[:1000]:
            if dynamic_bloom_filter.check_and_insert(query):
                dynamic_hit_count += 1
        dynamic_results.append((capacity, dynamic_hit_count / 1000))

    # Plot results for Dynamic Bloom Filter
    dynamic_capacities, dynamic_hit_rates = zip(*dynamic_results)
    plt.figure()
    plt.plot(dynamic_capacities, dynamic_hit_rates, marker="o", linestyle="--", label="Dynamic Bloom Filter")
    plt.title("Cache Hit Rates for Dynamic Bloom Filter")
    plt.xlabel("Initial Capacity")
    plt.ylabel("Cache Hit Rate")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "dynamic_bloom_filter_hit_rate.png"))
    logging.info("Dynamic Bloom Filter Experiment completed.")

    # Latency Experiment
    vector_db = VectorDatabase(embedding_dim=documents_embeddings.shape[1])
    vector_db.add_embeddings(documents_embeddings[:500])
    latency = experiment_latency(queries_embeddings[:50], vector_db)
    logging.info(f"Latency for 500 documents: {latency:.4f} seconds")
    
def fine_tune_model_and_evaluate(data_dir, queries, qrels, collection):
    # Fine-tune SentenceTransformer model
    logging.info("Fine-tuning model...")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    save_path = os.path.join(data_dir, f"fine_tuned_{model_name.split('/')[-1]}")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Check if fine-tuned model already exists
    if os.path.exists(save_path):
        logging.info(f"Fine-tuned model already exists at {save_path}. Skipping fine-tuning.")
    else:
        # Fine-tune SentenceTransformer model
        logging.info("Fine-tuning model...")
        fine_tune_model(
            model_name=model_name,
            train_data=(queries, qrels, collection),
            save_path=save_path,
            batch_size=32,
            epochs=1,
        )
        logging.info(f"Fine-tuned model saved at {save_path}")

    # Evaluate Fine-Tuned Model
    logging.info("Evaluating fine-tuned model...")
    
    precision, recall, f1_score = evaluate_model(
        model_path=save_path,
        queries=queries["Query"].tolist()[:50],
        qrels=qrels,
        documents=collection["Passage"].tolist()[:500],
    )

    # Ensure numerical values for precision, recall, and f1_score
    try:
        precision = float(precision)
        recall = float(recall)
        f1_score = float(f1_score)
        logging.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}")
    except ValueError as e:
        logging.error(f"Error formatting evaluation metrics: {e}")
        logging.info(f"Precision: {precision}, Recall: {recall}, F1-Score: {f1_score}")

def main():
    # Paths
    data_dir = "data/"
    embeddings_dir = "embeddings/"
    reports_dir = "reports/"
    os.makedirs(reports_dir, exist_ok=True)

    # Load Data
    logging.info("Loading data...")
    collection = load_collection(os.path.join(data_dir, "collection/collection.tsv"))
    queries, qrels = load_queries_and_qrels(
        os.path.join(data_dir, "queries/queries.train.tsv"),
        os.path.join(data_dir, "qrels.train.tsv"),
    )
    logging.info(f"Collection size: {len(collection)}")
    logging.info(f"Queries size: {len(queries)}")
    logging.info(f"QRELs size: {len(qrels)}")

    # Run existing experiments
    run_experiments(data_dir, embeddings_dir, reports_dir, collection, queries, qrels)
    
    # Fine-tune and evaluate the model
    fine_tune_model_and_evaluate(data_dir, queries, qrels, collection)


if __name__ == "__main__":
    main()