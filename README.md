# Intelligent Query Optimization

This project explores **Dynamic Bloom Filters (DBFs)** and the **fine-tuning of SentenceTransformer models** for optimizing semantic search in large-scale document retrieval systems. It focuses on improving scalability, precision, memory efficiency, and query performance in dynamic and real-time workloads.

---

## Key Features

- **Dynamic Bloom Filters**: Efficiently handle dynamic workloads by dynamically resizing to manage variable data efficiently.
- **Semantic Search**: Fine-tunes the `all-MiniLM-L6-v2` SentenceTransformer model for high-precision semantic search.
- **Interactive Querying**: Provides real-time query capability with ranked retrieval based on cosine similarity.
- **Performance Evaluation**: Uses metrics like cache hit rate, false positive rate, Precision@k, Recall@k, and F1-score.

---

## Directory Structure

.
├── bloom_filter.py            # Implements Dynamic Bloom Filter logic
├── dynamic_bloom_filter.py    # Code for enhanced Bloom Filter capabilities
├── embedding_generator.py     # Generates embeddings using SentenceTransformer
├── vector_database.py         # Manages vector space operations with FAISS
├── preprocessing.py           # Prepares data for training and testing
├── evaluation.py              # Evaluates fine-tuned models
├── experiment_latency.py      # Experiments measuring query latency
├── experiment_recall.py       # Experiments measuring recall
├── experiment_query_difficulty.py  # Evaluates performance on challenging queries
├── main.py                    # Entry point for running experiments
├── README.md                  # Project documentation
└── requirements.txt           # Python dependencies

---

## Installation

### Prerequisites
- Python 3.8 or above
- `pip` package manager
- GPU-enabled environment for faster training and inference (optional)

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/joymohanty8999/intelligent-query-optimization.git
   cd intelligent-query-optimization
2.	Set up a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
4.	Download the [MS MARCO dataset](https://microsoft.github.io/msmarco/) and place it in the appropriate directory.

### Usage

#### Running Experiments

To run the full suite of experiments:

```bash
python main.py
```

#### Interactive Query Mode

To test the semantic search with interactive queries:

```bash
python interactive_query.py
```
