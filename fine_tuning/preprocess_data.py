import pandas as pd
from sentence_transformers import InputExample

def prepare_training_data(queries_file, qrels_file, collection_file):
    """
    Prepares query-passage pairs for fine-tuning.
    Args:
        queries_file (str): Path to the queries file.
        qrels_file (str): Path to the QRELs file.
        collection_file (str): Path to the collection file.
    Returns:
        list: A list of InputExample objects.
    """
    queries = pd.read_csv(queries_file, sep="\t", header=None, names=["QueryID", "Query"])
    qrels = pd.read_csv(qrels_file, sep="\t", header=None, names=["QueryID", "PassageID", "Relevance"])
    collection = pd.read_csv(collection_file, sep="\t", header=None, names=["PassageID", "Passage"])

    query_dict = queries.set_index("QueryID")["Query"].to_dict()
    passage_dict = collection.set_index("PassageID")["Passage"].to_dict()

    train_samples = []
    for _, row in qrels.iterrows():
        query = query_dict.get(row["QueryID"])
        passage = passage_dict.get(row["PassageID"])
        relevance = float(row["Relevance"])
        train_samples.append(InputExample(texts=[query, passage], label=relevance))

    return train_samples