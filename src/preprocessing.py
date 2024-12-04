import os
import tarfile
import pandas as pd

def extract_tar_file(tar_path, extract_to="data/collection", tsv_file="data/collection/collection.tsv"):
    """
    Extracts a tar file to the specified directory if the .tsv file does not already exist.
    
    Parameters:
        tar_path (str): Path to the tar file.
        extract_to (str): Directory to extract the tar file.
        tsv_file (str): Path to the .tsv file to check for existence.
    """
    if os.path.exists(tsv_file):
        print(f"{tsv_file} already exists. Skipping extraction.")
        return

    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=extract_to)

    print(f"Extracted {tar_path} to {extract_to}")


def load_collection(collection_path):
    """
    Loads the collection file into a DataFrame.
    
    Parameters:
        collection_path (str): Path to the collection file.
    
    Returns:
        pd.DataFrame: DataFrame containing the collection data.
    """
    if not os.path.exists(collection_path):
        raise FileNotFoundError(f"{collection_path} does not exist. Please provide the collection file.")
    
    collection = pd.read_csv(
        collection_path, sep="\t", header=None, names=["PassageID", "Passage"]
    )
    print(f"Loaded {len(collection)} passages.")
    return collection


def load_queries_and_qrels(queries_path, qrels_path):
    """
    Loads queries and QRELs (Query Relevance Judgments) from given file paths.
    
    Parameters:
        queries_path (str): Path to the queries file.
        qrels_path (str): Path to the QRELs file.
    
    Returns:
        tuple: A tuple containing the queries DataFrame and QRELs DataFrame.
    """
    if not os.path.exists(queries_path):
        raise FileNotFoundError(f"{queries_path} does not exist.")
    if not os.path.exists(qrels_path):
        raise FileNotFoundError(f"{qrels_path} does not exist.")
    
    # Load queries
    queries = pd.read_csv(queries_path, sep="\t", header=None, names=["QueryID", "Query"])
    print(f"Loaded {len(queries)} queries.")

    # Load QRELs
    qrels = pd.read_csv(qrels_path, sep="\t", header=None, names=["QueryID", "PassageID", "Relevance"])
    print(f"Loaded {len(qrels)} QRELs.")

    return queries, qrels