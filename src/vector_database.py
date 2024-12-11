import numpy as np
import faiss
import torch

class VectorDatabase:
    def __init__(self, embedding_dim):
        """
        Initializes a FAISS index for storing embeddings.
        """
        self.index = faiss.IndexFlatL2(embedding_dim)

    def add_embeddings(self, embeddings):
        """
        Adds embeddings to the vector database.
        Handles tensors or NumPy arrays.
        """
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()  # Move to CPU and convert to NumPy
        else:
            embeddings = np.array(embeddings).astype(np.float32)  # Ensure float32

        self.index.add(embeddings)

    def query(self, embedding, top_k=10):
        """
        Queries the database for the top_k nearest neighbors of the given embedding.
        """
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().numpy()  # Move to CPU and convert to NumPy
        else:
            embedding = np.array(embedding).astype(np.float32)

        if embedding.ndim == 1:  # Single query
            embedding = embedding[np.newaxis, :]
        distances, indices = self.index.search(embedding, top_k)
        return distances, indices