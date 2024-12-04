import numpy as np
import faiss

class VectorDatabase:
    def __init__(self, embedding_dim):
        """
        Initializes a vector database using FAISS with L2 distance metric.
        
        Parameters:
            embedding_dim (int): Dimensionality of the embeddings.
        """
        self.index = faiss.IndexFlatL2(embedding_dim)  # L2 distance-based index

    def add_embeddings(self, embeddings):
        """
        Adds embeddings to the vector database.
        
        Parameters:
            embeddings (np.ndarray): Embeddings to be added.
        """
        self.index.add(np.array(embeddings).astype(np.float32))

    def query(self, embedding, top_k=10):
        """
        Queries the vector database to find the top-k similar embeddings.
        
        Parameters:
            embedding (np.ndarray): The query embedding.
            top_k (int): Number of top similar embeddings to retrieve.
        
        Returns:
            tuple: Distances and indices of the top-k similar embeddings.
        """
        # Ensure the embedding is in the correct shape (1, embedding_dim)
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        distances, indices = self.index.search(np.array(embedding).astype(np.float32), top_k)
        return distances, indices