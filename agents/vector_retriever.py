
'''
class Retriever:
    """
    Retrieves the top-k most relevant text chunks
    based on vector similarity search using FAISS.
    """

    def __init__(self, vector_store, chunks, embedder):
        """
        vector_store: instance of VectorStore
        chunks: list of text chunks corresponding to embeddings
        embedder: instance of EmbeddingAgent
        """
        self.vector_store = vector_store
        self.chunks = chunks
        self.embedder = embedder

    # ---------------------------------------
    # Retrieve relevant chunks for a query
    # ---------------------------------------
    def retrieve(self, query: str, top_k=3):
        """
        Returns top-k text chunks most relevant to the query.
        """
        # 1) Convert query to embedding
        query_embedding = self.embedder.embed([query])

        # 2) Search FAISS index for closest chunks
        top_indices = self.vector_store.search(query_embedding, top_k=top_k)

        # 3) Return the actual chunk texts
        results = [self.chunks[i] for i in top_indices]

        return results
'''


class Retriever:
    def __init__(self, vector_store, chunks, embedder, metadata=None):
        self.vector_store = vector_store
        self.chunks = chunks
        self.embedder = embedder  # EmbeddingAgent instance
        self.metadata = metadata  # Optional list of metadata dicts corresponding to chunks

    def retrieve(self, query: str, top_k=3, filter_dict=None):
        query_embedding = self.embedder.embed([query])
        
        # Post-query metadata filtering
        if filter_dict and self.metadata:
            # Query a larger subset to allow filtering candidates
            candidates_k = min(self.vector_store.count(), top_k * 5)
            indices = self.vector_store.search(query_embedding, candidates_k)
            
            filtered_results = []
            for idx in indices:
                if idx < len(self.metadata) and idx < len(self.chunks):
                    chunk_meta = self.metadata[idx]
                    
                    # Match criteria in filter_dict (e.g. {"page": 2})
                    match = True
                    for key, val in filter_dict.items():
                        if chunk_meta.get(key) != val:
                            match = False
                            break
                    
                    if match:
                        filtered_results.append(self.chunks[idx])
                        if len(filtered_results) == top_k:
                            break
            
            # Fallback if no elements match the filter
            if not filtered_results:
                indices = self.vector_store.search(query_embedding, top_k)
                return [self.chunks[i] for i in indices]
            return filtered_results
        else:
            indices = self.vector_store.search(query_embedding, top_k)
            return [self.chunks[i] for i in indices]
