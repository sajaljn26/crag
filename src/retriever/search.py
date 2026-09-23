from src.ingestion.embedder import Embedder
from src.ingestion.store import VectorStore
from src.config import TOP_K

class Retriever:
    def __init__(self):
        self.embedder = Embedder()
        self.store = VectorStore()

    def retrieve(self, query: str, k: int = TOP_K):
        query_embedding = self.embedder.embed(query)
        return self.store.query(query_embedding, n_results=k)
