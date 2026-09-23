import chromadb
from src.config import CHROMA_DB_PATH

class VectorStore:
    def __init__(self, collection_name: str = "rag_collection"):
        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_documents(self, documents: list[str], embeddings: list[list[float]], ids: list[str]):
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids
        )

    def query(self, query_embedding: list[float], n_results: int = 5):
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        # results['documents'] is a list of lists
        return results['documents'][0] if results['documents'] else []
