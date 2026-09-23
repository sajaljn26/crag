from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL_NAME

class Embedder:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def embed(self, text: str):
        return self.model.encode(text).tolist()

    def embed_batch(self, texts: list[str]):
        return self.model.encode(texts).tolist()
