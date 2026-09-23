import re

class DocumentProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def clean_text(self, text: str) -> str:
        # Basic cleaning: remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def chunk_text(self, text: str) -> list[str]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i : i + self.chunk_size])
        return chunks

    def process(self, text: str) -> list[str]:
        cleaned = self.clean_text(text)
        return self.chunk_text(cleaned)
