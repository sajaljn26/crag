import os
from dotenv import load_dotenv
from src.orchestrator.pipeline import RAGPipeline
from src.ingestion.processor import DocumentProcessor
from src.ingestion.embedder import Embedder
from src.ingestion.store import VectorStore

load_dotenv()

def seed_data():
    print("Seeding sample data...")
    texts = [
        "The capital of France is Paris. It is known for the Eiffel Tower.",
        "The Great Wall of China was built across the historical northern borders of ancient Chinese states.",
        "Python is a high-level, interpreted, general-purpose programming language.",
        "The self-correcting RAG pipeline uses Groq and ChromaDB to ensure answer quality.",
        "Llama-3.3-70b is a powerful model released by Meta for various NLP tasks."
    ]
    
    processor = DocumentProcessor()
    embedder = Embedder()
    store = VectorStore()
    
    all_docs = []
    all_embeddings = []
    all_ids = []
    
    for i, text in enumerate(texts):
        chunks = processor.process(text)
        for j, chunk in enumerate(chunks):
            all_docs.append(chunk)
            all_embeddings.append(embedder.embed(chunk))
            all_ids.append(f"doc_{i}_{j}")
            
    store.add_documents(all_docs, all_embeddings, all_ids)
    print("Seeding complete.")

def main():
    # Seed data if database doesn't exist or just for demonstration
    seed_data()
    
    pipeline = RAGPipeline()
    
    queries = [
        "What is the capital of France?",
        "Tell me about the Great Wall of China.",
        "How does the self-correcting RAG pipeline work?",
        "What is the weather in Tokyo?" # Should trigger 'no relevant context'
    ]
    
    for q in queries:
        print(f"\nQuery: {q}")
        answer = pipeline.run(q)
        print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
