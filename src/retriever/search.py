import numpy as np
from rank_bm25 import BM25Okapi
from src.ingestion.store import VectorStore
from src.ingestion.embedder import Embedder
from src.config import TOP_K
from sentence_transformers import CrossEncoder

class BM25Retriever:
    def __init__(self, store: VectorStore):
        self.store = store
        self.corpus = []
        self.bm25 = None
        self._initialize_index()

    def _initialize_index(self):
        # Get all documents from the vector store
        all_docs = self.store.collection.get()['documents']
        if not all_docs:
            self.corpus = []
            return
        
        self.corpus = all_docs
        # Tokenize corpus
        tokenized_corpus = [doc.lower().split() for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, k: int = TOP_K):
        if not self.bm25:
            return [], []
        
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_n = np.argsort(scores)[::-1][:k]
        
        return [self.corpus[i] for i in top_n], [scores[i] for i in top_n]

class Retriever:
    def __init__(self):
        self.embedder = Embedder()
        self.store = VectorStore()
        self.bm25_retriever = BM25Retriever(self.store)
        # Cross-Encoder for reranking
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def retrieve(self, query: str, k: int = TOP_K):
        # 1. Vector Search (Get top k)
        query_embedding = self.embedder.embed(query)
        vector_results = self.store.query(query_embedding, n_results=k) # List of (doc, dist)
        
        # 2. BM25 Search (Get top k)
        bm25_docs, _ = self.bm25_retriever.retrieve(query, k=k)
        
        # Merge results and keep track of source
        # We use a dict to deduplicate. 
        # Since vector_results and bm25_docs are already sorted, the first appearance 
        # of a doc is its best rank from that retriever.
        merged_results = {}
        
        # Process vector results first
        for doc, dist in vector_results:
            merged_results[doc] = "vector"
            
        # Process BM25 results
        for doc in bm25_docs:
            if doc in merged_results:
                merged_results[doc] = "both"
            else:
                merged_results[doc] = "bm25"
        
        # Convert to list of dicts
        combined = [{"doc": doc, "source": source} for doc, source in merged_results.items()]
        
        # 3. Reranking
        # The user wants to re-score the top 10 candidates.
        # We take the first 10 from our combined list.
        top_candidates = combined[:10]
        if not top_candidates:
            return [], []

        # Prepare pairs for CrossEncoder: (query, document)
        pairs = [[query, item["doc"]] for item in top_candidates]
        scores = self.reranker.predict(pairs)
        
        # Sort by score
        scored_candidates = []
        for i in range(len(top_candidates)):
            scored_candidates.append({
                "doc": top_candidates[i]["doc"],
                "source": top_candidates[i]["source"],
                "score": scores[i]
            })
        
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        
        # Keep top 3-5 (let's say top 5)
        final_candidates = scored_candidates[:5]
        
        final_docs = [item["doc"] for item in final_candidates]
        final_metadata = [{"doc": item["doc"], "source": item["source"]} for item in final_candidates]
        
        return final_docs, final_metadata
