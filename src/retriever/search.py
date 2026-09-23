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

    def retrieve(self, queries: list[str], k: int = TOP_K):
        # Handle single query input for backward compatibility if needed, 
        # but pipeline will now pass a list.
        if isinstance(queries, str):
            queries = [queries]

        # 1. Gather candidates from all query variants
        merged_results = {}
        
        for q in queries:
            # Vector Search for this variant
            query_embedding = self.embedder.embed(q)
            vector_results = self.store.query(query_embedding, n_results=k) # List of (doc, dist)
            
            # BM25 Search for this variant
            bm25_docs, _ = self.bm25_retriever.retrieve(q, k=k)
            
            # Merge and track source
            for doc, dist in vector_results:
                if doc in merged_results:
                    merged_results[doc] = "both"
                else:
                    merged_results[doc] = "vector"
                    
            for doc in bm25_docs:
                if doc in merged_results:
                    merged_results[doc] = "both"
                else:
                    merged_results[doc] = "bm25"
        
        # Convert to list of dicts for reranking
        combined = [{"doc": doc, "source": source} for doc, source in merged_results.items()]
        
        # 3. Reranking
        # The user wants to re-score the top 10 candidates.
        # We take the top 10 from the combined set.
        top_candidates = combined[:10]
        if not top_candidates:
            return [], []

        # The reranker needs the original primary query. 
        # We'll use the first query in the list (the original one).
        primary_query = queries[0]
        pairs = [[primary_query, item["doc"]] for item in top_candidates]
        scores = self.reranker.predict(pairs)
        
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
