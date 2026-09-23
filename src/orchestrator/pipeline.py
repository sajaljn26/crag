import json
import uuid
import os
from datetime import datetime
from src.config import MAX_RETRIEVAL_RETRIES, MAX_GENERATION_RETRIES, LOGS_DIR
from src.retriever.search import Retriever
from src.rewriter.query_rewriter import QueryRewriter
from src.graders.relevance import RelevanceGrader
from src.graders.groundedness import GroundednessGrader
from src.generator.llm import LLMClient
from src.generator.prompts import GENERATION_PROMPT, REGENERATION_PROMPT

class RAGPipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.rewriter = QueryRewriter()
        self.relevance_grader = RelevanceGrader()
        self.groundedness_grader = GroundednessGrader()
        self.llm = LLMClient()

    def _log_trace(self, trace: dict):
        os.makedirs(LOGS_DIR, exist_ok=True)
        filename = f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.json"
        with open(os.path.join(LOGS_DIR, filename), 'w') as f:
            json.dump(trace, f, indent=2)

    def run(self, query: str):
        trace = {
            "query": query,
            "retrieval_attempts": [],
            "generation_attempts": [],
            "final_answer": None
        }

        current_query = query
        documents = []
        is_relevant = False
        retrieval_count = 0

        # --- Phase A: Corrective Retrieval Loop ---
        while retrieval_count < MAX_RETRIEVAL_RETRIES:
            retrieval_count += 1
            documents, metadata = self.retriever.retrieve(current_query)
            is_relevant = self.relevance_grader.grade(current_query, documents)
            
            trace["retrieval_attempts"].append({
                "attempt": retrieval_count,
                "query": current_query,
                "docs": documents,
                "metadata": metadata,
                "is_relevant": is_relevant
            })

            if is_relevant:
                break
            
            current_query = self.rewriter.rewrite(current_query)

        if not is_relevant:
            trace["final_answer"] = "I could not find any relevant information to answer your question."
            self._log_trace(trace)
            return trace["final_answer"]

        # --- Phase B: Corrective Generation Loop ---
        generation_count = 0
        last_answer = ""
        last_reason = ""
        is_grounded = False

        while generation_count < MAX_GENERATION_RETRIES:
            generation_count += 1
            context_text = "\n\n".join(documents)
            
            if generation_count == 1:
                prompt = GENERATION_PROMPT.format(context=context_text, query=query)
            else:
                prompt = REGENERATION_PROMPT.format(
                    context=context_text, 
                    query=query, 
                    previous_answer=last_answer, 
                    reason=last_reason
                )
            
            answer = self.llm.generate(prompt)
            is_grounded, reason = self.groundedness_grader.grade(query, documents, answer)
            
            trace["generation_attempts"].append({
                "attempt": generation_count,
                "answer": answer,
                "is_grounded": is_grounded,
                "reason": reason
            })

            if is_grounded:
                last_answer = answer
                is_grounded = True
                break
            
            last_answer = answer
            last_reason = reason

        trace["final_answer"] = last_answer if last_answer else "Failed to generate a grounded answer."
        self._log_trace(trace)
        return trace["final_answer"]
