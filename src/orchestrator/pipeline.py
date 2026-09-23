import json
import uuid
import os
from datetime import datetime
from src.config import MAX_RETRIEVAL_RETRIES, MAX_GENERATION_RETRIES, LOGS_DIR, RELEVANCE_THRESHOLD, GROUNDEDNESS_THRESHOLD, INPUT_TOKEN_COST, OUTPUT_TOKEN_COST
from src.retriever.search import Retriever
from src.retriever.web_search import WebSearchTool
from src.rewriter.query_rewriter import QueryRewriter
from src.graders.relevance import RelevanceGrader
from src.graders.groundedness import GroundednessGrader
from src.generator.llm import LLMClient
from src.generator.prompts import GENERATION_PROMPT, REGENERATION_PROMPT
from src.security.injection_detector import InjectionDetector

class RAGPipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.web_search = WebSearchTool()
        self.rewriter = QueryRewriter()
        self.relevance_grader = RelevanceGrader()
        self.groundedness_grader = GroundednessGrader()
        self.llm = LLMClient()
        self.injection_detector = InjectionDetector()

    def _log_trace(self, trace: dict):
        os.makedirs(LOGS_DIR, exist_ok=True)
        filename = f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.json"
        with open(os.path.join(LOGS_DIR, filename), 'w') as f:
            json.dump(trace, f, indent=2)

    def run(self, query: str) -> tuple[str, dict]:
        trace = {
            "query": query,
            "retrieval_attempts": [],
            "generation_attempts": [],
            "final_answer": None,
            "path": None,
            "flagged_documents": [],
            "metrics": {
                "total_llm_calls": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_cost": 0.0
            }
        }

        def track_usage(usage):
            trace["metrics"]["total_llm_calls"] += 1
            trace["metrics"]["total_prompt_tokens"] += usage.get("prompt_tokens", 0)
            trace["metrics"]["total_completion_tokens"] += usage.get("completion_tokens", 0)
            
            # Cost calculation: (prompt / 1M * cost) + (completion / 1M * cost)
            cost = (usage.get("prompt_tokens", 0) / 1_000_000 * INPUT_TOKEN_COST) + \\
                   (usage.get("completion_tokens", 0) / 1_000_000 * OUTPUT_TOKEN_COST)
            trace["metrics"]["total_cost"] += cost

        current_query = query
        documents = []
        relevance_res = {"score": 0, "label": "incorrect", "reasoning": ""}
        retrieval_count = 0

        # --- Phase A: Corrective Retrieval Loop ---
        while retrieval_count < MAX_RETRIEVAL_RETRIES:
            retrieval_count += 1
            
            # Multi-Query Expansion: Generate variants for the current query
            queries_to_search, expansion_usage = self.rewriter.expand_query(current_query)
            track_usage(expansion_usage)
            
            documents, metadata = self.retriever.retrieve(queries_to_search)
            relevance_res, rel_usage = self.relevance_grader.grade(current_query, documents)
            track_usage(rel_usage)
            
            trace["retrieval_attempts"].append({
                "attempt": retrieval_count,
                "query": current_query,
                "expanded_queries": queries_to_search,
                "docs": documents,
                "metadata": metadata,
                "relevance_score": relevance_res.get("score"),
                "relevance_label": relevance_res.get("label"),
                "relevance_reasoning": relevance_res.get("reasoning")
            })

            if relevance_res.get("score", 0) >= RELEVANCE_THRESHOLD:
                trace["path"] = "local_retrieval"
                break
            
            if relevance_res.get("label") == "incorrect":
                # Trigger web search fallback immediately instead of retrying local retrieval
                trace["path"] = "web_search_fallback"
                documents = self.web_search.search(query)
                break
            
            # State is "ambiguous" - retry with rewritten query
            rewritten, rewrite_usage = self.rewriter.rewrite(current_query)
            track_usage(rewrite_usage)
            current_query = rewritten

        if not documents:
            trace["final_answer"] = "I could not find any relevant information locally or via web search."
            self._log_trace(trace)
            return trace["final_answer"], trace

        # --- Input Validation: Prompt Injection Detection ---
        documents, flagged = self.injection_detector.filter_documents(documents)
        trace["flagged_documents"] = flagged

        if not documents:
            trace["final_answer"] = "Retrieved documents were flagged as potentially unsafe or malicious."
            self._log_trace(trace)
            return trace["final_answer"], trace

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
            
            answer, gen_usage = self.llm.generate(prompt)
            track_usage(gen_usage)
            
            groundedness_res, grd_usage = self.groundedness_grader.grade(query, documents, answer)
            track_usage(grd_usage)
            
            trace["generation_attempts"].append({
                "attempt": generation_count,
                "answer": answer,
                "groundedness_score": groundedness_res.get("score"),
                "groundedness_reasoning": groundedness_res.get("reasoning")
            })

            if groundedness_res.get("score", 0) >= GROUNDEDNESS_THRESHOLD:
                last_answer = answer
                is_grounded = True
                break
            
            last_answer = answer
            last_reason = groundedness_res.get("reasoning", "Not grounded")

        trace["final_answer"] = last_answer if last_answer else "Failed to generate a grounded answer."
        self._log_trace(trace)
        return trace["final_answer"], trace
