from src.generator.llm import LLMClient
from src.generator.prompts import REWRITE_PROMPT, EXPANSION_PROMPT

class QueryRewriter:
    def __init__(self):
        self.llm = LLMClient()

    def rewrite(self, query: str) -> tuple[str, dict]:
        prompt = REWRITE_PROMPT.format(query=query)
        rewritten_query, usage = self.llm.generate(prompt, system_prompt="You are a search query optimization expert.")
        return rewritten_query.strip(), usage

    def expand_query(self, query: str) -> tuple[list[str], dict]:
        prompt = EXPANSION_PROMPT.format(query=query)
        expanded, usage = self.llm.generate(prompt, system_prompt="You are a search query expansion expert.")
        # Split by newline and remove empty strings/whitespace
        queries = [q.strip() for q in expanded.split('\n') if q.strip()]
        # Always include the original query as well
        return [query] + queries[:3], usage
