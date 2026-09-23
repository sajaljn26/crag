from src.generator.llm import LLMClient
from src.generator.prompts import REWRITE_PROMPT

class QueryRewriter:
    def __init__(self):
        self.llm = LLMClient()

    def rewrite(self, query: str) -> str:
        prompt = REWRITE_PROMPT.format(query=query)
        rewritten_query = self.llm.generate(prompt, system_prompt="You are a search query optimization expert.")
        return rewritten_query.strip()
