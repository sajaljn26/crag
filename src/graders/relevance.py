from src.generator.llm import LLMClient
from src.generator.prompts import RELEVANCE_PROMPT

class RelevanceGrader:
    def __init__(self):
        self.llm = LLMClient()

    def grade(self, query: str, documents: list[str]) -> bool:
        context = "\n\n".join(documents)
        prompt = RELEVANCE_PROMPT.format(query=query, context=context)
        response = self.llm.generate(prompt, system_prompt="You are an expert grader.")
        return "YES" in response.upper()
