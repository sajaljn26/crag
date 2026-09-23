from src.generator.llm import LLMClient
from src.generator.prompts import GROUNDEDNESS_PROMPT

class GroundednessGrader:
    def __init__(self):
        self.llm = LLMClient()

    def grade(self, query: str, context: list[str], answer: str) -> tuple[bool, str]:
        context_text = "\n\n".join(context)
        prompt = GROUNDEDNESS_PROMPT.format(context=context_text, answer=answer)
        result = self.llm.generate_json(prompt, system_prompt="You are an expert grader.")
        
        is_grounded = result.get("is_grounded", "NO").upper() == "YES"
        reason = result.get("reason", "No reason provided.")
        
        return is_grounded, reason
