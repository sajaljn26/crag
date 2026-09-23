import json
from src.generator.llm import LLMClient
from src.generator.prompts import GROUNDEDNESS_PROMPT

class GroundednessGrader:
    def __init__(self):
        self.llm = LLMClient()

    def grade(self, query: str, context: list[str], answer: str) -> tuple[dict, dict]:
        context_text = "\n\n".join(context)
        prompt = GROUNDEDNESS_PROMPT.format(context=context_text, answer=answer)
        
        try:
            # Use generate_json if available, otherwise handle manual parsing
            result, usage = self.llm.generate_json(prompt, system_prompt="You are an expert grader. Always return a valid JSON object.")
            return {
                "score": result.get("score", 0),
                "reasoning": result.get("reasoning", "No reasoning provided.")
            }, usage
        except Exception as e:
            return {
                "score": 0,
                "reasoning": f"Failed to grade groundedness: {e}"
            }, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
