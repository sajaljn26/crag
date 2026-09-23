import json
from src.generator.llm import LLMClient
from src.generator.prompts import RELEVANCE_PROMPT

class RelevanceGrader:
    def __init__(self):
        self.llm = LLMClient()

    def grade(self, query: str, documents: list[str]) -> tuple[dict, dict]:
        context = "\n\n".join(documents)
        prompt = RELEVANCE_PROMPT.format(query=query, context=context)
        response, usage = self.llm.generate(prompt, system_prompt="You are an expert grader. Always return a valid JSON object.")
        
        try:
            # Try to parse the response as JSON
            # The LLM might wrap it in markdown blocks
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:-3].strip()
            elif cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:-3].strip()
                
            return json.loads(cleaned_response), usage
        except Exception as e:
            # Fallback if JSON parsing fails
            return {
                "score": 0,
                "label": "incorrect",
                "reasoning": f"Failed to parse LLM response: {e}"
            }, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
