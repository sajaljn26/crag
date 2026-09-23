import json
from groq import Groq
from src.config import GROQ_API_KEY, LLM_MODEL

class LLMClient:
    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in environment variables.")
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = LLM_MODEL

    def generate(self, prompt: str, system_prompt: str = "You are a helpful assistant."):
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            usage = {
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens
            }
            return completion.choices[0].message.content, usage
        except Exception as e:
            print(f"Error during LLM generation: {e}")
            return "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def generate_json(self, prompt: str, system_prompt: str = "You are a helpful assistant."):
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            usage = {
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens
            }
            return json.loads(completion.choices[0].message.content), usage
        except Exception as e:
            print(f"Error during LLM JSON generation: {e}")
            return {}, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
