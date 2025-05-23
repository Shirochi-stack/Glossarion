# unified_api_client.py
import os
import json

# OpenAI SDK
try:
    import openai
    from openai import OpenAIError
except ImportError:
    openai = None
    class OpenAIError(Exception): pass

# Gemini SDK
try:
    import google.generativeai as genai
except ImportError:
    genai = None

class UnifiedClientError(Exception):
    """Generic exception for UnifiedClient errors."""
    pass

class UnifiedClient:
    def __init__(self, model: str, api_key: str):
        self.model = model.lower()
        self.api_key = api_key

        if 'gpt' in self.model:
            if openai is None:
                raise ImportError("OpenAI library not installed.")
            openai.api_key = self.api_key
            self.client_type = 'openai'

        elif 'gemini' in self.model:
            if genai is None:
                raise ImportError("Google Generative AI library not installed.")
            genai.configure(api_key=self.api_key)
            self.client_type = 'gemini'

        else:
            raise ValueError("Unsupported model type. Use a model starting with 'gpt' or 'gemini'")

    def send(self, messages, temperature=0.3, max_tokens=2048):
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            os.makedirs("Payloads", exist_ok=True)
            with open("Payloads/glossary_payload.json", "w", encoding="utf-8") as pf:
                json.dump(payload, pf, ensure_ascii=False, indent=2)

            if self.client_type == 'openai':
                return self._send_openai(messages, temperature)
            elif self.client_type == 'gemini':
                return self._send_gemini(messages, temperature, max_tokens)

        except Exception as e:
            raise UnifiedClientError(f"UnifiedClient error: {str(e)}") from e

    def _send_openai(self, messages, temperature):
        response = openai.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature
        )
        result = response.choices[0].message.content
        with open("Payloads/glossary_response.txt", "w", encoding="utf-8") as rf:
            rf.write(result)
        return result

    def _send_gemini(self, messages, temperature, max_tokens):
        # include both system & user messages so Gemini sees your instructions
        prompt_parts = []
        for m in messages:
            if m['role'] in ('system', 'user'):
                prompt_parts.append(m['content'])
        prompt = "\n\n".join(prompt_parts)
        model = genai.GenerativeModel(self.model)
        response = model.generate_content(prompt, generation_config={
            'temperature': temperature,
            'max_output_tokens': max_tokens
        })
        result = response.text
        with open("Payloads/glossary_response.txt", "w", encoding="utf-8") as rf:
            rf.write(result)
        return result
