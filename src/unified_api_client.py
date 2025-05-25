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

    def send(self, messages, temperature=0.3, max_tokens=8192):
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
                # will return (result, finish_reason)
                return self._send_openai(messages, temperature, max_tokens)
            elif self.client_type == 'gemini':
                result = self._send_gemini(messages, temperature, max_tokens)
                return result, None
        except Exception as e:
            raise UnifiedClientError(f"UnifiedClient error: {e}") from e

    def _send_openai(self, messages, temperature, max_tokens):
        resp   = openai.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        choice = resp.choices[0]
        text   = choice.message.content
        reason = choice.finish_reason

        # dump for debug
        with open("Payloads/glossary_response.txt", "w", encoding="utf-8") as rf:
            rf.write(text)

        return text, reason

    def _send_gemini(self, messages, temperature, max_tokens):
        # ─── Build one big prompt by concatenating your system/user/assistant turns ───
        parts = []
        for m in messages:
            # you can tweak how you interleave roles here if you like
            parts.append(m["content"])
        prompt = "\n\n".join(parts)

        # ─── Instantiate the Gemini model ───
        model = genai.GenerativeModel(self.model)

        # ─── Send it off ───
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature":      temperature,
                "max_output_tokens": max_tokens,
            }
        )

        # ─── Grab the text ───
        result = response.text

        # ─── (Optional) dump it for debug ───
        os.makedirs("Payloads", exist_ok=True)
        with open("Payloads/gemini_response.txt", "w", encoding="utf-8") as rf:
            rf.write(result)

        return result

        
