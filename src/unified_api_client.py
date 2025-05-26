# unified_api_client.py
import os
import json
import requests

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

        elif 'deepseek' in self.model:
            self.client_type = 'deepseek'

        elif 'sonnet' in self.model:
            self.client_type = 'anthropic'

        else:
            raise ValueError("Unsupported model type. Use a model starting with 'gpt', 'gemini', 'deepseek', or 'sonnet'")

    def send(self, messages, temperature=0.3, max_tokens=4196):
        try:
            os.makedirs("Payloads", exist_ok=True)
            with open("Payloads/glossary_payload.json", "w", encoding="utf-8") as pf:
                json.dump({"model": self.model, "messages": messages}, pf, ensure_ascii=False, indent=2)

            if self.client_type == 'openai':
                return self._send_openai(messages, temperature, max_tokens)
            elif self.client_type == 'gemini':
                result = self._send_gemini(messages, temperature, max_tokens)
                return result, None
            elif self.client_type == 'deepseek':
                result = self._send_openai_compatible(messages, temperature, max_tokens, base_url="https://api.deepseek.com/v1")
                return result, None
            elif self.client_type == 'anthropic':
                result = self._send_anthropic(messages, temperature, max_tokens)
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

        with open("Payloads/glossary_response.txt", "w", encoding="utf-8") as rf:
            rf.write(text)

        return text, reason

    def _send_gemini(self, messages, temperature, max_tokens):
        parts = [m["content"] for m in messages]
        prompt = "\n\n".join(parts)
        model = genai.GenerativeModel(self.model)
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
        )
        result = response.text

        with open("Payloads/gemini_response.txt", "w", encoding="utf-8") as rf:
            rf.write(result)

        return result

    def _send_openai_compatible(self, messages, temperature, max_tokens, base_url):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        resp = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=180
        )
        if resp.status_code != 200:
            raise UnifiedClientError(f"HTTP {resp.status_code}: {resp.text}")

        json_resp = resp.json()
        choice = json_resp["choices"][0]
        result = choice.get("message", {}).get("content") or choice.get("content", "")

        with open("Payloads/openai_compatible_response.txt", "w", encoding="utf-8") as rf:
            rf.write(result)

        return result

    def _send_anthropic(self, messages, temperature, max_tokens):
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }

        # Claude Sonnet does not support role: system
        # So prepend any system prompt to the first user message
        if messages and messages[0]["role"] == "system":
            system_prompt = messages[0]["content"]
            rest = messages[1:]
            if rest and rest[0]["role"] == "user":
                rest[0]["content"] = system_prompt + "\n\n" + rest[0]["content"]
            messages = rest

        data = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages
        }
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=data,
            timeout=180
        )
        if resp.status_code != 200:
            raise UnifiedClientError(f"HTTP {resp.status_code}: {resp.text}")

        raw = resp.json().get("content", "")
        if isinstance(raw, list):
            result = "".join(part.get("text", "") for part in raw)
        else:
            result = str(raw)
        with open("Payloads/anthropic_response.txt", "w", encoding="utf-8") as rf:
            rf.write(str(result))

        return result
