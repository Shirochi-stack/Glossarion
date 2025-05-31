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

    def send(self, messages, temperature=0.3, max_tokens=8192, context=None):
        """
        Send messages to the API
        context: Optional context string to help with payload naming (e.g., 'translation', 'glossary')
        """
        try:
            os.makedirs("Payloads", exist_ok=True)
            
            # Determine payload filename based on context or message content
            if context:
                payload_name = f"{context}_payload.json"
                response_name = f"{context}_response.txt"
            else:
                # Try to auto-detect context from messages
                messages_str = str(messages).lower()
                if 'glossary' in messages_str or 'character' in messages_str:
                    payload_name = "glossary_payload.json"
                    response_name = "glossary_response.txt"
                elif 'translat' in messages_str:
                    payload_name = "translation_payload.json"
                    response_name = "translation_response.txt"
                else:
                    # Default generic names with timestamp
                    import time
                    timestamp = int(time.time())
                    payload_name = f"api_payload_{timestamp}.json"
                    response_name = f"api_response_{timestamp}.txt"
            
            # Save the payload
            with open(f"Payloads/{payload_name}", "w", encoding="utf-8") as pf:
                json.dump({"model": self.model, "messages": messages}, pf, ensure_ascii=False, indent=2)

            if self.client_type == 'openai':
                return self._send_openai(messages, temperature, max_tokens, response_name)
            elif self.client_type == 'gemini':
                result = self._send_gemini(messages, temperature, max_tokens, response_name)
                return result, None
            elif self.client_type == 'deepseek':
                result = self._send_openai_compatible(messages, temperature, max_tokens, 
                                                     base_url="https://api.deepseek.com/v1",
                                                     response_name=response_name)
                return result, None
            elif self.client_type == 'anthropic':
                result = self._send_anthropic(messages, temperature, max_tokens, response_name)
                return result, None

        except Exception as e:
            raise UnifiedClientError(f"UnifiedClient error: {e}") from e

    def _send_openai(self, messages, temperature, max_tokens, response_name="response.txt"):
        resp = openai.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        choice = resp.choices[0]
        text = choice.message.content
        reason = choice.finish_reason

        with open(f"Payloads/{response_name}", "w", encoding="utf-8") as rf:
            rf.write(text)

        return text, reason

    def _send_gemini(self, messages, temperature, max_tokens, response_name="response.txt"):
        parts = [m["content"] for m in messages]
        prompt = "\n\n".join(parts)
        model = genai.GenerativeModel(self.model)
        
        # RETRY LOOP WITH INITIAL BOOST
        BOOST_FACTOR = 4
        attempts = 4
        attempt = 0
        result = None
        current_tok = max_tokens * BOOST_FACTOR

        while attempt < attempts:
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": current_tok
                }
            )
            # first try the quick accessor
            try:
                result = response.text
            except Exception:
                # if that fails, try candidates
                if getattr(response, 'candidates', None):
                    cand = response.candidates[0]
                    result = getattr(cand, 'content', None) or getattr(cand, 'text', "")
                else:
                    # warn and prepare to retry
                    print(f"⚠️ Attempt {attempt+1}: no text/candidates at {current_tok} tokens")
                    result = None
            # if we got something non-empty, break out
            if result:
                break
            # otherwise shrink the budget and retry
            current_tok = max(256, current_tok // 2)
            attempt += 1
            print(f"🔄 Retrying Gemini with max_output_tokens={current_tok}")

        if not result:
            # after exhausting retries, fall back to empty array
            print("⚠️ All retries failed; returning empty array")
            result = "[]"

        with open(f"Payloads/{response_name}", "w", encoding="utf-8") as rf:
            rf.write(result)

        return result

    def _send_openai_compatible(self, messages, temperature, max_tokens, base_url, response_name="response.txt"):
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

        with open(f"Payloads/{response_name}", "w", encoding="utf-8") as rf:
            rf.write(result)

        return result

    def _send_anthropic(self, messages, temperature, max_tokens, response_name="response.txt"):
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
            
        with open(f"Payloads/{response_name}", "w", encoding="utf-8") as rf:
            rf.write(str(result))

        return result
