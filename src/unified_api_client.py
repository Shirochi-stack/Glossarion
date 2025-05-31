# unified_api_client.py
import os
import json
import requests
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import logging

# Set up logging
logger = logging.getLogger(__name__)

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

@dataclass
class UnifiedResponse:
    """Standardized response format for all API providers"""
    content: str
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    raw_response: Optional[Any] = None
    
    @property
    def is_truncated(self) -> bool:
        """Check if the response was truncated"""
        return self.finish_reason in ['length', 'max_tokens', 'stop_sequence_limit']
    
    @property
    def is_complete(self) -> bool:
        """Check if the response completed normally"""
        return self.finish_reason in ['stop', 'complete', 'end_turn', None]

class UnifiedClientError(Exception):
    """Generic exception for UnifiedClient errors."""
    def __init__(self, message: str, http_status: Optional[int] = None):
        super().__init__(message)
        self.http_status = http_status

class UnifiedClient:
    def __init__(self, model: str, api_key: str):
        self.model = model.lower()
        self.api_key = api_key
        self.context = None  # Add context as instance variable

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

        elif 'sonnet' in self.model or 'claude' in self.model:
            self.client_type = 'anthropic'

        else:
            raise ValueError("Unsupported model type. Use a model starting with 'gpt', 'gemini', 'deepseek', 'claude', or 'sonnet'")

    def send(self, messages, temperature=0.3, max_tokens=8192, context=None) -> Tuple[str, Optional[str]]:
        """
        Send messages to the API
        Returns: (content, finish_reason) tuple for backward compatibility
        """
        self.context = context or self.context  # Use provided context or instance context
        
        try:
            os.makedirs("Payloads", exist_ok=True)
            
            # Determine payload filename based on context
            payload_name, response_name = self._get_file_names(messages, context)
            
            # Save the payload
            self._save_payload(messages, payload_name)
            
            # Get response based on client type
            response = self._get_response(messages, temperature, max_tokens, response_name)
            
            # Log if truncated
            if response.is_truncated:
                logger.warning(f"Response was truncated! finish_reason: {response.finish_reason}")
                print(f"⚠️ Response truncated: {response.finish_reason}")
            
            # Return in backward-compatible format
            return response.content, response.finish_reason

        except Exception as e:
            logger.error(f"UnifiedClient error: {e}")
            raise UnifiedClientError(f"UnifiedClient error: {e}") from e

    def _get_file_names(self, messages, context) -> Tuple[str, str]:
        """Determine appropriate filenames for payload and response"""
        if context:
            return f"{context}_payload.json", f"{context}_response.txt"
        
        # Try to auto-detect context from messages
        messages_str = str(messages).lower()
        if 'glossary' in messages_str or 'character' in messages_str:
            return "glossary_payload.json", "glossary_response.txt"
        elif 'translat' in messages_str:
            return "translation_payload.json", "translation_response.txt"
        else:
            # Default generic names with timestamp
            import time
            timestamp = int(time.time())
            return f"api_payload_{timestamp}.json", f"api_response_{timestamp}.txt"

    def _save_payload(self, messages, payload_name):
        """Save the request payload for debugging"""
        with open(f"Payloads/{payload_name}", "w", encoding="utf-8") as pf:
            json.dump({
                "model": self.model,
                "messages": messages,
                "timestamp": os.environ.get('TZ', 'UTC')
            }, pf, ensure_ascii=False, indent=2)

    def _save_response(self, content, response_name):
        """Save the response content for debugging"""
        with open(f"Payloads/{response_name}", "w", encoding="utf-8") as rf:
            rf.write(content)

    def _get_response(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Route to appropriate handler based on client type"""
        if self.client_type == 'openai':
            return self._send_openai(messages, temperature, max_tokens, response_name)
        elif self.client_type == 'gemini':
            return self._send_gemini(messages, temperature, max_tokens, response_name)
        elif self.client_type == 'deepseek':
            return self._send_deepseek(messages, temperature, max_tokens, response_name)
        elif self.client_type == 'anthropic':
            return self._send_anthropic(messages, temperature, max_tokens, response_name)

    def _send_openai(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to OpenAI API"""
        try:
            resp = openai.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            choice = resp.choices[0]
            content = choice.message.content
            finish_reason = choice.finish_reason
            
            # Extract usage if available
            usage = None
            if hasattr(resp, 'usage'):
                usage = {
                    'prompt_tokens': resp.usage.prompt_tokens,
                    'completion_tokens': resp.usage.completion_tokens,
                    'total_tokens': resp.usage.total_tokens
                }
            
            self._save_response(content, response_name)
            
            return UnifiedResponse(
                content=content,
                finish_reason=finish_reason,
                usage=usage,
                raw_response=resp
            )
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise UnifiedClientError(f"OpenAI API error: {e}")

    def _send_gemini(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to Gemini API with retry logic"""
        parts = [m["content"] for m in messages]
        prompt = "\n\n".join(parts)
        model = genai.GenerativeModel(self.model)
        
        # Retry loop with exponential backoff
        BOOST_FACTOR = 4
        attempts = 4
        attempt = 0
        result = None
        current_tokens = max_tokens * BOOST_FACTOR
        finish_reason = None

        while attempt < attempts:
            try:
                response = model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": current_tokens
                    }
                )
                
                # Try to extract text
                try:
                    result = response.text
                    finish_reason = 'stop'  # Gemini doesn't provide explicit finish reason
                except Exception:
                    if hasattr(response, 'candidates') and response.candidates:
                        candidate = response.candidates[0]
                        if hasattr(candidate, 'content'):
                            parts = candidate.content.parts
                            result = ''.join(part.text for part in parts if hasattr(part, 'text'))
                        
                        # Check finish reason if available
                        if hasattr(candidate, 'finish_reason'):
                            finish_reason = str(candidate.finish_reason)
                            if 'MAX_TOKENS' in finish_reason:
                                finish_reason = 'length'
                
                if result:
                    break
                    
            except Exception as e:
                logger.warning(f"Gemini attempt {attempt+1} failed: {e}")
            
            # Reduce token count and retry
            current_tokens = max(256, current_tokens // 2)
            attempt += 1
            print(f"🔄 Retrying Gemini with max_output_tokens={current_tokens}")

        if not result:
            print("⚠️ All Gemini retries failed; returning empty result")
            result = "[]" if self.context == 'glossary' else ""
            finish_reason = 'error'

        self._save_response(result, response_name)
        
        return UnifiedResponse(
            content=result,
            finish_reason=finish_reason,
            raw_response=response if 'response' in locals() else None
        )

    def _send_deepseek(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to DeepSeek API"""
        return self._send_openai_compatible(
            messages, temperature, max_tokens,
            base_url="https://api.deepseek.com/v1",
            response_name=response_name,
            provider="deepseek"
        )

    def _send_openai_compatible(self, messages, temperature, max_tokens, base_url, 
                                response_name, provider="generic") -> UnifiedResponse:
        """Send request to OpenAI-compatible APIs (DeepSeek, etc.)"""
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
        
        try:
            resp = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=180
            )
            
            if resp.status_code != 200:
                error_msg = f"HTTP {resp.status_code}: {resp.text}"
                logger.error(f"{provider} API error: {error_msg}")
                raise UnifiedClientError(error_msg, http_status=resp.status_code)

            json_resp = resp.json()
            
            # Extract response data
            choice = json_resp["choices"][0]
            content = choice.get("message", {}).get("content", "")
            finish_reason = choice.get("finish_reason")
            
            # Extract usage if available
            usage = json_resp.get("usage")
            if usage:
                usage = {
                    'prompt_tokens': usage.get('prompt_tokens', 0),
                    'completion_tokens': usage.get('completion_tokens', 0),
                    'total_tokens': usage.get('total_tokens', 0)
                }
            
            # Log token usage if available
            if usage:
                logger.info(f"{provider} token usage: {usage}")
                print(f"📊 Token usage - Prompt: {usage['prompt_tokens']}, "
                      f"Completion: {usage['completion_tokens']}, "
                      f"Total: {usage['total_tokens']}")

            self._save_response(content, response_name)
            
            return UnifiedResponse(
                content=content,
                finish_reason=finish_reason,
                usage=usage,
                raw_response=json_resp
            )
            
        except requests.exceptions.Timeout:
            raise UnifiedClientError(f"{provider} API timeout after 180 seconds")
        except requests.exceptions.RequestException as e:
            raise UnifiedClientError(f"{provider} API request error: {e}")

    def _send_anthropic(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to Anthropic API"""
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }

        # Handle system messages for Claude
        processed_messages = self._process_anthropic_messages(messages)

        data = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": processed_messages
        }
        
        try:
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=data,
                timeout=180
            )
            
            if resp.status_code != 200:
                error_msg = f"HTTP {resp.status_code}: {resp.text}"
                logger.error(f"Anthropic API error: {error_msg}")
                raise UnifiedClientError(error_msg, http_status=resp.status_code)

            json_resp = resp.json()
            
            # Extract content
            content_parts = json_resp.get("content", [])
            if isinstance(content_parts, list):
                content = "".join(part.get("text", "") for part in content_parts)
            else:
                content = str(content_parts)
            
            # Extract finish reason
            finish_reason = json_resp.get("stop_reason")
            if finish_reason == "max_tokens":
                finish_reason = "length"
            
            # Extract usage
            usage = json_resp.get("usage")
            if usage:
                usage = {
                    'prompt_tokens': usage.get('input_tokens', 0),
                    'completion_tokens': usage.get('output_tokens', 0),
                    'total_tokens': usage.get('input_tokens', 0) + usage.get('output_tokens', 0)
                }
                
            self._save_response(content, response_name)
            
            return UnifiedResponse(
                content=content,
                finish_reason=finish_reason,
                usage=usage,
                raw_response=json_resp
            )
            
        except requests.exceptions.Timeout:
            raise UnifiedClientError("Anthropic API timeout after 180 seconds")
        except requests.exceptions.RequestException as e:
            raise UnifiedClientError(f"Anthropic API request error: {e}")

    def _process_anthropic_messages(self, messages):
        """Process messages for Anthropic API (handle system messages)"""
        processed = []
        system_content = None
        
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                if msg["role"] == "user" and system_content:
                    # Prepend system content to first user message
                    msg = msg.copy()
                    msg["content"] = system_content + "\n\n" + msg["content"]
                    system_content = None
                processed.append(msg)
        
        return processed

    # Backward compatibility method
    def get_unified_response(self, messages, temperature=0.3, max_tokens=8192, 
                           context=None) -> UnifiedResponse:
        """
        Get a UnifiedResponse object instead of tuple
        Useful for new code that wants to use the full response object
        """
        self.context = context or self.context
        
        try:
            os.makedirs("Payloads", exist_ok=True)
            payload_name, response_name = self._get_file_names(messages, context)
            self._save_payload(messages, payload_name)
            return self._get_response(messages, temperature, max_tokens, response_name)
        except Exception as e:
            logger.error(f"UnifiedClient error: {e}")
            raise UnifiedClientError(f"UnifiedClient error: {e}") from e
