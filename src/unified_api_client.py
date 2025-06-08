# unified_api_client.py - REFACTORED with Pure Frequency-Based Reinforcement
import os
import json
import requests
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
import logging
import re
import base64
from PIL import Image
import io

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

class PromptReinforcer:
    """Pure frequency-based prompt reinforcement - no keyword detection"""
    
    def __init__(self):
        pass  # No keyword lists needed for pure reinforcement
        
    def needs_reinforcement(self, conversation_count: int, last_reinforcement: int, context: str = None) -> bool:
        """
        PURE frequency-based reinforcement logic
        Returns True ONLY when frequency interval is reached
        """
        # Skip glossary extraction
        if context == 'glossary':
            return False
            
        # Get reinforcement frequency from environment
        reinforce_freq = int(os.environ.get('REINFORCEMENT_FREQUENCY', '0'))
        if reinforce_freq == 0:
            return False  # Disabled
            
        # PURE: Only check frequency interval
        messages_since_last = conversation_count - last_reinforcement
        return messages_since_last >= reinforce_freq
    
    def create_reinforcement_prompt(self, system_content: str) -> str:
        """Simply return the original system prompt for reinforcement"""
        return system_content

class UnifiedClient:
    def __init__(self, model: str, api_key: str):
        self.model = model.lower()
        self.api_key = api_key
        self.context = None
        self.reinforcer = PromptReinforcer()
        
        # FIXED: Pure counter management with session tracking
        self.conversation_message_count = 0
        self.last_reinforcement_count = 0
        self.current_session_context = None

        if 'gpt' in self.model or 'o4' in self.model:
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
    def send_image(self, messages, image_data, temperature=0.3, max_tokens=8192, context=None) -> Tuple[str, Optional[str]]:
        """
        Send messages with image to vision-capable APIs
        
        Args:
            messages: List of message dicts
            image_data: Either bytes of image or base64 string
            temperature: Temperature for generation
            max_tokens: Max tokens to generate
            context: Context for the request (e.g., 'image_translation')
        
        Returns:
            (content, finish_reason) tuple
        """
        self.context = context or 'image_translation'
        self.conversation_message_count += 1
        
        # Convert image data to base64 if needed
        if isinstance(image_data, bytes):
            image_base64 = base64.b64encode(image_data).decode('utf-8')
        else:
            image_base64 = image_data
        
        try:
            os.makedirs("Payloads", exist_ok=True)
            
            # Apply reinforcement if needed
            messages = self._apply_pure_reinforcement(messages)
            
            # Determine payload filename
            payload_name = f"{self.context}_payload.json"
            response_name = f"{self.context}_response.txt"
            
            # Route to appropriate handler based on client type
            if self.client_type == 'gemini':
                response = self._send_gemini_image(messages, image_base64, temperature, max_tokens, response_name)
                return response.content, response.finish_reason
            elif self.client_type == 'openai':
                # Check if model supports vision
                vision_models = ['gpt-4.1-mini', 'gpt-4.1-nano', 'o4-mini']
                model_lower = self.model.lower()
                
                if ('gpt-4' in model_lower and ('vision' in model_lower or 'turbo' in model_lower or 'o' in model_lower)) or model_lower in vision_models:
                    response = self._send_openai_image(messages, image_base64, temperature, max_tokens, response_name)
                    return response.content, response.finish_reason
                else:
                    raise UnifiedClientError(f"Model {self.model} does not support image input")
            else:
                raise UnifiedClientError(f"Image input not supported for {self.client_type}")
                
        except Exception as e:
            logger.error(f"UnifiedClient image error: {e}")
            raise UnifiedClientError(f"Image processing error: {e}") from e

    def _send_gemini_image(self, messages, image_base64, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send image request to Gemini API"""
        try:
            # Format prompt for Gemini with image
            formatted_parts = []
            
            # Add system message if present
            for msg in messages:
                if msg.get('role') == 'system':
                    formatted_parts.append(f"Instructions: {msg['content']}")
                elif msg.get('role') == 'user':
                    formatted_parts.append(f"User: {msg['content']}")
            
            text_prompt = "\n\n".join(formatted_parts)
            
            # Create the model
            model = genai.GenerativeModel(self.model)
            
            # Decode base64 to PIL Image
            image_bytes = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Generate content with image
            response = model.generate_content(
                [text_prompt, image],
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens
                }
            )
            
            # Extract response
            try:
                result = response.text
                finish_reason = 'stop'
            except Exception:
                result = ""
                finish_reason = 'error'
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content'):
                        parts = candidate.content.parts
                        result = ''.join(part.text for part in parts if hasattr(part, 'text'))
            
            self._save_response(result, response_name)
            
            return UnifiedResponse(
                content=result,
                finish_reason=finish_reason,
                raw_response=response
            )
            
        except Exception as e:
            logger.error(f"Gemini image API error: {e}")
            raise UnifiedClientError(f"Gemini image API error: {e}")

    def _send_openai_image(self, messages, image_base64, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send image request to OpenAI Vision API"""
        try:
            # Format messages with image for OpenAI
            vision_messages = []
            
            for msg in messages[:-1]:  # All messages except the last
                vision_messages.append(msg)
            
            # Add the last user message with image
            last_msg = messages[-1]
            vision_messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": last_msg["content"]
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            })
            
            # Make API call
            response = openai.chat.completions.create(
                model=self.model,
                messages=vision_messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            choice = response.choices[0]
            content = choice.message.content
            finish_reason = choice.finish_reason
            
            # Extract usage
            usage = None
            if hasattr(response, 'usage'):
                usage = {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }
            
            self._save_response(content, response_name)
            
            return UnifiedResponse(
                content=content,
                finish_reason=finish_reason,
                usage=usage,
                raw_response=response
            )
            
        except Exception as e:
            logger.error(f"OpenAI Vision API error: {e}")
            raise UnifiedClientError(f"OpenAI Vision API error: {e}")
    def send(self, messages, temperature=0.3, max_tokens=8192, context=None) -> Tuple[str, Optional[str]]:
        """
        Send messages to the API with PURE frequency-based reinforcement
        Returns: (content, finish_reason) tuple for backward compatibility
        """
        # FIXED: Reset counters when context changes (new translation session)
        if context != self.current_session_context:
            self.reset_conversation_for_new_context(context)
        
        self.context = context or self.context
        self.conversation_message_count += 1
        
        try:
            os.makedirs("Payloads", exist_ok=True)
            
            # FIXED: Apply PURE frequency-based reinforcement
            messages = self._apply_pure_reinforcement(messages)
            
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

    def reset_conversation_for_new_context(self, new_context):
        """FIXED: Reset counters when switching between translation/glossary"""
        if new_context != self.current_session_context:
            print(f"🔄 Resetting reinforcement counters for context: {new_context}")
            self.conversation_message_count = 0
            self.last_reinforcement_count = 0
            self.current_session_context = new_context

    def _apply_pure_reinforcement(self, messages: List[Dict]) -> List[Dict]:
        """
        PURE frequency-based reinforcement - no keyword detection or random triggers
        """
        # Check if reinforcement is needed (PURE frequency check only)
        if not self.reinforcer.needs_reinforcement(
            self.conversation_message_count, 
            self.last_reinforcement_count,
            self.context
        ):
            return messages  # No reinforcement needed
            
        # Find system message
        system_msg = None
        system_idx = None
        for i, msg in enumerate(messages):
            if msg.get('role') == 'system':
                system_msg = msg
                system_idx = i
                break
                
        if not system_msg:
            return messages  # No system message to reinforce
            
        # Update reinforcement counter
        self.last_reinforcement_count = self.conversation_message_count
        
        print(f"🔄 Applying pure frequency-based reinforcement (message #{self.conversation_message_count})")
        
        # Apply simple reinforcement based on client type
        return self._apply_simple_reinforcement(messages, system_msg, system_idx)

    def _apply_simple_reinforcement(self, messages: List[Dict], system_msg: Dict, system_idx: int) -> List[Dict]:
        """Simple, consistent reinforcement for all API types"""
        reinforced_messages = messages.copy()
        
        if self.client_type in ['openai', 'deepseek']:
            # For OpenAI-style APIs: Enhance system message with reinforcement
            reinforced_messages[system_idx] = {
                'role': 'system',
                'content': f"{system_msg['content']}\n\n[REINFORCEMENT] Remember: {system_msg['content']}"
            }
            print(f"🔄 Reinforced system prompt for {self.client_type}")
            
        elif self.client_type == 'gemini':
            # For Gemini: Add emphasis to system message (simple, no restructuring)
            reinforced_messages[system_idx] = {
                'role': 'system',
                'content': f"IMPORTANT REMINDER: {system_msg['content']}\n\n{system_msg['content']}"
            }
            print(f"🔄 Reinforced system prompt for Gemini")
            
        elif self.client_type == 'anthropic':
            # For Claude: Add reminder to first user message (Claude doesn't use system role)
            for i, msg in enumerate(reinforced_messages):
                if msg.get('role') == 'user':
                    reinforced_messages[i] = {
                        'role': 'user',
                        'content': f"[REMINDER: {system_msg['content']}]\n\n{msg['content']}"
                    }
                    break
            # Remove original system message for Claude
            reinforced_messages = [msg for msg in reinforced_messages if msg.get('role') != 'system']
            print(f"🔄 Reinforced prompts for Anthropic")
                    
        return reinforced_messages

    def debug_reinforcement_status(self):
        """Debug method to verify pure reinforcement behavior"""
        reinforce_freq = int(os.environ.get('REINFORCEMENT_FREQUENCY', '0'))
        
        if reinforce_freq == 0:
            return f"Reinforcement DISABLED"
        
        messages_since_last = self.conversation_message_count - self.last_reinforcement_count
        messages_until_next = reinforce_freq - messages_since_last
        
        status = f"Reinforcement Status: {messages_since_last}/{reinforce_freq} messages since last"
        
        if messages_until_next <= 0:
            status += f" → WILL REINFORCE on next call"
        else:
            status += f" → {messages_until_next} messages until next reinforcement"
            
        status += f" (Context: {self.current_session_context})"
        
        return status

    def _get_file_names(self, messages, context) -> Tuple[str, str]:
        """Determine appropriate filenames for payload and response"""
        if context:
            return f"{context}_payload.json", f"{context}_response.txt"
        
        # Try to auto-detect context from messages
        messages_str = str(messages).lower()
        
        # Check for summary generation first (most specific)
        if any(phrase in messages_str for phrase in [
            'summarize the key events',
            'concise summary',
            'summary of the previous',
            'create a concise summary',
            'summarizer'
        ]):
            return "rolling_summary_payload.json", "rolling_summary_response.txt"
        
        # Check for glossary extraction
        elif any(phrase in messages_str for phrase in [
            'glossary extractor',
            'extract character information',
            'original_name',
            'traits'
        ]):
            return "glossary_payload.json", "glossary_response.txt"
        
        # Check for translation
        elif any(phrase in messages_str for phrase in [
            'translat',
            'korean to english',
            'japanese to english',
            'chinese to english',
            'novel translator'
        ]):
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
        """Send request to Gemini API with simple formatting (no complex restructuring)"""
        # Simple Gemini formatting - just concatenate messages
        formatted_prompt = self._format_gemini_prompt_simple(messages)
        
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
                    formatted_prompt,
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
    
    def _format_gemini_prompt_simple(self, messages) -> str:
        """Simple Gemini formatting - no complex restructuring"""
        formatted_parts = []
        
        for msg in messages:
            role = msg.get('role', 'user').upper()
            content = msg['content']
            
            if role == 'SYSTEM':
                formatted_parts.append(f"INSTRUCTIONS: {content}")
            else:
                formatted_parts.append(f"{role}: {content}")
        
        return "\n\n".join(formatted_parts)

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

        # Handle system messages for Claude (already handled by reinforcement)
        processed_messages = [msg for msg in messages if msg.get('role') != 'system']

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

    def reset_conversation(self):
        """Reset counters for completely new conversation"""
        self.conversation_message_count = 0
        self.last_reinforcement_count = 0
        self.current_session_context = None

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
            
            # Apply pure frequency-based reinforcement
            messages = self._apply_pure_reinforcement(messages)
            
            payload_name, response_name = self._get_file_names(messages, context)
            self._save_payload(messages, payload_name)
            return self._get_response(messages, temperature, max_tokens, response_name)
        except Exception as e:
            logger.error(f"UnifiedClient error: {e}")
            raise UnifiedClientError(f"UnifiedClient error: {e}") from e
