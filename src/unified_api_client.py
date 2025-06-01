# unified_api_client.py
import os
import json
import requests
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
import logging
import re

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
    """Handles prompt reinforcement logic for all APIs"""
    
    def __init__(self):
        self.translation_keywords = [
            'translate', 'translation', 'translator',
            'korean', 'japanese', 'chinese', 
            'retain honorifics', 'preserve html',
            'context rich', 'natural translation'
        ]
        
    def needs_reinforcement(self, messages: List[Dict], context: str = None) -> bool:
        """Determine if prompt reinforcement is needed"""
        if not messages:
            return False
            
        # Check context
        if context == 'glossary':
            return False  # Glossary extraction doesn't need reinforcement
            
        # Get reinforcement frequency from environment
        reinforce_freq = int(os.environ.get('REINFORCEMENT_FREQUENCY', '0'))
        if reinforce_freq == 0:
            return False  # Reinforcement disabled
            
        # Check message count
        if len(messages) > reinforce_freq:
            return True
            
        # Check if this appears to be a translation task
        all_content = ' '.join([m.get('content', '') for m in messages]).lower()
        return any(keyword in all_content for keyword in self.translation_keywords)
    
    def extract_key_rules(self, system_content: str) -> List[str]:
        """Extract key rules from system prompt - kept for compatibility"""
        return [system_content] if system_content else []
    
    def create_reinforcement_prompt(self, system_content: str, concise: bool = True) -> str:
        """Just return the system prompt as-is for reinforcement"""
        return system_content

class UnifiedClient:
    def __init__(self, model: str, api_key: str):
        self.model = model.lower()
        self.api_key = api_key
        self.context = None
        self.reinforcer = PromptReinforcer()
        
        # History tracking for reinforcement decisions
        self.message_count = 0
        self.last_reinforcement_count = 0

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
        Send messages to the API with automatic prompt reinforcement
        Returns: (content, finish_reason) tuple for backward compatibility
        """
        self.context = context or self.context
        self.message_count += 1
        
        try:
            os.makedirs("Payloads", exist_ok=True)
            
            # Apply prompt reinforcement if needed
            messages = self._apply_reinforcement(messages)
            
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

    def _apply_reinforcement(self, messages: List[Dict]) -> List[Dict]:
        """Apply prompt reinforcement based on API type and context"""
        if not self.reinforcer.needs_reinforcement(messages, self.context):
            return messages
            
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
            
        # Get reinforcement frequency from environment
        reinforce_freq = int(os.environ.get('REINFORCEMENT_FREQUENCY', '10'))
        if reinforce_freq == 0:
            return messages  # Disabled
            
        # Check if it's time to reinforce
        if self.message_count - self.last_reinforcement_count < reinforce_freq:
            return messages
            
        self.last_reinforcement_count = self.message_count
        
        # Apply reinforcement based on client type
        if self.client_type == 'openai' or self.client_type == 'deepseek':
            return self._reinforce_openai_format(messages, system_msg, system_idx)
        elif self.client_type == 'gemini':
            return self._reinforce_gemini_format(messages, system_msg)
        elif self.client_type == 'anthropic':
            return self._reinforce_anthropic_format(messages, system_msg)
        else:
            return messages
    
    def _reinforce_openai_format(self, messages: List[Dict], system_msg: Dict, system_idx: int) -> List[Dict]:
        """Reinforce prompts for OpenAI-style APIs"""
        reinforced_messages = messages.copy()
        
        # For OpenAI, we can add reinforcement to the system message
        reinforcement = self.reinforcer.create_reinforcement_prompt(system_msg['content'], concise=False)
        
        if reinforcement:
            # Create enhanced system message
            enhanced_content = system_msg['content']
            if reinforcement not in enhanced_content:
                enhanced_content = f"{system_msg['content']}\n\n{reinforcement}"
            
            reinforced_messages[system_idx] = {
                'role': 'system',
                'content': enhanced_content
            }
            
            print(f"🔄 Reinforced system prompt for {self.client_type}")
            
        return reinforced_messages
    
    def _reinforce_gemini_format(self, messages: List[Dict], system_msg: Dict) -> List[Dict]:
        """Special reinforcement for Gemini (which concatenates everything)"""
        # For Gemini, we need to restructure the messages
        system_content = system_msg['content']
        rules = self.reinforcer.extract_key_rules(system_content)
        
        # Keep only recent history for Gemini (last 6 messages)
        recent_messages = []
        non_system_messages = [m for m in messages if m.get('role') != 'system']
        
        if len(non_system_messages) > 6:
            recent_messages = non_system_messages[-6:]
        else:
            recent_messages = non_system_messages
            
        # Create a reformatted message list with reinforcement
        reinforced_messages = []
        
        # Always start with system instructions
        reinforced_messages.append({
            'role': 'system',
            'content': f"ACTIVE TRANSLATION RULES:\n{system_content}"
        })
        
        # Add recent context with labels
        if len(recent_messages) > 2:
            reinforced_messages.append({
                'role': 'context',
                'content': "RECENT TRANSLATION CONTEXT:"
            })
            
        # Add recent messages except the last one
        for msg in recent_messages[:-1]:
            reinforced_messages.append(msg)
            
        # Add reminder before the current request
        if rules:
            reminder = f"[CONTINUE FOLLOWING: {'; '.join(rules[:3])}]"
            last_msg = recent_messages[-1].copy()
            last_msg['content'] = f"{reminder}\n\n{last_msg['content']}"
            reinforced_messages.append(last_msg)
        else:
            reinforced_messages.append(recent_messages[-1])
            
        print(f"🔄 Restructured prompt for Gemini (kept {len(recent_messages)} recent messages)")
        
        return reinforced_messages
    
    def _reinforce_anthropic_format(self, messages: List[Dict], system_msg: Dict) -> List[Dict]:
        """Reinforce prompts for Anthropic Claude"""
        # For Claude, we need to handle system messages differently
        system_content = system_msg['content']
        reinforcement = self.reinforcer.create_reinforcement_prompt(system_content, concise=True)
        
        reinforced_messages = []
        for msg in messages:
            if msg.get('role') == 'system':
                # Skip original system message, we'll prepend to first user message
                continue
            elif msg.get('role') == 'user' and len(reinforced_messages) == 0:
                # First user message gets the full system content
                enhanced_content = f"{system_content}\n\n{msg['content']}"
                reinforced_messages.append({
                    'role': 'user',
                    'content': enhanced_content
                })
            elif msg.get('role') == 'user' and reinforcement:
                # Later user messages get brief reminders
                enhanced_content = f"{reinforcement}\n\n{msg['content']}"
                reinforced_messages.append({
                    'role': 'user',
                    'content': enhanced_content
                })
                reinforcement = None  # Only add once
            else:
                reinforced_messages.append(msg)
                
        print(f"🔄 Reinforced prompts for Anthropic")
        
        return reinforced_messages

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
        """Send request to Gemini API with special formatting"""
        # Apply Gemini-specific message formatting
        formatted_prompt = self._format_gemini_prompt(messages)
        
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
    
    def _format_gemini_prompt(self, messages) -> str:
        """Format messages specifically for Gemini's concatenation approach"""
        formatted_parts = []
        
        # Separate messages by role
        system_messages = [m for m in messages if m.get('role') == 'system']
        context_messages = [m for m in messages if m.get('role') == 'context']
        other_messages = [m for m in messages if m.get('role') not in ['system', 'context']]
        
        # System instructions first
        if system_messages:
            formatted_parts.append(system_messages[-1]['content'])  # Use most recent system message
            
        # Context marker if present
        if context_messages:
            formatted_parts.extend([m['content'] for m in context_messages])
            
        # Other messages with role labels
        for msg in other_messages:
            role = msg.get('role', 'user').upper()
            content = msg['content']
            
            # Don't label if it's a reminder or context marker
            if content.startswith('[') or content.startswith('RECENT'):
                formatted_parts.append(content)
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
        
        for msg in messages:
            if msg["role"] in ["system", "context"]:
                # Skip system/context messages, they've been prepended to user messages
                continue
            else:
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
            
            # Apply prompt reinforcement if needed
            messages = self._apply_reinforcement(messages)
            
            payload_name, response_name = self._get_file_names(messages, context)
            self._save_payload(messages, payload_name)
            return self._get_response(messages, temperature, max_tokens, response_name)
        except Exception as e:
            logger.error(f"UnifiedClient error: {e}")
            raise UnifiedClientError(f"UnifiedClient error: {e}") from e
