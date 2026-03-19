"""
grpc_gemini_client.py - Raw gRPC client for Google Generative Language API

Uses google.ai.generativelanguage.v1beta.GenerativeService directly via gRPC
for maximum performance:
  - Binary protobuf serialization (~30-40% smaller payloads vs JSON)
  - HTTP/2 multiplexing (multiple requests over one connection)
  - Native streaming support
  - Connection reuse (no TCP handshake per request)

Usage:
  The user enables this via the "Enable Gemini gRPC Endpoint" checkbox in
  Other Settings, and pastes the gRPC endpoint (default: generativelanguage.googleapis.com).
  The unified_api_client detects the gRPC mode and routes Gemini calls here.

Environment Variables:
  USE_GEMINI_GRPC_ENDPOINT: "1" to enable gRPC transport
  GEMINI_GRPC_ENDPOINT: gRPC host (default: generativelanguage.googleapis.com)
"""

import os
import time
import logging
import threading
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# gRPC availability flag
GRPC_AVAILABLE = False
_grpc = None
_generative_service_pb2 = None
_generative_service_pb2_grpc = None
_content_pb2 = None
_safety_setting_pb2 = None

try:
    import grpc as _grpc_module
    _grpc = _grpc_module
    
    from google.ai.generativelanguage_v1beta import (
        GenerativeServiceClient,
        GenerateContentRequest,
        Content,
        Part,
        GenerationConfig,
        SafetySetting,
        HarmCategory,
    )
    # ThinkingConfig for thinking budget/level support
    try:
        from google.ai.generativelanguage_v1beta.types.generative_service import ThinkingConfig as _ThinkingConfig
    except (ImportError, AttributeError):
        _ThinkingConfig = None
    # Also try to import the raw protobuf stubs for direct gRPC usage
    try:
        from google.ai.generativelanguage_v1beta.services.generative_service import transports
        from google.ai.generativelanguage_v1beta.types import generative_service as gs_types
        from google.ai.generativelanguage_v1beta.types import content as content_types
        from google.ai.generativelanguage_v1beta.types import safety_setting as safety_types
    except ImportError:
        gs_types = None
        content_types = None
        safety_types = None
    
    GRPC_AVAILABLE = True
    logger.debug("gRPC Gemini client: All dependencies available")
except ImportError as e:
    logger.warning(f"gRPC Gemini client: Missing dependencies - {e}")
    GenerativeServiceClient = None
    GenerateContentRequest = None
    Content = None
    Part = None
    GenerationConfig = None
    SafetySetting = None
    HarmCategory = None
    gs_types = None
    content_types = None
    safety_types = None


# Default endpoint
DEFAULT_GRPC_ENDPOINT = "generativelanguage.googleapis.com"


class GrpcGeminiError(Exception):
    """Error from gRPC Gemini client"""
    def __init__(self, message, error_type="api_error", details=None, status_code=None):
        super().__init__(message)
        self.error_type = error_type
        self.details = details or {}
        self.status_code = status_code


class GrpcGeminiResponse:
    """Standardized response from gRPC Gemini calls, compatible with UnifiedResponse"""
    def __init__(self, text="", finish_reason="stop", usage=None, raw_response=None,
                 thinking_tokens=0, raw_content_obj=None):
        self.text = text
        self.finish_reason = finish_reason
        self.usage = usage or {}
        self.raw_response = raw_response
        self.thinking_tokens = thinking_tokens
        self.raw_content_obj = raw_content_obj  # For thought signature support
        
    def __repr__(self):
        return f"GrpcGeminiResponse(text_len={len(self.text)}, finish_reason={self.finish_reason})"


class GrpcGeminiClient:
    """
    Raw gRPC client for Google's Generative Language API.
    
    This uses google.ai.generativelanguage_v1beta's client library in gRPC mode
    for maximum performance. The client library handles protobuf serialization,
    channel management, and authentication internally.
    """
    
    def __init__(self, api_key: str, endpoint: Optional[str] = None):
        """
        Initialize the gRPC Gemini client.
        
        Args:
            api_key: Google AI API key
            endpoint: gRPC endpoint (default: generativelanguage.googleapis.com)
        """
        if not GRPC_AVAILABLE:
            raise GrpcGeminiError(
                "gRPC dependencies not available. Install with:\n"
                "  pip install grpcio google-ai-generativelanguage",
                error_type="import_error"
            )
        
        self.api_key = api_key
        self.endpoint = self._normalize_endpoint(endpoint or DEFAULT_GRPC_ENDPOINT)
        self._client = None
        self._lock = threading.Lock()
        self._channel_created_at = 0
        self._request_count = 0
        
        # Create the client
        self._create_client()
    
    @staticmethod
    def _normalize_endpoint(endpoint: str) -> str:
        """Normalize endpoint string - strip protocols and trailing paths"""
        endpoint = endpoint.strip()
        # Strip common URL prefixes
        for prefix in ["https://", "http://", "grpc://", "grpcs://"]:
            if endpoint.lower().startswith(prefix):
                endpoint = endpoint[len(prefix):]
        # Strip trailing slashes and paths like /v1beta, /openai/, etc.
        # Keep only the host:port
        if "/" in endpoint:
            endpoint = endpoint.split("/")[0]
        # Strip trailing colon if port is missing
        endpoint = endpoint.rstrip(":")
        return endpoint
    
    def _create_client(self):
        """Create or recreate the gRPC client"""
        with self._lock:
            try:
                # Use the high-level client with API key auth and custom endpoint
                from google.api_core import client_options
                
                # Build client options with custom endpoint
                opts = client_options.ClientOptions(
                    api_endpoint=self.endpoint
                )
                
                # CRITICAL: Use AnonymousCredentials to prevent the client from
                # picking up GOOGLE_APPLICATION_CREDENTIALS (ADC) from the environment.
                # In multi-key mode, the ADC may belong to a different project than
                # the current API key, causing "CONSUMER_INVALID" errors.
                # Auth is handled entirely via the x-goog-api-key metadata header.
                try:
                    from google.auth.credentials import AnonymousCredentials
                    anon_creds = AnonymousCredentials()
                except ImportError:
                    anon_creds = None
                
                client_kwargs = {
                    "client_options": opts,
                    "client_info": None,  # Skip default client info
                }
                if anon_creds is not None:
                    client_kwargs["credentials"] = anon_creds
                
                self._client = GenerativeServiceClient(**client_kwargs)
                
                # Override the API key in transport metadata
                # The client uses x-goog-api-key header for API key auth
                self._api_key_header = ("x-goog-api-key", self.api_key)
                
                self._channel_created_at = time.time()
                self._request_count = 0
                
                logger.info(f"gRPC Gemini client created (endpoint: {self.endpoint})")
                
            except Exception as e:
                raise GrpcGeminiError(
                    f"Failed to create gRPC client: {e}",
                    error_type="connection_error"
                )
    
    def _ensure_client(self):
        """Ensure the client is initialized and healthy"""
        if self._client is None:
            self._create_client()
    
    def generate_content(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        safety_disabled: bool = False,
        thinking_budget: int = -1,
        thinking_level: Optional[str] = None,
        is_gemini_3: bool = False,
        supports_thinking: bool = False,
        anti_dupe_params: Optional[Dict] = None,
        stop_check_fn=None,
    ) -> GrpcGeminiResponse:
        """
        Send a generate_content request via gRPC.
        
        Args:
            model: Model name (e.g., "gemini-2.5-flash")
            messages: OpenAI-format messages list
            temperature: Sampling temperature  
            max_output_tokens: Max output tokens
            safety_disabled: Whether to disable safety filters
            thinking_budget: Thinking budget for Gemini 2.5 (-1=dynamic, 0=disabled)
            thinking_level: Thinking level for Gemini 3 (minimal/low/medium/high)
            is_gemini_3: Whether this is a Gemini 3 model
            supports_thinking: Whether model supports thinking
            anti_dupe_params: Anti-duplicate parameters (top_p, top_k, etc.)
            stop_check_fn: Callable that returns True if operation should stop
            
        Returns:
            GrpcGeminiResponse with extracted text and metadata
        """
        self._ensure_client()
        
        if stop_check_fn and stop_check_fn():
            raise GrpcGeminiError("Operation cancelled by user", error_type="cancelled")
        
        # Normalize model name - ensure it has the models/ prefix
        if not model.startswith("models/"):
            model = f"models/{model}"
        
        # Build contents from messages
        contents = self._build_contents(messages)
        
        # Build system instruction (if any system message exists)
        system_instruction = self._extract_system_instruction(messages)
        
        # Build generation config
        gen_config = self._build_generation_config(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            thinking_budget=thinking_budget if supports_thinking and not is_gemini_3 else None,
            thinking_level=thinking_level if supports_thinking and is_gemini_3 else None,
            anti_dupe_params=anti_dupe_params,
        )
        
        # Build safety settings
        safety_settings = self._build_safety_settings(safety_disabled)
        
        # Build the request
        request_kwargs = {
            "model": model,
            "contents": contents,
            "generation_config": gen_config,
        }
        
        if safety_settings:
            request_kwargs["safety_settings"] = safety_settings
        
        if system_instruction:
            request_kwargs["system_instruction"] = system_instruction
        
        try:
            request = GenerateContentRequest(**request_kwargs)
        except Exception as e:
            raise GrpcGeminiError(
                f"Failed to build gRPC request: {e}",
                error_type="validation"
            )
        
        # Make the gRPC call
        try:
            if stop_check_fn and stop_check_fn():
                raise GrpcGeminiError("Operation cancelled by user", error_type="cancelled")
            
            t0 = time.time()
            
            # Use metadata for API key authentication
            metadata = [self._api_key_header]
            
            response = self._client.generate_content(
                request=request,
                metadata=metadata,
            )
            
            elapsed = time.time() - t0
            self._request_count += 1
            
            logger.debug(f"gRPC call completed in {elapsed:.2f}s (request #{self._request_count})")
            
        except Exception as e:
            return self._handle_grpc_error(e)
        
        # Parse the response
        return self._parse_response(response, supports_thinking)
    
    def generate_content_stream(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        safety_disabled: bool = False,
        thinking_budget: int = -1,
        thinking_level: Optional[str] = None,
        is_gemini_3: bool = False,
        supports_thinking: bool = False,
        anti_dupe_params: Optional[Dict] = None,
        stop_check_fn=None,
        log_stream: bool = True,
    ) -> GrpcGeminiResponse:
        """
        Send a streaming generate_content request via gRPC.
        Collects all stream chunks and returns a complete GrpcGeminiResponse.
        """
        self._ensure_client()
        
        if stop_check_fn and stop_check_fn():
            raise GrpcGeminiError("Operation cancelled by user", error_type="cancelled")
        
        # Normalize model name
        if not model.startswith("models/"):
            model = f"models/{model}"
        
        # Build request components
        contents = self._build_contents(messages)
        system_instruction = self._extract_system_instruction(messages)
        gen_config = self._build_generation_config(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            thinking_budget=thinking_budget if supports_thinking and not is_gemini_3 else None,
            thinking_level=thinking_level if supports_thinking and is_gemini_3 else None,
            anti_dupe_params=anti_dupe_params,
        )
        safety_settings = self._build_safety_settings(safety_disabled)
        
        request_kwargs = {
            "model": model,
            "contents": contents,
            "generation_config": gen_config,
        }
        if safety_settings:
            request_kwargs["safety_settings"] = safety_settings
        if system_instruction:
            request_kwargs["system_instruction"] = system_instruction
        
        try:
            request = GenerateContentRequest(**request_kwargs)
        except Exception as e:
            raise GrpcGeminiError(f"Failed to build gRPC request: {e}", error_type="validation")
        
        # Make the streaming gRPC call
        try:
            t0 = time.time()
            metadata = [self._api_key_header]
            
            stream = self._client.stream_generate_content(
                request=request,
                metadata=metadata,
            )
            
            # Collect stream chunks
            text_parts = []
            finish_reason = "stop"
            last_response = None
            thinking_tokens = 0
            log_buf = []
            is_batch = os.getenv("BATCH_TRANSLATION", "0") == "1"
            allow_batch_logs = os.getenv("ALLOW_BATCH_STREAM_LOGS", "0").lower() not in ("0", "false")
            should_log = log_stream and (not is_batch or allow_batch_logs)
            # Thinking streaming state
            grpc_stream_thinking = os.getenv("STREAM_THINKING_LOGS", "1") not in ("0", "false")
            grpc_thinking_started = False
            grpc_thinking_chunks = 0
            grpc_thinking_start_ts = None
            grpc_thinking_log_buf = []  # buffer for accumulating thinking text before printing
            grpc_thinking_text_parts = []  # accumulate all thinking text for token counting
            
            for chunk in stream:
                if stop_check_fn and stop_check_fn():
                    raise GrpcGeminiError("Operation cancelled by user", error_type="cancelled")
                
                last_response = chunk
                
                if chunk.candidates:
                    candidate = chunk.candidates[0]
                    
                    # Check finish reason
                    if candidate.finish_reason:
                        fr_val = candidate.finish_reason
                        # Convert enum to string
                        fr_str = str(fr_val)
                        if "STOP" in fr_str:
                            finish_reason = "stop"
                        elif "MAX_TOKENS" in fr_str:
                            finish_reason = "length"
                        elif "SAFETY" in fr_str:
                            finish_reason = "safety"
                        elif "PROHIBITED" in fr_str:
                            finish_reason = "prohibited_content"
                    
                    # Extract text from parts
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            # Check for thinking/thought parts
                            is_thought = getattr(part, 'thought', False)
                            if is_thought and part.text:
                                grpc_thinking_chunks += 1
                                grpc_thinking_text_parts.append(part.text)
                                if grpc_thinking_start_ts is None:
                                    grpc_thinking_start_ts = time.time()
                                if not grpc_thinking_started:
                                    grpc_thinking_started = True
                                    if grpc_stream_thinking and should_log and not (stop_check_fn and stop_check_fn()):
                                        print(f"🧠 [gemini-grpc] Thinking...", flush=True)
                                if grpc_stream_thinking and should_log and not (stop_check_fn and stop_check_fn()):
                                    grpc_thinking_log_buf.append(part.text)
                                    combined = "".join(grpc_thinking_log_buf)
                                    if "\n" in combined:
                                        parts_split = combined.split("\n")
                                        for p in parts_split[:-1]:
                                            if p.strip().startswith("**") and grpc_thinking_chunks > 1:
                                                print("\u200b", flush=True)
                                            print(f"    {p}", flush=True)
                                        grpc_thinking_log_buf = [parts_split[-1]]
                                continue
                            if part.text:
                                # If switching from thinking to text, print completion
                                if grpc_thinking_started and grpc_stream_thinking and not (stop_check_fn and stop_check_fn()):
                                    # Flush remaining buffered thinking text
                                    remainder = "".join(grpc_thinking_log_buf).rstrip("\n")
                                    if remainder:
                                        for p in remainder.split("\n"):
                                            print(f"    {p}", flush=True)
                                    grpc_thinking_log_buf = []
                                    thinking_dur = time.time() - grpc_thinking_start_ts if grpc_thinking_start_ts else 0
                                    print(f"🧠 [gemini-grpc] Thinking complete ({grpc_thinking_chunks} chunks, {thinking_dur:.1f}s)", flush=True)
                                    grpc_thinking_started = False
                                text_parts.append(part.text)
                                if should_log and not (stop_check_fn and stop_check_fn()):
                                    # Line-buffered streaming output (matches OAI pattern)
                                    frag = part.text.replace("\r", "")
                                    combined = "".join(log_buf) + frag
                                    
                                    # Inject newlines after HTML closing tags for clean line breaks
                                    temp_combined = combined
                                    for tag in ['</h1>', '</h2>', '</h3>', '</h4>', '</h5>', '</h6>', '</p>']:
                                        temp_combined = temp_combined.replace(tag, tag + '\n')
                                    
                                    if "\n" in temp_combined:
                                        parts_split = temp_combined.split("\n")
                                        for ln in parts_split[:-1]:
                                            print(ln)
                                        log_buf = [parts_split[-1]]
                                    else:
                                        log_buf.append(frag)
                                        # Flush if buffer is getting long to show progress
                                        if len("".join(log_buf)) > 150:
                                            print("".join(log_buf), end="", flush=True)
                                            log_buf = []
                
                # Extract thinking tokens from usage metadata
                if chunk.usage_metadata:
                    if hasattr(chunk.usage_metadata, 'thoughts_token_count'):
                        tt = chunk.usage_metadata.thoughts_token_count
                        if tt is not None and tt > 0:
                            thinking_tokens = tt
            
            # Flush remaining log buffer
            if should_log and log_buf:
                remaining = "".join(log_buf)
                if remaining:
                    print(remaining)
                print()  # Final newline
            
            elapsed = time.time() - t0
            text_content = "".join(text_parts)
            self._request_count += 1
            
            if not (stop_check_fn and stop_check_fn()):
                print(f"🛰️ [gemini-grpc] Stream finished in {elapsed:.2f}s, tokens≈{len(text_content)//4}")
            
            # Extract usage info
            usage = {}
            if last_response and last_response.usage_metadata:
                um = last_response.usage_metadata
                usage = {
                    "prompt_tokens": getattr(um, "prompt_token_count", 0) or 0,
                    "completion_tokens": getattr(um, "candidates_token_count", 0) or 0,
                    "total_tokens": getattr(um, "total_token_count", 0) or 0,
                    "thinking_tokens": thinking_tokens,
                }
            
            # Check for prohibited content
            if finish_reason == "prohibited_content":
                raise GrpcGeminiError(
                    "Content blocked: FinishReason.PROHIBITED_CONTENT",
                    error_type="prohibited_content",
                    details={"finish_reason": "PROHIBITED_CONTENT", "thinking_tokens_wasted": thinking_tokens}
                )
            
            return GrpcGeminiResponse(
                text=text_content,
                finish_reason=finish_reason,
                usage=usage,
                raw_response=last_response,
                thinking_tokens=thinking_tokens,
            )
            
        except GrpcGeminiError:
            raise
        except Exception as e:
            return self._handle_grpc_error(e)
    
    def _build_contents(self, messages: List[Dict[str, Any]]) -> List:
        """Convert OpenAI-format messages to Gemini Content objects"""
        contents = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content_val = msg.get("content", "")
            
            # Skip system messages (handled separately as system_instruction)
            if role == "system":
                continue
            
            # Map roles
            gemini_role = "model" if role == "assistant" else "user"
            
            # Handle different content types
            if isinstance(content_val, str):
                parts = [Part(text=content_val)]
            elif isinstance(content_val, list):
                # Multi-part content (text + images)
                parts = []
                for item in content_val:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            parts.append(Part(text=item.get("text", "")))
                        elif item.get("type") == "image_url":
                            # Extract base64 image data
                            url_data = item.get("image_url", {})
                            url = url_data.get("url", "") if isinstance(url_data, dict) else str(url_data)
                            if url.startswith("data:"):
                                # Parse data URI
                                import base64
                                try:
                                    header, b64_data = url.split(",", 1)
                                    mime_type = header.split(":")[1].split(";")[0]
                                    image_bytes = base64.b64decode(b64_data)
                                    from google.ai.generativelanguage_v1beta.types import content as ct
                                    blob = ct.Blob(mime_type=mime_type, data=image_bytes)
                                    parts.append(Part(inline_data=blob))
                                except Exception as e:
                                    logger.warning(f"Failed to parse image data URI: {e}")
                    elif isinstance(item, str):
                        parts.append(Part(text=item))
                if not parts:
                    parts = [Part(text=str(content_val))]
            else:
                parts = [Part(text=str(content_val))]
            
            contents.append(Content(role=gemini_role, parts=parts))
        
        return contents
    
    def _extract_system_instruction(self, messages: List[Dict[str, Any]]) -> Optional[Content]:
        """Extract system message and convert to Gemini system_instruction"""
        system_texts = []
        for msg in messages:
            if msg.get("role") == "system":
                content_val = msg.get("content", "")
                if isinstance(content_val, str):
                    system_texts.append(content_val)
                elif isinstance(content_val, list):
                    for item in content_val:
                        if isinstance(item, dict) and item.get("type") == "text":
                            system_texts.append(item.get("text", ""))
                        elif isinstance(item, str):
                            system_texts.append(item)
        
        if not system_texts:
            return None
        
        combined = "\n".join(system_texts)
        return Content(parts=[Part(text=combined)])
    
    def _build_generation_config(
        self,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        thinking_budget: Optional[int] = None,
        thinking_level: Optional[str] = None,
        anti_dupe_params: Optional[Dict] = None,
    ) -> GenerationConfig:
        """Build GenerationConfig protobuf message"""
        config_kwargs = {}
        
        if temperature is not None:
            config_kwargs["temperature"] = temperature
        
        if max_output_tokens is not None:
            config_kwargs["max_output_tokens"] = max_output_tokens
        
        # Apply anti-duplicate parameters
        if anti_dupe_params:
            if "top_p" in anti_dupe_params and anti_dupe_params["top_p"] is not None:
                config_kwargs["top_p"] = anti_dupe_params["top_p"]
            if "top_k" in anti_dupe_params and anti_dupe_params["top_k"] is not None:
                config_kwargs["top_k"] = anti_dupe_params["top_k"]
            if "candidate_count" in anti_dupe_params and anti_dupe_params["candidate_count"] is not None:
                config_kwargs["candidate_count"] = anti_dupe_params["candidate_count"]
            if "stop_sequences" in anti_dupe_params and anti_dupe_params["stop_sequences"]:
                config_kwargs["stop_sequences"] = anti_dupe_params["stop_sequences"]
            if "presence_penalty" in anti_dupe_params:
                config_kwargs["presence_penalty"] = anti_dupe_params["presence_penalty"]
            if "frequency_penalty" in anti_dupe_params:
                config_kwargs["frequency_penalty"] = anti_dupe_params["frequency_penalty"]
        
        # Apply thinking configuration
        if _ThinkingConfig is not None and (thinking_budget is not None or thinking_level is not None):
            include_thoughts = os.getenv("ENABLE_THOUGHTS", "false").strip().lower() in ("1", "true", "yes", "on")
            thinking_kwargs = {'include_thoughts': include_thoughts}
            if thinking_budget is not None and thinking_budget != -1:
                thinking_kwargs['thinking_budget'] = thinking_budget
            elif thinking_level is not None:
                # Proto only supports thinking_budget (int), not thinking_level
                level_budget_map = {'minimal': 0, 'low': 4096, 'medium': 12288, 'high': 32768 }
                level_str = str(thinking_level).lower()
                budget_val = level_budget_map.get(level_str)
                if budget_val is not None:
                    thinking_kwargs['thinking_budget'] = budget_val
            try:
                config_kwargs['thinking_config'] = _ThinkingConfig(**thinking_kwargs)
            except Exception as e:
                logger.debug(f"Failed to set ThinkingConfig: {e}")
        
        return GenerationConfig(**config_kwargs)
    
    def _build_safety_settings(self, disabled: bool) -> Optional[List]:
        """Build safety settings list"""
        if not disabled:
            return None
        
        # Block none for all categories
        categories = [
            HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            HarmCategory.HARM_CATEGORY_HARASSMENT,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        ]
        
        # Try to add CIVIC_INTEGRITY if available
        try:
            categories.append(HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY)
        except (AttributeError, ValueError):
            pass
        
        settings = []
        for cat in categories:
            settings.append(
                SafetySetting(
                    category=cat,
                    threshold=SafetySetting.HarmBlockThreshold.BLOCK_NONE
                )
            )
        
        return settings
    
    def _parse_response(self, response, supports_thinking: bool = False) -> GrpcGeminiResponse:
        """Parse a gRPC GenerateContentResponse into GrpcGeminiResponse"""
        text_content = ""
        finish_reason = "stop"
        thinking_tokens = 0
        raw_content_obj = None
        
        # Check prompt feedback for blocks
        if response.prompt_feedback:
            feedback = response.prompt_feedback
            if feedback.block_reason:
                raise GrpcGeminiError(
                    f"Content blocked: {feedback.block_reason}",
                    error_type="prohibited_content",
                    details={"block_reason": str(feedback.block_reason)}
                )
        
        # Extract from candidates
        if response.candidates:
            candidate = response.candidates[0]
            
            # Check finish reason
            if candidate.finish_reason:
                fr_str = str(candidate.finish_reason)
                if "STOP" in fr_str:
                    finish_reason = "stop"
                elif "MAX_TOKENS" in fr_str:
                    finish_reason = "length"
                elif "SAFETY" in fr_str:
                    finish_reason = "safety"
                elif "PROHIBITED" in fr_str:
                    finish_reason = "prohibited_content"
            
            # Extract text from content parts
            if candidate.content:
                raw_content_obj = candidate.content
                text_parts = []
                for part in candidate.content.parts:
                    if part.text:
                        text_parts.append(part.text)
                text_content = "".join(text_parts)
        
        # Try simple .text accessor
        if not text_content:
            try:
                if hasattr(response, 'text') and response.text:
                    text_content = response.text
            except Exception:
                pass
        
        # Extract usage metadata
        usage = {}
        if response.usage_metadata:
            um = response.usage_metadata
            usage = {
                "prompt_tokens": getattr(um, "prompt_token_count", 0) or 0,
                "completion_tokens": getattr(um, "candidates_token_count", 0) or 0,
                "total_tokens": getattr(um, "total_token_count", 0) or 0,
            }
            if supports_thinking and hasattr(um, 'thoughts_token_count'):
                thinking_tokens = um.thoughts_token_count or 0
                usage["thinking_tokens"] = thinking_tokens
        
        # Check for prohibited content finish
        if finish_reason == "prohibited_content":
            raise GrpcGeminiError(
                "Content blocked: FinishReason.PROHIBITED_CONTENT",
                error_type="prohibited_content",
                details={
                    "finish_reason": "PROHIBITED_CONTENT",
                    "thinking_tokens_wasted": thinking_tokens
                }
            )
        
        return GrpcGeminiResponse(
            text=text_content,
            finish_reason=finish_reason,
            usage=usage,
            raw_response=response,
            thinking_tokens=thinking_tokens,
            raw_content_obj=raw_content_obj,
        )
    
    def _handle_grpc_error(self, error: Exception) -> GrpcGeminiResponse:
        """Convert gRPC/API errors to GrpcGeminiError"""
        error_str = str(error).lower()
        
        # Check for gRPC status codes
        if _grpc and isinstance(error, _grpc.RpcError):
            code = error.code()
            details = error.details() or str(error)
            
            if code == _grpc.StatusCode.RESOURCE_EXHAUSTED:
                raise GrpcGeminiError(
                    f"Rate limited (gRPC RESOURCE_EXHAUSTED): {details}",
                    error_type="rate_limit",
                    status_code=429
                )
            elif code == _grpc.StatusCode.UNAUTHENTICATED:
                raise GrpcGeminiError(
                    f"Authentication failed (gRPC UNAUTHENTICATED): {details}",
                    error_type="auth_error",
                    status_code=401
                )
            elif code == _grpc.StatusCode.PERMISSION_DENIED:
                raise GrpcGeminiError(
                    f"Permission denied (gRPC PERMISSION_DENIED): {details}",
                    error_type="auth_error",
                    status_code=403
                )
            elif code == _grpc.StatusCode.INVALID_ARGUMENT:
                raise GrpcGeminiError(
                    f"Invalid request (gRPC INVALID_ARGUMENT): {details}",
                    error_type="validation",
                    status_code=400
                )
            elif code == _grpc.StatusCode.NOT_FOUND:
                raise GrpcGeminiError(
                    f"Model not found (gRPC NOT_FOUND): {details}",
                    error_type="api_error",
                    status_code=404
                )
            elif code == _grpc.StatusCode.UNAVAILABLE:
                raise GrpcGeminiError(
                    f"Service unavailable (gRPC UNAVAILABLE): {details}",
                    error_type="api_error",
                    status_code=503
                )
            elif code == _grpc.StatusCode.DEADLINE_EXCEEDED:
                raise GrpcGeminiError(
                    f"Request timed out (gRPC DEADLINE_EXCEEDED): {details}",
                    error_type="timeout",
                    status_code=504
                )
            elif code == _grpc.StatusCode.CANCELLED:
                raise GrpcGeminiError(
                    "Operation cancelled",
                    error_type="cancelled"
                )
            else:
                raise GrpcGeminiError(
                    f"gRPC error ({code.name}): {details}",
                    error_type="api_error",
                    status_code=500
                )
        
        # Generic error handling
        if "cancelled" in error_str or "canceled" in error_str:
            raise GrpcGeminiError("Operation cancelled", error_type="cancelled")
        elif "rate" in error_str and "limit" in error_str:
            raise GrpcGeminiError(str(error), error_type="rate_limit", status_code=429)
        elif "timeout" in error_str:
            raise GrpcGeminiError(str(error), error_type="timeout", status_code=504)
        elif "auth" in error_str or "401" in error_str or "403" in error_str:
            raise GrpcGeminiError(str(error), error_type="auth_error", status_code=401)
        else:
            raise GrpcGeminiError(f"gRPC error: {error}", error_type="api_error")
    
    def close(self):
        """Close the gRPC channel"""
        with self._lock:
            if self._client:
                try:
                    transport = getattr(self._client, '_transport', None)
                    if transport and hasattr(transport, 'close'):
                        transport.close()
                except Exception:
                    pass
                self._client = None
    
    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def check_grpc_dependencies() -> Tuple[bool, str]:
    """
    Check if all dependencies for gRPC Gemini client are available.
    Returns (available, message).
    """
    issues = []
    
    try:
        import grpc
    except ImportError:
        issues.append("grpcio not installed (pip install grpcio)")
    
    try:
        from google.ai.generativelanguage_v1beta import GenerativeServiceClient
    except ImportError:
        issues.append("google-ai-generativelanguage not installed (pip install google-ai-generativelanguage)")
    
    if issues:
        return False, "Missing: " + "; ".join(issues)
    
    return True, "All gRPC dependencies available"
