# google_free_translate.py
"""
Free Google Translate API implementation using the web endpoint.
No API key or billing required!
"""
import logging
import threading
import time
import json
import requests
from typing import Optional, Dict, Any
import urllib.parse
import random

class GoogleFreeTranslateNew:
    """Google Translate API for free translation using public web endpoints.
    
    Uses the same endpoints as the Google Translate web interface and mobile apps.
    No API key or Google Cloud credentials required!
    """
    name = 'Google(Free)New'
    free = True
    endpoint: str = 'https://translate.googleapis.com/translate_a/single'
    
    def __init__(self, source_language: str = "auto", target_language: str = "en", logger=None):
        self.source_language = source_language
        self.target_language = target_language
        self.logger = logger or logging.getLogger(__name__)
        self.cache = {}  # Simple in-memory cache
        self.cache_lock = threading.Lock()
        self.request_lock = threading.Lock()
        self.last_request_time = 0
        self.rate_limit = 0.5  # 500ms between requests to avoid rate limiting
        
        # Use a CLASS-LEVEL flag for permanent fallback so it persists across instances
        if not hasattr(GoogleFreeTranslateNew, '_use_fallback_only'):
            GoogleFreeTranslateNew._use_fallback_only = False
        
        # Multiple user agents to rotate through (helps avoid 403)
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
        ]
        
    def _get_source_code(self):
        """Get the source language code."""
        lang_map = {
            "zh": "zh-CN", 
            "ja": "ja", 
            "ko": "ko", 
            "en": "en",
            "auto": "auto"
        }
        return lang_map.get(self.source_language, self.source_language)

    def _get_target_code(self):
        """Get the target language code."""
        lang_map = {
            "en": "en", 
            "zh": "zh-CN", 
            "ja": "ja", 
            "ko": "ko"
        }
        return lang_map.get(self.target_language, self.target_language)

    def get_headers(self):
        """Get request headers that mimic a browser with random user agent."""
        return {
            'Accept': '*/*',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'en-US,en;q=0.9',
            'Content-Type': 'application/x-www-form-urlencoded',
            'User-Agent': random.choice(self.user_agents),  # Random user agent to avoid detection
            'Referer': 'https://translate.google.com/',
            'Origin': 'https://translate.google.com',
        }
        
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for the translation."""
        return f"{self.source_language}_{self.target_language}_{hash(text)}"
    
    def _rate_limit_delay(self):
        """Ensure rate limiting between requests."""
        with self.request_lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.rate_limit:
                sleep_time = self.rate_limit - time_since_last
                time.sleep(sleep_time)
            self.last_request_time = time.time()
    
    def translate(self, text: str) -> Dict[str, Any]:
        """
        Translate text using Google's free web API.
        
        Args:
            text: Text to translate
            
        Returns:
            Dict with 'translatedText' and 'detectedSourceLanguage' keys
        """
        if not text or not text.strip():
            return {
                'translatedText': '',
                'detectedSourceLanguage': self.source_language
            }
            
        # Check cache first
        cache_key = self._get_cache_key(text)
        with self.cache_lock:
            if cache_key in self.cache:
                self.logger.debug(f"Cache hit for translation: {text[:50]}...")
                return self.cache[cache_key]
        
        # If we've already determined Google is blocking us, skip straight to fallback
        if GoogleFreeTranslateNew._use_fallback_only:
            try:
                # Prepare source lang for fallback logic (convert 'auto' if needed)
                # Note: _translate_via_argos handles 'auto' logic internally, so just pass self.source_language
                # but map Google codes if needed
                source_code = self._get_source_code()
                target_code = self._get_target_code()
                
                # Use fallback immediately
                # No logging needed here as we want to be quiet about it
                argos_result = self._translate_via_argos(text, source_code, target_code)
                if argos_result:
                    with self.cache_lock:
                        self.cache[cache_key] = argos_result
                    return argos_result
                
                # If fallback fails even in fallback-only mode, raise exception
                raise Exception("Argos fallback failed in permanent fallback mode")
            except Exception as e:
                # If fallback fails, log error and return original text
                self.logger.error(f"❌ Permanent fallback failed: {e}")
                return {
                    'translatedText': text,
                    'detectedSourceLanguage': self.source_language,
                    'error': str(e)
                }

        # Log start for Google endpoints (pre-call)
        try:
            self.logger.info(f"🌐 Google Translate Free: Translating {len(text)} characters")
        except Exception:
            pass

        # Rate limiting
        self._rate_limit_delay()
        
        try:
            # Prepare request data
            source_lang = self._get_source_code()
            target_lang = self._get_target_code()
            
            # Try multiple FREE endpoint formats (no credentials needed)
            # These are public endpoints used by the web interface and mobile apps
            # Ordered from most reliable to least reliable based on common usage
            endpoints_to_try = [
                'https://translate.googleapis.com/translate_a/single',  # Most reliable public endpoint
                'https://translate.google.com/translate_a/single',  # Direct web endpoint
                'https://clients5.google.com/translate_a/t',  # Mobile client endpoint
                'https://clients5.google.com/translate_a/single',  # Alternative client5
                'https://clients1.google.com/translate_a/single',
                'https://clients3.google.com/translate_a/t',
            ]
            
            # Collect all errors for detailed reporting
            all_errors = []
            
            for endpoint_url in endpoints_to_try:
                try:
                    # Check if it's a mobile endpoint (t) or single api
                    is_mobile = '/t' in endpoint_url
                    
                    if is_mobile:
                        # Use mobile client API format
                        result = self._translate_via_mobile_api(text, source_lang, target_lang, endpoint_url)
                    else:
                        # Use the public translate_a/single API format
                        result = self._translate_via_single_api(text, source_lang, target_lang, endpoint_url)
                        
                    if result:
                        # Cache successful result
                        with self.cache_lock:
                            self.cache[cache_key] = result
                            # Limit cache size
                            if len(self.cache) > 1000:
                                # Remove oldest entries
                                oldest_keys = list(self.cache.keys())[:100]
                                for key in oldest_keys:
                                    del self.cache[key]
                        
                        return result
                        
                except Exception as e:
                    error_msg = f"{endpoint_url}: {e}"
                    all_errors.append(error_msg)
                    self.logger.warning(f"⚠️ {error_msg}")
                    continue
            
            # If all endpoints failed, report all errors
            error_summary = "\n".join(f"  • {err}" for err in all_errors)
            detailed_error = f"All Google Translate endpoints failed:\n{error_summary}"
            self.logger.error(detailed_error)
            
            # Fallback to Argos Translate
            self.logger.info("🔄 All Google endpoints failed. Switching to permanent Argos Translate fallback...")
            GoogleFreeTranslateNew._use_fallback_only = True  # Set flag to skip Google endpoints for future requests
            self.logger.info("🧩 Using Argos Translate fallback for this request")
            
            argos_result = self._translate_via_argos(text, source_lang, target_lang)
            if argos_result:
                self.logger.info("✅ Argos Translate fallback successful")
                # Cache the result to avoid repeated fallback overhead
                with self.cache_lock:
                    self.cache[cache_key] = argos_result
                return argos_result
            
            raise Exception(detailed_error)
            
        except Exception as e:
            # Only log if it's not already logged above
            if "All Google Translate endpoints failed" not in str(e):
                self.logger.error(f"❌ Translation failed: {e}")
            # Return original text as fallback
            return {
                'translatedText': text,
                'detectedSourceLanguage': self.source_language,
                'error': str(e)
            }
    
    
    def _mask_html_tags(self, text: str):
        """Replace HTML tags/entities with Unicode PUA placeholders to avoid translation corruption."""
        import re
        if ('<' not in text or '>' not in text) and '&' not in text:
            return text, {}

        tag_pattern = re.compile(r'<[^>]+>')
        entity_pattern = re.compile(r'&[A-Za-z0-9#]+;')

        tag_list = []
        ent_list = []

        # Private Use Area base (U+E000)
        TAG_BASE = 0xE000
        ENT_BASE = 0xF000  # separate block to avoid collisions

        def _pua(codepoint):
            try:
                return chr(codepoint)
            except Exception:
                return ''

        def _tag_repl(match):
            idx = len(tag_list)
            tag_list.append(match.group(0))
            return _pua(TAG_BASE + idx)

        def _ent_repl(match):
            idx = len(ent_list)
            ent_list.append(match.group(0))
            return _pua(ENT_BASE + idx)

        # Mask tags first (protect inline attributes inside tags)
        masked = tag_pattern.sub(_tag_repl, text)
        # Mask entities in remaining text
        masked = entity_pattern.sub(_ent_repl, masked)

        tag_map = {chr(TAG_BASE + i): tag for i, tag in enumerate(tag_list)}
        ent_map = {chr(ENT_BASE + i): ent for i, ent in enumerate(ent_list)}
        tag_map.update(ent_map)
        return masked, tag_map

    def _unmask_html_tags(self, text: str, tag_map: dict):
        """Restore HTML tags/entities from Unicode PUA placeholders."""
        if not tag_map:
            return text
        for placeholder, tag in tag_map.items():
            if placeholder in text:
                text = text.replace(placeholder, tag)
        return text
    def _translate_via_argos(self, text: str, source_lang: str, target_lang: str) -> Optional[Dict[str, Any]]:
        """Fallback to Argos Translate (offline-capable)."""
        try:
            import argostranslate.package
            import argostranslate.translate
        except ImportError:
            self.logger.warning("⚠️ Argos Translate not installed. Installing...")
            import subprocess
            import sys
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "argostranslate"])
                import argostranslate.package
                import argostranslate.translate
            except Exception as e:
                self.logger.error(f"❌ Failed to install Argos Translate: {e}")
                return None

        try:
            # Map Google codes to Argos codes (ISO 639-1 generally)
            code_map = {
                'zh-CN': 'zh',
                'zh-TW': 'zh',
                'zh': 'zh',
                'ja': 'ja',
                'ko': 'ko',
                'en': 'en'
            }
            
            argos_source = code_map.get(source_lang, source_lang)
            argos_target = code_map.get(target_lang, target_lang)
            
            # Auto-detect source if 'auto' using improved heuristics
            if argos_source == 'auto':
                sample = text[:1000]  # Check first 1000 chars for better accuracy
                
                # 1. Check CJK ranges first (most reliable)
                if any('\u4e00' <= char <= '\u9fff' for char in sample):
                    argos_source = 'zh'
                elif any('\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff' for char in sample):
                    argos_source = 'ja'
                elif any('\uac00' <= char <= '\ud7af' for char in sample):
                    argos_source = 'ko'
                # 2. Check Cyrillic (Russian, Ukrainian, etc.)
                elif any('\u0400' <= char <= '\u04FF' for char in sample):
                    argos_source = 'ru'
                # 3. Check for specific language characteristics
                else:
                    # Simple heuristic based on common words/chars
                    lower_sample = sample.lower()
                    if any(x in lower_sample for x in ['the', 'and', 'is', 'it']):
                        argos_source = 'en'
                    elif any(x in lower_sample for x in ['le', 'la', 'les', 'et', 'est']):
                        argos_source = 'fr'
                    elif any(x in lower_sample for x in ['el', 'la', 'los', 'las', 'y', 'es']):
                        argos_source = 'es'
                    elif any(x in lower_sample for x in ['der', 'die', 'das', 'und', 'ist']):
                        argos_source = 'de'
                    elif any(x in lower_sample for x in ['il', 'lo', 'la', 'e', 'è']):
                        argos_source = 'it'
                    elif any(x in lower_sample for x in ['o', 'a', 'os', 'as', 'e', 'é']):
                        argos_source = 'pt'
                    elif any(x in lower_sample for x in ['bir', 've', 'bu', 'da']):
                        argos_source = 'tr'
                    else:
                        # Default to English if detection fails
                        argos_source = 'en' 
                
                self.logger.info(f"🔍 Argos auto-detected source: {argos_source}")

            # Check installed languages
            installed_languages = argostranslate.translate.get_installed_languages()
            from_lang = next((l for l in installed_languages if l.code == argos_source), None)
            to_lang = next((l for l in installed_languages if l.code == argos_target), None)
            
            translation_available = from_lang and to_lang and from_lang.get_translation(to_lang)
            
            if not translation_available:
                self.logger.info(f"⬇️ Downloading Argos model {argos_source}->{argos_target}...")
                try:
                    argostranslate.package.update_package_index()
                    available_packages = argostranslate.package.get_available_packages()
                    package_to_install = next(
                        filter(
                            lambda x: x.from_code == argos_source and x.to_code == argos_target, available_packages
                        ), None
                    )
                    if package_to_install:
                        argostranslate.package.install_from_path(package_to_install.download())
                        # Reload installed languages
                        installed_languages = argostranslate.translate.get_installed_languages()
                        from_lang = next((l for l in installed_languages if l.code == argos_source), None)
                        to_lang = next((l for l in installed_languages if l.code == argos_target), None)
                    else:
                        self.logger.error(f"❌ No Argos model found for {argos_source}->{argos_target}")
                        return None
                except Exception as e:
                    self.logger.error(f"❌ Failed to download Argos model (machine might be offline): {e}")
                    return None

            if from_lang and to_lang:
                translation = from_lang.get_translation(to_lang)
                try:
                    self.logger.info(f"🌐 Argos Translate: Translating {len(text)} characters")
                except Exception:
                    pass
                # Mask HTML tags to avoid corruption during Argos translation
                masked_text, tag_map = self._mask_html_tags(text)
                translated_text = translation.translate(masked_text)
                translated_text = self._unmask_html_tags(translated_text, tag_map)
                return {
                    'translatedText': translated_text,
                    'detectedSourceLanguage': argos_source,
                    'provider': 'argos'
                }
            
            return None

        except Exception as e:
            self.logger.error(f"❌ Argos fallback failed: {e}")
            return None
    
    def _translate_via_single_api(self, text: str, source_lang: str, target_lang: str, endpoint_url: str) -> Optional[Dict[str, Any]]:
        """Translate using the translate_a/single endpoint (older format)."""
        # Try multiple client types to avoid 403 errors
        client_types = ['gtx', 'webapp', 't', 'dict-chrome-ex', 'android']
        
        last_error = None
        for client_type in client_types:
            params = {
                'client': client_type,
                'sl': source_lang,
                'tl': target_lang,
                'dt': 't',
                'q': text
            }
            
            # Add additional parameters for certain clients
            if client_type == 'webapp':
                params['dj'] = '1'  # Return JSON format
                params['dt'] = ['t', 'bd', 'ex', 'ld', 'md', 'qca', 'rw', 'rm', 'ss', 'at']
            elif client_type == 'dict-chrome-ex':
                params['dt'] = ['t', 'bd']
            
            try:
                result = self._try_single_api_request(params, endpoint_url, client_type)
                if result:
                    return result
            except Exception as e:
                last_error = e
                self.logger.debug(f"Client type {client_type} failed: {e}")
                continue
        
        # If we got here, all client types failed - raise the last error
        if last_error:
            raise last_error
        return None
    
    def _try_single_api_request(self, params: dict, endpoint_url: str, client_type: str) -> Optional[Dict[str, Any]]:
        """Try a single API request with given parameters."""
        # Use POST to avoid URI too long errors
        response = requests.post(
            endpoint_url,
            data=params,
            headers=self.get_headers(),
            timeout=10
        )
        
        # Log response details for debugging
        if response.status_code != 200:
            # Extract error reason from HTML title if possible
            error_reason = "Unknown"
            if "403" in str(response.status_code):
                error_reason = "Forbidden (IP blocked or bot detected)"
            elif "429" in str(response.status_code):
                error_reason = "Rate Limited (too many requests)"
            elif "404" in str(response.status_code):
                error_reason = "Not Found (endpoint doesn't exist)"
            elif "500" in str(response.status_code):
                error_reason = "Server Error"
            
            error_msg = f"HTTP {response.status_code}: {error_reason}"
            raise Exception(error_msg)
        
        try:
            result = response.json()
            
            # Handle webapp format (dj=1 returns different structure)
            if client_type == 'webapp' and isinstance(result, dict):
                if 'sentences' in result:
                    translated_parts = []
                    for sentence in result['sentences']:
                        if 'trans' in sentence:
                            translated_parts.append(sentence['trans'])
                    
                    translated_text = ''.join(translated_parts)
                    detected_lang = result.get('src', params.get('sl', 'auto'))
                    
                    return {
                        'translatedText': translated_text,
                        'detectedSourceLanguage': detected_lang,
                        'provider': 'google'
                    }
            
            # Handle standard array format
            elif isinstance(result, list) and len(result) > 0:
                # Extract translated text
                translated_parts = []
                for item in result[0]:
                    if isinstance(item, list) and len(item) > 0:
                        translated_parts.append(item[0])
                
                translated_text = ''.join(translated_parts)
                
                # Try to detect source language from response
                detected_lang = params.get('sl', 'auto')
                if len(result) > 2 and isinstance(result[2], str):
                    detected_lang = result[2]
                
                return {
                    'translatedText': translated_text,
                    'detectedSourceLanguage': detected_lang,
                    'provider': 'google'
                }
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
            parse_error = f"Parse Error: {type(e).__name__}"
            raise Exception(parse_error)
            
        return None
    
    def _translate_via_mobile_api(self, text: str, source_lang: str, target_lang: str, endpoint_url: str) -> Optional[Dict[str, Any]]:
        """Translate using the mobile client endpoint (clients5.google.com/translate_a/t)."""
        params = {
            'client': 't',  # Mobile client
            'sl': source_lang,
            'tl': target_lang,
            'q': text
        }
        
        # Use different headers for mobile endpoint
        mobile_headers = {
            'Accept': '*/*',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'en-US,en;q=0.9',
            'User-Agent': 'GoogleTranslate/6.29.59279 (Linux; U; Android 11; Pixel 5)',
            'Content-Type': 'application/x-www-form-urlencoded',
        }
        
        response = requests.post(
            endpoint_url,
            data=params,
            headers=mobile_headers,
            timeout=10
        )
        
        if response.status_code != 200:
            # Extract error reason
            error_reason = "Unknown"
            if "403" in str(response.status_code):
                error_reason = "Forbidden (IP blocked or bot detected)"
            elif "429" in str(response.status_code):
                error_reason = "Rate Limited (too many requests)"
            elif "404" in str(response.status_code):
                error_reason = "Not Found (endpoint doesn't exist)"
            elif "500" in str(response.status_code):
                error_reason = "Server Error"
            
            error_msg = f"HTTP {response.status_code}: {error_reason}"
            raise Exception(error_msg)
        
        try:
            # This endpoint returns a simple JSON array
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                # Extract translated text from first element
                if isinstance(result[0], str):
                    translated_text = result[0]
                elif isinstance(result[0], list) and len(result[0]) > 0:
                    translated_text = result[0][0]
                else:
                    return None
                
                return {
                    'translatedText': translated_text,
                    'detectedSourceLanguage': source_lang,
                    'provider': 'google'
                }
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
            parse_error = f"Parse Error: {type(e).__name__}"
            raise Exception(parse_error)
            
        return None

# Convenience function for easy usage
def translate_text(text: str, source_lang: str = "auto", target_lang: str = "en") -> str:
    """
    Simple function to translate text using Google's free API.
    
    Args:
        text: Text to translate
        source_lang: Source language code (default: auto-detect)  
        target_lang: Target language code (default: en)
        
    Returns:
        Translated text
    """
    translator = GoogleFreeTranslateNew(source_lang, target_lang)
    result = translator.translate(text)
    return result.get('translatedText', text)

if __name__ == "__main__":
    from shutdown_utils import run_cli_main
    def _main():
        # Test the translator
        translator = GoogleFreeTranslateNew("auto", "en")
        
        test_texts = [
            "こんにちは、世界！",  # Japanese
            "안녕하세요 세계!",      # Korean  
            "你好世界！",           # Chinese
            "Hola mundo"           # Spanish
        ]
        
        for test_text in test_texts:
            result = translator.translate(test_text)
            print(f"Original: {test_text}")
            print(f"Translated: {result['translatedText']}")
            print(f"Detected language: {result['detectedSourceLanguage']}")
            print("-" * 50)
        return 0
    run_cli_main(_main)
