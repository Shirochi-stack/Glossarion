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
            "ko": "ko",
            "es": "es",
            "fr": "fr",
            "de": "de",
            "it": "it",
            "pt": "pt",
            "ru": "ru",
            "ar": "ar",
            "hi": "hi",
            "tr": "tr"
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
                'https://translate.google.cn/translate_a/single',
            ]
            
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
                    self.logger.warning(f"Failed with endpoint {endpoint_url}: {e}")
                    continue
            
            # If all endpoints failed
            raise Exception("All Google Translate endpoints failed")
            
        except Exception as e:
            self.logger.error(f"Translation failed: {e}")
            # Return original text as fallback
            return {
                'translatedText': text,
                'detectedSourceLanguage': self.source_language,
                'error': str(e)
            }
    
    def _try_single_api_request(self, params: dict, endpoint_url: str, client_type: str) -> Optional[Dict[str, Any]]:
                self.logger.warning(f"Chunk translation failed: {e}")
                translated_parts.append(chunk)
                
        return {
            'translatedText': "".join(translated_parts),
            'detectedSourceLanguage': detected_lang
        }

    def _translate_via_single_api(self, text: str, source_lang: str, target_lang: str, endpoint_url: str) -> Optional[Dict[str, Any]]:
        """Translate using the translate_a/single endpoint (older format)."""
        # Try multiple client types to avoid 403 errors
        client_types = ['gtx', 'webapp', 't', 'dict-chrome-ex', 'android']
        
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
                self.logger.debug(f"Client type {client_type} failed: {e}")
                continue
        
        return None
    
    def _try_single_api_request(self, params: dict, endpoint_url: str, client_type: str) -> Optional[Dict[str, Any]]:
        """Try a single API request with given parameters."""
        # Use POST to avoid URI too long errors for all requests
        response = requests.post(
            endpoint_url,
            data=params,
            headers=self.get_headers(),
            timeout=10
        )
        
        if response.status_code == 200:
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
                            'detectedSourceLanguage': detected_lang
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
                        'detectedSourceLanguage': detected_lang
                    }
            except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
                self.logger.warning(f"Failed to parse single API response with client {client_type}: {e}")
                
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
        
        if response.status_code == 200:
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
                        'detectedSourceLanguage': source_lang
                    }
            except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
                self.logger.warning(f"Failed to parse mobile API response: {e}")
                
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