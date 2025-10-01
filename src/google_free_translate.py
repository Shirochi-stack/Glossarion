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

class GoogleFreeTranslateNew:
    """Google Translate API for free translation using web endpoint."""
    name = 'Google(Free)New'
    free = True
    endpoint: str = 'https://translate-pa.googleapis.com/v1/translate'
    
    def __init__(self, source_language: str = "auto", target_language: str = "en", logger=None):
        self.source_language = source_language
        self.target_language = target_language
        self.logger = logger or logging.getLogger(__name__)
        self.cache = {}  # Simple in-memory cache
        self.cache_lock = threading.Lock()
        self.request_lock = threading.Lock()
        self.last_request_time = 0
        self.rate_limit = 0.5  # 500ms between requests to avoid rate limiting
        
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
        """Get request headers that mimic a browser."""
        return {
            'Accept': '*/*',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'en-US,en;q=0.9',
            'Content-Type': 'application/x-www-form-urlencoded',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                         'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 '
                         'Safari/537.36',
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
            
            # Try multiple endpoint formats
            endpoints_to_try = [
                'https://translate-pa.googleapis.com/v1/translate',
                'https://translate.googleapis.com/translate_a/single',
            ]
            
            for endpoint_url in endpoints_to_try:
                try:
                    if 'translate_a' in endpoint_url:
                        # Use the older public API format
                        result = self._translate_via_single_api(text, source_lang, target_lang, endpoint_url)
                    else:
                        # Use the newer pa API format  
                        result = self._translate_via_pa_api(text, source_lang, target_lang, endpoint_url)
                        
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
    
    def _translate_via_pa_api(self, text: str, source_lang: str, target_lang: str, endpoint_url: str) -> Optional[Dict[str, Any]]:
        """Translate using the translate-pa.googleapis.com endpoint."""
        data = {
            'q': text,
            'source': source_lang,
            'target': target_lang,
            'format': 'text'
        }
        
        response = requests.post(
            endpoint_url,
            data=data,
            headers=self.get_headers(),
            timeout=10
        )
        
        if response.status_code == 200:
            try:
                json_response = response.json()
                if 'data' in json_response and 'translations' in json_response['data']:
                    translations = json_response['data']['translations']
                    if translations:
                        translation = translations[0]
                        return {
                            'translatedText': translation.get('translatedText', text),
                            'detectedSourceLanguage': translation.get('detectedSourceLanguage', source_lang)
                        }
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                self.logger.warning(f"Failed to parse PA API response: {e}")
                
        return None
    
    def _translate_via_single_api(self, text: str, source_lang: str, target_lang: str, endpoint_url: str) -> Optional[Dict[str, Any]]:
        """Translate using the translate_a/single endpoint (older format)."""
        params = {
            'client': 'gtx',
            'sl': source_lang,
            'tl': target_lang,
            'dt': 't',
            'q': text
        }
        
        response = requests.get(
            endpoint_url,
            params=params,
            headers=self.get_headers(),
            timeout=10
        )
        
        if response.status_code == 200:
            try:
                # This endpoint returns a JSON array
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    # Extract translated text
                    translated_parts = []
                    for item in result[0]:
                        if isinstance(item, list) and len(item) > 0:
                            translated_parts.append(item[0])
                    
                    translated_text = ''.join(translated_parts)
                    
                    # Try to detect source language from response
                    detected_lang = source_lang
                    if len(result) > 2 and isinstance(result[2], str):
                        detected_lang = result[2]
                    
                    return {
                        'translatedText': translated_text,
                        'detectedSourceLanguage': detected_lang
                    }
            except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
                self.logger.warning(f"Failed to parse single API response: {e}")
                
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