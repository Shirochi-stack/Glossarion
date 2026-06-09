# google_free_translate.py
"""
Free Google Translate API implementation using the web endpoint.
No API key or billing required!
"""
import logging
import threading
import time
import json
import html as html_lib
import os
import re
import requests
from typing import Optional, Dict, Any
import urllib.parse
import random
from html.parser import HTMLParser

GOOGLE_TRANSLATE_CO_IN_ENDPOINT = 'https://translate.google.co.in/translate_a/single'
GOOGLETRANS_AJAX_ENDPOINT = 'https://translate.googleapis.com/translate_a/single'

GOOGLE_TRANSLATE_LANGUAGE_CODES = {
    "english": "en",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "portuguese": "pt",
    "russian": "ru",
    "arabic": "ar",
    "hindi": "hi",
    "chinese": "zh-CN",
    "chinese simplified": "zh-CN",
    "chinese (simplified)": "zh-CN",
    "simplified chinese": "zh-CN",
    "chinese traditional": "zh-TW",
    "chinese (traditional)": "zh-TW",
    "traditional chinese": "zh-TW",
    "japanese": "ja",
    "korean": "ko",
    "turkish": "tr",
    "vietnamese": "vi",
    "auto": "auto",
}

GOOGLE_TRANSLATE_CODE_ALIASES = {
    "zh": "zh-CN",
    "zh-cn": "zh-CN",
    "zh-hans": "zh-CN",
    "zh-tw": "zh-TW",
    "zh-hant": "zh-TW",
}


class _MarkedHtmlSegmentParser(HTMLParser):
    """Extract flat data-sdl-tip tagged segments without using an HTML repair parser."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.segments = []
        self._current = None

    def handle_starttag(self, tag, attrs):
        attrs = list(attrs or [])
        attr_map = {str(name).lower(): value for name, value in attrs}
        if self._current is None and "data-sdl-tip" in attr_map:
            self._current = {"tag": tag, "attrs": attrs, "text": []}
            return
        if self._current is not None:
            self._current["text"].append(self.get_starttag_text() or "")

    def handle_endtag(self, tag):
        if self._current is not None and tag.lower() == str(self._current.get("tag", "")).lower():
            self.segments.append(self._current)
            self._current = None
            return
        if self._current is not None:
            self._current["text"].append(f"</{tag}>")

    def handle_data(self, data):
        if self._current is not None:
            self._current["text"].append(data)

    def handle_entityref(self, name):
        if self._current is not None:
            self._current["text"].append(f"&{name};")

    def handle_charref(self, name):
        if self._current is not None:
            self._current["text"].append(f"&#{name};")


def _html_start_tag(tag, attrs):
    parts = [f"<{tag}"]
    for name, value in attrs or []:
        if value is None:
            parts.append(f" {name}")
        else:
            parts.append(f' {name}="{html_lib.escape(str(value), quote=True)}"')
    parts.append(">")
    return "".join(parts)


def google_translate_language_code(language: Any, default: str = "en") -> str:
    """Normalize GUI display names or language-code values for Google Translate."""
    if language is None:
        return default

    value = str(language).strip()
    if not value:
        return default

    normalized = " ".join(value.lower().replace("_", "-").split())
    normalized_name = normalized.replace("（", "(").replace("）", ")")
    normalized_name = normalized_name.replace("-", " ")

    if normalized_name in GOOGLE_TRANSLATE_LANGUAGE_CODES:
        return GOOGLE_TRANSLATE_LANGUAGE_CODES[normalized_name]

    normalized_code = normalized.replace(" ", "-")
    if normalized_code in GOOGLE_TRANSLATE_CODE_ALIASES:
        return GOOGLE_TRANSLATE_CODE_ALIASES[normalized_code]

    if 2 <= len(normalized_code) <= 3 and normalized_code.isalpha():
        return normalized_code

    if len(normalized_code) == 5 and normalized_code[2] == "-":
        lang, region = normalized_code.split("-", 1)
        if lang.isalpha() and region.isalpha():
            return f"{lang}-{region.upper()}"

    return default


class GoogleFreeTranslateNew:
    """Google Translate API for free translation using public web endpoints.
    
    Uses the same endpoints as the Google Translate web interface and mobile apps.
    No API key or Google Cloud credentials required!
    """
    name = 'Google(Free)New'
    free = True
    endpoint: str = GOOGLE_TRANSLATE_CO_IN_ENDPOINT
    
    def __init__(
        self,
        source_language: str = "auto",
        target_language: str = "en",
        logger=None,
        provider: str = "auto",
        api_keys: Optional[Dict[str, Any]] = None,
        honor_global_stop: bool = True,
    ):
        self.source_language = source_language
        self.target_language = target_language
        self.logger = logger or logging.getLogger(__name__)
        self.provider = self._normalize_provider(provider)
        self.api_keys = dict(api_keys or {})
        self.honor_global_stop = bool(honor_global_stop)
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
        return google_translate_language_code(self.source_language, default=self.source_language)

    def _get_target_code(self):
        """Get the target language code."""
        return google_translate_language_code(self.target_language, default=self.target_language)

    @staticmethod
    def _normalize_provider(provider: Any) -> str:
        normalized = str(provider or "auto").strip().lower().replace("_", "-").replace(" ", "-")
        aliases = {
            "": "auto",
            "machine": "auto",
            "machine-translation": "auto",
            "google-free": "google",
            "google-translate": "google",
            "google-translate-free": "google",
            "argos-translate": "argos",
            "argostranslate": "argos",
            "microsoft": "bing",
            "microsoft-translator": "bing",
            "azure": "bing",
            "azure-translator": "bing",
            "yandex-translate": "yandex",
        }
        normalized = aliases.get(normalized, normalized)
        return normalized if normalized in {"auto", "google", "deepl", "bing", "argos", "yandex"} else "auto"

    @staticmethod
    def _is_marked_html_batch(text: str) -> bool:
        return "data-sdl-tip" in str(text or "")

    @staticmethod
    def _deepl_target_language_code(target_lang: str) -> str:
        code = google_translate_language_code(target_lang, default=target_lang).upper()
        mapping = {
            "ZH-CN": "ZH-HANS",
            "ZH-TW": "ZH-HANT",
            "PT-BR": "PT-BR",
            "PT-PT": "PT-PT",
            "EN": "EN-US",
        }
        return mapping.get(code, code)

    @staticmethod
    def _yandex_language_code(lang: str) -> str:
        code = google_translate_language_code(lang, default=lang)
        aliases = {
            "zh-CN": "zh",
            "zh-TW": "zh",
        }
        return aliases.get(code, str(code or "").split("-", 1)[0].lower())

    def _api_option(self, provider: str, option: str, env_name: str = "") -> str:
        value = None
        provider_value = self.api_keys.get(provider)
        if isinstance(provider_value, dict):
            value = provider_value.get(option)
            if value is None and option == "api_key":
                value = provider_value.get("key")
        elif option == "api_key":
            value = provider_value
        if value is None:
            value = self.api_keys.get(f"{provider}_{option}")
        if value is None and env_name:
            value = os.environ.get(env_name, "")
        return str(value or "").strip()

    def _api_key(self, provider: str, *env_names: str) -> str:
        value = self.api_keys.get(provider)
        if isinstance(value, dict):
            value = value.get("api_key") or value.get("key")
        elif value is None:
            value = self.api_keys.get(f"{provider}_api_key")
        for env_name in env_names:
            if value:
                break
            value = os.environ.get(env_name, "")
        value = str(value or "").strip()
        if not value:
            raise ValueError(f"{provider} API key is not configured")
        return value

    def _folder_id(self, provider: str, env_name: str) -> str:
        value = self.api_keys.get(provider)
        if isinstance(value, dict):
            value = value.get("folder_id") or value.get("folderId")
        else:
            value = self.api_keys.get(f"{provider}_folder_id")
        value = str(value or os.environ.get(env_name, "") or "").strip()
        if not value:
            raise ValueError(f"{provider} folder id is not configured")
        return value

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
        return f"{self.provider}_{self.source_language}_{self.target_language}_{hash(text)}"

    def _cache_translation_result(self, cache_key: str, result: Dict[str, Any]) -> Dict[str, Any]:
        with self.cache_lock:
            self.cache[cache_key] = result
            if len(self.cache) > 1000:
                oldest_keys = list(self.cache.keys())[:100]
                for key in oldest_keys:
                    del self.cache[key]
        return result

    def _translation_error_result(self, provider: str, error: Any) -> Dict[str, Any]:
        return {
            'translatedText': '',
            'detectedSourceLanguage': self.source_language,
            'provider': provider,
            'error': str(error),
        }

    def _stop_requested(self) -> bool:
        if not getattr(self, "honor_global_stop", True):
            return False
        try:
            import unified_api_client as _uac
            return bool(getattr(_uac, "is_stop_requested", lambda: False)())
        except Exception:
            return False

    def _raise_if_stop_requested(self):
        if self._stop_requested():
            raise Exception("Operation cancelled")

    def _translate_via_configured_provider(
        self,
        provider: str,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> Optional[Dict[str, Any]]:
        if provider == "argos":
            return self._translate_via_argos(text, source_lang, target_lang)
        if provider == "deepl":
            return self._translate_via_deepl(text, source_lang, target_lang)
        if provider == "bing":
            return self._translate_via_bing(text, source_lang, target_lang)
        if provider == "yandex":
            return self._translate_via_yandex(text, source_lang, target_lang)
        raise ValueError(f"Unknown machine translation provider: {provider}")
    
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

        if self._stop_requested():
            return self._translation_error_result(self.provider, "Operation cancelled")
            
        # Check cache first
        cache_key = self._get_cache_key(text)
        with self.cache_lock:
            if cache_key in self.cache:
                self.logger.debug(f"Cache hit for translation: {text[:50]}...")
                return self.cache[cache_key]
        
        provider = self.provider
        source_lang = self._get_source_code()
        target_lang = self._get_target_code()

        if provider in {"argos", "deepl", "bing", "yandex"}:
            try:
                result = self._translate_via_configured_provider(provider, text, source_lang, target_lang)
                if result and str(result.get("translatedText") or "").strip():
                    return self._cache_translation_result(cache_key, result)
                raise Exception(f"{provider} returned no translation")
            except Exception as e:
                self.logger.error(f"❌ {provider} translation failed: {e}")
                return self._translation_error_result(provider, e)

        # If we've already determined Google is blocking us, skip straight to fallback.
        # Explicit Google mode should still try Google and report Google failures.
        if provider == "auto" and GoogleFreeTranslateNew._use_fallback_only:
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
                
                # If fallback fails even in fallback-only mode, and Argos is unavailable, disable fallback-only
                # and continue with Google endpoints.
                try:
                    import importlib.util
                    if importlib.util.find_spec("argostranslate") is None:
                        GoogleFreeTranslateNew._use_fallback_only = False
                        return self.translate(text)
                except Exception:
                    pass
                # If fallback fails even in fallback-only mode, raise exception
                raise Exception("Argos fallback failed in permanent fallback mode")
            except Exception as e:
                # If fallback fails, log error and return original text
                self.logger.error(f"❌ Permanent fallback failed: {e}")
                return self._translation_error_result("argos", e)

        # Log start for Google endpoints (pre-call)
        try:
            self._raise_if_stop_requested()
            self.logger.info(f"🌐 Google Translate Free: Translating {len(text)} characters")
        except Exception:
            pass

        # Rate limiting
        self._rate_limit_delay()
        
        try:
            # Prepare request data
            # Try multiple FREE endpoint formats (no credentials needed).
            # Keep the requested translate.google.co.in host first; it accepts longer POST bodies.
            endpoints_to_try = [
                self.endpoint,
                'https://translate.google.com/translate_a/single',  # Direct web endpoint
                'https://clients5.google.com/translate_a/t',  # Mobile client endpoint
                'https://clients5.google.com/translate_a/single',  # Alternative client5
                'https://clients1.google.com/translate_a/single',
                'https://clients3.google.com/translate_a/t',
                GOOGLETRANS_AJAX_ENDPOINT,  # Last-resort py-googletrans-style Ajax endpoint
            ]
            
            # Collect all errors for detailed reporting
            all_errors = []
            
            for endpoint_url in endpoints_to_try:
                # Honor global stop flags between requests
                self._raise_if_stop_requested()
                try:
                    # Check if it's the mobile /translate_a/t API, not merely a translate.* host.
                    is_mobile = endpoint_url.rstrip('/').endswith('/translate_a/t')
                    
                    if endpoint_url == GOOGLETRANS_AJAX_ENDPOINT:
                        result = self._translate_via_googletrans_ajax(text, source_lang, target_lang, endpoint_url)
                    elif is_mobile:
                        # Use mobile client API format
                        result = self._translate_via_mobile_api(text, source_lang, target_lang, endpoint_url)
                    else:
                        # Use the public translate_a/single API format
                        result = self._translate_via_single_api(text, source_lang, target_lang, endpoint_url)
                        
                    if result:
                        # Cache successful result
                        self._cache_translation_result(cache_key, result)
                        
                        # Honor stop flag after a successful call (before returning)
                        self._raise_if_stop_requested()
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

            if provider == "google":
                raise Exception(detailed_error)
            
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
            return self._translation_error_result(provider, e)

    @staticmethod
    def _sanitize_argos_text_tag_fragments(source_text: str, translated_text: str) -> str:
        translated = str(translated_text or "")
        if not translated:
            return translated
        source = str(source_text or "")
        if not re.search(r'(?i)(?:</?\s*(?:p|h[1-6])\b|/?\s*(?:p|h[1-6])\s*>)', translated):
            return translated

        # Argos can leak HTML-ish batch delimiters into plain text as "p>", "/p>",
        # "<p>", or "</p>". Strip only text-unit tag fragments, then normalize gaps.
        cleaned = re.sub(r'(?i)<\s*/?\s*(?:p|h[1-6])(?:\s+[^>]*)?\s*>', ' ', translated)
        cleaned = re.sub(r'(?i)(?<![\w/])/?\s*(?:p|h[1-6])\s*>', ' ', cleaned)
        cleaned = re.sub(r'(?i)<\s*/?\s*(?:p|h[1-6])\b[^<\s]*$', ' ', cleaned)
        cleaned = re.sub(r'[ \t]{2,}', ' ', cleaned)
        cleaned = re.sub(r' *\n *', '\n', cleaned)
        cleaned = cleaned.strip()
        if cleaned:
            return cleaned
        return translated if source.strip() else ""

    def _translate_marked_html_batch_via_argos(self, text: str, translation, stop_check=None) -> Optional[str]:
        if "data-sdl-tip" not in str(text or ""):
            return None
        parser = _MarkedHtmlSegmentParser()
        try:
            parser.feed(str(text or ""))
            parser.close()
        except Exception:
            return None
        if not parser.segments:
            return None

        rebuilt = []
        for segment in parser.segments:
            if stop_check and stop_check():
                raise Exception("Operation cancelled")
            source_text = html_lib.unescape("".join(segment.get("text") or []))
            translated = translation.translate(source_text) if source_text.strip() else ""
            translated = self._sanitize_argos_text_tag_fragments(source_text, translated)
            tag = segment.get("tag") or "p"
            rebuilt.append(
                f"{_html_start_tag(tag, segment.get('attrs') or [])}"
                f"{html_lib.escape(translated, quote=False)}"
                f"</{tag}>"
            )
        return "\n".join(rebuilt)
    
    def _translate_via_argos(self, text: str, source_lang: str, target_lang: str) -> Optional[Dict[str, Any]]:
        """Fallback to Argos Translate (offline-capable)."""
        try:
            import argostranslate.package
            import argostranslate.translate
        except ImportError:
            self.logger.warning("⚠️ Argos Translate not available on lite package. Skipping Argos fallback.")
            return None

        try:
            self._raise_if_stop_requested()
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
                self._raise_if_stop_requested()
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
                self._raise_if_stop_requested()
                translation = from_lang.get_translation(to_lang)
                try:
                    self.logger.info(f"🌐 Argos Translate: Translating {len(text)} characters")
                except Exception:
                    pass
                self._raise_if_stop_requested()
                translated_text = self._translate_marked_html_batch_via_argos(text, translation, self._stop_requested)
                if translated_text is None:
                    translated_text = translation.translate(text)
                    translated_text = self._sanitize_argos_text_tag_fragments(text, translated_text)
                return {
                    'translatedText': translated_text,
                    'detectedSourceLanguage': argos_source,
                    'provider': 'argos'
                }
            
            return None

        except Exception as e:
            self.logger.error(f"❌ Argos fallback failed: {e}")
            return None
    
    @staticmethod
    def _http_error_message(provider: str, response) -> str:
        try:
            body = response.text or ""
        except Exception:
            body = ""
        body = re.sub(r"\s+", " ", str(body or "")).strip()
        if len(body) > 240:
            body = body[:237] + "..."
        suffix = f": {body}" if body else ""
        return f"{provider} HTTP {getattr(response, 'status_code', 'unknown')}{suffix}"

    def _translate_via_deepl(self, text: str, source_lang: str, target_lang: str) -> Optional[Dict[str, Any]]:
        api_key = self._api_key("deepl", "DEEPL_API_KEY")
        provider_value = self.api_keys.get("deepl")
        endpoint = ""
        if isinstance(provider_value, dict):
            endpoint = str(provider_value.get("endpoint") or "").strip()
        endpoint = endpoint or self._api_option("deepl", "endpoint", "DEEPL_API_ENDPOINT")
        if not endpoint:
            endpoint = "https://api-free.deepl.com/v2/translate" if api_key.endswith(":fx") else "https://api.deepl.com/v2/translate"

        data = {
            "text": text,
            "target_lang": self._deepl_target_language_code(target_lang),
        }
        source_code = google_translate_language_code(source_lang, default=source_lang).upper()
        if source_code and source_code != "AUTO":
            data["source_lang"] = source_code.split("-", 1)[0]
        if self._is_marked_html_batch(text):
            data["tag_handling"] = "html"

        response = requests.post(
            endpoint,
            data=data,
            headers={"Authorization": f"DeepL-Auth-Key {api_key}"},
            timeout=20,
        )
        if response.status_code != 200:
            raise Exception(self._http_error_message("DeepL", response))
        payload = response.json()
        translations = payload.get("translations") if isinstance(payload, dict) else None
        if not translations:
            raise Exception("DeepL returned no translations")
        first = translations[0] or {}
        translated_text = str(first.get("text") or "").strip()
        if not translated_text:
            raise Exception("DeepL returned an empty translation")
        if not self._is_marked_html_batch(text):
            translated_text = self._sanitize_argos_text_tag_fragments(text, translated_text)
        return {
            "translatedText": translated_text,
            "detectedSourceLanguage": str(first.get("detected_source_language") or source_lang),
            "provider": "deepl",
            "endpoint": endpoint,
        }

    def _translate_via_bing(self, text: str, source_lang: str, target_lang: str) -> Optional[Dict[str, Any]]:
        api_key = self._api_key("bing", "BING_TRANSLATOR_API_KEY", "AZURE_TRANSLATOR_KEY")
        provider_value = self.api_keys.get("bing")
        endpoint = ""
        if isinstance(provider_value, dict):
            endpoint = str(provider_value.get("endpoint") or "").strip()
        endpoint = endpoint or self._api_option("bing", "endpoint", "AZURE_TRANSLATOR_ENDPOINT")
        endpoint = endpoint or "https://api.cognitive.microsofttranslator.com/translate"
        region = self._api_option("bing", "region", "AZURE_TRANSLATOR_REGION")

        params = {
            "api-version": "3.0",
            "to": google_translate_language_code(target_lang, default=target_lang),
        }
        source_code = google_translate_language_code(source_lang, default=source_lang)
        if source_code and source_code != "auto":
            params["from"] = source_code
        if self._is_marked_html_batch(text):
            params["textType"] = "html"

        headers = {
            "Ocp-Apim-Subscription-Key": api_key,
            "Content-Type": "application/json",
        }
        if region:
            headers["Ocp-Apim-Subscription-Region"] = region

        response = requests.post(
            endpoint,
            params=params,
            headers=headers,
            json=[{"Text": text}],
            timeout=20,
        )
        if response.status_code != 200:
            raise Exception(self._http_error_message("Microsoft Translator", response))
        payload = response.json()
        if not isinstance(payload, list) or not payload:
            raise Exception("Microsoft Translator returned no translations")
        translations = (payload[0] or {}).get("translations") or []
        if not translations:
            raise Exception("Microsoft Translator returned no translations")
        translated_text = str((translations[0] or {}).get("text") or "").strip()
        if not translated_text:
            raise Exception("Microsoft Translator returned an empty translation")
        if not self._is_marked_html_batch(text):
            translated_text = self._sanitize_argos_text_tag_fragments(text, translated_text)
        detected = (payload[0] or {}).get("detectedLanguage") or {}
        return {
            "translatedText": translated_text,
            "detectedSourceLanguage": str(detected.get("language") or source_lang),
            "provider": "bing",
            "endpoint": endpoint,
        }

    def _translate_via_yandex(self, text: str, source_lang: str, target_lang: str) -> Optional[Dict[str, Any]]:
        api_key = self._api_key("yandex", "YANDEX_TRANSLATE_API_KEY")
        folder_id = self._folder_id("yandex", "YANDEX_FOLDER_ID")
        provider_value = self.api_keys.get("yandex")
        endpoint = ""
        if isinstance(provider_value, dict):
            endpoint = str(provider_value.get("endpoint") or "").strip()
        endpoint = endpoint or self._api_option("yandex", "endpoint", "YANDEX_TRANSLATE_ENDPOINT")
        endpoint = endpoint or "https://translate.api.cloud.yandex.net/translate/v2/translate"

        body = {
            "targetLanguageCode": self._yandex_language_code(target_lang),
            "texts": [text],
            "folderId": folder_id,
            "format": "HTML" if self._is_marked_html_batch(text) else "PLAIN_TEXT",
        }
        source_code = self._yandex_language_code(source_lang)
        if source_code and source_code != "auto":
            body["sourceLanguageCode"] = source_code

        response = requests.post(
            endpoint,
            headers={
                "Authorization": f"Api-Key {api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=20,
        )
        if response.status_code != 200:
            raise Exception(self._http_error_message("Yandex Translate", response))
        payload = response.json()
        translations = payload.get("translations") if isinstance(payload, dict) else None
        if not translations:
            raise Exception("Yandex Translate returned no translations")
        first = translations[0] or {}
        translated_text = str(first.get("text") or "").strip()
        if not translated_text:
            raise Exception("Yandex Translate returned an empty translation")
        if not self._is_marked_html_batch(text):
            translated_text = self._sanitize_argos_text_tag_fragments(text, translated_text)
        return {
            "translatedText": translated_text,
            "detectedSourceLanguage": str(first.get("detectedLanguageCode") or source_lang),
            "provider": "yandex",
            "endpoint": endpoint,
        }

    def _translate_via_googletrans_ajax(self, text: str, source_lang: str, target_lang: str, endpoint_url: str) -> Optional[Dict[str, Any]]:
        """Translate using the tokenless Ajax route used by py-googletrans."""
        params = {
            'client': 'gtx',
            'sl': source_lang,
            'tl': target_lang,
            'dt': 't',
            'q': text
        }
        return self._try_single_api_request(params, endpoint_url, 'gtx', method='get')

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
    
    def _try_single_api_request(self, params: dict, endpoint_url: str, client_type: str, method: str = 'post') -> Optional[Dict[str, Any]]:
        """Try a single API request with given parameters."""
        if method == 'get':
            response = requests.get(
                endpoint_url,
                params=params,
                headers=self.get_headers(),
                timeout=10
            )
        else:
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
                        'provider': 'google',
                        'endpoint': endpoint_url
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
                    'provider': 'google',
                    'endpoint': endpoint_url
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
                    'provider': 'google',
                    'endpoint': endpoint_url
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
