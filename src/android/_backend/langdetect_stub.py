"""Stub - langdetect not available on Android."""
class LangDetectException(Exception): pass
def detect(text): return "unknown"
def detect_langs(text): return []
