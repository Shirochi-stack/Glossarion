#!/usr/bin/env python3
"""Generate Android stub modules for _backend/.

Called by the CI build workflow to create stub files for Python packages
that are not available on Android (tiktoken, ebooklib, httpx, etc.).
"""
import os
import sys


def main():
    backend_dir = os.path.join(os.path.dirname(__file__), '_backend')
    os.makedirs(backend_dir, exist_ok=True)

    stubs = {
        'image_translator.py': '''\
class ImageTranslator:
    """Stub — image translation not supported on Android."""
    pass
''',
        'ai_hunter_enhanced.py': '''\
class ImprovedAIHunterDetection:
    """Stub — AI hunter not available on Android."""
    pass
''',
        'pdf_extractor.py': '''\
def extract_text_from_pdf(*args, **kwargs): return ""
def extract_pdf_with_formatting(*args, **kwargs): return "", {}
def generate_css_from_pdf(*args, **kwargs): return ""
def create_pdf_from_html(*args, **kwargs): return False
def create_pdf_from_text(*args, **kwargs): return False
''',
        'httpx_stub.py': '''\
"""Stub — httpx not available on Android, use requests instead."""
class Client:
    def __init__(self, *a, **kw): pass
    def get(self, *a, **kw): raise NotImplementedError("httpx unavailable")
    def post(self, *a, **kw): raise NotImplementedError("httpx unavailable")
class AsyncClient(Client): pass
''',
        'rapidfuzz_stub.py': '''\
"""Stub — rapidfuzz not available on Android, falls back to difflib."""
import difflib
class fuzz:
    @staticmethod
    def ratio(s1, s2): return int(difflib.SequenceMatcher(None, s1, s2).ratio() * 100)
    @staticmethod
    def partial_ratio(s1, s2): return int(difflib.SequenceMatcher(None, s1, s2).ratio() * 100)
class process:
    @staticmethod
    def extractOne(query, choices, **kw):
        best = max(choices, key=lambda c: difflib.SequenceMatcher(None, query, c).ratio())
        score = int(difflib.SequenceMatcher(None, query, best).ratio() * 100)
        return (best, score, 0) if score > kw.get('score_cutoff', 0) else None
''',
        'langdetect_stub.py': '''\
"""Stub — langdetect not available on Android."""
def detect(text): return "unknown"
def detect_langs(text): return []
''',
        'ebooklib_stub.py': '''\
"""Stub — ebooklib not available on Android."""
ITEM_DOCUMENT = 9
ITEM_STYLE = 3
ITEM_IMAGE = 6
ITEM_NAVIGATION = 1
ITEM_COVER = 2
class EpubBook: pass
class EpubHtml: pass
class EpubItem: pass
def read_epub(*a, **kw): raise NotImplementedError("ebooklib unavailable on Android")
def write_epub(*a, **kw): raise NotImplementedError("ebooklib unavailable on Android")
# Sub-module alias so `from ebooklib import epub` works
import sys as _sys
class _EpubModule:
    ITEM_DOCUMENT = 9
    ITEM_STYLE = 3
    ITEM_IMAGE = 6
    ITEM_NAVIGATION = 1
    ITEM_COVER = 2
    EpubBook = EpubBook
    EpubHtml = EpubHtml
    EpubItem = EpubItem
    read_epub = staticmethod(read_epub)
    write_epub = staticmethod(write_epub)
epub = _EpubModule()
''',
        'tiktoken_stub.py': '''\
"""Stub — tiktoken not available on Android."""
class _DummyEncoding:
    def encode(self, text, *a, **kw): return list(range(len(text) // 4))
    def decode(self, tokens, *a, **kw): return ""
def encoding_for_model(model_name, **kw): return _DummyEncoding()
def get_encoding(name): return _DummyEncoding()
''',
        'sitecustomize.py': '''\
import importlib, sys
_stub_map = {
    'httpx': 'httpx_stub',
    'rapidfuzz': 'rapidfuzz_stub',
    'rapidfuzz.fuzz': 'rapidfuzz_stub',
    'rapidfuzz.process': 'rapidfuzz_stub',
    'langdetect': 'langdetect_stub',
    'ebooklib': 'ebooklib_stub',
    'ebooklib.epub': 'ebooklib_stub',
    'tiktoken': 'tiktoken_stub',
}
for mod_name, stub_name in _stub_map.items():
    if mod_name not in sys.modules:
        try:
            importlib.import_module(mod_name)
        except ImportError:
            try:
                stub = importlib.import_module(stub_name)
                sys.modules[mod_name] = stub
            except Exception:
                pass
''',
        '__init__.py': '',
    }

    for fname, content in stubs.items():
        path = os.path.join(backend_dir, fname)
        with open(path, 'w') as f:
            f.write(content)
        print(f"  ✅ {fname}")

    print(f"Generated {len(stubs)} stub files in {backend_dir}")


if __name__ == '__main__':
    main()
