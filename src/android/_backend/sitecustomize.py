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
