import importlib, importlib.util, sys
from pathlib import Path

_stub_map = {
    'httpx': 'httpx_stub',
    'rapidfuzz': 'rapidfuzz_stub',
    'rapidfuzz.fuzz': 'rapidfuzz_stub',
    'rapidfuzz.process': 'rapidfuzz_stub',
    'langdetect': 'langdetect_stub',
    'langdetect.lang_detect_exception': 'langdetect_stub',
    'ebooklib': 'ebooklib_stub',
    'ebooklib.epub': 'ebooklib_stub',
    'tiktoken': 'tiktoken_stub',
}

_stub_dir = Path(__file__).resolve().parent


def _load_stub(stub_name):
    try:
        return importlib.import_module(stub_name)
    except Exception:
        pass
    for suffix in ('.py', '.pyc'):
        candidate = _stub_dir / f'{stub_name}{suffix}'
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location(stub_name, candidate)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[stub_name] = module
                spec.loader.exec_module(module)
                return module
    raise ImportError(stub_name)


for mod_name, stub_name in _stub_map.items():
    if mod_name not in sys.modules:
        try:
            importlib.import_module(mod_name)
        except ImportError:
            try:
                stub = _load_stub(stub_name)
                sys.modules[mod_name] = stub
            except Exception:
                pass
