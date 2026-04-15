"""Stub - tiktoken not available on Android."""
class _DummyEncoding:
    def encode(self, text, *a, **kw): return list(range(len(text) // 4))
    def decode(self, tokens, *a, **kw): return ""
def encoding_for_model(model_name, **kw): return _DummyEncoding()
def get_encoding(name): return _DummyEncoding()
