"""Stub - httpx not available on Android, use requests instead."""
class Client:
    def __init__(self, *a, **kw): pass
    def get(self, *a, **kw): raise NotImplementedError("httpx unavailable")
    def post(self, *a, **kw): raise NotImplementedError("httpx unavailable")
class AsyncClient(Client): pass
