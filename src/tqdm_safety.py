# tqdm_safety.py
"""
A defensive patch for tqdm to prevent AttributeError at interpreter shutdown:
AttributeError: type object 'tqdm' has no attribute '_lock'

Root cause
- During interpreter shutdown, module globals/class attributes may be cleared before tqdm.__del__ runs.
- tqdm.close() calls a class method that uses cls._lock; if it's already deleted, AttributeError is raised.

Fix
- Ensure a class-level _lock exists and is a threading.RLock().
- Wrap __del__ and close() to guard against shutdown-time attribute loss.
- No-ops if core attributes are missing, preserving normal behavior during runtime.

This keeps tqdm enabled and visible; it only avoids the noisy traceback on exit.
"""
from __future__ import annotations

import threading


def apply_tqdm_safety_patch() -> None:
    try:
        import tqdm as _tqdm_mod
        # Prefer the tqdm.tqdm class
        tqdm_cls = getattr(_tqdm_mod, 'tqdm', None)
        if tqdm_cls is None:
            # Some variants might expose TqdmExperimentalWarning only; bail quietly
            return

        # Ensure a class-level lock exists
        if not hasattr(tqdm_cls, '_lock') or getattr(tqdm_cls, '_lock') is None:
            try:
                tqdm_cls._lock = threading.RLock()
            except Exception:
                # As last resort, set a dummy object with context manager protocol
                class _DummyLock:
                    def __enter__(self):
                        return self
                    def __exit__(self, exc_type, exc, tb):
                        return False
                tqdm_cls._lock = _DummyLock()

        # Patch the class method used during close to guard missing attributes
        _orig_decr = getattr(tqdm_cls, '_decr_instances', None)
        if callable(_orig_decr):
            def _safe_decr_instances(*args, **kwargs):
                try:
                    # cls._lock might be gone at shutdown
                    if not hasattr(tqdm_cls, '_lock') or tqdm_cls._lock is None:
                        return
                    return _orig_decr(*args, **kwargs)
                except Exception:
                    # Swallow shutdown-time errors only
                    return
            try:
                _safe_decr_instances.__name__ = _orig_decr.__name__
            except Exception:
                pass
            setattr(tqdm_cls, '_decr_instances', staticmethod(_safe_decr_instances))

        # Wrap instance .close() to be defensive
        _orig_close = getattr(tqdm_cls, 'close', None)
        if callable(_orig_close):
            def _safe_close(self, *args, **kwargs):
                try:
                    return _orig_close(self, *args, **kwargs)
                except AttributeError:
                    # Happens if class attrs are missing at shutdown
                    return
                except Exception:
                    # Avoid raising during shutdown
                    try:
                        # Best effort: clear display without relying on internals
                        fp = getattr(self, 'fp', None)
                        if fp and hasattr(fp, 'flush'):
                            fp.flush()
                    except Exception:
                        pass
                    return
            setattr(tqdm_cls, 'close', _safe_close)

        # Wrap destructor to ignore shutdown-time errors
        _orig_del = getattr(tqdm_cls, '__del__', None)
        if callable(_orig_del):
            def _safe_del(self):
                try:
                    _orig_del(self)
                except Exception:
                    # Ignore any errors during interpreter shutdown
                    return
            setattr(tqdm_cls, '__del__', _safe_del)

    except Exception:
        # Never let the safety patch break startup
        return
