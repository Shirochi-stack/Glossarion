"""Stub - rapidfuzz not available on Android, falls back to difflib."""
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
