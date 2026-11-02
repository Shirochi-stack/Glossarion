"""What a child interpreter of a QA scan runs after each batch (``job_kinds.qa._ScanChildren``)."""

from __future__ import annotations

__all__ = ["forget_caches"]


def forget_caches() -> None:
    """Empty every ``lru_cache`` of the scanner, the ones under another one too (``clear_qa_caches`` does not reach
    those: ``calculate_similarity_ratio`` is cached twice). In a thread pool their keys share the book's strings. In a
    child each batch's arguments are new strings, so that cache's limit of 20,000 pairs of texts would stay in memory per
    child, which a long book's all-pairs stage reaches. It costs no measurable time (host, Aggressive 70 chapters, 4
    workers, 6 rounds in rotated order: about 12.7 s with and without)."""
    import scan_html_folder

    for value in list(vars(scan_html_folder).values()):
        while callable(value):
            clear = getattr(value, "cache_clear", None)
            if clear is not None:
                clear()
            value = getattr(value, "__wrapped__", None)
