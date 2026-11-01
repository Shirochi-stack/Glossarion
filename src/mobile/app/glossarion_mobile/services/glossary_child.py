"""What a child interpreter of a glossary job runs (``job_kinds.glossary.filter_in_child``).

``GlossaryManager`` imports ``extract_glossary_from_epub`` for two names the smart filter does not use, and that brings the
whole API client, which a child cannot load (pydantic_core, grpc, tiktoken: refused on the device). Both are blocked
here, so its ``try: from extract_glossary_from_epub import`` takes the ``except ImportError`` it already has. Importing
this module blocks them, so only a child imports it: the job side names it by its path.

``test_interpreters.py`` reads from the source that nothing the filter reaches uses what is blocked here.
"""

from __future__ import annotations

import sys

for _blocked in ("extract_glossary_from_epub", "unified_api_client"):
    sys.modules.setdefault(_blocked, None)

import GlossaryManager  # noqa: E402

from glossarion_mobile.services.name_comparison import comparison_modules  # noqa: E402

__all__ = ["filter_text"]


def filter_text(args: tuple, should_stop) -> tuple:
    """``_filter_text_for_glossary(text, min_frequency, max_sentences)``. The filter asks for a Stop on every sentence and
    term; ``is_stop_requested`` looks at a file and at the API client for that, so here it asks the pool's signal
    instead (at most ``CHILD_POLL`` old). A Stop ends it as on a thread: its decorator returns ``("", [])``.

    ``ImportError`` when this interpreter compares names another way than the caller's: the job then filters on a
    thread."""
    text, min_frequency, max_sentences, expected = args
    if comparison_modules() != expected:
        raise ImportError(f"a child compares names with {comparison_modules()}, the app with {expected}")
    GlossaryManager.is_stop_requested = should_stop
    return GlossaryManager._filter_text_for_glossary(text, min_frequency, max_sentences)
