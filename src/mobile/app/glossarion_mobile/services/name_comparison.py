"""What decides how glossary names are compared, for an interpreter to check against another (``services.glossary_child``)."""

from __future__ import annotations

__all__ = ["comparison_modules"]


def comparison_modules() -> dict:
    """``duplicate_detection_config``'s optional modules: the same in every interpreter, or the same text gives other names.
    A child has RapidFuzz's pure Python where the app has its C++: the same scores (``test_interpreters.py``)."""
    import duplicate_detection_config as ddc

    return {"rapidfuzz": ddc._HAS_RAPIDFUZZ, "jellyfish": ddc._HAS_JELLYFISH, "jaro_winkler": ddc._rf_jaro_winkler is not None}
