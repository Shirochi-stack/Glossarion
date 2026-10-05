"""U3 name of the ModelSheet; the full sheet (UI_SPEC §2.2) lives in ``model_sheet`` (U4).

Kept as an alias so callers written against U3 (``chat_view.open_model_sheet`` /
``_chat_block``) keep working: ``ModelSheetMin`` is ``model_sheet.ModelSheet`` (same
constructor arguments, plus the U4 ones), and ``load_model_catalog`` / ``filter_models`` /
``excluded_route`` / ``EXCLUDED_ROUTE_PREFIXES`` are the same objects.
"""

from __future__ import annotations

from glossarion_mobile.ui.sheets.model_sheet import (  # noqa: F401 - re-exported
    EXCLUDED_ROUTE_PREFIXES,
    MAX_ROWS,
    ModelSheet,
    ModelSheetMin,
    excluded_route,
    filter_models,
    load_model_catalog,
)

__all__ = ["EXCLUDED_ROUTE_PREFIXES", "ModelSheetMin", "excluded_route", "filter_models", "load_model_catalog"]
