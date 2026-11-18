"""The manga settings (schema section ``manga.settings``) grouped like the desktop dialogs (no Flet import).

``group_manga_keys`` buckets the keys by the desktop group they belong to (OCR, bubble detection,
inpainting, rendering, ...): Manga › Settings' "All manga settings" card lists those groups.
``manga_settings_layout`` divides Settings › Manga › Manga Translator into tabs: the desktop manga
settings dialog's own tabs in its order (Preprocessing · Inpainting · Replicate API · Manual Edit · OCR ·
Advanced; the schema's ``group`` of each setting), then the main manga panel's Rendering and General
settings, which have no dialog tab. A setting without a dialog tab goes to the tab of its bucket.
"""

import re
from typing import Any, Sequence

__all__ = ["MANGA_SECTION", "MANGA_TABS", "SCHEMA_GROUPS", "group_manga_keys", "manga_settings_layout",
           "manga_tab_of"]

MANGA_SECTION = "manga.settings"

#: Schema groups of the "All manga settings" card (desktop dialog tab order).
SCHEMA_GROUPS = (
    ("ocr", "OCR"),
    ("detection", "Bubble detection"),
    ("context", "Context"),
    ("inpainting", "Inpainting & mask"),
    ("rendering", "Rendering"),
    ("font_sizing", "Font sizing"),
    ("preprocessing", "Preprocessing"),
    ("compression", "Image compression"),
    ("tiling", "Tiling"),
    ("advanced", "Advanced"),
    ("manual_edit", "Manual edit"),
    ("other", "Other"),
)
_DETECTION_OCR_KEYS = re.compile(r"^manga_settings\.ocr\.(bubble_|detector_type|detect_|rtdetr_|"
                                 r"use_rtdetr_for_ocr_regions|skip_rtdetr_merging|preserve_empty_blocks)")
_PREFIX_GROUPS = (
    (r"^manga_settings\.ocr\.", "ocr"),
    (r"^manga_settings\.inpainting\.", "inpainting"),
    (r"^manga_settings\.(mask_dilation|dilation_|use_all_iterations|all_iterations|"
     r"\w*dilation_iterations|auto_iterations|cloud_)", "inpainting"),
    (r"^manga_settings\.rendering\.", "rendering"),
    (r"^manga_settings\.font_sizing\.", "font_sizing"),
    (r"^manga_settings\.preprocessing\.", "preprocessing"),
    (r"^manga_settings\.compression\.", "compression"),
    (r"^manga_settings\.tiling\.", "tiling"),
    (r"^manga_settings\.advanced\.", "advanced"),
    (r"^manga_settings\.manual_edit\.", "manual_edit"),
    (r"^(rapidocr_|qwen2vl_|manga_ocr_)", "ocr"),
    (r"^manga_(bg_|font|text_|shadow_|safe_area|constrain|force_caps|strict_text|free_text|max_font|"
     r"min_readable)", "rendering"),
    (r"^manga_(inpaint|skip_inpaint|local_inpaint|disable_inpaint|custom|[\w-]*_model_path)", "inpainting"),
    (r"^manga_(full_page|visual_context|glossary)", "context"),
    (r"^experimental_translate_all$", "advanced"),
)

#: Settings › Manga › Manga Translator tabs: (tab label, desktop dialog tab = schema ``group``), in order.
#: The labels fit a phone tab bar (the bar scrolls when they do not all fit).
MANGA_TABS = (
    ("Preprocessing", "Preprocessing"),
    ("Inpainting", "Inpainting"),
    ("Replicate", "Replicate API"),
    ("Manual edit", "Manual Edit"),
    ("OCR", "OCR"),
    ("Advanced", "Advanced"),
    ("Rendering", ""),  # the main manga panel (rendering, font sizing): no dialog tab
    ("General", ""),  # the main manga panel (context, output folder) and anything else
)
#: The tab of a setting without a dialog tab: its ``group_manga_keys`` bucket -> tab label.
_BUCKET_TABS = {
    "ocr": "OCR", "detection": "OCR", "inpainting": "Inpainting", "preprocessing": "Preprocessing",
    "compression": "Preprocessing", "tiling": "Preprocessing", "advanced": "Advanced", "manual_edit": "Manual edit",
    "rendering": "Rendering", "font_sizing": "Rendering", "context": "General", "other": "General",
}
_FALLBACK_TAB = "General"


def _bucket(key: str) -> str:
    if _DETECTION_OCR_KEYS.search(key):
        return "detection"
    for pattern, gid in _PREFIX_GROUPS:
        if re.search(pattern, key):
            return gid
    return "other"


def group_manga_keys(keys: Sequence[str], *, exclude: frozenset = frozenset()) -> list:
    """``[(group id, title, keys)]`` for the schema keys of ``manga.settings`` (desktop tab order)."""
    buckets: dict = {gid: [] for gid, _title in SCHEMA_GROUPS}
    for key in keys:
        if key not in exclude:
            buckets[_bucket(key)].append(key)
    return [(gid, title, tuple(buckets[gid])) for gid, title in SCHEMA_GROUPS if buckets[gid]]


def manga_tab_of(spec: Any) -> str:
    """The tab label of one manga setting: its desktop dialog tab, else the tab of its bucket."""
    group = str(getattr(spec, "group", "") or "")
    for label, dialog_tab in MANGA_TABS:
        if dialog_tab and dialog_tab == group:
            return label
    return _BUCKET_TABS.get(_bucket(str(getattr(spec, "key", "") or "")), _FALLBACK_TAB)


def manga_settings_layout(specs: Sequence[Any]) -> tuple:
    """``(specs, headings, tabs)`` for a SectionPage: the specs ordered tab by tab (``MANGA_TABS``), inside a
    tab by bucket (``SCHEMA_GROUPS`` order, schema order within), headings = the bucket titles (none in a tab
    with one bucket), tabs = ``[(label, start, end)]`` without empty tabs. Every spec is in exactly one tab."""
    titles = dict(SCHEMA_GROUPS)
    order = {gid: index for index, (gid, _title) in enumerate(SCHEMA_GROUPS)}
    per_tab: dict = {label: [] for label, _dialog in MANGA_TABS}
    for index, spec in enumerate(specs):
        key = str(getattr(spec, "key", "") or "")
        per_tab.setdefault(manga_tab_of(spec), []).append((order.get(_bucket(key), len(order)), index, spec))
    ordered: list = []
    headings: dict = {}
    tabs: list = []
    for label, rows in per_tab.items():
        if not rows:
            continue
        rows.sort(key=lambda row: (row[0], row[1]))
        buckets = {_bucket(str(getattr(spec, "key", "") or "")) for _o, _i, spec in rows}
        start = len(ordered)
        for _o, _i, spec in rows:
            key = str(getattr(spec, "key", "") or "")
            ordered.append(spec)
            headings[key] = titles.get(_bucket(key), "") if len(buckets) > 1 else ""
        tabs.append((label, start, len(ordered)))
    return ordered, headings, tabs
