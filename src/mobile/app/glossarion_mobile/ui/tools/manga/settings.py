"""Manga › Settings (UI_SPEC §4.6 Settings; schema section ``manga.settings``).

Curated cards for the flows the desktop panel builds by hand, then every remaining key of the
schema section grouped like the desktop dialog tabs (``manga_settings.ocr`` / ``inpainting`` /
``rendering`` / ``font_sizing`` / ``preprocessing`` / ``advanced`` / ``compression`` /
``tiling`` / ``manual_edit`` + the top-level ``manga_*`` / ``rapidocr_*`` / ``qwen2vl_*``
keys), rendered with the shared Settings tiles (locks, defaults, reset and help as everywhere):

* **OCR provider** — one row per desktop provider with its status chip (Ready · Needs key ·
  Model not downloaded · Downloading NN% · Not in this build); credentials for the selected one
  (Google Vision service-account JSON — SDK or REST —, Azure Computer Vision key / endpoint,
  Azure Document Intelligence key / endpoint, custom-api OCR prompt + **Disable all thinking** +
  batch OCR requests + Vision keys). Torch-only providers (manga-ocr, Qwen2-VL, EasyOCR,
  PaddleOCR, DocTR) stay listed, disabled with "Needs PyTorch".
* **Detection** — AI bubble detection, the detector (RT-DETR ONNX; PyTorch / YOLO / custom
  disabled), its ONNX export (the desktop dialog's variant combo, from the model registry) and
  the RT-DETR ONNX ``ModelDownloadRow`` (status · Download · Load · Delete).
* **Context** — the current translation settings (model, profile, language, temperature,
  history) in place of the desktop "Refresh from Main GUI", full-page context + prompt, visual
  context, image request quality, the manga output token limit.
* **Glossary workflow** — use / generate / load / clear / auto-load / compress / debug subfolder.
* **Inpainting** — the method with its status chip (Preloaded · Loading · Not downloaded ·
  Needs key): Skip, Local / API model (ONNX aot / anime / lama with the model rows, the custom
  image-edit endpoint with Edit Prompt, Batch Image Requests and Test), Replicate (key), Hybrid
  and the Torch JIT / ollama / sd_local models disabled with their reasons.
* **Rendering** — a live sample bubble over font, size mode, background, text colour, shadow,
  safe area, constrain to bubble, caps, wrapping and line spacing; presets Manga / Manhwa /
  Large Text and Reset.

Every write uses the desktop config keys (``_save_rendering_settings``), so a desktop config
round-trips; runs read them from the job's config snapshot (changes apply to the next run).
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from typing import Any, Optional, Sequence

import flet as ft

from glossarion_mobile.services import manga as svc
from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.reason_chip import ReasonChip
from glossarion_mobile.ui.tools.common import hint_text, schema_tiles
from glossarion_mobile.ui.tools.manga.common import MangaTab, push, reason_or_chip
from glossarion_mobile.ui.theme import HIT_TARGET
from glossarion_mobile.ui.tools.manga.models import ModelDownloadRow, ModelManagerSheet

__all__ = ["CURATED_KEYS", "SCHEMA_GROUPS", "SettingsTab", "group_manga_keys", "preview_style"]

#: The desktop Reset to Defaults question (MangaTranslationTab._reset_rendering_to_defaults).
RESET_CONFIRM_TEXT = ("Are you sure you want to reset all rendering settings to their default values?\n\n"
                      "This will reset:\n"
                      "• Background opacity and style\n"
                      "• Font size and style settings\n"
                      "• Text color and shadow settings\n"
                      "• Auto-fit style and text wrapping\n"
                      "• All other rendering options")

log = logging.getLogger("glossarion.tools.manga")

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

#: Keys the curated cards edit (left out of the generic groups).
CURATED_KEYS = frozenset({
    "manga_settings.ocr.bubble_detection_enabled", "manga_settings.ocr.detector_type",
    "manga_settings.ocr.manga_ocr_disable_thinking", "manga_settings.ocr.rtdetr_onnx_variant",
    "manga_ocr_prompt", "manga_full_page_context", "manga_full_page_context_prompt", "manga_visual_context_enabled",
    "manga_skip_inpainting", "manga_inpaint_quality", "manga_local_inpaint_model",
    "manga_settings.inpainting.method", "manga_settings.inpainting.local_method",
    "manga_bg_opacity", "manga_bg_reduction", "manga_bg_style", "manga_free_text_only_bg_opacity",
    "manga_font_size_mode", "manga_font_size", "manga_font_size_multiplier", "manga_max_font_size",
    "manga_font_style", "manga_font_path", "manga_text_color", "manga_shadow_enabled", "manga_shadow_color",
    "manga_shadow_offset_x", "manga_shadow_offset_y", "manga_shadow_blur", "manga_constrain_to_bubble",
    "manga_force_caps_lock", "manga_strict_text_wrapping", "manga_settings.font_sizing.line_spacing",
    "manga_settings.font_sizing.algorithm",
    # Auto size mode: Minimum / Maximum Font Size (the desktop spin boxes; manga_max_font_size wins over
    # rendering.auto_max_size in manga_env, so both are written together)
    "manga_settings.rendering.auto_min_size", "manga_settings.rendering.auto_max_size",
})

#: The desktop Minimum / Maximum Font Size spin boxes' range (manga_integration ``min_size_spinbox``).
FONT_BOUNDS = (1, 999)

BG_STYLES = (("box", "Box"), ("circle", "Circle"), ("wrap", "Wrap"))
SIZE_MODES = (("auto", "Auto"), ("fixed", "Fixed Size"), ("multiplier", "Dynamic Multiplier"))
ALGORITHMS = (("smart", "Smart"), ("conservative", "Conservative"), ("aggressive", "Aggressive"))
PRESETS = (("small", "Manga"), ("balanced", "Manhwa"), ("large", "Large Text"))
COLOR_SWATCHES = ((0, 0, 0), (255, 255, 255), (102, 0, 0), (204, 128, 128), (30, 30, 120), (0, 90, 0))
SAMPLE_TEXT = "Wait… you're the one\nwho saved me?"
#: Config keys the Files tab persists on every list change (not settings: no re-render).
FILES_KEYS = frozenset({svc.K_SELECTED, svc.K_SKIPPED, svc.K_FOLDER_ROOTS, svc.K_SPLIT_SUBFOLDERS})


def group_manga_keys(keys: Sequence[str], *, exclude: frozenset = frozenset()) -> list:
    """``[(group id, title, keys)]`` for the schema keys of ``manga.settings`` (desktop tab order)."""
    buckets: dict = {gid: [] for gid, _title in SCHEMA_GROUPS}
    for key in keys:
        if key in exclude:
            continue
        if _DETECTION_OCR_KEYS.search(key):
            buckets["detection"].append(key)
            continue
        group = "other"
        for pattern, gid in _PREFIX_GROUPS:
            if re.search(pattern, key):
                group = gid
                break
        buckets[group].append(key)
    return [(gid, title, tuple(buckets[gid])) for gid, title in SCHEMA_GROUPS if buckets[gid]]


def _rgb(value: Any, default: tuple) -> tuple:
    try:
        r, g, b = (int(v) for v in list(value)[:3])
        return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))
    except Exception:
        return default


def _hex(rgb: tuple, alpha: Optional[int] = None) -> str:
    if alpha is None:
        return "#%02X%02X%02X" % rgb
    return "#%02X%02X%02X%02X" % (max(0, min(255, int(alpha))), *rgb)


def preview_style(config: Any) -> dict:
    """What the sample bubble shows for the current rendering settings (display only)."""
    get = (lambda key, default=None: svc.effective_setting(config, key, default))
    text_rgb = _rgb(get("manga_text_color", [102, 0, 0]), (102, 0, 0))
    shadow_rgb = _rgb(get("manga_shadow_color", [204, 128, 128]), (204, 128, 128))
    try:
        opacity = int(get("manga_bg_opacity", 130) or 0)
    except (TypeError, ValueError):
        opacity = 130
    mode = str(get("manga_font_size_mode", "fixed") or "fixed")
    try:
        size = int(get("manga_font_size", 0) or 0)
    except (TypeError, ValueError):
        size = 0
    try:
        multiplier = float(get("manga_font_size_multiplier", 1.0) or 1.0)
    except (TypeError, ValueError):
        multiplier = 1.0
    base = size if (mode == "fixed" and size > 0) else 16
    if mode == "multiplier":
        base = 16 * multiplier
    try:
        spacing = float(svc.effective_setting(config, ("manga_settings", "font_sizing", "line_spacing"), 1.3) or 1.3)
    except (TypeError, ValueError):
        spacing = 1.3
    return {
        "text": SAMPLE_TEXT.upper() if bool(get("manga_force_caps_lock", True)) else SAMPLE_TEXT,
        "color": _hex(text_rgb),
        "size": max(8.0, min(40.0, float(base))),
        "line_spacing": max(0.8, min(3.0, spacing)),
        "bg": _hex((255, 255, 255), max(0, min(255, opacity))),
        "bg_style": str(get("manga_bg_style", "circle") or "circle"),
        "shadow": bool(get("manga_shadow_enabled", True)),
        "shadow_color": _hex(shadow_rgb),
        "shadow_offset": (int(get("manga_shadow_offset_x", 2) or 0), int(get("manga_shadow_offset_y", 2) or 0)),
        "shadow_blur": max(0, int(get("manga_shadow_blur", 0) or 0)),
        "font_path": str(get("manga_font_path", "") or ""),
    }


def _muted(disabled: bool) -> Any:
    """Title colour of an unavailable option row (the row itself stays enabled so its ReasonChip opens)."""
    return ft.Colors.ON_SURFACE_VARIANT if disabled else None


class _EditorCtx:
    """What the shared full-screen editors (PromptEditor / SecretEditor / PathEditor) need when the
    tools context carries no SettingsContext (host tests)."""

    def __init__(self, ctx: Any) -> None:
        self._ctx = ctx
        self.page = ctx.page
        self.extras = ctx.extras
        self.file_picker_factory = None

    def show_dialog(self, dialog: Any) -> None:
        if self.page is not None:
            self.page.show_dialog(dialog)

    def pop_dialog(self, dialog: Any = None) -> None:
        """Close ``dialog`` itself (``close_dialog``, as ``SettingsContext.pop_dialog`` does);
        without one, the topmost open dialog."""
        if self.page is None:
            return
        if dialog is not None:
            from glossarion_mobile.ui.components.dialogs import close_dialog

            close_dialog(self.page, dialog)
            return
        try:
            self.page.pop_dialog()
        except Exception:
            pass

    def push(self, *controls: Any) -> None:
        push(*controls)

    def say(self, message: str, *args: Any) -> None:
        self._ctx.say(message)

    def spawn(self, coro: Any) -> Any:
        return self._ctx.spawn(coro)


class SettingsTab(MangaTab):
    key = "manga-settings"

    def __init__(self, ctx: Any, session: Any, *, screen: Any = None) -> None:
        super().__init__(ctx, session, screen=screen)
        self.cards: dict = {}
        self.tiles: dict = {}
        self.model_rows: dict = {}
        self._previous_rows: dict = {}  # the rows of the build before (kept while they show the same model)
        self.group_built: set = set()
        self._unsubs: list = []
        self._gen = 0
        self.editor: Any = None

    def k(self, name: str) -> str:
        """A per-build key: Flet 1.0.3 freezes a control that replaces one with the same key, so the
        card bodies rebuilt by ``refresh`` carry the build generation."""
        return f"{name}-g{self._gen}"

    # ---- config ------------------------------------------------------------------------------------

    def config(self) -> dict:
        return self.ctx.config_snapshot()

    def get(self, key: Any, default: Any = None) -> Any:
        return svc.effective_setting(self.config(), key, default)

    def set(self, key: Any, value: Any) -> None:
        self.ctx.set_cfg(tuple(key) if isinstance(key, (list, tuple)) else key, value)

    def set_many(self, updates: dict) -> None:
        for key, value in updates.items():
            self.set(key, value)

    def _editor_ctx(self) -> Any:
        settings = getattr(self.ctx, "settings", None)
        return settings if settings is not None else _EditorCtx(self.ctx)

    # ---- build ------------------------------------------------------------------------------------

    def build(self) -> ft.Control:
        self.ocr_card = self.section("OCR provider", [], icon="DOCUMENT_SCANNER", key="ms-ocr")
        self.preview: Optional[ft.Control] = None
        self.detection_card = self.section("Detection", [], icon="CENTER_FOCUS_STRONG", key="ms-detection")
        self.context_card = self.section("Context", [], icon="CHAT", key="ms-context")
        self.glossary_card = self.section("Glossary workflow", [], icon="SPELLCHECK", key="ms-glossary")
        self.inpaint_card = self.section("Inpainting", [], icon="AUTO_FIX_HIGH", key="ms-inpaint")
        self.render_card = self.section("Rendering", [], icon="FORMAT_COLOR_TEXT", key="ms-render")
        self.models_button = ft.TextButton(content="Manage models", icon=ft.Icons.MODEL_TRAINING,
                                           on_click=lambda e: self.open_models(), key="ms-models")
        self.all_card = self.section("All manga settings", [], icon="TUNE", key="ms-all",
                                     subtitle="Every manga_settings value, grouped like the desktop dialog")
        self.root = ft.ListView(
            controls=[ft.Row([hint_text("Changes apply to the next run."), self.models_button], wrap=True,
                             alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                      self.ocr_card, self.detection_card, self.context_card, self.glossary_card,
                      self.inpaint_card, self.render_card, self.all_card],
            expand=True, spacing=tokens.SPACING["md"], padding=ft.Padding.symmetric(horizontal=12, vertical=8),
            key=self.key)
        self.refresh(push_now=False)
        self._build_all_groups()
        return self.root

    def did_show(self) -> None:
        store = getattr(self.ctx, "store", None)
        observe_all = getattr(store, "observe_all", None)
        if callable(observe_all) and not self._unsubs:
            try:
                self._unsubs.append(observe_all(lambda key, value: self._on_config_changed(key)))
            except Exception:
                log.debug("observing the config failed", exc_info=True)

    def dispose(self) -> None:
        for unsub in self._unsubs:
            try:
                unsub()
            except Exception:
                pass
        self._unsubs = []

    def _on_config_changed(self, key: str) -> None:
        """A tile changed a value: re-render the card that shows it (UI loop). The Files tab's
        selection keys (persisted on every list change) are not settings."""
        key = str(key or "")
        if key in FILES_KEYS:
            return

        def run() -> None:
            if key.startswith("manga_") or key in ("model", "active_profile", "output_language"):
                self.refresh()

        dispatcher = getattr(self.ctx, "dispatcher", None)
        if dispatcher is not None and getattr(dispatcher, "bound", False) and not dispatcher.on_loop_thread():
            dispatcher.post(run)
        else:
            run()

    def refresh(self, push_now: bool = True) -> None:
        if self.root is None:
            return
        self._gen += 1
        # rebuilt with the cards (``_download_row``); a row whose download runs is kept for later builds
        # even while its card does not show it (another inpainting method picked meanwhile)
        downloading = {slot: row for slot, row in self._previous_rows.items() if row.downloading}
        self._previous_rows, self.model_rows = {**downloading, **self.model_rows}, {}
        cfg = self.config()
        self._card_body(self.ocr_card, self._ocr_controls(cfg))
        self._card_body(self.detection_card, self._detection_controls(cfg))
        self._card_body(self.context_card, self._context_controls(cfg))
        self._card_body(self.glossary_card, self._glossary_controls(cfg))
        self._card_body(self.inpaint_card, self._inpaint_controls(cfg))
        self._card_body(self.render_card, self._render_controls(cfg))
        if push_now:
            push(self.root)

    @staticmethod
    def _card_body(card: ft.Container, controls: list) -> None:
        column = card.content
        header = column.controls[0]
        column.controls = [header, *controls]

    def _download_row(self, slot: str, model_id: str, **kwargs: Any) -> ModelDownloadRow:
        """The download row of ``model_id`` in its card (``slot``). A rebuild keeps the row of the
        build before while it shows the same model (the same control, its status re-read unless its
        own download runs): a download keeps reporting into the row on screen, and the rebuild its
        status changes trigger does not orphan it."""
        previous = self._previous_rows.get(slot)
        if previous is not None and previous.model_id == model_id:
            previous.sync()
            row = previous
        else:
            row = ModelDownloadRow(self.ctx, self.session.models, model_id, **kwargs)
        self.model_rows[slot] = row
        return row

    # ---- OCR provider ------------------------------------------------------------------------------

    def _model_status(self, model_id: str) -> Any:
        manager = self.session.models
        if not manager.available:
            return None
        entry = manager.status(model_id)
        return None if entry.status == "unavailable" else entry

    def _ocr_controls(self, cfg: dict) -> list:
        current = str(cfg.get(svc.K_PROVIDER) or cfg.get("ocr_provider") or "custom-api")
        rows = svc.ocr_provider_rows(cfg, model_status=self._model_status)
        controls: list = []
        self.provider_tiles = {}
        for row in rows:
            selected = row.value == current
            tile = ft.ListTile(
                leading=ft.Icon(ft.Icons.RADIO_BUTTON_CHECKED if selected else ft.Icons.RADIO_BUTTON_UNCHECKED,
                                color=ft.Colors.PRIMARY if selected and not row.disabled else None),
                title=ft.Text(row.label, max_lines=2, color=_muted(row.disabled)),
                subtitle=ft.Text(row.detail, theme_style=ft.TextThemeStyle.BODY_SMALL) if row.detail else None,
                trailing=reason_or_chip(row, key=self.k(f"ms-ocr-chip-{row.value}"), dark=self.ctx.dark),
                # never disabled=True: Flet would disable the ReasonChip too (no InfoSheet, UI_SPEC §5.2)
                on_click=None if row.disabled else (lambda e, v=row.value: self.select_provider(v)),
                dense=True,
                key=self.k(f"ms-ocr-{row.value}-{self._gen}"),
            )
            self.provider_tiles[row.value] = tile
            controls.append(tile)
        controls.extend(self._provider_details(current, cfg))
        return controls

    def select_provider(self, value: str) -> bool:
        reason = svc.value_reason(svc.K_PROVIDER, value)
        if reason:
            self.ctx.say(reason)
            return False
        self.set(svc.K_PROVIDER, value)
        self.refresh()
        return True

    def _provider_details(self, provider: str, cfg: dict) -> list:
        out: list = [ft.Divider(height=1)]
        if provider == "google":
            path = str(cfg.get(svc.K_GOOGLE_CREDS) or cfg.get(svc.K_GOOGLE_CLOUD_CREDS) or "")
            out.append(ft.ListTile(
                title=ft.Text("Google Cloud credentials (service-account JSON)"),
                subtitle=ft.Text(os.path.basename(path) if path else "Not set", theme_style=ft.TextThemeStyle.BODY_SMALL),
                trailing=ft.Icon(ft.Icons.CHEVRON_RIGHT), on_click=lambda e: self.edit_google_credentials(),
                key=self.k("ms-google-creds"), dense=True))
            backend = svc.google_backend()
            out.append(hint_text("Uses google-cloud-vision" if backend == "SDK" else
                                 "Uses the Vision REST API (google-cloud-vision is not in this build)"))
        elif provider == "azure":
            out += [self._secret_row("Azure Computer Vision key", svc.K_AZURE_KEY, "ms-azure-key", cfg),
                    self._text_row("Azure endpoint", svc.K_AZURE_ENDPOINT, "ms-azure-endpoint", cfg,
                                   hint="https://<resource>.cognitiveservices.azure.com/")]
        elif provider == "azure-document-intelligence":
            out += [self._secret_row("Document Intelligence key", svc.K_DOCINTEL_KEY, "ms-docintel-key", cfg),
                    self._text_row("Document Intelligence endpoint", svc.K_DOCINTEL_ENDPOINT, "ms-docintel-endpoint",
                                   cfg, hint="https://<resource>.cognitiveservices.azure.com/"),
                    hint_text("Uses azure-ai-documentintelligence when it is in this build, otherwise its REST API.")]
        elif provider == "custom-api":
            disable_thinking = bool(svc.effective_setting(cfg, ("manga_settings", "ocr", "manga_ocr_disable_thinking"),
                                                          True))
            batch = bool(svc.effective_setting(cfg, "manga_custom_api_ocr_batch_enabled", True))
            try:
                batch_size = int(svc.effective_setting(cfg, "manga_custom_api_ocr_batch_size", 5) or 5)
            except (TypeError, ValueError):
                batch_size = 5
            out += [
                ft.Row([
                    ft.FilledTonalButton(content="Edit OCR prompt", icon=ft.Icons.EDIT_NOTE,
                                         on_click=lambda e: self.edit_prompt("manga_ocr_prompt", "OCR prompt",
                                                                             "ocr"), key=self.k("ms-ocr-prompt")),
                    ft.TextButton(content="Vision keys", icon=ft.Icons.KEY,
                                  on_click=lambda e: self.ctx.go("settings.keys.pool", {"pool": "qa_vision"}),
                                  key=self.k("ms-vision-keys")),
                ], wrap=True, spacing=8),
                ft.Switch(label="Disable all thinking", value=disable_thinking, key=self.k("ms-ocr-no-thinking"),
                          tooltip="Applies only to custom-api manga OCR requests. Removes thinking params for "
                                  "faster OCR.",
                          on_change=lambda e: self.set(("manga_settings", "ocr", "manga_ocr_disable_thinking"),
                                                       bool(e.control.value))),
                ft.Row([
                    ft.Switch(label="Batch OCR requests", value=batch, key=self.k("ms-ocr-batch"),
                              on_change=lambda e: self.set("manga_custom_api_ocr_batch_enabled",
                                                           bool(e.control.value))),
                    ft.Dropdown(options=[ft.DropdownOption(key=str(n), text=str(n)) for n in range(1, 33)],
                                value=str(max(1, min(32, batch_size))), width=96, dense=True, key=self.k("ms-ocr-batch-size"),
                                tooltip="Maximum concurrent custom-api OCR requests",
                                on_select=lambda e: self.set("manga_custom_api_ocr_batch_size",
                                                             int(e.control.value or 5))),
                ], wrap=True, spacing=8),
            ]
        elif provider == "rapidocr":
            controls, tiles = schema_tiles(self.ctx, ("rapidocr_use_recognition", "rapidocr_detection_mode",
                                                      "rapidocr_language"))
            self.tiles.update(tiles)
            out += controls
            if "rapidocr" in {e.id for e in self.session.models.entries()}:
                row = self._download_row("rapidocr", "rapidocr", key_prefix=self.k("ms-rapidocr-model"),
                                         allow_load=False, on_change=lambda entry: self.refresh())
                out.append(row.control)
        return out

    def _secret_row(self, title: str, key: str, ui_key: str, cfg: dict) -> ft.Control:
        value = str(cfg.get(key) or "")
        shown = ("•" * 8 + value[-4:]) if len(value) > 6 else ("Set" if value else "Not set")
        return ft.ListTile(title=ft.Text(title), subtitle=ft.Text(shown, theme_style=ft.TextThemeStyle.BODY_SMALL),
                           trailing=ft.Icon(ft.Icons.CHEVRON_RIGHT), dense=True, key=self.k(ui_key),
                           on_click=lambda e: self.edit_secret(title, key))

    def _text_row(self, title: str, key: str, ui_key: str, cfg: dict, *, hint: str = "") -> ft.Control:
        return ft.TextField(label=title, value=str(cfg.get(key) or ""), hint_text=hint or None, dense=True,
                            key=self.k(ui_key), on_blur=lambda e: self.set(key, str(e.control.value or "").strip()),
                            on_submit=lambda e: self.set(key, str(e.control.value or "").strip()))

    def edit_secret(self, title: str, key: str) -> Any:
        from glossarion_mobile.ui.settings.editors import SecretEditor

        def save(value: Any) -> Optional[str]:
            self.set(key, str(value or "").strip())
            self.refresh()
            return None

        current = str(self.config().get(key) or "")
        self.editor = SecretEditor(self._editor_ctx(), title=title, value="" if current.startswith("ENC:") else current,
                                   on_save=save, subtitle=key)
        return self.editor.show()

    def edit_google_credentials(self) -> Any:
        from glossarion_mobile.ui.settings.editors import PathEditor

        def save(value: Any) -> Optional[str]:
            path = str(value or "").strip()
            if path:
                problem = svc._google_credentials_problem(path)
                if problem:
                    return problem
                path = self._keep_private_copy(path, "credentials")
            self.set(svc.K_GOOGLE_CREDS, path)
            self.refresh()
            return None

        current = str(self.config().get(svc.K_GOOGLE_CREDS) or "")
        self.editor = PathEditor(self._editor_ctx(), title="Google Cloud credentials", value=current, on_save=save,
                                 import_dir=os.path.join(self.session.root, "credentials"),
                                 allowed_extensions=["json"], subtitle=svc.K_GOOGLE_CREDS)
        return self.editor.show()

    def _keep_private_copy(self, path: str, folder: str, move: bool = False) -> str:
        """Picked files live in the app's private manga folder (picker cache copies vanish);
        ``move`` moves the picker's own Inbox copy there instead of copying it."""
        target_dir = os.path.join(self.session.root, folder)
        try:
            if os.path.commonpath([os.path.abspath(path), os.path.abspath(target_dir)]) == os.path.abspath(target_dir):
                return path
        except ValueError:
            pass
        target = os.path.join(target_dir, os.path.basename(path))
        try:
            os.makedirs(target_dir, exist_ok=True)
            if move:
                shutil.move(path, target)
            else:
                shutil.copy2(path, target)
            return target
        except OSError:
            return path if os.path.isfile(path) or not os.path.isfile(target) else target

    @staticmethod
    def _discard(path: str) -> None:
        """Remove a picked file that is not used (its Inbox copy)."""
        try:
            os.remove(path)
        except OSError:
            pass

    # ---- prompts -----------------------------------------------------------------------------------

    def _default_prompt(self, kind: str) -> Optional[str]:
        names = {
            "ocr": ("default_manga_ocr_prompt", "DEFAULT_MANGA_OCR_PROMPT"),
            "context": ("default_full_page_context_prompt", "DEFAULT_FULL_PAGE_CONTEXT_PROMPT"),
            "glossary": ("default_manga_glossary_prompt", "DEFAULT_MANGA_GLOSSARY_PROMPT"),
            "image_edit": ("default_custom_image_edit_system_prompt", "DEFAULT_CUSTOM_IMAGE_EDIT_SYSTEM_PROMPT"),
        }.get(kind, ())
        for module_name in ("manga_settings_defaults", "manga_env"):
            value = svc.core_attr(module_name, *names)
            if callable(value):
                try:
                    value = value()
                except Exception:
                    value = None
            if isinstance(value, str) and value:
                return value
        return None

    def edit_prompt(self, key: str, title: str, kind: str) -> Any:
        from glossarion_mobile.ui.settings.editors import PromptEditor

        default = self._default_prompt(kind)
        current = str(self.config().get(key) or "") or (default or "")

        def save(value: Any) -> Optional[str]:
            self.set(key, str(value or ""))
            return None

        self.editor = PromptEditor(self._editor_ctx(), title=title, value=current, default=default, on_save=save,
                                   subtitle=key)
        return self.editor.show()

    def edit_image_edit_prompts(self) -> Any:
        """Custom image edit system prompt (+ optional user prompt) — ``_sync_custom_image_edit_prompt`` keys."""
        from glossarion_mobile.ui.settings.editors import PromptEditor

        cfg = self.config()
        default = self._default_prompt("image_edit")
        current = (str(cfg.get("custom_image_edit_system_prompt") or "") or str(cfg.get("custom_image_edit_prompt") or "")
                   or (default or ""))

        def save(value: Any) -> Optional[str]:
            text = str(value or "").strip() or (default or "")
            self.set("custom_image_edit_system_prompt", text)
            return None

        self.editor = PromptEditor(self._editor_ctx(), title="Custom image edit prompt", value=current,
                                   default=default, on_save=save, subtitle="custom_image_edit_system_prompt")
        return self.editor.show()

    # ---- detection ---------------------------------------------------------------------------------

    def _detection_controls(self, cfg: dict) -> list:
        enabled = bool(svc.effective_setting(cfg, svc.P_BUBBLE_DETECTION, True))
        current = str(svc.effective_setting(cfg, svc.P_DETECTOR, "rtdetr_onnx") or "rtdetr_onnx")
        controls: list = [ft.Switch(label="AI bubble detection", value=enabled, key=self.k("ms-bubble-detection"),
                                    on_change=lambda e: self.set(svc.P_BUBBLE_DETECTION, bool(e.control.value)))]
        for row in svc.detector_rows():
            selected = row.value == current
            controls.append(ft.ListTile(
                leading=ft.Icon(ft.Icons.RADIO_BUTTON_CHECKED if selected else ft.Icons.RADIO_BUTTON_UNCHECKED),
                title=ft.Text(row.label, color=_muted(row.disabled)), dense=True,
                trailing=ReasonChip(reason=row.chip or row.reason, detail=row.reason) if row.disabled else None,
                on_click=None if row.disabled else (lambda e, v=row.value: self._select_detector(v)),
                key=self.k(f"ms-detector-{row.value}-{self._gen}")))
        manager = self.session.models
        variants = manager.detector_variants() if manager.available and current == "rtdetr_onnx" else []
        if variants:
            # the desktop dialog's "ONNX Export" combo (the phone default is the small INT8 export)
            variant = str(svc.effective_setting(cfg, svc.P_RTDETR_VARIANT, "") or "")
            controls.append(ft.Text("ONNX export", theme_style=ft.TextThemeStyle.TITLE_SMALL))
            for selector, title, description, size in variants:
                selected = selector == variant
                controls.append(ft.ListTile(
                    leading=ft.Icon(ft.Icons.RADIO_BUTTON_CHECKED if selected else ft.Icons.RADIO_BUTTON_UNCHECKED),
                    title=ft.Text(f"{title} · {size}" if size else title),
                    subtitle=ft.Text(description or selector, theme_style=ft.TextThemeStyle.BODY_SMALL),
                    dense=True, on_click=lambda e, v=selector: self._select_variant(v),
                    key=self.k(f"ms-variant-{selector}")))
        model_id = manager.detector_id(cfg) if manager.available else None
        if model_id:
            row = self._download_row("detector", model_id, key_prefix=self.k("ms-detector-model"),
                                     on_change=lambda entry: None)
            controls.append(row.control)
        else:
            controls.append(ReasonChip(reason="Model downloads unavailable",
                                       detail=svc.MISSING_CORE + " (manga_models)."))
        return controls

    def _select_variant(self, selector: str) -> None:
        """The RT-DETR ONNX export the runs load (its download row follows)."""
        self.set(svc.P_RTDETR_VARIANT, selector)
        self.refresh()

    def _select_detector(self, value: str) -> None:
        if svc.value_reason(".".join(svc.P_DETECTOR), value):
            return
        self.set(svc.P_DETECTOR, value)
        self.refresh()

    # ---- context -----------------------------------------------------------------------------------

    def current_translation_summary(self, cfg: dict) -> list:
        """The run settings manga translation uses (desktop "Refresh from Main GUI" reads them live)."""
        def val(key: str, default: Any = "") -> str:
            value = cfg.get(key, default)
            return "" if value is None else str(value)

        rows = [("Model", val("model", "authgpt/gpt-6-luna")), ("Profile", val("active_profile", "")),
                ("Target language", val("output_language", "English")),
                ("Temperature", val("translation_temperature", "")),
                ("Contextual", "On" if cfg.get("contextual", True) else "Off"),
                ("History", val("translation_history_limit", "")),
                ("Batch translation", "On" if cfg.get("batch_translation") else "Off")]
        return [(k, v) for k, v in rows if v != ""]

    def _context_controls(self, cfg: dict) -> list:
        from glossarion_mobile.ui.tools.common import kv_line

        summary = [kv_line(k, v, key=self.k(f"ms-ctx-{k}")) for k, v in self.current_translation_summary(cfg)]
        full_page = bool(svc.effective_setting(cfg, "manga_full_page_context", False))
        visual = bool(svc.effective_setting(cfg, "manga_visual_context_enabled", True))
        controls = [
            hint_text("Current translation settings (from Settings › Model & Run):"),
            *summary,
            ft.TextButton(content="Open translation settings", icon=ft.Icons.SETTINGS,
                          on_click=lambda e: self.ctx.go("settings"), key=self.k("ms-ctx-open-settings")),
            ft.Row([
                ft.Switch(label="Full page context", value=full_page, key=self.k("ms-full-page"),
                          on_change=lambda e: self.set("manga_full_page_context", bool(e.control.value))),
                ft.TextButton(content="Edit prompt", icon=ft.Icons.EDIT_NOTE, key=self.k("ms-full-page-prompt"),
                              on_click=lambda e: self.edit_prompt("manga_full_page_context_prompt",
                                                                  "Full page context prompt", "context")),
            ], wrap=True, spacing=8),
            ft.Switch(label="Include page image in translation requests (visual context)", value=visual,
                      key=self.k("ms-visual"), on_change=lambda e: self.set("manga_visual_context_enabled",
                                                                    bool(e.control.value))),
        ]
        tiles, found = schema_tiles(self.ctx, ("manga_settings.compression.enabled", "manga_settings.compression.format",
                                               "manga_settings.compression.jpeg_quality",
                                               "manga_settings.manual_edit.manga_output_token_limit"))
        self.tiles.update(found)
        return controls + tiles

    # ---- glossary ----------------------------------------------------------------------------------

    def _glossary_controls(self, cfg: dict) -> list:
        enabled = bool(svc.effective_setting(cfg, "manga_glossary_enabled", False))
        compress = bool(cfg.get("compress_glossary_prompt", True))  # MangaEnvMixin's read of the shared key
        debug = bool(svc.effective_setting(cfg, "manga_glossary_debug_ocr_text", False))
        auto = not bool(svc.effective_setting(cfg, "manga_glossary_auto_load_suppressed", False))
        custom = str(cfg.get("manga_custom_glossary_path") or "")
        generated = str(cfg.get("manga_generated_glossary_path") or "")
        status = (f"Loaded: {os.path.basename(custom)}" if custom else
                  (f"Generated: {os.path.basename(generated)}" if generated else "No glossary loaded"))
        files_tab = getattr(self.screen, "files_tab", None) if self.screen is not None else None
        return [
            ft.Switch(label="Use loaded/generated glossary for translation", value=enabled, key=self.k("ms-glossary-on"),
                      on_change=lambda e: self.set("manga_glossary_enabled", bool(e.control.value))),
            ft.Row([
                ft.TextButton(content="Glossary prompt", icon=ft.Icons.EDIT_NOTE, key=self.k("ms-glossary-prompt"),
                              on_click=lambda e: self.edit_prompt("manga_glossary_prompt", "Manga glossary prompt",
                                                                  "glossary")),
                ft.FilledTonalButton(content="Generate glossary", icon=ft.Icons.AUTO_AWESOME, key=self.k("ms-glossary-generate"),
                                     on_click=lambda e: self.ctx.spawn(files_tab.start(glossary_only=True))
                                     if files_tab is not None else None),
            ], wrap=True, spacing=8),
            ft.Switch(label="Compress Glossary Prompt (same setting as Glossary Settings)", value=compress,
                      key=self.k("ms-glossary-compress"),
                      on_change=lambda e: self.set("compress_glossary_prompt", bool(e.control.value))),
            ft.Switch(label="Save OCR/glossary debug subfolder", value=debug, key=self.k("ms-glossary-debug"),
                      on_change=lambda e: self.set("manga_glossary_debug_ocr_text", bool(e.control.value))),
            ft.Switch(label="Auto-load the generated glossary for this folder", value=auto, key=self.k("ms-glossary-auto"),
                      on_change=lambda e: self._set_auto_load(bool(e.control.value))),
            ft.Row([
                ft.FilledTonalButton(content="Load glossary", icon=ft.Icons.FILE_OPEN, key=self.k("ms-glossary-load"),
                                     on_click=lambda e: self.ctx.spawn(self.load_glossary())),
                ft.TextButton(content="Clear", icon=ft.Icons.CLEAR, key=self.k("ms-glossary-clear"),
                              on_click=lambda e: self.clear_glossary(), disabled=not (custom or generated)),
            ], wrap=True, spacing=8),
            hint_text(status, key=self.k("ms-glossary-status")),
        ]

    def _set_auto_load(self, on: bool) -> None:
        self.set("manga_glossary_auto_load_suppressed", not on)
        if on:
            self.set("manga_glossary_auto_load_suppressed_root", "")

    async def load_glossary(self) -> Optional[str]:
        files = self.ctx.files
        if files is None:
            self.ctx.say("File picking is not available")
            return None
        picked = await files.pick_files(allowed_extensions=["csv", "json", "txt", "md"], allow_multiple=False,
                                        dialog_title="Load a glossary")
        if not picked:
            return None
        path = await self.ctx.io(self._keep_private_copy, picked[0].path, "glossaries")
        self.set("manga_custom_glossary_path", path)
        self.set("manga_glossary_auto_load_suppressed", False)
        self.refresh()
        self.ctx.say(f"Loaded glossary {os.path.basename(path)}")
        return path

    def clear_glossary(self) -> None:
        self.set("manga_custom_glossary_path", "")
        self.set("manga_generated_glossary_path", "")
        self.set("manga_glossary_auto_load_suppressed", True)
        self.refresh()

    # ---- inpainting --------------------------------------------------------------------------------

    def _inpaint_controls(self, cfg: dict) -> list:
        method, local = svc.current_inpaint_choice(cfg)
        status = svc.inpaint_status(cfg, model_status=self._local_model_status)
        controls: list = [ft.Row([ft.Text("Status", theme_style=ft.TextThemeStyle.BODY_SMALL),
                                  reason_or_chip(status, key=self.k("ms-inpaint-status"), dark=self.ctx.dark),
                                  hint_text(status.detail) if status.detail else ft.Container()],
                                 wrap=True, spacing=8, key=self.k(f"ms-inpaint-status-row-{self._gen}"))]
        for row in svc.inpaint_method_rows():
            selected = row.value == method
            controls.append(ft.ListTile(
                leading=ft.Icon(ft.Icons.RADIO_BUTTON_CHECKED if selected else ft.Icons.RADIO_BUTTON_UNCHECKED),
                title=ft.Text(row.label, color=_muted(row.disabled)), dense=True,
                trailing=ReasonChip(reason=row.chip or row.reason, detail=row.reason) if row.disabled else None,
                on_click=None if row.disabled else (lambda e, v=row.value: self.select_inpaint(v)),
                key=self.k(f"ms-inpaint-{row.value}-{self._gen}")))
        if method == "local":
            controls.append(ft.Row([
                ft.Text("Local / API model", theme_style=ft.TextThemeStyle.TITLE_SMALL),
                ft.IconButton(icon=ft.Icons.INFO_OUTLINE, tooltip="Model information", key=self.k("ms-model-info"),
                              size_constraints=HIT_TARGET, on_click=lambda e, m=local: self.show_model_info(m)),
            ], spacing=4, vertical_alignment=ft.CrossAxisAlignment.CENTER))
            for row in svc.local_model_rows():
                selected = row.value == local
                controls.append(ft.ListTile(
                    leading=ft.Icon(ft.Icons.RADIO_BUTTON_CHECKED if selected else ft.Icons.RADIO_BUTTON_UNCHECKED),
                    title=ft.Text(row.label, color=_muted(row.disabled)),
                    subtitle=ft.Text(row.value, theme_style=ft.TextThemeStyle.BODY_SMALL),
                    dense=True,
                    trailing=ReasonChip(reason=row.chip or row.reason, detail=row.reason) if row.disabled else None,
                    on_click=None if row.disabled else (lambda e, v=row.value: self.select_inpaint("local", v)),
                    key=self.k(f"ms-local-{row.value}-{self._gen}")))
            if local in svc.ONNX_INPAINT_MODELS:
                model_id = self.session.models.for_local_method(local) if self.session.models.available else None
                if model_id:
                    row = self._download_row("inpaint", model_id, key_prefix=self.k("ms-inpaint-model"),
                                             on_change=lambda entry: self._refresh_inpaint_status())
                    controls.append(row.control)
                else:
                    self.model_rows.pop("inpaint", None)
                    controls.append(ReasonChip(
                        reason="Model downloads unavailable",
                        detail=(f"The model registry (manga_models) has no download for {local}."
                                if self.session.models.available else svc.MISSING_CORE + " (manga_models).")))
            elif local == "custom-image-edit":
                controls += self._image_edit_controls(cfg)
            if local and local != "custom-image-edit":
                controls += self._model_file_controls(cfg, local)
        elif method == "cloud":
            quality = str(cfg.get("manga_inpaint_quality", "high") or "high")
            controls += [
                self._secret_row("Replicate API key", svc.K_REPLICATE_KEY, "ms-replicate-key", cfg),
                ft.SegmentedButton(segments=[ft.Segment(value="high", label=ft.Text("High Quality")),
                                             ft.Segment(value="fast", label=ft.Text("Fast"))],
                                   selected=[quality], key=self.k("ms-inpaint-quality"),
                                   on_change=lambda e: self.set("manga_inpaint_quality",
                                                                (list(e.control.selected) or ["high"])[0])),
                hint_text("Not recommended: this option has performed poorly in tests."),
            ]
        controls += self._mask_preset_controls()
        return controls

    #: The desktop Browse filter of a local inpainting model file (``_browse_local_model``).
    MODEL_FILE_EXTENSIONS = ("safetensors", "pt", "pth", "ckpt", "onnx")

    def _model_file_controls(self, cfg: dict, local: str) -> list:
        """Desktop Browse: a model file of your own for the local method (``manga_<method>_model_path``,
        which the run loads instead of the downloaded model)."""
        path = str(cfg.get(f"manga_{local}_model_path") or "")
        row: list = [ft.TextButton(content="Import model file…", icon=ft.Icons.UPLOAD_FILE,
                                   key=self.k("ms-model-import"),
                                   on_click=lambda e, m=local: self.ctx.spawn(self.import_model_file(m)))]
        if path:
            row += [hint_text(os.path.basename(path), key=self.k("ms-model-file")),
                    ft.IconButton(icon=ft.Icons.CLOSE, tooltip="Use the downloaded model", key=self.k("ms-model-clear"),
                                  size_constraints=HIT_TARGET, on_click=lambda e, m=local: self.clear_model_file(m))]
        return [ft.Row(row, wrap=True, spacing=4, vertical_alignment=ft.CrossAxisAlignment.CENTER)]

    def show_model_info(self, model_type: str) -> str:
        """ⓘ: the desktop Model Information text of the local model type (``manga_models.MODEL_INFO``)."""
        from glossarion_mobile.ui.components.info_sheet import InfoSheet

        text = svc.model_info_text(model_type) or "Please select a model type first"
        title = f"{model_type.upper()} Model" if model_type else "Model information"
        if self.ctx.page is not None:
            InfoSheet(title=title, body=text).show(self.ctx.page)
        return text

    async def import_model_file(self, model_type: str) -> Optional[str]:
        """Desktop ``_browse_local_model``: pick a model file and store its path in
        ``manga_<type>_model_path`` (kept as a private copy: picker cache copies vanish).

        The Android / iOS pickers map extensions to MIME types / UTIs and none of the model
        extensions has one, so there the picker shows every file and the extension is checked
        after the pick; the desktop keeps the Browse filter. The picker's Inbox copy is moved
        into the models folder, so a large model is not stored twice."""
        files = self.ctx.files
        if files is None:
            self.ctx.say("File picking is not available")
            return None
        platform = str(getattr(files, "platform", None) or getattr(self.ctx, "platform", "") or "")
        extensions = None if platform in ("android", "ios") else list(self.MODEL_FILE_EXTENSIONS)
        picked = await files.pick_files(allowed_extensions=extensions, allow_multiple=False,
                                        dialog_title=f"Select {model_type.upper()} Model")
        if not picked:
            return None
        chosen = picked[0]
        name = str(getattr(chosen, "name", "") or os.path.basename(chosen.path))
        inbox_copy = not getattr(chosen, "reused", False)  # an identical file already in the Inbox stays
        if os.path.splitext(name)[1].lower().lstrip(".") not in self.MODEL_FILE_EXTENSIONS:
            if inbox_copy:
                await self.ctx.io(self._discard, chosen.path)
            self.ctx.say(f"{name} is not a model file (.safetensors, .pt, .pth, .ckpt or .onnx)")
            return None
        path = await self.ctx.io(self._keep_private_copy, chosen.path, "models", inbox_copy)
        self.set(f"manga_{model_type}_model_path", path)
        self.refresh()
        self.ctx.say(f"Using {os.path.basename(path)} for {model_type}")
        return path

    def clear_model_file(self, model_type: str) -> None:
        self.set(f"manga_{model_type}_model_path", "")
        self.refresh()

    def _mask_preset_controls(self) -> list:
        """The manga settings dialog's mask dilation quick presets (B&W Manga / Colored / Uniform;
        shared ``manga_settings_defaults.MASK_PRESETS``)."""
        rows = svc.mask_preset_rows()
        if not rows:
            return []
        chips = [ft.OutlinedButton(content=label, key=self.k(f"ms-mask-{pid}"),
                                   on_click=lambda e, p=pid: self.apply_mask_preset(p))
                 for pid, label in rows]
        return [ft.Text("Mask presets", theme_style=ft.TextThemeStyle.TITLE_SMALL),
                ft.Row(chips, wrap=True, spacing=8, key=self.k(f"ms-mask-presets-{self._gen}")),
                hint_text("Sets the mask dilation and the per-region dilation iterations.")]

    def apply_mask_preset(self, preset: str) -> bool:
        """A mask quick preset (desktop ``_set_mask_preset`` + Save): mask dilation, all-iterations,
        and the text bubble / empty bubble / free text dilation iterations."""
        updates = svc.mask_preset_updates(preset)
        if not updates:
            self.ctx.say("Mask presets need the shared manga settings module")
            return False
        for key, value in updates.items():
            self.set(key, value)
        self.refresh()
        self.ctx.say(f"Applied the {dict(svc.mask_preset_rows()).get(preset, preset)} mask preset")
        return True

    def _local_model_status(self, method: str) -> Any:
        manager = self.session.models
        if not manager.available:
            return None
        model_id = manager.for_local_method(method)
        return manager.status(model_id) if model_id else None

    def _refresh_inpaint_status(self) -> None:
        self.refresh()

    def select_inpaint(self, method: str, local: Optional[str] = None) -> bool:
        if method != "skip":
            reason = svc.value_reason(svc.K_INPAINT_METHOD, method)
            if not reason and local:
                reason = svc.value_reason(svc.K_LOCAL_MODEL, local)
            if reason:
                self.ctx.say(reason)
                return False
        if method == "local" and local is None:
            local = svc.current_inpaint_choice(self.config())[1]
            if svc.value_reason(svc.K_LOCAL_MODEL, local):
                local = "anime_onnx"
        for key, value in svc.inpaint_choice_updates(method, local).items():
            self.set(key, value)
        if local == "custom-image-edit" and method == "local":
            endpoint = str(self.config().get(svc.K_EDIT_ENDPOINT) or "")
            self.set(svc.K_USE_EDIT_ENDPOINT, bool(endpoint))
        self.refresh()
        return True

    def _image_edit_controls(self, cfg: dict) -> list:
        endpoint = str(cfg.get(svc.K_EDIT_ENDPOINT) or cfg.get("manga_custom-image-edit_model_path") or "")
        batch = bool(svc.effective_setting(cfg, "manga_batch_image_requests_enabled", True))
        try:
            size = int(svc.effective_setting(cfg, "manga_batch_image_requests_size", 5) or 5)
        except (TypeError, ValueError):
            size = 5
        self.endpoint_field = ft.TextField(label="Image edit endpoint (blank = the main image provider)",
                                           value=endpoint, dense=True, key=self.k("ms-edit-endpoint"),
                                           on_blur=lambda e: self.set_image_edit_endpoint(e.control.value),
                                           on_submit=lambda e: self.set_image_edit_endpoint(e.control.value))
        self.test_result = hint_text("", key=self.k("ms-edit-test-result"))
        can_test = callable(svc.core_attr("manga_env", "test_custom_image_edit_endpoint",
                                          "check_custom_image_edit_endpoint"))
        test_row: list = [
            ft.FilledTonalButton(content="Edit Prompt", icon=ft.Icons.EDIT_NOTE, key=self.k("ms-edit-prompt"),
                                 on_click=lambda e: self.edit_image_edit_prompts()),
            ft.TextButton(content="Test", icon=ft.Icons.NETWORK_CHECK, key=self.k("ms-edit-test"), disabled=not can_test,
                          on_click=lambda e: self.ctx.spawn(self.test_image_edit())),
            ft.TextButton(content="Image keys", icon=ft.Icons.KEY, key=self.k("ms-edit-keys"),
                          on_click=lambda e: self.ctx.go("settings.keys.pool", {"pool": "inpainter"})),
        ]
        if not can_test:
            test_row.append(ReasonChip(reason="Test needs manga_env",
                                       detail="The endpoint check is the desktop one, shared through manga_env, "
                                              "which this build does not provide yet."))
        return [
            self.endpoint_field,
            ft.Row(test_row, wrap=True, spacing=8),
            ft.Row([
                ft.Switch(label="Batch Image Requests", value=batch, key=self.k("ms-edit-batch"),
                          on_change=lambda e: self.set("manga_batch_image_requests_enabled", bool(e.control.value))),
                ft.Dropdown(options=[ft.DropdownOption(key=str(n), text=str(n)) for n in range(1, 33)],
                            value=str(max(1, min(32, size))), width=96, dense=True, key=self.k("ms-edit-batch-size"),
                            on_select=lambda e: self.set("manga_batch_image_requests_size", int(e.control.value or 5))),
            ], wrap=True, spacing=8),
            self.test_result,
        ]

    def set_image_edit_endpoint(self, value: Any) -> None:
        """``_sync_custom_image_edit_controls``: the URL in every key the desktop keeps in lockstep."""
        url = str(value or "").strip()
        self.set(svc.K_EDIT_ENDPOINT, url)
        self.set("manga_custom-image-edit_model_path", url)
        self.set(svc.K_USE_EDIT_ENDPOINT, bool(url))

    async def test_image_edit(self) -> str:
        cfg = self.config()
        self.test_result.value = "Testing image edit endpoint…"
        push(self.test_result)
        try:
            message = await self.ctx.io(svc.test_image_edit_endpoint, cfg)
        except Exception as exc:
            message = f"Test failed: {exc}"
        self.test_result.value = message
        push(self.test_result)
        return message

    # ---- rendering ---------------------------------------------------------------------------------

    def _render_controls(self, cfg: dict) -> list:
        style = preview_style(cfg)
        self.preview = self._preview(style)
        get = (lambda key, default=None: svc.effective_setting(cfg, key, default))
        mode = str(get("manga_font_size_mode", "fixed") or "fixed")
        presets_ok = svc.presets_available()  # cheap; the presets themselves are measured on the io pool
        reset_ok = svc.rendering_reset_available()  # cheap; the reset values are measured on the io pool
        preset_row: list = [ft.OutlinedButton(content=label, key=self.k(f"ms-preset-{pid}"), disabled=not presets_ok,
                                              on_click=lambda e, p=pid: self.ctx.spawn(self.apply_preset_async(p)))
                            for pid, label in PRESETS]
        preset_row.append(ft.TextButton(content="Reset", icon=ft.Icons.RESTART_ALT, key=self.k("ms-render-reset"),
                                        disabled=not reset_ok,
                                        on_click=lambda e: self.ctx.spawn(self.reset_rendering())))
        if not presets_ok:
            preset_row.append(ReasonChip(reason="Needs the shared presets",
                                         detail="The Manga / Manhwa / Large Text presets come from the shared manga "
                                                "settings module (manga_settings_defaults), which this build does not "
                                                "provide."))
        elif not reset_ok:
            preset_row.append(ReasonChip(reason="Reset needs the shared defaults",
                                         detail="The Reset to Defaults values come from the shared manga settings "
                                                "module (manga_settings_defaults), which this build does not provide."))
        controls: list = [
            self.preview,
            ft.Row(preset_row, wrap=True, spacing=8),
            self._font_row(cfg),
            self._segmented("Font size mode", "manga_font_size_mode", SIZE_MODES, mode, "ms-size-mode"),
        ]
        if mode == "fixed":
            controls.append(self._slider("Font size (0 = Auto)", "manga_font_size", 0, 72, 72,
                                         int(get("manga_font_size", 0) or 0), "ms-font-size", integer=True))
        elif mode == "multiplier":
            controls.append(self._slider("Size multiplier", "manga_font_size_multiplier", 0.5, 2.0, 30,
                                         float(get("manga_font_size_multiplier", 1.0) or 1.0), "ms-font-mult"))
        else:  # Auto: the desktop Minimum / Maximum Font Size spin boxes
            controls.append(self._font_bounds_row(get))
        controls += [
            self._segmented("Algorithm", ("manga_settings", "font_sizing", "algorithm"), ALGORITHMS,
                            str(get(("manga_settings", "font_sizing", "algorithm"), "smart") or "smart"), "ms-algorithm"),
            self._slider("Line spacing", ("manga_settings", "font_sizing", "line_spacing"), 1.0, 2.0, 20,
                         float(get(("manga_settings", "font_sizing", "line_spacing"), 1.3) or 1.3), "ms-line-spacing"),
            self._segmented("Background style", "manga_bg_style", BG_STYLES,
                            str(get("manga_bg_style", "circle") or "circle"), "ms-bg-style"),
            self._slider("Background opacity", "manga_bg_opacity", 0, 255, 51, int(get("manga_bg_opacity", 130) or 0),
                         "ms-bg-opacity", integer=True),
            self._slider("Background size", "manga_bg_reduction", 0.5, 2.0, 30,
                         float(get("manga_bg_reduction", 1.0) or 1.0), "ms-bg-size"),
            self._switch("Free text only background opacity", "manga_free_text_only_bg_opacity",
                         bool(get("manga_free_text_only_bg_opacity", True)), "ms-free-text-bg"),
            self._color_row("Text colour", "manga_text_color", _rgb(get("manga_text_color", [102, 0, 0]), (102, 0, 0)),
                            "ms-text-color"),
            self._switch("Shadow / outline", "manga_shadow_enabled", bool(get("manga_shadow_enabled", True)),
                         "ms-shadow"),
        ]
        if bool(get("manga_shadow_enabled", True)):
            controls += [
                self._color_row("Shadow colour", "manga_shadow_color",
                                _rgb(get("manga_shadow_color", [204, 128, 128]), (204, 128, 128)), "ms-shadow-color"),
                self._slider("Shadow offset X", "manga_shadow_offset_x", -10, 10, 20,
                             int(get("manga_shadow_offset_x", 2) or 0), "ms-shadow-x", integer=True),
                self._slider("Shadow offset Y", "manga_shadow_offset_y", -10, 10, 20,
                             int(get("manga_shadow_offset_y", 2) or 0), "ms-shadow-y", integer=True),
                self._slider("Shadow blur", "manga_shadow_blur", 0, 10, 10, int(get("manga_shadow_blur", 0) or 0),
                             "ms-shadow-blur", integer=True),
            ]
        controls += [
            self._switch("Safe area (keep text inside the bubble's inner area)", "manga_safe_area_enabled",
                         bool(get("manga_safe_area_enabled", False)), "ms-safe-area"),
            self._slider("Safe area scale", "manga_safe_area_scale", 0.70, 1.10, 40,
                         float(get("manga_safe_area_scale", 1.0) or 1.0), "ms-safe-area-scale"),
            self._switch("Constrain text to bubble", "manga_constrain_to_bubble",
                         bool(get("manga_constrain_to_bubble", True)), "ms-constrain"),
            self._switch("Force CAPS LOCK", "manga_force_caps_lock", bool(get("manga_force_caps_lock", True)),
                         "ms-caps"),
            self._switch("Strict text wrapping", "manga_strict_text_wrapping",
                         bool(get("manga_strict_text_wrapping", True)), "ms-strict-wrap"),
        ]
        return controls

    def _preview(self, style: dict) -> ft.Control:
        # every preview is a new subtree under new keys (a live update replaces it in place)
        self._pgen = getattr(self, "_pgen", 0) + 1
        p = self._pgen
        shadow = None
        if style["shadow"]:
            dx, dy = style["shadow_offset"]
            shadow = ft.BoxShadow(color=style["shadow_color"], offset=ft.Offset(dx, dy),
                                  blur_radius=float(style["shadow_blur"]))
        text = ft.Text(style["text"], color=style["color"], size=style["size"], text_align=ft.TextAlign.CENTER,
                       weight=ft.FontWeight.W_700, key=self.k(f"ms-preview-text-p{p}"),
                       style=ft.TextStyle(height=style["line_spacing"], shadow=shadow))
        radius = 999 if style["bg_style"] == "circle" else (24 if style["bg_style"] == "wrap" else 6)
        bubble = ft.Container(content=text, bgcolor=style["bg"], border_radius=radius,
                              padding=ft.Padding.symmetric(horizontal=28, vertical=20),
                              border=ft.Border.all(2, ft.Colors.BLACK), key=self.k(f"ms-preview-bubble-p{p}"))
        return ft.Container(content=bubble, alignment=ft.Alignment.CENTER, height=180,
                            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST, border_radius=tokens.RADII["card"],
                            key=self.k(f"ms-preview-p{p}"),
                            tooltip="Sample bubble (the final page uses the desktop renderer)")

    def _font_row(self, cfg: dict) -> ft.Control:
        path = str(cfg.get("manga_font_path") or "")
        style = str(cfg.get("manga_font_style") or "Default")
        fonts = self.available_fonts(cfg)
        # "Default": on a phone the run renders with a scalable system font (manga_models.mobile_default_font)
        options = [ft.DropdownOption(key="", text="Default (system font)")] + [
            ft.DropdownOption(key=p, text=os.path.splitext(os.path.basename(p))[0]) for p in fonts]
        return ft.Row([
            ft.Dropdown(label="Font", options=options, value=path if path in fonts else "", dense=True, width=220,
                        key=self.k("ms-font"), on_select=lambda e: self.select_font(e.control.value or "")),
            ft.TextButton(content="Import font", icon=ft.Icons.FONT_DOWNLOAD, key=self.k("ms-font-import"),
                          on_click=lambda e: self.ctx.spawn(self.import_font())),
            hint_text(style if style != "Default" else ""),
        ], wrap=True, spacing=8)

    def available_fonts(self, cfg: dict) -> list:
        """Imported fonts (config ``custom_fonts``) + the font catalog of the shared core when present."""
        found: list = []
        catalog = svc.core_attr("manga_settings_defaults", "available_fonts", "list_fonts")
        if callable(catalog):
            try:
                found.extend(str(p) for p in catalog(cfg) or ())
            except Exception:
                log.debug("font catalog failed", exc_info=True)
        for entry in cfg.get("custom_fonts") or []:
            path = entry.get("path") if isinstance(entry, dict) else entry
            if path and os.path.isfile(str(path)) and str(path) not in found:
                found.append(str(path))
        return found

    def select_font(self, path: str) -> None:
        self.set("manga_font_path", path)
        self.set("manga_font_style", os.path.splitext(os.path.basename(path))[0] if path else "Default")
        self.refresh()

    async def import_font(self) -> Optional[str]:
        files = self.ctx.files
        if files is None:
            self.ctx.say("File picking is not available")
            return None
        picked = await files.pick_files(allowed_extensions=["ttf", "otf", "ttc"], allow_multiple=False,
                                        dialog_title="Import a font")
        if not picked:
            return None
        path = await self.ctx.io(self._keep_private_copy, picked[0].path, "fonts")
        fonts = list(self.config().get("custom_fonts") or [])
        name = os.path.splitext(os.path.basename(path))[0]
        if not any((f.get("path") if isinstance(f, dict) else f) == path for f in fonts):
            fonts.append({"name": name, "path": path})
            self.set("custom_fonts", fonts)
        self.select_font(path)
        return path

    def _switch(self, label: str, key: Any, value: bool, ui_key: str) -> ft.Control:
        return ft.Switch(label=label, value=bool(value), key=self.k(ui_key),
                         on_change=lambda e: self._set_and_preview(key, bool(e.control.value)))

    def _segmented(self, label: str, key: Any, options: Sequence[tuple], value: str, ui_key: str) -> ft.Control:
        return ft.Column([
            ft.Text(label, theme_style=ft.TextThemeStyle.BODY_SMALL),
            ft.SegmentedButton(segments=[ft.Segment(value=v, label=ft.Text(t)) for v, t in options],
                               selected=[value] if value in dict(options) else [], key=self.k(ui_key),
                               show_selected_icon=False, allow_empty_selection=True,
                               on_change=lambda e: self._set_and_refresh(key, (list(e.control.selected) or [value])[0])),
        ], spacing=2, tight=True)

    def _slider(self, label: str, key: Any, lo: float, hi: float, divisions: int, value: float, ui_key: str,
                *, integer: bool = False) -> ft.Control:
        value = max(lo, min(hi, value))
        text = ft.Text(f"{label}: {int(value) if integer else round(value, 2)}", theme_style=ft.TextThemeStyle.BODY_SMALL)

        def changed(e: Any) -> None:
            v = float(e.control.value)
            text.value = f"{label}: {int(round(v)) if integer else round(v, 2)}"
            push(text)

        def ended(e: Any) -> None:
            v = float(e.control.value)
            self._set_and_preview(key, int(round(v)) if integer else round(v, 2))

        return ft.Column([text, ft.Slider(min=lo, max=hi, divisions=divisions, value=value, key=self.k(ui_key),
                                          on_change=changed, on_change_end=ended)], spacing=0, tight=True)

    @staticmethod
    def _font_bounds(get: Any) -> tuple:
        """``(min, max)`` the Auto mode uses (manga_env: ``manga_max_font_size``, else rendering.auto_max_size;
        rendering.auto_min_size)."""
        def number(value: Any, default: int) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        high = number(get("manga_max_font_size", None), 0) or number(
            get(("manga_settings", "rendering", "auto_max_size"), 48), 48)
        low = number(get(("manga_settings", "rendering", "auto_min_size"), 8), 8)
        return low, high

    def _font_bounds_row(self, get: Any) -> ft.Control:
        low, high = self._font_bounds(get)
        lo, hi = FONT_BOUNDS

        def field(label: str, which: str, value: int, ui_key: str) -> ft.TextField:
            return ft.TextField(label=label, value=str(value), width=170, dense=True,
                                keyboard_type=ft.KeyboardType.NUMBER, helper=f"{lo}–{hi}", key=self.k(ui_key),
                                on_submit=lambda e: self.set_font_bound(which, e.control.value),
                                on_blur=lambda e: self.set_font_bound(which, e.control.value))

        return ft.Row([field("Minimum font size", "min", low, "ms-min-font"),
                       field("Maximum font size", "max", high, "ms-max-font")], wrap=True, spacing=12)

    def set_font_bound(self, which: str, raw: Any) -> Optional[tuple]:
        """Minimum / Maximum Font Size (Auto mode): clamped to 1-999, a minimum above the maximum is
        lowered to it (desktop ``_validate_font_size_range_manga``), saved like the desktop
        ``_save_rendering_settings`` (``manga_max_font_size`` plus the rendering / font_sizing mirrors)."""
        try:
            value = int(float(str(raw).strip()))
        except (TypeError, ValueError):
            self.refresh()
            return None
        lo, hi = FONT_BOUNDS
        value = max(lo, min(hi, value))
        low, high = self._font_bounds(self.get)
        if which == "min":
            low = value
        else:
            high = value
        if low > high:
            low = high
        self.set_many({
            "manga_max_font_size": high,
            ("manga_settings", "rendering", "auto_min_size"): low,
            ("manga_settings", "rendering", "auto_max_size"): high,
            ("manga_settings", "font_sizing", "min_size"): low,
            ("manga_settings", "font_sizing", "max_size"): high,
        })
        self.refresh()
        return low, high

    def _color_row(self, label: str, key: str, rgb: tuple, ui_key: str) -> ft.Control:
        swatches = []
        for index, color in enumerate(COLOR_SWATCHES):
            swatches.append(ft.Container(width=28, height=28, border_radius=14, bgcolor=_hex(color),
                                         border=ft.Border.all(3 if color == rgb else 1,
                                                              ft.Colors.PRIMARY if color == rgb else ft.Colors.OUTLINE),
                                         on_click=lambda e, c=color: self._set_and_refresh(key, list(c)),
                                         key=self.k(f"{ui_key}-{index}"), tooltip=_hex(color)))
        field = ft.TextField(value=_hex(rgb), width=110, dense=True, key=self.k(f"{ui_key}-hex"),
                             on_submit=lambda e: self._set_hex(key, e.control.value),
                             on_blur=lambda e: self._set_hex(key, e.control.value))
        return ft.Column([ft.Text(label, theme_style=ft.TextThemeStyle.BODY_SMALL),
                          ft.Row([*swatches, field], wrap=True, spacing=6)], spacing=2, tight=True)

    def _set_hex(self, key: str, value: Any) -> None:
        text = str(value or "").strip().lstrip("#")
        if not re.fullmatch(r"[0-9a-fA-F]{6}", text):
            return
        rgb = [int(text[i:i + 2], 16) for i in (0, 2, 4)]
        self._set_and_refresh(key, rgb)

    def _set_and_preview(self, key: Any, value: Any) -> None:
        self.set(key, value)
        self._refresh_preview()

    def _set_and_refresh(self, key: Any, value: Any) -> None:
        self.set(key, value)
        self.refresh()

    def _refresh_preview(self) -> None:
        if getattr(self, "preview", None) is None:
            return
        fresh = self._preview(preview_style(self.config()))
        column = self.render_card.content
        try:
            index = column.controls.index(self.preview)
        except ValueError:
            return
        column.controls[index] = fresh
        self.preview = fresh
        push(column)

    def apply_preset(self, preset: str, updates: Optional[dict] = None) -> bool:
        """Apply a preset's config writes (``updates``, or the session cache); False when they are
        not measured yet (``apply_preset_async`` measures them on the io pool)."""
        if updates is None:
            updates = svc.cached_font_preset_updates(preset)
        if not updates:
            return False
        for key, value in updates.items():
            self.set(key, value)
        self.refresh()
        self.ctx.say(f"Applied the {dict(PRESETS).get(preset, preset)} preset")
        return True

    async def apply_preset_async(self, preset: str) -> bool:
        """The preset buttons: the shared presets are measured once per session on the io pool
        (scratch manga tabs; never on the UI loop, never while a job owns the process state)."""
        if self.apply_preset(preset):
            return True
        try:
            updates = await self.ctx.io(svc.font_preset_updates, preset)
        except svc.PresetsBusy as exc:
            self.ctx.say(str(exc))
            return False
        except Exception as exc:
            self.ctx.say(f"Preset failed: {exc}")
            return False
        if not updates:
            self.ctx.say("Presets need the shared manga settings module")
            return False
        return self.apply_preset(preset, updates)

    async def reset_rendering(self) -> bool:
        from glossarion_mobile.ui.tools.common import ask

        updates = svc.cached_rendering_reset_updates()
        if updates is None:
            try:
                updates = await self.ctx.io(svc.rendering_reset_updates)
            except svc.PresetsBusy as exc:
                self.ctx.say(str(exc))
                return False
            except Exception as exc:
                self.ctx.say(f"Reset failed: {exc}")
                return False
        if not updates:
            self.ctx.say("Reset needs the shared manga settings module")
            return False
        answer = await ask(self.ctx, "Reset to Defaults", RESET_CONFIRM_TEXT,
                           (("no", "No", "text"), ("yes", "Yes", "filled")), key=self.k("ms-reset-confirm"))
        if answer != "yes":
            return False
        for key, value in updates.items():
            self.set(key, value)
        self.refresh()
        return True

    # ---- all settings (schema groups) ----------------------------------------------------------------

    def _section_keys(self) -> list:
        settings = getattr(self.ctx, "settings", None)
        schema = getattr(settings, "schema", None) if settings is not None else None
        section = schema.section("manga.settings") if schema is not None and getattr(schema, "available", False) else None
        return list(section.keys) if section is not None else []

    def _build_all_groups(self) -> None:
        keys = self._section_keys()
        column = self.all_card.content
        header = column.controls[0]
        if not keys:
            column.controls = [header, hint_text("The settings schema is not available in this session.")]
            return
        self.group_tiles: dict = {}
        controls: list = []
        for gid, title, group_keys in group_manga_keys(keys, exclude=CURATED_KEYS):
            tile = ft.ExpansionTile(title=ft.Text(f"{title} ({len(group_keys)})"), controls=[], key=f"ms-group-{gid}",
                                    on_change=lambda e, g=gid, k=group_keys: self._expand_group(g, k, e))
            self.group_tiles[gid] = (tile, group_keys)
            controls.append(tile)
        column.controls = [header, *controls]

    def _expand_group(self, gid: str, keys: Sequence[str], e: Any = None) -> None:
        """Tiles are built when a group first opens (166 settings would be slow to build up front)."""
        if gid in self.group_built:
            return
        tile, _keys = self.group_tiles[gid]
        controls, tiles = schema_tiles(self.ctx, keys)
        self.tiles.update(tiles)
        tile.controls = controls or [hint_text("No settings")]
        self.group_built.add(gid)
        push(tile)

    # ---- models ------------------------------------------------------------------------------------

    def open_models(self) -> ModelManagerSheet:
        sheet = ModelManagerSheet(self.ctx, self.session.models, on_change=lambda entry: self.refresh())
        self.ctx.extras["manga_models_sheet"] = sheet
        return sheet.show(self.ctx.page)
