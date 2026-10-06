"""ModeOptionsSheet (UI_SPEC §2.6): opened by tapping the active output mode.

Title "Output: <mode>", the "This chat only" switch (it scopes the mode itself: the chat
override instead of the global ``direct_text_output_mode``) and the mode's options - the shared
settings tiles of Settings bound to the global config keys (desktop Other Settings › Image
Translation & Vision API / main window), evaluated as if the mode were selected (the desktop
shows a mode's sub-settings only while it is selected):

* Vision: OCR prompts, skip translation, batch + slots, keep OCR image, long images, hide labels,
  Vision keys (``qa_scan`` pool);
* Image: output resolution 1K / 2K / 4K, batch + slots, Image keys (``inpainter`` pool, desktop
  "Image Keys"), the custom image-edit endpoint status chip -> Settings › Endpoints;
* Video: duration and resolution (NanoGPT video keys);
* Audio: TTS voice / file, TTS keys; Google Cloud TTS voices need google-cloud-texttospeech
  (a ReasonChip when the build lacks it - dependency rule);
* Refine: refinement mode, raw prompt role, refine prompts.

Image / Video / Audio end with "Generate from prompt (no input)": the composer text is the
prompt (a ``generate_media`` job); disabled with a reason when the composer is empty or has an
attachment. ``ModeOptionsContent`` is the body; the ＋ sheet shows it inline (§2.5).
"""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from typing import Any, Callable, Iterator, Optional

import flet as ft

from glossarion_mobile.ui.chat.media_model import GENERATIVE_MODES
from glossarion_mobile.ui.chat.output_modes import mode_label, normalize_mode, output_mode
from glossarion_mobile.ui.components.reason_chip import ReasonChip
from glossarion_mobile.ui.theme import icon_data

__all__ = [
    "GENERATE_LABEL",
    "GENERATE_REASONS",
    "MODE_HINTS",
    "MODE_KEY_POOLS",
    "MODE_OPTION_KEYS",
    "MODE_TILE_LABELS",
    "ModeOptionsContent",
    "ModeOptionsSheet",
    "TTS_SDK_REASON",
    "generate_block_reason",
    "image_edit_endpoint_status",
]

MODE_HINTS = {
    "text": "Translate text, documents, subtitles and books",
    "vision": "Vision OCR prompt · Skip translation (OCR only) · Batch Vision API requests · Keep OCR image · "
    "Process long images · Hide labels · Vision keys",
    "image": "Output resolution 1K / 2K / 4K · Batch requests · Image keys · Custom image-edit endpoint",
    "video": "Duration 5 / 10 / 15 / 20 / 30 / 60 s · Resolution 360p / 480p / 720p / 1080p",
    "audio": "TTS voice · TTS keys",
    "refinement": "Refinement mode (Full / Full + raw / Failed / Partial / Partial.b / Partial.b2) · "
    "Raw prompt role · Refine prompt",
}
#: Schema keys rendered as settings tiles per mode (UI_SPEC §2.6 "Options (bound schema keys)").
MODE_OPTION_KEYS = {
    "vision": (
        "vision_ocr_prompt", "vision_ocr_user_prompt", "vision_ocr_skip_translation",
        "vision_ocr_batch_translation", "vision_ocr_batch_size", "vision_ocr_keep_images",
        "process_webnovel_images", "hide_image_translation_label",
    ),
    "image": ("image_output_resolution", "vision_ocr_batch_translation", "vision_ocr_batch_size"),
    "video": ("nanogpt_video_duration", "nanogpt_video_resolution"),
    "audio": ("tts_voice",),
    "refinement": (
        "multipass_refinement_mode", "refinement_full_with_raw_raw_role",
        "refinement_system_prompt", "refinement_user_prompt",
    ),
}
#: The key pool of a mode (Settings › API keys page slug, desktop button label).
MODE_KEY_POOLS = {"vision": ("qa_vision", "Vision keys"), "image": ("inpainter", "Image keys"), "audio": ("tts", "TTS keys")}
#: Desktop labels where the generated schema label belongs to a neighbouring widget
#: (both video combos sit on the "Output Resolution:" row in the extractor's view).
MODE_TILE_LABELS = {"nanogpt_video_duration": "Video Duration", "nanogpt_video_resolution": "Video Resolution"}
GENERATE_LABEL = "Generate from prompt (no input)"
GENERATE_REASONS = {
    "empty": "Type a prompt in the composer first",
    "attachment": "Remove the attachment to generate from a prompt",
}
TTS_SDK_REASON = "Google Cloud TTS voices need google-cloud-texttospeech (not in this build)"


def generate_block_reason(has_text: bool, has_attachment: bool) -> Optional[str]:
    """Why "Generate from prompt (no input)" is disabled (None: it may run)."""
    if has_attachment:
        return GENERATE_REASONS["attachment"]
    if not has_text:
        return GENERATE_REASONS["empty"]
    return None


def image_edit_endpoint_status(get: Callable[[str, Any], Any]) -> str:
    """The custom image-edit endpoint chip: "Custom image-edit endpoint: On · host" / "Off"."""
    enabled = bool(get("use_custom_image_edit_endpoint", False))
    url = str(get("custom_image_edit_endpoint", "") or "").strip()
    if enabled and url:
        host = url.split("://", 1)[-1].split("/", 1)[0]
        return f"Custom image-edit endpoint: On · {host}"
    return "Custom image-edit endpoint: Off" if not enabled else "Custom image-edit endpoint: On · no URL"


def _google_tts_available() -> bool:
    try:
        return importlib.util.find_spec("google.cloud.texttospeech") is not None
    except (ImportError, ValueError):
        return False


class _ModeConfig(Mapping):
    """The effective config with the sheet's mode selected (rule evaluation only; never written)."""

    def __init__(self, base: Mapping, overlay: Mapping) -> None:
        self.base = base
        self.overlay = dict(overlay)

    def __getitem__(self, key: str) -> Any:
        if key in self.overlay:
            return self.overlay[key]
        return self.base[key]

    def __iter__(self) -> Iterator[str]:
        seen = set(self.overlay)
        yield from self.overlay
        for key in self.base:
            if key not in seen:
                yield key

    def __len__(self) -> int:
        return len(set(self.overlay) | set(self.base))


def _mode_config(ctx: Any, mode: str) -> Optional[Mapping]:
    try:
        from settings_rules import output_mode_flags  # shared (U4)

        from glossarion_mobile.ui.settings.tiles import EffectiveConfig

        return _ModeConfig(EffectiveConfig(ctx.store), output_mode_flags(mode).config_values)
    except Exception:
        return None


class ModeOptionsContent:
    """The options of one mode as a ``Column`` (sheet body and ＋ sheet inline section)."""

    def __init__(
        self,
        mode: str,
        *,
        ctx: Any = None,
        this_chat: Optional[bool] = None,
        on_this_chat: Optional[Callable[[bool], Any]] = None,
        has_text: bool = False,
        has_attachment: bool = False,
        on_generate: Optional[Callable[[str], Any]] = None,
        on_open_keys: Optional[Callable[[str], Any]] = None,
        on_open_endpoints: Optional[Callable[[], Any]] = None,
        config_get: Optional[Callable[[str, Any], Any]] = None,
        show_header: bool = True,
    ) -> None:
        self.mode = normalize_mode(mode)
        self.ctx = ctx  # SettingsContext for the option tiles (None: the options show a reason chip)
        self.tiles: dict = {}
        self.on_this_chat = on_this_chat
        self.on_generate = on_generate
        self.on_open_keys = on_open_keys
        self.on_open_endpoints = on_open_endpoints
        self.generate_reason = generate_block_reason(has_text, has_attachment)
        info = output_mode(self.mode)
        controls: list = []
        if show_header:
            controls.append(ft.Row(
                [ft.Icon(icon_data(info.icon)),
                 ft.Text(mode_label(self.mode), theme_style=ft.TextThemeStyle.TITLE_LARGE, weight=ft.FontWeight.W_600)],
                spacing=8,
            ))
        self.this_chat_switch = ft.Switch(label="This chat only", value=bool(this_chat),
                                          disabled=on_this_chat is None, on_change=self._this_chat_changed,
                                          key="mode-this-chat")
        if show_header:
            controls.append(self.this_chat_switch)
        controls.append(ft.Text(MODE_HINTS[self.mode], theme_style=ft.TextThemeStyle.BODY_MEDIUM))
        keys = MODE_OPTION_KEYS.get(self.mode, ())
        if keys:
            controls.extend(self._option_tiles(keys))
        pool = MODE_KEY_POOLS.get(self.mode)
        if pool is not None:
            slug, label = pool
            controls.append(ft.ListTile(
                leading=ft.Icon(ft.Icons.KEY), title=ft.Text(label), subtitle=ft.Text("Settings › API keys"),
                trailing=ft.Icon(ft.Icons.CHEVRON_RIGHT), on_click=lambda e, s=slug: self._open_keys(s),
                key=f"mode-keys-{slug}",
            ))
        if self.mode == "image":
            getter = config_get or (lambda k, d=None: (ctx.store.get(k, d) if ctx is not None else d))
            self.endpoint_chip = ft.Chip(label=ft.Text(image_edit_endpoint_status(getter)),
                                         leading=ft.Icon(ft.Icons.LINK, size=16),
                                         on_click=lambda e: self._open_endpoints(), key="mode-image-endpoint")
            controls.append(self.endpoint_chip)
        if self.mode == "audio" and not _google_tts_available():
            controls.append(ReasonChip(reason=TTS_SDK_REASON))
        self.generate_button: Optional[ft.Control] = None
        if self.mode in GENERATIVE_MODES:
            self.generate_button = ft.FilledTonalButton(
                content=GENERATE_LABEL, icon=ft.Icons.AUTO_AWESOME, disabled=self.generate_reason is not None,
                on_click=lambda e: self._generate(), key="mode-generate",
            )
            row: list = [self.generate_button]
            if self.generate_reason:
                row.append(ReasonChip(reason=self.generate_reason))
            controls.append(ft.Row(row, wrap=True, spacing=8))
        self.column = ft.Column(controls, tight=True, spacing=12)

    def _option_tiles(self, keys: tuple) -> list:
        """The shared settings tiles of ``keys`` (each skipped when the schema lacks it)."""
        ctx = self.ctx
        if ctx is None:
            return [ReasonChip(reason="These options are in Settings (settings unavailable in this session)")]
        from glossarion_mobile.ui.settings.tiles import make_tile

        config = _mode_config(ctx, self.mode)
        out = []
        for key in keys:
            try:
                spec = ctx.schema.spec(key)
            except Exception:
                spec = None
            if spec is None:
                continue
            try:
                tile = make_tile(spec, ctx, config=config) if config is not None else make_tile(spec, ctx)
            except Exception:
                continue
            label = MODE_TILE_LABELS.get(key)
            if label:
                tile.label = label
                tile.title_text.value = label
            self.tiles[key] = tile
            out.append(tile.control)
        return out

    def _this_chat_changed(self, e: Any = None) -> None:
        if self.on_this_chat is not None:
            self.on_this_chat(bool(self.this_chat_switch.value))

    def _open_keys(self, slug: str) -> None:
        if self.on_open_keys is not None:
            self.on_open_keys(slug)

    def _open_endpoints(self) -> None:
        if self.on_open_endpoints is not None:
            self.on_open_endpoints()

    def _generate(self) -> Any:
        if self.generate_reason is not None or self.on_generate is None:
            return None
        return self.on_generate(self.mode)


class ModeOptionsSheet:
    def __init__(self, mode: str, *, on_dismiss: Optional[Callable[..., Any]] = None, ctx: Any = None,
                 **content_kwargs: Any) -> None:
        self._page: Any = None
        on_generate = content_kwargs.pop("on_generate", None)
        on_open_keys = content_kwargs.pop("on_open_keys", None)
        on_open_endpoints = content_kwargs.pop("on_open_endpoints", None)
        self.content = ModeOptionsContent(
            mode, ctx=ctx,
            on_generate=(lambda m: self._close_then(on_generate, m)) if on_generate else None,
            on_open_keys=(lambda s: self._close_then(on_open_keys, s)) if on_open_keys else None,
            on_open_endpoints=(lambda: self._close_then(on_open_endpoints)) if on_open_endpoints else None,
            **content_kwargs,
        )
        self.mode = self.content.mode
        self.ctx = ctx
        self.tiles = self.content.tiles
        self.this_chat_switch = self.content.this_chat_switch
        self.generate_button = self.content.generate_button
        self.dialog = ft.BottomSheet(
            content=ft.Container(
                padding=ft.Padding.only(left=16, right=16, bottom=24),
                content=self.content.column,
            ),
            show_drag_handle=True,
            scrollable=True,
            on_dismiss=on_dismiss,
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGH,
        )

    def _close_then(self, handler: Callable[..., Any], *args: Any) -> Any:
        self.close()
        return handler(*args)

    def generate(self) -> Any:
        return self.content._generate()

    def show(self, page: Any) -> None:
        self._page = page
        page.show_dialog(self.dialog)

    def close(self) -> None:
        if self._page is not None and getattr(self.dialog, "open", False):
            try:
                self._page.pop_dialog()
            except Exception:
                pass
