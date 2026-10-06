"""Custom Mode Settings sheet (QA Scanner › Custom; desktop "Configure Custom Detection Settings").

Rows mirror the desktop dialog: "Detection Thresholds (%)" (five 10-100 sliders with the
desktop labels), "Processing Options" (consecutive chapters 1-10, sample size -1..∞ in steps
of 500 with "-1 = use all characters, 0 = disable duplicate detection", minimum text length
100-5000, "Check all file pairs (slower but more thorough)"). Defaults are
``qa_scan_runtime.DEFAULT_CUSTOM_MODE_SETTINGS``; **Save** writes
``qa_scanner_settings.custom_mode_settings`` (the desktop "Start Scan" shape, thresholds as
fractions) through the config store, **Reset** asks "Reset all values to default settings?".
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components._handlers import call_handler
from glossarion_mobile.ui.components.dialogs import ConfirmDialog
from glossarion_mobile.ui.tools import qa_model as qm
from glossarion_mobile.ui.tools.common import hint_text, section_header

__all__ = ["CustomModeSheet"]

CONFIG_PATH = ("qa_scanner_settings", "custom_mode_settings")


def _clamp(value: Any, low: int, high: int, default: int) -> int:
    try:
        number = int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


class CustomModeSheet:
    def __init__(self, ctx: Any, *, on_saved: Optional[Callable[[dict], Any]] = None) -> None:
        self.ctx = ctx
        self.on_saved = on_saved
        self.defaults = qm.custom_defaults() or {}
        saved = ctx.cfg(CONFIG_PATH, None)
        self.values = qm.custom_values(saved if isinstance(saved, dict) else None, self.defaults)
        self.saved_value: Optional[dict] = None
        self.sliders: dict = {}
        self.slider_labels: dict = {}
        rows: list[ft.Control] = [section_header("Detection Thresholds (%)")]
        low, high, _step = qm.CUSTOM_LIMITS["threshold"]
        for key, label, desc in qm.CUSTOM_THRESHOLDS:
            value = _clamp(self.values.get(key), low, high, int(self.defaults.get(key, 50) or 50))
            self.values[key] = value
            text = ft.Text(f"{value}%", width=48, text_align=ft.TextAlign.RIGHT, key=f"qa-custom-{key}-value")
            slider = ft.Slider(min=low, max=high, divisions=high - low, value=value, expand=True,
                               on_change=lambda e, k=key: self._on_slider(k, e), key=f"qa-custom-{key}")
            self.sliders[key] = slider
            self.slider_labels[key] = text
            rows.append(ft.Text(f"{label} - {desc}:", theme_style=ft.TextThemeStyle.BODY_MEDIUM))
            rows.append(ft.Row([slider, text], spacing=4))
        rows.append(section_header("Processing Options"))
        c_low, c_high, _ = qm.CUSTOM_LIMITS["consecutive_chapters"]
        self.consecutive = ft.TextField(label="Consecutive chapters to check:", dense=True,
                                        value=str(self.values.get("consecutive_chapters", 2)),
                                        keyboard_type=ft.KeyboardType.NUMBER, key="qa-custom-consecutive",
                                        helper=f"{c_low}-{c_high}")
        self.sample = ft.TextField(label="Sample size for comparison (characters):", dense=True,
                                   value=str(self.values.get("sample_size", 3000)),
                                   keyboard_type=ft.KeyboardType.NUMBER, key="qa-custom-sample",
                                   helper="-1 = use all characters, 0 = disable duplicate detection")
        m_low, m_high, _ = qm.CUSTOM_LIMITS["min_text_length"]
        self.min_length = ft.TextField(label="Minimum text length to process (characters):", dense=True,
                                       value=str(self.values.get("min_text_length", 500)),
                                       keyboard_type=ft.KeyboardType.NUMBER, key="qa-custom-min-length",
                                       helper=f"{m_low}-{m_high}")
        self.check_all = ft.Switch(label="Check all file pairs (slower but more thorough)",
                                   value=bool(self.values.get("check_all_pairs", False)), key="qa-custom-all-pairs")
        rows += [self.consecutive, self.sample, self.min_length, self.check_all]
        if not self.defaults:
            rows.insert(0, hint_text("This build has no Custom-mode defaults (qa_scan_runtime); saving is disabled.",
                                     color=ft.Colors.ERROR, key="qa-custom-missing"))
        self.save_button = ft.FilledButton(content="Save", icon=ft.Icons.SAVE, on_click=self._on_save,
                                           disabled=not self.defaults, key="qa-custom-save")
        actions = ft.Row([
            ft.TextButton(content="↺ Reset", on_click=self._on_reset, disabled=not self.defaults,
                          key="qa-custom-reset"),
            ft.TextButton(content="Cancel", on_click=lambda e: self.close(), key="qa-custom-cancel"),
            self.save_button,
        ], alignment=ft.MainAxisAlignment.END, wrap=True)
        content = ft.Column([
            ft.Text("Configure Custom Detection Settings", theme_style=ft.TextThemeStyle.TITLE_LARGE,
                    weight=ft.FontWeight.W_600),
            *rows,
            actions,
        ], tight=True, spacing=tokens.SPACING["sm"], scroll=ft.ScrollMode.AUTO)
        self.sheet = ft.BottomSheet(content=ft.Container(content=content, padding=tokens.SPACING["sheet_padding"]),
                                    show_drag_handle=True, scrollable=True, bgcolor=ft.Colors.SURFACE_CONTAINER_HIGH)
        self._page: Any = None

    def show(self, page: Any) -> "CustomModeSheet":
        self._page = page
        page.show_dialog(self.sheet)
        return self

    def close(self) -> None:
        if self._page is not None and getattr(self.sheet, "open", False):
            self._page.pop_dialog()

    def _on_slider(self, key: str, e: Any = None) -> None:
        slider = self.sliders[key]
        value = int(round(float(getattr(getattr(e, "control", None), "value", None) or slider.value or 0)))
        self.values[key] = value
        self.slider_labels[key].value = f"{value}%"
        self.ctx.push(self.slider_labels[key])

    def collect(self) -> dict:
        c_low, c_high, _ = qm.CUSTOM_LIMITS["consecutive_chapters"]
        s_low, s_high, _ = qm.CUSTOM_LIMITS["sample_size"]
        m_low, m_high, _ = qm.CUSTOM_LIMITS["min_text_length"]
        values = dict(self.values)
        for key in self.sliders:
            values[key] = int(round(float(self.sliders[key].value or 0)))
        values["consecutive_chapters"] = _clamp(self.consecutive.value, c_low, c_high,
                                                int(self.defaults.get("consecutive_chapters", 2)))
        values["sample_size"] = _clamp(self.sample.value, s_low, s_high, int(self.defaults.get("sample_size", 3000)))
        values["min_text_length"] = _clamp(self.min_length.value, m_low, m_high,
                                           int(self.defaults.get("min_text_length", 500)))
        values["check_all_pairs"] = bool(self.check_all.value)
        return values

    def save(self) -> dict:
        saved = qm.custom_saved(self.collect())
        self.ctx.set_cfg(CONFIG_PATH, saved)
        self.saved_value = saved
        self.ctx.say("✅ Custom detection settings saved")
        self.close()
        call_handler(self.on_saved, saved)
        return saved

    def _on_save(self, e: Any = None) -> dict:
        return self.save()

    def reset(self) -> None:
        """Back to the defaults on screen (Save still writes them)."""
        for key, slider in self.sliders.items():
            slider.value = int(self.defaults.get(key, slider.value))
            self.values[key] = int(slider.value)
            self.slider_labels[key].value = f"{int(slider.value)}%"
        self.consecutive.value = str(self.defaults.get("consecutive_chapters", 2))
        self.sample.value = str(self.defaults.get("sample_size", 3000))
        self.min_length.value = str(self.defaults.get("min_text_length", 500))
        self.check_all.value = bool(self.defaults.get("check_all_pairs", False))
        self.ctx.push(*self.sliders.values(), *self.slider_labels.values(), self.consecutive, self.sample,
                      self.min_length, self.check_all)
        self.ctx.say("ℹ️ Settings reset to defaults")

    def _on_reset(self, e: Any = None) -> None:
        dialog = ConfirmDialog(title="Reset to Defaults", body="Reset all values to default settings?",
                               confirm_label="Yes", cancel_label="No", on_confirm=self.reset)
        if self._page is not None:
            dialog.show(self._page)
        else:
            self.reset()
