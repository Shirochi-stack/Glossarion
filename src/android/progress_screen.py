# progress_screen.py
"""
Progress manager for Android.

Shows both:
1) Translation progress (translation_progress.json)
2) Glossary extraction progress (*_glossary_progress.json / glossary_progress.json)
"""

import json
import logging
import os
import threading
import time

from kivy.clock import Clock
from kivy.lang import Builder
from kivy.metrics import dp
from kivy.properties import ListProperty, ObjectProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout

from kivymd.toast import toast
from kivymd.uix.button import MDFlatButton, MDRaisedButton
from kivymd.uix.card import MDCard
from kivymd.uix.dialog import MDDialog
from kivymd.uix.label import MDLabel
from kivymd.uix.progressbar import MDProgressBar
from kivymd.uix.screen import MDScreen

logger = logging.getLogger(__name__)

TRANSLATION_STATUS_COLORS = {
    "completed": (0.18, 0.80, 0.44, 1),
    "completed_empty": (0.18, 0.80, 0.44, 0.7),
    "merged": (0.18, 0.80, 0.44, 0.7),
    "in_progress": (1.00, 0.76, 0.03, 1),
    "pending": (0.61, 0.61, 0.61, 1),
    "failed": (0.91, 0.30, 0.24, 1),
    "error": (0.91, 0.30, 0.24, 1),
    "qa_failed": (0.90, 0.49, 0.13, 1),
    "file_missing": (0.91, 0.30, 0.24, 0.7),
}

KV = """
<ProgressScreen>:
    BoxLayout:
        orientation: "vertical"

        MDTopAppBar:
            title: "Progress Manager"
            right_action_items: [["refresh", lambda x: root.refresh_all()]]
            md_bg_color: app.theme_cls.primary_color
            elevation: 2

        MDCard:
            size_hint: 1, None
            height: dp(60)
            padding: [dp(16), dp(8)]
            elevation: 1
            md_bg_color: app.theme_cls.bg_dark

            BoxLayout:
                spacing: dp(8)

                MDLabel:
                    id: summary_label
                    text: root.summary_text
                    font_style: "Subtitle2"
                    theme_text_color: "Secondary"

                MDRaisedButton:
                    text: "Library"
                    size_hint_x: 0.25
                    on_release: root.open_library_folder()

        ScrollView:
            id: progress_scroll

            BoxLayout:
                id: progress_list
                orientation: "vertical"
                size_hint_y: None
                height: self.minimum_height
                padding: [dp(8), dp(8)]
                spacing: dp(8)

        MDLabel:
            id: empty_label
            text: "No progress files found yet."
            halign: "center"
            theme_text_color: "Hint"
            font_style: "Body1"
            size_hint_y: None
            height: 0
            opacity: 0
"""


class ProgressScreen(MDScreen):
    """Unified progress manager screen."""

    app = ObjectProperty(None, allownone=True)
    summary_text = StringProperty("Scanning...")
    _entries = ListProperty([])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Builder.load_string(KV)
        self._detail_dialog = None
        self._scanning = False

    def on_enter(self, *args):
        self.refresh_all()

    def refresh_all(self):
        if self._scanning:
            return
        self._scanning = True
        self.summary_text = "Scanning..."
        threading.Thread(target=self._scan_worker, daemon=True).start()

    def _scan_worker(self):
        try:
            roots = self._get_scan_roots()
            translation_items = self._scan_translation_progress(roots)
            glossary_items = self._scan_glossary_progress(roots)

            results = translation_items + glossary_items
            results.sort(key=lambda r: r.get("last_modified", 0), reverse=True)
            Clock.schedule_once(lambda dt: self._on_scan_done(results), 0)
        except Exception as exc:
            logger.error("Progress scan error: %s", exc)
            Clock.schedule_once(lambda dt: self._on_scan_done([]), 0)

    def _get_scan_roots(self):
        from android_file_utils import get_documents_dir, get_library_dir

        roots = [get_library_dir()]
        docs = get_documents_dir()
        if docs not in roots:
            roots.append(docs)
        if self.app:
            for d in self.app.config_data.get("library_scan_dirs", []):
                if d and os.path.isdir(d) and d not in roots:
                    roots.append(d)
        return roots

    def _scan_translation_progress(self, roots):
        results = []
        visited_dirs = set()
        for root_dir in roots:
            if not os.path.isdir(root_dir):
                continue
            try:
                for entry in os.scandir(root_dir):
                    if not entry.is_dir():
                        continue
                    abs_dir = os.path.abspath(entry.path)
                    if abs_dir in visited_dirs:
                        continue
                    visited_dirs.add(abs_dir)

                    progress_path = os.path.join(entry.path, "translation_progress.json")
                    if not os.path.isfile(progress_path):
                        continue
                    info = self._parse_translation_progress(progress_path, entry.path, entry.name)
                    if info:
                        results.append(info)
            except Exception:
                continue
        return results

    def _scan_glossary_progress(self, roots):
        results = []
        seen_files = set()
        for root_dir in roots:
            if not os.path.isdir(root_dir):
                continue
            for current_dir, _, files in os.walk(root_dir):
                for filename in files:
                    lower = filename.lower()
                    if not (lower.endswith("_glossary_progress.json") or lower == "glossary_progress.json"):
                        continue
                    progress_path = os.path.abspath(os.path.join(current_dir, filename))
                    if progress_path in seen_files:
                        continue
                    seen_files.add(progress_path)
                    info = self._parse_glossary_progress(progress_path)
                    if info:
                        results.append(info)
        return results

    def _parse_translation_progress(self, progress_path, output_dir, dir_name):
        try:
            with open(progress_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            return None

        chapters = data.get("chapters", {})
        if not chapters:
            return None

        counts = {}
        last_updated = 0
        for ch_info in chapters.values():
            status = ch_info.get("status", "unknown")
            counts[status] = counts.get(status, 0) + 1
            ts = ch_info.get("last_updated", 0)
            if ts and ts > last_updated:
                last_updated = ts

        total = len(chapters)
        completed = counts.get("completed", 0) + counts.get("completed_empty", 0)
        merged = counts.get("merged", 0)
        failed = counts.get("failed", 0) + counts.get("error", 0)
        qa_failed = counts.get("qa_failed", 0)
        in_progress = counts.get("in_progress", 0)
        pending = counts.get("pending", 0)

        display_name = dir_name[:-7] if dir_name.endswith("_output") else dir_name
        if len(display_name) > 64:
            display_name = display_name[:61] + "..."

        done = completed + merged
        pct = int(done / total * 100) if total else 0

        return {
            "kind": "translation",
            "display_name": display_name,
            "progress_path": progress_path,
            "output_dir": output_dir,
            "total": total,
            "completed": completed,
            "merged": merged,
            "failed": failed,
            "qa_failed": qa_failed,
            "in_progress": in_progress,
            "pending": pending,
            "progress_pct": pct,
            "last_modified": max(last_updated, os.path.getmtime(progress_path)),
            "chapters": chapters,
        }

    def _parse_glossary_progress(self, progress_path):
        try:
            with open(progress_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            return None

        completed = self._to_int_set(data.get("completed", []))
        merged = self._to_int_set(data.get("merged_indices", []))
        failed = self._to_int_set(data.get("failed", []))
        done = completed | merged

        reported_total = data.get("total_chapters")
        total = 0
        if isinstance(reported_total, int) and reported_total > 0:
            total = reported_total
        elif completed or merged or failed:
            total = max(completed | merged | failed) + 1

        pct = int(len(done) / total * 100) if total else 0
        display_name = self._derive_glossary_name(progress_path)
        book_title = str(data.get("book_title", "") or "")

        return {
            "kind": "glossary",
            "display_name": display_name,
            "book_title": book_title,
            "progress_path": progress_path,
            "output_dir": os.path.dirname(progress_path),
            "total": total,
            "completed_indices": completed,
            "merged_indices": merged,
            "failed_indices": failed,
            "completed": max(0, len(done - merged)),
            "merged": len(merged),
            "failed": len(failed),
            "in_progress": 0,
            "pending": max(0, total - len(done | failed)),
            "progress_pct": pct,
            "last_modified": os.path.getmtime(progress_path),
        }

    @staticmethod
    def _to_int_set(values):
        out = set()
        if not isinstance(values, list):
            return out
        for value in values:
            try:
                out.add(int(value))
            except Exception:
                continue
        return out

    @staticmethod
    def _derive_glossary_name(progress_path):
        filename = os.path.basename(progress_path)
        lower = filename.lower()
        if lower.endswith("_glossary_progress.json"):
            return filename[:-len("_glossary_progress.json")]

        parent = os.path.basename(os.path.dirname(progress_path))
        if parent.lower() == "glossary":
            grandparent = os.path.basename(os.path.dirname(os.path.dirname(progress_path)))
            return grandparent or parent
        return parent or "glossary"

    def _on_scan_done(self, results):
        self._scanning = False
        self._entries = results

        translation = [r for r in results if r.get("kind") == "translation"]
        glossary = [r for r in results if r.get("kind") == "glossary"]
        t_total = sum(r.get("total", 0) for r in translation)
        t_done = sum(r.get("completed", 0) + r.get("merged", 0) for r in translation)
        g_total = sum(r.get("total", 0) for r in glossary)
        g_done = sum(r.get("completed", 0) + r.get("merged", 0) for r in glossary)

        self.summary_text = (
            f"{len(translation)} translation | {len(glossary)} glossary | "
            f"T {t_done}/{t_total} | G {g_done}/{g_total}"
        )

        self._populate_list()

    def _populate_list(self):
        progress_list = self.ids.progress_list
        progress_list.clear_widgets()

        if not self._entries:
            self.ids.empty_label.opacity = 1
            self.ids.empty_label.height = dp(180)
            return

        self.ids.empty_label.opacity = 0
        self.ids.empty_label.height = 0

        for info in self._entries:
            progress_list.add_widget(self._build_card(info))

    def _build_card(self, info):
        pct = info.get("progress_pct", 0)
        completed = info.get("completed", 0) + info.get("merged", 0)
        failed = info.get("failed", 0) + info.get("qa_failed", 0)
        total = info.get("total", 0)
        kind = info.get("kind", "translation")
        kind_label = "Glossary" if kind == "glossary" else "Translation"

        card = MDCard(
            size_hint=(1, None),
            height=dp(126),
            padding=dp(12),
            elevation=2,
            ripple_behavior=True,
            on_release=lambda _btn, i=info: self._show_detail(i),
        )

        outer = BoxLayout(orientation="vertical", spacing=dp(5))

        row1 = BoxLayout(size_hint_y=None, height=dp(26), spacing=dp(8))
        row1.add_widget(MDLabel(text=info.get("display_name", "unknown"), size_hint_x=0.62, shorten=True, shorten_from="right"))
        row1.add_widget(MDLabel(text=kind_label, size_hint_x=0.2, theme_text_color="Secondary", halign="center"))
        row1.add_widget(MDLabel(text=f"{pct}%", size_hint_x=0.18, halign="right"))
        outer.add_widget(row1)

        bar = MDProgressBar(
            value=pct,
            max=100,
            size_hint_y=None,
            height=dp(6),
            color=(0.18, 0.80, 0.44, 1) if pct == 100 else (1.0, 0.76, 0.03, 1),
        )
        outer.add_widget(bar)

        row2 = BoxLayout(size_hint_y=None, height=dp(24), spacing=dp(8))
        row2.add_widget(MDLabel(text=f"Done {completed}/{total}", size_hint_x=0.45, theme_text_color="Secondary"))
        if failed > 0:
            row2.add_widget(MDLabel(text=f"Failed {failed}", size_hint_x=0.25, theme_text_color="Error"))
        if info.get("in_progress", 0) > 0:
            row2.add_widget(MDLabel(text=f"Running {info.get('in_progress', 0)}", size_hint_x=0.3, theme_text_color="Custom", text_color=(1.0, 0.76, 0.03, 1)))
        else:
            row2.add_widget(MDLabel(text="", size_hint_x=0.3))
        outer.add_widget(row2)

        if kind == "glossary" and info.get("book_title"):
            outer.add_widget(MDLabel(
                text=f"Book: {info.get('book_title')}",
                font_style="Caption",
                theme_text_color="Hint",
                size_hint_y=None,
                height=dp(18),
                shorten=True,
                shorten_from="right",
            ))
        elif info.get("last_modified", 0) > 0:
            ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(info["last_modified"]))
            outer.add_widget(MDLabel(
                text=f"Updated: {ts}",
                font_style="Caption",
                theme_text_color="Hint",
                size_hint_y=None,
                height=dp(18),
            ))

        card.add_widget(outer)
        return card

    def _show_detail(self, info):
        if info.get("kind") == "glossary":
            self._show_glossary_detail(info)
        else:
            self._show_translation_detail(info)

    def _show_translation_detail(self, info):
        chapters = info.get("chapters", {})
        if not chapters:
            toast("No chapter data available")
            return

        sorted_chapters = self._sort_translation_chapters(chapters)
        content = BoxLayout(orientation="vertical", spacing=dp(2), size_hint_y=None)
        content.add_widget(MDLabel(
            text=f"{info.get('completed', 0)+info.get('merged', 0)}/{info.get('total', 0)} done | {info.get('failed', 0)+info.get('qa_failed', 0)} failed | {info.get('progress_pct', 0)}%",
            font_style="Caption",
            theme_text_color="Secondary",
            size_hint_y=None,
            height=dp(28),
        ))
        for ch_key, ch_info in sorted_chapters:
            content.add_widget(self._build_translation_row(ch_key, ch_info, info.get("output_dir", "")))
        content.height = dp(28) + dp(42) * len(sorted_chapters)
        self._open_detail_dialog(info.get("display_name", "Translation"), content, info.get("output_dir", ""))

    def _show_glossary_detail(self, info):
        completed = info.get("completed_indices", set())
        merged = info.get("merged_indices", set())
        failed = info.get("failed_indices", set())
        total = int(info.get("total", 0) or 0)

        if total <= 0:
            known = sorted(completed | merged | failed)
            total = (known[-1] + 1) if known else 0
        if total <= 0:
            toast("No glossary chapter progress available")
            return

        content = BoxLayout(orientation="vertical", spacing=dp(2), size_hint_y=None)
        done = len((completed | merged))
        content.add_widget(MDLabel(
            text=f"{done}/{total} done | failed {len(failed)} | {info.get('progress_pct', 0)}%",
            font_style="Caption",
            theme_text_color="Secondary",
            size_hint_y=None,
            height=dp(28),
        ))

        for idx in range(total):
            if idx in failed:
                label = "Failed"
                color = (0.91, 0.30, 0.24, 1)
            elif idx in merged:
                label = "Merged"
                color = (0.20, 0.64, 0.80, 1)
            elif idx in completed:
                label = "Completed"
                color = (0.18, 0.80, 0.44, 1)
            else:
                label = "Pending"
                color = (0.61, 0.61, 0.61, 1)

            row = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(8), padding=[dp(8), dp(2)])
            row.add_widget(MDLabel(text=f"Ch {idx + 1}", size_hint_x=0.28))
            row.add_widget(MDLabel(text=label, size_hint_x=0.72, halign="right", theme_text_color="Custom", text_color=color))
            content.add_widget(row)

        content.height = dp(28) + dp(40) * total
        self._open_detail_dialog(info.get("display_name", "Glossary"), content, info.get("output_dir", ""))

    def _open_detail_dialog(self, title, content_widget, folder_path):
        from kivy.uix.scrollview import ScrollView

        scroll = ScrollView(size_hint=(1, None), height=min(dp(430), max(dp(180), content_widget.height)), do_scroll_x=False)
        scroll.add_widget(content_widget)
        wrapper = BoxLayout(orientation="vertical", size_hint_y=None, height=scroll.height)
        wrapper.add_widget(scroll)

        if self._detail_dialog:
            try:
                self._detail_dialog.dismiss()
            except Exception:
                pass

        self._detail_dialog = MDDialog(
            title=title,
            type="custom",
            content_cls=wrapper,
            buttons=[
                MDFlatButton(text="OPEN FOLDER", on_release=lambda _b, p=folder_path: self._open_folder(p)),
                MDRaisedButton(text="CLOSE", on_release=lambda _b: self._detail_dialog.dismiss()),
            ],
        )
        self._detail_dialog.open()

    @staticmethod
    def _sort_translation_chapters(chapters):
        def _sort_key(item):
            key, info = item
            num = info.get("actual_num", 0)
            try:
                return (float(num), key)
            except Exception:
                return (0, key)
        return sorted(chapters.items(), key=_sort_key)

    def _build_translation_row(self, ch_key, ch_info, output_dir):
        status = ch_info.get("status", "unknown")
        actual_num = ch_info.get("actual_num", ch_key)
        output_file = ch_info.get("output_file", "")
        label = status.replace("_", " ").title()
        color = TRANSLATION_STATUS_COLORS.get(status, (0.5, 0.5, 0.5, 1))

        if output_file and status in ("completed", "completed_empty"):
            path = os.path.join(output_dir, output_file)
            if not os.path.isfile(path):
                label = "Missing Output"
                color = TRANSLATION_STATUS_COLORS["file_missing"]

        row = BoxLayout(size_hint_y=None, height=dp(42), spacing=dp(8), padding=[dp(8), dp(2)])
        row.add_widget(MDLabel(text=f"Ch {actual_num}", size_hint_x=0.2))
        row.add_widget(MDLabel(text=output_file if output_file else "-", size_hint_x=0.52, theme_text_color="Secondary", shorten=True, shorten_from="center"))
        row.add_widget(MDLabel(text=label, size_hint_x=0.28, halign="right", theme_text_color="Custom", text_color=color))
        return row

    def open_library_folder(self):
        try:
            from android_file_utils import get_library_dir
            self._open_folder(get_library_dir())
        except Exception as exc:
            toast(f"Could not open library: {exc}")

    def _open_folder(self, path):
        if not path:
            return
        try:
            from android_file_utils import is_android
            if is_android():
                toast(f"Folder: {path}")
                return
            import subprocess
            import sys

            if sys.platform == "win32":
                os.startfile(path)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as exc:
            toast(f"Could not open folder: {exc}")
