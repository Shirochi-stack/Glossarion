# progress_screen.py
"""
ProgressScreen — Translation progress viewer for all output directories.
Scans the Glossarion Library for translation_progress.json files and
displays per-book progress with chapter-level status detail.
KivyMD 1.2.0 compatible.
"""

import os
import json
import time
import logging
import threading

from kivy.properties import ObjectProperty, ListProperty, StringProperty
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.metrics import dp, sp
from kivy.uix.boxlayout import BoxLayout

from kivymd.uix.screen import MDScreen
from kivymd.uix.card import MDCard
from kivymd.uix.button import MDFlatButton, MDRaisedButton, MDIconButton
from kivymd.uix.label import MDLabel
from kivymd.uix.toolbar import MDTopAppBar
from kivymd.uix.dialog import MDDialog
from kivymd.uix.list import OneLineListItem, TwoLineListItem, ThreeLineListItem
from kivymd.uix.progressbar import MDProgressBar
from kivymd.toast import toast

logger = logging.getLogger(__name__)

# Status display info (color + label)
STATUS_COLORS = {
    'completed':       (0.18, 0.80, 0.44, 1),   # green
    'completed_empty': (0.18, 0.80, 0.44, 0.7),  # green faded
    'merged':          (0.18, 0.80, 0.44, 0.7),
    'in_progress':     (1.00, 0.76, 0.03, 1),    # amber
    'pending':         (0.61, 0.61, 0.61, 1),     # grey
    'failed':          (0.91, 0.30, 0.24, 1),     # red
    'error':           (0.91, 0.30, 0.24, 1),
    'qa_failed':       (0.90, 0.49, 0.13, 1),     # orange
    'file_missing':    (0.91, 0.30, 0.24, 0.7),
    'not_translated':  (0.50, 0.50, 0.50, 1),     # dark grey
}

STATUS_LABELS = {
    'completed':       '✓ Done',
    'completed_empty': '✓ Empty',
    'merged':          '↗ Merged',
    'in_progress':     '⟳ Running',
    'pending':         '⏳ Pending',
    'failed':          '✗ Failed',
    'error':           '✗ Error',
    'qa_failed':       '⚠ QA Failed',
    'file_missing':    '⚠ Missing',
    'not_translated':  '—',
}

KV = '''
<ProgressScreen>:
    BoxLayout:
        orientation: 'vertical'

        MDTopAppBar:
            title: "Progress Manager"
            right_action_items: [["refresh", lambda x: root.refresh_all()]]
            md_bg_color: app.theme_cls.primary_color
            elevation: 2

        # Summary bar
        MDCard:
            size_hint: 1, None
            height: dp(56)
            padding: [dp(16), dp(8)]
            elevation: 1
            md_bg_color: app.theme_cls.bg_dark

            BoxLayout:
                spacing: dp(12)

                MDLabel:
                    id: summary_label
                    text: root.summary_text
                    font_style: "Subtitle2"
                    theme_text_color: "Secondary"

                MDIconButton:
                    icon: "folder-open-outline"
                    pos_hint: {"center_y": 0.5}
                    on_release: root.open_library_folder()

        ScrollView:
            id: progress_scroll

            BoxLayout:
                id: progress_list
                orientation: 'vertical'
                size_hint_y: None
                height: self.minimum_height
                padding: [dp(8), dp(8)]
                spacing: dp(8)

        MDLabel:
            id: empty_label
            text: "No translations found.\\nTranslate an EPUB to see progress here."
            halign: "center"
            theme_text_color: "Hint"
            font_style: "Body1"
            size_hint_y: None
            height: 0
            opacity: 0
'''


class ProgressScreen(MDScreen):
    """Translation progress viewer for all output directories."""

    app = ObjectProperty(None, allownone=True)
    summary_text = StringProperty("Scanning…")
    _book_data = ListProperty([])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Builder.load_string(KV)
        self._detail_dialog = None
        self._scanning = False

    def on_enter(self, *args):
        """Called each time the screen is displayed."""
        self.refresh_all()

    # ── Scanning ──

    def refresh_all(self):
        """Scan all output directories and refresh the display."""
        if self._scanning:
            return
        self._scanning = True
        self.summary_text = "Scanning…"
        threading.Thread(target=self._scan_worker, daemon=True).start()

    def _scan_worker(self):
        """Background worker to scan for progress files."""
        try:
            from android_file_utils import get_library_dir, get_documents_dir
            results = []

            # Collect all directories to scan
            scan_roots = [get_library_dir()]
            docs = get_documents_dir()
            if docs != scan_roots[0]:
                scan_roots.append(docs)

            # Also scan user-configured dirs
            if self.app:
                extra = self.app.config_data.get('library_scan_dirs', [])
                for d in extra:
                    if d not in scan_roots and os.path.isdir(d):
                        scan_roots.append(d)

            visited = set()
            for root_dir in scan_roots:
                if not os.path.isdir(root_dir):
                    continue
                try:
                    for entry in os.scandir(root_dir):
                        if not entry.is_dir():
                            continue
                        abs_path = os.path.abspath(entry.path)
                        if abs_path in visited:
                            continue
                        visited.add(abs_path)

                        progress_file = os.path.join(entry.path,
                                                      "translation_progress.json")
                        if not os.path.isfile(progress_file):
                            continue

                        info = self._parse_progress_file(progress_file,
                                                          entry.path, entry.name)
                        if info:
                            results.append(info)
                except PermissionError:
                    pass

            # Sort by last modified (newest first)
            results.sort(key=lambda r: r.get('last_modified', 0), reverse=True)

            Clock.schedule_once(lambda dt: self._on_scan_done(results), 0)
        except Exception as e:
            logger.error(f"Progress scan error: {e}")
            Clock.schedule_once(
                lambda dt: self._on_scan_done([]), 0)

    def _parse_progress_file(self, progress_path, output_dir, dir_name):
        """Parse a single translation_progress.json and return summary dict."""
        try:
            with open(progress_path, 'r', encoding='utf-8') as f:
                prog = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.debug(f"Could not read {progress_path}: {e}")
            return None

        chapters = prog.get('chapters', {})
        if not chapters:
            return None

        # Count statuses
        counts = {}
        last_updated = 0
        for ch_info in chapters.values():
            st = ch_info.get('status', 'unknown')
            counts[st] = counts.get(st, 0) + 1
            ts = ch_info.get('last_updated', 0)
            if ts and ts > last_updated:
                last_updated = ts

        total = len(chapters)
        completed = counts.get('completed', 0) + counts.get('completed_empty', 0)
        merged = counts.get('merged', 0)
        failed = counts.get('failed', 0) + counts.get('error', 0)
        qa_failed = counts.get('qa_failed', 0)
        in_progress = counts.get('in_progress', 0)
        pending = counts.get('pending', 0)

        # Derive display name from folder name (strip _output suffix)
        display_name = dir_name
        if display_name.endswith('_output'):
            display_name = display_name[:-7]
        # Shorten if too long
        if len(display_name) > 50:
            display_name = display_name[:47] + '…'

        return {
            'display_name': display_name,
            'output_dir': output_dir,
            'progress_path': progress_path,
            'total': total,
            'completed': completed,
            'merged': merged,
            'failed': failed,
            'qa_failed': qa_failed,
            'in_progress': in_progress,
            'pending': pending,
            'counts': counts,
            'progress_pct': int((completed + merged) / total * 100) if total else 0,
            'last_modified': last_updated,
            'chapters': chapters,
        }

    def _on_scan_done(self, results):
        """Called on main thread when scan completes."""
        self._scanning = False
        self._book_data = results

        total_books = len(results)
        total_complete = sum(1 for r in results if r['progress_pct'] == 100)
        total_chapters = sum(r['total'] for r in results)
        total_done = sum(r['completed'] + r['merged'] for r in results)

        self.summary_text = (
            f"{total_books} books · {total_done}/{total_chapters} chapters"
            f" · {total_complete} complete"
        )

        self._populate_list()

    # ── UI ──

    def _populate_list(self):
        """Rebuild the scrollable list of progress cards."""
        progress_list = self.ids.progress_list
        progress_list.clear_widgets()

        empty_label = self.ids.empty_label
        if not self._book_data:
            empty_label.opacity = 1
            empty_label.height = dp(200)
            return
        empty_label.opacity = 0
        empty_label.height = 0

        for info in self._book_data:
            card = self._build_book_card(info)
            progress_list.add_widget(card)

    def _build_book_card(self, info):
        """Build a single book progress card."""
        pct = info['progress_pct']
        total = info['total']
        completed = info['completed'] + info['merged']
        failed = info['failed'] + info['qa_failed']

        # Card
        card = MDCard(
            size_hint=(1, None),
            height=dp(120),
            padding=dp(12),
            elevation=2,
            ripple_behavior=True,
            on_release=lambda x, i=info: self._show_detail(i),
        )

        outer = BoxLayout(orientation='vertical', spacing=dp(4))

        # Row 1: title + percentage
        row1 = BoxLayout(size_hint_y=None, height=dp(28))
        title_label = MDLabel(
            text=info['display_name'],
            font_style="Subtitle1",
            shorten=True,
            shorten_from="right",
            size_hint_x=0.75,
        )
        pct_label = MDLabel(
            text=f"{pct}%",
            font_style="H6",
            halign="right",
            size_hint_x=0.25,
            theme_text_color="Custom",
            text_color=(0.18, 0.80, 0.44, 1) if pct == 100
                        else (1.0, 0.76, 0.03, 1) if pct > 0
                        else (0.61, 0.61, 0.61, 1),
        )
        row1.add_widget(title_label)
        row1.add_widget(pct_label)
        outer.add_widget(row1)

        # Row 2: progress bar
        bar = MDProgressBar(
            value=pct,
            max=100,
            size_hint_y=None,
            height=dp(6),
            color=(0.18, 0.80, 0.44, 1) if pct == 100
                   else (1.0, 0.76, 0.03, 1),
        )
        outer.add_widget(bar)

        # Row 3: status chips
        row3 = BoxLayout(
            size_hint_y=None,
            height=dp(28),
            spacing=dp(8),
            padding=[0, dp(4)],
        )

        chips = [
            (f"✓ {completed}", (0.18, 0.80, 0.44, 1)),
        ]
        if failed > 0:
            chips.append((f"✗ {failed}", (0.91, 0.30, 0.24, 1)))
        if info['qa_failed'] > 0:
            chips.append((f"⚠ {info['qa_failed']}", (0.90, 0.49, 0.13, 1)))
        if info['in_progress'] > 0:
            chips.append((f"⟳ {info['in_progress']}", (1.0, 0.76, 0.03, 1)))

        remaining = total - completed
        if remaining > 0 and info['in_progress'] == 0:
            chips.append((f"— {remaining}", (0.50, 0.50, 0.50, 1)))

        for chip_text, chip_color in chips:
            lbl = MDLabel(
                text=chip_text,
                font_style="Caption",
                theme_text_color="Custom",
                text_color=chip_color,
                size_hint_x=None,
                width=dp(60),
                halign="center",
            )
            row3.add_widget(lbl)
        row3.add_widget(BoxLayout())  # spacer
        outer.add_widget(row3)

        # Row 4: last updated timestamp
        if info['last_modified'] > 0:
            ts = time.strftime("%Y-%m-%d %H:%M",
                               time.localtime(info['last_modified']))
            ts_label = MDLabel(
                text=f"Last updated: {ts}",
                font_style="Caption",
                theme_text_color="Hint",
                size_hint_y=None,
                height=dp(18),
            )
            outer.add_widget(ts_label)

        card.add_widget(outer)
        return card

    # ── Detail dialog ──

    def _show_detail(self, info):
        """Show a dialog with chapter-level progress for one book."""
        chapters = info.get('chapters', {})
        if not chapters:
            toast("No chapter data available")
            return

        # Build sorted chapter list
        sorted_chs = self._sort_chapters(chapters)

        # Content
        content = BoxLayout(
            orientation='vertical',
            spacing=dp(2),
            size_hint_y=None,
        )

        # Summary row
        summary = MDLabel(
            text=(f"{info['completed']+info['merged']}/{info['total']} completed"
                  f" · {info['failed']+info['qa_failed']} failed"
                  f" · {info['progress_pct']}%"),
            font_style="Caption",
            theme_text_color="Secondary",
            size_hint_y=None,
            height=dp(28),
            padding=[dp(4), 0],
        )
        content.add_widget(summary)

        # Chapter entries
        for ch_key, ch_info in sorted_chs:
            row = self._build_chapter_row(ch_key, ch_info, info['output_dir'])
            content.add_widget(row)

        content.height = dp(28) + dp(44) * len(sorted_chs)

        from kivy.uix.scrollview import ScrollView
        scroll = ScrollView(
            size_hint=(1, None),
            height=min(dp(400), dp(28) + dp(44) * len(sorted_chs)),
            do_scroll_x=False,
        )
        scroll.add_widget(content)

        wrapper = BoxLayout(orientation='vertical', size_hint_y=None,
                            height=scroll.height)
        wrapper.add_widget(scroll)

        if self._detail_dialog:
            try:
                self._detail_dialog.dismiss()
            except Exception:
                pass

        self._detail_dialog = MDDialog(
            title=info['display_name'],
            type="custom",
            content_cls=wrapper,
            buttons=[
                MDFlatButton(
                    text="OPEN FOLDER",
                    on_release=lambda x, d=info['output_dir']:
                        self._open_folder(d),
                ),
                MDRaisedButton(
                    text="CLOSE",
                    on_release=lambda x: self._detail_dialog.dismiss(),
                ),
            ],
        )
        self._detail_dialog.open()

    def _sort_chapters(self, chapters):
        """Sort chapters by actual number, then key."""
        def _sort_key(item):
            ch_key, ch_info = item
            num = ch_info.get('actual_num', 0)
            if num is None:
                num = 0
            try:
                return (float(num), ch_key)
            except (ValueError, TypeError):
                return (0, ch_key)
        return sorted(chapters.items(), key=_sort_key)

    def _build_chapter_row(self, ch_key, ch_info, output_dir):
        """Build a single chapter status row."""
        status = ch_info.get('status', 'unknown')
        actual_num = ch_info.get('actual_num', ch_key)
        output_file = ch_info.get('output_file', '')
        color = STATUS_COLORS.get(status, (0.5, 0.5, 0.5, 1))
        label = STATUS_LABELS.get(status, status)

        # Check if output file actually exists
        file_exists = False
        if output_file:
            file_path = os.path.join(output_dir, output_file)
            file_exists = os.path.isfile(file_path)

        # If status is 'completed' but file doesn't exist, show warning
        if status in ('completed', 'completed_empty') and not file_exists and output_file:
            label = '⚠ Missing'
            color = STATUS_COLORS['file_missing']

        row = BoxLayout(
            size_hint_y=None,
            height=dp(42),
            spacing=dp(8),
            padding=[dp(8), dp(2)],
        )

        # Chapter number
        num_label = MDLabel(
            text=f"Ch {actual_num}",
            font_style="Body2",
            size_hint_x=0.2,
        )
        row.add_widget(num_label)

        # Output file name
        file_label = MDLabel(
            text=output_file if output_file else "—",
            font_style="Caption",
            theme_text_color="Secondary",
            shorten=True,
            shorten_from="center",
            size_hint_x=0.5,
        )
        row.add_widget(file_label)

        # Status badge
        status_label = MDLabel(
            text=label,
            font_style="Caption",
            theme_text_color="Custom",
            text_color=color,
            halign="right",
            size_hint_x=0.3,
        )
        row.add_widget(status_label)

        return row

    # ── Actions ──

    def open_library_folder(self):
        """Open the library directory in the system file manager."""
        try:
            from android_file_utils import get_library_dir, is_android
            lib_dir = get_library_dir()
            if is_android():
                try:
                    from jnius import autoclass
                    Intent = autoclass('android.content.Intent')
                    Uri = autoclass('android.net.Uri')
                    PythonActivity = autoclass('org.kivy.android.PythonActivity')
                    intent = Intent(Intent.ACTION_VIEW)
                    intent.setDataAndType(Uri.parse(f"file://{lib_dir}"),
                                          "resource/folder")
                    intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                    PythonActivity.mActivity.startActivity(intent)
                except Exception:
                    toast(f"Library: {lib_dir}")
            else:
                import subprocess, sys
                if sys.platform == 'win32':
                    os.startfile(lib_dir)
                elif sys.platform == 'darwin':
                    subprocess.Popen(['open', lib_dir])
                else:
                    subprocess.Popen(['xdg-open', lib_dir])
        except Exception as e:
            toast(f"Could not open folder: {e}")

    def _open_folder(self, path):
        """Open a specific output folder."""
        try:
            from android_file_utils import is_android
            if is_android():
                toast(f"Output: {path}")
            else:
                import subprocess, sys
                if sys.platform == 'win32':
                    os.startfile(path)
                elif sys.platform == 'darwin':
                    subprocess.Popen(['open', path])
                else:
                    subprocess.Popen(['xdg-open', path])
        except Exception as e:
            toast(f"Could not open folder: {e}")
