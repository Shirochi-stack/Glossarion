"""
Retranslation GUI Module
Force retranslation functionality for EPUB, text, and image files
"""

import os
import sys
import json
import re
import copy
from PySide6.QtWidgets import (QWidget, QDialog, QLabel, QFrame, QListWidget, 
                                QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout,
                                QMessageBox, QFileDialog, QTabWidget, QListWidgetItem,
                                QScrollArea, QSizePolicy, QMenu, QAbstractItemView)
from PySide6.QtCore import Qt, Signal, QTimer, QPropertyAnimation, QEasingCurve, Property, QEventLoop, QUrl, QItemSelectionModel, QSize
from PySide6.QtGui import QFont, QColor, QTransform, QIcon, QPixmap, QDesktopServices
import xml.etree.ElementTree as ET
import zipfile
import shutil
import traceback
import subprocess
import platform
import time

_IS_MACOS = (sys.platform == 'darwin')

def _get_app_dir() -> str:
    """Return the application's base directory (Windows-safe)."""
    if platform.system() == 'Windows':
        if getattr(sys, 'frozen', False):
            return os.path.dirname(sys.executable)
        return os.path.dirname(os.path.abspath(__file__))
    return os.getcwd()


# WindowManager and UIHelper removed - not needed in PySide6
# Qt handles window management and UI utilities automatically


class AnimatedRefreshButton(QPushButton):
    """Custom QPushButton with rotation animation for refresh action using Halgakos.ico"""
    
    def __init__(self, text="Refresh", parent=None):
        super().__init__(text, parent)
        self._rotation = 0
        self._animation = None
        self._original_text = text
        self._timer = None
        self._animation_step = 0
        self._original_icon = None
        
        # Try to load Halgakos.ico
        try:
            # Get base directory
            base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
            ico_path = os.path.join(base_dir, 'Halgakos.ico')
            if os.path.isfile(ico_path):
                self._original_icon = QIcon(ico_path)
                self.setIcon(self._original_icon)
                self.setIconSize(self.iconSize() * 1.2)  # Make icon slightly larger
        except Exception as e:
            print(f"Could not load Halgakos.ico for refresh button: {e}")
        
    def get_rotation(self):
        return self._rotation
    
    def set_rotation(self, angle):
        self._rotation = angle
        self.update()  # Trigger repaint
    
    # Define rotation as a Qt Property for animation
    rotation = Property(float, get_rotation, set_rotation)
    
    def start_animation(self):
        """Start the spinning animation"""
        if self._timer and self._timer.isActive():
            return  # Already animating
        
        self.setProperty("refreshActive", True)
        self.style().unpolish(self)
        self.style().polish(self)
        
        # Start timer-based animation for icon rotation
        self._animation_step = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_animation_frame)
        self._timer.start(50)  # Update every 50ms for smooth rotation
    
    def _update_animation_frame(self):
        """Update animation frame by rotating the icon"""
        if self._original_icon:
            # Increment rotation angle (30 degrees per frame for smooth spinning)
            self._rotation = (self._rotation + 30) % 360
            
            # Create a rotated version of the icon
            pixmap = self._original_icon.pixmap(self.iconSize())
            transform = QTransform().rotate(self._rotation)
            rotated_pixmap = pixmap.transformed(transform, Qt.SmoothTransformation)
            
            # Set the rotated icon
            self.setIcon(QIcon(rotated_pixmap))
    
    def stop_animation(self):
        """Stop the spinning animation"""
        if self._timer:
            self._timer.stop()
            self._timer = None
            self._rotation = 0
            self._animation_step = 0
            
            # Restore original icon (unrotated)
            if self._original_icon:
                self.setIcon(self._original_icon)
            
            self.setProperty("refreshActive", False)
            self.style().unpolish(self)
            self.style().polish(self)
            
            self.update()


class RetranslationMixin:
    """Mixin class containing retranslation methods for TranslatorGUI"""

    _RETRANSLATION_SHOW_MODEL_INFO_CONFIG_KEY = "retranslation_show_model_info"

    def _get_retranslation_show_model_info_state(self, file_path=None):
        """Return the persisted Show Model Info preference, with live dialog cache first."""
        try:
            if file_path:
                file_key = os.path.abspath(file_path)
                cached = getattr(self, '_retranslation_dialog_cache', {}).get(file_key, {})
                if isinstance(cached, dict) and 'show_model_info_state' in cached:
                    return bool(cached.get('show_model_info_state'))
        except Exception:
            pass
        try:
            return bool(getattr(self, 'config', {}).get(self._RETRANSLATION_SHOW_MODEL_INFO_CONFIG_KEY, False))
        except Exception:
            return False

    def _persist_retranslation_show_model_info_state(self, enabled):
        """Persist the Show Model Info preference across app sessions."""
        try:
            if not hasattr(self, 'config') or not isinstance(self.config, dict):
                self.config = {}
            self.config[self._RETRANSLATION_SHOW_MODEL_INFO_CONFIG_KEY] = bool(enabled)
            if hasattr(self, 'save_config') and callable(self.save_config):
                self.save_config(show_message=False)
        except Exception as exc:
            try:
                print(f"⚠️ Could not persist Show Model Info state: {exc}")
            except Exception:
                pass

    def _progress_file_is_skipped_special(self, filename, fallback_is_special=False):
        """Return True only for special files that translation would skip."""
        translate_special = bool(
            getattr(self, 'translate_special_files_var', False)
            or getattr(self, 'config', {}).get('translate_special_files', False)
        )
        if hasattr(self, '_should_skip_special_file'):
            return self._should_skip_special_file(filename, translate_special)
        if translate_special:
            return False
        is_special = fallback_is_special
        if filename and hasattr(self, '_is_special_file'):
            is_special = self._is_special_file(filename)
        if not is_special:
            return False
        translate_all_numbered = bool(
            getattr(self, 'translate_all_numbered_html_var', True)
            or getattr(self, 'config', {}).get('translate_all_numbered_html', True)
        )
        if translate_all_numbered:
            stem = os.path.splitext(os.path.basename(str(filename or '')))[0]
            if stem.lower().startswith('response_'):
                stem = stem[len('response_'):]
            if re.search(r'\d', stem):
                return False
        return True

    def _apply_compact_inline_list_style(self, listbox, font=None, extra_row_px=0):
        """Use dense row spacing for inline status/list views."""
        try:
            if font is not None:
                listbox.setFont(font)
            listbox.setProperty("_compact_inline_extra_row_px", max(0, int(extra_row_px or 0)))
            listbox.setSpacing(0)
            listbox.setUniformItemSizes(True)
            listbox.setStyleSheet("""
                QListWidget {
                    outline: 0;
                }
                QListWidget::item {
                    margin: 0px;
                    padding: 0px 2px;
                }
            """)
        except Exception:
            pass

    def _set_compact_inline_item_size(self, listbox, item):
        try:
            extra_row_px = int(listbox.property("_compact_inline_extra_row_px") or 0)
            height = max(18, listbox.fontMetrics().lineSpacing() + 2) + extra_row_px
            item.setSizeHint(QSize(0, height))
        except Exception:
            pass
        return item

    def _add_compact_inline_list_item(self, listbox, item_or_text):
        item = item_or_text if isinstance(item_or_text, QListWidgetItem) else QListWidgetItem(str(item_or_text))
        self._set_compact_inline_item_size(listbox, item)
        listbox.addItem(item)
        return item
    
    def _ui_yield(self, ms=5):
        """Let the Qt event loop process pending events briefly."""
        try:
            if getattr(self, '_suspend_yield', False):
                return
            from PySide6.QtWidgets import QApplication
            QApplication.processEvents(QEventLoop.AllEvents, ms)
        except Exception:
            pass
    
    def _clear_layout(self, layout):
        """Safely clear all items from a layout"""
        if layout is None:
            return
        while layout.count():
            item = layout.takeAt(0)
            if item:
                widget = item.widget()
                if widget:
                    widget.setParent(None)
                    widget.deleteLater()
                elif item.layout():
                    self._clear_layout(item.layout())
    
    def _get_dialog_size(self, width_ratio=0.5, height_ratio=0.5):
        """Calculate dialog size as a ratio of screen size (default 50% width, 50% height)"""
        try:
            from PySide6.QtWidgets import QApplication
            from PySide6.QtGui import QScreen
            
            # Get primary screen
            screen = QApplication.primaryScreen()
            if screen:
                geometry = screen.availableGeometry()
                width = int(geometry.width() * width_ratio)
                height = int(geometry.height() * height_ratio)
                return width, height
        except:
            pass
        
        # Fallback to reasonable defaults if screen info unavailable
        return int(1920 * width_ratio), int(1080 * height_ratio)
    
    def _show_message(self, msg_type, title, message, parent=None):
        """Show message using PySide6 QMessageBox with Halgakos icon"""
        try:
            # Create message box
            msg_box = QMessageBox(parent)
            msg_box.setWindowTitle(title)
            msg_box.setText(message)
            
            # Set icon based on message type
            if msg_type == 'info':
                msg_box.setIcon(QMessageBox.Information)
            elif msg_type == 'warning':
                msg_box.setIcon(QMessageBox.Warning)
            elif msg_type == 'error':
                msg_box.setIcon(QMessageBox.Critical)
            elif msg_type == 'question':
                msg_box.setIcon(QMessageBox.Question)
                msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            
            # Center buttons
            msg_box.setStyleSheet("""
                QPushButton {
                    min-width: 80px;
                    min-height: 30px;
                    padding: 6px 20px;
                    font-size: 10pt;
                }
                QDialogButtonBox {
                    qproperty-centerButtons: true;
                }
            """)
            
            # Try to set Halgakos window icon
            try:
                from PySide6.QtGui import QIcon
                if hasattr(self, 'base_dir'):
                    base_dir = self.base_dir
                else:
                    import sys
                    base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
                ico_path = os.path.join(base_dir, 'Halgakos.ico')
                if os.path.isfile(ico_path):
                    msg_box.setWindowIcon(QIcon(ico_path))
            except:
                pass
            
            # Show message box
            if msg_type == 'question':
                # Ensure dialog stays on top if it's a critical question
                msg_box.setWindowFlags(msg_box.windowFlags() | Qt.WindowStaysOnTopHint)
                return msg_box.exec() == QMessageBox.Yes
            else:
                msg_box.setWindowFlags(msg_box.windowFlags() | Qt.WindowStaysOnTopHint)
                msg_box.exec()
                return True
                
        except Exception as e:
            # Fallback to console if dialog fails
            print(f"{title}: {message}")
            if msg_type == 'question':
                return False
            return False

    @staticmethod
    def _styled_msgbox(icon, parent, title, message, buttons=None):
        """Create a QMessageBox with centered buttons and return the result.
        
        Usage (replaces static convenience methods):
            QMessageBox.information(p, t, m)  →  self._styled_msgbox(QMessageBox.Information, p, t, m)
            QMessageBox.warning(p, t, m)      →  self._styled_msgbox(QMessageBox.Warning, p, t, m)
            QMessageBox.question(p, t, m, b)  →  self._styled_msgbox(QMessageBox.Question, p, t, m, b)
            QMessageBox.critical(p, t, m)     →  self._styled_msgbox(QMessageBox.Critical, p, t, m)
        """
        msg = QMessageBox(parent)
        msg.setIcon(icon)
        msg.setWindowTitle(title)
        msg.setText(message)
        if buttons is not None:
            msg.setStandardButtons(buttons)
        msg.setStyleSheet("""
            QPushButton {
                min-width: 80px;
                min-height: 30px;
                padding: 6px 20px;
                font-size: 10pt;
            }
            QDialogButtonBox {
                qproperty-centerButtons: true;
            }
        """)
        result = msg.exec()
        return result
 
    def _flash_pm_button_green(self, folder_path=None):
        """Flash the Progress Manager button green to indicate a new folder was created.
        Also plays a Windows sound and stores the folder path for the dialog status row."""
        try:
            pm_btn = getattr(self, 'pm_button', None)
            if pm_btn is None:
                return

            # Remember the definitive original style (first call wins)
            # This prevents re-capturing an already-green style on rapid re-calls
            if not hasattr(self, '_pm_original_style') or not self._pm_original_style:
                self._pm_original_style = pm_btn.styleSheet()

            # Flash to green using the definitive original as base
            import re as _re
            green_style = _re.sub(
                r'background-color:\s*#[0-9a-fA-F]+',
                'background-color: #27ae60',
                self._pm_original_style,
                count=1
            )
            pm_btn.setStyleSheet(green_style)

            # Restore using the definitive original style after 1.5 seconds
            def _restore_pm_style():
                try:
                    pm_btn.setStyleSheet(self._pm_original_style)
                except Exception:
                    pass
            QTimer.singleShot(1500, _restore_pm_style)

            # Play Windows system sound
            try:
                import platform
                if platform.system() == 'Windows':
                    import winsound
                    winsound.MessageBeep(winsound.MB_OK)
            except Exception:
                pass

            # Store the created folder path so the dialog stats row can show it
            if folder_path:
                self._pm_created_folder = folder_path
        except Exception as e:
            print(f"⚠️ Could not flash PM button: {e}")

    def _create_retranslation_shell_dialog(self, title="Progress Manager", width_ratio=0.38, height_ratio=0.4):
        from PySide6.QtWidgets import QApplication
        if not QApplication.instance():
            QApplication(sys.argv)

        parent_widget = self if isinstance(self, QWidget) else None
        dialog = QDialog(parent_widget)
        dialog.setWindowTitle(title)
        dialog.setWindowModality(Qt.NonModal)
        width, height = self._get_dialog_size(width_ratio, height_ratio)
        dialog.resize(width, height)
        dialog.setMinimumSize(width, height)

        try:
            if parent_widget is not None:
                ss = parent_widget.styleSheet()
                if ss:
                    dialog.setStyleSheet(ss)
        except Exception:
            pass

        base_dir = None
        ico_path = None
        try:
            if hasattr(self, 'base_dir'):
                base_dir = self.base_dir
            else:
                base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
            ico_path = os.path.join(base_dir, 'Halgakos.ico')
            if os.path.isfile(ico_path):
                dialog.setWindowIcon(QIcon(ico_path))
        except Exception as e:
            print(f"Failed to load icon: {e}")

        dialog_layout = QVBoxLayout(dialog)
        loading_widget = QWidget(dialog)
        loading_layout = QVBoxLayout(loading_widget)
        loading_layout.setContentsMargins(0, 0, 0, 0)
        loading_layout.setSpacing(10)
        loading_layout.addStretch(1)

        try:
            from spinning import animate_icon, create_icon_label
            loading_icon = create_icon_label(52, base_dir)
            loading_icon.setFixedSize(52, 52)
            loading_layout.addWidget(loading_icon, 0, Qt.AlignCenter)

            spin_timer = QTimer(dialog)

            def _spin_loading_icon():
                try:
                    animate_icon(loading_icon)
                except RuntimeError:
                    spin_timer.stop()

            spin_timer.timeout.connect(_spin_loading_icon)
            spin_timer.start(540)
            QTimer.singleShot(0, _spin_loading_icon)
            dialog._loading_icon_timer = spin_timer
        except Exception:
            pass

        loading_label = QLabel("Loading progress...")
        loading_label.setAlignment(Qt.AlignCenter)
        loading_label.setStyleSheet("color: #94a3b8; font-size: 12pt; font-weight: bold; padding: 24px;")
        loading_layout.addWidget(loading_label)
        loading_layout.addStretch(1)
        dialog_layout.addWidget(loading_widget)
        return dialog, dialog_layout, loading_widget, loading_label

    def _show_retranslation_shell_then_build(self, file_path, show_special_files_state=False):
        dialog, dialog_layout, loading_widget, loading_label = self._create_retranslation_shell_dialog("Progress Manager")
        file_key = os.path.abspath(file_path)

        def closeEvent(event):
            event.ignore()
            dialog.hide()

        dialog.closeEvent = closeEvent
        dialog.show()
        try:
            from PySide6.QtWidgets import QApplication
            QApplication.processEvents(QEventLoop.AllEvents, 50)
        except Exception:
            pass

        def _build_dialog_contents():
            try:
                content = QWidget(dialog)
                content_layout = QVBoxLayout(content)
                content_layout.setContentsMargins(0, 0, 0, 0)

                result = self._force_retranslation_epub_or_text(
                    file_path,
                    parent_dialog=dialog,
                    tab_frame=content,
                    show_special_files_state=show_special_files_state,
                )
                if not result:
                    dialog.hide()
                    return

                timer = getattr(dialog, '_loading_icon_timer', None)
                if timer:
                    timer.stop()
                loading_widget.hide()
                loading_widget.deleteLater()
                dialog_layout.addWidget(content)

                dialog.setWindowTitle("Progress Manager - OPF Based" if result.get('spine_chapters') else "Progress Manager")
                if not hasattr(self, '_retranslation_dialog_cache'):
                    self._retranslation_dialog_cache = {}
                self._retranslation_dialog_cache[file_key] = result
                QTimer.singleShot(50, lambda: self._populate_progress_listbox_streamed(result))
            except Exception as e:
                print(f"Failed to build progress manager contents: {e}")
                import traceback
                traceback.print_exc()
                try:
                    loading_label.setText(f"Failed to load progress:\n{e}")
                    loading_label.show()
                except Exception:
                    pass

        QTimer.singleShot(50, _build_dialog_contents)

    def force_retranslation(self):
        """Force retranslation of specific chapters or images with improved display"""
        
        # Check for multiple file selection first
        if hasattr(self, 'selected_files') and len(self.selected_files) > 1:
            self._force_retranslation_multiple_files()
            return
        
        # Check if it's a folder selection (for images)
        if hasattr(self, 'selected_files') and len(self.selected_files) > 0:
            # Check if the first selected file is actually a folder
            first_item = self.selected_files[0]
            if os.path.isdir(first_item):
                self._force_retranslation_images_folder(first_item)
                return
        
        # Original logic for single files
        # Get input path from QLineEdit widget
        if hasattr(self.entry_epub, 'text'):
            # PySide6 QLineEdit widget
            input_path = self.entry_epub.text()
        else:
            input_path = ""
        
        if not input_path or not os.path.isfile(input_path):
            self._show_message('error', "Error", "Please select a valid EPUB, text file, or image folder first.")
            return
        
        # Check if it's an image file
        image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')
        if input_path.lower().endswith(image_extensions):
            # For single image, pass the image file path itself
            self._force_retranslation_images_folder(input_path)
            return
        
        # Check if dialog already exists for this file and is just hidden
        file_key = os.path.abspath(input_path)
        if hasattr(self, '_retranslation_dialog_cache') and file_key in self._retranslation_dialog_cache:
            # Reuse existing dialog - just show it and refresh data
            cached_data = self._retranslation_dialog_cache[file_key]
            if cached_data and cached_data.get('dialog'):
                # Recompute output directory (override path can change, or cache can be stale)
                epub_base = os.path.splitext(os.path.basename(input_path))[0]
                override_dir = (os.environ.get('OUTPUT_DIRECTORY') or os.environ.get('OUTPUT_DIR'))
                if not override_dir and hasattr(self, 'config'):
                    try:
                        override_dir = self.config.get('output_directory')
                    except Exception:
                        override_dir = None
                expected_output_dir = os.path.join(override_dir, epub_base) if override_dir else epub_base
                # On macOS .app bundles, cwd can be '/' (read-only root).
                # Resolve relative output paths against the input file's directory.
                # Only on macOS — on Windows this would change the output dir and break progress tracking.
                if _IS_MACOS and not os.path.isabs(expected_output_dir):
                    expected_output_dir = os.path.join(os.path.dirname(os.path.abspath(input_path)), expected_output_dir)

                output_dir = cached_data.get('output_dir')
                progress_file = cached_data.get('progress_file')

                # If cache points at a different location than current override, force a rebuild.
                if output_dir and expected_output_dir and os.path.abspath(output_dir) != os.path.abspath(expected_output_dir):
                    del self._retranslation_dialog_cache[file_key]
                else:
                    # Check if output folder still exists before trying to refresh
                    if not output_dir:
                        output_dir = expected_output_dir
                        cached_data['output_dir'] = output_dir
                        cached_data['progress_file'] = os.path.join(output_dir, "translation_progress.json")
                        progress_file = cached_data['progress_file']

                    if not os.path.exists(output_dir):
                        # Output folder doesn't exist - create it with an empty progress file
                        try:
                            os.makedirs(output_dir, exist_ok=True)
                            empty_prog = {"chapters": {}, "chapter_chunks": {}, "version": "2.1"}
                            pf = os.path.join(output_dir, "translation_progress.json")
                            with open(pf, 'w', encoding='utf-8') as f:
                                json.dump(empty_prog, f, ensure_ascii=False, indent=2)
                            cached_data['output_dir'] = output_dir
                            cached_data['progress_file'] = pf
                            print(f"📁 Created output folder: {output_dir}")
                            # Flash the PM button green to signal folder creation
                            self._flash_pm_button_green(output_dir)
                        except Exception as e:
                            self._show_message('error', "Error", f"Could not create output folder: {e}")
                            del self._retranslation_dialog_cache[file_key]
                            return
                        del self._retranslation_dialog_cache[file_key]

                    if not progress_file or not os.path.exists(progress_file):
                        # Progress file was deleted - show message and remove from cache,
                        # but DO NOT return. Fall through so we rebuild the dialog and
                        # auto-discover completed chapters in a single click.
                        self._show_message('info', "Info", "No progress tracking found. Existing translations will be auto-discovered.")
                        del self._retranslation_dialog_cache[file_key]
                    else:
                        dialog = cached_data['dialog']
                        dialog.show()
                        dialog.raise_()
                        dialog.activateWindow()

                        # Trigger refresh after the dialog is visible so reopening
                        # a large progress file does not block the first paint.
                        def _refresh_cached_single_dialog():
                            _rf = cached_data.get('refresh_func')
                            if callable(_rf):
                                try:
                                    _rf()
                                except Exception:
                                    self._refresh_retranslation_data(cached_data)
                            else:
                                self._refresh_retranslation_data(cached_data)

                        QTimer.singleShot(50, _refresh_cached_single_dialog)
                        return
        
        # For EPUB/text files, use the shared logic
        # Get current toggle state if it exists, or default based on file type
        # Default to True for .txt, .pdf, .csv, and .json files, False for .epub
        show_special_extensions = ('.txt', '.pdf', '.csv', '.json')
        show_special = input_path.lower().endswith(show_special_extensions)
        
        if hasattr(self, '_retranslation_dialog_cache') and file_key in self._retranslation_dialog_cache:
            cached_data = self._retranslation_dialog_cache[file_key]
            if cached_data:
                show_special = cached_data.get('show_special_files_state', show_special)
        
        self._show_retranslation_shell_then_build(input_path, show_special_files_state=show_special)


    def _force_retranslation_epub_or_text(self, file_path, parent_dialog=None, tab_frame=None, show_special_files_state=False):
        """
        Shared logic for force retranslation of EPUB/text files with OPF support
        Can be used standalone or embedded in a tab
        
        Args:
            file_path: Path to the EPUB/text file
            parent_dialog: If provided, won't create its own dialog
            tab_frame: If provided, will render into this frame instead of creating dialog
            show_special_files_state: Initial state for showing special files toggle
        
        Returns:
            dict: Contains all the UI elements and data for external access
        """
        
        epub_base = os.path.splitext(os.path.basename(file_path))[0]
        
        # Check for output directory override
        override_dir = (os.environ.get('OUTPUT_DIRECTORY') or os.environ.get('OUTPUT_DIR'))
        if not override_dir and hasattr(self, 'config'):
            override_dir = self.config.get('output_directory')
            
        if override_dir:
            output_dir = os.path.join(override_dir, epub_base)
        else:
            output_dir = epub_base
        # On macOS .app bundles, cwd can be '/' (read-only root).
        # Resolve relative output paths against the input file's directory.
        # Only on macOS — on Windows this would change the output dir and break progress tracking.
        if _IS_MACOS and not os.path.isabs(output_dir):
            output_dir = os.path.join(os.path.dirname(os.path.abspath(file_path)), output_dir)
        
        if not os.path.exists(output_dir):
            # Output folder doesn't exist - create it with an empty progress file
            try:
                os.makedirs(output_dir, exist_ok=True)
                empty_prog = {"chapters": {}, "chapter_chunks": {}, "version": "2.1"}
                progress_file_path = os.path.join(output_dir, "translation_progress.json")
                with open(progress_file_path, 'w', encoding='utf-8') as f:
                    json.dump(empty_prog, f, ensure_ascii=False, indent=2)
                print(f"📁 Created output folder: {output_dir}")
                # Flash the PM button green to signal folder creation
                self._flash_pm_button_green(output_dir)
            except Exception as e:
                if not parent_dialog:
                    self._show_message('error', "Error", f"Could not create output folder: {e}")
                return None
        
        progress_file = os.path.join(output_dir, "translation_progress.json")
        if not os.path.exists(progress_file):
            # No progress file - create empty progress structure
            # This allows fuzzy matching to discover existing files
            print("⚠️ No progress file found - will attempt to discover existing translations")
            prog = {
                "chapters": {},
                "chapter_chunks": {},
                "version": "2.1"
            }
        else:
            with open(progress_file, 'r', encoding='utf-8') as f:
                prog = json.load(f)

        # Helper: auto-discover completed files when no OPF is available
        def _auto_discover_from_output_dir(output_dir, prog):
            updated = False
            try:
                # Only exclude _translated.* combined output files when the source
                # file itself does NOT contain "_translated" in its name
                source_has_translated = "_translated" in os.path.basename(file_path).lower()
                files = [
                    f for f in os.listdir(output_dir)
                    if os.path.isfile(os.path.join(output_dir, f))
                    # accept any extension except known non-chapter files
                    and (source_has_translated or not f.lower().endswith("_translated.txt"))
                    and (source_has_translated or not f.lower().endswith("_translated.pdf"))
                    and (source_has_translated or not f.lower().endswith("_translated.html"))
                    and f != "translation_progress.json"
                    and f.lower() not in ("glossary.csv", "metadata.json", "styles.css", "rolling_summary.txt")
                    and not f.lower().endswith(".epub")
                    and not f.lower().endswith(".cache")
                ]
                for fname in files:
                    base = os.path.basename(fname)
                    # Normalize by stripping response_ and all extensions
                    if base.startswith("response_"):
                        base = base[len("response_"):]
                    while True:
                        new_base, ext = os.path.splitext(base)
                        if not ext:
                            break
                        base = new_base

                    import re
                    m = re.findall(r"(\d+)", base)
                    chapter_num = int(m[-1]) if m else None
                    key = str(chapter_num) if chapter_num is not None else f"special_{base}"
                    actual_num = chapter_num if chapter_num is not None else 0

                    if key in prog.get("chapters", {}):
                        continue
                    
                    # Also check if any existing entry already references this output file
                    already_tracked = any(
                        ch.get("output_file") == fname
                        for ch in prog.get("chapters", {}).values()
                    )
                    if already_tracked:
                        continue

                    prog.setdefault("chapters", {})[key] = {
                        "actual_num": actual_num,
                        "content_hash": "",
                        "output_file": fname,
                        "status": "completed",
                        "last_updated": os.path.getmtime(os.path.join(output_dir, fname)),
                        "auto_discovered": True,
                        "original_basename": fname
                    }
                    updated = True
            except Exception as e:
                print(f"⚠️ Auto-discovery (no OPF) failed: {e}")
            return updated
        
        # Clean up missing files and merged children when opening the GUI
        # This handles the case where parent files were manually deleted
        from TransateKRtoEN import ProgressManager
        temp_progress = ProgressManager(os.path.dirname(progress_file))
        temp_progress.prog = prog
        temp_progress.cleanup_missing_files(output_dir)
        prog = temp_progress.prog
        
        # Save the cleaned progress back to file
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(prog, f, ensure_ascii=False, indent=2)
        
        # =====================================================
        # PARSE CONTENT.OPF FOR CHAPTER MANIFEST
        # =====================================================
        
        # State variables for title-row toggles (lists allow nested handlers to mutate them)
        show_special_files = [show_special_files_state]
        show_model_info_state = self._get_retranslation_show_model_info_state(file_path)
        show_model_info = [show_model_info_state]
        
        spine_chapters = []
        opf_chapter_order = {}
        is_epub = file_path.lower().endswith('.epub')
        opf_parsed = False

        if is_epub and os.path.exists(file_path):
            try:
                import xml.etree.ElementTree as ET
                import zipfile
                
                with zipfile.ZipFile(file_path, 'r') as zf:
                    # Find content.opf file
                    opf_path = None
                    opf_content = None
                    
                    # First try to find via container.xml
                    try:
                        container_content = zf.read('META-INF/container.xml')
                        container_root = ET.fromstring(container_content)
                        rootfile = container_root.find('.//{urn:oasis:names:tc:opendocument:xmlns:container}rootfile')
                        if rootfile is not None:
                            opf_path = rootfile.get('full-path')
                    except:
                        pass
                    
                    # Fallback: search for content.opf
                    if not opf_path:
                        for name in zf.namelist():
                            if name.endswith('content.opf'):
                                opf_path = name
                                break
                    
                    if opf_path:
                        opf_content = zf.read(opf_path)
                        
                        # Parse OPF
                        root = ET.fromstring(opf_content)
                        
                        # Handle namespaces
                        ns = {'opf': 'http://www.idpf.org/2007/opf'}
                        if root.tag.startswith('{'):
                            default_ns = root.tag[1:root.tag.index('}')]
                            ns = {'opf': default_ns}
                        
                        # Get manifest - all chapter files
                        manifest_chapters = {}
                        
                        for item in root.findall('.//opf:manifest/opf:item', ns):
                            item_id = item.get('id')
                            href = item.get('href')
                            media_type = item.get('media-type', '')
                            
                            if item_id and href and ('html' in media_type.lower() or href.endswith(('.html', '.xhtml', '.htm'))):
                                filename = os.path.basename(href)
                                
                                # Detect special files using configured keyword lists
                                # (mirrors TransateKRtoEN._is_configured_special_file)
                                is_special = self._is_special_file(filename) if hasattr(self, '_is_special_file') else (not bool(re.search(r'\d', filename)))
                                
                                # Add all files - UI will handle filtering based on toggle
                                manifest_chapters[item_id] = {
                                    'filename': filename,
                                    'href': href,
                                    'media_type': media_type,
                                    'is_special': is_special
                                }
                        
                        # Get spine order - the reading order
                        spine = root.find('.//opf:spine', ns)
                        
                        if spine is not None:
                            for itemref in spine.findall('opf:itemref', ns):
                                idref = itemref.get('idref')
                                if idref and idref in manifest_chapters:
                                    chapter_info = manifest_chapters[idref]
                                    filename = chapter_info['filename']
                                    is_special = chapter_info.get('is_special', False)
                                    
                                    # Extract chapter number from filename
                                    import re
                                    matches = re.findall(r'(\d+)', filename)
                                    if matches:
                                        file_chapter_num = 0 if is_special else int(matches[-1])
                                    elif is_special:
                                        # Special files without numbers should be chapter 0
                                        file_chapter_num = 0
                                    else:
                                        # Non-numbered OPF files like info.xhtml are
                                        # not real chapter numbers in the progress UI.
                                        file_chapter_num = 0
                                    
                                    # Add all files - UI will handle filtering based on toggle
                                    spine_chapters.append({
                                        'id': idref,
                                        'filename': filename,
                                        'position': len(spine_chapters),
                                        'file_chapter_num': file_chapter_num,
                                        'status': 'unknown',  # Will be updated
                                        'output_file': None,    # Will be updated
                                        'is_special': is_special
                                    })
                                    
                                    # Store the order for later use
                                    opf_chapter_order[filename] = len(spine_chapters) - 1
                                    
                                    # Also store without extension for matching
                                    filename_noext = os.path.splitext(filename)[0]
                                    opf_chapter_order[filename_noext] = len(spine_chapters) - 1
                                    opf_parsed = True
                        
            except Exception as e:
                print(f"Warning: Could not parse OPF: {e}")

        # If no OPF/spine, fall back to auto-discovery from output_dir
        if not opf_parsed or len(spine_chapters) == 0:
            if _auto_discover_from_output_dir(output_dir, prog):
                try:
                    with open(progress_file, 'w', encoding='utf-8') as f:
                        json.dump(prog, f, ensure_ascii=False, indent=2)
                    print("💾 Saved auto-discovered progress (no OPF available)")
                except Exception as e:
                    print(f"⚠️ Failed to save auto-discovered progress: {e}")
        else:
            # OPF-AWARE AUTO-DISCOVERY: Use OPF filenames as original_basename
            # This ensures correct mapping between OPF entries and response files
            progress_updated = False
            for spine_ch in spine_chapters:
                opf_filename = spine_ch['filename']  # e.g., "0009_10_.xhtml"
                base_name = os.path.splitext(opf_filename)[0]  # e.g., "0009_10_"
                
                # Look for corresponding response file on disk
                response_file = f"response_{base_name}.html"
                response_path = os.path.join(output_dir, response_file)
                
                if os.path.exists(response_path):
                    # Check if we already have a progress entry with correct original_basename
                    already_tracked = False
                    for ch_info in prog.get("chapters", {}).values():
                        if ch_info.get("original_basename") == opf_filename:
                            already_tracked = True
                            break
                        # Also check by output_file
                        if ch_info.get("output_file") == response_file:
                            # Update original_basename if missing or wrong
                            if ch_info.get("original_basename") != opf_filename:
                                ch_info["original_basename"] = opf_filename
                                progress_updated = True
                            already_tracked = True
                            break
                    
                    if not already_tracked:
                        # Create new progress entry with correct original_basename
                        chapter_num = spine_ch['file_chapter_num']
                        key = str(chapter_num) if chapter_num else f"special_{base_name}"
                        
                        # Avoid duplicate keys
                        if key not in prog.get("chapters", {}):
                            prog.setdefault("chapters", {})[key] = {
                                "actual_num": chapter_num,
                                "content_hash": "",
                                "output_file": response_file,
                                "status": "completed",
                                "last_updated": os.path.getmtime(response_path),
                                "auto_discovered": True,
                                "original_basename": opf_filename  # CORRECT: OPF filename
                            }
                            progress_updated = True
                            print(f"✅ OPF-aware discovery: {opf_filename} -> {response_file}")
            
            if progress_updated:
                try:
                    with open(progress_file, 'w', encoding='utf-8') as f:
                        json.dump(prog, f, ensure_ascii=False, indent=2)
                    #print("💾 Saved OPF-aware auto-discovered progress")
                except Exception as e:
                    print(f"⚠️ Failed to save progress: {e}")
        
        # =====================================================
        # MATCH OPF CHAPTERS WITH TRANSLATION PROGRESS
        # =====================================================
        
        # Helper: normalize filenames for OPF / progress matching
        # We intentionally strip a leading "response_" prefix so that
        # files like "chapter001.xhtml" and "response_chapter001.xhtml"
        # are treated as referring to the same logical entry.
        def _normalize_opf_match_name(name: str) -> str:
            if not name:
                return ""
            base = os.path.basename(name)
            # Remove response_ prefix
            if base.startswith("response_"):
                base = base[len("response_"):]
            # Remove all extensions so that .html, .xhtml, .htm, etc. all match
            # and double extensions like .html.xhtml collapse to the stem.
            while True:
                new_base, ext = os.path.splitext(base)
                if not ext:
                    break
                base = new_base
            return base

        def _opf_names_equal(a: str, b: str) -> bool:
            return _normalize_opf_match_name(a) == _normalize_opf_match_name(b)

        # Build a map of original basenames to progress entries (normalized)
        basename_to_progress = {}
        for chapter_key, chapter_info in prog.get("chapters", {}).items():
            original_basename = chapter_info.get("original_basename", "")
            if original_basename:
                norm_key = _normalize_opf_match_name(original_basename)
                if norm_key not in basename_to_progress:
                    basename_to_progress[norm_key] = []
                basename_to_progress[norm_key].append((chapter_key, chapter_info))
        
        # Also build a map of response files (include both exact and normalized keys)
        response_file_to_progress = {}
        for chapter_key, chapter_info in prog.get("chapters", {}).items():
            output_file = chapter_info.get("output_file", "")
            if output_file:
                # Exact key
                if output_file not in response_file_to_progress:
                    response_file_to_progress[output_file] = []
                response_file_to_progress[output_file].append((chapter_key, chapter_info))
                # Normalized key (ignoring response_ prefix)
                norm_key = _normalize_opf_match_name(output_file)
                if norm_key != output_file:
                    if norm_key not in response_file_to_progress:
                        response_file_to_progress[norm_key] = []
                    response_file_to_progress[norm_key].append((chapter_key, chapter_info))
        
        # Update spine chapters with translation status
        for idx, spine_ch in enumerate(spine_chapters):
            if idx % 80 == 0:
                self._ui_yield()
            filename = spine_ch['filename']
            chapter_num = spine_ch['file_chapter_num']
            is_special = spine_ch.get('is_special', False)
            
            # Find the actual response file that exists
            base_name = os.path.splitext(filename)[0]
            expected_response = None
            
            # Special files need to check what actually exists on disk
            if is_special:
                # Check for response_ prefix version
                response_with_prefix = f"response_{base_name}.html"
                retain = os.getenv('RETAIN_SOURCE_EXTENSION', '0') == '1' or self.config.get('retain_source_extension', False)
                
                if retain:
                    expected_response = filename
                elif os.path.exists(os.path.join(output_dir, response_with_prefix)):
                    expected_response = response_with_prefix
                else:
                    # Fallback to original filename
                    expected_response = filename
            else:
                # Use OPF filename directly to avoid mismatching
                retain = os.getenv('RETAIN_SOURCE_EXTENSION', '0') == '1' or self.config.get('retain_source_extension', False)
                if retain:
                    expected_response = filename
                else:
                    # Handle .htm.html -> .html conversion
                    stripped_base_name = base_name
                    if base_name.endswith('.htm'):
                        stripped_base_name = base_name[:-4]  # Remove .htm suffix
                    expected_response = filename  # Use exact OPF filename
                    
                    # Also check for response_ prefix version (used by the translator
                    # when TRANSLATE_ALL_NUMBERED_HTML overrides the special-file skip)
                    response_with_prefix = f"response_{base_name}.html"
                    if not os.path.exists(os.path.join(output_dir, expected_response)) and \
                       os.path.exists(os.path.join(output_dir, response_with_prefix)):
                        expected_response = response_with_prefix
            
            response_path = os.path.join(output_dir, expected_response)
            
            # Check various ways to find the translation progress info
            matched_info = None
            
            # Method 1: Check by original basename (ignoring response_ prefix)
            basename_key = _normalize_opf_match_name(filename)
            if basename_key in basename_to_progress:
                entries = basename_to_progress[basename_key]
                if entries:
                    _, chapter_info = entries[0]
                    # For in_progress/failed/qa_failed/pending, also verify actual_num matches
                    status = chapter_info.get('status', '')
                    if status in ['in_progress', 'failed', 'qa_failed', 'pending']:
                        if chapter_info.get('actual_num') == chapter_num:
                            matched_info = chapter_info
                    else:
                        matched_info = chapter_info
            
            # Method 2: Check by response file (with corrected extension)
            if not matched_info and expected_response in response_file_to_progress:
                entries = response_file_to_progress[expected_response]
                if entries:
                    _, chapter_info = entries[0]
                    # For in_progress/failed/qa_failed/pending, also verify actual_num matches
                    status = chapter_info.get('status', '')
                    if status in ['in_progress', 'failed', 'qa_failed', 'pending']:
                        if chapter_info.get('actual_num') == chapter_num:
                            matched_info = chapter_info
                    else:
                        matched_info = chapter_info
            
            # Method 3: Search through all progress entries for matching output file
            if not matched_info:
                for chapter_key, chapter_info in prog.get("chapters", {}).items():
                    out_file = chapter_info.get('output_file')
                    if out_file == expected_response or _opf_names_equal(out_file, expected_response):
                        # For in_progress/failed/qa_failed/pending, also verify actual_num matches
                        status = chapter_info.get('status', '')
                        if status in ['in_progress', 'failed', 'qa_failed', 'pending']:
                            if chapter_info.get('actual_num') == chapter_num:
                                matched_info = chapter_info
                                break
                        else:
                            matched_info = chapter_info
                            break
            
            # Method 4: CRUCIAL - Match by chapter number (actual_num vs file_chapter_num)
            # Also check composite keys for special files (e.g., "0_message", "0_TOC")
            if not matched_info:
                # First try simple chapter number key
                simple_key = str(chapter_num)
                if simple_key in prog.get("chapters", {}):
                    chapter_info = prog["chapters"][simple_key]
                    out_file = chapter_info.get('output_file')
                    status = chapter_info.get('status', '')
                    orig_base = chapter_info.get('original_basename', '')
                    if orig_base:
                        orig_base = os.path.basename(orig_base)
                    
                    # Merged chapters: check if parent exists AND original_basename matches
                    if status == 'merged':
                        parent_num = chapter_info.get('merged_parent_chapter')
                        # For merged chapters, match by original_basename (not output_file)
                        # because output_file points to parent's file, not this chapter's source file
                        # Strip extension for comparison since orig_base may not have it
                        filename_noext = os.path.splitext(filename)[0]
                        if parent_num is not None and (
                            _opf_names_equal(orig_base, filename)
                            or _opf_names_equal(orig_base, filename_noext)
                            or not orig_base
                        ):
                            parent_key = str(parent_num)
                            if parent_key in prog.get("chapters", {}):
                                # Just verify parent exists, don't enforce 'completed' status
                                # This ensures we show 'merged' even if parent is completed_empty or other states
                                matched_info = chapter_info
                    # In-progress/failed/pending chapters: require BOTH actual_num AND output_file
                    # to match to avoid cross-matching files.
                    elif status in ['in_progress', 'failed', 'pending']:
                        if chapter_info.get('actual_num') == chapter_num and (
                            out_file == expected_response or _opf_names_equal(out_file, expected_response)
                        ):
                            matched_info = chapter_info
                    # qa_failed chapters: match by chapter number only so they are always visible
                    elif status == 'qa_failed':
                        if chapter_info.get('actual_num') == chapter_num:
                            matched_info = chapter_info
                    # Normal match: output file matches expected (ignoring response_ prefix)
                    elif out_file == expected_response or _opf_names_equal(out_file, expected_response):
                        matched_info = chapter_info
                
                # If not found, check for composite key (chapter_num + filename)
                if not matched_info and is_special:
                    # For special files, try composite key format: "{chapter_num}_{filename_without_extension}"
                    base_name = os.path.splitext(filename)[0]
                    # Remove "response_" prefix if present in the filename
                    if base_name.startswith("response_"):
                        base_name = base_name[9:]
                    composite_key = f"{chapter_num}_{base_name}"
                    
                    if composite_key in prog.get("chapters", {}):
                        matched_info = prog["chapters"][composite_key]
                
                # Fallback: iterate through all entries matching chapter number,
                # but only accept when it clearly refers to the same source file.
                # This prevents files like "000_information.xhtml" and "0153_0.xhtml"
                # (both parsed as chapter 0) from being conflated.
                if not matched_info:
                    for chapter_key, chapter_info in prog.get("chapters", {}).items():
                        actual_num = chapter_info.get('actual_num')
                        # Also check 'chapter_num' as fallback
                        if actual_num is None:
                            actual_num = chapter_info.get('chapter_num')
                        
                        if actual_num is not None and actual_num == chapter_num:
                            orig_base = chapter_info.get('original_basename', '')
                            if orig_base:
                                orig_base = os.path.basename(orig_base)
                            out_file = chapter_info.get('output_file')
                            status = chapter_info.get('status', '')
                            qa_issues = chapter_info.get('qa_issues_found', [])
                            
                            # Merged chapters: match by actual_num AND original_basename
                            # For merged, output_file points to parent so we must match by source filename
                            if status == 'merged':
                                parent_num = chapter_info.get('merged_parent_chapter')
                                # Match by original_basename (the source file), not output_file (parent's file)
                                # Strip extension for comparison since orig_base may not have it
                                filename_noext = os.path.splitext(filename)[0]
                                if parent_num is not None and (
                                    _opf_names_equal(orig_base, filename)
                                    or _opf_names_equal(orig_base, filename_noext)
                                    or not orig_base
                                ):
                                    # Check if parent chapter exists
                                    parent_key = str(parent_num)
                                    if parent_key in prog.get("chapters", {}):
                                        # Just verify parent exists, don't enforce 'completed' status
                                        matched_info = chapter_info
                                        break
                            
                            # In-progress/failed/pending chapters: require BOTH actual_num AND output_file
                            # to match to avoid cross-matching files.
                            if status in ['in_progress', 'failed', 'pending']:
                                if actual_num == chapter_num and (
                                    out_file == expected_response or _opf_names_equal(out_file, expected_response)
                                ):
                                    matched_info = chapter_info
                                    break
                            # qa_failed chapters: match by chapter number only so they are always visible,
                            # even when filenames don't line up perfectly.
                            elif status == 'qa_failed':
                                if actual_num == chapter_num:
                                    matched_info = chapter_info
                                    break
                            
                            # Only treat as a match for other statuses if the original basename matches
                            # this filename, or, when original_basename is missing, the output_file matches
                            # what we expect.
                            if status not in ['in_progress', 'failed', 'qa_failed', 'pending']:
                                if (
                                    orig_base and _opf_names_equal(orig_base, filename)
                                ) or (
                                    not orig_base and out_file and (
                                        out_file == expected_response or _opf_names_equal(out_file, expected_response)
                                    )
                                ):
                                    matched_info = chapter_info
                                    break
            
            # Determine if translation file exists
            file_exists = os.path.exists(response_path)
            
            # Set status and output file based on findings
            if matched_info:
                # We found progress tracking info - use its status
                status = matched_info.get('status', 'unknown')
                spine_ch['progress_key'] = matched_info.get('_key')
                
                # CRITICAL: For failed/in_progress/qa_failed/pending, ALWAYS use progress status
                # Never let file existence override these statuses
                if status in ['failed', 'in_progress', 'qa_failed', 'pending']:
                    spine_ch['status'] = status
                    spine_ch['output_file'] = matched_info.get('output_file') or expected_response
                    spine_ch['progress_entry'] = matched_info
                    # Skip all other logic - don't check file existence
                    continue
                
                # For other statuses (completed, merged, etc.)
                spine_ch['status'] = status
                
                # For special files, always use the original filename (ignore what's in progress JSON)
                if is_special:
                    spine_ch['output_file'] = expected_response
                else:
                    spine_ch['output_file'] = matched_info.get('output_file', expected_response)
                
                spine_ch['progress_entry'] = matched_info
                
                # Handle null output_file
                if not spine_ch['output_file']:
                    spine_ch['output_file'] = expected_response
                
                # Verify file actually exists for completed status
                if status == 'completed':
                    output_path = os.path.join(output_dir, spine_ch['output_file'])
                    if not os.path.exists(output_path):
                        # If the expected_response file exists, prefer that and
                        # transparently update the progress entry.
                        if file_exists and expected_response:
                            fixed_output_path = os.path.join(output_dir, expected_response)
                            if os.path.exists(fixed_output_path):
                                spine_ch['output_file'] = expected_response

                                # If this spine chapter is tied to a concrete
                                # progress entry, keep it consistent.
                                if 'progress_entry' in spine_ch and spine_ch['progress_entry'] is not None:
                                    spine_ch['progress_entry']['output_file'] = expected_response

                                    # Also update the master prog dict so the
                                    # corrected value is written back later.
                                    for ch_key, ch_info in prog.get('chapters', {}).items():
                                        if ch_info is spine_ch['progress_entry']:
                                            prog['chapters'][ch_key]['output_file'] = expected_response
                                            break
                            else:
                                # No matching file anywhere – mark as missing.
                                spine_ch['status'] = 'not_translated'
                        else:
                            # Legacy behaviour: nothing on disk for this entry.
                            spine_ch['status'] = 'not_translated'
            
            elif file_exists:
                # File exists but no progress tracking - mark as completed
                spine_ch['status'] = 'completed'
                spine_ch['output_file'] = expected_response
            
            else:
                # No file and no progress tracking - LAST RESORT: Try exact filename matching
                # This handles the case where progress file was deleted but files exist
                # Match by filename only (ignore response_ prefix and all extensions)
                
                def normalize_filename(fname):
                    """Remove response_ prefix and all extensions for exact comparison"""
                    base = os.path.basename(fname)
                    # Remove response_ prefix
                    if base.startswith('response_'):
                        base = base[9:]
                    # Remove all extensions (including double extensions like .html.xhtml)
                    while True:
                        new_base, ext = os.path.splitext(base)
                        if not ext:
                            break
                        base = new_base
                    return base
                
                # Normalize the OPF filename
                normalized_opf = normalize_filename(filename)
                
                # Search for exact matching file in output directory
                matched_file = None
                if os.path.exists(output_dir):
                    try:
                        for existing_file in os.listdir(output_dir):
                            if os.path.isfile(os.path.join(output_dir, existing_file)):
                                normalized_existing = normalize_filename(existing_file)
                                # Exact match only - no fuzzy logic
                                if normalized_existing == normalized_opf:
                                    matched_file = existing_file
                                    break
                    except Exception as e:
                        print(f"Warning: Error scanning output directory for match: {e}")
                
                if matched_file:
                    # Found an exact matching file by normalized name - mark as completed
                    spine_ch['status'] = 'completed'
                    spine_ch['output_file'] = matched_file
                    print(f"📁 Matched: {filename} -> {matched_file}")
                else:
                    # No file and no progress tracking - not translated
                    spine_ch['status'] = 'not_translated'
                    spine_ch['output_file'] = expected_response
        
        # =====================================================
        # SAVE AUTO-DISCOVERED FILES TO PROGRESS
        # =====================================================
        
        # Check if we discovered any new completed files (exact matched by normalized filename)
        # and add them to the progress file
        progress_updated = False
        for spine_ch in spine_chapters:
            # Only add entries that were marked as completed but have no progress entry
            if spine_ch['status'] == 'completed' and 'progress_entry' not in spine_ch:
                chapter_num = spine_ch['file_chapter_num']
                output_file = spine_ch['output_file']
                filename = spine_ch['filename']
                
                # Create a progress entry for this auto-discovered file
                chapter_key = str(chapter_num)
                
                # Check if key already exists (avoid duplicates)
                # If the key exists but points to a DIFFERENT file, use a composite
                # key to avoid overwriting (e.g. chapter0003 vs chapter_notice0003).
                existing = prog.get("chapters", {}).get(chapter_key)
                if existing:
                    existing_out = existing.get('output_file', '')
                    existing_base = existing.get('original_basename', '')
                    # If same output file, this is already tracked
                    if existing_out == output_file:
                        continue
                    # Different file occupies this key — use composite key
                    base_noext = os.path.splitext(filename)[0]
                    chapter_key = f"{chapter_num}_{base_noext}"
                    # Also skip if composite key already exists
                    if chapter_key in prog.get("chapters", {}):
                        continue
                
                prog.setdefault("chapters", {})[chapter_key] = {
                    "actual_num": chapter_num,
                    "content_hash": "",  # Unknown since we don't have the source
                    "output_file": output_file,
                    "status": "completed",
                    "last_updated": os.path.getmtime(os.path.join(output_dir, output_file)),
                    "auto_discovered": True,
                    "original_basename": filename
                }
                progress_updated = True
                print(f"✅ Auto-discovered and tracked: {filename} -> {output_file} (key: {chapter_key})")
        
        # Save progress file if we added new entries
        if progress_updated:
            try:
                with open(progress_file, 'w', encoding='utf-8') as f:
                    json.dump(prog, f, ensure_ascii=False, indent=2)
                print(f"💾 Saved {sum(1 for ch in spine_chapters if ch['status'] == 'completed' and 'progress_entry' not in ch)} auto-discovered files to progress file")
            except Exception as e:
                print(f"⚠️ Warning: Failed to save progress file: {e}")
        
        # =====================================================
        # BUILD DISPLAY INFO
        # =====================================================
        
        chapter_display_info = []
        
        if spine_chapters:
            # Use OPF order
            for spine_ch in spine_chapters:
                display_info = {
                    'key': spine_ch.get('filename', ''),
                    'num': spine_ch['file_chapter_num'],
                    'info': spine_ch.get('progress_entry', {}),
                    'output_file': spine_ch['output_file'],
                    'status': spine_ch['status'],
                    'duplicate_count': 1,
                    'entries': [],
                    'opf_position': spine_ch['position'],
                    'original_filename': spine_ch['filename'],
                    'is_special': spine_ch.get('is_special', False)
                }
                chapter_display_info.append(display_info)
        else:
            # Fallback to original logic if no OPF
            # Known non-chapter files that should never appear in the progress list
            _non_chapter_files = {"glossary.csv", "metadata.json", "styles.css", "rolling_summary.txt"}
            _source_has_translated = "_translated" in os.path.basename(file_path).lower()
            files_to_entries = {}
            for chapter_key, chapter_info in prog.get("chapters", {}).items():
                output_file = chapter_info.get("output_file", "")
                status = chapter_info.get("status", "")
                
                # Skip known non-chapter files
                if output_file and output_file.lower() in _non_chapter_files:
                    continue
                # Skip combined _translated output files (unless source itself has _translated)
                if output_file and not _source_has_translated and any(
                    output_file.lower().endswith(s) for s in ("_translated.txt", "_translated.pdf", "_translated.html")
                ):
                    continue
                
                # Include chapters with output files OR transient statuses with null output file (legacy)
                # (composite keys like "0_TOC" should still be represented in the UI)
                if output_file or status in ["in_progress", "pending", "failed", "qa_failed"]:
                    # For merged chapters, use a unique key (chapter_key) instead of output_file
                    # This ensures merged chapters appear as separate entries in the list
                    if status == "merged":
                        file_key = f"_merged_{chapter_key}"
                    elif output_file:
                        file_key = output_file
                    elif status == "in_progress":
                        file_key = f"_in_progress_{chapter_key}"
                    elif status == "pending":
                        file_key = f"_pending_{chapter_key}"
                    elif status == "qa_failed":
                        file_key = f"_qa_failed_{chapter_key}"
                    else:  # failed
                        file_key = f"_failed_{chapter_key}"
                    
                    if file_key not in files_to_entries:
                        files_to_entries[file_key] = []
                    files_to_entries[file_key].append((chapter_key, chapter_info))
            
            for output_file, entries in files_to_entries.items():
                chapter_key, chapter_info = entries[0]
                
                # Get the actual output file (strip placeholder prefix if present)
                actual_output_file = output_file
                if (
                    output_file.startswith("_merged_")
                    or output_file.startswith("_in_progress_")
                    or output_file.startswith("_pending_")
                    or output_file.startswith("_failed_")
                    or output_file.startswith("_qa_failed_")
                ):
                    # For merged/in_progress/pending/failed/qa_failed, get the actual output_file from chapter_info
                    actual_output_file = chapter_info.get("output_file", "")
                    if not actual_output_file:
                        # Generate expected filename based on actual_num
                        actual_num = chapter_info.get("actual_num")
                        if actual_num is not None:
                            # Use .txt extension for text files, .html for EPUB
                            ext = ".txt" if file_path.endswith(".txt") else ".html"
                            actual_output_file = f"response_section_{actual_num}{ext}"
                
                # Check if this is a special file (files without numbers)
                original_basename = chapter_info.get("original_basename", "")
                filename_to_check = original_basename if original_basename else actual_output_file
                
                is_special = self._is_special_file(filename_to_check) if hasattr(self, '_is_special_file') else (not bool(re.search(r'\d', filename_to_check)))
                
                # Extract chapter number - prioritize stored values
                chapter_num = None
                if 'actual_num' in chapter_info and chapter_info['actual_num'] is not None:
                    chapter_num = chapter_info['actual_num']
                elif 'chapter_num' in chapter_info and chapter_info['chapter_num'] is not None:
                    chapter_num = chapter_info['chapter_num']
                
                # Fallback: extract from filename
                if chapter_num is None:
                    import re
                    matches = re.findall(r'(\d+)', actual_output_file)
                    if matches:
                        chapter_num = int(matches[-1])
                    else:
                        chapter_num = 999999
                
                status = chapter_info.get("status", "unknown")
                if status in ("completed_empty", "completed_image_only"):
                    status = "completed"
                
                # Check file existence
                if status == "completed":
                    output_path = os.path.join(output_dir, actual_output_file)
                    if not os.path.exists(output_path):
                        status = "file_missing"
                
                chapter_display_info.append({
                    'key': chapter_key,
                    'num': chapter_num,
                    'info': chapter_info,
                    'output_file': actual_output_file,  # Use actual output file, not placeholder
                    'status': status,
                    'duplicate_count': len(entries),
                    'entries': entries,
                    'is_special': is_special
                })
            
            # Sort by chapter number
            chapter_display_info.sort(key=lambda x: x['num'] if x['num'] is not None else 999999)

        self._append_pdf_ocr_display_info({'prog': prog, 'file_path': file_path, 'output_dir': output_dir}, chapter_display_info)
        self._append_image_gen_display_info({'prog': prog, 'file_path': file_path, 'output_dir': output_dir}, chapter_display_info)
        
        # =====================================================
        # CREATE UI
        # =====================================================
        
        # If no parent dialog or tab frame, create standalone dialog
        if not parent_dialog and not tab_frame:
            # Ensure QApplication exists for standalone PySide6 dialog
            from PySide6.QtWidgets import QApplication
            if not QApplication.instance():
                # Create QApplication if it doesn't exist
                import sys
                QApplication(sys.argv)

            # Create standalone PySide6 dialog.
            # IMPORTANT: If created without a parent, it will NOT inherit the main window's
            # dark stylesheet and will fall back to the OS theme (white on some Win10 setups).
            parent_widget = self if isinstance(self, QWidget) else None
            dialog = QDialog(parent_widget)
            dialog.setWindowTitle("Progress Manager - OPF Based" if spine_chapters else "Progress Manager")
            # Keep above the translator window but allow interaction with it
            # Parent-child windowing keeps this above the translator GUI
            dialog.setWindowModality(Qt.NonModal)
            # Use 38% width, 40% height for 1920x1080
            width, height = self._get_dialog_size(0.38, 0.4)
            dialog.resize(width, height)

            # Inherit/copy the main window stylesheet when available (ensures consistent dark theme).
            try:
                if parent_widget is not None:
                    ss = parent_widget.styleSheet()
                    if ss:
                        dialog.setStyleSheet(ss)
            except Exception:
                pass
            
            # Set icon
            try:
                from PySide6.QtGui import QIcon
                # Try to get base_dir from self (TranslatorGUI), fallback to calculating it
                if hasattr(self, 'base_dir'):
                    base_dir = self.base_dir
                else:
                    base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
                ico_path = os.path.join(base_dir, 'Halgakos.ico')
                if os.path.isfile(ico_path):
                    dialog.setWindowIcon(QIcon(ico_path))
            except Exception as e:
                print(f"Failed to load icon: {e}")
            dialog_layout = QVBoxLayout(dialog)
            container = QWidget(dialog)
            container_layout = QVBoxLayout(container)
            dialog_layout.addWidget(container)
        else:
            container = tab_frame or parent_dialog
            if not hasattr(container, 'layout') or container.layout() is None:
                container_layout = QVBoxLayout(container)
            else:
                container_layout = container.layout()
            dialog = parent_dialog
        
        # Title and toggle row
        title_row = QWidget()
        title_layout = QHBoxLayout(title_row)
        title_layout.setContentsMargins(0, 0, 0, 0)
        
        title_text = "Chapters from content.opf (in reading order):" if spine_chapters else "Select chapters to retranslate:"
        title_label = QLabel(title_text)
        title_font = QFont('Arial', 12 if not tab_frame else 11)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_layout.addWidget(title_label)
        
        title_layout.addStretch()
        
        # Add toggle for showing special files
        from PySide6.QtWidgets import QCheckBox
        show_special_files_cb = QCheckBox("Show skipped files")
        show_special_files_cb.setChecked(show_special_files[0])  # Preserve the current state
        show_special_files_cb.setToolTip("When enabled, shows files that would be skipped during translation\n(matching the special file keywords configured in Other Settings).")
        
        # Register this checkbox and checkmark with parent dialog for cross-tab syncing
        if parent_dialog and not hasattr(parent_dialog, '_all_toggle_checkboxes'):
            parent_dialog._all_toggle_checkboxes = []
            parent_dialog._all_checkmark_labels = []
            parent_dialog._tab_file_paths = {}  # Map file_path to index
        if parent_dialog:
            # Store the index for this file
            file_key = os.path.abspath(file_path)
            if file_key not in parent_dialog._tab_file_paths:
                parent_dialog._tab_file_paths[file_key] = len(parent_dialog._all_toggle_checkboxes)
                parent_dialog._all_toggle_checkboxes.append(show_special_files_cb)
            else:
                # Replace the old checkbox at this index
                idx = parent_dialog._tab_file_paths[file_key]
                if idx < len(parent_dialog._all_toggle_checkboxes):
                    parent_dialog._all_toggle_checkboxes[idx] = show_special_files_cb
        
        # Apply blue checkbox stylesheet (matching Other Settings dialog)
        show_special_files_cb.setStyleSheet("""
            QCheckBox {
                color: white;
                spacing: 6px;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid #5a9fd4;
                border-radius: 2px;
                background-color: #2d2d2d;
            }
            QCheckBox::indicator:checked {
                background-color: #5a9fd4;
                border-color: #5a9fd4;
            }
            QCheckBox::indicator:hover {
                border-color: #7bb3e0;
            }
            QCheckBox:disabled {
                color: #666666;
            }
            QCheckBox::indicator:disabled {
                background-color: #1a1a1a;
                border-color: #3a3a3a;
            }
        """)
        
        # Create checkmark overlay for the check symbol
        checkmark = QLabel("✓", show_special_files_cb)
        checkmark.setStyleSheet("""
            QLabel {
                color: white;
                background: transparent;
                font-weight: bold;
                font-size: 11px;
            }
        """)
        checkmark.setAlignment(Qt.AlignCenter)
        checkmark.hide()
        checkmark.setAttribute(Qt.WA_TransparentForMouseEvents)
        
        def position_checkmark():
            try:
                if checkmark:
                    checkmark.setGeometry(2, 1, 14, 14)
            except RuntimeError:
                pass
        
        def update_checkmark():
            try:
                if show_special_files_cb and checkmark:
                    if show_special_files_cb.isChecked():
                        position_checkmark()
                        checkmark.show()
                    else:
                        checkmark.hide()
            except RuntimeError:
                pass
        
        show_special_files_cb.stateChanged.connect(update_checkmark)
        
        def safe_init():
            try:
                position_checkmark()
                update_checkmark()
            except RuntimeError:
                pass
        
        QTimer.singleShot(0, safe_init)
        
        # Register checkmark for cross-tab syncing
        if parent_dialog:
            file_key = os.path.abspath(file_path)
            if file_key in parent_dialog._tab_file_paths:
                idx = parent_dialog._tab_file_paths[file_key]
                # Append if new, replace if exists
                if idx >= len(parent_dialog._all_checkmark_labels):
                    parent_dialog._all_checkmark_labels.append(checkmark)
                else:
                    parent_dialog._all_checkmark_labels[idx] = checkmark
        
        title_layout.addWidget(show_special_files_cb)

        show_model_info_cb = QCheckBox("Show Model Info")
        show_model_info_cb.setChecked(show_model_info[0])
        show_model_info_cb.setToolTip("When enabled, replaces the output-file column with the model used for each request.")
        show_model_info_cb.setStyleSheet(show_special_files_cb.styleSheet())

        model_checkmark = QLabel("\u2713", show_model_info_cb)
        model_checkmark.setStyleSheet("""
            QLabel {
                color: white;
                background: transparent;
                font-weight: bold;
                font-size: 11px;
            }
        """)
        model_checkmark.setAlignment(Qt.AlignCenter)
        model_checkmark.hide()
        model_checkmark.setAttribute(Qt.WA_TransparentForMouseEvents)

        def position_model_checkmark():
            try:
                if model_checkmark:
                    model_checkmark.setGeometry(2, 1, 14, 14)
            except RuntimeError:
                pass

        def update_model_checkmark():
            try:
                if show_model_info_cb and model_checkmark:
                    if show_model_info_cb.isChecked():
                        position_model_checkmark()
                        model_checkmark.show()
                    else:
                        model_checkmark.hide()
            except RuntimeError:
                pass

        show_model_info_cb.stateChanged.connect(update_model_checkmark)

        def safe_init_model_checkmark():
            try:
                position_model_checkmark()
                update_model_checkmark()
            except RuntimeError:
                pass

        QTimer.singleShot(0, safe_init_model_checkmark)
        title_layout.addWidget(show_model_info_cb)
        
        # ── Glossary Progress button ──
        # Find the glossary progress file based on automapping settings
        def _glossary_progress_search_dirs(base):
            """Return likely glossary progress locations, newest per-book layout first."""
            search_dirs = []
            seen = set()

            def _add(path):
                if not path:
                    return
                path = os.path.abspath(path)
                key = os.path.normcase(path)
                if key not in seen:
                    seen.add(key)
                    search_dirs.append(path)

            def _add_root(root):
                if not root:
                    return
                root = os.path.abspath(root)
                shared = os.path.join(root, 'Glossary')
                try:
                    from glossary_paths import get_book_glossary_dir
                    _add(get_book_glossary_dir(shared, base, create=False))
                except Exception:
                    _add(os.path.join(shared, base))
                _add(shared)
                _add(os.path.join(root, base, 'Glossary'))
                _add(os.path.join(root, base))

            _override_dir = (os.environ.get('OUTPUT_DIRECTORY') or os.environ.get('OUTPUT_DIR'))
            if not _override_dir and hasattr(self, 'config'):
                _override_dir = self.config.get('output_directory')
            if _override_dir:
                _add_root(_override_dir)

            try:
                from translator_gui import _get_app_dir
                _app_dir = _get_app_dir()
            except Exception:
                _app_dir = os.getcwd()
            _add_root(_app_dir)

            if hasattr(self, 'base_dir'):
                _add_root(self.base_dir)

            return search_dirs

        def _find_progress_in_dir(directory, progress_name):
            candidate = os.path.join(directory, progress_name)
            if os.path.isfile(candidate):
                return candidate
            generic = os.path.join(directory, 'glossary_progress.json')
            if os.path.isfile(generic):
                return generic
            if os.path.basename(directory).lower() != 'glossary':
                try:
                    matches = [
                        os.path.join(directory, name)
                        for name in os.listdir(directory)
                        if name.lower().endswith('_glossary_progress.json')
                    ]
                    if matches:
                        return max(matches, key=lambda path: os.path.getmtime(path))
                except Exception:
                    pass
            return None

        def _find_glossary_progress_file():
            """Locate the glossary progress file for the current EPUB."""
            try:
                base = os.path.splitext(os.path.basename(file_path))[0]
                progress_name = f"{base}_glossary_progress.json"
                for d in _glossary_progress_search_dirs(base):
                    if not os.path.isdir(d):
                        continue
                    found = _find_progress_in_dir(d, progress_name)
                    if found:
                        return found
            except Exception:
                pass
            return None
        
        glossary_progress_btn = QPushButton("📊 Glossary Progress")
        glossary_progress_btn.setCursor(Qt.PointingHandCursor)
        glossary_progress_btn.setStyleSheet("""
            QPushButton {
                background-color: #2d6a4f;
                color: #d8f3dc;
                border: 1px solid #40916c;
                border-radius: 4px;
                padding: 3px 10px;
                font-size: 9pt;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #40916c;
                border-color: #52b788;
            }
        """)
        # Always show; the dialog has an empty-state panel until progress exists.
        _initial_glossary_progress_file = _find_glossary_progress_file()
        glossary_progress_btn.setVisible(True)
        if _initial_glossary_progress_file:
            glossary_progress_btn.setToolTip(f"View glossary extraction progress\n{_initial_glossary_progress_file}")
        else:
            glossary_progress_btn.setToolTip("View glossary extraction progress")
        def _find_gp_for_file(fp):
            """Locate the glossary progress file for a given EPUB path."""
            try:
                base = os.path.splitext(os.path.basename(fp))[0]
                progress_name = f"{base}_glossary_progress.json"
                for d in _glossary_progress_search_dirs(base):
                    if not os.path.isdir(d):
                        continue
                    found = _find_progress_in_dir(d, progress_name)
                    if found:
                        return found
            except Exception:
                pass
            return None

        def _bool_setting(value):
            if isinstance(value, str):
                return value.strip().lower() in ('1', 'true', 'yes', 'on')
            return bool(value)

        def _glossary_refinement_settings_enabled():
            try:
                cfg = getattr(self, 'config', {}) or {}
                if _bool_setting(cfg.get('glossary_refinement_enabled', False)):
                    return True
                return os.getenv('GLOSSARY_REFINEMENT_ENABLED', '').strip().lower() in ('1', 'true', 'yes', 'on')
            except Exception:
                return False

        def _glossary_refinement_expected_entries():
            if not _glossary_refinement_settings_enabled():
                return {}
            cfg = getattr(self, 'config', {}) or {}
            custom_types = getattr(self, 'custom_entry_types', None) or cfg.get('custom_entry_types', {}) or {}
            if not isinstance(custom_types, dict) or not custom_types:
                custom_types = {
                    'character': {'enabled': True},
                    'terms': {'enabled': True},
                }

            active_types = []
            for type_name, type_cfg in custom_types.items():
                if isinstance(type_cfg, dict) and not type_cfg.get('enabled', True):
                    continue
                type_name = str(type_name or '').strip()
                if type_name:
                    active_types.append(type_name)

            type_mode = str(cfg.get('glossary_refinement_type_mode', 'all') or 'all').lower()
            if type_mode == 'selected':
                selected = cfg.get('glossary_refinement_selected_types', [])
                if isinstance(selected, str):
                    selected = [t.strip() for t in selected.split(',') if t.strip()]
                selected_lc = {str(t).strip().lower() for t in selected if str(t).strip()}
                active_types = [t for t in active_types if t.lower() in selected_lc]

            if not active_types:
                return {}

            chunking_mode = str(cfg.get('glossary_refinement_chunking_mode', 'separate') or 'separate').lower()
            if chunking_mode in ('all', 'all_types', 'all_in_one'):
                entry_type = 'all selected entry types'
                return {
                    f"all::{','.join(active_types)}": {
                        'entry_type': entry_type,
                        'status': 'not_refined',
                        'chunking_mode': 'all',
                    }
                }

            return {
                f"type::{entry_type}": {
                    'entry_type': entry_type,
                    'status': 'not_refined',
                    'chunking_mode': 'separate',
                }
                for entry_type in active_types
            }

        if _glossary_refinement_settings_enabled():
            glossary_progress_btn.setVisible(True)
        
        def _build_gp_panel(fp, gp_path, parent_widget):
            """Build a glossary progress panel for a single EPUB. Returns (panel_widget, refresh_func)."""
            from PySide6.QtWidgets import QStackedWidget, QComboBox
            
            def _gp_load_progress_dict(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        loaded = json.load(f)
                except Exception as e:
                    print(f"⚠️ Could not load glossary progress file {path}: {e}")
                    return {}
                if isinstance(loaded, dict):
                    return loaded
                print(f"⚠️ Glossary progress file has legacy non-dict shape: {type(loaded).__name__}")
                return {}

            def _gp_int_list(values):
                if values is None:
                    return []
                if isinstance(values, dict):
                    values = values.keys()
                elif isinstance(values, (str, int, float)):
                    values = [values]
                result = []
                seen = set()
                try:
                    iterator = iter(values)
                except TypeError:
                    iterator = iter([values])
                for value in iterator:
                    if isinstance(value, dict):
                        value = value.get('chapter_index', value.get('actual_num', value.get('chapter_num')))
                    try:
                        ivalue = int(value)
                    except (TypeError, ValueError):
                        continue
                    if ivalue not in seen:
                        seen.add(ivalue)
                        result.append(ivalue)
                return result

            def _gp_safe_int(value, default=0):
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return default

            gp_data = _gp_load_progress_dict(gp_path)

            completed_indices = _gp_int_list(gp_data.get('completed', []))
            failed_indices = _gp_int_list(gp_data.get('failed', []))
            merged_indices = _gp_int_list(gp_data.get('merged_indices', []))
            book_title = gp_data.get('book_title', '')

            def _gp_qa_issue_map(_d):
                if not isinstance(_d, dict):
                    _d = {}
                issues = {}

                def _add(idx, values):
                    try:
                        key = int(idx)
                    except (TypeError, ValueError):
                        return
                    if isinstance(values, str):
                        values = [values]
                    if not isinstance(values, list):
                        return
                    bucket = issues.setdefault(key, [])
                    for value in values:
                        text = str(value).strip()
                        if text and text not in bucket:
                            bucket.append(text)

                raw_map = _d.get('qa_issues_found', {})
                if isinstance(raw_map, dict):
                    for idx, values in raw_map.items():
                        if isinstance(values, dict):
                            values = values.get('qa_issues_found') or values.get('issues') or []
                        mapped_idx = _gp_index_for_progress_value(idx, _d)
                        _add(mapped_idx if mapped_idx is not None else idx, values)

                chapters = _d.get('chapters', {})
                if isinstance(chapters, dict):
                    for key, info in chapters.items():
                        if not isinstance(info, dict):
                            continue
                        idx = _gp_index_for_entry(info, key, _d)
                        _add(idx, info.get('qa_issues_found', []))
                return issues

            def _gp_filename_chapter_num(fname):
                import re as _re_gp_num
                nums = _re_gp_num.findall(r'[0-9]+', os.path.splitext(str(fname or ""))[0])
                if nums:
                    try:
                        return int(nums[-1])
                    except (TypeError, ValueError):
                        return None
                return None

            def _gp_display_chapter_num(ci, fname):
                ch_num = _gp_filename_chapter_num(fname)
                if ch_num is not None:
                    return ch_num
                return 0

            def _gp_filename_keys(name):
                """Normalize a filename into a set of lowercase lookup keys."""
                base = os.path.basename(str(name or ""))
                if not base:
                    return set()
                stem = os.path.splitext(base)[0]
                keys = {base.lower(), stem.lower()}
                if stem.lower().startswith('response_'):
                    keys.add(stem[9:].lower())
                return {k for k in keys if k}

            def _rebuild_reverse_lookups():
                """Rebuild O(1) lookup dicts from chapter_map. Call after chapter_map changes."""
                cmap = panel_state.get('chapter_map') or {}
                # filename key -> chapter index (first wins)
                fk_to_ci = {}
                for ci, mapped_name in cmap.items():
                    for key in _gp_filename_keys(mapped_name):
                        fk_to_ci.setdefault(key, ci)
                panel_state['_fk_to_ci'] = fk_to_ci
                # actual_num (from filename) -> list of chapter indices
                anum_to_ci = {}
                for ci, fname in cmap.items():
                    num = _gp_filename_chapter_num(fname)
                    if num is not None:
                        anum_to_ci.setdefault(num, []).append(ci)
                panel_state['_anum_to_ci'] = anum_to_ci
                # auto-completed (cover pages) — cached set
                auto_comp = set()
                for ci, fname in cmap.items():
                    stem = os.path.splitext(os.path.basename(str(fname or "")))[0].lower()
                    if stem == 'cover':
                        auto_comp.add(ci)
                panel_state['_auto_completed'] = auto_comp

            def _gp_auto_completed_indices():
                return panel_state.get('_auto_completed') or set()

            def _gp_index_for_actual_num(actual_num, _d=None):
                try:
                    actual_num = int(actual_num)
                except (TypeError, ValueError):
                    return None
                # O(1) reverse lookup from cached dict
                matches = list(panel_state.get('_anum_to_ci', {}).get(actual_num, []))
                if not matches:
                    chapter_numbers = (_d or gp_data).get('chapter_numbers', {})
                    if isinstance(chapter_numbers, dict):
                        for ci, num in chapter_numbers.items():
                            try:
                                if int(num) == actual_num:
                                    matches.append(int(ci))
                            except (TypeError, ValueError):
                                pass
                if len(matches) == 1:
                    return matches[0]
                return None

            def _gp_index_for_entry(info, key=None, _d=None):
                if not isinstance(info, dict):
                    info = {}

                had_filename_anchor = False
                fk_to_ci = panel_state.get('_fk_to_ci') or {}
                for fname_key in ('output_file', 'original_basename', 'chapter_file', 'source_filename', 'filename'):
                    fname = os.path.basename(str(info.get(fname_key, "") or ""))
                    if not fname:
                        continue
                    had_filename_anchor = True
                    # O(1) lookup via reverse dict instead of scanning chapter_map
                    for k in _gp_filename_keys(fname):
                        if k in fk_to_ci:
                            return fk_to_ci[k]
                if had_filename_anchor:
                    return None

                for num_key in ('actual_num', 'chapter_num'):
                    ci = _gp_index_for_actual_num(info.get(num_key), _d)
                    if ci is not None:
                        return ci

                if key is not None and 'chapter_index' not in info:
                    ci = _gp_index_for_actual_num(key, _d)
                    if ci is not None:
                        return ci

                try:
                    return int(info.get('chapter_index', key))
                except (TypeError, ValueError):
                    return None

            def _gp_index_for_progress_value(value, _d):
                if not isinstance(_d, dict):
                    _d = {}
                try:
                    ivalue = int(value)
                except (TypeError, ValueError):
                    return None
                if str(_d.get('indexing', '')).lower() == 'chapter_index_zero_based':
                    return ivalue
                positions = _d.get('chapter_positions', {})
                if isinstance(positions, dict) and str(ivalue) in positions:
                    return ivalue
                ci = _gp_index_for_actual_num(ivalue, _d)
                return ci if ci is not None else ivalue

            def _gp_sets(_d):
                if not isinstance(_d, dict):
                    _d = {}

                def _index_set(values):
                    result = set()
                    for value in _gp_int_list(values):
                        ci = _gp_index_for_progress_value(value, _d)
                        if ci is not None:
                            result.add(ci)
                    return result

                comp = set()
                fail = set()
                merg = set()
                chapters = _d.get('chapters', {})
                used_chapter_entries = False
                if isinstance(chapters, dict):
                    for key, info in chapters.items():
                        if not isinstance(info, dict):
                            continue
                        ci = _gp_index_for_entry(info, key, _d)
                        if ci is None:
                            continue
                        used_chapter_entries = True
                        status = str(info.get('status', '')).lower()
                        if status in ('failed', 'qa_failed', 'error'):
                            fail.add(ci)
                        elif status == 'merged':
                            merg.add(ci)
                        elif status == 'completed':
                            comp.add(ci)
                if not used_chapter_entries:
                    comp = _index_set(_d.get('completed', []))
                    fail = _index_set(_d.get('failed', []))
                    merg = _index_set(_d.get('merged_indices', []))
                comp |= _gp_auto_completed_indices()
                # Failed should win over completed in the UI.
                comp -= fail
                return comp, fail, merg

            def _gp_in_progress_set(_d, _precomputed_sets=None):
                if not isinstance(_d, dict):
                    _d = {}
                result = set()
                chapters = _d.get('chapters', {})
                used_chapter_entries = False
                if isinstance(chapters, dict):
                    for key, info in chapters.items():
                        if not isinstance(info, dict):
                            continue
                        status = str(info.get('status', '')).lower()
                        if status:
                            used_chapter_entries = True
                        if status != 'in_progress':
                            continue
                        ci = _gp_index_for_entry(info, key, _d)
                        if ci is not None:
                            result.add(ci)
                if not used_chapter_entries:
                    for value in _gp_int_list(_d.get('in_progress', [])):
                        ci = _gp_index_for_progress_value(value, _d)
                        if ci is not None:
                            result.add(ci)
                if _precomputed_sets:
                    comp, fail, merg = _precomputed_sets
                else:
                    comp, fail, merg = _gp_sets(_d)
                return result - comp - fail - merg

            def _gp_status_cache(_d):
                comp, fail, merg = _gp_sets(_d)
                in_prog = _gp_in_progress_set(_d, _precomputed_sets=(comp, fail, merg))
                issues = _gp_qa_issue_map(_d)
                qa_failed = set()
                chapters = _d.get('chapters', {}) if isinstance(_d, dict) else {}
                if isinstance(chapters, dict):
                    for key, info in chapters.items():
                        if not isinstance(info, dict):
                            continue
                        if str(info.get('status', '')).lower() != 'qa_failed':
                            continue
                        ci = _gp_index_for_entry(info, key, _d)
                        if ci is not None:
                            qa_failed.add(ci)
                return {
                    'completed': comp,
                    'failed': fail,
                    'merged': merg,
                    'in_progress': in_prog,
                    'issues': issues,
                    'qa_failed': qa_failed,
                }

            def _gp_status_for(ci, _d, cache=None):
                if not isinstance(_d, dict):
                    _d = {}
                cache = cache or _gp_status_cache(_d)
                comp = cache['completed']
                fail = cache['failed']
                merg = cache['merged']
                in_prog = cache['in_progress']
                issues = cache['issues']
                if ci in fail:
                    return ('qa_failed' if issues.get(ci) or ci in cache['qa_failed'] else 'failed'), issues.get(ci, [])
                if ci in merg:
                    return 'merged', []
                if ci in comp:
                    return 'completed', []
                if ci in in_prog:
                    return 'in_progress', []
                return 'not_completed', []

            def _gp_model_for(ci, _d):
                if not isinstance(_d, dict):
                    _d = {}
                chapters = _d.get('chapters', {})
                if isinstance(chapters, dict):
                    for key, info in chapters.items():
                        if not isinstance(info, dict):
                            continue
                        if _gp_index_for_entry(info, key, _d) != ci:
                            continue
                        model_name = str(info.get('model_name') or info.get('model') or '').strip()
                        if model_name:
                            return model_name
                return '(model unknown)'

            def _gp_entry_for(ci, _d):
                if not isinstance(_d, dict):
                    return {}
                chapters = _d.get('chapters', {})
                if isinstance(chapters, dict):
                    for key, info in chapters.items():
                        if not isinstance(info, dict):
                            continue
                        if _gp_index_for_entry(info, key, _d) == ci:
                            return info
                return {}

            def _gp_display_for(ci, fname, _d, cache=None):
                opf_pos = (panel_state.get('spine_index_map') or {}).get(ci, ci + 1)
                ch_num = _gp_display_chapter_num(ci, fname)
                status, issues = _gp_status_for(ci, _d, cache)
                model_name = _gp_model_for(ci, _d)
                icons = {
                    'completed': '\u2705',
                    'failed': '\u274c',
                    'qa_failed': '\u274c',
                    'merged': '\U0001f517',
                    'in_progress': '\U0001f504',
                    'not_completed': '\u2b1c',
                }
                icon = icons.get(status) or '\u2b1c'
                status_label = status.replace('_', ' ').title()
                entry = _gp_entry_for(ci, _d)
                if status == 'completed' and str(entry.get('refinement_status') or '').lower().strip() in ('refined', 'completed'):
                    status_label = f"{status_label} ⭐"
                display = f"[{opf_pos:03d}] Ch.{ch_num:03d} | {icon} {status_label:14s} | {fname} -> {model_name}"
                if issues:
                    issues_display = ', '.join(issues[:2])
                    if len(issues) > 2:
                        issues_display += f' (+{len(issues)-2} more)'
                    display += f" | {issues_display}"
                return display, status

            def _gp_refinement_rows(_d):
                refinement = _d.get('refinement', {}) if isinstance(_d, dict) else {}
                if not isinstance(refinement, dict):
                    refinement = {}
                refinement = dict(refinement)
                for expected_key, expected_info in _glossary_refinement_expected_entries().items():
                    refinement.setdefault(expected_key, expected_info)
                rows = []
                for key, info in sorted(refinement.items()):
                    if not isinstance(info, dict):
                        continue
                    entry_type = str(info.get('entry_type') or key.replace('type::', '')).strip() or 'entry type'
                    status = str(info.get('status') or 'unknown').lower()
                    before = info.get('entry_count_before')
                    after = info.get('entry_count_after')
                    total_chunks = info.get('total_chunks')
                    completed_chunks = info.get('completed_chunks')
                    model_name = str(info.get('model_name') or info.get('model') or '').strip() or '(model unknown)'
                    detail = ""
                    if before is not None and after is not None:
                        detail = f" | {before} -> {after} entries"
                    elif total_chunks:
                        detail = f" | chunks {completed_chunks or 0}/{total_chunks}"
                    icon_map = {
                        'completed': '\u2705',
                        'failed': '\u274c',
                        'qa_failed': '\u274c',
                        'in_progress': '\U0001f504',
                        'not_refined': '\u2728',
                    }
                    icon = icon_map.get(status, '\u2b1c')
                    display = f"Refinement | {icon} {status.replace('_', ' ').title():14s} | {entry_type} -> {model_name}{detail}"
                    rows.append((key, display, status))
                return rows

            def _gp_refinement_status_counts(_d):
                counts = {}
                for _key, _display, status in _gp_refinement_rows(_d):
                    status = str(status or 'unknown').lower().replace(' ', '_')
                    counts[status] = counts.get(status, 0) + 1
                return counts

            def _gp_color_for(status):
                if status == 'completed':
                    return '#27ae60'
                if status == 'merged':
                    return '#17a2b8'
                if status == 'in_progress':
                    return '#f59e0b'
                if status in ('failed', 'qa_failed'):
                    return '#e74c3c'
                return '#5a9fd4'

            def _gp_restore_in_progress_entry(info):
                if not isinstance(info, dict):
                    return None
                previous_status = str(info.get('previous_status', '') or '').lower()
                previous_entry = info.get('previous_progress_entry')
                if isinstance(previous_entry, dict):
                    restored = dict(previous_entry)
                    restored_status = str(restored.get('status', previous_status) or previous_status).lower()
                    if restored_status and restored_status not in ('in_progress', 'not_completed', 'not translated', 'not_translated'):
                        restored.pop('previous_status', None)
                        restored.pop('previous_progress_entry', None)
                        return restored
                if previous_status in ('qa_failed', 'failed', 'error', 'pending', 'merged', 'completed'):
                    restored = dict(info)
                    restored['status'] = 'failed' if previous_status == 'error' else previous_status
                    restored.pop('previous_status', None)
                    restored.pop('previous_progress_entry', None)
                    restored.pop('previous_status_unknown', None)
                    return restored
                if info.get('previous_status_unknown'):
                    restored = dict(info)
                    restored['status'] = 'failed'
                    restored.pop('previous_status', None)
                    restored.pop('previous_progress_entry', None)
                    restored.pop('previous_status_unknown', None)
                    return restored
                if previous_status in ('not_completed', 'not translated', 'not_translated'):
                    return None
                if info.get('output_file'):
                    restored = dict(info)
                    restored['status'] = 'failed'
                    restored.pop('previous_status', None)
                    restored.pop('previous_progress_entry', None)
                    restored.pop('previous_status_unknown', None)
                    return restored
                return None

            # Lightweight spine reader - returns (chapter_map, total_chapters, spine_index_map)
            def _read_spine_map(epub_path, translate_special):
                """Read OPF spine and return (chapter_map, total_chapters, spine_index_map)."""
                cmap = {}
                spine_index_map = {}
                if not (epub_path.lower().endswith('.epub') and os.path.exists(epub_path)):
                    return cmap, 0, spine_index_map
                try:
                    import zipfile
                    from xml.etree import ElementTree as ET
                    with zipfile.ZipFile(epub_path, 'r') as zf:
                        opf_path = None
                        try:
                            container = ET.fromstring(zf.read('META-INF/container.xml'))
                            ns = {'c': 'urn:oasis:names:tc:opendocument:xmlns:container'}
                            rootfile = container.find('.//c:rootfile', ns)
                            if rootfile is not None:
                                opf_path = rootfile.get('full-path')
                        except Exception:
                            opf_path = next((n for n in zf.namelist() if n.endswith('.opf')), None)
                        
                        if not opf_path:
                            return cmap, 0, spine_index_map
                        
                        opf_xml = ET.fromstring(zf.read(opf_path))
                        opf_ns = {'opf': 'http://www.idpf.org/2007/opf'}
                        
                        id_to_href = {}
                        html_types = {'application/xhtml+xml', 'text/html', 'application/html+xml'}
                        for item in opf_xml.findall('.//opf:manifest/opf:item', opf_ns):
                            mid = item.get('id', '')
                            mtype = item.get('media-type', '')
                            href = item.get('href', '')
                            if mtype in html_types:
                                id_to_href[mid] = href
                        
                        spine_hrefs = []
                        for itemref in opf_xml.findall('.//opf:spine/opf:itemref', opf_ns):
                            idref = itemref.get('idref', '')
                            if idref in id_to_href:
                                spine_hrefs.append(id_to_href[idref])
                        
                        _kw_env = os.environ.get('SPECIAL_FILE_KEYWORDS', '')
                        special_keywords = [k.strip().lower() for k in _kw_env.split(',') if k.strip()] if _kw_env else [
                            'title', 'toc', 'copyright', 'preface', 'nav',
                            'message', 'notice', 'colophon', 'dedication', 'epigraph',
                            'foreword', 'acknowledgment', 'author', 'appendix',
                            'bibliography'
                        ]
                        _exact_env = os.environ.get('SPECIAL_FILE_EXACT', '')
                        special_exact = [k.strip().lower() for k in _exact_env.split(',') if k.strip()] if _exact_env else ['index', 'glossary', 'glossary_extension']
                        import re as _re_spine
                        ci = 0
                        for opf_pos, href in enumerate(spine_hrefs, start=1):
                            basename = os.path.basename(href)
                            if not translate_special:
                                name_noext = os.path.splitext(basename)[0]
                                name_lower = name_noext.lower()
                                name_stripped = _re_spine.sub(r'\d+$', '', name_lower).rstrip('_- ')
                                # Exact match: these are special only when the basename matches exactly
                                if name_lower in special_exact:
                                    continue
                                if any(kw in name_lower for kw in special_keywords):
                                    has_digits = bool(_re_spine.search(r'\d', name_noext))
                                    if not has_digits or any(kw == name_stripped or kw in name_stripped for kw in special_keywords):
                                        continue
                            cmap[ci] = basename
                            spine_index_map[ci] = opf_pos
                            ci += 1
                        return cmap, ci, spine_index_map
                except Exception:
                    return cmap, 0, spine_index_map
            
            # Mutable state so refresh can update chapter_map when toggle changes
            _ts_init = os.getenv('TRANSLATE_SPECIAL_FILES', '0') == '1'
            _cmap_init, _total_init, _spine_idx_init = _read_spine_map(fp, _ts_init)
            
            if _total_init == 0:
                _total_init = _gp_safe_int(gp_data.get('chapter_count'), 0)
                if _total_init <= 0:
                    chapter_filenames = gp_data.get('chapter_filenames', {})
                    if isinstance(chapter_filenames, dict) and chapter_filenames:
                        _total_init = max((int(k) for k in chapter_filenames.keys() if str(k).isdigit()), default=-1) + 1
                if _total_init <= 0:
                    _idx_values = []
                    for _values in (completed_indices, failed_indices, merged_indices):
                        _idx_values.extend(_gp_int_list(_values))
                    _total_init = (max(_idx_values) + 1) if _idx_values else 1
            
            # Store in mutable dict so closures can update
            panel_state = {
                'chapter_map': _cmap_init,
                'spine_index_map': _spine_idx_init,
                'total': _total_init,
                'translate_special': _ts_init,
                'populate_generation': 0,
            }
            if not panel_state['chapter_map']:
                chapter_filenames = gp_data.get('chapter_filenames', {})
                if isinstance(chapter_filenames, dict):
                    try:
                        panel_state['chapter_map'] = {
                            int(k): os.path.basename(str(v or ""))
                            for k, v in chapter_filenames.items()
                            if str(k).lstrip('-').isdigit() and v
                        }
                        panel_state['spine_index_map'] = {
                            int(k): int(k) + 1
                            for k in chapter_filenames.keys()
                            if str(k).lstrip('-').isdigit()
                        }
                    except Exception:
                        panel_state['chapter_map'] = {}
                        panel_state['spine_index_map'] = {}
            
            # Build O(1) reverse lookups from chapter_map
            _rebuild_reverse_lookups()
            
            # Track file mtime for dirty-checking on refresh
            try:
                panel_state['_last_mtime'] = os.path.getmtime(gp_path) if os.path.isfile(gp_path) else 0
            except OSError:
                panel_state['_last_mtime'] = 0
            
            panel = QWidget(parent_widget)
            p_layout = QVBoxLayout(panel)
            p_layout.setContentsMargins(4, 4, 4, 4)
            
            if book_title:
                bt_label = QLabel(f"📖 {book_title}")
                bt_label.setStyleSheet("color: #94a3b8; font-style: italic; font-size: 10pt;")
                p_layout.addWidget(bt_label)
            else:
                bt_label = None
            
            # Stats row (clickable)
            _comp_set_init, _fail_set_init, _merg_set_init = _gp_sets(gp_data)
            _in_prog_set_init = _gp_in_progress_set(gp_data, _precomputed_sets=(_comp_set_init, _fail_set_init, _merg_set_init))
            # Completed count excludes chapters that are also merged
            n_completed = len(_comp_set_init - _merg_set_init - _fail_set_init)
            n_failed = len(_fail_set_init)
            n_merged = len(_merg_set_init)
            n_in_progress = len(_in_prog_set_init)
            n_remaining = max(0, panel_state['total'] - len(_comp_set_init | _fail_set_init | _merg_set_init | _in_prog_set_init))
            n_not_refined = _gp_refinement_status_counts(gp_data).get('not_refined', 0)
            
            gp_stats_frame = QWidget()
            gp_stats_layout = QHBoxLayout(gp_stats_frame)
            gp_stats_layout.setContentsMargins(0, 5, 0, 5)
            gp_stats_font = QFont('Arial', 10)
            
            lbl_total = QLabel(f"Total: {panel_state['total']} | ")
            lbl_total.setFont(gp_stats_font)
            gp_stats_layout.addWidget(lbl_total)
            
            lbl_gp_completed = QLabel(f"✅ Completed: {n_completed} | ")
            lbl_gp_completed.setFont(gp_stats_font)
            lbl_gp_completed.setStyleSheet("color: #27ae60;")
            lbl_gp_completed.setCursor(Qt.PointingHandCursor)
            gp_stats_layout.addWidget(lbl_gp_completed)

            lbl_gp_in_progress = QLabel(f"🔄 In Progress: {n_in_progress} | ")
            lbl_gp_in_progress.setFont(gp_stats_font)
            lbl_gp_in_progress.setStyleSheet("color: #f59e0b;")
            lbl_gp_in_progress.setCursor(Qt.PointingHandCursor)
            gp_stats_layout.addWidget(lbl_gp_in_progress)
            
            lbl_gp_failed = QLabel(f"❌ Failed: {n_failed} | ")
            lbl_gp_failed.setFont(gp_stats_font)
            lbl_gp_failed.setStyleSheet("color: #e74c3c;")
            lbl_gp_failed.setCursor(Qt.PointingHandCursor)
            gp_stats_layout.addWidget(lbl_gp_failed)
            
            lbl_gp_merged = QLabel(f"🔗 Merged: {n_merged} | ")
            lbl_gp_merged.setFont(gp_stats_font)
            lbl_gp_merged.setStyleSheet("color: #17a2b8;")
            lbl_gp_merged.setCursor(Qt.PointingHandCursor)
            gp_stats_layout.addWidget(lbl_gp_merged)
            if n_merged == 0:
                lbl_gp_merged.setVisible(False)
            
            lbl_gp_remaining = QLabel(f"⬜ Not Translated: {n_remaining}{' | ' if n_not_refined else ''}")
            lbl_gp_remaining.setFont(gp_stats_font)
            lbl_gp_remaining.setStyleSheet("color: #5a9fd4;")
            lbl_gp_remaining.setCursor(Qt.PointingHandCursor)
            gp_stats_layout.addWidget(lbl_gp_remaining)

            lbl_gp_not_refined = QLabel(f"✨ Not Refined: {n_not_refined}")
            lbl_gp_not_refined.setFont(gp_stats_font)
            lbl_gp_not_refined.setStyleSheet("color: #5a9fd4;")
            lbl_gp_not_refined.setCursor(Qt.PointingHandCursor)
            lbl_gp_not_refined.setVisible(n_not_refined > 0)
            gp_stats_layout.addWidget(lbl_gp_not_refined)
            
            gp_stats_layout.addStretch()
            p_layout.addWidget(gp_stats_frame)
            
            # Chapter list
            gp_listbox = QListWidget()
            self._apply_compact_inline_list_style(gp_listbox, QFont('Courier', 10))
            gp_listbox.setContextMenuPolicy(Qt.CustomContextMenu)
            gp_listbox.setSelectionMode(QListWidget.ExtendedSelection)
            
            completed_set, failed_set, merged_set = _gp_sets(gp_data)
            
            chapter_map = panel_state['chapter_map']
            total_epub_chapters = 0
            
            for ci in range(total_epub_chapters):
                fname = chapter_map.get(ci, f'chapter {ci + 1}')
                ch_num = _gp_display_chapter_num(ci, fname)
                
                if ci in merged_set:
                    icon, status, color = '🔗', 'merged', '#17a2b8'
                elif ci in completed_set:
                    icon, status, color = '✅', 'completed', '#27ae60'
                elif ci in failed_set:
                    icon, status, color = '❌', 'failed', '#e74c3c'
                else:
                    icon, status, color = '⬜', 'not_completed', '#5a9fd4'
                
                display = f"Ch.{ch_num:03d} | {icon} {status.replace('_', ' ').title():14s} | {fname}"
                display, status = _gp_display_for(ci, fname, gp_data)
                color = _gp_color_for(status)
                item = QListWidgetItem(display)
                item.setForeground(QColor(color))
                item.setData(Qt.UserRole, status)
                item.setData(Qt.UserRole + 1, ci)  # Store chapter index for deletion
                self._add_compact_inline_list_item(gp_listbox, item)

            def _refresh_refinement_rows(_d, keep_updates_disabled=False):
                selected_ref_keys = {
                    it.data(Qt.UserRole + 3)
                    for it in gp_listbox.selectedItems()
                    if it and it.data(Qt.UserRole + 3)
                }
                if not keep_updates_disabled:
                    gp_listbox.setUpdatesEnabled(False)
                try:
                    for row in range(gp_listbox.count() - 1, -1, -1):
                        item = gp_listbox.item(row)
                        if item and item.data(Qt.UserRole + 3):
                            gp_listbox.takeItem(row)

                    for ref_key, ref_display, ref_status in _gp_refinement_rows(_d):
                        item = QListWidgetItem(ref_display)
                        item.setForeground(QColor(_gp_color_for(ref_status)))
                        item.setData(Qt.UserRole, ref_status)
                        item.setData(Qt.UserRole + 1, None)
                        item.setData(Qt.UserRole + 3, ref_key)
                        self._add_compact_inline_list_item(gp_listbox, item)
                        if ref_key in selected_ref_keys:
                            item.setSelected(True)
                finally:
                    if not keep_updates_disabled:
                        gp_listbox.setUpdatesEnabled(True)
                        gp_listbox.viewport().update()
            
            def _populate_gp_listbox(_d, chunk_size=150):
                panel_state['populate_generation'] = panel_state.get('populate_generation', 0) + 1
                generation = panel_state['populate_generation']
                cache = _gp_status_cache(_d)
                gp_listbox.clear()
                gp_listbox.setUpdatesEnabled(False)
                total = panel_state['total']
                chapter_map = panel_state['chapter_map']
                state = {'ci': 0}

                def _add_chunk():
                    if generation != panel_state.get('populate_generation'):
                        return
                    start_ci = state['ci']
                    end_ci = min(start_ci + chunk_size, total)
                    for ci in range(start_ci, end_ci):
                        fname = chapter_map.get(ci, f'chapter {ci + 1}')
                        display, status = _gp_display_for(ci, fname, _d, cache)
                        item = QListWidgetItem(display)
                        item.setForeground(QColor(_gp_color_for(status)))
                        item.setData(Qt.UserRole, status)
                        item.setData(Qt.UserRole + 1, ci)
                        self._add_compact_inline_list_item(gp_listbox, item)
                        if panel_state.get('select_all_visible'):
                            item.setSelected(not item.isHidden())
                    state['ci'] = end_ci
                    if end_ci < total:
                        QTimer.singleShot(0, _add_chunk)
                    else:
                        _refresh_refinement_rows(_d, keep_updates_disabled=True)
                        gp_listbox.setUpdatesEnabled(True)
                        gp_listbox.viewport().update()

                QTimer.singleShot(0, _add_chunk)

            _populate_gp_listbox(gp_data)

            # Helper to refresh stats labels from a loaded progress dict
            def _refresh_stats_from_dict(_d):
                _comp2, _fail2, _merg2 = _gp_sets(_d)
                _prog2 = _gp_in_progress_set(_d, _precomputed_sets=(_comp2, _fail2, _merg2))
                _total = panel_state['total']
                _not_refined2 = _gp_refinement_status_counts(_d).get('not_refined', 0)
                lbl_total.setText(f"Total: {_total} | ")
                lbl_gp_completed.setText(f"✅ Completed: {len(_comp2 - _merg2)} | ")
                lbl_gp_in_progress.setText(f"🔄 In Progress: {len(_prog2)} | ")
                lbl_gp_in_progress.setVisible(True)
                lbl_gp_failed.setText(f"❌ Failed: {len(_fail2)} | ")
                lbl_gp_merged.setText(f"🔗 Merged: {len(_merg2)} | ")
                lbl_gp_merged.setVisible(len(_merg2) > 0)
                lbl_gp_remaining.setText(f"⬜ Not Translated: {max(0, _total - len(_comp2 | _fail2 | _merg2 | _prog2))}{' | ' if _not_refined2 else ''}")
                lbl_gp_not_refined.setText(f"✨ Not Refined: {_not_refined2}")
                lbl_gp_not_refined.setVisible(_not_refined2 > 0)
            
            # Right-click context menu to delete entries from progress
            def _gp_context_menu(pos):
                # Gather selected items that are deletable
                clicked_item = gp_listbox.itemAt(pos)
                if clicked_item is not None and not clicked_item.isSelected():
                    gp_listbox.clearSelection()
                    clicked_item.setSelected(True)
                selected = gp_listbox.selectedItems()
                deletable_statuses = ('completed', 'merged', 'in_progress', 'failed', 'qa_failed', 'not_refined')
                targets = []
                for it in selected:
                    if it.data(Qt.UserRole) not in deletable_statuses:
                        continue
                    refinement_key = it.data(Qt.UserRole + 3)
                    chapter_index = it.data(Qt.UserRole + 1)
                    if refinement_key:
                        targets.append((it, ('refinement', refinement_key)))
                    elif chapter_index is not None:
                        targets.append((it, ('chapter', chapter_index)))
                if not targets:
                    return
                
                from PySide6.QtWidgets import QMenu
                menu = QMenu(gp_listbox)
                menu.setStyleSheet(
                    "QMenu { background-color: #2d2d2d; color: white; border: 1px solid #555; }"
                    "QMenu::item:selected { background-color: #c0392b; }"
                )
                
                n = len(targets)
                if n == 1:
                    target_kind, target_value = targets[0][1]
                    status = targets[0][0].data(Qt.UserRole)
                    display_text = targets[0][0].text() or ""
                    if target_kind == 'refinement':
                        chapter_label = display_text.split('|')[-1].strip() or str(target_value)
                    else:
                        label_match = re.search(r'\bCh\.\d+(?:\.\d+)?\b', display_text)
                        chapter_label = label_match.group(0) if label_match else f"Ch.{target_value+1}"
                    action = menu.addAction(f"🗑️ Remove {chapter_label} from progress ({status})")
                else:
                    action = menu.addAction(f"🗑️ Remove {n} chapters from progress")
                
                chosen = menu.exec(gp_listbox.viewport().mapToGlobal(pos))
                if chosen != action:
                    return
                
                # Remove from progress JSON
                try:
                    _rp = _find_gp_for_file(fp)
                    if not _rp or not os.path.isfile(_rp):
                        return
                    _d = _gp_load_progress_dict(_rp)
                    
                    indices_to_remove = set(value for _, (kind, value) in targets if kind == 'chapter')
                    refinement_keys_to_remove = set(value for _, (kind, value) in targets if kind == 'refinement')
                    changed = False
                    if indices_to_remove:
                        removed_indices = _d.get('manual_removed_indices', [])
                        if not isinstance(removed_indices, list):
                            removed_indices = []
                        removed_set = set()
                        for value in removed_indices:
                            mapped_ci = _gp_index_for_progress_value(value, _d)
                            if mapped_ci is not None:
                                removed_set.add(mapped_ci)
                        removed_set.update(indices_to_remove)
                        _d['manual_removed_indices'] = sorted(removed_set)
                        _d['manual_removed_session_id'] = _d.get('progress_session_id')
                        changed = True

                    for key in ('completed', 'failed', 'merged_indices', 'in_progress'):
                        lst = _d.get(key, [])
                        new_lst = []
                        for v in lst:
                            mapped_ci = _gp_index_for_progress_value(v, _d)
                            if not indices_to_remove or mapped_ci not in indices_to_remove:
                                new_lst.append(v)
                        if len(new_lst) != len(lst):
                            _d[key] = new_lst
                            changed = True

                    qa_map = _d.get('qa_issues_found', {})
                    if indices_to_remove and isinstance(qa_map, dict):
                        new_qa_map = {}
                        for k, v in qa_map.items():
                            mapped_ci = _gp_index_for_progress_value(k, _d)
                            keep = mapped_ci not in indices_to_remove if mapped_ci is not None else True
                            if keep:
                                new_qa_map[k] = v
                        if len(new_qa_map) != len(qa_map):
                            _d['qa_issues_found'] = new_qa_map
                            changed = True

                    chapters = _d.get('chapters', {})
                    if indices_to_remove and isinstance(chapters, dict):
                        new_chapters = {}
                        chapters_changed = False
                        for k, v in chapters.items():
                            ci = _gp_index_for_entry(v, k, _d) if isinstance(v, dict) else None
                            keep = ci not in indices_to_remove if ci is not None else True
                            if keep:
                                new_chapters[k] = v
                            else:
                                chapters_changed = True
                                changed = True
                        if chapters_changed or len(new_chapters) != len(chapters):
                            _d['chapters'] = new_chapters
                            changed = True

                    if refinement_keys_to_remove and isinstance(_d.get('refinement'), dict):
                        for ref_key in refinement_keys_to_remove:
                            if ref_key in _d['refinement']:
                                del _d['refinement'][ref_key]
                                changed = True
                    
                    if changed:
                        with open(_rp, 'w', encoding='utf-8') as _f:
                            json.dump(_d, _f, ensure_ascii=False, indent=2)
                        # Update all affected items
                        _cmap = panel_state['chapter_map']
                        for it, (kind, value) in targets:
                            if kind == 'refinement':
                                row = next((r for r in _gp_refinement_rows(_d) if r[0] == value), None)
                                if row:
                                    _rk, display3, restored_status = row
                                    it.setText(display3)
                                    it.setForeground(QColor(_gp_color_for(restored_status)))
                                    it.setData(Qt.UserRole, restored_status)
                                else:
                                    gp_listbox.takeItem(gp_listbox.row(it))
                            else:
                                ci = value
                                fname = _cmap.get(ci, f'chapter {ci + 1}')
                                display3, restored_status = _gp_display_for(ci, fname, _d)
                                it.setText(display3)
                                it.setForeground(QColor(_gp_color_for(restored_status)))
                                it.setData(Qt.UserRole, restored_status)
                        _refresh_stats_from_dict(_d)
                except Exception as e:
                    print(f"⚠️ Error removing chapters from progress: {e}")
            
            gp_listbox.customContextMenuRequested.connect(_gp_context_menu)
            
            # Cycle handler
            def _gp_make_cycle(target_statuses, lb_ref):
                target_statuses = set(target_statuses)
                def _handler(_event=None):
                    lb = lb_ref
                    if not lb:
                        return
                    indices = []
                    for i in range(lb.count()):
                        item = lb.item(i)
                        if not item or item.isHidden():
                            continue
                        status = item.data(Qt.UserRole)
                        if isinstance(status, str):
                            status = status.lower().replace(' ', '_')
                        if status in target_statuses:
                            indices.append(i)
                    if not indices:
                        return
                    selected_rows = [lb.row(item) for item in lb.selectedItems()]
                    current = lb.currentRow()
                    if selected_rows and current not in selected_rows:
                        current = max(selected_rows)
                    nxt = next((i for i in indices if i > current), indices[0])
                    lb.setCurrentRow(nxt, QItemSelectionModel.ClearAndSelect)
                    lb.scrollToItem(lb.item(nxt), QListWidget.PositionAtCenter)
                return _handler
            
            lbl_gp_completed.mousePressEvent = _gp_make_cycle(('completed',), gp_listbox)
            lbl_gp_in_progress.mousePressEvent = _gp_make_cycle(('in_progress',), gp_listbox)
            lbl_gp_failed.mousePressEvent = _gp_make_cycle(('failed', 'qa_failed'), gp_listbox)
            lbl_gp_merged.mousePressEvent = _gp_make_cycle(('merged',), gp_listbox)
            lbl_gp_remaining.mousePressEvent = _gp_make_cycle(('not_completed', 'not_translated', 'no_tts'), gp_listbox)
            lbl_gp_not_refined.mousePressEvent = _gp_make_cycle(('not_refined',), gp_listbox)
            
            p_layout.addWidget(gp_listbox)
            
            # Progress file path + open folder button + open glossary button
            path_row = QHBoxLayout()
            path_label = QLabel(f"📁 {gp_path}")
            path_label.setStyleSheet("color: #666; font-size: 8pt;")
            path_label.setWordWrap(True)
            path_row.addWidget(path_label, stretch=1)

            select_all_btn = QPushButton("Select All")
            select_all_btn.setCursor(Qt.PointingHandCursor)
            select_all_btn.setStyleSheet(
                "QPushButton { background-color: #263445; color: #dbeafe; border: 1px solid #64748b; "
                "border-radius: 3px; padding: 2px 8px; font-size: 8pt; } "
                "QPushButton:hover { background-color: #334155; }"
            )
            select_all_btn.setFixedHeight(22)

            def _select_all_gp_visible(_checked=False):
                panel_state['select_all_visible'] = True
                first_selected = None
                gp_listbox.blockSignals(True)
                try:
                    gp_listbox.clearSelection()
                    for row in range(gp_listbox.count()):
                        item = gp_listbox.item(row)
                        if not item or item.isHidden():
                            continue
                        item.setSelected(True)
                        if first_selected is None:
                            first_selected = row
                    if first_selected is not None:
                        gp_listbox.setCurrentRow(first_selected, QItemSelectionModel.Select)
                finally:
                    gp_listbox.blockSignals(False)
                gp_listbox.viewport().update()

            select_all_btn.clicked.connect(_select_all_gp_visible)
            path_row.addWidget(select_all_btn)
            
            _gp_folder = os.path.dirname(gp_path)
            open_folder_btn = QPushButton("📂 Open Folder")
            open_folder_btn.setCursor(Qt.PointingHandCursor)
            open_folder_btn.setStyleSheet(
                "QPushButton { background-color: #3a3a3a; color: #d8f3dc; border: 1px solid #40916c; "
                "border-radius: 3px; padding: 2px 8px; font-size: 8pt; } "
                "QPushButton:hover { background-color: #40916c; }"
            )
            open_folder_btn.setFixedHeight(22)
            def _open_gp_folder(_checked=False, folder=_gp_folder):
                import subprocess, sys
                if sys.platform == 'win32':
                    os.startfile(folder)
                elif sys.platform == 'darwin':
                    subprocess.Popen(['open', folder])
                else:
                    subprocess.Popen(['xdg-open', folder])
            open_folder_btn.clicked.connect(_open_gp_folder)
            path_row.addWidget(open_folder_btn)
            
            # ── Open Glossary button ──
            open_glossary_btn = QPushButton("✏️ Open Glossary")
            open_glossary_btn.setCursor(Qt.PointingHandCursor)
            open_glossary_btn.setStyleSheet(
                "QPushButton { background-color: #1e3a5f; color: #93c5fd; border: 1px solid #3b82f6; "
                "border-radius: 3px; padding: 2px 8px; font-size: 8pt; } "
                "QPushButton:hover { background-color: #1e40af; }"
            )
            open_glossary_btn.setFixedHeight(22)
            
            def _find_glossary_file(_gp_dir=_gp_folder, _epub_path=fp):
                """Find the glossary file (csv/json/txt) in the same directory as the progress file."""
                import glob
                base = os.path.splitext(os.path.basename(_epub_path))[0]
                # Search priority: book-specific glossary > generic glossary
                for ext in ['.csv', '.json', '.txt', '.md']:
                    for pattern in [
                        os.path.join(_gp_dir, f"{base}_glossary{ext}"),
                        os.path.join(_gp_dir, f"{base}{ext}"),
                        os.path.join(_gp_dir, f"glossary{ext}"),
                    ]:
                        if os.path.isfile(pattern):
                            return pattern
                # Also check parent dir (if progress is in Glossary/ subfolder)
                parent = os.path.dirname(_gp_dir)
                if os.path.basename(_gp_dir).lower() == 'glossary':
                    for ext in ['.csv', '.json', '.txt', '.md']:
                        for pattern in [
                            os.path.join(parent, f"glossary{ext}"),
                            os.path.join(parent, f"{base}_glossary{ext}"),
                        ]:
                            if os.path.isfile(pattern):
                                return pattern
                return None
            
            def _open_glossary_file(_checked=False):
                """Open the glossary file in the best available text editor."""
                import subprocess, shutil, sys
                
                glossary_path = _find_glossary_file()
                if not glossary_path or not os.path.isfile(glossary_path):
                    from PySide6.QtWidgets import QMessageBox
                    QMessageBox.information(
                        panel, "No Glossary Found",
                        f"No glossary file found in:\n{_gp_folder}\n\n"
                        "Expected: glossary.csv, glossary.json, or <book>_glossary.csv"
                    )
                    return
                
                try:
                    if sys.platform == 'win32':
                        _npp_paths = [
                            r'C:\Program Files\Notepad++\notepad++.exe',
                            r'C:\Program Files (x86)\Notepad++\notepad++.exe',
                        ]
                        _npp = next((p for p in _npp_paths if os.path.exists(p)), None)
                        if _npp:
                            subprocess.Popen([_npp, glossary_path])
                        else:
                            subprocess.Popen(['notepad.exe', glossary_path])
                    elif sys.platform == 'darwin':
                        if shutil.which('code'):
                            subprocess.Popen(['code', glossary_path])
                        else:
                            subprocess.Popen(['open', '-t', glossary_path])
                    else:
                        if shutil.which('gedit'):
                            subprocess.Popen(['gedit', glossary_path])
                        elif shutil.which('kate'):
                            subprocess.Popen(['kate', glossary_path])
                        elif shutil.which('code'):
                            subprocess.Popen(['code', glossary_path])
                        else:
                            _linux_editors = ['mousepad', 'xed', 'pluma', 'nano', 'xdg-open']
                            _editor = next((e for e in _linux_editors if shutil.which(e)), 'xdg-open')
                            subprocess.Popen([_editor, glossary_path])
                except Exception as _e:
                    print(f"⚠️ Could not open glossary editor: {_e}")
            
            open_glossary_btn.clicked.connect(_open_glossary_file)
            # Show tooltip with glossary path if found
            _initial_glossary = _find_glossary_file()
            if _initial_glossary:
                open_glossary_btn.setToolTip(f"Open in text editor:\n{_initial_glossary}")
            else:
                open_glossary_btn.setToolTip("No glossary file found yet")
            path_row.addWidget(open_glossary_btn)
            
            p_layout.addLayout(path_row)
            
            # Helper to fully rebuild the listbox when chapter_map changes
            def _rebuild_listbox(_d):
                _cmap = panel_state['chapter_map']
                _total = 0
                _comp, _fail, _merg = _gp_sets(_d)
                
                gp_listbox.clear()
                for ci in range(_total):
                    fname = _cmap.get(ci, f'chapter {ci + 1}')
                    ch_num = _gp_display_chapter_num(ci, fname)
                    
                    if ci in _merg:
                        icon, status, color = '🔗', 'merged', '#17a2b8'
                    elif ci in _comp:
                        icon, status, color = '✅', 'completed', '#27ae60'
                    elif ci in _fail:
                        icon, status, color = '❌', 'failed', '#e74c3c'
                    else:
                        icon, status, color = '⬜', 'not_completed', '#5a9fd4'
                    
                    display, status = _gp_display_for(ci, fname, _d)
                    color = _gp_color_for(status)
                    item = QListWidgetItem(display)
                    item.setForeground(QColor(color))
                    item.setData(Qt.UserRole, status)
                    item.setData(Qt.UserRole + 1, ci)
                    self._add_compact_inline_list_item(gp_listbox, item)
            
                _populate_gp_listbox(_d)

            # Refresh function (called by timer)
            def _refresh():
                try:
                    _rp = _find_gp_for_file(fp)
                    if not _rp or not os.path.isfile(_rp):
                        return
                    
                    # Dirty-check: skip full recomputation if file hasn't changed
                    # and the special-files toggle is the same
                    try:
                        _cur_mtime = os.path.getmtime(_rp)
                    except OSError:
                        _cur_mtime = 0
                    _cur_ts = os.getenv('TRANSLATE_SPECIAL_FILES', '0') == '1'
                    if (_cur_mtime == panel_state.get('_last_mtime', -1)
                            and _cur_ts == panel_state.get('translate_special')):
                        return  # Nothing changed — skip
                    panel_state['_last_mtime'] = _cur_mtime
                    
                    _d = _gp_load_progress_dict(_rp)
                    
                    # Check if TRANSLATE_SPECIAL_FILES toggle changed — rebuild chapter map if so
                    if _cur_ts != panel_state['translate_special']:
                        panel_state['translate_special'] = _cur_ts
                        new_cmap, new_total, new_spine_idx = _read_spine_map(fp, _cur_ts)
                        if new_total > 0:
                            panel_state['chapter_map'] = new_cmap
                            panel_state['spine_index_map'] = new_spine_idx
                            panel_state['total'] = new_total
                        elif new_total == 0:
                            _comp0, _fail0, _merg0 = _gp_sets(_d)
                            _all_idx = _comp0 | _fail0 | _merg0
                            panel_state['total'] = (max(_all_idx, default=0) + 1) if _all_idx else 1
                        # Rebuild reverse lookups after chapter_map change
                        _rebuild_reverse_lookups()
                        _rebuild_listbox(_d)
                        _refresh_stats_from_dict(_d)
                        return
                    
                    _comp, _fail, _merg = _gp_sets(_d)
                    _prog = _gp_in_progress_set(_d, _precomputed_sets=(_comp, _fail, _merg))
                    _not_refined = _gp_refinement_status_counts(_d).get('not_refined', 0)
                    _total = panel_state['total']
                    _cmap = panel_state['chapter_map']
                    
                    _nr = max(0, _total - len(_comp | _fail | _merg | _prog))
                    lbl_total.setText(f"Total: {_total} | ")
                    lbl_gp_completed.setText(f"✅ Completed: {len(_comp - _merg)} | ")
                    lbl_gp_in_progress.setText(f"🔄 In Progress: {len(_prog)} | ")
                    lbl_gp_in_progress.setVisible(True)
                    lbl_gp_failed.setText(f"❌ Failed: {len(_fail)} | ")
                    lbl_gp_merged.setText(f"🔗 Merged: {len(_merg)} | ")
                    lbl_gp_merged.setVisible(len(_merg) > 0)
                    lbl_gp_remaining.setText(f"⬜ Not Translated: {_nr}{' | ' if _not_refined else ''}")
                    lbl_gp_not_refined.setText(f"✨ Not Refined: {_not_refined}")
                    lbl_gp_not_refined.setVisible(_not_refined > 0)
                    
                    _bt = _d.get('book_title', '')
                    if _bt and bt_label:
                        bt_label.setText(f"📖 {_bt}")
                    
                    _cache = _gp_status_cache(_d)
                    for ci in range(min(gp_listbox.count(), _total)):
                        item = gp_listbox.item(ci)
                        if not item:
                            continue
                        if item.data(Qt.UserRole + 3):
                            continue
                        old_status = item.data(Qt.UserRole)
                        new_status, _issues = _gp_status_for(ci, _d, _cache)
                        new_color = _gp_color_for(new_status)
                        
                        if new_status != old_status or _issues:
                            fname = _cmap.get(ci, f'chapter {ci + 1}')
                            ch_num2 = _gp_display_chapter_num(ci, fname)
                            _icons = {'completed': '✅', 'failed': '❌', 'merged': '🔗', 'not_completed': '⬜'}
                            display2 = f"Ch.{ch_num2:03d} | {_icons.get(new_status, '⬜')} {new_status.replace('_', ' ').title():14s} | {fname}"
                            display2, _ = _gp_display_for(ci, fname, _d, _cache)
                            item.setText(display2)
                            item.setForeground(QColor(new_color))
                            item.setData(Qt.UserRole, new_status)
                    _refresh_refinement_rows(_d)
                except Exception:
                    pass
            
            return panel, _refresh
        
        def _show_glossary_progress():
            """Show glossary extraction progress for all EPUBs (with or without progress files)."""
            try:
                # Reuse cached dialog if it still exists
                _cached = getattr(dialog, '_glossary_progress_dialog', None)
                if _cached is not None:
                    try:
                        _cached.show()
                        _cached.raise_()
                        _cached.activateWindow()

                        def _refresh_cached_gp_panels():
                            for rfn in getattr(dialog, '_gp_refresh_funcs', []):
                                try:
                                    rfn()
                                except Exception:
                                    pass

                        QTimer.singleShot(0, _refresh_cached_gp_panels)
                        return
                    except RuntimeError:
                        # Widget was deleted
                        dialog._glossary_progress_dialog = None
                        dialog._gp_refresh_funcs = []
                
                # Gather all EPUB paths from multi-file dialog or just this file
                all_files = [file_path]
                if parent_dialog and hasattr(parent_dialog, '_epub_files_in_dialog'):
                    all_files = [f for f in parent_dialog._epub_files_in_dialog if str(f).lower().endswith('.epub')]
                if not all_files:
                    all_files = [file_path]
                
                # Build entries for ALL EPUBs — gp_path is None when no progress file exists
                all_file_entries = []
                for fp in all_files:
                    gp = _find_gp_for_file(fp)
                    if gp and os.path.isfile(gp):
                        all_file_entries.append((fp, gp))
                    else:
                        all_file_entries.append((fp, None))
                
                # Create dialog
                gp_dialog = QDialog(dialog)
                gp_dialog.setAttribute(Qt.WA_DeleteOnClose, False)
                n_files = len(all_file_entries)
                if n_files == 1:
                    gp_dialog.setWindowTitle(f"Glossary Extraction Progress — {os.path.basename(all_file_entries[0][0])}")
                else:
                    gp_dialog.setWindowTitle(f"Glossary Extraction Progress — {n_files} files")
                gp_dialog.setWindowModality(Qt.NonModal)
                gp_width, gp_height = self._get_dialog_size(0.35, 0.45)
                gp_dialog.resize(gp_width, gp_height)
                
                try:
                    ss = dialog.styleSheet()
                    if ss:
                        gp_dialog.setStyleSheet(ss)
                except Exception:
                    pass
                
                gp_main_layout = QVBoxLayout(gp_dialog)
                
                # Title + note
                gp_title = QLabel("Glossary Extraction Progress")
                gp_title_font = QFont('Arial', 12)
                gp_title_font.setBold(True)
                gp_title.setFont(gp_title_font)
                gp_title.setStyleSheet("color: #52b788;")
                gp_main_layout.addWidget(gp_title)
                
                gp_note = QLabel("ℹ️ Tracks Balanced / Full auto glossary modes (Extract Glossary logic)")
                gp_note.setStyleSheet("color: #7a8a9e; font-size: 8pt; font-style: italic;")
                gp_note.setWordWrap(True)
                gp_main_layout.addWidget(gp_note)
                
                # Build panels and collect refresh functions
                all_refresh_funcs = []
                
                def _build_gp_empty_panel(fp, parent_widget):
                    """Build a placeholder panel for an EPUB without glossary progress yet.
                    Returns (panel_widget, refresh_func) — refresh auto-upgrades to full panel."""
                    epub_base = os.path.splitext(os.path.basename(fp))[0]
                    
                    panel = QWidget(parent_widget)
                    p_layout = QVBoxLayout(panel)
                    p_layout.setContentsMargins(12, 20, 12, 20)
                    
                    empty_icon = QLabel("📊")
                    empty_icon.setAlignment(Qt.AlignCenter)
                    empty_icon.setStyleSheet("font-size: 36pt;")
                    p_layout.addWidget(empty_icon)
                    
                    expected_refinement = _glossary_refinement_expected_entries()
                    empty_label = QLabel(f"No glossary extraction progress found for:\n{epub_base}")
                    empty_label.setAlignment(Qt.AlignCenter)
                    empty_label.setStyleSheet("color: #7a8a9e; font-size: 11pt;")
                    empty_label.setWordWrap(True)
                    p_layout.addWidget(empty_label)
                    
                    hint_text = "Run glossary extraction to see progress here."
                    if expected_refinement:
                        hint_text = "Glossary refinement is enabled; expected refinement entry types are listed below."
                    hint_label = QLabel(hint_text)
                    hint_label.setAlignment(Qt.AlignCenter)
                    hint_label.setStyleSheet("color: #555; font-size: 9pt; font-style: italic;")
                    p_layout.addWidget(hint_label)

                    if expected_refinement:
                        ref_list = QListWidget(panel)
                        ref_list.setSelectionMode(QAbstractItemView.NoSelection)
                        ref_list.setSpacing(0)
                        ref_list.setUniformItemSizes(True)
                        ref_list.setStyleSheet("""
                            QListWidget {
                                background-color: #1f1f1f;
                                color: white;
                                border: 1px solid #4a5568;
                                border-radius: 4px;
                                padding: 4px;
                            }
                            QListWidget::item {
                                margin: 0px;
                                padding: 0px 4px;
                            }
                        """)
                        ref_list.setMaximumHeight(160)
                        for _ref_key, ref_info in sorted(expected_refinement.items()):
                            entry_type = str(ref_info.get('entry_type') or _ref_key.replace('type::', '')).strip() or 'entry type'
                            item = QListWidgetItem(f"Refinement | \u2728 Not Refined    | {entry_type}")
                            item.setForeground(QColor('#5a9fd4'))
                            self._add_compact_inline_list_item(ref_list, item)
                        p_layout.addWidget(ref_list)
                    
                    p_layout.addStretch()
                    
                    # State container for upgrade tracking
                    _state = {'upgraded': False}
                    
                    def _empty_refresh():
                        """Check if a progress file appeared and upgrade the panel in-place."""
                        if _state['upgraded']:
                            return
                        gp = _find_gp_for_file(fp)
                        if gp and os.path.isfile(gp):
                            _state['upgraded'] = True
                            # Clear placeholder content
                            while p_layout.count():
                                item = p_layout.takeAt(0)
                                if item and item.widget():
                                    item.widget().deleteLater()
                            # Build the real panel content inside this existing panel
                            real_panel, real_refresh = _build_gp_panel(fp, gp, panel)
                            p_layout.addWidget(real_panel)
                            # Replace this refresh func in the parent list
                            try:
                                idx = all_refresh_funcs.index(_empty_refresh)
                                all_refresh_funcs[idx] = real_refresh
                            except ValueError:
                                all_refresh_funcs.append(real_refresh)
                    
                    return panel, _empty_refresh
                
                if n_files == 1:
                    # Single file — no tabs needed
                    fp, gp = all_file_entries[0]
                    if gp:
                        panel, refresh_fn = _build_gp_panel(fp, gp, gp_dialog)
                    else:
                        panel, refresh_fn = _build_gp_empty_panel(fp, gp_dialog)
                    gp_main_layout.addWidget(panel)
                    all_refresh_funcs.append(refresh_fn)
                
                elif n_files <= 3:
                    # Tabs for ≤3 files
                    notebook = QTabWidget()
                    notebook.setStyleSheet("""
                        QTabWidget::pane {
                            border: 2px solid #40916c;
                            border-radius: 4px;
                            background-color: #2d2d2d;
                        }
                        QTabBar::tab {
                            background-color: #3a3a3a;
                            color: white;
                            padding: 8px 16px;
                            margin-right: 2px;
                            border: 1px solid #40916c;
                            border-bottom: none;
                            border-top-left-radius: 4px;
                            border-top-right-radius: 4px;
                            font-size: 10pt;
                        }
                        QTabBar::tab:selected {
                            background-color: #2d6a4f;
                            color: #d8f3dc;
                            font-weight: bold;
                        }
                        QTabBar::tab:hover { background-color: #40916c; }
                    """)
                    for fp, gp in all_file_entries:
                        epub_base = os.path.splitext(os.path.basename(fp))[0]
                        if gp:
                            panel, refresh_fn = _build_gp_panel(fp, gp, notebook)
                        else:
                            panel, refresh_fn = _build_gp_empty_panel(fp, notebook)
                        notebook.addTab(panel, epub_base)
                        all_refresh_funcs.append(refresh_fn)
                    gp_main_layout.addWidget(notebook)
                
                else:
                    # Dropdown navigation for >3 files
                    from PySide6.QtWidgets import QComboBox, QStackedWidget
                    
                    nav_row = QHBoxLayout()
                    nav_row.setSpacing(6)
                    
                    nav_prev = QPushButton("◀")
                    nav_prev.setFixedWidth(36)
                    nav_prev.setStyleSheet(
                        "QPushButton { background-color:#3a3a3a; color:white; font-weight:bold; "
                        "font-size:13pt; border:1px solid #5a9fd4; border-radius:4px; padding:4px; }"
                        "QPushButton:hover { background-color:#4a8fc4; }"
                        "QPushButton:disabled { color:#666; background-color:#2a2a2a; }"
                    )
                    
                    combo = QComboBox()
                    combo.setStyleSheet(
                        "QComboBox { background-color:#3a3a3a; color:white; font-weight:bold; "
                        "font-size:11pt; padding:6px 10px; border:1px solid #5a9fd4; border-radius:4px; }"
                        "QComboBox::drop-down { border:none; }"
                        "QComboBox QAbstractItemView { background-color:#2d2d2d; color:white; "
                        "selection-background-color:#5a9fd4; }"
                    )
                    
                    nav_counter = QLabel("1 / 1")
                    nav_counter.setStyleSheet("color:#94a3b8; font-size:10pt; font-weight:bold;")
                    nav_counter.setFixedWidth(60)
                    nav_counter.setAlignment(Qt.AlignCenter)
                    
                    nav_next = QPushButton("▶")
                    nav_next.setFixedWidth(36)
                    nav_next.setStyleSheet(nav_prev.styleSheet())
                    
                    nav_row.addWidget(nav_prev)
                    nav_row.addWidget(combo, stretch=1)
                    nav_row.addWidget(nav_counter)
                    nav_row.addWidget(nav_next)
                    gp_main_layout.addLayout(nav_row)
                    
                    stack = QStackedWidget()
                    
                    for fp, gp in all_file_entries:
                        epub_base = os.path.splitext(os.path.basename(fp))[0]
                        if gp:
                            panel, refresh_fn = _build_gp_panel(fp, gp, stack)
                        else:
                            panel, refresh_fn = _build_gp_empty_panel(fp, stack)
                        stack.addWidget(panel)
                        combo.addItem(epub_base)
                        all_refresh_funcs.append(refresh_fn)
                    
                    def _update_nav():
                        idx = combo.currentIndex()
                        n = combo.count()
                        nav_prev.setEnabled(idx > 0)
                        nav_next.setEnabled(idx < n - 1)
                        nav_counter.setText(f"{idx + 1} / {n}")
                        stack.setCurrentIndex(idx)
                    
                    combo.currentIndexChanged.connect(lambda _: _update_nav())
                    nav_prev.clicked.connect(lambda: combo.setCurrentIndex(combo.currentIndex() - 1))
                    nav_next.clicked.connect(lambda: combo.setCurrentIndex(combo.currentIndex() + 1))
                    _update_nav()
                    
                    gp_main_layout.addWidget(stack)
                
                # Close button
                close_btn = QPushButton("Close")
                close_btn.setStyleSheet(
                    "QPushButton { background-color: #555; color: white; padding: 6px 20px; "
                    "border-radius: 4px; font-size: 10pt; } "
                    "QPushButton:hover { background-color: #666; }"
                )
                close_btn.clicked.connect(gp_dialog.hide)
                gp_main_layout.addWidget(close_btn, alignment=Qt.AlignCenter)
                
                # Auto-refresh timer (2s) — calls all panel refresh functions
                def _gp_refresh_all():
                    try:
                        if not gp_dialog.isVisible():
                            return
                        for rfn in all_refresh_funcs:
                            try:
                                rfn()
                            except Exception:
                                pass
                    except Exception:
                        pass
                
                _gp_timer = QTimer(gp_dialog)
                _gp_timer.setInterval(2000)
                _gp_timer.timeout.connect(_gp_refresh_all)
                _gp_timer.start()
                
                # Cache on parent dialog so all tabs share the same instance
                dialog._glossary_progress_dialog = gp_dialog
                dialog._gp_refresh_funcs = all_refresh_funcs
                
                gp_dialog.show()
            
            except Exception as e:
                print(f"⚠️ Error showing glossary progress: {e}")
                import traceback
                traceback.print_exc()
        
        glossary_progress_btn.clicked.connect(_show_glossary_progress)
        title_layout.addWidget(glossary_progress_btn)
        
        # Periodic check: show/hide button based on file existence (3s)
        # Uses single-pass caching to avoid redundant filesystem scans
        def _check_glossary_btn_visibility():
            try:
                # Skip if parent dialog is not visible (no point scanning filesystem)
                if hasattr(dialog, 'isVisible') and not dialog.isVisible():
                    return
                
                # Check all EPUBs from multi-file dialog, or just this file
                all_epubs = [file_path]
                if parent_dialog and hasattr(parent_dialog, '_epub_files_in_dialog'):
                    all_epubs = [f for f in parent_dialog._epub_files_in_dialog if str(f).lower().endswith('.epub')]
                if not all_epubs:
                    all_epubs = [file_path]
                
                is_multi = len(all_epubs) > 1
                # Single pass: resolve all paths once, cache results
                gp_results = {fp: _find_gp_for_file(fp) for fp in all_epubs}
                found_paths = {fp: gp for fp, gp in gp_results.items() if gp}
                any_exists = bool(found_paths)
                refinement_expected = _glossary_refinement_settings_enabled()
                glossary_progress_btn.setVisible(True)
                if any_exists:
                    count = len(found_paths)
                    if count == 1:
                        gp = next(iter(found_paths.values()))
                        glossary_progress_btn.setToolTip(f"View glossary extraction progress\n{gp}")
                    else:
                        glossary_progress_btn.setToolTip(f"View glossary extraction progress ({count}/{len(all_epubs)} files)")
                elif refinement_expected:
                    glossary_progress_btn.setToolTip("View glossary extraction and refinement progress")
                elif is_multi:
                    glossary_progress_btn.setToolTip(f"View glossary extraction progress ({len(all_epubs)} files)")
                else:
                    glossary_progress_btn.setToolTip("View glossary extraction progress")
            except RuntimeError:
                # Widget was deleted
                _gp_vis_timer.stop()
        
        _gp_vis_timer = QTimer()
        _gp_vis_timer.setInterval(3000)
        _gp_vis_timer.timeout.connect(_check_glossary_btn_visibility)
        _gp_vis_timer.start()
        # Parent timer to container so it dies with the dialog
        _gp_vis_timer.setParent(container)
        
        container_layout.addWidget(title_row)
        
        # Store reference to the listbox (will be created later)
        listbox_ref = [None]
        
        # Function to handle toggle change - will be defined after UI is created
        def on_toggle_special_files(state):
            """Filter the chapter list when the special files toggle is changed"""
            # Update the state variable
            show_special_files[0] = show_special_files_cb.isChecked()
            
            # Store the state persistently
            file_key = os.path.abspath(file_path)
            if not hasattr(self, '_retranslation_dialog_cache'):
                self._retranslation_dialog_cache = {}
            if file_key not in self._retranslation_dialog_cache:
                self._retranslation_dialog_cache[file_key] = {}
            self._retranslation_dialog_cache[file_key]['show_special_files_state'] = show_special_files[0]
            
            # For tabs in multi-file dialog, sync toggle state across tabs
            if tab_frame and parent_dialog:
                # Update cache for all files in the current selection
                if hasattr(parent_dialog, '_epub_files_in_dialog'):
                    for f_path in parent_dialog._epub_files_in_dialog:
                        f_key = os.path.abspath(f_path)
                        if f_key not in self._retranslation_dialog_cache:
                            self._retranslation_dialog_cache[f_key] = {}
                        self._retranslation_dialog_cache[f_key]['show_special_files_state'] = show_special_files[0]
                
                # Sync ALL toggle checkboxes and checkmarks in ALL tabs
                if hasattr(parent_dialog, '_all_toggle_checkboxes'):
                    for idx, other_checkbox in enumerate(parent_dialog._all_toggle_checkboxes):
                        if other_checkbox is None or other_checkbox == show_special_files_cb:
                            continue
                        
                        try:
                            other_checkbox.isChecked()
                            other_checkbox.blockSignals(True)
                            other_checkbox.setChecked(show_special_files[0])
                            other_checkbox.blockSignals(False)
                            
                            if hasattr(parent_dialog, '_all_checkmark_labels') and idx < len(parent_dialog._all_checkmark_labels):
                                other_checkmark = parent_dialog._all_checkmark_labels[idx]
                                if other_checkmark is not None:
                                    try:
                                        other_checkmark.isVisible()
                                        if show_special_files[0]:
                                            other_checkmark.setGeometry(2, 1, 14, 14)
                                            other_checkmark.show()
                                        else:
                                            other_checkmark.hide()
                                    except RuntimeError:
                                        parent_dialog._all_checkmark_labels[idx] = None
                        except (RuntimeError, AttributeError):
                            parent_dialog._all_toggle_checkboxes[idx] = None
            
            # Filter list items instead of rebuilding entire UI
            if listbox_ref[0]:
                listbox = listbox_ref[0]
                for i in range(listbox.count()):
                    item = listbox.item(i)
                    if item:
                        # Check if this item is marked as special
                        item_data = item.data(Qt.UserRole)
                        if item_data and isinstance(item_data, dict):
                            # Dynamically re-evaluate is_special to respect current
                            # translate_all_numbered_html setting.
                            _info = item_data.get('info') or {}
                            _fname = _info.get('original_filename', '') or _info.get('output_file', '') or _info.get('key', '')
                            is_skipped_special = self._progress_file_is_skipped_special(
                                _fname,
                                item_data.get('is_special', False),
                            )
                            # Show all items if toggle is on, hide only files that translation skips.
                            item.setHidden(is_skipped_special and not show_special_files[0])
        
        # Connect the checkbox to the handler
        show_special_files_cb.stateChanged.connect(on_toggle_special_files)

        def on_toggle_model_info(state):
            show_model_info[0] = show_model_info_cb.isChecked()
            file_key = os.path.abspath(file_path)
            if not hasattr(self, '_retranslation_dialog_cache'):
                self._retranslation_dialog_cache = {}
            if file_key not in self._retranslation_dialog_cache:
                self._retranslation_dialog_cache[file_key] = {}
            self._retranslation_dialog_cache[file_key]['show_model_info_state'] = show_model_info[0]
            self._persist_retranslation_show_model_info_state(show_model_info[0])

            if tab_frame and parent_dialog and hasattr(parent_dialog, '_epub_files_in_dialog'):
                for f_path in parent_dialog._epub_files_in_dialog:
                    f_key = os.path.abspath(f_path)
                    if f_key not in self._retranslation_dialog_cache:
                        self._retranslation_dialog_cache[f_key] = {}
                    self._retranslation_dialog_cache[f_key]['show_model_info_state'] = show_model_info[0]

            data = getattr(show_model_info_cb, '_progress_data_ref', None)
            if isinstance(data, dict):
                data['show_model_info_state'] = show_model_info[0]
                self._update_listbox_display(data)

        show_model_info_cb.stateChanged.connect(on_toggle_model_info)
        
        # Statistics - always show for both OPF and non-OPF files
        stats_frame = QWidget()
        stats_layout = QHBoxLayout(stats_frame)
        stats_layout.setContentsMargins(0, 5, 0, 5)
        container_layout.addWidget(stats_frame)
        
        # Calculate stats from the appropriate source
        _stats_data = {'prog': prog}
        if spine_chapters:
            total_chapters = len(spine_chapters)
            completed = sum(1 for ch in spine_chapters if self._progress_display_status(ch, _stats_data) == 'completed')
            merged = sum(1 for ch in spine_chapters if self._progress_display_status(ch, _stats_data) == 'merged')
            in_progress = sum(1 for ch in spine_chapters if self._progress_display_status(ch, _stats_data) == 'in_progress')
            pending = sum(1 for ch in spine_chapters if self._progress_display_status(ch, _stats_data) == 'pending')
            missing = sum(1 for ch in spine_chapters if self._progress_display_status(ch, _stats_data) in ['not_translated', 'not_refined', 'no_tts'])
            failed = sum(1 for ch in spine_chapters if self._progress_display_status(ch, _stats_data) in ['failed', 'qa_failed'])
        else:
            # For non-OPF files, calculate from chapter_display_info
            total_chapters = len(chapter_display_info)
            completed = sum(1 for ch in chapter_display_info if self._progress_display_status(ch, _stats_data) == 'completed')
            merged = sum(1 for ch in chapter_display_info if self._progress_display_status(ch, _stats_data) == 'merged')
            in_progress = sum(1 for ch in chapter_display_info if self._progress_display_status(ch, _stats_data) == 'in_progress')
            pending = sum(1 for ch in chapter_display_info if self._progress_display_status(ch, _stats_data) == 'pending')
            missing = sum(1 for ch in chapter_display_info if self._progress_display_status(ch, _stats_data) in ['not_translated', 'not_refined', 'no_tts'])
            failed = sum(1 for ch in chapter_display_info if self._progress_display_status(ch, _stats_data) in ['failed', 'qa_failed'])
        
        # Create labels (outside the if/else so they always appear)
        stats_font = QFont('Arial', 10)
        
        lbl_total = QLabel(f"Total: {total_chapters} | ")
        lbl_total.setFont(stats_font)
        stats_layout.addWidget(lbl_total)
        
        lbl_completed = QLabel(f"✅ Completed: {completed} | ")
        lbl_completed.setFont(stats_font)
        lbl_completed.setStyleSheet("color: green;")
        lbl_completed.setCursor(Qt.PointingHandCursor)
        stats_layout.addWidget(lbl_completed)
        
        # Merged: chapters combined into parent request (always create, hide if 0)
        lbl_merged = QLabel(f"🔗 Merged: {merged} | ")
        lbl_merged.setFont(stats_font)
        lbl_merged.setStyleSheet("color: #17a2b8;")  # Cyan/teal
        stats_layout.addWidget(lbl_merged)
        if merged == 0:
            lbl_merged.setVisible(False)
        
        # In Progress: currently being translated (always create, hide if 0)
        lbl_in_progress = QLabel(f"🔄 In Progress: {in_progress} | ")
        lbl_in_progress.setFont(stats_font)
        lbl_in_progress.setStyleSheet("color: orange;")
        lbl_in_progress.setCursor(Qt.PointingHandCursor)
        stats_layout.addWidget(lbl_in_progress)
        if in_progress == 0:
            lbl_in_progress.setVisible(False)
        
        # Pending: marked for retranslation (always create, hide if 0)
        lbl_pending = QLabel(f"❓ Pending: {pending} | ")
        lbl_pending.setFont(stats_font)
        lbl_pending.setStyleSheet("color: white;")
        lbl_pending.setCursor(Qt.PointingHandCursor)
        stats_layout.addWidget(lbl_pending)
        if pending == 0:
            lbl_pending.setVisible(False)
        
        # Not Translated: unique emoji/color (distinct from failures)
        _current_output_mode = self._current_progress_output_mode({'prog': prog})
        _missing_label_text = "✨ Not Refined" if _current_output_mode == 'refinement' else ("🔊 No TTS" if _current_output_mode == 'audio' else "⬜ Not Translated")
        lbl_missing = QLabel(f"{_missing_label_text}: {missing} | ")
        lbl_missing.setFont(stats_font)
        lbl_missing.setStyleSheet("color: #2b6cb0;")
        lbl_missing.setCursor(Qt.PointingHandCursor)
        stats_layout.addWidget(lbl_missing)
        
        # Match list status: failed/qa_failed use ❌ and red (clickable — jumps to next failure)
        lbl_failed = QLabel(f"❌ Failed: {failed} | ")
        lbl_failed.setFont(stats_font)
        lbl_failed.setStyleSheet("color: red;")
        lbl_failed.setCursor(Qt.PointingHandCursor)
        stats_layout.addWidget(lbl_failed)
        
        
        stats_layout.addStretch()
        
        # Show temporary "folder created" label in the stats row if a folder was just created
        created_folder = getattr(self, '_pm_created_folder', None)
        if created_folder:
            display_name = os.path.basename(created_folder) or created_folder
            lbl_created = QLabel(f"📁 Created: {display_name}")
            lbl_created.setFont(stats_font)
            lbl_created.setStyleSheet("color: #27ae60; font-weight: bold;")
            stats_layout.addWidget(lbl_created)
            # Auto-hide after 2000ms
            QTimer.singleShot(2000, lbl_created.hide)
            # Clear the stored path so it doesn't re-appear on refresh
            self._pm_created_folder = None
        
        # Main frame for listbox
        main_frame = QWidget()
        main_layout = QVBoxLayout(main_frame)
        main_layout.setContentsMargins(10 if not tab_frame else 5, 5, 10 if not tab_frame else 5, 5)
        container_layout.addWidget(main_frame)
        
        # Create listbox (QListWidget has built-in scrollbars)
        listbox = QListWidget()
        listbox.setSelectionMode(QListWidget.ExtendedSelection)
        listbox_font = QFont('Courier', 10)  # Fixed-width font for better alignment
        self._apply_compact_inline_list_style(listbox, listbox_font, extra_row_px=2)
        # Use 36% of screen width
        min_width, _ = self._get_dialog_size(0.36, 0)
        listbox.setMinimumWidth(min_width)
        main_layout.addWidget(listbox)
        
        # Store listbox reference for toggle handler
        listbox_ref[0] = listbox
        
        # Helper: cycle to next item matching given statuses
        def _make_cycle_handler(statuses):
            def _handler(_event=None):
                lb = listbox_ref[0]
                if not lb:
                    return
                status_data = {'prog': prog}
                indices = []
                for i in range(lb.count()):
                    item = lb.item(i)
                    if not item or item.isHidden():
                        continue
                    display_status = item.data(Qt.UserRole + 2)
                    if not display_status:
                        payload = item.data(Qt.UserRole) or {}
                        display_status = self._progress_display_status(payload.get('info', {}), status_data)
                    if display_status in statuses:
                        indices.append(i)
                if not indices:
                    return
                selected_rows = [lb.row(item) for item in lb.selectedItems()]
                current = lb.currentRow()
                if selected_rows and current not in selected_rows:
                    current = max(selected_rows)
                nxt = next((i for i in indices if i > current), indices[0])
                lb.setCurrentRow(nxt, QItemSelectionModel.ClearAndSelect)
                lb.scrollToItem(lb.item(nxt), QListWidget.PositionAtCenter)
            return _handler

        lbl_completed.mousePressEvent   = _make_cycle_handler(('completed',))
        lbl_in_progress.mousePressEvent = _make_cycle_handler(('in_progress',))
        lbl_pending.mousePressEvent     = _make_cycle_handler(('pending',))
        lbl_missing.mousePressEvent     = _make_cycle_handler(('not_translated', 'not_refined', 'no_tts'))
        lbl_failed.mousePressEvent      = _make_cycle_handler(('failed', 'qa_failed'))
        
        # Large progress lists are populated after result setup so the dialog can paint first.
        
        # Selection count label
        selection_count_label = QLabel("Selected: 0")
        selection_font = QFont('Arial', 10 if not tab_frame else 9)
        selection_count_label.setFont(selection_font)
        container_layout.addWidget(selection_count_label)
        
        def update_selection_count():
            count = len(listbox.selectedItems())
            selection_count_label.setText(f"Selected: {count}")
        
        listbox.itemSelectionChanged.connect(update_selection_count)
        
        # Return data structure for external access
        result = {
            'file_path': file_path,
            'output_dir': output_dir,
            'progress_file': progress_file,
            'prog': prog,
            'spine_chapters': spine_chapters,
            'opf_chapter_order': opf_chapter_order,
            'chapter_display_info': chapter_display_info,
            'listbox': listbox,
            'selection_count_label': selection_count_label,
            'dialog': dialog,
            'container': container,
            'show_special_files_state': show_special_files[0],  # Store current toggle state
            'show_special_files_cb': show_special_files_cb,  # Store checkbox reference
            'show_model_info_state': show_model_info[0],
            'show_model_info_cb': show_model_info_cb
        }
        show_model_info_cb._progress_data_ref = result
        
        # If standalone (no parent), add buttons and show dialog
        if not parent_dialog and not tab_frame:
            self._add_retranslation_buttons_opf(result)
            
            # Override close event to hide instead of destroy
            def closeEvent(event):
                event.ignore()  # Ignore the close event
                dialog.hide()   # Just hide the dialog
            
            dialog.closeEvent = closeEvent
            
            # Cache the dialog for reuse
            if not hasattr(self, '_retranslation_dialog_cache'):
                self._retranslation_dialog_cache = {}
            
            file_key = os.path.abspath(file_path)
            self._retranslation_dialog_cache[file_key] = result
            
            # Show the dialog (non-modal to allow interaction with other windows)
            dialog.show()
            QTimer.singleShot(50, lambda: self._populate_progress_listbox_streamed(result))
        elif not parent_dialog or tab_frame:
            # Embedded in tab - just add buttons
            self._add_retranslation_buttons_opf(result)
        
        return result


    def _add_retranslation_buttons_opf(self, data, button_frame=None):
        """Add the standard button set for retranslation dialogs with OPF support"""
        
        if not button_frame:
            button_frame = QWidget()
            button_layout = QGridLayout(button_frame)
            # Get container layout and add button frame
            container = data['container']
            if hasattr(container, 'layout') and container.layout():
                container.layout().addWidget(button_frame)
        else:
            button_layout = button_frame.layout() if button_frame.layout() else QGridLayout(button_frame)
        
        # Helper functions that work with the data dict
        def select_all():
            data['listbox'].selectAll()
            data['selection_count_label'].setText(f"Selected: {data['listbox'].count()}")
        
        def clear_selection():
            data['listbox'].clearSelection()
            data['selection_count_label'].setText("Selected: 0")
        
        def select_status(status_to_select):
            data['listbox'].clearSelection()
            for idx in range(data['listbox'].count()):
                item = data['listbox'].item(idx)
                if not item or item.isHidden():
                    continue
                display_status = item.data(Qt.UserRole + 2)
                if not display_status:
                    payload = item.data(Qt.UserRole) or {}
                    display_status = self._progress_display_status(payload.get('info', {}), data)
                if status_to_select == 'failed':
                    matched = display_status in ['failed', 'qa_failed']
                elif status_to_select == 'qa_failed':
                    matched = display_status == 'qa_failed'
                else:
                    matched = display_status == status_to_select
                if matched:
                    item.setSelected(True)
            count = len(data['listbox'].selectedItems())
            data['selection_count_label'].setText(f"Selected: {count}")

        def _normalize_filename(name: str) -> str:
            if not name:
                return ""
            base = os.path.basename(name)
            if base.startswith("response_"):
                base = base[len("response_"):]
            while True:
                new_base, ext = os.path.splitext(base)
                if not ext:
                    break
                base = new_base
            return base

        def _find_progress_entry(chapter_info, prog):
            """Strict: match only identical output_file string."""
            target_out = chapter_info.get('output_file')
            if not target_out:
                return None
            for key, ch in prog.get("chapters", {}).items():
                if ch.get('output_file') == target_out:
                    return key, ch
            return None

        def _clear_refinement_progress_fields(entry):
            """Remove stale refinement metadata when a chapter is queued again."""
            if not isinstance(entry, dict):
                return 0
            removed = 0
            for field in ("refinement_status", "refined_at", "refinement_error", "unrefined_backup_file"):
                if field in entry:
                    entry.pop(field, None)
                    removed += 1
            previous_entry = entry.get("previous_progress_entry")
            if isinstance(previous_entry, dict):
                for field in ("refinement_status", "refined_at", "refinement_error", "unrefined_backup_file"):
                    previous_entry.pop(field, None)
            return removed

        def _restore_regular_in_progress_entry(info):
            if not isinstance(info, dict):
                return None
            previous_status = str(info.get('previous_status', '') or '').lower()
            previous_entry = info.get('previous_progress_entry')
            transient_statuses = {'in_progress', 'not_translated', 'not translated', 'not_completed'}
            if isinstance(previous_entry, dict):
                restored = dict(previous_entry)
                restored_status = str(restored.get('status', previous_status) or previous_status).lower()
                if restored_status and restored_status not in transient_statuses:
                    restored.pop('previous_status', None)
                    restored.pop('previous_progress_entry', None)
                    return restored
            if previous_status in ('qa_failed', 'failed', 'error', 'pending', 'merged', 'completed'):
                restored = dict(info)
                restored['status'] = 'failed' if previous_status == 'error' else previous_status
                restored.pop('previous_status', None)
                restored.pop('previous_progress_entry', None)
                restored.pop('previous_status_unknown', None)
                return restored
            if info.get('previous_status_unknown'):
                restored = dict(info)
                restored['status'] = 'failed'
                restored.pop('previous_status', None)
                restored.pop('previous_progress_entry', None)
                restored.pop('previous_status_unknown', None)
                return restored
            output_file = info.get('output_file')
            output_exists = bool(output_file and os.path.exists(os.path.join(data['output_dir'], output_file)))
            if previous_status in ('not_translated', 'not translated', 'not_completed', ''):
                if previous_status and not output_exists:
                    return None
                if output_exists:
                    restored = dict(info)
                    restored['status'] = 'failed'
                    restored.pop('previous_status', None)
                    restored.pop('previous_progress_entry', None)
                    restored.pop('previous_status_unknown', None)
                    return restored
            return None

        def restore_in_progress_marks():
            selected_items = data['listbox'].selectedItems()
            if not selected_items:
                self._styled_msgbox(QMessageBox.Warning, data.get('dialog', self), "No Selection", "Please select at least one chapter.")
                return

            selected_indices = [data['listbox'].row(item) for item in selected_items]
            selected_chapters = [data['chapter_display_info'][i] for i in selected_indices]
            in_progress_chapters = [ch for ch in selected_chapters if ch.get('status') == 'in_progress']

            if not in_progress_chapters:
                self._styled_msgbox(QMessageBox.Warning, data.get('dialog', self), "No In Progress Chapters",
                                     "None of the selected chapters have 'in_progress' status.")
                return

            restored_count = 0
            deleted_count = 0
            failed_count = 0
            progress_updated = False

            for info in in_progress_chapters:
                match = None
                progress_key = info.get('progress_key')
                if progress_key and progress_key in data['prog'].get("chapters", {}):
                    match = (progress_key, data['prog']["chapters"][progress_key])
                else:
                    match = _find_progress_entry(info, data['prog'])

                if not match:
                    print(f"WARNING: Could not find in-progress entry for {info.get('num')} ({info.get('output_file')})")
                    continue

                key, entry = match
                restored = _restore_regular_in_progress_entry(entry)
                if restored:
                    data['prog']["chapters"][key] = restored
                    progress_updated = True
                    if restored.get('status') == 'failed':
                        failed_count += 1
                    else:
                        restored_count += 1
                else:
                    del data['prog']["chapters"][key]
                    progress_updated = True
                    deleted_count += 1

            if progress_updated:
                with open(data['progress_file'], 'w', encoding='utf-8') as f:
                    json.dump(data['prog'], f, ensure_ascii=False, indent=2)
                self._refresh_retranslation_data(data)

            message_parts = []
            if restored_count:
                message_parts.append(f"restored {restored_count}")
            if deleted_count:
                message_parts.append(f"removed {deleted_count} not-translated placeholder(s)")
            if failed_count:
                message_parts.append(f"marked {failed_count} as failed")
            message = "Successfully " + ", ".join(message_parts) + "." if message_parts else "No in-progress marks were changed."
            self._styled_msgbox(QMessageBox.Information, data.get('dialog', self), "In Progress Restored", message)
        
        def remove_qa_failed_mark():
            selected_items = data['listbox'].selectedItems()
            if not selected_items:
                self._styled_msgbox(QMessageBox.Warning, data.get('dialog', self), "No Selection", "Please select at least one chapter.")
                return

            # Skip dedup here to avoid merging distinct chapters that share filenames
            

            selected_indices = [data['listbox'].row(item) for item in selected_items]
            selected_chapters = [data['chapter_display_info'][i] for i in selected_indices]
            failed_chapters = [ch for ch in selected_chapters if ch['status'] in ['qa_failed', 'failed']]
            
            if not failed_chapters:
                self._styled_msgbox(QMessageBox.Warning, data.get('dialog', self), "No Failed Chapters", 
                                     "None of the selected chapters have 'qa_failed' or 'failed' status.")
                return
            
            count = len(failed_chapters)
            reply = self._styled_msgbox(QMessageBox.Question, data.get('dialog', self), "Confirm Remove Failed Mark", 
                                      f"Remove failed mark from {count} chapters?",
                                      QMessageBox.Yes | QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
            
            # Remove marks
            cleared_count = 0
            progress_updated = False
            for info in failed_chapters:
                match = None
                progress_key = info.get('progress_key')
                if progress_key and progress_key in data['prog'].get("chapters", {}):
                    match = (progress_key, data['prog']["chapters"][progress_key])
                else:
                    match = _find_progress_entry(info, data['prog'])

                # Normalize target output for multi-entry cleanup
                target_out = info.get('output_file')
                target_norm = _normalize_filename(target_out)
                if match:
                    # Clear failed/qa_failed on ALL entries sharing this output file (normalized)
                    fields_to_remove = ["qa_issues", "qa_timestamp", "qa_issues_found", "duplicate_confidence", "failure_reason", "error_message"]
                    for key, entry in data['prog'].get("chapters", {}).items():
                        entry_out = entry.get('output_file')
                        if not entry_out:
                            continue
                        if _normalize_filename(entry_out) == target_norm:
                            if entry.get('status') in ['qa_failed', 'failed']:
                                entry["status"] = "completed"
                                for field in fields_to_remove:
                                    entry.pop(field, None)
                                cleared_count += 1
                                progress_updated = True
                else:
                    print(f"WARNING: Could not find chapter entry for {info.get('num')} ({info.get('output_file')})")
            
            # Save the updated progress
            if progress_updated:
                with open(data['progress_file'], 'w', encoding='utf-8') as f:
                    json.dump(data['prog'], f, ensure_ascii=False, indent=2)
            
            # Auto-refresh the display
            self._refresh_retranslation_data(data)
            
            self._styled_msgbox(QMessageBox.Information, data.get('dialog', self), "Success", f"Removed failed mark from {cleared_count} chapters.")
        
        def retranslate_selected():
            selected_items = data['listbox'].selectedItems()
            if not selected_items:
                self._styled_msgbox(QMessageBox.Warning, data.get('dialog', self), "No Selection", "Please select at least one chapter.")
                return

            # Do NOT dedup here; it can collapse distinct chapters sharing filenames
            
            selected_indices = [data['listbox'].row(item) for item in selected_items]
            selected_chapters = [data['chapter_display_info'][i] for i in selected_indices]

            if self._current_progress_output_mode(data) == 'audio':
                count = len(selected_chapters)
                reply = self._styled_msgbox(
                    QMessageBox.Question,
                    data.get('dialog', self),
                    "Confirm TTS Reset",
                    f"This will delete only generated TTS audio for {count} selected chapter(s), mark them as No TTS, and leave translated HTML files untouched.\n\nContinue?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply != QMessageBox.Yes:
                    return

                deleted_count = 0
                status_reset_count = 0
                missing_audio_count = 0
                progress_updated = False

                def _audio_candidates(ch_entry, ch_info):
                    candidates = []
                    stored_tts_file = ch_entry.get('tts_file') if isinstance(ch_entry, dict) else None
                    if stored_tts_file:
                        candidates.append(stored_tts_file)

                    output_file = (ch_entry or {}).get('output_file') or ch_info.get('output_file')
                    if output_file:
                        configured_ext = str(os.environ.get('TTS_AUDIO_FORMAT') or 'mp3').lower().strip().lstrip('.')
                        for stem in self._audio_stem_variants(output_file):
                            for ext in [configured_ext, 'mp3', 'wav']:
                                if ext:
                                    candidates.append(os.path.join('text_to_speech', f"{stem}.{ext}"))

                    seen = set()
                    paths = []
                    for candidate in candidates:
                        normalized = str(candidate).replace('\\', '/')
                        if normalized in seen:
                            continue
                        seen.add(normalized)
                        full_path = normalized if os.path.isabs(normalized) else os.path.join(data['output_dir'], normalized)
                        paths.append(full_path)
                    return paths

                for ch_info in selected_chapters:
                    match = None
                    progress_key = ch_info.get('progress_key')
                    if progress_key and progress_key in data['prog'].get("chapters", {}):
                        match = (progress_key, data['prog']["chapters"][progress_key])
                    else:
                        match = _find_progress_entry(ch_info, data['prog'])

                    ch_entry = match[1] if match else {}
                    deleted_for_chapter = False
                    for audio_path in _audio_candidates(ch_entry, ch_info):
                        try:
                            if os.path.exists(audio_path):
                                os.remove(audio_path)
                                deleted_count += 1
                                deleted_for_chapter = True
                                print(f"Deleted TTS audio: {audio_path}")
                        except Exception as e:
                            print(f"Failed to delete TTS audio {audio_path}: {e}")

                    if not deleted_for_chapter:
                        missing_audio_count += 1

                    if match:
                        chapter_key, ch_entry = match
                        ch_entry["tts_status"] = "no_tts"
                        ch_entry.pop("tts_file", None)
                        ch_entry.pop("tts_at", None)
                        ch_entry.pop("tts_error", None)
                        ch_entry["last_updated"] = time.time()
                        progress_updated = True
                        status_reset_count += 1
                        print(f"Reset TTS status to no_tts for chapter {ch_info.get('num')} (key: {chapter_key})")
                    else:
                        print(f"WARNING: Could not find exact progress entry for {ch_info.get('output_file')}; skipped TTS status reset")

                if progress_updated:
                    try:
                        with open(data['progress_file'], 'w', encoding='utf-8') as f:
                            json.dump(data['prog'], f, ensure_ascii=False, indent=2)
                        print(f"Updated progress tracking file - reset {status_reset_count} TTS statuses to no_tts")
                    except Exception as e:
                        print(f"Failed to update progress file: {e}")

                data['skip_cleanup'] = True
                self._refresh_retranslation_data(data)

                success_parts = []
                if deleted_count > 0:
                    success_parts.append(f"deleted {deleted_count} TTS file(s)")
                if status_reset_count > 0:
                    success_parts.append(f"marked {status_reset_count} chapter(s) as No TTS")
                if missing_audio_count > 0:
                    success_parts.append(f"{missing_audio_count} chapter(s) had no audio file on disk")
                message = "Successfully " + ", ".join(success_parts) + "." if success_parts else "No TTS changes made."
                self._styled_msgbox(QMessageBox.Information, data.get('dialog', self), "TTS Reset", message)
                return
            
            # Count different types
            missing_count = sum(1 for ch in selected_chapters if ch['status'] == 'not_translated')
            existing_count = sum(1 for ch in selected_chapters if ch['status'] != 'not_translated')
            
            count = len(selected_chapters)
            if count > 10:
                if missing_count > 0 and existing_count > 0:
                    confirm_msg = f"This will:\n• Mark {missing_count} missing chapters for translation\n• Delete and retranslate {existing_count} existing chapters\n\nTotal: {count} chapters\n\nContinue?"
                elif missing_count > 0:
                    confirm_msg = f"This will mark {missing_count} missing chapters for translation.\n\nContinue?"
                else:
                    confirm_msg = f"This will delete {existing_count} translated chapters and mark them for retranslation.\n\nContinue?"
            else:
                chapters = [f"Ch.{ch['num']}" for ch in selected_chapters]
                confirm_msg = f"This will process:\n\n{', '.join(chapters)}\n\n"
                if missing_count > 0:
                    confirm_msg += f"• {missing_count} missing chapters will be marked for translation\n"
                if existing_count > 0:
                    confirm_msg += f"• {existing_count} existing chapters will be deleted and retranslated\n"
                confirm_msg += "\nContinue?"
            
            reply = self._styled_msgbox(QMessageBox.Question, data.get('dialog', self), "Confirm Retranslation", confirm_msg,
                                       QMessageBox.Yes | QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
            
            # Process chapters - DELETE FILES AND UPDATE PROGRESS
            deleted_count = 0
            marked_count = 0
            status_reset_count = 0
            refinement_cleared_count = 0
            merged_cleared_count = 0
            progress_updated = False

            for ch_info in selected_chapters:
                output_file = ch_info['output_file']
                actual_num = ch_info['num']
                progress_key = ch_info.get('progress_key')
                
                if ch_info['status'] != 'not_translated':
                    # Reset status to pending for ALL non-not_translated chapters, but only if we can match the exact progress entry
                    match = None
                    if progress_key and progress_key in data['prog']["chapters"]:
                        match = (progress_key, data['prog']["chapters"][progress_key])
                    else:
                        match = _find_progress_entry(ch_info, data['prog'])
                    old_status = ch_info['status']
                    
                    if match:
                        # Delete existing file only after we know which entry to update
                        if output_file:
                            output_path = os.path.join(data['output_dir'], output_file)
                            try:
                                if os.path.exists(output_path):
                                    os.remove(output_path)
                                    deleted_count += 1
                                    print(f"Deleted: {output_path}")
                            except Exception as e:
                                print(f"Failed to delete {output_path}: {e}")
                        chapter_key, ch_entry = match
                        target_output_file = ch_entry.get('output_file') or ch_info['output_file']
                        print(f"Resetting {old_status} status to pending for chapter {actual_num} (key: {chapter_key}, output file: {target_output_file})")
                        ch_entry["status"] = "pending"
                        ch_entry["failure_reason"] = ""
                        ch_entry["error_message"] = ""
                        if _clear_refinement_progress_fields(ch_entry):
                            refinement_cleared_count += 1
                        progress_updated = True
                        status_reset_count += 1
                    else:
                        print(f"WARNING: Could not find exact progress entry for {output_file}; skipped deletion and status reset")
                    
                    # MERGED CHILDREN FIX: Clear any merged children of this chapter
                    # ONLY clear children that still have "merged" status
                    # If split-the-merge succeeded, children will have their own status (completed/qa_failed)
                    # and should NOT be deleted when parent is retranslated
                    for child_key, child_data in list(data['prog']["chapters"].items()):
                        child_status = child_data.get("status")
                        if child_status == "merged" and child_data.get("merged_parent_chapter") == actual_num:
                            child_actual_num = child_data.get("actual_num")
                            print(f"🔓 Clearing merged status for child chapter {child_actual_num} (parent {actual_num} being retranslated)")
                            del data['prog']["chapters"][child_key]
                            merged_cleared_count += 1
                            progress_updated = True
                else:
                    # Just marking for translation (no file to delete)
                    marked_count += 1
            
            # Save the updated progress if we made changes
            if progress_updated:
                try:
                    with open(data['progress_file'], 'w', encoding='utf-8') as f:
                        json.dump(data['prog'], f, ensure_ascii=False, indent=2)
                    print(f"Updated progress tracking file - reset {status_reset_count} chapter statuses to pending")
                except Exception as e:
                    print(f"Failed to update progress file: {e}")
            
            # Auto-refresh the display to show updated status
            data['skip_cleanup'] = True  # Disable cleanup for this dialog after retranslate to avoid deleting pending/failed
            self._refresh_retranslation_data(data)
            
            # Build success message
            success_parts = []
            if deleted_count > 0:
                success_parts.append(f"Deleted {deleted_count} files")
            if marked_count > 0:
                success_parts.append(f"marked {marked_count} missing chapters for translation")
            if status_reset_count > 0:
                success_parts.append(f"reset {status_reset_count} chapter statuses to pending")
            if refinement_cleared_count > 0:
                success_parts.append(f"cleared refinement state for {refinement_cleared_count} chapter(s)")
            if merged_cleared_count > 0:
                success_parts.append(f"cleared {merged_cleared_count} merged child chapters")
            
            if success_parts:
                success_msg = "Successfully " + ", ".join(success_parts) + "."
                if deleted_count > 0 or marked_count > 0 or merged_cleared_count > 0:
                    total_to_translate = len(selected_indices) + merged_cleared_count
                    success_msg += f"\n\nTotal {total_to_translate} chapters ready for translation."
                self._styled_msgbox(QMessageBox.Information, data.get('dialog', self), "Success", success_msg)
            else:
                self._styled_msgbox(QMessageBox.Information, data.get('dialog', self), "Info", "No changes made.")
        
        # Add buttons - First row
        btn_select_all = QPushButton("Select All")
        btn_select_all.setMinimumHeight(32)
        btn_select_all.setStyleSheet("QPushButton { background-color: #17a2b8; color: white; padding: 6px 16px; font-weight: bold; font-size: 10pt; }")
        btn_select_all.clicked.connect(select_all)
        button_layout.addWidget(btn_select_all, 0, 0)
        
        btn_clear = QPushButton("Clear")
        btn_clear.setMinimumHeight(32)
        btn_clear.setStyleSheet("QPushButton { background-color: #6c757d; color: white; padding: 6px 16px; font-weight: bold; font-size: 10pt; }")
        btn_clear.clicked.connect(clear_selection)
        button_layout.addWidget(btn_clear, 0, 1)
        
        btn_select_completed = QPushButton("Select Completed")
        btn_select_completed.setMinimumHeight(32)
        btn_select_completed.setStyleSheet("QPushButton { background-color: #28a745; color: white; padding: 6px 16px; font-weight: bold; font-size: 10pt; }")
        btn_select_completed.clicked.connect(lambda: select_status('completed'))
        button_layout.addWidget(btn_select_completed, 0, 2)
        
        btn_select_qa_failed = QPushButton("Select QA Failed")
        btn_select_qa_failed.setMinimumHeight(32)
        # Use red for QA Failed
        btn_select_qa_failed.setStyleSheet("QPushButton { background-color: #dc3545; color: white; padding: 6px 16px; font-weight: bold; font-size: 10pt; }")
        btn_select_qa_failed.clicked.connect(lambda: select_status('qa_failed'))
        button_layout.addWidget(btn_select_qa_failed, 0, 3)
        
        btn_select_failed = QPushButton("Select Failed")
        btn_select_failed.setMinimumHeight(32)
        # Use red for Failed / QA Failed
        btn_select_failed.setStyleSheet("QPushButton { background-color: #dc3545; color: white; padding: 6px 16px; font-weight: bold; font-size: 10pt; }")
        btn_select_failed.clicked.connect(lambda: select_status('failed'))
        button_layout.addWidget(btn_select_failed, 0, 4)
        
        # Second row
        btn_retranslate = QPushButton("Reset TTS Selected" if self._current_progress_output_mode(data) == 'audio' else "Retranslate Selected")
        btn_retranslate.setMinimumHeight(32)
        btn_retranslate.setStyleSheet("QPushButton { background-color: #d39e00; color: white; padding: 6px 16px; font-weight: bold; font-size: 10pt; }")
        btn_retranslate.clicked.connect(retranslate_selected)
        button_layout.addWidget(btn_retranslate, 1, 0, 1, 2)
        
        btn_remove_qa = QPushButton("Remove QA Failed Mark")
        btn_remove_qa.setMinimumHeight(32)
        btn_remove_qa.setStyleSheet("QPushButton { background-color: #28a745; color: white; padding: 6px 16px; font-weight: bold; font-size: 10pt; }")
        btn_remove_qa.clicked.connect(remove_qa_failed_mark)
        button_layout.addWidget(btn_remove_qa, 1, 2, 1, 1)
        
        # Add animated refresh button
        btn_refresh = AnimatedRefreshButton("  Refresh")  # Double space for icon padding
        btn_refresh.setMinimumHeight(32)
        btn_refresh.setStyleSheet(
            "QPushButton { "
            "background-color: #17a2b8; "
            "color: white; "
            "padding: 6px 16px; "
            "font-weight: bold; "
            "font-size: 10pt; "
            "}"
            "QPushButton[refreshActive=\"true\"] { "
            "background-color: #138496; "
            "}"
        )
        
        # Create refresh handler with animation
        def animated_refresh():
            import time

            btn_refresh.start_animation()
            btn_refresh.setEnabled(False)

            # Track start time for minimum animation duration
            start_time = time.time()
            min_animation_duration = 0.8  # 800ms minimum

            # A token to prevent older timers from firing after a newer refresh click
            refresh_token = time.time()
            data['_last_refresh_token'] = refresh_token

            def _rebuild_gui_from_refresh():
                """Recreate the retranslation GUI if refresh appears to have failed to render."""
                try:
                    dlg = data.get('dialog')

                    # Best-effort capture current toggle state
                    show_special = data.get('show_special_files_state', False)
                    cb = data.get('show_special_files_cb')
                    if cb:
                        try:
                            show_special = cb.isChecked()
                        except RuntimeError:
                            pass

                    # Multi-file dialog: destroy and recreate the whole multi-tab window
                    if dlg and hasattr(dlg, '_tab_data'):
                        selection = None
                        if hasattr(self, '_multi_file_selection_key') and self._multi_file_selection_key:
                            try:
                                selection = list(self._multi_file_selection_key)
                            except Exception:
                                selection = None

                        def do_multi_rebuild():
                            try:
                                # Clear cached multi-file dialog so the recreate path is taken
                                if hasattr(self, '_multi_file_retranslation_dialog'):
                                    self._multi_file_retranslation_dialog = None
                                if hasattr(self, '_multi_file_selection_key'):
                                    self._multi_file_selection_key = None

                                try:
                                    dlg.hide()
                                except Exception:
                                    pass
                                try:
                                    dlg.deleteLater()
                                except Exception:
                                    pass

                                if selection is not None:
                                    self.selected_files = selection
                                    self._force_retranslation_multiple_files()
                            except Exception as e:
                                print(f"Error during multi-file rebuild: {e}")

                        QTimer.singleShot(0, do_multi_rebuild)
                        return

                    # Single-file dialog: remove cached entry and recreate
                    file_path = data.get('file_path')
                    if not file_path:
                        return

                    file_key = os.path.abspath(file_path)
                    if hasattr(self, '_retranslation_dialog_cache') and file_key in self._retranslation_dialog_cache:
                        try:
                            del self._retranslation_dialog_cache[file_key]
                        except Exception:
                            pass

                    old_dlg = dlg

                    def do_single_rebuild():
                        try:
                            if old_dlg:
                                try:
                                    old_dlg.hide()
                                except Exception:
                                    pass
                                try:
                                    old_dlg.deleteLater()
                                except Exception:
                                    pass
                            self._show_retranslation_shell_then_build(file_path, show_special_files_state=show_special)
                        except Exception as e:
                            print(f"Error during rebuild: {e}")

                    QTimer.singleShot(0, do_single_rebuild)

                except Exception as e:
                    print(f"Error during rebuild: {e}")

            # Use QTimer to run refresh after animation starts
            def do_refresh():
                try:
                    # Always refresh only this tab's data (not all tabs)
                    self._refresh_retranslation_data(data)

                    # Schedule watchdog: if after 3 seconds there are still no visible entries,
                    # but our data says there should be, rebuild the GUI.
                    def watchdog_check():
                        try:
                            if data.get('_last_refresh_token') != refresh_token:
                                return  # superseded by a newer refresh

                            expected_total = len(data.get('chapter_display_info', []) or [])
                            if expected_total <= 0:
                                return

                            listbox = data.get('listbox')
                            if not listbox:
                                _rebuild_gui_from_refresh()
                                return

                            try:
                                count = listbox.count()
                            except RuntimeError:
                                _rebuild_gui_from_refresh()
                                return

                            visible = 0
                            try:
                                for i in range(count):
                                    item = listbox.item(i)
                                    if item is not None and not item.isHidden():
                                        visible += 1
                            except RuntimeError:
                                _rebuild_gui_from_refresh()
                                return

                            if visible > 0:
                                return

                            # Don't rebuild if everything is hidden purely due to the special-files filter.
                            try:
                                show_special = data.get('show_special_files_state', False)
                                cb = data.get('show_special_files_cb')
                                if cb:
                                    show_special = cb.isChecked()
                                if not show_special:
                                    infos = data.get('chapter_display_info', []) or []
                                    if infos and all(bool(info.get('is_special', False)) for info in infos):
                                        return
                            except Exception:
                                pass

                            _rebuild_gui_from_refresh()
                        except Exception as e:
                            print(f"Watchdog check error: {e}")

                    QTimer.singleShot(3000, watchdog_check)

                    # Calculate remaining time to meet minimum animation duration
                    elapsed = time.time() - start_time
                    remaining = max(0, min_animation_duration - elapsed)

                    # Schedule animation stop after remaining time
                    def finish_animation():
                        btn_refresh.stop_animation()
                        btn_refresh.setEnabled(True)

                    if remaining > 0:
                        QTimer.singleShot(int(remaining * 1000), finish_animation)
                    else:
                        finish_animation()

                except Exception as e:
                    print(f"Error during refresh: {e}")
                    btn_refresh.stop_animation()
                    btn_refresh.setEnabled(True)

            QTimer.singleShot(50, do_refresh)  # Small delay to let animation start
        
        btn_refresh.clicked.connect(animated_refresh)
        button_layout.addWidget(btn_refresh, 1, 3, 1, 1)

        # Expose refresh handler for external triggers (e.g., Progress Manager reopen)
        data['refresh_func'] = animated_refresh
        if data.get('dialog'):
            setattr(data['dialog'], '_refresh_func', animated_refresh)

        # Auto-refresh every 3 seconds (silent, no animation)
        def _silent_refresh():
            try:
                # Skip if a manual refresh is already in progress
                if not btn_refresh.isEnabled():
                    return
                dlg = data.get('dialog')
                if dlg and dlg.isVisible():
                    self._refresh_retranslation_data(data)
            except Exception:
                pass

        _auto_refresh_timer = QTimer(data.get('dialog') or self)
        _auto_refresh_timer.setInterval(2000)
        _auto_refresh_timer.timeout.connect(_silent_refresh)
        _auto_refresh_timer.start()
        data['_auto_refresh_timer'] = _auto_refresh_timer

        # ==== Context menu on listbox ====
        listbox = data['listbox']
        listbox.setContextMenuPolicy(Qt.CustomContextMenu)

        def _open_file_for_item(display_info):
            """Open the output file for a chapter. Accepts pre-extracted display_info dict."""
            output_file = display_info.get('output_file')
            if not output_file:
                self._show_message('error', "File Missing", "No output file recorded for this entry.", parent=data.get('dialog', self))
                return
            resolved_file, path = self._resolve_existing_output_path(
                data['output_dir'],
                output_file,
                display_info,
                data.get('prog'),
            )
            if not path or not os.path.exists(path):
                missing_path = os.path.join(data['output_dir'], output_file)
                self._show_message('error', "File Missing", f"File not found:\n{missing_path}", parent=data.get('dialog', self))
                return
            if resolved_file and resolved_file != output_file:
                display_info['output_file'] = resolved_file
            try:
                QDesktopServices.openUrl(QUrl.fromLocalFile(path))
            except Exception as e:
                self._show_message('error', "Open Failed", str(e), parent=data.get('dialog', self))

        def _find_audio_file_for_item(display_info):
            """Return the generated TTS file path associated with an HTML row, if one exists."""
            progress_entry = display_info.get('info', {}) or {}
            output_file = display_info.get('output_file') or progress_entry.get('output_file')
            candidates = []

            stored_tts_file = progress_entry.get('tts_file')
            if stored_tts_file:
                candidates.append(stored_tts_file)

            progress_key = display_info.get('progress_key')
            if progress_key and progress_key in data.get('prog', {}).get('chapters', {}):
                tracked_tts_file = data['prog']['chapters'][progress_key].get('tts_file')
                if tracked_tts_file:
                    candidates.append(tracked_tts_file)

            if output_file:
                for _key, tracked in data.get('prog', {}).get('chapters', {}).items():
                    if isinstance(tracked, dict) and tracked.get('output_file') == output_file and tracked.get('tts_file'):
                        candidates.append(tracked.get('tts_file'))

                for stem in self._audio_stem_variants(output_file):
                    for ext in ("wav", "mp3", "pcm", "m4a", "ogg", "flac"):
                        candidates.append(os.path.join("text_to_speech", f"{stem}.{ext}"))

            seen = set()
            for candidate in candidates:
                if not candidate:
                    continue
                normalized = str(candidate).replace("\\", "/")
                if normalized in seen:
                    continue
                seen.add(normalized)
                path = normalized if os.path.isabs(normalized) else os.path.join(data['output_dir'], normalized)
                if os.path.exists(path):
                    return path
            return None

        def _reset_tts_progress_for_output(output_file):
            if not output_file:
                return 0
            updated = 0
            now = time.time()
            for _key, tracked in data.get('prog', {}).get('chapters', {}).items():
                if not isinstance(tracked, dict):
                    continue
                if tracked.get('output_file') != output_file:
                    continue
                tracked['tts_status'] = 'no_tts'
                tracked.pop('tts_file', None)
                tracked.pop('tts_at', None)
                tracked.pop('tts_error', None)
                tracked['last_updated'] = now
                updated += 1
            if updated:
                with open(data['progress_file'], 'w', encoding='utf-8') as f:
                    json.dump(data['prog'], f, ensure_ascii=False, indent=2)
            return updated

        def _open_audio_file_for_item(display_info):
            audio_path = _find_audio_file_for_item(display_info)
            if not audio_path:
                self._show_message('error', "Audio Missing", "No generated audio file was found for this HTML entry.", parent=data.get('dialog', self))
                return
            try:
                QDesktopServices.openUrl(QUrl.fromLocalFile(audio_path))
            except Exception as e:
                self._show_message('error', "Open Failed", str(e), parent=data.get('dialog', self))

        def _delete_audio_file_for_item(display_info):
            audio_path = _find_audio_file_for_item(display_info)
            if not audio_path:
                self._show_message('info', "Audio Missing", "No generated audio file was found for this HTML entry.", parent=data.get('dialog', self))
                return
            reply = self._styled_msgbox(
                QMessageBox.Question,
                data.get('dialog', self),
                "Delete Audio File",
                f"Delete this generated audio file?\n\n{audio_path}",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
            try:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            except Exception as e:
                self._show_message('error', "Delete Failed", str(e), parent=data.get('dialog', self))
                return
            _reset_tts_progress_for_output(display_info.get('output_file'))
            self._refresh_retranslation_data(data)
            self._show_message('info', "Audio Deleted", "Audio file deleted and TTS status reset to No TTS.", parent=data.get('dialog', self))

        def show_context_menu(pos):
            item = listbox.itemAt(pos)
            if not item:
                return
            if not item.isSelected():
                listbox.clearSelection()
                item.setSelected(True)
                listbox.setCurrentItem(item)
            
            # IMPORTANT: Extract ALL data from the item BEFORE menu.exec() blocks.
            # The auto-refresh timer (2s) can rebuild the listbox and delete C++ objects
            # while the context menu is open, making the item reference stale.
            try:
                info_wrapper = item.data(Qt.UserRole)
                if not info_wrapper:
                    return
                display_info = info_wrapper.get('info', {})
                item_text = item.text()
            except RuntimeError:
                # C++ object already deleted
                return
            
            # The actual progress entry is nested inside 'info' key of display_info
            progress_entry = display_info.get('info', {})
            
            # qa_issues is a boolean flag; the actual list is qa_issues_found
            qa_issues = progress_entry.get('qa_issues_found', [])
            if not isinstance(qa_issues, list):
                qa_issues = []
                
            has_missing_images = any('missing_images' in str(issue) for issue in qa_issues)
            
            # Fallback: Check item text directly as it definitely contains the issue if visible
            if not has_missing_images and 'missing_images' in item_text:
                has_missing_images = True
                print("DEBUG: Detected missing_images via list item text")
            
            # Determine file path for Notepad action
            _output_file = display_info.get('output_file')
            qa_file_path = os.path.join(data['output_dir'], _output_file) if _output_file else None
            
            menu = QMenu(listbox)
            # Remove extra left gutter reserved for icons to avoid empty space
            menu.setStyleSheet(
                "QMenu {"
                "  padding: 4px;"
                "  background-color: #2b2b2b;"
                "  color: white;"
                "  border: 1px solid #5a9fd4;"
                "} "
                "QMenu::icon { width: 0px; } "
                "QMenu::item {"
                "  padding: 6px 12px;"
                "  background-color: transparent;"
                "} "
                "QMenu::item:selected {"
                "  background-color: #17a2b8;"
                "  color: white;"
                "} "
                "QMenu::item:pressed {"
                "  background-color: #138496;"
                "}"
            )
            act_open = menu.addAction("📂 Open File")
            act_open_audio = None
            act_delete_audio = None
            if _find_audio_file_for_item(display_info):
                act_open_audio = menu.addAction("🔊 Open Audio File")
                act_delete_audio = menu.addAction("🗑️ Delete Audio File")
            act_notepad_qa = None
            if qa_file_path:
                _label = "✏️ Edit File (find QA issue)" if qa_issues else "✏️ Edit File"
                act_notepad_qa = menu.addAction(_label)
            act_retranslate = menu.addAction("🔁 Retranslate Selected")
            
            act_insert_img = None
            if has_missing_images:
                act_insert_img = menu.addAction("🖼️ Insert Missing Image")
                
            act_remove_qa = menu.addAction("🧹 Remove QA Failed Mark")
            selected_infos = []
            try:
                for selected_item in listbox.selectedItems():
                    wrapper = selected_item.data(Qt.UserRole) or {}
                    selected_infos.append(wrapper.get('info', {}))
            except RuntimeError:
                selected_infos = [display_info]
            act_restore_in_progress = None
            if any((info or {}).get('status') == 'in_progress' for info in selected_infos):
                act_restore_in_progress = menu.addAction("Restore In Progress Status")
            chosen = menu.exec(listbox.mapToGlobal(pos))
            if chosen == act_open:
                _open_file_for_item(display_info)
            elif act_open_audio and chosen == act_open_audio:
                _open_audio_file_for_item(display_info)
            elif act_delete_audio and chosen == act_delete_audio:
                _delete_audio_file_for_item(display_info)
            elif chosen == act_retranslate:
                retranslate_selected()
            elif act_insert_img and chosen == act_insert_img:
                # IN-PLACE RESTORATION LOGIC using ContentProcessor
                try:
                    from bs4 import BeautifulSoup
                    import zipfile
                    from TransateKRtoEN import ContentProcessor
                    
                    # Load rename map from output directory
                    rename_map = None
                    rename_map_path = os.path.join(data['output_dir'], 'image_rename_map.json')
                    if os.path.exists(rename_map_path):
                        try:
                            with open(rename_map_path, 'r', encoding='utf-8') as f:
                                rename_map = json.load(f) or {}
                        except Exception:
                            pass

                    # 1. Get Source Content from EPUB
                    epub_path = data['file_path']
                    original_filename = display_info.get('original_filename')
                    source_html = None
                    
                    if original_filename:
                        try:
                            def normalize_name(n):
                                base = os.path.basename(n)
                                if base.startswith('response_'):
                                    base = base[9:]
                                return os.path.splitext(base)[0].lower()
                                
                            target_base = normalize_name(original_filename)
                            
                            with zipfile.ZipFile(epub_path, 'r') as zf:
                                for fname in zf.namelist():
                                    if normalize_name(fname) == target_base:
                                        source_html = zf.read(fname).decode('utf-8', errors='ignore')
                                        break
                        except Exception as ex:
                            print(f"Extraction error: {ex}")
                    
                    if not source_html:
                        self._show_message('error', "Error", "Could not extract source HTML for this chapter.")
                    else:
                        # 2. Get Translated Content
                        output_file = display_info.get('output_file')
                        output_path = os.path.join(data['output_dir'], output_file)
                        
                        if os.path.exists(output_path):
                            with open(output_path, 'r', encoding='utf-8') as f:
                                translated_html = f.read()
                                
                            # 3. Restore using ContentProcessor (supports all image formats + rename map)
                            restored_html = ContentProcessor.emergency_restore_images(
                                translated_html, source_html, verbose=True, rename_map=rename_map
                            )
                            
                            if restored_html != translated_html:
                                # 4. Save
                                with open(output_path, 'w', encoding='utf-8') as f:
                                    f.write(restored_html)
                                    
                                # 5. Update Progress (Clear QA flags)
                                found_key = None
                                target_out = display_info.get('output_file')
                                
                                if target_out:
                                    for k, v in data['prog'].get('chapters', {}).items():
                                        if v.get('output_file') == target_out:
                                            found_key = k
                                            break
                                
                                if found_key:
                                    real_entry = data['prog']['chapters'][found_key]
                                    real_entry['status'] = 'completed'
                                    for key in ['qa_issues', 'qa_issues_found', 'qa_timestamp', 'failure_reason', 'error_message']:
                                        real_entry.pop(key, None)
                                else:
                                    progress_entry['status'] = 'completed'
                                    for key in ['qa_issues', 'qa_issues_found', 'qa_timestamp', 'failure_reason', 'error_message']:
                                        progress_entry.pop(key, None)
                                
                                # Save progress
                                with open(data['progress_file'], 'w', encoding='utf-8') as f:
                                    json.dump(data['prog'], f, ensure_ascii=False, indent=2)
                                    
                                # 6. Refresh
                                self._refresh_retranslation_data(data)
                                self._show_message('info', "Success", "Images restored and QA flags cleared.")
                            else:
                                self._show_message('info', "Info", "No missing images could be automatically restored.")
                        else:
                            self._show_message('error', "Error", "Output file not found.")
                            
                except Exception as e:
                    self._show_message('error', "Error", f"Failed to restore images: {e}")
                    import traceback
                    traceback.print_exc()
            elif act_restore_in_progress and chosen == act_restore_in_progress:
                restore_in_progress_marks()
            elif chosen == act_remove_qa:
                remove_qa_failed_mark()
            elif act_notepad_qa and chosen == act_notepad_qa:
                search_term = None
                _line_num = 1
                if qa_issues:
                    # Extract a meaningful search term from the QA issue strings
                    # Try all common delimiter styles in order
                    _QUOTE_PATTERNS = [
                        r"'([^']+)'",                    # single quotes: 'text'
                        r'"([^"]+)"',                   # double quotes: "text"
                        r"\u201c([^\u201d]+)\u201d",    # curly double quotes: “text”
                        r"\u2018([^\u2019]+)\u2019",    # curly single quotes: ‘text’
                        r"\u300c([^\u300d]+)\u300d",    # Japanese corner brackets: 「text」
                        r"\u300e([^\u300f]+)\u300f",    # Japanese white corner brackets: 『text』
                        r"\uff62([^\uff63]+)\uff63",    # Halfwidth corner brackets
                        r"\[([^\]]+)\]",              # square brackets: [text]
                        r"\(([^)]+)\)",               # parentheses: (text)
                    ]
                    for _issue in qa_issues:
                        _s = str(_issue)
                        for _pat in _QUOTE_PATTERNS:
                            _m = re.search(_pat, _s)
                            if _m and _m.group(1).strip():
                                search_term = _m.group(1)
                                break
                        if search_term:
                            break
                    # Fallback: scan file for any non-ASCII sequence
                    if not search_term:
                        try:
                            with open(qa_file_path, 'r', encoding='utf-8', errors='ignore') as _f:
                                _content = _f.read()
                            _m = re.search(r'[^\x00-\x7f]{1,30}', _content)
                            if _m:
                                search_term = _m.group(0)
                        except Exception:
                            pass
                    # Find line number of search term in file
                    # Try progressively shorter prefixes in case the QA term is truncated
                    if search_term and os.path.exists(qa_file_path):
                        try:
                            with open(qa_file_path, 'r', encoding='utf-8', errors='ignore') as _f:
                                _lines = _f.readlines()
                            # Strip surrounding quote/bracket chars so we search raw content
                            _STRIP_QUOTES = '\'"「」『』“”‘’｢｣《》〈〉（）'
                            _bare = search_term.strip(_STRIP_QUOTES)
                            _base = _bare if _bare else search_term
                            # Build candidates: full bare term, then shrinking prefixes (min 1 char)
                            _candidates = [_base[:_l] for _l in range(len(_base), 0, -1)]
                            for _cand in _candidates:
                                for _i, _ln in enumerate(_lines, 1):
                                    if _cand in _ln:
                                        _line_num = _i
                                        break
                                if _line_num > 1:
                                    break
                        except Exception:
                            pass
                    # Copy search term to clipboard
                    if search_term:
                        from PySide6.QtWidgets import QApplication
                        QApplication.clipboard().setText(search_term)
                # Open file in best available editor, jumping to line if supported
                try:
                    if sys.platform == 'win32':
                        _npp_paths = [
                            r'C:\Program Files\Notepad++\notepad++.exe',
                            r'C:\Program Files (x86)\Notepad++\notepad++.exe',
                        ]
                        _npp = next((p for p in _npp_paths if os.path.exists(p)), None)
                        if _npp:
                            subprocess.Popen([_npp, f'-n{_line_num}', qa_file_path])
                        else:
                            subprocess.Popen(['notepad.exe', qa_file_path])
                    elif sys.platform == 'darwin':
                        # Try TextEdit alternatives that support line jumping
                        if shutil.which('code'):
                            subprocess.Popen(['code', '--goto', f'{qa_file_path}:{_line_num}'])
                        else:
                            subprocess.Popen(['open', '-t', qa_file_path])
                    else:
                        # Linux: try editors with line-jump support first
                        if shutil.which('gedit'):
                            subprocess.Popen(['gedit', f'+{_line_num}', qa_file_path])
                        elif shutil.which('kate'):
                            subprocess.Popen(['kate', '-l', str(_line_num), qa_file_path])
                        elif shutil.which('code'):
                            subprocess.Popen(['code', '--goto', f'{qa_file_path}:{_line_num}'])
                        else:
                            _linux_editors = ['mousepad', 'xed', 'pluma', 'nano', 'xdg-open']
                            _editor = next((e for e in _linux_editors if shutil.which(e)), 'xdg-open')
                            subprocess.Popen([_editor, qa_file_path])
                except Exception as _e:
                    self._show_message('error', "Open Failed", f"Could not open editor:\n{_e}",
                                       parent=data.get('dialog', self))

        listbox.customContextMenuRequested.connect(show_context_menu)
        
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setMinimumHeight(32)
        btn_cancel.setStyleSheet("QPushButton { background-color: #6c757d; color: white; padding: 6px 16px; font-weight: bold; font-size: 10pt; }")
        btn_cancel.clicked.connect(lambda: data['dialog'].close() if data.get('dialog') else None)
        button_layout.addWidget(btn_cancel, 1, 4, 1, 1)

        # Automatically refresh once when dialog is opened
        # Skip for multi-tab dialogs — the parent will do a single bulk refresh
        is_multi_tab = data.get('dialog') and hasattr(data['dialog'], '_tab_data')
        if not is_multi_tab:
            animated_refresh()

    def _refresh_all_tabs(self, tab_data_list):
        """Refresh all tabs in a multi-file retranslation dialog"""
        try:
            print(f"🔄 Refreshing all {len(tab_data_list)} tabs...")
            
            refreshed_count = 0
            skipped_count = 0
            for idx, data in enumerate(tab_data_list):
                if data and data.get('type') != 'image_folder' and data.get('type') != 'individual_images':
                    # Only refresh EPUB/text tabs
                    try:
                        # Check if widgets are still valid before attempting refresh
                        if not self._is_data_valid(data):
                            print(f"[DEBUG] Skipping tab {idx + 1}/{len(tab_data_list)} - widgets deleted")
                            skipped_count += 1
                            continue
                        
                        print(f"[DEBUG] Refreshing tab {idx + 1}/{len(tab_data_list)}")
                        self._refresh_retranslation_data(data)
                        refreshed_count += 1
                    except RuntimeError as e:
                        # Widget was deleted
                        print(f"[WARN] Skipping tab {idx + 1} - widget deleted: {e}")
                        skipped_count += 1
                    except Exception as e:
                        print(f"[ERROR] Failed to refresh tab {idx + 1}: {e}")
            
            if skipped_count > 0:
                print(f"✅ Successfully refreshed {refreshed_count} tab(s), skipped {skipped_count} deleted tab(s)")
            else:
                print(f"✅ Successfully refreshed {refreshed_count} tab(s)")
            
        except Exception as e:
            print(f"❌ Failed to refresh all tabs: {e}")
            import traceback
            traceback.print_exc()
    
    def _is_data_valid(self, data):
        """Check if the data structure has valid (non-deleted) widgets"""
        try:
            if not data:
                return False
            
            # Check if listbox exists and is still valid
            listbox = data.get('listbox')
            if not listbox:
                return False
            
            # Try to access a simple property to check if widget is still alive
            # This will raise RuntimeError if the C++ object was deleted
            listbox.count()
            return True
            
        except (RuntimeError, AttributeError):
            return False
    
    def _refresh_retranslation_data(self, data):
        """Refresh the retranslation dialog data by reloading progress and updating display"""
        updates_were_enabled = True
        signals_were_blocked = False
        try:
            # First check if widgets are still valid
            if not self._is_data_valid(data):
                print("⚠️ Cannot refresh - widgets have been deleted")
                return

            # If the output override directory changed while the dialog is open,
            # re-resolve output_dir/progress_file so we don't keep reading the old progress JSON.
            try:
                file_path = data.get('file_path')
                if file_path:
                    epub_base = os.path.splitext(os.path.basename(file_path))[0]
                    override_dir = (os.environ.get('OUTPUT_DIRECTORY') or os.environ.get('OUTPUT_DIR'))
                    if not override_dir and hasattr(self, 'config'):
                        try:
                            override_dir = self.config.get('output_directory')
                        except Exception:
                            override_dir = None

                    expected_output_dir = os.path.join(override_dir, epub_base) if override_dir else epub_base
                    # On macOS .app bundles, cwd can be '/' (read-only root).
                    # Resolve relative output paths against the input file's directory.
                    # Only on macOS — on Windows this would change the output dir and break progress tracking.
                    if _IS_MACOS and not os.path.isabs(expected_output_dir):
                        expected_output_dir = os.path.join(os.path.dirname(os.path.abspath(file_path)), expected_output_dir)
                    expected_progress_file = os.path.join(expected_output_dir, "translation_progress.json")

                    # Update in-place if changed
                    if expected_output_dir and data.get('output_dir') != expected_output_dir:
                        data['output_dir'] = expected_output_dir
                    if expected_progress_file and data.get('progress_file') != expected_progress_file:
                        data['progress_file'] = expected_progress_file

                    # Keep cache consistent too (if present)
                    try:
                        file_key = os.path.abspath(file_path)
                        if hasattr(self, '_retranslation_dialog_cache') and file_key in self._retranslation_dialog_cache:
                            cached = self._retranslation_dialog_cache[file_key]
                            if isinstance(cached, dict):
                                cached['output_dir'] = data.get('output_dir')
                                cached['progress_file'] = data.get('progress_file')
                    except Exception:
                        pass
            except Exception as e:
                print(f"[WARN] Could not re-resolve output override on refresh: {e}")

            def _read_progress_json_safely(path):
                import random
                import time as _time
                last_error = None
                for _attempt in range(20):
                    try:
                        with open(path, 'r', encoding='utf-8') as f:
                            loaded = json.load(f)
                        data['_last_good_prog'] = copy.deepcopy(loaded)
                        return loaded
                    except (PermissionError, FileNotFoundError, json.JSONDecodeError, OSError) as e:
                        last_error = e
                        _time.sleep(min(0.5, 0.03 * (2 ** min(_attempt, 5))) + random.uniform(0, 0.03))
                snapshot = data.get('_last_good_prog') or data.get('prog')
                if isinstance(snapshot, dict):
                    print(f"⚠️ Progress file locked during refresh; using last good snapshot this tick: {last_error}")
                    return copy.deepcopy(snapshot)
                raise last_error

            def _write_progress_json_safely(path, payload):
                import random
                import tempfile
                import time as _time
                progress_dir = os.path.dirname(path) or '.'
                if progress_dir:
                    os.makedirs(progress_dir, exist_ok=True)
                last_error = None
                for _attempt in range(20):
                    temp_path = None
                    try:
                        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=progress_dir, delete=False, suffix='.tmp') as tmp:
                            temp_path = tmp.name
                            json.dump(payload, tmp, ensure_ascii=False, indent=2)
                            tmp.flush()
                            try:
                                os.fsync(tmp.fileno())
                            except Exception:
                                pass
                        os.replace(temp_path, path)
                        return True
                    except (PermissionError, OSError) as e:
                        last_error = e
                        if temp_path and os.path.exists(temp_path):
                            try:
                                os.remove(temp_path)
                            except Exception:
                                pass
                        _time.sleep(min(0.5, 0.03 * (2 ** min(_attempt, 5))) + random.uniform(0, 0.03))
                raise last_error
            
            # Save current scroll position (and first visible row/offset) to restore after refresh
            saved_scroll = None
            updates_were_enabled = True
            signals_were_blocked = False
            self._suspend_yield = True
            first_visible_row = None
            first_visible_offset = 0
            if 'listbox' in data and data['listbox']:
                try:
                    from PySide6.QtCore import QPoint
                    saved_scroll = data['listbox'].verticalScrollBar().value()
                    updates_were_enabled = data['listbox'].updatesEnabled()
                    signals_were_blocked = data['listbox'].signalsBlocked()
                    idx = data['listbox'].indexAt(QPoint(0, 0))
                    if idx and idx.isValid():
                        first_visible_row = idx.row()
                        rect = data['listbox'].visualItemRect(data['listbox'].item(first_visible_row))
                        first_visible_offset = -rect.top()
                    data['listbox'].blockSignals(True)
                    data['listbox'].setUpdatesEnabled(False)
                except Exception:
                    saved_scroll = None
            
            # Save current selections to restore after refresh
            selected_indices = []
            try:
                selected_indices = [data['listbox'].row(item) for item in data['listbox'].selectedItems()]
            except RuntimeError:
                print("⚠️ Could not save selection state - widget was deleted")
                return
            
            # Reload progress file - check if it exists first
            if not os.path.exists(data['progress_file']):
                print(f"⚠️ Progress file not found: {data['progress_file']}")
                # Recreate a minimal progress file and auto-discover completed files from output_dir
                prog = {
                    "chapters": {},
                    "chapter_chunks": {},
                    "version": "2.1"
                }

                def _auto_discover_from_output_dir(output_dir, prog):
                    updated = False
                    try:
                        files = [
                            f for f in os.listdir(output_dir)
                            if os.path.isfile(os.path.join(output_dir, f))
                            # accept any extension except .epub
                            and not f.lower().endswith("_translated.txt")
                            and f != "translation_progress.json"
                            and not f.lower().endswith(".epub")
                            and not f.lower().endswith(".cache")
                        ]
                        for fname in files:
                            base = os.path.basename(fname)
                            if base.startswith("response_"):
                                base = base[len("response_"):]
                            while True:
                                new_base, ext = os.path.splitext(base)
                                if not ext:
                                    break
                                base = new_base
                            import re
                            m = re.findall(r"(\\d+)", base)
                            chapter_num = int(m[-1]) if m else None
                            key = str(chapter_num) if chapter_num is not None else f"special_{base}"
                            actual_num = chapter_num if chapter_num is not None else 0
                            if key in prog.get("chapters", {}):
                                continue
                            prog.setdefault("chapters", {})[key] = {
                                "actual_num": actual_num,
                                "content_hash": "",
                                "output_file": fname,
                                "status": "completed",
                                "last_updated": os.path.getmtime(os.path.join(output_dir, fname)),
                                "auto_discovered": True,
                                "original_basename": fname
                            }
                            updated = True
                    except Exception as e:
                        print(f"⚠️ Auto-discovery (refresh no OPF) failed: {e}")
                    return updated

                if _auto_discover_from_output_dir(data['output_dir'], prog):
                    print("💾 Recreated progress file via auto-discovery (refresh)")
                try:
                    _write_progress_json_safely(data['progress_file'], prog)
                except PermissionError as e:
                    print(f"⚠️ Progress file locked during refresh recreate; will retry on next refresh tick: {e}")
                    return
                except Exception as e:
                    self._styled_msgbox(QMessageBox.Warning, data.get('dialog', self), "Progress File Error",
                                        f"Could not recreate progress file:\n{e}")
                    return
            
            # The translator may briefly lock/replace the JSON; retry and skip this tick if it stays locked.
            data['prog'] = _read_progress_json_safely(data['progress_file'])
            data['_last_good_prog'] = copy.deepcopy(data['prog'])

            def _progress_has_active_entries(prog):
                try:
                    return any(
                        isinstance(info, dict)
                        and str(info.get('status', '')).lower() == 'in_progress'
                        for info in (prog or {}).get('chapters', {}).values()
                    )
                except Exception:
                    return False
            
            # Clean up missing files and merged children before display unless disabled
            if not data.get('skip_cleanup', False) and not _progress_has_active_entries(data['prog']):
                from TransateKRtoEN import ProgressManager
                before_cleanup = copy.deepcopy(data['prog'])
                temp_progress = ProgressManager(os.path.dirname(data['progress_file']))
                temp_progress.prog = data['prog']
                temp_progress.cleanup_missing_files(data['output_dir'])
                data['prog'] = temp_progress.prog
                
                # Save only if cleanup really changed the file. During active translation
                # refresh should be a reader, not another progress writer.
                if data['prog'] != before_cleanup:
                    _write_progress_json_safely(data['progress_file'], data['prog'])

            if self._reconcile_tts_audio_files(data):
                _write_progress_json_safely(data['progress_file'], data['prog'])
            
            # Check if we're using OPF-based display or fallback
            if data.get('spine_chapters'):
                # OPF-based: Re-run full matching logic to update merged status correctly
                # We need to re-match spine chapters against the updated progress JSON
                self._rematch_spine_chapters(data)
            else:
                # Fallback mode: REBUILD chapter_display_info from scratch to pick up new entries
                # This is necessary for text files or EPUBs without OPF
                self._rebuild_chapter_display_info(data)
            
            # Note: chapter_display_info is already rebuilt/updated above
            # For OPF mode: _update_chapter_status_info updated existing entries
            # For fallback mode: _rebuild_chapter_display_info rebuilt from scratch
            
            # Update the listbox display
            self._update_listbox_display(data)
            
            # Update statistics if available
            self._update_statistics_display(data)

            # Ensure the special-files toggle is applied after every refresh.
            try:
                show_special = data.get('show_special_files_state', False)
                cb = data.get('show_special_files_cb')
                if cb:
                    show_special = cb.isChecked()
                listbox = data.get('listbox')
                if listbox:
                    for i in range(listbox.count()):
                        item = listbox.item(i)
                        if not item:
                            continue
                        meta = item.data(Qt.UserRole) or {}
                        _info = meta.get('info') or {}
                        _fname = _info.get('original_filename', '') or _info.get('output_file', '') or _info.get('key', '')
                        is_skipped_special = self._progress_file_is_skipped_special(
                            _fname,
                            meta.get('is_special', False),
                        )
                        item.setHidden(is_skipped_special and not show_special)
                data['show_special_files_state'] = show_special
            except Exception:
                pass
            
            # Restore scroll position and repaint immediately after rebuild
            if 'listbox' in data and data['listbox']:
                try:
                    sb = data['listbox'].verticalScrollBar()
                    if first_visible_row is not None and first_visible_row < data['listbox'].count():
                        item = data['listbox'].item(first_visible_row)
                        data['listbox'].scrollToItem(item, data['listbox'].PositionAtTop)
                        sb.setValue(sb.value() - first_visible_offset)
                    elif saved_scroll is not None:
                        target = min(saved_scroll, sb.maximum())
                        if sb.value() != target:
                            sb.setValue(target)
                    data['listbox'].setUpdatesEnabled(updates_were_enabled)
                    data['listbox'].blockSignals(signals_were_blocked)
                    data['listbox'].viewport().update()
                except Exception:
                    try:
                        data['listbox'].setUpdatesEnabled(updates_were_enabled)
                        data['listbox'].blockSignals(signals_were_blocked)
                    except Exception:
                        pass
            self._suspend_yield = False
            
            # Restore selections
            try:
                if selected_indices:
                    for idx in selected_indices:
                        if idx < data['listbox'].count():
                            data['listbox'].item(idx).setSelected(True)
                    # Update selection count
                    if 'selection_count_label' in data and data['selection_count_label']:
                        data['selection_count_label'].setText(f"Selected: {len(selected_indices)}")
                else:
                    # Clear selections if there were none
                    data['listbox'].clearSelection()
                    if 'selection_count_label' in data and data['selection_count_label']:
                        data['selection_count_label'].setText("Selected: 0")

                # Re-apply scroll AFTER selections (since selecting can auto-scroll)
                if saved_scroll is not None and 'listbox' in data and data['listbox']:
                    from PySide6.QtCore import QTimer
                    def _restore_scroll_again():
                        try:
                            sb = data['listbox'].verticalScrollBar()
                            target = min(saved_scroll, sb.maximum())
                            if sb.value() != target:
                                sb.setValue(target)
                        except Exception:
                            pass
                    QTimer.singleShot(0, _restore_scroll_again)
            except RuntimeError:
                print("⚠️ Could not restore selection state - widget was deleted during refresh")
            
            # print("✅ Retranslation data refreshed successfully")
            
        except RuntimeError as e:
            print(f"❌ Failed to refresh data - widget deleted: {e}")
        except FileNotFoundError as e:
            print(f"❌ Failed to refresh data - file not found: {e}")
            try:
                self._styled_msgbox(QMessageBox.Information, data.get('dialog', self), "Output Folder Not Found", 
                                      f"The output folder appears to have been deleted or moved.\n\n"
                                      f"File not found: {os.path.basename(str(e))}")
            except (RuntimeError, AttributeError):
                print(f"[WARN] Could not show error dialog - dialog was deleted")
        except PermissionError as e:
            # Refresh runs periodically. If the translator is writing/replacing
            # the progress JSON, skip this tick instead of interrupting the user.
            print(f"⚠️ Progress file locked during refresh; will retry on next refresh tick: {e}")
        except Exception as e:
            print(f"❌ Failed to refresh data: {e}")
            import traceback
            traceback.print_exc()
            try:
                # Show friendlier error message for common cases
                error_msg = str(e)
                if "No such file or directory" in error_msg or "cannot find the path" in error_msg:
                    self._styled_msgbox(QMessageBox.Information, data.get('dialog', self), "Output Folder Not Found", 
                                          f"The output folder appears to have been deleted or moved.\n\n"
                                          f"Error: {error_msg}")
                else:
                    self._styled_msgbox(QMessageBox.Warning, data.get('dialog', self), "Refresh Failed", 
                                      f"Failed to refresh data: {error_msg}")
            except (RuntimeError, AttributeError):
                # Dialog was also deleted, just print to console
                print(f"[WARN] Could not show error dialog - dialog was deleted")
        finally:
            self._suspend_yield = False
            try:
                listbox = data.get('listbox') if isinstance(data, dict) else None
                if listbox:
                    listbox.setUpdatesEnabled(updates_were_enabled)
                    listbox.blockSignals(signals_were_blocked)
                    listbox.viewport().update()
            except Exception:
                pass
    
    def _rematch_spine_chapters(self, data):
        """Re-run the full spine chapter matching logic against updated progress JSON"""
        prog = data['prog']
        output_dir = data['output_dir']
        spine_chapters = data['spine_chapters']

        def _normalize_opf_match_name(name: str) -> str:
            if not name:
                return ""
            base = os.path.basename(name)
            if base.startswith("response_"):
                base = base[len("response_"):]
            while True:
                new_base, ext = os.path.splitext(base)
                if not ext:
                    break
                base = new_base
            return base

        def _opf_names_equal(a: str, b: str) -> bool:
            return _normalize_opf_match_name(a) == _normalize_opf_match_name(b)

        # Build indexes once (O(n))
        basename_to_progress = {}
        response_to_progress = {}
        actualnum_to_progress = {}
        composite_to_progress = {}

        chapters_dict = prog.get("chapters", {})
        for ch in chapters_dict.values():
            orig = ch.get("original_basename", "")
            out = ch.get("output_file", "")
            actual_num = ch.get("actual_num")

            if orig:
                basename_to_progress.setdefault(_normalize_opf_match_name(orig), []).append(ch)
            if out:
                response_to_progress.setdefault(out, []).append(ch)
                norm_out = _normalize_opf_match_name(out)
                if norm_out != out:
                    response_to_progress.setdefault(norm_out, []).append(ch)
            if actual_num is not None:
                actualnum_to_progress.setdefault(actual_num, []).append(ch)

            fname_for_comp = orig or out
            if fname_for_comp and actual_num is not None:
                filename_noext = os.path.splitext(_normalize_opf_match_name(fname_for_comp))[0]
                composite_to_progress[f"{actual_num}_{filename_noext}"] = ch

        # Cache directory listing to avoid thousands of exists calls
        try:
            existing_files = {f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))}
        except Exception:
            existing_files = set()

        def file_exists_fast(fname: str) -> bool:
            return fname in existing_files

        for spine_ch in spine_chapters:
            filename = spine_ch['filename']
            chapter_num = spine_ch['file_chapter_num']
            is_special = spine_ch.get('is_special', False)

            base_name = os.path.splitext(filename)[0]
            retain = os.getenv('RETAIN_SOURCE_EXTENSION', '0') == '1' or self.config.get('retain_source_extension', False)

            if is_special:
                response_with_prefix = f"response_{base_name}.html"
                if retain:
                    expected_response = filename
                elif file_exists_fast(response_with_prefix):
                    expected_response = response_with_prefix
                else:
                    expected_response = filename
            else:
                expected_response = filename if retain else filename

            matched_info = None
            basename_key = _normalize_opf_match_name(filename)

            # 1) original basename map
            lst = basename_to_progress.get(basename_key)
            if lst:
                for ch in lst:
                    status = ch.get('status', '')
                    if status in ['in_progress', 'failed', 'qa_failed', 'pending']:
                        if ch.get('actual_num') == chapter_num:
                            matched_info = ch
                            break
                    else:
                        matched_info = ch
                        break

            # 2) response map (choose highest severity, prefer matching chapter_num)
            if not matched_info:
                lookup_keys = [
                    expected_response,
                    _normalize_opf_match_name(expected_response),
                    f"response_{expected_response}" if not expected_response.startswith("response_") else expected_response,
                    basename_key
                ]
                lst = None
                for k in lookup_keys:
                    if k in response_to_progress:
                        lst = response_to_progress[k]
                        break
                if lst:
                    has_qa = any(ch.get('status') == 'qa_failed' for ch in lst)
                    if has_qa:
                        lst = [ch for ch in lst if ch.get('status') != 'pending']
                    severity = {'qa_failed': 4, 'failed': 3, 'pending': 2, 'in_progress': 1, 'completed': 0}
                    best = None
                    best_score = -1
                    for ch in lst:
                        status = ch.get('status', '')
                        score = severity.get(status, -1)
                        matches_num = ch.get('actual_num') == chapter_num
                        if score > best_score or (score == best_score and matches_num):
                            best = ch
                            best_score = score
                            # If exact chapter match and highest severity, keep going in case of even higher severity
                    if best:
                        matched_info = best

            # 3) composite key
            if not matched_info:
                filename_noext = base_name
                if filename_noext.startswith("response_"):
                    filename_noext = filename_noext[len("response_"):]
                comp_key = f"{chapter_num}_{filename_noext}"
                matched_info = composite_to_progress.get(comp_key)

            # 4) actual_num map fallback (avoid mis-matching special files)
            if not matched_info and chapter_num in actualnum_to_progress:
                for ch in actualnum_to_progress[chapter_num]:
                    status = ch.get('status', '')
                    out_file = ch.get('output_file')
                    orig_base = os.path.basename(ch.get('original_basename', '') or '')

                    # If this spine entry is a special file (no digits), require filename match to avoid hijacking by other chapter 0 entries
                    if is_special:
                        fname_matches = (
                            (orig_base and _opf_names_equal(orig_base, filename)) or
                            (out_file and (_opf_names_equal(out_file, expected_response) or out_file == expected_response))
                        )
                        if not fname_matches:
                            continue

                    if status == 'merged':
                        if _opf_names_equal(orig_base, filename) or not orig_base:
                            matched_info = ch
                            break
                    elif status in ['in_progress', 'failed', 'pending', 'qa_failed']:
                        if out_file and (_opf_names_equal(out_file, expected_response) or out_file == expected_response):
                            matched_info = ch
                            break
                    else:
                        if (orig_base and _opf_names_equal(orig_base, filename)) or (out_file and (_opf_names_equal(out_file, expected_response) or out_file == expected_response)):
                            matched_info = ch
                            break

            file_exists = file_exists_fast(expected_response)

            if matched_info:
                status = matched_info.get('status', 'unknown')

                if status in ['failed', 'in_progress', 'qa_failed', 'pending']:
                    spine_ch['status'] = status
                    spine_ch['output_file'] = matched_info.get('output_file') or expected_response
                    spine_ch['progress_entry'] = matched_info
                    continue

                spine_ch['status'] = status
                spine_ch['output_file'] = expected_response if is_special else matched_info.get('output_file', expected_response)
                spine_ch['progress_entry'] = matched_info
                if not spine_ch['output_file']:
                    spine_ch['output_file'] = expected_response

                if status == 'completed':
                    output_file = spine_ch['output_file']
                    if not file_exists_fast(output_file):
                        if file_exists and expected_response:
                            spine_ch['output_file'] = expected_response
                            matched_info['output_file'] = expected_response
                        else:
                            spine_ch['status'] = 'not_translated'

            elif file_exists:
                spine_ch['status'] = 'completed'
                spine_ch['output_file'] = expected_response

            else:
                norm_target = _normalize_opf_match_name(filename)
                matched_file = None
                for f in existing_files:
                    if _normalize_opf_match_name(f) == norm_target:
                        matched_file = f
                        break
                if matched_file:
                    spine_ch['status'] = 'completed'
                    spine_ch['output_file'] = matched_file
                else:
                    spine_ch['status'] = 'not_translated'
                    spine_ch['output_file'] = expected_response
        
        # =====================================================
        # SAVE AUTO-DISCOVERED FILES TO PROGRESS (refresh path)
        # =====================================================
        
        progress_updated = False
        for spine_ch in spine_chapters:
            # Only add entries that were marked as completed but have no progress entry
            if spine_ch['status'] == 'completed' and 'progress_entry' not in spine_ch:
                chapter_num = spine_ch['file_chapter_num']
                output_file = spine_ch['output_file']
                filename = spine_ch['filename']

                # Require normalized filename match between spine file and output file, and the file must exist
                norm_spine = _normalize_opf_match_name(filename)
                norm_out = _normalize_opf_match_name(output_file)
                file_exists = os.path.exists(os.path.join(output_dir, output_file))
                if norm_spine != norm_out or not file_exists:
                    continue

                # Create a progress entry for this auto-discovered file
                chapter_key = str(chapter_num)
                
                # Check if key already exists (avoid duplicates)
                if chapter_key not in prog.get("chapters", {}):
                    prog.setdefault("chapters", {})[chapter_key] = {
                        "actual_num": chapter_num,
                        "content_hash": "",  # Unknown since we don't have the source
                        "output_file": output_file,
                        "status": "completed",
                        "last_updated": os.path.getmtime(os.path.join(output_dir, output_file)),
                        "auto_discovered": True,
                        "original_basename": filename
                    }
                    progress_updated = True
                    print(f"✅ Auto-discovered and tracked (refresh): {filename} -> {output_file}")
        
        # Save progress file if we added new entries
        if progress_updated:
            try:
                with open(data['progress_file'], 'w', encoding='utf-8') as f:
                    json.dump(prog, f, ensure_ascii=False, indent=2)
                print(f"💾 Saved {sum(1 for ch in spine_chapters if ch['status'] == 'completed' and 'progress_entry' not in ch)} auto-discovered files to progress file (refresh)")
            except Exception as e:
                print(f"⚠️ Warning: Failed to save progress file during refresh: {e}")
        
        # Rebuild chapter_display_info from updated spine_chapters
        chapter_display_info = []
        for spine_ch in spine_chapters:
            display_info = {
                'key': spine_ch.get('filename', ''),
                'num': spine_ch['file_chapter_num'],
                'info': spine_ch.get('progress_entry', {}),
                'output_file': spine_ch['output_file'],
                'status': spine_ch['status'],
                'duplicate_count': 1,
                'entries': [],
                'opf_position': spine_ch['position'],
                'original_filename': spine_ch['filename'],
                'is_special': spine_ch.get('is_special', False),
                'progress_key': spine_ch.get('progress_key')
            }
            chapter_display_info.append(display_info)
        
        self._append_pdf_ocr_display_info(data, chapter_display_info)
        self._append_image_gen_display_info(data, chapter_display_info)
        data['chapter_display_info'] = chapter_display_info
    
    def _rebuild_chapter_display_info(self, data):
        """Rebuild chapter_display_info from scratch (for fallback mode without OPF)"""
        # This is the same logic as the initial build in _force_retranslation_epub_or_text
        # but extracted here so refresh can use it
        
        prog = data['prog']
        output_dir = data['output_dir']
        file_path = data.get('file_path', '')
        show_special = data.get('show_special_files_state', False)
        
        # Known non-chapter files that should never appear in the progress list
        _non_chapter_files = {"glossary.csv", "metadata.json", "styles.css", "rolling_summary.txt"}
        _source_has_translated = "_translated" in os.path.basename(file_path).lower()
        files_to_entries = {}
        for chapter_key, chapter_info in prog.get("chapters", {}).items():
            output_file = chapter_info.get("output_file", "")
            status = chapter_info.get("status", "")
            
            # Skip known non-chapter files
            if output_file and output_file.lower() in _non_chapter_files:
                continue
            # Skip combined _translated output files (unless source itself has _translated)
            if output_file and not _source_has_translated and any(
                output_file.lower().endswith(s) for s in ("_translated.txt", "_translated.pdf", "_translated.html")
            ):
                continue
            
            # Include chapters with output files OR in_progress/failed/qa_failed with null output file (legacy)
            if output_file or status in ["in_progress", "failed", "qa_failed"]:
                # For merged chapters, use a unique key (chapter_key) instead of output_file
                # This ensures merged chapters appear as separate entries in the list
                if status == "merged":
                    file_key = f"_merged_{chapter_key}"
                elif output_file:
                    file_key = output_file
                elif status == "in_progress":
                    file_key = f"_in_progress_{chapter_key}"
                elif status == "qa_failed":
                    file_key = f"_qa_failed_{chapter_key}"
                else:  # failed
                    file_key = f"_failed_{chapter_key}"
                
                if file_key not in files_to_entries:
                    files_to_entries[file_key] = []
                files_to_entries[file_key].append((chapter_key, chapter_info))
        
        chapter_display_info = []
        
        for output_file, entries in files_to_entries.items():
            chapter_key, chapter_info = entries[0]
            
            # Get the actual output file (strip placeholder prefix if present)
            actual_output_file = output_file
            if output_file.startswith("_merged_") or output_file.startswith("_in_progress_") or output_file.startswith("_failed_") or output_file.startswith("_qa_failed_"):
                # For merged/in_progress/failed/qa_failed, get the actual output_file from chapter_info
                actual_output_file = chapter_info.get("output_file", "")
                if not actual_output_file:
                    # Generate expected filename based on actual_num
                    actual_num = chapter_info.get("actual_num")
                    if actual_num is not None:
                        # Use .txt extension for text files, .html for EPUB
                        ext = ".txt" if file_path.endswith(".txt") else ".html"
                        actual_output_file = f"response_section_{actual_num}{ext}"
            
            # Check if this is a special file using configured keyword lists
            original_basename = chapter_info.get("original_basename", "")
            filename_to_check = original_basename if original_basename else actual_output_file
            
            is_special = self._is_special_file(filename_to_check) if hasattr(self, '_is_special_file') else (not bool(re.search(r'\d', filename_to_check)))
            
            # Don't skip special files here - let the display logic handle hiding them
            # This ensures chapter_display_info contains all items, and the listbox
            # will properly hide/show items based on the toggle state
            
            # Extract chapter number - prioritize stored values
            chapter_num = None
            if 'actual_num' in chapter_info and chapter_info['actual_num'] is not None:
                chapter_num = chapter_info['actual_num']
            elif 'chapter_num' in chapter_info and chapter_info['chapter_num'] is not None:
                chapter_num = chapter_info['chapter_num']
            
            # Fallback: extract from filename
            if chapter_num is None:
                import re
                matches = re.findall(r'(\d+)', actual_output_file)
                if matches:
                    chapter_num = int(matches[-1])
                else:
                    chapter_num = 999999
            
            status = chapter_info.get("status", "unknown")
            if status in ("completed_empty", "completed_image_only"):
                status = "completed"
            
            # Check file existence
            if status == "completed":
                output_path = os.path.join(output_dir, actual_output_file)
                if not os.path.exists(output_path):
                    status = "not_translated"
            
            chapter_display_info.append({
                'key': chapter_key,
                'num': chapter_num,
                'info': chapter_info,
                'output_file': actual_output_file,  # Use actual output file, not placeholder
                'status': status,
                'duplicate_count': len(entries),
                'entries': entries,
                'is_special': is_special,
                'progress_key': chapter_key
            })
        
        # Sort by chapter number
        chapter_display_info.sort(key=lambda x: x['num'] if x['num'] is not None else 999999)
        
        self._append_pdf_ocr_display_info(data, chapter_display_info)
        self._append_image_gen_display_info(data, chapter_display_info)

        # Update data with rebuilt list
        data['chapter_display_info'] = chapter_display_info

    def _append_pdf_ocr_display_info(self, data, chapter_display_info):
        """Add a lightweight summary row for PDF Vision OCR progress."""
        try:
            prog = data.get('prog') or {}
            pdf_ocr = prog.get('pdf_ocr')
            progress_output_mode = str(prog.get('output_mode') or data.get('output_mode') or '').lower().strip()
            ui_output_mode = ""
            try:
                if hasattr(self, '_get_output_mode'):
                    ui_output_mode = str(self._get_output_mode() or '').lower().strip()
            except Exception:
                ui_output_mode = ""
            if ui_output_mode and ui_output_mode != 'vision':
                return
            if progress_output_mode and progress_output_mode != 'vision':
                return
            current_file = str(data.get('file_path') or '')
            if current_file and not current_file.lower().endswith('.pdf'):
                return
            if not current_file:
                pdf_source = ""
                if isinstance(pdf_ocr, dict):
                    pdf_source = str(pdf_ocr.get('source_file') or '')
                if not pdf_source or not pdf_source.lower().endswith('.pdf'):
                    return
            if not isinstance(pdf_ocr, dict):
                output_dir = data.get('output_dir') or ''
                image_exts = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif', '.tif', '.tiff')
                image_dir = os.path.join(output_dir, 'images')
                single_dir = os.path.join(output_dir, 'OCR', 'single')
                image_count = 0
                cached_count = 0
                try:
                    if os.path.isdir(image_dir):
                        image_count = sum(
                            1 for name in os.listdir(image_dir)
                            if os.path.isfile(os.path.join(image_dir, name)) and name.lower().endswith(image_exts)
                        )
                except Exception:
                    image_count = 0
                try:
                    if os.path.isdir(single_dir):
                        cached_count = sum(
                            1 for name in os.listdir(single_dir)
                            if os.path.isfile(os.path.join(single_dir, name)) and name.lower().endswith('.txt')
                        )
                except Exception:
                    cached_count = 0
                total_guess = max(image_count, cached_count)
                if total_guess <= 0:
                    return
                pdf_ocr = {
                    'source_file': data.get('file_path'),
                    'ocr_source_file': '',
                    'status': 'completed' if cached_count >= total_guess else 'in_progress',
                    'total': total_guess,
                    'done': cached_count,
                    'cached': cached_count,
                    'no_text': 0,
                    'failed': 0,
                    'cache_inferred': True,
                }
            total = int(pdf_ocr.get('total') or 0)
            pages = pdf_ocr.get('pages') if isinstance(pdf_ocr.get('pages'), dict) else {}
            if total <= 0 and pages:
                total = len(pages)
            if total <= 0:
                return
            done = int(pdf_ocr.get('done') or 0)
            cached = int(pdf_ocr.get('cached') or 0)
            failed = int(pdf_ocr.get('failed') or 0)
            no_text = int(pdf_ocr.get('no_text') or 0)
            status = str(pdf_ocr.get('status') or 'in_progress').lower().strip()
            if status not in ('completed', 'failed', 'cancelled'):
                status = 'in_progress'
            elif status == 'cancelled':
                status = 'failed'
            source_file = os.path.basename(str(pdf_ocr.get('source_file') or data.get('file_path') or 'PDF'))
            ocr_source_file = os.path.basename(str(pdf_ocr.get('ocr_source_file') or ''))
            label_bits = [f"{min(done, total)}/{total} pages"]
            if cached:
                label_bits.append(f"{cached} cached")
            if no_text:
                label_bits.append(f"{no_text} no-text")
            if failed:
                label_bits.append(f"{failed} failed")
            output_label = f"{source_file} -> {ocr_source_file or '_OCR.pdf'} ({', '.join(label_bits)})"
            info = dict(pdf_ocr)
            info['status'] = status
            info['ocr_progress'] = {
                'done': min(done, total),
                'total': total,
                'label': f"{min(done, total)}/{total}",
            }
            chapter_display_info.insert(0, {
                'key': '__pdf_ocr__',
                'num': 0,
                'info': info,
                'output_file': output_label,
                'status': status,
                'duplicate_count': 1,
                'entries': [],
                'is_special': False,
                'progress_key': '__pdf_ocr__',
                'pdf_ocr': True,
            })
        except Exception as e:
            print(f"Warning: could not read PDF OCR progress: {e}")

    def _append_image_gen_display_info(self, data, chapter_display_info):
        """Add a lightweight summary row for image generation progress (image output mode)."""
        try:
            prog = data.get('prog') or {}
            image_gen = prog.get('image_gen')
            if not isinstance(image_gen, dict):
                return
            # Hide the row when the user switches output mode away from 'image'.
            # Check the live GUI combo directly — it's the most reliable source.
            combo = getattr(self, '_output_mode_combo', None)
            if combo is not None:
                try:
                    live_mode = {0: 'text', 1: 'vision', 2: 'image', 3: 'video', 4: 'audio', 5: 'refinement'}.get(combo.currentIndex(), 'text')
                    if live_mode != 'image':
                        return
                except Exception:
                    pass
            # Verify the source is epub/pdf using the stored source_file or the current file.
            source_file = str(image_gen.get('source_file') or data.get('file_path') or '')
            if source_file and not source_file.lower().endswith(('.epub', '.pdf')):
                return

            total = int(image_gen.get('total') or 0)
            if total <= 0:
                return
            done = int(image_gen.get('done') or 0)
            success = int(image_gen.get('success') or 0)
            skipped = int(image_gen.get('skipped') or 0)
            failed = int(image_gen.get('failed') or 0)
            status = str(image_gen.get('status') or 'in_progress').lower().strip()
            if status not in ('completed', 'failed', 'cancelled'):
                status = 'in_progress'
            elif status == 'cancelled':
                status = 'failed'

            source_file = os.path.basename(str(image_gen.get('source_file') or data.get('file_path') or 'EPUB'))
            label_bits = [f"{min(done, total)}/{total} images"]
            if success:
                label_bits.append(f"{success} generated")
            if skipped:
                label_bits.append(f"{skipped} skipped")
            if failed:
                label_bits.append(f"{failed} failed")
            output_label = f"🎨 Image Generation: {source_file} ({', '.join(label_bits)})"

            info = dict(image_gen)
            info['status'] = status
            info['image_gen_progress'] = {
                'done': min(done, total),
                'total': total,
                'label': f"{min(done, total)}/{total}",
            }
            chapter_display_info.insert(0, {
                'key': '__image_gen__',
                'num': 0,
                'info': info,
                'output_file': output_label,
                'status': status,
                'duplicate_count': 1,
                'entries': [],
                'is_special': False,
                'progress_key': '__image_gen__',
                'image_gen': True,
            })
        except Exception as e:
            print(f"Warning: could not read image gen progress: {e}")
    
    def _current_progress_output_mode(self, data=None, entry=None):
        """Prefer the live GUI output mode over stale mode values saved in progress JSON."""
        candidates = []

        combo = getattr(self, '_output_mode_combo', None)
        if combo is not None:
            try:
                idx_mode = {0: 'text', 1: 'vision', 2: 'image', 3: 'video', 4: 'audio', 5: 'refinement'}.get(combo.currentIndex())
                if idx_mode:
                    candidates.append(idx_mode)
            except RuntimeError:
                pass
            except Exception:
                pass
            try:
                candidates.append(combo.currentText())
            except RuntimeError:
                pass
            except Exception:
                pass

        try:
            if hasattr(self, '_get_output_mode'):
                candidates.append(self._get_output_mode())
        except Exception:
            pass

        candidates.append(getattr(self, 'output_mode_var', None))

        config = getattr(self, 'config', None)
        if isinstance(config, dict):
            candidates.append(config.get('output_mode'))
            if config.get('enable_audio_output_mode'):
                candidates.append('audio')
            if config.get('enable_refinement_output_mode'):
                candidates.append('refinement')

        prog = (data or {}).get('prog') or {}
        candidates.append(prog.get('output_mode'))
        if isinstance(entry, dict):
            candidates.append(entry.get('output_mode'))

        for candidate in candidates:
            mode = str(candidate or '').lower().strip()
            if 'audio' in mode:
                return 'audio'
            if 'refine' in mode or 'refinement' in mode:
                return 'refinement'
            if mode in ('text', 'vision', 'image', 'video'):
                return mode
        return 'text'

    def _audio_stem_variants(self, output_file):
        stem = os.path.splitext(os.path.basename(output_file or ""))[0]
        if not stem:
            return []
        variants = [stem]
        if stem.startswith("response_"):
            variants.append(stem[len("response_"):])
        else:
            variants.append(f"response_{stem}")
        return list(dict.fromkeys(variants))

    def _normalize_progress_output_name(self, name: str) -> str:
        """Normalize translated/source output names for response_/extension-tolerant matching."""
        if not name:
            return ""
        base = os.path.basename(str(name).replace("\\", "/"))
        if base.lower().startswith("response_"):
            base = base[len("response_"):]
        while True:
            stem, ext = os.path.splitext(base)
            if not ext:
                break
            base = stem
        return base.lower()

    def _resolve_existing_output_path(self, output_dir, output_file=None, display_info=None, prog=None):
        """Resolve an output file while tolerating stale OCR rows and filename mode changes."""
        display_info = display_info or {}
        progress_entry = display_info.get("info") or display_info.get("progress_entry") or {}
        prog = prog or {}
        chapters = prog.get("chapters", {}) if isinstance(prog, dict) else {}
        candidates = []

        def add_candidate(value):
            if value:
                text = str(value).replace("\\", "/")
                if text not in candidates:
                    candidates.append(text)

        add_candidate(output_file)
        add_candidate(display_info.get("output_file"))
        add_candidate(progress_entry.get("output_file") if isinstance(progress_entry, dict) else None)
        if isinstance(progress_entry, dict):
            previous = progress_entry.get("previous_progress_entry")
            if isinstance(previous, dict):
                add_candidate(previous.get("output_file"))

        progress_key = display_info.get("progress_key")
        if progress_key and isinstance(chapters.get(progress_key), dict):
            tracked = chapters[progress_key]
            add_candidate(tracked.get("output_file"))
            previous = tracked.get("previous_progress_entry")
            if isinstance(previous, dict):
                add_candidate(previous.get("output_file"))

        target_num = display_info.get("num")
        original_names = {
            self._normalize_progress_output_name(display_info.get("original_filename")),
            self._normalize_progress_output_name(display_info.get("original_basename")),
        }
        original_names.discard("")

        for tracked in chapters.values():
            if not isinstance(tracked, dict):
                continue
            tracked_num = tracked.get("actual_num", tracked.get("chapter_num"))
            tracked_names = {
                self._normalize_progress_output_name(tracked.get("output_file")),
                self._normalize_progress_output_name(tracked.get("original_basename")),
                self._normalize_progress_output_name(tracked.get("original_filename")),
            }
            if str(tracked_num) == str(target_num) or (original_names and tracked_names & original_names):
                add_candidate(tracked.get("output_file"))
                previous = tracked.get("previous_progress_entry")
                if isinstance(previous, dict):
                    add_candidate(previous.get("output_file"))

        for candidate in candidates:
            path = candidate if os.path.isabs(candidate) else os.path.join(output_dir, candidate)
            if os.path.isfile(path):
                rel = os.path.relpath(path, output_dir).replace("\\", "/") if os.path.isabs(candidate) else candidate
                return rel, path

        target_norms = {self._normalize_progress_output_name(value) for value in candidates if value}
        target_norms |= original_names
        target_norms.discard("")
        if not target_norms:
            return None, None
        try:
            for fname in os.listdir(output_dir):
                path = os.path.join(output_dir, fname)
                if os.path.isfile(path) and self._normalize_progress_output_name(fname) in target_norms:
                    return fname, path
        except Exception:
            pass
        return None, None

    def _audio_candidates_for_entry(self, output_dir, entry):
        """Return possible audio files for a progress entry as (relative, absolute) pairs."""
        candidates = []
        if not isinstance(entry, dict):
            return candidates

        stored = entry.get('tts_file')
        if stored:
            candidates.append(stored)

        output_file = entry.get('output_file')
        if output_file:
            for stem in self._audio_stem_variants(output_file):
                for ext in ("wav", "mp3", "pcm", "m4a", "ogg", "flac"):
                    candidates.append(os.path.join("text_to_speech", f"{stem}.{ext}"))

        seen = set()
        resolved = []
        for candidate in candidates:
            if not candidate:
                continue
            normalized = str(candidate).replace("\\", "/")
            if normalized in seen:
                continue
            seen.add(normalized)
            abs_path = normalized if os.path.isabs(normalized) else os.path.join(output_dir, normalized)
            rel_path = os.path.relpath(abs_path, output_dir).replace("\\", "/") if os.path.isabs(normalized) else normalized
            resolved.append((rel_path, abs_path))
        return resolved

    def _existing_audio_for_entry(self, output_dir, entry):
        for rel_path, abs_path in self._audio_candidates_for_entry(output_dir, entry):
            if os.path.exists(abs_path):
                return rel_path, abs_path
        return None, None

    def _reconcile_tts_audio_files(self, data):
        """Keep progress TTS status aligned with generated audio files on disk."""
        prog = data.get('prog') or {}
        output_dir = data.get('output_dir')
        if not output_dir:
            return False

        changed = False
        now = time.time()
        for _key, entry in prog.get('chapters', {}).items():
            if not isinstance(entry, dict):
                continue
            output_file = entry.get('output_file')
            if not output_file:
                continue
            rel_audio, _abs_audio = self._existing_audio_for_entry(output_dir, entry)
            tts_status = str(entry.get('tts_status') or 'no_tts').lower().strip()

            if rel_audio:
                if tts_status not in ('tts_completed', 'completed') or entry.get('tts_file') != rel_audio:
                    entry['tts_status'] = 'tts_completed'
                    entry['tts_file'] = rel_audio
                    entry.pop('tts_error', None)
                    entry.setdefault('tts_at', now)
                    entry['last_updated'] = now
                    changed = True
                continue

            had_audio_state = (
                entry.get('tts_file')
                or tts_status in ('tts_completed', 'completed', 'in_progress')
            )
            if had_audio_state:
                entry['tts_status'] = 'no_tts'
                entry.pop('tts_file', None)
                entry.pop('tts_at', None)
                entry.pop('tts_error', None)
                entry['last_updated'] = now
                changed = True
        return changed

    def _progress_display_status(self, info, data=None):
        """Derive the status shown in Progress Manager for post-processing modes."""
        status = info.get('status', 'unknown')
        entry = info.get('progress_entry') or info.get('info') or {}
        mode = self._current_progress_output_mode(data, entry)

        if status in ('completed_empty', 'completed_image_only'):
            status = 'completed'

        ref_status = str(entry.get('refinement_status') or '').lower().strip()
        tts_status = str(entry.get('tts_status') or '').lower().strip()
        if status == 'in_progress' and (ref_status == 'in_progress' or tts_status == 'in_progress'):
            return 'in_progress'

        if status == 'in_progress' and data and data.get('output_dir'):
            previous_status = str(entry.get('previous_status') or '').lower().strip()
            previous_entry = entry.get('previous_progress_entry')
            if previous_status in ('completed', 'completed_empty', 'completed_image_only') or (
                isinstance(previous_entry, dict)
                and str(previous_entry.get('status') or '').lower().strip() in ('completed', 'completed_empty', 'completed_image_only')
            ):
                _resolved_file, resolved_path = self._resolve_existing_output_path(
                    data.get('output_dir'),
                    info.get('output_file') or entry.get('output_file'),
                    info,
                    data.get('prog'),
                )
                if resolved_path and os.path.exists(resolved_path):
                    return 'completed'

        if status in ('failed', 'qa_failed', 'in_progress', 'pending', 'merged', 'not_translated'):
            return status
        if mode == 'refinement':
            ref_status = ref_status or 'not_refined'
            if ref_status in ('failed', 'error'):
                return 'failed'
            if ref_status == 'in_progress':
                return 'in_progress'
            if ref_status not in ('refined', 'completed'):
                return 'not_refined'
        if mode == 'audio':
            tts_status = tts_status or 'no_tts'
            if tts_status in ('failed', 'error'):
                return 'failed'
            if tts_status == 'in_progress':
                return 'in_progress'
            if tts_status not in ('tts_completed', 'completed'):
                return 'no_tts'
        return status

    def _progress_entry_is_refined(self, info):
        """Return True when a completed progress entry also has refined output."""
        try:
            entry = info.get('progress_entry') or info.get('info') or info
            if not isinstance(entry, dict):
                return False
            return str(entry.get('refinement_status') or '').lower().strip() in ('refined', 'completed')
        except Exception:
            return False

    def _update_chapter_status_info(self, data):
        """Update chapter status information after refresh"""
        # Re-check file existence and update status for each chapter
        for info in data['chapter_display_info']:
            output_file = info['output_file']
            resolved_output_file, resolved_output_path = self._resolve_existing_output_path(
                data['output_dir'],
                output_file,
                info,
                data.get('prog'),
            )
            output_path = resolved_output_path or os.path.join(data['output_dir'], output_file)
            
            # Find matching progress entry
            matched_info = None
            
            # PRIORITY 1: Match by BOTH actual_num AND output_file
            # This prevents cross-matching between files with same chapter number but different filenames
            for chapter_key, chapter_info in data['prog'].get("chapters", {}).items():
                actual_num = chapter_info.get('actual_num') or chapter_info.get('chapter_num')
                ch_output = chapter_info.get('output_file')
                
                # BOTH must match - no fallback
                if actual_num is not None and actual_num == info['num'] and ch_output == output_file:
                    matched_info = chapter_info
                    break
            
            # PRIORITY 2: Fall back to output_file matching if no actual_num match
            if not matched_info:
                # Prefer completed over failed/pending/in_progress; keep qa_failed highest
                severity = {'qa_failed': 5, 'completed': 4, 'failed': 3, 'pending': 2, 'in_progress': 1}
                best = None
                best_score = -1
                for chapter_key, chapter_info in data['prog'].get("chapters", {}).items():
                    if chapter_info.get('output_file') == output_file:
                        status = chapter_info.get('status', 'unknown')
                        score = severity.get(status, -1)
                        # Prefer higher severity; tie-breaker: matching actual_num if present
                        matches_num = (chapter_info.get('actual_num') or chapter_info.get('chapter_num')) == info['num']
                        if score > best_score or (score == best_score and matches_num):
                            best_score = score
                            best = chapter_info
                if best:
                    matched_info = best
            
            # Update status based on current state from progress file
            if matched_info:
                new_status = matched_info.get('status', 'unknown')
                # Handle legacy completed variants as completed for display
                if new_status in ('completed_empty', 'completed_image_only'):
                    new_status = 'completed'
                # Verify file actually exists for completed status (but NOT for merged - merged chapters
                # don't have their own output files, they point to parent's file)
                if new_status == 'completed' and not os.path.exists(output_path):
                    new_status = 'not_translated'
                elif new_status == 'completed' and resolved_output_file:
                    info['output_file'] = resolved_output_file
                info['status'] = new_status
                info['info'] = matched_info
            elif os.path.exists(output_path):
                # Before marking as completed based on file existence, check if this chapter
                # is actually marked as merged in the progress file (by actual_num lookup)
                # This handles the case where old output files exist from before merging was enabled
                is_merged_chapter = False
                for chapter_key, chapter_info in data['prog'].get("chapters", {}).items():
                    actual_num = chapter_info.get('actual_num') or chapter_info.get('chapter_num')
                    if actual_num is not None and actual_num == info['num']:
                        if chapter_info.get('status') == 'merged':
                            is_merged_chapter = True
                            info['status'] = 'merged'
                            info['info'] = chapter_info
                            break
                
                if not is_merged_chapter:
                    info['status'] = 'completed'
                    info.pop('info', None)
                    info.pop('progress_entry', None)
            else:
                info['status'] = 'not_translated'
                info.pop('info', None)
                info.pop('progress_entry', None)

    def _progress_entry_model_name(self, info, data=None):
        """Return the model name attached to a progress row, with old-file fallbacks."""
        candidates = []
        if isinstance(info, dict):
            candidates.append(info)
            for key in ('info', 'progress_entry'):
                value = info.get(key)
                if isinstance(value, dict):
                    candidates.append(value)
                    previous = value.get('previous_progress_entry')
                    if isinstance(previous, dict):
                        candidates.append(previous)
        if isinstance(data, dict):
            prog = data.get('prog')
            if isinstance(prog, dict):
                progress_key = info.get('progress_key') if isinstance(info, dict) else None
                chapters = prog.get('chapters', {})
                if progress_key and isinstance(chapters, dict) and isinstance(chapters.get(progress_key), dict):
                    candidates.append(chapters[progress_key])

        for candidate in candidates:
            model_name = str(candidate.get('model_name') or candidate.get('model') or '').strip()
            if model_name:
                return model_name
        return "(model unknown)"

    def _progress_model_column_text(self, info, data, fallback_output):
        if isinstance(data, dict) and data.get('show_model_info_state'):
            return self._progress_entry_model_name(info, data)
        return fallback_output

    def _progress_list_column_widths(self, chapter_display_info, data):
        max_original_len = 0
        max_output_len = 0
        for info in chapter_display_info or []:
            if 'opf_position' not in info:
                continue
            original_file = info.get('original_filename', '')
            output_file = self._progress_model_column_text(info, data, info.get('output_file', ''))
            max_original_len = max(max_original_len, len(original_file))
            max_output_len = max(max_output_len, len(output_file))
        return max(max_original_len, 20), max(max_output_len, 25)

    def _progress_list_show_special(self, data):
        show_special_files = data.get('show_special_files_state', False) if isinstance(data, dict) else False
        cb = data.get('show_special_files_cb') if isinstance(data, dict) else None
        if cb:
            try:
                show_special_files = cb.isChecked()
            except RuntimeError:
                pass
        if isinstance(data, dict):
            data['show_special_files_state'] = show_special_files
        return show_special_files

    def _progress_list_sync_model_toggle(self, data):
        cb = data.get('show_model_info_cb') if isinstance(data, dict) else None
        if cb:
            try:
                data['show_model_info_state'] = cb.isChecked()
            except RuntimeError:
                pass

    def _progress_list_item_key(self, info):
        if not isinstance(info, dict):
            return None
        progress_key = info.get('progress_key')
        if progress_key:
            return f"progress:{progress_key}"
        output_file = info.get('output_file')
        if output_file:
            return f"output:{output_file}"
        return f"row:{info.get('num')}:{info.get('original_filename', '')}:{info.get('key', '')}"

    def _progress_list_display_text(self, info, data, max_original_len, max_output_len):
        status_icons = {
            'completed': '✅',
            'merged': '🔗',
            'failed': '❌',
            'qa_failed': '❌',
            'in_progress': '🔄',
            'pending': '❓',
            'not_translated': '⬜',
            'not_refined': '✨',
            'no_tts': '🔊',
            'unknown': '❓'
        }
        status_labels = {
            'completed': 'Completed',
            'merged': 'Merged',
            'failed': 'Failed',
            'qa_failed': 'QA Failed',
            'in_progress': 'In Progress',
            'pending': 'Pending',
            'not_translated': 'Not Translated',
            'not_refined': 'Not Refined',
            'no_tts': 'No TTS',
            'unknown': 'Unknown'
        }

        chapter_num = info['num']
        status = self._progress_display_status(info, data)
        output_file = info['output_file']
        output_display = self._progress_model_column_text(info, data, output_file)
        icon = status_icons.get(status, '❓')
        status_label = status_labels.get(status, status)
        if status == 'completed' and self._progress_entry_is_refined(info):
            status_label = f"{status_label} ⭐"
        chapter_info = info.get('info') or info.get('progress_entry') or {}
        ocr_progress = chapter_info.get('ocr_progress') if isinstance(chapter_info, dict) else None
        if status == 'in_progress' and isinstance(ocr_progress, dict):
            try:
                ocr_done = int(ocr_progress.get('done', 0))
                ocr_total = int(ocr_progress.get('total', 0))
            except (TypeError, ValueError):
                ocr_done = 0
                ocr_total = 0
            if ocr_total > 0:
                status_label = f"{status_label} ({min(ocr_done, ocr_total)}/{ocr_total})"

        if info.get('pdf_ocr'):
            display = f"PDF OCR | {icon} {status_label:18s} | {output_display}"
        elif 'opf_position' in info:
            original_file = info.get('original_filename', '')
            opf_pos = info['opf_position'] + 1
            if isinstance(chapter_num, float):
                if chapter_num.is_integer():
                    display = f"[{opf_pos:03d}] Ch.{int(chapter_num):03d} | {icon} {status_label:11s} | {original_file:<{max_original_len}} -> {output_display}"
                else:
                    display = f"[{opf_pos:03d}] Ch.{chapter_num:06.1f} | {icon} {status_label:11s} | {original_file:<{max_original_len}} -> {output_display}"
            else:
                display = f"[{opf_pos:03d}] Ch.{chapter_num:03d} | {icon} {status_label:11s} | {original_file:<{max_original_len}} -> {output_display}"
        else:
            if isinstance(chapter_num, float) and chapter_num.is_integer():
                display = f"Chapter {int(chapter_num):03d} | {icon} {status_label:11s} | {output_display}"
            elif isinstance(chapter_num, float):
                display = f"Chapter {chapter_num:06.1f} | {icon} {status_label:11s} | {output_display}"
            else:
                display = f"Chapter {chapter_num:03d} | {icon} {status_label:11s} | {output_display}"

        if status == 'qa_failed':
            qa_issues = chapter_info.get('qa_issues_found', []) if isinstance(chapter_info, dict) else []
            if qa_issues:
                issues_display = ', '.join(qa_issues[:2])
                if len(qa_issues) > 2:
                    issues_display += f' (+{len(qa_issues)-2} more)'
                display += f" | {issues_display}"

        if status == 'merged':
            parent_chapter = chapter_info.get('merged_parent_chapter') if isinstance(chapter_info, dict) else None
            if parent_chapter:
                display += f" | → Ch.{parent_chapter}"

        if info.get('duplicate_count', 1) > 1:
            display += f" | ({info['duplicate_count']} entries)"

        return display, status

    def _apply_progress_list_item_visuals(self, item, status):
        if status == 'completed':
            item.setForeground(QColor('green'))
        elif status == 'merged':
            item.setForeground(QColor('#17a2b8'))
        elif status in ['failed', 'qa_failed']:
            item.setForeground(QColor('red'))
        elif status == 'not_translated':
            item.setForeground(QColor('#2b6cb0'))
        elif status in ['not_refined', 'no_tts']:
            item.setForeground(QColor('#8a63d2'))
        elif status == 'in_progress':
            item.setForeground(QColor('orange'))
        else:
            item.setForeground(QColor('white'))

    def _set_progress_list_item_metadata(self, item, info, status, show_special_files):
        is_special = info.get('is_special', False)
        _fname = info.get('original_filename', '') or info.get('output_file', '') or info.get('key', '')
        is_skipped_special = self._progress_file_is_skipped_special(_fname, is_special)
        item.setData(Qt.UserRole, {
            'is_special': is_special,
            'info': info,
            'progress_key': info.get('progress_key'),
            'item_key': self._progress_list_item_key(info),
        })
        item.setData(Qt.UserRole + 2, status)
        item.setHidden(is_skipped_special and not show_special_files)

    def _populate_progress_listbox_streamed(self, data, chunk_size=150, preserve_selection=False, preserve_scroll=False):
        """Populate large progress lists over multiple event-loop turns."""
        if not self._is_data_valid(data):
            return

        listbox = data.get('listbox')
        if not listbox:
            return

        self._progress_list_sync_model_toggle(data)
        infos = list(data.get('chapter_display_info') or [])
        max_original_len, max_output_len = self._progress_list_column_widths(infos, data)

        selected_keys = set()
        if preserve_selection:
            try:
                for item in listbox.selectedItems():
                    payload = item.data(Qt.UserRole) or {}
                    key = payload.get('item_key') or self._progress_list_item_key(payload.get('info') or {})
                    if key:
                        selected_keys.add(key)
            except RuntimeError:
                selected_keys = set()

        saved_scroll = None
        if preserve_scroll:
            try:
                saved_scroll = listbox.verticalScrollBar().value()
            except RuntimeError:
                saved_scroll = None

        generation = int(data.get('_listbox_populate_generation', 0)) + 1
        data['_listbox_populate_generation'] = generation
        data['_listbox_populate_active'] = True

        try:
            listbox.blockSignals(True)
            listbox.setUpdatesEnabled(False)
            listbox.clear()
        except RuntimeError:
            data['_listbox_populate_active'] = False
            return

        state = {'idx': 0}

        def _finish():
            if generation != data.get('_listbox_populate_generation'):
                return
            data['_listbox_populate_active'] = False
            try:
                if saved_scroll is not None:
                    sb = listbox.verticalScrollBar()
                    sb.setValue(min(saved_scroll, sb.maximum()))
                listbox.blockSignals(False)
                listbox.setUpdatesEnabled(True)
                label = data.get('selection_count_label')
                if label:
                    label.setText(f"Selected: {len(listbox.selectedItems())}")
                listbox.viewport().update()
            except RuntimeError:
                pass

        def _add_chunk():
            if generation != data.get('_listbox_populate_generation'):
                return
            if not self._is_data_valid(data):
                return
            try:
                listbox.setUpdatesEnabled(False)
                show_special_files = self._progress_list_show_special(data)
                end_idx = min(state['idx'] + chunk_size, len(infos))
                for idx in range(state['idx'], end_idx):
                    info = infos[idx]
                    display, status = self._progress_list_display_text(
                        info,
                        data,
                        max_original_len,
                        max_output_len,
                    )
                    item = QListWidgetItem(display)
                    self._apply_progress_list_item_visuals(item, status)
                    self._set_progress_list_item_metadata(item, info, status, show_special_files)
                    self._add_compact_inline_list_item(listbox, item)
                    self._set_progress_list_item_metadata(item, info, status, show_special_files)
                    if selected_keys and self._progress_list_item_key(info) in selected_keys:
                        item.setSelected(True)
                state['idx'] = end_idx
                listbox.setUpdatesEnabled(True)
                listbox.viewport().update()
            except RuntimeError:
                return

            if state['idx'] < len(infos):
                QTimer.singleShot(0, _add_chunk)
            else:
                _finish()

        QTimer.singleShot(0, _add_chunk)

    def _update_listbox_display(self, data):
        """Update the listbox display with current chapter information"""
        if not self._is_data_valid(data):
            print("⚠️ Cannot update listbox display - widgets have been deleted")
            return

        listbox = data['listbox']
        self._progress_list_sync_model_toggle(data)
        count_existing = listbox.count()
        count_new = len(data.get('chapter_display_info') or [])
        if data.get('_listbox_populate_active') or count_existing != count_new:
            self._populate_progress_listbox_streamed(
                data,
                preserve_selection=True,
                preserve_scroll=True,
            )
            return

        show_special_files = self._progress_list_show_special(data)
        max_original_len, max_output_len = self._progress_list_column_widths(
            data.get('chapter_display_info') or [],
            data,
        )

        listbox.setUpdatesEnabled(False)
        listbox.blockSignals(True)
        try:
            for idx, info in enumerate(data.get('chapter_display_info') or []):
                if idx % 120 == 0:
                    self._ui_yield()
                item = listbox.item(idx)
                if not item:
                    continue
                display, display_status = self._progress_list_display_text(
                    info,
                    data,
                    max_original_len,
                    max_output_len,
                )
                item.setText(display)
                self._apply_progress_list_item_visuals(item, display_status)
                self._set_compact_inline_item_size(listbox, item)
                self._set_progress_list_item_metadata(item, info, display_status, show_special_files)
        finally:
            listbox.blockSignals(False)
            listbox.setUpdatesEnabled(True)

    def _update_statistics_display(self, data):
        """Update statistics display for both OPF and non-OPF files"""
        # Find statistics labels in the container
        container = data['container']
        
        # Search for statistics labels by traversing the widget hierarchy
        def find_stats_labels(widget):
            labels = {}
            if hasattr(widget, 'children'):
                for child in widget.children():
                    if hasattr(child, 'text'):
                        text = child.text()
                        if text.startswith('Total:'):
                            labels['total'] = child
                        elif text.startswith('✅ Completed:'):
                            labels['completed'] = child
                        elif text.startswith('🔗 Merged:'):
                            labels['merged'] = child
                        elif text.startswith('🔄 In Progress:'):
                            labels['in_progress'] = child
                        elif text.startswith('❓ Pending:'):
                            labels['pending'] = child
                        elif text.startswith('⬜ Not Translated:') or text.startswith('✨ Not Refined:') or text.startswith('🔊 No TTS:'):
                            labels['missing'] = child
                        elif text.startswith('❌ Failed:'):
                            labels['failed'] = child
                    
                    # Recursively search children
                    labels.update(find_stats_labels(child))
            return labels
        
        stats_labels = find_stats_labels(container)
        
        if stats_labels:
            # Recalculate statistics from chapter_display_info (works for both OPF and non-OPF)
            chapter_display_info = data.get('chapter_display_info', [])
            pdf_rows = [info for info in chapter_display_info if info.get('pdf_ocr')]
            if pdf_rows and len(pdf_rows) == len(chapter_display_info):
                pdf_info = pdf_rows[0].get('info') or {}
                try:
                    total_chapters = int(pdf_info.get('total') or 0)
                    completed = min(int(pdf_info.get('done') or 0), total_chapters)
                    failed = int(pdf_info.get('failed') or 0)
                except (TypeError, ValueError):
                    total_chapters = len(chapter_display_info)
                    completed = 0
                    failed = 0
                merged = 0
                pending = 0
                status = self._progress_display_status(pdf_rows[0], data)
                in_progress = 1 if status == 'in_progress' else 0
                missing = max(0, total_chapters - completed - failed)
            else:
                total_chapters = len(chapter_display_info)
                display_statuses = [self._progress_display_status(info, data) for info in chapter_display_info]
                completed = sum(1 for status in display_statuses if status == 'completed')
                merged = sum(1 for status in display_statuses if status == 'merged')
                in_progress = sum(1 for status in display_statuses if status == 'in_progress')
                pending = sum(1 for status in display_statuses if status == 'pending')
                missing = sum(1 for status in display_statuses if status in ['not_translated', 'not_refined', 'no_tts'])
                failed = sum(1 for status in display_statuses if status in ['failed', 'qa_failed'])
            
            # Update labels
            if 'total' in stats_labels:
                stats_labels['total'].setText(f"Total: {total_chapters} | ")
            if 'completed' in stats_labels:
                stats_labels['completed'].setText(f"✅ Completed: {completed} | ")
            if 'merged' in stats_labels:
                if merged > 0:
                    stats_labels['merged'].setText(f"🔗 Merged: {merged} | ")
                    stats_labels['merged'].setVisible(True)
                else:
                    stats_labels['merged'].setVisible(False)
            if 'in_progress' in stats_labels:
                if in_progress > 0:
                    stats_labels['in_progress'].setText(f"🔄 In Progress: {in_progress} | ")
                    stats_labels['in_progress'].setVisible(True)
                else:
                    stats_labels['in_progress'].setVisible(False)
            if 'pending' in stats_labels:
                if pending > 0:
                    stats_labels['pending'].setText(f"❓ Pending: {pending} | ")
                    stats_labels['pending'].setVisible(True)
                else:
                    stats_labels['pending'].setVisible(False)
            if 'missing' in stats_labels:
                mode = self._current_progress_output_mode(data)
                missing_label = "✨ Not Refined" if mode == 'refinement' else ("🔊 No TTS" if mode == 'audio' else "⬜ Not Translated")
                stats_labels['missing'].setText(f"{missing_label}: {missing} | ")
            if 'failed' in stats_labels:
                stats_labels['failed'].setText(f"❌ Failed: {failed} | ")

    def _refresh_image_folder_data(self, data):
        """Refresh the image folder retranslation dialog data by rescanning files"""
        try:
            # Validate that widgets still exist
            if not self._is_data_valid(data):
                print("⚠️ Cannot refresh - widgets have been deleted")
                return
            
            # Save current selections to restore after refresh
            selected_indices = []
            try:
                selected_indices = [data['listbox'].row(item) for item in data['listbox'].selectedItems()]
            except RuntimeError:
                print("⚠️ Could not save selection state - widget was deleted")
                return
            
            output_dir = data['output_dir']
            progress_file = data['progress_file']
            folder_path = data['folder_path']
            
            def _normalize_output_file(output_file, output_dir):
                if not output_file:
                    return None
                # Normalize separators
                normalized = str(output_file).replace('\\', '/')
                # If absolute, try to store relative to output_dir when possible
                if os.path.isabs(normalized):
                    try:
                        rel = os.path.relpath(normalized, output_dir)
                        if not rel.startswith('..'):
                            return rel.replace('\\', '/')
                    except Exception:
                        pass
                    return normalized
                # If it's a relative path, keep as-is (preserve subfolders)
                return normalized
            
            # ALWAYS reload progress data from file to catch deletions
            progress_data = None
            html_files = []
            has_progress_tracking = os.path.exists(progress_file)
            
            if has_progress_tracking:
                try:
                    with open(progress_file, 'r', encoding='utf-8') as f:
                        progress_data = json.load(f)
                    print(f"🔄 Reloaded progress file from disk")
                    
                    # Extract files from progress data (primary source)
                    # Check if this is the newer nested structure with 'images' key
                    images_dict = progress_data.get('images', {})
                    if images_dict:
                        # Newer structure: progress_data['images'][hash] = {entry}
                        for key, value in images_dict.items():
                            if isinstance(value, dict) and 'output_file' in value:
                                output_file = _normalize_output_file(value['output_file'], output_dir)
                                
                                # Only include if file actually exists on disk
                                if output_file and output_file not in html_files:
                                    full_path = output_file if os.path.isabs(output_file) else os.path.join(output_dir, output_file)
                                    if os.path.exists(full_path):
                                        html_files.append(output_file)
                                    else:
                                        #print(f"⚠️ File in progress but not on disk: {output_file}")
                                        pass
                    else:
                        # Older structure: progress_data[hash] = {entry}
                        for key, value in progress_data.items():
                            if isinstance(value, dict) and 'output_file' in value:
                                output_file = _normalize_output_file(value['output_file'], output_dir)
                                
                                # Only include if file actually exists on disk
                                if output_file and output_file not in html_files:
                                    full_path = output_file if os.path.isabs(output_file) else os.path.join(output_dir, output_file)
                                    if os.path.exists(full_path):
                                        html_files.append(output_file)
                                    else:
                                        #print(f"⚠️ File in progress but not on disk: {output_file}")
                                        pass
                except Exception as e:
                    print(f"Failed to load progress file: {e}")
                    has_progress_tracking = False
            
            # Also scan directory for any HTML files not in progress (fallback)
            if os.path.exists(output_dir):
                try:
                    for file in os.listdir(output_dir):
                        file_path = os.path.join(output_dir, file)
                        if (os.path.isfile(file_path) and 
                            file.lower().endswith(('.html', '.xhtml', '.htm')) and 
                            file not in html_files):
                            html_files.append(file)
                except Exception as e:
                    print(f"Error scanning directory: {e}")
            
            # Rescan cover images
            image_files = []
            images_dir = os.path.join(output_dir, "images")
            if os.path.exists(images_dir):
                try:
                    for file in os.listdir(images_dir):
                        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                            image_files.append(file)
                except Exception as e:
                    print(f"Error scanning images directory: {e}")
            
            # Rebuild file_info list
            file_info = []
            
            # Add translated files (both HTML and generated images)
            for html_file in sorted(set(html_files)):
                # Determine file type and extract info
                file_name = os.path.basename(html_file)
                is_html = file_name.lower().endswith(('.html', '.xhtml', '.htm'))
                is_image = file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif'))
                
                if is_html:
                    match = re.match(r'response_(\d+)_(.+)\.html', file_name)
                    if match:
                        index = match.group(1)
                        base_name = match.group(2)
                elif is_image:
                    # For generated images, just use the filename
                    base_name = os.path.splitext(file_name)[0]
                
                # Find hash key if progress tracking exists
                hash_key = None
                if progress_data:
                    # Check nested structure first
                    images_dict = progress_data.get('images', {})
                    if images_dict:
                        for key, value in images_dict.items():
                            if isinstance(value, dict) and 'output_file' in value:
                                output_file = _normalize_output_file(value['output_file'], output_dir)
                                if output_file and output_file == html_file:
                                    hash_key = key
                                    break
                    else:
                        # Check flat structure
                        for key, value in progress_data.items():
                            if isinstance(value, dict) and 'output_file' in value:
                                output_file = _normalize_output_file(value['output_file'], output_dir)
                                if output_file and output_file == html_file:
                                    hash_key = key
                                    break
                
                file_info.append({
                    'type': 'translated',
                    'file': html_file,
                    'path': html_file if os.path.isabs(html_file) else os.path.join(output_dir, html_file),
                    'hash_key': hash_key,
                    'output_dir': output_dir
                })
            
            # Add cover images
            for img_file in sorted(image_files):
                file_info.append({
                    'type': 'cover',
                    'file': img_file,
                    'path': os.path.join(images_dir, img_file),
                    'hash_key': None,
                    'output_dir': output_dir
                })
            
            # Update data dictionary with fresh data
            data['file_info'] = file_info
            data['progress_data'] = progress_data
            
            # IMPORTANT: Also update the original refresh_data dict so future operations use fresh data
            # This ensures delete operations after refresh work with current state
            if 'progress_data' in data:
                # Update the reference in the closure
                data['progress_data'] = progress_data
            
            # Clear and rebuild listbox
            listbox = data['listbox']
            listbox.clear()
            
            # Add all tracked files to display
            for info in file_info:
                if info['type'] == 'translated':
                    file_name = os.path.basename(info['file'])
                    # Check if it's an HTML file or a generated image
                    is_html = file_name.lower().endswith(('.html', '.xhtml', '.htm'))
                    is_image = file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif'))
                    
                    if is_html:
                        match = re.match(r'response_(\d+)_(.+)\.html', file_name)
                        if match:
                            index = match.group(1)
                            base_name = match.group(2)
                            display = f"📄 Image {index} | {base_name} | ✅ Completed"
                        else:
                            display = f"📄 {file_name} | ✅ Completed"
                    elif is_image:
                        # Generated image file (e.g., Test1.png from imagen)
                        base_name = os.path.splitext(file_name)[0]
                        display = f"🖼️ {base_name} | ✅ Completed"
                    else:
                        display = f"📄 {file_name} | ✅ Completed"
                elif info['type'] == 'cover':
                    display = f"🖼️ Cover | {info['file']} | ⏭️ Skipped (cover)"
                else:
                    display = f"📄 {info['file']}"
                
                self._add_compact_inline_list_item(listbox, display)
            
            # Restore selections
            try:
                if selected_indices:
                    for idx in selected_indices:
                        if idx < listbox.count():
                            listbox.item(idx).setSelected(True)
                    # Update selection count
                    if 'selection_count_label' in data and data['selection_count_label']:
                        data['selection_count_label'].setText(f"Selected: {len(selected_indices)}")
                else:
                    listbox.clearSelection()
                    if 'selection_count_label' in data and data['selection_count_label']:
                        data['selection_count_label'].setText("Selected: 0")
            except RuntimeError:
                print("⚠️ Could not restore selection state - widget was deleted during refresh")
            
            print(f"✅ Image folder data refreshed: {len(html_files)} HTML files, {len(image_files)} cover images")
            
        except Exception as e:
            print(f"❌ Failed to refresh image folder data: {e}")
            import traceback
            traceback.print_exc()

    def _force_retranslation_multiple_files(self):
        """Handle force retranslation when multiple files are selected - now uses shared logic"""
        try:
            print(f"[DEBUG] _force_retranslation_multiple_files called with {len(self.selected_files)} files")
            
            # First, check if all selected files are images from the same folder
            # This handles the case where folder selection results in individual file selections
            if len(self.selected_files) > 1:
                all_images = True
                parent_dirs = set()
                
                image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')
                
                for file_path in self.selected_files:
                    if os.path.isfile(file_path) and file_path.lower().endswith(image_extensions):
                        parent_dirs.add(os.path.dirname(file_path))
                    else:
                        all_images = False
                        break
                
                # If all files are images from the same directory, treat it as a folder selection
                if all_images and len(parent_dirs) == 1:
                    folder_path = parent_dirs.pop()
                    print(f"[DEBUG] Detected {len(self.selected_files)} images from same folder: {folder_path}")
                    print(f"[DEBUG] Treating as folder selection")
                    self._force_retranslation_images_folder(folder_path)
                    return
            
            # Otherwise, continue with normal categorization
            epub_files = []
            text_files = []
            image_files = []
            folders = []
            
            image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')
            
            for file_path in self.selected_files:
                if os.path.isdir(file_path):
                    folders.append(file_path)
                elif file_path.lower().endswith('.epub'):
                    epub_files.append(file_path)
                elif file_path.lower().endswith('.txt'):
                    text_files.append(file_path)
                elif file_path.lower().endswith(image_extensions):
                    image_files.append(file_path)
            
            # Build summary
            summary_parts = []
            if epub_files:
                summary_parts.append(f"{len(epub_files)} EPUB file(s)")
            if text_files:
                summary_parts.append(f"{len(text_files)} text file(s)")
            if image_files:
                summary_parts.append(f"{len(image_files)} image file(s)")
            if folders:
                summary_parts.append(f"{len(folders)} folder(s)")
            
            if not summary_parts:
                self._styled_msgbox(QMessageBox.Information, self, "Info", "No valid files selected.")
                return
            
            # Create a unique key for the current selection
            selection_key = tuple(sorted(self.selected_files))
            
            # Check if we already have a cached dialog for this exact selection
            if (hasattr(self, '_multi_file_retranslation_dialog') and 
                self._multi_file_retranslation_dialog and 
                hasattr(self, '_multi_file_selection_key') and 
                self._multi_file_selection_key == selection_key):
                # Reuse existing dialog - show first, then refresh tabs without blocking open.
                cached_dialog = self._multi_file_retranslation_dialog
                cached_dialog.show()
                cached_dialog.raise_()
                cached_dialog.activateWindow()
                if getattr(cached_dialog, '_multi_file_tabs_building', False):
                    return

                def _refresh_cached_tabs():
                    if not hasattr(cached_dialog, '_tab_data') or not cached_dialog._tab_data:
                        return
                    print(f"[DEBUG] Auto-clicking refresh on all {len(cached_dialog._tab_data)} tabs in cached dialog...")
                    for _td in cached_dialog._tab_data:
                        _rf = _td.get('refresh_func') if _td else None
                        if callable(_rf):
                            try:
                                _rf()
                            except Exception as _e:
                                print(f"[WARN] Auto-refresh failed for a tab: {_e}")

                QTimer.singleShot(50, _refresh_cached_tabs)
                return
            
            # If there's an existing dialog for a different selection, destroy it first
            if hasattr(self, '_multi_file_retranslation_dialog') and self._multi_file_retranslation_dialog:
                self._multi_file_retranslation_dialog.close()
                self._multi_file_retranslation_dialog.deleteLater()
                self._multi_file_retranslation_dialog = None
            
            # Create main dialog
            dialog = QDialog(self)
            dialog.setWindowTitle("Progress Manager - Multiple Files")
            # Parent-child windowing keeps this above the translator GUI
            dialog.setWindowModality(Qt.NonModal)
            # Store the list of EPUBs in the dialog for cross-tab state updates
            dialog._epub_files_in_dialog = epub_files + text_files
            # Increased height from 18% to 25% for better visibility
            width, height = self._get_dialog_size(0.25, 0.45)
            dialog.resize(width, height)
            
            # Set icon
            try:
                from PySide6.QtGui import QIcon
                if hasattr(self, 'base_dir'):
                    base_dir = self.base_dir
                else:
                    base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
                ico_path = os.path.join(base_dir, 'Halgakos.ico')
                if os.path.isfile(ico_path):
                    dialog.setWindowIcon(QIcon(ico_path))
            except Exception as e:
                print(f"Failed to load icon: {e}")
            
            dialog_layout = QVBoxLayout(dialog)
            
            # Summary label
            summary_label = QLabel(f"Selected: {', '.join(summary_parts)}")
            summary_font = QFont('Arial', 12)
            summary_font.setBold(True)
            summary_label.setFont(summary_font)
            dialog_layout.addWidget(summary_label)
            
            # Count total files for UI decision
            total_files = len(epub_files) + len(text_files) + len(folders) + len(image_files)
            use_dropdown = total_files > 3

            if use_dropdown:
                # ── Dropdown + arrows for many files ──
                from PySide6.QtWidgets import QComboBox, QStackedWidget

                nav_row = QHBoxLayout()
                nav_row.setSpacing(6)

                nav_prev = QPushButton("◀")
                nav_prev.setFixedWidth(36)
                nav_prev.setStyleSheet(
                    "QPushButton { background-color:#3a3a3a; color:white; font-weight:bold; "
                    "font-size:13pt; border:1px solid #5a9fd4; border-radius:4px; padding:4px; }"
                    "QPushButton:hover { background-color:#4a8fc4; }"
                    "QPushButton:disabled { color:#666; background-color:#2a2a2a; }"
                )

                combo = QComboBox()
                combo.setStyleSheet(
                    "QComboBox { background-color:#3a3a3a; color:white; font-weight:bold; "
                    "font-size:11pt; padding:6px 10px; border:1px solid #5a9fd4; border-radius:4px; }"
                    "QComboBox::drop-down { border:none; }"
                    "QComboBox QAbstractItemView { background-color:#2d2d2d; color:white; "
                    "selection-background-color:#5a9fd4; }"
                )

                nav_counter = QLabel("1 / 1")
                nav_counter.setStyleSheet("color:#94a3b8; font-size:10pt; font-weight:bold;")
                nav_counter.setFixedWidth(60)
                nav_counter.setAlignment(Qt.AlignCenter)

                nav_next = QPushButton("▶")
                nav_next.setFixedWidth(36)
                nav_next.setStyleSheet(nav_prev.styleSheet())

                nav_row.addWidget(nav_prev)
                nav_row.addWidget(combo, stretch=1)
                nav_row.addWidget(nav_counter)
                nav_row.addWidget(nav_next)
                dialog_layout.addLayout(nav_row)

                stack = QStackedWidget()
                dialog_layout.addWidget(stack)

                def _update_nav():
                    idx = combo.currentIndex()
                    n = combo.count()
                    nav_prev.setEnabled(idx > 0)
                    nav_next.setEnabled(idx < n - 1)
                    nav_counter.setText(f"{idx + 1} / {n}")
                    stack.setCurrentIndex(idx)

                combo.currentIndexChanged.connect(lambda _: _update_nav())
                nav_prev.clicked.connect(lambda: combo.setCurrentIndex(combo.currentIndex() - 1))
                nav_next.clicked.connect(lambda: combo.setCurrentIndex(combo.currentIndex() + 1))

                # Wrap stack+combo to behave like QTabWidget for the rest of the code
                class _DropdownNotebook:
                    """Thin adapter so addTab() works the same as QTabWidget."""
                    def __init__(self, stack, combo):
                        self._stack = stack
                        self._combo = combo
                    def addTab(self, widget, label):
                        self._stack.addWidget(widget)
                        self._combo.addItem(label)
                    def currentIndex(self):
                        return self._combo.currentIndex()
                    def setCurrentIndex(self, idx):
                        self._combo.setCurrentIndex(idx)

                notebook = _DropdownNotebook(stack, combo)
                dialog._dropdown_update_nav = _update_nav
            else:
                # ── Standard tabs for ≤7 files ──
                notebook = QTabWidget()
                notebook.setStyleSheet("""
                    QTabWidget::pane {
                        border: 2px solid #5a9fd4;
                        border-radius: 4px;
                        background-color: #2d2d2d;
                    }
                    QTabBar::tab {
                        background-color: #3a3a3a;
                        color: white;
                        padding: 8px 16px;
                        margin-right: 2px;
                        border: 1px solid #5a9fd4;
                        border-bottom: none;
                        border-top-left-radius: 4px;
                        border-top-right-radius: 4px;
                        font-weight: bold;
                        font-size: 11pt;
                    }
                    QTabBar::tab:selected {
                        background-color: #5a9fd4;
                        color: white;
                    }
                    QTabBar::tab:hover {
                        background-color: #4a8fc4;
                    }
                    QTabBar QToolButton {
                        background-color: #3a3a3a;
                        border: 1px solid #5a9fd4;
                        border-radius: 3px;
                        color: white;
                        font-weight: bold;
                        font-size: 14pt;
                        width: 36px;
                        padding: 4px;
                        margin: 2px 4px;
                    }
                    QTabBar QToolButton:hover {
                        background-color: #4a8fc4;
                    }
                    QTabBar::scroller {
                        width: 52px;
                    }
                """)
                dialog_layout.addWidget(notebook)
            
            # Track all tab data
            tab_data = []
            tabs_created = False
            
            # Store tab_data reference on the dialog for cross-tab operations
            dialog._tab_data = tab_data

            # Paint the full-size multi-file shell before scanning/building every EPUB tab.
            dialog.show()
            try:
                from PySide6.QtWidgets import QApplication
                QApplication.processEvents(QEventLoop.AllEvents, 50)
            except Exception:
                pass
            
            # Get the global show_special state from the first file that has it cached
            # Default to True if any text files are present, False otherwise
            global_show_special = True if text_files else False
            
            for file_path in epub_files + text_files:
                file_key = os.path.abspath(file_path)
                if hasattr(self, '_retranslation_dialog_cache') and file_key in self._retranslation_dialog_cache:
                    cached_data = self._retranslation_dialog_cache[file_key]
                    if cached_data and 'show_special_files_state' in cached_data:
                        global_show_special = cached_data['show_special_files_state']
                        break  # Use the first one we find
            
            # Determine output directory override (matches single-file logic)
            override_dir = (os.environ.get('OUTPUT_DIRECTORY') or os.environ.get('OUTPUT_DIR'))
            if not override_dir and hasattr(self, 'config'):
                try:
                    override_dir = self.config.get('output_directory')
                except Exception:
                    override_dir = None

            # Stream tab creation: add lightweight tab shells immediately, then build
            # each EPUB/text tab on its own event-loop turn.
            self._add_multi_file_buttons(dialog, notebook, tab_data)

            def closeEvent(event):
                event.ignore()
                dialog.hide()

            dialog.closeEvent = closeEvent
            self._multi_file_retranslation_dialog = dialog
            self._multi_file_selection_key = selection_key

            def _update_dropdown_nav_safe():
                if hasattr(dialog, '_dropdown_update_nav'):
                    try:
                        dialog._dropdown_update_nav()
                    except RuntimeError:
                        pass

            build_tasks = []
            for file_path in epub_files + text_files:
                file_base = os.path.splitext(os.path.basename(file_path))[0]
                print(f"[DEBUG] Queueing EPUB/text tab: {file_base}")

                output_dir = os.path.join(override_dir, file_base) if override_dir else file_base
                if not os.path.exists(output_dir):
                    print(f"[DEBUG] Output folder missing for {file_base}; will create via tab builder: {output_dir}")

                tab_frame = QWidget()
                tab_layout = QVBoxLayout(tab_frame)
                tab_layout.setContentsMargins(0, 0, 0, 0)
                loading_label = QLabel(f"Loading {file_base}...")
                loading_label.setAlignment(Qt.AlignCenter)
                loading_label.setStyleSheet("color: #94a3b8; font-size: 10pt; font-weight: bold; padding: 18px;")
                tab_layout.addWidget(loading_label)

                tab_name = file_base if use_dropdown else (file_base[:20] + "..." if len(file_base) > 20 else file_base)
                notebook.addTab(tab_frame, tab_name)
                _update_dropdown_nav_safe()
                build_tasks.append(('epub_text', file_path, file_base, tab_frame))

            for folder_path in folders:
                build_tasks.append(('folder', folder_path, os.path.basename(folder_path) or folder_path, None))

            build_state = {'idx': 0, 'tabs_created': False}
            dialog._multi_file_tabs_building = True

            def _refresh_tabs_streamed(idx=0):
                if idx >= len(tab_data):
                    return
                _td = tab_data[idx]
                _rf = _td.get('refresh_func') if _td else None
                if callable(_rf):
                    try:
                        _rf()
                    except Exception as _e:
                        print(f"[WARN] Auto-refresh failed for a tab: {_e}")
                QTimer.singleShot(25, lambda: _refresh_tabs_streamed(idx + 1))

            def _finish_streamed_tabs():
                dialog._multi_file_tabs_building = False

                if image_files and not build_state['tabs_created']:
                    image_tab_result = self._create_individual_images_tab(
                        image_files,
                        notebook,
                        dialog
                    )
                    if image_tab_result:
                        tab_data.append(image_tab_result)
                        build_state['tabs_created'] = True
                        _update_dropdown_nav_safe()

                if not build_state['tabs_created'] and folders:
                    scanned_images = []
                    for folder_path in folders:
                        if os.path.isdir(folder_path):
                            try:
                                for file in os.listdir(folder_path):
                                    file_path = os.path.join(folder_path, file)
                                    if os.path.isfile(file_path) and file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                                        scanned_images.append(file_path)
                            except Exception:
                                pass

                    if scanned_images:
                        image_tab_result = self._create_individual_images_tab(
                            scanned_images,
                            notebook,
                            dialog
                        )
                        if image_tab_result:
                            tab_data.append(image_tab_result)
                            build_state['tabs_created'] = True
                            _update_dropdown_nav_safe()

                if not build_state['tabs_created']:
                    self._styled_msgbox(QMessageBox.Information, self, "Info",
                        "No translation output found for any of the selected files.\n\n"
                        "Make sure the output folders exist in your script directory.")
                    dialog.hide()
                    return

                _update_dropdown_nav_safe()
                if tab_data:
                    print(f"[DEBUG] Auto-clicking refresh on all {len(tab_data)} tabs on dialog open...")
                    QTimer.singleShot(100, lambda: _refresh_tabs_streamed(0))
                else:
                    print(f"[WARN] No tab data to refresh on dialog open")

            def _build_next_tab():
                if build_state['idx'] >= len(build_tasks):
                    _finish_streamed_tabs()
                    return

                kind, path, label, tab_frame = build_tasks[build_state['idx']]
                build_state['idx'] += 1

                if kind == 'epub_text':
                    print(f"[DEBUG] Creating streamed tab for {label}")
                    try:
                        if tab_frame.layout():
                            self._clear_layout(tab_frame.layout())
                        tab_result = self._force_retranslation_epub_or_text(
                            path,
                            parent_dialog=dialog,
                            tab_frame=tab_frame,
                            show_special_files_state=global_show_special
                        )
                        if tab_result:
                            cdi = tab_result.get('chapter_display_info', [])
                            completed = sum(1 for info in cdi if info.get('status') == 'completed')
                            in_progress = sum(1 for info in cdi if info.get('status') == 'in_progress')
                            tab_data.append(tab_result)
                            build_state['tabs_created'] = True
                            QTimer.singleShot(25, lambda td=tab_result: self._populate_progress_listbox_streamed(td))
                            print(f"[DEBUG] Successfully created tab for {label} (progress: {completed} done, {in_progress} in-progress)")
                        else:
                            if tab_frame.layout():
                                self._clear_layout(tab_frame.layout())
                                failed_label = QLabel(f"Failed to load {label}")
                                failed_label.setAlignment(Qt.AlignCenter)
                                failed_label.setStyleSheet("color: #e74c3c; font-size: 10pt; font-weight: bold; padding: 18px;")
                                tab_frame.layout().addWidget(failed_label)
                            print(f"[DEBUG] Failed to create content for {label}")
                    except Exception as _e:
                        print(f"[WARN] Failed to create streamed tab for {label}: {_e}")
                elif kind == 'folder':
                    folder_result = self._create_image_folder_tab(
                        path,
                        notebook,
                        dialog
                    )
                    if folder_result:
                        tab_data.append(folder_result)
                        build_state['tabs_created'] = True
                        _update_dropdown_nav_safe()

                QTimer.singleShot(0, _build_next_tab)

            QTimer.singleShot(50, _build_next_tab)
            return

            # Create tabs for EPUB/text files using shared logic
            pending_tabs = []  # Collect before sorting
            for file_path in epub_files + text_files:
                file_base = os.path.splitext(os.path.basename(file_path))[0]
                
                print(f"[DEBUG] Checking EPUB/text: {file_base}")
                
                # Quick check if output exists (respect override output directory)
                # NOTE: For multi-file, don't skip when missing — the tab builder will
                # create the output folder (same as single-file behavior).
                output_dir = os.path.join(override_dir, file_base) if override_dir else file_base
                if not os.path.exists(output_dir):
                    print(f"[DEBUG] Output folder missing for {file_base}; will create via tab builder: {output_dir}")
                
                print(f"[DEBUG] Creating tab for {file_base}")
                
                # Create tab
                tab_frame = QWidget()
                tab_layout = QVBoxLayout(tab_frame)
                tab_name = file_base if use_dropdown else (file_base[:20] + "..." if len(file_base) > 20 else file_base)
                
                # Use shared logic to populate the tab with global state
                tab_result = self._force_retranslation_epub_or_text(
                    file_path, 
                    parent_dialog=dialog, 
                    tab_frame=tab_frame,
                    show_special_files_state=global_show_special
                )
                
                # Only keep the tab if content was successfully created
                if tab_result:
                    # Count progress for sorting
                    cdi = tab_result.get('chapter_display_info', [])
                    completed = sum(1 for info in cdi if info.get('status') == 'completed')
                    in_progress = sum(1 for info in cdi if info.get('status') == 'in_progress')
                    progress_score = completed + in_progress
                    pending_tabs.append((progress_score, tab_frame, tab_name, tab_result))
                    print(f"[DEBUG] Successfully created tab for {file_base} (progress: {completed} done, {in_progress} in-progress)")
                else:
                    print(f"[DEBUG] Failed to create content for {file_base}")
            
            # Sort tabs: most progress first
            pending_tabs.sort(key=lambda t: t[0], reverse=True)
            for _score, tab_frame, tab_name, tab_result in pending_tabs:
                notebook.addTab(tab_frame, tab_name)
                tab_data.append(tab_result)
                tabs_created = True
            
            # Create tabs for image folders (keeping existing logic for now)
            for folder_path in folders:
                folder_result = self._create_image_folder_tab(
                    folder_path, 
                    notebook, 
                    dialog
                )
                if folder_result:
                    tab_data.append(folder_result)
                    tabs_created = True
            
            # If only individual image files selected and no tabs created yet
            if image_files and not tabs_created:
                # Create a single tab for all individual images
                image_tab_result = self._create_individual_images_tab(
                    image_files,
                    notebook,
                    dialog
                )
                if image_tab_result:
                    tab_data.append(image_tab_result)
                    tabs_created = True
            
            # If no tabs were created from folders, try scanning folders for individual images
            if not tabs_created and folders:
                # Scan folders for individual image files
                scanned_images = []
                for folder_path in folders:
                    if os.path.isdir(folder_path):
                        try:
                            for file in os.listdir(folder_path):
                                file_path = os.path.join(folder_path, file)
                                if os.path.isfile(file_path) and file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                                    scanned_images.append(file_path)
                        except:
                            pass
                
                # If we found images, create a tab for them
                if scanned_images:
                    image_tab_result = self._create_individual_images_tab(
                        scanned_images,
                        notebook,
                        dialog
                    )
                    if image_tab_result:
                        tab_data.append(image_tab_result)
                        tabs_created = True
            
            # If still no tabs were created, show error
            if not tabs_created:
                self._styled_msgbox(QMessageBox.Information, self, "Info", 
                    "No translation output found for any of the selected files.\n\n"
                    "Make sure the output folders exist in your script directory.")
                dialog.close()
                return
        
            # Add unified button bar that works across all tabs
            self._add_multi_file_buttons(dialog, notebook, tab_data)
            
            # Override close event to minimize instead of destroy
            def closeEvent(event):
                event.ignore()  # Ignore the close event
                dialog.hide()   # Just hide (minimize) the dialog
            
            dialog.closeEvent = closeEvent
            
            # Cache the dialog and selection key for reuse
            self._multi_file_retranslation_dialog = dialog
            self._multi_file_selection_key = selection_key
            
            # Update dropdown nav state after all tabs are added
            if hasattr(dialog, '_dropdown_update_nav'):
                dialog._dropdown_update_nav()

            # Show the dialog (non-modal to allow interaction with other windows)
            dialog.show()

            def _populate_tabs_after_show():
                for _idx, _td in enumerate(tab_data):
                    if _td:
                        QTimer.singleShot(
                            _idx * 10,
                            lambda td=_td: self._populate_progress_listbox_streamed(td),
                        )

            # Trigger refresh after the dialog has painted so large tabs do not block opening.
            def _refresh_tabs_after_show():
                if tab_data:
                    print(f"[DEBUG] Auto-clicking refresh on all {len(tab_data)} tabs on dialog open...")
                    for _td in tab_data:
                        _rf = _td.get('refresh_func') if _td else None
                        if callable(_rf):
                            try:
                                _rf()
                            except Exception as _e:
                                print(f"[WARN] Auto-refresh failed for a tab: {_e}")
                else:
                    print(f"[WARN] No tab data to refresh on dialog open")

            QTimer.singleShot(50, _populate_tabs_after_show)
            QTimer.singleShot(150, _refresh_tabs_after_show)
            
        except Exception as e:
            print(f"[ERROR] _force_retranslation_multiple_files failed: {e}")
            import traceback
            traceback.print_exc()
            self._styled_msgbox(QMessageBox.Critical, self, "Error", f"Failed to open retranslation dialog:\n{str(e)}")

    def _add_multi_file_buttons(self, dialog, notebook, tab_data):
        """Placeholder for future multi-file button functionality"""
        # No buttons needed - dialog has standard close button
        pass
              
    def _create_individual_images_tab(self, image_files, notebook, parent_dialog):
        """Create a tab for individual image files"""
        # Create tab
        tab_frame = QWidget()
        tab_layout = QVBoxLayout(tab_frame)
        notebook.addTab(tab_frame, "Individual Images")
        
        # Instructions
        instruction_label = QLabel(f"Selected {len(image_files)} individual image(s):")
        instruction_font = QFont('Arial', 11)
        instruction_label.setFont(instruction_font)
        tab_layout.addWidget(instruction_label)
        
        # Listbox (QListWidget has built-in scrolling)
        listbox = QListWidget()
        listbox.setSelectionMode(QListWidget.ExtendedSelection)
        self._apply_compact_inline_list_style(listbox)
        # Use 16% of screen width (half of original ~31% for 1920px screen)
        min_width, _ = self._get_dialog_size(0.16, 0)
        listbox.setMinimumWidth(min_width)
        tab_layout.addWidget(listbox)
        
        # File info
        file_info = []
        script_dir = _get_app_dir()
        
        # Check each image for translations
        for img_path in sorted(image_files):
            img_name = os.path.basename(img_path)
            base_name = os.path.splitext(img_name)[0]
            
            # Look for translations in various possible locations
            found_translations = []
            
            # Check in script directory with base name
            possible_dirs = [
                os.path.join(script_dir, base_name),
                os.path.join(script_dir, f"{base_name}_translated"),
                base_name,
                f"{base_name}_translated"
            ]
            
            for output_dir in possible_dirs:
                if os.path.exists(output_dir) and os.path.isdir(output_dir):
                    # Look for HTML files
                    for file in os.listdir(output_dir):
                        if file.lower().endswith(('.html', '.xhtml', '.htm')) and base_name in file:
                            found_translations.append((output_dir, file))
            
            if found_translations:
                for output_dir, html_file in found_translations:
                    display = f"📄 {img_name} → {html_file} | ✅ Translated"
                    self._add_compact_inline_list_item(listbox, display)
                    
                    file_info.append({
                        'type': 'translated',
                        'source_image': img_path,
                        'output_dir': output_dir,
                        'file': html_file,
                        'path': os.path.join(output_dir, html_file)
                    })
            else:
                display = f"🖼️ {img_name} | ❌ No translation found"
                self._add_compact_inline_list_item(listbox, display)
        
        # Selection count
        selection_count_label = QLabel("Selected: 0")
        selection_font = QFont('Arial', 9)
        selection_count_label.setFont(selection_font)
        tab_layout.addWidget(selection_count_label)
        
        def update_selection_count():
            count = len(listbox.selectedItems())
            selection_count_label.setText(f"Selected: {count}")
        
        listbox.itemSelectionChanged.connect(update_selection_count)

        # Right-click context menu to open translated/cover files
        def _open_file_for_row(row):
            if row < 0 or row >= len(file_info):
                return
            info = file_info[row]
            path = info.get('path')
            if not path or not os.path.exists(path):
                self._show_message('error', "File Missing", f"File not found:\n{path}", parent=parent_dialog)
                return
            try:
                QDesktopServices.openUrl(QUrl.fromLocalFile(path))
            except Exception as e:
                self._show_message('error', "Open Failed", str(e), parent=parent_dialog)

        def _show_context_menu(pos):
            item = listbox.itemAt(pos)
            if not item:
                return
            row = listbox.row(item)
            menu = QMenu(listbox)
            menu.setStyleSheet(
                "QMenu {"
                "  padding: 4px;"
                "  background-color: #2b2b2b;"
                "  color: white;"
                "  border: 1px solid #5a9fd4;"
                "} "
                "QMenu::icon { width: 0px; } "
                "QMenu::item {"
                "  padding: 6px 12px;"
                "  background-color: transparent;"
                "} "
                "QMenu::item:selected {"
                "  background-color: #17a2b8;"
                "  color: white;"
                "} "
                "QMenu::item:pressed {"
                "  background-color: #138496;"
                "}"
            )
            act_open = menu.addAction("📂 Open File")
            chosen = menu.exec(listbox.mapToGlobal(pos))
            if chosen == act_open:
                _open_file_for_row(row)

        listbox.setContextMenuPolicy(Qt.CustomContextMenu)
        listbox.customContextMenuRequested.connect(_show_context_menu)
        
        return {
            'type': 'individual_images',
            'listbox': listbox,
            'file_info': file_info,
            'selection_count_label': selection_count_label
        }


    def _create_image_folder_tab(self, folder_path, notebook, parent_dialog):
        """Create a tab for image folder retranslation"""
        folder_name = os.path.basename(folder_path)
        output_dir = f"{folder_name}_translated"
        
        if not os.path.exists(output_dir):
            return None
        
        # Create tab
        tab_frame = QWidget()
        tab_layout = QVBoxLayout(tab_frame)
        tab_name = "📁 " + (folder_name[:17] + "..." if len(folder_name) > 17 else folder_name)
        notebook.addTab(tab_frame, tab_name)
        
        # Instructions
        instruction_label = QLabel("Select images to retranslate:")
        instruction_font = QFont('Arial', 11)
        instruction_label.setFont(instruction_font)
        tab_layout.addWidget(instruction_label)
        
        # Listbox (QListWidget has built-in scrolling)
        listbox = QListWidget()
        listbox.setSelectionMode(QListWidget.ExtendedSelection)
        self._apply_compact_inline_list_style(listbox)
        # Use 16% of screen width (half of original ~31% for 1920px screen)
        min_width, _ = self._get_dialog_size(0.16, 0)
        listbox.setMinimumWidth(min_width)
        tab_layout.addWidget(listbox)
        
        # Find files
        file_info = []
        
        # Add HTML files (any .html/.xhtml/.htm, not just response_*)
        for file in os.listdir(output_dir):
            if file.lower().endswith(('.html', '.xhtml', '.htm')):
                match = re.match(r'^response_(\d+)_([^.]*).(?:html?|xhtml|htm)(?:\.xhtml)?$', file, re.IGNORECASE)
                if match:
                    index = match.group(1)
                    base_name = match.group(2)
                    display = f"📄 Image {index} | {base_name} | ✅ Completed"
                else:
                    display = f"📄 {file} | ✅ Completed"
                
                self._add_compact_inline_list_item(listbox, display)
                file_info.append({
                    'type': 'translated',
                    'file': file,
                    'path': os.path.join(output_dir, file)
                })
        
        # Add cover images
        images_dir = os.path.join(output_dir, "images")
        if os.path.exists(images_dir):
            for file in sorted(os.listdir(images_dir)):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                    display = f"🖼️ Cover | {file} | ⏭️ Skipped"
                    self._add_compact_inline_list_item(listbox, display)
                    file_info.append({
                        'type': 'cover',
                        'file': file,
                        'path': os.path.join(images_dir, file)
                    })
        
        # Selection count
        selection_count_label = QLabel("Selected: 0")
        selection_font = QFont('Arial', 9)
        selection_count_label.setFont(selection_font)
        tab_layout.addWidget(selection_count_label)
        
        def update_selection_count():
            count = len(listbox.selectedItems())
            selection_count_label.setText(f"Selected: {count}")
        
        listbox.itemSelectionChanged.connect(update_selection_count)

        # Right-click context menu (Open File)
        def _open_file_for_row(row):
            if row < 0 or row >= len(file_info):
                return
            info = file_info[row]
            path = info.get('path')
            if not path or not os.path.exists(path):
                self._show_message('error', "File Missing", f"File not found:\n{path}", parent=parent_dialog)
                return
            try:
                QDesktopServices.openUrl(QUrl.fromLocalFile(path))
            except Exception as e:
                self._show_message('error', "Open Failed", str(e), parent=parent_dialog)

        def _show_context_menu(pos):
            item = listbox.itemAt(pos)
            if not item:
                return
            row = listbox.row(item)
            menu = QMenu(listbox)
            menu.setStyleSheet(
                "QMenu {"
                "  padding: 4px;"
                "  background-color: #2b2b2b;"
                "  color: white;"
                "  border: 1px solid #5a9fd4;"
                "} "
                "QMenu::icon { width: 0px; } "
                "QMenu::item {"
                "  padding: 6px 12px;"
                "  background-color: transparent;"
                "} "
                "QMenu::item:selected {"
                "  background-color: #17a2b8;"
                "  color: white;"
                "} "
                "QMenu::item:pressed {"
                "  background-color: #138496;"
                "}"
            )
            act_open = menu.addAction("📂 Open File")
            chosen = menu.exec(listbox.mapToGlobal(pos))
            if chosen == act_open:
                _open_file_for_row(row)

        listbox.setContextMenuPolicy(Qt.CustomContextMenu)
        listbox.customContextMenuRequested.connect(_show_context_menu)
        
        return {
            'type': 'image_folder',
            'folder_path': folder_path,
            'output_dir': output_dir,
            'listbox': listbox,
            'file_info': file_info,
            'selection_count_label': selection_count_label
        }


    def _force_retranslation_images_folder(self, folder_path):
        """Handle force retranslation for image folders"""
        # If folder_path is actually a file (single image), get its directory
        if os.path.isfile(folder_path):
            # Single image file - use basename without extension
            folder_name = os.path.splitext(os.path.basename(folder_path))[0]
        else:
            # Folder - use folder name as-is
            folder_name = os.path.basename(folder_path)
        
        # Check if we already have a cached dialog for this folder
        folder_key = os.path.abspath(folder_path)
        if hasattr(self, '_image_retranslation_dialog_cache') and folder_key in self._image_retranslation_dialog_cache:
            cached_dialog = self._image_retranslation_dialog_cache[folder_key]
            if cached_dialog:
                # Reuse existing dialog - just show it
                try:
                    # Click stored refresh button or call stored refresh func on reuse
                    if hasattr(cached_dialog, '_refresh_button') and cached_dialog._refresh_button:
                        QTimer.singleShot(0, cached_dialog._refresh_button.click)
                    elif hasattr(cached_dialog, '_refresh_func'):
                        QTimer.singleShot(0, cached_dialog._refresh_func)
                except Exception:
                    pass
                cached_dialog.show()
                cached_dialog.raise_()
                cached_dialog.activateWindow()
                return
        
        # Look for output folder in the SCRIPT'S directory, not relative to the selected folder
        script_dir = _get_app_dir()  # Application directory where output is generated
        
        # Check multiple possible output folder patterns IN THE SCRIPT DIRECTORY
        possible_output_dirs = [
            os.path.join(script_dir, folder_name),  # Script dir + folder name (without extension)
            os.path.join(script_dir, f"{folder_name}_translated"),  # Script dir + folder_translated
            folder_name,  # Just the folder name in current directory
            f"{folder_name}_translated",  # folder_translated in current directory
        ]
        
        # Check for output directory override
        override_dir = os.environ.get('OUTPUT_DIRECTORY')
        if not override_dir and hasattr(self, 'config'):
            override_dir = self.config.get('output_directory')
            
        if override_dir:
            # If override is set, check inside it for the folder name
            possible_output_dirs.insert(0, os.path.join(override_dir, folder_name))
            possible_output_dirs.insert(1, os.path.join(override_dir, f"{folder_name}_translated"))
        
        output_dir = None
        for possible_dir in possible_output_dirs:
            print(f"Checking: {possible_dir}")
            if os.path.exists(possible_dir):
                # Check if it has translation_progress.json or HTML files
                if os.path.exists(os.path.join(possible_dir, "translation_progress.json")):
                    output_dir = possible_dir
                    print(f"Found output directory with progress tracker: {output_dir}")
                    break
                # Check if it has any HTML files
                elif os.path.isdir(possible_dir):
                    try:
                        files = os.listdir(possible_dir)
                        if any(f.lower().endswith(('.html', '.xhtml', '.htm')) for f in files):
                            output_dir = possible_dir
                            print(f"Found output directory with HTML files: {output_dir}")
                            break
                    except:
                        pass
        
        if not output_dir:
            self._styled_msgbox(QMessageBox.Information, self, "Info", 
                f"No translation output found for '{folder_name}'.\n\n"
                f"Selected folder: {folder_path}\n"
                f"Script directory: {script_dir}\n\n"
                f"Checked locations:\n" + "\n".join(f"- {d}" for d in possible_output_dirs))
            return
        
        print(f"Using output directory: {output_dir}")
        
        # Check for progress tracking file
        progress_file = os.path.join(output_dir, "translation_progress.json")
        has_progress_tracking = os.path.exists(progress_file)
        
        print(f"Progress tracking: {has_progress_tracking} at {progress_file}")
        
        # Find all HTML files in the output directory
        html_files = []
        _html_seen = set()
        image_files = []
        progress_data = None
        
        if has_progress_tracking:
            # Load progress data for image translations
            try:
                with open(progress_file, 'r', encoding='utf-8') as f:
                    progress_data = json.load(f)
                    print(f"Loaded progress data with {len(progress_data)} entries")
                    
                # Extract files from progress data
                # The structure appears to use hash keys at the root level
                for key, value in progress_data.items():
                    if isinstance(value, dict) and 'output_file' in value:
                        output_file = value['output_file']
                        if not output_file:
                            continue
                        # Normalize path
                        output_norm = os.path.normpath(str(output_file))
                        # If absolute and under output_dir, store as relative
                        try:
                            if os.path.isabs(output_norm) and output_dir:
                                outdir_norm = os.path.normpath(output_dir)
                                if output_norm.startswith(outdir_norm):
                                    output_norm = os.path.relpath(output_norm, outdir_norm)
                        except Exception:
                            pass
                        if output_norm in _html_seen:
                            continue
                        _html_seen.add(output_norm)
                        html_files.append(output_norm)
                        print(f"Found tracked file: {output_norm}")
            except Exception as e:
                print(f"Error loading progress file: {e}")
                import traceback
                traceback.print_exc()
                has_progress_tracking = False
        
            # Also scan directory for any HTML files not in progress
            # Include all .html/.xhtml/.htm files plus generated image files
        try:
            for file in os.listdir(output_dir):
                file_path = os.path.join(output_dir, file)
                # Include HTML files (any name)
                if (os.path.isfile(file_path) and 
                    file.lower().endswith(('.html', '.xhtml', '.htm')) and 
                    file not in html_files and file not in _html_seen):
                    _html_seen.add(file)
                    html_files.append(file)
                    print(f"Found HTML file: {file}")
                # Also include generated image files (not in images/ subdirectory)
                elif (os.path.isfile(file_path) and 
                      file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif')) and
                      file not in html_files):
                    html_files.append(file)  # Add to html_files for now, will be handled separately
                    print(f"Found generated image file: {file}")
        except Exception as e:
            print(f"Error scanning directory: {e}")
        
        # Check for images subdirectory (cover images)
        images_dir = os.path.join(output_dir, "images")
        if os.path.exists(images_dir):
            try:
                for file in os.listdir(images_dir):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                        image_files.append(file)
            except Exception as e:
                print(f"Error scanning images directory: {e}")
        
        print(f"Total files found: {len(html_files)} HTML, {len(image_files)} images")
        
        if not html_files and not image_files:
            self._styled_msgbox(QMessageBox.Information, self, "Info", 
                f"No translated files found in: {output_dir}\n\n"
                f"Progress tracking: {'Yes' if has_progress_tracking else 'No'}")
            return
        
        # Create dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Progress Manager - Images")
        # Parent-child windowing keeps this above the translator GUI
        dialog.setWindowModality(Qt.NonModal)
        # Decreased width to 18%, increased height to 25% for better vertical space
        width, height = self._get_dialog_size(0.18, 0.25)
        dialog.resize(width, height)
        
        # Set icon
        try:
            from PySide6.QtGui import QIcon
            if hasattr(self, 'base_dir'):
                base_dir = self.base_dir
            else:
                base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
            ico_path = os.path.join(base_dir, 'Halgakos.ico')
            if os.path.isfile(ico_path):
                dialog.setWindowIcon(QIcon(ico_path))
        except Exception as e:
            print(f"Failed to load icon: {e}")
        
        dialog_layout = QVBoxLayout(dialog)
        
        # Create listbox (QListWidget has built-in scrolling)
        listbox = QListWidget()
        listbox.setSelectionMode(QListWidget.ExtendedSelection)
        self._apply_compact_inline_list_style(listbox)
        # Use 16% of screen width (half of original ~31% for 1920px screen)
        min_width, _ = self._get_dialog_size(0.16, 0)
        listbox.setMinimumWidth(min_width)
        dialog_layout.addWidget(listbox)
        
        # Keep track of file info
        file_info = []
        
        progress_data_current = progress_data
        
        # Add translated HTML files
        for html_file in sorted(set(html_files)):  # Use set to avoid duplicates
            display_name = os.path.basename(html_file)
            # Extract original image name from HTML filename
            # Expected format: response_001_imagename.html
            match = re.match(r'response_(\d+)_(.+)\.html', display_name)
            if match:
                index = match.group(1)
                base_name = match.group(2)
                display = f"📄 Image {index} | {base_name} | ✅ Completed"
            else:
                display = f"📄 {display_name} | ✅ Completed"
            
            self._add_compact_inline_list_item(listbox, display)
            
            # Find the hash key for this file if progress tracking exists
            hash_key = None
            if progress_data_current:
                for key, value in progress_data_current.items():
                    if isinstance(value, dict) and 'output_file' in value:
                        outp = str(value.get('output_file') or '')
                        if html_file == outp or display_name == os.path.basename(outp) or html_file in outp:
                            hash_key = key
                            break
            
            # Build absolute path (preserve subfolders if present)
            if os.path.isabs(html_file):
                abs_path = html_file
            else:
                abs_path = os.path.join(output_dir, html_file)
            file_info.append({
                'type': 'translated',
                'file': html_file,  # may include subfolders relative to output_dir
                'path': abs_path,
                'hash_key': hash_key,
                'output_dir': output_dir  # Store for later use
            })
        
        # Add cover images
        for img_file in sorted(image_files):
            display = f"🖼️ Cover | {img_file} | ⏭️ Skipped (cover)"
            self._add_compact_inline_list_item(listbox, display)
            file_info.append({
                'type': 'cover',
                'file': img_file,
                'path': os.path.join(images_dir, img_file),
                'hash_key': None,
                'output_dir': output_dir
            })
        
        # Selection count label
        selection_count_label = QLabel("Selected: 0")
        selection_font = QFont('Arial', 10)
        selection_count_label.setFont(selection_font)
        dialog_layout.addWidget(selection_count_label)
        
        def update_selection_count():
            count = len(listbox.selectedItems())
            selection_count_label.setText(f"Selected: {count}")
        
        listbox.itemSelectionChanged.connect(update_selection_count)

        # ==== Context menu for image list ====
        def _open_file_for_index(idx):
            info_list = refresh_data.get('file_info', file_info)
            if idx < 0 or idx >= len(info_list):
                return
            info = info_list[idx]
            path = info.get('path')
            if not path or not os.path.exists(path):
                self._show_message('error', "File Missing", f"File not found:\n{path}", parent=dialog)
                return
            try:
                QDesktopServices.openUrl(QUrl.fromLocalFile(path))
            except Exception as e:
                self._show_message('error', "Open Failed", str(e), parent=dialog)

        def _show_context_menu(pos):
            item = listbox.itemAt(pos)
            if not item:
                return
            row = listbox.row(item)
            menu = QMenu(listbox)
            menu.setStyleSheet(
                "QMenu {"
                "  padding: 4px;"
                "  background-color: #2b2b2b;"
                "  color: white;"
                "  border: 1px solid #5a9fd4;"
                "} "
                "QMenu::icon { width: 0px; } "
                "QMenu::item {"
                "  padding: 6px 12px;"
                "  background-color: transparent;"
                "} "
                "QMenu::item:selected {"
                "  background-color: #17a2b8;"
                "  color: white;"
                "} "
                "QMenu::item:pressed {"
                "  background-color: #138496;"
                "}"
            )
            act_open = menu.addAction("📂 Open File")
            act_delete = menu.addAction("🔁 Delete / Retranslate")
            chosen = menu.exec(listbox.mapToGlobal(pos))
            if chosen == act_open:
                _open_file_for_index(row)
            elif chosen == act_delete:
                retranslate_selected()

        listbox.setContextMenuPolicy(Qt.CustomContextMenu)
        listbox.customContextMenuRequested.connect(_show_context_menu)
        
        # Button frame
        button_frame = QWidget()
        button_layout = QGridLayout(button_frame)
        dialog_layout.addWidget(button_frame)
        
        def select_all():
            listbox.selectAll()
            update_selection_count()
        
        def clear_selection():
            listbox.clearSelection()
            update_selection_count()
        
        def select_translated():
            listbox.clearSelection()
            info_list = refresh_data.get('file_info', file_info)
            for idx, info in enumerate(info_list):
                if info['type'] == 'translated':
                    listbox.item(idx).setSelected(True)
            update_selection_count()
        
        def mark_as_skipped():
            """Move selected images to the images folder to be skipped"""
            selected_items = listbox.selectedItems()
            if not selected_items:
                self._styled_msgbox(QMessageBox.Warning, dialog, "No Selection", "Please select at least one image to mark as skipped.")
                return
            
            # Get all selected items
            selected_indices = [listbox.row(item) for item in selected_items]
            info_list = refresh_data.get('file_info', file_info)
            items_with_info = [(i, info_list[i]) for i in selected_indices]
            progress_data_current = refresh_data.get('progress_data', progress_data)
            
            # Filter out items already in images folder (covers)
            items_to_move = [(i, item) for i, item in items_with_info if item['type'] != 'cover']
            
            if not items_to_move:
                self._styled_msgbox(QMessageBox.Information, dialog, "Info", "Selected items are already in the images folder (skipped).")
                return
            
            count = len(items_to_move)
            reply = self._styled_msgbox(QMessageBox.Question, dialog, "Confirm Mark as Skipped", 
                                      f"Move {count} translated image(s) to the images folder?\n\n"
                                      "This will:\n"
                                      "• Delete the translated HTML files\n"
                                      "• Copy source images to the images folder\n"
                                      "• Skip these images in future translations",
                                      QMessageBox.Yes | QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
            
            # Create images directory if it doesn't exist
            images_dir = os.path.join(output_dir, "images")
            os.makedirs(images_dir, exist_ok=True)
            
            moved_count = 0
            failed_count = 0
            
            for idx, item in items_to_move:
                try:
                    # Extract the original image name from the HTML filename
                    # Expected format: response_001_imagename.html (also accept compound extensions)
                    html_file = item['file']
                    html_base = os.path.basename(html_file)
                    match = re.match(r'^response_\d+_([^\.]*)\.(?:html?|xhtml|htm)(?:\.xhtml)?$', html_base, re.IGNORECASE)
                    
                    if match:
                        base_name = match.group(1)
                        # Try to find the original image with common extensions
                        original_found = False
                        
                        # Look for the source image in multiple locations
                        search_paths = [
                            folder_path,  # Original folder path
                            os.path.dirname(folder_path),  # Parent of folder path
                            os.getcwd(),  # Script directory
                        ]
                        
                        for ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']:
                            for search_path in search_paths:
                                if not search_path or not os.path.exists(search_path):
                                    continue
                                    
                                # Check in the search path
                                possible_source = os.path.join(search_path, base_name + ext)
                                if os.path.exists(possible_source) and os.path.isfile(possible_source):
                                    # Copy to images folder
                                    dest_path = os.path.join(images_dir, base_name + ext)
                                    if not os.path.exists(dest_path):
                                        import shutil
                                        shutil.copy2(possible_source, dest_path)
                                        print(f"Copied {base_name + ext} from {possible_source} to images folder")
                                    original_found = True
                                    break
                            if original_found:
                                break
                        
                        if not original_found:
                            print(f"Warning: Could not find original image for {html_file} in: {search_paths}")
                            # Even if source not found, we can still delete the HTML and mark it
                    
                    # Delete the HTML translation file
                    if os.path.exists(item['path']):
                        os.remove(item['path'])
                        print(f"Deleted translation: {item['path']}")
                        
                        # Remove from progress tracking if applicable
                        if progress_data_current and item.get('hash_key'):
                            hash_key = item['hash_key']
                            # Check nested structure first
                            if 'images' in progress_data_current and hash_key in progress_data_current['images']:
                                del progress_data_current['images'][hash_key]
                            # Check flat structure
                            elif hash_key in progress_data_current:
                                del progress_data_current[hash_key]
                    
                    # Update the listbox display
                    display = f"🖼️ Skipped | {base_name if match else html_base} | ⏭️ Moved to images folder"
                    listbox.item(idx).setText(display)
                    
                    # Update file_info
                    info_list[idx] = {
                        'type': 'cover',  # Treat as cover type since it's in images folder
                        'file': base_name + ext if match and original_found else html_base,
                        'path': os.path.join(images_dir, base_name + ext if match and original_found else html_base),
                        'hash_key': None,
                        'output_dir': output_dir
                    }
                    refresh_data['file_info'] = info_list
                    
                    moved_count += 1
                    
                except Exception as e:
                    print(f"Failed to process {item['file']}: {e}")
                    failed_count += 1
            
            # Save updated progress if modified
            if progress_data_current:
                try:
                    with open(progress_file, 'w', encoding='utf-8') as f:
                        json.dump(progress_data_current, f, ensure_ascii=False, indent=2)
                    print(f"Updated progress tracking file")
                except Exception as e:
                    print(f"Failed to update progress file: {e}")
            
            # Auto-refresh the display to show updated status
            if 'refresh_data' in locals():
                self._refresh_image_folder_data(refresh_data)
            
            # Update selection count
            update_selection_count()
            
            # Show result
            if failed_count > 0:
                self._styled_msgbox(QMessageBox.Warning, dialog, "Partial Success", 
                    f"Moved {moved_count} image(s) to be skipped.\n"
                    f"Failed to process {failed_count} item(s).")
            else:
                self._styled_msgbox(QMessageBox.Information, dialog, "Success", 
                    f"Moved {moved_count} image(s) to the images folder.\n"
                    "They will be skipped in future translations.")
        
        def retranslate_selected():
            selected_items = listbox.selectedItems()
            if not selected_items:
                self._styled_msgbox(QMessageBox.Warning, dialog, "No Selection", "Please select at least one file.")
                return
            
            selected_indices = [listbox.row(item) for item in selected_items]
            info_list = refresh_data.get('file_info', file_info)
            progress_data_current = refresh_data.get('progress_data', progress_data)
            
            # Count types
            translated_count = sum(1 for i in selected_indices if info_list[i]['type'] == 'translated')
            cover_count = sum(1 for i in selected_indices if info_list[i]['type'] == 'cover')
            
            # Build confirmation message
            msg_parts = []
            if translated_count > 0:
                msg_parts.append(f"{translated_count} translated image(s)")
            if cover_count > 0:
                msg_parts.append(f"{cover_count} cover image(s)")
            
            confirm_msg = f"This will delete {' and '.join(msg_parts)}.\n\nContinue?"
            
            reply = self._styled_msgbox(QMessageBox.Question, dialog, "Confirm Deletion", confirm_msg,
                                       QMessageBox.Yes | QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
            
            # Delete selected files
            deleted_count = 0
            
            for idx in selected_indices:
                info = info_list[idx]
                try:
                    if os.path.exists(info['path']):
                        os.remove(info['path'])
                        deleted_count += 1
                        print(f"Deleted: {info['path']}")
                        
                        # Remove from progress tracking if applicable
                        if progress_data_current and info.get('hash_key'):
                            hash_key = info['hash_key']
                            # Check nested structure first
                            if 'images' in progress_data_current and hash_key in progress_data_current['images']:
                                del progress_data_current['images'][hash_key]
                                print(f"Removed {hash_key} from progress_data['images']")
                            # Check flat structure
                            elif hash_key in progress_data_current:
                                del progress_data_current[hash_key]
                                print(f"Removed {hash_key} from progress_data")
                            
                except Exception as e:
                    print(f"Failed to delete {info['path']}: {e}")
            
            # ALWAYS save progress file after any deletions
            if deleted_count > 0 and progress_data_current:
                try:
                    with open(progress_file, 'w', encoding='utf-8') as f:
                        json.dump(progress_data_current, f, ensure_ascii=False, indent=2)
                    print(f"Updated progress tracking file")
                except Exception as e:
                    print(f"Failed to update progress file: {e}")
            
            # Auto-refresh the display to show updated status
            if 'refresh_data' in locals():
                self._refresh_image_folder_data(refresh_data)
            
            self._styled_msgbox(QMessageBox.Information, dialog, "Success", 
                f"Deleted {deleted_count} file(s).\n\n"
                "They will be retranslated on the next run.")
            
            dialog.close()
        
        # Add buttons in grid layout (similar to EPUB/text retranslation)
        # Row 0: Selection buttons
        btn_select_all = QPushButton("Select All")
        btn_select_all.setStyleSheet("QPushButton { background-color: #17a2b8; color: white; padding: 5px 15px; font-weight: bold; }")
        btn_select_all.clicked.connect(select_all)
        button_layout.addWidget(btn_select_all, 0, 0)
        
        btn_clear_selection = QPushButton("Clear Selection")
        btn_clear_selection.setStyleSheet("QPushButton { background-color: #6c757d; color: white; padding: 5px 15px; font-weight: bold; }")
        btn_clear_selection.clicked.connect(clear_selection)
        button_layout.addWidget(btn_clear_selection, 0, 1)
        
        btn_select_translated = QPushButton("Select Translated")
        btn_select_translated.setStyleSheet("QPushButton { background-color: #28a745; color: white; padding: 5px 15px; font-weight: bold; }")
        btn_select_translated.clicked.connect(select_translated)
        button_layout.addWidget(btn_select_translated, 0, 2)
        
        btn_mark_skipped = QPushButton("Mark as Skipped")
        btn_mark_skipped.setStyleSheet("QPushButton { background-color: #e0a800; color: white; padding: 5px 15px; font-weight: bold; }")
        btn_mark_skipped.clicked.connect(mark_as_skipped)
        button_layout.addWidget(btn_mark_skipped, 0, 3)
        
        # Row 1: Action buttons
        btn_delete = QPushButton("Delete Selected")
        btn_delete.setStyleSheet("QPushButton { background-color: #dc3545; color: white; padding: 5px 15px; font-weight: bold; }")
        btn_delete.clicked.connect(retranslate_selected)
        button_layout.addWidget(btn_delete, 1, 0, 1, 1)
        
        # Add animated refresh button
        btn_refresh = AnimatedRefreshButton("  Refresh")  # Double space for icon padding
        btn_refresh.setStyleSheet(
            "QPushButton { "
            "background-color: #17a2b8; "
            "color: white; "
            "padding: 5px 15px; "
            "font-weight: bold; "
            "}"
            "QPushButton[refreshActive=\"true\"] { "
            "background-color: #138496; "
            "}"
        )
        
        # Create data dict for refresh function
        refresh_data = {
            'type': 'image_folder',
            'listbox': listbox,
            'file_info': file_info,
            'progress_file': progress_file,
            'progress_data': progress_data,
            'output_dir': output_dir,
            'folder_path': folder_path,
            'selection_count_label': selection_count_label,
            'dialog': dialog
        }
        
        # Create refresh handler with animation
        def animated_refresh():
            import time
            btn_refresh.start_animation()
            btn_refresh.setEnabled(False)
            
            # Track start time for minimum animation duration
            start_time = time.time()
            min_animation_duration = 0.8  # 800ms minimum
            
            # Use QTimer to run refresh after animation starts
            def do_refresh():
                try:
                    self._refresh_image_folder_data(refresh_data)
                    
                    # Calculate remaining time to meet minimum animation duration
                    elapsed = time.time() - start_time
                    remaining = max(0, min_animation_duration - elapsed)
                    
                    # Schedule animation stop after remaining time
                    def finish_animation():
                        btn_refresh.stop_animation()
                        btn_refresh.setEnabled(True)
                    
                    if remaining > 0:
                        QTimer.singleShot(int(remaining * 1000), finish_animation)
                    else:
                        finish_animation()
                        
                except Exception as e:
                    print(f"Error during refresh: {e}")
                    btn_refresh.stop_animation()
                    btn_refresh.setEnabled(True)
            
            QTimer.singleShot(50, do_refresh)  # Small delay to let animation start
        
        btn_refresh.clicked.connect(animated_refresh)
        button_layout.addWidget(btn_refresh, 1, 1, 1, 1)
        # Store for reuse-trigger
        dialog._refresh_button = btn_refresh

        # Auto-refresh every 3 seconds (silent, no animation)
        def _silent_refresh_images():
            try:
                # Skip if a manual refresh is already in progress
                if not btn_refresh.isEnabled():
                    return
                if dialog.isVisible():
                    self._refresh_image_folder_data(refresh_data)
            except Exception:
                pass

        _auto_refresh_timer = QTimer(dialog)
        _auto_refresh_timer.setInterval(2000)
        _auto_refresh_timer.timeout.connect(_silent_refresh_images)
        _auto_refresh_timer.start()
        dialog._auto_refresh_timer = _auto_refresh_timer
        
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setStyleSheet("QPushButton { background-color: #6c757d; color: white; padding: 5px 15px; font-weight: bold; }")
        btn_cancel.clicked.connect(dialog.close)
        button_layout.addWidget(btn_cancel, 1, 2, 1, 2)
        
        # Override close event to hide instead of destroy
        def closeEvent(event):
            event.ignore()  # Ignore the close event
            dialog.hide()   # Just hide the dialog
        
        dialog.closeEvent = closeEvent
        
        # Cache the dialog for reuse
        if not hasattr(self, '_image_retranslation_dialog_cache'):
            self._image_retranslation_dialog_cache = {}
        
        folder_key = os.path.abspath(folder_path)
        self._image_retranslation_dialog_cache[folder_key] = dialog
        
        # Programmatically click the Refresh button once on open to ensure latest data (fires same slot)
        QTimer.singleShot(0, btn_refresh.click)

        # Show the dialog (non-modal to allow interaction with other windows)
        dialog.show()
