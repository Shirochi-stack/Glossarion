
from safe_image import open_image
import sys
import os
import json
import threading
import time
import hashlib
import traceback
import concurrent.futures
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from PySide6.QtWidgets import (QWidget, QLabel, QFrame, QPushButton, QVBoxLayout, QHBoxLayout,
                               QGroupBox, QListWidget, QComboBox, QLineEdit, QCheckBox,
                               QRadioButton, QSlider, QSpinBox, QDoubleSpinBox, QTextEdit,
                               QProgressBar, QFileDialog, QMessageBox, QColorDialog, QScrollArea,
                               QDialog, QButtonGroup, QApplication, QSizePolicy, QToolButton)
from PySide6.QtCore import Qt, QTimer, Signal, QObject, Slot, QEvent, QPropertyAnimation, QEasingCurve, Property, QThread
from PySide6.QtGui import QFont, QColor, QTextCharFormat, QIcon, QKeyEvent, QPixmap, QTransform
from typing import List, Dict, Optional, Any
from queue import Queue, Empty
import logging
from manga_translator import MangaTranslator, GOOGLE_CLOUD_VISION_AVAILABLE
from manga_settings_dialog import MangaSettingsDialog
import manga_ocr_io
import manga_editor_core

# U8: the GUI-free half of the manga editor (Detect / Clean / Recognize / Translate / Translate All,
# the per-box actions, Save & Update Overlay re-render, per-image state) lives in manga_editor_core,
# shared with Glossarion Mobile. The desktop runs those same functions re-bound to this module's
# namespace: they keep calling the Qt helpers defined here and stay patchable as ImageRenderer.<name>.
manga_editor_core.bind_editor_namespace(globals())


def _apply_rectangle_clean_style(rect_item, *, is_recognized=False, excluded=False):
    try:
        from PySide6.QtGui import QPen, QBrush, QColor
        if excluded:
            pen = QPen(QColor(255, 140, 0), 3)
            brush = QBrush(QColor(255, 140, 0, 30))
        elif is_recognized:
            pen = QPen(QColor(0, 150, 255), 2)
            brush = QBrush(QColor(0, 150, 255, 50))
        elif _is_free_text_region_metadata(rect_item=rect_item):
            pen = QPen(QColor(0, 220, 180), 2)
            brush = QBrush(QColor(0, 220, 180, 45))
        else:
            pen = QPen(QColor(0, 255, 0), 2)
            brush = QBrush(QColor(0, 255, 0, 50))
        try:
            pen.setCosmetic(True)
        except Exception:
            pass
        rect_item.setPen(pen)
        rect_item.setBrush(brush)
    except Exception:
        pass

# Desktop halves of the Qt lines the shared editor functions hand off (U8). manga_editor_core
# holds GUI-free stubs of the same names, which the mobile editor session answers.
def _style_recognized_rectangle(rect_item):
    """Colour a rectangle blue once its text is recognized (Qt half of the shared OCR handlers)."""
    from PySide6.QtGui import QPen, QBrush, QColor
    rect_item.setPen(QPen(QColor(0, 150, 255), 2))  # Blue border
    rect_item.setBrush(QBrush(QColor(0, 150, 255, 50)))  # Semi-transparent blue fill


def _confirm_clean_excluded_rectangle(self, region_index):
    """Ask before cleaning a rectangle that is excluded from inpainting (Qt half of Clean This Rectangle)."""
    from PySide6.QtWidgets import QMessageBox
    reply = QMessageBox.question(
        self.dialog,
        "Rectangle Excluded",
        f"Rectangle {region_index} is currently excluded from inpainting.\n\nDo you want to clean it anyway?",
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.No
    )
    if reply == QMessageBox.No:
        print(f"[CLEAN_RECT] User cancelled - rectangle {region_index} is excluded")
        return False
    return True


def _schedule_rendered_output_refresh(self, rendered_pil, output_path, switch_tab):
    """Load a rendered page into the Output tab on the GUI thread (Qt half of _render_with_manga_translator)."""
    from PySide6.QtCore import QTimer
    QTimer.singleShot(0, lambda: _load_rendered_image_to_output_tab(self, rendered_pil, output_path, switch_tab))

# Optional: psutil/ctypes helpers to reduce GUI lag by lowering background thread priority
try:
    import psutil  # type: ignore
except Exception:
    psutil = None

import ctypes
import platform

_IS_WINDOWS = platform.system().lower().startswith('win')

# Windows thread priority constants
if _IS_WINDOWS:
    _THREAD_PRIORITY_IDLE = -15
    _THREAD_PRIORITY_LOWEST = -2
    _THREAD_PRIORITY_BELOW_NORMAL = -1
    _THREAD_PRIORITY_NORMAL = 0
    _THREAD_SET_INFORMATION = 0x0020
    _THREAD_QUERY_INFORMATION = 0x0040

    _kernel32 = ctypes.windll.kernel32

    def _win_set_current_thread_priority(level=_THREAD_PRIORITY_BELOW_NORMAL):
        try:
            _kernel32.SetThreadPriority(_kernel32.GetCurrentThread(), ctypes.c_int(level))
        except Exception:
            pass

    def _win_set_current_thread_affinity(reserve_cores: int = 1):
        try:
            # Build an affinity mask that leaves some low-index cores free for the GUI
            cpu_count = psutil.cpu_count(logical=True) if psutil else os.cpu_count() or 1
            allow = max(1, cpu_count - max(0, reserve_cores))
            mask = 0
            for i in range(allow):
                mask |= (1 << i)
            _kernel32.SetThreadAffinityMask(_kernel32.GetCurrentThread(), ctypes.c_size_t(mask))
        except Exception:
            pass

    def _win_set_thread_priority_by_tid(tid: int, level=_THREAD_PRIORITY_BELOW_NORMAL):
        try:
            handle = _kernel32.OpenThread(_THREAD_SET_INFORMATION | _THREAD_QUERY_INFORMATION, False, ctypes.c_uint32(tid))
            if handle:
                _kernel32.SetThreadPriority(handle, ctypes.c_int(level))
                _kernel32.CloseHandle(handle)
        except Exception:
            pass

    def _win_set_thread_affinity_by_tid(tid: int, reserve_cores: int = 1):
        try:
            handle = _kernel32.OpenThread(_THREAD_SET_INFORMATION | _THREAD_QUERY_INFORMATION, False, ctypes.c_uint32(tid))
            if handle:
                cpu_count = psutil.cpu_count(logical=True) if psutil else os.cpu_count() or 1
                allow = max(1, cpu_count - max(0, reserve_cores))
                mask = 0
                for i in range(allow):
                    mask |= (1 << i)
                _kernel32.SetThreadAffinityMask(handle, ctypes.c_size_t(mask))
                _kernel32.CloseHandle(handle)
        except Exception:
            pass

    def _lower_current_thread_priority_and_affinity(reserve_env_key: str = 'MANGA_RESERVE_CORES'):
        try:
            reserve = 1
            try:
                reserve = int(os.environ.get(reserve_env_key, '1'))
            except Exception:
                reserve = 1
            _win_set_current_thread_priority(_THREAD_PRIORITY_BELOW_NORMAL)
            _win_set_current_thread_affinity(reserve)
        except Exception:
            pass

    def _demote_non_main_threads(main_tid: int, reserve_env_key: str = 'MANGA_RESERVE_CORES'):
        try:
            reserve = 1
            try:
                reserve = int(os.environ.get(reserve_env_key, '1'))
            except Exception:
                reserve = 1
            if psutil is None:
                return
            p = psutil.Process()
            for th in p.threads():
                tid = int(th.id)
                if tid != main_tid:
                    _win_set_thread_priority_by_tid(tid, _THREAD_PRIORITY_BELOW_NORMAL)
                    _win_set_thread_affinity_by_tid(tid, reserve)
        except Exception:
            pass
else:
    def _lower_current_thread_priority_and_affinity(reserve_env_key: str = 'MANGA_RESERVE_CORES'):
        # Non-Windows: no-op (per-thread priority not easily portable without extra deps)
        return

    def _demote_non_main_threads(main_tid: int, reserve_env_key: str = 'MANGA_RESERVE_CORES'):
        return

# Try to import UnifiedClient for API initialization
try:
    from unified_api_client import UnifiedClient
except ImportError:
    UnifiedClient = None

# MODULE-LEVEL WORKER FUNCTION for parallel save processing (pickleable)
def _process_save_task_worker(region_index: int, current_image: str, trans_text: str, task: dict) -> dict:
    """Worker function for ProcessPoolExecutor - processes a single save task.
    
    This function is pickleable and runs in a separate process.
    It cannot access GUI state or non-pickleable objects.
    
    Args:
        region_index: Index of the region to process
        current_image: Path to the current image
        trans_text: Translation text for the region
        task: Task metadata dict
        
    Returns:
        dict with 'success' bool and optional 'error' string
    """
    import time
    start_time = time.time_ns()
    
    try:
        print(f"[PARALLEL_WORKER] Processing region {region_index} in process")
        
        # Since we can't access GUI state from a worker process,
        # we just return success to indicate the task was received
        # The actual GUI update will happen in the completion callback
        
        end_time = time.time_ns()
        duration_ms = (end_time - start_time) / 1_000_000
        
        print(f"[PARALLEL_WORKER] Region {region_index} processed in {duration_ms:.2f}ms")
        
        return {
            'success': True,
            'region_index': region_index,
            'duration_ms': duration_ms
        }
        
    except Exception as e:
        print(f"[PARALLEL_WORKER] Error processing region {region_index}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'region_index': region_index,
            'error': str(e)
        }

# MODULE-LEVEL RENDER FUNCTION (pickle-able for ProcessPoolExecutor)
def _render_single_region_overlay(region_data: dict, image_size: tuple, render_settings: dict) -> Image.Image:
    """
    Render a single region overlay as RGBA PIL Image (pickle-able for multiprocessing)
    
    Args:
        region_data: dict with 'text', 'bbox' (x,y,w,h), 'vertices'
        image_size: (width, height)
        render_settings: dict with font/color/outline settings
    
    Returns:
        PIL RGBA Image of full size with transparent overlay
    """
    try:
        # Create transparent overlay
        overlay = Image.new('RGBA', image_size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Extract data
        text = region_data.get('translated_text', '')
        if not text:
            return overlay
        
        x, y, w, h = region_data.get('bbox', (0, 0, 100, 100))
        
        # Get settings
        font_size = render_settings.get('font_size', 24)
        font_path = render_settings.get('font_path')
        text_color = tuple(render_settings.get('text_color', (102, 0, 0))) + (255,)
        outline_color = tuple(render_settings.get('outline_color', (255, 255, 255))) + (255,)
        outline_width = render_settings.get('outline_width', 2)
        force_caps = render_settings.get('force_caps_lock', False)
        
        if force_caps:
            text = text.upper()
        
        # Load font
        try:
            if font_path and os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
            else:
                font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()
        
        # Simple text wrapping
        lines = text.split('\n')
        line_height = int(font_size * 1.2)
        total_height = len(lines) * line_height
        start_y = y + (h - total_height) // 2
        
        # Render each line
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            
            # Get text width
            try:
                bbox = draw.textbbox((0, 0), line, font=font)
                text_width = bbox[2] - bbox[0]
            except Exception:
                text_width = len(line) * font_size * 0.6
            
            tx = x + (w - text_width) // 2
            ty = start_y + i * line_height
            
            # Clamp to bounds
            tx = max(0, min(tx, image_size[0] - 10))
            ty = max(0, min(ty, image_size[1] - 10))
            
            # Render with outline using PIL stroke parameter
            try:
                draw.text(
                    (tx, ty), line, font=font,
                    fill=text_color,
                    stroke_width=outline_width,
                    stroke_fill=outline_color
                )
            except TypeError:
                # Fallback for older PIL
                if outline_width > 0:
                    for dx in range(-outline_width, outline_width + 1):
                        for dy in range(-outline_width, outline_width + 1):
                            if dx != 0 or dy != 0:
                                draw.text((tx + dx, ty + dy), line, font=font, fill=outline_color)
                draw.text((tx, ty), line, font=font, fill=text_color)
        
        return overlay
    except Exception as e:
        print(f"[RENDER] Error rendering region: {e}")
        return Image.new('RGBA', image_size, (0, 0, 0, 0))
    
def _restore_detect_button(self):
    """Restore the detect button to its original state"""
    try:
        # Remove processing overlay effect for the image that was being processed
        image_path = getattr(self, '_detection_started_for_image', None)
        _remove_processing_overlay(self, image_path)
        
        if hasattr(self, 'image_preview_widget') and hasattr(self.image_preview_widget, 'detect_btn'):
            self.image_preview_widget.detect_btn.setEnabled(True)
            self.image_preview_widget.detect_btn.setText("Detect Text")
    except Exception:
        pass

def _restore_image_state(self, image_path: str):
    """Restore persisted state for an image (rectangles, overlays, paths)"""
    print = _manga_debug_print
    try:
        if not hasattr(self, 'image_state_manager'):
            return
        
        # CRITICAL: Always reset path tracking variables to current image to prevent
        # "Cannot render: Translation data is for X but you're viewing Y" errors
        # This must happen BEFORE we check for saved state
        self._translation_data_image_path = image_path
        self._translating_image_path = image_path
        self._current_state_image_path = image_path
        print(f"[STATE_ISOLATION] Reset path tracking to: {os.path.basename(image_path)}")
        
        # CRITICAL: Validate and clean stale state BEFORE restoration
        _validate_and_clean_stale_state(self, image_path)
        
        # Get saved state (after cleaning)
        state = self.image_state_manager.get_state(image_path)
        if not state:
            print(f"[STATE] No saved state for {os.path.basename(image_path)}")
            return
        
        print(f"[STATE] Restoring state for {os.path.basename(image_path)}")
        
        # Prefer viewer_rectangles (latest manual adjustments) to restore boxes; fallback to detection_regions
        used_boxes = False
        if 'viewer_rectangles' in state and state['viewer_rectangles']:
            viewer = self.image_preview_widget.viewer
            if hasattr(viewer, 'clear_rectangles'):
                viewer.clear_rectangles()
            from PySide6.QtCore import QRectF, Qt
            from PySide6.QtGui import QPen, QBrush, QColor, QPainterPath
            from manga_image_preview import MoveableRectItem, MoveableEllipseItem, MoveablePathItem

            # Determine coloring based on recognized/translated texts
            recognized_texts = state.get('recognized_texts', [])
            translated_texts = state.get('translated_texts', [])
            has_any_text = bool([t for t in (recognized_texts or []) if not (isinstance(t, dict) and t.get('deleted'))]) or \
                           bool([t for t in (translated_texts or []) if not (isinstance(t, dict) and t.get('deleted'))])

            for idx, rect_data in enumerate(state['viewer_rectangles']):
                shape = rect_data.get('shape', 'rect')
                rect = QRectF(rect_data['x'], rect_data['y'], rect_data['width'], rect_data['height'])

                # Blue if this rectangle has recognized/translated text, else green
                has_text_for_this_rect = False
                if has_any_text:
                    has_recognized = (idx < len(recognized_texts) and not (isinstance(recognized_texts[idx], dict) and recognized_texts[idx].get('deleted')))
                    has_translation = (idx < len(translated_texts) and not (isinstance(translated_texts[idx], dict) and translated_texts[idx].get('deleted')))
                    has_text_for_this_rect = has_recognized or has_translation

                if has_text_for_this_rect:
                    pen = QPen(QColor(0, 150, 255), 2)
                    brush = QBrush(QColor(0, 150, 255, 50))
                else:
                    pen = QPen(QColor(0, 255, 0), 1)
                    brush = QBrush(QColor(0, 255, 0, 50))
                pen.setCosmetic(True)

                if shape == 'ellipse':
                    item = MoveableEllipseItem(rect, pen=pen, brush=brush)
                elif shape == 'polygon' and rect_data.get('polygon'):
                    path = QPainterPath()
                    pts = rect_data.get('polygon') or []
                    if pts:
                        path.moveTo(pts[0][0], pts[0][1])
                        for px, py in pts[1:]:
                            path.lineTo(px, py)
                        path.closeSubpath()
                    item = MoveablePathItem(path, pen=pen, brush=brush)
                else:
                    item = MoveableRectItem(rect, pen=pen, brush=brush)
                metadata = _saved_rect_metadata(state, idx, rect_data)
                _copy_region_metadata_to_item(item, metadata)
                if not has_text_for_this_rect:
                    _apply_rectangle_clean_style(item, is_recognized=False, excluded=False)

                # Attach viewer and metadata
                try:
                    item._viewer = viewer
                    item.region_index = idx
                    # CRITICAL: Always set is_recognized based on text state, not just when true
                    item.is_recognized = has_text_for_this_rect
                    # Attach move sync and context menu
                    _attach_move_sync_to_rectangle(self, item, idx)
                    _add_context_menu_to_rectangle(self, item, idx)
                except Exception:
                    pass

                viewer._scene.addItem(item)
                viewer.rectangles.append(item)
            print(f"[STATE] Restored {len(state['viewer_rectangles'])} viewer shapes (preferred)")
            
            # CRITICAL: Rehydrate text state so OCR/translation data is available in memory
            ocr_count, trans_count = _rehydrate_text_state_from_persisted(self, image_path)
            if ocr_count and hasattr(self, '_update_rectangles_with_recognition'):
                try:
                    _update_rectangles_with_recognition(self, self._recognized_texts)
                    print(f"[STATE] Attached OCR tooltips to {ocr_count} rectangles (viewer_rectangles branch)")
                except Exception:
                    pass
            used_boxes = True
        
        if not used_boxes and 'detection_regions' in state:
            self._current_regions = state['detection_regions']
            print(f"[STATE] Restored {len(self._current_regions)} detection regions")
            # Redraw detection boxes on preview
            if hasattr(self.image_preview_widget.viewer, 'clear_rectangles'):
                self.image_preview_widget.viewer.clear_rectangles()
            _draw_detection_boxes_on_preview(self, )

            # If recognized/translated texts exist, promote matching rectangles to BLUE and attach metadata
            try:
                viewer = self.image_preview_widget.viewer
                rects = getattr(viewer, 'rectangles', []) or []
                recognized_texts = state.get('recognized_texts') or []
                translated_texts = state.get('translated_texts') or []
                max_len = max(len(recognized_texts), len(translated_texts))
                if max_len and rects:
                    from PySide6.QtGui import QPen, QBrush, QColor
                    for idx, rect_item in enumerate(rects):
                        has_recognized = (idx < len(recognized_texts) and not (isinstance(recognized_texts[idx], dict) and recognized_texts[idx].get('deleted')))
                        has_translation = (idx < len(translated_texts) and not (isinstance(translated_texts[idx], dict) and translated_texts[idx].get('deleted')))
                        has_text = has_recognized or has_translation
                        if has_text:
                            # Make blue and mark recognized
                            try:
                                rect_item.setPen(QPen(QColor(0, 150, 255), 2))
                                rect_item.setBrush(QBrush(QColor(0, 150, 255, 50)))
                            except Exception:
                                pass
                        # CRITICAL: Always set is_recognized based on text state, not just when true
                        try:
                            rect_item.is_recognized = has_text
                        except Exception:
                            pass
                        # Always ensure region_index and context menu are attached
                        try:
                            rect_item.region_index = idx
                            _attach_move_sync_to_rectangle(self, rect_item, idx)
                            _add_context_menu_to_rectangle(self, rect_item, idx)
                        except Exception:
                            pass
                    print(f"[STATE] Promoted {sum(1 for i in range(min(len(rects), max_len)) if (i < len(recognized_texts) and not (isinstance(recognized_texts[i], dict) and recognized_texts[i].get('deleted'))) or (i < len(translated_texts) and not (isinstance(translated_texts[i], dict) and translated_texts[i].get('deleted'))))} rectangles to BLUE from recognized/translated texts")
                # Populate in-memory text state and add OCR tooltips
                ocr_count, trans_count = _rehydrate_text_state_from_persisted(self, image_path)
                if ocr_count and hasattr(self, '_update_rectangles_with_recognition'):
                    try:
                        _update_rectangles_with_recognition(self, self._recognized_texts)
                        print(f"[STATE] Attached OCR tooltips to {ocr_count} rectangles (detection branch)")
                    except Exception:
                        pass
            except Exception:
                pass
        
        # Restore exclusion status after rectangles are drawn
        try:
            print(f"[RECT_0_DEBUG] About to call _restore_exclusion_status_from_state for: {image_path}")
            _restore_exclusion_status_from_state(self, image_path)
        except Exception as e:
            print(f"[STATE] Failed to restore exclusion status: {e}")
        
        # Restore custom inpainting iterations after rectangles are drawn
        try:
            _restore_inpainting_iterations_from_state(self, image_path)
        except Exception as e:
            print(f"[STATE] Failed to restore inpainting iterations: {e}")
        
        # Restore overlay rectangles
        if 'overlay_rects' in state and state['overlay_rects']:
            if hasattr(self.image_preview_widget.viewer, 'overlay_rects'):
                self.image_preview_widget.viewer.overlay_rects = state['overlay_rects'].copy()
                print(f"[STATE] Restored {len(state['overlay_rects'])} overlay rects")
        
        # Restore cleaned image path (with validation + filesystem discovery)
        resolved_cleaned = _resolve_cleaned_image_for_render(self, image_path)
        if resolved_cleaned:
            self._cleaned_image_path = resolved_cleaned
            print(f"[STATE] Restored cleaned image path: {os.path.basename(self._cleaned_image_path)}")
        elif 'cleaned_image_path' in state:
            print(f"[STATE] Skipped corrupted cleaned_image_path: {os.path.basename(state['cleaned_image_path'])}")
        
        # Restore rendered image path mapping if available
        if 'rendered_image_path' in state:
            rendered_path = state['rendered_image_path']
            if os.path.exists(rendered_path):
                # Store the translated path for reference
                if hasattr(self.image_preview_widget, 'current_translated_path'):
                    self.image_preview_widget.current_translated_path = rendered_path
                print(f"[STATE] Restored rendered image path reference: {os.path.basename(rendered_path)}")
                
                # Store mapping
                if not hasattr(self, '_rendered_images_map'):
                    self._rendered_images_map = {}
                self._rendered_images_map[image_path] = rendered_path
        
        # Restore viewer rectangles (if no detection regions were restored)
        if 'viewer_rectangles' in state and not ('detection_regions' in state):
            viewer = self.image_preview_widget.viewer
            if hasattr(viewer, 'clear_rectangles'):
                viewer.clear_rectangles()
            
            from PySide6.QtCore import QRectF, Qt
            from PySide6.QtGui import QPen, QBrush, QColor, QPainterPath
            from manga_image_preview import MoveableRectItem, MoveableEllipseItem, MoveablePathItem
            
            # Check for recognized/translated texts to determine rectangle colors
            recognized_texts = state.get('recognized_texts', [])
            translated_texts = state.get('translated_texts', [])
            
            for idx, rect_data in enumerate(state['viewer_rectangles']):
                shape = rect_data.get('shape', 'rect')
                rect = QRectF(rect_data['x'], rect_data['y'], rect_data['width'], rect_data['height'])
                
                # Check if this rectangle has OCR/translation data
                has_recognized = (idx < len(recognized_texts) and 
                                 not (isinstance(recognized_texts[idx], dict) and recognized_texts[idx].get('deleted')) and
                                 (isinstance(recognized_texts[idx], str) and recognized_texts[idx].strip() or
                                  isinstance(recognized_texts[idx], dict) and recognized_texts[idx].get('text', '').strip()))
                has_translation = (idx < len(translated_texts) and 
                                  not (isinstance(translated_texts[idx], dict) and translated_texts[idx].get('deleted')) and
                                  isinstance(translated_texts[idx], dict) and translated_texts[idx].get('translation', '').strip())
                has_text = has_recognized or has_translation
                
                # Use blue for recognized/translated, green for detection-only
                if has_text:
                    pen = QPen(QColor(0, 150, 255), 2)
                    brush = QBrush(QColor(0, 150, 255, 50))
                else:
                    pen = QPen(QColor(0, 255, 0), 1)
                    brush = QBrush(QColor(0, 255, 0, 50))
                pen.setCosmetic(True)
                
                if shape == 'ellipse':
                    item = MoveableEllipseItem(rect, pen=pen, brush=brush)
                elif shape == 'polygon' and rect_data.get('polygon'):
                    path = QPainterPath()
                    pts = rect_data.get('polygon') or []
                    if pts:
                        path.moveTo(pts[0][0], pts[0][1])
                        for px, py in pts[1:]:
                            path.lineTo(px, py)
                        path.closeSubpath()
                    item = MoveablePathItem(path, pen=pen, brush=brush)
                else:
                    item = MoveableRectItem(rect, pen=pen, brush=brush)
                metadata = _saved_rect_metadata(state, idx, rect_data)
                _copy_region_metadata_to_item(item, metadata)
                if not has_text:
                    _apply_rectangle_clean_style(item, is_recognized=False, excluded=False)
                
                # Attach viewer reference so moved emits
                try:
                    item._viewer = viewer
                except Exception:
                    pass
                viewer._scene.addItem(item)
                viewer.rectangles.append(item)
                
                # Attach region index, is_recognized flag, and handlers
                try:
                    item.region_index = idx
                    item.is_recognized = has_text  # CRITICAL: Set is_recognized flag
                    _attach_move_sync_to_rectangle(self, item, idx)
                    # Add context menu to restored rectangles
                    _add_context_menu_to_rectangle(self, item, idx)
                except Exception:
                    pass
            
            # Rehydrate in-memory text state for this path as well
            try:
                ocr_count, trans_count = _rehydrate_text_state_from_persisted(self, image_path)
                print(f"[STATE] Rehydrated {ocr_count} OCR and {trans_count} translation entries (viewer_rectangles branch)")
            except Exception:
                pass
            
            print(f"[STATE] Restored {len(state['viewer_rectangles'])} viewer shapes")
        
        print(f"[STATE] State restoration complete for {os.path.basename(image_path)}")
        
        # CRITICAL: Comprehensive graphics scene and overlay synchronization
        # This fixes the cosmetic issue where overlays appear disconnected until user interaction
        try:
            from PySide6.QtCore import QTimer
            viewer = self.image_preview_widget.viewer
            
            def comprehensive_refresh():
                try:
                    # 1. Force complete scene update
                    viewer._scene.update()
                    viewer.update()
                    viewer.repaint()
                    
                    # 2. Ensure text overlays are visible and properly positioned
                    if hasattr(self, 'show_text_overlays_for_image'):
                        self.show_text_overlays_for_image(image_path)
                    
                    # 3. Force overlay position synchronization with rectangles
                    try:
                        _synchronize_overlay_positions_with_rectangles(self, image_path)
                    except Exception:
                        pass
                    
                    # 4. Final scene update to reflect changes
                    viewer._scene.update()
                    viewer.viewport().update()
                    
                    _manga_debug_print(f"[STATE] Comprehensive refresh completed for {os.path.basename(image_path)}")
                except Exception as e:
                    print(f"[STATE] Comprehensive refresh failed: {e}")
            
            # Schedule multiple refreshes with increasing delays for maximum reliability
            QTimer.singleShot(10, comprehensive_refresh)
            QTimer.singleShot(100, comprehensive_refresh)
            QTimer.singleShot(250, comprehensive_refresh)
        except Exception:
            pass
        
    except Exception as e:
        print(f"[STATE] Failed to restore state: {e}")
        import traceback
        traceback.print_exc()

def _restore_image_state_overlays_only(self, image_path: str):
    """Restore ONLY rectangles/overlays for an image, without loading images
    
    This is used after manually loading the correct image to avoid double-loading.
    
    STATE ISOLATION: Validates that we're restoring the correct image's state.
    """
    print = _manga_debug_print
    try:
        if not hasattr(self, 'image_state_manager'):
            return
        
        # STATE ISOLATION: Verify we don't have stale state from another image
        if hasattr(self, '_current_state_image_path') and self._current_state_image_path:
            if self._current_state_image_path != image_path:
                print(f"[STATE_ISOLATION] WARNING: Stale state detected! Current={os.path.basename(self._current_state_image_path)}, Requested={os.path.basename(image_path)}")
                print(f"[STATE_ISOLATION] Clearing stale state before restoration")
                _clear_cross_image_state(self)
        
        # CRITICAL: Validate and clean stale state BEFORE restoration
        _validate_and_clean_stale_state(self, image_path)
        
        # Get saved state (after cleaning)
        state = self.image_state_manager.get_state(image_path)
        if not state:
            print(f"[STATE] No saved state for {os.path.basename(image_path)}")
            return
        
        print(f"[STATE] Restoring overlays for {os.path.basename(image_path)}")
        
        # Prefer viewer_rectangles (latest manual adjustments) to restore boxes; fallback to detection_regions
        used_boxes = False
        if 'viewer_rectangles' in state and state['viewer_rectangles']:
            viewer = self.image_preview_widget.viewer
            
            # STATE ISOLATION: Track rectangle count before and after clearing
            rect_count_before_clear = len(viewer.rectangles) if hasattr(viewer, 'rectangles') else 0
            print(f"[STATE_ISOLATION] Rectangle count BEFORE clear: {rect_count_before_clear}")
            
            if hasattr(viewer, 'clear_rectangles'):
                viewer.clear_rectangles()
            
            rect_count_after_clear = len(viewer.rectangles) if hasattr(viewer, 'rectangles') else 0
            print(f"[STATE_ISOLATION] Rectangle count AFTER clear: {rect_count_after_clear}")
            print(f"[STATE_ISOLATION] About to restore {len(state['viewer_rectangles'])} rectangles for {os.path.basename(image_path)}")
            
            from PySide6.QtCore import QRectF, Qt
            from PySide6.QtGui import QPen, QBrush, QColor, QPainterPath
            from manga_image_preview import MoveableRectItem, MoveableEllipseItem, MoveablePathItem
            # Check if there are OCR or translated texts to determine rectangle colors
            recognized_texts = state.get('recognized_texts', [])
            translated_texts = state.get('translated_texts', [])
            has_ocr_text = bool([t for t in recognized_texts if not (isinstance(t, dict) and t.get('deleted'))])
            has_translated_text = bool([t for t in translated_texts if not (isinstance(t, dict) and t.get('deleted'))])
            has_any_text = has_ocr_text or has_translated_text
            
            for idx, rect_data in enumerate(state['viewer_rectangles']):
                rect = QRectF(rect_data['x'], rect_data['y'], rect_data['width'], rect_data['height'])
                
                # Use blue color if this rectangle has any recognized text (OCR or translated), green otherwise
                has_text_for_this_rect = False
                if has_any_text and idx < max(len(recognized_texts), len(translated_texts)):
                    # Check if this specific rectangle has text (either OCR or translated)
                    has_recognized = False
                    if idx < len(recognized_texts):
                        text_entry = recognized_texts[idx]
                        # Check if it's a valid text entry (not deleted, not empty)
                        if isinstance(text_entry, dict):
                            if not text_entry.get('deleted') and text_entry.get('text', '').strip():
                                has_recognized = True
                        elif isinstance(text_entry, str) and text_entry.strip():
                            has_recognized = True
                    
                    has_translation = False
                    if idx < len(translated_texts):
                        trans_entry = translated_texts[idx]
                        # Check if it's a valid translation entry (not deleted, not empty)
                        if isinstance(trans_entry, dict):
                            if not trans_entry.get('deleted') and trans_entry.get('translation', '').strip():
                                has_translation = True
                    
                    has_text_for_this_rect = has_recognized or has_translation
                    
                    if has_text_for_this_rect:
                        print(f"[STATE] Rectangle {idx} has text - will be BLUE (OCR={has_recognized}, Trans={has_translation})")
                    else:
                        print(f"[STATE] Rectangle {idx} has no valid text - will be GREEN")
                
                if has_text_for_this_rect:
                    # Blue for rectangles with recognized/translated text
                    pen = QPen(QColor(0, 150, 255), 2)
                    brush = QBrush(QColor(0, 150, 255, 50))
                else:
                    # Green for detection-only rectangles
                    pen = QPen(QColor(0, 255, 0), 1)
                    brush = QBrush(QColor(0, 255, 0, 50))
                
                pen.setCosmetic(True)
                shape = rect_data.get('shape', 'rect')
                if shape == 'ellipse':
                    rect_item = MoveableEllipseItem(rect, pen=pen, brush=brush)
                elif shape == 'polygon' and rect_data.get('polygon'):
                    path = QPainterPath()
                    pts = rect_data.get('polygon') or []
                    if pts:
                        path.moveTo(pts[0][0], pts[0][1])
                        for px, py in pts[1:]:
                            path.lineTo(px, py)
                        path.closeSubpath()
                    rect_item = MoveablePathItem(path, pen=pen, brush=brush)
                else:
                    rect_item = MoveableRectItem(rect, pen=pen, brush=brush)
                metadata = _saved_rect_metadata(state, idx, rect_data)
                _copy_region_metadata_to_item(rect_item, metadata)
                if not has_text_for_this_rect:
                    _apply_rectangle_clean_style(rect_item, is_recognized=False, excluded=False)
                # Attach viewer for move signal
                try:
                    rect_item._viewer = viewer
                except Exception:
                    pass
                viewer._scene.addItem(rect_item)
                viewer.rectangles.append(rect_item)
                
                # Add context menu to restored rectangles
                try:
                    rect_item.region_index = idx
                    # CRITICAL: Always set is_recognized based on text state, not just when true
                    rect_item.is_recognized = has_text_for_this_rect
                    _attach_move_sync_to_rectangle(self, rect_item, idx)
                    _add_context_menu_to_rectangle(self, rect_item, idx)
                except Exception:
                    pass
                    
            print(f"[STATE] Restored {len(state['viewer_rectangles'])} viewer shapes (preferred)")
            # Populate in-memory text state and add OCR tooltips
            try:
                ocr_count, trans_count = _rehydrate_text_state_from_persisted(self, image_path)
                if ocr_count and hasattr(self, '_update_rectangles_with_recognition'):
                    _update_rectangles_with_recognition(self, self._recognized_texts)
                    print(f"[STATE] Attached OCR tooltips to {ocr_count} rectangles (viewer_rectangles branch)")
            except Exception:
                pass
            used_boxes = True
        
        if not used_boxes and 'detection_regions' in state:
            self._current_regions = state['detection_regions']
            print(f"[STATE] Restored {len(self._current_regions)} detection regions")
            # Redraw detection boxes on preview
            if hasattr(self.image_preview_widget.viewer, 'clear_rectangles'):
                self.image_preview_widget.viewer.clear_rectangles()
            _draw_detection_boxes_on_preview(self, )
        
        # Restore overlay rectangles
        if 'overlay_rects' in state and state['overlay_rects']:
            if hasattr(self.image_preview_widget.viewer, 'overlay_rects'):
                self.image_preview_widget.viewer.overlay_rects = state['overlay_rects'].copy()
                print(f"[STATE] Restored {len(state['overlay_rects'])} overlay rects")
        
        # Restore cleaned image path reference (with validation + filesystem discovery)
        resolved_cleaned = _resolve_cleaned_image_for_render(self, image_path)
        if resolved_cleaned:
            self._cleaned_image_path = resolved_cleaned
            print(f"[STATE] Restored cleaned image path: {os.path.basename(self._cleaned_image_path)}")
        elif 'cleaned_image_path' in state:
            print(f"[STATE] Skipped corrupted cleaned_image_path: {os.path.basename(state['cleaned_image_path'])}")
        
        # Restore recognition_data from persisted recognized_texts so Edit OCR menu works after reload
        try:
            recognized_texts = state.get('recognized_texts') or []
            if recognized_texts:
                self._recognition_data = {}
                for i, result in enumerate(recognized_texts):
                    # Skip deleted entries
                    if isinstance(result, dict) and result.get('deleted'):
                        continue
                    # Handle both simple string format and complex dict format
                    # Match the format expected by context menu (dict with 'text' and 'bbox' keys)
                    if isinstance(result, str):
                        self._recognition_data[int(i)] = {'text': result, 'bbox': [0, 0, 100, 100]}
                    elif isinstance(result, dict) and 'text' in result:
                        idx = result.get('region_index', i)
                        self._recognition_data[int(idx)] = {
                            'text': result.get('text', ''),
                            'bbox': result.get('bbox', [0, 0, 100, 100])
                        }
                print(f"[STATE] Restored recognition_data for {len(self._recognition_data)} regions")
        except Exception as re:
            print(f"[STATE] Failed to restore recognition_data: {re}")
        
        # Restore translation_data from persisted translated_texts so Edit Translation menu works after reload
        try:
            translated_texts = state.get('translated_texts') or []
            if translated_texts:
                self._translation_data = {}
                for i, result in enumerate(translated_texts):
                    # Skip deleted entries
                    if isinstance(result, dict) and result.get('deleted'):
                        continue
                    idx = result.get('original', {}).get('region_index', i)
                    self._translation_data[int(idx)] = {
                        'original': result.get('original', {}).get('text', ''),
                        'translation': _manga_output_text(result.get('translation'))
                    }
                print(f"[STATE] Restored translation_data for {len(self._translation_data)} regions")
        except Exception as te:
            print(f"[STATE] Failed to restore translation_data: {te}")
        
        # Reattach context menus for rectangles (after both recognition and translation data are restored)
        try:
            rects = getattr(self.image_preview_widget.viewer, 'rectangles', []) or []
            print(f"[STATE] Debug - Available recognition_data keys: {list(getattr(self, '_recognition_data', {}).keys())}")
            print(f"[STATE] Debug - Available translation_data keys: {list(getattr(self, '_translation_data', {}).keys())}")
            for idx, rect_item in enumerate(rects):
                try:
                    # Make sure region_index is set correctly on the rectangle
                    if not hasattr(rect_item, 'region_index'):
                        rect_item.region_index = idx
                    print(f"[STATE] Debug - Rectangle {idx} has region_index: {getattr(rect_item, 'region_index', 'None')}")
                    _add_context_menu_to_rectangle(self, rect_item, rect_item.region_index)
                except Exception as e:
                    print(f"[STATE] Error attaching context menu to rect {idx}: {e}")
            print(f"[STATE] Reattached context menus to {len(rects)} rectangles")
        except Exception as cm:
            print(f"[STATE] Failed to reattach context menus: {cm}")
        
        # Restore viewer rectangles (if no detection regions were restored)
        if 'viewer_rectangles' in state and not ('detection_regions' in state):
            viewer = self.image_preview_widget.viewer
            if hasattr(viewer, 'clear_rectangles'):
                viewer.clear_rectangles()
            
            from PySide6.QtCore import QRectF, Qt
            from PySide6.QtGui import QPen, QBrush, QColor, QPainterPath
            from manga_image_preview import MoveableRectItem, MoveableEllipseItem, MoveablePathItem
            
            # Check if there are OCR or translated texts to determine rectangle colors
            recognized_texts = state.get('recognized_texts', [])
            translated_texts = state.get('translated_texts', [])
            has_ocr_text = bool([t for t in recognized_texts if not (isinstance(t, dict) and t.get('deleted'))])
            has_translated_text = bool([t for t in translated_texts if not (isinstance(t, dict) and t.get('deleted'))])
            has_any_text = has_ocr_text or has_translated_text
            
            for idx, rect_data in enumerate(state['viewer_rectangles']):
                rect = QRectF(rect_data['x'], rect_data['y'], rect_data['width'], rect_data['height'])
                
                # Use blue color if this rectangle has any recognized text (OCR or translated), green otherwise
                has_text_for_this_rect = False
                if has_any_text and idx < max(len(recognized_texts), len(translated_texts)):
                    # Check if this specific rectangle has text (either OCR or translated)
                    has_recognized = (idx < len(recognized_texts) and 
                                    not (isinstance(recognized_texts[idx], dict) and recognized_texts[idx].get('deleted')))
                    has_translation = (idx < len(translated_texts) and 
                                     not (isinstance(translated_texts[idx], dict) and translated_texts[idx].get('deleted')))
                    has_text_for_this_rect = has_recognized or has_translation
                
                if has_text_for_this_rect:
                    # Blue for rectangles with recognized/translated text
                    pen = QPen(QColor(0, 150, 255), 2)
                    brush = QBrush(QColor(0, 150, 255, 50))
                else:
                    # Green for detection-only rectangles
                    pen = QPen(QColor(0, 255, 0), 1)
                    brush = QBrush(QColor(0, 255, 0, 50))
                
                pen.setCosmetic(True)
                shape = rect_data.get('shape', 'rect')
                if shape == 'ellipse':
                    rect_item = MoveableEllipseItem(rect, pen=pen, brush=brush)
                elif shape == 'polygon' and rect_data.get('polygon'):
                    path = QPainterPath()
                    pts = rect_data.get('polygon') or []
                    if pts:
                        path.moveTo(pts[0][0], pts[0][1])
                        for px, py in pts[1:]:
                            path.lineTo(px, py)
                        path.closeSubpath()
                    rect_item = MoveablePathItem(path, pen=pen, brush=brush)
                else:
                    rect_item = MoveableRectItem(rect, pen=pen, brush=brush)
                metadata = _saved_rect_metadata(state, idx, rect_data)
                _copy_region_metadata_to_item(rect_item, metadata)
                if not has_text_for_this_rect:
                    _apply_rectangle_clean_style(rect_item, is_recognized=False, excluded=False)
                # Attach viewer reference so moved emits
                try:
                    rect_item._viewer = viewer
                except Exception:
                    pass
                viewer._scene.addItem(rect_item)
                viewer.rectangles.append(rect_item)
                # Attach region index and move-sync handler for ALL rectangles
                try:
                    rect_item.region_index = idx
                    # CRITICAL: Always set is_recognized based on text state, not just when true
                    rect_item.is_recognized = has_text_for_this_rect
                    _attach_move_sync_to_rectangle(self, rect_item, idx)
                    # CRITICAL: Add context menu to ALL rectangles (both blue and green)
                    _add_context_menu_to_rectangle(self, rect_item, idx)
                except Exception:
                    pass
            
            print(f"[STATE] Restored {len(state['viewer_rectangles'])} viewer shapes")
        
        # If translated_texts exist and rectangles are present, restore text overlays on source viewer
        try:
            translated_texts = state.get('translated_texts') or []
            rects_exist = bool(getattr(self.image_preview_widget.viewer, 'rectangles', []))
            if translated_texts and rects_exist and hasattr(self, '_add_text_overlay_to_viewer'):
                # Filter out deleted text overlays
                active_translated_texts = []
                for i, result in enumerate(translated_texts):
                    if not (isinstance(result, dict) and result.get('deleted')):
                        active_translated_texts.append(result)
                
                if active_translated_texts:
                    _add_text_overlay_to_viewer(self, active_translated_texts)
                    print(f"[STATE] Restored {len(active_translated_texts)} text overlays from persisted state (skipped {len(translated_texts) - len(active_translated_texts)} deleted)")
        except Exception as e2:
            print(f"[STATE] Failed to restore text overlays: {e2}")
        
        # If translated_texts exist and rectangles are present, restore text overlays on source viewer
        try:
            translated_texts = state.get('translated_texts') or []
            rects_exist = bool(getattr(self.image_preview_widget.viewer, 'rectangles', []))
            if translated_texts and rects_exist and hasattr(self, '_add_text_overlay_to_viewer'):
                # Filter out deleted text overlays
                active_translated_texts = []
                for i, result in enumerate(translated_texts):
                    if not (isinstance(result, dict) and result.get('deleted')):
                        active_translated_texts.append(result)
                
                if active_translated_texts:
                    # Ensure _translation_data is populated for context menu (only for active overlays)
                    if not hasattr(self, '_translation_data') or not self._translation_data:
                        self._translation_data = {}
                        for i, result in enumerate(active_translated_texts):
                            idx = result.get('original', {}).get('region_index', i)
                            self._translation_data[int(idx)] = {
                                'original': result.get('original', {}).get('text', ''),
                                'translation': result.get('translation', '')
                            }
                    _add_text_overlay_to_viewer(self, active_translated_texts)
                    # Reattach context menus
                    rects = getattr(self.image_preview_widget.viewer, 'rectangles', []) or []
                    for idx, rect_item in enumerate(rects):
                        try:
                            _add_context_menu_to_rectangle(self, rect_item, idx)
                        except Exception:
                            pass
                    print(f"[STATE] Restored {len(active_translated_texts)} text overlays from persisted state (skipped {len(translated_texts) - len(active_translated_texts)} deleted)")
        except Exception as e2:
            print(f"[STATE] Failed to restore text overlays: {e2}")
        
        print(f"[STATE] Overlay restoration complete for {os.path.basename(image_path)}")
        
        # CRITICAL: Comprehensive graphics scene and overlay synchronization for overlays-only mode
        # This fixes the cosmetic issue where overlays appear disconnected until user interaction
        try:
            from PySide6.QtCore import QTimer
            viewer = self.image_preview_widget.viewer
            
            def comprehensive_overlay_refresh():
                try:
                    # 1. Force complete scene update
                    viewer._scene.update()
                    viewer.update()
                    viewer.repaint()
                    
                    # 2. Ensure text overlays are visible and properly positioned
                    if hasattr(self, 'show_text_overlays_for_image'):
                        self.show_text_overlays_for_image(image_path)
                    
                    # 3. Force overlay position synchronization with rectangles
                    try:
                        _synchronize_overlay_positions_with_rectangles(self, image_path)
                    except Exception:
                        pass
                    
                    # 4. Re-attach move sync handlers to ensure interactivity
                    try:
                        rectangles = getattr(viewer, 'rectangles', []) or []
                        for idx, rect_item in enumerate(rectangles):
                            if hasattr(rect_item, 'region_index'):
                                _attach_move_sync_to_rectangle(self, rect_item, rect_item.region_index)
                            else:
                                _attach_move_sync_to_rectangle(self, rect_item, idx)
                    except Exception:
                        pass
                    
                    # 5. Final comprehensive scene update
                    viewer._scene.update()
                    viewer.viewport().update()
                    viewer.repaint()
                    
                    _manga_debug_print(f"[STATE] Comprehensive overlay refresh completed for {os.path.basename(image_path)}")
                except Exception as e:
                    print(f"[STATE] Comprehensive overlay refresh failed: {e}")
            
            # Schedule multiple refreshes with increasing delays for maximum reliability
            QTimer.singleShot(15, comprehensive_overlay_refresh)
            QTimer.singleShot(100, comprehensive_overlay_refresh)
            QTimer.singleShot(300, comprehensive_overlay_refresh)
        except Exception:
            pass
        
    except Exception as e:
        print(f"[STATE] Failed to restore overlays: {e}")
        import traceback
        traceback.print_exc()

def _draw_detection_boxes_on_preview(self):
    """Draw detection boxes on the preview widget using region data"""
    try:
        if not hasattr(self, '_current_regions') or not self._current_regions:
            return
        
        viewer = self.image_preview_widget.viewer
        
        # If we already have rectangles, don't redraw (preserve existing rectangles during clean operations)
        if hasattr(viewer, 'rectangles') and viewer.rectangles and len(viewer.rectangles) > 0:
            print(f"[DRAW_BOXES] Skipping rectangle drawing - {len(viewer.rectangles)} rectangles already exist")
            return
        
        from PySide6.QtCore import QRectF, Qt
        from PySide6.QtGui import QPen, QBrush, QColor
        from manga_image_preview import MoveableRectItem, MoveableEllipseItem
        
        print(f"[DRAW_BOXES] Drawing {len(self._current_regions)} regions")
        print(f"[DRAW_BOXES] Current rectangles count before drawing: {len(viewer.rectangles)}")
        
        # Draw boxes for each region
        for i, region in enumerate(self._current_regions):
            bbox = region.get('bbox', [])
            if len(bbox) >= 4:
                x, y, width, height = bbox
                rect = QRectF(x, y, width, height)
                print(f"[DRAW_BOXES] Drawing shape {i}: x={x}, y={y}, w={width}, h={height}")
                
                is_free_text = _is_free_text_region_metadata(region)
                if is_free_text:
                    pen_color = QColor(0, 220, 180)
                    brush_color = QColor(0, 220, 180, 45)
                else:
                    pen_color = QColor(0, 255, 0)
                    brush_color = QColor(0, 255, 0, 50)

                # Create pen and brush with detection colors
                pen = QPen(pen_color, 2 if is_free_text else 1)  # Green/teal border
                pen.setCosmetic(True)  # Pen width stays constant regardless of zoom
                pen.setCapStyle(Qt.PenCapStyle.SquareCap)
                pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
                brush = QBrush(brush_color)
                
                # Create shape item (ellipse or rectangle)
                item = MoveableEllipseItem(rect, pen=pen, brush=brush) if getattr(self, '_use_circle_shapes', False) else MoveableRectItem(rect, pen=pen, brush=brush)
                
                # Explicitly disable antialiasing on this item to prevent blur artifacts
                from PySide6.QtWidgets import QGraphicsItem
                item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemUsesExtendedStyleOption, False)
                
                # Attach viewer reference so item can emit moved signal
                try:
                    item._viewer = viewer
                except Exception:
                    pass
                
                viewer._scene.addItem(item)
                viewer.rectangles.append(item)
                
                # Track region index on the item and attach move-sync handler
                try:
                    item.region_index = i
                    _copy_region_metadata_to_item(item, region)
                    _attach_move_sync_to_rectangle(self, item, i)
                    # Add context menu to green detection rectangles
                    _add_context_menu_to_rectangle(self, item, i)
                except Exception:
                    pass
        
        print(f"[DRAW_BOXES] Final rectangles count after drawing: {len(viewer.rectangles)}")
    
    except Exception as e:
        self._log(f"⚠️ Error drawing detection boxes: {str(e)}", "warning")

def _update_preview_after_clean(self, output_path: str):
    """Update preview on main thread after cleaning is complete"""
    try:
        # Store cleaned image path so Translate button can use it
        self._cleaned_image_path = output_path
        print(f"[CLEAN] Stored cleaned image path: {output_path}")
        
        # Before switching image, alias overlays from original path to cleaned path
        if hasattr(self, '_original_image_path') and self._original_image_path:
            _alias_text_overlays_for_image(self, self._original_image_path, output_path)
        
        # Load cleaned image while preserving rectangles and text overlays for workflow continuity
        self.image_preview_widget.load_image(output_path, preserve_rectangles=True, preserve_text_overlays=True)
    except Exception as e:
        self._log(f"❌ Failed to update preview: {str(e)}", "error")

def _restore_clean_button(self):
    """Restore the clean button to its original state and switch display mode to cleaned"""
    try:
        # Remove processing overlay effect for the image that was being processed
        # For clean button, use _original_image_path if available, otherwise current image
        image_path = getattr(self, '_original_image_path', None)
        if not image_path and hasattr(self, 'image_preview_widget'):
            image_path = getattr(self.image_preview_widget, 'current_image_path', None)
        _remove_processing_overlay(self, image_path)
        
        # Re-enable all workflow buttons and hide stop button
        _enable_workflow_buttons(self)
        
        # Switch display mode to 'cleaned' so user sees the result
        try:
            ipw = self.image_preview_widget
            ipw.source_display_mode = 'cleaned'
            ipw.cleaned_images_enabled = True  # Deprecated flag for compatibility
            
            # Update the toggle button appearance to match 'cleaned' state
            if hasattr(ipw, 'cleaned_toggle_btn') and ipw.cleaned_toggle_btn:
                ipw.cleaned_toggle_btn.setText("🧽")  # Sponge for cleaned
                ipw.cleaned_toggle_btn.setToolTip("Showing cleaned images (click to cycle)")
                ipw.cleaned_toggle_btn.setStyleSheet("""
                    QToolButton {
                        background-color: #4a7ba7;
                        border: 2px solid #5a9fd4;
                        font-size: 12pt;
                        min-width: 32px;
                        min-height: 32px;
                        max-width: 36px;
                        max-height: 36px;
                        padding: 3px;
                        border-radius: 3px;
                        color: white;
                    }
                    QToolButton:hover {
                        background-color: #5a9fd4;
                    }
                """)
            
            # Reload the image to show the cleaned version
            if image_path:
                ipw.load_image(image_path, preserve_rectangles=True, preserve_text_overlays=True)
            print(f"[CLEAN_RESTORE] Switched display mode to 'cleaned'")
        except Exception as mode_err:
            print(f"[CLEAN_RESTORE] Failed to switch display mode: {mode_err}")
    except Exception:
        pass

def _extract_regions_for_background(self):
    """Helper to extract regions from preview and store for background thread"""
    try:
        regions = _extract_regions_from_preview(self, )
        self._temp_regions_extracted = regions
        print(f"[EXTRACT] Extracted {len(regions)} regions for background thread")
    except Exception as e:
        print(f"[EXTRACT] Error extracting regions: {e}")
        self._temp_regions_extracted = []

def _restore_recognize_button(self):
    """Restore the recognize button to its original state"""
    try:
        # Remove processing overlay effect for the image that was being processed
        # For recognize, track the image path from the operation
        image_path = None
        if hasattr(self, '_recognized_texts_image_path'):
            image_path = self._recognized_texts_image_path
        elif hasattr(self, 'image_preview_widget'):
            image_path = getattr(self.image_preview_widget, 'current_image_path', None)
        _remove_processing_overlay(self, image_path)
        
        # Re-enable all workflow buttons and hide stop button
        _enable_workflow_buttons(self)
    except Exception:
        pass

def _is_custom_image_edit_workflow(self) -> bool:
    try:
        cfg = getattr(getattr(self, 'main_gui', None), 'config', {}) or {}
        return (
            str(cfg.get('manga_inpaint_method', 'local') or '').lower() == 'local'
            and str(cfg.get('manga_local_inpaint_model', '') or '').lower() == 'custom-image-edit'
            and not bool(cfg.get('manga_skip_inpainting', False))
        )
    except Exception:
        return False

def _regions_for_custom_image_edit(self, image_path: str, use_current_rectangles: bool = True):
    """Return only custom-image-edit regions confirmed to contain OCR text."""
    try:
        has_rectangles = (
            use_current_rectangles
            and hasattr(self, 'image_preview_widget')
            and hasattr(self.image_preview_widget, 'viewer')
            and getattr(self.image_preview_widget.viewer, 'rectangles', None)
            and len(self.image_preview_widget.viewer.rectangles) > 0
        )
        if has_rectangles:
            regions = _extract_regions_from_preview(self)
        else:
            detection_config = _get_detection_config(self) or {}
            if detection_config.get('detect_empty_bubbles', True):
                detection_config['detect_empty_bubbles'] = False
            regions = _run_detection_sync(self, image_path, detection_config)
            if regions:
                try:
                    self.update_queue.put(('detect_results', {
                        'image_path': image_path,
                        'regions': regions
                    }))
                except Exception:
                    pass
                try:
                    if hasattr(self, 'image_state_manager'):
                        self.image_state_manager.update_state(image_path, {'detection_regions': regions})
                except Exception:
                    pass

        if not regions:
            return []
        recognized_texts = _run_ocr_on_regions(self, image_path, regions, _get_ocr_config(self))
        confirmed_regions = _regions_with_ocr_text(regions, recognized_texts)
        self._log(
            f"🧹 Image edit: {len(confirmed_regions)}/{len(regions)} regions contain OCR text",
            "info",
        )
        return confirmed_regions
    except Exception as e:
        print(f"[CUSTOM_IMAGE_EDIT] Failed to resolve regions: {e}")
        return []

def _run_custom_image_edit_translate_clicked(self):
    """Use custom-image-edit as the Translate button workflow without text rendering."""
    try:
        image_path = self.image_preview_widget.current_image_path
        self._translating_image_path = image_path

        _disable_workflow_buttons(self, exclude=None)
        if hasattr(self.image_preview_widget, 'translate_btn'):
            self.image_preview_widget.translate_btn.setText("Translating...")
        if hasattr(self.image_preview_widget, 'thumbnail_list'):
            self.image_preview_widget.thumbnail_list.setEnabled(False)
            print("[TRANSLATE] Disabled thumbnail list during custom image edit")
        _add_processing_overlay(self)

        import threading
        thread = threading.Thread(
            target=_run_custom_image_edit_translate_background,
            args=(self, image_path),
            daemon=True,
        )
        thread.start()
    except Exception as e:
        import traceback
        self._log(f"âŒ Custom image edit setup failed: {str(e)}", "error")
        print(f"[CUSTOM_IMAGE_EDIT] Setup traceback: {traceback.format_exc()}")
        _restore_translate_button(self)

def _run_custom_image_edit_translate_background(self, image_path: str):
    _reset_cancellation_flags(self)
    try:
        self._log("Custom image edit mode: confirming OCR text before editing...", "info")
        regions = _regions_for_custom_image_edit(self, image_path, use_current_rectangles=True)
        if not regions:
            self._log("âš ï¸ No regions found for custom image edit", "warning")
            return

        translated_path = _run_inpainting_sync(
            self,
            image_path,
            regions,
            save_as='translated',
        )
        if not translated_path or not os.path.exists(translated_path):
            self._log("âš ï¸ Custom image edit did not produce an edited image", "warning")
            return

        try:
            if not hasattr(self, '_rendered_images_map'):
                self._rendered_images_map = {}
            self._rendered_images_map[image_path] = translated_path
            if hasattr(self.image_preview_widget, 'current_translated_path'):
                self.image_preview_widget.current_translated_path = translated_path
            if hasattr(self, 'image_state_manager'):
                self.image_state_manager.update_state(image_path, {
                    'rendered_image_path': translated_path,
                    'step': 'translated'
                })
        except Exception:
            pass

        self.update_queue.put(('preview_update', {
            'translated_path': translated_path,
            'source_path': image_path
        }))
        self.update_queue.put(('switch_to_translated_mode', {
            'image_path': image_path
        }))
        self._log(f"âœ… Custom image edit complete: {os.path.basename(image_path)}", "success")
    except Exception as e:
        import traceback
        self._log(f"âŒ Custom image edit failed: {str(e)}", "error")
        print(f"[CUSTOM_IMAGE_EDIT] Background traceback: {traceback.format_exc()}")
    finally:
        self.update_queue.put(('remove_processing_overlay', None))
        self.update_queue.put(('translate_button_restore', None))

def _update_rectangles_with_translations(self, translated_texts: list):
    """Add translated text as overlay layer without modifying original image"""
    try:
        if not hasattr(self, 'image_preview_widget') or not hasattr(self.image_preview_widget, 'viewer'):
            return
        
        print(f"[DEBUG] Adding translated text overlay for {len(translated_texts)} regions")
        
        # Store translation data for context menu access
        self._translation_data = {}
        for i, result in enumerate(translated_texts):
            region_index = result['original'].get('region_index', i)
            # Capture bbox for stable remapping via IoU during re-render
            bbox_val = result.get('bbox')
            if not bbox_val and 0 <= region_index < len(self.image_preview_widget.viewer.rectangles):
                rr = self.image_preview_widget.viewer.rectangles[region_index].sceneBoundingRect()
                bbox_val = [int(rr.x()), int(rr.y()), int(rr.width()), int(rr.height())]
            self._translation_data[region_index] = {
                'original': result['original']['text'],
                'translation': _manga_output_text(result.get('translation')),
                'bbox': bbox_val
            }
        
        # Add text overlay to the viewer
        _add_text_overlay_to_viewer(self, translated_texts)
        
        self._log(f"✅ Added {len(translated_texts)} translation overlays to image preview", "success")
        
    except Exception as e:
        print(f"[DEBUG] Error adding translation overlays: {str(e)}")
        import traceback
        print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        self._log(f"❌ Error adding translation overlays: {str(e)}", "error")

def _add_processing_overlay(self):
    """Add a pulsing overlay effect to indicate processing on the active viewer (source or output)."""
    try:
        # ===== CANCELLATION CHECK: Don't add overlay if stop was clicked =====
        if _is_translation_cancelled(self):
            print(f"[OVERLAY] Not adding overlay - stop was clicked")
            return
        
        if not hasattr(self, 'image_preview_widget'):
            return
        
        # Get current image path for per-image overlay tracking
        current_image = getattr(self.image_preview_widget, 'current_image_path', None)
        if not current_image:
            return
        
        from PySide6.QtWidgets import QGraphicsRectItem
        from PySide6.QtCore import QRectF, QPropertyAnimation, QEasingCurve, Qt, QObject, Property
        from PySide6.QtGui import QBrush, QColor
        
        # Initialize per-image overlay storage
        if not hasattr(self, '_processing_overlays_by_image'):
            self._processing_overlays_by_image = {}
        
        # Don't add overlay if one already exists for this image
        if current_image in self._processing_overlays_by_image:
            print(f"[OVERLAY] Processing overlay already exists for {os.path.basename(current_image)}")
            return
        
        # Use source viewer (no more separate output viewer)
        viewer = getattr(self.image_preview_widget, 'viewer', None)
        if viewer is None:
            return
        
        # Create overlay rectangle covering entire scene
        scene_rect = viewer._scene.sceneRect()
        overlay = QGraphicsRectItem(scene_rect)
        overlay.setBrush(QBrush(QColor(0, 150, 255, 30)))  # Blue semi-transparent
        overlay.setPen(Qt.NoPen)
        overlay.setZValue(1000)  # On top of everything
        
        # Add to scene
        viewer._scene.addItem(overlay)
        
        # Create pulsing animation using QObject wrapper
        class OpacityItem(QObject):
            def __init__(self, item, parent=None):
                super().__init__(parent)
                self._item = item
                self._opacity = 30
            
            def get_opacity(self):
                return self._opacity
            
            def set_opacity(self, value):
                self._opacity = value
                self._item.setBrush(QBrush(QColor(0, 150, 255, int(value))))
            
            opacity = Property(int, get_opacity, set_opacity)
        
        opacity_wrapper = OpacityItem(overlay)
        
        pulse_animation = QPropertyAnimation(opacity_wrapper, b"opacity")
        pulse_animation.setDuration(1500)  # 1.5 seconds
        pulse_animation.setStartValue(15)
        pulse_animation.setEndValue(50)
        pulse_animation.setEasingCurve(QEasingCurve.InOutQuad)
        pulse_animation.setLoopCount(-1)  # Infinite loop
        pulse_animation.start()
        
        # Store per-image
        self._processing_overlays_by_image[current_image] = {
            'overlay': overlay,
            'animation': pulse_animation,
            'wrapper': opacity_wrapper,
            'viewer': viewer
        }
        
        print(f"[OVERLAY] Added processing overlay for {os.path.basename(current_image)}")
        
    except Exception as e:
        print(f"[OVERLAY] Error adding processing overlay: {str(e)}")

def _add_context_menu_to_rectangle(self, rect_item, region_index: int):
    """Add context menu to rectangle for OCR and translation options on right-click"""
    try:
        from PySide6.QtWidgets import QMenu, QMessageBox
        from PySide6.QtCore import QPoint
        from PySide6.QtGui import QAction
        
        # Store the region index on the rectangle for later retrieval
        rect_item.region_index = region_index
        
        # Override the mouse press event to handle right-clicks
        original_mouse_press = rect_item.mousePressEvent
        
        def handle_mouse_press(event):
            try:
                from PySide6.QtCore import Qt
                from PySide6.QtGui import QCursor
                
                if event.button() == Qt.MouseButton.RightButton:
                    # Handle right-click for context menu
                    menu = QMenu()
                    
                    # Get recognition data (fresh lookup for up-to-date text)
                    # Use rect_item.region_index to get the actual stored index for this specific rectangle
                    actual_index = rect_item.region_index
                    is_free_text_rect = _is_free_text_region_metadata(rect_item=rect_item)
                    if is_free_text_rect:
                        type_action = QAction("🏷️ Type: Free text", menu)
                        type_action.setEnabled(False)
                        menu.addAction(type_action)
                    elif getattr(rect_item, 'bubble_type', None) or getattr(rect_item, 'region_type', None):
                        type_action = QAction("🏷️ Type: Bubble text", menu)
                        type_action.setEnabled(False)
                        menu.addAction(type_action)
                    mark_action = QAction("🫧 Mark as Bubble Text" if is_free_text_rect else "🦅 Mark as Free Text", menu)
                    def make_mark_type_handler(idx, rect):
                        return lambda: _handle_toggle_free_text_region(self, idx, rect)
                    mark_action.triggered.connect(make_mark_type_handler(actual_index, rect_item))
                    menu.addAction(mark_action)
                    if not menu.isEmpty():
                        menu.addSeparator()

                    # ALWAYS add "OCR this text" option first - available for all rectangles
                    ocr_this_action = QAction("🔍 OCR This Text", menu)
                    def make_ocr_this_handler(idx, rect):
                        return lambda: _handle_ocr_this_text(self, idx, rect)
                    ocr_this_action.triggered.connect(make_ocr_this_handler(actual_index, rect_item))
                    menu.addAction(ocr_this_action)
                    
                    # Add "Edit OCR" option if text already exists
                    if hasattr(self, '_recognition_data') and actual_index in self._recognition_data:
                        recognition_text = self._recognition_data[actual_index]['text']
                        # Add action to show OCR text (with better preview formatting)
                        preview_text = (recognition_text[:22] + "...") if len(recognition_text) > 25 else recognition_text
                        ocr_action = QAction(f"📝 Edit OCR: \"{preview_text}\"", menu)
                        # Create a proper closure by defining a function that captures the current index
                        def make_ocr_handler(idx):
                            return lambda: _show_ocr_popup(self, self._recognition_data[idx]['text'], idx)
                        ocr_action.triggered.connect(make_ocr_handler(actual_index))
                        menu.addAction(ocr_action)
                    
                    # Get translation data if available (fresh lookup)
                    if hasattr(self, '_translation_data') and actual_index in self._translation_data:
                        current_trans = self._translation_data[actual_index]['translation']
                        preview_trans = (current_trans[:22] + "...") if len(current_trans) > 25 else current_trans
                        trans_action = QAction(f"🌍 Edit Translation: \"{preview_trans}\"", menu)
                        # Create a proper closure by defining a function that captures the current index
                        def make_trans_handler(idx):
                            return lambda: _show_translation_popup(self, self._translation_data[idx], idx)
                        trans_action.triggered.connect(make_trans_handler(actual_index))
                        menu.addAction(trans_action)
                    
                    # Add "Translate This Text" option for manual editing (only if text already recognized)
                    if hasattr(self, '_recognition_data') and actual_index in self._recognition_data:
                        actual_prompt, target_language = _manual_translate_prompt(self)
                        
                        # Add the translate action
                        translate_action = QAction(f"📞 Translate This Text ({target_language})", menu)
                        def make_translate_handler(idx, prompt_text):
                            return lambda: _handle_translate_this_text(self, idx, prompt_text)
                        translate_action.triggered.connect(make_translate_handler(actual_index, actual_prompt))
                        menu.addAction(translate_action)
                    
                    # Add separator before utility options
                    if not menu.isEmpty():
                        menu.addSeparator()
                    
                    # Add "Exclude from Clean" toggle option (always available)
                    is_excluded = getattr(rect_item, 'exclude_from_clean', False)
                    if is_excluded:
                        exclude_action = QAction("❌ Exclude from Clean (ON)", menu)
                    else:
                        exclude_action = QAction("✅ Exclude from Clean (OFF)", menu)
                    def make_exclude_handler(idx, rect):
                        return lambda: _handle_toggle_exclude_clean(self, idx, rect)
                    exclude_action.triggered.connect(make_exclude_handler(actual_index, rect_item))
                    menu.addAction(exclude_action)
                    
                    # Add "Set Inpainting Iterations" option
                    current_iterations = getattr(rect_item, 'inpaint_iterations', None)
                    if current_iterations is not None:
                        iterations_text = f"🔧 Set Inpainting Iterations (Current: {current_iterations})"
                    else:
                        iterations_text = "🔧 Set Inpainting Iterations (Auto)"
                    iterations_action = QAction(iterations_text, menu)
                    def make_iterations_handler(idx, rect):
                        return lambda: _handle_set_inpainting_iterations(self, idx, rect)
                    iterations_action.triggered.connect(make_iterations_handler(actual_index, rect_item))
                    menu.addAction(iterations_action)
                    
                    # Add "Clean This Rectangle" option (disabled when waiting for model)
                    is_waiting = getattr(self, '_waiting_for_model', False)
                    if is_waiting:
                        clean_action = QAction("⏳ Clean This Rectangle (Waiting...)", menu)
                        clean_action.setEnabled(False)
                    else:
                        clean_action = QAction("🧽 Clean This Rectangle", menu)
                        def make_clean_handler(idx, rect):
                            return lambda: _handle_clean_this_rectangle(self, idx, rect)
                        clean_action.triggered.connect(make_clean_handler(actual_index, rect_item))
                    menu.addAction(clean_action)
                    
                    # Add "Delete Selected" option
                    delete_action = QAction("🗑️ Delete Selected", menu)
                    def make_delete_handler(idx, rect):
                        return lambda: _handle_delete_rectangle(self, idx, rect)
                    delete_action.triggered.connect(make_delete_handler(actual_index, rect_item))
                    menu.addAction(delete_action)
                    
                    if not menu.isEmpty():
                        # Set menu properties for better display
                        menu.setMinimumWidth(250)  # Increase minimum width for better readability
                        menu.setMaximumWidth(500)  # Allow wider menus for longer text
                        
                        # Apply better styling to the menu
                        menu.setStyleSheet("""
                            QMenu {
                                background-color: #2d2d2d;
                                border: 1px solid #5a9fd4;
                                color: white;
                                padding: 4px;
                                border-radius: 4px;
                            }
                            QMenu::item {
                                background-color: transparent;
                                padding: 8px 12px;
                                margin: 2px;
                                border-radius: 3px;
                                font-size: 11pt;
                                white-space: nowrap;
                            }
                            QMenu::item:selected {
                                background-color: #5a9fd4;
                                color: white;
                            }
                            QMenu::separator {
                                height: 1px;
                                background-color: #5a9fd4;
                                margin: 4px 8px;
                            }
                        """)
                        
                        # Show the context menu at the actual cursor position
                        menu.exec(QCursor.pos())
                    
                    # Don't propagate the right-click event
                    event.accept()
                    return
                
                # For other mouse buttons, use original behavior
                original_mouse_press(event)
                
            except Exception as e:
                print(f"[DEBUG] Mouse press error: {str(e)}")
                # Fallback to original behavior
                original_mouse_press(event)
        
        # Replace the rectangle's mouse press event
        rect_item.mousePressEvent = handle_mouse_press
        
    except Exception as e:
        print(f"[DEBUG] Error adding context menu: {str(e)}")

def _handle_toggle_exclude_clean(self, region_index: int, rect_item):
    """Toggle exclude from clean status for a rectangle"""
    try:
        print(f"[TOGGLE_DEBUG] === TOGGLE CALLED FOR RECTANGLE {region_index} ===")
        
        # Special verbose debug for rectangle 0
        if region_index == 0:
            print(f"[RECT_0_DEBUG] *** Special debugging for rectangle 0 ***")
            print(f"[RECT_0_DEBUG] This is the problematic rectangle!")
        
        # Get current exclude status
        current_status = getattr(rect_item, 'exclude_from_clean', False)
        new_status = not current_status
        
        print(f"[TOGGLE_DEBUG] Rectangle {region_index}: current_status={current_status}, new_status={new_status}")
        print(f"[TOGGLE_DEBUG] Rectangle object: {rect_item}")
        print(f"[TOGGLE_DEBUG] Rectangle has exclude_from_clean attr: {hasattr(rect_item, 'exclude_from_clean')}")
        
        # Set the new status on the rectangle
        rect_item.exclude_from_clean = new_status
        
        # Visual feedback - change rectangle appearance to indicate excluded status
        from PySide6.QtGui import QPen, QBrush, QColor
        if new_status:
            # Excluded - use red/orange styling
            rect_item.setPen(QPen(QColor(255, 140, 0), 3))  # Orange border, thicker
            rect_item.setBrush(QBrush(QColor(255, 140, 0, 30)))  # Semi-transparent orange fill
            self._log(f"❌ Rectangle {region_index} excluded from inpainting", "info")
        else:
            # Not excluded - restore normal styling based on rectangle type
            if hasattr(rect_item, 'is_recognized') and rect_item.is_recognized:
                # Blue for recognized text
                rect_item.setPen(QPen(QColor(0, 150, 255), 2))
                rect_item.setBrush(QBrush(QColor(0, 150, 255, 50)))
            else:
                # Green for detection boxes
                rect_item.setPen(QPen(QColor(0, 255, 0), 2))
                rect_item.setBrush(QBrush(QColor(0, 255, 0, 50)))
            _apply_rectangle_clean_style(rect_item, is_recognized=getattr(rect_item, 'is_recognized', False), excluded=False)
            self._log(f"✅ Rectangle {region_index} included in inpainting", "info")
        
        # EXCLUSION PERSISTENCE REMOVAL: Don't save exclusion state to persist across sessions
        # Exclusions are now session-only and reset when the app is restarted
        print(f"[EXCLUDE_CLEAN] Rectangle {region_index} exclude status: {new_status} (session-only, not persisted)")
        
        print(f"[EXCLUDE_CLEAN] Rectangle {region_index} exclude status: {new_status}")
        
    except Exception as e:
        print(f"[EXCLUDE_CLEAN] Error toggling exclude status: {e}")
        import traceback
        print(f"[EXCLUDE_CLEAN] Traceback: {traceback.format_exc()}")
        self._log(f"❌ Failed to toggle exclude status: {str(e)}", "error")

def _handle_set_inpainting_iterations(self, region_index: int, rect_item):
    """Handle setting custom inpainting iterations for a rectangle"""
    try:
        from PySide6.QtWidgets import QInputDialog, QMessageBox
        from PySide6.QtCore import Qt
        
        print(f"[INPAINT_ITERATIONS] Setting iterations for rectangle {region_index}")
        
        # Get current value
        current_iterations = getattr(rect_item, 'inpaint_iterations', None)
        current_display = current_iterations if current_iterations is not None else "Auto"
        
        # Show input dialog
        dialog_text = (
            f"Set inpainting iterations for rectangle {region_index}\n\n"
            f"Current: {current_display}\n\n"
            f"Enter number of iterations (-1 to 50):\n-1 = Auto, 0-50 = Custom iterations"
        )
        
        value, ok = QInputDialog.getInt(
            self.dialog,
            "Set Inpainting Iterations",
            dialog_text,
            current_iterations if current_iterations is not None else -1
        )
        
        if ok:
            # Validate input range (-1 to 50, where -1 means auto)
            if value < -1 or value > 50:
                QMessageBox.warning(
                    self.dialog,
                    "Invalid Input", 
                    f"Please enter a value between -1 and 50.\n-1 = Auto, 0-50 = Custom iterations"
                )
                return
            
            _apply_inpaint_iterations(self, region_index, rect_item, value)
            
            # Visual feedback - update rectangle appearance slightly
            from PySide6.QtGui import QPen, QBrush, QColor
            if value != -1:
                # Custom iterations - add slight blue tint to border
                if getattr(rect_item, 'exclude_from_clean', False):
                    # Keep orange if excluded
                    pass
                else:
                    # Add blue tint to show custom iterations
                    if hasattr(rect_item, 'is_recognized') and rect_item.is_recognized:
                        # Slightly brighter blue for recognized + custom iterations
                        rect_item.setPen(QPen(QColor(50, 170, 255), 2))
                    else:
                        # Slightly blue-green for detection + custom iterations
                        rect_item.setPen(QPen(QColor(0, 200, 150), 2))
        
    except Exception as e:
        print(f"[INPAINT_ITERATIONS] Error setting iterations: {e}")
        import traceback
        print(f"[INPAINT_ITERATIONS] Traceback: {traceback.format_exc()}")
        self._log(f"❌ Failed to set inpainting iterations: {str(e)}", "error")

def _add_rectangle_pulse_effect(self, rect_item, region_index, auto_remove=False):
    """Add a purple pulse effect to a specific rectangle during operations
    
    Args:
        rect_item: The rectangle item to pulse
        region_index: Index of the region
        auto_remove: If True, auto-remove after 0.75s (for save position). 
                    If False, loop indefinitely until manually removed (for OCR/clean/translate)
    """
    try:
        from PySide6.QtWidgets import QGraphicsRectItem
        from PySide6.QtCore import QRectF, QPropertyAnimation, QEasingCurve, Qt, QObject, Property
        from PySide6.QtGui import QPen, QBrush, QColor
        
        mode = "auto-remove" if auto_remove else "loop until complete"
        print(f"[RECT_PULSE] Adding pulse effect to rectangle {region_index} (mode: {mode})")
        
        # Store original pen for restoration
        original_pen = rect_item.pen()
        setattr(rect_item, '_original_pen', original_pen)
        
        # Create pulsing animation using QObject wrapper
        class PulsingRectangle(QObject):
            def __init__(self, rect_item, parent=None):
                super().__init__(parent)
                self._rect_item = rect_item
                self._intensity = 100
            
            def get_intensity(self):
                return self._intensity
            
            def set_intensity(self, value):
                self._intensity = value
                # Create purple pen with varying intensity
                purple_color = QColor(147, 112, 219, int(value))  # Medium slate blue/purple
                pen = QPen(purple_color, 3)  # Thicker pen for visibility
                self._rect_item.setPen(pen)
            
            intensity = Property(int, get_intensity, set_intensity)
        
        # Store pulse wrapper on the rectangle item
        pulse_wrapper = PulsingRectangle(rect_item)
        setattr(rect_item, '_pulse_wrapper', pulse_wrapper)
        
        # Create the animation
        pulse_animation = QPropertyAnimation(pulse_wrapper, b"intensity")
        pulse_animation.setDuration(750)  # 0.75 seconds (750ms)
        pulse_animation.setStartValue(80)
        pulse_animation.setEndValue(255)
        pulse_animation.setEasingCurve(QEasingCurve.InOutQuad)
        
        if auto_remove:
            # For save position: run once and auto-remove
            pulse_animation.setLoopCount(1)
            # Auto-remove pulse effect when animation finishes
            def on_animation_finished():
                try:
                    _remove_rectangle_pulse_effect(self, rect_item, region_index)
                except Exception as e:
                    print(f"[RECT_PULSE] Error removing pulse on finish: {e}")
            pulse_animation.finished.connect(on_animation_finished)
        else:
            # For other operations: loop indefinitely until manually removed
            pulse_animation.setLoopCount(-1)  # Loop forever
        
        pulse_animation.start()
        
        # Store animation on the rectangle item
        setattr(rect_item, '_pulse_animation', pulse_animation)
        
        print(f"[RECT_PULSE] Started pulse animation for rectangle {region_index}")
        
    except Exception as e:
        print(f"[RECT_PULSE] Error adding pulse effect: {e}")
        import traceback
        print(f"[RECT_PULSE] Traceback: {traceback.format_exc()}")

def _remove_rectangle_pulse_effect(self, rect_item, region_index):
    """Remove the pulse effect from a specific rectangle"""
    try:
        print(f"[RECT_PULSE] Removing pulse effect from rectangle {region_index}")
        
        # Stop animation
        if hasattr(rect_item, '_pulse_animation'):
            rect_item._pulse_animation.stop()
            delattr(rect_item, '_pulse_animation')
        
        # Remove pulse wrapper
        if hasattr(rect_item, '_pulse_wrapper'):
            delattr(rect_item, '_pulse_wrapper')
        
        # Restore original pen
        if hasattr(rect_item, '_original_pen'):
            rect_item.setPen(rect_item._original_pen)
            delattr(rect_item, '_original_pen')
        else:
            # Fallback: restore normal styling based on rectangle type
            from PySide6.QtGui import QPen, QBrush, QColor
            if getattr(rect_item, 'exclude_from_clean', False):
                # Orange for excluded
                rect_item.setPen(QPen(QColor(255, 165, 0), 2))
            elif hasattr(rect_item, 'is_recognized') and rect_item.is_recognized:
                # Blue for recognized text
                rect_item.setPen(QPen(QColor(0, 150, 255), 2))
            else:
                # Green for detection boxes
                rect_item.setPen(QPen(QColor(0, 255, 0), 2))
        
        print(f"[RECT_PULSE] Removed pulse effect from rectangle {region_index}")
        
    except Exception as e:
        print(f"[RECT_PULSE] Error removing pulse effect: {e}")

def _restore_exclusion_status_from_state(self, image_path: str):
    """Restore exclusion status for rectangles from saved state - DISABLED"""
    print = _manga_debug_print
    try:
        print(f"[EXCLUDE_RESTORE] === EXCLUSION RESTORATION DISABLED - ALWAYS START WITH NO EXCLUSIONS ===")
        print(f"[EXCLUDE_RESTORE] Exclusion toggle state no longer persists across sessions")
        
        # Ensure all rectangles start with no exclusion styling
        if hasattr(self.image_preview_widget, 'viewer') and hasattr(self.image_preview_widget.viewer, 'rectangles'):
            rectangles = self.image_preview_widget.viewer.rectangles
            print(f"[EXCLUDE_RESTORE] Ensuring {len(rectangles)} rectangles start with normal styling")
            
            from PySide6.QtGui import QPen, QBrush, QColor
            for i, rect_item in enumerate(rectangles):
                # Ensure exclude flag is False
                rect_item.exclude_from_clean = False
                
                # Apply normal styling based on rectangle type
                if hasattr(rect_item, 'is_recognized') and rect_item.is_recognized:
                    # Blue for recognized text
                    rect_item.setPen(QPen(QColor(0, 150, 255), 2))
                    rect_item.setBrush(QBrush(QColor(0, 150, 255, 50)))
                else:
                    # Green for detection boxes
                    rect_item.setPen(QPen(QColor(0, 255, 0), 2))
                    rect_item.setBrush(QBrush(QColor(0, 255, 0, 50)))
                _apply_rectangle_clean_style(rect_item, is_recognized=getattr(rect_item, 'is_recognized', False), excluded=False)
        
        print(f"[EXCLUDE_RESTORE] All rectangles initialized with no exclusions")
        
    except Exception as e:
        print(f"[EXCLUDE_RESTORE] Error initializing exclusion status: {e}")

def _restore_inpainting_iterations_from_state(self, image_path: str):
    """Restore custom inpainting iterations for rectangles from saved state"""
    print = _manga_debug_print
    try:
        print(f"[ITERATIONS_RESTORE] === ITERATION RESTORATION CALLED FOR IMAGE: {image_path} ===")
        
        if not hasattr(self, 'image_state_manager'):
            print(f"[ITERATIONS_RESTORE] No image_state_manager available")
            return
        
        # Get iterations dict from state
        state = self.image_state_manager.get_state(image_path)
        print(f"[ITERATIONS_RESTORE] State for {image_path}: {state}")
        
        custom_iterations = state.get('inpaint_iterations', {})
        print(f"[ITERATIONS_RESTORE] Custom iterations from state: {custom_iterations}")
        
        if not custom_iterations:
            print(f"[ITERATIONS_RESTORE] No custom iterations to restore")
            return
        
        # Apply custom iterations to rectangles
        if hasattr(self.image_preview_widget, 'viewer') and hasattr(self.image_preview_widget.viewer, 'rectangles'):
            rectangles = self.image_preview_widget.viewer.rectangles
            print(f"[ITERATIONS_RESTORE] Found {len(rectangles)} rectangles in viewer")
            
            for region_str, iterations in custom_iterations.items():
                try:
                    region_index = int(region_str)
                    print(f"[ITERATIONS_RESTORE] Processing custom iterations for region {region_index}: {iterations}")
                    
                    if 0 <= region_index < len(rectangles):
                        rect_item = rectangles[region_index]
                        
                        # Set custom iterations
                        rect_item.inpaint_iterations = iterations
                        print(f"[ITERATIONS_RESTORE] Set inpaint_iterations={iterations} for region {region_index}")
                        
                        # Apply visual styling to indicate custom iterations
                        from PySide6.QtGui import QPen, QBrush, QColor
                        if not getattr(rect_item, 'exclude_from_clean', False):
                            # Only apply styling if not excluded (excluded styling takes priority)
                            if hasattr(rect_item, 'is_recognized') and rect_item.is_recognized:
                                # Slightly brighter blue for recognized + custom iterations
                                rect_item.setPen(QPen(QColor(50, 170, 255), 2))
                            else:
                                # Slightly blue-green for detection + custom iterations
                                rect_item.setPen(QPen(QColor(0, 200, 150), 2))
                        
                        print(f"[ITERATIONS_RESTORE] Applied styling for custom iterations to region {region_index}")
                    else:
                        print(f"[ITERATIONS_RESTORE] Region index {region_index} out of bounds (rectangles: {len(rectangles)})")
                except (ValueError, TypeError) as e:
                    print(f"[ITERATIONS_RESTORE] Error processing region {region_str}: {e}")
        else:
            print(f"[ITERATIONS_RESTORE] No rectangles available in viewer")
        
        print(f"[ITERATIONS_RESTORE] Completed restoration for {len(custom_iterations)} custom iteration settings")
        
    except Exception as e:
        print(f"[ITERATIONS_RESTORE] Error restoring inpainting iterations: {e}")

def _debug_exclusion_status(self):
    """Debug function to show current exclusion status"""
    try:
        if not hasattr(self, 'image_preview_widget') or not self.image_preview_widget.current_image_path:
            print(f"[EXCLUSION_DEBUG] No current image")
            return
        
        current_image = self.image_preview_widget.current_image_path
        print(f"[EXCLUSION_DEBUG] Current image: {current_image}")
        
        # Check state
        if hasattr(self, 'image_state_manager'):
            state = self.image_state_manager.get_state(current_image)
            excluded_regions = state.get('excluded_from_clean', []) if state else []
            print(f"[EXCLUSION_DEBUG] Excluded regions in state: {excluded_regions}")
        
        # Check rectangles
        if hasattr(self.image_preview_widget, 'viewer') and hasattr(self.image_preview_widget.viewer, 'rectangles'):
            rectangles = self.image_preview_widget.viewer.rectangles
            print(f"[EXCLUSION_DEBUG] Total rectangles: {len(rectangles)}")
            
            for i, rect_item in enumerate(rectangles):
                is_excluded = getattr(rect_item, 'exclude_from_clean', False)
                print(f"[EXCLUSION_DEBUG] Rectangle {i}: excluded={is_excluded}")
        
    except Exception as e:
        print(f"[EXCLUSION_DEBUG] Error in debug: {e}")

def _clear_all_exclusion_states(self):
    """Clear exclusion states for all images - useful for debugging"""
    try:
        if not hasattr(self, 'image_state_manager'):
            print(f"[CLEAR_EXCLUSION] No image_state_manager available")
            return
        
        # Get current image if available
        current_image = None
        if hasattr(self, 'image_preview_widget') and self.image_preview_widget.current_image_path:
            current_image = self.image_preview_widget.current_image_path
        
        # Clear current image's exclusion state
        if current_image:
            state = self.image_state_manager.get_state(current_image)
            if 'excluded_from_clean' in state:
                old_exclusions = state['excluded_from_clean']
                state['excluded_from_clean'] = []
                self.image_state_manager.set_state(current_image, state)
                print(f"[CLEAR_EXCLUSION] Cleared exclusions for current image: {old_exclusions} -> []")
                
                # Also clear visual state from rectangles
                if hasattr(self.image_preview_widget, 'viewer') and hasattr(self.image_preview_widget.viewer, 'rectangles'):
                    rectangles = self.image_preview_widget.viewer.rectangles
                    for i, rect_item in enumerate(rectangles):
                        if getattr(rect_item, 'exclude_from_clean', False):
                            rect_item.exclude_from_clean = False
                            # Restore normal styling
                            from PySide6.QtGui import QPen, QBrush, QColor
                            if hasattr(rect_item, 'is_recognized') and rect_item.is_recognized:
                                rect_item.setPen(QPen(QColor(0, 150, 255), 2))
                                rect_item.setBrush(QBrush(QColor(0, 150, 255, 50)))
                            else:
                                rect_item.setPen(QPen(QColor(0, 255, 0), 2))
                                rect_item.setBrush(QBrush(QColor(0, 255, 0, 50)))
                            print(f"[CLEAR_EXCLUSION] Cleared visual exclusion for rectangle {i}")
                
                self._log("✅ Cleared all exclusion states for current image", "info")
            else:
                print(f"[CLEAR_EXCLUSION] No exclusions found for current image")
        
    except Exception as e:
        print(f"[CLEAR_EXCLUSION] Error clearing exclusions: {e}")
        import traceback
        print(f"[CLEAR_EXCLUSION] Traceback: {traceback.format_exc()}")

def _clear_all_exclusions(self):
    """Clear all exclusion flags and reset rectangles to normal appearance"""
    try:
        if not hasattr(self, 'image_preview_widget') or not self.image_preview_widget.current_image_path:
            print(f"[CLEAR_EXCLUSIONS] No current image")
            return
        
        current_image = self.image_preview_widget.current_image_path
        print(f"[CLEAR_EXCLUSIONS] Clearing exclusions for: {current_image}")
        
        # Clear state
        if hasattr(self, 'image_state_manager'):
            state = self.image_state_manager.get_state(current_image)
            if state is None:
                state = {}
            state['excluded_from_clean'] = []
            self.image_state_manager.set_state(current_image, state)
            print(f"[CLEAR_EXCLUSIONS] Cleared exclusion list in state")
        
        # Reset all rectangles
        if hasattr(self.image_preview_widget, 'viewer') and hasattr(self.image_preview_widget.viewer, 'rectangles'):
            rectangles = self.image_preview_widget.viewer.rectangles
            from PySide6.QtGui import QPen, QBrush, QColor
            
            for i, rect_item in enumerate(rectangles):
                # Clear exclude flag
                rect_item.exclude_from_clean = False
                
                # Reset to normal appearance based on rectangle type
                if hasattr(rect_item, 'is_recognized') and rect_item.is_recognized:
                    # Blue for recognized text
                    rect_item.setPen(QPen(QColor(0, 150, 255), 2))
                    rect_item.setBrush(QBrush(QColor(0, 150, 255, 50)))
                else:
                    # Green for detection boxes
                    rect_item.setPen(QPen(QColor(0, 255, 0), 2))
                    rect_item.setBrush(QBrush(QColor(0, 255, 0, 50)))
                
                print(f"[CLEAR_EXCLUSIONS] Reset rectangle {i} to normal appearance")
        
        self._log("✅ Cleared all exclusions - all rectangles will be included in cleaning", "info")
        print(f"[CLEAR_EXCLUSIONS] All exclusions cleared successfully")
        
    except Exception as e:
        print(f"[CLEAR_EXCLUSIONS] Error: {e}")
        import traceback
        print(f"[CLEAR_EXCLUSIONS] Traceback: {traceback.format_exc()}")

def _relayout_overlay_for_region(self, region_index: int):
    """Re-layout the overlay text to fit the current blue rectangle (auto-resize like pipeline)."""
    try:
        current_image = getattr(self.image_preview_widget, 'current_image_path', None)
        if not current_image:
            return
        overlays_map = getattr(self, '_text_overlays_by_image', {}) or {}
        groups = overlays_map.get(current_image, [])
        target = None
        for g in groups:
            if getattr(g, '_overlay_region_index', None) == region_index:
                target = g
                break
        if target is None:
            return
        # Get rectangle bounds
        rects = getattr(self.image_preview_widget.viewer, 'rectangles', []) or []
        if not (0 <= int(region_index) < len(rects)):
            return
        br = rects[int(region_index)].sceneBoundingRect()
        x, y, w, h = int(br.x()), int(br.y()), int(br.width()), int(br.height())
        if w <= 0 or h <= 0:
            return
        # Get text
        text = None
        try:
            if hasattr(self, '_translation_data') and isinstance(self._translation_data, dict):
                td = self._translation_data.get(int(region_index))
                if td:
                    text = td.get('translation')
        except Exception:
            pass
        if not text:
            text = getattr(target, '_overlay_original_text', '')
        if text is None:
            text = ''
        # Settings with background forced off for source tab
        settings = _get_manga_rendering_settings(self, )
        try:
            settings['show_background'] = False
            settings['bg_opacity'] = 0
        except Exception:
            pass
        # Create new text item sized to current rectangle
        new_text_item, _ = _create_manga_text_item(self, text, x, y, w, h, settings)
        if new_text_item is None:
            return
        viewer = self.image_preview_widget.viewer
        # Replace text item in group with proper cleanup
        try:
            old_text = getattr(target, '_overlay_text_item', None)
            if old_text is not None:
                try:
                    target.removeFromGroup(old_text)
                except Exception:
                    pass
                try:
                    viewer._scene.removeItem(old_text)
                    # Explicitly delete old text item to free memory
                    old_text.deleteLater()
                except Exception:
                    pass
        except Exception:
            pass
        # Add new text to scene and group
        viewer._scene.addItem(new_text_item)
        try:
            target.addToGroup(new_text_item)
        except Exception:
            pass
        try:
            target._overlay_text_item = new_text_item
            target._overlay_bbox_size = (w, h)
        except Exception:
            pass
        # Update transparent overlay rect to new size if present
        try:
            for child in target.childItems():
                if hasattr(child, 'region_index') and getattr(child, 'region_index') == region_index:
                    # This is the transparent overlay rect
                    child.setRect(x, y, w, h)
                    break
        except Exception:
            pass
        # Force scene update
        try:
            viewer._scene.update()
            viewer.viewport().update()
        except Exception:
            pass
    except Exception:
        pass

def _relayout_all_overlays_for_current_image(self):
    """Re-layout all overlays for the current image using current settings."""
    try:
        current_image = getattr(self.image_preview_widget, 'current_image_path', None)
        if not current_image:
            return
        overlays_map = getattr(self, '_text_overlays_by_image', {}) or {}
        groups = overlays_map.get(current_image, [])
        for g in groups:
            idx = getattr(g, '_overlay_region_index', None)
            if idx is not None:
                try:
                    _relayout_overlay_for_region(self, int(idx))
                except Exception:
                    continue
    except Exception:
        pass

def _attach_move_sync_to_rectangle(self, rect_item, region_index: int):
    """Attach a move handler so that when the blue rectangle is moved,
    the corresponding text overlay group moves with it and state is persisted.
    Safe to call multiple times; attaches once per item.
    """
    try:
        # Avoid duplicate attachment
        if getattr(rect_item, '_move_sync_attached', False):
            return
        rect_item._move_sync_attached = True
        # Ensure region_index is present on the item
        rect_item.region_index = region_index
        
        original_release = rect_item.mouseReleaseEvent
        
        def _on_rect_release(ev, r=rect_item, idx=region_index):
            try:
                # Call original release
                try:
                    original_release(ev)
                except Exception:
                    pass
                
                # Desired new top-left in SCENE coordinates (use sceneBoundingRect, not local rect)
                rr = r.sceneBoundingRect()
                new_x, new_y = int(rr.x()), int(rr.y())
                
                # Find the overlay group to move — prefer exact index, else best IoU match
                current_image = getattr(self.image_preview_widget, 'current_image_path', None)
                overlays_map = getattr(self, '_text_overlays_by_image', {}) or {}
                groups = overlays_map.get(current_image, [])
                target = None
                for g in groups:
                    if getattr(g, '_overlay_region_index', None) == idx:
                        target = g
                        break
                if target is None:
                    # Fallback to IoU-based match
                    def _iou(a, b):
                        try:
                            ax, ay, aw, ah = a
                            bx, by, bw, bh = b
                            ax2, ay2 = ax + aw, ay + ah
                            bx2, by2 = bx + bw, by + bh
                            x1 = max(ax, bx); y1 = max(ay, by)
                            x2 = min(ax2, bx2); y2 = min(ay2, by2)
                            inter = max(0, x2 - x1) * max(0, y2 - y1)
                            area_a = max(0, aw) * max(0, ah)
                            area_b = max(0, bw) * max(0, bh)
                            den = area_a + area_b - inter
                            return (inter / den) if den > 0 else 0.0
                        except Exception:
                            return 0.0
                    best_iou = 0.0
                    for g in groups:
                        brg = g.sceneBoundingRect()
                        iou = _iou([new_x, new_y, int(rr.width()), int(rr.height())], [int(brg.x()), int(brg.y()), int(brg.width()), int(brg.height())])
                        if iou > best_iou:
                            best_iou = iou
                            target = g
                
                if target is not None:
                    # Compute desired target position preserving any saved offset for this rectangle index
                    try:
                        saved_offsets = {}
                        if hasattr(self, 'image_state_manager') and current_image:
                            st = self.image_state_manager.get_state(current_image) or {}
                            saved_offsets = st.get('overlay_offsets') or {}
                        off = saved_offsets.get(str(int(idx))) or saved_offsets.get(int(idx)) or [0, 0]
                        off_x, off_y = int(off[0]) if isinstance(off, (list, tuple)) and len(off) > 0 else 0, int(off[1]) if isinstance(off, (list, tuple)) and len(off) > 1 else 0
                    except Exception:
                        off_x, off_y = 0, 0
                    br = target.sceneBoundingRect()
                    dx = (new_x + off_x) - int(br.x())
                    dy = (new_y + off_y) - int(br.y())
                    if dx != 0 or dy != 0:
                        try:
                            target.moveBy(dx, dy)
                        except Exception:
                            try:
                                from PySide6.QtCore import QPointF
                                target.setPos(target.pos() + QPointF(dx, dy))
                            except Exception:
                                pass
                        
                        # Toggle overlay visibility based on overlap IoU with original bbox
                        try:
                            def _iou_xywh(a, b):
                                try:
                                    ax, ay, aw, ah = int(a[0]), int(a[1]), int(a[2]), int(a[3])
                                    bx, by, bw, bh = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                                    ax2, ay2 = ax + aw, ay + ah
                                    bx2, by2 = bx + bw, by + bh
                                    x1 = max(ax, bx); y1 = max(ay, by)
                                    x2 = min(ax2, bx2); y2 = min(ay2, by2)
                                    inter = max(0, x2 - x1) * max(0, y2 - y1)
                                    area_a = max(0, aw) * max(0, ah)
                                    area_b = max(0, bw) * max(0, bh)
                                    den = area_a + area_b - inter
                                    return (inter / den) if den > 0 else 0.0
                                except Exception:
                                    return 0.0
                            brg = target.sceneBoundingRect()
                            cur = [int(brg.x()), int(brg.y()), int(brg.width()), int(brg.height())]
                            ob = getattr(target, '_overlay_original_bbox', None)
                            if ob and len(ob) >= 4:
                                overlap = _iou_xywh(cur, ob)
                                target.setVisible(overlap < 0.5)
                            else:
                                target.setVisible(True)
                        except Exception as vis_err:
                            print(f"[DEBUG] Error setting overlay visibility on move: {vis_err}")
                        
                        # Force scene update
                        try:
                            self.image_preview_widget.viewer._scene.update()
                        except Exception:
                            pass
                    
                    # Persist only this region's overlay offset to avoid touching others
                    try:
                        _persist_single_overlay_offset(self, current_image, idx, target)
                    except Exception:
                        pass

                    # Only re-layout if rectangle SIZE changed (width/height), not just position
                    # This avoids expensive text rendering on simple moves
                    try:
                        if hasattr(target, '_overlay_bbox_size'):
                            old_w, old_h = target._overlay_bbox_size
                            new_w, new_h = int(rr.width()), int(rr.height())
                            # Only re-layout if size changed by more than 2 pixels (avoid rounding noise)
                            if abs(new_w - old_w) > 2 or abs(new_h - old_h) > 2:
                                print(f"[PERF] Rectangle {idx} resized ({old_w}x{old_h} -> {new_w}x{new_h}), re-layouting text")
                                _relayout_overlay_for_region(self, idx)
                            # else: just moved, text overlay already moved with it via moveBy()
                    except Exception:
                        pass
                
                # Persist rectangles state
                try:
                    if hasattr(self.image_preview_widget, '_persist_rectangles_state'):
                        self.image_preview_widget._persist_rectangles_state()
                except Exception:
                    pass
            except Exception as e:
                print(f"[DEBUG] Rectangle move sync failed: {e}")
        
        rect_item.mouseReleaseEvent = _on_rect_release
    except Exception as e:
        print(f"[DEBUG] Failed to attach move sync: {e}")

def _persist_overlay_offsets_for_current_image(self):
    """Persist overlay offsets relative to the best-matching rectangle for the current image.
    Uses IoU to robustly tie each overlay group to a rectangle, then stores dx,dy per matched index.
    NOTE: Prefer _persist_single_overlay_offset during interactive edits to avoid global remap.
    """
    try:
        current_image = getattr(self.image_preview_widget, 'current_image_path', None)
        if not current_image:
            return
        overlays_map = getattr(self, '_text_overlays_by_image', {}) or {}
        groups = overlays_map.get(current_image, [])
        rects = getattr(self.image_preview_widget.viewer, 'rectangles', []) or []
        offsets = {}
        
        def _iou(a, b):
            try:
                ax, ay, aw, ah = a
                bx, by, bw, bh = b
                ax2, ay2 = ax + aw, ay + ah
                bx2, by2 = bx + bw, by + bh
                x1 = max(ax, bx); y1 = max(ay, by)
                x2 = min(ax2, bx2); y2 = min(ay2, by2)
                inter = max(0, x2 - x1) * max(0, y2 - y1)
                area_a = max(0, aw) * max(0, ah)
                area_b = max(0, bw) * max(0, bh)
                den = area_a + area_b - inter
                return (inter / den) if den > 0 else 0.0
            except Exception:
                return 0.0
        
        for g in groups:
            try:
                br_g = g.sceneBoundingRect()
                gx, gy, gw, gh = int(br_g.x()), int(br_g.y()), int(br_g.width()), int(br_g.height())
                # Find best rectangle by IoU
                best_idx, best_iou = -1, 0.0
                for i, r in enumerate(rects):
                    br_r = r.sceneBoundingRect()
                    rx, ry, rw, rh = int(br_r.x()), int(br_r.y()), int(br_r.width()), int(br_r.height())
                    iou = _iou([gx, gy, gw, gh], [rx, ry, rw, rh])
                    if iou > best_iou:
                        best_iou, best_idx = iou, i
                if best_idx != -1:
                    br_r = rects[best_idx].sceneBoundingRect()
                    dx = int(br_g.x() - br_r.x())
                    dy = int(br_g.y() - br_r.y())
                    offsets[str(best_idx)] = [dx, dy]
            except Exception:
                continue
        if hasattr(self, 'image_state_manager'):
            self.image_state_manager.update_state(current_image, {'overlay_offsets': offsets}, save=True)
            print(f"[STATE] Persisted overlay offsets for {os.path.basename(current_image)}: {len(offsets)} entries")
    except Exception as e:
        print(f"[STATE] Failed to persist overlay offsets: {e}")

def _persist_single_overlay_offset(self, image_path: str, rect_index: int, group):
    """Persist only one overlay offset (dx,dy) for the given rectangle index.
    Avoids recomputing offsets for other overlays to prevent global shifts.
    """
    try:
        if not image_path or rect_index is None:
            return
        rects = getattr(self.image_preview_widget.viewer, 'rectangles', []) or []
        if not (0 <= int(rect_index) < len(rects)):
            return
        br_g = group.sceneBoundingRect()
        br_r = rects[int(rect_index)].sceneBoundingRect()
        dx = int(br_g.x() - br_r.x())
        dy = int(br_g.y() - br_r.y())
        
        # Always keep overlay hidden
        try:
            group.setVisible(False)
        except Exception as vis_err:
            print(f"[STATE] Error updating overlay visibility: {vis_err}")
        
        if hasattr(self, 'image_state_manager'):
            state = self.image_state_manager.get_state(image_path) or {}
            off = state.get('overlay_offsets') or {}
            off[str(int(rect_index))] = [dx, dy]
            self.image_state_manager.update_state(image_path, {'overlay_offsets': off}, save=True)
            print(f"[STATE] Persisted single overlay offset for idx={rect_index}: ({dx},{dy})")
    except Exception as e:
        print(f"[STATE] Failed to persist single overlay offset: {e}")

def _synchronize_overlay_positions_with_rectangles(self, image_path: str):
    """Force synchronization of text overlay positions with their corresponding rectangles.
    This ensures overlays appear properly positioned after state restoration.
    """
    try:
        if not image_path:
            return
        
        viewer = self.image_preview_widget.viewer
        rectangles = getattr(viewer, 'rectangles', []) or []
        
        # Get overlay groups for this image
        overlays_map = getattr(self, '_text_overlays_by_image', {}) or {}
        groups = overlays_map.get(image_path, [])
        
        if not groups or not rectangles:
            return
        
        # Load saved overlay offsets
        saved_offsets = {}
        try:
            if hasattr(self, 'image_state_manager'):
                state = self.image_state_manager.get_state(image_path) or {}
                saved_offsets = state.get('overlay_offsets', {})
        except Exception:
            saved_offsets = {}
        
        print(f"[SYNC] Synchronizing {len(groups)} overlay groups with {len(rectangles)} rectangles")
        
        # For each overlay group, find its matching rectangle and sync position
        for group in groups:
            try:
                # Try to get the region index stored on the overlay group
                region_index = getattr(group, '_overlay_region_index', None)
                target_rect = None
                
                # First try direct index match
                if region_index is not None and 0 <= region_index < len(rectangles):
                    target_rect = rectangles[region_index]
                else:
                    # Fall back to IoU-based matching
                    best_iou = 0.0
                    group_rect = group.sceneBoundingRect()
                    gx, gy, gw, gh = int(group_rect.x()), int(group_rect.y()), int(group_rect.width()), int(group_rect.height())
                    
                    for idx, rect in enumerate(rectangles):
                        rect_bounds = rect.sceneBoundingRect()
                        rx, ry, rw, rh = int(rect_bounds.x()), int(rect_bounds.y()), int(rect_bounds.width()), int(rect_bounds.height())
                        
                        # Calculate IoU
                        iou = _calculate_iou(self, [gx, gy, gw, gh], [rx, ry, rw, rh])
                        if iou > best_iou:
                            best_iou = iou
                            target_rect = rect
                            region_index = idx
                
                if target_rect is not None and region_index is not None:
                    # Get saved offset for this region
                    offset_key = str(region_index)
                    saved_offset = saved_offsets.get(offset_key, [0, 0])
                    if isinstance(saved_offset, (list, tuple)) and len(saved_offset) >= 2:
                        off_x, off_y = int(saved_offset[0]), int(saved_offset[1])
                    else:
                        off_x, off_y = 0, 0
                    
                    # Calculate desired position
                    rect_bounds = target_rect.sceneBoundingRect()
                    desired_x = int(rect_bounds.x()) + off_x
                    desired_y = int(rect_bounds.y()) + off_y
                    
                    # Get current position
                    group_bounds = group.sceneBoundingRect()
                    current_x = int(group_bounds.x())
                    current_y = int(group_bounds.y())
                    
                    # Calculate movement needed
                    dx = desired_x - current_x
                    dy = desired_y - current_y
                    
                    # Apply movement if needed
                    if abs(dx) > 1 or abs(dy) > 1:  # Only move if significant difference
                        try:
                            group.moveBy(dx, dy)
                            print(f"[SYNC] Moved overlay for region {region_index} by ({dx}, {dy})")
                        except Exception:
                            try:
                                from PySide6.QtCore import QPointF
                                group.setPos(group.pos() + QPointF(dx, dy))
                                print(f"[SYNC] Set overlay position for region {region_index} with offset ({dx}, {dy})")
                            except Exception as e:
                                print(f"[SYNC] Failed to move overlay for region {region_index}: {e}")
                    
                    # Always keep overlay hidden
                    if hasattr(group, 'setVisible'):
                        group.setVisible(False)
                    
                    # Update the region index on the group if not set
                    if not hasattr(group, '_overlay_region_index'):
                        group._overlay_region_index = region_index
                
            except Exception as e:
                print(f"[SYNC] Failed to sync overlay group: {e}")
        
        # Force scene update after all synchronization
        try:
            viewer._scene.update()
            print(f"[SYNC] Overlay synchronization completed for {os.path.basename(image_path)}")
        except Exception:
            pass
            
    except Exception as e:
        print(f"[SYNC] Overlay synchronization failed: {e}")

def _calculate_iou(self, box1, box2):
    """Calculate Intersection over Union for two bounding boxes [x, y, w, h]"""
    try:
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        left = max(x1, x2)
        top = max(y1, y2)
        right = min(x1 + w1, x2 + w2)
        bottom = min(y1 + h1, y2 + h2)
        
        if left >= right or top >= bottom:
            return 0.0
        
        intersection = (right - left) * (bottom - top)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0
    except Exception:
        return 0.0

def _show_ocr_popup(self, ocr_text: str, region_index: int = None):
    """Show OCR text in a popup dialog with edit capability"""
    try:
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QTextEdit, QHBoxLayout
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QFont
        
        dialog = QDialog(self.image_preview_widget)
        dialog.setWindowTitle("📝 OCR Recognition Result")
        dialog.resize(400, 250)
        dialog.setModal(True)
        
        # Apply dark theme styling
        dialog.setStyleSheet("""
            QDialog {
                background-color: #2d2d2d;
                color: white;
            }
            QTextEdit {
                background-color: #1e1e1e;
                color: white;
                border: 1px solid #5a9fd4;
                border-radius: 4px;
                padding: 8px;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 11pt;
                selection-background-color: #5a9fd4;
            }
            QPushButton {
                background-color: #5a9fd4;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #7bb3e0;
            }
            QPushButton#save_btn {
                background-color: #28a745;
            }
            QPushButton#save_btn:hover {
                background-color: #34ce57;
            }
            QLabel {
                color: white;
                font-weight: bold;
                margin-bottom: 8px;
            }
        """)
        
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        # Title label
        title_label = QLabel("Recognized Text (editable):")
        layout.addWidget(title_label)
        
        # Text display - EDITABLE
        text_edit = QTextEdit()
        text_edit.setPlainText(ocr_text)
        text_edit.setReadOnly(False)  # Make it editable
        
        # Apply user's selected font from manga settings
        font_name = getattr(self, 'font_style_value', 'Comic Sans MS Bold')
        if font_name == 'Default':
            font_name = 'Comic Sans MS Bold'  # Use Comic Sans Bold as default
        edit_font = QFont(font_name, 11)  # 11pt size for readability in dialog
        text_edit.setFont(edit_font)
        
        layout.addWidget(text_edit)
        
        # Button layout
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        # Save button
        save_btn = QPushButton("💾 Save")
        save_btn.setObjectName("save_btn")
        def save_changes():
            new_text = text_edit.toPlainText()
            _apply_ocr_text_edit(self, region_index, ocr_text, new_text)
            dialog.accept()
        save_btn.clicked.connect(save_changes)
        button_layout.addWidget(save_btn)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        
        dialog.exec()
        
    except Exception as e:
        print(f"[DEBUG] Error showing OCR popup: {str(e)}")

def _show_translation_popup(self, translation_data: dict, region_index: int = None):
    """Show translation in a popup dialog with edit capability"""
    try:
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QTextEdit, QHBoxLayout, QFrame
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QFont
        
        original = translation_data['original']
        translation = translation_data['translation']
        
        dialog = QDialog(self.image_preview_widget)
        dialog.setWindowTitle("🌍 Translation Result")
        dialog.resize(500, 380)
        dialog.setModal(True)
        
        # Apply dark theme styling
        dialog.setStyleSheet("""
            QDialog {
                background-color: #2d2d2d;
                color: white;
            }
            QTextEdit {
                background-color: #1e1e1e;
                color: white;
                border: 1px solid #5a9fd4;
                border-radius: 4px;
                padding: 8px;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 11pt;
                selection-background-color: #5a9fd4;
            }
            QPushButton {
                background-color: #5a9fd4;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #7bb3e0;
            }
            QPushButton#save_btn {
                background-color: #28a745;
            }
            QPushButton#save_btn:hover {
                background-color: #34ce57;
            }
            QLabel {
                color: white;
                font-weight: bold;
                margin-bottom: 4px;
            }
            QFrame {
                border: none;
            }
        """)
        
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        # Original text section
        orig_label = QLabel("Original Text (editable):")
        layout.addWidget(orig_label)
        
        orig_text = QTextEdit()
        orig_text.setPlainText(original)
        orig_text.setReadOnly(False)  # Make it editable
        orig_text.setMaximumHeight(100)
        
        # Apply user's selected font from manga settings
        font_name = getattr(self, 'font_style_value', 'Comic Sans MS Bold')
        if font_name == 'Default':
            font_name = 'Comic Sans MS Bold'  # Use Comic Sans Bold as default
        edit_font = QFont(font_name, 11)  # 11pt size for readability in dialog
        orig_text.setFont(edit_font)
        
        layout.addWidget(orig_text)
        
        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet("color: #5a9fd4;")
        layout.addWidget(separator)
        
        # Translation section
        trans_label = QLabel("Translation (editable):")
        layout.addWidget(trans_label)
        
        trans_text = QTextEdit()
        trans_text.setPlainText(translation)
        trans_text.setReadOnly(False)  # Make it editable
        trans_text.setMaximumHeight(100)
        
        # Apply user's selected font from manga settings
        font_name = getattr(self, 'font_style_value', 'Comic Sans MS Bold')
        if font_name == 'Default':
            font_name = 'Comic Sans MS Bold'  # Use Comic Sans Bold as default
        edit_font = QFont(font_name, 11)  # 11pt size for readability in dialog
        trans_text.setFont(edit_font)
        
        layout.addWidget(trans_text)
        
        # Button layout
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        # Save button
        save_btn = QPushButton("💾 Save & Update Overlay")
        save_btn.setObjectName("save_btn")
        def save_changes():
            new_original = orig_text.toPlainText()
            new_translation = trans_text.toPlainText()
            changed = False
            
            if region_index is not None:
                changed = _apply_translation_text_edit(self, region_index, original, translation, new_original, new_translation)
                
                # Refresh the text overlay for this region
                if changed:
                    try:
                        if not _persist_translation_text_edit(self, region_index, new_original, new_translation):
                            dialog.accept()
                            return
                        
                        # Animate button during async operation
                        old_text = save_btn.text()
                        save_btn.setEnabled(False)
                        save_btn.setText("Saving…")
                        
                        # Use the async method that utilizes ThreadPoolExecutor
                        # This handles processing overlay internally
                        _save_overlay_async(self, region_index, new_translation)
                    finally:
                        # Restore button state
                        try:
                            save_btn.setText(old_text)
                            save_btn.setEnabled(True)
                        except Exception:
                            pass
            
            dialog.accept()
        save_btn.clicked.connect(save_changes)
        button_layout.addWidget(save_btn)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        
        dialog.exec()
        
    except Exception as e:
        print(f"[DEBUG] Error showing translation popup: {str(e)}")

def clear_text_overlays_for_image(self, image_path: str = None):
    """Clear text overlays for a specific image (or all if no path given) with proper Qt cleanup"""
    try:
        viewer = self.image_preview_widget.viewer
        
        # Initialize overlay dictionary if not exists
        if not hasattr(self, '_text_overlays_by_image'):
            self._text_overlays_by_image = {}
        
        if image_path is None:
            # Clear all overlays from all images with proper cleanup
            for overlays in self._text_overlays_by_image.values():
                for overlay in overlays:
                    try:
                        # Remove from scene first
                        viewer._scene.removeItem(overlay)
                        # Destroy all child items explicitly
                        for child in overlay.childItems():
                            try:
                                child.setParentItem(None)
                                child.deleteLater()
                            except Exception:
                                pass
                        # Destroy the group itself
                        overlay.deleteLater()
                    except Exception:
                        pass
            self._text_overlays_by_image = {}
            print("[DEBUG] Cleared all text overlays for all images with Qt cleanup")
        else:
            # Clear overlays for specific image with proper cleanup
            if image_path in self._text_overlays_by_image:
                for overlay in self._text_overlays_by_image[image_path]:
                    try:
                        # Remove from scene first
                        viewer._scene.removeItem(overlay)
                        # Destroy all child items explicitly
                        for child in overlay.childItems():
                            try:
                                child.setParentItem(None)
                                child.deleteLater()
                            except Exception:
                                pass
                        # Destroy the group itself
                        overlay.deleteLater()
                    except Exception:
                        pass
                del self._text_overlays_by_image[image_path]
                print(f"[DEBUG] Cleared text overlays for image with Qt cleanup: {os.path.basename(image_path)}")
    except Exception as e:
        print(f"[DEBUG] Error clearing text overlays: {e}")

def show_text_overlays_for_image(self, image_path: str):
    """Keep text overlays for a specific image (but always hidden)"""
    try:
        viewer = self.image_preview_widget.viewer
        
        # Initialize overlay dictionary if not exists
        if not hasattr(self, '_text_overlays_by_image'):
            self._text_overlays_by_image = {}
        
        # Always hide all overlays (overlays exist but are invisible)
        for overlays in self._text_overlays_by_image.values():
            for overlay in overlays:
                overlay.setVisible(False)
        
        # Keep overlays for the requested image (but hidden)
        if image_path in self._text_overlays_by_image:
            # Don't show them - keep them hidden
            for overlay in self._text_overlays_by_image[image_path]:
                overlay.setVisible(False)
            # Force scene update
            viewer._scene.update()
            viewer.update()
            print(f"[DEBUG] Kept {len(self._text_overlays_by_image[image_path])} overlays hidden for image: {os.path.basename(image_path)}")
        else:
            #print(f"[DEBUG] No overlays found for image: {os.path.basename(image_path)}")
            pass
    except Exception as e:
        print(f"[DEBUG] Error hiding text overlays: {str(e)}")

def _alias_text_overlays_for_image(self, from_path: str, to_path: str):
    """Alias the overlays list from one image path to another (e.g., original -> cleaned)"""
    try:
        if not hasattr(self, '_text_overlays_by_image'):
            self._text_overlays_by_image = {}
        if from_path in self._text_overlays_by_image:
            self._text_overlays_by_image[to_path] = self._text_overlays_by_image[from_path]
            print(f"[DEBUG] Aliased overlays from {os.path.basename(from_path)} to {os.path.basename(to_path)}")
    except Exception as e:
        print(f"[DEBUG] Error aliasing overlays: {str(e)}")

def _save_position_async(self, region_index: int):
    """Save position and update overlay for a specific region using thread pool executor with microsecond locks.
    
    Enhanced version that supports parallel processing with race condition protection.
    """
    try:
        # Initialize parallel processing system if not exists
        if not hasattr(self, '_parallel_save_system'):
            _init_parallel_save_system(self, )
        
        # Submit to parallel processing queue
        self._parallel_save_system.queue_save_task(region_index)
        
    except Exception as e:
        print(f"[PARALLEL] Error in _save_position_async: {e}")
        # Fallback to single region processing
        _fallback_single_save(self, region_index)

def _schedule_source_refresh(self):
    """Instant refresh of source preview - called from completion callback."""
    try:
        print(f"[REFRESH] Starting instant preview refresh...")
        _do_source_refresh(self)
        print(f"[REFRESH] Instant refresh completed")
    except Exception as e:
        print(f"[REFRESH] Refresh failed: {e}")
        import traceback
        traceback.print_exc()

def _do_source_refresh(self):
    """Actually perform the source refresh (must be called from main thread)
    
    This reloads the rendered/translated image to show updates from auto-save position.
    """
    try:
        ipw = getattr(self, 'image_preview_widget', None)
        if not ipw:
            print(f"[REFRESH] No image_preview_widget available")
            return
            
        current_image = getattr(ipw, 'current_image_path', None)
        if not current_image:
            print(f"[REFRESH] No current_image_path available")
            return
        
        print(f"[REFRESH] Executing preview refresh for: {os.path.basename(current_image)}")
        
        # Get the rendered image path from state
        rendered_path = None
        try:
            if hasattr(self, 'image_state_manager') and self.image_state_manager:
                state = self.image_state_manager.get_state(current_image) or {}
                rendered_path = state.get('rendered_image_path')
                if rendered_path and os.path.exists(rendered_path):
                    print(f"[REFRESH] Found rendered image: {os.path.basename(rendered_path)}")
                else:
                    print(f"[REFRESH] No valid rendered image in state")
                    rendered_path = None
        except Exception as e:
            print(f"[REFRESH] Error getting rendered path from state: {e}")
        
        # If no rendered path, check the rendered images map
        if not rendered_path and hasattr(self, '_rendered_images_map'):
            rendered_path = self._rendered_images_map.get(current_image)
            if rendered_path and os.path.exists(rendered_path):
                print(f"[REFRESH] Found rendered image in map: {os.path.basename(rendered_path)}")
            else:
                rendered_path = None
        
        # If still no rendered path, search in override directory and source directory
        if not rendered_path:
            source_dir = os.path.dirname(current_image)
            source_filename = os.path.basename(current_image)
            base_name = os.path.splitext(source_filename)[0]
            
            # Check for OUTPUT_DIRECTORY override
            override_dir = None
            if hasattr(self, 'main_gui') and self.main_gui and hasattr(self.main_gui, 'config'):
                override_dir = self.main_gui.config.get('output_directory', '')
            if not override_dir:
                override_dir = os.environ.get('OUTPUT_DIRECTORY', '')
            
            # Build search paths - check override first if set
            search_paths = []
            if override_dir:
                search_paths.append(os.path.join(override_dir, f"{base_name}_translated", source_filename))
            search_paths.append(os.path.join(source_dir, f"{base_name}_translated", source_filename))
            
            for path in search_paths:
                if os.path.exists(path):
                    rendered_path = path
                    print(f"[REFRESH] Found rendered image via directory search: {os.path.basename(rendered_path)}")
                    break
        
        # Reload the appropriate image
        if rendered_path:
            print(f"[REFRESH] Loading rendered image: {rendered_path}")
            ipw.load_image(rendered_path, preserve_rectangles=True, preserve_text_overlays=False)
        else:
            print(f"[REFRESH] No rendered image found, reloading source: {current_image}")
            ipw.load_image(current_image, preserve_rectangles=True, preserve_text_overlays=True)
        
        # Rehydrate text state
        try:
            ocr_count, trans_count = _rehydrate_text_state_from_persisted(self, current_image)
            if ocr_count and hasattr(self, '_update_rectangles_with_recognition'):
                _update_rectangles_with_recognition(self, self._recognized_texts)
        except Exception as e:
            print(f"[REFRESH] Error rehydrating state: {e}")
        
        print(f"[REFRESH] Preview refresh completed")
        
    except Exception as _e:
        print(f"[REFRESH] Preview refresh failed: {_e}")
        import traceback
        traceback.print_exc()

def _init_parallel_save_system(self):
    """Initialize the parallel save processing system with ProcessPoolExecutor."""
    try:
        import threading
        import time
        from queue import Queue
        from concurrent.futures import ThreadPoolExecutor
        
        class ParallelSaveSystem:
            def __init__(self, parent):
                self.parent = parent
                self.pending_tasks = Queue()
                self.active_tasks = set()  # Track active region indices
                self.microsecond_lock = threading.Lock()  # Microsecond precision lock
                
                # Get max_workers from manga settings
                max_workers = 3  # Default - lower for threads since they're lighter
                try:
                    if hasattr(parent, 'main_gui') and hasattr(parent.main_gui, 'config'):
                        manga_settings = parent.main_gui.config.get('manga_settings', {})
                        advanced_settings = manga_settings.get('advanced', {})
                        max_workers = advanced_settings.get('max_workers', 3)
                        print(f"[PARALLEL] Using max_workers from settings: {max_workers}")
                    else:
                        print(f"[PARALLEL] Using default max_workers: {max_workers}")
                except Exception as e:
                    print(f"[PARALLEL] Error reading max_workers from settings, using default: {e}")
                
                # Use ThreadPoolExecutor for instant response (no process startup overhead)
                self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="AutoSave")
                self.processing_count = 0  # Track number of active threads
                self.count_lock = threading.Lock()  # Lock for processing count
                
                # Start the coordinator thread
                self.coordinator_thread = threading.Thread(
                    target=self._coordinate_tasks, 
                    daemon=True, 
                    name="AutoSaveCoordinator"
                )
                self.coordinator_thread.start()
                print(f"[PARALLEL] Initialized parallel save system with {max_workers} worker threads (ThreadPoolExecutor - instant response)")
            
            def queue_save_task(self, region_index: int):
                """Queue a save task for the given region index."""
                try:
                    timestamp = time.time_ns()  # Microsecond precision timestamp
                    task = {
                        'region_index': region_index,
                        'timestamp': timestamp,
                        'retry_count': 0
                    }
                    
                    with self.microsecond_lock:
                        # Check if this region is already being processed
                        if region_index in self.active_tasks:
                            print(f"[PARALLEL] Region {region_index} already being processed, skipping")
                            return False
                        
                        # Add to pending tasks
                        self.pending_tasks.put(task)
                        print(f"[PARALLEL] Queued save task for region {region_index} at {timestamp}")
                        return True
                        
                except Exception as e:
                    print(f"[PARALLEL] Error queuing task for region {region_index}: {e}")
                    return False
            
            def queue_batch_save_tasks(self, region_indices: list) -> int:
                """Queue multiple save tasks for batch processing optimization.
                
                Args:
                    region_indices: List of region indices to queue for processing
                    
                Returns:
                    Number of tasks successfully queued
                """
                if not region_indices:
                    return 0
                    
                queued_count = 0
                batch_timestamp = time.time_ns()
                
                with self.microsecond_lock:
                    for region_index in region_indices:
                        # Check if this region is already being processed
                        if region_index in self.active_tasks:
                            print(f"[PARALLEL] Region {region_index} already being processed, skipping from batch")
                            continue
                        
                        # Create task with batch timestamp
                        task = {
                            'region_index': region_index,
                            'timestamp': batch_timestamp,
                            'retry_count': 0,
                            'batch_id': batch_timestamp  # Add batch identifier
                        }
                        
                        # Add to pending tasks
                        self.pending_tasks.put(task)
                        queued_count += 1
                
                print(f"[PARALLEL] Batch queued {queued_count}/{len(region_indices)} save tasks")
                return queued_count
            
            def _coordinate_tasks(self):
                """Coordinate parallel task execution with microsecond precision."""
                print(f"[PARALLEL] Coordinator thread started")
                
                while True:
                    try:
                        # Wait for a task (blocking)
                        task = self.pending_tasks.get(timeout=1.0)
                        
                        region_index = task['region_index']
                        
                        # Acquire microsecond lock
                        with self.microsecond_lock:
                            # Double-check the region isn't already active
                            if region_index in self.active_tasks:
                                print(f"[PARALLEL] Region {region_index} became active while waiting, skipping")
                                continue
                            
                            # Mark region as active
                            self.active_tasks.add(region_index)
                            
                            # Increment processing count
                            with self.count_lock:
                                self.processing_count += 1
                                print(f"[PARALLEL] Started processing region {region_index} (active: {self.processing_count})")
                                
                                # Update button state via main thread queue (NOT directly - Qt crash)
                                try:
                                    self.parent.update_queue.put(('parallel_button_state', self.processing_count))
                                except Exception:
                                    pass
                        
                        # Submit the work to thread pool (threads can access parent directly)
                        region_idx = task['region_index']
                        
                        # Get translation text
                        trans_text = _get_translation_text_for_region(self.parent, region_idx)
                        if not trans_text:
                            print(f"[PARALLEL] No translation text for region {region_idx}, skipping")
                            # Clean up
                            with self.microsecond_lock:
                                self.active_tasks.discard(region_idx)
                                with self.count_lock:
                                    self.processing_count = max(0, self.processing_count - 1)
                            continue
                        
                        # Submit to ThreadPoolExecutor (instant - no process startup)
                        future = self.executor.submit(self._execute_save_task, region_idx, trans_text)
                        
                        # Add callback to handle completion
                        future.add_done_callback(lambda f, idx=region_idx: self._handle_task_completion(f, idx))
                        
                        # Don't wait for completion - let it run in parallel
                        
                    except Empty:
                        continue
                    except Exception as e:
                        print(f"[PARALLEL] Coordinator error: {e}")
            
            def _execute_save_task(self, region_index, trans_text):
                """Execute save task — marshal GUI update to main thread via update_queue."""
                try:
                    print(f"[PARALLEL] Queuing GUI update for region {region_index} to main thread")
                    # CRITICAL: Do NOT call Qt from this thread. Route through update_queue.
                    self.parent.update_queue.put(('parallel_gui_update', {
                        'region_index': region_index,
                        'trans_text': trans_text
                    }))
                    return {'success': True, 'region_index': region_index}
                except Exception as e:
                    print(f"[PARALLEL] Thread worker error for region {region_index}: {e}")
                    return {'success': False, 'region_index': region_index}
            
            def _handle_task_completion(self, future, region_index):
                """Handle completion of a save task from ThreadPoolExecutor."""
                try:
                    result = future.result(timeout=5.0)
                    success = result.get('success', False) if isinstance(result, dict) else False
                    
                    if not success:
                        print(f"[PARALLEL] Task failed for region {region_index}")
                        
                except Exception as e:
                    print(f"[PARALLEL] Error handling completion for region {region_index}: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    # Always clean up
                    with self.microsecond_lock:
                        self.active_tasks.discard(region_index)
                        
                        with self.count_lock:
                            self.processing_count = max(0, self.processing_count - 1)
                            print(f"[PARALLEL] Finished processing region {region_index} (active: {self.processing_count})")
                            
                            # Update button state via main thread queue (NOT directly - Qt crash)
                            try:
                                self.parent.update_queue.put(('parallel_button_state', self.processing_count))
                            except Exception:
                                pass
            
            def get_active_count(self):
                """Get the number of currently active processing tasks."""
                with self.count_lock:
                    return self.processing_count
            
            def queue_batch_save_tasks(self, region_indices: list) -> int:
                """Queue multiple save tasks for batch processing optimization.
                
                Args:
                    region_indices: List of region indices to queue for saving
                    
                Returns:
                    Number of tasks successfully queued
                """
                if not region_indices:
                    return 0
                    
                queued_count = 0
                batch_timestamp = time.time_ns()
                
                with self.microsecond_lock:
                    for region_index in region_indices:
                        # Check if this region is already being processed
                        if region_index in self.active_tasks:
                            print(f"[PARALLEL] Region {region_index} already being processed, skipping from batch")
                            continue
                        
                        # Create task with batch timestamp
                        task = {
                            'region_index': region_index,
                            'timestamp': batch_timestamp,
                            'retry_count': 0,
                            'batch_id': batch_timestamp  # Add batch identifier
                        }
                        
                        # Add to pending tasks
                        self.pending_tasks.put(task)
                        queued_count += 1
                
                print(f"[PARALLEL] Batch queued {queued_count}/{len(region_indices)} save tasks")
                return queued_count
            
            def shutdown(self, wait=False):
                """Shutdown the parallel processing system."""
                try:
                    wait_label = "waiting for tasks" if wait else "canceling queued tasks"
                    print(f"[PARALLEL] Shutting down executor ({wait_label})...")
                    self.executor.shutdown(wait=wait, cancel_futures=True)
                    print(f"[PARALLEL] Parallel save system shutdown completed")
                except Exception as e:
                    print(f"[PARALLEL] Error during shutdown: {e}")
                    pass
        
        self._parallel_save_system = ParallelSaveSystem(self)
        
    except Exception as e:
        print(f"[PARALLEL] Failed to initialize parallel save system: {e}")
        self._parallel_save_system = None

def _save_positions_batch(self, region_indices: list) -> bool:
    """Save positions for multiple regions using batch processing.
    
    Args:
        region_indices: List of region indices to save
        
    Returns:
        True if batch queuing succeeded, False if fallback needed
    """
    if not region_indices:
        print(f"[PARALLEL] No region indices provided for batch save")
        return False
        
    try:
        # Initialize parallel save system if not already done
        if not hasattr(self, '_parallel_save_system') or not self._parallel_save_system:
            _init_parallel_save_system(self, )
        
        # Check if parallel system is available
        if not self._parallel_save_system:
            print(f"[PARALLEL] Parallel system unavailable, falling back to sequential saves")
            # Fall back to sequential single saves
            for region_index in region_indices:
                _fallback_single_save(self, region_index)
            return False
        
        # Queue batch save tasks
        queued_count = self._parallel_save_system.queue_batch_save_tasks(region_indices)
        
        if queued_count > 0:
            print(f"[PARALLEL] Successfully queued {queued_count} tasks for batch processing")
            return True
        else:
            print(f"[PARALLEL] No tasks queued, using fallback")
            # Fall back to sequential saves
            for region_index in region_indices:
                _fallback_single_save(self, region_index)
            return False
            
    except Exception as e:
        print(f"[PARALLEL] Batch save failed: {e}, using fallback")
        # Fall back to sequential saves
        for region_index in region_indices:
            _fallback_single_save(self, region_index)
        return False

def _fallback_single_save(self, region_index: int):
    """Fallback to single save processing when parallel system fails."""
    try:
        print(f"[PARALLEL] Using fallback single save for region {region_index}")
        
        # Mark auto-save as in progress
        self._auto_save_in_progress = True
        _update_save_overlay_button_state(self, )
        
        # Get translation text first
        trans_text = _get_translation_text_for_region(self, region_index)
        if not trans_text:
            self._auto_save_in_progress = False
            _update_save_overlay_button_state(self, )
            return
        
        # Use thread pool executor for background processing
        def render_task():
            try:
                # Update the single text overlay
                _update_single_text_overlay(self, region_index, trans_text)
                print(f"[FALLBACK] Region {region_index} saved successfully")
                # Signal-based refresh happens automatically after file save
                return True
            except Exception as e:
                print(f"[FALLBACK] Render task failed: {e}")
                import traceback
                traceback.print_exc()
                return False
            finally:
                self._auto_save_in_progress = False
                _update_save_overlay_button_state(self, )
                # Note: Pulse effect auto-removes after 0.1s via animation finished callback
        
        # Submit to executor if available
        if hasattr(self.main_gui, 'executor') and self.main_gui.executor:
            future = self.main_gui.executor.submit(render_task)
        else:
            render_task()
        
    except Exception as e:
        print(f"[PARALLEL] Fallback single save failed: {e}")
        self._auto_save_in_progress = False
        _update_save_overlay_button_state(self, )

def _update_save_overlay_button_state(self):
    """Update the save overlay button enabled/disabled state based on auto-save progress"""
    try:
        # Check if we have access to the button
        if hasattr(self, 'image_preview_widget') and hasattr(self.image_preview_widget, 'save_overlay_btn'):
            in_progress = getattr(self, '_auto_save_in_progress', False)
            button = self.image_preview_widget.save_overlay_btn
            button.setEnabled(not in_progress)
            
            if in_progress:
                # Store original text if not already stored
                if not hasattr(button, '_original_text'):
                    button._original_text = button.text()
                button.setText("⏳")
            else:
                # Restore original text
                if hasattr(button, '_original_text'):
                    button.setText(button._original_text)
                else:
                    button.setText("💾")  # Fallback text
    except Exception:
        pass

def _update_parallel_save_button_state(self, active_count: int):
    """Update the save overlay button state for parallel processing.
    Shows different ⏳ emojis based on the number of parallel tasks.
    """
    try:
        if hasattr(self, 'image_preview_widget') and hasattr(self.image_preview_widget, 'save_overlay_btn'):
            button = self.image_preview_widget.save_overlay_btn
            
            if active_count > 0:
                # Store original text if not already stored
                if not hasattr(button, '_original_text'):
                    button._original_text = button.text()
                
                # Show different indicators based on parallel count
                if active_count == 1:
                    button.setText("⏳1")
                    button.setToolTip("Auto-saving 1 rectangle position...")
                else:
                    button.setText(f"⏳{active_count}")
                    button.setToolTip(f"Auto-saving {active_count} rectangle positions in parallel...")
                
                button.setEnabled(False)
            else:
                # Restore original text and state
                if hasattr(button, '_original_text'):
                    button.setText(button._original_text)
                else:
                    button.setText("💾")
                
                button.setToolTip("Save & Update Overlay")
                button.setEnabled(True)
    except Exception as e:
        print(f"[PARALLEL] Error updating button state: {e}")


def _update_single_text_overlay_parallel(self, region_index: int, trans_text: str) -> bool:
    """Thread-safe version of overlay update for parallel processing.
    
    This method can be called from worker threads and handles thread safety.
    """
    try:
        # Import thread-safe utilities
        import time
        from PySide6.QtCore import QMetaObject, Qt
        
        # For thread safety, we'll queue the GUI update to the main thread
        def gui_update():
            try:
                # Persist rectangles state on main thread
                if hasattr(self.image_preview_widget, '_persist_rectangles_state'):
                    self.image_preview_widget._persist_rectangles_state()
                
                # Update the overlay using the existing method (main thread only)
                return _update_single_text_overlay(self, region_index, trans_text)
            except Exception as e:
                print(f"[PARALLEL] GUI update error for region {region_index}: {e}")
                return False
        
        # Use QMetaObject to invoke on main thread
        result = [False]  # Use list to allow modification in nested function
        
        def set_result(success):
            result[0] = success
        
        # Execute on main thread and wait for completion
        try:
        # Queue the GUI update to main thread
            if hasattr(self.main_gui, '_execute_parallel_gui_update'):
                success = self.main_gui._execute_parallel_gui_update(region_index, trans_text)
            else:
                # Fallback - call directly on main thread
                success = gui_update()
            
            if success is None:
                # Fallback: use the direct method (may cause thread issues but better than failing)
                return gui_update()
            
            return bool(success)
            
        except Exception:
            # Final fallback: direct call (not thread-safe but functional)
            print(f"[PARALLEL] Using direct GUI update fallback for region {region_index}")
            return gui_update()
            
    except Exception as e:
        print(f"[PARALLEL] Error in parallel overlay update for region {region_index}: {e}")
        return False

# NOTE: Auto-save position is now simplified to only update overlays without attempting
# to update the translated output preview. Users can manually click "Save & Update Overlay"
# if they want to see the updated preview in the output tab.
# The Save & Update Overlay button is disabled during auto-save to prevent conflicts.

def _extract_render_data_for_region(self, region_index: int) -> dict:
    """Extract all GUI data needed for rendering (main thread only)"""
    try:
        current_image = self.image_preview_widget.current_image_path
        if not current_image:
            print(f"[DEBUG] No current image path")
            return None
        
        # Extract the same data that _update_single_text_overlay uses
        if not (hasattr(self, '_translation_data') and self._translation_data):
            print(f"[DEBUG] No translation data available")
            return None
        
        from manga_translator import TextRegion
        rectangles = self.image_preview_widget.viewer.rectangles
        print(f"[DEBUG] Found {len(rectangles)} rectangles and {len(self._translation_data)} translations")
        
        # Prepare dimensions and positions (same as _update_single_text_overlay)
        from PIL import Image as _PILImage
        try:
            src_w, src_h = open_image(current_image).size
        except Exception:
            src_w, src_h = (1, 1)
        
        saved_offsets = {}
        last_pos = {}
        try:
            current_state = self.image_state_manager.get_state(current_image) if hasattr(self, 'image_state_manager') else None
            if current_state:
                saved_offsets = current_state.get('overlay_offsets') or {}
                last_pos = current_state.get('last_render_positions') or {}
        except Exception:
            saved_offsets, last_pos = {}, {}
        
        # Build TextRegion objects for ALL regions (same logic as _update_single_text_overlay)
        regions = []
        for idx_key in sorted(self._translation_data.keys()):
            trans_data = self._translation_data[idx_key]
            if region_index is not None and int(idx_key) == int(region_index):
                # Edited region — use current rectangle
                if int(idx_key) < len(rectangles):
                    rect = rectangles[int(idx_key)].sceneBoundingRect()
                    sx, sy, sw, sh = int(rect.x()), int(rect.y()), int(rect.width()), int(rect.height())
                else:
                    lp = last_pos.get(str(int(idx_key)))
                    if not lp:
                        continue
                    sx, sy, sw, sh = map(int, lp)
            else:
                # Unedited region — lock to last render position if available
                lp = last_pos.get(str(int(idx_key)))
                if not lp:
                    if int(idx_key) < len(rectangles):
                        rect = rectangles[int(idx_key)].sceneBoundingRect()
                        lp = [int(rect.x()), int(rect.y()), int(rect.width()), int(rect.height())]
                    else:
                        continue
                sx, sy, sw, sh = map(int, lp)
            
            region = TextRegion(
                text=trans_data['original'],
                vertices=[(sx, sy), (sx + sw, sy), (sx + sw, sy + sh), (sx, sy + sh)],
                bounding_box=(sx, sy, sw, sh),
                confidence=1.0,
                region_type='text_block'
            )
            region.translated_text = trans_data['translation']
            regions.append(region)
        
        if not regions:
            print(f"[DEBUG] No regions to render")
            return None
        
        # Choose base image (state -> memory -> filesystem discovery)
        base_image = _resolve_cleaned_image_for_render(self, current_image)
        if base_image is None:
            base_image = current_image
            print(f"[DEBUG] Using original image as base (no cleaned image available)")
        
        # Scale regions if needed (same logic as _update_single_text_overlay)
        try:
            base_w, base_h = open_image(base_image).size
            if (src_w, src_h) != (base_w, base_h):
                sx = base_w / max(1, float(src_w))
                sy = base_h / max(1, float(src_h))
                print(f"[DEBUG] Scaling regions from src ({src_w}x{src_h}) -> base ({base_w}x{base_h}) with factors (sx={sx:.4f}, sy={sy:.4f})")
                from manga_translator import TextRegion as _TR
                scaled = []
                for r in regions:
                    x, y, w, h = r.bounding_box
                    nx = int(round(x * sx)); ny = int(round(y * sy)); nw = int(round(w * sx)); nh = int(round(h * sy))
                    v = [(nx, ny), (nx + nw, ny), (nx + nw, ny + nh), (nx, ny + nh)]
                    nr = _TR(text=r.text, vertices=v, bounding_box=(nx, ny, nw, nh), confidence=r.confidence, region_type=r.region_type)
                    nr.translated_text = r.translated_text
                    scaled.append(nr)
                regions = scaled
        except Exception as scale_err:
            print(f"[DEBUG] Region scaling skipped due to error: {scale_err}")
        
        # Determine output path (same logic as _update_single_text_overlay)
        try:
            rendered_path = getattr(self.image_preview_widget, 'current_translated_path', None)
            if not rendered_path and hasattr(self, 'image_state_manager') and self.image_state_manager:
                st = self.image_state_manager.get_state(current_image) or {}
                rendered_path = st.get('rendered_image_path')
            if not rendered_path and hasattr(self, '_rendered_images_map'):
                rendered_path = self._rendered_images_map.get(current_image)
        except Exception:
            rendered_path = None
        output_path = rendered_path if (rendered_path and os.path.exists(os.path.dirname(rendered_path))) else None
        
        return {
            'current_image': current_image,
            'regions': regions,
            'base_image': base_image,
            'output_path': output_path
        }
        
    except Exception as e:
        print(f"[DEBUG] Failed to extract render data: {e}")
        import traceback
        traceback.print_exc()
        return None

@Slot(str)
def _on_save_progress(self, message: str):
    """Handle progress updates from worker"""
    print(f"[PROGRESS] {message}")

@Slot(bool, int, str)
def _on_save_position_finished(self, success: bool, region_index: int, rendered_path: str):
    """Handle completion from background worker (main thread)"""
    try:
        print(f"[DEBUG] Save position finished - success: {success}, region: {region_index}")
        
        if success and rendered_path:
            # Store the rendered image path for reference and refresh the source preview
            try:
                print(f"[DEBUG] Rendered image saved: {os.path.basename(rendered_path)}")
                if hasattr(self.image_preview_widget, 'current_translated_path'):
                    self.image_preview_widget.current_translated_path = rendered_path
                # Refresh source preview to show the newly rendered image
                if hasattr(self.image_preview_widget, 'current_image_path'):
                    from PySide6.QtCore import QTimer
                    QTimer.singleShot(500, lambda: self.image_preview_widget.load_image(
                        self.image_preview_widget.current_image_path, 
                        preserve_rectangles=True, 
                        preserve_text_overlays=True
                    ))
                print(f"[DEBUG] Source preview refresh scheduled")
            except Exception as e:
                print(f"[DEBUG] Error handling rendered output: {e}")
        
        print(f"[DEBUG] Save Position completed for region {region_index}")
        
    except Exception as err:
        print(f"[DEBUG] Save Position completion failed: {err}")
        import traceback
        traceback.print_exc()
    finally:
        # Always remove processing overlay
        _remove_processing_overlay(self, )

def _save_overlay_async(self, region_index: int = 0, new_translation: str = "", update_all_regions: bool = False):
    """Save & Update Overlay functionality using ThreadPoolExecutor on main thread.
    
    Args:
        region_index: Region index to update (default 0 for full re-render)
        new_translation: Specific translation text to use (empty string for original behavior)
        update_all_regions: If True, update all regions with current settings (not just one)
    """
    print(f"\n{'='*80}")
    print(f"[DEBUG] _save_overlay_async: METHOD ENTRY")
    print(f"[DEBUG] Args: region_index={region_index}, new_translation='{new_translation}', update_all_regions={update_all_regions}")
    print(f"{'='*80}\n")
    
    try:
        print(f"[DEBUG] Save & Update Overlay triggered for region {region_index}, translation='{new_translation}', update_all={update_all_regions}")
        
        # Show processing overlay immediately on main thread
        _add_processing_overlay(self, )
        
        def _save_overlay_task():
            """The actual save overlay task - runs via executor but stays on the main thread"""
            print(f"[DEBUG] _save_overlay_task: TASK FUNCTION ENTRY")
            try:
                print(f"[DEBUG] Save & Update Overlay task executing for region {region_index}")
                
                # Persist rectangles state
                try:
                    if hasattr(self.image_preview_widget, '_persist_rectangles_state'):
                        print(f"[DEBUG] Persisting rectangles state...")
                        self.image_preview_widget._persist_rectangles_state()
                        print(f"[DEBUG] Rectangles state persisted successfully")
                    else:
                        print(f"[DEBUG] No _persist_rectangles_state method available")
                except Exception as e:
                    print(f"[DEBUG] Failed to persist rectangles state: {e}")
                
                # Call _update_single_text_overlay directly with the provided parameters
                # This matches the original behavior exactly
                print(f"[DEBUG] Calling _update_single_text_overlay({region_index}, '{new_translation}', update_all={update_all_regions})")
                _update_single_text_overlay(self, region_index, new_translation, update_all_regions=update_all_regions)
                print(f"[DEBUG] _update_single_text_overlay call completed successfully")
                print(f"[DEBUG] Save & Update Overlay task completed for region {region_index}")
                return True
                
            except Exception as e:
                print(f"[DEBUG] Save & Update Overlay task failed for region {region_index}: {e}")
                import traceback
                print(f"[DEBUG] Traceback: {traceback.format_exc()}")
                return False
        
        # Check executor availability with detailed logging
        has_main_gui = hasattr(self, 'main_gui')
        has_executor_attr = has_main_gui and hasattr(self.main_gui, 'executor')
        executor_exists = has_executor_attr and self.main_gui.executor is not None
        
        print(f"[DEBUG] Executor check: has_main_gui={has_main_gui}, has_executor_attr={has_executor_attr}, executor_exists={executor_exists}")
        
        if executor_exists:
            print(f"[DEBUG] Using ThreadPoolExecutor for overlay task")
            future = self.main_gui.executor.submit(_save_overlay_task)
            print(f"[DEBUG] Task submitted to executor (fire-and-forget for responsiveness)")
            # Don't wait for completion - fire and forget for responsiveness
            from PySide6.QtCore import QTimer
            
            # Refresh source preview to show updated translated/cleaned image
            def _refresh_source_preview():
                try:
                    ipw = getattr(self, 'image_preview_widget', None)
                    if ipw and getattr(ipw, 'current_image_path', None):
                        ipw.load_image(ipw.current_image_path, preserve_rectangles=True, preserve_text_overlays=True)
                except Exception as _e:
                    print(f"[DEBUG] Source preview refresh failed: {_e}")
            QTimer.singleShot(1200, _refresh_source_preview)
            QTimer.singleShot(3000, _refresh_source_preview)
        else:
            print(f"[DEBUG] No executor available, running save overlay synchronously")
            _save_overlay_task()
            # Immediately refresh view since it was synchronous
            from PySide6.QtCore import QTimer
            def _refresh_source_preview_sync():
                try:
                    ipw = getattr(self, 'image_preview_widget', None)
                    if ipw and getattr(ipw, 'current_image_path', None):
                        ipw.load_image(ipw.current_image_path, preserve_rectangles=True, preserve_text_overlays=True)
                except Exception as _e:
                    print(f"[DEBUG] Source preview refresh (sync) failed: {_e}")
            QTimer.singleShot(600, _refresh_source_preview_sync)
        
        print(f"[DEBUG] _save_overlay_async: METHOD COMPLETION")
        
    except Exception as err:
        print(f"[DEBUG] Save & Update Overlay failed to start for region {region_index}: {err}")
        import traceback
        print(f"[DEBUG] Method error traceback: {traceback.format_exc()}")
    finally:
        # Always remove the processing overlay
        try:
            print(f"[DEBUG] Removing processing overlay...")
            _remove_processing_overlay(self, )
            print(f"[DEBUG] Processing overlay removed successfully")
        except Exception as e:
            print(f"[DEBUG] Failed to remove processing overlay: {e}")

def _start_output_refresh_check(self):
    """Start periodic checking for output image updates"""
    try:
        from PySide6.QtCore import QTimer
        # Check every 1 second for up to 10 seconds
        if not hasattr(self, '_output_refresh_timer'):
            self._output_refresh_timer = QTimer()
            self._output_refresh_timer.setSingleShot(False)
            self._output_refresh_timer.timeout.connect(self._check_and_refresh_output)
        
        # Store the check start time and count
        import time
        self._output_refresh_start_time = time.time()
        self._output_refresh_count = 0
        
        self._output_refresh_timer.start(2000)  # Check every 2 seconds (reduced frequency)
        print(f"[DEBUG] Started output refresh checking timer")
    except Exception as e:
        print(f"[DEBUG] Error starting output refresh check: {e}")

def _check_and_refresh_output(self):
    """Periodic check for output image updates"""
    try:
        import time
        current_time = time.time()
        elapsed = current_time - getattr(self, '_output_refresh_start_time', current_time)
        self._output_refresh_count = getattr(self, '_output_refresh_count', 0) + 1
        
        # Stop checking after 10 seconds or 10 attempts
        if elapsed > 10.0 or self._output_refresh_count > 10:
            if hasattr(self, '_output_refresh_timer'):
                self._output_refresh_timer.stop()
            print(f"[DEBUG] Stopped output refresh checking after {elapsed:.1f}s and {self._output_refresh_count} attempts")
            return
        
        # Try to refresh the output
        if _refresh_output_tab(self, ):
            # Success - stop checking
            if hasattr(self, '_output_refresh_timer'):
                self._output_refresh_timer.stop()
            print(f"[DEBUG] Output refreshed successfully, stopped checking")
            
    except Exception as e:
        print(f"[DEBUG] Error in output refresh check: {e}")

def _refresh_output_tab(self) -> bool:
    """Refresh the output tab with the latest rendered image"""
    try:
        current_image_path = getattr(self.image_preview_widget, 'current_image_path', None)
        if not current_image_path:
            return False
        
        # Look for rendered image in the expected location
        source_dir = os.path.dirname(current_image_path)
        source_filename = os.path.basename(current_image_path)
        
        # Check various possible locations for translated images
        possible_paths = [
            # 3_translated folder
            os.path.join(source_dir, "3_translated", source_filename),
            # isolated folder
            os.path.join(source_dir, f"{os.path.splitext(source_filename)[0]}_translated", source_filename),
            # same directory with _translated suffix
            os.path.join(source_dir, f"{os.path.splitext(source_filename)[0]}_translated{os.path.splitext(source_filename)[1]}")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                # Check if this file is newer than what we currently have loaded
                current_translated = getattr(self.image_preview_widget, 'current_translated_path', None)
                if current_translated != path or _is_file_newer(self, path, current_translated):
                    print(f"[DEBUG] Refreshing output tab with: {os.path.basename(path)}")
                    self.image_preview_widget.output_viewer.load_image(path)
                    self.image_preview_widget.current_translated_path = path
                    return True
        
        return False
        
    except Exception as e:
        print(f"[DEBUG] Error refreshing output tab: {e}")
        return False

def _is_file_newer(self, file_path: str, reference_path: str) -> bool:
    """Check if file_path is newer than reference_path"""
    try:
        if not reference_path or not os.path.exists(reference_path):
            return True  # New file is always "newer" than non-existent reference
        
        import os
        file_mtime = os.path.getmtime(file_path)
        ref_mtime = os.path.getmtime(reference_path)
        return file_mtime > ref_mtime
    except Exception:
        return True  # Assume newer on error

def _load_rendered_image_to_output_tab(self, rendered_pil, output_path, switch_tab=True):
    """Load rendered image into the output tab - must be called on main thread"""
    print(f"[GUI] === _load_rendered_image_to_output_tab CALLED ===")
    print(f"[GUI] output_path: {output_path}")
    print(f"[GUI] switch_tab: {switch_tab}")
    try:
        print(f"[GUI] Loading rendered image into output tab: {os.path.basename(output_path)}")
        
        # Check current thread for debugging
        from PySide6.QtCore import QThread
        from PySide6.QtWidgets import QApplication
        
        current_thread = QThread.currentThread()
        main_thread = QApplication.instance().thread() if QApplication.instance() else None
        print(f"[GUI] Thread check: current={current_thread}, main={main_thread}, same={current_thread == main_thread}")
        
        # Display in output viewer (using correct load_image method)
        try:
            if hasattr(self.image_preview_widget, 'output_viewer') and self.image_preview_widget.output_viewer:
                # Use the same method as _check_and_load_translated_output
                self.image_preview_widget.output_viewer.load_image(output_path)
                # Store the translated image path
                self.image_preview_widget.current_translated_path = output_path
                print(f"[GUI] Successfully loaded image into output viewer using load_image")
            else:
                print(f"[GUI] No output_viewer available")
        except Exception as output_err:
            print(f"[GUI] Error loading image to output viewer: {output_err}")
            import traceback
            traceback.print_exc()
        
        
    except Exception as e:
        print(f"[GUI] Error in _load_rendered_image_to_output_tab: {e}")
        import traceback
        traceback.print_exc()

def _load_save_position_output(self):
    """Helper method to load rendered output after save position completes - runs on main thread"""
    try:
        current_image_path = getattr(self.image_preview_widget, 'current_image_path', None)
        if not current_image_path:
            print(f"[DEBUG] Save Position Output: No current image path")
            return
        
        # Look for rendered image in the expected location
        source_dir = os.path.dirname(current_image_path)
        source_filename = os.path.basename(current_image_path)
        base_name = os.path.splitext(source_filename)[0]
        
        # Check for OUTPUT_DIRECTORY override (prefer config over env var)
        override_dir = None
        if hasattr(self, 'main_gui') and self.main_gui and hasattr(self.main_gui, 'config'):
            override_dir = self.main_gui.config.get('output_directory', '')
        if not override_dir:
            override_dir = os.environ.get('OUTPUT_DIRECTORY', '')
        
        # Build list of possible paths
        # STRICT: If OUTPUT_DIRECTORY is set, ONLY check there (no source fallback)
        possible_paths = []
        if override_dir:
            # Check override directory ONLY
            possible_paths.extend([
                os.path.join(override_dir, f"{base_name}_translated", source_filename),
                os.path.join(override_dir, "3_translated", source_filename),
            ])
            print(f"[DEBUG] Save Position Output: Using OUTPUT_DIRECTORY ONLY: {override_dir}")
        else:
            # No override, check source directory
            possible_paths.extend([
                # 3_translated folder
                os.path.join(source_dir, "3_translated", source_filename),
                # isolated folder
                os.path.join(source_dir, f"{base_name}_translated", source_filename),
                # same directory with _translated suffix
                os.path.join(source_dir, f"{base_name}_translated{os.path.splitext(source_filename)[1]}")
            ])
        
        print(f"[DEBUG] Save Position Output: Looking for rendered images at:")
        for path in possible_paths:
            print(f"[DEBUG] Save Position Output:   {path} - exists: {os.path.exists(path)}")
            if os.path.exists(path):
                print(f"[DEBUG] Save Position Output: Found rendered image, loading into output viewer...")
                self.image_preview_widget.output_viewer.load_image(path)
                self.image_preview_widget.current_translated_path = path
                # REMOVED: Don't auto-switch tabs - let user manually switch
                # if hasattr(self.image_preview_widget, 'viewer_tabs'):
                #     self.image_preview_widget.viewer_tabs.setCurrentIndex(1)  # Switch to output tab
                print(f"[DEBUG] Save Position Output: Successfully loaded {os.path.basename(path)} into output viewer")
                return
        
        print(f"[DEBUG] Save Position Output: No rendered image found in expected locations")
        
    except Exception as e:
        print(f"[DEBUG] Save Position Output: Error loading rendered output: {e}")
        import traceback
        traceback.print_exc()

@Slot()
def _load_rendered_output_direct(self):
    """Direct method to find and load rendered image into output tab - same as working button approach"""
    try:
        current_image_path = getattr(self.image_preview_widget, 'current_image_path', None)
        if not current_image_path:
            print(f"[DIRECT] No current image path")
            return
        
        # Look for rendered image in the expected location
        source_dir = os.path.dirname(current_image_path)
        source_filename = os.path.basename(current_image_path)
        base_name = os.path.splitext(source_filename)[0]
        
        # Check for OUTPUT_DIRECTORY override (prefer config over env var)
        override_dir = None
        if hasattr(self, 'main_gui') and self.main_gui and hasattr(self.main_gui, 'config'):
            override_dir = self.main_gui.config.get('output_directory', '')
        if not override_dir:
            override_dir = os.environ.get('OUTPUT_DIRECTORY', '')
        
        # Build list of possible paths
        # STRICT: If OUTPUT_DIRECTORY is set, ONLY check there (no source fallback)
        possible_paths = []
        if override_dir:
            # Check override directory ONLY
            possible_paths.extend([
                os.path.join(override_dir, f"{base_name}_translated", source_filename),
                os.path.join(override_dir, "3_translated", source_filename),
            ])
            print(f"[DIRECT] Using OUTPUT_DIRECTORY ONLY: {override_dir}")
        else:
            # No override, check source directory
            possible_paths.extend([
                # 3_translated folder
                os.path.join(source_dir, "3_translated", source_filename),
                # isolated folder
                os.path.join(source_dir, f"{base_name}_translated", source_filename),
                # same directory with _translated suffix
                os.path.join(source_dir, f"{base_name}_translated{os.path.splitext(source_filename)[1]}")
            ])
        
        print(f"[DIRECT] Looking for rendered images at:")
        for path in possible_paths:
            print(f"[DIRECT]   {path} - exists: {os.path.exists(path)}")
            if os.path.exists(path):
                print(f"[DIRECT] Found rendered image, loading into output viewer...")
                self.image_preview_widget.output_viewer.load_image(path)
                self.image_preview_widget.current_translated_path = path
                # REMOVED: Don't auto-switch tabs - let user manually switch
                # if hasattr(self.image_preview_widget, 'viewer_tabs'):
                #     self.image_preview_widget.viewer_tabs.setCurrentIndex(1)  # Switch to output tab
                print(f"[DIRECT] Successfully loaded {os.path.basename(path)} into output viewer")
                return
        
        print(f"[DIRECT] No rendered image found in expected locations")
        
    except Exception as e:
        print(f"[DIRECT] Error in _load_rendered_output_direct: {e}")
        import traceback
        traceback.print_exc()

def _add_text_overlay_to_viewer(self, translated_texts: list):
    """Add translated text as graphics items overlay on the viewer
    
    Overlays are hidden by default if at original position (overlaps with rendered output).
    When user moves blue rectangles (auto-save position), overlays move with them and become visible.
    """
    try:
        from PySide6.QtWidgets import QGraphicsTextItem, QGraphicsRectItem
        from PySide6.QtCore import QRectF
        from PySide6.QtGui import QColor, QBrush, QPen, QFont
        
        viewer = self.image_preview_widget.viewer
        
        # Get current image path
        current_image = self.image_preview_widget.current_image_path
        if not current_image:
            print("[DEBUG] No current image path, cannot add overlays")
            return
        
        # Initialize overlay dictionary if not exists
        if not hasattr(self, '_text_overlays_by_image'):
            self._text_overlays_by_image = {}
        
        # Clear any existing overlays for this specific image with proper Qt cleanup
        if current_image in self._text_overlays_by_image:
            for overlay in self._text_overlays_by_image[current_image]:
                try:
                    # Remove from scene
                    viewer._scene.removeItem(overlay)
                    # Destroy all child items to free memory
                    for child in overlay.childItems():
                        try:
                            child.setParentItem(None)
                            child.deleteLater()
                        except Exception:
                            pass
                    # Destroy the group itself
                    overlay.deleteLater()
                except Exception:
                    pass
        
        # Create new list for this image's overlays
        self._text_overlays_by_image[current_image] = []
        
        # Load any saved overlay offsets for this image
        saved_offsets = {}
        try:
            if hasattr(self, 'image_state_manager') and self.image_state_manager:
                st = self.image_state_manager.get_state(current_image) or {}
                saved_offsets = st.get('overlay_offsets') or {}
        except Exception:
            saved_offsets = {}
        
        # Get manga rendering settings
        manga_settings = _get_manga_rendering_settings(self, )
        
        # Source tab overlays should not force any background opacity
        try:
            manga_settings['show_background'] = False
            manga_settings['bg_opacity'] = 0
        except Exception:
            pass
        
        
        for i, result in enumerate(translated_texts):
            try:
                bbox = result.get('bbox')
                translation = _manga_output_text(result.get('translation'))
                if not translation.strip():
                    continue
                region_index = (result.get('original', {}) or {}).get('region_index', i)
                
                # Prefer current BLUE rectangle geometry if available; fallback to bbox
                x = y = w = h = None
                original_bbox = bbox  # Store original bbox for overlap detection
                try:
                    rects = getattr(self.image_preview_widget.viewer, 'rectangles', []) or []
                    if 0 <= int(region_index) < len(rects):
                        br = rects[int(region_index)].sceneBoundingRect()
                        x, y, w, h = int(br.x()), int(br.y()), int(br.width()), int(br.height())
                except Exception:
                    pass
                if (x is None or y is None or w is None or h is None) and bbox and len(bbox) >= 4:
                    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                
                if x is not None and w is not None and h is not None and w > 0 and h > 0:
                    
                    # Create background rectangle if enabled in settings
                    bg_rect = None
                    if manga_settings.get('show_background', True):
                        bg_rect = _create_background_shape(self, x, y, w, h, manga_settings)
                        if bg_rect:
                            bg_rect.setZValue(10)  # Above image, below text
                            viewer._scene.addItem(bg_rect)
                    
                    # Apply text formatting
                    text = translation.upper() if manga_settings.get('force_caps', False) else translation
                    
                    # Create text item with proper manga text rendering
                    text_item, final_font_size = _create_manga_text_item(self, text, x, y, w, h, manga_settings)
                    if text_item is None:
                        # Clean up orphan bg if created
                        try:
                            if bg_rect:
                                viewer._scene.removeItem(bg_rect)
                        except Exception:
                            pass
                        continue
                    
                    # Add text to scene (needed before grouping)
                    viewer._scene.addItem(text_item)
                    
                    # Make text item completely non-interactive
                    try:
                        from PySide6.QtCore import Qt
                        from PySide6.QtWidgets import QGraphicsItem
                        text_item.setAcceptedMouseButtons(Qt.NoButton)
                        text_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
                        text_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
                        text_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsFocusable, False)
                        text_item.setAcceptHoverEvents(False)
                    except Exception:
                        pass
                    
                    # Create a transparent rectangle overlay that covers the text and forwards clicks
                    overlay_rect = QGraphicsRectItem(x, y, w, h)
                    # Keep overlay rect in sync with rectangle moves by storing size
                    try:
                        overlay_rect._overlay_bbox_size = (w, h)
                    except Exception:
                        pass
                    overlay_rect.setBrush(QBrush(QColor(0, 0, 0, 0)))  # Completely transparent
                    overlay_rect.setPen(QPen(QColor(0, 0, 0, 0)))  # No border
                    overlay_rect.setZValue(20)  # Above everything else to capture clicks
                    
                    # Store the region index on the overlay for identification
                    overlay_rect.region_index = region_index
                    
                    # Make text item completely non-interactive to avoid conflicts
                    try:
                        text_item.setAcceptedMouseButtons(Qt.NoButton)
                        text_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
                        text_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
                        text_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsFocusable, False)
                        text_item.setAcceptHoverEvents(False)
                    except Exception:
                        pass
                    
                    viewer._scene.addItem(overlay_rect)
                    
                    # Group background + text + overlay so they stay together
                    group_items = [text_item, overlay_rect]
                    if bg_rect:
                        group_items.insert(0, bg_rect)  # bg_rect first (lowest z-order)
                        # Make bg_rect completely non-interactive
                        try:
                            from PySide6.QtCore import Qt
                            from PySide6.QtWidgets import QGraphicsItem
                            bg_rect.setAcceptedMouseButtons(Qt.NoButton)
                            bg_rect.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
                            bg_rect.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
                            bg_rect.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsFocusable, False)
                            bg_rect.setAcceptHoverEvents(False)
                        except Exception:
                            pass
                    
                    group = viewer._scene.createItemGroup(group_items)
                    group.setZValue(12)
                    # Store references for later reflowing/resizing
                    try:
                        group._overlay_text_item = text_item
                        group._overlay_bg_item = bg_rect
                        group._overlay_original_text = translation
                    except Exception:
                        pass
                    from PySide6.QtWidgets import QGraphicsItem
                    # Disable user interaction on overlays; movement controlled by rectangles only
                    # But allow child items (like text) to handle mouse events
                    try:
                        from PySide6.QtCore import Qt
                        group.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
                        group.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
                        # Don't block mouse buttons on group level - let child items handle them
                        # group.setAcceptedMouseButtons(Qt.NoButton)
                    except Exception:
                        pass
                    
                    # Attach metadata for sync from rectangle moves
                    group._overlay_region_index = region_index
                    group._overlay_bbox_size = (w, h)
                    group._overlay_image_path = current_image
                    
                    # Store original bbox for overlap detection
                    group._overlay_original_bbox = original_bbox
                    
                    # Apply saved offset for this region if present
                    has_offset = False
                    try:
                        off = None
                        # support str and int keys
                        if str(int(region_index)) in saved_offsets:
                            off = saved_offsets[str(int(region_index))]
                        elif int(region_index) in saved_offsets:
                            off = saved_offsets[int(region_index)]
                        if off and len(off) >= 2:
                            dx_off, dy_off = int(off[0]), int(off[1])
                            if dx_off != 0 or dy_off != 0:
                                group.moveBy(dx_off, dy_off)
                                has_offset = True
                    except Exception:
                        pass
                    
                    # Always hide text overlays (keep them hidden)
                    try:
                        group.setVisible(False)
                        #print(f"[DEBUG] Hiding overlay for region {region_index} - text overlays kept hidden")
                    except Exception as hide_err:
                        print(f"[DEBUG] Error setting overlay visibility: {hide_err}")
                    
                    # Monkey-patch mouse release remains (harmless since overlays are not movable)
                    original_release = group.mouseReleaseEvent
                    def _on_release(ev, grp=group):
                        try:
                            original_release(ev)
                        except Exception:
                            pass
                        try:
                            # Compute new top-left from scene bounding rect
                            br = grp.sceneBoundingRect()
                            new_x, new_y = int(br.x()), int(br.y())
                            w_, h_ = grp._overlay_bbox_size
                            # Update viewer rectangle for this region if available
                            rects = getattr(self.image_preview_widget.viewer, 'rectangles', [])
                            idx = int(grp._overlay_region_index) if grp._overlay_region_index is not None else -1
                            if 0 <= idx < len(rects):
                                from PySide6.QtCore import QRectF as _QRectF
                                rects[idx].setRect(_QRectF(new_x, new_y, w_, h_))
                                # Also trigger a scene update so handles refresh
                                try:
                                    self.image_preview_widget.viewer._scene.update()
                                except Exception:
                                    pass
                            # Persist new rectangles state
                            try:
                                if hasattr(self.image_preview_widget, '_persist_rectangles_state'):
                                    self.image_preview_widget._persist_rectangles_state()
                            except Exception:
                                pass
                        except Exception as move_err:
                            print(f"[DEBUG] Overlay move update failed: {move_err}")
                    group.mouseReleaseEvent = _on_release
                    
                    # Track overlay group for cleanup
                    self._text_overlays_by_image[current_image].append(group)
                    
                    print(f"[DEBUG] Added text overlay at ({x},{y}) with font size {final_font_size}: region={region_index}, text='{translation[:30]}...'")
            
            except Exception as text_error:
                print(f"[DEBUG] Error adding text overlay: {str(text_error)}")
                import traceback
                print(f"[DEBUG] Text overlay traceback: {traceback.format_exc()}")
                continue
        
        overlay_count = len(self._text_overlays_by_image.get(current_image, []))
        print(f"[DEBUG] Added {overlay_count} text overlay items for image: {os.path.basename(current_image)}")
        
        # Force scene update to ensure overlays are visible
        viewer._scene.update()
        print(f"[DEBUG] Forced scene update")
        
    except Exception as e:
        print(f"[DEBUG] Error adding text overlays: {str(e)}")
        import traceback
        print(f"[DEBUG] Traceback: {traceback.format_exc()}")

def _add_text_overlay_for_region(self, region_index: int, original_text: str, translation: str, bbox: list = None):
    """Add or replace a single text overlay for the given region on the main thread.
    Does NOT clear overlays for other regions.
    """
    try:
        from PySide6.QtWidgets import QGraphicsRectItem
        from PySide6.QtGui import QColor, QBrush, QPen
        from PySide6.QtCore import QRectF
        
        viewer = self.image_preview_widget.viewer
        current_image = getattr(self.image_preview_widget, 'current_image_path', None)
        if not current_image:
            return
        
        # Init tracking map
        if not hasattr(self, '_text_overlays_by_image'):
            self._text_overlays_by_image = {}
        if current_image not in self._text_overlays_by_image:
            self._text_overlays_by_image[current_image] = []
        
        # Remove existing overlay for this region with proper Qt cleanup
        to_remove = []
        for grp in list(self._text_overlays_by_image[current_image]):
            if getattr(grp, '_overlay_region_index', None) == int(region_index):
                to_remove.append(grp)
        for grp in to_remove:
            try:
                # Remove from scene
                viewer._scene.removeItem(grp)
                # Destroy all child items
                for child in grp.childItems():
                    try:
                        child.setParentItem(None)
                        child.deleteLater()
                    except Exception:
                        pass
                # Destroy the group
                grp.deleteLater()
                # Remove from tracking list
                self._text_overlays_by_image[current_image].remove(grp)
            except Exception:
                pass
        
        # Determine geometry from current rectangle if present
        x = y = w = h = None
        try:
            rects = getattr(viewer, 'rectangles', []) or []
            if 0 <= int(region_index) < len(rects):
                br = rects[int(region_index)].sceneBoundingRect()
                x, y, w, h = int(br.x()), int(br.y()), int(br.width()), int(br.height())
        except Exception:
            pass
        if (x is None or y is None or w is None or h is None) and bbox and len(bbox) >= 4:
            x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        if x is None or w is None or h is None or w <= 0 or h <= 0:
            return
        
        # Render text item using existing helper (uses current GUI settings)
        settings = _get_manga_rendering_settings(self, ) or {}
        try:
            settings['show_background'] = False
            settings['bg_opacity'] = 0
        except Exception:
            pass
        text_item, _ = _create_manga_text_item(self, translation, x, y, w, h, settings)
        if text_item is None:
            return
        viewer._scene.addItem(text_item)
        
        # Transparent overlay rect for click passthrough/context
        overlay_rect = QGraphicsRectItem(x, y, w, h)
        overlay_rect.setBrush(QBrush(QColor(0, 0, 0, 0)))
        overlay_rect.setPen(QPen(QColor(0, 0, 0, 0)))
        overlay_rect.setZValue(20)
        overlay_rect.region_index = int(region_index)
        viewer._scene.addItem(overlay_rect)
        
        # Optional background (disabled for source view)
        bg_rect = None
        
        # Group items and tag metadata
        group_items = [text_item, overlay_rect]
        group = viewer._scene.createItemGroup(group_items)
        group.setZValue(12)
        try:
            group._overlay_text_item = text_item
            group._overlay_bg_item = bg_rect
            group._overlay_original_text = translation
            group._overlay_region_index = int(region_index)
            group._overlay_bbox_size = (w, h)
            group._overlay_image_path = current_image
        except Exception:
            pass
        
        self._text_overlays_by_image[current_image].append(group)
        viewer._scene.update()
    except Exception:
        pass

def _get_manga_rendering_settings(self) -> dict:
    """Stub - overlays are hidden, rendering happens in manga_translator.py"""
    return {}

def show_recognized_overlays_for_image(self, image_path: str):
    """Restore recognized (OCR) overlays for the given image and attach tooltips.
    - Draw detection rectangles if not present
    - Apply recognition tooltips/updates if recognized_texts exist in state
    """
    try:
        if not hasattr(self, 'image_state_manager') or not image_path:
            return
        state = self.image_state_manager.get_state(image_path)
        if not state:
            return
        # Ensure rectangles exist; if not, draw detection regions first
        need_draw = False
        try:
            rects = getattr(self.image_preview_widget.viewer, 'rectangles', [])
            need_draw = (not rects)
        except Exception:
            need_draw = True
        if need_draw:
            regions = state.get('detection_regions') or []
            if regions:
                self._current_regions = regions
                try:
                    if hasattr(self.image_preview_widget.viewer, 'clear_rectangles'):
                        self.image_preview_widget.viewer.clear_rectangles()
                except Exception:
                    pass
                _draw_detection_boxes_on_preview(self, )
        # Apply recognition data if available
        recognized_texts = state.get('recognized_texts') or []
        if recognized_texts:
            _update_rectangles_with_recognition(self, recognized_texts)
    except Exception as e:
        print(f"[DEBUG] Error restoring recognized overlays: {e}")

def _create_manga_text_item(self, text: str, x: int, y: int, w: int, h: int, settings: dict):
    """Stub - overlays are hidden, actual rendering happens in manga_translator.py"""
    return None, 0


def _calculate_auto_font_size(self, text: str, width: int, height: int, settings: dict) -> int:
    """Stub - redundant, manga_translator.py has the real algorithm"""
    return 12


def _text_fits_in_bounds(self, text: str, width: int, height: int, font, settings: dict) -> bool:
    """Stub - redundant"""
    return True

def _wrap_text_for_bubble(self, text: str, max_width: int, font, settings: dict) -> str:
    """Stub - redundant"""
    return text


def _create_background_shape(self, x: int, y: int, w: int, h: int, settings: dict):
    """Stub - overlays are hidden"""
    return None

def _wrap_text_to_width(self, text: str, max_width: int, font) -> str:
    """Stub - redundant"""
    return text

def _disable_workflow_buttons(self, exclude=None, show_stop_button=True):
    """Disable all workflow buttons to prevent concurrent operations.
    
    Args:
        exclude: Button name to exclude from disabling (e.g., 'translate' keeps translate enabled)
        show_stop_button: Whether to show the workflow stop button (False for Start Translation which has its own stop)
    """
    try:
        # Mark that a workflow operation is actively running.
        # This prevents _set_translation_buttons_waiting from re-enabling
        # buttons or hiding the stop button while the operation is in progress.
        self._workflow_operation_active = True
        
        ipw = self.image_preview_widget if hasattr(self, 'image_preview_widget') else None
        if not ipw:
            return
        
        buttons = [
            ('detect_btn', 'Detect Text'),
            ('clean_btn', 'Clean'),
            ('recognize_btn', 'Recognize Text'),
            ('translate_btn', 'Translate'),
            ('import_ocr_btn', '📥 Import'),
            ('export_ocr_btn', '📤 Export'),
            ('translate_all_btn', 'Translate All'),
        ]
        
        for btn_name, _ in buttons:
            if exclude and btn_name == exclude:
                continue
            if hasattr(ipw, btn_name):
                btn = getattr(ipw, btn_name)
                btn.setEnabled(False)
        
        # Disable editing tools during processing
        editing_buttons = ['save_overlay_btn', 'delete_btn', 'clear_boxes_btn', 'box_draw_btn', 'circle_draw_btn', 'lasso_btn']
        for btn_name in editing_buttons:
            if hasattr(ipw, btn_name):
                btn = getattr(ipw, btn_name)
                btn.setEnabled(False)
        
        # Also disable start_button in manga_integration if it exists
        if exclude != 'start_button' and hasattr(self, 'start_button') and self.start_button:
            self.start_button.setEnabled(False)
        for btn_name in ('batch_ocr_import_btn', 'batch_ocr_open_btn'):
            if hasattr(self, btn_name):
                getattr(self, btn_name).setEnabled(False)
        
        # Show the stop button when an operation starts, but not during model loading.
        if show_stop_button and not getattr(self, '_waiting_for_model', False) and hasattr(ipw, 'stop_translation_btn'):
            ipw.stop_translation_btn.setVisible(True)
            ipw.stop_translation_btn.setEnabled(True)
            ipw.stop_translation_btn.setText("⏹ Stop")
        
        print(f"[WORKFLOW] Disabled workflow buttons (exclude={exclude}, show_stop={show_stop_button})")
    except Exception as e:
        print(f"[WORKFLOW] Error disabling buttons: {e}")

def _enable_workflow_buttons(self):
    """Re-enable all workflow buttons after operation completes."""
    try:
        self._workflow_operation_active = False
        ipw = self.image_preview_widget if hasattr(self, 'image_preview_widget') else None
        if not ipw:
            return
        
        buttons = [
            ('detect_btn', 'Detect Text'),
            ('clean_btn', 'Clean'),
            ('recognize_btn', 'Recognize Text'),
            ('translate_btn', 'Translate'),
            ('import_ocr_btn', '📥 Import'),
            ('export_ocr_btn', '📤 Export'),
            ('translate_all_btn', 'Translate All'),
        ]
        
        for btn_name, default_text in buttons:
            if hasattr(ipw, btn_name):
                btn = getattr(ipw, btn_name)
                btn.setEnabled(True)
                btn.setText(default_text)
        
        # Also re-enable start_button in manga_integration if it exists
        if hasattr(self, 'start_button') and self.start_button:
            self.start_button.setEnabled(True)
        for btn_name in ('batch_ocr_import_btn', 'batch_ocr_open_btn'):
            if hasattr(self, btn_name):
                getattr(self, btn_name).setEnabled(True)
        
        # Re-enable editing tools after processing
        editing_buttons = ['save_overlay_btn', 'delete_btn', 'clear_boxes_btn', 'box_draw_btn', 'circle_draw_btn', 'lasso_btn']
        for btn_name in editing_buttons:
            if hasattr(ipw, btn_name):
                btn = getattr(ipw, btn_name)
                btn.setEnabled(True)
        
        # Hide the stop button when operations complete
        if hasattr(ipw, 'stop_translation_btn'):
            ipw.stop_translation_btn.setVisible(False)
            ipw.stop_translation_btn.setEnabled(True)
            ipw.stop_translation_btn.setText("⏹ Stop")
        
        print(f"[WORKFLOW] Re-enabled all workflow buttons")
        
        # Re-apply waiting/failed state if models are still loading.
        # Without this, buttons briefly show as enabled after an operation
        # finishes even though the inpainter is still loading.
        try:
            if hasattr(self, '_check_preload_status'):
                from PySide6.QtCore import QTimer
                QTimer.singleShot(0, self._check_preload_status)
        except Exception:
            pass
    except Exception as e:
        print(f"[WORKFLOW] Error enabling buttons: {e}")

def _restore_translate_button(self):
    """Restore the translate button to its original state"""
    try:
        # Remove processing overlay effect for the image that was being processed
        image_path = getattr(self, '_translating_image_path', None)
        _remove_processing_overlay(self, image_path)
        
        # CRITICAL: Restore print hijacking if MangaTranslator exists
        if hasattr(self, '_manga_translator') and self._manga_translator:
            try:
                self._manga_translator.restore_print()
            except Exception:
                pass
        
        # Re-enable ALL workflow buttons (not just translate)
        _enable_workflow_buttons(self)
        
        # Re-enable thumbnail list
        if hasattr(self, 'image_preview_widget') and hasattr(self.image_preview_widget, 'thumbnail_list'):
            self.image_preview_widget.thumbnail_list.setEnabled(True)
            print(f"[TRANSLATE] Re-enabled thumbnail list after translation")
        
        # Switch display mode to 'translated' so user sees the result
        try:
            ipw = self.image_preview_widget if hasattr(self, 'image_preview_widget') else None
            if ipw:
                ipw.source_display_mode = 'translated'
                ipw.cleaned_images_enabled = True  # Deprecated flag for compatibility
                
                # Update the toggle button appearance to match 'translated' state
                if hasattr(ipw, 'cleaned_toggle_btn') and ipw.cleaned_toggle_btn:
                    ipw.cleaned_toggle_btn.setText("✒️")  # Pen for translated output
                    ipw.cleaned_toggle_btn.setToolTip("Showing translated output (click to cycle)")
                    ipw.cleaned_toggle_btn.setStyleSheet("""
                        QToolButton {
                            background-color: #28a745;
                            border: 2px solid #34ce57;
                            font-size: 12pt;
                            min-width: 32px;
                            min-height: 32px;
                            max-width: 36px;
                            max-height: 36px;
                            padding: 3px;
                            border-radius: 3px;
                            color: white;
                        }
                        QToolButton:hover {
                            background-color: #34ce57;
                        }
                    """)
                
                # Refresh preview to show translated output
                current_image = getattr(ipw, 'current_image_path', None)
                if current_image:
                    ipw.load_image(current_image, preserve_rectangles=True, preserve_text_overlays=True)
                
                print(f"[TRANSLATE] Switched display mode to 'translated'")
        except Exception as mode_err:
            print(f"[TRANSLATE] Failed to switch display mode: {mode_err}")
    except Exception:
        pass

def _run_custom_image_edit_translate_all_clicked(self, image_paths: list):
    """Use custom-image-edit as the Translate All workflow without text rendering."""
    try:
        total_images = len(image_paths)
        _disable_workflow_buttons(self, exclude=None)
        if hasattr(self.image_preview_widget, 'translate_all_btn'):
            self.image_preview_widget.translate_all_btn.setText(f"Translating... (0/{total_images})")
        if hasattr(self.image_preview_widget, 'thumbnail_list'):
            self.image_preview_widget.thumbnail_list.setEnabled(False)
            print("[TRANSLATE_ALL] Disabled thumbnail list during custom image edit batch")
        _add_processing_overlay(self)

        import threading
        thread = threading.Thread(
            target=_run_custom_image_edit_translate_all_background,
            args=(self, list(image_paths)),
            daemon=True,
        )
        thread.start()
    except Exception as e:
        import traceback
        self._log(f"âŒ Custom image edit batch setup failed: {str(e)}", "error")
        print(f"[CUSTOM_IMAGE_EDIT_ALL] Setup traceback: {traceback.format_exc()}")
        _restore_translate_all_button(self)

def _run_custom_image_edit_translate_all_background(self, image_paths: list):
    _reset_cancellation_flags(self)
    try:
        total = len(image_paths)
        success_count = 0
        failed_count = 0
        self._log(f"ðŸ§½ Starting custom image edit batch: {total} images", "info")

        for idx, image_path in enumerate(image_paths, 1):
            if _is_translation_cancelled(self):
                self._log(f"â¹ Custom image edit batch cancelled at image {idx}/{total}", "warning")
                break

            self.update_queue.put(('load_preview_image', {
                'path': image_path,
                'preserve_rectangles': False,
                'preserve_overlays': False
            }))
            self.update_queue.put(('sync_file_selection', {'image_path': image_path}))
            self.update_queue.put(('translate_all_progress', {
                'current': idx,
                'total': total
            }))
            self.update_queue.put(('add_processing_overlay', None))
            self._log(f"ðŸ“„ [{idx}/{total}] Custom image editing: {os.path.basename(image_path)}", "info")

            regions = _regions_for_custom_image_edit(self, image_path, use_current_rectangles=False)
            if not regions:
                self._log(f"âš ï¸ [{idx}/{total}] No regions found", "warning")
                failed_count += 1
                continue

            translated_path = _run_inpainting_sync(
                self,
                image_path,
                regions,
                save_as='translated',
            )
            if not translated_path or not os.path.exists(translated_path):
                self._log(f"âš ï¸ [{idx}/{total}] Custom image edit failed", "warning")
                failed_count += 1
                continue

            try:
                if not hasattr(self, '_rendered_images_map'):
                    self._rendered_images_map = {}
                self._rendered_images_map[image_path] = translated_path
                if hasattr(self.image_preview_widget, 'current_translated_path'):
                    self.image_preview_widget.current_translated_path = translated_path
                if hasattr(self, 'image_state_manager'):
                    self.image_state_manager.update_state(image_path, {
                        'rendered_image_path': translated_path,
                        'step': 'translated'
                    })
            except Exception:
                pass

            self.update_queue.put(('preview_update', {
                'translated_path': translated_path,
                'source_path': image_path
            }))
            self.update_queue.put(('switch_to_translated_mode', {
                'image_path': image_path
            }))
            success_count += 1

        self._log("Custom image edit batch complete", "success")
        self._log(f"   Successful: {success_count}/{total}", "success")
        if failed_count:
            self._log(f"   Failed: {failed_count}/{total}", "warning")
        self.update_queue.put(('update_preview_to_rendered', None))
    except Exception as e:
        import traceback
        self._log(f"âŒ Custom image edit batch failed: {str(e)}", "error")
        print(f"[CUSTOM_IMAGE_EDIT_ALL] Background traceback: {traceback.format_exc()}")
    finally:
        self.update_queue.put(('remove_processing_overlay', None))
        self.update_queue.put(('translate_all_button_restore', None))
        try:
            self._batch_mode_active = False
        except Exception:
            pass

def _restore_translate_all_button(self):
    """Restore the translate all button to its original state"""
    print(f"[TRANSLATE_ALL] _restore_translate_all_button called")
    try:
        # Re-enable ALL workflow buttons (not just translate all)
        _enable_workflow_buttons(self)
        
        # Re-enable thumbnail list
        if hasattr(self, 'image_preview_widget') and hasattr(self.image_preview_widget, 'thumbnail_list'):
            self.image_preview_widget.thumbnail_list.setEnabled(True)
            print(f"[TRANSLATE_ALL] Re-enabled thumbnail list after batch translation")
        
        # Explicitly reset translate_all_btn text (in case _enable_workflow_buttons missed it)
        if hasattr(self, 'image_preview_widget') and hasattr(self.image_preview_widget, 'translate_all_btn'):
            self.image_preview_widget.translate_all_btn.setText("Translate All")
            self.image_preview_widget.translate_all_btn.setEnabled(True)
            print(f"[TRANSLATE_ALL] Explicitly reset translate_all_btn text")
        
        # Switch display mode to 'translated' so user sees the result
        try:
            ipw = self.image_preview_widget
            if ipw:
                ipw.source_display_mode = 'translated'
                ipw.cleaned_images_enabled = True  # Deprecated flag for compatibility
                
                # Update the toggle button appearance to match 'translated' state
                if hasattr(ipw, 'cleaned_toggle_btn') and ipw.cleaned_toggle_btn:
                    ipw.cleaned_toggle_btn.setText("✒️")  # Pen for translated output
                    ipw.cleaned_toggle_btn.setToolTip("Showing translated output (click to cycle)")
                    ipw.cleaned_toggle_btn.setStyleSheet("""
                        QToolButton {
                            background-color: #28a745;
                            border: 2px solid #34ce57;
                            font-size: 12pt;
                            min-width: 32px;
                            min-height: 32px;
                            max-width: 36px;
                            max-height: 36px;
                            padding: 3px;
                            border-radius: 3px;
                            color: white;
                        }
                        QToolButton:hover {
                            background-color: #34ce57;
                        }
                    """)
                
                # Go back to the first active image. For manga range mode, row 1 means
                # the first row inside the current visible-order range, not row 0 overall.
                if hasattr(self, '_update_manga_preview_image_list_for_range'):
                    try:
                        self._update_manga_preview_image_list_for_range()
                    except Exception as range_sync_err:
                        print(f"[TRANSLATE_ALL] Range thumbnail sync failed: {range_sync_err}")

                first_image = None
                file_row = None
                try:
                    if hasattr(self, '_manga_range_filtered_files'):
                        ranged_files, range_error = self._manga_range_filtered_files()
                        if not range_error and ranged_files:
                            first_image = ranged_files[0]
                except Exception as range_err:
                    print(f"[TRANSLATE_ALL] Failed to read image range: {range_err}")

                if not first_image and hasattr(ipw, 'image_paths') and ipw.image_paths:
                    first_image = ipw.image_paths[0]

                if first_image:
                    try:
                        if hasattr(self, 'selected_files') and first_image in self.selected_files:
                            file_row = self.selected_files.index(first_image)
                    except Exception:
                        file_row = None

                    ipw.load_image(first_image, preserve_rectangles=True, preserve_text_overlays=True)
                    # Also update thumbnail selection to the matching ranged thumbnail.
                    if hasattr(ipw, 'thumbnail_list') and ipw.thumbnail_list.count() > 0:
                        thumb_row = 0
                        try:
                            if hasattr(ipw, 'image_paths') and first_image in ipw.image_paths:
                                thumb_row = ipw.image_paths.index(first_image)
                        except Exception:
                            thumb_row = 0
                        ipw.thumbnail_list.setCurrentRow(thumb_row)
                    # Sync file_listbox selection to the original visible row.
                    if file_row is not None and hasattr(self, 'file_listbox') and self.file_listbox and self.file_listbox.count() > file_row:
                        self.file_listbox.setCurrentRow(file_row)
                        print(f"[TRANSLATE_ALL] Synced file_listbox to ranged row {file_row + 1}")
                    print(f"[TRANSLATE_ALL] Returned to first active image: {os.path.basename(first_image)}")
                else:
                    # Fallback: refresh current image
                    current_image = getattr(ipw, 'current_image_path', None)
                    if current_image:
                        ipw.load_image(current_image, preserve_rectangles=True, preserve_text_overlays=True)
                
                print(f"[TRANSLATE_ALL] Switched display mode to 'translated'")
        except Exception as mode_err:
            print(f"[TRANSLATE_ALL] Failed to switch display mode: {mode_err}")
    except Exception as e:
        print(f"[TRANSLATE_ALL] Error in _restore_translate_all_button: {e}")
        import traceback
        traceback.print_exc()

def _update_preview_to_rendered_images(self):
    """On batch end, show translated result for the CURRENT selection only; do not change lists or selection."""
    try:
        if not hasattr(self, '_rendered_images_map') or not self._rendered_images_map:
            print("[UPDATE_PREVIEW] No rendered images to show")
            return
        
        # Determine current selection
        current_row = self.file_listbox.currentRow() if hasattr(self, 'file_listbox') else -1
        if current_row is None or current_row < 0 or current_row >= len(self.selected_files):
            print("[UPDATE_PREVIEW] No current selection; not changing preview")
            return
        current_source = self.selected_files[current_row]
        
        # If a rendered image exists for the currently selected source, store the path and refresh preview
        rendered_path = self._rendered_images_map.get(current_source)
        if rendered_path and os.path.exists(rendered_path):
            if hasattr(self, 'image_preview_widget'):
                # Store the translated path
                self.image_preview_widget.current_translated_path = rendered_path
                # Refresh the preview to show the rendered result
                self.image_preview_widget.load_image(current_source, preserve_rectangles=True, preserve_text_overlays=True)
                print(f"[UPDATE_PREVIEW] Updated preview for current image: {os.path.basename(rendered_path)}")
        else:
            print("[UPDATE_PREVIEW] No rendered image for current selection")
    
    except Exception as e:
        print(f"[UPDATE_PREVIEW] Error: {str(e)}")
        import traceback
        print(f"[UPDATE_PREVIEW] Traceback: {traceback.format_exc()}")

def _render_with_manga_translator_thread_safe(self, base_image_path, regions, output_path, original_image_path):
    """Thread-safe rendering using existing translator instance (no heavy initialization)"""
    try:
        import sys
        import cv2
        import os
        
        sys.__stdout__.write(f"[WORKER] Starting render: base={os.path.basename(base_image_path)}\n")
        sys.__stdout__.write(f"[WORKER] Regions to render: {len(regions)}\n")
        sys.__stdout__.write(f"[WORKER] Output path: {output_path}\n")
        sys.__stdout__.flush()
        
        # Use existing translator instance from manga_integration (already has loaded models)
        translator = self.manga_integration.translator
        if not translator:
            sys.__stdout__.write("[WORKER] ERROR: No translator instance available\n")
            sys.__stdout__.flush()
            return False
        
        # Load the base image for rendering
        base_image_array = cv2.imread(base_image_path)
        if base_image_array is None:
            raise ValueError(f"Failed to load base image: {base_image_path}")
        
        # Ensure all regions have translated_text set
        filtered_regions = []
        for region in regions:
            if not hasattr(region, 'translated_text') or not region.translated_text:
                # Fallback to original text if translation missing
                region.translated_text = region.text
            filtered_regions.append(region)
        
        sys.__stdout__.write(f"[WORKER] Rendering {len(filtered_regions)} regions with translations\n")
        sys.__stdout__.flush()
        
        # Render using the existing translator's render method (thread-safe for rendering)
        # The translator already has loaded models in the shared pool
        rendered_image_array = translator.render_translated_text(base_image_array, filtered_regions)
        
        # Save the rendered image
        if output_path:
            result_success = cv2.imwrite(output_path, rendered_image_array)
            result_path = output_path if result_success else None
        else:
            # Generate output path if not provided
            base_dir = os.path.dirname(base_image_path)
            base_name = os.path.splitext(os.path.basename(base_image_path))[0]
            output_path = os.path.join(base_dir, f"{base_name}_translated.png")
            result_success = cv2.imwrite(output_path, rendered_image_array)
            result_path = output_path if result_success else None
        
        # Clean up only local resources (images)
        # DO NOT clean up translator or models - they're shared and reused!
        del base_image_array
        del rendered_image_array
        
        sys.__stdout__.write(f"[WORKER] Render completed: {result_path}\n")
        sys.__stdout__.flush()
        return result_path is not None
        
    except Exception as e:
        sys.__stdout__.write(f"[WORKER] Render failed: {e}\n")
        sys.__stdout__.flush()
        import traceback
        traceback.print_exc()
        return False
