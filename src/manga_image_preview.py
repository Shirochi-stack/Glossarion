# manga_image_preview.py
"""
Compact image preview widget with manual editing features for manga translation
Based on comic-translate's ImageViewer but simplified for integration into Glossarion
"""

import os
import threading
from typing import List, Dict, Optional
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGraphicsView, 
                               QGraphicsScene, QGraphicsPixmapItem, QToolButton, 
                               QLabel, QSlider, QFrame, QPushButton, QGraphicsRectItem,
                               QSizePolicy, QListWidget, QListWidgetItem, QTabWidget)
from PySide6.QtCore import Qt, Signal, QRectF, QPointF, QSize, QThread, QObject, Slot
from PySide6.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QBrush, QCursor, QIcon
import numpy as np


class MoveableRectItem(QGraphicsRectItem):
    """Simple moveable rectangle for text box selection"""
    
    def __init__(self, rect=None, parent=None, pen=None, brush=None):
        super().__init__(rect, parent)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        
        # Disable caching to prevent double-draw artifacts
        self.setCacheMode(QGraphicsRectItem.CacheMode.NoCache)
        
        # Only set default colors if not provided
        if brush is not None:
            self.setBrush(brush)
        else:
            self.setBrush(QBrush(QColor(255, 192, 203, 125)))  # Transparent pink (default)
        
        if pen is not None:
            self.setPen(pen)
        else:
            self.setPen(QPen(QColor(255, 0, 0), 2))  # Red border (default)
        
        self.setZValue(1)
        self.selected = False
        
    def hoverEnterEvent(self, event):
        self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        super().hoverEnterEvent(event)
        
    def hoverLeaveEvent(self, event):
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        super().hoverLeaveEvent(event)
        
    def mousePressEvent(self, event):
        """Ensure only this rectangle is selected to avoid multi-move of multiple items."""
        try:
            if event.button() == Qt.MouseButton.LeftButton:
                sc = self.scene()
                if sc is not None:
                    sc.clearSelection()
                self.setSelected(True)
        except Exception:
            pass
        super().mousePressEvent(event)


class ImageLoaderWorker(QObject):
    """Worker for loading images in background thread"""
    
    finished = Signal(QPixmap, str)  # pixmap, file_path
    error = Signal(str)  # error message
    
    def __init__(self):
        super().__init__()
        self.image_path = None
        self.cancelled = False
    
    @Slot()
    def load_image(self, image_path: str):
        """Load image in background thread"""
        self.image_path = image_path
        self.cancelled = False
        
        try:
            if not os.path.exists(image_path):
                self.error.emit(f"File not found: {image_path}")
                return
            
            if self.cancelled:
                return
            
            # Load pixmap (this can be slow for large images)
            pixmap = QPixmap(image_path)
            
            if self.cancelled:
                return
            
            if pixmap.isNull():
                self.error.emit(f"Failed to load: {image_path}")
                return
            
            # Emit the loaded pixmap
            self.finished.emit(pixmap, image_path)
            
        except Exception as e:
            if not self.cancelled:
                self.error.emit(str(e))
    
    def cancel(self):
        """Cancel the current load operation"""
        self.cancelled = True


class CompactImageViewer(QGraphicsView):
    """Compact image viewer with basic editing capabilities"""
    
    rectangle_created = Signal(QRectF)
    rectangle_selected = Signal(QRectF)
    rectangle_deleted = Signal(QRectF)
    image_loading = Signal(str)  # emitted when starting to load
    image_loaded = Signal(str)   # emitted when image successfully loaded
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Scene setup
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.photo = QGraphicsPixmapItem()
        self._scene.addItem(self.photo)
        
        # View settings
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)  # Default to pan mode
        
        # High quality image scaling to preserve image quality
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        # Disable antialiasing for shapes to prevent double-line artifacts on rectangles
        self.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        
        # State
        self.empty = True
        self.zoom_level = 0
        self.user_zoomed = False  # Prevent auto-fit after manual zoom
        self.current_tool = 'pan'  # Default to pan tool
        self.rectangles: List[MoveableRectItem] = []
        self.selected_rect: Optional[MoveableRectItem] = None
        
        # Drawing state
        self.start_point: Optional[QPointF] = None
        self.current_rect: Optional[MoveableRectItem] = None
        
        # Inpainting
        self.brush_size = 20
        self.eraser_size = 20
        self.drawing = False
        self.last_pos = None
        self.brush_strokes = []
        
        # Background image loading
        self._loader_thread = None
        self._loader_worker = None
        self._loading = False
    
    def resizeEvent(self, event):
        """Refit only if user hasn't manually zoomed."""
        super().resizeEvent(event)
        if self.hasPhoto() and not self.user_zoomed:
            self.fitInView()
    
    def hasPhoto(self) -> bool:
        return not self.empty
    
    def load_image(self, image_path: str) -> bool:
        """Load an image from file path in background thread"""
        if not os.path.exists(image_path):
            return False
        
        # Cancel any existing load operation
        if self._loader_worker:
            self._loader_worker.cancel()
        
        # Clean up old thread if it exists
        if self._loader_thread and self._loader_thread.isRunning():
            self._loader_thread.quit()
            self._loader_thread.wait(100)  # Wait up to 100ms
        
        # Emit loading signal
        self._loading = True
        self.image_loading.emit(image_path)
        
        # Create new thread and worker
        self._loader_thread = QThread()
        self._loader_worker = ImageLoaderWorker()
        self._loader_worker.moveToThread(self._loader_thread)
        
        # Connect signals
        self._loader_worker.finished.connect(self._on_image_loaded)
        self._loader_worker.error.connect(self._on_image_error)
        self._loader_thread.started.connect(lambda: self._loader_worker.load_image(image_path))
        
        # Start the thread
        self._loader_thread.start()
        
        return True
    
    @Slot(QPixmap, str)
    def _on_image_loaded(self, pixmap: QPixmap, image_path: str):
        """Handle image loaded from background thread"""
        self._loading = False
        self.setPhoto(pixmap)
        
        # Emit success signal
        self.image_loaded.emit(image_path)
        
        # Clean up thread
        if self._loader_thread:
            self._loader_thread.quit()
            self._loader_thread.wait()
            self._loader_thread = None
        self._loader_worker = None
    
    @Slot(str)
    def _on_image_error(self, error_msg: str):
        """Handle image load error"""
        self._loading = False
        print(f"Image load error: {error_msg}")
        
        # Clean up thread
        if self._loader_thread:
            self._loader_thread.quit()
            self._loader_thread.wait()
            self._loader_thread = None
        self._loader_worker = None
    
    def setPhoto(self, pixmap: QPixmap = None):
        """Set the photo to display with high quality rendering"""
        if pixmap and not pixmap.isNull():
            self.empty = False
            self.photo.setPixmap(pixmap)
            # Reset zoom state and fit once on load
            self.user_zoomed = False
            from PySide6.QtCore import QTimer
            QTimer.singleShot(10, self.fitInView)
        else:
            self.empty = True
            self.photo.setPixmap(QPixmap())
        self.zoom_level = 0
    
    def fitInView(self):
        """Fit the image to view and clear manual zoom state."""
        if not self.hasPhoto():
            return
        
        rect = self.photo.boundingRect()
        if not rect.isNull():
            self.setSceneRect(rect)
            unity = self.transform().mapRect(QRectF(0, 0, 1, 1))
            self.scale(1 / unity.width(), 1 / unity.height())
            viewrect = self.viewport().rect()
            scenerect = self.transform().mapRect(rect)
            factor = min(viewrect.width() / scenerect.width(),
                        viewrect.height() / scenerect.height())
            self.scale(factor, factor)
            self.centerOn(rect.center())
            # Reset manual zoom flag when performing an explicit fit
            self.user_zoomed = False
    
    def set_tool(self, tool: str):
        """Set current tool (box_draw, pan, brush, eraser)"""
        self.current_tool = tool
        if tool == 'pan':
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        elif tool == 'brush':
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.setCursor(Qt.CursorShape.CrossCursor)
        elif tool == 'eraser':
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.setCursor(Qt.CursorShape.ArrowCursor)
    
    def wheelEvent(self, event):
        """Handle zoom with mouse wheel (Shift+Wheel). Set user_zoomed to prevent auto-fit resets."""
        if self.hasPhoto() and event.modifiers() == Qt.KeyboardModifier.ShiftModifier:
            factor = 1.15
            if event.angleDelta().y() > 0:
                self.scale(factor, factor)
                self.zoom_level += 1
            else:
                self.scale(1 / factor, 1 / factor)
                self.zoom_level -= 1
            self.user_zoomed = True
        else:
            super().wheelEvent(event)
    
    def mousePressEvent(self, event):
        """Handle mouse press for drawing boxes or inpainting"""
        if self.current_tool == 'box_draw' and event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            self.start_point = scene_pos
            rect = QRectF(scene_pos, scene_pos)
            self.current_rect = MoveableRectItem(rect)
            self._scene.addItem(self.current_rect)
        elif self.current_tool in ['brush', 'eraser'] and event.button() == Qt.MouseButton.LeftButton:
            self.drawing = True
            self.last_pos = self.mapToScene(event.pos())
        elif self.current_tool == 'pan' and event.button() == Qt.MouseButton.LeftButton:
            # Check if we clicked on a rectangle to select it
            item = self.itemAt(event.pos())
            if isinstance(item, MoveableRectItem):
                self._select_rectangle(item)
            else:
                self._select_rectangle(None)  # Deselect all
            super().mousePressEvent(event)
        else:
            super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for drawing boxes or inpainting"""
        if self.current_tool == 'box_draw' and self.current_rect:
            scene_pos = self.mapToScene(event.pos())
            rect = QRectF(self.start_point, scene_pos).normalized()
            self.current_rect.setRect(rect)
        elif self.current_tool in ['brush', 'eraser'] and self.drawing:
            current_pos = self.mapToScene(event.pos())
            if self.last_pos:
                # Store stroke for inpainting
                stroke = {
                    'start': (self.last_pos.x(), self.last_pos.y()),
                    'end': (current_pos.x(), current_pos.y()),
                    'size': self.brush_size if self.current_tool == 'brush' else self.eraser_size,
                    'type': self.current_tool
                }
                self.brush_strokes.append(stroke)
                
                # Draw stroke visually on the scene
                self._draw_brush_stroke(stroke)
            self.last_pos = current_pos
        else:
            super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release"""
        if self.current_tool == 'box_draw' and self.current_rect:
            if self.current_rect.rect().width() > 10 and self.current_rect.rect().height() > 10:
                self.rectangles.append(self.current_rect)
                self.rectangle_created.emit(self.current_rect.rect())
            else:
                self._scene.removeItem(self.current_rect)
            self.current_rect = None
            self.start_point = None
        elif self.current_tool in ['brush', 'eraser']:
            self.drawing = False
            self.last_pos = None
        else:
            super().mouseReleaseEvent(event)
    
    def _select_rectangle(self, rect_item: Optional[MoveableRectItem]):
        """Select a rectangle and update visual feedback"""
        # Deselect previous selection
        if self.selected_rect:
            self.selected_rect.setPen(QPen(QColor(0, 255, 0), 2))  # Green for normal
        
        # Select new rectangle
        self.selected_rect = rect_item
        if self.selected_rect:
            self.selected_rect.setPen(QPen(QColor(255, 255, 0), 3))  # Yellow for selected, thicker
            self.rectangle_selected.emit(self.selected_rect.rect())
    
    def _draw_brush_stroke(self, stroke: dict):
        """Draw a brush stroke visually on the scene"""
        from PySide6.QtWidgets import QGraphicsLineItem
        
        start_x, start_y = stroke['start']
        end_x, end_y = stroke['end']
        size = stroke['size']
        stroke_type = stroke['type']
        
        # Create line item
        line = QGraphicsLineItem(start_x, start_y, end_x, end_y)
        
        # Style based on tool type
        if stroke_type == 'brush':
            # Red brush strokes for areas to inpaint (areas that need cleaning)
            pen = QPen(QColor(255, 0, 0, 180), size)  # Semi-transparent red
        else:  # eraser
            # White eraser strokes (areas to keep clean/erase text)
            pen = QPen(QColor(255, 255, 255, 200), size)  # Semi-transparent white
        
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        line.setPen(pen)
        line.setZValue(2)  # Above rectangles
        
        # Add to scene and track it
        self._scene.addItem(line)
        
        # Store reference in stroke for later removal
        stroke['graphics_item'] = line
    
    def clear_rectangles(self):
        """Clear all rectangles"""
        for rect in self.rectangles:
            self._scene.removeItem(rect)
        self.rectangles.clear()
        self.selected_rect = None
    
    def clear_brush_strokes(self):
        """Clear all brush strokes from scene and memory"""
        for stroke in self.brush_strokes:
            if 'graphics_item' in stroke:
                self._scene.removeItem(stroke['graphics_item'])
        self.brush_strokes.clear()
    
    def draw_detection_boxes(self, regions: list, color: QColor = None):
        """Draw detection boxes from text regions"""
        if color is None:
            color = QColor(0, 255, 0)  # Green for detections
        
        # Clear existing rectangles first
        self.clear_rectangles()
        
        # Draw boxes for each region
        for region in regions:
            x1, y1, x2, y2 = region.x1, region.y1, region.x2, region.y2
            rect = QRectF(x1, y1, x2 - x1, y2 - y1)
            
            # Create rectangle item
            rect_item = MoveableRectItem(rect)
            rect_item.setPen(QPen(color, 2))
            rect_item.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 50)))  # Semi-transparent
            
            self._scene.addItem(rect_item)
            self.rectangles.append(rect_item)
    
    def overlay_image(self, image_array):
        """Overlay a numpy image array (e.g., cleaned or rendered image) on top of current image"""
        try:
            import cv2
            # Convert numpy array (BGR) to QPixmap
            rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            height, width, channel = rgb_image.shape
            bytes_per_line = 3 * width
            q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.setPhoto(pixmap)
        except Exception as e:
            print(f"Error overlaying image: {e}")
    
    def delete_selected_rectangle(self):
        """Delete currently selected rectangle"""
        if self.selected_rect:
            self._scene.removeItem(self.selected_rect)
            self.rectangles.remove(self.selected_rect)
            self.rectangle_deleted.emit(self.selected_rect.rect())
            self.selected_rect = None
    
    def clear_scene(self):
        """Clear entire scene"""
        # Cancel any pending load
        if self._loader_worker:
            self._loader_worker.cancel()
        if self._loader_thread and self._loader_thread.isRunning():
            self._loader_thread.quit()
            self._loader_thread.wait(100)
        
        self._scene.clear()
        self.rectangles.clear()
        self.selected_rect = None
        self.photo = QGraphicsPixmapItem()
        self._scene.addItem(self.photo)
        self.empty = True
        self._loading = False


class MangaImagePreviewWidget(QWidget):
    """Complete compact image preview widget with tools"""
    
    # Signals for translation workflow
    detect_text_clicked = Signal()
    clean_image_clicked = Signal()
    recognize_text_clicked = Signal()
    translate_text_clicked = Signal()
    translate_all_clicked = Signal()
    
    def __init__(self, parent=None, main_gui=None):
        super().__init__(parent)
        self.current_image_path = None
        self.current_translated_path = None  # Track translated output path
        self.main_gui = main_gui  # Store reference to main GUI for config persistence
        self.manual_editing_enabled = False  # Manual editing is disabled by default (preview mode)
        self.translated_folder_path = None  # Path to translated images folder
        self._build_ui()
    
    def _build_ui(self):
        """Build the compact UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Title with toggle switch and manual file load button
        title_container = QWidget()
        title_layout = QHBoxLayout(title_container)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(10)
        
        title_label = QLabel("📷 Image Preview & Editing")
        title_font = title_label.font()
        title_font.setBold(True)
        title_font.setPointSize(10)
        title_label.setFont(title_font)
        title_layout.addWidget(title_label)
        
        # Manual editing toggle with spinning icon
        from PySide6.QtWidgets import QCheckBox
        from spinning import create_spinning_checkbox_with_icon
        
        self.manual_editing_toggle = QCheckBox("Enable Manual Editing")
        
        # Load saved state from config or default to disabled (preview mode)
        if self.main_gui and hasattr(self.main_gui, 'config'):
            saved_state = self.main_gui.config.get('manga_manual_editing_enabled', False)
            self.manual_editing_toggle.setChecked(saved_state)
            self.manual_editing_enabled = saved_state
        else:
            self.manual_editing_toggle.setChecked(False)  # Disabled by default
            self.manual_editing_enabled = False
        self.manual_editing_toggle.setStyleSheet("""
            QCheckBox {
                color: white;
                font-size: 9pt;
                font-weight: bold;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border: 2px solid #5a9fd4;
                border-radius: 3px;
                background-color: #2d2d2d;
            }
            QCheckBox::indicator:checked {
                background-color: #5a9fd4;
                border-color: #7bb3e0;
            }
            QCheckBox::indicator:unchecked {
                background-color: #1a1a1a;
                border-color: #5a5a5a;
            }
            QCheckBox::indicator:hover {
                border-color: #7bb3e0;
            }
            QCheckBox::indicator:unchecked:hover {
                background-color: #2d2d2d;
            }
            QCheckBox::indicator:checked:hover {
                background-color: #7bb3e0;
            }
        """)
        
        # Create checkmark overlay (same technique as manga_settings_dialog.py)
        self.checkmark_overlay = QLabel("✓", self.manual_editing_toggle)
        self.checkmark_overlay.setStyleSheet("""
            QLabel {
                color: white;
                background: transparent;
                font-weight: bold;
                font-size: 14px;
            }
        """)
        self.checkmark_overlay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.checkmark_overlay.hide()
        self.checkmark_overlay.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)  # Make checkmark click-through
        
        # Position checkmark properly
        def position_checkmark():
            self.checkmark_overlay.setGeometry(2, 2, 20, 20)
        
        # Show/hide checkmark based on checked state
        def update_checkmark_visibility():
            if self.manual_editing_toggle.isChecked():
                position_checkmark()
                self.checkmark_overlay.show()
            else:
                self.checkmark_overlay.hide()
        
        self.manual_editing_toggle.stateChanged.connect(update_checkmark_visibility)
        # Delay initial positioning to ensure widget is properly rendered
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, lambda: (position_checkmark(), update_checkmark_visibility()))
        self.manual_editing_toggle.stateChanged.connect(self._on_manual_editing_toggled)
        self.manual_editing_toggle.stateChanged.connect(self._update_toggle_label)
        title_layout.addWidget(self.manual_editing_toggle)
        
        # Add spinning Halgakos icon next to checkbox
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.halgakos_icon = create_spinning_checkbox_with_icon(
            self.manual_editing_toggle,
            icon_size=20,
            base_dir=script_dir
        )
        title_layout.addWidget(self.halgakos_icon)
        
        title_layout.addStretch()
        layout.addWidget(title_container)
        
        # Create tabbed viewer for source/output switching
        self.viewer_tabs = QTabWidget()
        self.viewer_tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.viewer_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #3a3a3a;
                border-radius: 3px;
            }
            QTabBar::tab {
                background-color: #2d2d2d;
                color: white;
                padding: 6px 12px;
                border: 1px solid #3a3a3a;
                border-bottom: none;
                border-top-left-radius: 3px;
                border-top-right-radius: 3px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #5a9fd4;
                border-color: #7bb3e0;
            }
            QTabBar::tab:hover:!selected {
                background-color: #3a3a3a;
            }
        """)
        
        # Create horizontal layout for viewer + thumbnails
        viewer_container = QWidget()
        viewer_layout = QHBoxLayout(viewer_container)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        viewer_layout.setSpacing(5)
        
        # Source image viewer
        self.viewer = CompactImageViewer()
        self.viewer.setMinimumHeight(300)
        viewer_layout.addWidget(self.viewer, stretch=1)
        
        # Add source tab
        self.viewer_tabs.addTab(viewer_container, "📄 Source")
        
        # Create output viewer container
        output_container = QWidget()
        output_layout = QHBoxLayout(output_container)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.setSpacing(5)
        
        # Output image viewer
        self.output_viewer = CompactImageViewer()
        self.output_viewer.setMinimumHeight(300)
        output_layout.addWidget(self.output_viewer, stretch=1)
        
        # Add output tab
        self.viewer_tabs.addTab(output_container, "✨ Translated Output")
        
        # Thumbnail list (right side) - shared across tabs
        from PySide6.QtWidgets import QListWidget, QListWidgetItem, QScrollArea
        self.thumbnail_list = QListWidget()
        self.thumbnail_list.setMaximumWidth(150)
        self.thumbnail_list.setMinimumWidth(120)
        self.thumbnail_list.setIconSize(QSize(100, 100))
        self.thumbnail_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.thumbnail_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.thumbnail_list.setMovement(QListWidget.Movement.Static)
        self.thumbnail_list.setFlow(QListWidget.Flow.TopToBottom)
        self.thumbnail_list.setWrapping(False)
        self.thumbnail_list.setSpacing(5)
        self.thumbnail_list.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        self.thumbnail_list.setStyleSheet("""
            QListWidget {
                background-color: #1e1e1e;
                border: 1px solid #3a3a3a;
                border-radius: 3px;
            }
            QListWidget::item {
                background-color: #2d2d2d;
                border: 2px solid #3a3a3a;
                border-radius: 3px;
                padding: 2px;
                margin: 2px;
            }
            QListWidget::item:selected {
                background-color: #3a3a3a;
                border-color: #5a9fd4;
            }
            QListWidget::item:hover {
                border-color: #7bb3e0;
            }
        """)
        self.thumbnail_list.itemClicked.connect(self._on_thumbnail_clicked)
        viewer_layout.addWidget(self.thumbnail_list)
        
        # Create a second thumbnail list for output tab (can't share same widget)
        self.output_thumbnail_list = QListWidget()
        self.output_thumbnail_list.setMaximumWidth(150)
        self.output_thumbnail_list.setMinimumWidth(120)
        self.output_thumbnail_list.setIconSize(QSize(100, 100))
        self.output_thumbnail_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.output_thumbnail_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.output_thumbnail_list.setMovement(QListWidget.Movement.Static)
        self.output_thumbnail_list.setFlow(QListWidget.Flow.TopToBottom)
        self.output_thumbnail_list.setWrapping(False)
        self.output_thumbnail_list.setSpacing(5)
        self.output_thumbnail_list.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        self.output_thumbnail_list.setStyleSheet(self.thumbnail_list.styleSheet())
        self.output_thumbnail_list.itemClicked.connect(self._on_thumbnail_clicked)
        output_layout.addWidget(self.output_thumbnail_list)
        
        # Both thumbnail lists start hidden
        self.thumbnail_list.setVisible(False)
        self.output_thumbnail_list.setVisible(False)
        
        # Store image paths for thumbnails
        self.image_paths = []
        
        layout.addWidget(self.viewer_tabs, stretch=1)
        
        # Connect loading signals to show status
        self.viewer.image_loading.connect(self._on_image_loading_started)
        self.viewer.image_loaded.connect(self._on_image_loaded_success)
        
        # Refitting on tab change ensures current image fills the available space
        self.viewer_tabs.currentChanged.connect(self._on_tab_changed)
        
        # Display placeholder icon when no image is loaded
        self._show_placeholder_icon()
        self._show_output_placeholder()
        
        # Manual Tools - Row 1: Box Drawing & Inpainting
        self.tools_frame = self._create_tool_frame("Manual Tools")
        tools_layout = QHBoxLayout(self.tools_frame)
        tools_layout.setContentsMargins(3, 3, 3, 3)
        tools_layout.setSpacing(3)
        
        # ALWAYS VISIBLE: Pan and zoom controls at the start
        self.hand_tool_btn = self._create_tool_button("✅", "Pan")
        self.hand_tool_btn.setCheckable(True)
        self.hand_tool_btn.setChecked(True)  # Default to Pan
        self.hand_tool_btn.clicked.connect(lambda: self._set_tool('pan'))
        tools_layout.addWidget(self.hand_tool_btn)
        
        self.fit_btn = self._create_tool_button("⊟", "Fit to View")
        self.fit_btn.clicked.connect(self._fit_active_view)
        tools_layout.addWidget(self.fit_btn)
        
        self.zoom_in_btn = self._create_tool_button("🔍+", "Zoom In (Shift+Wheel)")
        self.zoom_in_btn.clicked.connect(self._zoom_in)
        tools_layout.addWidget(self.zoom_in_btn)
        
        self.zoom_out_btn = self._create_tool_button("🔍-", "Zoom Out (Shift+Wheel)")
        self.zoom_out_btn.clicked.connect(self._zoom_out)
        tools_layout.addWidget(self.zoom_out_btn)
        
        # Add a stretch spacer after zoom controls
        tools_layout.addStretch()
        
        # MANUAL EDITING TOOLS (hidden when manual editing is off)
        self.box_draw_btn = self._create_tool_button("◭", "Draw Box")
        self.box_draw_btn.setCheckable(True)
        self.box_draw_btn.clicked.connect(lambda: self._set_tool('box_draw'))
        tools_layout.addWidget(self.box_draw_btn)
        
        self.brush_btn = self._create_tool_button("🖌", "Brush")
        self.brush_btn.setCheckable(True)
        self.brush_btn.clicked.connect(lambda: self._set_tool('brush'))
        tools_layout.addWidget(self.brush_btn)
        
        self.eraser_btn = self._create_tool_button("✏️", "Eraser")
        self.eraser_btn.setCheckable(True)
        self.eraser_btn.clicked.connect(lambda: self._set_tool('eraser'))
        tools_layout.addWidget(self.eraser_btn)
        
        self.delete_btn = self._create_tool_button("🗑", "Delete Selected")
        self.delete_btn.clicked.connect(self.viewer.delete_selected_rectangle)
        tools_layout.addWidget(self.delete_btn)
        
        self.clear_boxes_btn = self._create_tool_button("🧹", "Clear Boxes")
        self.clear_boxes_btn.clicked.connect(self._on_clear_boxes_clicked)
        tools_layout.addWidget(self.clear_boxes_btn)
        
        self.clear_strokes_btn = self._create_tool_button("❌", "Clear Strokes")
        self.clear_strokes_btn.clicked.connect(self._clear_strokes)
        tools_layout.addWidget(self.clear_strokes_btn)
        
        # Brush size slider (stretches to fill space)
        self.size_slider = QSlider(Qt.Orientation.Horizontal)
        self.size_slider.setMinimum(5)
        self.size_slider.setMaximum(50)
        self.size_slider.setValue(20)
        self.size_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 6px;
                background: #3a3a3a;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #ffd700;
                width: 14px;
                height: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
        """)
        self.size_slider.valueChanged.connect(self._update_brush_size)
        self.size_slider.wheelEvent = lambda event: None
        tools_layout.addWidget(self.size_slider, stretch=1)  # Stretch to fill
        
        self.size_label = QLabel("20")
        self.size_label.setStyleSheet("color: white; min-width: 25px;")
        tools_layout.addWidget(self.size_label)
        
        # Box count
        self.box_count_label = QLabel("0")
        self.box_count_label.setStyleSheet("color: white; font-weight: bold; min-width: 20px;")
        self.box_count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tools_layout.addWidget(self.box_count_label)
        self.viewer.rectangle_created.connect(self._update_box_count)
        self.viewer.rectangle_deleted.connect(self._update_box_count)
        # Persist rectangles when created/deleted
        self.viewer.rectangle_created.connect(lambda _: self._persist_rectangles_state())
        self.viewer.rectangle_deleted.connect(lambda _: self._persist_rectangles_state())
        
        layout.addWidget(self.tools_frame)
        
        # Translation workflow buttons (compact, single row)
        self.workflow_frame = self._create_tool_frame("Translation Workflow")
        workflow_layout = QHBoxLayout(self.workflow_frame)
        workflow_layout.setContentsMargins(3, 3, 3, 3)
        workflow_layout.setSpacing(2)
        
        self.detect_btn = self._create_compact_button("Detect Text")
        self.detect_btn.clicked.connect(self.detect_text_clicked.emit)
        workflow_layout.addWidget(self.detect_btn, stretch=1)
        
        self.clean_btn = self._create_compact_button("Clean")
        self.clean_btn.clicked.connect(self.clean_image_clicked.emit)
        workflow_layout.addWidget(self.clean_btn, stretch=1)
        
        self.recognize_btn = self._create_compact_button("Recognize Text")
        self.recognize_btn.clicked.connect(lambda: self._emit_recognize_signal())
        workflow_layout.addWidget(self.recognize_btn, stretch=1)
        
        self.translate_btn = self._create_compact_button("Translate")
        self.translate_btn.clicked.connect(lambda: self._emit_translate_signal())
        workflow_layout.addWidget(self.translate_btn, stretch=1)
        
        # Conditionally show Translate All only when experimental flag is enabled
        enable_translate_all = False
        try:
            if self.main_gui and hasattr(self.main_gui, 'config') and isinstance(self.main_gui.config, dict):
                cfg = self.main_gui.config
                # Either flat key or nested under an experimental block
                enable_translate_all = bool(
                    cfg.get('experimental_translate_all', False) or
                    (isinstance(cfg.get('experimental'), dict) and cfg.get('experimental', {}).get('translate_all', False))
                )
            # Environment override
            if not enable_translate_all:
                enable_translate_all = (os.getenv('EXPERIMENTAL_TRANSLATE_ALL', '0') == '1')
        except Exception:
            enable_translate_all = False
        
        if enable_translate_all:
            self.translate_all_btn = self._create_compact_button("Translate All")
            self.translate_all_btn.clicked.connect(lambda: self._emit_translate_all_signal())
            self.translate_all_btn.setToolTip("Translate all images in preview (experimental)")
            workflow_layout.addWidget(self.translate_all_btn, stretch=1)
        else:
            # Do not create the button at all when experimental is disabled
            self.translate_all_btn = None
        
        layout.addWidget(self.workflow_frame)
        
        # Current file label with download button
        file_info_container = QWidget()
        file_info_layout = QHBoxLayout(file_info_container)
        file_info_layout.setContentsMargins(0, 0, 0, 0)
        file_info_layout.setSpacing(8)
        
        self.file_label = QLabel("No image loaded")
        self.file_label.setStyleSheet("color: gray; font-size: 8pt;")
        self.file_label.setWordWrap(True)
        file_info_layout.addWidget(self.file_label, stretch=1)
        
        # Download Images button
        self.download_btn = QPushButton("📥 Download Images")
        self.download_btn.setMinimumHeight(24)
        self.download_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: 1px solid #28a745;
                border-radius: 3px;
                padding: 4px 10px;
                font-size: 8pt;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #218838;
                border-color: #218838;
            }
            QPushButton:pressed {
                background-color: #1e7e34;
            }
            QPushButton:disabled {
                background-color: #1a1a1a;
                color: #666666;
                border-color: #2a2a2a;
            }
        """)
        self.download_btn.wheelEvent = lambda event: None  # Disable scroll wheel
        self.download_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # Prevent focus on click
        self.download_btn.clicked.connect(self._on_download_images_clicked)
        self.download_btn.setEnabled(False)  # Initially disabled until images are translated
        file_info_layout.addWidget(self.download_btn)
        
        layout.addWidget(file_info_container)
        
        # Apply initial button visibility based on saved manual editing state
        # (Call the toggle handler to set visibility correctly on startup)
        initial_state = Qt.CheckState.Checked.value if self.manual_editing_enabled else Qt.CheckState.Unchecked.value
        self._on_manual_editing_toggled(initial_state)
    
    def _create_tool_button(self, text: str, tooltip: str = "") -> QToolButton:
        """Create a compact tool button"""
        btn = QToolButton()
        btn.setText(text)
        btn.setToolTip(tooltip)
        btn.setStyleSheet("""
            QToolButton {
                background-color: #2d2d2d;
                color: white;
                border: 1px solid #3a3a3a;
                border-radius: 3px;
                padding: 3px;
                font-size: 11pt;
                min-width: 28px;
                min-height: 28px;
                max-width: 32px;
                max-height: 32px;
            }
            QToolButton:hover {
                background-color: #3a3a3a;
                border-color: #5a5a5a;
            }
            QToolButton:checked {
                background-color: #5a9fd4;
                border-color: #7bb3e0;
            }
        """)
        # Prevent focus and scroll issues
        btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        btn.wheelEvent = lambda event: None
        return btn
    
    def _create_compact_button(self, text: str) -> QPushButton:
        """Create a compact workflow button"""
        btn = QPushButton(text)
        btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        btn.setStyleSheet("""
            QPushButton {
                background-color: #2d2d2d;
                color: white;
                border: 1px solid #3a3a3a;
                border-radius: 3px;
                padding: 4px 6px;
                font-size: 8pt;
            }
            QPushButton:hover {
                background-color: #3a3a3a;
            }
            QPushButton:pressed {
                background-color: #1a1a1a;
            }
            QPushButton:disabled {
                background-color: #1a1a1a;
                color: #666666;
                border-color: #2a2a2a;
            }
        """)
        # Disable scroll wheel focus
        btn.wheelEvent = lambda event: None
        # Prevent button from taking keyboard/mouse focus (prevents scroll on click)
        btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        return btn
    
    def _create_tool_frame(self, title: str) -> QFrame:
        """Create a labeled frame for tool groups"""
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        frame.setStyleSheet(f"""
            QFrame {{
                background-color: #1e1e1e;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
            }}
        """)
        
        # Add title as data attribute
        frame.setProperty("title", title)
        return frame
    
    def _active_viewer(self) -> CompactImageViewer:
        """Return the viewer for the active tab (source or output)."""
        try:
            return self.output_viewer if (self.viewer_tabs.currentIndex() == 1) else self.viewer
        except Exception:
            return self.viewer
    
    def _set_tool(self, tool: str):
        """Set active tool and update button states"""
        print(f"[DEBUG] Setting tool to: {tool}")
        # Always set on source viewer
        self.viewer.set_tool(tool)
        # For pan tool, also apply to output viewer so panning works there
        try:
            if tool == 'pan' and hasattr(self, 'output_viewer') and self.output_viewer:
                self.output_viewer.set_tool('pan')
        except Exception:
            pass
        
        # Update button states
        self.hand_tool_btn.setChecked(tool == 'pan')
        self.box_draw_btn.setChecked(tool == 'box_draw')
        self.brush_btn.setChecked(tool == 'brush')
        self.eraser_btn.setChecked(tool == 'eraser')
        
        # Show user feedback
        tool_names = {
            'pan': 'Pan/Select Tool',
            'box_draw': 'Box Drawing Tool', 
            'brush': 'Brush Tool (Red strokes)',
            'eraser': 'Eraser Tool (White strokes)'
        }
        print(f"[DEBUG] Active tool: {tool_names.get(tool, tool)}")
    
    def _fit_active_view(self):
        """Fit the active viewer to view (source or output)."""
        v = self._active_viewer()
        try:
            v.fitInView()
        except Exception:
            pass
    
    def _zoom_in(self):
        """Zoom in on the active viewer"""
        v = self._active_viewer()
        try:
            v.scale(1.15, 1.15)
            if hasattr(v, 'user_zoomed'):
                v.user_zoomed = True
        except Exception:
            pass
    
    def _zoom_out(self):
        """Zoom out on the active viewer"""
        v = self._active_viewer()
        try:
            v.scale(1 / 1.15, 1 / 1.15)
            if hasattr(v, 'user_zoomed'):
                v.user_zoomed = True
        except Exception:
            pass
    
    def _update_box_count(self):
        """Update box count display"""
        self.box_count_label.setText(str(len(self.viewer.rectangles)))
    
    def _persist_rectangles_state(self):
        """Persist current rectangles (viewer_rectangles only) to image state manager.
        Avoid updating detection_regions here to prevent unintended remapping of all regions.
        """
        try:
            if not hasattr(self, 'manga_integration') or not self.manga_integration:
                return
            image_path = getattr(self, 'current_image_path', None)
            if not image_path:
                return
            # Collect current viewer rectangles geometry for persistence
            rect_data = []
            try:
                for rect_item in getattr(self.viewer, 'rectangles', []) or []:
                    r = rect_item.sceneBoundingRect()
                    rect_data.append({'x': r.x(), 'y': r.y(), 'width': r.width(), 'height': r.height()})
            except Exception:
                rect_data = []
            # Persist ONLY viewer_rectangles to avoid cascading changes to all overlays
            if hasattr(self.manga_integration, 'image_state_manager') and self.manga_integration.image_state_manager:
                # Merge into existing state without touching detection_regions
                prev = self.manga_integration.image_state_manager.get_state(image_path) or {}
                prev['viewer_rectangles'] = rect_data
                self.manga_integration.image_state_manager.set_state(image_path, prev, save=True)
        except Exception:
            pass
    
    def _update_brush_size(self, value: int):
        """Update brush/eraser size"""
        self.size_label.setText(str(value))
        self.viewer.brush_size = value
        self.viewer.eraser_size = value
    
    def _clear_strokes(self):
        """Clear brush strokes only"""
        self.viewer.clear_brush_strokes()
    
    def _on_clear_boxes_clicked(self):
        """Clear all rectangles and persist deletion to state."""
        try:
            self.viewer.clear_rectangles()
            self._update_box_count()
            # Clear detection state aggressively (both memory and persisted)
            try:
                if hasattr(self, 'manga_integration') and self.manga_integration:
                    if hasattr(self.manga_integration, '_clear_detection_state_for_image'):
                        self.manga_integration._clear_detection_state_for_image(self.current_image_path)
            except Exception:
                pass
            # Persist empty state (viewer_rectangles/detection_regions)
            self._persist_rectangles_state()
        except Exception:
            pass
    
    def _clear_all(self):
        """Clear all boxes and strokes"""
        self.viewer.clear_rectangles()
        self.viewer.clear_brush_strokes()
        self._update_box_count()
        self._persist_rectangles_state()
    
    def load_image(self, image_path: str, preserve_rectangles: bool = False, preserve_text_overlays: bool = False):
        """Load an image into the preview (async)
        
        Args:
            image_path: Path to the image to load
            preserve_rectangles: If True, don't clear rectangles when loading (for workflow continuity)
            preserve_text_overlays: If True, keep text overlays visible (for cleaned images)
        """
        # Store the preserve flags for use in _on_image_loading_started
        self._preserve_rectangles_on_load = preserve_rectangles
        self._preserve_text_overlays_on_load = preserve_text_overlays
        
        # Start loading (happens in background)
        self.viewer.load_image(image_path)
        self.current_image_path = image_path
        
        # Check for translated output and load if available
        self._check_and_load_translated_output(image_path)
        
        # Update thumbnail selection
        self._update_thumbnail_selection(image_path)
    
    def _emit_recognize_signal(self):
        """Emit recognize text signal with debugging"""
        print("[DEBUG] Recognize button clicked - emitting signal")
        self.recognize_text_clicked.emit()
        print("[DEBUG] Recognize signal emitted")
    
    def _emit_translate_signal(self):
        """Emit translate text signal with debugging"""
        print("[TRANSLATE_BUTTON] Translate button clicked - emitting signal", flush=True)
        self.translate_text_clicked.emit()
        print("[TRANSLATE_BUTTON] Translate signal emitted", flush=True)
    
    def _emit_translate_all_signal(self):
        """Emit translate all signal with debugging"""
        print("[TRANSLATE_ALL_BUTTON] Translate All button clicked - emitting signal", flush=True)
        self.translate_all_clicked.emit()
        print("[TRANSLATE_ALL_BUTTON] Translate All signal emitted", flush=True)
    
    def set_image_list(self, image_paths: list):
        """Set the list of images and populate thumbnails"""
        self.image_paths = image_paths
        self._populate_thumbnails()
        
        # Show/hide thumbnail lists based on image count
        if len(image_paths) <= 1:
            self.thumbnail_list.setVisible(False)
            self.output_thumbnail_list.setVisible(False)
        else:
            self.thumbnail_list.setVisible(True)
            self.output_thumbnail_list.setVisible(True)
    
    def _populate_thumbnails(self):
        """Populate thumbnail list with images"""
        self.thumbnail_list.clear()
        self.output_thumbnail_list.clear()
        
        from PySide6.QtWidgets import QListWidgetItem
        from PySide6.QtCore import Qt, QSize
        import threading
        
        # Load thumbnails in background to avoid blocking
        def load_thumb(path, index):
            try:
                pixmap = QPixmap(path)
                if not pixmap.isNull():
                    # Scale to thumbnail size
                    thumb = pixmap.scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio, 
                                         Qt.TransformationMode.SmoothTransformation)
                    return (index, path, thumb)
            except:
                pass
            return None
        
        # Add items with loading placeholders first
        for i, path in enumerate(self.image_paths):
            # Add to source thumbnail list
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, path)  # Store path
            item.setText(os.path.basename(path))
            item.setSizeHint(QSize(110, 110))
            self.thumbnail_list.addItem(item)
            
            # Add to output thumbnail list
            output_item = QListWidgetItem()
            output_item.setData(Qt.ItemDataRole.UserRole, path)
            output_item.setText(os.path.basename(path))
            output_item.setSizeHint(QSize(110, 110))
            self.output_thumbnail_list.addItem(output_item)
        
        # Load thumbnails asynchronously
        def load_all_thumbs():
            for i, path in enumerate(self.image_paths):
                result = load_thumb(path, i)
                if result:
                    idx, img_path, thumb = result
                    # Update items on main thread (both lists)
                    try:
                        from PySide6.QtCore import QMetaObject, Q_ARG
                        QMetaObject.invokeMethod(self, "_update_thumbnail_item",
                                                Qt.ConnectionType.QueuedConnection,
                                                Q_ARG(int, idx),
                                                Q_ARG(QPixmap, thumb),
                                                Q_ARG(str, img_path))
                        QMetaObject.invokeMethod(self, "_update_output_thumbnail_item",
                                                Qt.ConnectionType.QueuedConnection,
                                                Q_ARG(int, idx),
                                                Q_ARG(QPixmap, thumb),
                                                Q_ARG(str, img_path))
                    except:
                        pass
        
        # Start background loading
        thread = threading.Thread(target=load_all_thumbs, daemon=True)
        thread.start()
    
    @Slot(int, QPixmap, str)
    def _update_thumbnail_item(self, index: int, pixmap: QPixmap, path: str):
        """Update thumbnail item with loaded pixmap (called from background thread)"""
        try:
            if index < self.thumbnail_list.count():
                item = self.thumbnail_list.item(index)
                icon = QIcon(pixmap)
                item.setIcon(icon)
                item.setText("")  # Remove text once icon is loaded
        except Exception as e:
            print(f"Error updating thumbnail: {e}")
    
    @Slot(int, QPixmap, str)
    def _update_output_thumbnail_item(self, index: int, pixmap: QPixmap, path: str):
        """Update output thumbnail item with loaded pixmap (called from background thread)"""
        try:
            if index < self.output_thumbnail_list.count():
                item = self.output_thumbnail_list.item(index)
                icon = QIcon(pixmap)
                item.setIcon(icon)
                item.setText("")  # Remove text once icon is loaded
        except Exception as e:
            print(f"Error updating output thumbnail: {e}")
    
    def _update_thumbnail_selection(self, image_path: str):
        """Update which thumbnail is selected based on current image"""
        # Update both thumbnail lists
        for i in range(self.thumbnail_list.count()):
            item = self.thumbnail_list.item(i)
            item_path = item.data(Qt.ItemDataRole.UserRole)
            if item_path == image_path:
                self.thumbnail_list.setCurrentItem(item)
                break
        
        for i in range(self.output_thumbnail_list.count()):
            item = self.output_thumbnail_list.item(i)
            item_path = item.data(Qt.ItemDataRole.UserRole)
            if item_path == image_path:
                self.output_thumbnail_list.setCurrentItem(item)
                break
    
    def _on_thumbnail_clicked(self, item):
        """Handle thumbnail click - load the corresponding image"""
        from PySide6.QtCore import Qt
        image_path = item.data(Qt.ItemDataRole.UserRole)
        if image_path and os.path.exists(image_path):
            self.load_image(image_path)
    
    @Slot(str)
    def _on_image_loading_started(self, image_path: str):
        """Handle image loading started"""
        # Show loading status immediately
        self.file_label.setText(f"⌛ Loading {os.path.basename(image_path)}...")
        self.file_label.setStyleSheet("color: #5a9fd4; font-size: 8pt;")  # Blue while loading
        
        # Clear previous image data (unless preserve flag is set for workflow continuity)
        preserve_rectangles = getattr(self, '_preserve_rectangles_on_load', False)
        if not preserve_rectangles:
            self.viewer.clear_rectangles()
            self._clear_strokes()
            self._update_box_count()
        else:
            # Only clear strokes, keep rectangles for workflow continuity (e.g., after cleaning)
            self._clear_strokes()
        
        # Also hide any existing translated text overlays immediately to prevent carryover
        try:
            if hasattr(self, 'manga_integration') and self.manga_integration:
                # This hides all overlays and will only show those for the incoming image if any
                self.manga_integration.show_text_overlays_for_image(image_path)
                # IMPORTANT: Clear per-image recognition/translation caches to prevent cross-image reuse
                try:
                    if hasattr(self.manga_integration, '_recognized_texts'):
                        del self.manga_integration._recognized_texts
                    if hasattr(self.manga_integration, '_recognized_texts_image_path'):
                        del self.manga_integration._recognized_texts_image_path
                    if hasattr(self.manga_integration, '_recognition_data'):
                        del self.manga_integration._recognition_data
                    if hasattr(self.manga_integration, '_translation_data'):
                        del self.manga_integration._translation_data
                except Exception:
                    pass
        except Exception:
            pass
        
        # Reset the preserve flags after use
        self._preserve_rectangles_on_load = False
        self._preserve_text_overlays_on_load = False
    
    @Slot(str)
    def _on_image_loaded_success(self, image_path: str):
        """Handle successful image load"""
        if self.current_image_path:
            self.file_label.setText(f"📄 {os.path.basename(self.current_image_path)}")
            self.file_label.setStyleSheet("color: gray; font-size: 8pt;")  # Back to normal
        
        # Restore overlays for the newly loaded image (and hide others)
        try:
            if hasattr(self, 'manga_integration') and self.manga_integration:
                # Restore detection/recognition overlays
                self.manga_integration._restore_image_state_overlays_only(self.current_image_path)
                # Attach recognition tooltips/updates
                if hasattr(self.manga_integration, 'show_recognized_overlays_for_image'):
                    self.manga_integration.show_recognized_overlays_for_image(self.current_image_path)
                # Restore translated text overlays
                self.manga_integration.show_text_overlays_for_image(self.current_image_path)
        except Exception:
            pass
        
    def _on_tab_changed(self, index: int):
        """Refit only if user hasn't manually zoomed, to avoid resetting zoom."""
        try:
            if index == 0 and self.viewer.hasPhoto() and not getattr(self.viewer, 'user_zoomed', False):
                self.viewer.fitInView()
            elif index == 1 and self.output_viewer.hasPhoto() and not getattr(self.output_viewer, 'user_zoomed', False):
                self.output_viewer.fitInView()
        except Exception:
            pass
    
    def clear(self):
        """Clear the preview"""
        # Clear source viewer
        self.viewer.clear_scene()
        self.current_image_path = None
        self.file_label.setText("No image loaded")
        self._update_box_count()
        self._show_placeholder_icon()
        
        # Clear translated output viewer as well (cosmetic reset)
        try:
            if hasattr(self, 'output_viewer') and self.output_viewer:
                self.output_viewer.clear_scene()
                self.current_translated_path = None
                self._show_output_placeholder()
        except Exception:
            pass
    
    def _show_placeholder_icon(self):
        """Display Halgakos_NoChibi.png as placeholder when no image is loaded"""
        # Try NoChibi PNG first (high resolution), then regular PNG, fallback to ICO
        nochibi_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Halgakos_NoChibi.png')
        png_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Halgakos.png')
        ico_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Halgakos.ico')
        
        if os.path.exists(nochibi_path):
            icon_path = nochibi_path
        elif os.path.exists(png_path):
            icon_path = png_path
        else:
            icon_path = ico_path
        
        if os.path.exists(icon_path) and not self.viewer.hasPhoto():
            try:
                from PySide6.QtGui import QPainter
                
                # Load the high-resolution PNG at full size (no scaling)
                placeholder_pixmap = QPixmap(icon_path)
                
                if not placeholder_pixmap.isNull():
                    # Create a semi-transparent version without any scaling
                    transparent_pixmap = QPixmap(placeholder_pixmap.size())
                    transparent_pixmap.fill(Qt.GlobalColor.transparent)
                    
                    painter = QPainter(transparent_pixmap)
                    painter.setOpacity(0.15)  # 15% opacity for subtle watermark
                    painter.drawPixmap(0, 0, placeholder_pixmap)
                    painter.end()
                    
                    # Set the full-size transparent image
                    self.viewer.setPhoto(transparent_pixmap)
            except Exception as e:
                # Silently fail if icon can't be loaded
                pass
    
    def _show_output_placeholder(self):
        """Display WhereIsMyOutput.png as placeholder for no output, scaled to fill like Halgakos_NoChibi."""
        placeholder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'WhereIsMyOutput.png')
        
        if os.path.exists(placeholder_path):
            try:
                # Load at full size (no pre-scaling)
                placeholder_pixmap = QPixmap(placeholder_path)
                
                if not placeholder_pixmap.isNull():
                    # Set and fit; resizeEvent and tab change will keep it filling the area
                    self.output_viewer.setPhoto(placeholder_pixmap)
            except Exception as e:
                print(f"Error loading output placeholder: {e}")
    
    def _update_toggle_label(self, state):
        """Update the toggle label to show visual feedback"""
        from PySide6.QtCore import Qt
        if state == Qt.CheckState.Checked.value:
            # Blue text when enabled
            self.manual_editing_toggle.setStyleSheet(self.manual_editing_toggle.styleSheet().replace(
                "color: white;",
                "color: #5a9fd4;"
            ))
        else:
            # White text when disabled
            self.manual_editing_toggle.setStyleSheet(self.manual_editing_toggle.styleSheet().replace(
                "color: #5a9fd4;",
                "color: white;"
            ))
    
    def _on_manual_editing_toggled(self, state):
        """Handle manual editing toggle - show/hide manual tools and workflow buttons"""
        from PySide6.QtCore import Qt
        enabled = (state == Qt.CheckState.Checked.value)
        
        print(f"[DEBUG] Manual editing toggled: {enabled}")
        
        # Show/hide manual editing tools ONLY (not pan/zoom which are always visible)
        self.box_draw_btn.setVisible(enabled)
        self.brush_btn.setVisible(enabled)
        self.eraser_btn.setVisible(enabled)
        self.delete_btn.setVisible(enabled)
        self.clear_boxes_btn.setVisible(enabled)
        self.clear_strokes_btn.setVisible(enabled)
        self.size_slider.setVisible(enabled)
        self.size_label.setVisible(enabled)
        self.box_count_label.setVisible(enabled)
        
        # Keep the Translation Workflow row ALWAYS visible; only toggle manual tools
        # Pan and zoom buttons are ALWAYS visible (explicitly ensure they stay visible)
        self.hand_tool_btn.setVisible(True)
        self.fit_btn.setVisible(True)
        self.zoom_in_btn.setVisible(True)
        self.zoom_out_btn.setVisible(True)
        
        # If hiding tools, reset to pan tool
        if not enabled:
            self._set_tool('pan')
        
        # Store state for later use in preview mode
        self.manual_editing_enabled = enabled
        
        # Persist state to config (same pattern as other checkboxes)
        try:
            if self.main_gui and hasattr(self.main_gui, 'config'):
                self.main_gui.config['manga_manual_editing_enabled'] = enabled
                print(f"[DEBUG] Set manga_manual_editing_enabled = {enabled}")
                # Save config to disk
                if hasattr(self.main_gui, 'save_config'):
                    self.main_gui.save_config(show_message=False)
                    print(f"[DEBUG] Config saved to disk")
                else:
                    print(f"[DEBUG] save_config method not found on main_gui")
            else:
                print(f"[DEBUG] main_gui or config not available")
        except Exception as e:
            print(f"[DEBUG] Error saving config: {e}")
            import traceback
            traceback.print_exc()
    
            print(f"[DEBUG] Error loading manual file: {str(e)}")
            import traceback
            print(traceback.format_exc())
    
    def _on_download_images_clicked(self):
        """Handle download images button - consolidate isolated images into structured folder"""
        try:
            from PySide6.QtWidgets import QFileDialog, QMessageBox
            import shutil
            import glob
            
            # Get the parent directory of translated images
            # Since each image is now in its own *_translated/ folder, we need to find them
            if not hasattr(self, 'manga_integration') or not self.manga_integration:
                print("[DEBUG] No manga_integration reference available")
                return
            
            # Get the source directory from selected files
            if not hasattr(self.manga_integration, 'selected_files') or not self.manga_integration.selected_files:
                print("[DEBUG] No selected files available")
                return
            
            # Get parent directory of the first selected file
            first_file = self.manga_integration.selected_files[0]
            parent_dir = os.path.dirname(first_file)
            
            # Find all *_translated folders in the parent directory
            translated_folders = []
            for item in os.listdir(parent_dir):
                item_path = os.path.join(parent_dir, item)
                if os.path.isdir(item_path) and item.endswith('_translated'):
                    translated_folders.append(item_path)
            
            if not translated_folders:
                QMessageBox.warning(
                    self,
                    "No Translated Images",
                    "No translated images found. Please translate some images first."
                )
                return
            
            print(f"[DEBUG] Found {len(translated_folders)} translated folders")
            
            # Create consolidated "translated" folder in parent directory
            consolidated_folder = os.path.join(parent_dir, 'translated')
            os.makedirs(consolidated_folder, exist_ok=True)
            
            # Copy all images from isolated folders to consolidated folder
            image_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif')
            copied_count = 0
            
            for folder in translated_folders:
                for filename in os.listdir(folder):
                    if filename.lower().endswith(image_extensions):
                        src_path = os.path.join(folder, filename)
                        dest_path = os.path.join(consolidated_folder, filename)
                        
                        # Handle potential filename conflicts
                        if os.path.exists(dest_path):
                            base, ext = os.path.splitext(filename)
                            counter = 1
                            while os.path.exists(dest_path):
                                dest_path = os.path.join(consolidated_folder, f"{base}_{counter}{ext}")
                                counter += 1
                        
                        shutil.copy2(src_path, dest_path)
                        copied_count += 1
            
            print(f"[DEBUG] Successfully consolidated {copied_count} translated images to {consolidated_folder}")
            
            # Show success message
            QMessageBox.information(
                self,
                "Download Complete",
                f"Successfully consolidated {copied_count} translated images into:\n{consolidated_folder}\n\n"
                f"All images from separate folders have been combined into a single 'translated' folder."
            )
            
        except Exception as e:
            print(f"[DEBUG] Error downloading images: {str(e)}")
            import traceback
            print(traceback.format_exc())
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(
                self,
                "Download Error",
                f"Failed to consolidate images:\n{str(e)}"
            )
    
    def set_translated_folder(self, folder_path: str):
        """Set the translated folder path and enable download button"""
        self.translated_folder_path = folder_path
        
        # Check for isolated *_translated folders in parent directory
        has_translated_images = False
        
        if folder_path and os.path.exists(folder_path):
            parent_dir = os.path.dirname(folder_path) if os.path.isfile(folder_path) else folder_path
            
            # Look for isolated *_translated folders
            image_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif')
            for item in os.listdir(parent_dir):
                item_path = os.path.join(parent_dir, item)
                if os.path.isdir(item_path) and item.endswith('_translated'):
                    # Check if this folder has images
                    if any(f.lower().endswith(image_extensions) for f in os.listdir(item_path)):
                        has_translated_images = True
                        break
            
            # Also check the folder_path itself for backward compatibility
            if not has_translated_images and os.path.isdir(folder_path):
                has_translated_images = any(
                    f.lower().endswith(image_extensions) 
                    for f in os.listdir(folder_path)
                )
            
            self.download_btn.setEnabled(has_translated_images)
            print(f"[DEBUG] Translated images found: {has_translated_images}")
        else:
            self.download_btn.setEnabled(False)
    
    def _check_and_load_translated_output(self, source_image_path: str):
        """Check for translated output and load into output tab if available.
        If none found, fall back to cleaned image if present."""
        try:
            # Build the expected translated path based on isolated folder structure
            source_dir = os.path.dirname(source_image_path)
            source_filename = os.path.basename(source_image_path)
            source_name_no_ext = os.path.splitext(source_filename)[0]

            # Look for translated version in isolated folder
            translated_folder = os.path.join(source_dir, f"{source_name_no_ext}_translated")

            # Helper to load an output image
            def _load_output(path: str):
                if path and os.path.exists(path):
                    print(f"[DEBUG] Loading output image into output tab: {path}")
                    self.output_viewer.load_image(path)
                    self.current_translated_path = path
                    return True
                return False

            # 1) Prefer any translated image inside the isolated folder
            if os.path.exists(translated_folder) and os.path.isdir(translated_folder):
                image_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif')
                for filename in os.listdir(translated_folder):
                    if filename.lower().endswith(image_extensions):
                        translated_path = os.path.join(translated_folder, filename)
                        if _load_output(translated_path):
                            return

            # 2) No translated image found — try cleaned image from state or naming convention
            cleaned_path = None
            try:
                if hasattr(self, 'manga_integration') and self.manga_integration and \
                   hasattr(self.manga_integration, 'image_state_manager') and self.manga_integration.image_state_manager:
                    state = self.manga_integration.image_state_manager.get_state(source_image_path) or {}
                    cand = state.get('cleaned_image_path')
                    if cand and os.path.exists(cand):
                        cleaned_path = cand
            except Exception:
                pass

            if not cleaned_path and os.path.exists(translated_folder):
                # Look for files matching *_cleaned.* inside the isolated folder
                image_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif')
                for filename in os.listdir(translated_folder):
                    name_lower = filename.lower()
                    if name_lower.startswith(f"{source_name_no_ext.lower()}_cleaned") and name_lower.endswith(image_extensions):
                        cleaned_path = os.path.join(translated_folder, filename)
                        break

            if cleaned_path and _load_output(cleaned_path):
                print(f"[DEBUG] No translated image found; showing cleaned image instead: {cleaned_path}")
                return

            # 3) Nothing found — show placeholder
            print(f"[DEBUG] No translated or cleaned output found for: {source_filename}")
            self._show_output_placeholder()
            self.current_translated_path = None

        except Exception as e:
            print(f"[DEBUG] Error checking translated output: {str(e)}")
            import traceback
            print(traceback.format_exc())
