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
                               QSizePolicy, QListWidget, QListWidgetItem)
from PySide6.QtCore import Qt, Signal, QRectF, QPointF, QSize, QThread, QObject, Slot
from PySide6.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QBrush, QCursor, QIcon
import numpy as np


class MoveableRectItem(QGraphicsRectItem):
    """Simple moveable rectangle for text box selection"""
    
    def __init__(self, rect=None, parent=None):
        super().__init__(rect, parent)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self.setBrush(QBrush(QColor(255, 192, 203, 125)))  # Transparent pink
        self.setPen(QPen(QColor(255, 0, 0), 2))
        self.setZValue(1)
        self.selected = False
        
    def hoverEnterEvent(self, event):
        self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        super().hoverEnterEvent(event)
        
    def hoverLeaveEvent(self, event):
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        super().hoverLeaveEvent(event)


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
        
        # State
        self.empty = True
        self.zoom_level = 0
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
        """Set the photo to display"""
        if pixmap and not pixmap.isNull():
            self.empty = False
            self.photo.setPixmap(pixmap)
            # Schedule fitInView after pixmap is set to ensure proper sizing
            from PySide6.QtCore import QTimer
            QTimer.singleShot(10, self.fitInView)
        else:
            self.empty = True
            self.photo.setPixmap(QPixmap())
        self.zoom_level = 0
    
    def fitInView(self):
        """Fit the image to view"""
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
        """Handle zoom with mouse wheel (requires Shift key)"""
        # Only zoom if Shift key is held, otherwise allow normal scrolling
        if self.hasPhoto() and event.modifiers() == Qt.KeyboardModifier.ShiftModifier:
            factor = 1.15
            if event.angleDelta().y() > 0:
                self.scale(factor, factor)
                self.zoom_level += 1
            else:
                self.scale(1 / factor, 1 / factor)
                self.zoom_level -= 1
        else:
            # Pass event to parent for normal scrolling
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
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_image_path = None
        self._build_ui()
    
    def _build_ui(self):
        """Build the compact UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Title
        title_label = QLabel("📷 Image Preview & Editing")
        title_font = title_label.font()
        title_font.setBold(True)
        title_font.setPointSize(10)
        title_label.setFont(title_font)
        layout.addWidget(title_label)
        
        # Create horizontal layout for viewer + thumbnails
        viewer_container = QWidget()
        viewer_layout = QHBoxLayout(viewer_container)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        viewer_layout.setSpacing(5)
        
        # Image viewer (main)
        self.viewer = CompactImageViewer()
        self.viewer.setMinimumHeight(300)
        viewer_layout.addWidget(self.viewer, stretch=1)
        
        # Thumbnail list (right side)
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
        self.thumbnail_list.setVisible(False)  # Hidden by default
        viewer_layout.addWidget(self.thumbnail_list)
        
        # Store image paths for thumbnails
        self.image_paths = []
        
        layout.addWidget(viewer_container, stretch=1)
        
        # Connect loading signals to show status
        self.viewer.image_loading.connect(self._on_image_loading_started)
        self.viewer.image_loaded.connect(self._on_image_loaded_success)
        
        # Display placeholder icon when no image is loaded
        self._show_placeholder_icon()
        
        # Manual Tools - Row 1: Box Drawing & Inpainting
        tools_frame = self._create_tool_frame("Manual Tools")
        tools_layout = QHBoxLayout(tools_frame)
        tools_layout.setContentsMargins(3, 3, 3, 3)
        tools_layout.setSpacing(3)
        
        self.hand_tool_btn = self._create_tool_button("✅", "Pan")
        self.hand_tool_btn.setCheckable(True)
        self.hand_tool_btn.setChecked(True)  # Default to Pan
        self.hand_tool_btn.clicked.connect(lambda: self._set_tool('pan'))
        tools_layout.addWidget(self.hand_tool_btn)
        
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
        self.clear_boxes_btn.clicked.connect(self.viewer.clear_rectangles)
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
        
        # Add zoom controls to the same row
        self.fit_btn = self._create_tool_button("⊟", "Fit to View")
        self.fit_btn.clicked.connect(self.viewer.fitInView)
        tools_layout.addWidget(self.fit_btn)
        
        self.zoom_in_btn = self._create_tool_button("🔍+", "Zoom In (Shift+Wheel)")
        self.zoom_in_btn.clicked.connect(self._zoom_in)
        tools_layout.addWidget(self.zoom_in_btn)
        
        self.zoom_out_btn = self._create_tool_button("🔍-", "Zoom Out (Shift+Wheel)")
        self.zoom_out_btn.clicked.connect(self._zoom_out)
        tools_layout.addWidget(self.zoom_out_btn)
        
        layout.addWidget(tools_frame)
        
        # Translation workflow buttons (compact, single row)
        workflow_frame = self._create_tool_frame("Translation Workflow")
        workflow_layout = QHBoxLayout(workflow_frame)
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
        
        layout.addWidget(workflow_frame)
        
        # Current file label
        self.file_label = QLabel("No image loaded")
        self.file_label.setStyleSheet("color: gray; font-size: 8pt;")
        self.file_label.setWordWrap(True)
        layout.addWidget(self.file_label)
    
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
        """)
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
    
    def _set_tool(self, tool: str):
        """Set active tool and update button states"""
        print(f"[DEBUG] Setting tool to: {tool}")
        self.viewer.set_tool(tool)
        
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
    
    def _zoom_in(self):
        """Zoom in"""
        self.viewer.scale(1.15, 1.15)
    
    def _zoom_out(self):
        """Zoom out"""
        self.viewer.scale(1 / 1.15, 1 / 1.15)
    
    def _update_box_count(self):
        """Update box count display"""
        self.box_count_label.setText(str(len(self.viewer.rectangles)))
    
    def _update_brush_size(self, value: int):
        """Update brush/eraser size"""
        self.size_label.setText(str(value))
        self.viewer.brush_size = value
        self.viewer.eraser_size = value
    
    def _clear_strokes(self):
        """Clear brush strokes only"""
        self.viewer.clear_brush_strokes()
    
    def _clear_all(self):
        """Clear all boxes and strokes"""
        self.viewer.clear_rectangles()
        self.viewer.clear_brush_strokes()
        self._update_box_count()
    
    def load_image(self, image_path: str):
        """Load an image into the preview (async)"""
        # Start loading (happens in background)
        self.viewer.load_image(image_path)
        self.current_image_path = image_path
        
        # Update thumbnail selection
        self._update_thumbnail_selection(image_path)
    
    def _emit_recognize_signal(self):
        """Emit recognize text signal with debugging"""
        print("[DEBUG] Recognize button clicked - emitting signal")
        self.recognize_text_clicked.emit()
        print("[DEBUG] Recognize signal emitted")
    
    def _emit_translate_signal(self):
        """Emit translate text signal with debugging"""
        print("[DEBUG] Translate button clicked - emitting signal")
        self.translate_text_clicked.emit()
        print("[DEBUG] Translate signal emitted")
    
    def set_image_list(self, image_paths: list):
        """Set the list of images and populate thumbnails"""
        self.image_paths = image_paths
        self._populate_thumbnails()
        
        # Hide thumbnail list if there's 0 or 1 images (nothing to switch between)
        if len(image_paths) <= 1:
            self.thumbnail_list.setVisible(False)
            self.thumbnail_list.setMaximumWidth(0)  # Collapse completely
        else:
            self.thumbnail_list.setVisible(True)
            self.thumbnail_list.setMaximumWidth(150)  # Restore width
    
    def _populate_thumbnails(self):
        """Populate thumbnail list with images"""
        self.thumbnail_list.clear()
        
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
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, path)  # Store path
            item.setText(os.path.basename(path))
            item.setSizeHint(QSize(110, 110))
            self.thumbnail_list.addItem(item)
        
        # Load thumbnails asynchronously
        def load_all_thumbs():
            for i, path in enumerate(self.image_paths):
                result = load_thumb(path, i)
                if result:
                    idx, img_path, thumb = result
                    # Update item on main thread
                    try:
                        from PySide6.QtCore import QMetaObject, Q_ARG
                        QMetaObject.invokeMethod(self, "_update_thumbnail_item",
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
    
    def _update_thumbnail_selection(self, image_path: str):
        """Update which thumbnail is selected based on current image"""
        for i in range(self.thumbnail_list.count()):
            item = self.thumbnail_list.item(i)
            item_path = item.data(Qt.ItemDataRole.UserRole)
            if item_path == image_path:
                self.thumbnail_list.setCurrentItem(item)
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
        
        # Clear previous image data
        self.viewer.clear_rectangles()
        self._clear_strokes()
        self._update_box_count()
    
    @Slot(str)
    def _on_image_loaded_success(self, image_path: str):
        """Handle successful image load"""
        if self.current_image_path:
            self.file_label.setText(f"📄 {os.path.basename(self.current_image_path)}")
            self.file_label.setStyleSheet("color: gray; font-size: 8pt;")  # Back to normal
    
    def clear(self):
        """Clear the preview"""
        self.viewer.clear_scene()
        self.current_image_path = None
        self.file_label.setText("No image loaded")
        self._update_box_count()
        # Show placeholder icon when cleared
        self._show_placeholder_icon()
    
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
