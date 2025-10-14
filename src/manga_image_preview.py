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
                               QLabel, QSlider, QFrame, QPushButton, QGraphicsRectItem)
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
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        
        # State
        self.empty = True
        self.zoom_level = 0
        self.current_tool = None
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
        """Handle zoom with mouse wheel"""
        if self.hasPhoto():
            factor = 1.15
            if event.angleDelta().y() > 0:
                self.scale(factor, factor)
                self.zoom_level += 1
            else:
                self.scale(1 / factor, 1 / factor)
                self.zoom_level -= 1
    
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
                self.brush_strokes.append({
                    'start': (self.last_pos.x(), self.last_pos.y()),
                    'end': (current_pos.x(), current_pos.y()),
                    'size': self.brush_size if self.current_tool == 'brush' else self.eraser_size,
                    'type': self.current_tool
                })
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
    
    def clear_rectangles(self):
        """Clear all rectangles"""
        for rect in self.rectangles:
            self._scene.removeItem(rect)
        self.rectangles.clear()
        self.selected_rect = None
    
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
    recognize_text_clicked = Signal()
    get_translations_clicked = Signal()
    segment_text_clicked = Signal()
    clean_image_clicked = Signal()
    render_clicked = Signal()
    
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
        
        # Image viewer
        self.viewer = CompactImageViewer()
        self.viewer.setMinimumHeight(300)
        layout.addWidget(self.viewer, stretch=1)
        
        # Connect loading signals to show status
        self.viewer.image_loading.connect(self._on_image_loading_started)
        self.viewer.image_loaded.connect(self._on_image_loaded_success)
        
        # Zoom controls
        zoom_frame = QFrame()
        zoom_frame.setFrameShape(QFrame.Shape.StyledPanel)
        zoom_layout = QHBoxLayout(zoom_frame)
        zoom_layout.setContentsMargins(3, 3, 3, 3)
        zoom_layout.setSpacing(5)
        
        self.fit_btn = self._create_tool_button("⊡", "Fit to View")
        self.fit_btn.clicked.connect(self.viewer.fitInView)
        zoom_layout.addWidget(self.fit_btn)
        
        self.zoom_in_btn = self._create_tool_button("🔍+", "Zoom In")
        self.zoom_in_btn.clicked.connect(self._zoom_in)
        zoom_layout.addWidget(self.zoom_in_btn)
        
        self.zoom_out_btn = self._create_tool_button("🔍-", "Zoom Out")
        self.zoom_out_btn.clicked.connect(self._zoom_out)
        zoom_layout.addWidget(self.zoom_out_btn)
        
        zoom_layout.addStretch()
        layout.addWidget(zoom_frame)
        
        # Box Drawing tools
        box_frame = self._create_tool_frame("Box Drawing")
        box_layout = QHBoxLayout(box_frame)
        box_layout.setContentsMargins(3, 3, 3, 3)
        box_layout.setSpacing(3)
        
        self.hand_tool_btn = self._create_tool_button("✋", "Pan")
        self.hand_tool_btn.setCheckable(True)
        self.hand_tool_btn.clicked.connect(lambda: self._set_tool('pan'))
        box_layout.addWidget(self.hand_tool_btn)
        
        self.box_draw_btn = self._create_tool_button("▭", "Draw Box")
        self.box_draw_btn.setCheckable(True)
        self.box_draw_btn.setChecked(True)
        self.box_draw_btn.clicked.connect(lambda: self._set_tool('box_draw'))
        box_layout.addWidget(self.box_draw_btn)
        
        self.delete_btn = self._create_tool_button("🗑", "Delete Selected")
        self.delete_btn.clicked.connect(self.viewer.delete_selected_rectangle)
        box_layout.addWidget(self.delete_btn)
        
        self.clear_boxes_btn = self._create_tool_button("🧹", "Clear All Boxes")
        self.clear_boxes_btn.clicked.connect(self.viewer.clear_rectangles)
        box_layout.addWidget(self.clear_boxes_btn)
        
        # Box count
        self.box_count_label = QLabel("0")
        self.box_count_label.setStyleSheet("color: white; font-weight: bold;")
        box_layout.addWidget(self.box_count_label)
        self.viewer.rectangle_created.connect(self._update_box_count)
        self.viewer.rectangle_deleted.connect(self._update_box_count)
        
        box_layout.addStretch()
        layout.addWidget(box_frame)
        
        # Inpainting tools
        inpaint_frame = self._create_tool_frame("Inpainting")
        inpaint_layout = QVBoxLayout(inpaint_frame)
        inpaint_layout.setContentsMargins(3, 3, 3, 3)
        inpaint_layout.setSpacing(3)
        
        # Tool buttons
        tool_row = QHBoxLayout()
        tool_row.setSpacing(3)
        
        self.brush_btn = self._create_tool_button("🖌", "Brush")
        self.brush_btn.setCheckable(True)
        self.brush_btn.clicked.connect(lambda: self._set_tool('brush'))
        tool_row.addWidget(self.brush_btn)
        
        self.eraser_btn = self._create_tool_button("🧽", "Eraser")
        self.eraser_btn.setCheckable(True)
        self.eraser_btn.clicked.connect(lambda: self._set_tool('eraser'))
        tool_row.addWidget(self.eraser_btn)
        
        self.clear_strokes_btn = self._create_tool_button("🧹", "Clear Strokes")
        self.clear_strokes_btn.clicked.connect(self._clear_strokes)
        tool_row.addWidget(self.clear_strokes_btn)
        
        tool_row.addStretch()
        inpaint_layout.addLayout(tool_row)
        
        # Brush size slider
        slider_row = QHBoxLayout()
        slider_row.setSpacing(5)
        
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
        # Disable mouse wheel to prevent accidental scrolling
        self.size_slider.wheelEvent = lambda event: None
        slider_row.addWidget(self.size_slider)
        
        self.size_label = QLabel("20")
        self.size_label.setStyleSheet("color: white; min-width: 25px;")
        slider_row.addWidget(self.size_label)
        
        inpaint_layout.addLayout(slider_row)
        layout.addWidget(inpaint_frame)
        
        # Translation workflow buttons (compact)
        workflow_frame = self._create_tool_frame("Translation Workflow")
        workflow_layout = QVBoxLayout(workflow_frame)
        workflow_layout.setContentsMargins(3, 3, 3, 3)
        workflow_layout.setSpacing(3)
        
        # Row 1
        row1 = QHBoxLayout()
        row1.setSpacing(2)
        
        self.detect_btn = self._create_compact_button("Detect Text")
        self.detect_btn.clicked.connect(self.detect_text_clicked.emit)
        row1.addWidget(self.detect_btn)
        
        self.recognize_btn = self._create_compact_button("Recognize")
        self.recognize_btn.clicked.connect(self.recognize_text_clicked.emit)
        row1.addWidget(self.recognize_btn)
        
        # Translate button removed - use main Start Translation button instead
        
        workflow_layout.addLayout(row1)
        
        # Row 2
        row2 = QHBoxLayout()
        row2.setSpacing(2)
        
        self.segment_btn = self._create_compact_button("Segment")
        self.segment_btn.clicked.connect(self.segment_text_clicked.emit)
        row2.addWidget(self.segment_btn)
        
        self.clean_btn = self._create_compact_button("Clean")
        self.clean_btn.clicked.connect(self.clean_image_clicked.emit)
        row2.addWidget(self.clean_btn)
        
        self.render_btn = self._create_compact_button("Render")
        self.render_btn.clicked.connect(self.render_clicked.emit)
        row2.addWidget(self.render_btn)
        
        workflow_layout.addLayout(row2)
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
                padding: 5px;
                font-size: 12pt;
                min-width: 30px;
                min-height: 30px;
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
        self.viewer.set_tool(tool)
        
        # Update button states
        self.hand_tool_btn.setChecked(tool == 'pan')
        self.box_draw_btn.setChecked(tool == 'box_draw')
        self.brush_btn.setChecked(tool == 'brush')
        self.eraser_btn.setChecked(tool == 'eraser')
    
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
        """Clear brush strokes"""
        self.viewer.brush_strokes.clear()
    
    def load_image(self, image_path: str):
        """Load an image into the preview (async)"""
        # Start loading (happens in background)
        self.viewer.load_image(image_path)
        self.current_image_path = image_path
    
    @Slot(str)
    def _on_image_loading_started(self, image_path: str):
        """Handle image loading started"""
        # Show loading status immediately
        self.file_label.setText(f"⏳ Loading {os.path.basename(image_path)}...")
        self.file_label.setStyleSheet("color: #5a9fd4; font-size: 8pt;")  # Blue while loading
        
        # Clear previous image data
        self.viewer.clear_rectangles()
        self._clear_strokes()
        self._update_box_count()
    
    def _on_image_loaded_success(self):
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
