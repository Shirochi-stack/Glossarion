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
                               QLabel, QSlider, QFrame, QPushButton, QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsPathItem,
                               QSizePolicy, QListWidget, QListWidgetItem, QTabWidget)
from PySide6.QtCore import Qt, Signal, QRectF, QPointF, QSize, QTimer, Slot
from PySide6.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QBrush, QCursor, QIcon, QPainterPath
import numpy as np


class MoveableRectItem(QGraphicsRectItem):
    """Simple moveable and resizable rectangle for text box selection"""
    
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
        self.shape_type = 'rect'
        
        # Resize functionality
        self._resize_mode = None  # None, 'corner', 'edge'
        self._resize_start_pos = None
        self._resize_start_rect = None
        self._resize_handle_size = 8
        self._is_resizing = False
        
        # Movement/resize tracking
        self._initial_pos = None
        self._initial_rect = None
        
    def _get_resize_mode(self, pos):
        """Determine resize mode based on position relative to rectangle"""
        rect = self.rect()
        handle_size = self._resize_handle_size
        
        # Always use pos as-is since it's already in item coordinates
        local_pos = pos
        
        # Check corners first (priority over edges)
        if (abs(local_pos.x() - rect.left()) <= handle_size and 
            abs(local_pos.y() - rect.top()) <= handle_size):
            return 'top_left'
        elif (abs(local_pos.x() - rect.right()) <= handle_size and 
              abs(local_pos.y() - rect.top()) <= handle_size):
            return 'top_right'
        elif (abs(local_pos.x() - rect.left()) <= handle_size and 
              abs(local_pos.y() - rect.bottom()) <= handle_size):
            return 'bottom_left'
        elif (abs(local_pos.x() - rect.right()) <= handle_size and 
              abs(local_pos.y() - rect.bottom()) <= handle_size):
            return 'bottom_right'
        
        # Check edges
        elif abs(local_pos.y() - rect.top()) <= handle_size and rect.left() <= local_pos.x() <= rect.right():
            return 'top'
        elif abs(local_pos.y() - rect.bottom()) <= handle_size and rect.left() <= local_pos.x() <= rect.right():
            return 'bottom'
        elif abs(local_pos.x() - rect.left()) <= handle_size and rect.top() <= local_pos.y() <= rect.bottom():
            return 'left'
        elif abs(local_pos.x() - rect.right()) <= handle_size and rect.top() <= local_pos.y() <= rect.bottom():
            return 'right'
        
        return None
    
    def _get_cursor_for_resize_mode(self, mode):
        """Get appropriate cursor for resize mode"""
        cursor_map = {
            'top_left': Qt.CursorShape.SizeFDiagCursor,
            'top_right': Qt.CursorShape.SizeBDiagCursor,
            'bottom_left': Qt.CursorShape.SizeBDiagCursor,
            'bottom_right': Qt.CursorShape.SizeFDiagCursor,
            'top': Qt.CursorShape.SizeVerCursor,
            'bottom': Qt.CursorShape.SizeVerCursor,
            'left': Qt.CursorShape.SizeHorCursor,
            'right': Qt.CursorShape.SizeHorCursor,
        }
        return cursor_map.get(mode, Qt.CursorShape.PointingHandCursor)
    
    def hoverMoveEvent(self, event):
        """Update cursor based on position for resize handles"""
        if not self._is_resizing:
            resize_mode = self._get_resize_mode(event.pos())
            if resize_mode:
                self.setCursor(QCursor(self._get_cursor_for_resize_mode(resize_mode)))
            else:
                self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        super().hoverMoveEvent(event)
    
    def itemChange(self, change, value):
        """Handle item changes - ensure hover works after position changes"""
        result = super().itemChange(change, value)
        
        # When position changes, ensure we can still detect resize after the move
        from PySide6.QtWidgets import QGraphicsItem
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            # Reset cursor to allow proper hover detection
            if not self._is_resizing:
                self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        
        return result
    
    def hoverEnterEvent(self, event):
        self.hoverMoveEvent(event)  # Use the same logic
        super().hoverEnterEvent(event)
        
    def hoverLeaveEvent(self, event):
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        super().hoverLeaveEvent(event)
        
    def mousePressEvent(self, event):
        """Handle mouse press for resize or move operations"""
        try:
            if event.button() == Qt.MouseButton.LeftButton:
                # Store initial state for change detection
                self._initial_pos = self.pos()
                self._initial_rect = self.rect()
                
                # Check for resize mode BEFORE clearing selection
                resize_mode = self._get_resize_mode(event.pos())
                
                # Clear other selections and select this item
                sc = self.scene()
                if sc is not None:
                    sc.clearSelection()
                self.setSelected(True)
                
                if resize_mode:
                    self._resize_mode = resize_mode
                    self._resize_start_pos = event.pos()
                    self._resize_start_rect = self.rect()
                    self._is_resizing = True
                    # Disable item movement during resize
                    self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsMovable, False)
                    # Don't call super() for resize - handle it ourselves
                    event.accept()
                    return
                else:
                    self._resize_mode = None
                    self._is_resizing = False
                    # Disable Qt's automatic movement and handle it manually
                    self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsMovable, False)
                    # Set up manual movement tracking
                    self._manual_move_start = event.scenePos()
                    self._manual_move_initial_pos = self.pos()
        except Exception:
            pass
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for resize operations"""
        if self._is_resizing and self._resize_mode:
            try:
                current_pos = event.pos()
                start_pos = self._resize_start_pos
                start_rect = self._resize_start_rect
                
                dx = current_pos.x() - start_pos.x()
                dy = current_pos.y() - start_pos.y()
                
                new_rect = QRectF(start_rect)
                
                # Apply resize based on mode
                if 'left' in self._resize_mode:
                    new_rect.setLeft(start_rect.left() + dx)
                if 'right' in self._resize_mode:
                    new_rect.setRight(start_rect.right() + dx)
                if 'top' in self._resize_mode:
                    new_rect.setTop(start_rect.top() + dy)
                if 'bottom' in self._resize_mode:
                    new_rect.setBottom(start_rect.bottom() + dy)
                
                # Ensure minimum size
                min_size = 10
                if new_rect.width() < min_size:
                    if 'left' in self._resize_mode:
                        new_rect.setLeft(new_rect.right() - min_size)
                    else:
                        new_rect.setRight(new_rect.left() + min_size)
                if new_rect.height() < min_size:
                    if 'top' in self._resize_mode:
                        new_rect.setTop(new_rect.bottom() - min_size)
                    else:
                        new_rect.setBottom(new_rect.top() + min_size)
                
                self.setRect(new_rect)
                event.accept()
                return
            except Exception as e:
                print(f"Error during resize: {e}")
        elif not self._is_resizing:
            # Handle manual movement to avoid Qt's movement system interfering
            if hasattr(self, '_manual_move_start'):
                current_pos = event.scenePos()
                start_pos = self._manual_move_start
                delta = current_pos - start_pos
                new_pos = self._manual_move_initial_pos + delta
                self.setPos(new_pos)
                event.accept()
                return
        
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        was_resizing = self._is_resizing
        was_actually_moved_or_resized = False
        
        # Check if we actually moved or resized (not just clicked)
        if (hasattr(self, '_initial_pos') and self._initial_pos is not None and 
            hasattr(self, '_initial_rect') and self._initial_rect is not None):
            current_pos = self.pos()
            current_rect = self.rect()
            # Consider it moved/resized if position changed by more than 1 pixel or rect changed
            pos_changed = (abs(current_pos.x() - self._initial_pos.x()) > 1 or 
                          abs(current_pos.y() - self._initial_pos.y()) > 1)
            rect_changed = (abs(current_rect.x() - self._initial_rect.x()) > 1 or
                           abs(current_rect.y() - self._initial_rect.y()) > 1 or
                           abs(current_rect.width() - self._initial_rect.width()) > 1 or
                           abs(current_rect.height() - self._initial_rect.height()) > 1)
            was_actually_moved_or_resized = pos_changed or rect_changed or was_resizing
        
        try:
            if self._is_resizing:
                self._is_resizing = False
                self._resize_mode = None
                # Re-enable movement
                self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsMovable, True)
            
            super().mouseReleaseEvent(event)
        finally:
            try:
                # Only notify viewer if we actually moved or resized (not just clicked)
                if was_actually_moved_or_resized and hasattr(self, '_viewer') and self._viewer:
                    r = self.sceneBoundingRect()
                    # Emit moved signal with SCENE coordinates (for both move and resize)
                    self._viewer.rectangle_moved.emit(r)
            except Exception:
                pass
            
            # Clean up tracking variables
            self._initial_pos = None
            self._initial_rect = None
            
            # Clean up manual movement tracking
            if hasattr(self, '_manual_move_start'):
                delattr(self, '_manual_move_start')
            if hasattr(self, '_manual_move_initial_pos'):
                delattr(self, '_manual_move_initial_pos')
            
            # Re-enable hover events and reset cursor after move/resize
            self.setAcceptHoverEvents(True)
            if not self._is_resizing:
                self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

    def paint(self, painter, option, widget=None):
        """Custom paint to avoid Qt's default dashed selection rectangle; draw only our pen/brush."""
        try:
            painter.setPen(self.pen())
            painter.setBrush(self.brush())
            painter.drawRect(self.rect())
        except Exception:
            # Fallback to default painter if something goes wrong
            super().paint(painter, option, widget)


class MoveableEllipseItem(QGraphicsEllipseItem):
    """Moveable and resizable ellipse for circular text region selection"""
    
    def __init__(self, rect=None, parent=None, pen=None, brush=None):
        super().__init__(rect, parent)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        
        # Disable caching to prevent double-draw artifacts
        try:
            self.setCacheMode(QGraphicsEllipseItem.CacheMode.NoCache)
        except Exception:
            pass
        
        # Colors
        if brush is not None:
            self.setBrush(brush)
        else:
            self.setBrush(QBrush(QColor(255, 192, 203, 125)))
        if pen is not None:
            self.setPen(pen)
        else:
            self.setPen(QPen(QColor(255, 0, 0), 2))
        
        self.setZValue(1)
        self.selected = False
        self.shape_type = 'ellipse'
        
        # Resize functionality
        self._resize_mode = None  # None, 'corner', 'edge'
        self._resize_start_pos = None
        self._resize_start_rect = None
        self._resize_handle_size = 8
        self._is_resizing = False
        
        # Movement/resize tracking
        self._initial_pos = None
        self._initial_rect = None
        
    def _get_resize_mode(self, pos):
        """Determine resize mode based on position relative to ellipse bounding rectangle"""
        rect = self.rect()
        handle_size = self._resize_handle_size
        
        # Always use pos as-is since it's already in item coordinates
        local_pos = pos
        
        # Check corners of bounding rectangle first (priority over edges)
        if (abs(local_pos.x() - rect.left()) <= handle_size and 
            abs(local_pos.y() - rect.top()) <= handle_size):
            return 'top_left'
        elif (abs(local_pos.x() - rect.right()) <= handle_size and 
              abs(local_pos.y() - rect.top()) <= handle_size):
            return 'top_right'
        elif (abs(local_pos.x() - rect.left()) <= handle_size and 
              abs(local_pos.y() - rect.bottom()) <= handle_size):
            return 'bottom_left'
        elif (abs(local_pos.x() - rect.right()) <= handle_size and 
              abs(local_pos.y() - rect.bottom()) <= handle_size):
            return 'bottom_right'
        
        # Check edges
        elif abs(local_pos.y() - rect.top()) <= handle_size and rect.left() <= local_pos.x() <= rect.right():
            return 'top'
        elif abs(local_pos.y() - rect.bottom()) <= handle_size and rect.left() <= local_pos.x() <= rect.right():
            return 'bottom'
        elif abs(local_pos.x() - rect.left()) <= handle_size and rect.top() <= local_pos.y() <= rect.bottom():
            return 'left'
        elif abs(local_pos.x() - rect.right()) <= handle_size and rect.top() <= local_pos.y() <= rect.bottom():
            return 'right'
        
        return None
    
    def _get_cursor_for_resize_mode(self, mode):
        """Get appropriate cursor for resize mode"""
        cursor_map = {
            'top_left': Qt.CursorShape.SizeFDiagCursor,
            'top_right': Qt.CursorShape.SizeBDiagCursor,
            'bottom_left': Qt.CursorShape.SizeBDiagCursor,
            'bottom_right': Qt.CursorShape.SizeFDiagCursor,
            'top': Qt.CursorShape.SizeVerCursor,
            'bottom': Qt.CursorShape.SizeVerCursor,
            'left': Qt.CursorShape.SizeHorCursor,
            'right': Qt.CursorShape.SizeHorCursor,
        }
        return cursor_map.get(mode, Qt.CursorShape.PointingHandCursor)
    
    def hoverMoveEvent(self, event):
        """Update cursor based on position for resize handles"""
        if not self._is_resizing:
            resize_mode = self._get_resize_mode(event.pos())
            if resize_mode:
                self.setCursor(QCursor(self._get_cursor_for_resize_mode(resize_mode)))
            else:
                self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        super().hoverMoveEvent(event)
    
    def itemChange(self, change, value):
        """Handle item changes - ensure hover works after position changes"""
        result = super().itemChange(change, value)
        
        # When position changes, ensure we can still detect resize after the move
        from PySide6.QtWidgets import QGraphicsItem
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            # Reset cursor to allow proper hover detection
            if not self._is_resizing:
                self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        
        return result
    
    def hoverEnterEvent(self, event):
        self.hoverMoveEvent(event)  # Use the same logic
        super().hoverEnterEvent(event)
        
    def hoverLeaveEvent(self, event):
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        super().hoverLeaveEvent(event)
        
    def mousePressEvent(self, event):
        """Handle mouse press for resize or move operations"""
        try:
            if event.button() == Qt.MouseButton.LeftButton:
                # Store initial state for change detection
                self._initial_pos = self.pos()
                self._initial_rect = self.rect()
                
                # Check for resize mode BEFORE clearing selection
                resize_mode = self._get_resize_mode(event.pos())
                
                # Clear other selections and select this item
                sc = self.scene()
                if sc is not None:
                    sc.clearSelection()
                self.setSelected(True)
                
                if resize_mode:
                    self._resize_mode = resize_mode
                    self._resize_start_pos = event.pos()
                    self._resize_start_rect = self.rect()
                    self._is_resizing = True
                    # Disable item movement during resize
                    self.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemIsMovable, False)
                    # Don't call super() for resize - handle it ourselves
                    event.accept()
                    return
                else:
                    self._resize_mode = None
                    self._is_resizing = False
                    # Disable Qt's automatic movement and handle it manually
                    self.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemIsMovable, False)
                    # Set up manual movement tracking
                    self._manual_move_start = event.scenePos()
                    self._manual_move_initial_pos = self.pos()
        except Exception:
            pass
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for resize operations"""
        if self._is_resizing and self._resize_mode:
            try:
                current_pos = event.pos()
                start_pos = self._resize_start_pos
                start_rect = self._resize_start_rect
                
                dx = current_pos.x() - start_pos.x()
                dy = current_pos.y() - start_pos.y()
                
                new_rect = QRectF(start_rect)
                
                # Apply resize based on mode
                if 'left' in self._resize_mode:
                    new_rect.setLeft(start_rect.left() + dx)
                if 'right' in self._resize_mode:
                    new_rect.setRight(start_rect.right() + dx)
                if 'top' in self._resize_mode:
                    new_rect.setTop(start_rect.top() + dy)
                if 'bottom' in self._resize_mode:
                    new_rect.setBottom(start_rect.bottom() + dy)
                
                # Ensure minimum size
                min_size = 10
                if new_rect.width() < min_size:
                    if 'left' in self._resize_mode:
                        new_rect.setLeft(new_rect.right() - min_size)
                    else:
                        new_rect.setRight(new_rect.left() + min_size)
                if new_rect.height() < min_size:
                    if 'top' in self._resize_mode:
                        new_rect.setTop(new_rect.bottom() - min_size)
                    else:
                        new_rect.setBottom(new_rect.top() + min_size)
                
                self.setRect(new_rect)
                event.accept()
                return
            except Exception as e:
                print(f"Error during ellipse resize: {e}")
        elif not self._is_resizing:
            # Handle manual movement to avoid Qt's movement system interfering
            if hasattr(self, '_manual_move_start'):
                current_pos = event.scenePos()
                start_pos = self._manual_move_start
                delta = current_pos - start_pos
                new_pos = self._manual_move_initial_pos + delta
                self.setPos(new_pos)
                event.accept()
                return
        
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        was_resizing = self._is_resizing
        was_actually_moved_or_resized = False
        
        # Check if we actually moved or resized (not just clicked)
        if (hasattr(self, '_initial_pos') and self._initial_pos is not None and 
            hasattr(self, '_initial_rect') and self._initial_rect is not None):
            current_pos = self.pos()
            current_rect = self.rect()
            # Consider it moved/resized if position changed by more than 1 pixel or rect changed
            pos_changed = (abs(current_pos.x() - self._initial_pos.x()) > 1 or 
                          abs(current_pos.y() - self._initial_pos.y()) > 1)
            rect_changed = (abs(current_rect.x() - self._initial_rect.x()) > 1 or
                           abs(current_rect.y() - self._initial_rect.y()) > 1 or
                           abs(current_rect.width() - self._initial_rect.width()) > 1 or
                           abs(current_rect.height() - self._initial_rect.height()) > 1)
            was_actually_moved_or_resized = pos_changed or rect_changed or was_resizing
        
        try:
            if self._is_resizing:
                self._is_resizing = False
                self._resize_mode = None
                # Re-enable movement
                self.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemIsMovable, True)
            
            super().mouseReleaseEvent(event)
        finally:
            try:
                # Only notify viewer if we actually moved or resized (not just clicked)
                if was_actually_moved_or_resized and hasattr(self, '_viewer') and self._viewer:
                    r = self.sceneBoundingRect()
                    # Emit moved signal with SCENE coordinates (for both move and resize)
                    self._viewer.rectangle_moved.emit(r)
            except Exception:
                pass
            
            # Clean up tracking variables
            self._initial_pos = None
            self._initial_rect = None
            
            # Clean up manual movement tracking
            if hasattr(self, '_manual_move_start'):
                delattr(self, '_manual_move_start')
            if hasattr(self, '_manual_move_initial_pos'):
                delattr(self, '_manual_move_initial_pos')
            
            # Re-enable hover events and reset cursor after move/resize
            self.setAcceptHoverEvents(True)
            if not self._is_resizing:
                self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

    def paint(self, painter, option, widget=None):
        """Custom paint to avoid Qt's default dashed selection rectangle; draw only our pen/brush."""
        try:
            painter.setPen(self.pen())
            painter.setBrush(self.brush())
            painter.drawEllipse(self.rect())
        except Exception:
            super().paint(painter, option, widget)


class MoveablePathItem(QGraphicsPathItem):
    """Moveable and resizable freehand path (lasso) for arbitrary-shaped regions"""
    def __init__(self, path=None, parent=None, pen=None, brush=None):
        super().__init__(parent)
        if path is not None:
            self.setPath(path)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setFlag(QGraphicsPathItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsPathItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        try:
            self.setCacheMode(QGraphicsPathItem.CacheMode.NoCache)
        except Exception:
            pass
        if brush is not None:
            self.setBrush(brush)
        else:
            self.setBrush(QBrush(QColor(255, 192, 203, 125)))
        if pen is not None:
            self.setPen(pen)
        else:
            self.setPen(QPen(QColor(255, 0, 0), 2))
        self.setZValue(1)
        self.selected = False
        self.shape_type = 'polygon'
        
        # Resize functionality
        self._resize_mode = None
        self._resize_start_pos = None
        self._resize_start_rect = None
        self._resize_handle_size = 8
        self._is_resizing = False
        self._original_path = None
        
        # Movement/resize tracking
        self._initial_pos = None
        self._initial_rect = None
    def _get_resize_mode(self, pos):
        """Determine resize mode based on position relative to path bounding rectangle"""
        rect = self.boundingRect()
        handle_size = self._resize_handle_size
        
        # Always use pos as-is since it's already in item coordinates
        local_pos = pos
        
        # Check corners of bounding rectangle first (priority over edges)
        if (abs(local_pos.x() - rect.left()) <= handle_size and 
            abs(local_pos.y() - rect.top()) <= handle_size):
            return 'top_left'
        elif (abs(local_pos.x() - rect.right()) <= handle_size and 
              abs(local_pos.y() - rect.top()) <= handle_size):
            return 'top_right'
        elif (abs(local_pos.x() - rect.left()) <= handle_size and 
              abs(local_pos.y() - rect.bottom()) <= handle_size):
            return 'bottom_left'
        elif (abs(local_pos.x() - rect.right()) <= handle_size and 
              abs(local_pos.y() - rect.bottom()) <= handle_size):
            return 'bottom_right'
        
        # Check edges
        elif abs(local_pos.y() - rect.top()) <= handle_size and rect.left() <= local_pos.x() <= rect.right():
            return 'top'
        elif abs(local_pos.y() - rect.bottom()) <= handle_size and rect.left() <= local_pos.x() <= rect.right():
            return 'bottom'
        elif abs(local_pos.x() - rect.left()) <= handle_size and rect.top() <= local_pos.y() <= rect.bottom():
            return 'left'
        elif abs(local_pos.x() - rect.right()) <= handle_size and rect.top() <= local_pos.y() <= rect.bottom():
            return 'right'
        
        return None
    
    def _get_cursor_for_resize_mode(self, mode):
        """Get appropriate cursor for resize mode"""
        cursor_map = {
            'top_left': Qt.CursorShape.SizeFDiagCursor,
            'top_right': Qt.CursorShape.SizeBDiagCursor,
            'bottom_left': Qt.CursorShape.SizeBDiagCursor,
            'bottom_right': Qt.CursorShape.SizeFDiagCursor,
            'top': Qt.CursorShape.SizeVerCursor,
            'bottom': Qt.CursorShape.SizeVerCursor,
            'left': Qt.CursorShape.SizeHorCursor,
            'right': Qt.CursorShape.SizeHorCursor,
        }
        return cursor_map.get(mode, Qt.CursorShape.PointingHandCursor)
    
    def hoverMoveEvent(self, event):
        """Update cursor based on position for resize handles"""
        if not self._is_resizing:
            resize_mode = self._get_resize_mode(event.pos())
            if resize_mode:
                self.setCursor(QCursor(self._get_cursor_for_resize_mode(resize_mode)))
            else:
                self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        super().hoverMoveEvent(event)
    
    def itemChange(self, change, value):
        """Handle item changes - ensure hover works after position changes"""
        result = super().itemChange(change, value)
        
        # When position changes, ensure we can still detect resize after the move
        from PySide6.QtWidgets import QGraphicsItem
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            # Reset cursor to allow proper hover detection
            if not self._is_resizing:
                self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        
        return result
    
    def hoverEnterEvent(self, event):
        self.hoverMoveEvent(event)  # Use the same logic
        super().hoverEnterEvent(event)
        
    def hoverLeaveEvent(self, event):
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        super().hoverLeaveEvent(event)
    def _scale_path(self, path, old_rect, new_rect):
        """Scale a path from old_rect to new_rect"""
        from PySide6.QtGui import QTransform
        
        if old_rect.width() == 0 or old_rect.height() == 0:
            return path
        
        # Calculate scale factors
        sx = new_rect.width() / old_rect.width()
        sy = new_rect.height() / old_rect.height()
        
        # Create transform to scale from old rect to new rect
        transform = QTransform()
        transform.translate(new_rect.left(), new_rect.top())
        transform.scale(sx, sy)
        transform.translate(-old_rect.left(), -old_rect.top())
        
        return transform.map(path)
    
    def mousePressEvent(self, event):
        """Handle mouse press for resize or move operations"""
        try:
            if event.button() == Qt.MouseButton.LeftButton:
                # Store initial state for change detection
                self._initial_pos = self.pos()
                self._initial_rect = self.boundingRect()
                
                # Check for resize mode BEFORE clearing selection
                resize_mode = self._get_resize_mode(event.pos())
                
                # Clear other selections and select this item
                sc = self.scene()
                if sc is not None:
                    sc.clearSelection()
                self.setSelected(True)
                
                if resize_mode:
                    self._resize_mode = resize_mode
                    self._resize_start_pos = event.pos()
                    self._resize_start_rect = self.boundingRect()
                    self._original_path = QPainterPath(self.path())  # Store original path
                    self._is_resizing = True
                    # Disable item movement during resize
                    self.setFlag(QGraphicsPathItem.GraphicsItemFlag.ItemIsMovable, False)
                    # Don't call super() for resize - handle it ourselves
                    event.accept()
                    return
                else:
                    self._resize_mode = None
                    self._is_resizing = False
                    # Disable Qt's automatic movement and handle it manually
                    self.setFlag(QGraphicsPathItem.GraphicsItemFlag.ItemIsMovable, False)
                    # Set up manual movement tracking
                    self._manual_move_start = event.scenePos()
                    self._manual_move_initial_pos = self.pos()
        except Exception:
            pass
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for resize operations"""
        if self._is_resizing and self._resize_mode and self._original_path:
            try:
                current_pos = event.pos()
                start_pos = self._resize_start_pos
                start_rect = self._resize_start_rect
                
                dx = current_pos.x() - start_pos.x()
                dy = current_pos.y() - start_pos.y()
                
                new_rect = QRectF(start_rect)
                
                # Apply resize based on mode
                if 'left' in self._resize_mode:
                    new_rect.setLeft(start_rect.left() + dx)
                if 'right' in self._resize_mode:
                    new_rect.setRight(start_rect.right() + dx)
                if 'top' in self._resize_mode:
                    new_rect.setTop(start_rect.top() + dy)
                if 'bottom' in self._resize_mode:
                    new_rect.setBottom(start_rect.bottom() + dy)
                
                # Ensure minimum size
                min_size = 10
                if new_rect.width() < min_size:
                    if 'left' in self._resize_mode:
                        new_rect.setLeft(new_rect.right() - min_size)
                    else:
                        new_rect.setRight(new_rect.left() + min_size)
                if new_rect.height() < min_size:
                    if 'top' in self._resize_mode:
                        new_rect.setTop(new_rect.bottom() - min_size)
                    else:
                        new_rect.setBottom(new_rect.top() + min_size)
                
                # Scale the path to fit the new rect
                scaled_path = self._scale_path(self._original_path, start_rect, new_rect)
                self.setPath(scaled_path)
                event.accept()
                return
            except Exception as e:
                print(f"Error during path resize: {e}")
        elif not self._is_resizing:
            # Handle manual movement to avoid Qt's movement system interfering
            if hasattr(self, '_manual_move_start'):
                current_pos = event.scenePos()
                start_pos = self._manual_move_start
                delta = current_pos - start_pos
                new_pos = self._manual_move_initial_pos + delta
                self.setPos(new_pos)
                event.accept()
                return
        
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        was_resizing = self._is_resizing
        was_actually_moved_or_resized = False
        
        # Check if we actually moved or resized (not just clicked)
        if (hasattr(self, '_initial_pos') and self._initial_pos is not None and 
            hasattr(self, '_initial_rect') and self._initial_rect is not None):
            current_pos = self.pos()
            current_rect = self.boundingRect()
            # Consider it moved/resized if position changed by more than 1 pixel or rect changed
            pos_changed = (abs(current_pos.x() - self._initial_pos.x()) > 1 or 
                          abs(current_pos.y() - self._initial_pos.y()) > 1)
            rect_changed = (abs(current_rect.x() - self._initial_rect.x()) > 1 or
                           abs(current_rect.y() - self._initial_rect.y()) > 1 or
                           abs(current_rect.width() - self._initial_rect.width()) > 1 or
                           abs(current_rect.height() - self._initial_rect.height()) > 1)
            was_actually_moved_or_resized = pos_changed or rect_changed or was_resizing
        
        try:
            if self._is_resizing:
                self._is_resizing = False
                self._resize_mode = None
                self._original_path = None
                # Re-enable movement
                self.setFlag(QGraphicsPathItem.GraphicsItemFlag.ItemIsMovable, True)
            
            super().mouseReleaseEvent(event)
        finally:
            try:
                # Only notify viewer if we actually moved or resized (not just clicked)
                if was_actually_moved_or_resized and hasattr(self, '_viewer') and self._viewer:
                    r = self.sceneBoundingRect()
                    # Emit moved signal with SCENE coordinates (for both move and resize)
                    self._viewer.rectangle_moved.emit(r)
            except Exception:
                pass
            
            # Clean up tracking variables
            self._initial_pos = None
            self._initial_rect = None
            
            # Clean up manual movement tracking
            if hasattr(self, '_manual_move_start'):
                delattr(self, '_manual_move_start')
            if hasattr(self, '_manual_move_initial_pos'):
                delattr(self, '_manual_move_initial_pos')
            
            # Re-enable hover events and reset cursor after move/resize
            self.setAcceptHoverEvents(True)
            if not self._is_resizing:
                self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
    def paint(self, painter, option, widget=None):
        try:
            painter.setPen(self.pen())
            painter.setBrush(self.brush())
            painter.drawPath(self.path())
        except Exception:
            super().paint(painter, option, widget)



class CompactImageViewer(QGraphicsView):
    """Compact image viewer with basic editing capabilities"""
    
    rectangle_created = Signal(QRectF)
    rectangle_selected = Signal(QRectF)
    rectangle_deleted = Signal(QRectF)
    rectangle_moved = Signal(QRectF)
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
        
        # Lasso state
        self.lasso_path: Optional[QPainterPath] = None
        self.lasso_item: Optional[MoveablePathItem] = None
        self.lasso_active = False
        
        # Image loading state
        self._loading = False
    
    def resizeEvent(self, event):
        """Refit only if user hasn't manually zoomed."""
        super().resizeEvent(event)
        if self.hasPhoto() and not self.user_zoomed:
            self.fitInView()
    
    def hasPhoto(self) -> bool:
        return not self.empty
    
    def load_image(self, image_path: str) -> bool:
        """Load an image from file path (synchronous on main thread)"""
        if not os.path.exists(image_path):
            return False
        
        # Emit loading signal
        self._loading = True
        self.image_loading.emit(image_path)
        
        try:
            # Load directly with QPixmap on main thread
            pixmap = QPixmap(image_path)
            
            if pixmap.isNull():
                print(f"Failed to load: {image_path}")
                self._loading = False
                return False
            
            self._loading = False
            self.setPhoto(pixmap)
            self.image_loaded.emit(image_path)
            return True
            
        except Exception as e:
            print(f"Error loading image: {e}")
            self._loading = False
            return False
    
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
        """Set current tool (box_draw, circle_draw, lasso_draw, pan, brush, eraser)"""
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
            # box_draw, circle_draw, lasso_draw
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
        # Handle right-clicks on rectangles first to ensure proper context menu routing
        if event.button() == Qt.MouseButton.RightButton:
            item = self.itemAt(event.pos())
            print(f"[CONTEXT_MENU] Right-click - item at pos: {type(item).__name__ if item else 'None'}")
            
            # If right-clicking on a rectangle, route the event to that rectangle
            if isinstance(item, (MoveableRectItem, MoveableEllipseItem, MoveablePathItem)):
                print(f"[CONTEXT_MENU] Routing right-click to rectangle with region_index: {getattr(item, 'region_index', 'None')}")
                # Let the item handle the right-click event
                scene_pos = self.mapToScene(event.pos())
                # Convert to item coordinates and deliver the event
                item_pos = item.mapFromScene(scene_pos)
                from PySide6.QtGui import QMouseEvent
                from PySide6.QtCore import QPointF
                # Create a new mouse event in item coordinates
                item_event = QMouseEvent(
                    event.type(),
                    item_pos,
                    event.button(),
                    event.buttons(),
                    event.modifiers()
                )
                item.mousePressEvent(item_event)
                event.accept()
                return
            # If right-clicking on something else, let default handling occur
            super().mousePressEvent(event)
            return
        
        if self.current_tool in ['box_draw', 'circle_draw'] and event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            self.start_point = scene_pos
            rect = QRectF(scene_pos, scene_pos)
            if self.current_tool == 'circle_draw':
                self.current_rect = MoveableEllipseItem(rect)
            else:
                self.current_rect = MoveableRectItem(rect)
            # Attach viewer reference so item can emit moved signal
            try:
                self.current_rect._viewer = self
            except Exception:
                pass
            self._scene.addItem(self.current_rect)
        elif self.current_tool == 'lasso_draw' and event.button() == Qt.MouseButton.LeftButton:
            self.lasso_active = True
            scene_pos = self.mapToScene(event.pos())
            self.lasso_path = QPainterPath(scene_pos)
            self.lasso_item = MoveablePathItem(self.lasso_path)
            try:
                self.lasso_item._viewer = self
            except Exception:
                pass
            self._scene.addItem(self.lasso_item)
        elif self.current_tool in ['brush', 'eraser'] and event.button() == Qt.MouseButton.LeftButton:
            self.drawing = True
            self.last_pos = self.mapToScene(event.pos())
        elif self.current_tool == 'pan' and event.button() == Qt.MouseButton.LeftButton:
            # Check if we clicked on a rectangle to select it
            item = self.itemAt(event.pos())
            print(f"[SELECTION] Pan click - item at pos: {type(item).__name__ if item else 'None'}")
            
            if isinstance(item, (MoveableRectItem, MoveableEllipseItem, MoveablePathItem)):
                print(f"[SELECTION] Selecting rectangle directly")
                self._select_rectangle(item)
            elif hasattr(item, 'region_index'):
                # This is a text overlay - find the corresponding blue rectangle
                region_idx = item.region_index
                print(f"[SELECTION] Clicked text overlay for region {region_idx}")
                if 0 <= region_idx < len(self.rectangles):
                    target_rect = self.rectangles[region_idx]
                    print(f"[SELECTION] Selecting blue rectangle {region_idx}")
                    self._select_rectangle(target_rect)
                else:
                    print(f"[SELECTION] No rectangle found for region {region_idx}")
                    self._select_rectangle(None)
            else:
                print(f"[SELECTION] Deselecting all")
                self._select_rectangle(None)  # Deselect all
            super().mousePressEvent(event)
        else:
            super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for drawing boxes or inpainting"""
        if self.current_tool in ['box_draw', 'circle_draw'] and self.current_rect:
            scene_pos = self.mapToScene(event.pos())
            rect = QRectF(self.start_point, scene_pos).normalized()
            self.current_rect.setRect(rect)
        elif self.current_tool == 'lasso_draw' and self.lasso_active and self.lasso_item is not None and self.lasso_path is not None:
            scene_pos = self.mapToScene(event.pos())
            self.lasso_path.lineTo(scene_pos)
            self.lasso_item.setPath(self.lasso_path)
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
        if self.current_tool in ['box_draw', 'circle_draw'] and self.current_rect:
            if self.current_rect.rect().width() > 10 and self.current_rect.rect().height() > 10:
                self.rectangles.append(self.current_rect)
                self.rectangle_created.emit(self.current_rect.rect())
            else:
                self._scene.removeItem(self.current_rect)
            self.current_rect = None
            self.start_point = None
        elif self.current_tool == 'lasso_draw' and self.lasso_active and self.lasso_item is not None and self.lasso_path is not None:
            # Close and validate polygon
            self.lasso_path.closeSubpath()
            self.lasso_item.setPath(self.lasso_path)
            br = self.lasso_item.boundingRect()
            if br.width() > 10 and br.height() > 10:
                self.rectangles.append(self.lasso_item)
                # Emit bounding rect
                self.rectangle_created.emit(self.lasso_item.sceneBoundingRect())
            else:
                self._scene.removeItem(self.lasso_item)
            # Reset lasso state
            self.lasso_active = False
            self.lasso_item = None
            self.lasso_path = None
        elif self.current_tool in ['brush', 'eraser']:
            self.drawing = False
            self.last_pos = None
        else:
            super().mouseReleaseEvent(event)
    
    def _select_rectangle(self, rect_item: Optional[MoveableRectItem]):
        """Select a rectangle and update visual feedback"""
        print(f"[SELECTION] _select_rectangle called with: {type(rect_item).__name__ if rect_item else 'None'}")
        
        # Deselect previous selection
        if self.selected_rect:
            print(f"[SELECTION] Deselecting previous rectangle")
            # Determine base color from recognition flag, not by presence of region_index
            was_recognized = getattr(self.selected_rect, 'is_recognized', False)
            pen = self.selected_rect.pen()
            current_color = pen.color()
            
            # If it's currently yellow (selected), restore to proper base color
            if current_color == QColor(255, 255, 0):  # Currently yellow (selected)
                if was_recognized:
                    # Restore blue color for rectangles with recognized text
                    self.selected_rect.setPen(QPen(QColor(0, 150, 255), 2))  # Blue for recognized text
                    self.selected_rect.setBrush(QBrush(QColor(0, 150, 255, 50)))  # Semi-transparent blue fill
                else:
                    # Restore green for detection-only rectangles
                    self.selected_rect.setPen(QPen(QColor(0, 255, 0), 2))
                    self.selected_rect.setBrush(QBrush(QColor(0, 255, 0, 50)))
        
        # Select new rectangle
        self.selected_rect = rect_item
        if self.selected_rect:
            print(f"[SELECTION] Rectangle selected - turning yellow")
            self.selected_rect.setPen(QPen(QColor(255, 255, 0), 3))  # Yellow for selected, thicker
            try:
                self.rectangle_selected.emit(self.selected_rect.rect())
            except Exception:
                pass
        else:
            print(f"[SELECTION] No rectangle selected")
    
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
            # Attach viewer reference so item can emit moved signal
            try:
                rect_item._viewer = self
            except Exception:
                pass
            
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
        """Delete currently selected rectangle and associated text overlays"""
        print(f"[DELETE] delete_selected_rectangle called. selected_rect = {type(self.selected_rect).__name__ if self.selected_rect else 'None'}")
        if self.selected_rect:
            print(f"[DELETE] Attempting to delete selected rectangle")
            
            # Get region index for blue rectangles (ones with recognized text)
            region_index = None
            rect_to_delete = self.selected_rect
            
            # Check if this rectangle has a region_index (blue rectangle with recognized text)
            if hasattr(rect_to_delete, 'region_index'):
                region_index = rect_to_delete.region_index
                print(f"[DELETE] Found region_index attribute: {region_index}")
            else:
                # Try to find index by position in rectangles list
                try:
                    region_index = self.rectangles.index(rect_to_delete)
                    print(f"[DELETE] Using list position as region_index: {region_index}")
                except ValueError:
                    print(f"[DELETE] Rectangle not found in list, using None")
                    region_index = None
            
            # Check if this is a blue rectangle (has blue color)
            is_blue_rect = False
            try:
                pen_color = rect_to_delete.pen().color()
                if pen_color in [QColor(0, 150, 255), QColor(255, 255, 0)]:  # Blue or selected yellow
                    is_blue_rect = True
                    print(f"[DELETE] Detected blue rectangle (color: {pen_color.name()})")
            except Exception:
                pass
            
            print(f"[DELETE] Deleting rectangle - region_index: {region_index}, is_blue: {is_blue_rect}")
            
            # Remove from scene and list
            self._scene.removeItem(rect_to_delete)
            self.rectangles.remove(rect_to_delete)
            
            # Clear selection
            self.selected_rect = None
            
            # Notify parent widget about deletion
            self.rectangle_deleted.emit(rect_to_delete.rect())
            
            # If this is a blue rectangle, clean up overlays
            if (region_index is not None and is_blue_rect and 
                hasattr(self, 'manga_integration') and self.manga_integration):
                try:
                    print(f"[DELETE] Calling cleanup for blue rectangle region {region_index}")
                    import ImageRenderer
                    ImageRenderer._clean_up_deleted_rectangle_overlays(self.manga_integration, region_index)
                except Exception as e:
                    print(f"[DELETE] Cleanup failed: {e}")
            else:
                print(f"[DELETE] No cleanup needed - not a blue rectangle or no region_index")
    
    def clear_scene(self):
        """Clear entire scene"""
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
    
    # Internal signal for thumbnail loading (thread-safe)
    _thumbnail_loaded_signal = Signal(str, QImage)
    
    def __init__(self, parent=None, main_gui=None):
        super().__init__(parent)
        self.current_image_path = None
        self.current_translated_path = None  # Track translated output path
        self.main_gui = main_gui  # Store reference to main GUI for config persistence
        self.manual_editing_enabled = False  # Manual editing is disabled by default (preview mode)
        self.translated_folder_path = None  # Path to translated images folder
        self.source_display_mode = 'translated'  # Source tab display: 'translated', 'cleaned', or 'original'
        self.cleaned_images_enabled = True  # Deprecated: kept for compatibility, use source_display_mode instead
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
        
        # Create horizontal layout for viewer + thumbnails
        viewer_container = QWidget()
        viewer_layout = QHBoxLayout(viewer_container)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        viewer_layout.setSpacing(5)
        
        # Source image viewer
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
        viewer_layout.addWidget(self.thumbnail_list)
        
        # Thumbnail list starts hidden
        self.thumbnail_list.setVisible(False)
        
        # Store image paths for thumbnails
        self.image_paths = []
        
        layout.addWidget(viewer_container, stretch=1)
        
        # Connect loading signals to show status
        self.viewer.image_loading.connect(self._on_image_loading_started)
        self.viewer.image_loaded.connect(self._on_image_loaded_success)
        
        # Connect thumbnail loading signal
        self._thumbnail_loaded_signal.connect(self._on_thumbnail_loaded)
        
        # Display placeholder icon when no image is loaded
        self._show_placeholder_icon()
        
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
        
        # Source tab display mode toggle button (3 states: translated output, cleaned, original)
        self.source_display_mode = 'translated'  # Default to translated output
        self.cleaned_toggle_btn = self._create_tool_button("✒️", "Show translated output (click to cycle)")
        self.cleaned_toggle_btn.setCheckable(False)  # Not a checkbox, cycles through states
        self.cleaned_toggle_btn.clicked.connect(self._cycle_source_display_mode)
        # Make this button slightly larger for better emoji display
        self.cleaned_toggle_btn.setStyleSheet("""
            QToolButton {
                background-color: #2d2d2d;
                color: white;
                border: 1px solid #3a3a3a;
                border-radius: 3px;
                padding: 3px;
                font-size: 12pt;
                min-width: 32px;
                min-height: 32px;
                max-width: 36px;
                max-height: 36px;
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
        tools_layout.addWidget(self.cleaned_toggle_btn)
        
        # Add a stretch spacer after zoom and toggle controls
        tools_layout.addStretch()
        
        # MANUAL EDITING TOOLS (hidden when manual editing is off)
        self.box_draw_btn = self._create_tool_button("◭", "Draw Box")
        self.box_draw_btn.setCheckable(True)
        self.box_draw_btn.clicked.connect(lambda: self._set_tool('box_draw'))
        tools_layout.addWidget(self.box_draw_btn)
        
        # Circle draw / Circles mode button
        self.circle_draw_btn = self._create_tool_button("◯", "Draw Circle (also use circles in pipeline)")
        self.circle_draw_btn.setCheckable(True)
        def on_circle_clicked():
            self._set_tool('circle_draw')
            self._set_use_circle_shapes(True)
        self.circle_draw_btn.clicked.connect(on_circle_clicked)
        tools_layout.addWidget(self.circle_draw_btn)
        
        # Lasso draw button
        self.lasso_btn = self._create_tool_button("L", "Lasso (free shape)")
        self.lasso_btn.setCheckable(True)
        self.lasso_btn.clicked.connect(lambda: self._set_tool('lasso_draw'))
        tools_layout.addWidget(self.lasso_btn)
        
        self.save_overlay_btn = self._create_tool_button("💾", "Save & Update Overlay")
        self.save_overlay_btn.clicked.connect(self._on_save_overlay_clicked)
        tools_layout.addWidget(self.save_overlay_btn)
        
        # Conditionally show brush/eraser tools only when experimental flag is enabled
        enable_experimental_tools = False
        try:
            if self.main_gui and hasattr(self.main_gui, 'config') and isinstance(self.main_gui.config, dict):
                cfg = self.main_gui.config
                # Either flat key or nested under an experimental block
                enable_experimental_tools = bool(
                    cfg.get('experimental_translate_all', False) or
                    (isinstance(cfg.get('experimental'), dict) and cfg.get('experimental', {}).get('translate_all', False))
                )
            # Environment override
            if not enable_experimental_tools:
                enable_experimental_tools = (os.getenv('EXPERIMENTAL_TRANSLATE_ALL', '0') == '1')
        except Exception:
            enable_experimental_tools = False
        
        if enable_experimental_tools:
            self.brush_btn = self._create_tool_button("🖌", "Brush")
            self.brush_btn.setCheckable(True)
            self.brush_btn.clicked.connect(lambda: self._set_tool('brush'))
            tools_layout.addWidget(self.brush_btn)
            
            self.eraser_btn = self._create_tool_button("✏️", "Eraser")
            self.eraser_btn.setCheckable(True)
            self.eraser_btn.clicked.connect(lambda: self._set_tool('eraser'))
            tools_layout.addWidget(self.eraser_btn)
        else:
            # Do not create the buttons at all when experimental is disabled
            self.brush_btn = None
            self.eraser_btn = None
        
        self.delete_btn = self._create_tool_button("🗑", "Delete Selected")
        self.delete_btn.clicked.connect(self.viewer.delete_selected_rectangle)
        tools_layout.addWidget(self.delete_btn)
        
        self.clear_boxes_btn = self._create_tool_button("❌", "Clear Boxes")
        self.clear_boxes_btn.clicked.connect(self._on_clear_boxes_clicked)
        tools_layout.addWidget(self.clear_boxes_btn)
        
        # Only show clear strokes button when experimental tools are enabled
        if enable_experimental_tools:
            self.clear_strokes_btn = self._create_tool_button("🧹", "Clear Strokes")
            self.clear_strokes_btn.clicked.connect(self._clear_strokes)
            tools_layout.addWidget(self.clear_strokes_btn)
        else:
            # Do not create the button at all when experimental is disabled
            self.clear_strokes_btn = None
        
        # Brush size slider (stretches to fill space) - only show when experimental tools are enabled
        if enable_experimental_tools:
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
        else:
            # Do not create the slider/label at all when experimental is disabled
            self.size_slider = None
            self.size_label = None
        
        # Box count
        self.box_count_label = QLabel("0")
        self.box_count_label.setStyleSheet("color: white; font-weight: bold; min-width: 20px;")
        self.box_count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tools_layout.addWidget(self.box_count_label)
        self.viewer.rectangle_created.connect(self._update_box_count)
        self.viewer.rectangle_deleted.connect(self._update_box_count)
        # Persist rectangles when created/deleted/moved
        self.viewer.rectangle_created.connect(lambda _: self._persist_rectangles_state())
        self.viewer.rectangle_deleted.connect(lambda _: self._persist_rectangles_state())
        self.viewer.rectangle_moved.connect(lambda _: self._persist_rectangles_state())
        # Attach context menu to newly created rectangles (red or ellipse)
        self.viewer.rectangle_created.connect(self._on_rectangle_created)
        # Auto-apply save position on rectangle movement
        self.viewer.rectangle_moved.connect(self._on_rectangle_moved)
        
        layout.addWidget(self.tools_frame)
        
        # Translation workflow buttons (compact, single row)
        self.workflow_frame = self._create_tool_frame("Translation Workflow")
        workflow_layout = QHBoxLayout(self.workflow_frame)
        workflow_layout.setContentsMargins(3, 3, 3, 3)
        workflow_layout.setSpacing(2)
        
        self.detect_btn = self._create_compact_button("Detect Text")
        self.detect_btn.setToolTip("Detect text regions in the current image")
        self.detect_btn.clicked.connect(self.detect_text_clicked.emit)
        workflow_layout.addWidget(self.detect_btn, stretch=1)
        
        self.clean_btn = self._create_compact_button("Clean")
        self.clean_btn.setToolTip("Remove text from detected regions (inpainting)")
        self.clean_btn.clicked.connect(self.clean_image_clicked.emit)
        workflow_layout.addWidget(self.clean_btn, stretch=1)
        
        self.recognize_btn = self._create_compact_button("Recognize Text")
        self.recognize_btn.setToolTip("Perform OCR on detected text regions")
        self.recognize_btn.clicked.connect(lambda: self._emit_recognize_signal())
        workflow_layout.addWidget(self.recognize_btn, stretch=1)
        
        self.translate_btn = self._create_compact_button("Translate")
        self.translate_btn.setToolTip("Translate recognized text to target language")
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
        
        # Ensure tooltips always work by setting mouse tracking
        btn.setMouseTracking(True)
        
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
        
        # Ensure tooltips always work by setting mouse tracking
        btn.setMouseTracking(True)
        
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
        """Return the active viewer (always source viewer now)."""
        return self.viewer
    
    def _set_use_circle_shapes(self, enabled: bool):
        """Toggle using circle shapes across pipeline (detect/clean/recognize/translate)."""
        try:
            self.use_circle_shapes = bool(enabled)
            if hasattr(self, 'manga_integration') and self.manga_integration:
                if hasattr(self.manga_integration, 'set_use_circle_shapes'):
                    self.manga_integration.set_use_circle_shapes(bool(enabled))
        except Exception:
            pass
    
    def _set_tool(self, tool: str):
        """Set active tool and update button states"""
        self.viewer.set_tool(tool)
        
        # Update button states
        self.hand_tool_btn.setChecked(tool == 'pan')
        self.box_draw_btn.setChecked(tool == 'box_draw')
        if hasattr(self, 'circle_draw_btn') and self.circle_draw_btn is not None:
            self.circle_draw_btn.setChecked(tool == 'circle_draw')
            # Sync pipeline circle usage
            try:
                self._set_use_circle_shapes(tool == 'circle_draw')
            except Exception:
                pass
        if hasattr(self, 'lasso_btn') and self.lasso_btn is not None:
            self.lasso_btn.setChecked(tool == 'lasso_draw')
        # Only update experimental tool buttons if they exist
        if self.brush_btn is not None:
            self.brush_btn.setChecked(tool == 'brush')
        if self.eraser_btn is not None:
            self.eraser_btn.setChecked(tool == 'eraser')
    
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
            # Collect current viewer shapes for persistence (rect/ellipse/polygon)
            rect_data = []
            try:
                for rect_item in getattr(self.viewer, 'rectangles', []) or []:
                    r = rect_item.sceneBoundingRect()
                    entry = {
                        'x': r.x(),
                        'y': r.y(),
                        'width': r.width(),
                        'height': r.height(),
                        'shape': getattr(rect_item, 'shape_type', 'rect')
                    }
                    # Persist polygon points if lasso
                    try:
                        if entry['shape'] == 'polygon' and hasattr(rect_item, 'path'):
                            poly = rect_item.mapToScene(rect_item.path().toFillPolygon())
                            pts = []
                            for p in poly:
                                pts.append([float(p.x()), float(p.y())])
                            if len(pts) >= 3:
                                entry['polygon'] = pts
                    except Exception:
                        pass
                    rect_data.append(entry)
            except Exception:
                rect_data = []
            # Persist ONLY viewer_rectangles/shapes to avoid cascading changes to all overlays
            if hasattr(self.manga_integration, 'image_state_manager') and self.manga_integration.image_state_manager:
                # Merge into existing state without touching detection_regions
                prev = self.manga_integration.image_state_manager.get_state(image_path) or {}
                prev['viewer_rectangles'] = rect_data
                self.manga_integration.image_state_manager.set_state(image_path, prev, save=True)
        except Exception:
            pass
    
    def _update_brush_size(self, value: int):
        """Update brush/eraser size"""
        # Only update size label if it exists
        if self.size_label is not None:
            self.size_label.setText(str(value))
        self.viewer.brush_size = value
        self.viewer.eraser_size = value

    
    def _clear_strokes(self):
        """Clear brush strokes only"""
        self.viewer.clear_brush_strokes()
    
    def _on_clear_boxes_clicked(self):
        """Clear all rectangles and persist deletion to state. Also clear text overlays and their persisted offsets."""
        try:
            self.viewer.clear_rectangles()
            self._update_box_count()
            # Clear detection state aggressively (both memory and persisted)
            try:
                if hasattr(self, 'manga_integration') and self.manga_integration:
                    import ImageRenderer
                    ImageRenderer._clear_detection_state_for_image(self.manga_integration, self.current_image_path)
                    # Clear overlays from scene and persisted offsets/positions
                    try:
                        ImageRenderer.clear_text_overlays_for_image(self.manga_integration, self.current_image_path)
                    except Exception:
                        pass
                    # Clear in-memory translation and recognition data that causes "Edit Translation" tooltips
                    try:
                        if hasattr(self.manga_integration, '_translation_data'):
                            self.manga_integration._translation_data.clear()
                        if hasattr(self.manga_integration, '_recognition_data'):
                            self.manga_integration._recognition_data.clear()
                        print(f"[CLEAR] Cleared in-memory translation and recognition data")
                    except Exception:
                        pass
                    # Delete translated output image file
                    try:
                        if self.current_image_path:
                            source_dir = os.path.dirname(self.current_image_path)
                            source_filename = os.path.basename(self.current_image_path)
                            source_name_no_ext = os.path.splitext(source_filename)[0]
                            translated_folder = os.path.join(source_dir, f"{source_name_no_ext}_translated")
                            
                            # Delete translated output file (non-cleaned file) from isolated folder
                            if os.path.exists(translated_folder):
                                image_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif')
                                for filename in os.listdir(translated_folder):
                                    name_lower = filename.lower()
                                    # Find and delete files that match the source name but NOT cleaned files
                                    if (name_lower.startswith(source_name_no_ext.lower()) and 
                                        name_lower.endswith(image_extensions) and
                                        '_cleaned' not in name_lower):
                                        translated_path = os.path.join(translated_folder, filename)
                                        try:
                                            os.remove(translated_path)
                                            print(f"[CLEAR] Deleted translated output: {os.path.basename(translated_path)}")
                                        except Exception as e:
                                            print(f"[CLEAR] Failed to delete translated output: {e}")
                    except Exception as e:
                        print(f"[CLEAR] Error deleting translated output: {e}")
                    try:
                        if hasattr(self.manga_integration, 'image_state_manager') and self.manga_integration.image_state_manager and self.current_image_path:
                            st = self.manga_integration.image_state_manager.get_state(self.current_image_path) or {}
                            # Clear ALL saved state data - OCR, translations, overlays, etc.
                            st.pop('overlay_offsets', None)
                            st.pop('last_render_positions', None)
                            st.pop('translated_texts', None)
                            st.pop('recognized_texts', None)  # Clear OCR data
                            st.pop('detection_regions', None)  # Clear detection data
                            st.pop('viewer_rectangles', None)  # Clear rectangle data
                            self.manga_integration.image_state_manager.set_state(self.current_image_path, st, save=True)
                    except Exception:
                        pass
            except Exception:
                pass
            # Persist empty state (viewer_rectangles/detection_regions)
            self._persist_rectangles_state()
            # Reload the image to reflect the deletion (will fallback to cleaned or original)
            try:
                if self.current_image_path and os.path.exists(self.current_image_path):
                    self.load_image(self.current_image_path, preserve_rectangles=False, preserve_text_overlays=False)
            except Exception:
                pass
        except Exception:
            pass
    
    def _clear_all(self):
        """Clear all boxes and strokes"""
        self.viewer.clear_rectangles()
        self.viewer.clear_brush_strokes()
        self._update_box_count()
        # Also clear overlays and persisted overlay state
        try:
            if hasattr(self, 'manga_integration') and self.manga_integration and self.current_image_path:
                try:
                    import ImageRenderer
                    ImageRenderer.clear_text_overlays_for_image(self.manga_integration, self.current_image_path)
                except Exception:
                    pass
                # Clear in-memory translation and recognition data that causes "Edit Translation" tooltips
                try:
                    if hasattr(self.manga_integration, '_translation_data'):
                        self.manga_integration._translation_data.clear()
                    if hasattr(self.manga_integration, '_recognition_data'):
                        self.manga_integration._recognition_data.clear()
                    print(f"[CLEAR_ALL] Cleared in-memory translation and recognition data")
                except Exception:
                    pass
                try:
                    if hasattr(self.manga_integration, 'image_state_manager') and self.manga_integration.image_state_manager:
                        st = self.manga_integration.image_state_manager.get_state(self.current_image_path) or {}
                        # Clear ALL saved state data - OCR, translations, overlays, etc.
                        st.pop('overlay_offsets', None)
                        st.pop('last_render_positions', None)
                        st.pop('translated_texts', None)
                        st.pop('recognized_texts', None)  # Clear OCR data
                        st.pop('detection_regions', None)  # Clear detection data
                        st.pop('viewer_rectangles', None)  # Clear rectangle data
                        self.manga_integration.image_state_manager.set_state(self.current_image_path, st, save=True)
                except Exception:
                    pass
        except Exception:
            pass
        self._persist_rectangles_state()
    
    def load_image(self, image_path: str, preserve_rectangles: bool = False, preserve_text_overlays: bool = False):
        """Load an image into the preview (async) with explicit cleanup of previous image
        
        Args:
            image_path: Path to the image to load
            preserve_rectangles: If True, don't clear rectangles when loading (for workflow continuity)
            preserve_text_overlays: If True, keep text overlays visible (for cleaned images)
        """
        # MEMORY OPTIMIZATION: Explicitly clear previous image before loading new one
        try:
            if hasattr(self.viewer, '_pixmap_item') and self.viewer._pixmap_item:
                old_pixmap = self.viewer._pixmap_item.pixmap()
                if old_pixmap and not old_pixmap.isNull():
                    # Clear reference and schedule deletion
                    self.viewer._pixmap_item.setPixmap(QPixmap())
                    old_pixmap = None
        except Exception:
            pass
        
        # Store the preserve flags for use in _on_image_loading_started
        self._preserve_rectangles_on_load = preserve_rectangles
        self._preserve_text_overlays_on_load = preserve_text_overlays
        
        # Check for cleaned image and use it instead of original in src tab if available
        src_image_path = self._check_for_cleaned_image(image_path)
        
        # CRITICAL: Set current_image_path BEFORE starting async load
        # Otherwise _on_image_loaded_success will use the old/stale path
        self.current_image_path = image_path  # Still track original path for state management
        print(f"[STATE_ISOLATION] Set current_image_path to: {os.path.basename(image_path)}")
        
        # Start loading (happens in background)
        self.viewer.load_image(src_image_path)
        
        # Update thumbnail selection
        self._update_thumbnail_selection(image_path)
    
    def _emit_recognize_signal(self):
        """Emit recognize text signal"""
        self.recognize_text_clicked.emit()
    
    def _emit_translate_signal(self):
        """Emit translate text signal"""
        self.translate_text_clicked.emit()
    
    def _emit_translate_all_signal(self):
        """Emit translate all signal"""
        self.translate_all_clicked.emit()
    
    def _on_save_overlay_clicked(self):
        """Re-render overlay with current rectangle positions and rendering settings"""
        print("[SAVE_OVERLAY] Button clicked - scheduling background render")
        
        try:
            # Check if we have manga integration available
            if not hasattr(self, 'manga_integration') or not self.manga_integration:
                print("[SAVE_OVERLAY] No manga_integration available")
                return
            
            # Persist current rectangle positions first
            try:
                if hasattr(self, '_persist_rectangles_state'):
                    self._persist_rectangles_state()
                    print("[SAVE_OVERLAY] Rectangle positions persisted")
            except Exception as e:
                print(f"[SAVE_OVERLAY] Failed to persist rectangles: {e}")
            
            from PySide6.QtCore import QTimer
            import ImageRenderer
            
            # Run save_positions_and_rerender in background thread using executor
            if hasattr(self, 'main_gui') and hasattr(self.main_gui, 'executor') and self.main_gui.executor:
                print("[SAVE_OVERLAY] Submitting to background thread executor")
                future = self.main_gui.executor.submit(ImageRenderer.save_positions_and_rerender, self.manga_integration)
                
                # Schedule refreshes after rendering completes
                def _refresh_source():
                    try:
                        if getattr(self, 'current_image_path', None):
                            # Preserve rectangles and overlays while reloading to show updated render
                            self.load_image(self.current_image_path, preserve_rectangles=True, preserve_text_overlays=True)
                            print("[SAVE_OVERLAY] Source preview refreshed")
                    except Exception as _e:
                        print(f"[SAVE_OVERLAY] Source refresh failed: {_e}")
                
                # Refresh after rendering completes
                QTimer.singleShot(500, _refresh_source)
                QTimer.singleShot(1500, _refresh_source)
                QTimer.singleShot(3000, _refresh_source)
            else:
                print("[SAVE_OVERLAY] No executor available, falling back to main thread")
                # Fallback to QTimer if no executor (shouldn't happen)
                QTimer.singleShot(0, lambda: ImageRenderer.save_positions_and_rerender(self.manga_integration))
                
                def _refresh_source():
                    try:
                        if getattr(self, 'current_image_path', None):
                            self.load_image(self.current_image_path, preserve_rectangles=True, preserve_text_overlays=True)
                            print("[SAVE_OVERLAY] Source preview refreshed")
                    except Exception as _e:
                        print(f"[SAVE_OVERLAY] Source refresh failed: {_e}")
                
                QTimer.singleShot(500, _refresh_source)
                QTimer.singleShot(1500, _refresh_source)
                QTimer.singleShot(3000, _refresh_source)
        
        except Exception as e:
            print(f"[SAVE_OVERLAY] Exception: {e}")
            import traceback
            print(f"[SAVE_OVERLAY] Traceback:\n{traceback.format_exc()}")
    
    
    def set_image_list(self, image_paths: list):
        """Set the list of images and populate thumbnails"""
        self.image_paths = image_paths
        self._populate_thumbnails()
        
        # Show/hide thumbnail list based on image count
        if len(image_paths) <= 1:
            self.thumbnail_list.setVisible(False)
        else:
            self.thumbnail_list.setVisible(True)
    
    def _populate_thumbnails(self):
        """Populate thumbnail list with images"""
        self.thumbnail_list.clear()
        
        from PySide6.QtWidgets import QListWidgetItem
        from PySide6.QtCore import Qt, QSize
        import threading
        
        # Add items with loading placeholders first
        for i, path in enumerate(self.image_paths):
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, path)  # Store path
            item.setText(os.path.basename(path))
            item.setSizeHint(QSize(110, 110))
            self.thumbnail_list.addItem(item)
        
        # Load thumbnails in background thread
        def load_thumb(path):
            try:
                # Use QImage in background thread (thread-safe), not QPixmap
                image = QImage(path)
                if not image.isNull():
                    # Scale the QImage in background thread
                    scaled_image = image.scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio, 
                                               Qt.TransformationMode.SmoothTransformation)
                    # Make a deep copy to ensure it survives thread transition
                    img_copy = QImage(scaled_image)
                    # Emit signal to update thumbnail on main thread (thread-safe)
                    self._thumbnail_loaded_signal.emit(path, img_copy)
            except Exception as e:
                print(f"Failed to load thumbnail: {e}")
        
        def load_all_thumbs():
            for path in self.image_paths:
                load_thumb(path)
        
        thread = threading.Thread(target=load_all_thumbs, daemon=True)
        thread.start()
    
    @Slot(str, QImage)
    def _on_thumbnail_loaded(self, path: str, image: QImage):
        """Handle loaded thumbnail and update the list item"""
        try:
            # Convert QImage to QPixmap on main thread (thread-safe)
            pixmap = QPixmap.fromImage(image)
            
            # Find the item with matching path
            for i in range(self.thumbnail_list.count()):
                item = self.thumbnail_list.item(i)
                item_path = item.data(Qt.ItemDataRole.UserRole)
                if item_path == path:
                    icon = QIcon(pixmap)
                    item.setIcon(icon)
                    item.setText("")  # Remove text once icon is loaded
                    break
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
    
    def _on_rectangle_created(self, _rect: QRectF):
        """Assign index/flags and attach context menu to the newest rectangle."""
        try:
            # Get latest rectangle item
            if not hasattr(self.viewer, 'rectangles') or not self.viewer.rectangles:
                return
            rect_item = self.viewer.rectangles[-1]
            idx = len(self.viewer.rectangles) - 1
            # Mark as detection/red (no OCR yet)
            try:
                rect_item.region_index = idx
                rect_item.is_recognized = False
            except Exception:
                pass
            # Attach context menu via integration if available
            try:
                if hasattr(self, 'manga_integration') and self.manga_integration:
                    import ImageRenderer
                    ImageRenderer._add_context_menu_to_rectangle(self.manga_integration, rect_item, idx)
            except Exception:
                pass
        except Exception:
            pass
    
    def _on_thumbnail_clicked(self, item):
        """Handle thumbnail click - load the corresponding image"""
        from PySide6.QtCore import Qt
        image_path = item.data(Qt.ItemDataRole.UserRole)
        if image_path and os.path.exists(image_path):
            # Remove any processing overlay when switching images
            try:
                if hasattr(self, 'manga_integration') and self.manga_integration:
                    import ImageRenderer
                    ImageRenderer._remove_processing_overlay(self.manga_integration)
                    print(f"[THUMBNAIL_CLICK] Removed processing overlay when switching to: {os.path.basename(image_path)}")
            except Exception as e:
                print(f"[THUMBNAIL_CLICK] Failed to remove processing overlay: {e}")
            
            self.load_image(image_path)
    
    @Slot(str)
    def _on_image_loading_started(self, image_path: str):
        """Handle image loading started"""
        # Show loading status immediately
        self.file_label.setText(f"⌛ Loading {os.path.basename(image_path)}...")
        self.file_label.setStyleSheet("color: #5a9fd4; font-size: 8pt;")  # Blue while loading
        
        # Remove any processing overlay when switching images (e.g., during batch translation)
        try:
            if hasattr(self, 'manga_integration') and self.manga_integration:
                import ImageRenderer
                ImageRenderer._remove_processing_overlay(self.manga_integration)
                print(f"[IMAGE_LOADING] Removed processing overlay when loading: {os.path.basename(image_path)}")
        except Exception as e:
            print(f"[IMAGE_LOADING] Failed to remove processing overlay: {e}")
        
        # STATE ISOLATION FIX: Clear cross-image state FIRST to prevent leakage
        # This must happen before any restoration or overlay showing
        preserve_overlays = getattr(self, '_preserve_text_overlays_on_load', False)
        if not preserve_overlays:
            try:
                if hasattr(self, 'manga_integration') and self.manga_integration:
                    import ImageRenderer
                    ImageRenderer._clear_cross_image_state(self.manga_integration)
                    print(f"[STATE_ISOLATION] Cleared cross-image state before loading {os.path.basename(image_path)}")
            except Exception as e:
                print(f"[STATE_ISOLATION] Failed to clear cross-image state: {e}")
        
        # Clear previous image data (unless preserve flag is set for workflow continuity)
        preserve_rectangles = getattr(self, '_preserve_rectangles_on_load', False)
        if not preserve_rectangles:
            self.viewer.clear_rectangles()
            self._clear_strokes()
            self._update_box_count()
        else:
            # Only clear strokes, keep rectangles for workflow continuity (e.g., after cleaning)
            self._clear_strokes()
        
        # Handle text overlays based on preserve flag
        try:
            if hasattr(self, 'manga_integration') and self.manga_integration:
                if preserve_overlays:
                    # Keep existing overlays - just ensure they're visible for the current image
                    # Use current_image_path (original) for overlay lookup since overlays are mapped to original path
                    overlay_path = getattr(self, 'current_image_path', image_path) or image_path
                    print(f"[LOAD] Preserving text overlays for: {os.path.basename(overlay_path)}")
                    import ImageRenderer
                    ImageRenderer.show_text_overlays_for_image(self.manga_integration, overlay_path)
                else:
                    # Hide existing overlays for new image
                    import ImageRenderer
                    ImageRenderer.show_text_overlays_for_image(self.manga_integration, image_path)
        except Exception:
            pass
        
        # Store the preserve flags for _on_image_loaded_success to use
        self._preserve_rectangles_after_load = getattr(self, '_preserve_rectangles_on_load', False)
        self._preserve_text_overlays_after_load = getattr(self, '_preserve_text_overlays_on_load', False)
        
        # Reset the load flags after storing them
        self._preserve_rectangles_on_load = False
        self._preserve_text_overlays_on_load = False
    
    @Slot(str)
    def _on_image_loaded_success(self, image_path: str):
        """Handle successful image load"""
        if self.current_image_path:
            self.file_label.setText(f"📄 {os.path.basename(self.current_image_path)}")
            self.file_label.setStyleSheet("color: gray; font-size: 8pt;")  # Back to normal
        
        # Restore overlays for the newly loaded image (but only if not preserving)
        preserve_overlays_after = getattr(self, '_preserve_text_overlays_after_load', False)
        preserve_rectangles_after = getattr(self, '_preserve_rectangles_after_load', False)
        
        try:
            if hasattr(self, 'manga_integration') and self.manga_integration:
                if not preserve_rectangles_after and not preserve_overlays_after:
                    # Normal load - restore full state from persistence
                    print(f"[LOADED] Normal state restoration for: {os.path.basename(self.current_image_path)}")
                    
                    # STATE ISOLATION: Explicitly clear rectangles BEFORE restoration to prevent accumulation
                    rect_count_before = len(self.viewer.rectangles) if hasattr(self.viewer, 'rectangles') else 0
                    if rect_count_before > 0:
                        print(f"[STATE_ISOLATION] Found {rect_count_before} lingering rectangles - clearing before restoration")
                        self.viewer.clear_rectangles()
                    
                    # Restore detection/recognition overlays
                    import ImageRenderer
                    ImageRenderer._restore_image_state_overlays_only(self.manga_integration, self.current_image_path)
                    # Attach recognition tooltips/updates
                    if hasattr(self.manga_integration, 'show_recognized_overlays_for_image'):
                        ImageRenderer.show_recognized_overlays_for_image(self.manga_integration, self.current_image_path)
                    # Restore translated text overlays using current_image_path (original)
                    # Rebuild overlays with current settings to ensure they match
                    if hasattr(self.manga_integration, '_translated_texts') and self.manga_integration._translated_texts:
                        ImageRenderer._add_text_overlay_to_viewer(self.manga_integration, self.manga_integration._translated_texts)
                    else:
                        ImageRenderer.show_text_overlays_for_image(self.manga_integration, self.current_image_path)
                else:
                    # Preserve mode - don't restore from state, keep current rectangles/overlays
                    print(f"[LOADED] Preserve mode - skipping state restoration for: {os.path.basename(self.current_image_path)}")
        except Exception:
            pass
        
        # Reset the preserve flags after use
        self._preserve_rectangles_after_load = False
        self._preserve_text_overlays_after_load = False
        
    
    def clear(self):
        """Clear the preview"""
        # Clear source viewer
        try:
            # Also clear overlays for current image persistently
            if hasattr(self, 'manga_integration') and self.manga_integration and self.current_image_path:
                try:
                    import ImageRenderer
                    ImageRenderer.clear_text_overlays_for_image(self.manga_integration, self.current_image_path)
                except Exception:
                    pass
                try:
                    if hasattr(self.manga_integration, 'image_state_manager') and self.manga_integration.image_state_manager:
                        st = self.manga_integration.image_state_manager.get_state(self.current_image_path) or {}
                        st.pop('overlay_offsets', None)
                        st.pop('last_render_positions', None)
                        self.manga_integration.image_state_manager.set_state(self.current_image_path, st, save=True)
                except Exception:
                    pass
                # STATE ISOLATION FIX: Clear cross-image state when clearing preview
                try:
                    ImageRenderer._clear_cross_image_state(self.manga_integration)
                except Exception:
                    pass
        except Exception:
            pass
        self.viewer.clear_scene()
        self.current_image_path = None
        self.file_label.setText("No image loaded")
        self._update_box_count()
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
        
        
        # Show/hide manual editing tools ONLY (not pan/zoom which are always visible)
        self.box_draw_btn.setVisible(enabled)
        if hasattr(self, 'circle_draw_btn') and self.circle_draw_btn is not None:
            self.circle_draw_btn.setVisible(enabled)
        if hasattr(self, 'lasso_btn') and self.lasso_btn is not None:
            self.lasso_btn.setVisible(enabled)
        self.save_overlay_btn.setVisible(enabled)
        # Only set visibility for experimental tools if they exist
        if self.brush_btn is not None:
            self.brush_btn.setVisible(enabled)
        if self.eraser_btn is not None:
            self.eraser_btn.setVisible(enabled)
        self.delete_btn.setVisible(enabled)
        self.clear_boxes_btn.setVisible(enabled)
        # Only set visibility for clear strokes button if it exists
        if self.clear_strokes_btn is not None:
            self.clear_strokes_btn.setVisible(enabled)
        # Only set visibility for size controls if they exist
        if self.size_slider is not None:
            self.size_slider.setVisible(enabled)
        if self.size_label is not None:
            self.size_label.setVisible(enabled)
        self.box_count_label.setVisible(enabled)
        
        # Show/hide the entire Translation Workflow frame based on manual editing toggle
        self.workflow_frame.setVisible(enabled)
        
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
                # Save config to disk
                if hasattr(self.main_gui, 'save_config'):
                    self.main_gui.save_config(show_message=False)
        except Exception:
            pass
    
    def _cycle_source_display_mode(self):
        """Cycle through 3 source display modes: translated -> cleaned -> original -> translated"""
        try:
            # Cycle to next state
            if self.source_display_mode == 'translated':
                self.source_display_mode = 'cleaned'
            elif self.source_display_mode == 'cleaned':
                self.source_display_mode = 'original'
            else:  # original
                self.source_display_mode = 'translated'
            
            # Update deprecated flag for compatibility
            self.cleaned_images_enabled = (self.source_display_mode != 'original')
            
            # Update button appearance and tooltip based on mode
            if self.source_display_mode == 'translated':
                self.cleaned_toggle_btn.setText("✒️")  # Pen for translated output (default)
                self.cleaned_toggle_btn.setToolTip("Showing translated output (click to cycle)")
                self.cleaned_toggle_btn.setStyleSheet("""
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
            elif self.source_display_mode == 'cleaned':
                self.cleaned_toggle_btn.setText("🧽")  # Sponge for cleaned
                self.cleaned_toggle_btn.setToolTip("Showing cleaned images (click to cycle)")
                self.cleaned_toggle_btn.setStyleSheet("""
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
            else:  # original
                self.cleaned_toggle_btn.setText("📄")  # Document for original
                self.cleaned_toggle_btn.setToolTip("Showing original images (click to cycle)")
                self.cleaned_toggle_btn.setStyleSheet("""
                    QToolButton {
                        background-color: #6c757d;
                        border: 2px solid #777777;
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
                        background-color: #777777;
                    }
                """)
            
            # If we have a current image, reload it to apply the new mode
            if self.current_image_path and os.path.exists(self.current_image_path):
                print(f"[DISPLAY_MODE] Reloading image with mode: {self.source_display_mode}")
                self.load_image(self.current_image_path, preserve_rectangles=True, preserve_text_overlays=True)
            
        except Exception as e:
            print(f"[ERROR] Failed to cycle source display mode: {e}")
    
    def _on_download_images_clicked(self):
        """Handle download images button - consolidate isolated images into structured folder"""
        try:
            from PySide6.QtWidgets import QFileDialog, QMessageBox
            import shutil
            import glob
            
            # Get the parent directory of translated images
            # Since each image is now in its own *_translated/ folder, we need to find them
            if not hasattr(self, 'manga_integration') or not self.manga_integration:
                return
            
            # Get the source directory from selected files
            if not hasattr(self.manga_integration, 'selected_files') or not self.manga_integration.selected_files:
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
            
            
            # Create consolidated "translated" folder in parent directory
            consolidated_folder = os.path.join(parent_dir, 'translated')
            os.makedirs(consolidated_folder, exist_ok=True)
            
            # Copy all images from isolated folders to consolidated folder
            image_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif')
            copied_count = 0
            
            for folder in translated_folders:
                for filename in os.listdir(folder):
                    if filename.lower().endswith(image_extensions):
                        # Skip cleaned images (intermediate files)
                        if '_cleaned' in filename.lower():
                            continue
                        
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
            
            
            # Show success message
            QMessageBox.information(
                self,
                "Download Complete",
                f"Successfully consolidated {copied_count} translated images into:\n{consolidated_folder}\n\n"
                f"All images from separate folders have been combined into a single 'translated' folder."
            )
            
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(
                self,
                "Download Error",
                f"Failed to consolidate images:\n{str(e)}"
            )
    
    def _on_rectangle_moved(self, rect: QRectF):
        """Handle rectangle movement - automatically apply save position"""
        try:
            # Find which rectangle was moved by comparing scene coordinates
            moved_index = -1
            for idx, rectangle in enumerate(self.viewer.rectangles):
                if rectangle.sceneBoundingRect() == rect:
                    moved_index = idx
                    break
            
            if moved_index >= 0:
                print(f"[DEBUG] Rectangle {moved_index} moved, auto-applying save position")
                
                # Add purple pulse effect to show processing (auto-remove after 0.75s for save position)
                try:
                    rect_item = self.viewer.rectangles[moved_index]
                    if hasattr(self, 'manga_integration') and self.manga_integration:
                        import ImageRenderer
                        ImageRenderer._add_rectangle_pulse_effect(self.manga_integration, rect_item, moved_index, auto_remove=True)
                except Exception as e:
                    print(f"[PULSE] Failed to add pulse effect: {e}")
                
                # IMMEDIATE REFRESH: Reload current translated image first (optimistic update)
                try:
                    if self.current_image_path:
                        print(f"[INSTANT] Reloading current preview immediately for instant feedback")
                        self.load_image(self.current_image_path, preserve_rectangles=True, preserve_text_overlays=True)
                except Exception as e:
                    print(f"[INSTANT] Immediate refresh failed: {e}")
                
                # THEN trigger background re-render IMMEDIATELY (no delay)
                if hasattr(self, 'manga_integration') and self.manga_integration:
                    import ImageRenderer
                    ImageRenderer._save_position_async(self.manga_integration, moved_index)
        except Exception as e:
            print(f"[DEBUG] Error in _on_rectangle_moved: {e}")
    
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
                    # Check if this folder has images (excluding cleaned images)
                    if any(f.lower().endswith(image_extensions) and '_cleaned' not in f.lower() for f in os.listdir(item_path)):
                        has_translated_images = True
                        break
            
            # Also check the folder_path itself for backward compatibility
            if not has_translated_images and os.path.isdir(folder_path):
                has_translated_images = any(
                    f.lower().endswith(image_extensions) and '_cleaned' not in f.lower()
                    for f in os.listdir(folder_path)
                )
            
            self.download_btn.setEnabled(has_translated_images)
        else:
            self.download_btn.setEnabled(False)
    
    
    def _check_for_cleaned_image(self, source_image_path: str) -> str:
        """Check for appropriate image based on source display mode.
        
        Args:
            source_image_path: Path to the original source image
            
        Returns:
            Path based on current display mode: translated output, cleaned, or original
        """
        try:
            # Get current display mode (translated, cleaned, or original)
            mode = getattr(self, 'source_display_mode', 'translated')
            
            # If mode is 'original', return original immediately
            if mode == 'original':
                print(f"[SRC] Display mode is 'original', using: {os.path.basename(source_image_path)}")
                return source_image_path
            
            # Build paths to check
            source_dir = os.path.dirname(source_image_path)
            source_filename = os.path.basename(source_image_path)
            source_name_no_ext = os.path.splitext(source_filename)[0]
            source_ext = os.path.splitext(source_filename)[1]
            
            # Isolated folder path
            translated_folder = os.path.join(source_dir, f"{source_name_no_ext}_translated")
            
            # If mode is 'translated', check for translated output first
            if mode == 'translated':
                # Check for translated output (non-cleaned file in isolated folder)
                translated_path = None
                if os.path.exists(translated_folder):
                    image_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif')
                    for filename in os.listdir(translated_folder):
                        name_lower = filename.lower()
                        # Find files that match the source name but NOT cleaned files
                        if (name_lower.startswith(source_name_no_ext.lower()) and 
                            name_lower.endswith(image_extensions) and
                            '_cleaned' not in name_lower):
                            translated_path = os.path.join(translated_folder, filename)
                            print(f"[SRC] Found translated output: {os.path.basename(translated_path)}")
                            return translated_path
                
                # Fall back to cleaned if no translated output found
                print(f"[SRC] No translated output found, falling back to cleaned image check")
            
            # For 'translated' (fallback) and 'cleaned' modes, look for cleaned image
            if mode == 'cleaned':
                print(f"[SRC] Display mode is 'cleaned', looking for cleaned image")
            
            # 1) Check state manager for saved cleaned image path (but only if it's actually a cleaned image)
            cleaned_path = None
            try:
                if hasattr(self, 'manga_integration') and self.manga_integration and \
                   hasattr(self.manga_integration, 'image_state_manager') and self.manga_integration.image_state_manager:
                    state = self.manga_integration.image_state_manager.get_state(source_image_path) or {}
                    cand = state.get('cleaned_image_path')
                    if cand and os.path.exists(cand):
                        # IMPORTANT: Validate that this is actually a cleaned image, not a translated image
                        cand_filename = os.path.basename(cand).lower()
                        print(f"[SRC] State has cleaned_image_path: {os.path.basename(cand)}")
                        print(f"[SRC] Validating filename: '{cand_filename}' contains '_cleaned': {'_cleaned' in cand_filename}")
                        if '_cleaned' in cand_filename and not cand_filename.endswith('.json'):
                            # Accept only if the path is inside the expected translated folder or the same directory
                            try:
                                cand_abs = os.path.abspath(cand)
                                tdir_abs = os.path.abspath(translated_folder)
                                sdir_abs = os.path.abspath(source_dir)
                                def _under(p, base):
                                    try:
                                        return os.path.commonpath([os.path.normcase(p), os.path.normcase(base)]) == os.path.normcase(base)
                                    except Exception:
                                        return False
                                if _under(cand_abs, tdir_abs) or _under(cand_abs, sdir_abs):
                                    cleaned_path = cand_abs
                                    print(f"[SRC] Found validated cleaned image from state: {os.path.basename(cleaned_path)}")
                                else:
                                    # Ignore stale path from another folder/session
                                    print(f"[SRC] Ignoring cleaned image path from different folder: {os.path.basename(cand)}")
                                    cleaned_path = None
                            except Exception:
                                cleaned_path = None
                        else:
                            # This path doesn't look like a cleaned image - might be a translated image
                            print(f"[SRC] Ignoring non-cleaned image path from state: {os.path.basename(cand)}")
                            cleaned_path = None
            except Exception:
                pass
            
            # 2) Check isolated folder for *_cleaned.* files
            if not cleaned_path and os.path.exists(translated_folder):
                image_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif')
                for filename in os.listdir(translated_folder):
                    name_lower = filename.lower()
                    if (name_lower.startswith(f"{source_name_no_ext.lower()}_cleaned") and 
                        name_lower.endswith(image_extensions)):
                        cleaned_path = os.path.join(translated_folder, filename)
                        print(f"[SRC] Found cleaned image in isolated folder: {os.path.basename(cleaned_path)}")
                        break
            
            # 3) Check same directory with _cleaned suffix
            if not cleaned_path:
                cleaned_candidate = os.path.join(source_dir, f"{source_name_no_ext}_cleaned{source_ext}")
                if os.path.exists(cleaned_candidate):
                    cleaned_path = cleaned_candidate
                    print(f"[SRC] Found cleaned image in same directory: {os.path.basename(cleaned_path)}")
            
            # Return cleaned image if found, otherwise original
            if cleaned_path and os.path.exists(cleaned_path):
                print(f"[SRC] Mode '{mode}': Using cleaned image: {os.path.basename(cleaned_path)}")
                return cleaned_path
            else:
                print(f"[SRC] Mode '{mode}': No cleaned/translated found, using original: {os.path.basename(source_image_path)}")
                return source_image_path
                
        except Exception as e:
            print(f"[SRC] Error in display mode '{mode}': {e}")
            return source_image_path
