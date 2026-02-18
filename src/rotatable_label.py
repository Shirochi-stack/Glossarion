from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Qt, Property, QSize
from PySide6.QtGui import QPainter, QPixmap

class RotatableLabel(QLabel):
    """
    A QLabel that supports rotation of its pixmap via a 'rotation' property,
    optimized for smooth animation by avoiding expensive scaling/transformations
    during the paint loop.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._rotation = 0.0
        self._original_pixmap = None
        self._scaled_pixmap = None
        self._last_size = QSize()
        self._painting = False  # Prevent concurrent painting
    
    def set_rotation(self, angle):
        self._rotation = angle
        self.update()  # Trigger paintEvent
    
    def get_rotation(self):
        return self._rotation
    
    # Define rotation as a Qt Property for animation
    rotation = Property(float, get_rotation, set_rotation)
    
    def set_original_pixmap(self, pixmap):
        self._original_pixmap = pixmap
        self._update_scaled_pixmap()
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self):
        if not self._original_pixmap or self.size().isEmpty():
            self._scaled_pixmap = None
            return
            
        w, h = self.width(), self.height()
        # Ensure we don't scale to 0
        if w <= 0 or h <= 0:
            return

        # Handle High-DPI
        dpr = self.devicePixelRatioF()
        side = min(w, h)
        target_px = int(side * dpr)
        
        # Scale the original pixmap to fit the label (in device pixels)
        # Check if we even need to scale (optimization)
        if self._original_pixmap.width() == target_px and self._original_pixmap.height() == target_px and self._original_pixmap.devicePixelRatio() == dpr:
             self._scaled_pixmap = self._original_pixmap
        else:
            self._scaled_pixmap = self._original_pixmap.scaled(
                target_px, target_px, 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            self._scaled_pixmap.setDevicePixelRatio(dpr)
            
        self._last_size = self.size()

    def paintEvent(self, event):
        # Prevent re-entrant painting
        if self._painting:
            return
            
        if not self._scaled_pixmap:
            super().paintEvent(event)
            return

        self._painting = True
        try:
            painter = QPainter(self)
            painter.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
            
            w, h = self.width(), self.height()
            
            # Translate to center
            painter.translate(w / 2, h / 2)
            
            # Rotate
            painter.rotate(self._rotation)
            
            # Draw the pre-scaled pixmap centered
            pix = self._scaled_pixmap
            painter.translate(-pix.width() / 2, -pix.height() / 2)
            painter.drawPixmap(0, 0, pix)
            
            # Let QPainter be destroyed automatically - no explicit end() needed
        finally:
            self._painting = False
