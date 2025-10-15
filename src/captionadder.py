#!/usr/bin/env python3
"""
Create output placeholder with large pink text with black outline
"""

import os
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPixmap, QPainter, QPen, QColor, QFont, QPainterPath
from PySide6.QtCore import Qt, QPointF
import sys

def create_output_placeholder():
    """Create output placeholder with large pink text with black outline"""
    
    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, 'Surprise_blushing.png')
    output_path = os.path.join(script_dir, 'WhereIsMyOutput.png')
    
    if not os.path.exists(input_path):
        print(f"ERROR: Input image not found: {input_path}")
        return False
    
    print(f"Loading: {input_path}")
    
    # Load the surprise image
    surprise_pixmap = QPixmap(input_path)
    
    if surprise_pixmap.isNull():
        print("ERROR: Failed to load input image")
        return False
    
    print(f"Image loaded: {surprise_pixmap.width()}x{surprise_pixmap.height()}")
    
    # Create a copy to draw on
    canvas = QPixmap(surprise_pixmap)
    
    # Start painting
    painter = QPainter(canvas)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    
    # Very large font for caption
    text_font = QFont("Comic Sans MS", 160, QFont.Weight.Bold)
    painter.setFont(text_font)
    
    # Text very close to top
    y_position = -50  # Move up a few more pixels
    text = "WHERE IS MY OUTPUT ?"
    
    # Get text bounds for centering
    font_metrics = painter.fontMetrics()
    text_width = font_metrics.horizontalAdvance(text)
    text_height = font_metrics.height()
    
    # Center horizontally
    x_position = (canvas.width() - text_width) // 2
    
    # Create text path for outline effect
    text_path = QPainterPath()
    text_path.addText(QPointF(x_position, y_position + text_height), text_font, text)
    
    # Draw black outline (stroke)
    painter.strokePath(text_path, QPen(QColor(0, 0, 0), 12, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    
    # Fill with pink
    pink_color = QColor(255, 105, 180)  # Hot pink
    painter.fillPath(text_path, pink_color)
    
    print("Large pink text with black outline drawn (moved up)")
    
    painter.end()
    
    # Save the result
    success = canvas.save(output_path, "PNG")
    
    if success:
        print(f"✅ Successfully created: {output_path}")
        print(f"   Size: {canvas.width()}x{canvas.height()}")
        return True
    else:
        print(f"❌ Failed to save: {output_path}")
        return False

if __name__ == "__main__":
    # QApplication is required for QPixmap operations
    app = QApplication(sys.argv)
    
    print("=" * 60)
    print("Creating output placeholder with pink outlined text...")
    print("=" * 60)
    
    success = create_output_placeholder()
    
    print("=" * 60)
    if success:
        print("DONE!")
    else:
        print("FAILED!")
    print("=" * 60)
    
    sys.exit(0 if success else 1)
