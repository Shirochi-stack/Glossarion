#!/usr/bin/env python3
"""
Simple test script for the manga image preview widget with async loading
Run this to test the widget independently
"""

import sys
import os
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget
from PySide6.QtCore import Qt

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from manga_image_preview import MangaImagePreviewWidget


class TestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Manga Image Preview Test - Async Loading")
        self.setGeometry(100, 100, 1400, 900)
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        
        # Left side: file list + buttons
        left_layout = QVBoxLayout()
        
        self.file_list = QListWidget()
        self.file_list.setMinimumWidth(300)
        self.file_list.itemSelectionChanged.connect(self.on_selection_changed)
        left_layout.addWidget(self.file_list)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        add_btn = QPushButton("Add Test Images")
        add_btn.clicked.connect(self.add_test_images)
        btn_layout.addWidget(add_btn)
        
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear_all)
        btn_layout.addWidget(clear_btn)
        
        left_layout.addLayout(btn_layout)
        
        # Add to main layout
        layout.addLayout(left_layout, stretch=1)
        
        # Right side: image preview widget
        self.preview = MangaImagePreviewWidget()
        layout.addWidget(self.preview, stretch=0)
        
        # Status
        self.statusBar().showMessage("Ready - Select images from a folder or use 'Add Test Images'")
        
        # Test images list
        self.image_paths = []
    
    def add_test_images(self):
        """Add some test images from the user's Pictures folder or current directory"""
        from PySide6.QtWidgets import QFileDialog
        
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Test Images",
            "",
            "Images (*.png *.jpg *.jpeg *.gif *.bmp *.webp);;All files (*.*)"
        )
        
        if files:
            self.file_list.clear()
            self.image_paths = files
            
            for path in files:
                self.file_list.addItem(os.path.basename(path))
            
            self.statusBar().showMessage(f"Added {len(files)} images - Click rapidly to test async loading!")
    
    def on_selection_changed(self):
        """Handle file selection - demonstrates async loading"""
        selected = self.file_list.selectedItems()
        if not selected:
            return
        
        row = self.file_list.row(selected[0])
        if 0 <= row < len(self.image_paths):
            path = self.image_paths[row]
            
            # This call returns immediately - loading happens in background!
            self.preview.load_image(path)
            
            self.statusBar().showMessage(f"Loading {os.path.basename(path)}... (async)")
    
    def clear_all(self):
        """Clear everything"""
        self.file_list.clear()
        self.image_paths.clear()
        self.preview.clear()
        self.statusBar().showMessage("Cleared")


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    
    # Set dark theme
    app.setStyle("Fusion")
    from PySide6.QtGui import QPalette, QColor
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(35, 35, 35))
    palette.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Button, QColor(45, 45, 45))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
    app.setPalette(palette)
    
    window = TestWindow()
    window.show()
    
    print("=" * 60)
    print("Manga Image Preview Test - Async Loading")
    print("=" * 60)
    print("✅ GUI widgets created on main thread")
    print("✅ Image loading happens in background threads")
    print("✅ UI stays responsive during image loads")
    print()
    print("Instructions:")
    print("1. Click 'Add Test Images' to select some manga images")
    print("2. Click different images in the list rapidly")
    print("3. Notice the UI never freezes! ⚡")
    print("4. Loading indicator shows while images load")
    print("5. Previous loads are cancelled automatically")
    print()
    print("Try with large images (5-10MB) to see the difference!")
    print("=" * 60)
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
