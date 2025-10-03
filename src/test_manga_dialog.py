#!/usr/bin/env python3

"""
Test script for the manga settings dialog
"""

import sys
import os
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget
from PySide6.QtCore import Qt

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from manga_settings_dialog import MangaSettingsDialog

class TestMainWindow(QMainWindow):
    """Simple test main window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Manga Settings Dialog Test")
        self.setGeometry(100, 100, 300, 150)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create layout
        layout = QVBoxLayout(central_widget)
        
        # Create test button
        test_button = QPushButton("Open Manga Settings Dialog")
        test_button.clicked.connect(self.open_dialog)
        layout.addWidget(test_button)
        
        # Mock config
        self.config = {}
        
    def open_dialog(self):
        """Open the manga settings dialog"""
        try:
            dialog = MangaSettingsDialog(
                parent=self,
                main_gui=self,  # Use self as mock main_gui
                config=self.config,
                callback=self.on_settings_saved
            )
            print("Dialog opened successfully!")
        except Exception as e:
            print(f"Error opening dialog: {e}")
            import traceback
            traceback.print_exc()
    
    def on_settings_saved(self, settings):
        """Callback when settings are saved"""
        print(f"Settings saved: {settings}")
        self.config['manga_settings'] = settings
    
    def save_config(self):
        """Mock save_config method"""
        print("Mock save_config called")

def main():
    """Run the test"""
    app = QApplication(sys.argv)
    
    # Create and show main window
    window = TestMainWindow()
    window.show()
    
    # Run the application
    sys.exit(app.exec())

if __name__ == "__main__":
    main()