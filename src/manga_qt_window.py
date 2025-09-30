# manga_qt_window.py
"""
Standalone PySide6 manga translator window.
Runs as a separate process and communicates with the main tkinter app via Redis.
"""

import sys
import os
from pathlib import Path

# Add src directory to path
src_dir = Path(__file__).parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QListWidget, QProgressBar, QTextEdit,
    QFileDialog, QMessageBox, QGroupBox, QCheckBox, QSpinBox
)
from PySide6.QtCore import Qt, QTimer, Signal, QThread
from PySide6.QtGui import QFont, QIcon

from redis_gui_bridge import RedisGUIBridge


class MangaQtWindow(QMainWindow):
    """PySide6 manga translator window."""
    
    def __init__(self):
        super().__init__()
        
        # Connect to Redis
        self.bridge = RedisGUIBridge()
        if not self.bridge.is_connected():
            QMessageBox.critical(
                self,
                "Redis Not Available",
                "Could not connect to Redis.\n\n"
                "Make sure Redis is running:\n"
                "  docker-compose up -d\n\n"
                "Or see REDIS_SETUP.md for instructions."
            )
            sys.exit(1)
        
        # Load state from Redis
        self.config = self.bridge.get_manga_config()
        self.files = self.bridge.get_manga_files()
        
        print(f"📥 Loaded {len(self.files)} files from Redis")
        print(f"📋 Config keys: {list(self.config.keys())}")
        
        # Setup UI
        self.setWindowTitle("🎌 Manga Panel Translator (PySide6)")
        self.setMinimumSize(900, 700)
        
        # Try to load icon
        self._load_icon()
        
        # Create central widget
        central = QWidget()
        self.setCentralWidget(central)
        
        # Main layout
        layout = QVBoxLayout(central)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("🎌 Manga Translation")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(title)
        
        # File selection group
        file_group = QGroupBox("Selected Manga Images")
        file_layout = QVBoxLayout()
        
        self.file_list = QListWidget()
        self.file_list.addItems(self.files)
        file_layout.addWidget(self.file_list)
        
        # File buttons
        btn_layout = QHBoxLayout()
        
        add_files_btn = QPushButton("Add Files")
        add_files_btn.clicked.connect(self.add_files)
        btn_layout.addWidget(add_files_btn)
        
        add_folder_btn = QPushButton("Add Folder")
        add_folder_btn.clicked.connect(self.add_folder)
        btn_layout.addWidget(add_folder_btn)
        
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear_files)
        btn_layout.addWidget(clear_btn)
        
        btn_layout.addStretch()
        file_layout.addLayout(btn_layout)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # Progress
        progress_group = QGroupBox("Translation Progress")
        progress_layout = QVBoxLayout()
        
        self.status_label = QLabel("Ready")
        progress_layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # Log area
        log_group = QGroupBox("Translation Log")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        # Control buttons
        control_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Start Translation")
        self.start_btn.clicked.connect(self.start_translation)
        self.start_btn.setMinimumHeight(40)
        control_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_translation)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setMinimumHeight(40)
        control_layout.addWidget(self.stop_btn)
        
        layout.addLayout(control_layout)
        
        # Setup state synchronization
        self._setup_sync()
        
        self.log("✅ Manga translator ready (PySide6 + Redis)")
        self.log(f"📁 Loaded {len(self.files)} file(s) from main app")
    
    def _load_icon(self):
        """Try to load application icon."""
        try:
            icon_path = Path(__file__).parent / 'Halgakos.ico'
            if icon_path.exists():
                self.setWindowIcon(QIcon(str(icon_path)))
        except:
            pass
    
    def _setup_sync(self):
        """Setup bi-directional state sync with Redis."""
        # Watch for file list changes from main app
        self.bridge.watch_state('manga:files', self._on_files_changed)
        
        # Watch for config changes
        self.bridge.watch_state('manga:config', self._on_config_changed)
        
        # Subscribe to events
        self.bridge.subscribe_event('manga:events', self._on_event)
    
    def _on_files_changed(self, files):
        """Called when file list changes in Redis."""
        if files != self.files:
            self.files = files or []
            self.file_list.clear()
            self.file_list.addItems(self.files)
            self.log(f"🔄 File list updated: {len(self.files)} file(s)")
    
    def _on_config_changed(self, config):
        """Called when config changes in Redis."""
        self.config = config or {}
        self.log("🔄 Configuration updated from main app")
    
    def _on_event(self, event):
        """Called when event is published."""
        event_type = event.get('type')
        message = event.get('message', '')
        
        if event_type == 'log':
            self.log(message)
        elif event_type == 'progress':
            self.progress_bar.setValue(event.get('value', 0))
        elif event_type == 'status':
            self.status_label.setText(message)
    
    def log(self, message):
        """Add message to log."""
        self.log_text.append(message)
        print(message)
    
    def add_files(self):
        """Add files to list."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Manga Images",
            "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp)"
        )
        
        if files:
            self.files.extend(files)
            self.file_list.addItems(files)
            # Sync to Redis
            self.bridge.set_manga_files(self.files)
            self.log(f"➕ Added {len(files)} file(s)")
    
    def add_folder(self):
        """Add folder of images."""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            # Find all image files
            from pathlib import Path
            extensions = ['.png', '.jpg', '.jpeg', '.webp', '.bmp']
            image_files = []
            for ext in extensions:
                image_files.extend([str(p) for p in Path(folder).rglob(f'*{ext}')])
            
            if image_files:
                self.files.extend(image_files)
                self.file_list.addItems(image_files)
                # Sync to Redis
                self.bridge.set_manga_files(self.files)
                self.log(f"📁 Added {len(image_files)} file(s) from folder")
            else:
                self.log("⚠️ No image files found in folder")
    
    def clear_files(self):
        """Clear file list."""
        self.files = []
        self.file_list.clear()
        # Sync to Redis
        self.bridge.set_manga_files(self.files)
        self.log("🗑️ File list cleared")
    
    def start_translation(self):
        """Start translation."""
        if not self.files:
            QMessageBox.warning(self, "No Files", "Please add some manga images first.")
            return
        
        self.log("🚀 Starting translation...")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        
        # TODO: Implement actual translation logic
        # For now, just simulate
        self._simulate_translation()
    
    def _simulate_translation(self):
        """Simulate translation progress."""
        self.status_label.setText("Translating...")
        
        def update_progress():
            current = self.progress_bar.value()
            if current < 100:
                self.progress_bar.setValue(current + 10)
                QTimer.singleShot(500, update_progress)
            else:
                self.status_label.setText("✅ Translation complete!")
                self.start_btn.setEnabled(True)
                self.stop_btn.setEnabled(False)
                self.log("✅ Translation finished!")
        
        QTimer.singleShot(500, update_progress)
    
    def stop_translation(self):
        """Stop translation."""
        self.log("🛑 Stopping translation...")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Stopped")
    
    def closeEvent(self, event):
        """Handle window close."""
        self.bridge.cleanup()
        event.accept()


def main():
    """Main entry point."""
    print("🚀 Starting PySide6 manga translator...")
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern style
    
    window = MangaQtWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()