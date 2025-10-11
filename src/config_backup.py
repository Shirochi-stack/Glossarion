"""Config Backup Management Methods for Glossarion

These methods handle automatic and manual config backup/restore functionality.
They are designed to be bound to the TranslatorGUI instance.
"""

import os
import sys
import time
import shutil
import json
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTreeWidget, QTreeWidgetItem, QMessageBox, QFrame, QGroupBox,
    QApplication
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QFont

# Import required from translator_gui
from translator_gui import CONFIG_FILE, decrypt_config


def _backup_config_file(self):
    """Create backup of the existing config file before saving."""
    try:
        # Skip if config file doesn't exist yet
        if not os.path.exists(CONFIG_FILE):
            return
            
        # Get base directory that works in both development and frozen environments
        base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        
        # Resolve config file path for backup directory
        if os.path.isabs(CONFIG_FILE):
            config_dir = os.path.dirname(CONFIG_FILE)
        else:
            config_dir = os.path.dirname(os.path.abspath(CONFIG_FILE))
        
        # Create backup directory
        backup_dir = os.path.join(config_dir, "config_backups")
        os.makedirs(backup_dir, exist_ok=True)
        
        # Create timestamped backup name
        backup_name = f"config_{time.strftime('%Y%m%d_%H%M%S')}.json.bak"
        backup_path = os.path.join(backup_dir, backup_name)
        
        # Copy the file
        shutil.copy2(CONFIG_FILE, backup_path)
        
        # Clean backups older than 72 hours
        cutoff_time = time.time() - (72 * 60 * 60)  # 72 hours in seconds
        backups = [os.path.join(backup_dir, f) for f in os.listdir(backup_dir) 
                   if f.startswith("config_") and f.endswith(".json.bak")]
        
        # Remove backups older than 72 hours
        for backup_file in backups:
            try:
                if os.path.getmtime(backup_file) <= cutoff_time:
                    os.remove(backup_file)
            except Exception:
                pass  # Ignore errors when cleaning old backups
    
    except Exception as e:
        # Silent exception - don't interrupt normal operation if backup fails
        print(f"Warning: Could not create config backup: {e}")

def _restore_config_from_backup(self):
    """Attempt to restore config from the most recent backup."""
    try:
        # Locate backups directory
        if os.path.isabs(CONFIG_FILE):
            config_dir = os.path.dirname(CONFIG_FILE)
        else:
            config_dir = os.path.dirname(os.path.abspath(CONFIG_FILE))
        backup_dir = os.path.join(config_dir, "config_backups")
        
        if not os.path.exists(backup_dir):
            return
        
        # Find most recent backup
        backups = [os.path.join(backup_dir, f) for f in os.listdir(backup_dir) 
                  if f.startswith("config_") and f.endswith(".json.bak")]
        
        if not backups:
            return
            
        backups.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        latest_backup = backups[0]
        
        # Copy backup to config file
        shutil.copy2(latest_backup, CONFIG_FILE)
        from PySide6.QtWidgets import QMessageBox
        from PySide6.QtGui import QIcon
        
        # Get icon path
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "halgakos.ico")
        icon = QIcon(icon_path) if os.path.exists(icon_path) else QIcon()
        
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setWindowTitle("Config Restored")
        msg_box.setText(f"Configuration was restored from backup: {os.path.basename(latest_backup)}")
        msg_box.setWindowIcon(icon)
        msg_box.exec()
        
        # Reload config
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
                self.config = decrypt_config(self.config)
        except Exception as e:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Critical)
            msg_box.setWindowTitle("Error")
            msg_box.setText(f"Failed to reload configuration: {e}")
            msg_box.setWindowIcon(icon)
            msg_box.exec()
            
    except Exception as e:
        from PySide6.QtWidgets import QMessageBox
        from PySide6.QtGui import QIcon
        
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "halgakos.ico")
        icon = QIcon(icon_path) if os.path.exists(icon_path) else QIcon()
        
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle("Restore Failed")
        msg_box.setText(f"Could not restore config from backup: {e}")
        msg_box.setWindowIcon(icon)
        msg_box.exec()
        
def _create_manual_config_backup(self):
    """Create a manual config backup."""
    try:
        # Force create backup even if config file doesn't exist
        self._backup_config_file()
        from PySide6.QtWidgets import QMessageBox
        from PySide6.QtGui import QIcon
        
        # Get icon path
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "halgakos.ico")
        icon = QIcon(icon_path) if os.path.exists(icon_path) else QIcon()
        
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setWindowTitle("Backup Created")
        msg_box.setText("Configuration backup created successfully!")
        msg_box.setWindowIcon(icon)
        msg_box.exec()
    except Exception as e:
        from PySide6.QtWidgets import QMessageBox
        from PySide6.QtGui import QIcon
        
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "halgakos.ico")
        icon = QIcon(icon_path) if os.path.exists(icon_path) else QIcon()
        
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle("Backup Failed")
        msg_box.setText(f"Failed to create backup: {e}")
        msg_box.setWindowIcon(icon)
        msg_box.exec()

def _open_backup_folder(self):
    """Open the config backups folder in file explorer."""
    try:
        from PySide6.QtGui import QIcon
        
        # Get icon path
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "halgakos.ico")
        icon = QIcon(icon_path) if os.path.exists(icon_path) else QIcon()
        
        if os.path.isabs(CONFIG_FILE):
            config_dir = os.path.dirname(CONFIG_FILE)
        else:
            config_dir = os.path.dirname(os.path.abspath(CONFIG_FILE))
        backup_dir = os.path.join(config_dir, "config_backups")
        
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir, exist_ok=True)
            from PySide6.QtWidgets import QMessageBox
            
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setWindowTitle("Backup Folder")
            msg_box.setText(f"Created backup folder: {backup_dir}")
            msg_box.setWindowIcon(icon)
            msg_box.exec()
        
        # Open folder in explorer (cross-platform)
        import subprocess
        import platform
        
        if platform.system() == "Windows":
            os.startfile(backup_dir)
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["open", backup_dir])
        else:  # Linux
            subprocess.run(["xdg-open", backup_dir])
            
    except Exception as e:
        from PySide6.QtWidgets import QMessageBox
        from PySide6.QtGui import QIcon
        
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "halgakos.ico")
        icon = QIcon(icon_path) if os.path.exists(icon_path) else QIcon()
        
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle("Error")
        msg_box.setText(f"Could not open backup folder: {e}")
        msg_box.setWindowIcon(icon)
        msg_box.exec()

def _manual_restore_config(self):
    """Show dialog to manually select and restore a config backup."""
    try:
        # Ensure QApplication exists
        app = QApplication.instance()
        if not app:
            try:
                QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
            except:
                pass
            app = QApplication(sys.argv)
        
        if os.path.isabs(CONFIG_FILE):
            config_dir = os.path.dirname(CONFIG_FILE)
        else:
            config_dir = os.path.dirname(os.path.abspath(CONFIG_FILE))
        backup_dir = os.path.join(config_dir, "config_backups")
        
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "halgakos.ico")
        icon = QIcon(icon_path) if os.path.exists(icon_path) else QIcon()
        
        if not os.path.exists(backup_dir):
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setWindowTitle("No Backups")
            msg_box.setText("No backup folder found. No backups have been created yet.")
            msg_box.setWindowIcon(icon)
            msg_box.exec()
            return
        
        # Get list of available backups
        backups = [f for f in os.listdir(backup_dir) 
                  if f.startswith("config_") and f.endswith(".json.bak")]
        
        if not backups:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setWindowTitle("No Backups")
            msg_box.setText("No config backups found.")
            msg_box.setWindowIcon(icon)
            msg_box.exec()
            return
        
        # Sort by creation time (newest first)
        backups.sort(key=lambda x: os.path.getmtime(os.path.join(backup_dir, x)), reverse=True)
        
        # Create PySide6 dialog
        dialog = QDialog(None)
        dialog.setWindowTitle("Config Backup Manager")
        dialog.setWindowIcon(icon)
        
        # Get screen dimensions and calculate size
        screen = app.primaryScreen().geometry()
        dialog_width = int(screen.width() * 0.3)  # Reduced from 0.6 to 0.3
        dialog_height = int(screen.height() * 0.4)  # Reduced from 0.8 to 0.4
        dialog.resize(dialog_width, dialog_height)
        
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # Header
        title_label = QLabel("Configuration Backup Manager")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        main_layout.addWidget(title_label)
        
        desc_label = QLabel("Select a backup to restore or manage your configuration backups.")
        desc_label.setStyleSheet("color: gray; font-size: 10pt;")
        main_layout.addWidget(desc_label)
        
        # Info section
        info_group = QGroupBox("Backup Information")
        info_layout = QVBoxLayout()
        info_layout.setContentsMargins(10, 10, 10, 10)
        
        info_text = f"📁 Backup Location: {backup_dir}\n📊 Total Backups: {len(backups)}"
        info_label = QLabel(info_text)
        info_label.setStyleSheet("color: white; font-size: 10pt;")
        info_layout.addWidget(info_label)
        info_group.setLayout(info_layout)
        main_layout.addWidget(info_group)
        
        # Backup list section
        list_group = QGroupBox("Available Backups (Newest First)")
        list_layout = QVBoxLayout()
        list_layout.setContentsMargins(10, 10, 10, 10)
        
        # Create QTreeWidget for better display
        tree = QTreeWidget()
        tree.setColumnCount(3)
        tree.setHeaderLabels(['Date & Time', 'Backup File', 'Size'])
        tree.setColumnWidth(0, 200)
        tree.setColumnWidth(1, 300)
        tree.setColumnWidth(2, 100)
        tree.setMinimumHeight(300)
        tree.setSelectionMode(QTreeWidget.SingleSelection)
        tree.setAlternatingRowColors(True)
        
        # Populate tree with backup information
        backup_items = []
        for backup in backups:
            backup_path = os.path.join(backup_dir, backup)
            
            # Extract timestamp from filename
            try:
                timestamp_part = backup.replace("config_", "").replace(".json.bak", "")
                formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", 
                                              time.strptime(timestamp_part, "%Y%m%d_%H%M%S"))
            except:
                formatted_time = "Unknown"
            
            # Get file size
            try:
                size_bytes = os.path.getsize(backup_path)
                if size_bytes < 1024:
                    size_str = f"{size_bytes} B"
                elif size_bytes < 1024 * 1024:
                    size_str = f"{size_bytes // 1024} KB"
                else:
                    size_str = f"{size_bytes // (1024 * 1024)} MB"
            except:
                size_str = "Unknown"
            
            # Insert into tree
            item = QTreeWidgetItem([formatted_time, backup, size_str])
            tree.addTopLevelItem(item)
            backup_items.append((item, backup, formatted_time))
        
        # Select first item by default
        if backup_items:
            tree.setCurrentItem(backup_items[0][0])
        
        list_layout.addWidget(tree)
        list_group.setLayout(list_layout)
        main_layout.addWidget(list_group)
        
        # Action buttons
        button_group = QGroupBox("Actions")
        button_main_layout = QVBoxLayout()
        button_main_layout.setContentsMargins(10, 10, 10, 10)
        button_main_layout.setSpacing(10)
        
        # Button row 1
        button_row1 = QHBoxLayout()
        button_row1.setSpacing(10)
        
        # Button row 2
        button_row2 = QHBoxLayout()
        button_row2.setSpacing(10)
        
        def get_selected_backup():
            """Get currently selected backup from tree"""
            current_item = tree.currentItem()
            if not current_item:
                return None
                
            for item, backup_filename, formatted_time in backup_items:
                if item == current_item:
                    return backup_filename, formatted_time
            return None
        
        def restore_selected():
            selected = get_selected_backup()
            if not selected:
                msg = QMessageBox(dialog)
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("No Selection")
                msg.setText("Please select a backup to restore.")
                msg.setWindowIcon(icon)
                msg.exec()
                return
            
            selected_backup, formatted_time = selected
            backup_path = os.path.join(backup_dir, selected_backup)
            
            # Confirm restore
            confirm = QMessageBox(dialog)
            confirm.setIcon(QMessageBox.Question)
            confirm.setWindowTitle("Confirm Restore")
            confirm.setText(f"This will replace your current configuration with the backup from:\n\n"
                          f"{formatted_time}\n{selected_backup}\n\n"
                          f"A backup of your current config will be created first.\n\n"
                          f"Are you sure you want to continue?")
            confirm.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            confirm.setDefaultButton(QMessageBox.No)
            confirm.setWindowIcon(icon)
            
            if confirm.exec() == QMessageBox.Yes:
                try:
                    # Create backup of current config before restore
                    self._backup_config_file()
                    
                    # Copy backup to config file
                    shutil.copy2(backup_path, CONFIG_FILE)
                    
                    msg = QMessageBox(dialog)
                    msg.setIcon(QMessageBox.Information)
                    msg.setWindowTitle("Restore Complete")
                    msg.setText(f"Configuration restored from: {selected_backup}\n\n"
                              f"Please restart the application for changes to take effect.")
                    msg.setWindowIcon(icon)
                    msg.exec()
                    dialog.close()
                    
                except Exception as e:
                    msg = QMessageBox(dialog)
                    msg.setIcon(QMessageBox.Critical)
                    msg.setWindowTitle("Restore Failed")
                    msg.setText(f"Failed to restore backup: {e}")
                    msg.setWindowIcon(icon)
                    msg.exec()
        
        def delete_selected():
            selected = get_selected_backup()
            if not selected:
                msg = QMessageBox(dialog)
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("No Selection")
                msg.setText("Please select a backup to delete.")
                msg.setWindowIcon(icon)
                msg.exec()
                return
            
            selected_backup, formatted_time = selected
            
            confirm = QMessageBox(dialog)
            confirm.setIcon(QMessageBox.Question)
            confirm.setWindowTitle("Confirm Delete")
            confirm.setText(f"Delete backup from {formatted_time}?\n\n{selected_backup}\n\n"
                          f"This action cannot be undone.")
            confirm.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            confirm.setDefaultButton(QMessageBox.No)
            confirm.setWindowIcon(icon)
            
            if confirm.exec() == QMessageBox.Yes:
                try:
                    os.remove(os.path.join(backup_dir, selected_backup))
                    
                    # Remove from tree
                    current_item = tree.currentItem()
                    if current_item:
                        index = tree.indexOfTopLevelItem(current_item)
                        tree.takeTopLevelItem(index)
                    
                    # Update backup items list
                    backup_items[:] = [(item, backup, time_str) 
                                     for item, backup, time_str in backup_items 
                                     if backup != selected_backup]
                    
                    msg = QMessageBox(dialog)
                    msg.setIcon(QMessageBox.Information)
                    msg.setWindowTitle("Deleted")
                    msg.setText("Backup deleted successfully.")
                    msg.setWindowIcon(icon)
                    msg.exec()
                except Exception as e:
                    msg = QMessageBox(dialog)
                    msg.setIcon(QMessageBox.Critical)
                    msg.setWindowTitle("Delete Failed")
                    msg.setText(f"Failed to delete backup: {e}")
                    msg.setWindowIcon(icon)
                    msg.exec()
        
        def create_new_backup():
            """Create a new manual backup"""
            try:
                self._backup_config_file()
                msg = QMessageBox(dialog)
                msg.setIcon(QMessageBox.Information)
                msg.setWindowTitle("Backup Created")
                msg.setText("New configuration backup created successfully!")
                msg.setWindowIcon(icon)
                msg.exec()
                # Refresh the dialog
                dialog.close()
                self._manual_restore_config()  # Reopen with updated list
            except Exception as e:
                msg = QMessageBox(dialog)
                msg.setIcon(QMessageBox.Critical)
                msg.setWindowTitle("Backup Failed")
                msg.setText(f"Failed to create backup: {e}")
                msg.setWindowIcon(icon)
                msg.exec()
        
        def open_backup_folder():
            """Open backup folder in file explorer"""
            self._open_backup_folder()
        
        # Primary action buttons (Row 1)
        restore_btn = QPushButton("✅ Restore Selected")
        restore_btn.setMinimumWidth(180)
        restore_btn.setMinimumHeight(35)
        restore_btn.clicked.connect(restore_selected)
        restore_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                padding: 8px 20px;
                font-size: 11pt;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        button_row1.addWidget(restore_btn)
        
        create_btn = QPushButton("💾 Create New Backup")
        create_btn.setMinimumWidth(180)
        create_btn.setMinimumHeight(35)
        create_btn.clicked.connect(create_new_backup)
        create_btn.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                padding: 8px 20px;
                font-size: 11pt;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
        """)
        button_row1.addWidget(create_btn)
        
        folder_btn = QPushButton("📁 Open Folder")
        folder_btn.setMinimumWidth(180)
        folder_btn.setMinimumHeight(35)
        folder_btn.clicked.connect(open_backup_folder)
        folder_btn.setStyleSheet("""
            QPushButton {
                background-color: #17a2b8;
                color: white;
                padding: 8px 20px;
                font-size: 11pt;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #117a8b;
            }
        """)
        button_row1.addWidget(folder_btn)
        button_row1.addStretch()
        
        # Secondary action buttons (Row 2)
        delete_btn = QPushButton("🗑️ Delete Selected")
        delete_btn.setMinimumWidth(180)
        delete_btn.setMinimumHeight(35)
        delete_btn.clicked.connect(delete_selected)
        delete_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                padding: 8px 20px;
                font-size: 11pt;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
        """)
        button_row2.addWidget(delete_btn)
        button_row2.addStretch()
        
        close_btn = QPushButton("❌ Close")
        close_btn.setMinimumWidth(120)
        close_btn.setMinimumHeight(35)
        close_btn.clicked.connect(dialog.close)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                padding: 8px 20px;
                font-size: 11pt;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
        """)
        button_row2.addWidget(close_btn)
        
        button_main_layout.addLayout(button_row1)
        button_main_layout.addLayout(button_row2)
        button_group.setLayout(button_main_layout)
        main_layout.addWidget(button_group)
        
        # Set dialog layout and show
        dialog.setLayout(main_layout)
        
        # Run dialog in separate thread to avoid GIL conflicts
        import threading
        def run_dialog():
            dialog.exec()
        
        thread = threading.Thread(target=run_dialog, daemon=True)
        thread.start()
        
        # Keep reference to prevent garbage collection
        self._backup_dialog = dialog
        
    except Exception as e:
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("Error")
        msg.setText(f"Failed to open backup restore dialog: {e}")
        if icon:
            msg.setWindowIcon(icon)
        msg.exec()
