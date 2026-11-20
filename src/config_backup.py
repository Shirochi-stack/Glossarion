"""Config Backup Management Methods for Glossarion

These methods handle automatic and manual config backup/restore functionality.
They are designed to be bound to the TranslatorGUI instance.
"""

import os
import sys
import time
import shutil
import json
import tempfile
try:
    import dpi_setup
    dpi_setup.configure()
except Exception:
    pass
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTreeWidget, QTreeWidgetItem, QMessageBox, QFrame, QGroupBox,
    QApplication
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QIcon, QFont
from qt_widget_helpers import BackupBrowserDialog, format_backup_size

# Config path and backup/restore core (GUI-free, shared with mobile)
from app_paths import config_file_path
from config_store import (
    backup_config_file,
    config_backup_dir,
    list_config_backups,
    load_config,
    restore_config_backup_file,
    restore_latest_backup,
)


def _backup_config_file(self):
    """Create backup of the existing config file before saving."""
    # Skips a missing config; failures are printed, never raised (config_store).
    backup_config_file(config_file_path())

def _restore_config_from_backup(self):
    """Attempt to restore config from the most recent backup."""
    try:
        # Copy the most recent backup over the config file (config_store)
        latest_backup = restore_latest_backup(config_file_path())
        if not latest_backup:
            return
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
            self.config = load_config(config_file_path())
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
        
        backup_dir = config_backup_dir(config_file_path())
        
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

def _restore_config_backup_file(self, backup_path):
    """Validate and atomically restore a backup before requesting a restart."""
    # Read first, then a safety backup of the current config, then an atomic
    # replace (config_store); raises and leaves the config intact on failure.
    restore_config_backup_file(
        config_file_path(), backup_path, safety_backup=self._backup_config_file,
    )
    # Modal messages process timers too; prevent pending saves from overwriting
    # the restored file while the user acknowledges the restart message.
    self._config_restore_pending = True
    self._restart_command = (
        [sys.executable, *sys.argv[1:]] if getattr(sys, 'frozen', False)
        else [sys.executable, *sys.argv]
    )


def _manual_restore_config(self):
    """Show dialog to manually select and restore a config backup."""
    try:
        existing = getattr(self, '_backup_dialog', None)
        if existing is not None and existing.isVisible():
            existing.raise_()
            existing.activateWindow()
            return
        # Ensure QApplication exists
        app = QApplication.instance()
        if not app:
            try:
                import dpi_setup
                dpi_setup.configure()
            except Exception:
                pass
            app = QApplication(sys.argv)
        
        backup_dir = config_backup_dir(config_file_path())
        
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
        
        # Get list of available backups (newest first, config_store)
        backups = [entry['name'] for entry in list_config_backups(config_file_path())]
        
        if not backups:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setWindowTitle("No Backups")
            msg_box.setText("No config backups found.")
            msg_box.setWindowIcon(icon)
            msg_box.exec()
            return
        
        # One row per backup for the backup browser (qt_widget_helpers.BackupBrowserDialog, the
        # window the Glossary Editor's Restore Backup also uses)
        rows = []
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
                size_str = format_backup_size(os.path.getsize(backup_path))
            except:
                size_str = "Unknown"

            rows.append({'cells': [formatted_time, backup, size_str], 'key': (backup, formatted_time)})

        def restore_selected(selected):
            selected_backup, formatted_time = selected
            backup_path = os.path.join(backup_dir, selected_backup)

            # Confirm restore
            if dialog.confirm("Confirm Restore",
                              f"This will replace your current configuration with the backup from:\n\n"
                              f"{formatted_time}\n{selected_backup}\n\n"
                              f"A backup of your current config will be created first.\n\n"
                              f"Are you sure you want to continue?"):
                try:
                    _restore_config_backup_file(self, backup_path)

                    dialog.message(QMessageBox.Information, "Restore Complete",
                                   f"Configuration restored from: {selected_backup}\n\n"
                                   f"The application will now restart for changes to take effect.")
                    dialog.close()

                    # Let Qt unwind this callback and run normal shutdown.
                    # The replacement is launched only after child cleanup.
                    QTimer.singleShot(0, self.close)

                except Exception as e:
                    dialog.message(QMessageBox.Critical, "Restore Failed", f"Failed to restore backup: {e}")

        def delete_selected(selected):
            """Delete the selected backup; True when it is gone (its row is then removed)."""
            selected_backup, formatted_time = selected

            if not dialog.confirm("Confirm Delete",
                                  f"Delete backup from {formatted_time}?\n\n{selected_backup}\n\n"
                                  f"This action cannot be undone."):
                return False
            try:
                os.remove(os.path.join(backup_dir, selected_backup))
            except Exception as e:
                dialog.message(QMessageBox.Critical, "Delete Failed", f"Failed to delete backup: {e}")
                return False
            dialog.message(QMessageBox.Information, "Deleted", "Backup deleted successfully.")
            return True

        def create_new_backup():
            """Create a new manual backup"""
            try:
                self._backup_config_file()
                dialog.message(QMessageBox.Information, "Backup Created",
                               "New configuration backup created successfully!")
                # Refresh the dialog
                dialog.close()
                self._manual_restore_config()  # Reopen with updated list
            except Exception as e:
                dialog.message(QMessageBox.Critical, "Backup Failed", f"Failed to create backup: {e}")

        def open_backup_folder():
            """Open backup folder in file explorer"""
            self._open_backup_folder()

        dialog = BackupBrowserDialog(
            self,
            window_title="Config Backup Manager",
            heading="Configuration Backup Manager",
            description="Select a backup to restore or manage your configuration backups.",
            location=backup_dir,
            columns=[('Date & Time', 200), ('Backup File', 300), ('Size', 100)],
            rows=rows,
            on_restore=restore_selected,
            on_create=create_new_backup,
            on_open_folder=open_backup_folder,
            on_delete=delete_selected,
            icon=icon,
        )
        dialog.setAttribute(Qt.WA_DeleteOnClose, True)
        dialog.finished.connect(lambda *_: setattr(self, '_backup_dialog', None))

        # All Qt widgets and their event loops must stay on the GUI thread.
        self._backup_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        
    except Exception as e:
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("Error")
        msg.setText(f"Failed to open backup restore dialog: {e}")
        if icon:
            msg.setWindowIcon(icon)
        msg.exec()
