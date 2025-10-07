"""Config Backup Management Methods for Glossarion

These methods handle automatic and manual config backup/restore functionality.
They are designed to be bound to the TranslatorGUI instance.
"""

import os
import sys
import time
import shutil
import json
import tkinter as tk
from tkinter import messagebox, ttk
import ttkbootstrap as tb

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
        
        # Maintain only the last 10 backups
        backups = [os.path.join(backup_dir, f) for f in os.listdir(backup_dir) 
                   if f.startswith("config_") and f.endswith(".json.bak")]
        backups.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        # Remove oldest backups if more than 10
        for old_backup in backups[10:]:
            try:
                os.remove(old_backup)
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
        QMessageBox.information(None, "Config Restored", 
                          f"Configuration was restored from backup: {os.path.basename(latest_backup)}")
        
        # Reload config
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
                self.config = decrypt_config(self.config)
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Failed to reload configuration: {e}")
            
    except Exception as e:
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.critical(None, "Restore Failed", f"Could not restore config from backup: {e}")
        
def _create_manual_config_backup(self):
    """Create a manual config backup."""
    try:
        # Force create backup even if config file doesn't exist
        self._backup_config_file()
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(None, "Backup Created", "Configuration backup created successfully!")
    except Exception as e:
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.critical(None, "Backup Failed", f"Failed to create backup: {e}")

def _open_backup_folder(self):
    """Open the config backups folder in file explorer."""
    try:
        if os.path.isabs(CONFIG_FILE):
            config_dir = os.path.dirname(CONFIG_FILE)
        else:
            config_dir = os.path.dirname(os.path.abspath(CONFIG_FILE))
        backup_dir = os.path.join(config_dir, "config_backups")
        
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir, exist_ok=True)
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(None, "Backup Folder", f"Created backup folder: {backup_dir}")
        
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
        QMessageBox.critical(None, "Error", f"Could not open backup folder: {e}")

def _manual_restore_config(self):
    """Show dialog to manually select and restore a config backup."""
    try:
        if os.path.isabs(CONFIG_FILE):
            config_dir = os.path.dirname(CONFIG_FILE)
        else:
            config_dir = os.path.dirname(os.path.abspath(CONFIG_FILE))
        backup_dir = os.path.join(config_dir, "config_backups")
        
        if not os.path.exists(backup_dir):
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(None, "No Backups", "No backup folder found. No backups have been created yet.")
            return
        
        # Get list of available backups
        backups = [f for f in os.listdir(backup_dir) 
                  if f.startswith("config_") and f.endswith(".json.bak")]
        
        if not backups:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(None, "No Backups", "No config backups found.")
            return
        
        # Sort by creation time (newest first)
        backups.sort(key=lambda x: os.path.getmtime(os.path.join(backup_dir, x)), reverse=True)
        
        # Use WindowManager to create scrollable dialog
        dialog, scrollable_frame, canvas = self.wm.setup_scrollable(
            self.master,
            "Config Backup Manager",
            width=0,
            height=None,
            max_width_ratio=0.6,
            max_height_ratio=0.8
        )
        
        # Main content
        header_frame = tk.Frame(scrollable_frame)
        header_frame.pack(fill=tk.X, padx=20, pady=(20, 10))
        
        tk.Label(header_frame, text="Configuration Backup Manager", 
                font=('TkDefaultFont', 14, 'bold')).pack(anchor=tk.W)
        
        tk.Label(header_frame, 
                text="Select a backup to restore or manage your configuration backups.",
                font=('TkDefaultFont', 10), fg='gray').pack(anchor=tk.W, pady=(5, 0))
        
        # Info section
        info_frame = tk.LabelFrame(scrollable_frame, text="Backup Information", padx=10, pady=10)
        info_frame.pack(fill=tk.X, padx=20, pady=(0, 10))
        
        info_text = f"📁 Backup Location: {backup_dir}\n📊 Total Backups: {len(backups)}"
        tk.Label(info_frame, text=info_text, font=('TkDefaultFont', 10), 
                fg='#333', justify=tk.LEFT).pack(anchor=tk.W)
        
        # Backup list section
        list_frame = tk.LabelFrame(scrollable_frame, text="Available Backups (Newest First)", padx=10, pady=10)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 10))
        
        # Create treeview for better display
        columns = ('timestamp', 'filename', 'size')
        tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=8)
        
        # Define headings
        tree.heading('timestamp', text='Date & Time')
        tree.heading('filename', text='Backup File')
        tree.heading('size', text='Size')
        
        # Configure column widths
        tree.column('timestamp', width=150, anchor='center')
        tree.column('filename', width=200)
        tree.column('size', width=80, anchor='center')
        
        # Add scrollbars for treeview
        v_scrollbar = ttk.Scrollbar(list_frame, orient='vertical', command=tree.yview)
        h_scrollbar = ttk.Scrollbar(list_frame, orient='horizontal', command=tree.xview)
        tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack treeview and scrollbars
        tree.pack(side='left', fill='both', expand=True)
        v_scrollbar.pack(side='right', fill='y')
        h_scrollbar.pack(side='bottom', fill='x')
        
        # Populate treeview with backup information
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
            
            # Insert into treeview
            item_id = tree.insert('', 'end', values=(formatted_time, backup, size_str))
            backup_items.append((item_id, backup, formatted_time))
        
        # Select first item by default
        if backup_items:
            tree.selection_set(backup_items[0][0])
            tree.focus(backup_items[0][0])
        
        # Action buttons frame
        button_frame = tk.LabelFrame(scrollable_frame, text="Actions", padx=10, pady=10)
        button_frame.pack(fill=tk.X, padx=20, pady=(0, 10))
        
        # Create button layout
        button_row1 = tk.Frame(button_frame)
        button_row1.pack(fill=tk.X, pady=(0, 5))
        
        button_row2 = tk.Frame(button_frame)
        button_row2.pack(fill=tk.X)
        
        def get_selected_backup():
            """Get currently selected backup from treeview"""
            selection = tree.selection()
            if not selection:
                return None
                
            selected_item = selection[0]
            for item_id, backup_filename, formatted_time in backup_items:
                if item_id == selected_item:
                    return backup_filename, formatted_time
            return None
        
        def restore_selected():
            selected = get_selected_backup()
            if not selected:
                messagebox.showwarning("No Selection", "Please select a backup to restore.")
                return
            
            selected_backup, formatted_time = selected
            backup_path = os.path.join(backup_dir, selected_backup)
            
            # Confirm restore
            if messagebox.askyesno("Confirm Restore", 
                                 f"This will replace your current configuration with the backup from:\n\n"
                                 f"{formatted_time}\n{selected_backup}\n\n"
                                 f"A backup of your current config will be created first.\n\n"
                                 f"Are you sure you want to continue?"):
                
                try:
                    # Create backup of current config before restore
                    self._backup_config_file()
                    
                    # Copy backup to config file
                    shutil.copy2(backup_path, CONFIG_FILE)
                    
                    messagebox.showinfo("Restore Complete", 
                                      f"Configuration restored from: {selected_backup}\n\n"
                                      f"Please restart the application for changes to take effect.")
                    dialog._cleanup_scrolling()
                    dialog.destroy()
                    
                except Exception as e:
                    messagebox.showerror("Restore Failed", f"Failed to restore backup: {e}")
        
        def delete_selected():
            selected = get_selected_backup()
            if not selected:
                messagebox.showwarning("No Selection", "Please select a backup to delete.")
                return
            
            selected_backup, formatted_time = selected
            
            if messagebox.askyesno("Confirm Delete", 
                                 f"Delete backup from {formatted_time}?\n\n{selected_backup}\n\n"
                                 f"This action cannot be undone."):
                try:
                    os.remove(os.path.join(backup_dir, selected_backup))
                    
                    # Remove from treeview
                    selection = tree.selection()
                    if selection:
                        tree.delete(selection[0])
                    
                    # Update backup items list
                    backup_items[:] = [(item_id, backup, time_str) 
                                     for item_id, backup, time_str in backup_items 
                                     if backup != selected_backup]
                    
                    messagebox.showinfo("Deleted", "Backup deleted successfully.")
                except Exception as e:
                    messagebox.showerror("Delete Failed", f"Failed to delete backup: {e}")
        
        def create_new_backup():
            """Create a new manual backup"""
            try:
                self._backup_config_file()
                messagebox.showinfo("Backup Created", "New configuration backup created successfully!")
                # Refresh the dialog
                dialog._cleanup_scrolling()
                dialog.destroy()
                self._manual_restore_config()  # Reopen with updated list
            except Exception as e:
                messagebox.showerror("Backup Failed", f"Failed to create backup: {e}")
        
        def open_backup_folder():
            """Open backup folder in file explorer"""
            self._open_backup_folder()
        
        # Primary action buttons (Row 1)
        tb.Button(button_row1, text="✅ Restore Selected", 
                 command=restore_selected, bootstyle="success", 
                 width=20).pack(side=tk.LEFT, padx=(0, 10))
                 
        tb.Button(button_row1, text="💾 Create New Backup", 
                 command=create_new_backup, bootstyle="primary-outline", 
                 width=20).pack(side=tk.LEFT, padx=(0, 10))
                 
        tb.Button(button_row1, text="📁 Open Folder", 
                 command=open_backup_folder, bootstyle="info-outline", 
                 width=20).pack(side=tk.LEFT)
        
        # Secondary action buttons (Row 2)
        tb.Button(button_row2, text="🗑️ Delete Selected", 
                 command=delete_selected, bootstyle="danger-outline", 
                 width=20).pack(side=tk.LEFT, padx=(0, 10))
                 
        tb.Button(button_row2, text="❌ Close", 
                 command=lambda: [dialog._cleanup_scrolling(), dialog.destroy()], 
                 bootstyle="secondary", 
                 width=20).pack(side=tk.RIGHT)
        
        # Auto-resize and show dialog
        self.wm.auto_resize_dialog(dialog, canvas, max_width_ratio=0.7, max_height_ratio=0.9)
        
        # Handle window close
        dialog.protocol("WM_DELETE_WINDOW", lambda: [dialog._cleanup_scrolling(), dialog.destroy()])
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to open backup restore dialog: {e}")