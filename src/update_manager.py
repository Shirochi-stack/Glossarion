# update_manager.py - Auto-update functionality for Glossarion
import os
import json
import requests
import threading
import time
from typing import Optional, Dict, Tuple
from packaging import version
import tkinter as tk
from tkinter import ttk, messagebox
import ttkbootstrap as tb
from datetime import datetime

class UpdateManager:
    """Handles automatic update checking and installation for Glossarion"""
    
    GITHUB_API_URL = "https://api.github.com/repos/Shirochi-stack/Glossarion/releases/latest"
    
    def __init__(self, main_gui, base_dir):
        self.main_gui = main_gui
        self.base_dir = base_dir
        self.update_available = False
        self.latest_release = None
        self.download_progress = 0
        self.is_downloading = False
        
        # Get version from the main GUI's __version__ variable
        if hasattr(main_gui, '__version__'):
            self.CURRENT_VERSION = main_gui.__version__
        else:
            # Extract from window title as fallback
            title = self.main_gui.master.title()
            if 'v' in title:
                self.CURRENT_VERSION = title.split('v')[-1].strip()
            else:
                self.CURRENT_VERSION = "0.0.0"
        
    def check_for_updates(self, silent=True) -> Tuple[bool, Optional[Dict]]:
        """Check GitHub for newer releases
        
        Args:
            silent: If True, only show dialog if update available
            
        Returns:
            Tuple of (update_available, release_info)
        """
        try:
            headers = {
                'Accept': 'application/vnd.github.v3+json',
                'User-Agent': 'Glossarion-Updater'
            }
            
            response = requests.get(self.GITHUB_API_URL, headers=headers, timeout=10)
            response.raise_for_status()
            
            release_data = response.json()
            latest_version = release_data['tag_name'].lstrip('v')
            
            # Compare versions
            if version.parse(latest_version) > version.parse(self.CURRENT_VERSION):
                self.update_available = True
                self.latest_release = release_data
                
                if not silent:
                    self.show_update_dialog()
                    
                return True, release_data
            else:
                if not silent:
                    messagebox.showinfo("Update Check", 
                                      f"You are running the latest version ({self.CURRENT_VERSION})")
                return False, None
                
        except requests.RequestException as e:
            if not silent:
                messagebox.showerror("Update Check Failed", 
                                   f"Could not check for updates:\n{str(e)}")
            return False, None
    
    def show_update_dialog(self):
        """Show update available dialog using WindowManager style"""
        if not self.latest_release:
            return
            
        # Create dialog using WindowManager
        dialog, scrollable_frame, canvas = self.main_gui.wm.setup_scrollable(
            self.main_gui.master,
            "Update Available",
            width=600,
            height=400,
            max_width_ratio=0.4,  # 40% of screen width
            max_height_ratio=0.5  # 50% of screen height
        )
        
        # Main container
        main_frame = ttk.Frame(scrollable_frame)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Version info
        version_frame = ttk.LabelFrame(main_frame, text="Version Information", padding=10)
        version_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Label(version_frame, 
                 text=f"Current Version: {self.CURRENT_VERSION}").pack(anchor='w')
        ttk.Label(version_frame, 
                 text=f"Latest Version: {self.latest_release['tag_name']}",
                 font=('TkDefaultFont', 10, 'bold')).pack(anchor='w')
        
        # Release notes
        notes_frame = ttk.LabelFrame(main_frame, text="What's New", padding=10)
        notes_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        # Use UIHelper for scrollable text
        notes_text = self.main_gui.ui.setup_scrollable_text(
            notes_frame, 
            height=10, 
            wrap='word'
        )
        notes_text.pack(fill='both', expand=True)
        notes_text.insert('1.0', self.latest_release.get('body', 'No release notes available'))
        self.main_gui.ui.block_text_editing(notes_text)
        
        # Download progress (initially hidden)
        self.progress_frame = ttk.Frame(main_frame)
        self.progress_label = ttk.Label(self.progress_frame, text="Downloading update...")
        self.progress_label.pack(anchor='w')
        self.progress_bar = ttk.Progressbar(self.progress_frame, mode='determinate')
        self.progress_bar.pack(fill='x', pady=5)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x', pady=(10, 0))
        
        def start_download():
            self.progress_frame.pack(fill='x', pady=(0, 10))
            download_btn.config(state='disabled')
            remind_btn.config(state='disabled')
            
            # Start download in background thread
            thread = threading.Thread(target=self.download_update, 
                                    args=(dialog,), daemon=True)
            thread.start()
        
        download_btn = tb.Button(button_frame, text="Download Update", 
                               command=start_download, bootstyle="success")
        download_btn.pack(side='left', padx=(0, 5))
        
        remind_btn = tb.Button(button_frame, text="Remind Me Later", 
                             command=dialog.destroy, bootstyle="secondary")
        remind_btn.pack(side='left', padx=5)
        
        tb.Button(button_frame, text="Skip This Version", 
                 command=lambda: self.skip_version(dialog), 
                 bootstyle="link").pack(side='left', padx=5)
        
        # Auto-resize and show
        self.main_gui.wm.auto_resize_dialog(dialog, canvas, max_width_ratio=0.4, max_height_ratio=0.62)
        
        # Handle window close
        dialog.protocol("WM_DELETE_WINDOW", lambda: [dialog._cleanup_scrolling(), dialog.destroy()])
    
    def download_update(self, dialog):
        """Download the update file"""
        try:
            # Find the appropriate asset (Windows .exe file)
            asset = None
            for a in self.latest_release['assets']:
                if a['name'].endswith('.exe'):
                    asset = a
                    break
                    
            if not asset:
                dialog.after(0, lambda: messagebox.showerror("Download Error", 
                                                           "No Windows executable found in release"))
                return
            
            # Use the exact filename from GitHub
            original_filename = asset['name']  # e.g., "Glossarion v3.1.1.exe"
            download_path = os.path.join(self.base_dir, original_filename)
            
            # Check if old version exists and delete it
            for file in os.listdir(self.base_dir):
                if file.endswith('.exe') and file.startswith('Glossarion') and file != original_filename:
                    try:
                        old_file = os.path.join(self.base_dir, file)
                        os.remove(old_file)
                        print(f"Deleted old version: {file}")
                    except Exception as e:
                        print(f"Could not delete old version {file}: {e}")
            
            # Download with progress tracking
            response = requests.get(asset['browser_download_url'], stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            downloaded = 0
            with open(download_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Update progress bar
                        if total_size > 0:
                            progress = int((downloaded / total_size) * 100)
                            dialog.after(0, lambda p=progress: self.progress_bar.config(value=p))
            
            # Download complete
            dialog.after(0, lambda: self.download_complete(dialog, download_path))
            
        except Exception as e:
            dialog.after(0, lambda: messagebox.showerror("Download Failed", str(e)))
            
        except Exception as e:
            dialog.after(0, lambda: messagebox.showerror("Download Failed", str(e)))
    
    def download_complete(self, dialog, file_path):
        """Handle completed download"""
        dialog.destroy()
        
        result = messagebox.askyesno(
            "Download Complete",
            "Update downloaded successfully.\n\n"
            "Would you like to install it now?\n"
            "(The application will need to restart)"
        )
        
        if result:
            self.install_update(file_path)
    
    def install_update(self, update_file):
        """Launch the update installer and exit current app"""
        try:
            # Save current state/config if needed
            self.main_gui.save_config()
            
            # Launch the installer
            import subprocess
            subprocess.Popen([update_file], shell=True)
            
            # Exit current application
            self.main_gui.on_close()  # Changed from on_closing to on_close
            
        except Exception as e:
            messagebox.showerror("Installation Error", 
                               f"Could not start installer:\n{str(e)}")
