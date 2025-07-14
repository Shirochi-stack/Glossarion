# update_manager.py - Auto-update functionality for Glossarion
import os
import sys
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
        self._last_check_time = 0
        self._check_cache_duration = 3600  # Cache for 1 hour
        
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
            silent: If True, don't show "up to date" message, but STILL show update dialog
            
        Returns:
            Tuple of (update_available, release_info)
        """
        try:
            # Check if this version was previously skipped
            skipped_versions = self.main_gui.config.get('skipped_versions', [])
            
            headers = {
                'Accept': 'application/vnd.github.v3+json',
                'User-Agent': 'Glossarion-Updater'
            }
            
            response = requests.get(self.GITHUB_API_URL, headers=headers, timeout=10)
            response.raise_for_status()
            
            release_data = response.json()
            latest_version = release_data['tag_name'].lstrip('v')
            
            # Check if this version was skipped by user
            if release_data['tag_name'] in skipped_versions:
                return False, None
            
            # Compare versions
            if version.parse(latest_version) > version.parse(self.CURRENT_VERSION):
                self.update_available = True
                self.latest_release = release_data
                
                # Pre-process release notes to improve dialog loading speed
                if 'body' in release_data and release_data['body']:
                    # Limit release notes length and clean up markdown
                    max_length = 20000
                    body = release_data['body']
                    
                    # Remove excessive newlines
                    body = '\n'.join(line for line in body.split('\n') if line.strip())
                    
                    # Truncate if too long
                    if len(body) > max_length:
                        body = body[:max_length] + "\n\n... (see full notes on GitHub)"
                    
                    release_data['body'] = body
                
                # ALWAYS show update dialog when update is available
                print(f"[DEBUG] Showing update dialog for version {latest_version}")
                self.main_gui.master.after(100, self.show_update_dialog)
                    
                return True, release_data
            else:
                # We're up to date - only show message if not silent
                if not silent:
                    messagebox.showinfo("Update Check", 
                                      f"You are running the latest version ({self.CURRENT_VERSION})")
                return False, None
                
        except requests.Timeout:
            if not silent:
                messagebox.showerror("Update Check Failed", 
                                   "Connection timed out. Please check your internet connection.")
            return False, None
            
        except requests.ConnectionError:
            if not silent:
                messagebox.showerror("Update Check Failed", 
                                   "Could not connect to GitHub. Please check your internet connection.")
            return False, None
            
        except requests.HTTPError as e:
            if not silent:
                if e.response.status_code == 403:
                    messagebox.showerror("Update Check Failed", 
                                       "GitHub API rate limit exceeded. Please try again later.")
                else:
                    messagebox.showerror("Update Check Failed", 
                                       f"GitHub returned error: {e.response.status_code}")
            return False, None
            
        except ValueError as e:
            if not silent:
                messagebox.showerror("Update Check Failed", 
                                   "Invalid response from GitHub. The update service may be temporarily unavailable.")
            return False, None
            
        except Exception as e:
            if not silent:
                messagebox.showerror("Update Check Failed", 
                                   f"An unexpected error occurred:\n{str(e)}")
            return False, None
    
    def show_update_dialog(self):
        """Show update available dialog using WindowManager style"""
        if not self.latest_release:
            return
        
        # Create dialog first without content
        dialog, scrollable_frame, canvas = self.main_gui.wm.setup_scrollable(
            self.main_gui.master,
            "Update Available",
            width=600,
            height=400,
            max_width_ratio=0.4,
            max_height_ratio=0.5
        )
        
        # Show dialog immediately
        dialog.update_idletasks()
        
        # Then populate content
        self.main_gui.master.after(10, lambda: self._populate_update_dialog(dialog, scrollable_frame, canvas))

    def _populate_update_dialog(self, dialog, scrollable_frame, canvas):
        """Populate the update dialog content"""
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
        
        # Create text widget without using setup_scrollable_text (which might be slow)
        notes_text = tk.Text(notes_frame, height=10, wrap='word')
        notes_scroll = ttk.Scrollbar(notes_frame, command=notes_text.yview)
        notes_text.config(yscrollcommand=notes_scroll.set)
        
        notes_text.pack(side='left', fill='both', expand=True)
        notes_scroll.pack(side='right', fill='y')
        
        # Insert text in chunks to avoid blocking
        release_notes = self.latest_release.get('body', 'No release notes available')
        if len(release_notes) > 1000:
            # Insert first chunk
            notes_text.insert('1.0', release_notes[:1000])
            # Insert rest after a moment
            dialog.after(50, lambda: notes_text.insert('end', release_notes[1000:]))
        else:
            notes_text.insert('1.0', release_notes)
        
        notes_text.config(state='disabled')  # Make read-only
        
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
        
        # Auto-resize at the end
        dialog.after(100, lambda: self.main_gui.wm.auto_resize_dialog(dialog, canvas, max_width_ratio=0.4, max_height_ratio=0.62))
    
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
            
            # Get the current executable path
            if getattr(sys, 'frozen', False):
                # Running as compiled executable
                current_exe = sys.executable
                download_dir = os.path.dirname(current_exe)
            else:
                # Running as script
                current_exe = None
                download_dir = self.base_dir
            
            # Use the exact filename from GitHub
            original_filename = asset['name']  # e.g., "Glossarion v3.1.3.exe"
            new_exe_path = os.path.join(download_dir, original_filename)
            
            # If new file would overwrite current executable, download to temp name first
            if current_exe and os.path.normpath(new_exe_path) == os.path.normpath(current_exe):
                temp_path = new_exe_path + ".new"
                download_path = temp_path
            else:
                download_path = new_exe_path
            
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
            # Capture the error message immediately
            error_msg = str(e)
            dialog.after(0, lambda: messagebox.showerror("Download Failed", error_msg))
    
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
            self.main_gui.save_config(show_message=False)
            
            # Get current executable path
            if getattr(sys, 'frozen', False):
                current_exe = sys.executable
                current_dir = os.path.dirname(current_exe)
                
                # Create a batch file to handle the update
                batch_content = f"""@echo off
    echo Updating Glossarion...
    echo Waiting for current version to close...
    timeout /t 3 /nobreak > nul

    :: Delete the old executable
    echo Deleting old version...
    if exist "{current_exe}" (
        del /f /q "{current_exe}"
        if exist "{current_exe}" (
            echo Failed to delete old version, retrying...
            timeout /t 2 /nobreak > nul
            del /f /q "{current_exe}"
        )
    )

    :: Start the new version
    echo Starting new version...
    start "" "{update_file}"

    :: Clean up this batch file
    del "%~f0"
    """
                batch_path = os.path.join(current_dir, "update_glossarion.bat")
                with open(batch_path, 'w') as f:
                    f.write(batch_content)
                
                # Run the batch file
                import subprocess
                subprocess.Popen([batch_path], shell=True, creationflags=subprocess.CREATE_NO_WINDOW)
                
                print(f"[DEBUG] Update batch file created: {batch_path}")
                print(f"[DEBUG] Will delete: {current_exe}")
                print(f"[DEBUG] Will start: {update_file}")
            else:
                # Running as script, just start the new exe
                import subprocess
                subprocess.Popen([update_file], shell=True)
            
            # Exit current application
            print("[DEBUG] Closing application for update...")
            self.main_gui.master.quit()
            sys.exit(0)
            
        except Exception as e:
            messagebox.showerror("Installation Error", 
                               f"Could not start update process:\n{str(e)}")
