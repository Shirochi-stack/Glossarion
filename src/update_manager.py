# update_manager.py - Auto-update functionality for Glossarion
import os
import sys
import json
import requests
import threading
import time
import re
from typing import Optional, Dict, Tuple, List
from packaging import version
import tkinter as tk
from tkinter import ttk, messagebox, font
import ttkbootstrap as tb
from datetime import datetime

class UpdateManager:
    """Handles automatic update checking and installation for Glossarion"""
    
    GITHUB_API_URL = "https://api.github.com/repos/Shirochi-stack/Glossarion/releases"
    GITHUB_LATEST_URL = "https://api.github.com/repos/Shirochi-stack/Glossarion/releases/latest"
    
    def __init__(self, main_gui, base_dir):
        self.main_gui = main_gui
        self.base_dir = base_dir
        self.update_available = False
        self.latest_release = None
        self.all_releases = []  # Store all fetched releases
        self.download_progress = 0
        self.is_downloading = False
        self._last_check_time = 0
        self._check_cache_duration = 3600  # Cache for 1 hour
        self.selected_asset = None  # Store selected asset for download
        
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
    
    def fetch_multiple_releases(self, count=10) -> List[Dict]:
        """Fetch multiple releases from GitHub
        
        Args:
            count: Number of releases to fetch
            
        Returns:
            List of release data dictionaries
        """
        try:
            headers = {
                'Accept': 'application/vnd.github.v3+json',
                'User-Agent': 'Glossarion-Updater'
            }
            
            # Fetch multiple releases
            response = requests.get(
                f"{self.GITHUB_API_URL}?per_page={count}", 
                headers=headers, 
                timeout=10
            )
            response.raise_for_status()
            
            releases = response.json()
            
            # Process each release's notes
            for release in releases:
                if 'body' in release and release['body']:
                    # Clean up but don't truncate for history viewing
                    body = release['body']
                    # Just clean up excessive newlines
                    body = re.sub(r'\n{3,}', '\n\n', body)
                    release['body'] = body
            
            return releases
            
        except Exception as e:
            print(f"Error fetching releases: {e}")
            return []
    
    def check_for_updates(self, silent=True, force_show=False) -> Tuple[bool, Optional[Dict]]:
        """Check GitHub for newer releases
        
        Args:
            silent: If True, don't show error messages
            force_show: If True, show the dialog even when up to date
            
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
            
            response = requests.get(self.GITHUB_LATEST_URL, headers=headers, timeout=10)
            response.raise_for_status()
            
            release_data = response.json()
            latest_version = release_data['tag_name'].lstrip('v')
            
            # Fetch all releases for history regardless
            self.all_releases = self.fetch_multiple_releases(count=10)
            self.latest_release = release_data
            
            # Check if this version was skipped by user
            if release_data['tag_name'] in skipped_versions and not force_show:
                return False, None
            
            # Compare versions
            if version.parse(latest_version) > version.parse(self.CURRENT_VERSION):
                self.update_available = True
                
                # Show update dialog when update is available
                print(f"[DEBUG] Showing update dialog for version {latest_version}")
                self.main_gui.master.after(100, self.show_update_dialog)
                    
                return True, release_data
            else:
                # We're up to date
                self.update_available = False
                
                # Show dialog if explicitly requested (from menu)
                if force_show or not silent:
                    self.main_gui.master.after(100, self.show_update_dialog)
                
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
    
    def check_for_updates_manual(self):
        """Manual update check from menu - always shows dialog"""
        return self.check_for_updates(silent=False, force_show=True)
    
    def format_markdown_to_tkinter(self, text_widget, markdown_text):
        """Convert GitHub markdown to formatted tkinter text - simplified version
        
        Args:
            text_widget: The Text widget to insert formatted text into
            markdown_text: The markdown source text
        """
        # Configure minimal tags
        text_widget.tag_config("heading", font=('TkDefaultFont', 12, 'bold'))
        text_widget.tag_config("bold", font=('TkDefaultFont', 10, 'bold'))
        
        # Process text line by line with minimal formatting
        lines = markdown_text.split('\n')
        
        for line in lines:
            # Strip any weird unicode characters that might cause display issues
            line = ''.join(char for char in line if ord(char) < 65536)
            
            # Handle headings
            if line.startswith('#'):
                # Remove all # symbols and get the heading text
                heading_text = line.lstrip('#').strip()
                if heading_text:
                    text_widget.insert('end', heading_text + '\n', 'heading')
            
            # Handle bullet points
            elif line.strip().startswith(('- ', '* ')):
                # Get the text after the bullet
                bullet_text = line.strip()[2:].strip()
                # Clean the text of markdown formatting
                bullet_text = self._clean_markdown_text(bullet_text)
                text_widget.insert('end', '    • ' + bullet_text + '\n')
            
            # Handle numbered lists  
            elif re.match(r'^\s*\d+\.\s', line):
                # Extract number and text
                match = re.match(r'^(\s*)(\d+)\.\s(.+)', line)
                if match:
                    indent, num, text = match.groups()
                    clean_text = self._clean_markdown_text(text.strip())
                    text_widget.insert('end', f'    {num}. {clean_text}\n')
            
            # Handle separator lines
            elif line.strip() in ['---', '***', '___']:
                text_widget.insert('end', '─' * 40 + '\n')
            
            # Handle code blocks - just skip the markers
            elif line.strip().startswith('```'):
                continue  # Skip code fence markers
            
            # Regular text
            elif line.strip():
                # Clean and insert the line
                clean_text = self._clean_markdown_text(line)
                # Check if this looks like it should be bold (common pattern)
                if clean_text.endswith(':') and len(clean_text) < 50:
                    text_widget.insert('end', clean_text + '\n', 'bold')
                else:
                    text_widget.insert('end', clean_text + '\n')
            
            # Empty lines
            else:
                text_widget.insert('end', '\n')
    
    def _clean_markdown_text(self, text):
        """Remove markdown formatting from text
        
        Args:
            text: Text with markdown formatting
            
        Returns:
            Clean text without markdown symbols
        """
        # Remove inline code backticks
        text = re.sub(r'`([^`]+)`', r'\1', text)
        
        # Remove bold markers
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)
        
        # Remove italic markers
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'_([^_]+)_', r'\1', text)
        
        # Remove links but keep link text
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        
        # Remove any remaining special characters that might cause issues
        text = text.replace('\u200b', '')  # Remove zero-width spaces
        text = text.replace('\ufeff', '')  # Remove BOM
        
        return text.strip()
    
    def show_update_dialog(self):
        """Show update dialog (for updates or version history)"""
        if not self.latest_release and not self.all_releases:
            # Try to fetch releases if we don't have them
            self.all_releases = self.fetch_multiple_releases(count=10)
            if self.all_releases:
                self.latest_release = self.all_releases[0]
            else:
                messagebox.showerror("Error", "Unable to fetch version information from GitHub.")
                return
        
        # Set appropriate title
        if self.update_available:
            title = "Update Available"
        else:
            title = "Version History"
        
        # Create dialog first without content
        dialog, scrollable_frame, canvas = self.main_gui.wm.setup_scrollable(
            self.main_gui.master,
            title,
            width=None,
            height=None,
            max_width_ratio=0.5,
            max_height_ratio=0.8
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
        
        # Initialize selected_asset to None
        self.selected_asset = None
        
        # Version info
        version_frame = ttk.LabelFrame(main_frame, text="Version Information", padding=10)
        version_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Label(version_frame, 
                 text=f"Current Version: {self.CURRENT_VERSION}").pack(anchor='w')
        
        if self.latest_release:
            latest_version = self.latest_release['tag_name']
            if self.update_available:
                ttk.Label(version_frame, 
                         text=f"Latest Version: {latest_version}",
                         font=('TkDefaultFont', 10, 'bold')).pack(anchor='w')
            else:
                ttk.Label(version_frame, 
                         text=f"Latest Version: {latest_version} ✓ You are up to date!",
                         foreground='green',
                         font=('TkDefaultFont', 10, 'bold')).pack(anchor='w')
        
        # ALWAYS show asset selection when we have the first release data (current or latest)
        release_to_check = self.all_releases[0] if self.all_releases else self.latest_release
        
        if release_to_check:
            # Get exe files from the first/latest release
            exe_assets = [a for a in release_to_check.get('assets', []) 
                         if a['name'].lower().endswith('.exe')]
            
            print(f"[DEBUG] Found {len(exe_assets)} exe files in release {release_to_check.get('tag_name')}")
            
            # Show selection UI if there are exe files
            if exe_assets:
                # Determine the title based on whether there are multiple variants
                if len(exe_assets) > 1:
                    frame_title = "Select Version to Download"
                else:
                    frame_title = "Available Download"
                
                asset_frame = ttk.LabelFrame(main_frame, text=frame_title, padding=10)
                asset_frame.pack(fill='x', pady=(0, 10))
                
                if len(exe_assets) > 1:
                    # Multiple exe files - show radio buttons to choose
                    self.asset_var = tk.StringVar()
                    for i, asset in enumerate(exe_assets):
                        filename = asset['name']
                        size_mb = asset['size'] / (1024 * 1024)
                        
                        # Try to identify variant type from filename
                        if 'full' in filename.lower():
                            variant_label = f"Full Version - {filename} ({size_mb:.1f} MB)"
                        else:
                            variant_label = f"Standard Version - {filename} ({size_mb:.1f} MB)"
                        
                        rb = ttk.Radiobutton(asset_frame, text=variant_label, 
                                            variable=self.asset_var, 
                                            value=str(i))
                        rb.pack(anchor='w', pady=2)
                        
                        # Select first option by default
                        if i == 0:
                            self.asset_var.set(str(i))
                            self.selected_asset = asset
                    
                    # Add listener for selection changes
                    def on_asset_change(*args):
                        idx = int(self.asset_var.get())
                        self.selected_asset = exe_assets[idx]
                    
                    self.asset_var.trace_add('write', on_asset_change)
                else:
                    # Only one exe file - just show it and set it as selected
                    self.selected_asset = exe_assets[0]
                    filename = exe_assets[0]['name']
                    size_mb = exe_assets[0]['size'] / (1024 * 1024)
                    ttk.Label(asset_frame, 
                             text=f"{filename} ({size_mb:.1f} MB)").pack(anchor='w')
        
        # Create notebook for version history
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill='both', expand=True, pady=(0, 10))
        
        # Add tabs for different versions
        if self.all_releases:
            for i, release in enumerate(self.all_releases[:5]):  # Show up to 5 versions
                version_tag = release['tag_name']
                version_num = version_tag.lstrip('v')
                is_current = version_num == self.CURRENT_VERSION
                is_latest = i == 0
                
                # Create tab label
                tab_label = version_tag
                if is_current and is_latest:
                    tab_label += " (Current)"
                elif is_current:
                    tab_label += " (Current)"
                elif is_latest:
                    tab_label += " (Latest)"
                
                # Create frame for this version
                tab_frame = ttk.Frame(notebook)
                notebook.add(tab_frame, text=tab_label)
                
                # Add release date
                if 'published_at' in release:
                    date_str = release['published_at'][:10]  # Get YYYY-MM-DD
                    date_label = ttk.Label(tab_frame, text=f"Released: {date_str}", 
                                         font=('TkDefaultFont', 9, 'italic'))
                    date_label.pack(anchor='w', padx=10, pady=(10, 5))
                
                # Create text widget for release notes
                text_frame = ttk.Frame(tab_frame)
                text_frame.pack(fill='both', expand=True, padx=10, pady=(0, 10))
                
                notes_text = tk.Text(text_frame, height=12, wrap='word', width=60)
                notes_scroll = ttk.Scrollbar(text_frame, command=notes_text.yview)
                notes_text.config(yscrollcommand=notes_scroll.set)
                
                notes_text.pack(side='left', fill='both', expand=True)
                notes_scroll.pack(side='right', fill='y')
                
                # Format and insert release notes with markdown support
                release_notes = release.get('body', 'No release notes available')
                self.format_markdown_to_tkinter(notes_text, release_notes)
                
                notes_text.config(state='disabled')  # Make read-only
                
                # Don't set background color as it causes rendering artifacts
        else:
            # Fallback to simple display if no releases fetched
            notes_frame = ttk.LabelFrame(main_frame, text="Release Notes", padding=10)
            notes_frame.pack(fill='both', expand=True, pady=(0, 10))
            
            notes_text = tk.Text(notes_frame, height=10, wrap='word')
            notes_scroll = ttk.Scrollbar(notes_frame, command=notes_text.yview)
            notes_text.config(yscrollcommand=notes_scroll.set)
            
            notes_text.pack(side='left', fill='both', expand=True)
            notes_scroll.pack(side='right', fill='y')
            
            if self.latest_release:
                release_notes = self.latest_release.get('body', 'No release notes available')
                self.format_markdown_to_tkinter(notes_text, release_notes)
            else:
                notes_text.insert('1.0', 'Unable to fetch release notes.')
            
            notes_text.config(state='disabled')
        
        # Download progress (initially hidden)
        self.progress_frame = ttk.Frame(main_frame)
        self.progress_label = ttk.Label(self.progress_frame, text="Downloading update...")
        self.progress_label.pack(anchor='w')
        self.progress_bar = ttk.Progressbar(self.progress_frame, mode='determinate', length=400)
        self.progress_bar.pack(fill='x', pady=5)
        
        # Add status label for download details
        self.status_label = ttk.Label(self.progress_frame, text="", font=('TkDefaultFont', 8))
        self.status_label.pack(anchor='w')
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x', pady=(10, 0))
        
        def start_download():
            if not self.selected_asset:
                messagebox.showerror("No File Selected", 
                                   "Please select a version to download.")
                return
                
            self.progress_frame.pack(fill='x', pady=(0, 10), before=button_frame)
            download_btn.config(state='disabled')
            if 'remind_btn' in locals():
                remind_btn.config(state='disabled')
            if 'skip_btn' in locals():
                skip_btn.config(state='disabled')
            if 'close_btn' in locals():
                close_btn.config(state='disabled')
            
            # Reset progress
            self.progress_bar['value'] = 0
            self.download_progress = 0
            
            # Start download in background thread
            thread = threading.Thread(target=self.download_update, 
                                    args=(dialog,), daemon=True)
            thread.start()
        
        # Always show download button if we have exe files
        has_exe_files = self.selected_asset is not None
        
        if self.update_available:
            # Show update-specific buttons
            download_btn = tb.Button(button_frame, text="Download Update", 
                                   command=start_download, bootstyle="success")
            download_btn.pack(side='left', padx=(0, 5))
            
            remind_btn = tb.Button(button_frame, text="Remind Me Later", 
                                 command=dialog.destroy, bootstyle="secondary")
            remind_btn.pack(side='left', padx=5)
            
            skip_btn = tb.Button(button_frame, text="Skip This Version", 
                               command=lambda: self.skip_version(dialog), 
                               bootstyle="link")
            skip_btn.pack(side='left', padx=5)
        elif has_exe_files:
            # We're up to date but have downloadable files
            # Check if there are multiple exe files
            release_to_check = self.all_releases[0] if self.all_releases else self.latest_release
            exe_count = 0
            if release_to_check:
                exe_count = len([a for a in release_to_check.get('assets', []) 
                               if a['name'].lower().endswith('.exe')])
            
            if exe_count > 1:
                # Multiple versions available
                download_btn = tb.Button(button_frame, text="Download Different Path", 
                                       command=start_download, bootstyle="info")
            else:
                # Single version available
                download_btn = tb.Button(button_frame, text="Re-download", 
                                       command=start_download, bootstyle="secondary")
            download_btn.pack(side='left', padx=(0, 5))
            
            close_btn = tb.Button(button_frame, text="Close", 
                                command=dialog.destroy, 
                                bootstyle="secondary")
            close_btn.pack(side='left', padx=(0, 5))
        else:
            # No downloadable files
            close_btn = tb.Button(button_frame, text="Close", 
                                command=dialog.destroy, 
                                bootstyle="primary")
            close_btn.pack(side='left', padx=(0, 5))
        
        # Add "View All Releases" link button
        def open_releases_page():
            import webbrowser
            webbrowser.open("https://github.com/Shirochi-stack/Glossarion/releases")
        
        tb.Button(button_frame, text="View All Releases", 
                 command=open_releases_page, 
                 bootstyle="link").pack(side='right', padx=5)
        
        # Auto-resize at the end
        dialog.after(100, lambda: self.main_gui.wm.auto_resize_dialog(dialog, canvas, max_width_ratio=0.5, max_height_ratio=0.8))
    
        # Handle window close
        dialog.protocol("WM_DELETE_WINDOW", lambda: [dialog._cleanup_scrolling(), dialog.destroy()])
    
    def skip_version(self, dialog):
        """Mark this version as skipped and close dialog"""
        if not self.latest_release:
            dialog.destroy()
            return
        
        # Get current skipped versions list
        if 'skipped_versions' not in self.main_gui.config:
            self.main_gui.config['skipped_versions'] = []
        
        # Add this version to skipped list
        version_tag = self.latest_release['tag_name']
        if version_tag not in self.main_gui.config['skipped_versions']:
            self.main_gui.config['skipped_versions'].append(version_tag)
        
        # Save config
        self.main_gui.save_config(show_message=False)
        
        # Close dialog
        dialog.destroy()
        
        # Show confirmation
        messagebox.showinfo("Version Skipped", 
                          f"Version {version_tag} will be skipped in future update checks.\n"
                          "You can manually check for updates from the Help menu.")
    
    def download_update(self, dialog):
        """Download the update file"""
        try:
            # Use the selected asset
            asset = self.selected_asset
                    
            if not asset:
                dialog.after(0, lambda: messagebox.showerror("Download Error", 
                                                           "No file selected for download."))
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
            chunk_size = 8192
            
            with open(download_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Update progress bar
                        if total_size > 0:
                            progress = int((downloaded / total_size) * 100)
                            size_mb = downloaded / (1024 * 1024)
                            total_mb = total_size / (1024 * 1024)
                            
                            # Use after_idle for smoother updates
                            def update_progress(p=progress, d=size_mb, t=total_mb):
                                try:
                                    self.progress_bar['value'] = p
                                    self.progress_label.config(text=f"Downloading update... {p}%")
                                    self.status_label.config(text=f"{d:.1f} MB / {t:.1f} MB")
                                except:
                                    pass  # Dialog might have been closed
                            
                            dialog.after_idle(update_progress)
            
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
