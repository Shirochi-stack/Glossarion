# update_manager.py - Auto-update functionality for Glossarion
import os
import sys
import json
import requests
import threading
import concurrent.futures
import time
import re
from typing import Optional, Dict, Tuple, List
from packaging import version
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QRadioButton, QButtonGroup, QGroupBox, QTabWidget, QWidget,
    QTextEdit, QProgressBar, QMessageBox, QApplication
)
from PySide6.QtCore import Qt, QTimer, QObject, QThread, Signal
from PySide6.QtGui import QFont, QIcon, QTextCursor
from datetime import datetime

class UpdateCheckWorker(QThread):
    """Worker thread for checking updates in background"""
    update_checked = Signal(bool, object)  # (update_available, release_data)
    error_occurred = Signal(str)  # error message
    
    def __init__(self, update_manager, silent=True, force_show=False):
        super().__init__()
        self.update_manager = update_manager
        self.silent = silent
        self.force_show = force_show
    
    def run(self):
        """Run update check in background thread"""
        try:
            print("[DEBUG] Worker thread starting update check...")
            result = self.update_manager._check_for_updates_internal(self.silent, self.force_show)
            self.update_checked.emit(*result)
        except Exception as e:
            print(f"[DEBUG] Worker thread error: {e}")
            self.error_occurred.emit(str(e))

class UpdateManager(QObject):
    """Handles automatic update checking and installation for Glossarion"""
    
    GITHUB_API_URL = "https://api.github.com/repos/Shirochi-stack/Glossarion/releases"
    GITHUB_LATEST_URL = "https://api.github.com/repos/Shirochi-stack/Glossarion/releases/latest"
    
    def __init__(self, main_gui, base_dir):
        super().__init__()
        self.main_gui = main_gui
        self.dialog = main_gui  # Set dialog as the main GUI window for message box parent
        self.base_dir = base_dir
        self.update_available = False
        self._check_in_progress = False  # Prevent concurrent checks
        # Use shared executor from main GUI if available
        try:
            if hasattr(self.main_gui, '_ensure_executor'):
                self.main_gui._ensure_executor()
            self.executor = getattr(self.main_gui, 'executor', None)
        except Exception:
            self.executor = None
        self.latest_release = None
        self.all_releases = []  # Store all fetched releases
        self.download_progress = 0
        self.is_downloading = False
        # Load persistent check time from config
        self._last_check_time = self.main_gui.config.get('last_update_check_time', 0)
        self._check_cache_duration = 1800  # Cache for 30 minutes
        self.selected_asset = None  # Store selected asset for download
        
        # Get version from the main GUI's __version__ variable
        if hasattr(main_gui, '__version__'):
            self.CURRENT_VERSION = main_gui.__version__
        else:
            # Extract from window title as fallback
            title = self.main_gui.windowTitle()
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
            
            # Fetch multiple releases with minimal retry logic
            max_retries = 1  # Reduced to prevent hanging
            timeout = 5  # Very short timeout
            
            for attempt in range(max_retries + 1):
                try:
                    response = requests.get(
                        f"{self.GITHUB_API_URL}?per_page={count}", 
                        headers=headers, 
                        timeout=timeout
                    )
                    response.raise_for_status()
                    break  # Success
                except (requests.Timeout, requests.ConnectionError) as e:
                    if attempt == max_retries:
                        raise  # Re-raise after final attempt
                    time.sleep(1)
            
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
    
    def check_for_updates_async(self, silent=True, force_show=False):
        """Run check_for_updates in background using QThread (PySide6 compatible).
        """
        print("[DEBUG] Starting background update check with QThread")
        
        # Prevent concurrent update checks
        if self._check_in_progress:
            print("[DEBUG] Update check already in progress, skipping...")
            return None
            
        self._check_in_progress = True
        
        # Show loading dialog for manual checks (when not silent)
        if not silent:
            self._show_loading_dialog()
        
        # Create and start worker thread
        self.worker = UpdateCheckWorker(self, silent, force_show)
        self.worker.update_checked.connect(self._on_update_checked)
        self.worker.error_occurred.connect(self._on_update_error)
        self.worker.finished.connect(self._on_update_finished)  # Clean up flag
        self.worker.start()
        
        return None  # Async, results will come via signals
    
    def _on_update_checked(self, update_available, release_data):
        """Handle update check results from worker thread"""
        print(f"[DEBUG] Update check completed: available={update_available}")
        if update_available or self.worker.force_show:
            self.show_update_dialog()
    
    def _on_update_error(self, error_msg):
        """Handle update check error from worker thread"""
        print(f"[DEBUG] Update check error: {error_msg}")
        # Close loading dialog first
        self._close_loading_dialog()
        if not self.worker.silent:
            msg = QMessageBox(self.dialog if hasattr(self.dialog, 'show') else None)
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Update Check Failed")
            msg.setText(f"Failed to check for updates: {error_msg}")
            msg.exec()
    
    def _on_update_finished(self):
        """Clean up after update check is finished"""
        print("[DEBUG] Update check finished, resetting progress flag")
        self._check_in_progress = False
        # Close loading dialog if it exists
        self._close_loading_dialog()

    def _check_for_updates_internal(self, silent=True, force_show=False) -> Tuple[bool, Optional[Dict]]:
        """Check GitHub for newer releases
        
        Args:
            silent: If True, don't show error messages
            force_show: If True, show the dialog even when up to date
            
        Returns:
            Tuple of (update_available, release_info)
        """
        print("[DEBUG] _check_for_updates_internal called")
        try:
            # Check if we need to skip the check due to cache
            current_time = time.time()
            if not force_show and (current_time - self._last_check_time) < self._check_cache_duration:
                print(f"[DEBUG] Skipping update check - cache still valid for {int(self._check_cache_duration - (current_time - self._last_check_time))} seconds")
                return False, None
            
            # Check if this version was previously skipped
            skipped_versions = self.main_gui.config.get('skipped_versions', [])
            
            headers = {
                'Accept': 'application/vnd.github.v3+json',
                'User-Agent': 'Glossarion-Updater'
            }
            
            # Try with reasonable timeout and minimal retries to prevent hanging
            max_retries = 0  # No retries to prevent hanging
            timeout = 10  # Reasonable timeout
            
            for attempt in range(max_retries + 1):
                try:
                    print(f"[DEBUG] Update check attempt {attempt + 1}/{max_retries + 1}")
                    response = requests.get(self.GITHUB_LATEST_URL, headers=headers, timeout=timeout)
                    response.raise_for_status()
                    break  # Success, exit retry loop
                except (requests.Timeout, requests.ConnectionError) as e:
                    if attempt == max_retries:
                        # Last attempt failed, save check time and re-raise
                        self._save_last_check_time()
                        raise
                    print(f"[DEBUG] Network error on attempt {attempt + 1}: {e}")
                    time.sleep(1)  # Short delay before retry
            
            release_data = response.json()
            latest_version = release_data['tag_name'].lstrip('v')
            
            # Save successful check time
            self._save_last_check_time()
            
            # Fetch all releases for history regardless (with timeout protection)
            try:
                self.all_releases = self.fetch_multiple_releases(count=10)
            except Exception as e:
                print(f"[DEBUG] Could not fetch release history: {e}")
                self.all_releases = [release_data]  # Use just the latest release
            self.latest_release = release_data
            
            # Check if this version was skipped by user
            if release_data['tag_name'] in skipped_versions and not force_show:
                return False, None
            
            # Compare versions
            if version.parse(latest_version) > version.parse(self.CURRENT_VERSION):
                self.update_available = True
                
                # Update available - will be handled by signal
                print(f"[DEBUG] Update available for version {latest_version}")
                    
                return True, release_data
            else:
                # We're up to date
                self.update_available = False
                
                # Dialog will be shown via signal if force_show is True
                return False, None
                
        except requests.Timeout:
            if not silent:
                msg = QMessageBox(self.dialog if hasattr(self.dialog, 'show') else None)
                msg.setIcon(QMessageBox.Critical)
                msg.setWindowTitle("Update Check Failed")
                msg.setText("Connection timed out while checking for updates.\n\n"
                          "This is usually due to network connectivity issues.\n"
                          "The next update check will be in 1 hour.")
                msg.exec()
            return False, None
            
        except requests.ConnectionError as e:
            if not silent:
                msg = QMessageBox(self.dialog if hasattr(self.dialog, 'show') else None)
                msg.setIcon(QMessageBox.Critical)
                msg.setWindowTitle("Update Check Failed")
                if 'api.github.com' in str(e):
                    msg.setText("Cannot reach GitHub servers for update check.\n\n"
                              "This may be due to:\n"
                              "• Internet connectivity issues\n"
                              "• Firewall blocking GitHub API\n"
                              "• GitHub API temporarily unavailable\n\n"
                              "The next update check will be in 1 hour.")
                else:
                    msg.setText(f"Network error: {str(e)}\n\n"
                              "The next update check will be in 1 hour.")
                msg.exec()
            return False, None
            
        except requests.HTTPError as e:
            if not silent:
                msg = QMessageBox(self.dialog if hasattr(self.dialog, 'show') else None)
                msg.setIcon(QMessageBox.Critical)
                msg.setWindowTitle("Update Check Failed")
                if e.response.status_code == 403:
                    msg.setText("GitHub API rate limit exceeded. Please try again later.")
                else:
                    msg.setText(f"GitHub returned error: {e.response.status_code}")
                msg.exec()
            return False, None
            
        except ValueError as e:
            if not silent:
                msg = QMessageBox(self.dialog if hasattr(self.dialog, 'show') else None)
                msg.setIcon(QMessageBox.Critical)
                msg.setWindowTitle("Update Check Failed")
                msg.setText("Invalid response from GitHub. The update service may be temporarily unavailable.")
                msg.exec()
            return False, None
            
        except Exception as e:
            if not silent:
                msg = QMessageBox(self.dialog if hasattr(self.dialog, 'show') else None)
                msg.setIcon(QMessageBox.Critical)
                msg.setWindowTitle("Update Check Failed")
                msg.setText(f"An unexpected error occurred:\n{str(e)}")
                msg.exec()
            return False, None
    
    def check_for_updates_manual(self):
        """Manual update check from menu - always shows dialog (async)"""
        return self.check_for_updates_async(silent=False, force_show=True)
    
    def check_for_updates(self, silent=True, force_show=False):
        """Public method for checking updates - delegates to async method"""
        return self.check_for_updates_async(silent=silent, force_show=force_show)
    
    def _save_last_check_time(self):
        """Save the last update check time to config"""
        try:
            current_time = time.time()
            self._last_check_time = current_time
            self.main_gui.config['last_update_check_time'] = current_time
            # Save config without showing message
            self.main_gui.save_config(show_message=False)
        except Exception as e:
            print(f"[DEBUG] Failed to save last check time: {e}")
    
    def format_markdown_to_qt(self, text_widget, markdown_text):
        """Convert GitHub markdown to formatted Qt text - simplified version
        
        Args:
            text_widget: The QTextEdit widget to insert formatted text into
            markdown_text: The markdown source text
        """
        # Set default font
        default_font = QFont()
        default_font.setPointSize(10)
        text_widget.setFont(default_font)
        
        # Process text line by line with minimal formatting
        lines = markdown_text.split('\n')
        cursor = text_widget.textCursor()
        cursor.movePosition(QTextCursor.End)
        
        for line in lines:
            # Strip any weird unicode characters that might cause display issues
            line = ''.join(char for char in line if ord(char) < 65536)
            
            # Handle headings
            if line.startswith('#'):
                # Remove all # symbols and get the heading text
                heading_text = line.lstrip('#').strip()
                if heading_text:
                    cursor.insertText(heading_text + '\n')
                    # Make it bold by moving back and applying format
                    cursor.movePosition(QTextCursor.PreviousBlock)
                    cursor.select(QTextCursor.BlockUnderCursor)
                    fmt = cursor.charFormat()
                    font = QFont()
                    font.setBold(True)
                    font.setPointSize(12)
                    fmt.setFont(font)
                    cursor.mergeCharFormat(fmt)
                    cursor.movePosition(QTextCursor.End)
            
            # Handle bullet points
            elif line.strip().startswith(('- ', '* ')):
                # Get the text after the bullet
                bullet_text = line.strip()[2:].strip()
                # Clean the text of markdown formatting
                bullet_text = self._clean_markdown_text(bullet_text)
                cursor.insertText('    • ' + bullet_text + '\n')
            
            # Handle numbered lists  
            elif re.match(r'^\s*\d+\.\s', line):
                # Extract number and text
                match = re.match(r'^(\s*)(\d+)\.\s(.+)', line)
                if match:
                    indent, num, text = match.groups()
                    clean_text = self._clean_markdown_text(text.strip())
                    cursor.insertText(f'    {num}. {clean_text}\n')
            
            # Handle separator lines
            elif line.strip() in ['---', '***', '___']:
                cursor.insertText('─' * 40 + '\n')
            
            # Handle code blocks - just skip the markers
            elif line.strip().startswith('```'):
                continue  # Skip code fence markers
            
            # Regular text
            elif line.strip():
                # Clean and insert the line
                clean_text = self._clean_markdown_text(line)
                cursor.insertText(clean_text + '\n')
            
            # Empty lines
            else:
                cursor.insertText('\n')
        
        # Move cursor to start and scroll to top
        cursor.movePosition(QTextCursor.Start)
        text_widget.setTextCursor(cursor)
    
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
    
    def _show_loading_dialog(self):
        """Show loading dialog during update check"""
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar
        from PySide6.QtCore import Qt, QTimer
        from PySide6.QtGui import QIcon, QPixmap
        import os
        
        # Get the main GUI window for parenting - ensure it's a proper QWidget
        from PySide6.QtWidgets import QWidget
        parent = None
        if hasattr(self.dialog, 'show') and isinstance(self.dialog, QWidget):
            parent = self.dialog
        
        # Create loading dialog
        self.loading_dialog = QDialog(parent)
        self.loading_dialog.setWindowTitle("Checking for Updates")
        self.loading_dialog.setFixedSize(300, 150)
        self.loading_dialog.setModal(True)
        
        # Set the proper application icon for the dialog
        try:
            ico_path = os.path.join(self.base_dir, 'Halgakos.ico')
            if os.path.isfile(ico_path):
                self.loading_dialog.setWindowIcon(QIcon(ico_path))
        except Exception as e:
            print(f"Could not load icon for loading dialog: {e}")
        
        # Position dialog at center of parent
        if parent:
            self.loading_dialog.move(parent.geometry().center() - self.loading_dialog.rect().center())
        
        # Create main layout
        main_layout = QVBoxLayout(self.loading_dialog)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(10)
        
        # Try to load and resize the icon
        try:
            ico_path = os.path.join(self.base_dir, 'Halgakos.ico')
            if os.path.isfile(ico_path):
                # Load and resize image
                icon_pixmap = QPixmap(ico_path).scaled(48, 48, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                icon_label = QLabel()
                icon_label.setPixmap(icon_pixmap)
                icon_label.setAlignment(Qt.AlignCenter)
                main_layout.addWidget(icon_label)
        except Exception as e:
            print(f"Could not load loading icon: {e}")
        
        # Add loading text
        self.loading_text = QLabel("Checking for updates...")
        self.loading_text.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.loading_text)
        
        # Add progress bar
        progress_bar = QProgressBar()
        progress_bar.setRange(0, 0)  # Indeterminate mode
        main_layout.addWidget(progress_bar)
        
        # Animation state
        self.loading_animation_active = True
        self.loading_rotation = 0
        
        def animate_text():
            """Animate the loading text"""
            if not self.loading_animation_active or not hasattr(self, 'loading_text'):
                return
                
            try:
                # Simple text-based animation
                dots = "." * ((self.loading_rotation // 10) % 4)
                self.loading_text.setText(f"Checking for updates{dots}")
                self.loading_rotation += 1
                
                # Schedule next animation frame
                QTimer.singleShot(100, animate_text)
            except:
                pass  # Dialog might have been destroyed
        
        # Start text animation
        animate_text()
        
        # Show the dialog
        self.loading_dialog.show()
    
    def _close_loading_dialog(self):
        """Close the loading dialog if it exists"""
        try:
            if hasattr(self, 'loading_animation_active'):
                self.loading_animation_active = False
            if hasattr(self, 'loading_dialog') and self.loading_dialog:
                self.loading_dialog.close()
                delattr(self, 'loading_dialog')
        except:
            pass  # Dialog might already be destroyed
    
    def show_update_dialog(self):
        """Show update dialog (for updates or version history)"""
        print("[DEBUG] show_update_dialog called")
        
        if not self.latest_release and not self.all_releases:
            print("[DEBUG] No release data, trying to fetch...")
            # Try to fetch releases if we don't have them
            try:
                self.all_releases = self.fetch_multiple_releases(count=10)
                if self.all_releases:
                    self.latest_release = self.all_releases[0]
                    print(f"[DEBUG] Fetched {len(self.all_releases)} releases")
                else:
                    print("[DEBUG] No releases fetched")
                    msg = QMessageBox(self.dialog if hasattr(self.dialog, 'show') else None)
                    msg.setIcon(QMessageBox.Critical)
                    msg.setWindowTitle("Error")
                    msg.setText("Unable to fetch version information from GitHub.")
                    msg.exec()
                    return
            except Exception as e:
                print(f"[DEBUG] Error fetching releases in show_update_dialog: {e}")
                return
        
        # Set appropriate title
        if self.update_available:
            title = "Update Available"
        else:
            title = "Version History"
        
        print(f"[DEBUG] Creating update dialog with title: {title}")
        
        # Use existing QApplication instance - never create a new one
        app = QApplication.instance()
        if not app:
            print("[ERROR] No QApplication instance found - update dialog cannot be shown")
            return
        
        # Determine parent window - use active modal widget or main GUI
        from PySide6.QtWidgets import QWidget
        parent_widget = None
        try:
            # Check if there's an active modal widget (e.g., Other Settings dialog)
            parent_widget = app.activeModalWidget()
            if not parent_widget:
                # Fall back to active window
                parent_widget = app.activeWindow()
            if not parent_widget:
                # Fall back to main GUI if it's a proper QWidget
                if hasattr(self.dialog, 'show') and isinstance(self.dialog, QWidget):
                    parent_widget = self.dialog
        except Exception:
            if hasattr(self.dialog, 'show') and isinstance(self.dialog, QWidget):
                parent_widget = self.dialog
        
        # Create simple non-blocking dialog with appropriate parent
        dialog = QDialog(parent_widget)
        dialog.setWindowTitle(title)
        dialog.setModal(False)  # Ensure non-modal
        
        # Apply dark theme styling to fix white background
        dialog.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
                color: white;
            }
            QWidget {
                background-color: #1e1e1e;
                color: white;
            }
            QLabel {
                color: white;
                background-color: transparent;
            }
        """)
        
        print(f"[DEBUG] Dialog created successfully with dark theme")
        
        # Get screen dimensions and calculate size
        screen = app.primaryScreen().geometry()
        dialog_width = int(screen.width() * 0.25)  # Half of 0.5
        dialog_height = int(screen.height() * 0.4)  # Half of 0.8
        dialog.resize(dialog_width, dialog_height)
        
        # Set icon if available
        icon_path = os.path.join(self.base_dir, 'halgakos.ico')
        if os.path.exists(icon_path):
            dialog.setWindowIcon(QIcon(icon_path))
        
        # Apply global stylesheet for radio buttons
        dialog.setStyleSheet("""
            QRadioButton {
                color: white;
                spacing: 6px;
            }
            QRadioButton::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid #5a9fd4;
                border-radius: 7px;
                background-color: #2d2d2d;
            }
            QRadioButton::indicator:hover {
                border: 1px solid #7ab8e8;
                background-color: #3d3d3d;
            }
            QRadioButton::indicator:checked {
                background-color: #5a9fd4;
                border: 1px solid #5a9fd4;
            }
            QRadioButton::indicator:checked:hover {
                background-color: #7ab8e8;
                border: 1px solid #7ab8e8;
            }
            QRadioButton::indicator:disabled {
                border: 1px solid #555555;
                background-color: #1e1e1e;
            }
            QGroupBox {
                color: white;
                border: 1px solid #555555;
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 5px;
                color: #5a9fd4;
                font-weight: bold;
            }
        """)
        
        # Populate content
        self._populate_update_dialog(dialog)

    def _populate_update_dialog(self, dialog):
        """Populate the update dialog content"""
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(10)
        
        # Initialize selected_asset to None
        self.selected_asset = None
        
        # Version info
        version_group = QGroupBox("Version Information")
        version_layout = QVBoxLayout()
        version_layout.setContentsMargins(10, 10, 10, 10)
        
        current_label = QLabel(f"Current Version: {self.CURRENT_VERSION}")
        version_layout.addWidget(current_label)
        
        if self.latest_release:
            latest_version = self.latest_release['tag_name']
            if self.update_available:
                latest_label = QLabel(f"Latest Version: {latest_version}")
                latest_font = QFont()
                latest_font.setBold(True)
                latest_font.setPointSize(10)
                latest_label.setFont(latest_font)
                version_layout.addWidget(latest_label)
            else:
                latest_label = QLabel(f"Latest Version: {latest_version} ✓ You are up to date!")
                latest_font = QFont()
                latest_font.setBold(True)
                latest_font.setPointSize(10)
                latest_label.setFont(latest_font)
                latest_label.setStyleSheet("color: green;")
                version_layout.addWidget(latest_label)
        
        version_group.setLayout(version_layout)
        main_layout.addWidget(version_group)
        
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
                
                asset_group = QGroupBox(frame_title)
                asset_layout = QVBoxLayout()
                asset_layout.setContentsMargins(10, 10, 10, 10)
                
                if len(exe_assets) > 1:
                    # Multiple exe files - show radio buttons to choose
                    self.asset_button_group = QButtonGroup()
                    for i, asset in enumerate(exe_assets):
                        filename = asset['name']
                        size_mb = asset['size'] / (1024 * 1024)
                        
                        # Identify variant type based on first letter of filename
                        first_letter = filename[0].upper() if filename else ''
                        if first_letter == 'G':
                            variant_type = "Standard"
                        elif first_letter == 'L':
                            variant_type = "Lite"
                        elif first_letter == 'N':
                            variant_type = "No CUDA"
                        else:
                            # Fallback: check for keywords in filename
                            if 'lite' in filename.lower():
                                variant_type = "Lite"
                            elif 'cuda' in filename.lower():
                                variant_type = "No CUDA"
                            elif 'full' in filename.lower():
                                variant_type = "Full"
                            else:
                                variant_type = "Standard"
                        
                        variant_label = f"{variant_type} - {filename} ({size_mb:.1f} MB)"
                        
                        rb = QRadioButton(variant_label)
                        rb.setProperty("asset_index", i)
                        self.asset_button_group.addButton(rb, i)
                        asset_layout.addWidget(rb)
                        
                        # Select first option by default
                        if i == 0:
                            rb.setChecked(True)
                            self.selected_asset = asset
                    
                    # Add listener for selection changes
                    def on_asset_change(button_id):
                        self.selected_asset = exe_assets[button_id]
                    
                    self.asset_button_group.idClicked.connect(on_asset_change)
                else:
                    # Only one exe file - just show it and set it as selected
                    self.selected_asset = exe_assets[0]
                    filename = exe_assets[0]['name']
                    size_mb = exe_assets[0]['size'] / (1024 * 1024)
                    asset_label = QLabel(f"{filename} ({size_mb:.1f} MB)")
                    asset_layout.addWidget(asset_label)
                
                asset_group.setLayout(asset_layout)
                main_layout.addWidget(asset_group)
        
        # Create tab widget for version history
        tab_widget = QTabWidget()
        tab_widget.setMinimumHeight(300)
        
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
                
                # Create widget for this version
                tab_widget_container = QWidget()
                tab_layout = QVBoxLayout(tab_widget_container)
                tab_layout.setContentsMargins(10, 10, 10, 10)
                
                # Add release date
                if 'published_at' in release:
                    date_str = release['published_at'][:10]  # Get YYYY-MM-DD
                    date_label = QLabel(f"Released: {date_str}")
                    date_font = QFont()
                    date_font.setItalic(True)
                    date_font.setPointSize(9)
                    date_label.setFont(date_font)
                    tab_layout.addWidget(date_label)
                
                # Create text widget for release notes
                notes_text = QTextEdit()
                notes_text.setReadOnly(True)
                notes_text.setMinimumHeight(200)
                
                # Format and insert release notes with markdown support
                release_notes = release.get('body', 'No release notes available')
                self.format_markdown_to_qt(notes_text, release_notes)
                
                tab_layout.addWidget(notes_text)
                tab_widget.addTab(tab_widget_container, tab_label)
        else:
            # Fallback to simple display if no releases fetched
            tab_widget_container = QWidget()
            tab_layout = QVBoxLayout(tab_widget_container)
            tab_layout.setContentsMargins(10, 10, 10, 10)
            
            notes_text = QTextEdit()
            notes_text.setReadOnly(True)
            notes_text.setMinimumHeight(200)
            
            if self.latest_release:
                release_notes = self.latest_release.get('body', 'No release notes available')
                self.format_markdown_to_qt(notes_text, release_notes)
            else:
                notes_text.setPlainText('Unable to fetch release notes.')
            
            tab_layout.addWidget(notes_text)
            tab_widget.addTab(tab_widget_container, "Release Notes")
        
        main_layout.addWidget(tab_widget)
        
        # Download progress (initially hidden)
        self.progress_widget = QWidget()
        progress_layout = QVBoxLayout(self.progress_widget)
        progress_layout.setContentsMargins(0, 10, 0, 10)
        
        self.progress_label = QLabel("Downloading update...")
        progress_layout.addWidget(self.progress_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setTextVisible(True)
        progress_layout.addWidget(self.progress_bar)
        
        # Add status label for download details
        self.status_label = QLabel("")
        status_font = QFont()
        status_font.setPointSize(8)
        self.status_label.setFont(status_font)
        progress_layout.addWidget(self.status_label)
        
        # Hide progress initially
        self.progress_widget.setVisible(False)
        
        # Add progress widget to layout (hidden initially)
        main_layout.addWidget(self.progress_widget)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        
        def start_download():
            if not self.selected_asset:
                msg = QMessageBox(dialog)
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("No File Selected")
                msg.setText("Please select a version to download.")
                msg.exec()
                return
            
            # Show progress
            self.progress_widget.setVisible(True)
            
            # Disable all buttons
            for i in range(button_layout.count()):
                widget = button_layout.itemAt(i).widget()
                if widget and isinstance(widget, QPushButton):
                    widget.setEnabled(False)
            
            # Reset progress
            self.progress_bar.setValue(0)
            self.download_progress = 0
            
            # Start download using shared executor if available
            try:
                if hasattr(self.main_gui, '_ensure_executor'):
                    self.main_gui._ensure_executor()
                execu = getattr(self, 'executor', None) or getattr(self.main_gui, 'executor', None)
                if execu:
                    execu.submit(self.download_update, dialog)
                else:
                    thread = threading.Thread(target=self.download_update, args=(dialog,), daemon=True)
                    thread.start()
            except Exception:
                thread = threading.Thread(target=self.download_update, args=(dialog,), daemon=True)
                thread.start()
        
        # Always show download button if we have exe files
        has_exe_files = self.selected_asset is not None
        
        if self.update_available:
            # Show update-specific buttons
            download_btn = QPushButton("Download Update")
            download_btn.setMinimumHeight(35)
            download_btn.clicked.connect(start_download)
            download_btn.setStyleSheet("""
                QPushButton {
                    background-color: #28a745;
                    color: white;
                    padding: 8px 20px;
                    font-size: 11pt;
                    font-weight: bold;
                    border-radius: 4px;
                }
                QPushButton:hover { background-color: #218838; }
            """)
            button_layout.addWidget(download_btn)
            
            remind_btn = QPushButton("Remind Me Later")
            remind_btn.setMinimumHeight(35)
            remind_btn.clicked.connect(dialog.close)
            remind_btn.setStyleSheet("""
                QPushButton {
                    background-color: #6c757d;
                    color: white;
                    padding: 8px 20px;
                    font-size: 11pt;
                    font-weight: bold;
                    border-radius: 4px;
                }
                QPushButton:hover { background-color: #5a6268; }
            """)
            button_layout.addWidget(remind_btn)
            
            skip_btn = QPushButton("Skip This Version")
            skip_btn.setMinimumHeight(35)
            skip_btn.clicked.connect(lambda: self.skip_version(dialog))
            skip_btn.setStyleSheet("""
                QPushButton {
                    background-color: transparent;
                    color: #007bff;
                    padding: 8px 20px;
                    font-size: 11pt;
                    border: none;
                    text-decoration: underline;
                }
                QPushButton:hover { color: #0056b3; }
            """)
            button_layout.addWidget(skip_btn)
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
                download_btn = QPushButton("Download Different Path")
                download_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #17a2b8;
                        color: white;
                        padding: 8px 20px;
                        font-size: 11pt;
                        font-weight: bold;
                        border-radius: 4px;
                    }
                    QPushButton:hover { background-color: #117a8b; }
                """)
            else:
                # Single version available
                download_btn = QPushButton("Re-download")
                download_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #6c757d;
                        color: white;
                        padding: 8px 20px;
                        font-size: 11pt;
                        font-weight: bold;
                        border-radius: 4px;
                    }
                    QPushButton:hover { background-color: #5a6268; }
                """)
            download_btn.setMinimumHeight(35)
            download_btn.clicked.connect(start_download)
            button_layout.addWidget(download_btn)
            
            close_btn = QPushButton("Close")
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
                QPushButton:hover { background-color: #5a6268; }
            """)
            button_layout.addWidget(close_btn)
        else:
            # No downloadable files
            close_btn = QPushButton("Close")
            close_btn.setMinimumHeight(35)
            close_btn.clicked.connect(dialog.close)
            close_btn.setStyleSheet("""
                QPushButton {
                    background-color: #007bff;
                    color: white;
                    padding: 8px 20px;
                    font-size: 11pt;
                    font-weight: bold;
                    border-radius: 4px;
                }
                QPushButton:hover { background-color: #0056b3; }
            """)
            button_layout.addWidget(close_btn)
        
        # Add "View All Releases" link button
        def open_releases_page():
            import webbrowser
            webbrowser.open("https://github.com/Shirochi-stack/Glossarion/releases")
        
        view_releases_btn = QPushButton("View All Releases")
        view_releases_btn.setMinimumHeight(35)
        view_releases_btn.clicked.connect(open_releases_page)
        view_releases_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #007bff;
                padding: 8px 20px;
                font-size: 11pt;
                border: none;
                text-decoration: underline;
            }
            QPushButton:hover { color: #0056b3; }
        """)
        button_layout.addStretch()
        button_layout.addWidget(view_releases_btn)
        
        # Add button layout to main layout
        main_layout.addLayout(button_layout)
        
        # Set dialog layout and show
        dialog.setLayout(main_layout)
        
        # Show dialog as non-modal window that won't block other dialogs
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        
        # Keep reference to prevent garbage collection
        self._update_dialog = dialog
    
    def skip_version(self, dialog):
        """Mark this version as skipped and close dialog"""
        if not self.latest_release:
            dialog.close()
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
        dialog.close()
        
        # Show confirmation
        msg = QMessageBox(self.dialog if hasattr(self.dialog, 'show') else None)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Version Skipped")
        msg.setText(f"Version {version_tag} will be skipped in future update checks.\n"
                   "You can manually check for updates from the Help menu.")
        msg.exec()
    
    def download_update(self, dialog):
        """Download the update file"""
        try:
            # Use the selected asset
            asset = self.selected_asset
                    
            if not asset:
                def show_error():
                    msg = QMessageBox(dialog)
                    msg.setIcon(QMessageBox.Critical)
                    msg.setWindowTitle("Download Error")
                    msg.setText("No file selected for download.")
                    msg.exec()
                QTimer.singleShot(0, show_error)
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
            
            # Download with progress tracking and shorter timeout
            response = requests.get(asset['browser_download_url'], stream=True, timeout=15)
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
                            
                            # Use QTimer for smoother updates
                            def update_progress(p=progress, d=size_mb, t=total_mb):
                                try:
                                    self.progress_bar.setValue(p)
                                    self.progress_label.setText(f"Downloading update... {p}%")
                                    self.status_label.setText(f"{d:.1f} MB / {t:.1f} MB")
                                except:
                                    pass  # Dialog might have been closed
                            
                            QTimer.singleShot(0, update_progress)
            
            # Download complete
            QTimer.singleShot(0, lambda: self.download_complete(dialog, download_path))
            
        except Exception as e:
            # Capture the error message immediately
            error_msg = str(e)
            def show_error():
                msg = QMessageBox(dialog)
                msg.setIcon(QMessageBox.Critical)
                msg.setWindowTitle("Download Failed")
                msg.setText(error_msg)
                msg.exec()
            QTimer.singleShot(0, show_error)
    
    def download_complete(self, dialog, file_path):
        """Handle completed download"""
        dialog.close()
        
        msg = QMessageBox(self.dialog if hasattr(self.dialog, 'show') else None)
        msg.setIcon(QMessageBox.Question)
        msg.setWindowTitle("Download Complete")
        msg.setText("Update downloaded successfully.\n\n"
                   "Would you like to install it now?\n"
                   "(The application will need to restart)")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.Yes)
        
        if msg.exec() == QMessageBox.Yes:
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
            QApplication.quit()
            sys.exit(0)
            
        except Exception as e:
            msg = QMessageBox(self.dialog if hasattr(self.dialog, 'show') else None)
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Installation Error")
            msg.setText(f"Could not start update process:\n{str(e)}")
            msg.exec()
