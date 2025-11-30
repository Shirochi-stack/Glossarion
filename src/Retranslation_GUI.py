"""
Retranslation GUI Module
Force retranslation functionality for EPUB, text, and image files
"""

import os
import sys
import json
import re
from PySide6.QtWidgets import (QWidget, QDialog, QLabel, QFrame, QListWidget, 
                                QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout,
                                QMessageBox, QFileDialog, QTabWidget, QListWidgetItem,
                                QScrollArea, QSizePolicy)
from PySide6.QtCore import Qt, Signal, QTimer, QPropertyAnimation, QEasingCurve, Property
from PySide6.QtGui import QFont, QColor, QTransform, QIcon, QPixmap
import xml.etree.ElementTree as ET
import zipfile
import shutil
import traceback

# WindowManager and UIHelper removed - not needed in PySide6
# Qt handles window management and UI utilities automatically


class AnimatedRefreshButton(QPushButton):
    """Custom QPushButton with rotation animation for refresh action using Halgakos.ico"""
    
    def __init__(self, text="Refresh", parent=None):
        super().__init__(text, parent)
        self._rotation = 0
        self._animation = None
        self._original_text = text
        self._timer = None
        self._animation_step = 0
        self._original_icon = None
        
        # Try to load Halgakos.ico
        try:
            # Get base directory
            base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
            ico_path = os.path.join(base_dir, 'Halgakos.ico')
            if os.path.isfile(ico_path):
                self._original_icon = QIcon(ico_path)
                self.setIcon(self._original_icon)
                self.setIconSize(self.iconSize() * 1.2)  # Make icon slightly larger
        except Exception as e:
            print(f"Could not load Halgakos.ico for refresh button: {e}")
        
    def get_rotation(self):
        return self._rotation
    
    def set_rotation(self, angle):
        self._rotation = angle
        self.update()  # Trigger repaint
    
    # Define rotation as a Qt Property for animation
    rotation = Property(float, get_rotation, set_rotation)
    
    def start_animation(self):
        """Start the spinning animation"""
        if self._timer and self._timer.isActive():
            return  # Already animating
        
        # Update stylesheet to show active state
        current_style = self.styleSheet()
        if "background-color: #17a2b8" in current_style:
            self.setStyleSheet(current_style.replace(
                "background-color: #17a2b8",
                "background-color: #138496"
            ))
        
        # Start timer-based animation for icon rotation
        self._animation_step = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_animation_frame)
        self._timer.start(50)  # Update every 50ms for smooth rotation
    
    def _update_animation_frame(self):
        """Update animation frame by rotating the icon"""
        if self._original_icon:
            # Increment rotation angle (30 degrees per frame for smooth spinning)
            self._rotation = (self._rotation + 30) % 360
            
            # Create a rotated version of the icon
            pixmap = self._original_icon.pixmap(self.iconSize())
            transform = QTransform().rotate(self._rotation)
            rotated_pixmap = pixmap.transformed(transform, Qt.SmoothTransformation)
            
            # Set the rotated icon
            self.setIcon(QIcon(rotated_pixmap))
    
    def stop_animation(self):
        """Stop the spinning animation"""
        if self._timer:
            self._timer.stop()
            self._timer = None
            self._rotation = 0
            self._animation_step = 0
            
            # Restore original icon (unrotated)
            if self._original_icon:
                self.setIcon(self._original_icon)
            
            # Restore original stylesheet
            current_style = self.styleSheet()
            if "background-color: #138496" in current_style:
                self.setStyleSheet(current_style.replace(
                    "background-color: #138496",
                    "background-color: #17a2b8"
                ))
            
            self.update()


class RetranslationMixin:
    """Mixin class containing retranslation methods for TranslatorGUI"""
    
    def _clear_layout(self, layout):
        """Safely clear all items from a layout"""
        if layout is None:
            return
        while layout.count():
            item = layout.takeAt(0)
            if item:
                widget = item.widget()
                if widget:
                    widget.setParent(None)
                    widget.deleteLater()
                elif item.layout():
                    self._clear_layout(item.layout())
    
    def _get_dialog_size(self, width_ratio=0.5, height_ratio=0.5):
        """Calculate dialog size as a ratio of screen size (default 50% width, 50% height)"""
        try:
            from PySide6.QtWidgets import QApplication
            from PySide6.QtGui import QScreen
            
            # Get primary screen
            screen = QApplication.primaryScreen()
            if screen:
                geometry = screen.availableGeometry()
                width = int(geometry.width() * width_ratio)
                height = int(geometry.height() * height_ratio)
                return width, height
        except:
            pass
        
        # Fallback to reasonable defaults if screen info unavailable
        return int(1920 * width_ratio), int(1080 * height_ratio)
    
    def _show_message(self, msg_type, title, message, parent=None):
        """Show message using PySide6 QMessageBox with Halgakos icon"""
        try:
            # Create message box
            msg_box = QMessageBox(parent)
            msg_box.setWindowTitle(title)
            msg_box.setText(message)
            
            # Set icon based on message type
            if msg_type == 'info':
                msg_box.setIcon(QMessageBox.Information)
            elif msg_type == 'warning':
                msg_box.setIcon(QMessageBox.Warning)
            elif msg_type == 'error':
                msg_box.setIcon(QMessageBox.Critical)
            elif msg_type == 'question':
                msg_box.setIcon(QMessageBox.Question)
                msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            
            # Try to set Halgakos window icon
            try:
                from PySide6.QtGui import QIcon
                if hasattr(self, 'base_dir'):
                    base_dir = self.base_dir
                else:
                    import sys
                    base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
                ico_path = os.path.join(base_dir, 'Halgakos.ico')
                if os.path.isfile(ico_path):
                    msg_box.setWindowIcon(QIcon(ico_path))
            except:
                pass
            
            # Show message box
            if msg_type == 'question':
                return msg_box.exec() == QMessageBox.Yes
            else:
                msg_box.exec()
                return True
                
        except Exception as e:
            # Fallback to console if dialog fails
            print(f"{title}: {message}")
            if msg_type == 'question':
                return False
            return False
 
    def force_retranslation(self):
        """Force retranslation of specific chapters or images with improved display"""
        
        # Check for multiple file selection first
        if hasattr(self, 'selected_files') and len(self.selected_files) > 1:
            self._force_retranslation_multiple_files()
            return
        
        # Check if it's a folder selection (for images)
        if hasattr(self, 'selected_files') and len(self.selected_files) > 0:
            # Check if the first selected file is actually a folder
            first_item = self.selected_files[0]
            if os.path.isdir(first_item):
                self._force_retranslation_images_folder(first_item)
                return
        
        # Original logic for single files
        # Get input path from QLineEdit widget
        if hasattr(self.entry_epub, 'text'):
            # PySide6 QLineEdit widget
            input_path = self.entry_epub.text()
        else:
            input_path = ""
        
        if not input_path or not os.path.isfile(input_path):
            self._show_message('error', "Error", "Please select a valid EPUB, text file, or image folder first.")
            return
        
        # Check if it's an image file
        image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')
        if input_path.lower().endswith(image_extensions):
            # For single image, pass the image file path itself
            self._force_retranslation_images_folder(input_path)
            return
        
        # Check if dialog already exists for this file and is just hidden
        file_key = os.path.abspath(input_path)
        if hasattr(self, '_retranslation_dialog_cache') and file_key in self._retranslation_dialog_cache:
            # Reuse existing dialog - just show it and refresh data
            cached_data = self._retranslation_dialog_cache[file_key]
            if cached_data and cached_data.get('dialog'):
                # Check if output folder still exists before trying to refresh
                output_dir = cached_data.get('output_dir')
                progress_file = cached_data.get('progress_file')
                
                if not output_dir or not os.path.exists(output_dir):
                    # Output folder was deleted - show message and remove from cache
                    self._show_message('info', "Info", "No translation output found for this file.")
                    del self._retranslation_dialog_cache[file_key]
                    return
                
                if not progress_file or not os.path.exists(progress_file):
                    # Progress file was deleted - show message and remove from cache
                    self._show_message('info', "Info", "No progress tracking found.")
                    del self._retranslation_dialog_cache[file_key]
                    return
                
                dialog = cached_data['dialog']
                # Refresh the data before showing
                self._refresh_retranslation_data(cached_data)
                dialog.show()
                dialog.raise_()
                dialog.activateWindow()
                return
        
        # For EPUB/text files, use the shared logic
        # Get current toggle state if it exists
        show_special = False
        if hasattr(self, '_retranslation_dialog_cache') and file_key in self._retranslation_dialog_cache:
            cached_data = self._retranslation_dialog_cache[file_key]
            if cached_data:
                show_special = cached_data.get('show_special_files_state', False)
        
        self._force_retranslation_epub_or_text(input_path, show_special_files_state=show_special)


    def _force_retranslation_epub_or_text(self, file_path, parent_dialog=None, tab_frame=None, show_special_files_state=False):
        """
        Shared logic for force retranslation of EPUB/text files with OPF support
        Can be used standalone or embedded in a tab
        
        Args:
            file_path: Path to the EPUB/text file
            parent_dialog: If provided, won't create its own dialog
            tab_frame: If provided, will render into this frame instead of creating dialog
            show_special_files_state: Initial state for showing special files toggle
        
        Returns:
            dict: Contains all the UI elements and data for external access
        """
        
        epub_base = os.path.splitext(os.path.basename(file_path))[0]
        output_dir = epub_base
        
        if not os.path.exists(output_dir):
            if not parent_dialog:
                self._show_message('info', "Info", "No translation output found for this file.")
            return None
        
        progress_file = os.path.join(output_dir, "translation_progress.json")
        if not os.path.exists(progress_file):
            if not parent_dialog:
                self._show_message('info', "Info", "No progress tracking found.")
            return None
        
        with open(progress_file, 'r', encoding='utf-8') as f:
            prog = json.load(f)
        
        # =====================================================
        # PARSE CONTENT.OPF FOR CHAPTER MANIFEST
        # =====================================================
        
        # State variable for special files toggle (will be set later by checkbox)
        show_special_files = [show_special_files_state]  # Use list to allow modification in nested function
        
        spine_chapters = []
        opf_chapter_order = {}
        is_epub = file_path.lower().endswith('.epub')
        
        if is_epub and os.path.exists(file_path):
            try:
                import xml.etree.ElementTree as ET
                import zipfile
                
                with zipfile.ZipFile(file_path, 'r') as zf:
                    # Find content.opf file
                    opf_path = None
                    opf_content = None
                    
                    # First try to find via container.xml
                    try:
                        container_content = zf.read('META-INF/container.xml')
                        container_root = ET.fromstring(container_content)
                        rootfile = container_root.find('.//{urn:oasis:names:tc:opendocument:xmlns:container}rootfile')
                        if rootfile is not None:
                            opf_path = rootfile.get('full-path')
                    except:
                        pass
                    
                    # Fallback: search for content.opf
                    if not opf_path:
                        for name in zf.namelist():
                            if name.endswith('content.opf'):
                                opf_path = name
                                break
                    
                    if opf_path:
                        opf_content = zf.read(opf_path)
                        
                        # Parse OPF
                        root = ET.fromstring(opf_content)
                        
                        # Handle namespaces
                        ns = {'opf': 'http://www.idpf.org/2007/opf'}
                        if root.tag.startswith('{'):
                            default_ns = root.tag[1:root.tag.index('}')]
                            ns = {'opf': default_ns}
                        
                        # Get manifest - all chapter files
                        manifest_chapters = {}
                        
                        for item in root.findall('.//opf:manifest/opf:item', ns):
                            item_id = item.get('id')
                            href = item.get('href')
                            media_type = item.get('media-type', '')
                            
                            if item_id and href and ('html' in media_type.lower() or href.endswith(('.html', '.xhtml', '.htm'))):
                                filename = os.path.basename(href)
                                
                                # Skip special files (files without numbers) unless show_special_files is enabled
                                import re
                                # Check if filename contains any digits
                                has_numbers = bool(re.search(r'\d', filename))
                                is_special = not has_numbers
                                if not is_special or show_special_files[0]:
                                    manifest_chapters[item_id] = {
                                        'filename': filename,
                                        'href': href,
                                        'media_type': media_type,
                                        'is_special': is_special
                                    }
                        
                        # Get spine order - the reading order
                        spine = root.find('.//opf:spine', ns)
                        
                        if spine is not None:
                            for itemref in spine.findall('opf:itemref', ns):
                                idref = itemref.get('idref')
                                if idref and idref in manifest_chapters:
                                    chapter_info = manifest_chapters[idref]
                                    filename = chapter_info['filename']
                                    is_special = chapter_info.get('is_special', False)
                                    
                                    # Skip navigation, toc, and cover files (unless show_special_files is enabled)
                                    if not is_special or show_special_files[0]:
                                        # Extract chapter number from filename
                                        import re
                                        matches = re.findall(r'(\d+)', filename)
                                        if matches:
                                            file_chapter_num = int(matches[-1])
                                        elif is_special:
                                            # Special files without numbers should be chapter 0
                                            file_chapter_num = 0
                                        else:
                                            file_chapter_num = len(spine_chapters)
                                        
                                        spine_chapters.append({
                                            'id': idref,
                                            'filename': filename,
                                            'position': len(spine_chapters),
                                            'file_chapter_num': file_chapter_num,
                                            'status': 'unknown',  # Will be updated
                                            'output_file': None,    # Will be updated
                                            'is_special': is_special
                                        })
                                        
                                        # Store the order for later use
                                        opf_chapter_order[filename] = len(spine_chapters) - 1
                                        
                                        # Also store without extension for matching
                                        filename_noext = os.path.splitext(filename)[0]
                                        opf_chapter_order[filename_noext] = len(spine_chapters) - 1
                        
            except Exception as e:
                print(f"Warning: Could not parse OPF: {e}")
        
        # =====================================================
        # MATCH OPF CHAPTERS WITH TRANSLATION PROGRESS
        # =====================================================
        
        # Build a map of original basenames to progress entries
        basename_to_progress = {}
        for chapter_key, chapter_info in prog.get("chapters", {}).items():
            original_basename = chapter_info.get("original_basename", "")
            if original_basename:
                if original_basename not in basename_to_progress:
                    basename_to_progress[original_basename] = []
                basename_to_progress[original_basename].append((chapter_key, chapter_info))
        
        # Also build a map of response files
        response_file_to_progress = {}
        for chapter_key, chapter_info in prog.get("chapters", {}).items():
            output_file = chapter_info.get("output_file", "")
            if output_file:
                if output_file not in response_file_to_progress:
                    response_file_to_progress[output_file] = []
                response_file_to_progress[output_file].append((chapter_key, chapter_info))
        
        # Update spine chapters with translation status
        for spine_ch in spine_chapters:
            filename = spine_ch['filename']
            chapter_num = spine_ch['file_chapter_num']
            is_special = spine_ch.get('is_special', False)
            
            # Find the actual response file that exists
            base_name = os.path.splitext(filename)[0]
            expected_response = None
            
            # Special files need to check what actually exists on disk
            if is_special:
                # Check for response_ prefix version
                response_with_prefix = f"response_{base_name}.html"
                retain = os.getenv('RETAIN_SOURCE_EXTENSION', '0') == '1' or self.config.get('retain_source_extension', False)
                
                if retain:
                    expected_response = filename
                elif os.path.exists(os.path.join(output_dir, response_with_prefix)):
                    expected_response = response_with_prefix
                else:
                    # Fallback to original filename
                    expected_response = filename
            else:
                # Handle .htm.html -> .html conversion
                stripped_base_name = base_name
                if base_name.endswith('.htm'):
                    stripped_base_name = base_name[:-4]  # Remove .htm suffix

                # Look for translated file matching base name, with or without 'response_' and with allowed extensions
                allowed_exts = ('.html', '.xhtml', '.htm', '.png', '.jpg', '.jpeg', '.webp', '.gif')
                for file in os.listdir(output_dir):
                    f_low = file.lower()
                    if f_low.endswith(allowed_exts):
                        name_no_ext = os.path.splitext(file)[0]
                        core = name_no_ext[9:] if name_no_ext.startswith('response_') else name_no_ext
                        # Accept matches for:
                        # - OPF filename without last extension (base_name)
                        # - Stripped base for .htm cases
                        # - OPF filename as-is (e.g., 'chapter_02.htm') when the output file is 'chapter_02.htm.xhtml'
                        if core == base_name or core == stripped_base_name or core == filename:
                            expected_response = file
                            break

                # Fallback - per mode, prefer OPF filename when retain mode is on
                if not expected_response:
                    retain = os.getenv('RETAIN_SOURCE_EXTENSION', '0') == '1' or self.config.get('retain_source_extension', False)
                    if retain:
                        expected_response = filename
                    else:
                        expected_response = f"response_{stripped_base_name}.html"
            
            response_path = os.path.join(output_dir, expected_response)
            
            # Check various ways to find the translation progress info
            matched_info = None
            
            # Method 1: Check by original basename
            if filename in basename_to_progress:
                entries = basename_to_progress[filename]
                if entries:
                    _, chapter_info = entries[0]
                    matched_info = chapter_info
            
            # Method 2: Check by response file (with corrected extension)
            if not matched_info and expected_response in response_file_to_progress:
                entries = response_file_to_progress[expected_response]
                if entries:
                    _, chapter_info = entries[0]
                    matched_info = chapter_info
            
            # Method 3: Search through all progress entries for matching output file
            if not matched_info:
                for chapter_key, chapter_info in prog.get("chapters", {}).items():
                    if chapter_info.get('output_file') == expected_response:
                        matched_info = chapter_info
                        break
            
            # Method 4: CRUCIAL - Match by chapter number (actual_num vs file_chapter_num)
            # Also check composite keys for special files (e.g., "0_message", "0_TOC")
            if not matched_info:
                # First try simple chapter number key
                simple_key = str(chapter_num)
                if simple_key in prog.get("chapters", {}):
                    chapter_info = prog["chapters"][simple_key]
                    out_file = chapter_info.get('output_file')
                    status = chapter_info.get('status', '')
                    
                    # Merged chapters: check if parent is completed
                    if status == 'merged':
                        parent_num = chapter_info.get('merged_parent_chapter')
                        if parent_num is not None:
                            parent_key = str(parent_num)
                            if parent_key in prog.get("chapters", {}):
                                parent_info = prog["chapters"][parent_key]
                                if parent_info.get('status') == 'completed':
                                    matched_info = chapter_info
                    # In-progress chapters: match by key (they have null output_file)
                    elif status == 'in_progress' and not out_file:
                        matched_info = chapter_info
                    # Normal match: output file matches expected
                    elif out_file == expected_response:
                        matched_info = chapter_info
                
                # If not found, check for composite key (chapter_num + filename)
                if not matched_info and is_special:
                    # For special files, try composite key format: "{chapter_num}_{filename_without_extension}"
                    base_name = os.path.splitext(filename)[0]
                    # Remove "response_" prefix if present in the filename
                    if base_name.startswith("response_"):
                        base_name = base_name[9:]
                    composite_key = f"{chapter_num}_{base_name}"
                    
                    if composite_key in prog.get("chapters", {}):
                        matched_info = prog["chapters"][composite_key]
                
                # Fallback: iterate through all entries matching chapter number,
                # but only accept when it clearly refers to the same source file.
                # This prevents files like "000_information.xhtml" and "0153_0.xhtml"
                # (both parsed as chapter 0) from being conflated.
                if not matched_info:
                    for chapter_key, chapter_info in prog.get("chapters", {}).items():
                        actual_num = chapter_info.get('actual_num')
                        # Also check 'chapter_num' as fallback
                        if actual_num is None:
                            actual_num = chapter_info.get('chapter_num')
                        
                        if actual_num is not None and actual_num == chapter_num:
                            orig_base = os.path.basename(chapter_info.get('original_basename', '') or '')
                            out_file = chapter_info.get('output_file')
                            status = chapter_info.get('status', '')
                            
                            # Merged chapters: match by actual_num alone (they point to parent's output)
                            # But only treat as merged if the parent chapter is actually completed
                            if status == 'merged':
                                parent_num = chapter_info.get('merged_parent_chapter')
                                parent_completed = False
                                if parent_num is not None:
                                    # Check if parent chapter is completed
                                    parent_key = str(parent_num)
                                    if parent_key in prog.get("chapters", {}):
                                        parent_info = prog["chapters"][parent_key]
                                        if parent_info.get('status') == 'completed':
                                            parent_completed = True
                                
                                if parent_completed:
                                    matched_info = chapter_info
                                    break
                                # else: don't match - will fall through to not_translated
                            
                            # In-progress chapters: match by actual_num (they have null output_file)
                            if status == 'in_progress' and not out_file:
                                matched_info = chapter_info
                                break
                            
                            # Only treat as a match if the original basename matches this filename,
                            # or, when original_basename is missing, the output_file matches what we expect.
                            if (orig_base and orig_base == filename) or (not orig_base and out_file and out_file == expected_response):
                                matched_info = chapter_info
                                break
            
            # Determine if translation file exists
            file_exists = os.path.exists(response_path)
            
            # Set status and output file based on findings
            if matched_info:
                # We found progress tracking info - use its status
                spine_ch['status'] = matched_info.get('status', 'unknown')
                
                # For special files, always use the original filename (ignore what's in progress JSON)
                if is_special:
                    spine_ch['output_file'] = expected_response
                else:
                    spine_ch['output_file'] = matched_info.get('output_file', expected_response)
                
                spine_ch['progress_entry'] = matched_info
                
                # Handle null output_file (common for failed/in_progress chapters)
                if not spine_ch['output_file']:
                    spine_ch['output_file'] = expected_response
                
                # Keep original extension (html/xhtml/htm) as written on disk
                
                # Verify file actually exists for completed status
                if spine_ch['status'] == 'completed':
                    output_path = os.path.join(output_dir, spine_ch['output_file'])
                    if not os.path.exists(output_path):
                        spine_ch['status'] = 'file_missing'
            
            elif file_exists:
                # File exists but no progress tracking - mark as completed
                spine_ch['status'] = 'completed'
                spine_ch['output_file'] = expected_response
            
            else:
                # No file and no progress tracking - not translated
                spine_ch['status'] = 'not_translated'
                spine_ch['output_file'] = expected_response
        
        # =====================================================
        # BUILD DISPLAY INFO
        # =====================================================
        
        chapter_display_info = []
        
        if spine_chapters:
            # Use OPF order
            for spine_ch in spine_chapters:
                display_info = {
                    'key': spine_ch.get('filename', ''),
                    'num': spine_ch['file_chapter_num'],
                    'info': spine_ch.get('progress_entry', {}),
                    'output_file': spine_ch['output_file'],
                    'status': spine_ch['status'],
                    'duplicate_count': 1,
                    'entries': [],
                    'opf_position': spine_ch['position'],
                    'original_filename': spine_ch['filename']
                }
                chapter_display_info.append(display_info)
        else:
            # Fallback to original logic if no OPF
            files_to_entries = {}
            for chapter_key, chapter_info in prog.get("chapters", {}).items():
                output_file = chapter_info.get("output_file", "")
                if output_file:
                    if output_file not in files_to_entries:
                        files_to_entries[output_file] = []
                    files_to_entries[output_file].append((chapter_key, chapter_info))
            
            for output_file, entries in files_to_entries.items():
                chapter_key, chapter_info = entries[0]
                
                # Check if this is a special file (files without numbers)
                original_basename = chapter_info.get("original_basename", "")
                filename_to_check = original_basename if original_basename else output_file
                
                # Check if filename contains any digits
                import re
                has_numbers = bool(re.search(r'\d', filename_to_check))
                is_special = not has_numbers
                
                # Skip special files if the toggle is off
                if is_special and not show_special_files[0]:
                    continue
                
                # Extract chapter number
                import re
                matches = re.findall(r'(\d+)', output_file)
                if matches:
                    chapter_num = int(matches[-1])
                else:
                    chapter_num = 999999
                
                # Override with stored values if available
                if 'actual_num' in chapter_info and chapter_info['actual_num'] is not None:
                    chapter_num = chapter_info['actual_num']
                elif 'chapter_num' in chapter_info and chapter_info['chapter_num'] is not None:
                    chapter_num = chapter_info['chapter_num']
                
                status = chapter_info.get("status", "unknown")
                if status == "completed_empty":
                    status = "completed"
                
                # Check file existence
                if status == "completed":
                    output_path = os.path.join(output_dir, output_file)
                    if not os.path.exists(output_path):
                        status = "file_missing"
                
                chapter_display_info.append({
                    'key': chapter_key,
                    'num': chapter_num,
                    'info': chapter_info,
                    'output_file': output_file,
                    'status': status,
                    'duplicate_count': len(entries),
                    'entries': entries
                })
            
            # Sort by chapter number
            chapter_display_info.sort(key=lambda x: x['num'] if x['num'] is not None else 999999)
        
        # =====================================================
        # CREATE UI
        # =====================================================
        
        # If no parent dialog or tab frame, create standalone dialog
        if not parent_dialog and not tab_frame:
            # Ensure QApplication exists for standalone PySide6 dialog
            from PySide6.QtWidgets import QApplication
            if not QApplication.instance():
                # Create QApplication if it doesn't exist
                import sys
                QApplication(sys.argv)
            
            # Create standalone PySide6 dialog
            dialog = QDialog()
            dialog.setWindowTitle("Force Retranslation - OPF Based" if spine_chapters else "Force Retranslation")
            # Make it stay on top so it doesn't hide behind main GUI
            dialog.setWindowFlags(dialog.windowFlags() | Qt.WindowStaysOnTopHint)
            # Use 38% width, 36% height for 1920x1080
            width, height = self._get_dialog_size(0.38, 0.36)
            dialog.resize(width, height)
            
            # Set icon
            try:
                from PySide6.QtGui import QIcon
                # Try to get base_dir from self (TranslatorGUI), fallback to calculating it
                if hasattr(self, 'base_dir'):
                    base_dir = self.base_dir
                else:
                    base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
                ico_path = os.path.join(base_dir, 'Halgakos.ico')
                if os.path.isfile(ico_path):
                    dialog.setWindowIcon(QIcon(ico_path))
            except Exception as e:
                print(f"Failed to load icon: {e}")
            dialog_layout = QVBoxLayout(dialog)
            container = QWidget()
            container_layout = QVBoxLayout(container)
            dialog_layout.addWidget(container)
        else:
            container = tab_frame or parent_dialog
            if not hasattr(container, 'layout') or container.layout() is None:
                container_layout = QVBoxLayout(container)
            else:
                container_layout = container.layout()
            dialog = parent_dialog
        
        # Title and toggle row
        title_row = QWidget()
        title_layout = QHBoxLayout(title_row)
        title_layout.setContentsMargins(0, 0, 0, 0)
        
        title_text = "Chapters from content.opf (in reading order):" if spine_chapters else "Select chapters to retranslate:"
        title_label = QLabel(title_text)
        title_font = QFont('Arial', 12 if not tab_frame else 11)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_layout.addWidget(title_label)
        
        title_layout.addStretch()
        
        # Add toggle for showing special files
        from PySide6.QtWidgets import QCheckBox
        show_special_files_cb = QCheckBox("Show special files (cover, nav, toc)")
        show_special_files_cb.setChecked(show_special_files[0])  # Preserve the current state
        show_special_files_cb.setToolTip("When enabled, shows special files (files without chapter numbers like cover, nav, toc, info, message, etc.)")
        
        # Register this checkbox and checkmark with parent dialog for cross-tab syncing
        if parent_dialog and not hasattr(parent_dialog, '_all_toggle_checkboxes'):
            parent_dialog._all_toggle_checkboxes = []
            parent_dialog._all_checkmark_labels = []
            parent_dialog._tab_file_paths = {}  # Map file_path to index
        if parent_dialog:
            # Store the index for this file
            file_key = os.path.abspath(file_path)
            if file_key not in parent_dialog._tab_file_paths:
                parent_dialog._tab_file_paths[file_key] = len(parent_dialog._all_toggle_checkboxes)
                parent_dialog._all_toggle_checkboxes.append(show_special_files_cb)
            else:
                # Replace the old checkbox at this index
                idx = parent_dialog._tab_file_paths[file_key]
                if idx < len(parent_dialog._all_toggle_checkboxes):
                    parent_dialog._all_toggle_checkboxes[idx] = show_special_files_cb
        
        # Apply blue checkbox stylesheet (matching Other Settings dialog)
        show_special_files_cb.setStyleSheet("""
            QCheckBox {
                color: white;
                spacing: 6px;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid #5a9fd4;
                border-radius: 2px;
                background-color: #2d2d2d;
            }
            QCheckBox::indicator:checked {
                background-color: #5a9fd4;
                border-color: #5a9fd4;
            }
            QCheckBox::indicator:hover {
                border-color: #7bb3e0;
            }
            QCheckBox:disabled {
                color: #666666;
            }
            QCheckBox::indicator:disabled {
                background-color: #1a1a1a;
                border-color: #3a3a3a;
            }
        """)
        
        # Create checkmark overlay for the check symbol
        checkmark = QLabel("✓", show_special_files_cb)
        checkmark.setStyleSheet("""
            QLabel {
                color: white;
                background: transparent;
                font-weight: bold;
                font-size: 11px;
            }
        """)
        checkmark.setAlignment(Qt.AlignCenter)
        checkmark.hide()
        checkmark.setAttribute(Qt.WA_TransparentForMouseEvents)
        
        def position_checkmark():
            try:
                if checkmark:
                    checkmark.setGeometry(2, 1, 14, 14)
            except RuntimeError:
                pass
        
        def update_checkmark():
            try:
                if show_special_files_cb and checkmark:
                    if show_special_files_cb.isChecked():
                        position_checkmark()
                        checkmark.show()
                    else:
                        checkmark.hide()
            except RuntimeError:
                pass
        
        show_special_files_cb.stateChanged.connect(update_checkmark)
        
        def safe_init():
            try:
                position_checkmark()
                update_checkmark()
            except RuntimeError:
                pass
        
        QTimer.singleShot(0, safe_init)
        
        # Register checkmark for cross-tab syncing
        if parent_dialog:
            file_key = os.path.abspath(file_path)
            if file_key in parent_dialog._tab_file_paths:
                idx = parent_dialog._tab_file_paths[file_key]
                # Append if new, replace if exists
                if idx >= len(parent_dialog._all_checkmark_labels):
                    parent_dialog._all_checkmark_labels.append(checkmark)
                else:
                    parent_dialog._all_checkmark_labels[idx] = checkmark
        
        title_layout.addWidget(show_special_files_cb)
        
        container_layout.addWidget(title_row)
        
        # Function to handle toggle change - will be defined after UI is created
        def on_toggle_special_files(state):
            """Rebuild the chapter list when the special files toggle is changed"""
            # Update the state variable
            show_special_files[0] = show_special_files_cb.isChecked()
            
            # For tabs in multi-file dialog, update ALL tabs and all cached states
            if tab_frame and parent_dialog:
                # Store the state persistently for ALL files in this multi-file dialog
                if not hasattr(self, '_retranslation_dialog_cache'):
                    self._retranslation_dialog_cache = {}
                
                # Update cache for all files in the current selection
                if hasattr(parent_dialog, '_epub_files_in_dialog'):
                    for f_path in parent_dialog._epub_files_in_dialog:
                        f_key = os.path.abspath(f_path)
                        if f_key not in self._retranslation_dialog_cache:
                            self._retranslation_dialog_cache[f_key] = {}
                        self._retranslation_dialog_cache[f_key]['show_special_files_state'] = show_special_files[0]
                
                # Find and update ALL toggle checkboxes and checkmarks in ALL tabs
                if hasattr(parent_dialog, '_all_toggle_checkboxes'):
                    for idx, other_checkbox in enumerate(parent_dialog._all_toggle_checkboxes):
                        if other_checkbox is None or other_checkbox == show_special_files_cb:
                            continue
                        
                        try:
                            # Try to check if widget is valid by calling a simple method
                            other_checkbox.isChecked()
                            
                            # Widget is valid, update it
                            # Block signals to avoid triggering its handler
                            other_checkbox.blockSignals(True)
                            other_checkbox.setChecked(show_special_files[0])
                            other_checkbox.blockSignals(False)
                            
                            # Update the corresponding checkmark visual
                            if hasattr(parent_dialog, '_all_checkmark_labels') and idx < len(parent_dialog._all_checkmark_labels):
                                other_checkmark = parent_dialog._all_checkmark_labels[idx]
                                if other_checkmark is not None:
                                    try:
                                        # Check if checkmark is valid
                                        other_checkmark.isVisible()
                                        
                                        # Update checkmark visibility
                                        if show_special_files[0]:
                                            other_checkmark.setGeometry(2, 1, 14, 14)
                                            other_checkmark.show()
                                        else:
                                            other_checkmark.hide()
                                    except RuntimeError:
                                        # Checkmark was deleted
                                        parent_dialog._all_checkmark_labels[idx] = None
                        except (RuntimeError, AttributeError):
                            # Widget was deleted or invalid
                            parent_dialog._all_toggle_checkboxes[idx] = None
                
                # Clear the tab frame's layout
                for i in reversed(range(container_layout.count())):
                    widget = container_layout.itemAt(i).widget()
                    if widget:
                        widget.setParent(None)
                        widget.deleteLater()
                
                # Rebuild the tab content with new toggle state
                # The rebuild will replace the checkbox/checkmark at the same index
                self._force_retranslation_epub_or_text(file_path, parent_dialog, tab_frame, show_special_files[0])
                return
            
            # For standalone dialogs - refresh in place like tabs
            # Store the state persistently
            file_key = os.path.abspath(file_path)
            if not hasattr(self, '_retranslation_dialog_cache'):
                self._retranslation_dialog_cache = {}
            if file_key not in self._retranslation_dialog_cache:
                self._retranslation_dialog_cache[file_key] = {}
            self._retranslation_dialog_cache[file_key]['show_special_files_state'] = show_special_files[0]
            
            # Refresh in place - clear and rebuild container content
            if dialog and not parent_dialog and container:
                # Temporarily disconnect the checkbox to prevent recursion
                show_special_files_cb.stateChanged.disconnect()
                
                # Store dialog position and size
                dialog_pos = dialog.pos()
                dialog_size = dialog.size()
                
                # Clear all widgets from the container
                while container_layout.count():
                    item = container_layout.takeAt(0)
                    if item:
                        widget = item.widget()
                        if widget:
                            widget.setParent(None)
                            widget.deleteLater()
                        elif item.layout():
                            # Handle nested layouts
                            self._clear_layout(item.layout())
                
                # Remove from cache to force rebuild
                if file_key in self._retranslation_dialog_cache:
                    del self._retranslation_dialog_cache[file_key]
                
                # Now we need to rebuild the content by calling the function with the existing container
                # The trick is to pass the container as if it's a tab_frame
                self._force_retranslation_epub_or_text(
                    file_path, 
                    parent_dialog=dialog,  # Pass as parent 
                    tab_frame=container,   # Use container as tab frame to rebuild in place
                    show_special_files_state=show_special_files[0]
                )
                
                # Restore dialog position and size
                dialog.move(dialog_pos)
                dialog.resize(dialog_size)
                return
        
        # Connect the checkbox to the handler
        show_special_files_cb.stateChanged.connect(on_toggle_special_files)
        
        # Statistics if OPF is available
        if spine_chapters:
            stats_frame = QWidget()
            stats_layout = QHBoxLayout(stats_frame)
            stats_layout.setContentsMargins(0, 5, 0, 5)
            container_layout.addWidget(stats_frame)
            
            total_chapters = len(spine_chapters)
            completed = sum(1 for ch in spine_chapters if ch['status'] == 'completed')
            merged = sum(1 for ch in spine_chapters if ch['status'] == 'merged')
            in_progress = sum(1 for ch in spine_chapters if ch['status'] == 'in_progress')
            missing = sum(1 for ch in spine_chapters if ch['status'] == 'not_translated')
            failed = sum(1 for ch in spine_chapters if ch['status'] in ['failed', 'qa_failed'])
            file_missing = sum(1 for ch in spine_chapters if ch['status'] == 'file_missing')
            
            stats_font = QFont('Arial', 10)
            
            lbl_total = QLabel(f"Total: {total_chapters} | ")
            lbl_total.setFont(stats_font)
            stats_layout.addWidget(lbl_total)
            
            lbl_completed = QLabel(f"✅ Completed: {completed} | ")
            lbl_completed.setFont(stats_font)
            lbl_completed.setStyleSheet("color: green;")
            stats_layout.addWidget(lbl_completed)
            
            # Merged: chapters combined into parent request (always create, hide if 0)
            lbl_merged = QLabel(f"🔗 Merged: {merged} | ")
            lbl_merged.setFont(stats_font)
            lbl_merged.setStyleSheet("color: #17a2b8;")  # Cyan/teal
            stats_layout.addWidget(lbl_merged)
            if merged == 0:
                lbl_merged.setVisible(False)
            
            # In Progress: currently being translated (always create, hide if 0)
            lbl_in_progress = QLabel(f"🔄 In Progress: {in_progress} | ")
            lbl_in_progress.setFont(stats_font)
            lbl_in_progress.setStyleSheet("color: orange;")
            stats_layout.addWidget(lbl_in_progress)
            if in_progress == 0:
                lbl_in_progress.setVisible(False)
            
            # Not Translated: unique emoji/color (distinct from failures)
            lbl_missing = QLabel(f"⬜ Not Translated: {missing} | ")
            lbl_missing.setFont(stats_font)
            lbl_missing.setStyleSheet("color: #2b6cb0;")
            stats_layout.addWidget(lbl_missing)
            
            # Match list status: failed/qa_failed use ❌ and red
            lbl_failed = QLabel(f"❌ Failed: {failed} | ")
            lbl_failed.setFont(stats_font)
            lbl_failed.setStyleSheet("color: red;")
            stats_layout.addWidget(lbl_failed)
            
            # Match list status: file_missing uses ⚠️ and purple
            lbl_file_missing = QLabel(f"⚠️ File Missing: {file_missing}")
            lbl_file_missing.setFont(stats_font)
            lbl_file_missing.setStyleSheet("color: purple;")
            stats_layout.addWidget(lbl_file_missing)
            
            stats_layout.addStretch()
        
        # Main frame for listbox
        main_frame = QWidget()
        main_layout = QVBoxLayout(main_frame)
        main_layout.setContentsMargins(10 if not tab_frame else 5, 5, 10 if not tab_frame else 5, 5)
        container_layout.addWidget(main_frame)
        
        # Create listbox (QListWidget has built-in scrollbars)
        listbox = QListWidget()
        listbox.setSelectionMode(QListWidget.ExtendedSelection)
        listbox_font = QFont('Courier', 10)  # Fixed-width font for better alignment
        listbox.setFont(listbox_font)
        # Use 36% of screen width
        min_width, _ = self._get_dialog_size(0.36, 0)
        listbox.setMinimumWidth(min_width)
        main_layout.addWidget(listbox)
        
        # Populate listbox with dynamic column widths
        status_icons = {
            'completed': '✅',
            'merged': '🔗',
            'failed': '❌',
            'qa_failed': '❌',
            'file_missing': '⚠️',
            'in_progress': '🔄',
            'not_translated': '⬜',
            'unknown': '❓'
        }
        
        status_labels = {
            'completed': 'Completed',
            'merged': 'Merged',
            'failed': 'Failed',
            'qa_failed': 'QA Failed',
            'file_missing': 'File Missing',
            'in_progress': 'In Progress',
            'not_translated': 'Not Translated',
            'unknown': 'Unknown'
        }
        
        # Calculate maximum widths for dynamic column sizing
        max_original_len = 0
        max_output_len = 0
        
        for info in chapter_display_info:
            if 'opf_position' in info:
                original_file = info.get('original_filename', '')
                output_file = info['output_file']
                max_original_len = max(max_original_len, len(original_file))
                max_output_len = max(max_output_len, len(output_file))
        
        # Set minimum widths to prevent too narrow columns
        max_original_len = max(max_original_len, 20)
        max_output_len = max(max_output_len, 25)
        
        for info in chapter_display_info:
            chapter_num = info['num']
            status = info['status']
            output_file = info['output_file']
            icon = status_icons.get(status, '❓')
            status_label = status_labels.get(status, status)
            
            # Format display with OPF info if available
            if 'opf_position' in info:
                # OPF-based display with dynamic widths
                original_file = info.get('original_filename', '')
                opf_pos = info['opf_position'] + 1  # 1-based for display
                
                # Format: [OPF Position] Chapter Number | Status | Original File -> Response File
                if isinstance(chapter_num, float) and chapter_num.is_integer():
                    display = f"[{opf_pos:03d}] Ch.{int(chapter_num):03d} | {icon} {status_label:11s} | {original_file:<{max_original_len}} -> {output_file}"
                else:
                    display = f"[{opf_pos:03d}] Ch.{chapter_num:03d} | {icon} {status_label:11s} | {original_file:<{max_original_len}} -> {output_file}"
            else:
                # Original format
                if isinstance(chapter_num, float) and chapter_num.is_integer():
                    display = f"Chapter {int(chapter_num):03d} | {icon} {status_label:11s} | {output_file}"
                elif isinstance(chapter_num, float):
                    display = f"Chapter {chapter_num:06.1f} | {icon} {status_label:11s} | {output_file}"
                else:
                    display = f"Chapter {chapter_num:03d} | {icon} {status_label:11s} | {output_file}"
            
            # Add QA issues if status is qa_failed
            if status == 'qa_failed':
                chapter_info = info.get('info', {})
                qa_issues = chapter_info.get('qa_issues_found', [])
                if qa_issues:
                    # Format issues for display (show first 2)
                    issues_display = ', '.join(qa_issues[:2])
                    if len(qa_issues) > 2:
                        issues_display += f' (+{len(qa_issues)-2} more)'
                    display += f" | {issues_display}"
            
            # Add parent chapter info if status is merged
            if status == 'merged':
                chapter_info = info.get('info', {})
                parent_chapter = chapter_info.get('merged_parent_chapter')
                if parent_chapter:
                    display += f" | → Ch.{parent_chapter}"
            
            if info.get('duplicate_count', 1) > 1:
                display += f" | ({info['duplicate_count']} entries)"
            
            item = QListWidgetItem(display)
            
            # Color code based on status
            if status == 'completed':
                item.setForeground(QColor('green'))
            elif status == 'merged':
                item.setForeground(QColor('#17a2b8'))  # Cyan/teal for merged
            elif status in ['failed', 'qa_failed']:
                item.setForeground(QColor('red'))
            elif status == 'not_translated':
                item.setForeground(QColor('#2b6cb0'))
            elif status == 'file_missing':
                item.setForeground(QColor('purple'))
            elif status == 'in_progress':
                item.setForeground(QColor('orange'))
            
            listbox.addItem(item)
        
        # Selection count label
        selection_count_label = QLabel("Selected: 0")
        selection_font = QFont('Arial', 10 if not tab_frame else 9)
        selection_count_label.setFont(selection_font)
        container_layout.addWidget(selection_count_label)
        
        def update_selection_count():
            count = len(listbox.selectedItems())
            selection_count_label.setText(f"Selected: {count}")
        
        listbox.itemSelectionChanged.connect(update_selection_count)
        
        # Return data structure for external access
        result = {
            'file_path': file_path,
            'output_dir': output_dir,
            'progress_file': progress_file,
            'prog': prog,
            'spine_chapters': spine_chapters,
            'opf_chapter_order': opf_chapter_order,
            'chapter_display_info': chapter_display_info,
            'listbox': listbox,
            'selection_count_label': selection_count_label,
            'dialog': dialog,
            'container': container,
            'show_special_files_state': show_special_files[0],  # Store current toggle state
            'show_special_files_cb': show_special_files_cb  # Store checkbox reference
        }
        
        # If standalone (no parent), add buttons and show dialog
        if not parent_dialog and not tab_frame:
            self._add_retranslation_buttons_opf(result)
            
            # Override close event to hide instead of destroy
            def closeEvent(event):
                event.ignore()  # Ignore the close event
                dialog.hide()   # Just hide the dialog
            
            dialog.closeEvent = closeEvent
            
            # Cache the dialog for reuse
            if not hasattr(self, '_retranslation_dialog_cache'):
                self._retranslation_dialog_cache = {}
            
            file_key = os.path.abspath(file_path)
            self._retranslation_dialog_cache[file_key] = result
            
            # Show the dialog (non-modal to allow interaction with other windows)
            dialog.show()
        elif not parent_dialog or tab_frame:
            # Embedded in tab - just add buttons
            self._add_retranslation_buttons_opf(result)
        
        return result


    def _add_retranslation_buttons_opf(self, data, button_frame=None):
        """Add the standard button set for retranslation dialogs with OPF support"""
        
        if not button_frame:
            button_frame = QWidget()
            button_layout = QGridLayout(button_frame)
            # Get container layout and add button frame
            container = data['container']
            if hasattr(container, 'layout') and container.layout():
                container.layout().addWidget(button_frame)
        else:
            button_layout = button_frame.layout() if button_frame.layout() else QGridLayout(button_frame)
        
        # Helper functions that work with the data dict
        def select_all():
            data['listbox'].selectAll()
            data['selection_count_label'].setText(f"Selected: {data['listbox'].count()}")
        
        def clear_selection():
            data['listbox'].clearSelection()
            data['selection_count_label'].setText("Selected: 0")
        
        def select_status(status_to_select):
            data['listbox'].clearSelection()
            for idx, info in enumerate(data['chapter_display_info']):
                if status_to_select == 'failed':
                    if info['status'] in ['failed', 'qa_failed']:
                        data['listbox'].item(idx).setSelected(True)
                elif status_to_select == 'missing':
                    if info['status'] in ['not_translated', 'file_missing']:
                        data['listbox'].item(idx).setSelected(True)
                else:
                    if info['status'] == status_to_select:
                        data['listbox'].item(idx).setSelected(True)
            count = len(data['listbox'].selectedItems())
            data['selection_count_label'].setText(f"Selected: {count}")
        
        def remove_qa_failed_mark():
            selected_items = data['listbox'].selectedItems()
            if not selected_items:
                QMessageBox.warning(data.get('dialog', self), "No Selection", "Please select at least one chapter.")
                return
            
            selected_indices = [data['listbox'].row(item) for item in selected_items]
            selected_chapters = [data['chapter_display_info'][i] for i in selected_indices]
            qa_failed_chapters = [ch for ch in selected_chapters if ch['status'] == 'qa_failed']
            
            if not qa_failed_chapters:
                QMessageBox.warning(data.get('dialog', self), "No QA Failed Chapters", 
                                     "None of the selected chapters have 'qa_failed' status.")
                return
            
            count = len(qa_failed_chapters)
            reply = QMessageBox.question(data.get('dialog', self), "Confirm Remove QA Failed Mark", 
                                      f"Remove QA failed mark from {count} chapters?",
                                      QMessageBox.Yes | QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
            
            # Remove marks
            cleared_count = 0
            for info in qa_failed_chapters:
                # Find the actual numeric key in progress by matching output_file
                target_output_file = info['output_file']
                chapter_key = None
                
                # Search through all chapters to find the one with matching output_file
                for key, ch_info in data['prog']["chapters"].items():
                    if ch_info.get('output_file') == target_output_file:
                        chapter_key = key
                        break
                
                # Update the chapter status if we found the key
                if chapter_key and chapter_key in data['prog']["chapters"]:
                    print(f"Updating chapter key {chapter_key} (output file: {target_output_file})")
                    data['prog']["chapters"][chapter_key]["status"] = "completed"
                    
                    # Remove all QA-related fields
                    fields_to_remove = ["qa_issues", "qa_timestamp", "qa_issues_found", "duplicate_confidence"]
                    for field in fields_to_remove:
                        if field in data['prog']["chapters"][chapter_key]:
                            del data['prog']["chapters"][chapter_key][field]
                    
                    cleared_count += 1
                else:
                    print(f"WARNING: Could not find chapter key for output file: {target_output_file}")
            
            # Save the updated progress
            with open(data['progress_file'], 'w', encoding='utf-8') as f:
                json.dump(data['prog'], f, ensure_ascii=False, indent=2)
            
            # Auto-refresh the display
            self._refresh_retranslation_data(data)
            
            QMessageBox.information(data.get('dialog', self), "Success", f"Removed QA failed mark from {cleared_count} chapters.")
        
        def retranslate_selected():
            selected_items = data['listbox'].selectedItems()
            if not selected_items:
                QMessageBox.warning(data.get('dialog', self), "No Selection", "Please select at least one chapter.")
                return
            
            selected_indices = [data['listbox'].row(item) for item in selected_items]
            selected_chapters = [data['chapter_display_info'][i] for i in selected_indices]
            
            # Count different types
            missing_count = sum(1 for ch in selected_chapters if ch['status'] == 'not_translated')
            existing_count = sum(1 for ch in selected_chapters if ch['status'] != 'not_translated')
            
            count = len(selected_chapters)
            if count > 10:
                if missing_count > 0 and existing_count > 0:
                    confirm_msg = f"This will:\n• Mark {missing_count} missing chapters for translation\n• Delete and retranslate {existing_count} existing chapters\n\nTotal: {count} chapters\n\nContinue?"
                elif missing_count > 0:
                    confirm_msg = f"This will mark {missing_count} missing chapters for translation.\n\nContinue?"
                else:
                    confirm_msg = f"This will delete {existing_count} translated chapters and mark them for retranslation.\n\nContinue?"
            else:
                chapters = [f"Ch.{ch['num']}" for ch in selected_chapters]
                confirm_msg = f"This will process:\n\n{', '.join(chapters)}\n\n"
                if missing_count > 0:
                    confirm_msg += f"• {missing_count} missing chapters will be marked for translation\n"
                if existing_count > 0:
                    confirm_msg += f"• {existing_count} existing chapters will be deleted and retranslated\n"
                confirm_msg += "\nContinue?"
            
            reply = QMessageBox.question(data.get('dialog', self), "Confirm Retranslation", confirm_msg,
                                       QMessageBox.Yes | QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
            
            # Process chapters - DELETE FILES AND UPDATE PROGRESS
            deleted_count = 0
            marked_count = 0
            status_reset_count = 0
            progress_updated = False

            for ch_info in selected_chapters:
                output_file = ch_info['output_file']
                
                if ch_info['status'] != 'not_translated':
                    # Delete existing file
                    if output_file:
                        output_path = os.path.join(data['output_dir'], output_file)
                        try:
                            if os.path.exists(output_path):
                                os.remove(output_path)
                                deleted_count += 1
                                print(f"Deleted: {output_path}")
                        except Exception as e:
                            print(f"Failed to delete {output_path}: {e}")
                    
                    # Reset status for any completed, completed_empty, or qa_failed chapters
                    if ch_info['status'] in ['completed', 'completed_empty', 'completed_image_only', 'qa_failed']:
                        target_output_file = ch_info['output_file']
                        chapter_key = None
                        old_status = ch_info['status']  # Define old_status before using it
                        
                        # Search through all chapters to find the one with matching output_file
                        for key, ch_data in data['prog']["chapters"].items():
                            if ch_data.get('output_file') == target_output_file:
                                chapter_key = key
                                break
                        
                        # Update the chapter status if we found the key
                        if chapter_key and chapter_key in data['prog']["chapters"]:
                            print(f"Resetting {old_status} status to pending for chapter key {chapter_key} (output file: {target_output_file})")
                            
                            # Reset status to pending for retranslation
                            data['prog']["chapters"][chapter_key]["status"] = "pending"
                            
                            # Remove completion-related fields if they exist
                            fields_to_remove = []
                            if old_status == 'qa_failed':
                                # Remove QA-related fields for qa_failed chapters
                                fields_to_remove = ["qa_issues", "qa_timestamp", "qa_issues_found", "duplicate_confidence"]
                            elif old_status == 'completed':
                                # Remove completion-related fields if any exist for completed chapters
                                fields_to_remove = ["completion_timestamp", "final_word_count", "translation_quality_score"]
                            
                            for field in fields_to_remove:
                                if field in data['prog']["chapters"][chapter_key]:
                                    del data['prog']["chapters"][chapter_key][field]
                            
                            status_reset_count += 1
                            progress_updated = True
                        else:
                            print(f"WARNING: Could not find chapter key for {old_status} output file: {target_output_file}")
                else:
                    # Just marking for translation (no file to delete)
                    marked_count += 1
            
            # Save the updated progress if we made changes
            if progress_updated:
                try:
                    with open(data['progress_file'], 'w', encoding='utf-8') as f:
                        json.dump(data['prog'], f, ensure_ascii=False, indent=2)
                    print(f"Updated progress tracking file - reset {status_reset_count} chapter statuses to pending")
                except Exception as e:
                    print(f"Failed to update progress file: {e}")
            
            # Auto-refresh the display to show updated status
            self._refresh_retranslation_data(data)
            
            # Build success message
            success_parts = []
            if deleted_count > 0:
                success_parts.append(f"Deleted {deleted_count} files")
            if marked_count > 0:
                success_parts.append(f"marked {marked_count} missing chapters for translation")
            if status_reset_count > 0:
                success_parts.append(f"reset {status_reset_count} chapter statuses to pending")
            
            if success_parts:
                success_msg = "Successfully " + ", ".join(success_parts) + "."
                if deleted_count > 0 or marked_count > 0:
                    success_msg += f"\n\nTotal {len(selected_indices)} chapters ready for translation."
                QMessageBox.information(data.get('dialog', self), "Success", success_msg)
            else:
                QMessageBox.information(data.get('dialog', self), "Info", "No changes made.")
        
        # Add buttons - First row
        btn_select_all = QPushButton("Select All")
        btn_select_all.setMinimumHeight(32)
        btn_select_all.setStyleSheet("QPushButton { background-color: #17a2b8; color: white; padding: 6px 16px; font-weight: bold; font-size: 10pt; }")
        btn_select_all.clicked.connect(select_all)
        button_layout.addWidget(btn_select_all, 0, 0)
        
        btn_clear = QPushButton("Clear")
        btn_clear.setMinimumHeight(32)
        btn_clear.setStyleSheet("QPushButton { background-color: #6c757d; color: white; padding: 6px 16px; font-weight: bold; font-size: 10pt; }")
        btn_clear.clicked.connect(clear_selection)
        button_layout.addWidget(btn_clear, 0, 1)
        
        btn_select_completed = QPushButton("Select Completed")
        btn_select_completed.setMinimumHeight(32)
        btn_select_completed.setStyleSheet("QPushButton { background-color: #28a745; color: white; padding: 6px 16px; font-weight: bold; font-size: 10pt; }")
        btn_select_completed.clicked.connect(lambda: select_status('completed'))
        button_layout.addWidget(btn_select_completed, 0, 2)
        
        btn_select_missing = QPushButton("Select Missing")
        btn_select_missing.setMinimumHeight(32)
        # Use amber for Not Translated / Missing (distinct from failures)
        btn_select_missing.setStyleSheet("QPushButton { background-color: #d39e00; color: white; padding: 6px 16px; font-weight: bold; font-size: 10pt; }")
        btn_select_missing.clicked.connect(lambda: select_status('missing'))
        button_layout.addWidget(btn_select_missing, 0, 3)
        
        btn_select_failed = QPushButton("Select Failed")
        btn_select_failed.setMinimumHeight(32)
        # Use red for Failed / QA Failed
        btn_select_failed.setStyleSheet("QPushButton { background-color: #dc3545; color: white; padding: 6px 16px; font-weight: bold; font-size: 10pt; }")
        btn_select_failed.clicked.connect(lambda: select_status('failed'))
        button_layout.addWidget(btn_select_failed, 0, 4)
        
        # Second row
        btn_retranslate = QPushButton("Retranslate Selected")
        btn_retranslate.setMinimumHeight(32)
        btn_retranslate.setStyleSheet("QPushButton { background-color: #d39e00; color: white; padding: 6px 16px; font-weight: bold; font-size: 10pt; }")
        btn_retranslate.clicked.connect(retranslate_selected)
        button_layout.addWidget(btn_retranslate, 1, 0, 1, 2)
        
        btn_remove_qa = QPushButton("Remove QA Failed Mark")
        btn_remove_qa.setMinimumHeight(32)
        btn_remove_qa.setStyleSheet("QPushButton { background-color: #28a745; color: white; padding: 6px 16px; font-weight: bold; font-size: 10pt; }")
        btn_remove_qa.clicked.connect(remove_qa_failed_mark)
        button_layout.addWidget(btn_remove_qa, 1, 2, 1, 1)
        
        # Add animated refresh button
        btn_refresh = AnimatedRefreshButton("  Refresh")  # Double space for icon padding
        btn_refresh.setMinimumHeight(32)
        btn_refresh.setStyleSheet(
            "QPushButton { "
            "background-color: #17a2b8; "
            "color: white; "
            "padding: 6px 16px; "
            "font-weight: bold; "
            "font-size: 10pt; "
            "}"
        )
        
        # Create refresh handler with animation
        def animated_refresh():
            import time
            btn_refresh.start_animation()
            btn_refresh.setEnabled(False)
            
            # Track start time for minimum animation duration
            start_time = time.time()
            min_animation_duration = 0.8  # 800ms minimum
            
            # Use QTimer to run refresh after animation starts
            def do_refresh():
                try:
                    # Check if this is part of a multi-tab dialog and refresh all tabs, otherwise just refresh current
                    if data.get('dialog') and hasattr(data['dialog'], '_tab_data'):
                        self._refresh_all_tabs(data['dialog']._tab_data)
                    else:
                        self._refresh_retranslation_data(data)
                    
                    # Calculate remaining time to meet minimum animation duration
                    elapsed = time.time() - start_time
                    remaining = max(0, min_animation_duration - elapsed)
                    
                    # Schedule animation stop after remaining time
                    def finish_animation():
                        btn_refresh.stop_animation()
                        btn_refresh.setEnabled(True)
                    
                    if remaining > 0:
                        QTimer.singleShot(int(remaining * 1000), finish_animation)
                    else:
                        finish_animation()
                        
                except Exception as e:
                    print(f"Error during refresh: {e}")
                    btn_refresh.stop_animation()
                    btn_refresh.setEnabled(True)
            
            QTimer.singleShot(50, do_refresh)  # Small delay to let animation start
        
        btn_refresh.clicked.connect(animated_refresh)
        button_layout.addWidget(btn_refresh, 1, 3, 1, 1)
        
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setMinimumHeight(32)
        btn_cancel.setStyleSheet("QPushButton { background-color: #6c757d; color: white; padding: 6px 16px; font-weight: bold; font-size: 10pt; }")
        btn_cancel.clicked.connect(lambda: data['dialog'].close() if data.get('dialog') else None)
        button_layout.addWidget(btn_cancel, 1, 4, 1, 1)

    def _refresh_all_tabs(self, tab_data_list):
        """Refresh all tabs in a multi-file retranslation dialog"""
        try:
            print(f"🔄 Refreshing all {len(tab_data_list)} tabs...")
            
            refreshed_count = 0
            skipped_count = 0
            for idx, data in enumerate(tab_data_list):
                if data and data.get('type') != 'image_folder' and data.get('type') != 'individual_images':
                    # Only refresh EPUB/text tabs
                    try:
                        # Check if widgets are still valid before attempting refresh
                        if not self._is_data_valid(data):
                            print(f"[DEBUG] Skipping tab {idx + 1}/{len(tab_data_list)} - widgets deleted")
                            skipped_count += 1
                            continue
                        
                        print(f"[DEBUG] Refreshing tab {idx + 1}/{len(tab_data_list)}")
                        self._refresh_retranslation_data(data)
                        refreshed_count += 1
                    except RuntimeError as e:
                        # Widget was deleted
                        print(f"[WARN] Skipping tab {idx + 1} - widget deleted: {e}")
                        skipped_count += 1
                    except Exception as e:
                        print(f"[ERROR] Failed to refresh tab {idx + 1}: {e}")
            
            if skipped_count > 0:
                print(f"✅ Successfully refreshed {refreshed_count} tab(s), skipped {skipped_count} deleted tab(s)")
            else:
                print(f"✅ Successfully refreshed {refreshed_count} tab(s)")
            
        except Exception as e:
            print(f"❌ Failed to refresh all tabs: {e}")
            import traceback
            traceback.print_exc()
    
    def _is_data_valid(self, data):
        """Check if the data structure has valid (non-deleted) widgets"""
        try:
            if not data:
                return False
            
            # Check if listbox exists and is still valid
            listbox = data.get('listbox')
            if not listbox:
                return False
            
            # Try to access a simple property to check if widget is still alive
            # This will raise RuntimeError if the C++ object was deleted
            listbox.count()
            return True
            
        except (RuntimeError, AttributeError):
            return False
    
    def _refresh_retranslation_data(self, data):
        """Refresh the retranslation dialog data by reloading progress and updating display"""
        try:
            # First check if widgets are still valid
            if not self._is_data_valid(data):
                print("⚠️ Cannot refresh - widgets have been deleted")
                return
            
            # print("🔄 Refreshing retranslation data...")
            
            # Save current selections to restore after refresh
            selected_indices = []
            try:
                selected_indices = [data['listbox'].row(item) for item in data['listbox'].selectedItems()]
            except RuntimeError:
                print("⚠️ Could not save selection state - widget was deleted")
                return
            
            # Reload progress file - check if it exists first
            if not os.path.exists(data['progress_file']):
                print(f"⚠️ Progress file not found: {data['progress_file']}")
                QMessageBox.information(data.get('dialog', self), "Output Folder Not Found", 
                                      f"The output folder appears to have been deleted.\n\n"
                                      f"Progress file not found:\n{data['progress_file']}")
                return
            
            with open(data['progress_file'], 'r', encoding='utf-8') as f:
                data['prog'] = json.load(f)
            
            # Re-parse and update chapter status information
            self._update_chapter_status_info(data)
            
            # Update the listbox display
            self._update_listbox_display(data)
            
            # Update statistics if available
            self._update_statistics_display(data)
            
            # Restore selections
            try:
                if selected_indices:
                    for idx in selected_indices:
                        if idx < data['listbox'].count():
                            data['listbox'].item(idx).setSelected(True)
                    # Update selection count
                    if 'selection_count_label' in data and data['selection_count_label']:
                        data['selection_count_label'].setText(f"Selected: {len(selected_indices)}")
                else:
                    # Clear selections if there were none
                    data['listbox'].clearSelection()
                    if 'selection_count_label' in data and data['selection_count_label']:
                        data['selection_count_label'].setText("Selected: 0")
            except RuntimeError:
                print("⚠️ Could not restore selection state - widget was deleted during refresh")
            
            # print("✅ Retranslation data refreshed successfully")
            
        except RuntimeError as e:
            print(f"❌ Failed to refresh data - widget deleted: {e}")
        except FileNotFoundError as e:
            print(f"❌ Failed to refresh data - file not found: {e}")
            try:
                QMessageBox.information(data.get('dialog', self), "Output Folder Not Found", 
                                      f"The output folder appears to have been deleted or moved.\n\n"
                                      f"File not found: {os.path.basename(str(e))}")
            except (RuntimeError, AttributeError):
                print(f"[WARN] Could not show error dialog - dialog was deleted")
        except Exception as e:
            print(f"❌ Failed to refresh data: {e}")
            import traceback
            traceback.print_exc()
            try:
                # Show friendlier error message for common cases
                error_msg = str(e)
                if "No such file or directory" in error_msg or "cannot find the path" in error_msg:
                    QMessageBox.information(data.get('dialog', self), "Output Folder Not Found", 
                                          f"The output folder appears to have been deleted or moved.\n\n"
                                          f"Error: {error_msg}")
                else:
                    QMessageBox.warning(data.get('dialog', self), "Refresh Failed", 
                                      f"Failed to refresh data: {error_msg}")
            except (RuntimeError, AttributeError):
                # Dialog was also deleted, just print to console
                print(f"[WARN] Could not show error dialog - dialog was deleted")
    
    def _update_chapter_status_info(self, data):
        """Update chapter status information after refresh"""
        # Re-check file existence and update status for each chapter
        for info in data['chapter_display_info']:
            output_file = info['output_file']
            output_path = os.path.join(data['output_dir'], output_file)
            
            # Find matching progress entry
            matched_info = None
            
            # PRIORITY 1: Try to match by actual_num first (most reliable for merged chapters)
            # This prevents merged chapters from matching the parent's entry by output_file
            for chapter_key, chapter_info in data['prog'].get("chapters", {}).items():
                actual_num = chapter_info.get('actual_num') or chapter_info.get('chapter_num')
                if actual_num is not None and actual_num == info['num']:
                    matched_info = chapter_info
                    break
            
            # PRIORITY 2: Fall back to output_file matching if no actual_num match
            if not matched_info:
                for chapter_key, chapter_info in data['prog'].get("chapters", {}).items():
                    if chapter_info.get('output_file') == output_file:
                        matched_info = chapter_info
                        break
            
            # Update status based on current state from progress file
            if matched_info:
                new_status = matched_info.get('status', 'unknown')
                # Handle completed_empty as completed for display
                if new_status == 'completed_empty':
                    new_status = 'completed'
                # Verify file actually exists for completed status (but NOT for merged - merged chapters
                # don't have their own output files, they point to parent's file)
                if new_status == 'completed' and not os.path.exists(output_path):
                    new_status = 'file_missing'
                info['status'] = new_status
                info['info'] = matched_info
            elif os.path.exists(output_path):
                # Before marking as completed based on file existence, check if this chapter
                # is actually marked as merged in the progress file (by actual_num lookup)
                # This handles the case where old output files exist from before merging was enabled
                is_merged_chapter = False
                for chapter_key, chapter_info in data['prog'].get("chapters", {}).items():
                    actual_num = chapter_info.get('actual_num') or chapter_info.get('chapter_num')
                    if actual_num is not None and actual_num == info['num']:
                        if chapter_info.get('status') == 'merged':
                            is_merged_chapter = True
                            info['status'] = 'merged'
                            info['info'] = chapter_info
                            break
                
                if not is_merged_chapter:
                    info['status'] = 'completed'
            else:
                info['status'] = 'not_translated'
    
    def _update_listbox_display(self, data):
        """Update the listbox display with current chapter information"""
        listbox = data['listbox']
        
        # Clear existing items
        listbox.clear()
        
        # Status icons and labels
        status_icons = {
            'completed': '✅',
            'merged': '🔗',
            'failed': '❌',
            'qa_failed': '❌',
            'file_missing': '⚠️',
            'in_progress': '🔄',
            'not_translated': '⬜',
            'unknown': '❓'
        }
        
        status_labels = {
            'completed': 'Completed',
            'merged': 'Merged',
            'failed': 'Failed',
            'qa_failed': 'QA Failed',
            'file_missing': 'File Missing',
            'in_progress': 'In Progress',
            'not_translated': 'Not Translated',
            'unknown': 'Unknown'
        }
        
        # Calculate maximum widths for dynamic column sizing
        max_original_len = 0
        max_output_len = 0
        
        for info in data['chapter_display_info']:
            if 'opf_position' in info:
                original_file = info.get('original_filename', '')
                output_file = info['output_file']
                max_original_len = max(max_original_len, len(original_file))
                max_output_len = max(max_output_len, len(output_file))
        
        # Set minimum widths to prevent too narrow columns
        max_original_len = max(max_original_len, 20)
        max_output_len = max(max_output_len, 25)
        
        # Rebuild listbox items
        for info in data['chapter_display_info']:
            chapter_num = info['num']
            status = info['status']
            output_file = info['output_file']
            icon = status_icons.get(status, '❓')
            status_label = status_labels.get(status, status)
            
            # Format display with OPF info if available
            if 'opf_position' in info:
                # OPF-based display with dynamic widths
                original_file = info.get('original_filename', '')
                opf_pos = info['opf_position'] + 1  # 1-based for display
                
                # Format: [OPF Position] Chapter Number | Status | Original File -> Response File
                if isinstance(chapter_num, float) and chapter_num.is_integer():
                    display = f"[{opf_pos:03d}] Ch.{int(chapter_num):03d} | {icon} {status_label:11s} | {original_file:<{max_original_len}} -> {output_file}"
                else:
                    display = f"[{opf_pos:03d}] Ch.{chapter_num:03d} | {icon} {status_label:11s} | {original_file:<{max_original_len}} -> {output_file}"
            else:
                # Original format
                if isinstance(chapter_num, float) and chapter_num.is_integer():
                    display = f"Chapter {int(chapter_num):03d} | {icon} {status_label:11s} | {output_file}"
                elif isinstance(chapter_num, float):
                    display = f"Chapter {chapter_num:06.1f} | {icon} {status_label:11s} | {output_file}"
                else:
                    display = f"Chapter {chapter_num:03d} | {icon} {status_label:11s} | {output_file}"
            
            # Add QA issues if status is qa_failed
            if status == 'qa_failed':
                chapter_info = info.get('info', {})
                qa_issues = chapter_info.get('qa_issues_found', [])
                if qa_issues:
                    # Format issues for display (show first 2)
                    issues_display = ', '.join(qa_issues[:2])
                    if len(qa_issues) > 2:
                        issues_display += f' (+{len(qa_issues)-2} more)'
                    display += f" | {issues_display}"
            
            # Add parent chapter info if status is merged
            if status == 'merged':
                chapter_info = info.get('info', {})
                parent_chapter = chapter_info.get('merged_parent_chapter')
                if parent_chapter:
                    display += f" | → Ch.{parent_chapter}"
            
            if info.get('duplicate_count', 1) > 1:
                display += f" | ({info['duplicate_count']} entries)"
            
            from PySide6.QtWidgets import QListWidgetItem
            from PySide6.QtGui import QColor
            item = QListWidgetItem(display)
            
            # Color code based on status
            if status == 'completed':
                item.setForeground(QColor('green'))
            elif status == 'merged':
                item.setForeground(QColor('#17a2b8'))  # Cyan/teal for merged
            elif status in ['failed', 'qa_failed']:
                item.setForeground(QColor('red'))
            elif status == 'not_translated':
                item.setForeground(QColor('#2b6cb0'))
            elif status == 'file_missing':
                item.setForeground(QColor('purple'))
            elif status == 'in_progress':
                item.setForeground(QColor('orange'))
            
            listbox.addItem(item)
    
    def _update_statistics_display(self, data):
        """Update statistics display if spine chapters are available"""
        if not data.get('spine_chapters'):
            return
        
        # Find statistics labels in the container
        container = data['container']
        
        # Search for statistics labels by traversing the widget hierarchy
        def find_stats_labels(widget):
            labels = {}
            if hasattr(widget, 'children'):
                for child in widget.children():
                    if hasattr(child, 'text'):
                        text = child.text()
                        if text.startswith('Total:'):
                            labels['total'] = child
                        elif text.startswith('✅ Completed:'):
                            labels['completed'] = child
                        elif text.startswith('🔗 Merged:'):
                            labels['merged'] = child
                        elif text.startswith('🔄 In Progress:'):
                            labels['in_progress'] = child
                        elif text.startswith('⬜ Not Translated:'):
                            labels['missing'] = child
                        elif text.startswith('❌ Failed:'):
                            labels['failed'] = child
                        elif text.startswith('⚠️ File Missing:'):
                            labels['file_missing'] = child
                    
                    # Recursively search children
                    labels.update(find_stats_labels(child))
            return labels
        
        stats_labels = find_stats_labels(container)
        
        if stats_labels:
            # Recalculate statistics
            spine_chapters = data['spine_chapters']
            total_chapters = len(spine_chapters)
            completed = sum(1 for info in data['chapter_display_info'] if info['status'] == 'completed')
            merged = sum(1 for info in data['chapter_display_info'] if info['status'] == 'merged')
            in_progress = sum(1 for info in data['chapter_display_info'] if info['status'] == 'in_progress')
            missing = sum(1 for info in data['chapter_display_info'] if info['status'] == 'not_translated')
            failed = sum(1 for info in data['chapter_display_info'] if info['status'] in ['failed', 'qa_failed'])
            file_missing = sum(1 for info in data['chapter_display_info'] if info['status'] == 'file_missing')
            
            # Update labels
            if 'total' in stats_labels:
                stats_labels['total'].setText(f"Total: {total_chapters} | ")
            if 'completed' in stats_labels:
                stats_labels['completed'].setText(f"✅ Completed: {completed} | ")
            if 'merged' in stats_labels:
                if merged > 0:
                    stats_labels['merged'].setText(f"🔗 Merged: {merged} | ")
                    stats_labels['merged'].setVisible(True)
                else:
                    stats_labels['merged'].setVisible(False)
            if 'in_progress' in stats_labels:
                if in_progress > 0:
                    stats_labels['in_progress'].setText(f"🔄 In Progress: {in_progress} | ")
                    stats_labels['in_progress'].setVisible(True)
                else:
                    stats_labels['in_progress'].setVisible(False)
            if 'missing' in stats_labels:
                stats_labels['missing'].setText(f"⬜ Not Translated: {missing} | ")
            if 'failed' in stats_labels:
                stats_labels['failed'].setText(f"❌ Failed: {failed} | ")
            if 'file_missing' in stats_labels:
                stats_labels['file_missing'].setText(f"⚠️ File Missing: {file_missing}")


    def _force_retranslation_multiple_files(self):
        """Handle force retranslation when multiple files are selected - now uses shared logic"""
        try:
            print(f"[DEBUG] _force_retranslation_multiple_files called with {len(self.selected_files)} files")
            
            # First, check if all selected files are images from the same folder
            # This handles the case where folder selection results in individual file selections
            if len(self.selected_files) > 1:
                all_images = True
                parent_dirs = set()
                
                image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')
                
                for file_path in self.selected_files:
                    if os.path.isfile(file_path) and file_path.lower().endswith(image_extensions):
                        parent_dirs.add(os.path.dirname(file_path))
                    else:
                        all_images = False
                        break
                
                # If all files are images from the same directory, treat it as a folder selection
                if all_images and len(parent_dirs) == 1:
                    folder_path = parent_dirs.pop()
                    print(f"[DEBUG] Detected {len(self.selected_files)} images from same folder: {folder_path}")
                    print(f"[DEBUG] Treating as folder selection")
                    self._force_retranslation_images_folder(folder_path)
                    return
            
            # Otherwise, continue with normal categorization
            epub_files = []
            text_files = []
            image_files = []
            folders = []
            
            image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')
            
            for file_path in self.selected_files:
                if os.path.isdir(file_path):
                    folders.append(file_path)
                elif file_path.lower().endswith('.epub'):
                    epub_files.append(file_path)
                elif file_path.lower().endswith('.txt'):
                    text_files.append(file_path)
                elif file_path.lower().endswith(image_extensions):
                    image_files.append(file_path)
            
            # Build summary
            summary_parts = []
            if epub_files:
                summary_parts.append(f"{len(epub_files)} EPUB file(s)")
            if text_files:
                summary_parts.append(f"{len(text_files)} text file(s)")
            if image_files:
                summary_parts.append(f"{len(image_files)} image file(s)")
            if folders:
                summary_parts.append(f"{len(folders)} folder(s)")
            
            if not summary_parts:
                QMessageBox.information(self, "Info", "No valid files selected.")
                return
            
            # Create a unique key for the current selection
            selection_key = tuple(sorted(self.selected_files))
            
            # Check if we already have a cached dialog for this exact selection
            if (hasattr(self, '_multi_file_retranslation_dialog') and 
                self._multi_file_retranslation_dialog and 
                hasattr(self, '_multi_file_selection_key') and 
                self._multi_file_selection_key == selection_key):
                # Reuse existing dialog - refresh all tabs before showing
                cached_dialog = self._multi_file_retranslation_dialog
                if hasattr(cached_dialog, '_tab_data') and cached_dialog._tab_data:
                    print(f"[DEBUG] Refreshing all {len(cached_dialog._tab_data)} tabs in cached dialog...")
                    self._refresh_all_tabs(cached_dialog._tab_data)
                cached_dialog.show()
                cached_dialog.raise_()
                cached_dialog.activateWindow()
                return
            
            # If there's an existing dialog for a different selection, destroy it first
            if hasattr(self, '_multi_file_retranslation_dialog') and self._multi_file_retranslation_dialog:
                self._multi_file_retranslation_dialog.close()
                self._multi_file_retranslation_dialog.deleteLater()
                self._multi_file_retranslation_dialog = None
            
            # Create main dialog
            dialog = QDialog(self)
            dialog.setWindowTitle("Force Retranslation - Multiple Files")
            # Store the list of EPUBs in the dialog for cross-tab state updates
            dialog._epub_files_in_dialog = epub_files + text_files
            # Increased height from 18% to 25% for better visibility
            width, height = self._get_dialog_size(0.25, 0.45)
            dialog.resize(width, height)
            
            # Set icon
            try:
                from PySide6.QtGui import QIcon
                if hasattr(self, 'base_dir'):
                    base_dir = self.base_dir
                else:
                    base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
                ico_path = os.path.join(base_dir, 'Halgakos.ico')
                if os.path.isfile(ico_path):
                    dialog.setWindowIcon(QIcon(ico_path))
            except Exception as e:
                print(f"Failed to load icon: {e}")
            
            dialog_layout = QVBoxLayout(dialog)
            
            # Summary label
            summary_label = QLabel(f"Selected: {', '.join(summary_parts)}")
            summary_font = QFont('Arial', 12)
            summary_font.setBold(True)
            summary_label.setFont(summary_font)
            dialog_layout.addWidget(summary_label)
            
            # Create tab widget with custom styling
            notebook = QTabWidget()
            notebook.setStyleSheet("""
                QTabWidget::pane {
                    border: 2px solid #5a9fd4;
                    border-radius: 4px;
                    background-color: #2d2d2d;
                }
                QTabBar::tab {
                    background-color: #3a3a3a;
                    color: white;
                    padding: 8px 16px;
                    margin-right: 2px;
                    border: 1px solid #5a9fd4;
                    border-bottom: none;
                    border-top-left-radius: 4px;
                    border-top-right-radius: 4px;
                    font-weight: bold;
                    font-size: 11pt;
                }
                QTabBar::tab:selected {
                    background-color: #5a9fd4;
                    color: white;
                }
                QTabBar::tab:hover {
                    background-color: #4a8fc4;
                }
            """)
            dialog_layout.addWidget(notebook)
            
            # Track all tab data
            tab_data = []
            tabs_created = False
            
            # Store tab_data reference on the dialog for cross-tab operations
            dialog._tab_data = tab_data
            
            # Get the global show_special state from the first file that has it cached, or default to False
            global_show_special = False
            for file_path in epub_files + text_files:
                file_key = os.path.abspath(file_path)
                if hasattr(self, '_retranslation_dialog_cache') and file_key in self._retranslation_dialog_cache:
                    cached_data = self._retranslation_dialog_cache[file_key]
                    if cached_data and 'show_special_files_state' in cached_data:
                        global_show_special = cached_data['show_special_files_state']
                        break  # Use the first one we find
            
            # Create tabs for EPUB/text files using shared logic
            for file_path in epub_files + text_files:
                file_base = os.path.splitext(os.path.basename(file_path))[0]
                
                print(f"[DEBUG] Checking EPUB/text: {file_base}")
                
                # Quick check if output exists
                if not os.path.exists(file_base):
                    print(f"[DEBUG] Skipping {file_base} - output folder doesn't exist")
                    continue
                
                print(f"[DEBUG] Creating tab for {file_base}")
                
                # Create tab
                tab_frame = QWidget()
                tab_layout = QVBoxLayout(tab_frame)
                tab_name = file_base[:20] + "..." if len(file_base) > 20 else file_base
                
                # Use shared logic to populate the tab with global state
                tab_result = self._force_retranslation_epub_or_text(
                    file_path, 
                    parent_dialog=dialog, 
                    tab_frame=tab_frame,
                    show_special_files_state=global_show_special
                )
                
                # Only add the tab if content was successfully created
                if tab_result:
                    notebook.addTab(tab_frame, tab_name)
                    tab_data.append(tab_result)
                    tabs_created = True
                    print(f"[DEBUG] Successfully created tab for {file_base}")
                else:
                    print(f"[DEBUG] Failed to create content for {file_base}")
            
            # Create tabs for image folders (keeping existing logic for now)
            for folder_path in folders:
                folder_result = self._create_image_folder_tab(
                    folder_path, 
                    notebook, 
                    dialog
                )
                if folder_result:
                    tab_data.append(folder_result)
                    tabs_created = True
            
            # If only individual image files selected and no tabs created yet
            if image_files and not tabs_created:
                # Create a single tab for all individual images
                image_tab_result = self._create_individual_images_tab(
                    image_files,
                    notebook,
                    dialog
                )
                if image_tab_result:
                    tab_data.append(image_tab_result)
                    tabs_created = True
            
            # If no tabs were created from folders, try scanning folders for individual images
            if not tabs_created and folders:
                # Scan folders for individual image files
                scanned_images = []
                for folder_path in folders:
                    if os.path.isdir(folder_path):
                        try:
                            for file in os.listdir(folder_path):
                                file_path = os.path.join(folder_path, file)
                                if os.path.isfile(file_path) and file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                                    scanned_images.append(file_path)
                        except:
                            pass
                
                # If we found images, create a tab for them
                if scanned_images:
                    image_tab_result = self._create_individual_images_tab(
                        scanned_images,
                        notebook,
                        dialog
                    )
                    if image_tab_result:
                        tab_data.append(image_tab_result)
                        tabs_created = True
            
            # If still no tabs were created, show error
            if not tabs_created:
                QMessageBox.information(self, "Info", 
                    "No translation output found for any of the selected files.\n\n"
                    "Make sure the output folders exist in your script directory.")
                dialog.close()
                return
        
            # Add unified button bar that works across all tabs
            self._add_multi_file_buttons(dialog, notebook, tab_data)
            
            # Override close event to minimize instead of destroy
            def closeEvent(event):
                event.ignore()  # Ignore the close event
                dialog.hide()   # Just hide (minimize) the dialog
            
            dialog.closeEvent = closeEvent
            
            # Cache the dialog and selection key for reuse
            self._multi_file_retranslation_dialog = dialog
            self._multi_file_selection_key = selection_key
            
            # Refresh all tabs before showing the dialog
            if tab_data:
                print(f"[DEBUG] Refreshing all {len(tab_data)} tabs on dialog open...")
                self._refresh_all_tabs(tab_data)
            else:
                print(f"[WARN] No tab data to refresh on dialog open")
            
            # Show the dialog (non-modal to allow interaction with other windows)
            dialog.show()
            
        except Exception as e:
            print(f"[ERROR] _force_retranslation_multiple_files failed: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to open retranslation dialog:\n{str(e)}")

    def _add_multi_file_buttons(self, dialog, notebook, tab_data):
        """Placeholder for future multi-file button functionality"""
        # No buttons needed - dialog has standard close button
        pass
              
    def _create_individual_images_tab(self, image_files, notebook, parent_dialog):
        """Create a tab for individual image files"""
        # Create tab
        tab_frame = QWidget()
        tab_layout = QVBoxLayout(tab_frame)
        notebook.addTab(tab_frame, "Individual Images")
        
        # Instructions
        instruction_label = QLabel(f"Selected {len(image_files)} individual image(s):")
        instruction_font = QFont('Arial', 11)
        instruction_label.setFont(instruction_font)
        tab_layout.addWidget(instruction_label)
        
        # Listbox (QListWidget has built-in scrolling)
        listbox = QListWidget()
        listbox.setSelectionMode(QListWidget.ExtendedSelection)
        # Use 16% of screen width (half of original ~31% for 1920px screen)
        min_width, _ = self._get_dialog_size(0.16, 0)
        listbox.setMinimumWidth(min_width)
        tab_layout.addWidget(listbox)
        
        # File info
        file_info = []
        script_dir = os.getcwd()
        
        # Check each image for translations
        for img_path in sorted(image_files):
            img_name = os.path.basename(img_path)
            base_name = os.path.splitext(img_name)[0]
            
            # Look for translations in various possible locations
            found_translations = []
            
            # Check in script directory with base name
            possible_dirs = [
                os.path.join(script_dir, base_name),
                os.path.join(script_dir, f"{base_name}_translated"),
                base_name,
                f"{base_name}_translated"
            ]
            
            for output_dir in possible_dirs:
                if os.path.exists(output_dir) and os.path.isdir(output_dir):
                    # Look for HTML files
                    for file in os.listdir(output_dir):
                        if file.lower().endswith(('.html', '.xhtml', '.htm')) and base_name in file:
                            found_translations.append((output_dir, file))
            
            if found_translations:
                for output_dir, html_file in found_translations:
                    display = f"📄 {img_name} → {html_file} | ✅ Translated"
                    listbox.addItem(display)
                    
                    file_info.append({
                        'type': 'translated',
                        'source_image': img_path,
                        'output_dir': output_dir,
                        'file': html_file,
                        'path': os.path.join(output_dir, html_file)
                    })
            else:
                display = f"🖼️ {img_name} | ❌ No translation found"
                listbox.addItem(display)
        
        # Selection count
        selection_count_label = QLabel("Selected: 0")
        selection_font = QFont('Arial', 9)
        selection_count_label.setFont(selection_font)
        tab_layout.addWidget(selection_count_label)
        
        def update_selection_count():
            count = len(listbox.selectedItems())
            selection_count_label.setText(f"Selected: {count}")
        
        listbox.itemSelectionChanged.connect(update_selection_count)
        
        return {
            'type': 'individual_images',
            'listbox': listbox,
            'file_info': file_info,
            'selection_count_label': selection_count_label
        }


    def _create_image_folder_tab(self, folder_path, notebook, parent_dialog):
        """Create a tab for image folder retranslation"""
        folder_name = os.path.basename(folder_path)
        output_dir = f"{folder_name}_translated"
        
        if not os.path.exists(output_dir):
            return None
        
        # Create tab
        tab_frame = QWidget()
        tab_layout = QVBoxLayout(tab_frame)
        tab_name = "📁 " + (folder_name[:17] + "..." if len(folder_name) > 17 else folder_name)
        notebook.addTab(tab_frame, tab_name)
        
        # Instructions
        instruction_label = QLabel("Select images to retranslate:")
        instruction_font = QFont('Arial', 11)
        instruction_label.setFont(instruction_font)
        tab_layout.addWidget(instruction_label)
        
        # Listbox (QListWidget has built-in scrolling)
        listbox = QListWidget()
        listbox.setSelectionMode(QListWidget.ExtendedSelection)
        # Use 16% of screen width (half of original ~31% for 1920px screen)
        min_width, _ = self._get_dialog_size(0.16, 0)
        listbox.setMinimumWidth(min_width)
        tab_layout.addWidget(listbox)
        
        # Find files
        file_info = []
        
        # Add HTML files
        for file in os.listdir(output_dir):
            if file.startswith('response_'):
                # Allow response_{index}_{name}.html and compound extensions like .html.xhtml
                match = re.match(r'^response_(\d+)_([^\.]*)\.(?:html?|xhtml|htm)(?:\.xhtml)?$', file, re.IGNORECASE)
                if match:
                    index = match.group(1)
                    base_name = match.group(2)
                    display = f"📄 Image {index} | {base_name} | ✅ Translated"
                else:
                    display = f"📄 {file} | ✅ Translated"
                
                listbox.addItem(display)
                file_info.append({
                    'type': 'translated',
                    'file': file,
                    'path': os.path.join(output_dir, file)
                })
        
        # Add cover images
        images_dir = os.path.join(output_dir, "images")
        if os.path.exists(images_dir):
            for file in sorted(os.listdir(images_dir)):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                    display = f"🖼️ Cover | {file} | ⏭️ Skipped"
                    listbox.addItem(display)
                    file_info.append({
                        'type': 'cover',
                        'file': file,
                        'path': os.path.join(images_dir, file)
                    })
        
        # Selection count
        selection_count_label = QLabel("Selected: 0")
        selection_font = QFont('Arial', 9)
        selection_count_label.setFont(selection_font)
        tab_layout.addWidget(selection_count_label)
        
        def update_selection_count():
            count = len(listbox.selectedItems())
            selection_count_label.setText(f"Selected: {count}")
        
        listbox.itemSelectionChanged.connect(update_selection_count)
        
        return {
            'type': 'image_folder',
            'folder_path': folder_path,
            'output_dir': output_dir,
            'listbox': listbox,
            'file_info': file_info,
            'selection_count_label': selection_count_label
        }


    def _force_retranslation_images_folder(self, folder_path):
        """Handle force retranslation for image folders"""
        # If folder_path is actually a file (single image), get its directory
        if os.path.isfile(folder_path):
            # Single image file - use basename without extension
            folder_name = os.path.splitext(os.path.basename(folder_path))[0]
        else:
            # Folder - use folder name as-is
            folder_name = os.path.basename(folder_path)
        
        # Check if we already have a cached dialog for this folder
        folder_key = os.path.abspath(folder_path)
        if hasattr(self, '_image_retranslation_dialog_cache') and folder_key in self._image_retranslation_dialog_cache:
            cached_dialog = self._image_retranslation_dialog_cache[folder_key]
            if cached_dialog:
                # Reuse existing dialog - just show it
                cached_dialog.show()
                cached_dialog.raise_()
                cached_dialog.activateWindow()
                return
        
        # Look for output folder in the SCRIPT'S directory, not relative to the selected folder
        script_dir = os.getcwd()  # Current working directory where the script is running
        
        # Check multiple possible output folder patterns IN THE SCRIPT DIRECTORY
        possible_output_dirs = [
            os.path.join(script_dir, folder_name),  # Script dir + folder name (without extension)
            os.path.join(script_dir, f"{folder_name}_translated"),  # Script dir + folder_translated
            folder_name,  # Just the folder name in current directory
            f"{folder_name}_translated",  # folder_translated in current directory
        ]
        
        output_dir = None
        for possible_dir in possible_output_dirs:
            print(f"Checking: {possible_dir}")
            if os.path.exists(possible_dir):
                # Check if it has translation_progress.json or HTML files
                if os.path.exists(os.path.join(possible_dir, "translation_progress.json")):
                    output_dir = possible_dir
                    print(f"Found output directory with progress tracker: {output_dir}")
                    break
                # Check if it has any HTML files
                elif os.path.isdir(possible_dir):
                    try:
                        files = os.listdir(possible_dir)
                        if any(f.lower().endswith(('.html', '.xhtml', '.htm')) for f in files):
                            output_dir = possible_dir
                            print(f"Found output directory with HTML files: {output_dir}")
                            break
                    except:
                        pass
        
        if not output_dir:
            QMessageBox.information(self, "Info", 
                f"No translation output found for '{folder_name}'.\n\n"
                f"Selected folder: {folder_path}\n"
                f"Script directory: {script_dir}\n\n"
                f"Checked locations:\n" + "\n".join(f"- {d}" for d in possible_output_dirs))
            return
        
        print(f"Using output directory: {output_dir}")
        
        # Check for progress tracking file
        progress_file = os.path.join(output_dir, "translation_progress.json")
        has_progress_tracking = os.path.exists(progress_file)
        
        print(f"Progress tracking: {has_progress_tracking} at {progress_file}")
        
        # Find all HTML files in the output directory
        html_files = []
        image_files = []
        progress_data = None
        
        if has_progress_tracking:
            # Load progress data for image translations
            try:
                with open(progress_file, 'r', encoding='utf-8') as f:
                    progress_data = json.load(f)
                    print(f"Loaded progress data with {len(progress_data)} entries")
                    
                # Extract files from progress data
                # The structure appears to use hash keys at the root level
                for key, value in progress_data.items():
                    if isinstance(value, dict) and 'output_file' in value:
                        output_file = value['output_file']
                        # Handle both forward and backslashes in paths
                        output_file = output_file.replace('\\', '/')
                        if '/' in output_file:
                            output_file = os.path.basename(output_file)
                        html_files.append(output_file)
                        print(f"Found tracked file: {output_file}")
            except Exception as e:
                print(f"Error loading progress file: {e}")
                import traceback
                traceback.print_exc()
                has_progress_tracking = False
        
        # Also scan directory for any HTML files not in progress
        # Only include translated image files (response_*.html pattern)
        # Also include generated image files (.png, .jpg, etc.)
        try:
            for file in os.listdir(output_dir):
                file_path = os.path.join(output_dir, file)
                # Include HTML files matching response_NNN_*.html pattern
                if (os.path.isfile(file_path) and 
                    file.endswith('.html') and 
                    file not in html_files and
                    re.match(r'response_\d+_', file)):
                    html_files.append(file)
                    print(f"Found untracked HTML file: {file}")
                # Also include generated image files (not in images/ subdirectory)
                elif (os.path.isfile(file_path) and 
                      file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif')) and
                      file not in html_files):
                    html_files.append(file)  # Add to html_files for now, will be handled separately
                    print(f"Found generated image file: {file}")
        except Exception as e:
            print(f"Error scanning directory: {e}")
        
        # Check for images subdirectory (cover images)
        images_dir = os.path.join(output_dir, "images")
        if os.path.exists(images_dir):
            try:
                for file in os.listdir(images_dir):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                        image_files.append(file)
            except Exception as e:
                print(f"Error scanning images directory: {e}")
        
        print(f"Total files found: {len(html_files)} HTML, {len(image_files)} images")
        
        if not html_files and not image_files:
            QMessageBox.information(self, "Info", 
                f"No translated files found in: {output_dir}\n\n"
                f"Progress tracking: {'Yes' if has_progress_tracking else 'No'}")
            return
        
        # Create dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Force Retranslation - Images")
        # Decreased width to 18%, increased height to 25% for better vertical space
        width, height = self._get_dialog_size(0.18, 0.25)
        dialog.resize(width, height)
        
        # Set icon
        try:
            from PySide6.QtGui import QIcon
            if hasattr(self, 'base_dir'):
                base_dir = self.base_dir
            else:
                base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
            ico_path = os.path.join(base_dir, 'Halgakos.ico')
            if os.path.isfile(ico_path):
                dialog.setWindowIcon(QIcon(ico_path))
        except Exception as e:
            print(f"Failed to load icon: {e}")
        
        dialog_layout = QVBoxLayout(dialog)
        
        # Create listbox (QListWidget has built-in scrolling)
        listbox = QListWidget()
        listbox.setSelectionMode(QListWidget.ExtendedSelection)
        # Use 16% of screen width (half of original ~31% for 1920px screen)
        min_width, _ = self._get_dialog_size(0.16, 0)
        listbox.setMinimumWidth(min_width)
        dialog_layout.addWidget(listbox)
        
        # Keep track of file info
        file_info = []
        
        # Add translated HTML files
        for html_file in sorted(set(html_files)):  # Use set to avoid duplicates
            # Extract original image name from HTML filename
            # Expected format: response_001_imagename.html
            match = re.match(r'response_(\d+)_(.+)\.html', html_file)
            if match:
                index = match.group(1)
                base_name = match.group(2)
                display = f"📄 Image {index} | {base_name} | ✅ Translated"
            else:
                display = f"📄 {html_file} | ✅ Translated"
            
            listbox.addItem(display)
            
            # Find the hash key for this file if progress tracking exists
            hash_key = None
            if progress_data:
                for key, value in progress_data.items():
                    if isinstance(value, dict) and 'output_file' in value:
                        if html_file in value['output_file']:
                            hash_key = key
                            break
            
            file_info.append({
                'type': 'translated',
                'file': html_file,
                'path': os.path.join(output_dir, html_file),
                'hash_key': hash_key,
                'output_dir': output_dir  # Store for later use
            })
        
        # Add cover images
        for img_file in sorted(image_files):
            display = f"🖼️ Cover | {img_file} | ⏭️ Skipped (cover)"
            listbox.addItem(display)
            file_info.append({
                'type': 'cover',
                'file': img_file,
                'path': os.path.join(images_dir, img_file),
                'hash_key': None,
                'output_dir': output_dir
            })
        
        # Selection count label
        selection_count_label = QLabel("Selected: 0")
        selection_font = QFont('Arial', 10)
        selection_count_label.setFont(selection_font)
        dialog_layout.addWidget(selection_count_label)
        
        def update_selection_count():
            count = len(listbox.selectedItems())
            selection_count_label.setText(f"Selected: {count}")
        
        listbox.itemSelectionChanged.connect(update_selection_count)
        
        # Button frame
        button_frame = QWidget()
        button_layout = QGridLayout(button_frame)
        dialog_layout.addWidget(button_frame)
        
        def select_all():
            listbox.selectAll()
            update_selection_count()
        
        def clear_selection():
            listbox.clearSelection()
            update_selection_count()
        
        def select_translated():
            listbox.clearSelection()
            for idx, info in enumerate(file_info):
                if info['type'] == 'translated':
                    listbox.item(idx).setSelected(True)
            update_selection_count()
        
        def mark_as_skipped():
            """Move selected images to the images folder to be skipped"""
            selected_items = listbox.selectedItems()
            if not selected_items:
                QMessageBox.warning(self, "No Selection", "Please select at least one image to mark as skipped.")
                return
            
            # Get all selected items
            selected_indices = [listbox.row(item) for item in selected_items]
            items_with_info = [(i, file_info[i]) for i in selected_indices]
            
            # Filter out items already in images folder (covers)
            items_to_move = [(i, item) for i, item in items_with_info if item['type'] != 'cover']
            
            if not items_to_move:
                QMessageBox.information(self, "Info", "Selected items are already in the images folder (skipped).")
                return
            
            count = len(items_to_move)
            reply = QMessageBox.question(self, "Confirm Mark as Skipped", 
                                      f"Move {count} translated image(s) to the images folder?\n\n"
                                      "This will:\n"
                                      "• Delete the translated HTML files\n"
                                      "• Copy source images to the images folder\n"
                                      "• Skip these images in future translations",
                                      QMessageBox.Yes | QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
            
            # Create images directory if it doesn't exist
            images_dir = os.path.join(output_dir, "images")
            os.makedirs(images_dir, exist_ok=True)
            
            moved_count = 0
            failed_count = 0
            
            for idx, item in items_to_move:
                try:
                    # Extract the original image name from the HTML filename
                    # Expected format: response_001_imagename.html (also accept compound extensions)
                    html_file = item['file']
                    match = re.match(r'^response_\d+_([^\.]*)\.(?:html?|xhtml|htm)(?:\.xhtml)?$', html_file, re.IGNORECASE)
                    
                    if match:
                        base_name = match.group(1)
                        # Try to find the original image with common extensions
                        original_found = False
                        
                        # Look for the source image in multiple locations
                        search_paths = [
                            folder_path,  # Original folder path
                            os.path.dirname(folder_path),  # Parent of folder path
                            os.getcwd(),  # Script directory
                        ]
                        
                        for ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']:
                            for search_path in search_paths:
                                if not search_path or not os.path.exists(search_path):
                                    continue
                                    
                                # Check in the search path
                                possible_source = os.path.join(search_path, base_name + ext)
                                if os.path.exists(possible_source) and os.path.isfile(possible_source):
                                    # Copy to images folder
                                    dest_path = os.path.join(images_dir, base_name + ext)
                                    if not os.path.exists(dest_path):
                                        import shutil
                                        shutil.copy2(possible_source, dest_path)
                                        print(f"Copied {base_name + ext} from {possible_source} to images folder")
                                    original_found = True
                                    break
                            if original_found:
                                break
                        
                        if not original_found:
                            print(f"Warning: Could not find original image for {html_file} in: {search_paths}")
                            # Even if source not found, we can still delete the HTML and mark it
                    
                    # Delete the HTML translation file
                    if os.path.exists(item['path']):
                        os.remove(item['path'])
                        print(f"Deleted translation: {item['path']}")
                        
                        # Remove from progress tracking if applicable
                        if progress_data and item.get('hash_key') and item['hash_key'] in progress_data:
                            del progress_data[item['hash_key']]
                    
                    # Update the listbox display
                    display = f"🖼️ Skipped | {base_name if match else item['file']} | ⏭️ Moved to images folder"
                    listbox.item(idx).setText(display)
                    
                    # Update file_info
                    file_info[idx] = {
                        'type': 'cover',  # Treat as cover type since it's in images folder
                        'file': base_name + ext if match and original_found else item['file'],
                        'path': os.path.join(images_dir, base_name + ext if match and original_found else item['file']),
                        'hash_key': None,
                        'output_dir': output_dir
                    }
                    
                    moved_count += 1
                    
                except Exception as e:
                    print(f"Failed to process {item['file']}: {e}")
                    failed_count += 1
            
            # Save updated progress if modified
            if progress_data:
                try:
                    with open(progress_file, 'w', encoding='utf-8') as f:
                        json.dump(progress_data, f, ensure_ascii=False, indent=2)
                    print(f"Updated progress tracking file")
                except Exception as e:
                    print(f"Failed to update progress file: {e}")
            
            # Update selection count
            update_selection_count()
            
            # Show result
            if failed_count > 0:
                QMessageBox.warning(self, "Partial Success", 
                    f"Moved {moved_count} image(s) to be skipped.\n"
                    f"Failed to process {failed_count} item(s).")
            else:
                QMessageBox.information(self, "Success", 
                    f"Moved {moved_count} image(s) to the images folder.\n"
                    "They will be skipped in future translations.")
        
        def retranslate_selected():
            selected_items = listbox.selectedItems()
            if not selected_items:
                QMessageBox.warning(self, "No Selection", "Please select at least one file.")
                return
            
            selected_indices = [listbox.row(item) for item in selected_items]
            
            # Count types
            translated_count = sum(1 for i in selected_indices if file_info[i]['type'] == 'translated')
            cover_count = sum(1 for i in selected_indices if file_info[i]['type'] == 'cover')
            
            # Build confirmation message
            msg_parts = []
            if translated_count > 0:
                msg_parts.append(f"{translated_count} translated image(s)")
            if cover_count > 0:
                msg_parts.append(f"{cover_count} cover image(s)")
            
            confirm_msg = f"This will delete {' and '.join(msg_parts)}.\n\nContinue?"
            
            reply = QMessageBox.question(self, "Confirm Deletion", confirm_msg,
                                       QMessageBox.Yes | QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
            
            # Delete selected files
            deleted_count = 0
            progress_updated = False
            
            for idx in selected_indices:
                info = file_info[idx]
                try:
                    if os.path.exists(info['path']):
                        os.remove(info['path'])
                        deleted_count += 1
                        print(f"Deleted: {info['path']}")
                        
                        # Remove from progress tracking if applicable
                        if progress_data and info['hash_key'] and info['hash_key'] in progress_data:
                            del progress_data[info['hash_key']]
                            progress_updated = True
                            
                except Exception as e:
                    print(f"Failed to delete {info['path']}: {e}")
            
            # Save updated progress if modified
            if progress_updated and progress_data:
                try:
                    with open(progress_file, 'w', encoding='utf-8') as f:
                        json.dump(progress_data, f, ensure_ascii=False, indent=2)
                    print(f"Updated progress tracking file")
                except Exception as e:
                    print(f"Failed to update progress file: {e}")
            
            QMessageBox.information(self, "Success", 
                f"Deleted {deleted_count} file(s).\n\n"
                "They will be retranslated on the next run.")
            
            dialog.close()
        
        # Add buttons in grid layout (similar to EPUB/text retranslation)
        # Row 0: Selection buttons
        btn_select_all = QPushButton("Select All")
        btn_select_all.setStyleSheet("QPushButton { background-color: #17a2b8; color: white; padding: 5px 15px; font-weight: bold; }")
        btn_select_all.clicked.connect(select_all)
        button_layout.addWidget(btn_select_all, 0, 0)
        
        btn_clear_selection = QPushButton("Clear Selection")
        btn_clear_selection.setStyleSheet("QPushButton { background-color: #6c757d; color: white; padding: 5px 15px; font-weight: bold; }")
        btn_clear_selection.clicked.connect(clear_selection)
        button_layout.addWidget(btn_clear_selection, 0, 1)
        
        btn_select_translated = QPushButton("Select Translated")
        btn_select_translated.setStyleSheet("QPushButton { background-color: #28a745; color: white; padding: 5px 15px; font-weight: bold; }")
        btn_select_translated.clicked.connect(select_translated)
        button_layout.addWidget(btn_select_translated, 0, 2)
        
        btn_mark_skipped = QPushButton("Mark as Skipped")
        btn_mark_skipped.setStyleSheet("QPushButton { background-color: #e0a800; color: white; padding: 5px 15px; font-weight: bold; }")
        btn_mark_skipped.clicked.connect(mark_as_skipped)
        button_layout.addWidget(btn_mark_skipped, 0, 3)
        
        # Row 1: Action buttons
        btn_delete = QPushButton("Delete Selected")
        btn_delete.setStyleSheet("QPushButton { background-color: #dc3545; color: white; padding: 5px 15px; font-weight: bold; }")
        btn_delete.clicked.connect(retranslate_selected)
        button_layout.addWidget(btn_delete, 1, 0, 1, 2)
        
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setStyleSheet("QPushButton { background-color: #6c757d; color: white; padding: 5px 15px; font-weight: bold; }")
        btn_cancel.clicked.connect(dialog.close)
        button_layout.addWidget(btn_cancel, 1, 2, 1, 2)
        
        # Override close event to hide instead of destroy
        def closeEvent(event):
            event.ignore()  # Ignore the close event
            dialog.hide()   # Just hide the dialog
        
        dialog.closeEvent = closeEvent
        
        # Cache the dialog for reuse
        if not hasattr(self, '_image_retranslation_dialog_cache'):
            self._image_retranslation_dialog_cache = {}
        
        folder_key = os.path.abspath(folder_path)
        self._image_retranslation_dialog_cache[folder_key] = dialog
        
        # Show the dialog (non-modal to allow interaction with other windows)
        dialog.show()
