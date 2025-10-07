"""
Retranslation GUI Module
Force retranslation functionality for EPUB, text, and image files
"""

import os
import sys
import json
import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import ttkbootstrap as tb
import xml.etree.ElementTree as ET
import zipfile
import shutil
import traceback

# Import from translator_gui if available
try:
    from translator_gui import WindowManager, UIHelper
except ImportError:
    WindowManager = None
    UIHelper = None


class RetranslationMixin:
    """Mixin class containing retranslation methods for TranslatorGUI"""
 
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
        input_path = self.entry_epub.get()
        if not input_path or not os.path.isfile(input_path):
            messagebox.showerror("Error", "Please select a valid EPUB, text file, or image folder first.")
            return
        
        # Check if it's an image file
        image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')
        if input_path.lower().endswith(image_extensions):
            self._force_retranslation_single_image(input_path)
            return
        
        # For EPUB/text files, use the shared logic
        self._force_retranslation_epub_or_text(input_path)


    def _force_retranslation_epub_or_text(self, file_path, parent_dialog=None, tab_frame=None):
        """
        Shared logic for force retranslation of EPUB/text files with OPF support
        Can be used standalone or embedded in a tab
        
        Args:
            file_path: Path to the EPUB/text file
            parent_dialog: If provided, won't create its own dialog
            tab_frame: If provided, will render into this frame instead of creating dialog
        
        Returns:
            dict: Contains all the UI elements and data for external access
        """
        
        epub_base = os.path.splitext(os.path.basename(file_path))[0]
        output_dir = epub_base
        
        if not os.path.exists(output_dir):
            if not parent_dialog:
                messagebox.showinfo("Info", "No translation output found for this file.")
            return None
        
        progress_file = os.path.join(output_dir, "translation_progress.json")
        if not os.path.exists(progress_file):
            if not parent_dialog:
                messagebox.showinfo("Info", "No progress tracking found.")
            return None
        
        with open(progress_file, 'r', encoding='utf-8') as f:
            prog = json.load(f)
        
        # =====================================================
        # PARSE CONTENT.OPF FOR CHAPTER MANIFEST
        # =====================================================
        
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
                                
                                # Skip navigation, toc, and cover files
                                if not any(skip in filename.lower() for skip in ['nav.', 'toc.', 'cover.']):
                                    manifest_chapters[item_id] = {
                                        'filename': filename,
                                        'href': href,
                                        'media_type': media_type
                                    }
                        
                        # Get spine order - the reading order
                        spine = root.find('.//opf:spine', ns)
                        
                        if spine is not None:
                            for itemref in spine.findall('opf:itemref', ns):
                                idref = itemref.get('idref')
                                if idref and idref in manifest_chapters:
                                    chapter_info = manifest_chapters[idref]
                                    filename = chapter_info['filename']
                                    
                                    # Skip navigation, toc, and cover files
                                    if not any(skip in filename.lower() for skip in ['nav.', 'toc.', 'cover.']):
                                        # Extract chapter number from filename
                                        import re
                                        matches = re.findall(r'(\d+)', filename)
                                        if matches:
                                            file_chapter_num = int(matches[-1])
                                        else:
                                            file_chapter_num = len(spine_chapters)
                                        
                                        spine_chapters.append({
                                            'id': idref,
                                            'filename': filename,
                                            'position': len(spine_chapters),
                                            'file_chapter_num': file_chapter_num,
                                            'status': 'unknown',  # Will be updated
                                            'output_file': None    # Will be updated
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
            
            # Find the actual response file that exists
            base_name = os.path.splitext(filename)[0]
            expected_response = None
            
            # Handle .htm.html -> .html conversion
            stripped_base_name = base_name
            if base_name.endswith('.htm'):
                stripped_base_name = base_name[:-4]  # Remove .htm suffix

            # Look for translated file matching base name, with or without 'response_' and with allowed extensions
            allowed_exts = ('.html', '.xhtml', '.htm')
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
            if not matched_info:
                for chapter_key, chapter_info in prog.get("chapters", {}).items():
                    actual_num = chapter_info.get('actual_num')
                    # Also check 'chapter_num' as fallback
                    if actual_num is None:
                        actual_num = chapter_info.get('chapter_num')
                    
                    if actual_num is not None and actual_num == chapter_num:
                        matched_info = chapter_info
                        break
            
            # Determine if translation file exists
            file_exists = os.path.exists(response_path)
            
            # Set status and output file based on findings
            if matched_info:
                # We found progress tracking info - use its status
                spine_ch['status'] = matched_info.get('status', 'unknown')
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
            dialog = self.wm.create_simple_dialog(
                self.master,
                "Force Retranslation - OPF Based" if spine_chapters else "Force Retranslation",
                width=1000,
                height=700
            )
            container = dialog
        else:
            container = tab_frame or parent_dialog
            dialog = parent_dialog
        
        # Title
        title_text = "Chapters from content.opf (in reading order):" if spine_chapters else "Select chapters to retranslate:"
        tk.Label(container, text=title_text, 
                font=('Arial', 12 if not tab_frame else 11, 'bold')).pack(pady=5)
        
        # Statistics if OPF is available
        if spine_chapters:
            stats_frame = tk.Frame(container)
            stats_frame.pack(pady=5)
            
            total_chapters = len(spine_chapters)
            completed = sum(1 for ch in spine_chapters if ch['status'] == 'completed')
            missing = sum(1 for ch in spine_chapters if ch['status'] == 'not_translated')
            failed = sum(1 for ch in spine_chapters if ch['status'] in ['failed', 'qa_failed'])
            file_missing = sum(1 for ch in spine_chapters if ch['status'] == 'file_missing')
            
            tk.Label(stats_frame, text=f"Total: {total_chapters} | ", font=('Arial', 10)).pack(side=tk.LEFT)
            tk.Label(stats_frame, text=f"✅ Completed: {completed} | ", font=('Arial', 10), fg='green').pack(side=tk.LEFT)
            tk.Label(stats_frame, text=f"❌ Missing: {missing} | ", font=('Arial', 10), fg='red').pack(side=tk.LEFT)
            tk.Label(stats_frame, text=f"⚠️ Failed: {failed} | ", font=('Arial', 10), fg='orange').pack(side=tk.LEFT)
            tk.Label(stats_frame, text=f"📁 File Missing: {file_missing}", font=('Arial', 10), fg='purple').pack(side=tk.LEFT)
        
        # Main frame for listbox
        main_frame = tk.Frame(container)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10 if not tab_frame else 5, pady=5)
        
        # Create scrollbars and listbox
        h_scrollbar = ttk.Scrollbar(main_frame, orient=tk.HORIZONTAL)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        v_scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        listbox = tk.Listbox(
            main_frame, 
            selectmode=tk.EXTENDED, 
            yscrollcommand=v_scrollbar.set,
            xscrollcommand=h_scrollbar.set,
            width=120,
            font=('Courier', 10)  # Fixed-width font for better alignment
        )
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        v_scrollbar.config(command=listbox.yview)
        h_scrollbar.config(command=listbox.xview)
        
        # Populate listbox
        status_icons = {
            'completed': '✅',
            'failed': '❌',
            'qa_failed': '❌',
            'file_missing': '⚠️',
            'in_progress': '🔄',
            'not_translated': '❌',
            'unknown': '❓'
        }
        
        status_labels = {
            'completed': 'Completed',
            'failed': 'Failed',
            'qa_failed': 'QA Failed',
            'file_missing': 'File Missing',
            'in_progress': 'In Progress',
            'not_translated': 'Not Translated',
            'unknown': 'Unknown'
        }
        
        for info in chapter_display_info:
            chapter_num = info['num']
            status = info['status']
            output_file = info['output_file']
            icon = status_icons.get(status, '❓')
            status_label = status_labels.get(status, status)
            
            # Format display with OPF info if available
            if 'opf_position' in info:
                # OPF-based display
                original_file = info.get('original_filename', '')
                opf_pos = info['opf_position'] + 1  # 1-based for display
                
                # Format: [OPF Position] Chapter Number | Status | Original File -> Response File
                if isinstance(chapter_num, float) and chapter_num.is_integer():
                    display = f"[{opf_pos:03d}] Ch.{int(chapter_num):03d} | {icon} {status_label:15s} | {original_file:30s} -> {output_file}"
                else:
                    display = f"[{opf_pos:03d}] Ch.{chapter_num:03d} | {icon} {status_label:15s} | {original_file:30s} -> {output_file}"
            else:
                # Original format
                if isinstance(chapter_num, float) and chapter_num.is_integer():
                    display = f"Chapter {int(chapter_num):03d} | {icon} {status_label:15s} | {output_file}"
                elif isinstance(chapter_num, float):
                    display = f"Chapter {chapter_num:06.1f} | {icon} {status_label:15s} | {output_file}"
                else:
                    display = f"Chapter {chapter_num:03d} | {icon} {status_label:15s} | {output_file}"
            
            if info.get('duplicate_count', 1) > 1:
                display += f" | ({info['duplicate_count']} entries)"
            
            listbox.insert(tk.END, display)
            
            # Color code based on status
            if status == 'completed':
                listbox.itemconfig(tk.END, fg='green')
            elif status in ['failed', 'qa_failed', 'not_translated']:
                listbox.itemconfig(tk.END, fg='red')
            elif status == 'file_missing':
                listbox.itemconfig(tk.END, fg='purple')
            elif status == 'in_progress':
                listbox.itemconfig(tk.END, fg='orange')
        
        # Selection count label
        selection_count_label = tk.Label(container, text="Selected: 0", 
                                       font=('Arial', 10 if not tab_frame else 9))
        selection_count_label.pack(pady=(5, 10) if not tab_frame else 2)
        
        def update_selection_count(*args):
            count = len(listbox.curselection())
            selection_count_label.config(text=f"Selected: {count}")
        
        listbox.bind('<<ListboxSelect>>', update_selection_count)
        
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
            'container': container
        }
        
        # If standalone (no parent), add buttons
        if not parent_dialog or tab_frame:
            self._add_retranslation_buttons_opf(result)
        
        return result


    def _add_retranslation_buttons_opf(self, data, button_frame=None):
        """Add the standard button set for retranslation dialogs with OPF support"""
        
        if not button_frame:
            button_frame = tk.Frame(data['container'])
            button_frame.pack(pady=10)
        
        # Configure column weights
        for i in range(5):
            button_frame.columnconfigure(i, weight=1)
        
        # Helper functions that work with the data dict
        def select_all():
            data['listbox'].select_set(0, tk.END)
            data['selection_count_label'].config(text=f"Selected: {data['listbox'].size()}")
        
        def clear_selection():
            data['listbox'].select_clear(0, tk.END)
            data['selection_count_label'].config(text="Selected: 0")
        
        def select_status(status_to_select):
            data['listbox'].select_clear(0, tk.END)
            for idx, info in enumerate(data['chapter_display_info']):
                if status_to_select == 'failed':
                    if info['status'] in ['failed', 'qa_failed']:
                        data['listbox'].select_set(idx)
                elif status_to_select == 'missing':
                    if info['status'] in ['not_translated', 'file_missing']:
                        data['listbox'].select_set(idx)
                else:
                    if info['status'] == status_to_select:
                        data['listbox'].select_set(idx)
            count = len(data['listbox'].curselection())
            data['selection_count_label'].config(text=f"Selected: {count}")
        
        def remove_qa_failed_mark():
            selected = data['listbox'].curselection()
            if not selected:
                messagebox.showwarning("No Selection", "Please select at least one chapter.")
                return
            
            selected_chapters = [data['chapter_display_info'][i] for i in selected]
            qa_failed_chapters = [ch for ch in selected_chapters if ch['status'] == 'qa_failed']
            
            if not qa_failed_chapters:
                messagebox.showwarning("No QA Failed Chapters", 
                                     "None of the selected chapters have 'qa_failed' status.")
                return
            
            count = len(qa_failed_chapters)
            if not messagebox.askyesno("Confirm Remove QA Failed Mark", 
                                      f"Remove QA failed mark from {count} chapters?"):
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
            
            messagebox.showinfo("Success", f"Removed QA failed mark from {cleared_count} chapters.")
            if data.get('dialog'):
                data['dialog'].destroy()
        
        def retranslate_selected():
            selected = data['listbox'].curselection()
            if not selected:
                messagebox.showwarning("No Selection", "Please select at least one chapter.")
                return
            
            selected_chapters = [data['chapter_display_info'][i] for i in selected]
            
            # Count different types
            missing_count = sum(1 for ch in selected_chapters if ch['status'] == 'not_translated')
            existing_count = sum(1 for ch in selected_chapters if ch['status'] != 'not_translated')
            
            count = len(selected)
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
            
            if not messagebox.askyesno("Confirm Retranslation", confirm_msg):
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
                    
                    # Reset status for any completed or qa_failed chapters
                    if ch_info['status'] in ['completed', 'qa_failed']:
                        target_output_file = ch_info['output_file']
                        chapter_key = None
                        
                        # Search through all chapters to find the one with matching output_file
                        for key, ch_data in data['prog']["chapters"].items():
                            if ch_data.get('output_file') == target_output_file:
                                chapter_key = key
                                break
                        
                        # Update the chapter status if we found the key
                        if chapter_key and chapter_key in data['prog']["chapters"]:
                            old_status = ch_info['status']
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
                    success_msg += f"\n\nTotal {len(selected)} chapters ready for translation."
                messagebox.showinfo("Success", success_msg)
            else:
                messagebox.showinfo("Info", "No changes made.")
            
            if data.get('dialog'):
                data['dialog'].destroy()
        
        # Add buttons - First row
        tb.Button(button_frame, text="Select All", command=select_all, 
                  bootstyle="info").grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        tb.Button(button_frame, text="Clear", command=clear_selection, 
                  bootstyle="secondary").grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        tb.Button(button_frame, text="Select Completed", command=lambda: select_status('completed'), 
                  bootstyle="success").grid(row=0, column=2, padx=5, pady=5, sticky="ew")
        tb.Button(button_frame, text="Select Missing", command=lambda: select_status('missing'), 
                  bootstyle="danger").grid(row=0, column=3, padx=5, pady=5, sticky="ew")
        tb.Button(button_frame, text="Select Failed", command=lambda: select_status('failed'), 
                  bootstyle="warning").grid(row=0, column=4, padx=5, pady=5, sticky="ew")
        
        # Second row
        tb.Button(button_frame, text="Retranslate Selected", command=retranslate_selected, 
                  bootstyle="warning").grid(row=1, column=0, columnspan=2, padx=5, pady=10, sticky="ew")
        tb.Button(button_frame, text="Remove QA Failed Mark", command=remove_qa_failed_mark, 
                  bootstyle="success").grid(row=1, column=2, columnspan=1, padx=5, pady=10, sticky="ew")
        tb.Button(button_frame, text="Cancel", command=lambda: data['dialog'].destroy() if data.get('dialog') else None, 
                  bootstyle="secondary").grid(row=1, column=3, columnspan=2, padx=5, pady=10, sticky="ew")


    def _force_retranslation_multiple_files(self):
        """Handle force retranslation when multiple files are selected - now uses shared logic"""
        
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
            messagebox.showinfo("Info", "No valid files selected.")
            return
        
        # Create main dialog
        dialog = self.wm.create_simple_dialog(
            self.master,
            "Force Retranslation - Multiple Files",
            width=950,
            height=700
        )
        
        # Summary label
        tk.Label(dialog, text=f"Selected: {', '.join(summary_parts)}", 
                font=('Arial', 12, 'bold')).pack(pady=10)
        
        # Create notebook
        notebook = ttk.Notebook(dialog)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Track all tab data
        tab_data = []
        tabs_created = False
        
        # Create tabs for EPUB/text files using shared logic
        for file_path in epub_files + text_files:
            file_base = os.path.splitext(os.path.basename(file_path))[0]
            
            # Quick check if output exists
            if not os.path.exists(file_base):
                continue
            
            # Create tab
            tab_frame = tk.Frame(notebook)
            tab_name = file_base[:20] + "..." if len(file_base) > 20 else file_base
            notebook.add(tab_frame, text=tab_name)
            tabs_created = True
            
            # Use shared logic to populate the tab
            tab_result = self._force_retranslation_epub_or_text(
                file_path, 
                parent_dialog=dialog, 
                tab_frame=tab_frame
            )
            
            if tab_result:
                tab_data.append(tab_result)
        
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
        
        # If no tabs were created, show error
        if not tabs_created:
            messagebox.showinfo("Info", 
                "No translation output found for any of the selected files.\n\n"
                "Make sure the output folders exist in your script directory.")
            dialog.destroy()
            return
        
        # Add unified button bar that works across all tabs
        self._add_multi_file_buttons(dialog, notebook, tab_data)

    def _add_multi_file_buttons(self, dialog, notebook, tab_data):
        """Add a simple cancel button at the bottom of the dialog"""
        button_frame = tk.Frame(dialog)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
        tb.Button(button_frame, text="Close All", command=dialog.destroy, 
                  bootstyle="secondary").pack(side=tk.RIGHT, padx=5)
              
    def _create_individual_images_tab(self, image_files, notebook, parent_dialog):
        """Create a tab for individual image files"""
        # Create tab
        tab_frame = tk.Frame(notebook)
        notebook.add(tab_frame, text="Individual Images")
        
        # Instructions
        tk.Label(tab_frame, text=f"Selected {len(image_files)} individual image(s):", 
                 font=('Arial', 11)).pack(pady=5)
        
        # Main frame
        main_frame = tk.Frame(tab_frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Scrollbars and listbox
        h_scrollbar = ttk.Scrollbar(main_frame, orient=tk.HORIZONTAL)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        v_scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        listbox = tk.Listbox(
            main_frame,
            selectmode=tk.EXTENDED,
            yscrollcommand=v_scrollbar.set,
            xscrollcommand=h_scrollbar.set,
            width=100
        )
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        v_scrollbar.config(command=listbox.yview)
        h_scrollbar.config(command=listbox.xview)
        
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
                    listbox.insert(tk.END, display)
                    
                    file_info.append({
                        'type': 'translated',
                        'source_image': img_path,
                        'output_dir': output_dir,
                        'file': html_file,
                        'path': os.path.join(output_dir, html_file)
                    })
            else:
                display = f"🖼️ {img_name} | ❌ No translation found"
                listbox.insert(tk.END, display)
        
        # Selection count
        selection_count_label = tk.Label(tab_frame, text="Selected: 0", font=('Arial', 9))
        selection_count_label.pack(pady=2)
        
        def update_selection_count(*args):
            count = len(listbox.curselection())
            selection_count_label.config(text=f"Selected: {count}")
        
        listbox.bind('<<ListboxSelect>>', update_selection_count)
        
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
        tab_frame = tk.Frame(notebook)
        tab_name = "📁 " + (folder_name[:17] + "..." if len(folder_name) > 17 else folder_name)
        notebook.add(tab_frame, text=tab_name)
        
        # Instructions
        tk.Label(tab_frame, text="Select images to retranslate:", font=('Arial', 11)).pack(pady=5)
        
        # Main frame
        main_frame = tk.Frame(tab_frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Scrollbars and listbox
        h_scrollbar = ttk.Scrollbar(main_frame, orient=tk.HORIZONTAL)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        v_scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        listbox = tk.Listbox(
            main_frame,
            selectmode=tk.EXTENDED,
            yscrollcommand=v_scrollbar.set,
            xscrollcommand=h_scrollbar.set,
            width=100
        )
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        v_scrollbar.config(command=listbox.yview)
        h_scrollbar.config(command=listbox.xview)
        
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
                
                listbox.insert(tk.END, display)
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
                    listbox.insert(tk.END, display)
                    file_info.append({
                        'type': 'cover',
                        'file': file,
                        'path': os.path.join(images_dir, file)
                    })
        
        # Selection count
        selection_count_label = tk.Label(tab_frame, text="Selected: 0", font=('Arial', 9))
        selection_count_label.pack(pady=2)
        
        def update_selection_count(*args):
            count = len(listbox.curselection())
            selection_count_label.config(text=f"Selected: {count}")
        
        listbox.bind('<<ListboxSelect>>', update_selection_count)
        
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
        folder_name = os.path.basename(folder_path)
        
        # Look for output folder in the SCRIPT'S directory, not relative to the selected folder
        script_dir = os.getcwd()  # Current working directory where the script is running
        
        # Check multiple possible output folder patterns IN THE SCRIPT DIRECTORY
        possible_output_dirs = [
            os.path.join(script_dir, folder_name),  # Script dir + folder name
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
            messagebox.showinfo("Info", 
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
        try:
            for file in os.listdir(output_dir):
                file_path = os.path.join(output_dir, file)
                if os.path.isfile(file_path) and file.endswith('.html') and file not in html_files:
                    html_files.append(file)
                    print(f"Found untracked HTML file: {file}")
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
            messagebox.showinfo("Info", 
                f"No translated files found in: {output_dir}\n\n"
                f"Progress tracking: {'Yes' if has_progress_tracking else 'No'}")
            return
        
        # Create dialog
        dialog = self.wm.create_simple_dialog(
            self.master,
            "Force Retranslation - Images",
            width=800,
            height=600
        )
        
        # Add instructions with more detail
        instruction_text = f"Output folder: {output_dir}\n"
        instruction_text += f"Found {len(html_files)} translated images and {len(image_files)} cover images"
        if has_progress_tracking:
            instruction_text += " (with progress tracking)"
        tk.Label(dialog, text=instruction_text, font=('Arial', 11), justify=tk.LEFT).pack(pady=10)
        
        # Create main frame for listbox and scrollbars
        main_frame = tk.Frame(dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create scrollbars
        h_scrollbar = ttk.Scrollbar(main_frame, orient=tk.HORIZONTAL)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        v_scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create listbox
        listbox = tk.Listbox(
            main_frame, 
            selectmode=tk.EXTENDED, 
            yscrollcommand=v_scrollbar.set,
            xscrollcommand=h_scrollbar.set,
            width=100
        )
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Configure scrollbars
        v_scrollbar.config(command=listbox.yview)
        h_scrollbar.config(command=listbox.xview)
        
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
            
            listbox.insert(tk.END, display)
            
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
            listbox.insert(tk.END, display)
            file_info.append({
                'type': 'cover',
                'file': img_file,
                'path': os.path.join(images_dir, img_file),
                'hash_key': None,
                'output_dir': output_dir
            })
        
        # Selection count label
        selection_count_label = tk.Label(dialog, text="Selected: 0", font=('Arial', 10))
        selection_count_label.pack(pady=(5, 10))
        
        def update_selection_count(*args):
            count = len(listbox.curselection())
            selection_count_label.config(text=f"Selected: {count}")
        
        listbox.bind('<<ListboxSelect>>', update_selection_count)
        
        # Button frame
        button_frame = tk.Frame(dialog)
        button_frame.pack(pady=10)
        
        # Configure grid columns
        for i in range(4):
            button_frame.columnconfigure(i, weight=1)
        
        def select_all():
            listbox.select_set(0, tk.END)
            update_selection_count()
        
        def clear_selection():
            listbox.select_clear(0, tk.END)
            update_selection_count()
        
        def select_translated():
            listbox.select_clear(0, tk.END)
            for idx, info in enumerate(file_info):
                if info['type'] == 'translated':
                    listbox.select_set(idx)
            update_selection_count()
        
        def mark_as_skipped():
            """Move selected images to the images folder to be skipped"""
            selected = listbox.curselection()
            if not selected:
                messagebox.showwarning("No Selection", "Please select at least one image to mark as skipped.")
                return
            
            # Get all selected items
            selected_items = [(i, file_info[i]) for i in selected]
            
            # Filter out items already in images folder (covers)
            items_to_move = [(i, item) for i, item in selected_items if item['type'] != 'cover']
            
            if not items_to_move:
                messagebox.showinfo("Info", "Selected items are already in the images folder (skipped).")
                return
            
            count = len(items_to_move)
            if not messagebox.askyesno("Confirm Mark as Skipped", 
                                      f"Move {count} translated image(s) to the images folder?\n\n"
                                      "This will:\n"
                                      "• Delete the translated HTML files\n"
                                      "• Copy source images to the images folder\n"
                                      "• Skip these images in future translations"):
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
                        
                        for ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']:
                            # Check in the parent folder (where source images are)
                            possible_source = os.path.join(folder_path, base_name + ext)
                            if os.path.exists(possible_source):
                                # Copy to images folder
                                dest_path = os.path.join(images_dir, base_name + ext)
                                if not os.path.exists(dest_path):
                                    import shutil
                                    shutil.copy2(possible_source, dest_path)
                                    print(f"Copied {base_name + ext} to images folder")
                                original_found = True
                                break
                        
                        if not original_found:
                            print(f"Warning: Could not find original image for {html_file}")
                    
                    # Delete the HTML translation file
                    if os.path.exists(item['path']):
                        os.remove(item['path'])
                        print(f"Deleted translation: {item['path']}")
                        
                        # Remove from progress tracking if applicable
                        if progress_data and item.get('hash_key') and item['hash_key'] in progress_data:
                            del progress_data[item['hash_key']]
                    
                    # Update the listbox display
                    display = f"🖼️ Skipped | {base_name if match else item['file']} | ⏭️ Moved to images folder"
                    listbox.delete(idx)
                    listbox.insert(idx, display)
                    
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
                messagebox.showwarning("Partial Success", 
                    f"Moved {moved_count} image(s) to be skipped.\n"
                    f"Failed to process {failed_count} item(s).")
            else:
                messagebox.showinfo("Success", 
                    f"Moved {moved_count} image(s) to the images folder.\n"
                    "They will be skipped in future translations.")
        
        def retranslate_selected():
            selected = listbox.curselection()
            if not selected:
                messagebox.showwarning("No Selection", "Please select at least one file.")
                return
            
            # Count types
            translated_count = sum(1 for i in selected if file_info[i]['type'] == 'translated')
            cover_count = sum(1 for i in selected if file_info[i]['type'] == 'cover')
            
            # Build confirmation message
            msg_parts = []
            if translated_count > 0:
                msg_parts.append(f"{translated_count} translated image(s)")
            if cover_count > 0:
                msg_parts.append(f"{cover_count} cover image(s)")
            
            confirm_msg = f"This will delete {' and '.join(msg_parts)}.\n\nContinue?"
            
            if not messagebox.askyesno("Confirm Deletion", confirm_msg):
                return
            
            # Delete selected files
            deleted_count = 0
            progress_updated = False
            
            for idx in selected:
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
            
            messagebox.showinfo("Success", 
                f"Deleted {deleted_count} file(s).\n\n"
                "They will be retranslated on the next run.")
            
            dialog.destroy()
        
        # Add buttons in grid layout (similar to EPUB/text retranslation)
        # Row 0: Selection buttons
        tb.Button(button_frame, text="Select All", command=select_all, 
                  bootstyle="info").grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        tb.Button(button_frame, text="Clear Selection", command=clear_selection, 
                  bootstyle="secondary").grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        tb.Button(button_frame, text="Select Translated", command=select_translated, 
                  bootstyle="success").grid(row=0, column=2, padx=5, pady=5, sticky="ew")
        tb.Button(button_frame, text="Mark as Skipped", command=mark_as_skipped, 
                  bootstyle="warning").grid(row=0, column=3, padx=5, pady=5, sticky="ew")
        
        # Row 1: Action buttons
        tb.Button(button_frame, text="Delete Selected", command=retranslate_selected, 
                  bootstyle="danger").grid(row=1, column=0, columnspan=2, padx=5, pady=10, sticky="ew")
        tb.Button(button_frame, text="Cancel", command=dialog.destroy, 
                  bootstyle="secondary").grid(row=1, column=2, columnspan=2, padx=5, pady=10, sticky="ew")