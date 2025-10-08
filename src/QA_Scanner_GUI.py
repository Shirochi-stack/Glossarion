"""
QA Scanner GUI Methods
These methods can be integrated into TranslatorGUI or used standalone
"""

import os
import sys
import re
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import ttkbootstrap as tb
import threading
import traceback

# Import required modules from translator_gui
try:
    from translator_gui import WindowManager, UIHelper
    scan_html_folder = None  # Will be lazy-loaded from translator_gui
except ImportError as e:
    # Fallback if translator_gui not available (shouldn't happen in normal use)
    WindowManager = None
    UIHelper = None
    scan_html_folder = None


def check_epub_folder_match(epub_name, folder_name, custom_suffixes=''):
    """
    Check if EPUB name and folder name likely refer to the same content
    Uses strict matching to avoid false positives with similar numbered titles
    """
    # Normalize names for comparison
    epub_norm = normalize_name_for_comparison(epub_name)
    folder_norm = normalize_name_for_comparison(folder_name)
    
    # Direct match
    if epub_norm == folder_norm:
        return True
    
    # Check if folder has common output suffixes that should be ignored
    output_suffixes = ['_output', '_translated', '_trans', '_en', '_english', '_done', '_complete', '_final']
    if custom_suffixes:
        custom_list = [s.strip() for s in custom_suffixes.split(',') if s.strip()]
        output_suffixes.extend(custom_list)
    
    for suffix in output_suffixes:
        if folder_norm.endswith(suffix):
            folder_base = folder_norm[:-len(suffix)]
            if folder_base == epub_norm:
                return True
        if epub_norm.endswith(suffix):
            epub_base = epub_norm[:-len(suffix)]
            if epub_base == folder_norm:
                return True
    
    # Check for exact match with version numbers removed
    version_pattern = r'[\s_-]v\d+$'
    epub_no_version = re.sub(version_pattern, '', epub_norm)
    folder_no_version = re.sub(version_pattern, '', folder_norm)
    
    if epub_no_version == folder_no_version and (epub_no_version != epub_norm or folder_no_version != folder_norm):
        return True
    
    # STRICT NUMBER CHECK - all numbers must match exactly
    epub_numbers = re.findall(r'\d+', epub_name)
    folder_numbers = re.findall(r'\d+', folder_name)
    
    if epub_numbers != folder_numbers:
        return False
    
    # If we get here, numbers match, so check if the text parts are similar enough
    epub_text_only = re.sub(r'\d+', '', epub_norm).strip()
    folder_text_only = re.sub(r'\d+', '', folder_norm).strip()
    
    if epub_numbers and folder_numbers:
        return epub_text_only == folder_text_only
    
    return False


def normalize_name_for_comparison(name):
    """Normalize a filename for comparison - preserving number positions"""
    name = name.lower()
    name = re.sub(r'\.(epub|txt|html?)$', '', name)
    name = re.sub(r'[-_\s]+', ' ', name)
    name = re.sub(r'\[(?![^\]]*\d)[^\]]*\]', '', name)
    name = re.sub(r'\((?![^)]*\d)[^)]*\)', '', name)
    name = re.sub(r'[^\w\s\-]', ' ', name)
    name = ' '.join(name.split())
    return name.strip()


class QAScannerMixin:
    """Mixin class containing QA Scanner methods for TranslatorGUI"""
    
    def run_qa_scan(self, mode_override=None, non_interactive=False, preselected_files=None):
        """Run QA scan with mode selection and settings"""
        # Create a small loading window with icon
        loading_window = self.wm.create_simple_dialog(
            self.master,
            "Loading QA Scanner",
            width=300,
            height=120,
            modal=True,
            hide_initially=False
        )
        
        # Create content frame
        content_frame = tk.Frame(loading_window, padx=20, pady=20)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Try to add icon image if available
        status_label = None
        try:
            from PIL import Image, ImageTk
            ico_path = os.path.join(self.base_dir, 'Halgakos.ico')
            if os.path.isfile(ico_path):
                # Load icon at small size
                icon_image = Image.open(ico_path)
                icon_image = icon_image.resize((32, 32), Image.Resampling.LANCZOS)
                icon_photo = ImageTk.PhotoImage(icon_image)
                
                # Create horizontal layout
                icon_label = tk.Label(content_frame, image=icon_photo)
                icon_label.image = icon_photo  # Keep reference
                icon_label.pack(side=tk.LEFT, padx=(0, 10))
                
                # Text on the right
                text_frame = tk.Frame(content_frame)
                text_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                tk.Label(text_frame, text="Initializing QA Scanner...", 
                        font=('TkDefaultFont', 11)).pack(anchor=tk.W)
                status_label = tk.Label(text_frame, text="Loading modules...", 
                                      font=('TkDefaultFont', 9), fg='gray')
                status_label.pack(anchor=tk.W, pady=(5, 0))
            else:
                # Fallback without icon
                tk.Label(content_frame, text="Initializing QA Scanner...", 
                        font=('TkDefaultFont', 11)).pack()
                status_label = tk.Label(content_frame, text="Loading modules...", 
                                      font=('TkDefaultFont', 9), fg='gray')
                status_label.pack(pady=(10, 0))
        except ImportError:
            # No PIL, simple text only
            tk.Label(content_frame, text="Initializing QA Scanner...", 
                    font=('TkDefaultFont', 11)).pack()
            status_label = tk.Label(content_frame, text="Loading modules...", 
                                  font=('TkDefaultFont', 9), fg='gray')
            status_label.pack(pady=(10, 0))
        

        self.master.update_idletasks()
        
        try:
            # Update status
            if status_label:
                status_label.config(text="Loading translation modules...")
            loading_window.update_idletasks()
            
            if not self._lazy_load_modules():
                loading_window.destroy()
                self.append_log("❌ Failed to load QA scanner modules")
                return
            
            if status_label:
                status_label.config(text="Preparing scanner...")
            loading_window.update_idletasks()
            
            # Check for scan_html_folder in the global scope from translator_gui
            import sys
            translator_module = sys.modules.get('translator_gui')
            if translator_module is None or not hasattr(translator_module, 'scan_html_folder') or translator_module.scan_html_folder is None:
                loading_window.destroy()
                self.append_log("❌ QA scanner module is not available")
                messagebox.showerror("Module Error", "QA scanner module is not available.")
                return
            
            if hasattr(self, 'qa_thread') and self.qa_thread and self.qa_thread.is_alive():
                loading_window.destroy()
                self.stop_requested = True
                self.append_log("⛔ QA scan stop requested.")
                return
            
            # Close loading window
            loading_window.destroy()
            self.append_log("✅ QA scanner initialized successfully")
            
        except Exception as e:
            loading_window.destroy()
            self.append_log(f"❌ Error initializing QA scanner: {e}")
            return
        
        # Load QA scanner settings from config
        qa_settings = self.config.get('qa_scanner_settings', {
            'foreign_char_threshold': 10,
            'excluded_characters': '',
            'check_encoding_issues': False,
            'check_repetition': True,
            'check_translation_artifacts': False,
            'min_file_length': 0,
            'report_format': 'detailed',
            'auto_save_report': True,
            'check_missing_html_tag': True,
            'check_invalid_nesting': False,
            'check_word_count_ratio': False,
            'check_multiple_headers': True,
            'warn_name_mismatch': True,
            'cache_enabled': True,
            'cache_auto_size': False,
            'cache_show_stats': False,
            'cache_normalize_text': 10000,
            'cache_similarity_ratio': 20000,
            'cache_content_hashes': 5000,
            'cache_semantic_fingerprint': 2000,
            'cache_structural_signature': 2000,
            'cache_translation_artifacts': 1000             
        })
        # Debug: Print current settings
        print(f"[DEBUG] QA Settings: {qa_settings}")
        print(f"[DEBUG] Word count check enabled: {qa_settings.get('check_word_count_ratio', False)}")
        
        # Optionally skip mode dialog if a mode override was provided (e.g., scanning phase)
        selected_mode_value = mode_override if mode_override else None
        if selected_mode_value is None:
            # Show mode selection dialog with settings - calculate proportional sizing
            screen_width = self.master.winfo_screenwidth()
            screen_height = self.master.winfo_screenheight()
            dialog_width = int(screen_width * 0.98)  # 98% of screen width
            dialog_height = int(screen_height * 0.80)  # 80% of screen height
            
            mode_dialog = self.wm.create_simple_dialog(
                self.master,
                "Select QA Scanner Mode",
                width=dialog_width,  # Proportional width for 4-card layout
                height=dialog_height,  # Proportional height
                hide_initially=True
            )
        
        if selected_mode_value is None:
            # Set minimum size to prevent dialog from being too small
            mode_dialog.minsize(1200, 600)
            
            # Variables
            # selected_mode_value already set above
            
            # Main container with constrained expansion
            main_container = tk.Frame(mode_dialog)
            main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)  # Add padding
            
            # Content with padding
            main_frame = tk.Frame(main_container, padx=30, pady=20)  # Reduced padding
            main_frame.pack(fill=tk.X)  # Only fill horizontally, don't expand
            
            # Title with subtitle
            title_frame = tk.Frame(main_frame)
            title_frame.pack(pady=(0, 15))  # Further reduced
            
            tk.Label(title_frame, text="Select Detection Mode", 
                     font=('Arial', 28, 'bold'), fg='#f0f0f0').pack()  # Further reduced
            tk.Label(title_frame, text="Choose how sensitive the duplicate detection should be",
                     font=('Arial', 16), fg='#d0d0d0').pack(pady=(3, 0))  # Further reduced
            
            # Mode cards container - don't expand vertically to leave room for buttons
            modes_container = tk.Frame(main_frame)
            modes_container.pack(fill=tk.X, pady=(0, 10))  # Reduced bottom padding
                    
            mode_data = [
            {
                "value": "ai-hunter",
                "emoji": "🤖",
                "title": "AI HUNTER",
                "subtitle": "30% threshold",
                "features": [
                    "✓ Catches AI retranslations",
                    "✓ Different translation styles",
                    "⚠ MANY false positives",
                    "✓ Same chapter, different words",
                    "✓ Detects paraphrasing",
                    "✓ Ultimate duplicate finder"
                ],
                "bg_color": "#2a1a3e",  # Dark purple
                "hover_color": "#6a4c93",  # Medium purple
                "border_color": "#8b5cf6",
                "accent_color": "#a78bfa",
                "recommendation": "⚡ Best for finding ALL similar content"
            },
            {
                "value": "aggressive",
                "emoji": "🔥",
                "title": "AGGRESSIVE",
                "subtitle": "75% threshold",
                "features": [
                    "✓ Catches most duplicates",
                    "✓ Good for similar chapters",
                    "⚠ Some false positives",
                    "✓ Finds edited duplicates",
                    "✓ Moderate detection",
                    "✓ Balanced approach"
                ],
                "bg_color": "#3a1f1f",  # Dark red
                "hover_color": "#8b3a3a",  # Medium red
                "border_color": "#dc2626",
                "accent_color": "#ef4444",
                "recommendation": None
            },
            {
                "value": "quick-scan",
                "emoji": "⚡",
                "title": "QUICK SCAN",
                "subtitle": "85% threshold, Speed optimized",
                "features": [
                    "✓ 3-5x faster scanning",
                    "✓ Checks consecutive chapters only",
                    "✓ Simplified analysis",
                    "✓ Skips AI Hunter",
                    "✓ Good for large libraries",
                    "✓ Minimal resource usage"
                ],
                "bg_color": "#1f2937",  # Dark gray
                "hover_color": "#374151",  # Medium gray
                "border_color": "#059669",
                "accent_color": "#10b981",
                "recommendation": "✅ Recommended for quick checks & large folders"
            },
            {
                "value": "custom",
                "emoji": "⚙️",
                "title": "CUSTOM",
                "subtitle": "Configurable",
                "features": [
                    "✓ Fully customizable",
                    "✓ Set your own thresholds",
                    "✓ Advanced controls",
                    "✓ Fine-tune detection",
                    "✓ Expert mode",
                    "✓ Maximum flexibility"
                ],
                "bg_color": "#1e3a5f",  # Dark blue
                "hover_color": "#2c5aa0",  # Medium blue
                "border_color": "#3b82f6",
                "accent_color": "#60a5fa",
                "recommendation": None
            }
        ]
        
        # Restore original single-row layout (four cards across)
        if selected_mode_value is None:
            # Make each column share space evenly
            for col in range(len(mode_data)):
                modes_container.columnconfigure(col, weight=1)
            # Keep row height stable
            modes_container.rowconfigure(0, weight=0)
            
            for idx, mi in enumerate(mode_data):
                # Main card frame with initial background
                card = tk.Frame(
                    modes_container,
                    bg=mi["bg_color"],
                    highlightbackground=mi["border_color"],
                    highlightthickness=2,
                    relief='flat'
                )
                card.grid(row=0, column=idx, padx=10, pady=5, sticky='nsew')
                
                # Content frame
                content_frame = tk.Frame(card, bg=mi["bg_color"], cursor='hand2')
                content_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
                
                # Emoji
                emoji_label = tk.Label(content_frame, text=mi["emoji"], font=('Arial', 48), bg=mi["bg_color"]) 
                emoji_label.pack(pady=(0, 5))
                
                # Title
                title_label = tk.Label(content_frame, text=mi["title"], font=('Arial', 24, 'bold'), fg='white', bg=mi["bg_color"]) 
                title_label.pack()
                
                # Subtitle
                tk.Label(content_frame, text=mi["subtitle"], font=('Arial', 14), fg=mi["accent_color"], bg=mi["bg_color"]).pack(pady=(3, 10))
                
                # Features
                features_frame = tk.Frame(content_frame, bg=mi["bg_color"]) 
                features_frame.pack(fill=tk.X)
                for feature in mi["features"]:
                    tk.Label(features_frame, text=feature, font=('Arial', 11), fg='#e0e0e0', bg=mi["bg_color"], justify=tk.LEFT).pack(anchor=tk.W, pady=1)
                
                # Recommendation badge if present
                rec_frame = None
                rec_label = None
                if mi["recommendation"]:
                    rec_frame = tk.Frame(content_frame, bg=mi["accent_color"]) 
                    rec_frame.pack(pady=(10, 0), fill=tk.X)
                    rec_label = tk.Label(rec_frame, text=mi["recommendation"], font=('Arial', 11, 'bold'), fg='white', bg=mi["accent_color"], padx=8, pady=4)
                    rec_label.pack()
                
                # Click handler
                def make_click_handler(mode_value):
                    def handler(event=None):
                        nonlocal selected_mode_value
                        selected_mode_value = mode_value
                        mode_dialog.destroy()
                    return handler
                click_handler = make_click_handler(mi["value"]) 
                
                # Hover effects for this card only
                def create_hover_handlers(md, widgets):
                    def on_enter(event=None):
                        for w in widgets:
                            try:
                                w.config(bg=md["hover_color"])
                            except Exception:
                                pass
                    def on_leave(event=None):
                        for w in widgets:
                            try:
                                w.config(bg=md["bg_color"])
                            except Exception:
                                pass
                    return on_enter, on_leave
                
                all_widgets = [content_frame, emoji_label, title_label, features_frame]
                all_widgets += [child for child in features_frame.winfo_children() if isinstance(child, tk.Label)]
                if rec_frame is not None:
                    all_widgets += [rec_frame, rec_label]
                on_enter, on_leave = create_hover_handlers(mi, all_widgets)
                
                for widget in [card, content_frame, emoji_label, title_label, features_frame] + list(features_frame.winfo_children()):
                    widget.bind("<Enter>", on_enter)
                    widget.bind("<Leave>", on_leave)
                    widget.bind("<Button-1>", click_handler)
                    try:
                        widget.config(cursor='hand2')
                    except Exception:
                        pass
        
        if selected_mode_value is None:
            # Add separator line before buttons
            separator = tk.Frame(main_frame, height=1, bg='#cccccc')  # Thinner separator
            separator.pack(fill=tk.X, pady=(10, 0))
            
            # Add settings button at the bottom
            button_frame = tk.Frame(main_frame)
            button_frame.pack(fill=tk.X, pady=(10, 5))  # Reduced padding
            
            # Create inner frame for centering buttons
            button_inner = tk.Frame(button_frame)
            button_inner.pack()
            
            def show_qa_settings():
                """Show QA Scanner settings dialog"""
                self.show_qa_scanner_settings(mode_dialog, qa_settings)
            
            # Auto-search checkbox - moved to left side of Scanner Settings
            if not hasattr(self, 'qa_auto_search_output_var'):
                self.qa_auto_search_output_var = tk.BooleanVar(value=self.config.get('qa_auto_search_output', True))
            tb.Checkbutton(
                button_inner,
                text="Auto-search output",  # Renamed from "Auto-search output folder"
                variable=self.qa_auto_search_output_var,
                bootstyle="round-toggle"
            ).pack(side=tk.LEFT, padx=10)
            
            settings_btn = tb.Button(
                button_inner,
                text="⚙️  Scanner Settings",  # Added extra space
                command=show_qa_settings,
                bootstyle="info-outline",  # Changed to be more visible
                width=18,  # Slightly smaller
                padding=(8, 10)  # Reduced padding
            )
            settings_btn.pack(side=tk.LEFT, padx=10)
            
            cancel_btn = tb.Button(
                button_inner,
                text="Cancel",
                command=lambda: mode_dialog.destroy(),
                bootstyle="danger",  # Changed from outline to solid
                width=12,  # Smaller
                padding=(8, 10)  # Reduced padding
            )
            cancel_btn.pack(side=tk.LEFT, padx=10)
            
            # Handle window close (X button)
            def on_close():
                nonlocal selected_mode_value
                selected_mode_value = None
                mode_dialog.destroy()
            
            mode_dialog.protocol("WM_DELETE_WINDOW", on_close)
            
            # Show dialog
            mode_dialog.deiconify()
            mode_dialog.update_idletasks()  # Force geometry update
            mode_dialog.wait_window()
            
            # Check if user selected a mode
            if selected_mode_value is None:
                self.append_log("⚠️ QA scan canceled.")
                return

        # End of optional mode dialog
        
        # Show custom settings dialog if custom mode is selected

        # Show custom settings dialog if custom mode is selected
        if selected_mode_value == "custom":
            # Use WindowManager's setup_scrollable for proper scrolling support
            dialog, scrollable_frame, canvas = self.wm.setup_scrollable(
                self.master,
                "Custom Mode Settings",
                width=800,
                height=650,
                max_width_ratio=0.9,
                max_height_ratio=0.85
            )
            
            # Variables for custom settings
            custom_settings = {
                'similarity': tk.IntVar(value=85),
                'semantic': tk.IntVar(value=80),
                'structural': tk.IntVar(value=90),
                'word_overlap': tk.IntVar(value=75),
                'minhash_threshold': tk.IntVar(value=80),
                'consecutive_chapters': tk.IntVar(value=2),
                'check_all_pairs': tk.BooleanVar(value=False),
                'sample_size': tk.IntVar(value=3000),
                'min_text_length': tk.IntVar(value=500)
            }
            
            # Title using consistent styling
            title_label = tk.Label(scrollable_frame, text="Configure Custom Detection Settings", 
                                  font=('Arial', 20, 'bold'))
            title_label.pack(pady=(0, 20))
            
            # Detection Thresholds Section using ttkbootstrap
            threshold_frame = tb.LabelFrame(scrollable_frame, text="Detection Thresholds (%)", 
                                            padding=25, bootstyle="secondary")
            threshold_frame.pack(fill='x', padx=20, pady=(0, 25))
            
            threshold_descriptions = {
                'similarity': ('Text Similarity', 'Character-by-character comparison'),
                'semantic': ('Semantic Analysis', 'Meaning and context matching'),
                'structural': ('Structural Patterns', 'Document structure similarity'),
                'word_overlap': ('Word Overlap', 'Common words between texts'),
                'minhash_threshold': ('MinHash Similarity', 'Fast approximate matching')
            }
            
            # Create percentage labels dictionary to store references
            percentage_labels = {}
            
            for setting_key, (label_text, description) in threshold_descriptions.items():
                # Container for each threshold
                row_frame = tk.Frame(threshold_frame)
                row_frame.pack(fill='x', pady=8)
                
                # Left side - labels
                label_container = tk.Frame(row_frame)
                label_container.pack(side='left', fill='x', expand=True)
                
                main_label = tk.Label(label_container, text=f"{label_text} - {description}:",
                                     font=('TkDefaultFont', 11))
                main_label.pack(anchor='w')
                
                # Right side - slider and percentage
                slider_container = tk.Frame(row_frame)
                slider_container.pack(side='right', padx=(20, 0))
                
                # Percentage label (shows current value)
                percentage_label = tk.Label(slider_container, text=f"{custom_settings[setting_key].get()}%",
                                           font=('TkDefaultFont', 12, 'bold'), width=5, anchor='e')
                percentage_label.pack(side='right', padx=(10, 0))
                percentage_labels[setting_key] = percentage_label
                
                # Create slider
                slider = tb.Scale(slider_container, 
                                 from_=10, to=100,
                                 variable=custom_settings[setting_key],
                                 bootstyle="info",
                                 length=300,
                                 orient='horizontal')
                slider.pack(side='right')
                
                # Update percentage label when slider moves
                def create_update_function(key, label):
                    def update_percentage(*args):
                        value = custom_settings[key].get()
                        label.config(text=f"{value}%")
                    return update_percentage
                
                # Bind the update function
                update_func = create_update_function(setting_key, percentage_label)
                custom_settings[setting_key].trace('w', update_func)
            
            # Processing Options Section
            options_frame = tb.LabelFrame(scrollable_frame, text="Processing Options", 
                                          padding=20, bootstyle="secondary")
            options_frame.pack(fill='x', padx=20, pady=15)
            
            # Consecutive chapters option with spinbox
            consec_frame = tk.Frame(options_frame)
            consec_frame.pack(fill='x', pady=5)
            
            tk.Label(consec_frame, text="Consecutive chapters to check:", 
                     font=('TkDefaultFont', 11)).pack(side='left')
            
            tb.Spinbox(consec_frame, from_=1, to=10, 
                       textvariable=custom_settings['consecutive_chapters'],
                       width=10, bootstyle="info").pack(side='left', padx=(10, 0))
            
            # Sample size option
            sample_frame = tk.Frame(options_frame)
            sample_frame.pack(fill='x', pady=5)
            
            tk.Label(sample_frame, text="Sample size for comparison (characters):", 
                     font=('TkDefaultFont', 11)).pack(side='left')
            
            # Sample size spinbox with larger range
            sample_spinbox = tb.Spinbox(sample_frame, from_=1000, to=10000, increment=500,
                                        textvariable=custom_settings['sample_size'],
                                        width=10, bootstyle="info")
            sample_spinbox.pack(side='left', padx=(10, 0))
            
            # Minimum text length option
            min_length_frame = tk.Frame(options_frame)
            min_length_frame.pack(fill='x', pady=5)
            
            tk.Label(min_length_frame, text="Minimum text length to process (characters):", 
                     font=('TkDefaultFont', 11)).pack(side='left')
            
            # Minimum length spinbox
            min_length_spinbox = tb.Spinbox(min_length_frame, from_=100, to=5000, increment=100,
                                            textvariable=custom_settings['min_text_length'],
                                            width=10, bootstyle="info")
            min_length_spinbox.pack(side='left', padx=(10, 0))
            
            # Check all file pairs option
            tb.Checkbutton(options_frame, text="Check all file pairs (slower but more thorough)",
                           variable=custom_settings['check_all_pairs'],
                           bootstyle="primary").pack(anchor='w', pady=8)
            
            # Create button frame at bottom (inside scrollable_frame)
            button_frame = tk.Frame(scrollable_frame)
            button_frame.pack(fill='x', pady=(30, 20))
            
            # Center buttons using inner frame
            button_inner = tk.Frame(button_frame)
            button_inner.pack()
            
            # Flag to track if settings were saved
            settings_saved = False
            
            def save_custom_settings():
                """Save custom settings and close dialog"""
                nonlocal settings_saved
                qa_settings['custom_mode_settings'] = {
                    'thresholds': {
                        'similarity': custom_settings['similarity'].get() / 100,
                        'semantic': custom_settings['semantic'].get() / 100,
                        'structural': custom_settings['structural'].get() / 100,
                        'word_overlap': custom_settings['word_overlap'].get() / 100,
                        'minhash_threshold': custom_settings['minhash_threshold'].get() / 100
                    },
                    'consecutive_chapters': custom_settings['consecutive_chapters'].get(),
                    'check_all_pairs': custom_settings['check_all_pairs'].get(),
                    'sample_size': custom_settings['sample_size'].get(),
                    'min_text_length': custom_settings['min_text_length'].get()
                }
                settings_saved = True
                self.append_log("✅ Custom detection settings saved")
                dialog._cleanup_scrolling()  # Clean up scrolling bindings
                dialog.destroy()
            
            def reset_to_defaults():
                """Reset all values to default settings"""
                if messagebox.askyesno("Reset to Defaults", 
                                       "Reset all values to default settings?",
                                       parent=dialog):
                    custom_settings['similarity'].set(85)
                    custom_settings['semantic'].set(80)
                    custom_settings['structural'].set(90)
                    custom_settings['word_overlap'].set(75)
                    custom_settings['minhash_threshold'].set(80)
                    custom_settings['consecutive_chapters'].set(2)
                    custom_settings['check_all_pairs'].set(False)
                    custom_settings['sample_size'].set(3000)
                    custom_settings['min_text_length'].set(500)
                    self.append_log("ℹ️ Settings reset to defaults")
            
            def cancel_settings():
                """Cancel without saving"""
                nonlocal settings_saved
                if not settings_saved:
                    # Check if any settings were changed
                    defaults = {
                        'similarity': 85,
                        'semantic': 80,
                        'structural': 90,
                        'word_overlap': 75,
                        'minhash_threshold': 80,
                        'consecutive_chapters': 2,
                        'check_all_pairs': False,
                        'sample_size': 3000,
                        'min_text_length': 500
                    }
                    
                    changed = False
                    for key, default_val in defaults.items():
                        if custom_settings[key].get() != default_val:
                            changed = True
                            break
                    
                    if changed:
                        if messagebox.askyesno("Unsaved Changes", 
                                              "You have unsaved changes. Are you sure you want to cancel?",
                                              parent=dialog):
                            dialog._cleanup_scrolling()
                            dialog.destroy()
                    else:
                        dialog._cleanup_scrolling()
                        dialog.destroy()
                else:
                    dialog._cleanup_scrolling()
                    dialog.destroy()
            
            # Use ttkbootstrap buttons with better styling
            tb.Button(button_inner, text="Cancel", 
                     command=cancel_settings,
                     bootstyle="secondary", width=15).pack(side='left', padx=5)
            
            tb.Button(button_inner, text="Reset Defaults", 
                     command=reset_to_defaults,
                     bootstyle="warning", width=15).pack(side='left', padx=5)
            
            tb.Button(button_inner, text="Start Scan", 
                     command=save_custom_settings,
                     bootstyle="success", width=15).pack(side='left', padx=5)
            
            # Use WindowManager's auto-resize
            self.wm.auto_resize_dialog(dialog, canvas, max_width_ratio=0.9, max_height_ratio=0.72)
            
            # Handle window close properly - treat as cancel
            dialog.protocol("WM_DELETE_WINDOW", cancel_settings)
            
            # Wait for dialog to close
            dialog.wait_window()
            
            # If user cancelled at this dialog, cancel the whole scan
            if not settings_saved:
                self.append_log("⚠️ QA scan canceled - no custom settings were saved.")
                return
        # Check if word count cross-reference is enabled but no EPUB is selected
        check_word_count = qa_settings.get('check_word_count_ratio', False)
        epub_files_to_scan = []
        primary_epub_path = None
        
        # ALWAYS populate epub_files_to_scan for auto-search, regardless of word count checking
        # First check if current selection actually contains EPUBs
        current_epub_files = []
        if hasattr(self, 'selected_files') and self.selected_files:
            current_epub_files = [f for f in self.selected_files if f.lower().endswith('.epub')]
            print(f"[DEBUG] Current selection contains {len(current_epub_files)} EPUB files")
        
        if current_epub_files:
            # Use EPUBs from current selection
            epub_files_to_scan = current_epub_files
            print(f"[DEBUG] Using {len(epub_files_to_scan)} EPUB files from current selection")
        else:
            # No EPUBs in current selection - check if we have stored EPUBs
            primary_epub_path = self.get_current_epub_path()
            print(f"[DEBUG] get_current_epub_path returned: {primary_epub_path}")
            
            if primary_epub_path:
                epub_files_to_scan = [primary_epub_path]
                print(f"[DEBUG] Using stored EPUB file for auto-search")
        
        # Now handle word count specific logic if enabled
        if check_word_count:
            print("[DEBUG] Word count check is enabled, validating EPUB availability...")
            
            # Check if we have EPUBs for word count analysis
            if not epub_files_to_scan:
                # No EPUBs available for word count analysis
                result = messagebox.askyesnocancel(
                    "No Source EPUB Selected",
                    "Word count cross-reference is enabled but no source EPUB file is selected.\n\n" +
                    "Would you like to:\n" +
                    "• YES - Continue scan without word count analysis\n" +
                    "• NO - Select an EPUB file now\n" +
                    "• CANCEL - Cancel the scan",
                    icon='warning'
                )
                
                if result is None:  # Cancel
                    self.append_log("⚠️ QA scan canceled.")
                    return
                elif result is False:  # No - Select EPUB now
                    epub_path = filedialog.askopenfilename(
                        title="Select Source EPUB File",
                        filetypes=[("EPUB files", "*.epub"), ("All files", "*.*")]
                    )
                    
                    if not epub_path:
                        retry = messagebox.askyesno(
                            "No File Selected",
                            "No EPUB file was selected.\n\n" +
                            "Do you want to continue the scan without word count analysis?",
                            icon='question'
                        )
                        
                        if not retry:
                            self.append_log("⚠️ QA scan canceled.")
                            return
                        else:
                            qa_settings = qa_settings.copy()
                            qa_settings['check_word_count_ratio'] = False
                            self.append_log("ℹ️ Proceeding without word count analysis.")
                            epub_files_to_scan = []
                    else:
                        self.selected_epub_path = epub_path
                        self.config['last_epub_path'] = epub_path
                        self.save_config(show_message=False)
                        self.append_log(f"✅ Selected EPUB: {os.path.basename(epub_path)}")
                        epub_files_to_scan = [epub_path]
                else:  # Yes - Continue without word count
                    qa_settings = qa_settings.copy()
                    qa_settings['check_word_count_ratio'] = False
                    self.append_log("ℹ️ Proceeding without word count analysis.")
                    epub_files_to_scan = []
        # Persist latest auto-search preference
        try:
            self.config['qa_auto_search_output'] = bool(self.qa_auto_search_output_var.get())
            self.save_config(show_message=False)
        except Exception:
            pass
        
        # Try to auto-detect output folders based on EPUB files
        folders_to_scan = []
        auto_search_enabled = self.config.get('qa_auto_search_output', True)
        try:
            if hasattr(self, 'qa_auto_search_output_var'):
                auto_search_enabled = bool(self.qa_auto_search_output_var.get())
        except Exception:
            pass
        
        # Debug output for scanning phase
        if non_interactive:
            self.append_log(f"📝 Debug: auto_search_enabled = {auto_search_enabled}")
            self.append_log(f"📝 Debug: epub_files_to_scan = {len(epub_files_to_scan)} files")
            self.append_log(f"📝 Debug: Will run folder detection = {auto_search_enabled and epub_files_to_scan}")
        
        if auto_search_enabled and epub_files_to_scan:
            # Process each EPUB file to find its corresponding output folder
            self.append_log(f"🔍 DEBUG: Auto-search running with {len(epub_files_to_scan)} EPUB files")
            for epub_path in epub_files_to_scan:
                self.append_log(f"🔍 DEBUG: Processing EPUB: {epub_path}")
                try:
                    epub_base = os.path.splitext(os.path.basename(epub_path))[0]
                    current_dir = os.getcwd()
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    
                    self.append_log(f"🔍 DEBUG: EPUB base name: '{epub_base}'")
                    self.append_log(f"🔍 DEBUG: Current dir: {current_dir}")
                    self.append_log(f"🔍 DEBUG: Script dir: {script_dir}")
                    
                    # Check the most common locations in order of priority
                    candidates = [
                        os.path.join(current_dir, epub_base),        # current working directory
                        os.path.join(script_dir, epub_base),         # src directory (where output typically goes)
                        os.path.join(current_dir, 'src', epub_base), # src subdirectory from current dir
                    ]
                    
                    folder_found = None
                    for i, candidate in enumerate(candidates):
                        exists = os.path.isdir(candidate)
                        self.append_log(f"  [{epub_base}] Checking candidate {i+1}: {candidate} - {'EXISTS' if exists else 'NOT FOUND'}")
                        
                        if exists:
                            # Verify the folder actually contains HTML/XHTML files
                            try:
                                files = os.listdir(candidate)
                                html_files = [f for f in files if f.lower().endswith(('.html', '.xhtml', '.htm'))]
                                if html_files:
                                    folder_found = candidate
                                    self.append_log(f"📁 Auto-selected output folder for {epub_base}: {folder_found}")
                                    self.append_log(f"   Found {len(html_files)} HTML/XHTML files to scan")
                                    break
                                else:
                                    self.append_log(f"  [{epub_base}] Folder exists but contains no HTML/XHTML files: {candidate}")
                            except Exception as e:
                                self.append_log(f"  [{epub_base}] Error checking files in {candidate}: {e}")
                    
                    if folder_found:
                        folders_to_scan.append(folder_found)
                        self.append_log(f"🔍 DEBUG: Added to folders_to_scan: {folder_found}")
                    else:
                        self.append_log(f"  ⚠️ No output folder found for {epub_base}")
                            
                except Exception as e:
                    self.append_log(f"  ❌ Error processing {epub_base}: {e}")
            
            self.append_log(f"🔍 DEBUG: Final folders_to_scan: {folders_to_scan}")
        
        # Fallback behavior - if no folders found through auto-detection
        if not folders_to_scan:
            if auto_search_enabled:
                # Auto-search failed, offer manual selection as fallback
                self.append_log("⚠️ Auto-search enabled but no matching output folder found")
                self.append_log("📁 Falling back to manual folder selection...")
                
                selected_folder = filedialog.askdirectory(title="Auto-search failed - Select Output Folder to Scan")
                if not selected_folder:
                    self.append_log("⚠️ QA scan canceled - no folder selected.")
                    return
                
                # Verify the selected folder contains scannable files
                try:
                    files = os.listdir(selected_folder)
                    html_files = [f for f in files if f.lower().endswith(('.html', '.xhtml', '.htm'))]
                    if html_files:
                        folders_to_scan.append(selected_folder)
                        self.append_log(f"✓ Manual selection: {os.path.basename(selected_folder)} ({len(html_files)} HTML/XHTML files)")
                    else:
                        self.append_log(f"❌ Selected folder contains no HTML/XHTML files: {selected_folder}")
                        return
                except Exception as e:
                    self.append_log(f"❌ Error checking selected folder: {e}")
                    return
            if non_interactive:
                # Add debug info for scanning phase
                if epub_files_to_scan:
                    self.append_log(f"⚠️ Scanning phase: No matching output folders found for {len(epub_files_to_scan)} EPUB file(s)")
                    for epub_path in epub_files_to_scan:
                        epub_base = os.path.splitext(os.path.basename(epub_path))[0]
                        current_dir = os.getcwd()
                        expected_folder = os.path.join(current_dir, epub_base)
                        self.append_log(f"  [{epub_base}] Expected: {expected_folder}")
                        self.append_log(f"  [{epub_base}] Exists: {os.path.isdir(expected_folder)}")
                    
                    # List actual folders in current directory for debugging
                    try:
                        current_dir = os.getcwd()
                        actual_folders = [d for d in os.listdir(current_dir) if os.path.isdir(os.path.join(current_dir, d)) and not d.startswith('.')]
                        if actual_folders:
                            self.append_log(f"  Available folders: {', '.join(actual_folders[:10])}{'...' if len(actual_folders) > 10 else ''}")
                    except Exception:
                        pass
                else:
                    self.append_log("⚠️ Scanning phase: No EPUB files available for folder detection")
                
                self.append_log("⚠️ Skipping scan")
                return
            
            # Clean single folder selection - no messageboxes, no harassment
            self.append_log("📁 Select folder to scan...")
            
            folders_to_scan = []
            
            # Simply select one folder - clean and simple
            selected_folder = filedialog.askdirectory(title="Select Folder with HTML Files")
            if not selected_folder:
                self.append_log("⚠️ QA scan canceled - no folder selected.")
                return
            
            folders_to_scan.append(selected_folder)
            self.append_log(f"  ✓ Selected folder: {os.path.basename(selected_folder)}")
            self.append_log(f"📁 Single folder scan mode - scanning: {os.path.basename(folders_to_scan[0])}")

        mode = selected_mode_value
        
        # Initialize epub_path for use in run_scan() function
        # This ensures epub_path is always defined even when manually selecting folders
        epub_path = None
        if epub_files_to_scan:
            epub_path = epub_files_to_scan[0]  # Use first EPUB if multiple
            self.append_log(f"📚 Using EPUB from scan list: {os.path.basename(epub_path)}")
        elif hasattr(self, 'selected_epub_path') and self.selected_epub_path:
            epub_path = self.selected_epub_path
            self.append_log(f"📚 Using stored EPUB: {os.path.basename(epub_path)}")
        elif primary_epub_path:
            epub_path = primary_epub_path
            self.append_log(f"📚 Using primary EPUB: {os.path.basename(epub_path)}")
        else:
            self.append_log("ℹ️ No EPUB file configured (word count analysis will be disabled if needed)")
        
        # Initialize global selected_files that applies to single-folder scans
        global_selected_files = None
        if len(folders_to_scan) == 1 and preselected_files:
            global_selected_files = list(preselected_files)
        elif len(folders_to_scan) == 1 and (not non_interactive) and (not auto_search_enabled):
            # Scan all files in the folder - no messageboxes asking about specific files
            # User can set up file preselection if they need specific files
            pass
        
        # Log bulk scan start
        if len(folders_to_scan) == 1:
            self.append_log(f"🔍 Starting QA scan in {mode.upper()} mode for folder: {folders_to_scan[0]}")
        else:
            self.append_log(f"🔍 Starting bulk QA scan in {mode.upper()} mode for {len(folders_to_scan)} folders")
        
        self.stop_requested = False

        # Extract cache configuration from qa_settings
        cache_config = {
            'enabled': qa_settings.get('cache_enabled', True),
            'auto_size': qa_settings.get('cache_auto_size', False),
            'show_stats': qa_settings.get('cache_show_stats', False),
            'sizes': {}
        }
        
        # Get individual cache sizes
        for cache_name in ['normalize_text', 'similarity_ratio', 'content_hashes', 
                          'semantic_fingerprint', 'structural_signature', 'translation_artifacts']:
            size = qa_settings.get(f'cache_{cache_name}', None)
            if size is not None:
                # Convert -1 to None for unlimited
                cache_config['sizes'][cache_name] = None if size == -1 else size
        
        # Create custom settings that includes cache config
        custom_settings = {
            'qa_settings': qa_settings,
            'cache_config': cache_config,
            'log_cache_stats': qa_settings.get('cache_show_stats', False)
        }
 
        def run_scan():
            # Update UI on the main thread
            self.master.after(0, self.update_run_button)
            self.master.after(0, lambda: self.qa_button.config(text="Stop Scan", command=self.stop_qa_scan, bootstyle="danger"))
            
            try:
                # Extract cache configuration from qa_settings
                cache_config = {
                    'enabled': qa_settings.get('cache_enabled', True),
                    'auto_size': qa_settings.get('cache_auto_size', False),
                    'show_stats': qa_settings.get('cache_show_stats', False),
                    'sizes': {}
                }
                
                # Get individual cache sizes
                for cache_name in ['normalize_text', 'similarity_ratio', 'content_hashes', 
                                  'semantic_fingerprint', 'structural_signature', 'translation_artifacts']:
                    size = qa_settings.get(f'cache_{cache_name}', None)
                    if size is not None:
                        # Convert -1 to None for unlimited
                        cache_config['sizes'][cache_name] = None if size == -1 else size
                
                # Configure the cache BEFORE calling scan_html_folder
                from scan_html_folder import configure_qa_cache
                configure_qa_cache(cache_config)
                
                # Loop through all selected folders for bulk scanning
                successful_scans = 0
                failed_scans = 0
                
                for i, current_folder in enumerate(folders_to_scan):
                    if self.stop_requested:
                        self.append_log(f"⚠️ Bulk scan stopped by user at folder {i+1}/{len(folders_to_scan)}")
                        break
                    
                    folder_name = os.path.basename(current_folder)
                    if len(folders_to_scan) > 1:
                        self.append_log(f"\n📁 [{i+1}/{len(folders_to_scan)}] Scanning folder: {folder_name}")
                    
                    # Determine the correct EPUB path for this specific folder
                    current_epub_path = epub_path
                    current_qa_settings = qa_settings.copy()
                    
                    # For bulk scanning, try to find a matching EPUB for each folder
                    if len(folders_to_scan) > 1 and current_qa_settings.get('check_word_count_ratio', False):
                        # Try to find EPUB file matching this specific folder
                        folder_basename = os.path.basename(current_folder.rstrip('/\\'))
                        self.append_log(f"  🔍 Searching for EPUB matching folder: {folder_basename}")
                        
                        # Look for EPUB in various locations
                        folder_parent = os.path.dirname(current_folder)
                        
                        # Simple exact matching first, with minimal suffix handling
                        base_name = folder_basename
                        
                        # Only handle the most common output suffixes
                        common_suffixes = ['_output', '_translated', '_en']
                        for suffix in common_suffixes:
                            if base_name.endswith(suffix):
                                base_name = base_name[:-len(suffix)]
                                break
                        
                        # Simple EPUB search - focus on exact matching
                        search_names = [folder_basename]  # Start with exact folder name
                        if base_name != folder_basename:  # Add base name only if different
                            search_names.append(base_name)
                        
                        potential_epub_paths = [
                            # Most common locations in order of priority
                            os.path.join(folder_parent, f"{folder_basename}.epub"),  # Same directory as output folder
                            os.path.join(folder_parent, f"{base_name}.epub"),        # Same directory with base name
                            os.path.join(current_folder, f"{folder_basename}.epub"), # Inside the output folder
                            os.path.join(current_folder, f"{base_name}.epub"),       # Inside with base name
                        ]
                        
                        # Find the first existing EPUB
                        folder_epub_path = None
                        for potential_path in potential_epub_paths:
                            if os.path.isfile(potential_path):
                                folder_epub_path = potential_path
                                if len(folders_to_scan) > 1:
                                    self.append_log(f"      Found matching EPUB: {os.path.basename(potential_path)}")
                                break
                        
                        if folder_epub_path:
                            current_epub_path = folder_epub_path
                            if len(folders_to_scan) > 1:  # Only log for bulk scans
                                self.append_log(f"  📖 Using EPUB: {os.path.basename(current_epub_path)}")
                        else:
                            # NO FALLBACK TO GLOBAL EPUB FOR BULK SCANS - This prevents wrong EPUB usage!
                            if len(folders_to_scan) > 1:
                                self.append_log(f"  ⚠️ No matching EPUB found for folder '{folder_name}' - disabling word count analysis")
                                expected_names = ', '.join([f"{name}.epub" for name in search_names])
                                self.append_log(f"      Expected EPUB names: {expected_names}")
                                current_epub_path = None
                            elif current_epub_path:  # Single folder scan can use global EPUB
                                self.append_log(f"  📖 Using global EPUB: {os.path.basename(current_epub_path)} (no folder-specific EPUB found)")
                            else:
                                current_epub_path = None
                            
                            # Disable word count analysis when no matching EPUB is found
                            if not current_epub_path:
                                current_qa_settings = current_qa_settings.copy()
                                current_qa_settings['check_word_count_ratio'] = False
                    
                    # Check for EPUB/folder name mismatch
                    if current_epub_path and current_qa_settings.get('check_word_count_ratio', False) and current_qa_settings.get('warn_name_mismatch', True):
                        epub_name = os.path.splitext(os.path.basename(current_epub_path))[0]
                        folder_name_for_check = os.path.basename(current_folder.rstrip('/\\'))
                        
                        if not check_epub_folder_match(epub_name, folder_name_for_check, current_qa_settings.get('custom_output_suffixes', '')):
                            if len(folders_to_scan) == 1:
                                # Interactive dialog for single folder scans
                                result = messagebox.askyesnocancel(
                                    "EPUB/Folder Name Mismatch",
                                    f"The source EPUB and output folder names don't match:\n\n" +
                                    f"📖 EPUB: {epub_name}\n" +
                                    f"📁 Folder: {folder_name_for_check}\n\n" +
                                    "This might mean you're comparing the wrong files.\n" +
                                    "Would you like to:\n" +
                                    "• YES - Continue anyway (I'm sure these match)\n" +
                                    "• NO - Select a different EPUB file\n" +
                                    "• CANCEL - Cancel the scan",
                                    icon='warning'
                                )
                                
                                if result is None:  # Cancel
                                    self.append_log("⚠️ QA scan canceled due to EPUB/folder mismatch.")
                                    return
                                elif result is False:  # No - select different EPUB
                                    new_epub_path = filedialog.askopenfilename(
                                        title="Select Different Source EPUB File",
                                        filetypes=[("EPUB files", "*.epub"), ("All files", "*.*")]
                                    )
                                    
                                    if new_epub_path:
                                        current_epub_path = new_epub_path
                                        self.selected_epub_path = new_epub_path
                                        self.config['last_epub_path'] = new_epub_path
                                        self.save_config(show_message=False)
                                        self.append_log(f"✅ Updated EPUB: {os.path.basename(new_epub_path)}")
                                    else:
                                        proceed = messagebox.askyesno(
                                            "No File Selected",
                                            "No EPUB file was selected.\n\n" +
                                            "Continue scan without word count analysis?",
                                            icon='question'
                                        )
                                        if not proceed:
                                            self.append_log("⚠️ QA scan canceled.")
                                            return
                                        else:
                                            current_qa_settings = current_qa_settings.copy()
                                            current_qa_settings['check_word_count_ratio'] = False
                                            current_epub_path = None
                                            self.append_log("ℹ️ Proceeding without word count analysis.")
                                # If YES, just continue with warning
                            else:
                                # For bulk scans, just warn and continue
                                self.append_log(f"  ⚠️ Warning: EPUB/folder name mismatch - {epub_name} vs {folder_name_for_check}")
                    
                    try:
                        # Determine selected_files for this folder
                        current_selected_files = None
                        if global_selected_files and len(folders_to_scan) == 1:
                            current_selected_files = global_selected_files
                        
                        # Pass the QA settings to scan_html_folder
                        # Get scan_html_folder from translator_gui's global scope
                        import translator_gui
                        scan_func = translator_gui.scan_html_folder
                        scan_func(
                            current_folder, 
                            log=self.append_log, 
                            stop_flag=lambda: self.stop_requested, 
                            mode=mode,
                            qa_settings=current_qa_settings,
                            epub_path=current_epub_path,
                            selected_files=current_selected_files
                        )
                        
                        successful_scans += 1
                        if len(folders_to_scan) > 1:
                            self.append_log(f"✅ Folder '{folder_name}' scan completed successfully")
                    
                    except Exception as folder_error:
                        failed_scans += 1
                        self.append_log(f"❌ Folder '{folder_name}' scan failed: {folder_error}")
                        if len(folders_to_scan) == 1:
                            # Re-raise for single folder scans
                            raise
                
                # Final summary for bulk scans
                if len(folders_to_scan) > 1:
                    self.append_log(f"\n📋 Bulk scan summary: {successful_scans} successful, {failed_scans} failed")
                
                # If show_stats is enabled, log cache statistics
                if qa_settings.get('cache_show_stats', False):
                    from scan_html_folder import get_cache_info
                    cache_stats = get_cache_info()
                    self.append_log("\n📊 Cache Performance Statistics:")
                    for name, info in cache_stats.items():
                        if info:  # Check if info exists
                            hit_rate = info.hits / (info.hits + info.misses) if (info.hits + info.misses) > 0 else 0
                            self.append_log(f"  {name}: {info.hits} hits, {info.misses} misses ({hit_rate:.1%} hit rate)")
                
                if len(folders_to_scan) == 1:
                    self.append_log("✅ QA scan completed successfully.")
                else:
                    self.append_log("✅ Bulk QA scan completed.")
    
            except Exception as e:
                self.append_log(f"❌ QA scan error: {e}")
                self.append_log(f"Traceback: {traceback.format_exc()}")
            finally:
                # Clear thread/future refs so buttons re-enable
                self.qa_thread = None
                if hasattr(self, 'qa_future'):
                    try:
                        self.qa_future = None
                    except Exception:
                        pass
                self.master.after(0, self.update_run_button)
                # Check if scan_html_folder is available in translator_gui
                import translator_gui
                scan_available = hasattr(translator_gui, 'scan_html_folder') and translator_gui.scan_html_folder is not None
                self.master.after(0, lambda: self.qa_button.config(
                    text="QA Scan", 
                    command=self.run_qa_scan, 
                    bootstyle="warning",
                    state=tk.NORMAL if scan_available else tk.DISABLED
                ))
        
        # Run via shared executor
        self._ensure_executor()
        if self.executor:
            self.qa_future = self.executor.submit(run_scan)
            # Ensure UI is refreshed when QA work completes
            def _qa_done_callback(f):
                try:
                    self.master.after(0, lambda: (setattr(self, 'qa_future', None), self.update_run_button()))
                except Exception:
                    pass
            try:
                self.qa_future.add_done_callback(_qa_done_callback)
            except Exception:
                pass
        else:
            self.qa_thread = threading.Thread(target=run_scan, daemon=True)
            self.qa_thread.start()

    def show_qa_scanner_settings(self, parent_dialog, qa_settings):
        """Show QA Scanner settings dialog using WindowManager properly"""
        # Use setup_scrollable from WindowManager - NOT create_scrollable_dialog
        dialog, scrollable_frame, canvas = self.wm.setup_scrollable(
            parent_dialog,
            "QA Scanner Settings",
            width=800,
            height=None,  # Let WindowManager calculate optimal height
            modal=True,
            resizable=True,
            max_width_ratio=0.9,
            max_height_ratio=0.9
        )
        
        # Main settings frame
        main_frame = tk.Frame(scrollable_frame, padx=30, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = tk.Label(
            main_frame,
            text="QA Scanner Settings",
            font=('Arial', 24, 'bold')
        )
        title_label.pack(pady=(0, 20))
        
        # Foreign Character Settings Section
        foreign_section = tk.LabelFrame(
            main_frame,
            text="Foreign Character Detection",
            font=('Arial', 12, 'bold'),
            padx=20,
            pady=15
        )
        foreign_section.pack(fill=tk.X, pady=(0, 20))
        
        # Target Language setting
        target_lang_frame = tk.Frame(foreign_section)
        target_lang_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(
            target_lang_frame,
            text="Target language:",
            font=('Arial', 10)
        ).pack(side=tk.LEFT)
        
        # Capitalize the stored value for display in combobox
        stored_language = qa_settings.get('target_language', 'english')
        display_language = stored_language.capitalize()
        target_language_var = tk.StringVar(value=display_language)
        target_language_options = [
            'English', 'Spanish', 'French', 'German', 'Portuguese', 
            'Italian', 'Russian', 'Japanese', 'Korean', 'Chinese', 
            'Arabic', 'Hebrew', 'Thai'
        ]
        
        target_language_combo = tb.Combobox(
            target_lang_frame,
            textvariable=target_language_var,
            values=target_language_options,
            state='readonly',
            width=15,
            bootstyle="primary"
        )
        target_language_combo.pack(side=tk.LEFT, padx=(10, 0))
        
        # Disable mousewheel scrolling to prevent accidental changes
        self.ui.disable_spinbox_mousewheel(target_language_combo)
        
        tk.Label(
            target_lang_frame,
            text="(characters from other scripts will be flagged)",
            font=('Arial', 9),
            fg='gray'
        ).pack(side=tk.LEFT, padx=(10, 0))
        
        # Threshold setting
        threshold_frame = tk.Frame(foreign_section)
        threshold_frame.pack(fill=tk.X, pady=(10, 10))
        
        tk.Label(
            threshold_frame,
            text="Minimum foreign characters to flag:",
            font=('Arial', 10)
        ).pack(side=tk.LEFT)
        
        threshold_var = tk.IntVar(value=qa_settings.get('foreign_char_threshold', 10))
        threshold_spinbox = tb.Spinbox(
            threshold_frame,
            from_=0,
            to=1000,
            textvariable=threshold_var,
            width=10,
            bootstyle="primary"
        )
        threshold_spinbox.pack(side=tk.LEFT, padx=(10, 0))
        
        # Disable mousewheel scrolling on spinbox
        self.ui.disable_spinbox_mousewheel(threshold_spinbox)
        
        tk.Label(
            threshold_frame,
            text="(0 = always flag, higher = more tolerant)",
            font=('Arial', 9),
            fg='gray'
        ).pack(side=tk.LEFT, padx=(10, 0))
        
        # Excluded characters - using UIHelper for scrollable text
        excluded_frame = tk.Frame(foreign_section)
        excluded_frame.pack(fill=tk.X, pady=(10, 0))
        
        tk.Label(
            excluded_frame,
            text="Additional characters to exclude from detection:",
            font=('Arial', 10)
        ).pack(anchor=tk.W)
        
        # Use regular Text widget with manual scroll setup instead of ScrolledText
        excluded_text_frame = tk.Frame(excluded_frame)
        excluded_text_frame.pack(fill=tk.X, pady=(5, 0))
        
        excluded_text = tk.Text(
            excluded_text_frame,
            height=7,
            width=60,
            font=('Consolas', 10),
            wrap=tk.WORD,
            undo=True
        )
        excluded_text.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Add scrollbar manually
        excluded_scrollbar = ttk.Scrollbar(excluded_text_frame, orient="vertical", command=excluded_text.yview)
        excluded_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        excluded_text.configure(yscrollcommand=excluded_scrollbar.set)
        
        # Setup undo/redo for the text widget
        self.ui.setup_text_undo_redo(excluded_text)
        
        excluded_text.insert(1.0, qa_settings.get('excluded_characters', ''))
        
        tk.Label(
            excluded_frame,
            text="Enter characters separated by spaces (e.g., ™ © ® • …)",
            font=('Arial', 9),
            fg='gray'
        ).pack(anchor=tk.W)
        
        # Detection Options Section
        detection_section = tk.LabelFrame(
            main_frame,
            text="Detection Options",
            font=('Arial', 12, 'bold'),
            padx=20,
            pady=15
        )
        detection_section.pack(fill=tk.X, pady=(0, 20))
        
        # Checkboxes for detection options
        check_encoding_var = tk.BooleanVar(value=qa_settings.get('check_encoding_issues', False))
        check_repetition_var = tk.BooleanVar(value=qa_settings.get('check_repetition', True))
        check_artifacts_var = tk.BooleanVar(value=qa_settings.get('check_translation_artifacts', False))
        check_glossary_var = tk.BooleanVar(value=qa_settings.get('check_glossary_leakage', True))
        
        tb.Checkbutton(
            detection_section,
            text="Check for encoding issues (�, □, ◇)",
            variable=check_encoding_var,
            bootstyle="primary"
        ).pack(anchor=tk.W, pady=2)
        
        tb.Checkbutton(
            detection_section,
            text="Check for excessive repetition",
            variable=check_repetition_var,
            bootstyle="primary"
        ).pack(anchor=tk.W, pady=2)
        
        tb.Checkbutton(
            detection_section,
            text="Check for translation artifacts (MTL notes, watermarks)",
            variable=check_artifacts_var,
            bootstyle="primary"
        ).pack(anchor=tk.W, pady=2)
        tb.Checkbutton(
            detection_section,
            text="Check for glossary leakage (raw glossary entries in translation)",
            variable=check_glossary_var,
            bootstyle="primary"
        ).pack(anchor=tk.W, pady=2)
        
        # File Processing Section
        file_section = tk.LabelFrame(
            main_frame,
            text="File Processing",
            font=('Arial', 12, 'bold'),
            padx=20,
            pady=15
        )
        file_section.pack(fill=tk.X, pady=(0, 20))
        
        # Minimum file length
        min_length_frame = tk.Frame(file_section)
        min_length_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(
            min_length_frame,
            text="Minimum file length (characters):",
            font=('Arial', 10)
        ).pack(side=tk.LEFT)
        
        min_length_var = tk.IntVar(value=qa_settings.get('min_file_length', 0))
        min_length_spinbox = tb.Spinbox(
            min_length_frame,
            from_=0,
            to=10000,
            textvariable=min_length_var,
            width=10,
            bootstyle="primary"
        )
        min_length_spinbox.pack(side=tk.LEFT, padx=(10, 0))
        
        # Disable mousewheel scrolling on spinbox
        self.ui.disable_spinbox_mousewheel(min_length_spinbox)

        # Add a separator
        separator = ttk.Separator(main_frame, orient='horizontal')
        separator.pack(fill=tk.X, pady=15)
        
        # Word Count Cross-Reference Section
        wordcount_section = tk.LabelFrame(
            main_frame,
            text="Word Count Analysis",
            font=('Arial', 12, 'bold'),
            padx=20,
            pady=15
        )
        wordcount_section.pack(fill=tk.X, pady=(0, 20))
        
        check_word_count_var = tk.BooleanVar(value=qa_settings.get('check_word_count_ratio', False))
        tb.Checkbutton(
            wordcount_section,
            text="Cross-reference word counts with original EPUB",
            variable=check_word_count_var,
            bootstyle="primary"
        ).pack(anchor=tk.W, pady=(0, 5))
        
        tk.Label(
            wordcount_section,
            text="Compares word counts between original and translated files to detect missing content.\n" +
                 "Accounts for typical expansion ratios when translating from CJK to English.",
            wraplength=700,
            justify=tk.LEFT,
            fg='gray'
        ).pack(anchor=tk.W, padx=(20, 0))
 
        # Show current EPUB status and allow selection
        epub_frame = tk.Frame(wordcount_section)
        epub_frame.pack(anchor=tk.W, pady=(10, 5))

        # Get EPUBs from actual current selection (not stored config)
        current_epub_files = []
        if hasattr(self, 'selected_files') and self.selected_files:
            current_epub_files = [f for f in self.selected_files if f.lower().endswith('.epub')]
        
        if len(current_epub_files) > 1:
            # Multiple EPUBs in current selection
            primary_epub = os.path.basename(current_epub_files[0])
            status_text = f"📖 {len(current_epub_files)} EPUB files selected (Primary: {primary_epub})"
            status_color = 'green'
        elif len(current_epub_files) == 1:
            # Single EPUB in current selection
            status_text = f"📖 Current EPUB: {os.path.basename(current_epub_files[0])}"
            status_color = 'green'
        else:
            # No EPUB files in current selection
            status_text = "📖 No EPUB in current selection"
            status_color = 'orange'

        status_label = tk.Label(
            epub_frame,
            text=status_text,
            fg=status_color,
            font=('Arial', 10)
        )
        status_label.pack(side=tk.LEFT)

        def select_epub_for_qa():
            epub_path = filedialog.askopenfilename(
                title="Select Source EPUB File",
                filetypes=[("EPUB files", "*.epub"), ("All files", "*.*")],
                parent=dialog
            )
            if epub_path:
                self.selected_epub_path = epub_path
                self.config['last_epub_path'] = epub_path
                self.save_config(show_message=False)
                
                # Clear multiple EPUB tracking when manually selecting a single EPUB
                if hasattr(self, 'selected_epub_files'):
                    self.selected_epub_files = [epub_path]
                
                status_label.config(
                    text=f"📖 Current EPUB: {os.path.basename(epub_path)}",
                    fg='green'
                )
                self.append_log(f"✅ Selected EPUB for QA: {os.path.basename(epub_path)}")

        tk.Button(
            epub_frame,
            text="Select EPUB",
            command=select_epub_for_qa,
            font=('Arial', 9)
        ).pack(side=tk.LEFT, padx=(10, 0))

        # Add option to disable mismatch warning
        warn_mismatch_var = tk.BooleanVar(value=qa_settings.get('warn_name_mismatch', True))
        tb.Checkbutton(
            wordcount_section,
            text="Warn when EPUB and folder names don't match",
            variable=warn_mismatch_var,
            bootstyle="primary"
        ).pack(anchor=tk.W, pady=(10, 5))

        # Additional Checks Section
        additional_section = tk.LabelFrame(
            main_frame,
            text="Additional Checks",
            font=('Arial', 12, 'bold'),
            padx=20,
            pady=15
        )
        additional_section.pack(fill=tk.X, pady=(20, 0))

        # Multiple headers check
        check_multiple_headers_var = tk.BooleanVar(value=qa_settings.get('check_multiple_headers', True))
        tb.Checkbutton(
            additional_section,
            text="Detect files with 2 or more headers (h1-h6 tags)",
            variable=check_multiple_headers_var,
            bootstyle="primary"
        ).pack(anchor=tk.W, pady=(5, 5))

        tk.Label(
            additional_section,
            text="Identifies files that may have been incorrectly split or merged.\n" +
                 "Useful for detecting chapters that contain multiple sections.",
            wraplength=700,
            justify=tk.LEFT,
            fg='gray'
        ).pack(anchor=tk.W, padx=(20, 0))

        # Missing HTML tag check
        html_tag_frame = tk.Frame(additional_section)
        html_tag_frame.pack(fill=tk.X, pady=(10, 5))

        check_missing_html_tag_var = tk.BooleanVar(value=qa_settings.get('check_missing_html_tag', True))
        check_missing_html_tag_check = tb.Checkbutton(
            html_tag_frame,
            text="Flag HTML files with missing <html> tag",
            variable=check_missing_html_tag_var,
            bootstyle="primary"
        )
        check_missing_html_tag_check.pack(side=tk.LEFT)

        tk.Label(
            html_tag_frame,
            text="(Checks if HTML files have proper structure)",
            font=('Arial', 9),
            foreground='gray'
        ).pack(side=tk.LEFT, padx=(10, 0))

        # Invalid nesting check (separate toggle)
        check_invalid_nesting_var = tk.BooleanVar(value=qa_settings.get('check_invalid_nesting', False))
        tb.Checkbutton(
            additional_section,
            text="Check for invalid tag nesting",
            variable=check_invalid_nesting_var,
            bootstyle="primary"
        ).pack(anchor=tk.W, pady=(5, 5))

        # NEW: Paragraph Structure Check
        paragraph_section_frame = tk.Frame(additional_section)
        paragraph_section_frame.pack(fill=tk.X, pady=(15, 5))
        
        # Separator line
        ttk.Separator(paragraph_section_frame, orient='horizontal').pack(fill=tk.X, pady=(0, 10))
        
        # Checkbox for paragraph structure check
        check_paragraph_structure_var = tk.BooleanVar(value=qa_settings.get('check_paragraph_structure', True))
        paragraph_check = tb.Checkbutton(
            paragraph_section_frame,
            text="Check for insufficient paragraph tags",
            variable=check_paragraph_structure_var,
            bootstyle="primary"
        )
        paragraph_check.pack(anchor=tk.W)
        
        # Threshold setting frame
        threshold_container = tk.Frame(paragraph_section_frame)
        threshold_container.pack(fill=tk.X, pady=(10, 5), padx=(20, 0))
        
        tk.Label(
            threshold_container,
            text="Minimum text in <p> tags:",
            font=('Arial', 10)
        ).pack(side=tk.LEFT)
        
        # Get current threshold value (default 30%)
        current_threshold = int(qa_settings.get('paragraph_threshold', 0.3) * 100)
        paragraph_threshold_var = tk.IntVar(value=current_threshold)
        
        # Spinbox for threshold
        paragraph_threshold_spinbox = tb.Spinbox(
            threshold_container,
            from_=0,
            to=100,
            textvariable=paragraph_threshold_var,
            width=8,
            bootstyle="primary"
        )
        paragraph_threshold_spinbox.pack(side=tk.LEFT, padx=(10, 5))
        
        # Disable mousewheel scrolling on the spinbox
        self.ui.disable_spinbox_mousewheel(paragraph_threshold_spinbox)
        
        tk.Label(
            threshold_container,
            text="%",
            font=('Arial', 10)
        ).pack(side=tk.LEFT)
        
        # Threshold value label
        threshold_value_label = tk.Label(
            threshold_container,
            text=f"(currently {current_threshold}%)",
            font=('Arial', 9),
            fg='gray'
        )
        threshold_value_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Update label when spinbox changes
        def update_threshold_label(*args):
            try:
                value = paragraph_threshold_var.get()
                threshold_value_label.config(text=f"(currently {value}%)")
            except (tk.TclError, ValueError):
                # Handle empty or invalid input
                threshold_value_label.config(text="(currently --%)")
        paragraph_threshold_var.trace('w', update_threshold_label)
        
        # Description
        tk.Label(
            paragraph_section_frame,
            text="Detects HTML files where text content is not properly wrapped in paragraph tags.\n" +
                 "Files with less than the specified percentage of text in <p> tags will be flagged.\n" +
                 "Also checks for large blocks of unwrapped text directly in the body element.",
            wraplength=700,
            justify=tk.LEFT,
            fg='gray'
        ).pack(anchor=tk.W, padx=(20, 0), pady=(5, 0))
        
        # Enable/disable threshold setting based on checkbox
        def toggle_paragraph_threshold(*args):
            if check_paragraph_structure_var.get():
                paragraph_threshold_spinbox.config(state='normal')
            else:
                paragraph_threshold_spinbox.config(state='disabled')
        
        check_paragraph_structure_var.trace('w', toggle_paragraph_threshold)
        toggle_paragraph_threshold()  # Set initial state

        # Report Settings Section
        report_section = tk.LabelFrame(
            main_frame,
            text="Report Settings",
            font=('Arial', 12, 'bold'),
            padx=20,
            pady=15
        )
        report_section.pack(fill=tk.X, pady=(0, 20))

        # Cache Settings Section
        cache_section = tk.LabelFrame(
            main_frame,
            text="Performance Cache Settings",
            font=('Arial', 12, 'bold'),
            padx=20,
            pady=15
        )
        cache_section.pack(fill=tk.X, pady=(0, 20))
        
        # Enable cache checkbox
        cache_enabled_var = tk.BooleanVar(value=qa_settings.get('cache_enabled', True))
        cache_checkbox = tb.Checkbutton(
            cache_section,
            text="Enable performance cache (speeds up duplicate detection)",
            variable=cache_enabled_var,
            bootstyle="primary"
        )
        cache_checkbox.pack(anchor=tk.W, pady=(0, 10))
        
        # Cache size settings frame
        cache_sizes_frame = tk.Frame(cache_section)
        cache_sizes_frame.pack(fill=tk.X, padx=(20, 0))
        
        # Description
        tk.Label(
            cache_sizes_frame,
            text="Cache sizes (0 = disabled, -1 = unlimited):",
            font=('Arial', 10)
        ).pack(anchor=tk.W, pady=(0, 5))
        
        # Cache size variables
        cache_vars = {}
        cache_defaults = {
            'normalize_text': 10000,
            'similarity_ratio': 20000,
            'content_hashes': 5000,
            'semantic_fingerprint': 2000,
            'structural_signature': 2000,
            'translation_artifacts': 1000
        }
        
        # Create input fields for each cache type
        for cache_name, default_value in cache_defaults.items():
            row_frame = tk.Frame(cache_sizes_frame)
            row_frame.pack(fill=tk.X, pady=2)
            
            # Label
            label_text = cache_name.replace('_', ' ').title() + ":"
            tk.Label(
                row_frame,
                text=label_text,
                width=25,
                anchor='w',
                font=('Arial', 9)
            ).pack(side=tk.LEFT)
            
            # Get current value
            current_value = qa_settings.get(f'cache_{cache_name}', default_value)
            cache_var = tk.IntVar(value=current_value)
            cache_vars[cache_name] = cache_var
            
            # Spinbox
            spinbox = tb.Spinbox(
                row_frame,
                from_=-1,
                to=50000,
                textvariable=cache_var,
                width=10,
                bootstyle="primary"
            )
            spinbox.pack(side=tk.LEFT, padx=(0, 10))
            
            # Disable mousewheel scrolling
            self.ui.disable_spinbox_mousewheel(spinbox)
            
            # Quick preset buttons
            button_frame = tk.Frame(row_frame)
            button_frame.pack(side=tk.LEFT)
            
            tk.Button(
                button_frame,
                text="Off",
                width=4,
                font=('Arial', 8),
                command=lambda v=cache_var: v.set(0)
            ).pack(side=tk.LEFT, padx=1)
            
            tk.Button(
                button_frame,
                text="Small",
                width=5,
                font=('Arial', 8),
                command=lambda v=cache_var: v.set(1000)
            ).pack(side=tk.LEFT, padx=1)
            
            tk.Button(
                button_frame,
                text="Medium",
                width=7,
                font=('Arial', 8),
                command=lambda v=cache_var, d=default_value: v.set(d)
            ).pack(side=tk.LEFT, padx=1)
            
            tk.Button(
                button_frame,
                text="Large",
                width=5,
                font=('Arial', 8),
                command=lambda v=cache_var, d=default_value: v.set(d * 2)
            ).pack(side=tk.LEFT, padx=1)
            
            tk.Button(
                button_frame,
                text="Max",
                width=4,
                font=('Arial', 8),
                command=lambda v=cache_var: v.set(-1)
            ).pack(side=tk.LEFT, padx=1)
        
        # Enable/disable cache size controls based on checkbox
        def toggle_cache_controls(*args):
            state = 'normal' if cache_enabled_var.get() else 'disabled'
            for widget in cache_sizes_frame.winfo_children():
                if isinstance(widget, tk.Frame):
                    for child in widget.winfo_children():
                        if isinstance(child, (tb.Spinbox, tk.Button)):
                            child.config(state=state)
        
        cache_enabled_var.trace('w', toggle_cache_controls)
        toggle_cache_controls()  # Set initial state
        
        # Auto-size cache option
        auto_size_frame = tk.Frame(cache_section)
        auto_size_frame.pack(fill=tk.X, pady=(10, 5))
        
        auto_size_var = tk.BooleanVar(value=qa_settings.get('cache_auto_size', False))
        auto_size_check = tb.Checkbutton(
            auto_size_frame,
            text="Auto-size caches based on available RAM",
            variable=auto_size_var,
            bootstyle="primary"
        )
        auto_size_check.pack(side=tk.LEFT)
        
        tk.Label(
            auto_size_frame,
            text="(overrides manual settings)",
            font=('Arial', 9),
            fg='gray'
        ).pack(side=tk.LEFT, padx=(10, 0))
        
        # Cache statistics display
        stats_frame = tk.Frame(cache_section)
        stats_frame.pack(fill=tk.X, pady=(10, 0))
        
        show_stats_var = tk.BooleanVar(value=qa_settings.get('cache_show_stats', False))
        tb.Checkbutton(
            stats_frame,
            text="Show cache hit/miss statistics after scan",
            variable=show_stats_var,
            bootstyle="primary"
        ).pack(anchor=tk.W)
        
        # Info about cache
        tk.Label(
            cache_section,
            text="Larger cache sizes use more memory but improve performance for:\n" +
                 "• Large datasets (100+ files)\n" +
                 "• AI Hunter mode (all file pairs compared)\n" +
                 "• Repeated scans of the same folder",
            wraplength=700,
            justify=tk.LEFT,
            fg='gray',
            font=('Arial', 9)
        ).pack(anchor=tk.W, padx=(20, 0), pady=(10, 0))

        # AI Hunter Performance Section
        ai_hunter_section = tk.LabelFrame(
            main_frame,
            text="AI Hunter Performance Settings",
            font=('Arial', 12, 'bold'),
            padx=20,
            pady=15
        )
        ai_hunter_section.pack(fill=tk.X, pady=(0, 20))

        # Description
        tk.Label(
            ai_hunter_section,
            text="AI Hunter mode performs exhaustive duplicate detection by comparing every file pair.\n" +
                 "Parallel processing can significantly speed up this process on multi-core systems.",
            wraplength=700,
            justify=tk.LEFT,
            fg='gray',
            font=('Arial', 9)
        ).pack(anchor=tk.W, pady=(0, 10))

        # Parallel workers setting
        workers_frame = tk.Frame(ai_hunter_section)
        workers_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            workers_frame,
            text="Maximum parallel workers:",
            font=('Arial', 10)
        ).pack(side=tk.LEFT)

        # Get current value from AI Hunter config
        ai_hunter_config = self.config.get('ai_hunter_config', {})
        current_max_workers = ai_hunter_config.get('ai_hunter_max_workers', 1)

        ai_hunter_workers_var = tk.IntVar(value=current_max_workers)
        workers_spinbox = tb.Spinbox(
            workers_frame,
            from_=0,
            to=64,
            textvariable=ai_hunter_workers_var,
            width=10,
            bootstyle="primary"
        )
        workers_spinbox.pack(side=tk.LEFT, padx=(10, 0))

        # Disable mousewheel scrolling on spinbox
        self.ui.disable_spinbox_mousewheel(workers_spinbox)

        # CPU count display
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        cpu_label = tk.Label(
            workers_frame,
            text=f"(0 = use all {cpu_count} cores)",
            font=('Arial', 9),
            fg='gray'
        )
        cpu_label.pack(side=tk.LEFT, padx=(10, 0))

        # Quick preset buttons
        preset_frame = tk.Frame(ai_hunter_section)
        preset_frame.pack(fill=tk.X)

        tk.Label(
            preset_frame,
            text="Quick presets:",
            font=('Arial', 9)
        ).pack(side=tk.LEFT, padx=(0, 10))

        tk.Button(
            preset_frame,
            text=f"All cores ({cpu_count})",
            font=('Arial', 9),
            command=lambda: ai_hunter_workers_var.set(0)
        ).pack(side=tk.LEFT, padx=2)

        tk.Button(
            preset_frame,
            text="Half cores",
            font=('Arial', 9),
            command=lambda: ai_hunter_workers_var.set(max(1, cpu_count // 2))
        ).pack(side=tk.LEFT, padx=2)

        tk.Button(
            preset_frame,
            text="4 cores",
            font=('Arial', 9),
            command=lambda: ai_hunter_workers_var.set(4)
        ).pack(side=tk.LEFT, padx=2)

        tk.Button(
            preset_frame,
            text="8 cores",
            font=('Arial', 9),
            command=lambda: ai_hunter_workers_var.set(8)
        ).pack(side=tk.LEFT, padx=2)

        tk.Button(
            preset_frame,
            text="Single thread",
            font=('Arial', 9),
            command=lambda: ai_hunter_workers_var.set(1)
        ).pack(side=tk.LEFT, padx=2)

        # Performance tips
        tips_text = "Performance Tips:\n" + \
                    f"• Your system has {cpu_count} CPU cores available\n" + \
                    "• Using all cores provides maximum speed but may slow other applications\n" + \
                    "• 4-8 cores usually provides good balance of speed and system responsiveness\n" + \
                    "• Single thread (1) disables parallel processing for debugging"

        tk.Label(
            ai_hunter_section,
            text=tips_text,
            wraplength=700,
            justify=tk.LEFT,
            fg='gray',
            font=('Arial', 9)
        ).pack(anchor=tk.W, padx=(20, 0), pady=(10, 0))

        # Report format
        format_frame = tk.Frame(report_section)
        format_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            format_frame,
            text="Report format:",
            font=('Arial', 10)
        ).pack(side=tk.LEFT)

        format_var = tk.StringVar(value=qa_settings.get('report_format', 'detailed'))
        format_options = [
            ("Summary only", "summary"),
            ("Detailed (recommended)", "detailed"),
            ("Verbose (all data)", "verbose")
        ]

        for idx, (text, value) in enumerate(format_options):
            rb = tb.Radiobutton(
                format_frame,
                text=text,
                variable=format_var,
                value=value,
                bootstyle="primary"
            )
            rb.pack(side=tk.LEFT, padx=(10 if idx == 0 else 5, 0))

        # Auto-save report
        auto_save_var = tk.BooleanVar(value=qa_settings.get('auto_save_report', True))
        tb.Checkbutton(
            report_section,
            text="Automatically save report after scan",
            variable=auto_save_var,
            bootstyle="primary"
        ).pack(anchor=tk.W)

        # Buttons
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(20, 0))
        button_inner = tk.Frame(button_frame)
        button_inner.pack()
        
        def save_settings():
            """Save QA scanner settings with comprehensive debugging"""
            try:
                self.append_log("🔍 [DEBUG] Starting QA Scanner settings save process...")
                
                # Core QA Settings with debugging
                core_settings_to_save = {
                    'foreign_char_threshold': (threshold_var, lambda x: x.get()),
                    'excluded_characters': (excluded_text, lambda x: x.get(1.0, tk.END).strip()),
                    'target_language': (target_language_var, lambda x: x.get().lower()),
                    'check_encoding_issues': (check_encoding_var, lambda x: x.get()),
                    'check_repetition': (check_repetition_var, lambda x: x.get()),
                    'check_translation_artifacts': (check_artifacts_var, lambda x: x.get()),
                    'check_glossary_leakage': (check_glossary_var, lambda x: x.get()),
                    'min_file_length': (min_length_var, lambda x: x.get()),
                    'report_format': (format_var, lambda x: x.get()),
                    'auto_save_report': (auto_save_var, lambda x: x.get()),
                    'check_word_count_ratio': (check_word_count_var, lambda x: x.get()),
                    'check_multiple_headers': (check_multiple_headers_var, lambda x: x.get()),
                    'warn_name_mismatch': (warn_mismatch_var, lambda x: x.get()),
                    'check_missing_html_tag': (check_missing_html_tag_var, lambda x: x.get()),
                    'check_paragraph_structure': (check_paragraph_structure_var, lambda x: x.get()),
                    'check_invalid_nesting': (check_invalid_nesting_var, lambda x: x.get()),
                }
                
                failed_core_settings = []
                for setting_name, (var_obj, converter) in core_settings_to_save.items():
                    try:
                        old_value = qa_settings.get(setting_name, '<NOT SET>')
                        new_value = converter(var_obj)
                        qa_settings[setting_name] = new_value
                        
                        if old_value != new_value:
                            self.append_log(f"🔍 [DEBUG] QA {setting_name}: '{old_value}' → '{new_value}'")
                        else:
                            self.append_log(f"🔍 [DEBUG] QA {setting_name}: unchanged ('{new_value}')")
                            
                    except Exception as e:
                        failed_core_settings.append(f"{setting_name} ({str(e)})")
                        self.append_log(f"❌ [DEBUG] Failed to save QA {setting_name}: {e}")
                
                if failed_core_settings:
                    self.append_log(f"⚠️ [DEBUG] Failed QA core settings: {', '.join(failed_core_settings)}")
                
                # Cache settings with debugging
                self.append_log("🔍 [DEBUG] Saving QA cache settings...")
                cache_settings_to_save = {
                    'cache_enabled': (cache_enabled_var, lambda x: x.get()),
                    'cache_auto_size': (auto_size_var, lambda x: x.get()),
                    'cache_show_stats': (show_stats_var, lambda x: x.get()),
                }
                
                failed_cache_settings = []
                for setting_name, (var_obj, converter) in cache_settings_to_save.items():
                    try:
                        old_value = qa_settings.get(setting_name, '<NOT SET>')
                        new_value = converter(var_obj)
                        qa_settings[setting_name] = new_value
                        
                        if old_value != new_value:
                            self.append_log(f"🔍 [DEBUG] QA {setting_name}: '{old_value}' → '{new_value}'")
                        else:
                            self.append_log(f"🔍 [DEBUG] QA {setting_name}: unchanged ('{new_value}')")
                    except Exception as e:
                        failed_cache_settings.append(f"{setting_name} ({str(e)})")
                        self.append_log(f"❌ [DEBUG] Failed to save QA {setting_name}: {e}")
                
                # Save individual cache sizes with debugging
                saved_cache_vars = []
                failed_cache_vars = []
                for cache_name, cache_var in cache_vars.items():
                    try:
                        cache_key = f'cache_{cache_name}'
                        old_value = qa_settings.get(cache_key, '<NOT SET>')
                        new_value = cache_var.get()
                        qa_settings[cache_key] = new_value
                        saved_cache_vars.append(cache_name)
                        
                        if old_value != new_value:
                            self.append_log(f"🔍 [DEBUG] QA {cache_key}: '{old_value}' → '{new_value}'")
                    except Exception as e:
                        failed_cache_vars.append(f"{cache_name} ({str(e)})")
                        self.append_log(f"❌ [DEBUG] Failed to save QA cache_{cache_name}: {e}")
                
                if saved_cache_vars:
                    self.append_log(f"🔍 [DEBUG] Saved {len(saved_cache_vars)} cache settings: {', '.join(saved_cache_vars)}")
                if failed_cache_vars:
                    self.append_log(f"⚠️ [DEBUG] Failed cache settings: {', '.join(failed_cache_vars)}")
                
                # AI Hunter config with debugging
                self.append_log("🔍 [DEBUG] Saving AI Hunter config...")
                try:
                    if 'ai_hunter_config' not in self.config:
                        self.config['ai_hunter_config'] = {}
                        self.append_log("🔍 [DEBUG] Created new ai_hunter_config section")
                    
                    old_workers = self.config['ai_hunter_config'].get('ai_hunter_max_workers', '<NOT SET>')
                    new_workers = ai_hunter_workers_var.get()
                    self.config['ai_hunter_config']['ai_hunter_max_workers'] = new_workers
                    
                    if old_workers != new_workers:
                        self.append_log(f"🔍 [DEBUG] AI Hunter max_workers: '{old_workers}' → '{new_workers}'")
                    else:
                        self.append_log(f"🔍 [DEBUG] AI Hunter max_workers: unchanged ('{new_workers}')")
                        
                except Exception as e:
                    self.append_log(f"❌ [DEBUG] Failed to save AI Hunter config: {e}")
    
                # Validate and save paragraph threshold with debugging
                self.append_log("🔍 [DEBUG] Validating paragraph threshold...")
                try:
                    threshold_value = paragraph_threshold_var.get()
                    old_threshold = qa_settings.get('paragraph_threshold', '<NOT SET>')
                    
                    if 0 <= threshold_value <= 100:
                        new_threshold = threshold_value / 100.0  # Convert to decimal
                        qa_settings['paragraph_threshold'] = new_threshold
                        
                        if old_threshold != new_threshold:
                            self.append_log(f"🔍 [DEBUG] QA paragraph_threshold: '{old_threshold}' → '{new_threshold}' ({threshold_value}%)")
                        else:
                            self.append_log(f"🔍 [DEBUG] QA paragraph_threshold: unchanged ('{new_threshold}' / {threshold_value}%)")
                    else:
                        raise ValueError("Threshold must be between 0 and 100")
                        
                except (tk.TclError, ValueError) as e:
                    # Default to 30% if invalid
                    qa_settings['paragraph_threshold'] = 0.3
                    self.append_log(f"❌ [DEBUG] Invalid paragraph threshold ({e}), using default 30%")
                    self.append_log("⚠️ Invalid paragraph threshold, using default 30%")

                # Save to main config with debugging
                self.append_log("🔍 [DEBUG] Saving QA settings to main config...")
                try:
                    old_qa_config = self.config.get('qa_scanner_settings', {})
                    self.config['qa_scanner_settings'] = qa_settings
                    
                    # Count changed settings
                    changed_settings = []
                    for key, new_value in qa_settings.items():
                        if old_qa_config.get(key) != new_value:
                            changed_settings.append(key)
                    
                    if changed_settings:
                        self.append_log(f"🔍 [DEBUG] Changed {len(changed_settings)} QA settings: {', '.join(changed_settings[:5])}{'...' if len(changed_settings) > 5 else ''}")
                    else:
                        self.append_log("🔍 [DEBUG] No QA settings changed")
                        
                except Exception as e:
                    self.append_log(f"❌ [DEBUG] Failed to update main config: {e}")
                
                # Environment variables setup for QA Scanner
                self.append_log("🔍 [DEBUG] Setting QA Scanner environment variables...")
                qa_env_vars_set = []
                
                try:
                    # QA Scanner environment variables
                    qa_env_mappings = [
                        ('QA_FOREIGN_CHAR_THRESHOLD', str(qa_settings.get('foreign_char_threshold', 10))),
                        ('QA_TARGET_LANGUAGE', qa_settings.get('target_language', 'english')),
                        ('QA_CHECK_ENCODING', '1' if qa_settings.get('check_encoding_issues', False) else '0'),
                        ('QA_CHECK_REPETITION', '1' if qa_settings.get('check_repetition', True) else '0'),
                        ('QA_CHECK_ARTIFACTS', '1' if qa_settings.get('check_translation_artifacts', False) else '0'),
                        ('QA_CHECK_GLOSSARY_LEAKAGE', '1' if qa_settings.get('check_glossary_leakage', True) else '0'),
                        ('QA_MIN_FILE_LENGTH', str(qa_settings.get('min_file_length', 0))),
                        ('QA_REPORT_FORMAT', qa_settings.get('report_format', 'detailed')),
                        ('QA_AUTO_SAVE_REPORT', '1' if qa_settings.get('auto_save_report', True) else '0'),
                        ('QA_CACHE_ENABLED', '1' if qa_settings.get('cache_enabled', True) else '0'),
                        ('QA_PARAGRAPH_THRESHOLD', str(qa_settings.get('paragraph_threshold', 0.3))),
                        ('AI_HUNTER_MAX_WORKERS', str(self.config.get('ai_hunter_config', {}).get('ai_hunter_max_workers', 1))),
                    ]
                    
                    for env_key, env_value in qa_env_mappings:
                        try:
                            old_value = os.environ.get(env_key, '<NOT SET>')
                            os.environ[env_key] = str(env_value)
                            new_value = os.environ[env_key]
                            qa_env_vars_set.append(env_key)
                            
                            if old_value != new_value:
                                self.append_log(f"🔍 [DEBUG] ENV {env_key}: '{old_value}' → '{new_value}'")
                            else:
                                self.append_log(f"🔍 [DEBUG] ENV {env_key}: unchanged ('{new_value}')")
                                
                        except Exception as e:
                            self.append_log(f"❌ [DEBUG] Failed to set {env_key}: {e}")
                    
                    self.append_log(f"🔍 [DEBUG] Successfully set {len(qa_env_vars_set)} QA environment variables")
                    
                except Exception as e:
                    self.append_log(f"❌ [DEBUG] QA environment variable setup failed: {e}")
                    import traceback
                    self.append_log(f"❌ [DEBUG] Traceback: {traceback.format_exc()}")
                
                # Call save_config with show_message=False to avoid the error
                self.append_log("🔍 [DEBUG] Calling main save_config method...")
                try:
                    self.save_config(show_message=False)
                    self.append_log("🔍 [DEBUG] Main save_config completed successfully")
                except Exception as e:
                    self.append_log(f"❌ [DEBUG] Main save_config failed: {e}")
                    raise
                
                # Final QA environment variable verification
                self.append_log("🔍 [DEBUG] Final QA environment variable check:")
                critical_qa_vars = ['QA_FOREIGN_CHAR_THRESHOLD', 'QA_TARGET_LANGUAGE', 'QA_REPORT_FORMAT', 'AI_HUNTER_MAX_WORKERS']
                for var in critical_qa_vars:
                    value = os.environ.get(var, '<NOT SET>')
                    if value == '<NOT SET>' or not value:
                        self.append_log(f"❌ [DEBUG] CRITICAL QA: {var} is not set or empty!")
                    else:
                        self.append_log(f"✅ [DEBUG] QA {var}: {value}")
                
                self.append_log("✅ QA Scanner settings saved successfully")
                dialog._cleanup_scrolling()  # Clean up scrolling bindings
                dialog.destroy()
                
            except Exception as e:
                self.append_log(f"❌ [DEBUG] QA save_settings full exception: {str(e)}")
                import traceback
                self.append_log(f"❌ [DEBUG] QA save_settings traceback: {traceback.format_exc()}")
                self.append_log(f"❌ Error saving QA settings: {str(e)}")
                messagebox.showerror("Error", f"Failed to save settings: {str(e)}")
        
        def reset_defaults():
            """Reset to default settings"""
            result = messagebox.askyesno(
                "Reset to Defaults", 
                "Are you sure you want to reset all settings to defaults?",
                parent=dialog
            )
            if result:
                threshold_var.set(10)
                excluded_text.delete(1.0, tk.END)
                target_language_var.set('English')
                check_encoding_var.set(False)
                check_repetition_var.set(True)
                check_artifacts_var.set(False)

                check_glossary_var.set(True)
                min_length_var.set(0)
                format_var.set('detailed')
                auto_save_var.set(True)
                check_word_count_var.set(False)
                check_multiple_headers_var.set(True)
                warn_mismatch_var.set(False)
                check_missing_html_tag_var.set(True)
                check_paragraph_structure_var.set(True)
                check_invalid_nesting_var.set(False)
                paragraph_threshold_var.set(30)  # 30% default
                paragraph_threshold_var.set(30)  # 30% default
                
                # Reset cache settings
                cache_enabled_var.set(True)
                auto_size_var.set(False)
                show_stats_var.set(False)
                
                # Reset cache sizes to defaults
                for cache_name, default_value in cache_defaults.items():
                    cache_vars[cache_name].set(default_value)
                    
                ai_hunter_workers_var.set(1)
        
        # Create buttons using ttkbootstrap styles
        save_btn = tb.Button(
            button_inner,
            text="Save Settings",
            command=save_settings,
            bootstyle="success",
            width=15
        )
        save_btn.pack(side=tk.LEFT, padx=5)
        
        reset_btn = tb.Button(
            button_inner,
            text="Reset Defaults",
            command=reset_defaults,
            bootstyle="warning",
            width=15
        )
        reset_btn.pack(side=tk.RIGHT, padx=(5, 0))
        
        cancel_btn = tb.Button(
            button_inner,
            text="Cancel",
            command=lambda: [dialog._cleanup_scrolling(), dialog.destroy()],
            bootstyle="secondary",
            width=15
        )
        cancel_btn.pack(side=tk.RIGHT)
        
        # Use WindowManager's auto_resize_dialog to properly size the window
        self.wm.auto_resize_dialog(dialog, canvas, max_width_ratio=0.9, max_height_ratio=0.85)
        
        # Handle window close - setup_scrollable adds _cleanup_scrolling method
        dialog.protocol("WM_DELETE_WINDOW", lambda: [dialog._cleanup_scrolling(), dialog.destroy()])