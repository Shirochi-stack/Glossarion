# ai_hunter_enhanced.py
# Combined AI Hunter configuration GUI and detection logic

import tkinter as tk
from tkinter import ttk
import ttkbootstrap as tb
import json
import os
import re
import unicodedata
from difflib import SequenceMatcher
from collections import Counter

class AIHunterConfigGUI:
    """GUI for configuring AI Hunter detection parameters"""
    def __init__(self, parent, config_dict, callback=None):
        """
        Initialize with reference to main config dictionary
        
        Args:
            parent: Parent window
            config_dict: Reference to main translator config dictionary
            callback: Function to call after saving
        """
        self.parent = parent
        self.config = config_dict  # Reference to main config
        self.callback = callback
        self.window = None
        
        # Default AI Hunter settings structure
        self.default_ai_hunter = {
            'enabled': True,
            'sample_size': 3000,
            'thresholds': {
                'exact': 90,
                'text': 85,
                'semantic': 85,
                'structural': 85,
                'character': 80,
                'pattern': 80
            },
            'weights': {
                'exact': 1.5,
                'text': 1.2,
                'semantic': 1.0,
                'structural': 1.0,
                'character': 0.8,
                'pattern': 0.8
            },
            'detection_mode': 'multi_method',
            'multi_method_requirements': {
                'methods_required': 3,
                'min_methods': ['semantic', 'structural']
            },
            'preprocessing': {
                'remove_html_spacing': True,
                'normalize_unicode': True,
                'ignore_case': True,
                'remove_extra_whitespace': True
            },
            'edge_filters': {
                'min_text_length': 500,
                'max_length_ratio': 1.3,
                'min_length_ratio': 0.7
            }
        }
        
        # Initialize AI Hunter config in main config if not present
        if 'ai_hunter_config' not in self.config:
            self.config['ai_hunter_config'] = self.default_ai_hunter.copy()
        else:
            # Merge with defaults to ensure all keys exist
            self.config['ai_hunter_config'] = self._merge_configs(
                self.default_ai_hunter, 
                self.config['ai_hunter_config']
            )
    
    def _merge_configs(self, default, existing):
        """Recursively merge existing config with defaults"""
        result = default.copy()
        for key, value in existing.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result
    
    def get_ai_config(self):
        """Get AI Hunter configuration from main config"""
        return self.config.get('ai_hunter_config', self.default_ai_hunter)
    
    def show_ai_hunter_config(self):
        """Display the AI Hunter configuration window with scrollbar using WindowManager"""
        if self.window and self.window.winfo_exists():
            self.window.lift()
            return
        
        # Import WindowManager if not already available
        if not hasattr(self, 'wm'):
            from translator_gui import WindowManager
            import sys
            import os
            base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
            self.wm = WindowManager(base_dir)
        
        # Create scrollable dialog using WindowManager
        dialog, scrollable_frame, canvas = self.wm.setup_scrollable(
            self.parent,
            "AI Hunter Configuration",
            width=820,
            height=None,  # Will use default height
            max_width_ratio=0.9,
            max_height_ratio=0.85
        )
        
        self.window = dialog
        
        # Create notebook inside scrollable frame
        notebook = ttk.Notebook(scrollable_frame)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Tab 1: Detection Thresholds
        self.create_thresholds_tab(notebook)
        
        # Tab 2: Detection Mode
        self.create_mode_tab(notebook)
        
        # Tab 3: Preprocessing
        self.create_preprocessing_tab(notebook)
        
        # Tab 4: Advanced Settings
        self.create_advanced_tab(notebook)
        
        # Buttons at the bottom (inside scrollable frame)
        button_frame = tk.Frame(scrollable_frame)
        button_frame.pack(fill='x', padx=10, pady=(10, 20))
        
        tb.Button(button_frame, text="Save", command=self.apply_ai_hunter_settings, 
                 bootstyle="success").pack(side='right', padx=5)
        tb.Button(button_frame, text="Cancel", command=self.window.destroy,
                 bootstyle="secondary").pack(side='right')
        tb.Button(button_frame, text="Reset to Defaults", command=self.reset_defaults,
                 bootstyle="warning").pack(side='left')
        
        # Auto-resize and show
        self.wm.auto_resize_dialog(dialog, canvas, max_width_ratio=0.9, max_height_ratio=1.1)
        
        # Handle window close
        dialog.protocol("WM_DELETE_WINDOW", lambda: [dialog._cleanup_scrolling(), dialog.destroy()])
    
    def create_thresholds_tab(self, notebook):
        """Create the thresholds configuration tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Detection Thresholds")
        
        # Title
        tk.Label(frame, text="Detection Method Thresholds", 
                font=('TkDefaultFont', 12, 'bold')).pack(pady=10)
        
        tk.Label(frame, text="Higher values = fewer false positives (more strict)\n"
                           "Lower values = more false positives (more sensitive)",
                font=('TkDefaultFont', 10), fg='gray').pack(pady=(0, 20))
        
        # Threshold controls
        self.threshold_vars = {}
        threshold_frame = tk.Frame(frame)
        threshold_frame.pack(fill='both', expand=True, padx=20)
        
        descriptions = {
            'exact': 'Exact Text Match - Direct character-by-character comparison',
            'text': 'Smart Text Similarity - Intelligent text comparison with sampling',
            'semantic': 'Semantic Analysis - Character names, dialogue patterns, numbers',
            'structural': 'Structural Patterns - Paragraph structure, dialogue distribution',
            'character': 'Character Overlap - Common character names between chapters',
            'pattern': 'Pattern Analysis - Narrative flow and structure patterns'
        }
        
        ai_config = self.get_ai_config()
        
        for method, desc in descriptions.items():
            method_frame = tk.Frame(threshold_frame)
            method_frame.pack(fill='x', pady=10)
            
            # Method name and description
            label_frame = tk.Frame(method_frame)
            label_frame.pack(fill='x')
            
            tk.Label(label_frame, text=f"{method.title()}:", 
                    font=('TkDefaultFont', 10, 'bold')).pack(side='left')
            tk.Label(label_frame, text=f" {desc}",
                    font=('TkDefaultFont', 9), fg='gray').pack(side='left', padx=(10, 0))
            
            # Slider and value
            slider_frame = tk.Frame(method_frame)
            slider_frame.pack(fill='x', pady=(5, 0))
            
            self.threshold_vars[method] = tk.IntVar(value=ai_config['thresholds'][method])
            
            slider = tb.Scale(slider_frame, from_=10, to=100, 
                            variable=self.threshold_vars[method],
                            bootstyle="info", length=400)
            slider.pack(side='left', padx=(20, 10))
            
            value_label = tk.Label(slider_frame, text="", width=4)
            value_label.pack(side='left')
            
            # Update label when slider changes
            def update_label(val, label=value_label, var=self.threshold_vars[method]):
                label.config(text=f"{int(var.get())}%")
            
            self.threshold_vars[method].trace('w', lambda *args, f=update_label: f(None))
            update_label(None)
        
        # Weight configuration
        tk.Label(frame, text="Method Weights (for weighted average mode)", 
                font=('TkDefaultFont', 11, 'bold')).pack(pady=(30, 10))
        
        self.weight_vars = {}
        weight_frame = tk.Frame(frame)
        weight_frame.pack(fill='x', padx=20)
        
        for method in descriptions.keys():
            w_frame = tk.Frame(weight_frame)
            w_frame.pack(fill='x', pady=5)
            
            tk.Label(w_frame, text=f"{method.title()} weight:", width=20, 
                    anchor='w').pack(side='left')
            
            self.weight_vars[method] = tk.DoubleVar(value=ai_config['weights'][method])
            
            tb.Spinbox(w_frame, from_=0.1, to=2.0, increment=0.1,
                      textvariable=self.weight_vars[method],
                      width=10).pack(side='left', padx=10)
    
    def create_mode_tab(self, notebook):
        """Create the detection mode configuration tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Detection Mode")
        
        tk.Label(frame, text="Detection Mode Configuration", 
                font=('TkDefaultFont', 12, 'bold')).pack(pady=10)
        
        # Detection mode selection
        mode_frame = tk.LabelFrame(frame, text="Detection Mode", padx=20, pady=20)
        mode_frame.pack(fill='x', padx=20, pady=10)
        
        ai_config = self.get_ai_config()
        self.mode_var = tk.StringVar(value=ai_config['detection_mode'])
        
        modes = [
            ('single_method', 'Single Method', 
             'Flag as duplicate if ANY method exceeds its threshold\n(Most sensitive, most false positives)'),
            ('multi_method', 'Multi-Method Agreement', 
             'Require multiple methods to agree before flagging\n(Balanced approach)'),
            ('weighted_average', 'Weighted Average', 
             'Calculate weighted average of all methods\n(Most nuanced, least false positives)')
        ]
        
        for value, text, desc in modes:
            rb_frame = tk.Frame(mode_frame)
            rb_frame.pack(fill='x', pady=10)
            
            tb.Radiobutton(rb_frame, text=text, variable=self.mode_var, 
                          value=value, bootstyle="primary").pack(anchor='w')
            tk.Label(rb_frame, text=desc, font=('TkDefaultFont', 9), 
                    fg='gray').pack(anchor='w', padx=(25, 0))
        
        # Multi-method configuration
        multi_frame = tk.LabelFrame(frame, text="Multi-Method Settings", padx=20, pady=20)
        multi_frame.pack(fill='x', padx=20, pady=10)
        
        tk.Label(multi_frame, text="Number of methods required to agree:",
                font=('TkDefaultFont', 10)).pack(anchor='w')
        
        self.methods_required_var = tk.IntVar(
            value=ai_config['multi_method_requirements']['methods_required'])
        
        tb.Spinbox(multi_frame, from_=1, to=6, textvariable=self.methods_required_var,
                  width=10).pack(anchor='w', pady=5)
        
        tk.Label(multi_frame, text="Required methods (at least one must be included):",
                font=('TkDefaultFont', 10)).pack(anchor='w', pady=(10, 5))
        
        self.required_method_vars = {}
        for method in ['exact', 'text', 'semantic', 'structural', 'character', 'pattern']:
            var = tk.BooleanVar(
                value=method in ai_config['multi_method_requirements']['min_methods'])
            self.required_method_vars[method] = var
            
            tb.Checkbutton(multi_frame, text=method.title(), variable=var,
                          bootstyle="round-toggle").pack(anchor='w', padx=20)
    
    def create_preprocessing_tab(self, notebook):
        """Create the preprocessing configuration tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Preprocessing")
        
        tk.Label(frame, text="Text Preprocessing Options", 
                font=('TkDefaultFont', 12, 'bold')).pack(pady=10)
        
        tk.Label(frame, text="Configure how text is processed before comparison",
                font=('TkDefaultFont', 10), fg='gray').pack(pady=(0, 20))
        
        # Preprocessing options
        prep_frame = tk.Frame(frame)
        prep_frame.pack(fill='both', expand=True, padx=20)
        
        self.prep_vars = {}
        ai_config = self.get_ai_config()
        
        options = [
            ('remove_html_spacing', 'Remove HTML with spacing', 
             'Replace HTML tags with spaces instead of removing completely'),
            ('normalize_unicode', 'Normalize Unicode', 
             'Normalize unicode characters (recommended)'),
            ('ignore_case', 'Case-insensitive comparison', 
             'Ignore character case when comparing'),
            ('remove_extra_whitespace', 'Remove extra whitespace', 
             'Collapse multiple spaces/newlines into single spaces')
        ]
        
        for key, text, desc in options:
            var = tk.BooleanVar(value=ai_config['preprocessing'][key])
            self.prep_vars[key] = var
            
            opt_frame = tk.Frame(prep_frame)
            opt_frame.pack(fill='x', pady=10)
            
            tb.Checkbutton(opt_frame, text=text, variable=var,
                          bootstyle="round-toggle").pack(anchor='w')
            tk.Label(opt_frame, text=desc, font=('TkDefaultFont', 9),
                    fg='gray').pack(anchor='w', padx=(25, 0))
    
    def create_advanced_tab(self, notebook):
        """Create the advanced settings tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Advanced")
        
        tk.Label(frame, text="Advanced Settings", 
                font=('TkDefaultFont', 12, 'bold')).pack(pady=10)
        
        # General settings
        general_frame = tk.LabelFrame(frame, text="General", padx=20, pady=20)
        general_frame.pack(fill='x', padx=20, pady=10)
        
        ai_config = self.get_ai_config()
        
        
        # Sample size
        ss_frame = tk.Frame(general_frame)
        ss_frame.pack(fill='x', pady=5)
        
        tk.Label(ss_frame, text="Sample size:", width=20, anchor='w').pack(side='left')
        self.sample_size_var = tk.IntVar(value=ai_config['sample_size'])
        tb.Spinbox(ss_frame, from_=1000, to=10000, increment=500,
                  textvariable=self.sample_size_var, width=10).pack(side='left', padx=10)
        tk.Label(ss_frame, text="characters",
                font=('TkDefaultFont', 9)).pack(side='left')
        
        # Edge filters
        edge_frame = tk.LabelFrame(frame, text="Edge Case Filters", padx=20, pady=20)
        edge_frame.pack(fill='x', padx=20, pady=10)
        
        # Min text length
        min_frame = tk.Frame(edge_frame)
        min_frame.pack(fill='x', pady=5)
        
        tk.Label(min_frame, text="Minimum text length:", width=20, anchor='w').pack(side='left')
        self.min_length_var = tk.IntVar(value=ai_config['edge_filters']['min_text_length'])
        tb.Spinbox(min_frame, from_=100, to=2000, increment=100,
                  textvariable=self.min_length_var, width=10).pack(side='left', padx=10)
        tk.Label(min_frame, text="characters",
                font=('TkDefaultFont', 9)).pack(side='left')
        
        # Length ratios
        ratio_frame = tk.Frame(edge_frame)
        ratio_frame.pack(fill='x', pady=10)
        
        tk.Label(ratio_frame, text="Length ratio limits:").pack(anchor='w')
        
        r_frame = tk.Frame(ratio_frame)
        r_frame.pack(fill='x', pady=5)
        
        tk.Label(r_frame, text="Min ratio:", width=10, anchor='w').pack(side='left', padx=(20, 5))
        self.min_ratio_var = tk.DoubleVar(value=ai_config['edge_filters']['min_length_ratio'])
        tb.Spinbox(r_frame, from_=0.5, to=0.9, increment=0.1,
                  textvariable=self.min_ratio_var, width=8).pack(side='left')
        
        tk.Label(r_frame, text="Max ratio:", width=10, anchor='w').pack(side='left', padx=(20, 5))
        self.max_ratio_var = tk.DoubleVar(value=ai_config['edge_filters']['max_length_ratio'])
        tb.Spinbox(r_frame, from_=1.1, to=2.0, increment=0.1,
                  textvariable=self.max_ratio_var, width=8).pack(side='left')
        
        tk.Label(edge_frame, text="Chapters with vastly different lengths won't be compared",
                font=('TkDefaultFont', 9), fg='gray').pack(anchor='w', padx=20)
    
    def apply_ai_hunter_settings(self):
        """Apply AI Hunter settings to the main config"""
        ai_config = self.get_ai_config()
        
        # Update from GUI variables
        for method, var in self.threshold_vars.items():
            ai_config['thresholds'][method] = var.get()
        
        for method, var in self.weight_vars.items():
            ai_config['weights'][method] = var.get()
        
        ai_config['detection_mode'] = self.mode_var.get()
        ai_config['multi_method_requirements']['methods_required'] = self.methods_required_var.get()
        
        min_methods = [method for method, var in self.required_method_vars.items() if var.get()]
        ai_config['multi_method_requirements']['min_methods'] = min_methods
        
        for key, var in self.prep_vars.items():
            ai_config['preprocessing'][key] = var.get()
        
        ai_config['sample_size'] = self.sample_size_var.get()
        
        ai_config['edge_filters']['min_text_length'] = self.min_length_var.get()
        ai_config['edge_filters']['min_length_ratio'] = self.min_ratio_var.get()
        ai_config['edge_filters']['max_length_ratio'] = self.max_ratio_var.get()
        
        # Update main config
        self.config['ai_hunter_config'] = ai_config
        
        # Call callback if provided (this should trigger main save_configuration)
        if self.callback:
            self.callback()
        
        self.window.destroy()
    
    def reset_defaults(self):
        """Reset all values to defaults"""
        import tkinter.messagebox as messagebox
        result = messagebox.askyesno("Reset to Defaults", 
                                   "Are you sure you want to reset all settings to defaults?")
        if result:
            self.config['ai_hunter_config'] = self.default_ai_hunter.copy()
            self.window.destroy()
            self.show_ai_hunter_config()  # Reopen with default values


class ImprovedAIHunterDetection:
    """Improved AI Hunter detection methods for TranslateKRtoEN"""
    
    def __init__(self, main_config):
        """
        Initialize with reference to main config
        
        Args:
            main_config: Reference to main translator config dictionary
        """
        self.main_config = main_config
        
        # Default AI Hunter settings
        self.default_ai_hunter = {
            'enabled': True,
            'lookback_chapters': 5,
            'sample_size': 3000,
            'thresholds': {
                'exact': 90,
                'text': 85,
                'semantic': 85,
                'structural': 85,
                'character': 80,
                'pattern': 80
            },
            'weights': {
                'exact': 1.5,
                'text': 1.2,
                'semantic': 1.0,
                'structural': 1.0,
                'character': 0.8,
                'pattern': 0.8
            },
            'detection_mode': 'multi_method',
            'multi_method_requirements': {
                'methods_required': 2,
                'min_methods': ['semantic', 'structural']
            },
            'preprocessing': {
                'remove_html_spacing': True,
                'normalize_unicode': True,
                'ignore_case': True,
                'remove_extra_whitespace': True
            },
            'edge_filters': {
                'min_text_length': 500,
                'max_length_ratio': 1.3,
                'min_length_ratio': 0.7
            }
        }
    
    def get_ai_config(self):
        """Get AI Hunter configuration from main config"""
        return self.main_config.get('ai_hunter_config', self.default_ai_hunter)

    def detect_duplicate_ai_hunter_enhanced(self, result, idx, prog, out, current_chapter_num=None):
        """Enhanced AI Hunter duplicate detection with configurable parameters"""
        try:
            print(f"\n    ========== AI HUNTER DEBUG START ==========")
            print(f"    📍 Current chapter index: {idx}")
            if current_chapter_num:
                print(f"    📖 Current chapter number: {current_chapter_num}")
            
            # Get configuration
            config = self.get_ai_config()
            
            if not config.get('enabled', True):
                print(f"    ⚠️ AI Hunter is disabled")
                print(f"    ========== AI HUNTER DEBUG END ==========\n")
                return False, 0
            
            # Preprocess text
            result_clean = self._preprocess_text(result, config['preprocessing'])
            print(f"    📄 Text length after preprocessing: {len(result_clean)} chars")
            
            # Check edge cases
            if len(result_clean) < config['edge_filters']['min_text_length']:
                print(f"    ⚠️ Text too short ({len(result_clean)} < {config['edge_filters']['min_text_length']})")
                print(f"    ========== AI HUNTER DEBUG END ==========\n")
                return False, 0
            
            # Extract features
            print(f"    🔬 Extracting text features...")
            result_features = self._extract_text_features(result_clean)
            
            # Get lookback from main config, then fall back to env var if not found
            lookback = self.main_config.get('duplicate_lookback_chapters', 
                                           int(os.getenv('DUPLICATE_LOOKBACK_CHAPTERS', '5')))
            
            # Log configuration
            print(f"\n    🔧 Configuration:")
            print(f"       Detection mode: {config['detection_mode']}")
            print(f"       Lookback chapters: {lookback}")
            print(f"       Sample size: {config['sample_size']}")
            
            # FIX: Get all completed chapters sorted by actual chapter number
            completed_chapters = []
            for chapter_key, chapter_info in prog["chapters"].items():
                if chapter_info.get("status") == "completed" and chapter_info.get("output_file"):
                    # Handle both numeric and hash-based chapter keys
                    try:
                        # Try to get actual_num first (this is what's stored in progress)
                        chapter_num = chapter_info.get("actual_num")
                        if chapter_num is None:
                            # Try chapter_num as fallback
                            chapter_num = chapter_info.get("chapter_num")
                        if chapter_num is None:
                            # Try to parse from key if it's numeric
                            try:
                                chapter_num = int(chapter_key) + 1
                            except ValueError:
                                # Skip chapters without valid numbers
                                continue
                        
                        completed_chapters.append({
                            'key': chapter_key,
                            'num': chapter_num,
                            'file': chapter_info.get("output_file"),
                            'ai_features': chapter_info.get("ai_features")
                        })
                    except Exception as e:
                        print(f"       ⚠️ Error processing chapter {chapter_key}: {e}")
                        continue
            
            # Sort by actual chapter number
            completed_chapters.sort(key=lambda x: x['num'])
            
            # If no current chapter number provided, try to infer it
            if current_chapter_num is None:
                # Try to get from progress if this chapter is already partially processed
                chapter_key = str(idx)
                if chapter_key in prog["chapters"]:
                    current_chapter_num = prog["chapters"][chapter_key].get("actual_num")
                    if current_chapter_num is None:
                        current_chapter_num = prog["chapters"][chapter_key].get("chapter_num")
                    print(f"    🔍 Found in progress: chapter_key={chapter_key}, actual_num={prog['chapters'][chapter_key].get('actual_num')}, chapter_num={prog['chapters'][chapter_key].get('chapter_num')}")
                
                # If still None, use index + 1 as fallback
                if current_chapter_num is None:
                    current_chapter_num = idx + 1
                    print(f"    ⚠️ Using index-based chapter number: {current_chapter_num}")
            
            print(f"\n    📚 Found {len(completed_chapters)} completed chapters in progress")
            if completed_chapters:
                chapter_nums = [ch['num'] for ch in completed_chapters]
                print(f"    📊 Chapter numbers in progress: {sorted(chapter_nums)[:10]}{'...' if len(chapter_nums) > 10 else ''}")
            print(f"    🎯 Current chapter number: {current_chapter_num}")
            print(f"    🔍 Will check against last {lookback} chapters before chapter {current_chapter_num}")
            
            # Check previous chapters
            all_similarities = []
            highest_similarity = 0.0
            detected_method = None
            detected_chapter = None
            
            # FIX: Look at chapters by actual number, not index
            chapters_checked = 0
            for completed_chapter in reversed(completed_chapters):
                # Only check chapters that come before the current one
                # Ensure both values are integers for comparison
                try:
                    completed_chapter_num = int(completed_chapter['num'])
                    current_num = int(current_chapter_num)
                except (ValueError, TypeError) as e:
                    print(f"       ⚠️ Error converting chapter numbers to int: {e}")
                    continue
                
                if completed_chapter_num >= current_num:
                    continue
                    
                # Only check up to lookback number of chapters
                if chapters_checked >= lookback:
                    break
                    
                chapters_checked += 1
                
                print(f"\n    📝 Checking against chapter {completed_chapter['num']}...")
                
                # Get previous chapter features
                prev_features = completed_chapter.get('ai_features')
                prev_clean = None
                
                # Try to get cached features first
                if prev_features:
                    print(f"       ✅ Using cached features")
                else:
                    # Read and extract features
                    prev_path = os.path.join(out, completed_chapter['file'])
                    
                    if os.path.exists(prev_path):
                        try:
                            with open(prev_path, 'r', encoding='utf-8') as f:
                                prev_content = f.read()
                                prev_clean = self._preprocess_text(prev_content, config['preprocessing'])
                                
                                # Check length ratio
                                len_ratio = len(result_clean) / max(1, len(prev_clean))
                                if (len_ratio < config['edge_filters']['min_length_ratio'] or 
                                    len_ratio > config['edge_filters']['max_length_ratio']):
                                    print(f"       ⚠️ Length ratio out of bounds: {len_ratio:.2f}")
                                    continue
                                
                                prev_features = self._extract_text_features(prev_clean)
                                print(f"       📄 Extracted features from file")
                        except Exception as e:
                            print(f"       ❌ Failed to read file: {e}")
                            continue
                    else:
                        print(f"       ❌ File not found: {prev_path}")
                        continue
                
                # Calculate similarities
                print(f"       🔍 Calculating similarities...")
                similarities = self._calculate_all_similarities(
                    result_clean, result_features, 
                    prev_clean, prev_features, config
                )
                
                # Store for reporting
                all_similarities.append({
                    'chapter': completed_chapter['num'],
                    'similarities': similarities
                })
                
                # Log similarity scores
                for method, score in similarities.items():
                    if score > 0:
                        print(f"          {method}: {int(score*100)}%")
                
                # Check if duplicate based on configured mode
                is_duplicate, confidence, methods_triggered = self._evaluate_duplicate(
                    similarities, config
                )
                
                if is_duplicate:
                    print(f"\n    🚨 DUPLICATE DETECTED!")
                    print(f"       Detection mode: {config['detection_mode']}")
                    print(f"       Confidence: {int(confidence*100)}%")
                    print(f"       Triggered methods: {', '.join(methods_triggered)}")
                    print(f"       Match with: Chapter {completed_chapter['num']}")
                    print(f"    ========== AI HUNTER DEBUG END ==========\n")
                    return True, int(confidence * 100)
                
                # Track highest for reporting
                for method, sim in similarities.items():
                    if sim > highest_similarity:
                        highest_similarity = sim
                        detected_method = method
                        detected_chapter = completed_chapter['num']
            
            # No duplicate found
            print(f"\n    ✅ No duplicate found")
            if detected_method:
                print(f"       Highest similarity: {int(highest_similarity*100)}% via {detected_method}")
                print(f"       Closest match: Chapter {detected_chapter}")
            
            # Show top 3 closest matches
            if all_similarities:
                print(f"\n    📊 Top 3 closest matches:")
                sorted_chapters = sorted(all_similarities, 
                                       key=lambda x: self._get_chapter_score(x['similarities'], config), 
                                       reverse=True)[:3]
                for i, chapter_data in enumerate(sorted_chapters, 1):
                    score = self._get_chapter_score(chapter_data['similarities'], config)
                    print(f"       {i}. Chapter {chapter_data['chapter']}: {int(score*100)}%")
            
            print(f"    ========== AI HUNTER DEBUG END ==========\n")
            return False, 0
            
        except Exception as e:
            print(f"    ❌ AI Hunter detection failed with error: {e}")
            import traceback
            print(f"    {traceback.format_exc()}")
            print(f"    ========== AI HUNTER DEBUG END ==========\n")
            return False, 0
    
    def _preprocess_text(self, text, prep_config):
        """Preprocess text according to configuration"""
        # Remove HTML
        if prep_config.get('remove_html_spacing', True):
            text = re.sub(r'<[^>]+>', ' ', text)
        else:
            text = re.sub(r'<[^>]+>', '', text)
        
        # Normalize unicode
        if prep_config.get('normalize_unicode', True):
            text = unicodedata.normalize('NFKD', text)
        
        # Remove extra whitespace
        if prep_config.get('remove_extra_whitespace', True):
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\n\s*\n', '\n\n', text)
        
        text = text.strip()
        
        # Convert to lowercase if case-insensitive
        if prep_config.get('ignore_case', True):
            text = text.lower()
        
        return text
    
    def _calculate_all_similarities(self, result_clean, result_features, 
                                   prev_clean, prev_features, config):
        """Calculate all similarity metrics"""
        similarities = {}
        
        # Method 1: Exact content match
        if prev_clean is not None:
            sample_size = min(config['sample_size'], len(result_clean), len(prev_clean))
            exact_sim = self._calculate_exact_similarity(
                result_clean[:sample_size], 
                prev_clean[:sample_size]
            )
            similarities['exact'] = exact_sim
            
            # Method 2: Smart text similarity
            text_sim = self._calculate_smart_similarity(
                result_clean, prev_clean, config['sample_size']
            )
            similarities['text'] = text_sim
        else:
            similarities['exact'] = 0.0
            similarities['text'] = 0.0
        
        # Method 3: Semantic fingerprint
        semantic_sim = self._calculate_semantic_similarity(
            result_features.get('semantic', {}), 
            prev_features.get('semantic', {})
        )
        similarities['semantic'] = semantic_sim
        
        # Method 4: Structural signature
        structural_sim = self._calculate_structural_similarity(
            result_features.get('structural', {}), 
            prev_features.get('structural', {})
        )
        similarities['structural'] = structural_sim
        
        # Method 5: Character analysis
        char_sim = self._calculate_character_similarity(
            result_features.get('characters', []), 
            prev_features.get('characters', [])
        )
        similarities['character'] = char_sim
        
        # Method 6: Pattern analysis
        pattern_sim = self._calculate_pattern_similarity(
            result_features.get('patterns', {}), 
            prev_features.get('patterns', {})
        )
        similarities['pattern'] = pattern_sim
        
        return similarities
    
    def _evaluate_duplicate(self, similarities, config):
        """Evaluate if similarities indicate a duplicate based on detection mode"""
        mode = config['detection_mode']
        thresholds = {k: v/100.0 for k, v in config['thresholds'].items()}
        
        if mode == 'single_method':
            # Any method exceeding threshold
            for method, sim in similarities.items():
                if sim >= thresholds.get(method, 0.85):
                    return True, sim, [method]
            return False, 0, []
        
        elif mode == 'multi_method':
            # Multiple methods must agree
            triggered_methods = []
            for method, sim in similarities.items():
                if sim >= thresholds.get(method, 0.85):
                    triggered_methods.append(method)
            
            # Check if enough methods triggered
            required = config.get('multi_method_requirements', {}).get('methods_required', 2)
            min_methods = config.get('multi_method_requirements', {}).get('min_methods', [])
            
            if len(triggered_methods) >= required:
                # Check if at least one required method is included
                if not min_methods or any(m in triggered_methods for m in min_methods):
                    # Calculate average confidence of triggered methods
                    confidence = sum(similarities[m] for m in triggered_methods) / len(triggered_methods)
                    return True, confidence, triggered_methods
            
            return False, 0, []
        
        elif mode == 'weighted_average':
            # Calculate weighted average
            weights = config.get('weights', {})
            total_weight = sum(weights.get(m, 1.0) for m in similarities)
            weighted_sum = sum(similarities[m] * weights.get(m, 1.0) for m in similarities)
            weighted_avg = weighted_sum / total_weight if total_weight > 0 else 0
            
            # Check if weighted average exceeds average threshold
            avg_threshold = sum(thresholds.values()) / len(thresholds) if thresholds else 0.85
            
            if weighted_avg >= avg_threshold:
                # Find which methods contributed most
                triggered = [m for m, sim in similarities.items() 
                           if sim >= thresholds.get(m, 0.85)]
                return True, weighted_avg, triggered
            
            return False, 0, []
        
        return False, 0, []
    
    def _get_chapter_score(self, similarities, config):
        """Calculate overall score for a chapter comparison"""
        if config['detection_mode'] == 'weighted_average':
            weights = config.get('weights', {})
            total_weight = sum(weights.get(m, 1.0) for m in similarities)
            return sum(similarities.get(m, 0) * weights.get(m, 1.0) for m in similarities) / total_weight if total_weight > 0 else 0
        else:
            return max(similarities.values()) if similarities else 0
    
    def _extract_text_features(self, text):
        """Extract multiple features from text for AI Hunter analysis"""
        features = {
            'semantic': {},
            'structural': {},
            'characters': [],
            'patterns': {}
        }
        
        # Semantic fingerprint
        lines = text.split('\n')
        
        # Character extraction (names that appear 3+ times)
        words = re.findall(r'\b[A-Z][a-z]+\b', text)
        word_freq = Counter(words)
        features['characters'] = [name for name, count in word_freq.items() 
                                 if count >= 3 and name not in {
                                     'The', 'A', 'An', 'In', 'On', 'At', 'To', 
                                     'From', 'With', 'By', 'For', 'Of', 'As', 
                                     'But', 'And', 'Or', 'He', 'She', 'It', 
                                     'They', 'We', 'You', 'What', 'When', 'Where',
                                     'Who', 'Why', 'How', 'That', 'This', 'These'
                                 }]
        
        # Dialogue patterns
        dialogue_patterns = re.findall(r'"([^"]+)"', text)
        features['semantic']['dialogue_count'] = len(dialogue_patterns)
        features['semantic']['dialogue_lengths'] = [len(d) for d in dialogue_patterns[:10]]
        
        # Speaker patterns
        speaker_patterns = re.findall(r'(\w+)\s+(?:said|asked|replied|shouted|whispered)', text.lower())
        features['semantic']['speakers'] = list(set(speaker_patterns[:20]))
        
        # Number extraction
        numbers = re.findall(r'\b\d+\b', text)
        features['patterns']['numbers'] = numbers[:20]
        
        # Structural signature
        para_lengths = []
        dialogue_count = 0
        for para in text.split('\n\n'):
            if para.strip():
                para_lengths.append(len(para))
                if '"' in para:
                    dialogue_count += 1
        
        features['structural']['para_count'] = len(para_lengths)
        features['structural']['avg_para_length'] = sum(para_lengths) / max(1, len(para_lengths))
        features['structural']['dialogue_ratio'] = dialogue_count / max(1, len(para_lengths))
        
        # Create structural pattern string
        pattern = []
        for para in text.split('\n\n')[:20]:  # First 20 paragraphs
            if para.strip():
                if '"' in para:
                    pattern.append('D')  # Dialogue
                elif len(para) > 300:
                    pattern.append('L')  # Long
                elif len(para) < 100:
                    pattern.append('S')  # Short
                else:
                    pattern.append('M')  # Medium
        features['structural']['pattern'] = ''.join(pattern)
        
        # Action density
        action_verbs = len(re.findall(r'\b\w+ed\b', text))
        features['semantic']['action_density'] = action_verbs / max(1, len(text.split()))
        
        # Text length
        features['semantic']['text_length'] = len(text)
        
        return features
    
    def _calculate_exact_similarity(self, text1, text2):
        """Calculate exact text similarity"""
        return SequenceMatcher(None, text1, text2).ratio()
    
    def _calculate_smart_similarity(self, text1, text2, sample_size):
        """Smart similarity with configurable sample size"""
        if len(text1) > sample_size * 3 and len(text2) > sample_size * 3:
            # Use multiple samples
            samples1 = [
                text1[:sample_size],
                text1[len(text1)//2 - sample_size//2:len(text1)//2 + sample_size//2],
                text1[-sample_size:]
            ]
            samples2 = [
                text2[:sample_size],
                text2[len(text2)//2 - sample_size//2:len(text2)//2 + sample_size//2],
                text2[-sample_size:]
            ]
            similarities = [SequenceMatcher(None, s1, s2).ratio() 
                           for s1, s2 in zip(samples1, samples2)]
            return sum(similarities) / len(similarities)
        else:
            # Use full text up to sample size
            return SequenceMatcher(None, text1[:sample_size], text2[:sample_size]).ratio()
    
    def _calculate_semantic_similarity(self, sem1, sem2):
        """Calculate semantic fingerprint similarity"""
        score = 0.0
        weights = 0.0
        
        # Compare dialogue counts
        if 'dialogue_count' in sem1 and 'dialogue_count' in sem2:
            weights += 0.3
            if sem1['dialogue_count'] > 0 or sem2['dialogue_count'] > 0:
                ratio = min(sem1['dialogue_count'], sem2['dialogue_count']) / \
                       max(1, max(sem1['dialogue_count'], sem2['dialogue_count']))
                score += ratio * 0.3
        
        # Compare speakers
        if 'speakers' in sem1 and 'speakers' in sem2:
            weights += 0.4
            if sem1['speakers'] and sem2['speakers']:
                overlap = len(set(sem1['speakers']) & set(sem2['speakers']))
                total = len(set(sem1['speakers']) | set(sem2['speakers']))
                score += (overlap / max(1, total)) * 0.4
            elif not sem1['speakers'] and not sem2['speakers']:
                score += 0.4  # Both have no speakers
        
        # Compare dialogue lengths pattern
        if 'dialogue_lengths' in sem1 and 'dialogue_lengths' in sem2:
            weights += 0.2
            if sem1['dialogue_lengths'] and sem2['dialogue_lengths']:
                len1 = sem1['dialogue_lengths'][:10]
                len2 = sem2['dialogue_lengths'][:10]
                if len1 and len2:
                    avg1 = sum(len1) / len(len1)
                    avg2 = sum(len2) / len(len2)
                    ratio = min(avg1, avg2) / max(1, max(avg1, avg2))
                    score += ratio * 0.2
            elif not sem1['dialogue_lengths'] and not sem2['dialogue_lengths']:
                score += 0.2  # Both have no dialogue
        
        # Action density
        if 'action_density' in sem1 and 'action_density' in sem2:
            weights += 0.1
            act_sim = 1 - abs(sem1['action_density'] - sem2['action_density'])
            score += act_sim * 0.1
        
        return score / max(0.1, weights)
    
    def _calculate_structural_similarity(self, struct1, struct2):
        """Calculate structural signature similarity"""
        score = 0.0
        
        # Compare paragraph patterns
        if 'pattern' in struct1 and 'pattern' in struct2:
            pattern_sim = SequenceMatcher(None, struct1['pattern'], struct2['pattern']).ratio()
            score += pattern_sim * 0.5
        
        # Compare paragraph statistics
        if all(k in struct1 for k in ['para_count', 'avg_para_length', 'dialogue_ratio']) and \
           all(k in struct2 for k in ['para_count', 'avg_para_length', 'dialogue_ratio']):
            
            # Paragraph count ratio
            para_ratio = min(struct1['para_count'], struct2['para_count']) / \
                        max(1, max(struct1['para_count'], struct2['para_count']))
            score += para_ratio * 0.2
            
            # Average length ratio
            avg_ratio = min(struct1['avg_para_length'], struct2['avg_para_length']) / \
                       max(1, max(struct1['avg_para_length'], struct2['avg_para_length']))
            score += avg_ratio * 0.15
            
            # Dialogue ratio similarity
            dialogue_diff = abs(struct1['dialogue_ratio'] - struct2['dialogue_ratio'])
            score += (1 - min(1, dialogue_diff)) * 0.15
        
        return score
    
    def _calculate_character_similarity(self, chars1, chars2):
        """Calculate character overlap similarity"""
        if not chars1 and not chars2:
            return 1.0
        if not chars1 or not chars2:
            return 0.0
        
        set1 = set(chars1)
        set2 = set(chars2)
        
        overlap = len(set1 & set2)
        total = len(set1 | set2)
        
        return overlap / max(1, total)
    
    def _calculate_pattern_similarity(self, pat1, pat2):
        """Calculate pattern similarity (numbers, etc.)"""
        score = 0.0
        
        # Number overlap
        if 'numbers' in pat1 and 'numbers' in pat2:
            nums1 = set(pat1['numbers'])
            nums2 = set(pat2['numbers'])
            
            if nums1 or nums2:
                overlap = len(nums1 & nums2)
                total = len(nums1 | nums2)
                score = overlap / max(1, total)
            else:
                score = 1.0  # Both have no numbers
        
        return score
